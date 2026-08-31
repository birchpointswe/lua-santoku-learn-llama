# santoku-learn-llama usage

Worked flows for the llama.cpp embedder and the text generator. Every section that has one names the
regress suite that anchors it. The `fvec`/`mtx` types are santoku-matrix
([matrix README](../../lua-santoku-matrix/README.md)); the KRR modelling is santoku-learn
([learn README](../../lua-santoku-learn/README.md)). All suites require a built llama.cpp and a gguf
model and skip when their model environment variable is unset.

## Load an embedder and encode a batch

`create(path)` (or `embedder(path)`) loads a gguf model in embed mode. `:dims()` is the embedding
size (`n_embd`). `:encode(texts)` takes an array of strings and returns a dense `fvec` of shape
`n_samples x n_embd` (row-major) plus that dimension as a second return value, with each row
L2-normalized.

```lua
local llama = require("santoku.learn.llama")

local enc    = llama.create(model_path)
local n_dims = enc:dims()

local codes, dim = enc:encode({ "first document", "second document" })  -- fvec, 2 x n_dims
```

Pass a batch width and thread count through the options form when you need them:

```lua
local enc = llama.create(model_path, { n_seq = 32, n_threads = 8 })
-- equivalently: llama.embedder(model_path, { n_seq = 32 })
-- or just a width: llama.create(model_path, 32)
```

Anchor: `test/spec/santoku/learn/regress/imdb-llama.lua`.

## Hand the codes to the KRR pipeline

The embeddings are the dense `pool_codes` that `optimize.krr` consumes. Wrap the returned `fvec` in
an `mtx` (`n_rows` = split size, `n_cols` = `enc:dims()`) and pass it with the label `csr` from the
dataset loader; `optimize.krr` cross-validates over `folds` internally, so there is no separate
validation split to encode. This is the same modelling call the sparse n-gram path uses, with model
embeddings standing in for the `csr` codes.

`optimize.krr` returns `spectral_enc, ridge, deploy, params, decider`: `spectral_enc` is the
Nyström encoder, `deploy(codes)` is a function wrapping `spectral_enc:encode`, and `decider` is the
calibrated decision layer. Deploy by encoding the test split, pushing it through that encoder, then
`ridge:regress` (dense scores) or `ridge:label` (top-`k` `csr`), and scoring with `decider`.

```lua
local mtx      = require("santoku.mtx")
local optimize = require("santoku.learn.optimize")
local util     = require("santoku.learn.util")
local utc      = require("santoku.utc")

local stopwatch   = utc.stopwatch()
local train_codes = enc:encode(train.problems)
train_codes = mtx.create({ n_rows = train.n, n_cols = enc:dims(), data = train_codes })

local sp_enc, ridge_obj, _, best, decider = optimize.krr({
  pool_codes = train_codes,
  pool_labels = train.labels,
  pool_class = train.labels:neighbors(),
  n_labels = n_classes,
  folds = 3,
  n_landmarks = 1024 * 8,
  relevance = { "auc" },
  exponent = { { def = 0.0120756 } },
  lambda = { def = 2.35441e-05 },
  k = 1,
  search_trials = 0,
  each = util.make_ridge_log(stopwatch),
})

local test_codes = enc:encode(test_set.problems)
test_codes = mtx.create({ n_rows = test_set.n, n_cols = enc:dims(), data = test_codes })
local scores = ridge_obj:regress(sp_enc:encode(test_codes))
local _, m = decider:score({
  scores = scores, expected = test_set.labels, n_samples = test_set.n })
print(best.lambda, util.fmt_metrics(m))
```

Free each code matrix once it has been consumed (`train_codes = nil; collectgarbage("collect")`):
the encoded splits are the memory ceiling.

Anchor: `test/spec/santoku/learn/regress/newsgroups-llama.lua`.

## Sentence-pair inputs

The model sees one string per sample, so a pair task concatenates the two sides into a single text
before encoding. The encoded `fvec` then flows into `optimize.krr` unchanged. No in-tree suite
covers a pair dataset today.

```lua
local function pair_texts (lefts, rights, n)
  local texts = {}
  for i = 1, n do
    texts[i] = lefts[i] .. "\n" .. rights[i]
  end
  return texts
end

local train_codes = enc:encode(pair_texts(lefts, rights, #lefts))
```

## Extreme multi-label

For many labels, encode the splits the same way, then drive `optimize.krr` with a top-`k`
(`k = 256` in the suite) and a Matern kernel. Labels come in as the multi-label `csr`, and the
decider calibrates a threshold over the ranked labels rather than taking an argmax. The embedder's
role is unchanged: produce the dense codes.

```lua
local pool_codes = enc:encode(collect_texts(train.text_iter, train.n))
pool_codes = mtx.create({ n_rows = train.n, n_cols = enc:dims(), data = pool_codes })

local _, ridge_obj, deploy, best, decider = optimize.krr({
  pool_codes = pool_codes,
  pool_labels = train.labels,
  n_labels = train.n_labels,
  folds = 3,
  kernel = { "matern" },
  nu = { def = 3 },
  gamma = { def = 1.10565 },
  lambda = { def = 5.23323e-06 },
  relevance = { "auc" },
  exponent = { { def = 4.78458 } },
  n_landmarks = 1024 * 8,
  k = 256,
  search_trials = 0,
})

local test_codes = enc:encode(collect_texts(test_set.text_iter, test_set.n))
test_codes = mtx.create({ n_rows = test_set.n, n_cols = enc:dims(), data = test_codes })
local P = ridge_obj:label(deploy(test_codes), 256)
local _, m = decider:score({ pred = P, expected = test_set.labels, n_samples = test_set.n })
```

Anchor: `test/spec/santoku/learn/regress/eurlex-llama.lua`.

## Text generation

`generator(path[, opts])` (or `create(path, { mode = "generate" })`) loads a model in decode mode;
constructor options are `n_ctx` (default 4096, capped at the model's training context),
`n_threads`, `seed`, and `template` (`raw` default, `llama3`, `chatml`).
`:generate(prompt[, opts])` returns the completion string; `opts` covers `max_tokens`,
`temperature` (0 is greedy), `top_k`, `top_p`, `min_p`, `repeat_penalty`, `seed`, and a `stop`
list of strings that truncate the output. `:chat(messages[, opts])` requires a template
(`template = "llama3"` or `"chatml"` at construction) and builds the prompt from
`system`/`user`/`assistant_prefix` fields. `:dims()` returns the vocabulary size.

```lua
local g = llama.generator(model_path, { n_ctx = 2048 })
local out = g:generate(
  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
  .. "Say hi in one word.<|eot_id|>"
  .. "<|start_header_id|>assistant<|end_header_id|>\n\n",
  { max_tokens = 16, temperature = 0.0, stop = { "\n", "<|" } })

local gc  = llama.generator(model_path, { n_ctx = 2048, template = "llama3" })
local ans = gc:chat(
  { system = "You write terse one-line answers.", user = "Say hi in one word." },
  { max_tokens = 16, temperature = 0.0, stop = { "\n" } })
```

Anchor: `test/spec/santoku/learn/regress/generate-llama.lua`.

## Gotchas

- The embedder returns an `fvec` plus its dimension, not a `csr`. Wrap the `fvec` in an `mtx` and
  use it as `pool_codes`, not through the sparse `tokenizer` path.
- Each text is truncated to the model's training context, and the batch is encoded `n_seq`
  sequences at a time (default 32).
- L2 normalization is on by default. Pass `false` as the second `:encode` argument to turn it off.
- To bound memory, pass your own `fvec` as the third `:encode` argument (for example an mmap-backed
  one); in that form `:encode` writes in place and returns nothing.
- Pooling is fixed by the model: with a pooling type it pools, otherwise it uses the last-token
  embedding. There is no pooling option.
- `:chat` errors when no template was configured; use `generate()` for raw prompts or construct with
  `template = "llama3"`/`"chatml"`.
- The suites read `LLAMA_MODEL` (embedders) or `LLAMA_GEN_MODEL` (generator) and skip when unset;
  the classification suites also need their `test/res` datasets.
