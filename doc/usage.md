# santoku-learn-llama usage

Worked flows for the llama.cpp embedder and the text generator. Every section names the regress
suite that anchors it. The `fvec`/`mtx` types are santoku-matrix
([matrix README](../../lua-santoku-matrix/README.md)); the ridge/KRR modelling is santoku-learn
([learn README](../../lua-santoku-learn/README.md)). All suites require a built llama.cpp and a gguf
model and skip when their model environment variable is unset.

## Load an embedder and encode a batch

`create(path)` (or `embedder(path)`) loads a gguf model in embed mode. `:dims()` is the embedding
size (`n_embd`). `:encode(texts)` takes an array of strings and returns a dense `fvec` of shape
`n_samples x n_embd` (row-major), with each row L2-normalized.

```lua
local llama = require("santoku.learn.llama")

local enc    = llama.create(model_path)
local n_dims = enc:dims()

local codes = enc:encode({ "first document", "second document" })  -- fvec, 2 x n_dims
```

Pass a batch width and thread count through the options form when you need them:

```lua
local enc = llama.create(model_path, { n_seq = 32, n_threads = 8 })
-- equivalently: llama.embedder(model_path, { n_seq = 32 })
-- or just a width: llama.create(model_path, 32)
```

Anchor: `test/spec/santoku/learn/regress/imdb-llama.lua`.

## Hand the codes to the ridge pipeline

The embeddings are the dense `train_codes` that `optimize.ridge` consumes. Pass the `fvec`,
`n_samples`, and `n_dims = enc:dims()` alongside the label CSR offsets/neighbors; the validation
split is passed the same way. This is the same modelling call the BPE/vectorizer path uses, with
model embeddings standing in for the sparse n-gram codes.

```lua
local optimize = require("santoku.learn.optimize")
local eval     = require("santoku.learn.evaluator")

local train_codes = enc:encode(train.problems)
local val_codes   = enc:encode(validate.problems)

local ridge_obj, best_params = optimize.ridge({
  train_codes = train_codes, n_samples = train.n, n_dims = enc:dims(),
  label_offsets = train.sol_offsets, label_neighbors = train.sol_neighbors,
  n_labels = n_classes,
  val_codes = val_codes, val_n_samples = validate.n,
  val_expected_offsets = validate.sol_offsets, val_expected_neighbors = validate.sol_neighbors,
  lambda = { def = 5.49e-02 }, propensity_a = { def = 0.16 }, propensity_b = { def = 2.06 },
  k = 1, search_trials = 400,
})

local test_codes = enc:encode(test_set.problems)
local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
local stats = eval.class_accuracy(
  test_labels, test_set.sol_offsets, test_set.sol_neighbors, test_set.n, n_classes)
```

Free each code matrix once it has been consumed (`train_codes = nil; collectgarbage("collect")`):
the encoded splits are the memory ceiling.

Anchor: `test/spec/santoku/learn/regress/newsgroups-llama.lua`.

## Sentence-pair inputs

The model sees one string per sample, so a pair task concatenates the two sides into a single text
before encoding. The encoded `fvec` then flows into `optimize.ridge` unchanged.

```lua
local function build_pair_texts (split)
  local texts = {}
  for i = 1, split.n do
    local a = split.unique_texts[split.idx1:get(i - 1) + 1]
    local b = split.unique_texts[split.idx2:get(i - 1) + 1]
    texts[i] = a .. "\n" .. b
  end
  return texts
end

local train_codes = enc:encode(build_pair_texts(train))
```

Anchor: `test/spec/santoku/learn/regress/snli-llama.lua`.

## Extreme multi-label

For many labels, encode the splits the same way, then drive `optimize.ridge` with `gfm = true` and a
top-`k` (`k = 256` in the suite). The returned `gfm_obj` predicts per-sample cutoffs over the
ranked labels; `evaluator.retrieval_ks` and `rp_at_k` score the results. The embedder's role is
unchanged: produce the dense codes.

```lua
local train_codes = enc:encode(collect_texts(train.text_iter, train.n))
local dev_codes   = enc:encode(collect_texts(dev.text_iter, dev.n))

local ridge_obj, best_params, gfm_obj = optimize.ridge({
  train_codes = train_codes, n_samples = train.n, n_dims = enc:dims(),
  label_offsets = train.sol_offsets, label_neighbors = train.sol_neighbors,
  n_labels = train.n_labels,
  val_codes = dev_codes, val_n_samples = dev.n,
  val_expected_offsets = dev.sol_offsets, val_expected_neighbors = dev.sol_neighbors,
  lambda = { def = 2.73e-02 }, k = 256, search_trials = 400, gfm = true,
})
```

Anchor: `test/spec/santoku/learn/regress/eurlex-llama.lua`.

## Text generation

`generator(path[, opts])` (or `create(path, { mode = "generate" })`) loads a model in decode mode.
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

- The embedder returns an `fvec`, not a `csr`. Use it as `train_codes` with `n_dims = enc:dims()`,
  not through the sparse `vectorizer`/`spectral.encode` path.
- L2 normalization is on by default. Pass `false` as the second `:encode` argument to turn it off.
- To bound memory, pass your own `fvec` as the third `:encode` argument (for example an mmap-backed
  one); in that form `:encode` writes in place and returns nothing.
- Pooling is fixed by the model: with a pooling type it pools, otherwise it uses the last-token
  embedding. There is no pooling option.
- `:chat` errors when no template was configured; use `generate()` for raw prompts or construct with
  `template = "llama3"`/`"chatml"`.
- The suites read `LLAMA_MODEL` (embedders) or `LLAMA_GEN_MODEL` (generator) and skip when unset;
  the classification suites also need their `test/res` datasets.
