# santoku-learn-llama

A C binding to llama.cpp that turns text into dense embedding vectors, plus a small
text-generation entry point. The embeddings are produced as a santoku-matrix `fvec` (row-major
`n_samples x n_embd`), the same dense code shape that the santoku-learn ridge/KRR pipeline
consumes. This is the alternative to the bag-of-tokens `vectorizer` path in santoku-learn: instead
of BPE token n-grams, you feed model embeddings as the dense codes.

This README is a usage guide, not an API reference. **The tests are the spec**: each component
points at the regress suite that exercises its surface. Worked flows are in
[`doc/usage.md`](doc/usage.md).

The codes this module emits are handed to santoku-learn's `optimize.ridge`; see the
[learn README](../lua-santoku-learn/README.md) for the modelling pipeline. The `fvec`/`mtx` types
are santoku-matrix; this doc uses them but does not re-explain them, see the
[matrix README](../lua-santoku-matrix/README.md). Dataset loading and file helpers in the regress
suites come from santoku-learn's `dataset` module and [santoku-fs](../lua-santoku-fs/README.md).

## Runtime requirement

This binding links a built llama.cpp and needs a gguf model file at runtime. The regress suites are
benchmarks, not fast unit tests: each one reads the model path from the `LLAMA_MODEL` (or
`LLAMA_GEN_MODEL`) environment variable and prints `... not set. Skipping.` when it is absent, and
the classification suites also need their dataset directories under `test/res`. Run them only with a
model and data present.

## Module map

The module name is `santoku.learn.llama` (from `luaopen_santoku_learn_llama`). The module table
exposes three constructors; objects carry their own methods.

| Entry | Role | Anchor test |
|-------|------|-------------|
| `create(path[, n_seq])` / `create(path, opts)` | constructor; dispatches on `opts.mode` (`"embed"` default, `"generate"`) | `regress/newsgroups-llama.lua` |
| `embedder(path[, n_seq])` / `embedder(path, opts)` | explicit embedder constructor | `regress/imdb-llama.lua` |
| `generator(path[, opts])` | explicit text-generation constructor | `regress/generate-llama.lua` |
| embedder `:encode` / `:dims` | text batch to embeddings `fvec`; embedding dimension | `regress/newsgroups-llama.lua` |
| generator `:generate` / `:chat` / `:dims` | raw / templated text generation; vocab size | `regress/generate-llama.lua` |

`embedder`/`create` (embed mode) is the path the learn benchmarks use; `generator` is a separate
decode path for raw or chat-templated generation (`raw`, `llama3`, `chatml` templates).

## The core flow

Load an embedder, encode a batch of texts into an `fvec` of shape `n_samples x n_embd`, hand that
plus its dimension to santoku-learn's ridge solver, then label new splits.

```lua
local llama    = require("santoku.learn.llama")
local optimize = require("santoku.learn.optimize")   -- santoku-learn

local enc    = llama.create(model_path)              -- embed mode by default
local n_dims = enc:dims()                            -- model n_embd

local train_codes = enc:encode(train.problems)       -- {text,...} -> fvec (n x n_embd), L2-normalized
local val_codes   = enc:encode(validate.problems)

local ridge_obj, best_params = optimize.ridge({
  train_codes = train_codes, n_samples = train.n, n_dims = n_dims,
  label_offsets = train.sol_offsets, label_neighbors = train.sol_neighbors,
  n_labels = n_classes,
  val_codes = val_codes, val_n_samples = validate.n,
  val_expected_offsets = validate.sol_offsets, val_expected_neighbors = validate.sol_neighbors,
  lambda = { def = 5.49e-02 }, k = 1, search_trials = 400,
})

local test_codes = enc:encode(test_set.problems)
local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
```

See `test/spec/santoku/learn/regress/newsgroups-llama.lua` for this flow end to end.

## Conventions

- **Embeddings are an `fvec`, not a `csr`.** `:encode` returns a dense `fvec` plus the dimension;
  you pass it to `optimize.ridge` as `train_codes` with `n_dims = enc:dims()`. This is the dense
  codes the spectral/ridge pipeline consumes, in place of the sparse `csr` from the BPE
  `vectorizer` path.
- **L2 normalization is on by default.** `:encode(texts)` normalizes each row; pass `false` as the
  second argument to disable.
- **In-place into mmap.** `:encode(texts, normalize, out_fvec)` writes into a caller-supplied
  `fvec` (for example an mmap-backed one) instead of allocating; with `out` it returns nothing.
- **Pooling is model-driven.** The embedder uses the model's own pooling type when it has one,
  otherwise it takes the last-token embedding. There is no pooling option to set.
- **n_seq is the batch width.** The embedder defaults to 32 sequences per forward pass; the context
  per sequence is the model's training context. Encoding chunks the batch to fit.

## Scenarios to which test to read

| Task | Test |
|------|------|
| multiclass (single-label) | `regress/newsgroups-llama.lua` |
| binary classification | `regress/imdb-llama.lua` |
| sentence-pair classification (NLI) | `regress/snli-llama.lua` |
| extreme multi-label (XMLC) | `regress/eurlex-llama.lua` |
| raw / chat text generation | `regress/generate-llama.lua` |

## License

This binding is MIT-licensed. llama.cpp, which it links, is licensed separately under its own terms.

MIT License

Copyright 2025 Birch Point SWE

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
