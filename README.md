<p align="center">
  <img src="https://santoku.dev/logo-santoku-learn-llama.png" height="64" alt="santoku-learn-llama">
</p>

# santoku-learn-llama

A C binding to [llama.cpp](https://github.com/ggml-org/llama.cpp) with two modes: turn a
batch of texts into dense embedding vectors, or generate text (raw, or through a chat
template). The embeddings come back as a santoku-matrix `fvec` ready to wrap in an `mtx`
and hand to the santoku-learn KRR pipeline, as the dense alternative to sparse ngrams.

## Install

```sh
luarocks install santoku-learn-llama
```

## Example

```lua
local llama = require("santoku.learn.llama")
local mtx = require("santoku.mtx")

local enc = llama.embedder(model_path)

local data, dim = enc:encode(texts)

local codes = mtx.create({ data = data, n_rows = #texts, n_cols = dim })
```

A generator is the same shape: `llama.generator(model_path, { n_ctx = 2048, template =
"llama3" })` gives you `:generate(prompt, opts)` and `:chat({ system = ..., user = ... },
opts)`.

## Documentation

Runnable examples and the full API: [santoku.dev](https://santoku.dev/#santoku-learn-llama).

For agents and LLM tooling: [llms.txt](https://santoku.dev/llms.txt) for the index,
[llms-full.txt](https://santoku.dev/llms-full.txt) for every documented example.

## Tests

The binding links a built llama.cpp and needs a gguf model at runtime, so the suites are
benchmarks rather than fast unit tests: each reads its model path from `LLAMA_MODEL` or
`LLAMA_GEN_MODEL` and skips when it is unset. Read them for the exhaustive surface:
[`test/spec/santoku/learn/regress/generate-llama.lua`](test/spec/santoku/learn/regress/generate-llama.lua),
[`newsgroups-llama.lua`](test/spec/santoku/learn/regress/newsgroups-llama.lua),
[`imdb-llama.lua`](test/spec/santoku/learn/regress/imdb-llama.lua), and
[`eurlex-llama.lua`](test/spec/santoku/learn/regress/eurlex-llama.lua).

## License

MIT, see [LICENSE](LICENSE).

## More examples

```lua
local test = require("santoku.test")

local err = require("santoku.error")
local assert = err.assert
local pcall = err.pcall

local validate = require("santoku.validate")
local eq = validate.isequal
local hascall = validate.hascall

local env = require("santoku.env")
local mtx = require("santoku.mtx")

local llama = require("santoku.learn.llama")

test("one module, two modes: embedding and generation", function ()
  assert(eq(true, hascall(llama.create)))
  assert(eq(true, hascall(llama.embedder)))
  assert(eq(true, hascall(llama.generator)))
end)

test("a model that will not load raises instead of aborting", function ()
  assert(eq(false, (pcall(function ()
    return llama.embedder("test/res/no-such-model.gguf")
  end))))
end)

local model_path = env.var("LLAMA_MODEL", nil)

if not model_path then
  print("LLAMA_MODEL not set. Skipping the encode example.")
  return
end

test("encode a batch of texts into dense codes for the krr pipeline", function ()
  local enc = llama.embedder(model_path)
  local texts = { "the cat sat on the mat", "kernel ridge regression" }
  local data, dim = enc:encode(texts)
  assert(eq(dim, enc:dims()))
  local codes = mtx.create({ data = data, n_rows = #texts, n_cols = dim })
  local rows, cols = codes:shape()
  assert(eq(2, rows))
  assert(eq(dim, cols))
end)
```
