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
