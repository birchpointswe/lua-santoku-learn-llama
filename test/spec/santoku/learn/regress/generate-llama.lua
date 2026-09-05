local env = require("santoku.env")
local str = require("santoku.string")
local test = require("santoku.test")
local fs = require("santoku.fs")

fs.stdout:setvbuf("line")

local model_path = env.var("LLAMA_GEN_MODEL", nil)
if not model_path then
  print("LLAMA_GEN_MODEL not set. Skipping.")
  return
end

local llama = require("santoku.learn.llama")

test("llama generator raw", function ()
  local g = llama.generator(model_path, { n_ctx = 2048 })
  local prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    .. "Say hi in one word.<|eot_id|>"
    .. "<|start_header_id|>assistant<|end_header_id|>\n\n"
  local out = g:generate(prompt, { max_tokens = 16, temperature = 0.0, stop = { "\n", "<|" } })
  assert(out and #out > 0, "empty output")
  str.printf("[gen] vocab=%d out=%q\n", g:dims(), out)
end)

test("llama generator chat llama3", function ()
  local g = llama.generator(model_path, { n_ctx = 2048, template = "llama3" })
  local out = g:chat(
    { system = "You write terse one-line answers.", user = "Say hi in one word." },
    { max_tokens = 16, temperature = 0.0, stop = { "\n" } })
  assert(out and #out > 0, "empty output")
  str.printf("[chat] out=%q\n", out)
end)
