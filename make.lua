local fs = require("santoku.fs")
local vendor = require("santoku.make.vendor")

local vendored = {
  {
    file = "deps/llama/llama.cpp-e6b4acfe86af380c4e631973b9caa14337954423.tar.gz",
    url = "https://github.com/ggml-org/llama.cpp/archive/e6b4acfe86af380c4e631973b9caa14337954423.tar.gz",
    sha256 = "e771e8e99c0195a8e532f6e9b44e482742614aa83d779878f957cf76c84a5728",
  },
}

local include = {}
for i = 1, #vendored do
  include[i] = vendored[i].file
end

local env = {
  name = "santoku-learn-llama",
  version = "2.1.1-1",
  license = "MIT",
  public = true,
  rules = {
    include = include,
  },
  dependencies = {
    "lua == 5.1",
    "santoku >= 2.0.0, < 3.0.0",
    "santoku-matrix >= 2.0.0, < 3.0.0",
  },
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wno-unused-parameter", "-fopenmp",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
    "-I$(PWD)/deps/llama/llama.cpp/include",
    "-I$(PWD)/deps/llama/llama.cpp/ggml/include",
  },
  ldflags = {
    "-Wl,--start-group",
    "$(shell find $(PWD)/deps/llama/llama.cpp/build -name '*.a' | tr '\\n' ' ')",
    "-Wl,--end-group",
    "-lstdc++", "-lm", "-fopenmp",
    "$(shell pkg-config --libs blas lapack lapacke)",
  },
  test = {
    dependencies = {
      "santoku-learn >= 2.0.0, < 3.0.0",
      "santoku-fs >= 2.0.0, < 3.0.0",
      "lua-cjson >= 2.1.0.10-1",
    }
  },
  configure = function (submake, envs)
    for i = 1, #vendored do
      local v = vendored[i]
      local dest = fs.join(envs.root.build_dir, v.file)
      submake.target({ dest }, { "make.lua" }, function ()
        vendor.fetch(v, dest)
      end)
    end
  end,
}

env.homepage = "https://github.com/birchpointswe/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }
