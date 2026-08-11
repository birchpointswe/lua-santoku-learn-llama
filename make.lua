local env = {
  name = "santoku-learn-llama",
  version = "2.0.0-1",
  license = "MIT",
  public = true,
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
  }
}

env.homepage = "https://github.com/birchpointswe/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }
