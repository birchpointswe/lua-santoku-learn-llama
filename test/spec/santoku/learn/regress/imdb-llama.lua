local env = require("santoku.env")
local ds = require("santoku.learn.dataset")
local mtx = require("santoku.mtx")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local model_path = env.var("LLAMA_MODEL", nil)
if not model_path then
  print("LLAMA_MODEL not set. Skipping.")
  return
end

local llama = require("santoku.learn.llama")

local cfg = {
  data = { max = nil, ttr = 0.5 },
  ridge = {
    lambda = { def = 0.223725 },
    classes = 2,
    relevance = { "auc" },
    exponent = { { def = 0.001507 } },
    search_trials = 0,
    k = 1,
  },
}

test("imdb classifier (llama)", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set = ds.split_imdb(dataset, cfg.data.ttr)
  local n_classes = cfg.ridge.classes
  str.printf("[Data] train=%d test=%d classes=%d %s\n", train.n, test_set.n, n_classes, sw())

  str.printf("[Llama] Loading model\n")
  local enc = llama.create(model_path)
  local n_dims = enc:dims()
  str.printf("[Llama] n_embd=%d %s\n", n_dims, sw())

  str.printf("[Llama] Encoding train (%d texts)\n", train.n)
  local train_codes = enc:encode(train.problems)
  train_codes = mtx.create({ n_rows = train.n, n_cols = n_dims, data = train_codes })
  train.problems = nil
  str.printf("[Llama] Train encoded %s\n", sw())

  str.printf("[Ridge] Fitting\n")
  local pool_y = train.labels
  local enc2, ridge_obj, _, best_params, decider = optimize.krr({
    pool_codes = train_codes,
    pool_labels = pool_y,
    pool_class = pool_y:neighbors(),
    n_labels = n_classes,
    folds = 3,
    n_landmarks = 1024 * 8,
    relevance = cfg.ridge.relevance,
    exponent = cfg.ridge.exponent,
    lambda = cfg.ridge.lambda,
    k = cfg.ridge.k,
    search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  train_codes = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e %s\n", best_params.lambda, sw())

  str.printf("[Llama] Encoding test (%d texts)\n", test_set.n)
  local test_codes = enc:encode(test_set.problems)
  test_set.problems = nil
  test_codes = mtx.create({ n_rows = test_set.n, n_cols = n_dims, data = test_codes })
  local test_emb = enc2:encode(test_codes)
  test_codes = nil -- luacheck: ignore
  local test_y = test_set.labels
  local P = ridge_obj:label(test_emb, cfg.ridge.k)
  local _, test_m = decider:score({ pred = P, expected = test_y, n_samples = test_set.n })
  str.printf("[Class] test %s %s\n", util.fmt_metrics(test_m), sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
