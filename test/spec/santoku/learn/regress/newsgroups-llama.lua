local env = require("santoku.env")
local ds = require("santoku.learn.dataset")
local mtx = require("santoku.mtx")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local fs = require("santoku.fs")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

fs.stdout:setvbuf("line")

local model_path = env.var("LLAMA_MODEL", nil)
if not model_path then
  print("LLAMA_MODEL not set. Skipping.")
  return
end

local llama = require("santoku.learn.llama")

local cfg = {
  data = { max = nil },
  ridge = {
    lambda = { def = 2.35441e-05 },
    classes = 20,
    relevance = { "auc" },
    exponent = { { def = 0.0120756 } },
    search_trials = 0,
    k = 1,
  },
}

test("newsgroups classifier (llama)", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, test_set = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max)
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
  local _, test_m = decider:score({ scores = ridge_obj:regress(test_emb),
    n_samples = test_set.n, expected = test_y })
  str.printf("[Class] test %s %s\n", util.fmt_metrics(test_m), sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
