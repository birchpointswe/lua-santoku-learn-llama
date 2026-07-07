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
  data = { max = nil },
  n_landmarks = 1024 * 8,
  kernel = { "matern" },
  nu = { def = 3 },
  gamma = { def = 1.10565 },
  lambda = { def = 5.23323e-06 },
  relevance = { "auc" },
  exponent = { { def = 4.78458 } },
  k = 256,
  search_trials = 0,
  folds = 3,
}

local function collect_texts (text_iter_fn, n)
  local texts = {}
  local iter = text_iter_fn()
  for i = 1, n do texts[i] = iter() end
  return texts
end

test("eurlex classifier (llama)", function ()

  local stopwatch = utc.stopwatch()
  local function sw ()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, _, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  str.printf("[Data] pool=%d test=%d labels=%d folds=%d trials=%d %s\n",
    train.n, test_set.n, n_labels, cfg.folds, cfg.search_trials, sw())

  str.printf("[Llama] Loading model\n")
  local enc = llama.create(model_path)
  local n_dims = enc:dims()
  str.printf("[Llama] n_embd=%d %s\n", n_dims, sw())

  str.printf("[Llama] Encoding pool (%d texts)\n", train.n)
  local pool_codes = enc:encode(collect_texts(train.text_iter, train.n))
  pool_codes = mtx.create({ n_rows = train.n, n_cols = n_dims, data = pool_codes })
  str.printf("[Llama] Pool encoded %s\n", sw())

  str.printf("[Llama] Encoding test (%d texts)\n", test_set.n)
  local test_codes = enc:encode(collect_texts(test_set.text_iter, test_set.n))
  test_codes = mtx.create({ n_rows = test_set.n, n_cols = n_dims, data = test_codes })
  str.printf("[Llama] Test encoded %s\n", sw())

  local pool_y = train.labels
  local test_y = test_set.labels

  str.printf("[Ridge] Fitting\n")
  local buf = "test/res/eurlex57k/cv_"
  local _, ridge_obj, deploy, best, decider = optimize.krr({
    pool_codes = pool_codes,
    pool_labels = pool_y,
    n_labels = n_labels,
    folds = cfg.folds,
    relevance = cfg.relevance, exponent = cfg.exponent,
    kernel = cfg.kernel, nu = cfg.nu, gamma = cfg.gamma,
    lambda = cfg.lambda,
    n_landmarks = cfg.n_landmarks, k = cfg.k, cv_buf_path = buf,
    search_trials = cfg.search_trials, each = util.make_ridge_log(stopwatch),
  })
  pool_codes = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e %s\n", best.lambda, sw())

  local P = ridge_obj:label(deploy(test_codes), cfg.k)
  local _, m = decider:score({ pred = P, expected = test_y, n_samples = test_set.n })
  str.printf("[Result] offset=%.6g | test %s %s\n", decider:offset(), util.fmt_metrics(m), sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
