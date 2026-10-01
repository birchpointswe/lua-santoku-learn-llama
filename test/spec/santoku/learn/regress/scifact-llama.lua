-- SPDX-License-Identifier: MIT
-- SPDX-FileCopyrightText: 2026 Birch Point SWE
local env = require("santoku.env")
local ds = require("santoku.learn.dataset")
local retrieval = require("santoku.learn.retrieval")
local optimize = require("santoku.learn.optimize")
local mtx = require("santoku.mtx")
local num = require("santoku.num")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")
local fs = require("santoku.fs")

fs.stdout:setvbuf("line")

local model_path = env.var("LLAMA_RETRIEVAL_MODEL", nil)
if not model_path then
  print("LLAMA_RETRIEVAL_MODEL not set. Skipping.")
  return
end

local llama = require("santoku.learn.llama")

local cfg = {
  depth = 100,
  query_prefix = "Represent this sentence for searching relevant passages: ",
  bm25_ndcg = 0.6663,
  rerank_ndcg = 0.7092,
}

test("scifact FTS5 top 100 re-sorted by llama (bge-small-en-v1.5)", function ()
  local stopwatch = utc.stopwatch()
  local d = ds.read_beir("test/res/scifact", "test")
  str.printf("[Data] corpus=%d queries=%d\n", d.n_corpus, d.n_queries)

  local X, Q = retrieval.lexical({ corpus_texts = d.corpus_texts, query_texts = d.query_texts })
  local R = retrieval.bm25_ranker(X, Q)(1.2, 0.75, cfg.depth)
  local nd, m = R:ndcg(d.qrels, 10)
  str.printf("[Lexical] ndcg@10=%.4f\n", m)
  assert(num.abs(m - cfg.bm25_ndcg) < 1e-4)

  local enc = llama.create(model_path)
  local dim = enc:dims()
  local qtexts = {}
  for i = 1, d.n_queries do qtexts[i] = cfg.query_prefix .. d.query_texts[i] end
  local D = mtx.create({ data = enc:encode(d.corpus_texts), n_rows = d.n_corpus, n_cols = dim })
  local Qc = mtx.create({ data = enc:encode(qtexts), n_rows = d.n_queries, n_cols = dim })
  local d1, t1 = stopwatch()
  str.printf("[Llama] dim=%d encoded corpus and queries (%.1fs +%.1fs)\n", dim, t1, d1)

  local best = optimize.retrieval({ datasets = { {
    name = "scifact", candidates = R, qrels = d.qrels, query_codes = Qc, doc_codes = D } } })
  local R1 = retrieval.rerank({ candidates = R, query_codes = Qc, doc_codes = D, alpha = best.alpha })
  local n1, m1 = R1:ndcg(d.qrels, 10)
  local delta, p = nd:paired_test(n1, 2000, 1)
  local _, total = stopwatch()
  str.printf("[Rerank] alpha=%.2f ndcg@10=%.4f delta=%+.4f p=%.4f\nTotal: %.1fs\n", best.alpha, m1, delta, p, total)
  assert(num.abs(m1 - cfg.rerank_ndcg) < 1e-3, "llama rerank ndcg drifted from the pin")
end)
