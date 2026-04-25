#include <lua.h>
#include <lauxlib.h>
#include <santoku/lua/utils.h>
#include <santoku/fvec.h>
#include <llama.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TK_LLAMA_MT "tk_llama_t"
#define TK_LLAMA_GEN_MT "tk_llama_gen_t"
#define TK_LLAMA_DEFAULT_N_SEQ 32
#define TK_LLAMA_GEN_DEFAULT_N_CTX 4096
#define TK_LLAMA_GEN_DEFAULT_MAX_TOKENS 64
#define TK_LLAMA_GEN_DEFAULT_TOP_K 40
#define TK_LLAMA_GEN_DEFAULT_TOP_P 0.95f
#define TK_LLAMA_GEN_DEFAULT_MIN_P 0.05f
#define TK_LLAMA_GEN_DEFAULT_TEMP 0.0f
#define TK_LLAMA_GEN_DEFAULT_REPEAT_PENALTY 1.0f
#define TK_LLAMA_GEN_DEFAULT_PENALTY_LAST_N 64

typedef enum {
  TK_LLAMA_TEMPLATE_RAW = 0,
  TK_LLAMA_TEMPLATE_LLAMA3,
  TK_LLAMA_TEMPLATE_CHATML,
} tk_llama_template_t;

static int tk_llama_parse_template (const char *s, tk_llama_template_t *out) {
  if (!s || !*s || strcmp(s, "raw") == 0) { *out = TK_LLAMA_TEMPLATE_RAW; return 0; }
  if (strcmp(s, "llama3") == 0) { *out = TK_LLAMA_TEMPLATE_LLAMA3; return 0; }
  if (strcmp(s, "chatml") == 0) { *out = TK_LLAMA_TEMPLATE_CHATML; return 0; }
  return -1;
}

static int tk_llama_refs = 0;

static void tk_llama_log_noop (enum ggml_log_level level, const char *text, void *user_data) {
  (void)level; (void)text; (void)user_data;
}

typedef struct {
  struct llama_model *model;
  struct llama_context *ctx;
  int32_t n_embd;
  int32_t n_ctx;
  int32_t n_seq;
  bool has_encoder;
  bool pooled;
  bool destroyed;
} tk_llama_t;

static inline tk_llama_t *tk_llama_peek (lua_State *L, int i) {
  return (tk_llama_t *)luaL_checkudata(L, i, TK_LLAMA_MT);
}

static inline int tk_llama_gc (lua_State *L) {
  tk_llama_t *ll = tk_llama_peek(L, 1);
  if (!ll->destroyed) {
    if (ll->ctx) llama_free(ll->ctx);
    if (ll->model) llama_model_free(ll->model);
    ll->destroyed = true;
    tk_llama_refs--;
    if (tk_llama_refs <= 0)
      llama_backend_free();
  }
  return 0;
}

static inline void tk_llama_l2_normalize (float *v, int32_t n) {
  float sum = 0.0f;
  for (int32_t i = 0; i < n; i++)
    sum += v[i] * v[i];
  if (sum > 0.0f) {
    float inv = 1.0f / sqrtf(sum);
    for (int32_t i = 0; i < n; i++)
      v[i] *= inv;
  }
}

static inline int tk_llama_encode_lua (lua_State *L) {
  tk_llama_t *ll = tk_llama_peek(L, 1);
  if (ll->destroyed)
    return luaL_error(L, "encode: encoder destroyed");
  luaL_checktype(L, 2, LUA_TTABLE);
  int n = (int)lua_objlen(L, 2);
  int32_t dim = ll->n_embd;
  int do_norm = lua_isnoneornil(L, 3) ? 1 : lua_toboolean(L, 3);

  int has_output = !lua_isnoneornil(L, 4);
  tk_fvec_t *out;
  if (has_output) {
    out = tk_fvec_peek(L, 4, "output");
  } else {
    out = tk_fvec_create(L, (uint64_t)n * (uint64_t)dim);
    out->n = (uint64_t)n * (uint64_t)dim;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(ll->model);
  int32_t max_tok = ll->n_ctx;
  int32_t max_seq = ll->n_seq;
  int32_t batch_tokens = max_tok * max_seq;

  int32_t *tok_lens = NULL;
  llama_token *tok_all = NULL;
  llama_token *tmp_buf = NULL;
  struct llama_batch batch = { 0 };
  int rc = 0;

  tok_lens = (int32_t *)malloc((uint64_t)n * sizeof(int32_t));
  tok_all = (llama_token *)malloc((uint64_t)n * (uint64_t)max_tok * sizeof(llama_token));
  if (!tok_lens || !tok_all) {
    rc = luaL_error(L, "encode: out of memory");
    goto done;
  }

  int32_t tmp_cap = max_tok * 2;
  tmp_buf = (llama_token *)malloc((uint64_t)tmp_cap * sizeof(llama_token));
  if (!tmp_buf) {
    rc = luaL_error(L, "encode: out of memory");
    goto done;
  }

  for (int i = 0; i < n; i++) {
    lua_rawgeti(L, 2, i + 1);
    size_t len;
    const char *text = lua_tolstring(L, -1, &len);
    lua_pop(L, 1);
    int32_t nt = llama_tokenize(vocab, text, (int32_t)len,
                                tmp_buf, tmp_cap, true, true);
    if (nt < 0) {
      int32_t need = -nt;
      if (need > tmp_cap) {
        tmp_cap = need;
        llama_token *t = (llama_token *)realloc(tmp_buf, (uint64_t)tmp_cap * sizeof(llama_token));
        if (!t) {
          rc = luaL_error(L, "encode: out of memory");
          goto done;
        }
        tmp_buf = t;
      }
      nt = llama_tokenize(vocab, text, (int32_t)len,
                          tmp_buf, tmp_cap, true, true);
      if (nt < 0) {
        rc = luaL_error(L, "encode: tokenization failed for text %d", i + 1);
        goto done;
      }
    }
    if (nt > max_tok)
      nt = max_tok;
    tok_lens[i] = nt;
    memcpy(tok_all + (uint64_t)i * (uint64_t)max_tok, tmp_buf,
           (uint64_t)nt * sizeof(llama_token));
  }
  free(tmp_buf);
  tmp_buf = NULL;

  batch = llama_batch_init(batch_tokens, 0, 1);

  int cur = 0;
  while (cur < n) {
    batch.n_tokens = 0;
    int32_t total = 0;
    int start = cur;
    int count = 0;

    while (cur < n && count < max_seq) {
      int32_t nt = tok_lens[cur];
      if (total + nt > batch_tokens && count > 0)
        break;
      llama_token *src = tok_all + (uint64_t)cur * (uint64_t)max_tok;
      for (int32_t j = 0; j < nt; j++) {
        int32_t idx = batch.n_tokens++;
        batch.token[idx] = src[j];
        batch.pos[idx] = j;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = (llama_seq_id)count;
        batch.logits[idx] = 0;
      }
      if (!ll->pooled)
        batch.logits[batch.n_tokens - 1] = 1;
      total += nt;
      count++;
      cur++;
    }

    int err = ll->has_encoder
      ? llama_encode(ll->ctx, batch)
      : llama_decode(ll->ctx, batch);
    if (err != 0) {
      rc = luaL_error(L, "encode: forward pass failed (batch at text %d)", start + 1);
      goto done;
    }

    if (ll->pooled) {
      for (int s = 0; s < count; s++) {
        float *emb = llama_get_embeddings_seq(ll->ctx, (llama_seq_id)s);
        if (!emb) {
          rc = luaL_error(L, "encode: no embeddings for text %d", start + s + 1);
          goto done;
        }
        memcpy(out->a + (uint64_t)(start + s) * (uint64_t)dim, emb,
               (uint64_t)dim * sizeof(float));
        if (do_norm)
          tk_llama_l2_normalize(out->a + (uint64_t)(start + s) * (uint64_t)dim, dim);
      }
    } else {
      int32_t tok_off = 0;
      for (int s = 0; s < count; s++) {
        tok_off += tok_lens[start + s];
        float *emb = llama_get_embeddings_ith(ll->ctx, tok_off - 1);
        if (!emb) {
          rc = luaL_error(L, "encode: no embeddings for text %d", start + s + 1);
          goto done;
        }
        memcpy(out->a + (uint64_t)(start + s) * (uint64_t)dim, emb,
               (uint64_t)dim * sizeof(float));
        if (do_norm)
          tk_llama_l2_normalize(out->a + (uint64_t)(start + s) * (uint64_t)dim, dim);
      }
    }

    if (!ll->has_encoder)
      llama_memory_clear(llama_get_memory(ll->ctx), true);
  }

  rc = 0;

done:
  if (batch.token) llama_batch_free(batch);
  free(tmp_buf);
  free(tok_all);
  free(tok_lens);
  if (rc != 0)
    return rc;
  if (has_output) {
    return 0;
  }
  lua_pushinteger(L, dim);
  return 2;
}

static inline int tk_llama_dims_lua (lua_State *L) {
  tk_llama_t *ll = tk_llama_peek(L, 1);
  lua_pushinteger(L, ll->n_embd);
  return 1;
}

static luaL_Reg tk_llama_mt_fns[] = {
  { "encode", tk_llama_encode_lua },
  { "dims", tk_llama_dims_lua },
  { NULL, NULL }
};

typedef struct {
  struct llama_model *model;
  struct llama_context *ctx;
  int32_t n_vocab;
  int32_t n_ctx;
  uint32_t seed;
  tk_llama_template_t tmpl;
  bool destroyed;
} tk_llama_gen_t;

static inline tk_llama_gen_t *tk_llama_gen_peek (lua_State *L, int i) {
  return (tk_llama_gen_t *)luaL_checkudata(L, i, TK_LLAMA_GEN_MT);
}

static inline int tk_llama_gen_gc (lua_State *L) {
  tk_llama_gen_t *g = tk_llama_gen_peek(L, 1);
  if (!g->destroyed) {
    if (g->ctx) llama_free(g->ctx);
    if (g->model) llama_model_free(g->model);
    g->destroyed = true;
    tk_llama_refs--;
    if (tk_llama_refs <= 0)
      llama_backend_free();
  }
  return 0;
}

static inline int tk_llama_gen_dims_lua (lua_State *L) {
  tk_llama_gen_t *g = tk_llama_gen_peek(L, 1);
  lua_pushinteger(L, g->n_vocab);
  return 1;
}

static int tk_llama_do_generate (lua_State *L, tk_llama_gen_t *g, const char *prompt, size_t plen, int opts_idx) {
  int has_opts = opts_idx != 0;

  int32_t max_tokens = has_opts
    ? (int32_t)tk_lua_foptinteger(L, opts_idx, "generate", "max_tokens", TK_LLAMA_GEN_DEFAULT_MAX_TOKENS)
    : TK_LLAMA_GEN_DEFAULT_MAX_TOKENS;
  float temperature = has_opts
    ? (float)tk_lua_foptnumber(L, opts_idx, "generate", "temperature", TK_LLAMA_GEN_DEFAULT_TEMP)
    : TK_LLAMA_GEN_DEFAULT_TEMP;
  int32_t top_k = has_opts
    ? (int32_t)tk_lua_foptinteger(L, opts_idx, "generate", "top_k", TK_LLAMA_GEN_DEFAULT_TOP_K)
    : TK_LLAMA_GEN_DEFAULT_TOP_K;
  float top_p = has_opts
    ? (float)tk_lua_foptnumber(L, opts_idx, "generate", "top_p", TK_LLAMA_GEN_DEFAULT_TOP_P)
    : TK_LLAMA_GEN_DEFAULT_TOP_P;
  float min_p = has_opts
    ? (float)tk_lua_foptnumber(L, opts_idx, "generate", "min_p", TK_LLAMA_GEN_DEFAULT_MIN_P)
    : TK_LLAMA_GEN_DEFAULT_MIN_P;
  float repeat_penalty = has_opts
    ? (float)tk_lua_foptnumber(L, opts_idx, "generate", "repeat_penalty", TK_LLAMA_GEN_DEFAULT_REPEAT_PENALTY)
    : TK_LLAMA_GEN_DEFAULT_REPEAT_PENALTY;
  uint32_t seed = has_opts
    ? (uint32_t)tk_lua_foptinteger(L, opts_idx, "generate", "seed", g->seed)
    : g->seed;

  int n_stops = 0;
  const char **stop_strs = NULL;
  size_t *stop_lens = NULL;
  int stop_ref = LUA_NOREF;
  if (has_opts) {
    lua_getfield(L, opts_idx, "stop");
    if (lua_type(L, -1) == LUA_TTABLE) {
      n_stops = (int)lua_objlen(L, -1);
      if (n_stops > 0) {
        stop_strs = (const char **)malloc((size_t)n_stops * sizeof(char *));
        stop_lens = (size_t *)malloc((size_t)n_stops * sizeof(size_t));
        if (!stop_strs || !stop_lens) {
          free(stop_strs); free(stop_lens);
          return luaL_error(L, "generate: out of memory");
        }
        for (int i = 0; i < n_stops; i++) {
          lua_rawgeti(L, -1, i + 1);
          if (lua_type(L, -1) != LUA_TSTRING) {
            free(stop_strs); free(stop_lens);
            return luaL_error(L, "generate: stop[%d] not a string", i + 1);
          }
          stop_strs[i] = lua_tolstring(L, -1, &stop_lens[i]);
          lua_pop(L, 1);
        }
      }
      stop_ref = luaL_ref(L, LUA_REGISTRYINDEX);
    } else {
      lua_pop(L, 1);
    }
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(g->model);

  int32_t cap = (int32_t)plen + 16;
  llama_token *ptoks = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
  char *out = NULL;
  size_t out_len = 0, out_cap = 0;
  struct llama_sampler *smpl = NULL;
  struct llama_batch batch = { 0 };
  int rc = 0;

  if (!ptoks) {
    rc = luaL_error(L, "generate: out of memory");
    goto done;
  }

  int32_t nt = llama_tokenize(vocab, prompt, (int32_t)plen, ptoks, cap, false, true);
  if (nt < 0) {
    cap = -nt;
    llama_token *t = (llama_token *)realloc(ptoks, (size_t)cap * sizeof(llama_token));
    if (!t) {
      rc = luaL_error(L, "generate: out of memory");
      goto done;
    }
    ptoks = t;
    nt = llama_tokenize(vocab, prompt, (int32_t)plen, ptoks, cap, false, true);
    if (nt < 0) {
      rc = luaL_error(L, "generate: tokenization failed");
      goto done;
    }
  }
  if (nt == 0) {
    rc = luaL_error(L, "generate: empty prompt");
    goto done;
  }
  if (nt >= g->n_ctx) {
    rc = luaL_error(L, "generate: prompt (%d tokens) exceeds n_ctx (%d)", nt, g->n_ctx);
    goto done;
  }

  struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
  smpl = llama_sampler_chain_init(sp);
  if (temperature <= 0.0f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  } else {
    if (repeat_penalty > 1.0f)
      llama_sampler_chain_add(smpl,
        llama_sampler_init_penalties(TK_LLAMA_GEN_DEFAULT_PENALTY_LAST_N, repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
  }

  batch = llama_batch_init(g->n_ctx, 0, 1);
  batch.n_tokens = 0;
  for (int32_t i = 0; i < nt; i++) {
    int32_t idx = batch.n_tokens++;
    batch.token[idx] = ptoks[i];
    batch.pos[idx] = i;
    batch.n_seq_id[idx] = 1;
    batch.seq_id[idx][0] = 0;
    batch.logits[idx] = (i == nt - 1) ? 1 : 0;
  }
  if (llama_decode(g->ctx, batch) != 0) {
    rc = luaL_error(L, "generate: prompt decode failed");
    goto done;
  }
  int32_t n_past = nt;

  char piece_buf[256];
  int stopped = 0;

  for (int32_t i = 0; i < max_tokens; i++) {
    llama_token tok = llama_sampler_sample(smpl, g->ctx, -1);
    llama_sampler_accept(smpl, tok);
    if (llama_vocab_is_eog(vocab, tok)) break;

    char *piece = piece_buf;
    char *piece_dyn = NULL;
    int32_t pn = llama_token_to_piece(vocab, tok, piece_buf, sizeof piece_buf, 0, false);
    if (pn < 0) {
      int32_t need = -pn;
      piece_dyn = (char *)malloc((size_t)need);
      if (!piece_dyn) {
        rc = luaL_error(L, "generate: out of memory");
        goto done;
      }
      pn = llama_token_to_piece(vocab, tok, piece_dyn, need, 0, false);
      if (pn < 0) {
        free(piece_dyn);
        rc = luaL_error(L, "generate: token_to_piece failed");
        goto done;
      }
      piece = piece_dyn;
    }

    if (pn > 0) {
      if (out_len + (size_t)pn > out_cap) {
        size_t ncap = out_cap ? out_cap * 2 : 256;
        while (ncap < out_len + (size_t)pn) ncap *= 2;
        char *no = (char *)realloc(out, ncap);
        if (!no) {
          free(piece_dyn);
          rc = luaL_error(L, "generate: out of memory");
          goto done;
        }
        out = no; out_cap = ncap;
      }
      memcpy(out + out_len, piece, (size_t)pn);
      out_len += (size_t)pn;
    }
    free(piece_dyn);

    for (int si = 0; si < n_stops; si++) {
      size_t sl = stop_lens[si];
      if (sl == 0 || out_len < sl) continue;
      if (memcmp(out + out_len - sl, stop_strs[si], sl) == 0) {
        out_len -= sl;
        stopped = 1;
        break;
      }
    }
    if (stopped) break;

    if (i + 1 < max_tokens) {
      batch.n_tokens = 1;
      batch.token[0] = tok;
      batch.pos[0] = n_past;
      batch.n_seq_id[0] = 1;
      batch.seq_id[0][0] = 0;
      batch.logits[0] = 1;
      if (llama_decode(g->ctx, batch) != 0) {
        rc = luaL_error(L, "generate: decode failed at token %d", i);
        goto done;
      }
      n_past++;
    }
  }

done:
  if (batch.token) llama_batch_free(batch);
  if (smpl) llama_sampler_free(smpl);
  if (g->ctx) llama_memory_clear(llama_get_memory(g->ctx), true);
  free(ptoks);
  free(stop_strs);
  free(stop_lens);
  if (stop_ref != LUA_NOREF)
    luaL_unref(L, LUA_REGISTRYINDEX, stop_ref);
  if (rc != 0) {
    free(out);
    return rc;
  }
  lua_pushlstring(L, out ? out : "", out_len);
  free(out);
  return 1;
}

static inline int tk_llama_generate_lua (lua_State *L) {
  tk_llama_gen_t *g = tk_llama_gen_peek(L, 1);
  if (g->destroyed)
    return luaL_error(L, "generate: generator destroyed");
  size_t plen;
  const char *prompt = luaL_checklstring(L, 2, &plen);
  int opts_idx = 0;
  if (!lua_isnoneornil(L, 3)) {
    luaL_checktype(L, 3, LUA_TTABLE);
    opts_idx = 3;
  }
  return tk_llama_do_generate(L, g, prompt, plen, opts_idx);
}

static inline int tk_llama_chat_lua (lua_State *L) {
  tk_llama_gen_t *g = tk_llama_gen_peek(L, 1);
  if (g->destroyed)
    return luaL_error(L, "chat: generator destroyed");
  if (g->tmpl == TK_LLAMA_TEMPLATE_RAW)
    return luaL_error(L, "chat: no template configured; pass template='llama3'|'chatml' or use generate()");
  luaL_checktype(L, 2, LUA_TTABLE);
  int opts_idx = 0;
  if (!lua_isnoneornil(L, 3)) {
    luaL_checktype(L, 3, LUA_TTABLE);
    opts_idx = 3;
  }

  lua_getfield(L, 2, "system");
  const char *system = lua_isnil(L, -1) ? NULL : luaL_checkstring(L, -1);
  lua_getfield(L, 2, "user");
  if (lua_isnil(L, -1))
    return luaL_error(L, "chat: user field required");
  const char *user = luaL_checkstring(L, -1);
  lua_getfield(L, 2, "assistant_prefix");
  const char *prefix = lua_isnil(L, -1) ? NULL : luaL_checkstring(L, -1);

  luaL_Buffer b;
  luaL_buffinit(L, &b);
  switch (g->tmpl) {
    case TK_LLAMA_TEMPLATE_LLAMA3:
      luaL_addstring(&b, "<|begin_of_text|>");
      if (system) {
        luaL_addstring(&b, "<|start_header_id|>system<|end_header_id|>\n\n");
        luaL_addstring(&b, system);
        luaL_addstring(&b, "<|eot_id|>");
      }
      luaL_addstring(&b, "<|start_header_id|>user<|end_header_id|>\n\n");
      luaL_addstring(&b, user);
      luaL_addstring(&b, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      if (prefix) luaL_addstring(&b, prefix);
      break;
    case TK_LLAMA_TEMPLATE_CHATML:
      if (system) {
        luaL_addstring(&b, "<|im_start|>system\n");
        luaL_addstring(&b, system);
        luaL_addstring(&b, "<|im_end|>\n");
      }
      luaL_addstring(&b, "<|im_start|>user\n");
      luaL_addstring(&b, user);
      luaL_addstring(&b, "<|im_end|>\n<|im_start|>assistant\n");
      if (prefix) luaL_addstring(&b, prefix);
      break;
    default: break;
  }
  luaL_pushresult(&b);
  size_t plen;
  const char *prompt = lua_tolstring(L, -1, &plen);
  return tk_llama_do_generate(L, g, prompt, plen, opts_idx);
}

static luaL_Reg tk_llama_gen_mt_fns[] = {
  { "generate", tk_llama_generate_lua },
  { "chat", tk_llama_chat_lua },
  { "dims", tk_llama_gen_dims_lua },
  { NULL, NULL }
};

static int tk_llama_do_embedder (lua_State *L, const char *path, int32_t n_seq, int32_t n_threads) {
  if (tk_llama_refs == 0) {
    llama_backend_init();
    llama_log_set(tk_llama_log_noop, NULL);
  }
  struct llama_model_params mp = llama_model_default_params();
  struct llama_model *model = llama_model_load_from_file(path, mp);
  if (!model) {
    if (tk_llama_refs == 0)
      llama_backend_free();
    return luaL_error(L, "embedder: failed to load model '%s'", path);
  }
  int32_t n_embd = llama_model_n_embd(model);
  int32_t n_ctx = llama_model_n_ctx_train(model);
  int32_t n_total = n_ctx * n_seq;
  struct llama_context_params cp = llama_context_default_params();
  cp.embeddings = true;
  cp.n_ctx = (uint32_t)n_total;
  cp.n_batch = (uint32_t)n_total;
  cp.n_ubatch = (uint32_t)n_total;
  cp.n_seq_max = (uint32_t)n_seq;
  cp.n_threads = n_threads;
  cp.n_threads_batch = n_threads;
  struct llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    llama_model_free(model);
    if (tk_llama_refs == 0)
      llama_backend_free();
    return luaL_error(L, "embedder: failed to create context (n_seq=%d)", n_seq);
  }
  tk_llama_t *ll = tk_lua_newuserdata(L, tk_llama_t,
    TK_LLAMA_MT, tk_llama_mt_fns, tk_llama_gc);
  ll->model = model;
  ll->ctx = ctx;
  ll->n_embd = n_embd;
  ll->n_ctx = n_ctx;
  ll->n_seq = n_seq;
  ll->has_encoder = llama_model_has_encoder(model);
  ll->pooled = llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE;
  ll->destroyed = false;
  tk_llama_refs++;
  return 1;
}

static int tk_llama_do_generator (lua_State *L, const char *path, int32_t n_ctx_req, int32_t n_threads, uint32_t seed, tk_llama_template_t tmpl) {
  if (tk_llama_refs == 0) {
    llama_backend_init();
    llama_log_set(tk_llama_log_noop, NULL);
  }
  struct llama_model_params mp = llama_model_default_params();
  struct llama_model *model = llama_model_load_from_file(path, mp);
  if (!model) {
    if (tk_llama_refs == 0)
      llama_backend_free();
    return luaL_error(L, "generator: failed to load model '%s'", path);
  }
  int32_t n_ctx_train = llama_model_n_ctx_train(model);
  int32_t n_ctx = n_ctx_req > 0 ? n_ctx_req : TK_LLAMA_GEN_DEFAULT_N_CTX;
  if (n_ctx > n_ctx_train) n_ctx = n_ctx_train;
  struct llama_context_params cp = llama_context_default_params();
  cp.embeddings = false;
  cp.n_ctx = (uint32_t)n_ctx;
  cp.n_batch = (uint32_t)n_ctx;
  cp.n_ubatch = (uint32_t)n_ctx;
  cp.n_seq_max = 1;
  cp.n_threads = n_threads;
  cp.n_threads_batch = n_threads;
  cp.no_perf = true;
  struct llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    llama_model_free(model);
    if (tk_llama_refs == 0)
      llama_backend_free();
    return luaL_error(L, "generator: failed to create context (n_ctx=%d)", n_ctx);
  }
  tk_llama_gen_t *gg = tk_lua_newuserdata(L, tk_llama_gen_t,
    TK_LLAMA_GEN_MT, tk_llama_gen_mt_fns, tk_llama_gen_gc);
  gg->model = model;
  gg->ctx = ctx;
  gg->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  gg->n_ctx = n_ctx;
  gg->seed = seed;
  gg->tmpl = tmpl;
  gg->destroyed = false;
  tk_llama_refs++;
  return 1;
}

static int tk_llama_embedder_lua (lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  int32_t n_seq = TK_LLAMA_DEFAULT_N_SEQ;
  int32_t n_threads = omp_get_max_threads();
  int t = lua_type(L, 2);
  if (t == LUA_TNUMBER) {
    n_seq = (int32_t)luaL_checkinteger(L, 2);
  } else if (t == LUA_TTABLE) {
    n_seq = (int32_t)tk_lua_foptinteger(L, 2, "embedder", "n_seq", TK_LLAMA_DEFAULT_N_SEQ);
    n_threads = (int32_t)tk_lua_foptinteger(L, 2, "embedder", "n_threads", n_threads);
  } else if (t != LUA_TNONE && t != LUA_TNIL) {
    return luaL_error(L, "embedder: arg 2 must be integer or table");
  }
  return tk_llama_do_embedder(L, path, n_seq, n_threads);
}

static int tk_llama_generator_lua (lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  int32_t n_ctx = TK_LLAMA_GEN_DEFAULT_N_CTX;
  int32_t n_threads = omp_get_max_threads();
  uint32_t seed = 0;
  tk_llama_template_t tmpl = TK_LLAMA_TEMPLATE_RAW;
  int t = lua_type(L, 2);
  if (t == LUA_TTABLE) {
    n_ctx = (int32_t)tk_lua_foptinteger(L, 2, "generator", "n_ctx", TK_LLAMA_GEN_DEFAULT_N_CTX);
    n_threads = (int32_t)tk_lua_foptinteger(L, 2, "generator", "n_threads", n_threads);
    seed = (uint32_t)tk_lua_foptinteger(L, 2, "generator", "seed", 0);
    const char *ts = tk_lua_foptstring(L, 2, "generator", "template", NULL);
    if (tk_llama_parse_template(ts, &tmpl) != 0)
      return luaL_error(L, "generator: invalid template '%s'", ts);
  } else if (t != LUA_TNONE && t != LUA_TNIL) {
    return luaL_error(L, "generator: arg 2 must be a table");
  }
  return tk_llama_do_generator(L, path, n_ctx, n_threads, seed, tmpl);
}

static int tk_llama_create_lua (lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  int t = lua_type(L, 2);
  if (t == LUA_TNONE || t == LUA_TNIL)
    return tk_llama_do_embedder(L, path, TK_LLAMA_DEFAULT_N_SEQ, omp_get_max_threads());
  if (t == LUA_TNUMBER)
    return tk_llama_do_embedder(L, path, (int32_t)luaL_checkinteger(L, 2), omp_get_max_threads());
  if (t == LUA_TTABLE) {
    lua_getfield(L, 2, "mode");
    const char *mode = lua_type(L, -1) == LUA_TSTRING ? lua_tostring(L, -1) : NULL;
    lua_pop(L, 1);
    if (mode == NULL || strcmp(mode, "embed") == 0) {
      int32_t n_seq = (int32_t)tk_lua_foptinteger(L, 2, "create", "n_seq", TK_LLAMA_DEFAULT_N_SEQ);
      int32_t n_threads = (int32_t)tk_lua_foptinteger(L, 2, "create", "n_threads", omp_get_max_threads());
      return tk_llama_do_embedder(L, path, n_seq, n_threads);
    }
    if (strcmp(mode, "generate") == 0) {
      int32_t n_ctx = (int32_t)tk_lua_foptinteger(L, 2, "create", "n_ctx", TK_LLAMA_GEN_DEFAULT_N_CTX);
      int32_t n_threads = (int32_t)tk_lua_foptinteger(L, 2, "create", "n_threads", omp_get_max_threads());
      uint32_t seed = (uint32_t)tk_lua_foptinteger(L, 2, "create", "seed", 0);
      tk_llama_template_t tmpl = TK_LLAMA_TEMPLATE_RAW;
      const char *ts = tk_lua_foptstring(L, 2, "create", "template", NULL);
      if (tk_llama_parse_template(ts, &tmpl) != 0)
        return luaL_error(L, "create: invalid template '%s'", ts);
      return tk_llama_do_generator(L, path, n_ctx, n_threads, seed, tmpl);
    }
    return luaL_error(L, "create: invalid mode '%s'", mode);
  }
  return luaL_error(L, "create: arg 2 must be integer, table, or nil");
}

static luaL_Reg tk_llama_fns[] = {
  { "create", tk_llama_create_lua },
  { "embedder", tk_llama_embedder_lua },
  { "generator", tk_llama_generator_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_llama (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_llama_fns, 0);
  return 1;
}
