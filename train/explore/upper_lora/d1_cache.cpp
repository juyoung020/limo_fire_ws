// d1_cache — 위층 판단 모델 LoRA 학습의 특징 캐시 (upper_lora/README.md).
// 요청 본문(scripts/decider.cpp 가 만든 /v1/systemone JSON, combine_run --decider-log)마다 llama-server 와 같은 길로 프롬프트를 만들고
// (server_decision_context::fill_task — 서버 코드 그대로 링크) 모델을 한 번 돌려, 28 층(마지막 shortconv 두 층의 첫 층) 입력 = l_out-27 의 마지막 5 토큰 은닉,
// pick 질문의 선택지 글자 토큰 묶음, 그 토큰들의 마지막 자리 로짓(확인용)을 저장한다.
// 왜 5 토큰: 28 · 29 층은 커널 3 의 인과 shortconv 라 마지막 토큰 출력은 28 층 입력의 마지막 5 토큰에만 달림 (src/models/lfm-base.cpp build_lfm_shortconv).
//
//   d1_cache -m d1-3B-Q8_0.gguf --mmproj mmproj.gguf --out cache.bin  req1.json [req2.json …]   (또는 --list 파일)
//   --list 파일은 줄마다 "<요청 경로> [정답 선택지 번호]" (번호 없으면 −1)
//   출력 (작은 끝, 표본마다): u32 n_opt · u32 n_tok(label 토큰 수) · i32 target · u32 group[n_opt] · i32 tok[n_tok] · f32 logit[n_tok] (llama 가 낸 값) · f32 h[5][2048]
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-task.h"

namespace {
constexpr int kT = 5;
struct Cap {
  int n_embd = 0;
  std::vector<float> last;   // 마지막 kT 행 (구르는 고리)
  int seen = 0;
  std::string want = "l_out-27";
};
bool cb_eval(struct ggml_tensor* t, bool ask, void* ud) {
  Cap* c = (Cap*)ud;
  if (std::strcmp(t->name, c->want.c_str()) != 0) return ask ? false : true;
  if (ask) return true;
  const int ne0 = (int)t->ne[0], ne1 = (int)t->ne[1];
  std::vector<float> buf((size_t)ne0 * ne1);
  ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
  c->n_embd = ne0;
  if ((int)c->last.size() != kT * ne0) c->last.assign((size_t)kT * ne0, 0.f);
  for (int r = 0; r < ne1; ++r) {   // 고리에 행 차례로
    std::memmove(c->last.data(), c->last.data() + ne0, sizeof(float) * (size_t)(kT - 1) * ne0);
    std::memcpy(c->last.data() + (size_t)(kT - 1) * ne0, buf.data() + (size_t)r * ne0, sizeof(float) * ne0);
    c->seen += 1;
  }
  return true;
}
std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
}  // namespace

int main(int argc, char** argv) {
  std::string model, mmproj, out, list, lora;   // --lora: LoRA GGUF 를 켜고 돌림 (lora_export 결과 확인용 — 은닉은 28 층 입력이라 LoRA 와 무관, 로짓만 달라짐)
  std::vector<std::string> reqs;
  std::vector<int> tgt;
  int ngl = 99, nctx = 8192;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "-m") model = argv[++i];
    else if (k == "--mmproj") mmproj = argv[++i];
    else if (k == "--out") out = argv[++i];
    else if (k == "--list") list = argv[++i];
    else if (k == "--lora") lora = argv[++i];
    else if (k == "-ngl") ngl = std::stoi(argv[++i]);
    else if (k == "-c") nctx = std::stoi(argv[++i]);
    else reqs.push_back(k);
  }
  tgt.assign(reqs.size(), -1);
  if (!list.empty()) {
    std::ifstream f(list);
    for (std::string l; std::getline(f, l);) {
      if (l.empty()) continue;
      std::istringstream is(l);
      std::string p; int t = -1;
      is >> p >> t;
      reqs.push_back(p); tgt.push_back(t);
    }
  }
  if (model.empty() || out.empty() || reqs.empty()) { std::fprintf(stderr, "d1_cache -m model.gguf --mmproj mmproj.gguf --out cache.bin req.json …\n"); return 2; }
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = ngl;
  llama_model* m = llama_model_load_from_file(model.c_str(), mp);
  if (!m) { std::fprintf(stderr, "모델을 못 엶\n"); return 1; }
  Cap cap;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = nctx;
  cp.n_batch = nctx;
  cp.n_ubatch = 512;
  cp.cb_eval = cb_eval;
  cp.cb_eval_user_data = &cap;
  llama_context* ctx = llama_init_from_model(m, cp);
  if (!lora.empty()) {
    llama_adapter_lora* ad = llama_adapter_lora_init(m, lora.c_str());
    float sc = 1.f;
    if (!ad || llama_set_adapters_lora(ctx, &ad, 1, &sc) != 0) { std::fprintf(stderr, "LoRA 를 못 붙임\n"); return 1; }
  }
  mtmd_context_params mcp = mtmd_context_params_default();
  mcp.use_gpu = ngl > 0;
  mcp.media_marker = get_media_marker();   // 서버와 같은 표지 (server-context.cpp)
  { common_params cpar; mcp.image_min_tokens = cpar.image_min_tokens; mcp.image_max_tokens = cpar.image_max_tokens; mcp.batch_max_tokens = cpar.mtmd_batch_max_tokens; }
  mtmd_context* mctx = mtmd_init_from_file(mmproj.c_str(), m, mcp);
  if (!ctx || !mctx) { std::fprintf(stderr, "문맥을 못 만듦\n"); return 1; }
  server_decision_context dc;
  dc.init(m);
  const mtmd_helper_init_opt iopt = mtmd_helper_init_opt_default();
  std::FILE* fo = std::fopen(out.c_str(), "wb");
  const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
  long n_ok = 0;
  for (size_t ri = 0; ri < reqs.size(); ++ri) {
    const std::string& rp = reqs[ri];
    try {
      const json body = json::parse(slurp(rp));
      std::vector<raw_buffer> files;
      const json state = dc.parse_state(body, files);
      const std::vector<server_decision_question> qs = dc.parse_questions(body);
      const server_decision_question* q = nullptr;
      for (const auto& x : qs) if (x.id == "pick") q = &x;
      if (!q) throw std::runtime_error("pick 질문 없음");
      server_task task(SERVER_TASK_TYPE_DECISION);
      dc.fill_task(state, qs, *q, 0, files, mctx, iopt, task);
      llama_memory_clear(llama_get_memory(ctx), true);
      cap.seen = 0;
      llama_pos n_past = 0;
      const server_tokens& toks = task.tokens;
      size_t idx = 0;
      std::vector<llama_token> run;
      const auto flush = [&](bool last) {
        if (run.empty()) return;
        llama_batch b = llama_batch_init((int)run.size(), 0, 1);
        for (size_t k = 0; k < run.size(); ++k) {
          b.token[k] = run[k]; b.pos[k] = n_past + (llama_pos)k; b.n_seq_id[k] = 1; b.seq_id[k][0] = 0; b.logits[k] = last && k + 1 == run.size();
        }
        b.n_tokens = (int32_t)run.size();
        if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode 실패");
        n_past += (llama_pos)run.size();
        llama_batch_free(b);
        run.clear();
      };
      while (idx < toks.size()) {
        if (toks[idx] == LLAMA_TOKEN_NULL) {
          flush(false);
          const auto& ch = toks.find_chunk(idx);
          llama_pos np = n_past;
          if (mtmd_helper_eval_chunk_single(mctx, ctx, ch.get(), n_past, 0, (int)cp.n_batch, false, &np) != 0) throw std::runtime_error("그림 조각 실패");
          idx += mtmd_input_chunk_get_n_tokens(ch.get());
          n_past = np;
        } else {
          run.push_back(toks[idx]);
          ++idx;
        }
      }
      flush(true);
      const float* lg = llama_get_logits_ith(ctx, -1);
      const auto& labels = task.decision.labels;
      const auto& groups = task.decision.label_groups;
      if (cap.seen < kT || labels.empty()) throw std::runtime_error("은닉·라벨 없음");
      const uint32_t no = (uint32_t)groups.size(), nt = (uint32_t)labels.size();
      const int32_t tg = tgt[ri];
      std::fwrite(&no, 4, 1, fo); std::fwrite(&nt, 4, 1, fo); std::fwrite(&tg, 4, 1, fo);
      for (int32_t g : groups) { const uint32_t u = (uint32_t)g; std::fwrite(&u, 4, 1, fo); }
      for (llama_token t : labels) { const int32_t v = t; std::fwrite(&v, 4, 1, fo); }
      for (llama_token t : labels) { const float v = (t >= 0 && t < n_vocab) ? lg[t] : 0.f; std::fwrite(&v, 4, 1, fo); }
      std::fwrite(cap.last.data(), 4, cap.last.size(), fo);
      ++n_ok;
      std::fprintf(stderr, "%s: 선택지 %u · 라벨 토큰 %u · 프롬프트 %d\n", rp.c_str(), no, nt, (int)n_past);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: 실패 %s\n", rp.c_str(), e.what());
    }
  }
  std::fclose(fo);
  std::fprintf(stderr, "표본 %ld / %zu\n", n_ok, reqs.size());
  mtmd_free(mctx);
  llama_free(ctx);
  llama_model_free(m);
  return 0;
}
