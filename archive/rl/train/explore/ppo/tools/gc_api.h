// 기울기 검사 결과 (정밀도와 무관한 평범한 형) — fp32·fp64 두 빌드가 같이 쓴다.
#pragma once

#include <string>
#include <vector>

struct GcBlock {
  long off, n;
  int kind;  // PKind
  std::string name;
};
struct GcOut {
  double loss = 0;
  long n_params = 0;
  std::vector<double> ga;     // 해석 기울기 (전부)
  std::vector<GcBlock> blocks;
  std::vector<long> fd_ids;   // 차분한 변수
  std::vector<double> fd;     // 차분 기울기
  bool proj_ok = false;       // σ 범위 투영: 범위 밖에서 update 한 번 뒤 log σ 가 [proj_lo, proj_hi] 안인가
  double proj_lo = 0, proj_hi = 0;
  std::vector<double> proj_ls;
};

namespace xppo {
GcOut gc_run(bool full, bool negative, bool do_fd, int K, int mlp);
GcOut gc_run_graph(bool share, bool negative, bool do_fd, int K);   // 그래프 어텐션 정책 (gc_graph_impl.cuh)
}
namespace xppo64 {
GcOut gc_run(bool full, bool negative, bool do_fd, int K, int mlp);
GcOut gc_run_graph(bool share, bool negative, bool do_fd, int K);
}

// ---- 그래프 어텐션 신경망을 직접 돌린 결과 (tools/graph_net_check — gk_impl.cuh) ----
struct GkSpec {
  int feat = 8, d = 128, heads = 8, layers = 6, ff = 512, glob = 0;
};
struct GkCase {   // 그래프 n 개 (격자 gw × gh)
  int gw = 0, gh = 0, n = 0;
  std::vector<unsigned char> feat;   // [n][gw·gh][feat]
  std::vector<unsigned> edge;        // [n][gw·gh]
  std::vector<int> cur;              // [n]
  std::vector<float> glob;           // [n][glob]
};
struct GkOut {
  std::vector<GcBlock> blocks;
  std::vector<double> params, logits, E, Hc, mask, grad;   // logits·mask [n][24], E·Hc [n][d], grad = ∂(Σ coef·logits)/∂변수 (coef 를 줬을 때)
  std::vector<unsigned char> bits;                          // logits | E | Hc 원래 바이트 (비트 비교용)
  long rows = 0;
};
namespace xppo {
// 씨앗 고정 초기화(+ perturb 면 모든 변수를 조금 흔듦 — 전역 벡터 가중치가 0 이 아니게) → 희소 정방향(+ 역방향)
GkOut gk_run(const GkSpec& s, const GkCase& c, const std::vector<double>& coef, bool perturb);
}
namespace xppo64 {
GkOut gk_run(const GkSpec& s, const GkCase& c, const std::vector<double>& coef, bool perturb);
}
