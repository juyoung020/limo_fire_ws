// 유한 차분 기울기 검사 — PPO 손실(가우시안·베르누이 log π, 자른 대리 손실, 자른 가치 손실, 엔트로피) → 신경망 전체
// (u8 입력 합성곱 첫 층, float 합성곱, 평탄화, 정규화된 1D 입력, 선형, ELU, log σ, 덩이 누적) 기울기를 검사한다.
//
//   두 단계, 같은 씨앗·같은 입력·같은 가중치:
//   [1] fp64 빌드: 해석 기울기 vs 중앙 차분 (h = 1e-6) — 역방향 수식이 맞는지. 층마다 |g| 큰 원소 K 개.
//   [2] fp32 빌드(학습에 쓰는 것): 해석 기울기 vs [1] 의 fp64 해석 기울기 — 층마다 모든 원소.
//   상대 오차 = ‖g − g_기준‖ / max(‖g‖, ‖g_기준‖) (층 묶음 노름). 기준 < 1e-3 (ROADMAP S5).
//
//   fp32 로 직접 차분하면 손실 반올림(~1e-7)/h 잡음이 작은 층(합성곱 편향 등)에서 1e-3 을 넘는다 — 그래서 차분은 fp64 로,
//   fp32 는 fp64 해석값과 비교한다. fp32 직접 차분 결과도 참고로 찍는다(--fp32-fd).
//
//   덧붙여 σ 범위: 마지막 성분 log σ 를 상한(ln std_max) 위에 두고 차분 비교, 범위 밖 log σ 로 update 한 번 뒤 범위 안인지(PPOConfig std_min/std_max).
//
//   gradcheck            작은 구조(같은 코드 길, 작은 크기)
//   gradcheck --full     obs_layout.h 실제 크기 구조
//   gradcheck --negative ELU 역방향을 일부러 틀리게 → 검사가 잡아야(실패를 찾아야) 종료 0
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>

#include "gc_api.h"

static const char* kind_name(int k) {
  static const char* n[] = {"합성곱 W", "합성곱 b", "선형 W", "선형 b", "log σ"};
  return n[k];
}

int main(int argc, char** argv) {
  bool full = false, negative = false;
  int mlp = 1;  // fp32 쪽 MLP: 1 융합 GEMM fp32(정확, 기본), 0 cuBLAS, 2 bf16 (기준 1e-3 는 0·1 에만)
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--full")) full = true;
    if (!std::strcmp(argv[i], "--negative")) negative = true;
    if (!std::strcmp(argv[i], "--mlp") && i + 1 < argc) mlp = std::atoi(argv[++i]);
  }
  const int K = full ? 8 : 16;
  GcOut r64 = xppo64::gc_run(full, negative, true, K, 0);
  GcOut r32 = xppo::gc_run(full, negative, false, K, mlp);
  std::printf("fp32 쪽 MLP 길 --mlp %d\n", mlp);
  std::printf("구조 %s, 변수 %ld 개, 샘플 12 (덩이 4 × 3), 손실 fp64 %.9f / fp32 %.9f%s\n", full ? "실제(obs_layout)" : "작은",
              r64.n_params, r64.loss, r32.loss, negative ? "  [--negative: ELU 역방향 틀림]" : "");
  std::map<long, double> fd;
  for (size_t i = 0; i < r64.fd_ids.size(); ++i) fd[r64.fd_ids[i]] = r64.fd[i];
  std::map<std::string, double> wk1, wk2;
  double w1 = 0, w2 = 0;
  std::printf("%-22s %-9s %11s %16s %16s\n", "층", "종류", "|g|", "[1]64 vs 차분", "[2]32 vs 64");
  for (auto& b : r64.blocks) {
    double na = 0, nf = 0, nd = 0;
    for (long id = b.off; id < b.off + b.n; ++id) {
      auto it = fd.find(id);
      if (it == fd.end()) continue;
      na += r64.ga[id] * r64.ga[id];
      nf += it->second * it->second;
      nd += (r64.ga[id] - it->second) * (r64.ga[id] - it->second);
    }
    double e1 = std::sqrt(nd) / std::max({std::sqrt(na), std::sqrt(nf), 1e-300});
    double n64 = 0, n32 = 0, d = 0;
    for (long id = b.off; id < b.off + b.n; ++id) {
      n64 += r64.ga[id] * r64.ga[id];
      n32 += r32.ga[id] * r32.ga[id];
      d += (r64.ga[id] - r32.ga[id]) * (r64.ga[id] - r32.ga[id]);
    }
    double e2 = std::sqrt(d) / std::max({std::sqrt(n64), std::sqrt(n32), 1e-300});
    std::printf("%-22s %-9s %11.3e %16.2e %16.2e%s\n", b.name.c_str(), kind_name(b.kind), std::sqrt(n64), e1, e2,
                (e1 < 1e-3 && e2 < 1e-3) ? "" : "  ← 넘음");
    if (b.kind == 4)
      for (long id = b.off; id < b.off + b.n; ++id)
        std::printf("    [%ld] 해석64 %.9f  차분 %.9f  해석32 %.9f\n", id - b.off, r64.ga[id], fd.count(id) ? fd[id] : 0.0, r32.ga[id]);
    if (std::sqrt(n64) < 1e-12) std::printf("    경고: 기울기 0 — 이 층으로 기울기가 안 흐름\n");
    wk1[kind_name(b.kind)] = std::max(wk1[kind_name(b.kind)], e1);
    wk2[kind_name(b.kind)] = std::max(wk2[kind_name(b.kind)], e2);
    w1 = std::max(w1, e1);
    w2 = std::max(w2, e2);
  }
  std::printf("층 종류별 최대 [1] fp64 해석 vs 차분:");
  for (auto& [k, v] : wk1) std::printf("  %s %.2e", k.c_str(), v);
  std::printf("\n층 종류별 최대 [2] fp32 vs fp64 해석:");
  for (auto& [k, v] : wk2) std::printf("  %s %.2e", k.c_str(), v);
  std::printf("\n");
  for (const GcOut* r : {&r64, &r32}) {
    std::printf("σ 범위 투영 (%s): log σ (+3, −10, +0.5) → update 뒤", r == &r64 ? "fp64" : "fp32");
    for (double l : r->proj_ls) std::printf(" %.4f", l);
    std::printf("  [%.4f, %.4f] %s\n", r->proj_lo, r->proj_hi, r->proj_ok ? "안" : "← 밖");
  }
  bool pass = w1 < 1e-3 && w2 < 1e-3 && r64.proj_ok && r32.proj_ok;
  if (negative) {
    bool caught = w1 >= 1e-3;
    std::printf("%s: 틀린 기울기를 %s ([1] 최대 %.2e)\n", caught ? "통과" : "실패", caught ? "잡음" : "못 잡음", w1);
    return caught ? 0 : 1;
  }
  std::printf("%s: [1] 최대 %.2e, [2] 최대 %.2e (기준 < 1e-3)\n", pass ? "통과" : "실패", w1, w2);
  return pass ? 0 : 1;
}
