// GRU 한 층(network/gru.cuh) 검사 — 작은 크기, 판 경계(reset) 섞음.
//   gru_check [--negative]
// 1) CPU 참조(double, 같은 식)와 GPU fp64·fp32 정방향 비교
// 2) fp64 유한 차분(중앙, ε 1e-6, CPU 참조로 손실) vs GPU fp64 해석 기울기: 모든 변수 + 입력 F 열 (손실 = Σ w ⊙ h, w 무작위)
// 3) fp32 해석 기울기 vs fp64 해석 기울기 (같은 변수 값)
// 4) step() 을 걸음마다 부른 결과 = forward_seq (판 경계는 부르는 쪽이 h 를 0 으로)
// --negative: 걸음 1 의 출력 기울기 부호를 뒤집어 해석 기울기를 일부러 틀리게 → 검사가 잡아야 종료 0
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gru_check.h"

using namespace gruchk;
namespace {
// CPU 참조: gru.cuh 머리말의 식 그대로 (double)
std::vector<double> cpu_forward(const Data& d, const Out& L) {
  std::vector<double> out((size_t)NR * H);
  std::vector<double> hp(H);
  const double* P = d.params.data();
  for (int t = 0; t < T; ++t)
    for (int b = 0; b < B; ++b) {
      const int r = t * B + b;
      for (int j = 0; j < H; ++j) hp[j] = d.reset[r] != 0 ? 0.0 : (t == 0 ? d.h0[b * H + j] : out[(size_t)(r - B) * H + j]);
      for (int j = 0; j < H; ++j) {
        double g[3], gh[3];
        for (int q = 0; q < 3; ++q) {
          const int o = q * H + j;
          double a = P[L.bx + o];
          for (int k = 0; k < NF; ++k) a += P[L.wx + (long)o * (NF + ND) + k] * d.x[(size_t)r * LD + k];
          for (int k = 0; k < ND; ++k) a += P[L.wx + (long)o * (NF + ND) + NF + k] * d.x[(size_t)r * LD + OFFD + k];
          g[q] = a;
          double c = P[L.bh + o];
          for (int k = 0; k < H; ++k) c += P[L.wh + (long)k * 3 * H + o] * hp[k];
          gh[q] = c;
        }
        const double rr = 1 / (1 + std::exp(-(g[0] + gh[0]))), zz = 1 / (1 + std::exp(-(g[1] + gh[1])));
        const double nn = std::tanh(g[2] + rr * gh[2]);
        out[(size_t)r * H + j] = (1 - zz) * nn + zz * hp[j];
      }
    }
  return out;
}
double loss_of(const std::vector<double>& h, const std::vector<double>& w) {
  double s = 0;
  for (size_t i = 0; i < h.size(); ++i) s += h[i] * w[i];
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  const bool negative = argc > 1 && !std::strcmp(argv[1], "--negative");
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> U(-1, 1);
  const Out L = xnet64::gru_layout();
  Data d;
  d.params.resize(L.n);
  for (auto& v : d.params) v = 0.4 * U(rng);
  d.x.resize((size_t)NR * LD);
  for (auto& v : d.x) v = U(rng);
  d.h0.resize((size_t)B * H);
  for (auto& v : d.h0) v = 0.5 * U(rng);
  d.reset.assign(NR, 0.0);
  d.reset[2 * B + 1] = 1;   // 걸음 2 · 환경 1 에서 판이 새로 시작
  d.reset[0 * B + 2] = 1;   // 걸음 0 · 환경 2 (h0 무시)
  d.w.resize((size_t)NR * H);
  for (auto& v : d.w) v = U(rng);

  const Out o64 = xnet64::gru_run(d, negative), o32 = xnet::gru_run(d, negative);
  const auto ref = cpu_forward(d, L);
  double e64 = 0, e32 = 0;
  for (size_t i = 0; i < ref.size(); ++i) e64 = std::max(e64, std::fabs(o64.h[i] - ref[i])), e32 = std::max(e32, std::fabs(o32.h[i] - ref[i]));
  std::printf("정방향: GPU fp64 vs CPU 최대 차 %.2e · GPU fp32 vs CPU %.2e\n", e64, e32);
  bool ok = e64 < 1e-12 && e32 < 1e-5;

  const double eps = 1e-6;
  // 상대 오차, 바닥 1e-3: 유한 차분의 반올림 잡음(손실 크기 × 1e-16 / ε ≈ 3e-9 절대)이 작은 기울기(~1e-5)에서 상대 오차를 부풀리지 않게
  auto rel = [](double a, double b) { return std::fabs(a - b) / std::max(1e-3, std::fabs(a) + std::fabs(b)); };
  double worst = 0;
  std::string wname;
  for (auto& bl : L.blocks) {
    double bw = 0;
    for (long i = bl.off; i < bl.off + bl.n; ++i) {
      Data dp = d, dm = d;
      dp.params[i] += eps;
      dm.params[i] -= eps;
      const double fd = (loss_of(cpu_forward(dp, L), d.w) - loss_of(cpu_forward(dm, L), d.w)) / (2 * eps);
      const double re = rel(fd, o64.grad[i]);
      if (re > bw && re > 1e-5) std::printf("    %s[%ld]: 유한 차분 %.10e 해석 %.10e\n", bl.name.c_str(), i - bl.off, fd, o64.grad[i]);
      bw = std::max(bw, re);
    }
    std::printf("  %-5s 최대 상대 오차 %.2e\n", bl.name.c_str(), bw);
    if (bw > worst) worst = bw, wname = bl.name;
  }
  {
    double bw = 0;
    for (int r = 0; r < NR; ++r)
      for (int k = 0; k < NF; ++k) {
        Data dp = d, dm = d;
        dp.x[(size_t)r * LD + k] += eps;
        dm.x[(size_t)r * LD + k] -= eps;
        const double fd = (loss_of(cpu_forward(dp, L), d.w) - loss_of(cpu_forward(dm, L), d.w)) / (2 * eps);
        bw = std::max(bw, rel(fd, o64.dxf[(size_t)r * NF + k]));
      }
    std::printf("  입력 F 최대 상대 오차 %.2e\n", bw);
    if (bw > worst) worst = bw, wname = "입력 F";
  }
  std::printf("유한 차분 (fp64): 최대 %.2e (%s)\n", worst, wname.c_str());
  const bool fd_ok = worst < 1e-5;
  double e = 0;
  for (long i = 0; i < L.n; ++i) e = std::max(e, std::fabs(o32.grad[i] - o64.grad[i]) / (1e-3 + std::fabs(o64.grad[i])));
  std::printf("기울기 fp32 vs fp64: 최대 상대 차 %.2e\n", e);
  ok = ok && e < 1e-3;
  std::printf("step 걸음마다 vs forward_seq: fp64 최대 차 %.2e · fp32 %.2e\n", o64.step_diff, o32.step_diff);
  ok = ok && o64.step_diff < 1e-12 && o32.step_diff < 1e-5;
  if (negative) {
    std::printf(fd_ok ? "실패: 틀린 기울기를 못 잡음\n" : "통과: 틀린 기울기를 잡음\n");
    return fd_ok ? 1 : 0;
  }
  ok = ok && fd_ok;
  std::printf(ok ? "통과\n" : "실패\n");
  return ok ? 0 : 1;
}
