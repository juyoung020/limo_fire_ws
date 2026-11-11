// 그래프 어텐션 신경망(network/graph_net.cu)의 희소 구현 확인. GPU 는 PPO 없이 몸통 + 포인터 머리만 (gk_impl.cuh).
//   [1] 조밀 기준: 간선 가림을 씌운 조밀 어텐션(network/graph_policy_cpu.h, CPU fp64) vs 희소(GPU fp64) — 로짓·E·Hc 최대 차 < 1e-9,
//       가림 같음, 기울기(∂ Σ c·로짓 / ∂ 변수: GPU 해석 vs 조밀 기준의 중앙 차분, 블록마다 |g| 큰 K 개) 상대 오차 < 1e-3.
//       작은 구조(d 16, 머리 4, 인코더 3 층, FF 24, 전역 3, 격자 9×7), 그래프 6 개.
//   [2] 무효 노드: 무효 노드의 특징·간선, 간선 윗 8 비트를 다른 쓰레기 값으로 → 출력 비트 같음 (HEADER 크기, fp32)
//   [3] 격자 크기: 16×16 격자의 그래프를 32×32 격자 안 (9, 5) 에 옮겨 놓음 → 출력 비트 같음 (같은 가중치)
//   [4] CPU 정방향(fp32, 배포용) vs GPU fp32: 로짓 최대 차 ≤ 1e-3, 가림·결정 행동 같음 (HEADER 크기, 유효 노드 약 150)
//   graph_net_check [--k 4]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../network/graph_policy_cpu.h"
#include "gc_api.h"

namespace {
constexpr int F = 8, NB = 24;

// 무작위 그래프: 유효 확률 pv, 간선 비트 확률 1/2 (한쪽 방향만 있는 간선·무효 이웃을 가리키는 비트·윗 8 비트 쓰레기 포함), 무효 노드에도 쓰레기
GkCase make_case(int gw, int gh, int n, int glob, int pv_255, uint32_t seed) {
  GkCase c;
  c.gw = gw;
  c.gh = gh;
  c.n = n;
  const int N = gw * gh;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> U8(0, 255);
  std::normal_distribution<float> G(0.f, 1.f);
  c.feat.resize((size_t)n * N * F);
  c.edge.resize((size_t)n * N);
  c.cur.resize(n);
  for (int i = 0; i < n; ++i) {
    std::vector<int> valid;
    for (int j = 0; j < N; ++j) {
      unsigned char* f = &c.feat[((size_t)i * N + j) * F];
      for (int k = 0; k < F; ++k) f[k] = (unsigned char)U8(rng);
      f[0] = U8(rng) < pv_255 ? 1 : 0;
      if (f[0]) valid.push_back(j);
      c.edge[(size_t)i * N + j] = (unsigned)rng();
    }
    if (valid.empty()) {
      c.feat[(size_t)i * N * F] = 1;
      valid.push_back(0);
    }
    c.cur[i] = valid[rng() % valid.size()];
    if (i == 1) c.edge[(size_t)i * N + c.cur[i]] &= 0xFF000000u;   // 고를 이웃이 없는 그래프
  }
  c.glob.resize((size_t)n * glob);
  for (auto& v : c.glob) v = G(rng);
  return c;
}

template <class T>
xnet_cpu::GraphPolicyT<T> make_cpu(const GkOut& o, int heads) {
  xnet_cpu::GraphPolicyT<T> p;
  for (auto& b : o.blocks) p.blk[b.name].assign(o.params.begin() + b.off, o.params.begin() + b.off + b.n);
  p.configure(heads);
  p.feat_stride = F;   // 버퍼는 8 칸, 신경망은 1 … 5 번만 읽는다
  return p;
}

// 조밀 기준 정방향 (그래프마다) → logits·mask [n][24], E·Hc [n][d], 결정 행동
template <class T>
void cpu_forward(const xnet_cpu::GraphPolicyT<T>& p, const GkCase& c, std::vector<double>& lg, std::vector<double>& mk, std::vector<double>& E,
                 std::vector<double>& Hc, std::vector<int>& act) {
  const int N = c.gw * c.gh, d = p.d;
  lg.assign((size_t)c.n * NB, 0);
  mk.assign((size_t)c.n * NB, 0);
  E.assign((size_t)c.n * d, 0);
  Hc.assign((size_t)c.n * d, 0);
  act.assign(c.n, 0);
  std::vector<T> l(NB), m(NB), e(d), h(d), g(p.glob + 1);
  for (int i = 0; i < c.n; ++i) {
    for (int k = 0; k < p.glob; ++k) g[k] = (T)c.glob[(size_t)i * p.glob + k];
    act[i] = p.act(c.gw, c.gh, &c.feat[(size_t)i * N * F], &c.edge[(size_t)i * N], c.cur[i], g.data(), l.data(), m.data(), e.data(), h.data());
    for (int k = 0; k < NB; ++k) {
      lg[(size_t)i * NB + k] = l[k];
      mk[(size_t)i * NB + k] = m[k];
    }
    for (int k = 0; k < d; ++k) {
      E[(size_t)i * d + k] = e[k];
      Hc[(size_t)i * d + k] = h[k];
    }
  }
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = a.size() == b.size() ? 0 : 1e30;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
long bit_diff(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
  if (a.size() != b.size()) return -1;
  long n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}
}  // namespace

int main(int argc, char** argv) {
  int K = 4;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--k") && i + 1 < argc) K = std::atoi(argv[++i]);
  bool ok = true;

  // ---------- [1] 조밀 기준 (fp64) ----------
  {
    GkSpec s;
    s.d = 16;
    s.heads = 4;
    s.layers = 3;
    s.ff = 24;
    s.glob = 3;
    const GkCase c = make_case(9, 7, 6, s.glob, 150, 11);
    std::mt19937 rng(5);
    std::normal_distribution<float> G(0.f, 1.f);
    std::vector<double> coef((size_t)c.n * NB);
    for (auto& v : coef) v = G(rng);
    const GkOut g = xppo64::gk_run(s, c, coef, true);
    auto cpu = make_cpu<double>(g, s.heads);
    std::vector<double> lg, mk, E, Hc;
    std::vector<int> act;
    cpu_forward(cpu, c, lg, mk, E, Hc, act);
    const double dl = max_diff(lg, g.logits), dE = max_diff(E, g.E), dH = max_diff(Hc, g.Hc), dm = max_diff(mk, g.mask);
    int n_free = 0;
    for (double v : g.mask) n_free += v > 0.5;
    std::printf("[1] 조밀 기준(CPU fp64) vs 희소(GPU fp64): 그래프 %d 개, 유효 노드 합 %ld, 고를 수 있는 칸 %d / %d\n", c.n, g.rows, n_free, c.n * NB);
    std::printf("    최대 차: 로짓 %.3e  E %.3e  Hc %.3e  가림 %.0f   (기준 < 1e-9, 가림 0)\n", dl, dE, dH, dm);
    // 기울기: 조밀 기준의 중앙 차분
    auto L = [&]() {
      std::vector<double> l2, m2, e2, h2;
      std::vector<int> a2;
      cpu_forward(cpu, c, l2, m2, e2, h2, a2);
      double s2 = 0;
      for (size_t i = 0; i < l2.size(); ++i) s2 += coef[i] * l2[i];
      return s2;
    };
    double worst = 0;
    std::string worst_name;
    int n_fd = 0;
    for (auto& b : g.blocks) {
      if (b.name.rfind("gactor.", 0) != 0) continue;   // 가치 머리는 여기서 안 돈다
      std::vector<long> ids(b.n);
      for (long i = 0; i < b.n; ++i) ids[i] = i;
      const long k = std::min<long>(K, b.n);
      std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                        [&](long x, long y) { return std::fabs(g.grad[b.off + x]) > std::fabs(g.grad[b.off + y]); });
      auto& v = cpu.blk[b.name];
      for (long q = 0; q < k; ++q) {
        const long id = ids[q];
        const double o = v[id], h = 1e-6 * std::max(1.0, std::fabs(o));
        v[id] = o + h;
        const double Lp = L();
        v[id] = o - h;
        const double Lm = L();
        v[id] = o;
        const double fd = (Lp - Lm) / (2 * h), ga = g.grad[b.off + id];
        const double rel = std::fabs(fd - ga) / (std::max(std::fabs(fd), std::fabs(ga)) + 1e-12);
        ++n_fd;
        if (rel > worst) {
          worst = rel;
          worst_name = b.name;
        }
      }
    }
    std::printf("    기울기 (GPU 해석 vs 조밀 기준 중앙 차분, 변수 %d 개): 최대 상대 오차 %.3e (%s)   (기준 < 1e-3)\n", n_fd, worst, worst_name.c_str());
    const bool p = dl < 1e-9 && dE < 1e-9 && dH < 1e-9 && dm == 0 && worst < 1e-3 && n_free > 0;
    std::printf("    %s\n", p ? "통과" : "실패");
    ok = ok && p;
  }

  // ---------- [2]·[3]·[4] HEADER 크기, fp32 ----------
  GkSpec s;
  s.glob = 16;
  const GkCase small = make_case(16, 16, 4, s.glob, 150, 21);   // 유효 약 150 / 256
  // 32×32 안 (9, 5) 에 옮김. 바깥 자리는 무효 + 쓰레기
  auto embed = [&](uint32_t garbage_seed) {
    GkCase big = make_case(32, 32, small.n, s.glob, 0, garbage_seed);
    big.glob = small.glob;
    for (int i = 0; i < small.n; ++i) {
      big.feat[(size_t)i * 1024 * F] = 0;   // make_case 가 유효 노드 없을 때 살린 0 번 자리를 다시 무효로
      for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
          const size_t a = (size_t)i * 256 + y * 16 + x, b = (size_t)i * 1024 + (y + 5) * 32 + x + 9;
          std::memcpy(&big.feat[b * F], &small.feat[a * F], F);
          big.edge[b] = small.edge[a];
        }
      for (size_t j = 0; j < 1024; ++j) big.feat[((size_t)i * 1024 + j) * F] = big.feat[((size_t)i * 1024 + j) * F] ? 1 : 0;
      big.cur[i] = (small.cur[i] / 16 + 5) * 32 + small.cur[i] % 16 + 9;
    }
    return big;
  };
  const GkCase bigA = embed(31), bigB = embed(77);
  // bigC: 모든 무효 노드(상자 안팎)의 특징·간선이 bigA 와 다른 쓰레기, 유효 노드 간선의 윗 8 비트도 뒤집음
  GkCase bigC = bigB;
  for (size_t j = 0; j < bigC.edge.size(); ++j) {
    bigC.edge[j] ^= bigC.feat[j * F] ? 0xFF000000u : 0xFFFFFFFFu;
    if (!bigC.feat[j * F])
      for (int k = 1; k < F; ++k) bigC.feat[j * F + k] ^= 0x5A;
  }
  const GkOut o16 = xppo::gk_run(s, small, {}, true);
  const GkOut oA = xppo::gk_run(s, bigA, {}, true), oC = xppo::gk_run(s, bigC, {}, true);
  {
    size_t n_inv_diff = 0;
    for (size_t j = 0; j < bigA.edge.size(); ++j) n_inv_diff += bigA.feat[j * F] == 0 && std::memcmp(&bigA.feat[j * F + 1], &bigC.feat[j * F + 1], F - 1) != 0;
    const long b2 = bit_diff(oA.bits, oC.bits);
    std::printf("[2] 무효 노드 쓰레기 값 (HEADER 크기 fp32, 그래프 %d 개, 유효 노드 합 %ld / %d, 특징이 달라진 무효 노드 %zu 개 + 모든 간선 윗 8 비트 뒤집음):\n"
                "    출력(로짓·E·Hc) 다른 바이트 %ld / %zu, 가림 차 %.0f  → %s\n",
                bigA.n, oA.rows, bigA.n * 1024, n_inv_diff, b2, oA.bits.size(), max_diff(oA.mask, oC.mask),
                b2 == 0 && max_diff(oA.mask, oC.mask) == 0 && n_inv_diff > 0 ? "통과 (비트 같음)" : "실패");
    ok = ok && b2 == 0 && max_diff(oA.mask, oC.mask) == 0 && n_inv_diff > 0;
    // 예약 칸(6·7 번 — HEADER v' 5 칸 밖): 유효 노드의 예약 칸을 뒤집어도 출력이 비트까지 같아야 하고, 5 번 칸(전역 길잡이)을 뒤집으면 달라야 한다
    GkCase bigD = bigA, bigE = bigA;
    size_t n_res = 0;
    for (size_t j = 0; j < bigD.edge.size(); ++j)
      if (bigD.feat[j * F]) {
        for (int k = 6; k < F; ++k) bigD.feat[j * F + k] ^= 0xA5;
        bigE.feat[j * F + 5] ^= 0xA5;
        ++n_res;
      }
    const GkOut oD = xppo::gk_run(s, bigD, {}, true), oE = xppo::gk_run(s, bigE, {}, true);
    const long b4 = bit_diff(oA.bits, oD.bits), b5 = bit_diff(oA.bits, oE.bits);
    std::printf("    예약 칸(6·7 번)을 뒤집은 유효 노드 %zu 개: 다른 바이트 %ld (0 이어야 함) · 5 번 칸을 뒤집으면 %ld (> 0 이어야 함)  → %s\n", n_res, b4, b5,
                b4 == 0 && b5 > 0 && n_res > 0 ? "통과 (입력 5 칸)" : "실패");
    ok = ok && b4 == 0 && b5 > 0 && n_res > 0;
    const long b3 = bit_diff(o16.bits, oA.bits);
    std::printf("[3] 격자 16×16 ↔ 32×32 (같은 그래프를 (9, 5) 에 옮김, 같은 가중치): 다른 바이트 %ld / %zu, 최대 차 로짓 %.3e E %.3e, 가림 차 %.0f  → %s\n", b3,
                o16.bits.size(), max_diff(o16.logits, oA.logits), max_diff(o16.E, oA.E), max_diff(o16.mask, oA.mask),
                b3 == 0 && max_diff(o16.mask, oA.mask) == 0 ? "통과 (비트 같음)" : "실패");
    ok = ok && b3 == 0 && max_diff(o16.mask, oA.mask) == 0;
  }
  {
    auto cpu = make_cpu<float>(oA, s.heads);
    std::vector<double> lg, mk, E, Hc;
    std::vector<int> act;
    const auto t0 = std::chrono::steady_clock::now();
    cpu_forward(cpu, bigA, lg, mk, E, Hc, act);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / bigA.n;
    int same_act = 0, n_free = 0;
    double lmax = 0;
    for (int i = 0; i < bigA.n; ++i) {
      int a = -1;
      for (int k = 0; k < NB; ++k) {
        const size_t q = (size_t)i * NB + k;
        if (oA.mask[q] > 0.5) {
          ++n_free;
          lmax = std::max(lmax, std::fabs(oA.logits[q]));
          if (a < 0 || oA.logits[q] > oA.logits[(size_t)i * NB + a]) a = k;
        }
      }
      same_act += (a < 0 ? 0 : a) == act[i];
    }
    const double dl = max_diff(lg, oA.logits), dE = max_diff(E, oA.E), dm = max_diff(mk, oA.mask);
    const bool p = dl <= 1e-3 && dm == 0 && same_act == bigA.n;
    std::printf("[4] CPU 정방향(fp32) vs GPU fp32 (HEADER 크기, 그래프 %d 개, 고를 수 있는 칸 %d, max |로짓| %.2f):\n"
                "    최대 차 로짓 %.3e (기준 ≤ 1e-3)  E %.3e, 가림 차 %.0f, 결정 행동 같음 %d / %d, CPU 한 그래프 %.1f ms (스레드 1)  → %s\n",
                bigA.n, n_free, lmax, dl, dE, dm, same_act, bigA.n, ms, p ? "통과" : "실패");
    ok = ok && p;
  }
  std::printf("%s\n", ok ? "전체 통과" : "실패 있음");
  return ok ? 0 : 1;
}
