// 그래프 어텐션 정책 + PPO 검증: 장난감 환경 "미로 그래프에서 목표 쪽 이웃 고르기" — 여러 홉 떨어진 정보를 간선을 따라 모아야 풀린다.
//   환경(판 길이 1, 문맥 띠): 8×8 격자, 노드는 0.8 확률로 유효, 간선은 상하좌우 이웃끼리 0.6 확률(양방향) → 미로. 현재 노드(이웃 ≥ 2)와 같은 연결 성분에서
//   2 홉 이상 떨어진 목표 노드 하나(특징 3 번 칸 = 255). 고른 이웃이 목표까지의 그래프 거리(BFS)를 줄이면 보상 1, 아니면 0.
//   노드 특징: 유효, dx·dy(현재 노드 기준, 128 + 8·칸), 목표 표시. 전역 2 칸은 잡음.
//   기준 둘: 무작위(줄이는 이웃 수 / 이웃 수의 평균), 기하(목표와 직선 거리가 가장 가까운 이웃 — 간선 구조를 안 봄).
//   toy_graph [--iters 300] [--envs 256] [--seed 1] [--share]   학습 → 결정 정확도(새 그래프 4096) > 0.9 이고 가린 칸 고름 0 이면 통과
//   toy_graph --check-determinism [--iters 20]                  같은 씨앗 두 번 → 반복마다 손실 통계·체크포인트(가중치·Adam) 비트 같음
// 신경망은 HEADER 구조 그대로, 크기만 작게: d 32, 머리 4, 인코더 4 층, FF 64.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "ppo.h"

using namespace xnet;

namespace {
constexpr int G = 8, N = G * G, F = 8, NB = 24, GLOB = 2, ADIM = 4;

ACSpec toy_spec(bool share) {
  ACSpec s;
  s.graph = true;
  s.gspec = GraphSpec{G, G, F, 32, 4, 4, 64};
  s.graph_glob = GLOB;
  s.graph_share = share;
  s.graph_rows = 16384;
  NetSpec c;
  c.hidden = {32};
  c.out = 1;
  s.critic = c;
  s.n_cont = 0;
  return s;
}

int stencil(int dx, int dy) {   // mdp::graph: k = (dy+2)·5 + (dx+2), 가운데(12) 뒤는 −1
  const int i5 = (dy + 2) * 5 + dx + 2;
  return i5 > 12 ? i5 - 1 : i5;
}

struct Toy {
  int n;
  std::mt19937 rng;
  std::vector<uint8_t> feat;
  std::vector<uint32_t> edge;
  std::vector<int> cur, hops, geo;
  std::vector<float> glob, p_rand;
  std::vector<uint32_t> good;   // 거리를 줄이는 이웃 비트
  explicit Toy(int n_, uint64_t seed)
      : n(n_), rng((uint32_t)seed), feat((size_t)n_ * N * F), edge((size_t)n_ * N), cur(n_), hops(n_), geo(n_), glob((size_t)n_ * GLOB),
        p_rand(n_), good(n_) {}
  void sample() {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> Gn(0.f, 1.f);
    const int dxs[4] = {1, -1, 0, 0}, dys[4] = {0, 0, 1, -1};
    for (int i = 0; i < n; ++i) {
      uint8_t* f = &feat[(size_t)i * N * F];
      uint32_t* e = &edge[(size_t)i * N];
      for (;;) {
        bool val[N];
        for (int j = 0; j < N; ++j) {
          val[j] = U(rng) < 0.8f;
          e[j] = 0;
        }
        for (int j = 0; j < N; ++j) {
          if (!val[j]) continue;
          const int x = j % G, y = j / G;
          if (x + 1 < G && val[j + 1] && U(rng) < 0.6f) {
            e[j] |= 1u << stencil(1, 0);
            e[j + 1] |= 1u << stencil(-1, 0);
          }
          if (y + 1 < G && val[j + G] && U(rng) < 0.6f) {
            e[j] |= 1u << stencil(0, 1);
            e[j + G] |= 1u << stencil(0, -1);
          }
        }
        std::vector<int> cand;
        for (int j = 0; j < N; ++j)
          if (val[j] && __builtin_popcount(e[j]) >= 2) cand.push_back(j);
        if (cand.empty()) continue;
        const int c = cand[rng() % cand.size()];
        // 목표 후보: c 에서 2 홉 이상
        int dist[N], q[N];
        auto bfs = [&](int src) {
          for (int j = 0; j < N; ++j) dist[j] = -1;
          int qh = 0, qt = 0;
          dist[src] = 0;
          q[qt++] = src;
          while (qh < qt) {
            const int a = q[qh++];
            for (int k = 0; k < 4; ++k)
              if ((e[a] >> stencil(dxs[k], dys[k])) & 1u) {
                const int b = a + dxs[k] + dys[k] * G;
                if (dist[b] < 0) {
                  dist[b] = dist[a] + 1;
                  q[qt++] = b;
                }
              }
          }
        };
        bfs(c);
        std::vector<int> far;
        for (int j = 0; j < N; ++j)
          if (dist[j] >= 2) far.push_back(j);
        if (far.empty()) continue;
        const int goal = far[rng() % far.size()];
        bfs(goal);
        cur[i] = c;
        hops[i] = dist[c];
        good[i] = 0;
        int nn = 0, ng = 0, gbest = -1;
        float gd = 1e9f;
        for (int k = 0; k < 4; ++k) {
          const int s = stencil(dxs[k], dys[k]);
          if (!((e[c] >> s) & 1u)) continue;
          const int b = c + dxs[k] + dys[k] * G;
          ++nn;
          if (dist[b] < dist[c]) {
            good[i] |= 1u << s;
            ++ng;
          }
          const float ddx = (float)(b % G - goal % G), ddy = (float)(b / G - goal / G), dd = ddx * ddx + ddy * ddy;
          if (dd < gd) {
            gd = dd;
            gbest = s;
          }
        }
        geo[i] = gbest;
        p_rand[i] = (float)ng / (float)nn;
        for (int j = 0; j < N; ++j) {
          uint8_t* fj = f + (size_t)j * F;
          std::memset(fj, 0, F);
          fj[0] = val[j] ? 1 : 0;
          fj[1] = (uint8_t)(128 + 8 * (j % G - c % G));
          fj[2] = (uint8_t)(128 + 8 * (j / G - c / G));
          fj[3] = j == goal ? 255 : 0;
        }
        break;
      }
      glob[(size_t)i * GLOB] = Gn(rng);
      glob[(size_t)i * GLOB + 1] = Gn(rng);
    }
  }
};

struct Dev {
  uint8_t* feat;
  uint32_t* edge;
  int *cur, *done;
  float *glob, *act, *rew;
  explicit Dev(int n) {
    CK(cudaMalloc(&feat, (size_t)n * N * F));
    CK(cudaMalloc(&edge, (size_t)n * N * 4));
    CK(cudaMalloc(&cur, (size_t)n * 4));
    CK(cudaMalloc(&done, (size_t)n * 4));
    CK(cudaMalloc(&glob, sizeof(float) * n * GLOB));
    CK(cudaMalloc(&act, sizeof(float) * n * ADIM));
    CK(cudaMalloc(&rew, sizeof(float) * n));
  }
  ~Dev() {
    for (void* p : std::initializer_list<void*>{feat, edge, cur, done, glob, act, rew}) cudaFree(p);
  }
  void up(const Toy& t) {
    CK(cudaMemcpy(feat, t.feat.data(), t.feat.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(edge, t.edge.data(), t.edge.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cur, t.cur.data(), t.cur.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(glob, t.glob.data(), sizeof(float) * t.glob.size(), cudaMemcpyHostToDevice));
  }
  xppo::GraphObs obs() const { return {feat, edge, cur, glob}; }
};

struct Result {
  double acc = 0, acc_rand = 0, acc_geo = 0;
  double hop_ok[4] = {0, 0, 0, 0}, hop_n[4] = {0, 0, 0, 0};   // 2, 3, 4, 5 홉 이상
  long bad = 0;
  std::vector<float> stats;   // 반복마다 손실 통계 (결정성 비교)
};

Result train(int iters, int n, uint64_t seed, bool share, const std::string& save, bool verbose) {
  static_assert(mdp::act::DIM == ADIM, "행동 칸 수");
  xppo::PPOConfig cfg;
  cfg.num_envs = n;
  cfg.steps_per_env = 8;
  cfg.minibatches = 4;
  cfg.gamma = 0.f;   // 판 길이 1
  xppo::PPO ppo(cfg, toy_spec(share), seed);
  Toy env(n, seed * 7 + 3);
  Dev d(n);
  std::vector<float> act((size_t)n * ADIM), rew(n);
  std::vector<int> done(n, 1);
  CK(cudaMemcpy(d.done, done.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  Result R;
  env.sample();
  d.up(env);
  if (verbose) std::printf("변수 %ld 개 (가치 몸통 %s), 환경 %d × 걸음 %d\n", ppo.ac().n_params, share ? "공유" : "따로", n, cfg.steps_per_env);
  for (int it = 0; it < iters; ++it) {
    double hit = 0, pr = 0;
    for (int t = 0; t < cfg.steps_per_env; ++t) {
      ppo.act(d.obs(), d.act);
      CK(cudaMemcpy(act.data(), d.act, sizeof(float) * act.size(), cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        const int a = (int)act[(size_t)i * ADIM];
        if (a < 0 || a >= NB || !((env.edge[(size_t)i * N + env.cur[i]] >> a) & 1u)) ++R.bad;
        rew[i] = (a >= 0 && a < NB && ((env.good[i] >> a) & 1u)) ? 1.f : 0.f;
        hit += rew[i];
        pr += env.p_rand[i];
      }
      CK(cudaMemcpy(d.rew, rew.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
      ppo.store(d.rew, d.done);
      env.sample();
      d.up(env);
    }
    const auto us = ppo.update(d.obs());
    for (float v : {us.surrogate, us.value, us.entropy, us.kl, us.lr, us.grad_norm}) R.stats.push_back(v);
    if (verbose && (it % 20 == 0 || it == iters - 1))
      std::printf("반복 %3d: 학습 정확도 %.3f (무작위 %.3f), 엔트로피 %.3f, kl %.4f, value %.4f, 기울기 노름 %.3g, lr %.2e\n", it,
                  hit / (n * cfg.steps_per_env), pr / (n * cfg.steps_per_env), us.entropy, us.kl, us.value, us.grad_norm, us.lr);
  }
  // 결정 평가 (새 그래프) + 평가 샘플의 가림 검사
  Toy ev(n, seed * 13 + 5);
  double tot = 0;
  for (int r = 0; r < (4096 + n - 1) / n; ++r) {
    ev.sample();
    d.up(ev);
    ppo.act_inference(d.obs(), d.act, n);
    CK(cudaMemcpy(act.data(), d.act, sizeof(float) * act.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      const int a = (int)act[(size_t)i * ADIM];
      const bool ok = a >= 0 && a < NB && ((ev.good[i] >> a) & 1u);
      const int hb = std::min(ev.hops[i], 5) - 2;
      R.acc += ok;
      R.hop_ok[hb] += ok;
      R.hop_n[hb] += 1;
      R.acc_rand += ev.p_rand[i];
      R.acc_geo += (ev.good[i] >> ev.geo[i]) & 1u;
      tot += 1;
    }
    ppo.act_inference(d.obs(), d.act, n, r);
    CK(cudaMemcpy(act.data(), d.act, sizeof(float) * act.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      const int a = (int)act[(size_t)i * ADIM];
      if (a < 0 || a >= NB || !((ev.edge[(size_t)i * N + ev.cur[i]] >> a) & 1u)) ++R.bad;
    }
  }
  R.acc /= tot;
  R.acc_rand /= tot;
  R.acc_geo /= tot;
  if (!save.empty()) ppo.save(save);
  return R;
}
}  // namespace

int main(int argc, char** argv) {
  int iters = 300, n = 256;
  uint64_t seed = 1;
  bool det = false, share = false, iters_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) { iters = std::atoi(argv[++i]); iters_set = true; }
    else if (a == "--envs" && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--check-determinism") det = true;
    else if (a == "--share") share = true;
  }
  if (det) {
    if (!iters_set) iters = 20;
    const std::string pa = "toy_graph_a.bin", pb = "toy_graph_b.bin";   // 지금 폴더에 잠깐
    const Result ra = train(iters, n, seed, share, pa, false), rb = train(iters, n, seed, share, pb, false);
    FILE *fa = std::fopen(pa.c_str(), "rb"), *fb = std::fopen(pb.c_str(), "rb");
    std::vector<char> A, B;
    for (int c; (c = std::fgetc(fa)) != EOF;) A.push_back((char)c);
    for (int c; (c = std::fgetc(fb)) != EOF;) B.push_back((char)c);
    std::fclose(fa);
    std::fclose(fb);
    const bool same_w = A == B && !A.empty();
    const bool same_l = ra.stats.size() == rb.stats.size() && !ra.stats.empty() &&
                        std::memcmp(ra.stats.data(), rb.stats.data(), ra.stats.size() * sizeof(float)) == 0;
    std::printf("결정성 (같은 씨앗 %d 반복 두 번, 가치 몸통 %s): 손실 통계 %zu 개 %s, 체크포인트 %zu 바이트 %s — %s\n", iters, share ? "공유" : "따로",
                ra.stats.size(), same_l ? "비트 같음" : "다름", A.size(), same_w ? "비트 같음" : "다름", same_w && same_l ? "통과" : "실패");
    std::remove(pa.c_str());
    std::remove(pb.c_str());
    return same_w && same_l ? 0 : 1;
  }
  const Result R = train(iters, n, seed, share, "", true);
  std::printf("결정 정확도 %.3f  (무작위 %.3f, 기하 기준 %.3f)\n", R.acc, R.acc_rand, R.acc_geo);
  std::printf("홉 수별: 2 홉 %.3f (%0.f)  3 홉 %.3f (%.0f)  4 홉 %.3f (%.0f)  5 홉 이상 %.3f (%.0f)\n", R.hop_ok[0] / R.hop_n[0], R.hop_n[0],
              R.hop_ok[1] / R.hop_n[1], R.hop_n[1], R.hop_ok[2] / R.hop_n[2], R.hop_n[2], R.hop_ok[3] / R.hop_n[3], R.hop_n[3]);
  const bool ok = R.acc > 0.9 && R.bad == 0;
  std::printf("가린 칸 고름 %ld — %s (기준: 결정 정확도 > 0.9, 가린 칸 0)\n", R.bad, ok ? "통과" : "실패");
  return ok ? 0 : 1;
}
