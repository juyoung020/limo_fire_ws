// 장난감 GPU 환경(점 로봇 2D 목표 도달)에서 PPO 학습 — S5 검증 (학습·같은 씨앗·체크포인트 이어 학습).
//
// 환경: 관측 크기는 obs_layout.h 그대로(대부분 0). 로봇은 옆으로도 가는 점(v → x, vy → y, ω 무시), 0.1 s 걸음, 최고 1 m/s.
//   목표는 시작점에서 0.5~2.5 m. 1D 의 FRONTIER 칸에 목표 상대 위치 (dx, dy, 거리), 가까운 지도 0 번 채널에 목표 점 3×3.
//   보상 = 2·(가까워진 거리) − 0.01/걸음.
//   선언: 목표 0.25 m 안이면 +5·성공·종료(done=1), 밖이면 −1·실패·종료(done=1). 100 걸음이면 끊김(done=3, 실패로 셈).
//
//   toy_train [--envs N] [--iters I] [--seed S]               학습, 반복마다 곡선 한 줄, 끝에 결정적 평가
//   toy_train --check-determinism [--iters I]                 같은 씨앗 두 번 → 손실·가중치가 비트 단위로 같은지
//   toy_train --check-resume [--iters I] [--ckpt 경로]         I 번 연속 vs (I/2 번 → 저장 → 새로 만들어 불러옴 → I/2 번) 비교
//   추가: --eval-every K (K 반복마다 결정적 평가), --lr-max L (KL 적응 학습률 상한, 기본 1e-2 = rsl_rl)
#include <curand_kernel.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../../mdp/obs_layout.h"
#include "ppo.h"

using namespace xnet;
namespace o = mdp::obs;

namespace {
constexpr int TMAX = 100;
constexpr float DT = 0.1f, VMAX = 1.0f, R_GOAL = 0.25f;
constexpr int V0 = o::LIDAR + o::INTENSITY;  // VEC 시작

struct EnvState {  // GPU 배열, 환경마다
  float *x, *y, *gx, *gy;
  int *t, *resets;
  int *n_done, *n_succ, *sum_len;  // 반복 통계 (host 가 읽고 0 으로)
};

__device__ void reset_env(int i, uint64_t seed, EnvState s) {
  curandStatePhilox4_32_10_t rs;
  curand_init(seed, (uint64_t)i, (uint64_t)s.resets[i] * 4ull, &rs);
  float4 u = curand_uniform4(&rs);
  s.x[i] = 4.f * u.x - 2.f;
  s.y[i] = 4.f * u.y - 2.f;
  float r = 0.5f + 2.0f * u.z, th = 6.2831853f * u.w;
  s.gx[i] = s.x[i] + r * cosf(th);
  s.gy[i] = s.y[i] + r * sinf(th);
  s.t[i] = 0;
  s.resets[i] += 1;
}

__global__ void k_reset_all(int n, uint64_t seed, EnvState s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) reset_env(i, seed, s);
}

__global__ void k_step(int n, uint64_t seed, EnvState s, const float* act, float* rew, int* done) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* a = act + i * mdp::act::DIM;
  float ax = fminf(fmaxf(a[0], -1.f), 1.f), ay = fminf(fmaxf(a[1], -1.f), 1.f);
  bool declare = a[3] > 0.5f;
  float dx0 = s.gx[i] - s.x[i], dy0 = s.gy[i] - s.y[i];
  float d0 = sqrtf(dx0 * dx0 + dy0 * dy0);
  float r = -0.01f;
  int d = 0;
  bool succ = false;
  if (declare) {
    succ = d0 < R_GOAL;
    r += succ ? 5.f : -1.f;
    d = 1;
  } else {
    s.x[i] += DT * VMAX * ax;
    s.y[i] += DT * VMAX * ay;
    float dx = s.gx[i] - s.x[i], dy = s.gy[i] - s.y[i];
    r += 2.f * (d0 - sqrtf(dx * dx + dy * dy));
    if (s.t[i] + 1 >= TMAX) d = 3;
  }
  s.t[i] += 1;
  rew[i] = r;
  done[i] = d;
  if (d) {
    s.n_done[i] += 1;
    s.n_succ[i] += succ ? 1 : 0;
    s.sum_len[i] += s.t[i];
    reset_env(i, seed, s);
  }
}

// 관측: 이미지는 미리 0 으로 지운 뒤 목표 점만
__global__ void k_obs(int n, EnvState s, uint8_t* img, float* o1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float dx = s.gx[i] - s.x[i], dy = s.gy[i] - s.y[i];
  float dist = sqrtf(dx * dx + dy * dy);
  float* v = o1 + (long)i * o::POLICY_1D + V0;
  v[o::FRONTIER + 0] = dx;
  v[o::FRONTIER + 1] = dy;
  v[o::FRONTIER + 2] = dist;
  int cx = (int)floorf(dx / 0.1f) + o::NEAR_W / 2, cy = (int)floorf(dy / 0.1f) + o::NEAR_H / 2;
  uint8_t* im = img + (long)i * o::IMG_SIZE;  // 가까운 지도 0 번 채널
  for (int yy = cy - 1; yy <= cy + 1; ++yy)
    for (int xx = cx - 1; xx <= cx + 1; ++xx)
      if (yy >= 0 && yy < o::NEAR_H && xx >= 0 && xx < o::NEAR_W) im[yy * o::NEAR_W + xx] = 255;
}

struct ToyEnv {
  int n;
  uint64_t seed;
  EnvState s;
  uint8_t* img;
  float *o1, *rew;
  int* done;
  std::vector<void*> all;
  template <typename T>
  T* mk(long k) {
    T* p;
    CK(cudaMalloc(&p, k * sizeof(T)));
    CK(cudaMemset(p, 0, k * sizeof(T)));
    all.push_back(p);
    return p;
  }
  ToyEnv(int n_, uint64_t seed_) : n(n_), seed(seed_) {
    s.x = mk<float>(n), s.y = mk<float>(n), s.gx = mk<float>(n), s.gy = mk<float>(n);
    s.t = mk<int>(n), s.resets = mk<int>(n), s.n_done = mk<int>(n), s.n_succ = mk<int>(n), s.sum_len = mk<int>(n);
    img = mk<uint8_t>((long)n * o::IMG_SIZE);
    o1 = mk<float>((long)n * o::POLICY_1D);
    rew = mk<float>(n);
    done = mk<int>(n);
    k_reset_all<<<nblk(n, 128), 128>>>(n, seed, s);
    observe();
  }
  ~ToyEnv() {
    for (void* p : all) cudaFree(p);
  }
  void observe() {
    CK(cudaMemset(img, 0, (size_t)n * o::IMG_SIZE));
    k_obs<<<nblk(n, 128), 128>>>(n, s, img, o1);
  }
  void step(const float* act) {
    k_step<<<nblk(n, 128), 128>>>(n, seed, s, act, rew, done);
    observe();
  }
  // 반복 통계 꺼내고 0 으로
  void take_stats(long& nd, long& ns, long& len) {
    std::vector<int> a(n), b(n), c(n);
    CK(cudaMemcpy(a.data(), s.n_done, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), s.n_succ, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), s.sum_len, n * 4, cudaMemcpyDeviceToHost));
    nd = ns = len = 0;
    for (int i = 0; i < n; ++i) nd += a[i], ns += b[i], len += c[i];
    CK(cudaMemset(s.n_done, 0, n * 4));
    CK(cudaMemset(s.n_succ, 0, n * 4));
    CK(cudaMemset(s.sum_len, 0, n * 4));
  }
  // 환경 상태 저장·불러오기 (이어 학습 검사용 — 체크포인트는 PPO 것, 이건 장난감 환경 것)
  void save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    float* fs[] = {s.x, s.y, s.gx, s.gy};
    int* is[] = {s.t, s.resets};
    std::vector<char> buf(n * 4);
    for (auto p : fs) {
      CK(cudaMemcpy(buf.data(), p, n * 4, cudaMemcpyDeviceToHost));
      f.write(buf.data(), n * 4);
    }
    for (auto p : is) {
      CK(cudaMemcpy(buf.data(), p, n * 4, cudaMemcpyDeviceToHost));
      f.write(buf.data(), n * 4);
    }
  }
  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    float* fs[] = {s.x, s.y, s.gx, s.gy};
    int* is[] = {s.t, s.resets};
    std::vector<char> buf(n * 4);
    for (auto p : fs) {
      f.read(buf.data(), n * 4);
      CK(cudaMemcpy(p, buf.data(), n * 4, cudaMemcpyHostToDevice));
    }
    for (auto p : is) {
      f.read(buf.data(), n * 4);
      CK(cudaMemcpy(p, buf.data(), n * 4, cudaMemcpyHostToDevice));
    }
    if (!f) throw std::runtime_error("환경 상태 읽기 실패");
    observe();
  }
};

struct IterLog {
  int it;
  xppo::UpdateStats st;
  double succ_rate, mean_len, rew_per_step, sec;
  long n_ep;
};

// 반복 하나 (롤아웃 + 갱신)
IterLog one_iter(xppo::PPO& ppo, ToyEnv& env, float* d_act) {
  auto t0 = std::chrono::steady_clock::now();
  int T = ppo.config().steps_per_env;
  double rsum = 0;
  for (int t = 0; t < T; ++t) {
    ppo.act(env.img, env.o1, d_act);
    env.step(d_act);
    ppo.store(env.rew, env.done);
    rsum += sum_all(env.rew, env.n, 0);
  }
  IterLog L;
  L.it = ppo.iteration();
  L.st = ppo.update(env.img, env.o1);
  long nd, ns, len;
  env.take_stats(nd, ns, len);
  L.n_ep = nd;
  L.succ_rate = nd ? (double)ns / nd : 0;
  L.mean_len = nd ? (double)len / nd : 0;
  L.rew_per_step = rsum / ((double)T * env.n);
  CK(cudaDeviceSynchronize());
  L.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return L;
}

void print_head() {
  std::printf("%5s %8s %6s %7s %9s %10s %9s %8s %8s %7s %6s %8s %7s %7s\n", "반복", "성공률", "판수", "판길이", "보상/걸음",
              "surrogate", "value", "kl", "lr", "σ(v)", "초", "|g|", "V평균", "R평균");
}
void print_iter(const IterLog& L, xppo::PPO& ppo) {
  float ls;
  CK(cudaMemcpy(&ls, ppo.ac().log_std(), 4, cudaMemcpyDeviceToHost));
  std::printf("%5d %7.1f%% %6ld %7.1f %9.4f %10.6f %9.5f %8.5f %8.2e %7.3f %6.2f %8.3f %7.3f %7.3f\n", L.it, 100 * L.succ_rate, L.n_ep,
              L.mean_len, L.rew_per_step, L.st.surrogate, L.st.value, L.st.kl, L.st.lr, std::exp(ls), L.sec, L.st.grad_norm,
              L.st.mean_value, L.st.mean_return);
  std::fflush(stdout);
}

// 결정적 평가: 환경마다 첫 판 결과
void evaluate(xppo::PPO& ppo, int n, uint64_t seed) {
  ToyEnv env(n, seed);
  float* d_act;
  CK(cudaMalloc(&d_act, (size_t)n * mdp::act::DIM * 4));
  std::vector<int> first(n, -1), hd(n), hlen(n, 0);
  std::vector<float> hr(n);
  for (int t = 0; t < TMAX; ++t) {
    ppo.act_inference(env.img, env.o1, d_act, n);
    env.step(d_act);
    CK(cudaMemcpy(hd.data(), env.done, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(), env.rew, n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      if (first[i] < 0 && hd[i]) {
        first[i] = (hd[i] == 1 && hr[i] > 1.f) ? 1 : (hd[i] == 1 ? 0 : 2);  // 1 성공, 0 일찍 선언, 2 끊김
        hlen[i] = t + 1;
      }
  }
  int ns = 0, nf = 0, nt = 0;
  double len = 0;
  for (int i = 0; i < n; ++i) {
    ns += first[i] == 1, nf += first[i] == 0, nt += first[i] == 2 || first[i] < 0;
    if (first[i] == 1) len += hlen[i];
  }
  std::printf("결정적 평가 (판 %d 개, 평균 행동·선언 확률 > 0.5): 성공 %.1f%% (%d), 일찍 선언 %d, 끊김 %d, 성공 판 평균 %.1f 걸음\n", n,
              100.0 * ns / n, ns, nf, nt, ns ? len / ns : 0.0);
  cudaFree(d_act);
}

uint64_t param_hash(xppo::PPO& ppo) {
  long P = ppo.ac().n_params;
  std::vector<uint32_t> h(P);
  CK(cudaMemcpy(h.data(), ppo.ac().d_params, P * 4, cudaMemcpyDeviceToHost));
  uint64_t x = 1469598103934665603ull;
  for (uint32_t v : h) x = (x ^ v) * 1099511628211ull;
  return x;
}
}  // namespace

int main(int argc, char** argv) {
  int envs = 1024, iters = 150, eval_every = 0;
  float lr_max = 1e-2f;
  int mlp = 2;
  uint64_t seed = 1;
  std::string mode = "train", ckpt = "toy_ckpt.bin";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--envs") envs = std::stoi(nxt());
    else if (a == "--iters") iters = std::stoi(nxt());
    else if (a == "--seed") seed = std::stoull(nxt());
    else if (a == "--ckpt") ckpt = nxt();
    else if (a == "--lr-max") lr_max = std::stof(nxt());
    else if (a == "--mlp") mlp = std::stoi(nxt());
    else if (a == "--eval-every") eval_every = std::stoi(nxt());
    else if (a == "--check-determinism") mode = "det";
    else if (a == "--check-resume") mode = "resume";
    else {
      std::fprintf(stderr, "모르는 인자 %s\n", a.c_str());
      return 2;
    }
  }
  xppo::PPOConfig cfg;
  cfg.num_envs = envs;
  cfg.lr_max = lr_max;
  cfg.mlp = mlp;
  ACSpec spec = explore_spec();
  float* d_act;
  CK(cudaMalloc(&d_act, (size_t)envs * mdp::act::DIM * 4));

  if (mode == "train") {
    std::printf("장난감 목표 도달: 환경 %d, 반복 %d, 씨앗 %llu, 걸음/반복 %d\n", envs, iters, (unsigned long long)seed,
                cfg.steps_per_env);
    xppo::PPO ppo(cfg, spec, seed);
    ToyEnv env(envs, seed + 100);
    print_head();
    for (int it = 0; it < iters; ++it) {
      print_iter(one_iter(ppo, env, d_act), ppo);
      if (eval_every > 0 && (it + 1) % eval_every == 0 && it + 1 < iters) evaluate(ppo, envs, seed + 999);
    }
    evaluate(ppo, envs, seed + 999);
    return 0;
  }
  if (mode == "det") {
    std::vector<std::vector<float>> runs;
    std::vector<uint64_t> hashes;
    for (int r = 0; r < 2; ++r) {
      xppo::PPO ppo(cfg, spec, seed);
      ToyEnv env(envs, seed + 100);
      std::vector<float> v;
      for (int it = 0; it < iters; ++it) {
        auto L = one_iter(ppo, env, d_act);
        v.push_back(L.st.surrogate), v.push_back(L.st.value), v.push_back(L.st.kl);
        std::printf("실행 %d 반복 %d: surrogate %.9g value %.9g kl %.9g\n", r + 1, it, L.st.surrogate, L.st.value, L.st.kl);
      }
      runs.push_back(v);
      hashes.push_back(param_hash(ppo));
      std::printf("실행 %d 가중치 해시 %016llx\n", r + 1, (unsigned long long)hashes.back());
    }
    bool same = std::memcmp(runs[0].data(), runs[1].data(), runs[0].size() * 4) == 0 && hashes[0] == hashes[1];
    std::printf("%s: 같은 씨앗 두 번 → 손실 %zu 개·가중치 %s\n", same ? "통과" : "실패", runs[0].size(),
                same ? "비트 단위로 같음" : "다름");
    return same ? 0 : 1;
  }
  // resume
  int half = iters / 2;
  std::vector<float> a, b;
  uint64_t ha, hb;
  {
    xppo::PPO ppo(cfg, spec, seed);
    ToyEnv env(envs, seed + 100);
    for (int it = 0; it < 2 * half; ++it) {
      auto L = one_iter(ppo, env, d_act);
      if (it >= half) a.push_back(L.st.surrogate), a.push_back(L.st.value), a.push_back(L.st.kl);
    }
    ha = param_hash(ppo);
  }
  {
    xppo::PPO ppo(cfg, spec, seed);
    ToyEnv env(envs, seed + 100);
    for (int it = 0; it < half; ++it) one_iter(ppo, env, d_act);
    ppo.save(ckpt);
    env.save(ckpt + ".env");
    std::printf("반복 %d 에서 저장: %s\n", ppo.iteration(), ckpt.c_str());
  }
  {
    xppo::PPO ppo(cfg, spec, seed + 12345);  // 다른 씨앗으로 만들어도 불러오면 같아야 함
    ToyEnv env(envs, seed + 100);
    ppo.load(ckpt);
    env.load(ckpt + ".env");
    std::printf("불러옴: 반복 %d, 학습률 %.3e\n", ppo.iteration(), ppo.learning_rate());
    for (int it = 0; it < half; ++it) {
      auto L = one_iter(ppo, env, d_act);
      b.push_back(L.st.surrogate), b.push_back(L.st.value), b.push_back(L.st.kl);
      std::printf("이어서 반복 %d: surrogate %.9g (연속 %.9g)  value %.9g (연속 %.9g)\n", L.it, L.st.surrogate,
                  a[3 * it], L.st.value, a[3 * it + 1]);
    }
    hb = param_hash(ppo);
  }
  bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0 && ha == hb;
  std::printf("%s: 연속 %d 반복 vs 저장·불러와 이어 %d+%d 반복 → 손실·가중치 %s (해시 %016llx / %016llx)\n",
              same ? "통과" : "실패", 2 * half, half, half, same ? "비트 단위로 같음" : "다름", (unsigned long long)ha,
              (unsigned long long)hb);
  return same ? 0 : 1;
}
