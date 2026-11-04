// RT 코어 빠른 길(하드웨어 삼각형) vs 정확한 길(사용자 정의 프리미티브 + ray_tri) 통계 동등성 (include/optix_rt.h).
// 통과 기준은 결과를 보기 전에 정한 값 (아래 kMax*). 바꾸면 까닭을 적는다.
//   1. 광선: 같은 상태에서 만든 같은 광선(깊이 가운데 픽셀 · 라이다 가운데 빔 · 라이다 옆 광선 4 개, 사람 원기둥 포함)을 두 길로 쏴
//      다른 삼각형을 고른 비율 ≤ kMaxDiffTri, 거리 차가 실물 라이다 표면 잡음 σ(4.7 mm, analysis/real_limo.md) 를 넘는 비율 ≤ kMaxOverSigma.
//      다른 곳이 어디인지: 두 삼각형이 꼭짓점 2 개(모서리)·1 개를 같이 쓰나, 따로인가.
//   2. 롤아웃: 같은 씨앗·같은 무작위 행동열로 Env 둘(정확·빠름)을 돌려(한 광선이라도 다르면 길이 갈라지므로 통계만) 걸음당 보상 평균·항마다 평균·
//      끝 덮음·판 끝 종류 비율이 표준오차 3 배 안인지.
// rt_fastcheck --list 목록 [--envs 2048] [--ray-steps 10] [--roll-steps 300] [--people 4]
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "env_api.h"
#include "optix_rt.h"

namespace {
constexpr double kMaxDiffTri = 1e-4;     // 다른 삼각형을 고른 광선 비율
constexpr double kMaxOverSigma = 1e-5;   // |거리 차| > 4.7 mm 인 광선 비율
constexpr double kSigma = 0.0047;        // m — 실물 라이다 표면 빔 σ 중앙 (sensor_model.yaml lidar.sigma 근거)
constexpr double kZ = 3.0;               // 롤아웃 통계: |평균 차| ≤ 3 × 합친 표준오차

struct Ray {
  sim::V3 o, d;
  float tmax;
  int map, tri0;
};

// 사람 원기둥까지 (trav_init 과 같은 best)
Ray make_ray(const sim::EnvState& e, sim::V3 o, sim::V3 d, float tmax) {
  Ray r{o, d, tmax, e.map, -1};
  for (int k = 0; k < e.people.n; ++k) {
    const float t = sim::cyl_hit(e.people.cyl[k], o.x, o.y, o.z, d.x, d.y, d.z);
    if (t < r.tmax) { r.tmax = t; r.tri0 = -2 - k; }
  }
  return r;
}

struct Res {
  std::vector<float> t;
  std::vector<int> tri;
};

Res trace(sim::OptixRt& rt, const std::vector<Ray>& rays, int fast) {
  const int n = (int)rays.size();
  std::vector<float4> ho(n), hd(n);
  std::vector<int> h0(n);
  for (int i = 0; i < n; ++i) {
    ho[i] = make_float4(rays[i].o.x, rays[i].o.y, rays[i].o.z, rays[i].tmax);
    int m = rays[i].map;
    float mf;
    std::memcpy(&mf, &m, 4);
    hd[i] = make_float4(rays[i].d.x, rays[i].d.y, rays[i].d.z, mf);
    h0[i] = rays[i].tri0;
  }
  float4 *o, *d;
  int *t0, *tri;
  float* t;
  cudaMalloc(&o, 16 * n);
  cudaMalloc(&d, 16 * n);
  cudaMalloc(&t0, 4 * n);
  cudaMalloc(&tri, 4 * n);
  cudaMalloc(&t, 4 * n);
  cudaMemcpy(o, ho.data(), 16 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d, hd.data(), 16 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(t0, h0.data(), 4 * n, cudaMemcpyHostToDevice);
  sim::RtParams p{};
  p.o = o;
  p.d = d;
  p.tri0 = t0;
  p.out_t = t;
  p.out_tri = tri;
  p.n = n;
  p.fast = fast;
  rt.launch(p, 0);
  cudaDeviceSynchronize();
  Res r;
  r.t.resize(n);
  r.tri.resize(n);
  cudaMemcpy(r.t.data(), t, 4 * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(r.tri.data(), tri, 4 * n, cudaMemcpyDeviceToHost);
  for (void* q : {(void*)o, (void*)d, (void*)t0, (void*)tri, (void*)t}) cudaFree(q);
  return r;
}

int shared_vertices(const sim::Tri& a, const sim::Tri& b) {
  const sim::V3 va[3] = {a.a, a.b, a.c}, vb[3] = {b.a, b.b, b.c};
  int n = 0;
  for (auto& x : va)
    for (auto& y : vb)
      if (x.x == y.x && x.y == y.y && x.z == y.z) { ++n; break; }
  return n;
}

struct Stat {   // 평균·표준오차
  double s = 0, s2 = 0;
  long n = 0;
  void add(double x) { s += x; s2 += x * x; ++n; }
  double mean() const { return n ? s / n : 0; }
  double se() const { return n > 1 ? std::sqrt(std::max(0.0, (s2 / n - mean() * mean())) / (n - 1)) : 0; }
};
}  // namespace

int main(int argc, char** argv) {
  std::string list;
  int n_env = 2048, ray_steps = 10, roll_steps = 300, people = 4;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--list") list = argv[i + 1];
    else if (a == "--envs") n_env = std::stoi(argv[i + 1]);
    else if (a == "--ray-steps") ray_steps = std::stoi(argv[i + 1]);
    else if (a == "--roll-steps") roll_steps = std::stoi(argv[i + 1]);
    else if (a == "--people") people = std::stoi(argv[i + 1]);
  }
  sim::EnvHostCfg hc = sim::load_env_cfg(SIM_CONFIG_DIR "/env.yaml");
  hc.cfg.curriculum = 0;   // 사람 수를 --people 로 (모든 맵 고르게)
  const sim::LidarModel lidar = sim::load_lidar_model(SIM_CONFIG_DIR "/sensor_model.yaml");
  std::vector<sim::World> worlds = sim::load_scene_list(list, hc, nullptr);
  bool pass = true;

  // ---- 1. 광선
  {
    setenv("SIM_EXACT", "1", 1);
    sim::Env env(hc, worlds, lidar, n_env, 11, true);
    env.set_people_max(std::vector<int>(n_env, people));
    env.reset();
    auto* rt = (sim::OptixRt*)env.batch().rt;
    if (!rt) { std::fprintf(stderr, "RT 코어 경로 없음 (SIM_OPTIX 빌드·OptiX 드라이버)\n"); return 2; }
    const sim::EnvCfg& c = hc.cfg;
    float* d_act;
    cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * n_env);
    uint64_t rng = 3;
    long n_ray = 0, n_difftri = 0, n_over = 0, n_over1mm = 0, by_kind[3] = {0, 0, 0}, by_type[3][2] = {};
    double max_dt = 0;
    std::vector<double> dts;
    const char* type_name[3] = {"깊이 가운데", "라이다 가운데", "라이다 옆"};
    for (int s = 0; s < ray_steps; ++s) {
      std::vector<float> act((size_t)mdp::act::DIM * n_env);
      for (int i = 0; i < n_env; ++i) sim::random_action(c, rng, &act[(size_t)mdp::act::DIM * i]);
      cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
      env.step(d_act);
      env.sync();
      std::vector<sim::EnvState> st;
      env.get_state(st);
      std::vector<Ray> rays;
      std::vector<int> type;
      for (int e = 0; e < n_env; ++e) {
        const sim::EnvState& E = st[e];
        // 깊이 가운데
        const sim::Pose cam = mdp::obsgen::depth_camera(c, E);
        for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
          float u, v;
          mdp::obsgen::depth_uv(c.depth, i, &u, &v);
          if (!sim::depth_valid(c.depth.spec, u, v)) continue;
          rays.push_back(make_ray(E, cam.p, sim::depth_dir(c.depth.spec, cam, u, v), sim::depth_tmax(c.depth.spec)));
          type.push_back(0);
        }
        // 라이다: 지금 자세로 장 하나 준비 (scan_begin 은 상태를 안 바꿈)
        sim::Shared sh;
        sim::scan_begin(env.batch(), E, sh);
        for (int j = 0; j < sh.n_beams; ++j) {
          sim::BeamGeom g;
          sim::beam_center_ray(E.noise, sh.scan, j, g);
          rays.push_back(make_ray(E, g.o, g.d, E.noise.max_range));
          type.push_back(1);
          for (int k = 0; k < 4; ++k) {
            rays.push_back(make_ray(E, g.o, sim::beam_side_dir(E.noise, sh.scan, g, k), E.noise.max_range));
            type.push_back(2);
          }
        }
      }
      const Res ex = trace(*rt, rays, 0), fa = trace(*rt, rays, 1);
      for (size_t i = 0; i < rays.size(); ++i) {
        ++n_ray;
        by_type[type[i]][0]++;
        const float a = ex.t[i], b = fa.t[i];
        const bool hit_a = ex.tri[i] != -1, hit_b = fa.tri[i] != -1;
        double dt = 0;
        if (hit_a != hit_b) dt = 1e9;   // 한쪽만 맞음
        else if (hit_a) dt = std::fabs((double)a - (double)b);
        if (ex.tri[i] != fa.tri[i]) {
          ++n_difftri;
          by_type[type[i]][1]++;
          if (ex.tri[i] >= 0 && fa.tri[i] >= 0) {
            const auto& tris = worlds[rays[i].map].scene.tris;
            const int sv = shared_vertices(tris[ex.tri[i]], tris[fa.tri[i]]);
            by_kind[sv >= 2 ? 0 : (sv == 1 ? 1 : 2)]++;
          } else {
            by_kind[2]++;
          }
          dts.push_back(dt);
        }
        if (dt > kSigma) ++n_over;
        if (dt > 0.001) ++n_over1mm;
        if (dt < 1e8) max_dt = std::max(max_dt, dt);
      }
    }
    std::sort(dts.begin(), dts.end());
    auto pct = [&](double q) { return dts.empty() ? 0.0 : dts[std::min(dts.size() - 1, (size_t)(q * dts.size()))]; };
    const double f_diff = (double)n_difftri / n_ray, f_over = (double)n_over / n_ray;
    std::printf("광선 %ld 개 (환경 %d × %d 걸음): 다른 삼각형 %ld (%.2e, 기준 ≤ %.0e) — 모서리 공유 %ld · 꼭짓점 공유 %ld · 따로(또는 한쪽 못 맞음) %ld\n", n_ray,
                n_env, ray_steps, n_difftri, f_diff, kMaxDiffTri, by_kind[0], by_kind[1], by_kind[2]);
    for (int k = 0; k < 3; ++k)
      std::printf("  %s: %ld 개 중 다름 %ld (%.2e)\n", type_name[k], by_type[k][0], by_type[k][1], by_type[k][0] ? (double)by_type[k][1] / by_type[k][0] : 0.0);
    std::printf("  다른 광선의 |거리 차|: 중앙 %.3g m, 90 %% %.3g m, 99 %% %.3g m, 최대(둘 다 맞음) %.3g m\n", pct(0.5), pct(0.9), pct(0.99), max_dt);
    std::printf("  |거리 차| > 1 mm %ld (%.2e), > σ 4.7 mm %ld (%.2e, 기준 ≤ %.0e)\n", n_over1mm, (double)n_over1mm / n_ray, n_over, f_over, kMaxOverSigma);
    const bool ok = f_diff <= kMaxDiffTri && f_over <= kMaxOverSigma;
    std::printf("광선 동등성 %s\n", ok ? "통과" : "실패");
    pass = pass && ok;
    cudaFree(d_act);
  }

  // ---- 2. 롤아웃 (Env 둘, 같은 씨앗·행동열)
  {
    struct Run {
      Stat rew, cov;
      std::vector<Stat> term;
      long done[mdp::cur::kNumDone] = {};
      long n_ep = 0;
    };
    Run runs[2];
    for (int fast = 0; fast < 2; ++fast) {
      setenv("SIM_EXACT", fast ? "0" : "1", 1);
      sim::Env env(hc, worlds, lidar, n_env, 23, true);
      env.set_people_max(std::vector<int>(n_env, people));
      env.reset();
      float* d_act;
      cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * n_env);
      uint64_t rng = 7;
      Run& R = runs[fast];
      R.term.resize(mdp::rew::kTerms);
      std::vector<float> r, tm;
      std::vector<int32_t> d;
      std::vector<double> env_rew(n_env, 0.0);
      std::vector<std::vector<double>> env_term(n_env, std::vector<double>(mdp::rew::kTerms, 0.0));
      for (int s = 0; s < roll_steps; ++s) {
        std::vector<float> act((size_t)mdp::act::DIM * n_env);
        for (int i = 0; i < n_env; ++i) sim::random_action(hc.cfg, rng, &act[(size_t)mdp::act::DIM * i], 0.003f);
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
        env.step(d_act);
        env.sync();
        env.get_reward(r, d);
        env.get_terms(tm);
        for (int i = 0; i < n_env; ++i) {
          env_rew[i] += r[i];
          for (int k = 0; k < mdp::rew::kTerms; ++k) env_term[i][k] += tm[(size_t)i * mdp::rew::kTerms + k];
          if (d[i]) { R.done[d[i]]++; R.n_ep++; }
        }
      }
      std::vector<sim::EnvState> st;
      env.get_state(st);
      for (int i = 0; i < n_env; ++i) {   // 환경마다 걸음당 평균 = 표본 하나 (환경끼리 독립)
        R.rew.add(env_rew[i] / roll_steps);
        for (int k = 0; k < mdp::rew::kTerms; ++k) R.term[k].add(env_term[i][k] / roll_steps);
        R.cov.add(st[i].coverage);
      }
      cudaFree(d_act);
    }
    bool ok = true;
    auto cmp = [&](const char* name, const Stat& a, const Stat& b) {
      const double se = std::sqrt(a.se() * a.se() + b.se() * b.se()), diff = b.mean() - a.mean();
      const bool good = std::fabs(diff) <= kZ * se + 1e-12;
      ok = ok && good;
      std::printf("  %-14s 정확 %+.5f · 빠름 %+.5f · 차 %+.5f (표준오차 %.5f, %.2f σ) %s\n", name, a.mean(), b.mean(), diff, se, se > 0 ? diff / se : 0.0,
                  good ? "" : "← 넘음");
    };
    std::printf("롤아웃 (환경 %d × %d 걸음, 사람 0~%d, 같은 씨앗·행동열, 기준 |차| ≤ %.0f × 표준오차):\n", n_env, roll_steps, people, kZ);
    cmp("걸음당 보상", runs[0].rew, runs[1].rew);
    for (int k = 0; k < mdp::rew::kTerms; ++k) cmp(mdp::rew::term_name(k), runs[0].term[k], runs[1].term[k]);
    cmp("끝 덮음", runs[0].cov, runs[1].cov);
    for (int k = 1; k < mdp::cur::kNumDone; ++k) {   // 판 끝 종류 비율 (끝난 판 중)
      const double n0 = std::max<long>(1, runs[0].n_ep), n1 = std::max<long>(1, runs[1].n_ep);
      const double p0 = runs[0].done[k] / n0, p1 = runs[1].done[k] / n1, p = (runs[0].done[k] + runs[1].done[k]) / (n0 + n1);
      const double se = std::sqrt(std::max(1e-12, p * (1 - p) * (1 / n0 + 1 / n1)));
      const bool good = std::fabs(p1 - p0) <= kZ * se;
      ok = ok && good;
      std::printf("  끝 종류 %d      정확 %.4f (%ld) · 빠름 %.4f (%ld) · %.2f σ %s\n", k, p0, runs[0].done[k], p1, runs[1].done[k], (p1 - p0) / se,
                  good ? "" : "← 넘음");
    }
    std::printf("판 수: 정확 %ld · 빠름 %ld\n롤아웃 동등성 %s\n", runs[0].n_ep, runs[1].n_ep, ok ? "통과" : "실패");
    pass = pass && ok;
  }
  return pass ? 0 : 1;
}
