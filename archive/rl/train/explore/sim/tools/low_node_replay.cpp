// 실물 아래층 관측 다시 돌리기: low_node_check --dump 녹화를 RobotEnv 아래층 모드만으로 (GPU·시뮬 없이) 다시 돌려 기대 관측·명령과 비트 비교하고,
// 한 걸음 시간(low_act · step · low_obs)을 잰다. 젯슨에서도 돈다:
//   g++ -O2 -std=c++17 -ffp-contract=off -I<explore>/sim/include -I<explore>/mdp -I<explore>/sim/tools -I<limo_fire_rl>/include \
//       low_node_replay.cpp <limo_fire_rl>/src/robot_env.cpp <explore>/sim/src/env_cfg.cpp <explore>/sim/src/sensor_model.cpp -lyaml-cpp
//   low_node_replay 녹화.bin [--ckpt 아래층.bin]   (--ckpt: 걸음마다 실물 CPU 정책 정방향 시간도 — 행동은 녹화 값)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "limo_fire_rl/robot_env.h"
#include "low_node_rec.h"
#include "policy_cpu.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "low_node_replay 녹화.bin [--ckpt 아래층.bin]\n");
    return 2;
  }
  std::string ckpt;
  for (int i = 2; i < argc; ++i)
    if (std::string(argv[i]) == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
  try {
    lowrec::Reader rd(argv[1]);
    limo_fire_rl::RobotEnv robot(rd.hc, true);
    if (robot.low_img_n() != rd.ni || robot.low_1d_n() != rd.n1) throw std::runtime_error("녹화 관측 크기가 설정과 다름");
    xnet_cpu::LowPolicy pol;
    if (!ckpt.empty()) pol.load(ckpt);
    std::vector<uint8_t> ei((size_t)rd.ni), dep((size_t)mdp::obs::DEPTH_SIZE);
    std::vector<float> e1((size_t)rd.n1), a((size_t)mdp::low::ACT), out((size_t)mdp::low::ACT);
    limo_fire_rl::Proprio p;
    limo_fire_rl::Scan s;
    long n_obs = 0, n_act = 0, n_step = 0, d_img = 0, d_1d = 0, d_cmd = 0, n_ep = 0;
    int first_bad = -1;
    using clk = std::chrono::steady_clock;
    std::vector<double> t_act, t_step, t_obs, t_pol;
    const auto ms = [](clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); };
    for (char c = rd.next(); c; c = rd.next()) {
      if (c == 'R') {
        float xyz[3], fov;
        lowrec::Window win;
        rd.reset(xyz, &p, &fov, &win);
        robot.reset(xyz[0], xyz[1], xyz[2], p, fov, &win);
        if (!ckpt.empty()) pol.reset();
        ++n_ep;
      } else if (c == 'G') {
        robot.set_goal(rd.goal());
      } else if (c == 'A') {
        float v0, w0, v, w;
        rd.act(a.data(), &v0, &w0);
        const auto t0 = clk::now();
        robot.low_act(a.data(), &v, &w);
        t_act.push_back(ms(t0));
        ++n_act;
        if (std::memcmp(&v, &v0, 4) != 0 || std::memcmp(&w, &w0, 4) != 0) {
          ++d_cmd;
          if (first_bad < 0) first_bad = (int)n_obs;
        }
      } else if (c == 'S') {
        rd.step(&p, &s);
        const auto t0 = clk::now();
        robot.step(p, s);
        t_step.push_back(ms(t0));
        ++n_step;
      } else if (c == 'D') {
        rd.depth(dep.data());
        robot.set_depth(dep.data());
      } else if (c == 'O') {
        rd.obs(ei.data(), e1.data());
        const auto t0 = clk::now();
        robot.low_obs();
        t_obs.push_back(ms(t0));
        long di = 0, d1 = 0;
        for (int i = 0; i < rd.ni; ++i) di += ei[(size_t)i] != robot.low_img()[i];
        for (int i = 0; i < rd.n1; ++i) d1 += std::memcmp(&e1[(size_t)i], &robot.low_1d()[i], 4) != 0;
        d_img += di;
        d_1d += d1;
        if ((di || d1) && first_bad < 0) first_bad = (int)n_obs;
        if (!ckpt.empty()) {
          const auto t1 = clk::now();
          pol.act(robot.low_img(), robot.low_1d(), out.data());
          t_pol.push_back(ms(t1));
        }
        ++n_obs;
      } else {
        throw std::runtime_error(std::string("녹화 사건 종류를 모름: ") + c);
      }
    }
    const auto stat = [](std::vector<double> v, const char* nm) {
      if (v.empty()) return;
      double s = 0;
      for (double x : v) s += x;
      std::sort(v.begin(), v.end());
      std::printf("  %s 평균 %.3f ms · 중앙 %.3f · 99%% %.3f · 최대 %.3f (%zu 번)\n", nm, s / v.size(), v[v.size() / 2], v[(size_t)(0.99 * (v.size() - 1))], v.back(), v.size());
    };
    std::printf("녹화: 판 %ld · 행동 %ld · 걸음 %ld · 관측 %ld (그림 %d 바이트 · 1D %d 칸)\n", n_ep, n_act, n_step, n_obs, rd.ni, rd.n1);
    std::printf("RobotEnv 다시 돌리기 vs 녹화 기대값: 그림 다른 바이트 %ld · 1D 다른 칸 %ld · 명령 다름 %ld\n", d_img, d_1d, d_cmd);
    std::printf("시간 (이 기기 한 코어):\n");
    stat(t_act, "low_act (명령·추적)");
    stat(t_step, "step (라이다 장 → 지도·위층 관측)");
    stat(t_obs, "low_obs (아래층 관측)");
    stat(t_pol, "정책 정방향 (LowPolicy)");
    const bool ok = d_img == 0 && d_1d == 0 && d_cmd == 0 && n_obs > 0;
    std::printf("%s%s\n", ok ? "같음 (비트)" : "다름", first_bad >= 0 ? (" — 처음 다른 관측 " + std::to_string(first_bad)).c_str() : "");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_node_replay: %s\n", ex.what());
    return 2;
  }
}
