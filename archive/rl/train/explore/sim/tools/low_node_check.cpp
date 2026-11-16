// 실물 노드 아래층 관측 == 학습 시뮬 아래층 관측 (src/limo_fire/limo_fire_rl RobotEnv 아래층 모드 — 아래층 정책을 실물에 붙일 때 관측이 갈라지는 것을 막는다).
// 학습 판(train --low 와 같은 설정 — 목표를 환경이 뽑음, 목적지만)의 CPU 시뮬을 환경 하나로 돌리며, 걸음마다 시뮬 라이다 장(빔마다 거리·세기)·지도 자세·바퀴·자이로·IMU
// 를 RobotEnv 에 넣고(rl_node_check 와 같은 탭), 시뮬이 뽑은 목표를 위층이 준 것처럼 set_goal, 깊이 바이트(센서 입력)를 set_depth 로 넣은 뒤
// 아래층 관측(그림 low_img — 가까운 지도·라이다 4 장·깊이·넓은 지도, 1D low_1d — 바퀴·자이로·헛돎·EKF·목적지·직전 행동·움직이는 것 칸)과 명령을 비트 비교한다.
//   --gpu: 같은 씨앗·같은 행동의 GPU 시뮬도 같이 돌려 RobotEnv 관측을 GPU 아래층 관측과도 직접 비교 (SIM_EXACT=1)
//   --dump 녹화.bin: RobotEnv 가 받은 입력(리셋·목표·행동·감각·라이다 장·깊이)과 GPU(없으면 CPU) 관측·명령을 차례대로 녹화 →
//                   sim/tools/low_node_replay 가 RobotEnv 만으로 다시 돌려 비교 (GPU 없이 — 젯슨에서도)
//   low_node_check <장면.scene> … [--steps 600] [--seed 1] [--people 4] [--ckpt 아래층.bin] [--gpu] [--dump 녹화.bin]
//   --negative: 가운데 걸음에 RobotEnv 쪽 깊이 바이트 하나를 바꿔 넣는다 (다름이 나와야 함)
//   --ckpt: 행동 = 실물 CPU 정책(network/policy_cpu.h LowPolicy)이 RobotEnv 관측으로 낸 것 (없으면 고정 씨앗 무작위 v·ω)
// 지도 자세 오차·관측 지연은 끈다 (실물은 Cartographer 자세 하나 · 지연은 센서 값이 이미 품음 — rl_node_check 와 같은 까닭). 판이 끝나면 시뮬은 리셋되고
// RobotEnv 도 새 판 시작 상태로 다시 reset 한다. 1D·그림 설정(넓은 지도·움직이는 것 칸 넷 등)은 학습처럼 환경변수(SIM_LOW_*)로.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "env_api.h"
#include "limo_fire_rl/robot_env.h"
#include "low_node_rec.h"
#include "policy_cpu.h"

namespace {

struct Tap : sim::CpuTap {
  limo_fire_rl::RobotEnv* robot = nullptr;
  limo_fire_rl::Scan sc;
  bool have = false;
  lowrec::Writer* rec = nullptr;
  void scan(int, const sim::EnvState& e, const sim::Shared& sh, const std::vector<float>& r, const std::vector<float>& inten) override {
    sc.ranges = r;
    sc.intensities = inten;
    sc.angle_min = -0.5f * e.noise.fov;
    sc.angle_increment = sh.inc;
    sc.n_rev = sh.scan.n_rev;
    sc.scan_time = e.noise.scan_time;
    sc.time_increment = e.noise.scan_time / sh.scan.n_rev;
    sc.x = e.x + e.ex;
    sc.y = e.y + e.ey;
    sc.th = e.th + e.eth;
    sc.w = e.w;
    have = true;
  }
  void step_end(int, const sim::EnvState& e) override {
    if (!have) throw std::runtime_error("걸음에 라이다 장이 없음 (lidar_every ≠ substeps?)");
    limo_fire_rl::Proprio p;
    p.wheel_v = e.drive.wv;
    p.wheel_vy = e.drive.wvy;
    p.wheel_w = e.drive.ww;
    p.gyro_z = e.gyro;
    for (int i = 0; i < 7; ++i) p.imu[i] = e.imu.out[i];
    if (rec) rec->step(p, sc);
    robot->step(p, sc);
    have = false;
  }
};

limo_fire_rl::Proprio proprio_of(const sim::EnvState& e) {
  limo_fire_rl::Proprio p;
  p.wheel_v = e.drive.wv;
  p.wheel_vy = e.drive.wvy;
  p.wheel_w = e.drive.ww;
  p.gyro_z = e.gyro;
  for (int i = 0; i < 7; ++i) p.imu[i] = e.imu.out[i];
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", ckpt, dump;
  int steps = 600, people = 4;
  uint64_t seed = 1;
  bool use_gpu = false, negative = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    else if (a == "--people" && i + 1 < argc) people = std::stoi(argv[++i]);
    else if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--gpu") use_gpu = true;
    else if (a == "--negative") negative = true;   // 가운데 걸음에 RobotEnv 쪽 깊이 바이트 하나를 바꿈 (검사가 틀림을 잡는지)
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else scenes.push_back(a);
  }
  if (scenes.empty()) {
    std::fprintf(stderr, "low_node_check <장면.scene> … [--steps 600] [--seed 1] [--people 4] [--ckpt 아래층.bin] [--gpu] [--dump 녹화.bin]\n");
    return 2;
  }
  try {
    sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    hc.cfg.action_mode = sim::kActPath;   // train --low 와 같음 (scripts/train.cpp run_low)
    hc.cfg.curriculum = 0;
    hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = hc.cfg.low_auto = 1;
    hc.cfg.low_act_w = mdp::low::ACT;
    hc.cfg.lo_aux = 0;
    hc.cfg.pe_xy = hc.cfg.pe_yaw = 0.f;
    if (!hc.cfg.lo_goalonly) throw std::runtime_error("SIM_LOW_GOALONLY=1 로 (실물 아래층 모드는 목적지만 정책)");
    const sim::LidarModel lidar = sim::load_lidar_model(model_path);
    std::vector<sim::World> worlds;
    for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
    sim::set_cpu_threads(1);
    sim::Env cpu(hc, worlds, lidar, 1, seed, false);
    std::unique_ptr<sim::Env> gpu;
    if (use_gpu) gpu.reset(new sim::Env(hc, worlds, lidar, 1, seed, true));
    cpu.set_people_max(std::vector<int>(1, people));
    if (gpu) gpu->set_people_max(std::vector<int>(1, people));
    cpu.reset();
    if (gpu) { gpu->reset(); gpu->sync(); }
    std::vector<sim::EnvState> st, sg;
    const auto no_delay = [&]() {   // 관측 지연 끔 (판마다 새로 뽑히므로 판 시작마다)
      cpu.get_state(st);
      st[0].delay = 0;
      cpu.set_state(0, st[0]);
      if (gpu) {
        gpu->get_state(sg);
        sg[0].delay = 0;
        gpu->set_state(0, sg[0]);
      }
    };
    no_delay();
    limo_fire_rl::RobotEnv robot(hc, true);
    const int NI = robot.low_img_n(), N1 = robot.low_1d_n();
    xnet_cpu::LowPolicy pol;
    if (!ckpt.empty()) {
      pol.load(ckpt);
      if (pol.n1d() != N1) throw std::runtime_error("체크포인트 1D 칸 수 ≠ 설정 (SIM_LOW_MOV4 확인)");
    }
    std::unique_ptr<lowrec::Writer> rec;
    if (!dump.empty()) rec.reset(new lowrec::Writer(dump, hc, NI, N1));
    Tap tap;
    tap.robot = &robot;
    tap.rec = rec.get();
    float* d_act = nullptr;
    if (gpu) cudaMalloc(&d_act, sizeof(float) * mdp::low::ACT);
    std::vector<uint8_t> img, li, lig;
    std::vector<float> o1d, l1, l1g;
    std::vector<sim::LowCmd> cmd;
    std::vector<sim::LowRl> R;
    long n_obs = 0, d_img = 0, d_1d = 0, d_cmd = 0, g_img = 0, g_1d = 0, n_ep = 0, n_goal = 0, n_mov = 0, n_mov4 = 0, n_wgoal = 0, n_body = 0;
    int first_bad = -1;
    double max_1d = 0;
    const auto start_episode = [&]() {   // RobotEnv 를 시뮬 판 시작 상태로 (rl_node_check 와 같음) + 시뮬이 뽑은 첫 목표
      cpu.get_state(st);
      const sim::EnvState& e = st[0];
      const sim::World& w = worlds[(size_t)e.map];
      const limo_fire_rl::Proprio p0 = proprio_of(e);
      const limo_fire_rl::RobotEnv::Window win{w.ox, w.oy, w.gw, w.gh};
      robot.reset(e.x0, e.y0, e.th0, p0, e.noise.fov, &win);
      if (rec) rec->reset(e.x0, e.y0, e.th0, p0, e.noise.fov, win);
      ++n_ep;
    };
    int goal_step = -1, goal_tgt = -2;
    const auto sync_goal = [&]() {   // 시뮬이 이번 준비 단계에 새 목표를 뽑았으면(lowrl_goal_set: g_step·목표) 위층이 준 것처럼
      cpu.get_low_rl(R);
      cpu.get_low_cmd(cmd);
      if (R[0].g_step != goal_step || cmd[0].target != goal_tgt) {
        goal_step = R[0].g_step;
        goal_tgt = cmd[0].target;
        robot.set_goal(cmd[0].target);
        if (rec) rec->goal(cmd[0].target);
        ++n_goal;
      }
    };
    const auto observe = [&](int t) {   // 깊이(센서 입력) → RobotEnv 아래층 관측 → 시뮬과 비교
      cpu.get_obs(img, o1d);
      const uint8_t* depth = img.data() + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE;
      if (negative && t == steps / 2) {
        std::vector<uint8_t> d2(depth, depth + mdp::obs::DEPTH_SIZE);
        d2[(size_t)mdp::obs::DEPTH_SIZE / 2] ^= 0x40;
        robot.set_depth(d2.data());
        std::printf("일부러 어긋나게: 걸음 %d 의 RobotEnv 깊이 바이트 하나를 바꿈\n", t);
      } else {
        robot.set_depth(depth);
      }
      if (rec) rec->depth(depth);
      robot.low_obs();
      cpu.get_low_obs(li, l1);
      ++n_obs;
      long di = 0, d1 = 0;
      for (int i = 0; i < NI; ++i) di += li[(size_t)i] != robot.low_img()[i];
      for (int i = 0; i < N1; ++i)
        if (std::memcmp(&l1[(size_t)i], &robot.low_1d()[i], 4) != 0) {
          ++d1;
          max_1d = std::max(max_1d, (double)std::fabs(l1[(size_t)i] - robot.low_1d()[i]));
        }
      d_img += di;
      d_1d += d1;
      const uint8_t* exp_i = li.data();
      const float* exp_1 = l1.data();
      if (gpu) {
        gpu->get_low_obs(lig, l1g);
        long gi = 0, g1 = 0;
        for (int i = 0; i < NI; ++i) gi += lig[(size_t)i] != robot.low_img()[i];
        for (int i = 0; i < N1; ++i) g1 += std::memcmp(&l1g[(size_t)i], &robot.low_1d()[i], 4) != 0;
        g_img += gi;
        g_1d += g1;
        if ((gi || g1) && first_bad < 0) first_bad = t;
        exp_i = lig.data();
        exp_1 = l1g.data();
      }
      if ((di || d1) && first_bad < 0) first_bad = t;
      if (std::getenv("LOW_CHECK_VERBOSE") && (di || d1))
        for (int i = 0; i < N1; ++i)
          if (std::memcmp(&l1[(size_t)i], &robot.low_1d()[i], 4) != 0) std::printf("  걸음 %d 1D 칸 %d 시뮬 %g 실물 %g\n", t, i, l1[(size_t)i], robot.low_1d()[i]);
      if (rec) rec->obs(exp_i, exp_1);
      n_mov += robot.low_1d()[sim::kLoMov] > 0.5f;
      if (hc.cfg.lo_mov4) n_mov4 += robot.low_1d()[sim::kLoMov4 + mdp::low::MOV_W] > 0.5f;   // 둘째 칸까지 찬 걸음
      if (robot.low_cmd().target >= 0) ++n_wgoal;
      if (hc.cfg.lo_body) {   // 몸 둘레 간격: 한 구역이라도 BODY_RANGE 안에 막힘이 있는 관측
        const float* B = robot.low_1d() + mdp::low::body_off(hc.cfg.lo_mov4 != 0);
        bool near = false;
        for (int k = 0; k < mdp::low::BODY_SEC; ++k) near = near || B[k] < 1.f;
        n_body += near;
      }
    };
    start_episode();
    sync_goal();
    observe(-1);
    sim::set_cpu_tap(&tap);
    uint64_t rng = 777 + seed;
    std::vector<float> a((size_t)mdp::low::ACT, 0.f);
    std::vector<float> rw;
    std::vector<int32_t> dn;
    for (int t = 0; t < steps; ++t) {
      if (!ckpt.empty()) {
        pol.act(robot.low_img(), robot.low_1d(), a.data());
      } else {
        a[0] = dm::rand_range(rng, -0.2f, 1.1f);
        a[1] = dm::rand_range(rng, -1.1f, 1.1f);
        a[2] = 0.f;
      }
      float v, w;
      robot.low_act(a.data(), &v, &w);
      if (rec) rec->act(a.data(), v, w);
      cpu.step(a.data());
      if (gpu) {
        cudaMemcpy(d_act, a.data(), sizeof(float) * mdp::low::ACT, cudaMemcpyHostToDevice);
        gpu->step(d_act);
        gpu->sync();
      }
      cpu.get_state(st);
      // 명령 = 시뮬 lowrl_act 가 낸 것 (판이 끝나 리셋된 걸음은 새 판 상태라 견주지 않음)
      cpu.get_reward(rw, dn);
      if (dn[0] != 0) {
        if (!ckpt.empty()) pol.reset();
        no_delay();
        start_episode();
        goal_step = -1;
        goal_tgt = -2;
        sync_goal();
        observe(t);
        continue;
      }
      if (std::memcmp(&st[0].cmd_v, &v, 4) != 0 || std::memcmp(&st[0].cmd_w, &w, 4) != 0) {
        ++d_cmd;
        if (first_bad < 0) first_bad = t;
      }
      sync_goal();
      observe(t);
    }
    sim::set_cpu_tap(nullptr);
    if (d_act) cudaFree(d_act);
    std::printf("설정: 그림 %d 바이트 · 1D %d 칸 (넓은 지도 %d · 움직이는 것 %d · 칸 넷 %d · 몸 둘레 %d) · 행동 %s\n", NI, N1, hc.cfg.lo_wide, hc.cfg.lo_mov, hc.cfg.lo_mov4, hc.cfg.lo_body,
                ckpt.empty() ? "무작위" : "실물 CPU 정책");
    std::printf("관측 %ld 번 · 판 %ld · 목표 받음 %ld · 목표 있는 관측 %ld · 움직이는 것 찬 관측 %ld (칸 넷 둘째 칸까지 %ld) · 몸 둘레 1 m 안 막힘 관측 %ld\n", n_obs, n_ep, n_goal,
                n_wgoal, n_mov, n_mov4, n_body);
    std::printf("RobotEnv vs CPU 시뮬: 그림 다른 바이트 %ld · 1D 다른 칸 %ld (최대 차 %.3g) · 명령 다름 %ld\n", d_img, d_1d, max_1d, d_cmd);
    if (gpu) std::printf("RobotEnv vs GPU 시뮬: 그림 다른 바이트 %ld · 1D 다른 칸 %ld\n", g_img, g_1d);
    const bool ok = d_img == 0 && d_1d == 0 && d_cmd == 0 && g_img == 0 && g_1d == 0;
    std::printf("%s%s\n", ok ? "통과: 실물 노드 아래층 관측 == 시뮬 (비트)" : "다름", first_bad >= 0 ? (" — 처음 다른 걸음 " + std::to_string(first_bad)).c_str() : "");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_node_check: %s\n", ex.what());
    return 1;
  }
}
