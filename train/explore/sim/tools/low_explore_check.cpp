// 실물 합친 탐사 노드(위층 fuel + 아래층 RL — limo_fire_rl low_rl explore 모드) == CPU 시뮬 합친 탐사 (combine_run --upper fuel --policy 와 같은 설정).
// 시뮬(환경 하나, CPU 참조판)과 RobotEnv(합친 탐사 모드)가 각자 자기 상태로 위층 결정(scripts/low_explore_ctl.h — combine_run 의 환경 하나 몫)을 하고,
// 걸음마다 시뮬 라이다 장·감각·깊이를 RobotEnv 에 넣어(rl_node_check 와 같은 탭) 두 쪽의 위층 명령(목표 칸·follow)·아래층 관측(그림·1D)·명령(v, ω)을 비트 비교한다.
// 아래층 행동은 실물 CPU 정책(network/policy_cpu.h LowPolicy)이 RobotEnv 관측으로 낸 것을 두 쪽에 같이 준다.
//   low_explore_check <장면.scene> … --ckpt 아래층.bin [--steps 1500] [--seed 1] [--people 0] [--fuel-v 0.981 --fuel-w 3.232 --fuel-t0 0.10] [--fuel-time 1 --fuel-global 1 --fuel-vp 1]
// 지도 자세 오차·관측 지연은 끈다 (rl_node_check 와 같은 까닭). 판이 끝나면 둘 다 새 판 (위층 기억도 비움). 관측 설정은 체크포인트 꼬리(mdp/ckpt_cfg.h)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ckpt_cfg.h"
#include "env_api.h"
#include "limo_fire_rl/robot_env.h"
#include "low_explore_ctl.h"
#include "policy_cpu.h"

namespace {
struct Tap : sim::CpuTap {
  limo_fire_rl::RobotEnv* robot = nullptr;
  limo_fire_rl::Scan sc;
  bool have = false;
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
    if (!have) throw std::runtime_error("걸음에 라이다 장이 없음");
    limo_fire_rl::Proprio p;
    p.wheel_v = e.drive.wv;
    p.wheel_vy = e.drive.wvy;
    p.wheel_w = e.drive.ww;
    p.gyro_z = e.gyro;
    for (int i = 0; i < 7; ++i) p.imu[i] = e.imu.out[i];
    robot->step(p, sc);
    have = false;
  }
};
}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", ckpt;
  int steps = 1500, people = 0;
  uint64_t seed = 1;
  fu::Opt fo;
  fo.time = fo.global = fo.vp = true;   // 고른 조합 (scripts/README "combine_run — 위층 fuel": 시간 + 전역 + 시점)
  fo.v_eff = 0.981f;
  fo.w_eff = 3.232f;
  fo.t0 = 0.10f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " 뒤에 값"); return argv[++i]; };
    if (a == "--steps") steps = std::stoi(nx());
    else if (a == "--seed") seed = std::stoull(nx());
    else if (a == "--people") people = std::stoi(nx());
    else if (a == "--ckpt") ckpt = nx();
    else if (a == "--env") env_path = nx();
    else if (a == "--fuel-v") fo.v_eff = std::stof(nx());
    else if (a == "--fuel-w") fo.w_eff = std::stof(nx());
    else if (a == "--fuel-t0") fo.t0 = std::stof(nx());
    else if (a == "--fuel-time") fo.time = std::stoi(nx()) != 0;
    else if (a == "--fuel-global") fo.global = std::stoi(nx()) != 0;
    else if (a == "--fuel-vp") fo.vp = std::stoi(nx()) != 0;
    else scenes.push_back(a);
  }
  if (scenes.empty() || ckpt.empty()) {
    std::fprintf(stderr, "low_explore_check <장면.scene> … --ckpt 아래층.bin [--steps 1500] [--seed 1] [--people 0] [--fuel-*]\n");
    return 2;
  }
  try {
    std::string log;
    ckcfg::apply(ckpt, &log);
    std::printf("%s\n", log.c_str());
    sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    // combine_run.cpp:224-231 와 같은 설정 (max_steps 는 판 길이라 관측과 무관 — 여기서는 env.yaml 그대로)
    hc.cfg.action_mode = sim::kActPath;
    hc.cfg.curriculum = 0;
    hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
    hc.cfg.low_auto = 0;
    hc.cfg.low_act_w = 0;
    hc.cfg.low_explore = 1;
    hc.cfg.lo_aux = 0;
    hc.cfg.pe_xy = hc.cfg.pe_yaw = 0.f;
    const sim::LidarModel lidar = sim::load_lidar_model(model_path);
    std::vector<sim::World> worlds;
    for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
    sim::set_cpu_threads(1);
    sim::Env cpu(hc, worlds, lidar, 1, seed, false);
    if (people > 0) cpu.set_people_max(std::vector<int>(1, people));
    cpu.reset();
    std::vector<sim::EnvState> st;
    const auto no_delay = [&]() { cpu.get_state(st); st[0].delay = 0; cpu.set_state(0, st[0]); };
    no_delay();
    limo_fire_rl::RobotEnv robot(hc, true, true);
    xnet_cpu::LowPolicy pol;
    pol.load(ckpt);
    if (pol.n1d() != robot.low_1d_n()) throw std::runtime_error("체크포인트 1D 칸 수 ≠ 설정");
    const int NI = robot.low_img_n(), N1 = robot.low_1d_n(), AW = mdp::act::DIM, K = mdp::hier::K;
    lxc::Params lp = lxc::Params::from_cfg(hc.cfg);
    lp.fuel = fo;
    lxc::Ctl cs(lp), cr(lp);
    fu::Work ws, wr;
    Tap tap;
    tap.robot = &robot;
    std::vector<int16_t> lm;
    std::vector<uint32_t> tr;
    std::vector<uint8_t> vox, img, li;
    std::vector<float> pts, o1d, l1;
    std::vector<int32_t> cand, ccur;
    std::vector<sim::LowRl> R;
    std::vector<sim::LowEp> L;
    sim::World rw;   // RobotEnv 지도 창 (gw · gh · ox · oy 만)
    long n_obs = 0, d_img = 0, d_1d = 0, d_cmd = 0, d_dec = 0, n_dec = 0, n_ep = 0, why_n[3] = {0, 0, 0}, n_sw = 0;
    long ng_1d = 0;   // 목표 없음 걸음과 그 뒤 4 걸음의 1D 다름 (시뮬은 규칙 추종기가 돌고 실물은 0 — 직전 행동 칸이 다름, 따로 셈)
    int first_bad = -1, prev_tgt = -2, ng_until = -1;
    const auto start = [&]() {
      cpu.get_state(st);
      const sim::EnvState& e = st[0];
      const sim::World& w = worlds[(size_t)e.map];
      limo_fire_rl::Proprio p0;
      p0.wheel_v = e.drive.wv; p0.wheel_vy = e.drive.wvy; p0.wheel_w = e.drive.ww; p0.gyro_z = e.gyro;
      for (int i = 0; i < 7; ++i) p0.imu[i] = e.imu.out[i];
      const limo_fire_rl::RobotEnv::Window win{w.ox, w.oy, w.gw, w.gh};
      robot.reset(e.x0, e.y0, e.th0, p0, e.noise.fov, &win);
      rw.gw = w.gw; rw.gh = w.gh; rw.ox = w.ox; rw.oy = w.oy;
      cs.restart(); cr.restart();
      cs.set_geom(w.gw, hc.cfg.res); cr.set_geom(w.gw, hc.cfg.res);
      ++n_ep;
    };
    start();
    sim::set_cpu_tap(&tap);
    std::vector<float> act((size_t)AW, 0.f), a3((size_t)mdp::low::ACT, 0.f);
    std::vector<float> rew;
    std::vector<int32_t> dn;
    for (int t = 0; t < steps; ++t) {
      cpu.get_state(st);
      // 위층 (시뮬 쪽: combine_run 과 같은 입력 — Env 에서 꺼낸 지도·지나간 길·깊이 층·후보)
      if (cs.due(st[0])) {
        cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts); cpu.get_cand(cand); cpu.get_cand_cur(ccur);
        const sim::World& w = worlds[(size_t)st[0].map];
        const lxc::View vs{&hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
        int why = 0;
        const sim::LowCmd c1 = cs.decide(vs, ws, &why);
        why_n[why] += 1;
        cpu.set_low_cmd(std::vector<sim::LowCmd>{c1});
        ++n_dec;
        if (c1.target != prev_tgt) { ++n_sw; prev_tgt = c1.target; }
      }
      // 위층 (실물 쪽: RobotEnv 자기 값)
      if (cr.due(robot.state())) {
        const lxc::View vr{&robot.cfg(), &rw, robot.lmap(), robot.trail(), robot.cam_layer(), &robot.state(), robot.cand(), (uint32_t)robot.cand_cur()};
        const sim::LowCmd c2 = cr.decide(vr, wr);
        robot.set_cmd(c2);
      }
      if (cs.cmd().target != cr.cmd().target || cs.cmd().follow != cr.cmd().follow) { ++d_dec; if (first_bad < 0) first_bad = t; }
      // 준비 단계 (시뮬 Env::low_prepare · 실물 low_obs) — 목표 없음이면 행동 칸 −1 (combine_run.cpp:1144-1146)
      std::fill(act.begin(), act.end(), 0.f);
      if (cs.cmd().follow != 2) act[0] = -1.f;
      cpu.low_prepare(act.data());
      cpu.get_obs(img, o1d);
      robot.set_depth(img.data() + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE);
      robot.low_obs();
      cpu.get_low_obs(li, l1);
      ++n_obs;
      long di = 0, d1 = 0;
      for (int i = 0; i < NI; ++i) di += li[(size_t)i] != robot.low_img()[i];
      for (int i = 0; i < N1; ++i) d1 += std::memcmp(&l1[(size_t)i], &robot.low_1d()[i], 4) != 0;
      if (cs.cmd().follow != 2) ng_until = t + 5;
      if (t < ng_until && di == 0) { ng_1d += d1; d1 = 0; }
      d_img += di; d_1d += d1;
      if ((di || d1) && first_bad < 0) first_bad = t;
      // 아래층 정책 (RobotEnv 관측) → 두 쪽에 같은 행동
      pol.act(robot.low_img(), robot.low_1d(), a3.data());
      if (cs.cmd().follow == 2) { act[0] = a3[0]; act[1] = a3[1]; }
      float v, w;
      robot.low_act(a3.data(), &v, &w);
      cpu.step(act.data());
      cpu.get_state(st);
      cpu.get_reward(rew, dn);
      if (dn[0] != 0) {   // 판 끝 (시뮬은 리셋됨) — 실물 쪽 판 끝 코드가 같은지
        if (robot.done() != dn[0]) { ++d_cmd; if (first_bad < 0) first_bad = t; }
        pol.reset();
        no_delay();
        start();
        continue;
      }
      if (cs.cmd().follow == 2 && (std::memcmp(&st[0].cmd_v, &v, 4) != 0 || std::memcmp(&st[0].cmd_w, &w, 4) != 0)) { ++d_cmd; if (first_bad < 0) first_bad = t; }
      cpu.get_low_rl(R); cpu.get_low(L);
      cs.after_step(st[0], R[0], L[0]);
      cr.after_step(robot.state(), robot.low_rl(), robot.low_ep());
    }
    sim::set_cpu_tap(nullptr);
    std::printf("관측 %ld 번 · 판 %ld · 위층 결정 %ld (fuel %ld · slot0 대신 %ld · 목표 없음 %ld) · 목표 바뀜 %ld · 통과 %ld · 막힘 신호 %ld\n", n_obs, n_ep, n_dec, why_n[0], why_n[1],
                why_n[2], n_sw, cs.n_reach(), cs.n_stall());
    std::printf("RobotEnv vs CPU 시뮬: 위층 명령 다름 %ld · 그림 다른 바이트 %ld · 1D 다른 칸 %ld · 아래층 명령·판 끝 다름 %ld (목표 없음 걸음 뒤 직전 행동 칸 다름 %ld — 따로)\n", d_dec, d_img,
                d_1d, d_cmd, ng_1d);
    const bool ok = d_dec == 0 && d_img == 0 && d_1d == 0 && d_cmd == 0;
    std::printf("%s%s\n", ok ? "통과: 실물 합친 탐사 노드 == 시뮬 (비트)" : "다름", first_bad >= 0 ? (" — 처음 다른 걸음 " + std::to_string(first_bad)).c_str() : "");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_explore_check: %s\n", ex.what());
    return 1;
  }
}
