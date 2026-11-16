// 실물 방 재현: 실물 합친 탐사 노드(limo_fire_rl low_rl — 위층 fuel/slot0 + 아래층 규칙 추종기)를 시뮬(CPU 참조판, 환경 하나) 안에서 노드와 같은 차례로 돌리고
// 노드와 같은 꼴(lowrec — low_rl record:=)로 녹화한다. 실물 녹화와 같은 도구(node_stats)로 견주려고.
//   --follow 걸음.csv th tx ty [--follow-ahead 3]: 위층 대신 실물 궤적을 따라감 — node_stats --csv 의 (t, x, y)를 장면 좌표로 (회전 th · 이동 tx ty) 옮겨
//     결정마다 실물이 ahead 초 뒤 있던 자리를 목표로 (규칙 추종기 · 목표 칸 그대로, fuel 과 같은 길). 판단 모델처럼 다시 못 내는 위층의 실물 주행과 아래층·센서·지도를 견줄 때.
//     실물 시작 자세도 그 표 첫 줄에서 (--start 가 없으면)
//   room_run <장면.scene> [--start x y θ (없으면 장면 출발점·방향 0)] [--upper fuel|slot0] [--seed 1] [--max-s 300] [--rec 녹화.bin] [--exact] [--env env.yaml] [--model sensor_model.yaml]
// 장면 좌표 = 실물 지도 좌표 (map2scene 로 만든 장면이면 --start 는 그 실물 지도에서 로봇이 판을 시작한 자세 — node_stats 의 R 사건 / 노드 로그 "시작 자세").
// 시뮬 쪽: 센서(라이다 장·바퀴·IMU·깊이)를 RobotEnv 에 넣는 탭은 low_explore_check 와 같다. 위층 결정·판 끝은 RobotEnv(노드 몫)만 — 시뮬 환경은 같은 목표·행동 칸을
// 받아 자기 규칙 추종기로 몬다(env.set_low_cmd · low_prepare). --exact 가 아니면 지도 자세 오차·관측 지연은 env.yaml 그대로 — 노드가 받는 Cartographer 자세 대신 시뮬 오차 모델.
// 위층 결정은 깊이를 넣기 전에 한다 (노드는 깊이 뒤지만 결정은 앞 걸음 깊이 층만 써서 같은 값 — 시뮬 준비 단계가 이 걸음의 목표·행동 칸을 받게).
// 끝: 노드 판 끝(닫힘 성공 · 진척 없음 · 시간) 또는 시뮬 판 끝(부딪힘 등 — 노드라면 닿음으로 멈춤) 또는 --max-s.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "env_api.h"
#include "limo_fire_rl/robot_env.h"
#include "low_explore_ctl.h"
#include "low_node_rec.h"

namespace {
struct Tap : sim::CpuTap {
  limo_fire_rl::RobotEnv* robot = nullptr;
  lowrec::Writer* rec = nullptr;
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
    if (rec) rec->step(p, sc);
    robot->step(p, sc);
    have = false;
  }
};
}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::string scene, env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", rec_path;
  float start[3] = {0.f, 0.f, 0.f};
  bool have_start = false, exact = false, model_set = false;
  double max_s = 300.0, follow_ahead = 3.0;
  std::string follow_path;
  float ftf[3] = {0.f, 0.f, 0.f};
  int depth_lag = 0;
  uint64_t seed = 1;
  bool use_fuel = true;
  fu::Opt fo;
  fo.time = fo.global = fo.vp = true;   // 노드 기본 (config/low_rl.yaml)
  fo.v_eff = 0.981f;
  fo.w_eff = 3.232f;
  fo.t0 = 0.10f;
  fo.hyst = 0.2f;      // 실물 노드 기본 (limo_fire_rl config/low_rl.yaml fuel_hyst · fuel_vp_min · fuel_vp_clear — low_explore_replay 기본과 같게)
  fo.vp_min = 0.75f;
  fo.vp_clear = 0.21f;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " 뒤에 값"); return argv[++i]; };
      if (a == "--start") { for (int k = 0; k < 3; ++k) start[k] = std::stof(nx()); have_start = true; }
      else if (a == "--upper") { const std::string u = nx(); if (u != "fuel" && u != "slot0") throw std::runtime_error("--upper 는 fuel · slot0"); use_fuel = u == "fuel"; }
      else if (a == "--seed") seed = std::stoull(nx());
      else if (a == "--max-s") max_s = std::stod(nx());
      else if (a == "--rec") rec_path = nx();
      else if (a == "--exact") exact = true;
      else if (a == "--follow") { follow_path = nx(); for (int k = 0; k < 3; ++k) ftf[k] = std::stof(nx()); }
      else if (a == "--follow-ahead") follow_ahead = std::stod(nx());
      else if (a == "--depth-lag") depth_lag = std::stoi(nx());
      else if (a == "--env") env_path = nx();
      else if (a == "--model") { model_path = nx(); model_set = true; }
      else if (a[0] == '-') throw std::runtime_error("모르는 인자 " + a);
      else scene = a;
    }
    if (scene.empty())
      throw std::runtime_error("room_run <장면.scene> [--start x y θ] [--upper fuel|slot0] [--seed 1] [--max-s 300] [--rec 녹화.bin] [--exact]");
    struct FP { double t; float x, y, th; };
    std::vector<FP> fol;   // 실물 궤적 (장면 좌표)
    if (!follow_path.empty()) {
      FILE* f = std::fopen(follow_path.c_str(), "r");
      if (!f) throw std::runtime_error("못 읽음: " + follow_path);
      char line[1024];
      if (!std::fgets(line, sizeof line, f)) throw std::runtime_error("빈 표: " + follow_path);   // 머리 (ep,k,t,x,y,th,…)
      const float c = std::cos(ftf[0]), sn = std::sin(ftf[0]);
      while (std::fgets(line, sizeof line, f)) {
        int ep, k;
        double t;
        float x, y, th;
        if (std::sscanf(line, "%d,%d,%lf,%f,%f,%f", &ep, &k, &t, &x, &y, &th) != 6 || ep != 0) continue;
        fol.push_back(FP{t, c * x - sn * y + ftf[1], sn * x + c * y + ftf[2], th + ftf[0]});
      }
      std::fclose(f);
      if (fol.empty()) throw std::runtime_error("궤적이 없음: " + follow_path);
      if (!have_start) { start[0] = fol[0].x; start[1] = fol[0].y; start[2] = fol[0].th; have_start = true; }
    }
    sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    // combine_run.cpp 합친 탐사 설정 (low_explore_check 와 같음)
    hc.cfg.action_mode = sim::kActPath;
    hc.cfg.curriculum = 0;
    hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
    hc.cfg.low_auto = 0;
    hc.cfg.low_act_w = 0;
    hc.cfg.low_explore = 1;
    hc.cfg.lo_aux = 0;
    if (exact) hc.cfg.pe_xy = hc.cfg.pe_yaw = 0.f;
    if (model_set) {   // --model: 깊이·IMU 도 그 파일로 (env.yaml 의 sensor_model 대신 — 방마다 다른 바닥 구멍 같은 값을 갈아 끼워 볼 때)
      hc.cfg.depth = sim::load_depth_model(model_path);
      hc.cfg.imu = sim::load_imu_model(model_path, hc.cfg.dt);
    }
    const sim::LidarModel lidar = sim::load_lidar_model(model_path);
    sim::Scene sc = sim::load_scene(scene);
    if (have_start) {
      sc.start[0] = start[0];
      sc.start[1] = start[1];
    } else {   // --start 없음: 장면 출발점, 방향 0
      start[0] = sc.start[0];
      start[1] = sc.start[1];
    }
    std::vector<sim::World> worlds{sim::build_world(scene, sc, hc, false)};
    if (!worlds[0].closed) std::printf("장면이 닫히지 않음 (판 끝 규칙은 로봇 지도만 봄 — 그대로 돎): %s\n", worlds[0].leak.c_str());
    sim::set_cpu_threads(1);
    sim::Env cpu(hc, worlds, lidar, 1, seed, false);
    cpu.reset();
    std::vector<sim::EnvState> st;
    cpu.get_state(st);
    {
      sim::EnvState e = st[0];
      e.x = e.x0 = e.gx = start[0];
      e.y = e.y0 = e.gy = start[1];
      e.th = e.th0 = start[2];
      e.ex = e.ey = e.eth = 0.f;
      if (exact) e.delay = 0;
      sim::cam_hist_fill(e);   // 깊이 카메라 늦음 기록도 시작 자세로
      cpu.set_state(0, e);
      cpu.get_state(st);
    }
    limo_fire_rl::RobotEnv robot(hc, true, true);
    std::unique_ptr<lowrec::Writer> rec;
    if (!rec_path.empty()) rec.reset(new lowrec::Writer(rec_path, hc, robot.low_img_n(), robot.low_1d_n()));
    lxc::Params lp = lxc::Params::from_cfg(hc.cfg);
    lp.fuel = fo;
    lp.use_fuel = use_fuel;
    lxc::Ctl ctl(lp);
    fu::Work work;
    Tap tap;
    tap.robot = &robot;
    tap.rec = rec.get();
    std::vector<uint8_t> img;
    std::vector<float> o1d;
    std::vector<std::vector<uint8_t>> dep_hist;
    sim::World rw;
    {
      const sim::EnvState& e = st[0];
      const sim::World& w = worlds[0];
      limo_fire_rl::Proprio p0;
      p0.wheel_v = e.drive.wv; p0.wheel_vy = e.drive.wvy; p0.wheel_w = e.drive.ww; p0.gyro_z = e.gyro;
      for (int i = 0; i < 7; ++i) p0.imu[i] = e.imu.out[i];
      const limo_fire_rl::RobotEnv::Window win{w.ox, w.oy, w.gw, w.gh};
      robot.reset(e.x0, e.y0, e.th0, p0, e.noise.fov, &win);
      if (rec) rec->reset(e.x0, e.y0, e.th0, p0, e.noise.fov, win);
      rw.gw = w.gw; rw.gh = w.gh; rw.ox = w.ox; rw.oy = w.oy;
      ctl.restart();
      ctl.set_geom(w.gw, hc.cfg.res);
    }
    sim::set_cpu_tap(&tap);
    const int AW = mdp::act::DIM;
    std::vector<float> act((size_t)AW, 0.f), a3((size_t)mdp::low::ACT, 0.f), rew;   // a3: RobotEnv::low_act 행동 칸 (규칙 추종기는 안 씀 — 0)
    std::vector<int32_t> dn;
    const double dt = (double)hc.cfg.substeps * hc.cfg.dt;
    const int max_steps = (int)std::lround(max_s / dt);
    const char* why = "시간 (--max-s)";
    int t = 0, sim_end = 0;
    for (; t < max_steps; ++t) {
      // 노드 차례: (앞 걸음 step 은 탭에서) → 위층 → 깊이 → set_slot · low_obs → low_act
      if (ctl.due(robot.state())) {
        const lxc::View vr{&robot.cfg(), &rw, robot.lmap(), robot.trail(), robot.cam_layer(), &robot.state(), robot.cand(), (uint32_t)robot.cand_cur()};
        sim::LowCmd c = ctl.decide(vr, work);   // 위층 기록(결정 수 등)은 그대로 — --follow 면 목표만 실물 궤적으로 바꿈
        if (!fol.empty()) {
          const double tt = t * dt + follow_ahead;
          size_t k = 0;
          while (k + 1 < fol.size() && fol[k + 1].t <= tt) ++k;
          const int gx = (int)std::floor((fol[k].x - rw.ox) / hc.cfg.res), gy = (int)std::floor((fol[k].y - rw.oy) / hc.cfg.res);
          c = (gx >= 0 && gy >= 0 && gx < rw.gw && gy < rw.gh) ? sim::LowCmd{gy * rw.gw + gx, 1} : sim::LowCmd{-1, 0};
          ctl.force_cmd(c);
        }
        robot.set_cmd(c);
        cpu.set_low_cmd(std::vector<sim::LowCmd>{c});   // 시뮬 환경도 같은 목표
      }
      act.assign((size_t)AW, 0.f);
      act[0] = (float)ctl.slot();   // combine_run 규칙: 결정 때 정한 행동 칸 (다음 결정까지 그대로)
      cpu.low_prepare(act.data());
      cpu.get_obs(img, o1d);
      const uint8_t* dep_now = img.data() + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE;
      // --depth-lag 1: 노드가 앞 걸음(0.1 s 전) 자세에서 찍힌 깊이를 받는 셈 (실물 깊이가 자세보다 늦음 — node_stats "깊이 시각 어긋남")
      dep_hist.insert(dep_hist.begin(), std::vector<uint8_t>(dep_now, dep_now + mdp::obs::DEPTH_SIZE));
      if ((int)dep_hist.size() > depth_lag + 1) dep_hist.pop_back();
      const uint8_t* dep = dep_hist.back().data();
      robot.set_depth(dep);
      if (rec) rec->depth(dep);
      robot.set_slot(ctl.slot());
      robot.low_obs();
      if (rec) rec->obs(robot.low_img(), robot.low_1d());
      float v, w;
      robot.low_act(a3.data(), &v, &w);
      if (rec) rec->act(a3.data(), v, w);
      cpu.step(act.data());   // 탭이 라이다 장·감각을 RobotEnv 로 (step)
      cpu.get_reward(rew, dn);
      const int rd = robot.done();
      if (rd == mdp::cur::kDeclareOk || rd == mdp::cur::kNoProgress || rd == mdp::cur::kTruncated) {
        why = rd == mdp::cur::kDeclareOk ? "닫힘 성공" : rd == mdp::cur::kNoProgress ? "진척 없음" : "시간";
        ++t;
        break;
      }
      if (dn[0] != 0) { sim_end = dn[0]; why = "시뮬 판 끝 (부딪힘 등)"; ++t; break; }
      cpu.get_state(st);
      ctl.after_step(robot.state(), robot.low_rl(), robot.low_ep());
    }
    sim::set_cpu_tap(nullptr);
    std::printf("room_run: %s · 걸음 %d (%.1f s) · 끝 %s%s · 결정 %ld · 통과 %ld · 막힘 %ld · %s\n", scene.c_str(), t, t * dt, why,
                sim_end ? (" (시뮬 끝 코드 " + std::to_string(sim_end) + ")").c_str() : "", ctl.n_dec(), ctl.n_reach(), ctl.n_stall(), exact ? "자세 오차·지연 끔" : "자세 오차·지연 env.yaml");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "room_run: %s\n", ex.what());
    return 1;
  }
}
