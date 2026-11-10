// 실물 노드 관측 == 학습 시뮬 관측 확인 (sim-to-real 의 가장 큰 위험 — 관측 만들기가 갈라짐 — 을 막는다).
// CPU 시뮬(계층형, 환경 1 개)을 돌리며 걸음마다 시뮬이 만든 라이다 장(빔마다 거리·세기)·지도 자세·바퀴·자이로·IMU·명령을 실물 노드의
// RobotEnv(src/limo_fire/limo_fire_rl — /scan·TF·/odom·/imu·/cmd_vel 을 받는 것과 같은 입력)에 그대로 넣고, 결정마다 두 관측(2D 바이트·1D float·후보 칸)과
// 지도(로그 오즈)·닫힘 길·닫힘·진행 없음 시계·판 끝 코드를 비트까지 비교한다. 깊이 바이트는 센서 입력이라 시뮬 관측 값을 그대로 넣는다(깊이 변환은 따로).
//   rl_node_check <장면.scene> … [--decisions 120] [--seed 1] [--ckpt 정책.bin] [--pose-error] [--dump 관측.bin]
//   --ckpt: 결정을 정책 CPU 정방향(network/policy_cpu.h)으로 (없으면 있는 후보 중 무작위). --pose-error: 지도 자세 오차를 켬(기본 끔 — 아래)
//   --dump: 결정마다 시뮬 관측(이미지 IMG_SIZE 바이트 + 1D hier::POLICY_1D float)을 이어 씀 → ppo/tools/policy_cpu_check 입력
// 지도 자세 오차를 끄는 까닭: 시뮬 깊이 카메라 장애물 층은 참값 자세(e.x)로 칸을 찍고 실물은 Cartographer 자세 하나뿐이라, 오차를 켜면 그 층(→ 후보)만
// 달라진다 — 켠 판은 그 차이를 따로 센다(관측의 나머지는 지도 자세만 써서 같아야 함).
// 시뮬 복구(목표 포기 — hier_path.h path_recovery)는 실물에선 Nav2 가 맡으므로, 시뮬이 고른 목표 칸이 RobotEnv::choose 와 다르면 시뮬 값으로 맞추고 센다.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "env_api.h"
#include "limo_fire_rl/robot_env.h"
#include "policy_cpu.h"

namespace {

struct Tap : sim::CpuTap {
  limo_fire_rl::RobotEnv* robot = nullptr;
  limo_fire_rl::Scan sc;
  bool have = false;
  int steps = 0;
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
    p.cmd_v = e.cmd_v;
    p.cmd_w = e.cmd_w;
    robot->step(p, sc);
    have = false;
    ++steps;
  }
};

struct Cmp {
  long n = 0, diff = 0;
  double maxd = 0;
  int first = -1;
};
template <class T>
Cmp cmp(const T* a, const T* b, size_t n) {
  Cmp c;
  c.n = (long)n;
  for (size_t i = 0; i < n; ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0) continue;
    ++c.diff;
    const double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > c.maxd) c.maxd = d;
    if (c.first < 0) c.first = (int)i;
  }
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", ckpt, dump;
  int decisions = 120;
  uint64_t seed = 1;
  bool pose_err = false, nearest = false, bf16 = false, keep_delay = false;   // --keep-delay: 시뮬 관측 지연(0·1 걸음)을 그대로 (실물 노드 비교는 다를 수 있음 — 정책 성공률만 볼 때)   // --nearest: 결정 = BFS 거리 가장 짧은 있는 후보 (비교 규칙)
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--decisions" && i + 1 < argc) decisions = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    else if (a == "--ckpt" && i + 1 < argc) ckpt = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--pose-error") pose_err = true;
    else if (a == "--nearest") nearest = true;
    else if (a == "--bf16") bf16 = true;
    else if (a == "--keep-delay") keep_delay = true;
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else scenes.push_back(a);
  }
  if (scenes.empty()) {
    std::fprintf(stderr, "rl_node_check <장면.scene> … [--decisions 120] [--seed 1] [--ckpt 정책.bin] [--pose-error] [--dump 관측.bin]\n");
    return 2;
  }
  try {
    sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
    hc.cfg.action_mode = sim::kActPath;   // train --hier 와 같음
    hc.cfg.curriculum = 0;
    if (!pose_err) hc.cfg.pe_xy = hc.cfg.pe_yaw = 0.f;
    const sim::LidarModel lidar = sim::load_lidar_model(model_path);
    std::vector<sim::World> worlds;
    for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
    sim::set_cpu_threads(1);
    sim::Env env(hc, worlds, lidar, 1, seed, false);
    env.reset();
    std::vector<sim::EnvState> st;
    env.get_state(st);
    if (!keep_delay) st[0].delay = 0;   // 실물은 지연을 센서 값이 이미 품음 (RobotEnv delay 0) — 시뮬도 0 으로
    env.set_state(0, st[0]);
    xnet_cpu::HierPolicy pol;
    if (!ckpt.empty()) pol.load(ckpt);
    pol.set_bf16(bf16);
    FILE* fd = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    limo_fire_rl::RobotEnv robot(hc);
    Tap tap;
    tap.robot = &robot;
    const sim::World& w = worlds[(size_t)st[0].map];
    {
      const sim::EnvState& e = st[0];
      limo_fire_rl::Proprio p0;
      p0.wheel_v = e.drive.wv;
      p0.wheel_vy = e.drive.wvy;
      p0.wheel_w = e.drive.ww;
      p0.gyro_z = e.gyro;
      for (int i = 0; i < 7; ++i) p0.imu[i] = e.imu.out[i];
      const limo_fire_rl::RobotEnv::Window win{w.ox, w.oy, w.gw, w.gh};
      robot.reset(e.x0, e.y0, e.th0, p0, e.noise.fov, &win);
    }
    sim::set_cpu_tap(&tap);
    const auto& c = hc.cfg;
    const size_t n = (size_t)c.size * c.size;
    std::vector<uint8_t> img;
    std::vector<float> o1d, pv;
    std::vector<int16_t> lm;
    std::vector<uint32_t> tr;
    double st_cm = 0, st_sp = 0, st_mx = 0;
    long st_n = 0;
    long n_cam_diff = 0, n_img_diff = 0, n_1d_diff = 0, n_cand_diff = 0, n_map_diff = 0, n_trail_diff = 0, n_state_diff = 0, n_goal_sync = 0, n_dec = 0;
    double max_1d = 0;
    uint64_t rng = 12345 + seed;
    int done = 0;
    for (int k = 0; k < decisions; ++k) {
      env.get_obs(img, o1d, pv);
      env.get_state(st);
      const sim::EnvState& e = st[0];
      robot.set_depth(img.data() + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE);   // 깊이 = 센서 입력
      // 비교: 관측 2D·1D, 후보 칸, 지도, 닫힘 길, 상태 칸
      const Cmp ci = cmp(img.data(), robot.obs_img(), (size_t)mdp::obs::IMG_SIZE);
      const Cmp c1 = cmp(o1d.data(), robot.obs_1d(), (size_t)mdp::hier::POLICY_1D);
      const sim::EnvBatch& rb = robot.batch();
      int cand_diff = 0;
      {
        // 시뮬 후보 칸은 관측 1D 와 같이 나옴: 칸 번호는 Env 안 (b.cand) — 관측 후보 칸(1D)이 같으면 같은 덩어리. 칸 번호는 batch 로
        const sim::EnvBatch& sb = env.batch();
        for (int q = 0; q < mdp::hier::K; ++q) cand_diff += sb.cand[q] != rb.cand[q];
        cand_diff += sb.cand_cur[0] != rb.cand_cur[0];
      }
      env.get_map(lm);
      env.get_trail(tr);
      const Cmp cm = cmp(lm.data(), rb.lmap, n);
      const Cmp ct = cmp(tr.data(), rb.trail, tr.size());
      long cam_diff = 0;   // 깊이 카메라 장애물 층 (시뮬은 참값 자세로 찍음 — 자세 오차를 켜면 다를 수 있음, 따로 셈)
      {
        std::vector<uint8_t> vox;
        std::vector<float> pts;
        env.get_camlayer(vox, pts);
        if (!vox.empty() && rb.cvox) cam_diff = cmp(vox.data(), rb.cvox, n).diff;
      }
      n_cam_diff += cam_diff;
      const sim::EnvState& r = robot.state();
      if (std::getenv("RL_CHECK_VERBOSE")) {
        const char* nm[] = {"step", "since_occ", "map_closed", "known_free", "prog_sum", "occ_sum", "kb_x0", "kb_x1", "kb_y0", "kb_y1", "lid_head", "goal_cell", "ekf_v", "ekf_w"};
        const bool df[] = {e.step != r.step, e.since_occ != r.since_occ, e.map_closed != r.map_closed, e.known_free != r.known_free, e.prog_sum != r.prog_sum,
                           e.occ_sum != r.occ_sum, r.kb_x0 < e.kb_x0, r.kb_x1 > e.kb_x1, r.kb_y0 < e.kb_y0, r.kb_y1 > e.kb_y1, e.lid_head != r.lid_head,
                           e.goal_cell != r.goal_cell, e.ekf_v != r.ekf_v, e.ekf_w != r.ekf_w};
        for (int q = 0; q < 14; ++q)
          if (df[q]) std::printf("  상태 다름 %s\n", nm[q]);
      }
      const int sd = (e.step != r.step) + (e.since_occ != r.since_occ) + (e.map_closed != r.map_closed) + (e.known_free != r.known_free) +
                     (e.prog_sum != r.prog_sum) + (e.occ_sum != r.occ_sum) +
                     // 건드린 칸 범위: 시뮬은 덮음용 정답 벽 칸(wall_credit)도 넓혀 실물 범위 ⊆ 시뮬 범위 (프론티어 찾기는 아는 칸만 보므로 관측 같음)
                     (r.kb_x0 < e.kb_x0) + (r.kb_x1 > e.kb_x1) + (r.kb_y0 < e.kb_y0) + (r.kb_y1 > e.kb_y1) + (e.lid_head != r.lid_head) + (e.goal_cell != r.goal_cell) + (e.ekf_v != r.ekf_v) + (e.ekf_w != r.ekf_w);
      n_img_diff += ci.diff;
      n_1d_diff += c1.diff;
      max_1d = std::max(max_1d, c1.maxd);
      n_cand_diff += cand_diff;
      n_map_diff += cm.diff;
      n_trail_diff += ct.diff;
      n_state_diff += sd;
      ++n_dec;
      int valid = 0;
      for (int q = 0; q < mdp::hier::K; ++q) valid += o1d[(size_t)mdp::obs::POLICY_1D + q * mdp::hier::FEAT + mdp::hier::F_VALID] > 0.5f;
      std::printf("결정 %3d 걸음 %4d  2D 다름 %ld/%ld (첫 %d)  1D 다름 %ld/%ld 최대 %.3g (첫 %d)  후보 다름 %d  지도 다름 %ld  닫힘길 다름 %ld  상태 다름 %d  "
                  "후보 %d 닫힘 %d 진행없음 %.1fs\n",
                  k, e.step, ci.diff, ci.n, ci.first, c1.diff, c1.n, c1.maxd, c1.first, cand_diff, cm.diff, ct.diff, sd, valid, e.map_closed,
                  e.since_occ * 0.1);
      if (fd) {
        std::fwrite(img.data(), 1, (size_t)mdp::obs::IMG_SIZE, fd);
        std::fwrite(o1d.data(), sizeof(float), (size_t)mdp::hier::POLICY_1D, fd);
      }
      if (done) break;
      // 행동: 정책(CPU) 또는 무작위 있는 후보
      int slot = 0;
      float lg[mdp::hier::K], mk[mdp::hier::K];
      if (nearest) {
        float best = 1e9f;
        for (int q = 0; q < mdp::hier::K; ++q) {
          const float* F = &o1d[(size_t)mdp::obs::POLICY_1D + q * mdp::hier::FEAT];
          if (F[mdp::hier::F_VALID] > 0.5f && F[mdp::hier::F_DIST] < best) { best = F[mdp::hier::F_DIST]; slot = q; }
        }
      } else if (!ckpt.empty()) {
        slot = pol.act(img.data(), o1d.data(), lg, mk);
        // 로짓 통계 (학습 progress.jsonl logit_cm·logit_spread·logit_absmax 와 같은 정의: 있는 칸 평균·표준편차·최대 |l|)
        int nv = 0;
        double sm = 0, sq = 0, mx = 0;
        for (int q = 0; q < mdp::hier::K; ++q)
          if (mk[q] > 0.5f) { ++nv; sm += lg[q]; sq += (double)lg[q] * lg[q]; mx = std::max(mx, (double)std::fabs(lg[q])); }
        if (nv > 0) {
          const double mean = sm / nv;
          st_cm += std::fabs(mean);
          st_sp += std::sqrt(std::max(0.0, sq / nv - mean * mean));
          st_mx += mx;
          ++st_n;
        }
        if (std::getenv("RL_CHECK_SLOTS")) std::printf("  고름 %d 로짓 %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", slot, lg[0], lg[1], lg[2], lg[3], lg[4], lg[5], lg[6], lg[7]);
      }
      else {
        std::vector<int> ok;
        for (int q = 0; q < mdp::hier::K; ++q)
          if (o1d[(size_t)mdp::obs::POLICY_1D + q * mdp::hier::FEAT + mdp::hier::F_VALID] > 0.5f) ok.push_back(q);
        slot = ok.empty() ? 0 : ok[dm::splitmix64(rng) % ok.size()];
      }
      robot.choose(slot);
      robot.camlayer_update();   // 시뮬: 다음 결정 묶음 앞 hier_camlayer_cpu (같은 깊이 바이트)
      std::vector<float> act(mdp::act::DIM, 0.f);
      act[0] = (float)slot;
      tap.steps = 0;
      env.step(act.data());
      std::vector<float> rew;
      std::vector<int32_t> dn;
      env.get_reward(rew, dn);
      env.get_state(st);
      if (dn[0] != 0) {
        std::vector<sim::EpInfo> ep;
        env.get_ep(ep);
        std::printf("판 끝: 시뮬 코드 %d, 실물 노드 코드 %d (걸음 %d)\n", ep[0].done, robot.done(), tap.steps);
        if (ep[0].done != robot.done()) ++n_state_diff;
        done = 1;
        break;   // 시뮬은 리셋됨 — 비교 끝
      }
      if (st[0].goal_cell != robot.goal_cell()) {   // 시뮬 복구(목표 포기)는 실물에선 Nav2 몫 — 맞추고 셈
        robot.state_mut().goal_cell = st[0].goal_cell;
        robot.refresh_obs();
        ++n_goal_sync;
      }
    }
    if (fd) std::fclose(fd);
    sim::set_cpu_tap(nullptr);
    std::printf("\n합계 결정 %ld: 2D 다른 바이트 %ld · 1D 다른 칸 %ld (최대 차 %.3g) · 후보 칸 다름 %ld · 지도 칸 다름 %ld · 닫힘 길 낱말 다름 %ld · 상태 칸 다름 %ld · "
                "시뮬 복구로 목표 맞춤 %ld · (참고) 깊이 카메라 층 칸 다름 %ld\n",
                n_dec, n_img_diff, n_1d_diff, max_1d, n_cand_diff, n_map_diff, n_trail_diff, n_state_diff, n_goal_sync, n_cam_diff);
    if (st_n) std::printf("정책 로짓 (있는 칸): |평균| %.3f · 표준편차 %.3f · 최대 |l| %.3f (결정 %ld)\n", st_cm / st_n, st_sp / st_n, st_mx / st_n, st_n);
    const bool ok = n_img_diff == 0 && n_1d_diff == 0 && n_cand_diff == 0 && n_map_diff == 0 && n_trail_diff == 0 && n_state_diff == 0;
    std::printf("%s\n", ok ? "통과: 실물 노드 관측 == 시뮬 관측 (비트)" : "다름");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "rl_node_check: %s\n", ex.what());
    return 1;
  }
}
