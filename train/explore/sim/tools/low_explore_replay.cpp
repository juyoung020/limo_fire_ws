// 실물 합친 탐사 노드(limo_fire_explore explorer) 녹화를 RobotEnv 합친 탐사 모드 + 위층(scripts/low_explore_ctl.h)으로 다시 돌린다 (GPU·ROS 없이).
// 노드가 부르는 차례 그대로 (explorer_node.cpp on_scan): 'R' reset → ['S' step → 판 끝 → 위층 걸음 뒤] → 'D' 깊이 → [결정 차례면 위층 결정 → set_cmd] → 'A' set_slot · low_prepare · low_act.
// 아래층은 규칙 추종기 (2026-11-16 RL 을 뺀 뒤의 노드 — 행동 칸 = 위층 Ctl::slot()). 녹화 꼴은 sim/tools/low_node_rec.h (판 1 · 2 모두 읽음).
// 행동은 녹화 값을 쓰므로 궤적·지도는 녹화 그대로이고, 위층만 다시 결정한다. 녹화와 같은 위층 설정이면 다시 낸 아래층 명령이 녹화 명령과 비트까지 같아야 한다.
// 다른 위층 설정(--upper · --fuel-*)이면 명령은 다르지만(목표가 다름) 지도·궤적은 녹화 그대로 — "그 자리에서 이 위층이면 무엇을 골랐나" 를 본다.
//   low_explore_replay 녹화.bin [--upper fuel|slot0|cand] [--fuel-* …] [--log 결정.tsv] [--pgm 접두사 --pgm-every N] [--depth-stat]
//   fuel 기본 = 지금 실물 노드 기본(config/explorer.yaml). 옛 설정으로 녹화한 것을 비트로 다시 내려면 그때 값을 준다 (2026-11-16 앞: --fuel-hyst 0 --fuel-vp-min 0 --fuel-vp-clear 0)
//   (RL 을 빼기 전 판으로 1116_fuel_rl_1 659 / 659 · 1116_slot0_1 --upper slot0 86 / 86 비트 같음). 노드가 close_lidar 로 돌았으면 녹화 머리 hc 의 low_explore 2 를 그대로 이어받는다.
//   --pgm: 결정마다(--pgm-every N 째마다) 로봇 지도 그림 PPM (모름 회색 · 막힘 검정 · 닫힘 쪽 번진 칸 연두 · 깊이 층 주황 · 번진 곳 프론티어 빨강 · 밖 파랑 · 목표 초록)
//   --upper cand: 지금 실물 노드의 위층 후보(scripts/cand_view.h) + 판단 모델이 없을 때 대신 고르는 규칙(Ctl::cand_pick) — 판단 모델이 안 돈 녹화는 이것이 노드와 같은 길.
//     '대충 다 앎' 끝(Ctl::rough_done)이 난 걸음도 적는다 (노드는 거기서 끝).
//   --cam-pitch rad · --cam-z m: 녹화 머리의 깊이 카메라 숙임·높이를 바꿔 다시 돌림 (궤적·라이다 지도는 녹화 그대로, 깊이 층·후보만 바뀜 — 맞춤 확인용)
//   --depth-stat: 깊이 카메라 층이 표시하는 점의 (거리, 높이) 표 — 바닥이 장애물로 찍히는지
//   --depth-pix 접두사: 관측 깊이 픽셀(40 × 64)마다 층 표시 몫·평균 깊이·base_link 점 — 로봇에 붙은 것(발사대 등)은 돌아도 같은 픽셀·같은 깊이로 남는다.
//     접두사_mark.pgm (표시 몫 0~1 → 0~255, 8 배 확대) · 접두사_mean.pgm (평균 깊이 0~3 m → 검정~흰, 구멍 0) · 접두사_pix.tsv (표시 몫 > 0 픽셀) ·
//   --floor-fit: 바닥 면으로 깊이 카메라 높이·숙임 맞춤 — 녹화 설정(camera_mount·camera_pitch)으로 높이 ±8 cm 안·센서 수평 2 m 안 점을 바닥 후보로 모아
//     (높이 H, 숙임 p) 를 격자 찾기(최소제곱, 잔차 1.5 cm 밖 버리고 한 번 더). 바닥이 띠 아래 끝(min_obstacle_height)을 넘어 층에 찍히는지 본다
//   --depth-pix-every N 이면 N 장마다 그 장 깊이 그림 접두사_f장번호.pgm
//   녹화에 없는 것: 꺼짐·리모컨 멈춤 동안 노드는 녹화를 안 쓰고, 다시 켤 때의 즉시 결정(force_redo)도 안 남는다 — 멈춤이 끼인 녹화는 그 뒤 결정 차례가 조금 다를 수 있다.
// 결정마다: 시각·로봇 자세·왜 결정했나(주기 · 통과 · 경로 끝 밖 · 경로 없음 · 막힘 · 판 시작)·고른 목표(세계 좌표)·바뀜·거리·노드 수·지금 목표 노드·순회 값·목표 없음 까닭.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "limo_fire_explore/robot_env.h"
#include "low_explore_ctl.h"
#include "low_node_rec.h"
#include "cand_quality.h"
#include "cam_floor.h"

namespace {
const char* kWhyNo[] = {"목표 냄", "설 수 있는 시작 칸 없음", "번진 곳에 프론티어 덩어리 없음", "덩어리 있으나 자격 칸 없음", "모두 피하는 칸", "시점 자격 되는 덩어리 없음"};

// 닫힘 판정과 같은 frontier_search (fuel_upper.h decide 와 같은 SearchBuf — 지나간 칸·본 칸·close_min·몸통 반경 rad·깊이 층). n_front_all = 지도 전체 프론티어 칸 (번짐 무관)
std::vector<uint8_t> flag;
std::vector<int32_t> qa, qb, dist, label;
std::vector<mdp::Frontier> out;
struct Clos {
  mdp::SearchResult r;
  long n_front_all = 0;
};
Clos closure(const limo_fire_explore::RobotEnv& rb, float rad, bool use_cam = true) {
  const sim::MapDev& md = rb.map();
  const sim::EnvCfg& c = rb.cfg();
  const sim::EnvState& s = rb.state();
  const int gw = md.gw, gh = md.gh, n = gw * gh;
  flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); label.resize(n); out.resize(n / 2 + 1);
  const mdp::MapView mv{rb.lmap(), c.size, gw, gh};
  mdp::SearchBuf sb{flag.data(), qa.data(), qb.data(), dist.data(), label.data()};
  const int ws = (c.size + 31) / 32, nf = c.size * ws;
  sb.vis = rb.trail();
  sb.look = rb.trail() + nf;
  sb.tws = ws;
  sb.close_min = c.close_min;
  sb.rbc = mdp::robot_cells(rad, c.res);
  sb.cam = use_cam ? rb.cam_layer() : nullptr;
  const int rx = (int)std::floor((s.x + s.ex - md.ox) * c.inv_res), ry = (int)std::floor((s.y + s.ey - md.oy) * c.inv_res);
  Clos o;
  o.r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, out.data(), (int)out.size());
  for (int y = 0; y < gh; ++y) for (int x = 0; x < gw; ++x) o.n_front_all += mdp::frontier_cell(mv, x, y);
  return o;
}
// 로봇 지도 그림 PPM (closure 바로 뒤에 — 그 번짐 결과를 씀): 모름 회색 · 막힘 검정 · 빈 흰색 · 번진 칸(닫힘 쪽) 연두 · 깊이 카메라 층 막힘 주황 ·
// 번진 곳 프론티어 칸 빨강 · 번진 곳 밖 프론티어 칸 파랑 · 로봇 자홍 · 목표 초록 십자. 위가 +y
void write_ppm(const std::string& path, const limo_fire_explore::RobotEnv& rb, int tgt) {
  const sim::MapDev& md = rb.map();
  const sim::EnvCfg& c = rb.cfg();
  const int gw = md.gw, gh = md.gh;
  const mdp::MapView mv{rb.lmap(), c.size, gw, gh};
  std::vector<uint8_t> im((size_t)gw * gh * 3);
  const auto put = [&](int x, int y, int r, int g, int b) {
    if (x < 0 || y < 0 || x >= gw || y >= gh) return;
    uint8_t* q = &im[((size_t)(gh - 1 - y) * gw + x) * 3];
    q[0] = (uint8_t)r; q[1] = (uint8_t)g; q[2] = (uint8_t)b;
  };
  const uint8_t* cam = rb.cam_layer();
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) {
      const int i = y * gw + x;
      const uint8_t k = mdp::cls_at(mv, x, y);
      if (k == mdp::kUnk) put(x, y, 150, 150, 150);
      else if (k == mdp::kOcc) put(x, y, 0, 0, 0);
      else if (mdp::frontier_cell(mv, x, y)) { if (flag[i] & 1) put(x, y, 230, 0, 0); else put(x, y, 0, 60, 255); }
      else if (cam && cam[y * c.size + x]) put(x, y, 255, 150, 0);
      else if (flag[i] & 1) put(x, y, 190, 240, 190);
      else put(x, y, 255, 255, 255);
    }
  const sim::EnvState& s = rb.state();
  const int rx = (int)std::floor((s.x - md.ox) * c.inv_res), ry = (int)std::floor((s.y - md.oy) * c.inv_res);
  for (int d = -2; d <= 2; ++d) for (int e = -2; e <= 2; ++e) put(rx + d, ry + e, 220, 0, 220);
  if (tgt >= 0) for (int d = -5; d <= 5; ++d) { put(tgt % gw + d, tgt / gw, 0, 160, 0); put(tgt % gw, tgt / gw + d, 0, 160, 0); }
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%d %d\n255\n", gw, gh);
  std::fwrite(im.data(), 1, im.size(), f);
  std::fclose(f);
}
// 관측 깊이 크기(64 × 40) 회색 그림을 8 배 확대해 PGM 으로
void write_pgm8(const std::string& path, const uint8_t* g) {
  const int W = mdp::obs::DEPTH_W, H = mdp::obs::DEPTH_H, k = 8;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fprintf(f, "P5\n%d %d\n255\n", W * k, H * k);
  for (int y = 0; y < H * k; ++y)
    for (int x = 0; x < W * k; ++x) std::fputc(g[(y / k) * W + x / k], f);
  std::fclose(f);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "low_explore_replay 녹화.bin [--upper fuel|slot0] [--fuel-v 0.981 --fuel-w 3.232 --fuel-t0 0.10 --fuel-hyst 0 …] [--log 결정.tsv] [--pgm 접두사 --pgm-every N]\n");
    return 2;
  }
  std::string logp, pgm;
  int pgm_every = 0;
  bool depth_stat = false;
  bool cand_on = false;   // --cand: 위층 결정 때마다 후보 생성(cand_view.h)과 옛 hier 후보 품질 (정답 없음 — 녹화)
  std::string cand_dump;
  int stage_step = -1;
  std::string stage_pre;
  int cand_every = 10;   // 깊이 카메라 층이 표시하는 점(띠 0.03~0.25 m 안)의 높이·거리 표 — 바닥이 장애물로 찍히는지
  long dh[6][8] = {};        // [거리 0.5 m 칸 0~3 m][높이 칸 띠 8]
  bool floor_fit = false;    // --floor-fit
  sim::FloorFit ffit(2000000);
  std::string pix_pre;       // --depth-pix
  int pix_every = 0;
  long pix_frames = 0;
  std::vector<long> pix_valid(mdp::obs::DEPTH_SIZE, 0), pix_mark(mdp::obs::DEPTH_SIZE, 0);
  std::vector<double> pix_z(mdp::obs::DEPTH_SIZE, 0.0), pix_z2(mdp::obs::DEPTH_SIZE, 0.0), pix_b[3];
  for (auto& v : pix_b) v.assign(mdp::obs::DEPTH_SIZE, 0.0);
  bool use_fuel = true;
  bool use_cand = false;   // --upper cand
  float cam_pitch = NAN, cam_z = NAN;
  fu::Opt fo;   // 실물 노드 기본 (config/explorer.yaml)
  fo.time = fo.global = fo.vp = true;
  fo.v_eff = 0.981f;
  fo.w_eff = 3.232f;
  fo.t0 = 0.10f;
  fo.hyst = 0.2f;      // 실물 노드가 fuel 위층을 갖던 때(~2026-11-17)의 기본 fuel_hyst · fuel_vp_min · fuel_vp_clear (1116 녹화 다시 돌리기)
  fo.vp_min = 0.75f;
  fo.vp_clear = 0.21f;
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " 뒤에 값"); return argv[++i]; };
      if (a == "--upper") {
        const std::string u = nx();
        if (u != "fuel" && u != "slot0" && u != "cand") throw std::runtime_error("--upper 는 fuel · slot0 · cand");
        use_fuel = u == "fuel";
        use_cand = u == "cand";
      }
      else if (a == "--cam-pitch") cam_pitch = std::stof(nx());
      else if (a == "--cam-z") cam_z = std::stof(nx());
      else if (a == "--fuel-v") fo.v_eff = std::stof(nx());
      else if (a == "--fuel-w") fo.w_eff = std::stof(nx());
      else if (a == "--fuel-t0") fo.t0 = std::stof(nx());
      else if (a == "--fuel-hyst") fo.hyst = std::stof(nx());
      else if (a == "--fuel-vp-min") fo.vp_min = std::stof(nx());
      else if (a == "--fuel-vp-clear") fo.vp_clear = std::stof(nx());
      else if (a == "--fuel-first") fo.first = std::stoi(nx()) != 0;
      else if (a == "--fuel-time") fo.time = std::stoi(nx()) != 0;
      else if (a == "--fuel-global") fo.global = std::stoi(nx()) != 0;
      else if (a == "--fuel-vp") fo.vp = std::stoi(nx()) != 0;
      else if (a == "--log") logp = nx();
      else if (a == "--depth-stat") depth_stat = true;
      else if (a == "--depth-pix") pix_pre = nx();
      else if (a == "--floor-fit") floor_fit = true;
      else if (a == "--depth-pix-every") pix_every = std::stoi(nx());
      else if (a == "--cand") cand_on = true;
      else if (a == "--cand-dump") { cand_on = true; cand_dump = nx(); }
      else if (a == "--cand-every") cand_every = std::stoi(nx());
      else if (a == "--cand-stages") { cand_on = true; stage_step = std::stoi(nx()); stage_pre = nx(); }   // 이 걸음 뒤 첫 결정의 단계별 그림 재료 (candq::dump_stages)
      else if (a == "--pgm") pgm = nx();
      else if (a == "--pgm-every") pgm_every = std::stoi(nx());
      else throw std::runtime_error("모르는 인자 " + a);
    }
    lowrec::Reader rd(argv[1]);
    if (std::isfinite(cam_pitch)) { rd.hc.cfg.cam_cp = std::cos(cam_pitch); rd.hc.cfg.cam_sp = std::sin(cam_pitch); }
    if (std::isfinite(cam_z)) rd.hc.cfg.cam_z = cam_z;
    limo_fire_explore::RobotEnv robot(rd.hc);
    const sim::EnvCfg& c = robot.cfg();
    lxc::Params lp = lxc::Params::from_cfg(c);
    lp.fuel = fo;
    lp.use_fuel = use_fuel;
    lp.use_cand = use_cand;   // 노드와 같게 (explorer_node.cpp: use_cand true · use_fuel false)
    int rough_step = -1;      // --upper cand: '대충 다 앎' 끝 걸음 (노드는 여기서 끝)
    long n_cand_sum = 0, n_cand0 = 0;
    lxc::Ctl ctl(lp);
    fu::Diag dg;
    ctl.set_diag(&dg);
    fu::Work work;
    cand::Opt copt;
    cand::Mem cmem;
    cand::Work cwork;
    candq::Acc qacc;
    long n_cdec = 0;
    double c_ms = 0, c_ms_max = 0;
    sim::World wd;
    FILE* lf = logp.empty() ? nullptr : std::fopen(logp.c_str(), "w");
    if (lf) std::fprintf(lf, "t_s\tstep\tx\ty\tth\tcause\tcmd_why\ttgt\ttx\tty\tchanged\tjump_m\trobot_tgt_m\tn_clu\tn_node\tcur\tpick\tbest_first\tbest\tcur_cost\tno_goal_why\tn_block\tn_front_reach\tn_front_all\tn_block_nocam\tclosed\tn_cand\tc_clu\tc_dormant\tc_pocket\tc_start_free\n");
    std::vector<uint8_t> dep((size_t)mdp::obs::DEPTH_SIZE);
    limo_fire_explore::Proprio p;
    limo_fire_explore::Scan sc;
    long n_act = 0, d_cmd = 0, n_dec = 0, n_chg = 0, n_nogoal = 0, n_slot = 0, nog_why[6] = {0, 0, 0, 0, 0, 0}, nog_steps = 0, n_steps = 0;
    long cause_n[6] = {0, 0, 0, 0, 0, 0};
    double jump_sum = 0;
    int n_ep = 0;
    int prev_tgt = -2, done = 0, last_dec = 0, cause = 5;   // cause: 0 주기 · 1 통과 · 2 경로 끝 밖 · 3 경로 없음 · 4 막힘 · 5 판 시작
    int sig_prev = 0;
    bool started = false;
    const double dt = (double)c.substeps * c.dt;
    float px = 0, py = 0;
    bool cut = false;
    for (char ch = rd.next(); ch; ch = rd.next()) {
     try {
      if (ch == 'R') {
        float xyz[3], fov;
        lowrec::Window win;
        rd.reset(xyz, &p, &fov, &win);
        robot.reset(xyz[0], xyz[1], xyz[2], p, fov, &win);
        const sim::MapDev& md = robot.map();
        wd.gw = md.gw; wd.gh = md.gh; wd.ox = md.ox; wd.oy = md.oy;
        ctl.restart();
        cmem = cand::Mem();
        ctl.set_geom(md.gw, c.res);
        started = true;
        ++n_ep;
        cause = 5;
        prev_tgt = -2;
        sig_prev = 0;
      } else if (ch == 'S') {
        rd.step(&p, &sc);
        robot.step(p, sc);
        ++n_steps;
        const int dn = robot.done();
        if (dn == mdp::cur::kDeclareOk || dn == mdp::cur::kNoProgress || dn == mdp::cur::kTruncated) { done = dn; continue; }
        // 걸음 뒤 (Ctl::after_step 과 같은 판정 — 까닭만 적으려고 앞서 셈)
        const sim::EnvState& s = robot.state();
        const sim::LowRl& rl = robot.low_rl();
        const sim::LowEp& L = robot.low_ep();
        const sim::LowCmd cm = ctl.cmd();
        if (rl.stall && !sig_prev) cause = 4;
        sig_prev = rl.stall;
        if (cm.follow == 1) {   // fuel 목표 칸을 따라가는 중 (Ctl::after_step 과 같은 조건)
          const bool reached = L.reach_step == s.step && L.reached > 0;
          const bool off = !reached && L.hold != 0;
          const bool nopath = !reached && !off && s.step > last_dec && (s.goal_cell != cm.target || s.p_n <= 0);
          if (reached) cause = 1; else if (off) cause = 2; else if (nopath) cause = 3;
        }
        ctl.after_step(s, rl, L);
        nog_steps += ctl.cmd().follow == 0 && ctl.slot() < 0;
      } else if (ch == 'L') {   // 판 4: 깊이 층 전용 격자 ('D' 뒤)
        std::vector<uint8_t> ld(rd.layer_n());
        rd.layer(ld.data());
        robot.set_layer_depth(ld.data());
      } else if (ch == 'D') {
        rd.depth(dep.data());
        robot.set_depth(dep.data());
        if (depth_stat)
          for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
            float bx, by, bz;
            if (!sim::cam_point_base(c, dep[(size_t)i], i, &bx, &by, &bz)) continue;
            const int kv = sim::cam_voxel(c, bz);
            if (kv < 0) continue;
            const float lh = std::hypot(bx - c.cam_x, by - c.cam_y);
            if (lh > c.cam_obs_range) continue;
            dh[std::min(5, (int)(lh / 0.5f))][kv] += 1;
          }
        if (floor_fit) ffit.add(c, dep.data());
        if (!pix_pre.empty()) {
          const sim::DepthSpec& D = c.depth.spec;
          ++pix_frames;
          for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
            const uint8_t q = dep[(size_t)i];
            if (q == 0 || q == 255) continue;
            const double z = D.min_z + ((double)(q - 1) + 0.5) * ((D.obs_max_z - D.min_z) / 254.0);
            ++pix_valid[i];
            pix_z[i] += z;
            pix_z2[i] += z * z;
            float bx, by, bz;
            if (!sim::cam_point_base(c, q, i, &bx, &by, &bz) || sim::cam_voxel(c, bz) < 0) continue;
            if (std::hypot(bx - c.cam_x, by - c.cam_y) > c.cam_obs_range) continue;
            ++pix_mark[i];
            pix_b[0][i] += bx; pix_b[1][i] += by; pix_b[2][i] += bz;
          }
          if (pix_every > 0 && (pix_frames - 1) % pix_every == 0) {
            std::vector<uint8_t> g((size_t)mdp::obs::DEPTH_SIZE);
            for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) g[(size_t)i] = dep[(size_t)i] == 255 ? 255 : dep[(size_t)i];
            write_pgm8(pix_pre + "_f" + std::to_string(pix_frames - 1) + ".pgm", g.data());   // 장 번호 (판이 여럿이면 걸음 번호가 겹쳐서)
          }
        }
        if (!started || !ctl.due(robot.state())) continue;
        const sim::EnvState& s = robot.state();
        const lxc::View v{&c, &wd, robot.lmap(), robot.trail(), robot.cam_layer(), &s, robot.cand(), (uint32_t)robot.cand_cur(), robot.reach_keep()};
        int why = 0;
        dg = fu::Diag();
        if (cand_on) {
          const cand::In ci{&c, &wd, robot.lmap(), robot.trail(), robot.cam_layer(), &s, [&](int cell) { return ctl.avoided(cell, s.step); }, ctl.cmd().target};
          const bool stage_now = stage_step >= 0 && s.step >= stage_step;
          if (stage_now) { candq::dump_stages(stage_pre, ci, copt, cmem); stage_step = -1; }
          const cand::Out co_ = cand::build(ci, copt, cmem, cwork);
          if (stage_now) candq::dump(stage_pre + "_map", ci, co_, robot.cand());
          ++n_cdec;
          c_ms += co_.ms; c_ms_max = std::max(c_ms_max, co_.ms);
          candq::eval(ci, copt, cwork, co_, robot.cand(), nullptr, s.map_closed != 0, qacc);
          if (!cand_dump.empty() && (n_cdec - 1) % cand_every == 0) candq::dump(cand_dump + "_s" + std::to_string(s.step), ci, co_, robot.cand());
        }
        const sim::LowCmd cmd = ctl.decide(v, work, &why);
        robot.set_cmd(cmd);
        if (use_cand) {
          const cand::Out& co = ctl.cands();
          n_cand_sum += (long)co.c.size();
          n_cand0 += co.c.empty();
          if (rough_step < 0 && ctl.rough_done(s.step)) rough_step = s.step;
        }
        const int cs = s.step % lp.decide_steps == 0 && cause != 5 ? 0 : cause;
        cause_n[cs] += 1;
        ++n_dec;
        last_dec = s.step;
        float tx = NAN, ty = NAN;
        const bool has = cmd.target >= 0 && robot.cell_xy(cmd.target, &tx, &ty);
        const bool chg = cmd.target != prev_tgt && prev_tgt != -2;
        double jump = 0;
        if (chg && has && prev_tgt >= 0) { jump = std::hypot(tx - px, ty - py); jump_sum += jump; }
        n_chg += chg;
        if (why == 2) { ++n_nogoal; if (use_fuel) nog_why[dg.why] += 1; }
        if (why == 1) ++n_slot;
        const sim::EnvState& e = robot.state();
        const Clos cl2 = closure(robot, c.close_radius, false);   // 깊이 카메라 층 빼고 (라이다 지도만)
        const Clos cl = closure(robot, c.close_radius);
        if (lf)
          std::fprintf(lf, "%.1f\t%d\t%.2f\t%.2f\t%.1f\t%s\t%s\t%d\t%.2f\t%.2f\t%d\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%s\t%d\t%d\t%ld\t%d\t%d", e.step * dt, e.step, e.x, e.y, e.th * 57.2958,
                       cs == 0 ? "주기" : cs == 1 ? "통과" : cs == 2 ? "경로끝밖" : cs == 3 ? "경로없음" : cs == 4 ? "막힘" : "판시작", why == 0 ? (use_cand ? "cand" : "fuel") : why == 1 ? "slot0" : "없음",
                       cmd.target, has ? tx : NAN, has ? ty : NAN, chg ? 1 : 0, jump, has ? std::hypot(tx - e.x, ty - e.y) : NAN, dg.n_clu, dg.n_node, dg.cur, dg.pick,
                       dg.best_first, dg.best, dg.cur_cost, why == 2 && use_fuel ? kWhyNo[dg.why] : "-", cl.r.n_block, cl.r.n_cells, cl.n_front_all, cl2.r.n_block, (int)e.map_closed);
        if (lf) {
          const cand::Out& co = ctl.cands();
          if (use_cand) std::fprintf(lf, "\t%zu\t%d\t%d\t%d\t%d\n", co.c.size(), co.n_clu, co.n_dormant, co.n_pocket, co.start_free);
          else std::fprintf(lf, "\t-\t-\t-\t-\t-\n");
        }
        if (!pgm.empty() && pgm_every > 0 && (n_dec - 1) % pgm_every == 0) write_ppm(pgm + (n_ep > 1 ? "_e" + std::to_string(n_ep) : std::string()) + "_" + std::to_string(e.step) + ".ppm", robot, cmd.target);   // 판이 여럿이면 판 번호 (걸음 번호가 겹침)
        if (has) { px = tx; py = ty; }
        prev_tgt = cmd.target;
        cause = 0;
      } else if (ch == 'A') {   // 노드 차례: set_slot · low_prepare → low_act (명령)
        float v0, w0, v, w;
        lowrec::Goal rg;
        rd.act(&v0, &w0, &rg);
        robot.set_slot(ctl.slot());   // 규칙 추종기: 경로 계획 후보 (노드와 같은 차례)
        robot.low_prepare();
        robot.low_act(&v, &w);
        ++n_act;
        d_cmd += std::memcmp(&v, &v0, 4) != 0 || std::memcmp(&w, &w0, 4) != 0;
      } else {
        throw std::runtime_error(std::string("녹화 사건 종류를 모름: ") + ch);
      }
     } catch (const std::runtime_error& ex) {
      if (std::string(ex.what()) != "녹화가 짧음") throw;
      cut = true;   // 노드를 끌 때 마지막 사건이 덜 쓰인 녹화
      break;
     }
    }
    if (cut || rd.truncated) std::printf("녹화 끝 사건이 잘림 (노드를 끌 때) — 거기까지만\n");
    if (lf) std::fclose(lf);
    const double T = n_steps * dt;
    std::printf("위층 %s (fuel 시간 %d 전역 %d 시점 %d hyst %.2f) · 걸음 %ld (%.1f s) · 판 끝 %d\n", use_cand ? "cand" : use_fuel ? "fuel" : "slot0", fo.time, fo.global, fo.vp, fo.hyst, n_steps, T, done);
    std::printf("결정 %ld (주기 %ld · 통과 %ld · 경로 끝 밖 %ld · 경로 없음 %ld · 막힘 %ld · 판 시작 %ld) · 목표 바뀜 %ld (%.1f /분, 바뀔 때 평균 %.2f m 뜀) · slot0 %ld · 목표 없음 결정 %ld (%.0f %%) · 목표 없음 걸음 %.0f %%\n",
                n_dec, cause_n[0], cause_n[1], cause_n[2], cause_n[3], cause_n[4], cause_n[5], n_chg, T > 0 ? n_chg * 60.0 / T : 0.0, n_chg ? jump_sum / n_chg : 0.0, n_slot, n_nogoal,
                n_dec ? 100.0 * n_nogoal / n_dec : 0.0, n_steps ? 100.0 * nog_steps / n_steps : 0.0);
    if (cand_on && n_cdec) {
      candq::print(qacc, "실물 녹화");
      std::printf("후보 생성 시간 (이 기기): 평균 %.2f ms · 최대 %.2f ms (%ld 결정)\n", c_ms / n_cdec, c_ms_max, n_cdec);
    }
    if (depth_stat) {
      std::printf("깊이 카메라 층 표시 점 (장마다 합): 행 = 센서 수평 거리 0.5 m 칸, 열 = 높이 칸 (%.3f m 부터 %.4f m 씩)\n", c.cam_min_h, (c.cam_max_h - c.cam_min_h) / c.cam_nz);
      for (int r = 0; r < 6; ++r) { std::printf("  %.1f-%.1f m:", r * 0.5, r * 0.5 + 0.5); for (int k = 0; k < 8; ++k) std::printf(" %7ld", dh[r][k]); std::printf("\n"); }
    }
    if (floor_fit) {
      const sim::FloorFit::Res fr = ffit.fit(c);
      if (!fr.ok) std::printf("바닥 맞춤 (--floor-fit): 바닥 점이 모자람 (%zu)\n", ffit.size());
      else {
        const double cp0 = sim::cfg_pitch(c);
        std::printf("바닥 맞춤 (--floor-fit): 숙임 %.4f rad (%.2f°) · 높이 %.4f m · 잔차 %.4f m · 점 %ld / %ld — 녹화 설정 숙임 %.4f rad (%.2f°) · 높이 %.4f m\n", fr.pitch,
                    fr.pitch * 57.2958, fr.h, fr.rms, fr.n, fr.n_all, cp0, cp0 * 57.2958, c.cam_z);
        // 녹화 설정으로 본 맞춘 바닥의 높이 (센서 앞 수평 거리 0.5 m 마다, 광축 위 열): 띠 아래 끝 cam_min_h 를 넘는 거리부터 바닥이 층에 찍힘
        std::printf("  맞춘 바닥이 녹화 설정으로는 높이 (cm) — 수평 0.5 · 1.0 · 1.5 · 2.0 · 2.5 m:");
        for (int k = 1; k <= 5; ++k) {
          const double d = 0.5 * k;   // 맞춘 카메라에서 바닥 점 (센서 앞 수평 d, 높이 0)
          const double X = d * std::cos(fr.pitch) + fr.h * std::sin(fr.pitch), Z = d * std::sin(fr.pitch) - fr.h * std::cos(fr.pitch);   // 센서 앞 · 위 (맞춘 숙임 — cam_point_uv 의 역)
          std::printf(" %.1f", 100.0 * (c.cam_z + (c.cam_cp * Z - c.cam_sp * X)));
        }
        std::printf(" (띠 아래 끝 %.1f cm)\n", c.cam_min_h * 100.0);
      }
    }
    if (!pix_pre.empty() && pix_frames > 0) {
      const int N = mdp::obs::DEPTH_SIZE;
      std::vector<uint8_t> gm((size_t)N), gz((size_t)N);
      FILE* tf = std::fopen((pix_pre + "_pix.tsv").c_str(), "w");
      if (tf) std::fprintf(tf, "row\tcol\tmark_frac\tvalid_frac\tmean_z_m\tsd_z_m\tbx\tby\tbz\n");
      long n_always = 0;
      for (int i = 0; i < N; ++i) {
        const double fm = (double)pix_mark[i] / pix_frames;
        gm[(size_t)i] = (uint8_t)std::lround(255.0 * fm);
        const double mz = pix_valid[i] ? pix_z[i] / pix_valid[i] : 0.0;
        gz[(size_t)i] = pix_valid[i] ? (uint8_t)std::lround(std::min(255.0, 255.0 * mz / 3.0)) : 0;
        n_always += fm >= 0.9;
        if (tf && pix_mark[i] > 0) {
          const double sd = pix_valid[i] ? std::sqrt(std::max(0.0, pix_z2[i] / pix_valid[i] - mz * mz)) : 0.0;
          std::fprintf(tf, "%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", i / mdp::obs::DEPTH_W, i % mdp::obs::DEPTH_W, fm, (double)pix_valid[i] / pix_frames, mz, sd,
                       pix_b[0][i] / pix_mark[i], pix_b[1][i] / pix_mark[i], pix_b[2][i] / pix_mark[i]);
        }
      }
      if (tf) std::fclose(tf);
      write_pgm8(pix_pre + "_mark.pgm", gm.data());
      write_pgm8(pix_pre + "_mean.pgm", gz.data());
      std::printf("깊이 픽셀 (--depth-pix): 장 %ld · 장의 90 %% 이상에서 층에 찍히는 픽셀 %ld / %d → %s_{mark,mean}.pgm · _pix.tsv\n", pix_frames, n_always, N, pix_pre.c_str());
    }
    if (use_cand)
      std::printf("후보 (cand): 결정마다 평균 %.2f 개 · 후보 0 결정 %ld / %ld · '대충 다 앎' 끝 %s\n", n_dec ? (double)n_cand_sum / n_dec : 0.0, n_cand0, n_dec,
                  rough_step >= 0 ? (std::to_string(rough_step) + " 걸음 (" + std::to_string(rough_step * dt).substr(0, 5) + " s)").c_str() : "없음");
    if (use_fuel) std::printf("목표 없음 까닭: 시작 칸 없음 %ld · 덩어리 없음 %ld · 자격 칸 없음 %ld · 모두 피함 %ld · 시점 자격 없음 %ld\n", nog_why[1], nog_why[2], nog_why[3], nog_why[4], nog_why[5]);
    std::printf("녹화 명령과 다른 명령 %ld / %ld (녹화 판 %d — 같은 위층 설정이면 0 이어야 함)\n", d_cmd, n_act, rd.version);
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "low_explore_replay: %s\n", ex.what());
    return 2;
  }
}
