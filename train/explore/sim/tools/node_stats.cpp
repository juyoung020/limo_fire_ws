// 합친 탐사 노드 녹화(lowrec — 실물 low_rl record:= 또는 시뮬 room_run --rec)를 RobotEnv + 위층(scripts/low_explore_ctl.h)으로 노드와 같은 차례로 다시 돌려
// 실물·시뮬이 같은 잣대로 견줄 값을 낸다: 지도 자란 넓이 · 프론티어 칸(로봇 발밑 포함) · 후보 · 목표 차례 · 정책 명령 분포(v≈0 몫·ω 부호 바뀜) · 명령 대 바퀴·자이로 ·
// 라이다 빔(유효 몫 · 앞/옆/뒤) · 끝 시각. 궤적은 녹화 그대로 — 다시 낸 아래층 명령이 녹화 명령과 비트 같으면 위층 결정도 노드와 같다는 뜻.
//   node_stats <녹화.bin> [--upper fuel|slot0] [--csv 걸음.csv] [--label 이름]
// 노드 차례 (src/limo_fire/limo_fire_rl/src/low_rl_node.cpp on_scan): R 또는 S(step · 판 끝 · after_step) → D(set_depth → 위층 결정) → A(set_slot · low_prepare · low_act)
// 다른 점: 노드는 다시 켤 때(paused → running) 위층을 바로 다시 결정(force_redo)하는데 녹화에는 켠 때가 없다 — 멈췄던 녹화는 그 결정 때만 다를 수 있음(명령 비교로 드러남).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "frontier.h"
#include "limo_fire_rl/robot_env.h"
#include "low_explore_ctl.h"
#include "low_node_rec.h"

namespace {
struct Row {
  int k;
  float x, y, th, wv, ww, gyro, cmd_v, cmd_w;
  int goal, follow, dec, why;
  float gdist, gx, gy;
  float rgx = NAN, rgy = NAN;   // 녹화에 남은 노드의 아래층 목적지 (세계) — 판단 모델 위층처럼 다시 못 내는 위층의 목표
  int known, freec, occ, front, front_near, cand_n, ep;
  int scan_n, scan_ok, ok_front, n_front, ok_side, n_side;
  float rmin;
  int done;
};
const float kPi = 3.14159265f;
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "node_stats <녹화.bin> [--upper fuel|slot0] [--csv 걸음.csv] [--label 이름] [--map 지도.ppm 초]\n");
    return 2;
  }
  std::string csv, label = argv[1], upper = "fuel", map_out, floor_csv, vox_path;
  float vox_tau = 0.075f;
  float map_at = -1.f;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    else if (a == "--label" && i + 1 < argc) label = argv[++i];
    else if (a == "--upper" && i + 1 < argc) upper = argv[++i];
    else if (a == "--floor" && i + 1 < argc) floor_csv = argv[++i];
    else if (a == "--vox" && i + 1 < argc) vox_path = argv[++i];
    else if (a == "--vox-tau" && i + 1 < argc) vox_tau = std::stof(argv[++i]);
    else if (a == "--map" && i + 2 < argc) { map_out = argv[++i]; map_at = std::stof(argv[++i]); }
    else { std::fprintf(stderr, "모르는 인자 %s\n", a.c_str()); return 2; }
  }
  if (upper != "fuel" && upper != "slot0") { std::fprintf(stderr, "--upper 는 fuel · slot0\n"); return 2; }
  try {
    lowrec::Reader rd(argv[1]);
    limo_fire_rl::RobotEnv robot(rd.hc);
    const sim::EnvCfg& c = robot.cfg();
    // 위층 값: 노드 기본 (config/low_rl.yaml — fuel 시간·전역·시점, v 0.981 · ω 3.232 · t0 0.10)
    lxc::Params lp = lxc::Params::from_cfg(c);
    lp.fuel.time = lp.fuel.global = lp.fuel.vp = true;
    lp.fuel.v_eff = 0.981f;
    lp.fuel.w_eff = 3.232f;
    lp.fuel.t0 = 0.10f;
    lp.use_fuel = upper == "fuel";
    lxc::Ctl ctl(lp);
    fu::Work work;
    sim::World w;
    std::vector<uint8_t> dep((size_t)mdp::obs::DEPTH_SIZE);
    limo_fire_rl::Proprio p{};
    limo_fire_rl::Scan s;
    std::vector<Row> rows;
    Row cur{};
    long d_cmd = 0, n_act = 0;
    constexpr int kAB = 11;   // 라이다 각 구간 20° (−110° … +110°)
    double ab_n[kAB] = {}, ab_ok[kAB] = {};
    double rh[13] = {};       // 유효 빔 거리 구간 1 m (0…12 m+)
    double dh[6] = {};        // 깊이 관측 바이트: 구멍 0 · 1~42 · 43~85 · 86~170 · 171~254 · 255
    // 깊이 층 칸 나눔 (1.5 m 안): 라이다 막힘(로봇 지도 kOcc)에서 2 칸 안 · 지금 카메라 시야(수평 ±33°, 2.5 m) 안 · 시야 밖(지금 장으로 못 지움 — 쌓인 것)
    double cam_lid = 0, cam_fov = 0;
    double cam_near = 0, cam_n = 0;   // 깊이 카메라 장애물 층이 로봇 1.5 m 안에 찍은 칸 수 (걸음 평균)
    // 깊이 점 높이 (sim::cam_point_base — 층과 같은 셈): 센서 수평 0.5~2.5 m 점 중 높이 구간 몫, 바닥 같은 점(|z| < 0.15)의 높이 = a + b·거리 최소제곱
    double zb[5] = {};   // < −0.03 · −0.03~0.03 · 0.03~0.08 · 0.08~0.25 · > 0.25
    double fs[5] = {};   // 바닥 같은 점 Σ1 Σd Σd² Σz Σdz
    // 바닥 구멍 (sensor_model.yaml depth.holes.floor_cos 구간): 픽셀 광선이 바닥 면(z = 0)에 닿을 깊이 z_f 가 [min_z, obs_max) 이고 값이 구멍이거나 z_f 근처(바닥이 보임)인 픽셀만.
    // 구멍 / (구멍 + 바닥 보임). 앞에 물체가 보인 픽셀(z < z_f − 여유)은 뺌 — 실물·시뮬 같은 셈
    double fh_hole[8] = {}, fh_tot[8] = {};
    // --vox: 깊이 점을 지도 칸 · 높이 띠별로 셈 (몸통 띠 아래·라이다 평면 아래 · 몸통 띠 위쪽 · 좌판 · 상판). 자세는 τ(--vox-tau, 기본 0.075 s — 깊이 영상 늦음 맞춤값)만큼
    // 자이로로 되돌림. 칸마다 그 칸을 시야에 둔 걸음 수(see — 띠 점이 없었던 걸음 포함)도 셈 → 점 수 / see = 늘 보이는 물체인가. 실물 방 장면에 라이다가 못 보는 가구를 넣는 근거
    constexpr int kVoxB = 5;
    const float vb[kVoxB + 1] = {0.03f, 0.10f, 0.25f, 0.40f, 0.60f, 0.85f};   // [0.03,0.10) 라이다 평면(0.111) 아래 · [0.10,0.25) · [0.25,0.40) · [0.40,0.60) 좌판 · [0.60,0.85) 상판
    std::vector<uint32_t> vox, vsee;
    if (!vox_path.empty()) { vox.assign((size_t)c.size * c.size * kVoxB, 0u); vsee.assign((size_t)c.size * c.size, 0u); }
    // 깊이 시각 어긋남: 빨리 돌 때(|자이로| > 1.5 rad/s) 띠 안 깊이 점을 τ 만큼 앞 자세(θ − ω·τ)로 놓았을 때 로봇 지도 막힘 칸(한 칸 둘레)에 드는 몫 — 가장 큰 τ = 깊이가 자세보다 늦은 시간
    constexpr int kTau = 16;   // τ = 0, 0.025, …, 0.375 s
    double lat_hit[kTau] = {}, lat_n = 0;
    // --floor: 걸음마다 IMU roll·pitch 와 바닥 같은 깊이 점(센서 수평 0.5~1.5 m, |높이| < 0.10 m)의 높이 = a + b·거리 (바닥 기울어 보임 ↔ 몸 기울기) 와 띠 안 바닥 점 수
    FILE* ff = floor_csv.empty() ? nullptr : std::fopen(floor_csv.c_str(), "w");
    if (ff) std::fprintf(ff, "k,roll,pitch,ax,wv,gyro,n,a,b,n_band\n");
    double zh[4][30] = {};   // 거리 0.5 m 구간 넷(0.5~2.5)마다 높이 1 cm 칸(−10~+20 cm) 개수 — 최빈값 = 바닥 높이
    int first_bad = -1, ep_i = -1;
    const float dt = (float)c.substeps * c.dt;
    const float near_r = 0.6f;    // 발밑 프론티어: 로봇 중심 이 안 (몸통 0.32 × 0.22 외접 0.19 + 라이다 최소 거리 둘레) — 잠정
    const auto map_stats = [&](Row& r) {
      const mdp::MapView mv{robot.lmap(), c.size, w.gw, w.gh};
      r.known = r.freec = r.occ = r.front = r.front_near = 0;
      const int R = (int)std::ceil(near_r / c.res);
      const int cx = (int)std::floor((r.x - w.ox) / c.res), cy = (int)std::floor((r.y - w.oy) / c.res);
      for (int y = 0; y < w.gh; ++y)
        for (int x = 0; x < w.gw; ++x) {
          const uint8_t k = mdp::cls_at(mv, x, y);
          if (k == mdp::kUnk) continue;
          ++r.known;
          if (k == mdp::kOcc) { ++r.occ; continue; }
          ++r.freec;
          if (mdp::frontier_cell(mv, x, y)) {
            ++r.front;
            const int dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= R * R) ++r.front_near;
          }
        }
      r.cand_n = 0;
      for (int j = 0; j < mdp::hier::K; ++j) r.cand_n += robot.cand()[j] >= 0;
    };
    // 첫 판 그 시각의 로봇 지도 그림 (PPM: 흰 빈 · 검정 막힘 · 회색 모름 · 빨강 프론티어 · 파랑 로봇 · 초록 목표)
    bool map_done = false;
    const auto dump_map = [&](const Row& r) {
      const mdp::MapView mv{robot.lmap(), c.size, w.gw, w.gh};
      std::vector<uint8_t> px((size_t)w.gw * w.gh * 3);
      for (int y = 0; y < w.gh; ++y)
        for (int x = 0; x < w.gw; ++x) {
          const uint8_t k = mdp::cls_at(mv, x, y);
          uint8_t* q = &px[((size_t)(w.gh - 1 - y) * w.gw + x) * 3];
          const uint8_t g = k == mdp::kUnk ? 128 : k == mdp::kOcc ? 0 : 255;
          q[0] = q[1] = q[2] = g;
          if (k == mdp::kFree && mdp::frontier_cell(mv, x, y)) { q[0] = 255; q[1] = q[2] = 0; }
          if (robot.cam_layer() && robot.cam_layer()[(size_t)y * c.size + x]) { q[0] = 255; q[1] = 150; q[2] = 0; }   // 주황 = 깊이 카메라 장애물 층
        }
      const auto dot = [&](float wx, float wy, uint8_t R, uint8_t G, uint8_t B) {
        const int cx = (int)std::floor((wx - w.ox) / c.res), cy = (int)std::floor((wy - w.oy) / c.res);
        for (int dy = -2; dy <= 2; ++dy)
          for (int dx = -2; dx <= 2; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= w.gw || y >= w.gh) continue;
            uint8_t* q = &px[((size_t)(w.gh - 1 - y) * w.gw + x) * 3];
            q[0] = R; q[1] = G; q[2] = B;
          }
      };
      for (const Row& o : rows) if (o.ep == r.ep) dot(o.x, o.y, 0, 0, 255);
      if (r.goal >= 0) dot(r.gx, r.gy, 0, 200, 0);
      FILE* f = std::fopen(map_out.c_str(), "wb");
      if (!f) throw std::runtime_error("지도 그림을 못 엶: " + map_out);
      std::fprintf(f, "P6\n%d %d\n255\n", w.gw, w.gh);
      std::fwrite(px.data(), 1, px.size(), f);
      std::fclose(f);
      map_done = true;
    };
    const auto scan_stats = [&](Row& r) {
      r.scan_n = (int)s.ranges.size();
      r.scan_ok = r.ok_front = r.n_front = r.ok_side = r.n_side = 0;
      r.rmin = 99.f;
      for (int i = 0; i < r.scan_n; ++i) {
        const float an = s.angle_min + (float)i * s.angle_increment, rr = s.ranges[(size_t)i];
        const bool ok = std::isfinite(rr) && rr > 0.f;
        r.scan_ok += ok;
        if (std::fabs(an) < kPi / 3.f) { ++r.n_front; r.ok_front += ok; }
        else { ++r.n_side; r.ok_side += ok; }
        if (ok) r.rmin = std::min(r.rmin, rr);
        const int ab = std::min(kAB - 1, std::max(0, (int)std::floor((an + 1.9198622f) / (3.8397244f / kAB))));
        ab_n[ab] += 1; ab_ok[ab] += ok;
        if (ok) rh[std::min(12, (int)rr)] += 1;
      }
    };
    bool cut = false;   // 노드를 끈 순간 버퍼가 덜 써진 녹화: 마지막 사건이 잘림 — 거기까지만
    for (char ch = rd.next(); ch; ch = rd.next()) {
      try {
      if (ch == 'R') {
        float xyz[3], fov;
        lowrec::Window win;
        rd.reset(xyz, &p, &fov, &win);
        robot.reset(xyz[0], xyz[1], xyz[2], p, fov, &win);
        w.gw = win.gw; w.gh = win.gh; w.ox = win.ox; w.oy = win.oy;
        ctl.restart();
        ctl.set_geom(win.gw, c.res);
        ++ep_i;
        cur = Row{};
        cur.x = xyz[0]; cur.y = xyz[1]; cur.th = xyz[2];
        cur.scan_n = 0;
        cur.rmin = 99.f;
      } else if (ch == 'S') {
        rd.step(&p, &s);
        robot.step(p, s);
        cur = Row{};
        cur.k = robot.steps();
        cur.x = s.x; cur.y = s.y; cur.th = s.th;
        cur.wv = p.wheel_v; cur.ww = p.wheel_w; cur.gyro = p.gyro_z;
        scan_stats(cur);
        const int dn = robot.done();
        cur.done = dn;
        if (!dn) ctl.after_step(robot.state(), robot.low_rl(), robot.low_ep());
        if (dn == mdp::cur::kDeclareOk || dn == mdp::cur::kNoProgress || dn == mdp::cur::kTruncated) {   // 노드는 판 끝 걸음에 D·O·A 를 안 씀 — 그 걸음을 여기서 넣음 (명령 0)
          cur.goal = -1;
          cur.gdist = -1.f;
          cur.ep = ep_i;
          map_stats(cur);
          rows.push_back(cur);
          if (!map_out.empty() && !map_done && cur.ep == 0) dump_map(cur);
        }
      } else if (ch == 'L') {   // 판 4: 깊이 층 전용 격자 ('D' 뒤)
        std::vector<uint8_t> ld(rd.layer_n());
        rd.layer(ld.data());
        robot.set_layer_depth(ld.data());
      } else if (ch == 'D') {
        rd.depth(dep.data());
        robot.set_depth(dep.data());
        for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
          const int v = dep[(size_t)i];
          dh[v == 0 ? 0 : v < 43 ? 1 : v < 86 ? 2 : v < 171 ? 3 : v < 255 ? 4 : 5] += 1;
        }
        {
          const sim::DepthSpec& D = c.depth.spec;
          const float su = (float)D.w / (float)mdp::obs::DEPTH_W, sv = (float)D.h / (float)mdp::obs::DEPTH_H;
          for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
            const float u = ((float)(i % mdp::obs::DEPTH_W) + 0.5f) * su - 0.5f, v = ((float)(i / mdp::obs::DEPTH_W) + 0.5f) * sv - 0.5f;
            if (!sim::depth_valid(D, u, v)) continue;
            const float Yd = -(u - D.cx) / D.fx, Zd = -(v - D.cy) / D.fy;   // 광축 깊이 1 당 센서 FLU (cam_point_base 와 같음)
            const float dx = c.cam_cp + c.cam_sp * Zd, dz = c.cam_cp * Zd - c.cam_sp;   // base 방향 (광축 깊이 1 당)
            if (!(dz < 0.f)) continue;
            const float zf = -c.cam_z / dz;   // 바닥에 닿는 광축 깊이
            if (zf < D.min_z || zf >= D.obs_max_z) continue;
            const float cs = -dz / std::sqrt(dx * dx + Yd * Yd + dz * dz);
            int b = -1;
            for (int k = 0; k < sim::kDepthFloorBins; ++k) if (cs >= c.depth.floor_cos[k] && cs < c.depth.floor_cos[k + 1]) b = k;
            if (b < 0) continue;
            const uint8_t q = dep[(size_t)i];
            if (q == 0) { fh_hole[b] += 1; fh_tot[b] += 1; continue; }
            const float z = D.min_z + ((float)(q - 1) + 0.5f) * ((D.obs_max_z - D.min_z) / 254.f);
            if (q != 255 && std::fabs(z - zf) <= 0.05f * zf + 0.02f) fh_tot[b] += 1;   // 바닥이 보임 (그 밖 — 앞 물체·바닥 너머 — 는 안 셈)
          }
        }
        for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
          float bx, by, bz;
          if (!sim::cam_point_base(c, dep[(size_t)i], i, &bx, &by, &bz)) continue;
          const float lh = std::hypot(bx - c.cam_x, by - c.cam_y);
          if (lh < 0.5f || lh > 2.5f) continue;
          zb[bz < -0.03f ? 0 : bz < 0.03f ? 1 : bz < 0.08f ? 2 : bz < 0.25f ? 3 : 4] += 1;
          { const int hb = (int)std::floor((bz + 0.10f) / 0.01f); if (hb >= 0 && hb < 30) zh[std::min(3, (int)((lh - 0.5f) / 0.5f))][hb] += 1; }
          if (std::fabs(bz) < 0.15f) { fs[0] += 1; fs[1] += lh; fs[2] += lh * lh; fs[3] += bz; fs[4] += lh * bz; }
        }
        if (!vox.empty()) {
          const float th = cur.th - p.gyro_z * vox_tau, cs = std::cos(th), sn = std::sin(th);
          std::vector<uint8_t> seen_now((size_t)c.size * c.size, 0);
          for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
            const uint8_t q = dep[(size_t)i];
            if (q == 0) continue;
            float bx, by, bz;
            if (q == 255) {   // 3 m 넘음: 그 광선이 지나는 칸 표시용으로 3 m 점
              if (!sim::cam_point_base(c, 254, i, &bx, &by, &bz)) continue;
            } else if (!sim::cam_point_base(c, q, i, &bx, &by, &bz)) continue;
            const float lh = std::hypot(bx - c.cam_x, by - c.cam_y);
            // 광선이 지난 칸 = 본 칸 (0.5 칸 간격)
            const int n = (int)std::ceil(lh / (0.5f * c.res));
            for (int k = 1; k <= n; ++k) {
              const float t = (float)k / (float)n, hx = c.cam_x + t * (bx - c.cam_x), hy = c.cam_y + t * (by - c.cam_y);
              const int x = (int)std::floor((cur.x + cs * hx - sn * hy - w.ox) / c.res), y = (int)std::floor((cur.y + sn * hx + cs * hy - w.oy) / c.res);
              if (x >= 0 && y >= 0 && x < w.gw && y < w.gh) seen_now[(size_t)y * c.size + x] = 1;
            }
            if (q == 255 || lh > 2.5f) continue;   // 층과 같은 표시 거리 (obstacle_max_range 2.5 m)
            int b = -1;
            for (int k = 0; k < kVoxB; ++k) if (bz >= vb[k] && bz < vb[k + 1]) b = k;
            if (b < 0) continue;
            const int x = (int)std::floor((cur.x + cs * bx - sn * by - w.ox) / c.res), y = (int)std::floor((cur.y + sn * bx + cs * by - w.oy) / c.res);
            if (x >= 0 && y >= 0 && x < w.gw && y < w.gh) vox[((size_t)y * c.size + x) * kVoxB + b] += 1;
          }
          for (size_t k = 0; k < seen_now.size(); ++k) vsee[k] += seen_now[k];
        }
        if (ff) {
          double S0 = 0, S1 = 0, S2 = 0, Sz = 0, Sdz = 0;
          int nb = 0;
          for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
            float bx, by, bz;
            if (!sim::cam_point_base(c, dep[(size_t)i], i, &bx, &by, &bz)) continue;
            const float lh = std::hypot(bx - c.cam_x, by - c.cam_y);
            if (lh < 0.5f || lh > 1.5f || std::fabs(bz) > 0.10f) continue;
            S0 += 1; S1 += lh; S2 += lh * lh; Sz += bz; Sdz += lh * bz;
            nb += bz >= c.cam_min_h;
          }
          const double det = S0 * S2 - S1 * S1, b = det > 0 ? (S0 * Sdz - S1 * Sz) / det : 0.0, a0 = S0 > 0 ? (Sz - b * S1) / S0 : 0.0;
          std::fprintf(ff, "%d,%.5f,%.5f,%.3f,%.3f,%.3f,%.0f,%.4f,%.4f,%d\n", robot.steps(), p.imu[5], p.imu[6], p.imu[0], p.wheel_v, p.gyro_z, S0, a0, b, nb);
        }
        if (std::fabs(p.gyro_z) > 1.5f) {
          const mdp::MapView mv{robot.lmap(), c.size, w.gw, w.gh};
          const auto occ_near = [&](float wx, float wy) {
            const int cx = (int)std::floor((wx - w.ox) / c.res), cy = (int)std::floor((wy - w.oy) / c.res);
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const int x = cx + dx, y = cy + dy;
                if (x >= 0 && y >= 0 && x < w.gw && y < w.gh && mdp::cls_at(mv, x, y) == mdp::kOcc) return true;
              }
            return false;
          };
          for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) {
            float bx, by, bz;
            if (!sim::cam_point_base(c, dep[(size_t)i], i, &bx, &by, &bz) || sim::cam_voxel(c, bz) < 0) continue;
            for (int k = 0; k < kTau; ++k) {
              const float th = cur.th - p.gyro_z * 0.025f * (float)k;
              const float cs = std::cos(th), sn = std::sin(th);
              lat_hit[k] += occ_near(cur.x + cs * bx - sn * by, cur.y + sn * bx + cs * by);
            }
            lat_n += 1;
          }
        }
        if (robot.cam_layer()) {
          const int R = (int)std::ceil(1.5f / c.res);
          const int cx = (int)std::floor((cur.x - w.ox) / c.res), cy = (int)std::floor((cur.y - w.oy) / c.res);
          long k = 0;
          for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
              const int x = cx + dx, y = cy + dy;
              if (!(x >= 0 && y >= 0 && x < w.gw && y < w.gh && dx * dx + dy * dy <= R * R && robot.cam_layer()[(size_t)y * c.size + x])) continue;
              ++k;
              bool lid = false;
              const mdp::MapView mv{robot.lmap(), c.size, w.gw, w.gh};
              for (int ey = -2; ey <= 2 && !lid; ++ey)
                for (int ex = -2; ex <= 2 && !lid; ++ex) {
                  const int u = x + ex, v = y + ey;
                  lid = u >= 0 && v >= 0 && u < w.gw && v < w.gh && mdp::cls_at(mv, u, v) == mdp::kOcc;
                }
              cam_lid += lid;
              const float bx = w.ox + ((float)x + 0.5f) * c.res - cur.x, by = w.oy + ((float)y + 0.5f) * c.res - cur.y;
              const float br = std::remainder(std::atan2(by, bx) - cur.th, 6.2831853f);
              cam_fov += std::fabs(br) < 0.58f && std::hypot(bx, by) <= 2.5f;
            }
          cam_near += k;
          cam_n += 1;
        }
        if (ctl.due(robot.state())) {
          const lxc::View v{&c, &w, robot.lmap(), robot.trail(), robot.cam_layer(), &robot.state(), robot.cand(), (uint32_t)robot.cand_cur()};
          int why = 0;
          const sim::LowCmd cmd = ctl.decide(v, work, &why);
          robot.set_cmd(cmd);
          cur.dec = 1;
          cur.why = why;
        }
      } else if (ch == 'A') {   // 노드 차례: set_slot · low_prepare → low_act (명령)
        float v0, w0, v, wv;
        lowrec::Goal rg;
        rd.act(&v0, &w0, &rg);
        if (rg.kind == 1) { cur.rgx = rg.x; cur.rgy = rg.y; }   // 녹화의 노드 목적지 (세계)
        else if (rg.kind == 2) { cur.rgx = cur.x + rg.x * std::cos(cur.th + rg.y); cur.rgy = cur.y + rg.x * std::sin(cur.th + rg.y); }   // 판 1: 지도 자세 기준
        else cur.rgx = cur.rgy = NAN;
        robot.set_slot(ctl.slot());   // 규칙 추종기: 경로 계획 후보 (노드와 같은 차례)
        robot.low_prepare();
        robot.low_act(&v, &wv);
        ++n_act;
        if (std::memcmp(&v, &v0, 4) != 0 || std::memcmp(&wv, &w0, 4) != 0) { if (first_bad < 0) first_bad = (int)n_act - 1; ++d_cmd; }
        cur.cmd_v = v0; cur.cmd_w = w0;
        const sim::LowCmd& lc = robot.low_cmd();
        cur.goal = lc.target; cur.follow = lc.follow;
        cur.gx = cur.gy = 0.f;
        cur.gdist = lc.target >= 0 && robot.cell_xy(lc.target, &cur.gx, &cur.gy) ? std::hypot(cur.gx - cur.x, cur.gy - cur.y) : -1.f;
        cur.ep = ep_i;
        map_stats(cur);
        rows.push_back(cur);
        if (!map_out.empty() && !map_done && cur.ep == 0 && cur.k * (float)c.substeps * c.dt >= map_at) dump_map(cur);
        cur.dec = 0;
      } else {
        throw std::runtime_error(std::string("녹화 사건 종류를 모름: ") + ch);
      }
      } catch (const std::runtime_error& ex) {
        if (std::string(ex.what()) != "녹화가 짧음") throw;
        cut = true;
        break;
      }
    }
    if (cut || rd.truncated) std::fprintf(stderr, "node_stats: 녹화 끝이 잘림 — 온전한 걸음까지만\n");
    if (rows.empty()) throw std::runtime_error("걸음이 없음");
    if (!csv.empty()) {
      FILE* f = std::fopen(csv.c_str(), "w");
      if (!f) throw std::runtime_error("csv 를 못 엶: " + csv);
      std::fprintf(f, "ep,k,t,x,y,th,wv,ww,gyro,cmd_v,cmd_w,goal,gx,gy,rgx,rgy,follow,gdist,dec,why,known_m2,free_m2,occ_m2,front,front_near,cand_n,scan_n,scan_ok,ok_front,n_front,ok_side,n_side,rmin,done\n");
      const float a2 = c.res * c.res;
      for (const Row& r : rows)
        std::fprintf(f, "%d,%d,%.1f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.2f,%.2f,%.3f,%.3f,%d,%.3f,%d,%d,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%d\n", r.ep, r.k, r.k * dt, r.x, r.y, r.th, r.wv,
                     r.ww, r.gyro, r.cmd_v, r.cmd_w, r.goal, r.gx, r.gy, r.rgx, r.rgy, r.follow, r.gdist, r.dec, r.why, r.known * a2, r.freec * a2, r.occ * a2, r.front, r.front_near, r.cand_n, r.scan_n,
                     r.scan_ok, r.ok_front, r.n_front, r.ok_side, r.n_side, r.rmin, r.done);
      std::fclose(f);
    }
    // 요약: 판(R 사건)마다 — 노드는 판이 끝난 뒤 다시 켜면 새 판을 시작한다
    const float a2 = c.res * c.res, vmax = c.act_v_fwd, wmax = c.act_w_max;
    size_t b = 0;
    for (int ep = 0; b < rows.size(); ++ep) {
      size_t e = b + 1;
      while (e < rows.size() && rows[e].ep == rows[b].ep) ++e;
      const size_t n = e - b;
      long v0n = 0, wmaxn = 0, flips = 0, ngoal = 0, near_goal = 0, ndec = 0, dec_near = 0, dec_with_front_near = 0, n_sw = 0;
      double sv = 0, sw = 0, cand = 0, ok = 0, okf = 0, nf = 0, oks = 0, ns = 0, fn = 0, fr = 0;
      int last_sign = 0, last_tgt = -2, done_k = -1, done_code = 0;
      float lgx = 0.f, lgy = 0.f;
      std::vector<float> js;
      for (size_t i = b; i < e; ++i) {
        const Row& r = rows[i];
        v0n += std::fabs(r.cmd_v) < 0.05f;
        wmaxn += std::fabs(r.cmd_w) >= 0.95f * wmax;
        sv += std::fabs(r.cmd_v);
        sw += std::fabs(r.cmd_w);
        const int sg = r.cmd_w > 0.1f ? 1 : r.cmd_w < -0.1f ? -1 : 0;
        if (sg) { if (last_sign && sg != last_sign) ++flips; last_sign = sg; }
        if (r.goal >= 0) { ++ngoal; near_goal += r.gdist >= 0.f && r.gdist < 0.5f; }
        if (r.dec) {
          ++ndec;
          dec_near += r.goal >= 0 && r.gdist < 0.5f;
          dec_with_front_near += r.front_near > 0;
          if (r.goal != last_tgt) {
            if (last_tgt >= 0 && r.goal >= 0) js.push_back(std::hypot(r.gx - lgx, r.gy - lgy));
            lgx = r.gx;
            lgy = r.gy;
            if (last_tgt != -2) ++n_sw;
            last_tgt = r.goal;
          }
        }
        if (done_k < 0 && (r.done == mdp::cur::kDeclareOk || r.done == mdp::cur::kNoProgress || r.done == mdp::cur::kTruncated)) { done_k = r.k; done_code = r.done; }
        cand += r.cand_n;
        ok += r.scan_ok; okf += r.ok_front; nf += r.n_front; oks += r.ok_side; ns += r.n_side;
        fn += r.front_near; fr += r.front;
      }
      // 명령 → 바퀴·자이로 (명령은 걸음 끝에 냄 → 다음 걸음 감각과 견줌): 움직이라는 걸음만 최소제곱 비
      double num_v = 0, den_v = 0, num_w = 0, den_w = 0, num_g = 0;
      long nv = 0, nw = 0;
      for (size_t i = b; i + 1 < e; ++i) {
        const Row& r = rows[i];
        const Row& q = rows[i + 1];
        if (std::fabs(r.cmd_v) > 0.1f) { num_v += q.wv * r.cmd_v; den_v += r.cmd_v * r.cmd_v; ++nv; }
        if (std::fabs(r.cmd_w) > 0.3f) { num_w += q.ww * r.cmd_w; num_g += q.gyro * r.cmd_w; den_w += r.cmd_w * r.cmd_w; ++nw; }
      }
      const auto known_at = [&](float t) -> float {
        for (size_t i = b; i < e; ++i)
          if (rows[i].k * dt >= t) return rows[i].known * a2;
        return rows[e - 1].known * a2;
      };
      std::sort(js.begin(), js.end());
      const float jm = js.empty() ? 0.f : js[js.size() / 2];
      const float T = std::max(rows[e - 1].k * dt, 1e-3f);
      std::printf("[%s 판 %d] 걸음 %zu (%.1f s) · 끝 %s · 다시 돌린 명령 다름(녹화 전체) %ld%s\n", label.c_str(), ep, n, T,
                  done_k >= 0 ? (std::to_string(done_k * dt).substr(0, 5) + (done_code == mdp::cur::kDeclareOk ? " s (닫힘 성공)" : done_code == mdp::cur::kNoProgress ? " s (진척 없음)" : " s (시간)")).c_str() : "없음",
                  d_cmd, first_bad >= 0 ? (" (처음 " + std::to_string(first_bad) + ")").c_str() : "");
      std::printf("  지도 아는 넓이 m²: 5 s %.1f · 10 s %.1f · 20 s %.1f · 30 s %.1f · 끝 %.1f (빈 %.1f · 막힘 %.1f)\n", known_at(5), known_at(10), known_at(20), known_at(30), rows[e - 1].known * a2,
                  rows[e - 1].freec * a2, rows[e - 1].occ * a2);
      std::printf("  프론티어 칸 평균 %.0f · 발밑(%.1f m 안) 평균 %.1f · 결정 %ld 중 발밑 프론티어 있는 결정 %.2f · 후보 평균 %.2f\n", fr / n, near_r, fn / n, ndec, ndec ? (double)dec_with_front_near / ndec : 0.0,
                  cand / n);
      std::printf("  목표: 바뀜 %ld (%.2f /s) · 바뀔 때 옛→새 거리 중앙 %.2f m · 0.5 m 안 목표인 걸음 몫 %.2f · 결정 때 목표가 0.5 m 안 %.2f\n", n_sw, n_sw / T, jm, ngoal ? (double)near_goal / ngoal : 0.0,
                  ndec ? (double)dec_near / ndec : 0.0);
      std::printf("  정책 명령: |v|<0.05 몫 %.2f · 평균 |v| %.2f m/s (한도 %.2f) · 평균 |ω| %.2f rad/s · |ω| ≥ 0.95 한도 몫 %.2f · ω 부호 바뀜 %.2f /s\n", (double)v0n / n, sv / n, vmax, sw / n,
                  (double)wmaxn / n, flips / T);
      std::printf("  명령 → 다음 걸음 감각 (최소제곱 비): 바퀴 v / 명령 v %.2f (%ld 걸음) · 바퀴 ω / 명령 ω %.2f · 자이로 / 명령 ω %.2f (%ld 걸음)\n", den_v > 0 ? num_v / den_v : 0.0, nv,
                  den_w > 0 ? num_w / den_w : 0.0, den_w > 0 ? num_g / den_w : 0.0, nw);
      std::printf("  라이다: 빔 %d · 유효 몫 %.3f (앞 ±60° %.3f · 옆 %.3f)\n", rows[e - 1].scan_n, ok / std::max(1.0, (double)rows[e - 1].scan_n * n), nf > 0 ? okf / nf : 0.0, ns > 0 ? oks / ns : 0.0);
      b = e;
    }
    std::printf("[%s] 라이다 각 구간(20°, −110° → +110°) 유효 몫:", label.c_str());
    for (int i = 0; i < kAB; ++i) std::printf(" %.2f", ab_n[i] > 0 ? ab_ok[i] / ab_n[i] : 0.0);
    double rt = 0;
    for (double v : rh) rt += v;
    std::printf("\n[%s] 유효 빔 거리 분포(1 m 구간 0…12+):", label.c_str());
    for (double v : rh) std::printf(" %.3f", rt > 0 ? v / rt : 0.0);
    double dt_ = 0;
    for (double v : dh) dt_ += v;
    std::printf("\n[%s] 깊이 관측 바이트 몫: 구멍 %.3f · 1~42 %.3f · 43~85 %.3f · 86~170 %.3f · 171~254 %.3f · 255 %.3f | 깊이 장애물 층 1.5 m 안 칸 평균 %.1f\n", label.c_str(),
                dh[0] / dt_, dh[1] / dt_, dh[2] / dt_, dh[3] / dt_, dh[4] / dt_, dh[5] / dt_, cam_n > 0 ? cam_near / cam_n : 0.0);
    if (cam_near > 0)
      std::printf("[%s] 깊이 층 1.5 m 안 칸 나눔: 라이다 막힘 2 칸 안 %.2f · 지금 카메라 시야 안 %.2f · 시야 밖(쌓인 것) %.2f\n", label.c_str(),
                  cam_lid / cam_near, cam_fov / cam_near, 1.0 - cam_fov / cam_near);
    {
      double zt = 0;
      for (double v : zb) zt += v;
      const double det = fs[0] * fs[2] - fs[1] * fs[1];
      const double sb = det != 0 ? (fs[0] * fs[4] - fs[1] * fs[3]) / det : 0.0, sa = fs[0] > 0 ? (fs[3] - sb * fs[1]) / fs[0] : 0.0;
      std::printf("[%s] 깊이 점 높이 (센서 수평 0.5~2.5 m) 몫: < −3 cm %.3f · ±3 cm %.3f · 3~8 cm %.3f · 8~25 cm %.3f · > 25 cm %.3f | 바닥 같은 점 높이 = %.3f + %.4f·거리 (기울기 → 숙임 차 %.2f°)\n",
                  label.c_str(), zb[0] / zt, zb[1] / zt, zb[2] / zt, zb[3] / zt, zb[4] / zt, sa, sb, std::atan(sb) * 57.2958);
      std::printf("[%s] 바닥 높이 최빈 (cm, 수평 0.5~1 · 1~1.5 · 1.5~2 · 2~2.5 m):", label.c_str());
      for (int d = 0; d < 4; ++d) {
        int bi = 0;
        for (int k = 1; k < 30; ++k) if (zh[d][k] > zh[d][bi]) bi = k;
        std::printf(" %d", bi - 10);
      }
      std::printf("\n");
      std::printf("[%s] 바닥 구멍 몫 (입사 cos 구간 경계 sensor_model floor_cos):", label.c_str());
      for (int k = 0; k < sim::kDepthFloorBins; ++k) std::printf(" [%.2f,%.2f) %.3f (%.0f)", c.depth.floor_cos[k], c.depth.floor_cos[k + 1], fh_tot[k] > 0 ? fh_hole[k] / fh_tot[k] : -1.0, fh_tot[k]);
      std::printf("\n");
      if (std::getenv("NODE_STATS_ZHIST"))
        for (int d = 0; d < 4; ++d) {
          std::printf("  z 칸 %d:", d);
          for (int k = 0; k < 30; ++k) std::printf(" %.0f", zh[d][k]);
          std::printf("\n");
        }
    }
    if (ff) std::fclose(ff);
    if (!vox.empty()) {   // 칸 가운데 세계 좌표 · 본 걸음 · 띠별 점 수 · 라이다 막힘(로봇 지도 끝)
      FILE* fv = std::fopen(vox_path.c_str(), "w");
      if (!fv) throw std::runtime_error("vox 를 못 엶: " + vox_path);
      std::fprintf(fv, "x\ty\tsee\tz03_10\tz10_25\tz25_40\tz40_60\tz60_85\tlidar_occ\n");
      const mdp::MapView mv{robot.lmap(), c.size, w.gw, w.gh};
      for (int y = 0; y < w.gh; ++y)
        for (int x = 0; x < w.gw; ++x) {
          const size_t k = (size_t)y * c.size + x;
          uint32_t tot = 0;
          for (int b = 0; b < kVoxB; ++b) tot += vox[k * kVoxB + b];
          const int occ = mdp::cls_at(mv, x, y) == mdp::kOcc;
          if (!tot && !occ) continue;
          std::fprintf(fv, "%.3f\t%.3f\t%u", w.ox + (x + 0.5f) * c.res, w.oy + (y + 0.5f) * c.res, vsee[k]);
          for (int b = 0; b < kVoxB; ++b) std::fprintf(fv, "\t%u", vox[k * kVoxB + b]);
          std::fprintf(fv, "\t%d\n", occ);
        }
      std::fclose(fv);
    }
    if (lat_n > 0) {
      int kb = 0;
      for (int k = 1; k < kTau; ++k) if (lat_hit[k] > lat_hit[kb]) kb = k;
      std::printf("[%s] 깊이 시각 어긋남 (빨리 돌 때 띠 안 깊이 점 %.0f 개가 로봇 지도 막힘 둘레에 드는 몫, τ = 0 … 0.375 s):", label.c_str(), lat_n);
      for (int k = 0; k < kTau; ++k) std::printf(" %.3f", lat_hit[k] / lat_n);
      std::printf(" → 가장 잘 맞는 τ %.3f s\n", 0.025 * kb);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "node_stats: %s\n", ex.what());
    return 1;
  }
}
