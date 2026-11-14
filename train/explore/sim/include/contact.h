// 로봇이 재는 간격·닿음 (2026-11-14) — 보상·판 끝이 시뮬 정답(정답 격자·참 자세·참 사람) 대신 쓰는 값. env.h 가운데서 include (low_rl.h 뒤). 모두 스레드 0.
//
// 실물 LIMO 에는 범퍼가 없다. 그래서 "부딪힘"을 로봇 센서 셋으로 정한다 (mdp/README "닿음 — 센서"):
//   라이다  : 이번 장 점(±110°, 높이 0.111 m)이 몸통 사각형에서 con_lid 안 (안쪽이면 음수 간격)
//   깊이    : 깊이 카메라 장애물 층(hier_path.h — 라이다 면 아래 낮은 물체, 높이 0.03~0.25 m)의 칸이 지도 자세 몸통에서 con_cam 안
//   멈춤    : 명령은 움직이라는데 바퀴 오도메트리·자이로가 안 움직인 걸음이 con_stall 번 이어짐 — 센서가 못 보는 옆·뒤 닿음 (실물: 모터가 막힘)
//   (지도   : 로봇 지도 막힘 칸이 지도 자세 몸통에서 con_map 안 — 기억으로 옆·뒤까지 보지만 자세 오차를 탄다. 기본은 닿음에 안 넣고 간격(근접 항)에만)
// 닿음 = con_mask 에 든 것 가운데 하나라도. 시뮬 정답 충돌(e.collided — 몸통이 정답 격자·사람과 겹침)은 물리(멈춤)와 기록에만 쓴다.
// 시뮬은 충돌 충격(가속 튐)을 모델에 안 넣었다(env.h physics_step "충돌 충격은 모델에 없음") — 그래서 IMU 몫은 "명령 대비 안 움직임" 으로만 잰다.
#pragma once

// 몸통 좌표 점 (u 앞, v 왼쪽) → 몸통 사각형(반 길이·반 폭)까지 간격 (m). 안쪽이면 음수 (가장 가까운 변까지)
DEV float rect_gap(const EnvCfg& c, float u, float v) {
  const float dx = dm::absf(u) - c.half_len, dy = dm::absf(v) - c.half_wid;
  if (dx > 0.f || dy > 0.f) {
    const float ax = dm::maxf(dx, 0.f), ay = dm::maxf(dy, 0.f);
    return sqrtf(ax * ax + ay * ay);
  }
  return dm::maxf(dx, dy);
}

// 이번 행동의 라이다 장(Shared::bins — 관측 라이다 칸과 같은 248 칸, 칸마다 가장 가까운 점)의 점 → 몸통 간격 최솟값.
// 장은 행동 끝 자세에서 쏜 것이라(lidar_every = substeps) 몸통 좌표 = 장착 위치 + 빔. 장이 없었으면 kSenseFar
DEV float sense_lidar_gap(const EnvCfg& c, const Shared& sh) {
  if (!sh.scanned) return kSenseFar;
  float best = kSenseFar;
  for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
    if (sh.bins[k] == kBinNone) continue;
    const float r = (float)(sh.bins[k] >> 16) * 0.001f;
    if (r >= c.map_max_range) continue;
    const float a = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
    float s, co;
    dm::sincosf_d(a, &s, &co);
    best = dm::minf(best, rect_gap(c, c.mount_x + r * co, c.mount_y + r * s));
  }
  return best;
}

// 지도 자세 몸통 둘레 kSenseWin 안 칸: 로봇 지도 막힘 → *g_map, 깊이 카메라 층 → *g_cam (칸 가운데 간격 − 칸 반 폭, 없으면 kSenseWin)
DEV void sense_map_gaps(const EnvBatch& b, const EnvState& e, int env, float* g_map, float* g_cam) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const float px = e.x + e.ex, py = e.y + e.ey;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  const float hl = c.half_len + kSenseWin, hw = c.half_wid + kSenseWin, rad = sqrtf(hl * hl + hw * hw);
  int x0, y0, x1, y1;
  cell_of(c, m, px - rad, py - rad, &x0, &y0);
  cell_of(c, m, px + rad, py + rad, &x1, &y1);
  float gm = kSenseWin, gc = kSenseWin;
  y0 = y0 < 0 ? 0 : y0;
  x0 = x0 < 0 ? 0 : x0;
  y1 = y1 > m.gh - 1 ? m.gh - 1 : y1;
  x1 = x1 > m.gw - 1 ? m.gw - 1 : x1;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      const bool occ = mdp::cell_class(L[y * c.size + x]) == mdp::kOcc, cam = cam_lethal(b, env, x, y);
      if (!occ && !cam) continue;
      const float wx = m.ox + ((float)x + 0.5f) * c.res - px, wy = m.oy + ((float)y + 0.5f) * c.res - py;
      const float g = rect_gap(c, co * wx + s * wy, co * wy - s * wx) - 0.5f * c.res;
      if (occ) gm = dm::minf(gm, g);
      if (cam) gc = dm::minf(gc, g);
    }
  *g_map = gm;
  *g_cam = gc;
}

// 사람 (라이다 덩어리 칼만 추적 — 아래층 RL 의 lowrl_trk, 센서만): 지도 자세 몸통 ~ 추적한 움직이는 덩어리(원) 간격 최솟값 (m). 추적이 없으면 kSenseFar.
// 닿음 때 이 간격 < kConPersonGap 이면 사람 닿음으로 셈 (몸통 쪽 덩어리 가장자리가 원 반지름만큼 흔들리므로 여유)
constexpr float kConPersonGap = 0.15f;   // 잠정 — 추적 덩어리 자리 σ 0.05 m(low_rl.h kLrTrkSz)의 3 배
DEV float tracked_person_gap(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.lowrl || !b.cfg.ps_trk) return kSenseFar;
  LrMover M[kLrMovers + 1];
  const int n = lowrl_trk_movers(b, env, M);
  const float px = e.x + e.ex, py = e.y + e.ey;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  float g = kSenseFar;
  for (int i = 0; i < n; ++i) {
    const float dx = M[i].x - px, dy = M[i].y - py;
    g = dm::minf(g, rect_gap(b.cfg, co * dx + s * dy, co * dy - s * dx) - M[i].r);
  }
  return g;
}

// 걸음 끝 (finish_step 바로 앞, 물리·지도 갱신 뒤): 간격 넷·멈춤 시계·닿음·추적 사람 간격. sh = 이번 행동의 라이다 장 칸
DEV void sense_step(const EnvBatch& b, EnvState& e, int env, const Shared& sh) {
  const EnvCfg& c = b.cfg;
  if (e.collided) e.gt_touch |= e.people.hit_person ? 2 : 1;   // 정답 충돌 (기록 전용 — 학습에 안 씀)
  e.s_lid = sense_lidar_gap(c, sh);
  sense_map_gaps(b, e, env, &e.s_map, &e.s_cam);
  e.s_clear = dm::minf(dm::minf(e.s_lid, e.s_cam), e.s_map);
  // 멈춤: 보낸 명령(/cmd_vel)이 움직이라는데 바퀴 오도메트리(drive.wv·ww)·자이로가 거의 0
  const bool cmd = dm::absf(e.cmd_v) >= c.con_cmd_v || dm::absf(e.cmd_w) >= c.con_cmd_w;
  const bool still = dm::absf(e.drive.wv) < c.con_odo_v && dm::absf(e.drive.ww) < c.con_odo_w && dm::absf(e.gyro) < c.con_odo_w;
  e.s_stall = cmd && still ? e.s_stall + 1 : 0;
  int k = 0;
  if (e.s_lid < c.con_lid) k |= kConLidar;
  if (e.s_cam < c.con_cam) k |= kConCam;
  if (e.s_stall >= c.con_stall) k |= kConStall;
  if (e.s_map < c.con_map) k |= kConMap;
  e.contact = (k & c.con_mask) != 0 ? 1 : 0;
  e.s_pgap = tracked_person_gap(b, e, env);
  if (e.contact && e.s_pgap < kConPersonGap) k |= kConPerson;   // 닿음 때 추적 사람이 몸통 곁 → 사람 닿음 (판 끝 코드·보상 항)
  e.s_kind = k;
}
