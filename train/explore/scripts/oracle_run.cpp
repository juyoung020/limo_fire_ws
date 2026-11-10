// oracle_run — 정답(GT)을 쓰는 상한 측정 도구. 학습 환경·정책 관측에는 아무것도 안 넣는다: 시뮬은 env_api.h 로만 몰고(속도 행동, 물리 단위),
// 정답은 이 러너가 호스트에서만 읽는다 (World::grid, EnvState 참 자세·사람). 판 끝·성공 정의는 시뮬 그대로(지도 닫힘(mdp/frontier.h 닫힘 규칙) + 지도 진행 없음
// 시계 ≥ 10 s · 충돌 · 진행 없음 60 s · 안전 시간) — 이 도구는 "이 시뮬 + 이 성공 정의로 학습 맵에서 거의 100 % 가 되나"를 잰다.
//
//   oracle_run --policy oracle|slot0 --stage 0..3 [--people on|off] [--envs 256] [--per-stage 32] [--seed 1]
//              [--list scenes/train/list.tsv] [--out 판.tsv] [--threads 8] [--dump-fail 폴더]
//
// 판 = 환경마다 첫 판 (커리큘럼 켬, 시작 단계를 --stage 로 놓고 리셋 → 맵·오차 배율·사람 수가 학습과 같은 규칙. 첫 판 뒤 단계 변화는 안 셈).
// 맵 = 학습 목록에서 단계(등급)마다 고르게 --per-stage 개 (목록 차례에서 같은 간격) — 환경마다 그 단계 맵 중 무작위(시뮬 리셋 규칙).
// 같은 --seed 면 oracle·slot0 이 환경마다 같은 맵·시작 자세·바닥·사람으로 시작한다 (리셋 난수는 행동 모드와 무관).
// --people off: 사람 놓기 거리(spawn_keep)를 맵보다 크게 해 사람 0 명 (사람 난수 줄기는 로봇 쪽과 따로라 로봇 쪽 시작은 같음).
//
// oracle (정답 허용, 속도 행동 물리 단위. 시뮬 안전층은 끔 — 정답으로 더 엄격히 스스로 막음, ORACLE_SAFETY=1 이면 켬):
//   목표 = 성공 판정과 같은 닫힘 규칙(mdp/frontier.h frontier_search — 지나간 칸 씨앗·덩어리 문턱·본 칸, sim 의 지나간 길 비트 판 Env::get_trail)에서
//          닫힘을 막는 칸을 품은 굵은 칸(20 cm). 그 굵은 칸 가운데에서 max(0.5, close_look − 0.25) m 안 설 수 있는(정답) 칸 중, 막힘까지 ≥ 0.25 m(제자리
//          회전 여유)이고 라이다 높이 정답 장면 광선으로 그 굵은 칸이 보이는 칸을 먼저(아니면 경로 값 + 40), 정답 경로 거리가 가장 짧은 칸.
//          볼 자리가 없는 프론티어(집 밖·가구 속으로 샌 칸)는 3 굵은 칸 안의 덜 굳은 벽(정답 막힘이 있는데 지도는 막힘 아님)을 보러 감.
//   경로 = 정답 격자에서 계층형과 같은 팽창(로봇 반경 0.2 m, 갈 수 있는 바닥만)·같은 무게(막힘까지 ≤ 0.3 m → 3 · ≤ 0.45 m → 2 · 밖 1)로 Dial 최단 경로.
//          사람: 정답 위치·속도로 0 · 0.5 · 1 · 1.5 s 뒤 자리를 (반지름 + 0.3 m) 막힘으로. 결정(1 s)마다 다시 짬.
//   추종 = 참 자세로 pure pursuit (앞 0.45 m, 방향 오차 > 0.7 rad 면 제자리 회전, 최고 0.8 m/s, 벽 0.3 m 안이면 0.35 m/s).
//   안전 = 명령 후보 99 개(v × ω)를 보수적 구동(지연 0.1 s, 빨라짐 5 m/s² · 30 rad/s², 느려짐 2 m/s² · 8 rad/s²)으로 굴려,
//          한 걸음 쓰고 멈춰도 정답 격자 몸통(+1 cm)이 안 닿는 것(DWA admissible) 중 — 그대로 1 s 굴려도 안 닿고 사람(정답 위치·속도 등속 예측,
//          로봇 1 m 안의 비키는 사람은 로봇 반대쪽으로)과 몸통 간격이 2 s 동안 0.2 m 넘는 경로 명령이면 그대로, 아니면 사람 간격·경로 명령과의 차이 점수가 가장 좋은 것.
//          (멈춤·물러나기 포함). 없으면 멈춤.
//   도착하면 굵은 프론티어 쪽으로 돌아서 3 s 기다리고, 그래도 안 풀리면 30 s 동안 그 칸을 뺀다(8 s 못 움직임·시간 넘김도). 프론티어가 없거나 볼 곳이 없으면 멈춰 기다림.
// slot0: 계층형 규칙 정책(늘 0 번 후보) — frontier_run slot0 과 같은 정책을 같은 판으로.
//
// 출력: 판마다 한 줄 TSV(--out) + 끝 요약(성공·벽 충돌·사람 충돌·진행 없음·끊김·거짓 닫힘(정답 덮음 < 0.8 인 성공)·성공까지 시간, 표준오차).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "env_api.h"
#include "frontier.h"

namespace {

struct Args {
  std::string policy = "oracle", list = "scenes/train/list.tsv", out, env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml";
  int stage = 0, envs = 256, per_stage = 32, threads = 8, dump_env = -1;
  std::string dump_fail;   // --dump-fail 폴더: 성공 못 한 판·거짓 닫힘(정답 덮음 < 0.8) 판의 마지막 결정 지도 그림 (dump_ppm)
  std::string dump;   // --dump 파일.ppm --dump-env i: 그 환경 첫 판의 결정마다 지도 그림을 덮어씀 (판 끝 1 s 안 모습이 남음)
  bool people = true;
  uint64_t seed = 1;
};

// ---------- 정답 격자 (맵마다 한 번) ----------
struct Gt {
  int gw = 0, gh = 0;
  float ox = 0, oy = 0, res = 0.05f;
  std::vector<uint8_t> blocked, stand, wgt;   // 막힘 · 설 수 있음(갈 수 있는 바닥 + 막힘까지 > 0.2 m) · 경로 무게 1~3
  std::vector<float> dist;                    // 막힘까지 거리 (m, 0.6 에서 자름)
  sim::SceneView sv{};                        // 정답 장면 (라이다 높이 보임 — 가구 다리 사이·밑은 라이다가 지나감)
  float lidar_z = 0.f;
};

Gt make_gt(const sim::World& W, float res) {
  Gt g;
  g.gw = W.gw; g.gh = W.gh; g.ox = W.ox; g.oy = W.oy; g.res = res;
  g.sv = W.scene.view();
  const int n = W.gw * W.gh;
  g.blocked.assign(n, 0); g.stand.assign(n, 0); g.wgt.assign(n, 0); g.dist.assign(n, 0.6f);
  for (int k = 0; k < n; ++k) g.blocked[k] = (W.grid[k] & sim::kCellBlocked) ? 1 : 0;
  const int R = 12;   // 0.6 m
  for (int y = 0; y < W.gh; ++y)
    for (int x = 0; x < W.gw; ++x) {
      int best = 1 << 30;
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int u = x + dx, v = y + dy;
          const bool b = u < 0 || v < 0 || u >= W.gw || v >= W.gh || g.blocked[v * W.gw + u];
          if (b) best = std::min(best, dx * dx + dy * dy);
        }
      const float d = best == (1 << 30) ? 0.6f : std::min(0.6f, std::sqrt((float)best) * res);
      const int k = y * W.gw + x;
      g.dist[k] = d;
      // 계층형 설 수 있음과 같은 팽창: 칸 가운데끼리 거리² ≤ (0.2/res)² 인 칸에 막힘이 있으면 못 섬 (mdp::robot_cells(0.2, 0.05) r2 = 16)
      g.stand[k] = (W.grid[k] & sim::kCellFloor) && best > 16 ? 1 : 0;
      g.wgt[k] = !g.stand[k] ? 0 : (d > 0.45f ? 1 : (d > 0.3f ? 2 : 3));
    }
  return g;
}

bool gt_blocked(const Gt& g, float x, float y) {
  const int cx = (int)std::floor((x - g.ox) / g.res), cy = (int)std::floor((y - g.oy) / g.res);
  if (cx < 0 || cy < 0 || cx >= g.gw || cy >= g.gh) return true;
  return g.blocked[cy * g.gw + cx] != 0;
}
// 몸통(+여유) 둘레 점 + 가운데가 막힌 칸에 닿나
bool body_hit(const Gt& g, float x, float y, float th, float hl, float hw) {
  const float s = std::sin(th), c = std::cos(th);
  const int nl = 7, nw = 5;
  for (int i = 0; i <= nl; ++i)
    for (int j = 0; j <= nw; ++j) {
      if (i != 0 && i != nl && j != 0 && j != nw) continue;
      const float u = -hl + 2.f * hl * i / nl, v = -hw + 2.f * hw * j / nw;
      if (gt_blocked(g, x + c * u - s * v, y + s * u + c * v)) return true;
    }
  return gt_blocked(g, x, y);
}
// 칸 (x0, y0) 가운데 라이다 높이에서 굵은 칸 [bx0, bx0 + f) × [by0, by0 + f) 가운데로 쏜 정답 장면 광선이 그 굵은 칸에 닿나 (가는 길에 형상에 안 막힘 —
// 맞으면 그 칸이 막힘이 되고, 지나가면 모름이 풀림). 몸통 높이 충돌 격자가 아니라 라이다 평면 보임 — 다리 있는 가구 밑은 라이다가 지나감
bool gt_sees_block(const Gt& g, int x0, int y0, int bx0, int by0, int f) {
  const float ox = g.ox + (x0 + 0.5f) * g.res, oy = g.oy + (y0 + 0.5f) * g.res;
  const float tx = g.ox + (bx0 + 0.5f * f) * g.res, ty = g.oy + (by0 + 0.5f * f) * g.res;
  const float dx = tx - ox, dy = ty - oy, d = std::hypot(dx, dy);
  if (d < 1e-3f) return true;
  uint16_t obj;
  const float r = sim::raycast(g.sv, sim::V3{ox, oy, g.lidar_z}, sim::V3{dx / d, dy / d, 0.f}, d + 0.2f, &obj);
  return r >= d - 0.5f * f * g.res;   // 굵은 칸 가장자리(반 칸) 안까지 감
}

// ---------- 사람 예측 ----------
struct PPred { float x, y, vx, vy, r; };
std::vector<PPred> predict_people(const sim::EnvCfg& c, const sim::EnvState& s) {
  std::vector<PPred> out;
  const sim::People& P = s.people;
  for (int i = 0; i < P.n; ++i) {
    PPred p{P.cyl[i].x, P.cyl[i].y, 0.f, 0.f, P.cyl[i].r};
    const bool ignore = P.mode[i] & sim::kPersonIgnore, walk = P.mode[i] & sim::kPersonWalk;
    const float rdx = p.x - s.x, rdy = p.y - s.y, rd = std::hypot(rdx, rdy);
    if (!ignore && rd < c.people.yield_dist && rd > 1e-3f) {   // 비켜섬
      p.vx = c.people.yield_speed * rdx / rd;
      p.vy = c.people.yield_speed * rdy / rd;
    } else if (walk && P.wait[i] <= 0.f) {
      const float gx = P.gx[i] - p.x, gy = P.gy[i] - p.y, gd = std::hypot(gx, gy);
      if (gd > 1e-3f) { p.vx = P.speed[i] * gx / gd; p.vy = P.speed[i] * gy / gd; }
    }
    out.push_back(p);
  }
  return out;
}
// 몸통 사각형(로봇 기준) ~ 원기둥 간격
float rect_gap(float rx, float ry, float rth, float hl, float hw, float px, float py, float pr) {
  const float s = std::sin(rth), c = std::cos(rth), dx = px - rx, dy = py - ry;
  const float u = c * dx + s * dy, v = -s * dx + c * dy;
  const float ex = std::max(std::fabs(u) - hl, 0.f), ey = std::max(std::fabs(v) - hw, 0.f);
  return std::hypot(ex, ey) - pr;
}

// ---------- 환경 하나의 oracle 상태 ----------
enum Phase { kGo = 0, kArrive = 1, kClosedWait = 2, kNoTarget = 3 };
int g_dbg_env = -1;   // ORACLE_DEBUG=i: 그 환경의 결정·뺌 사유를 찍음
struct Oracle {
  int id = -1;
  std::vector<std::pair<float, float>> path;   // 세계 좌표 (참 자세 기준 계획)
  int pi = 0;
  int tgt_coarse = -1;          // 굵은 칸 번호
  float tgt_fx = 0, tgt_fy = 0; // 굵은 프론티어 가운데 (세계)
  float tgt_cost = 0;
  int phase = kNoTarget;
  float arrive_t = 0, tgt_t0 = 0, last_move_t = 0, mvx = 0, mvy = 0;
  std::map<int, float> black;   // 굵은 칸 → 뺄 시각 끝
  // 진단
  float t_closed_first = -1;
  int last_unview = 0, wall_target = 0, n_wall_dec = 0, n_wedge = 0, last_nf_floor = 0;
  sim::EnvState last_s{};
  int n_dec = 0, n_unreach = 0, n_black = 0, n_closed_dec = 0, last_nf = 0, last_nreach = 0, n_people_override = 0, n_stop = 0;
};

struct Work {   // 스레드마다 버퍼
  std::vector<int32_t> dist;
  std::vector<uint8_t> par, dyn;
  std::vector<int16_t> cL;
  std::vector<uint8_t> cflag;
  std::vector<int32_t> qa, qb;
  std::vector<std::vector<int32_t>> buckets;
  std::vector<uint8_t> fflag;                 // 닫힘 찾기 (mdp::frontier_search) 버퍼
  std::vector<int32_t> fqa, fqb, flabel;
  std::vector<mdp::Frontier> fout;
};
constexpr int kF = 4;   // 목표를 묶는 굵은 칸 (4 × 5 cm = 20 cm) — 볼 자리 찾기·덜 굳은 벽 찾기 단위
inline int cdim(int n) { return (n + kF - 1) / kF; }

constexpr int kInf = 1 << 29;

// 정답 격자 Dial 최단 경로 (무게 1~3, 4 이웃). 시작 칸 둘레 0.3 m 는 설 수 없는 바닥(몸통 반폭보다 넓은 곳)도 무게 6 으로 — 벽에 붙어 서 있을 때 빠져나오게
void gt_dijkstra(const Gt& g, int sx, int sy, Work& w) {
  const int n = g.gw * g.gh;
  w.dist.assign(n, kInf);
  w.par.assign(n, 255);
  if (sx < 0 || sy < 0 || sx >= g.gw || sy >= g.gh) return;
  const int s0 = sy * g.gw + sx;
  w.buckets.assign(8, {});
  w.dist[s0] = 0;
  w.buckets[0].push_back(s0);
  int cur = 0, left = 1;
  const int X[4] = {-1, 1, 0, 0}, Y[4] = {0, 0, -1, 1};
  while (left > 0) {
    auto& B = w.buckets[cur % 8];
    if (B.empty()) { ++cur; continue; }
    const int k = B.back();
    B.pop_back();
    --left;
    if (w.dist[k] != cur) continue;
    const int x = k % g.gw, y = k / g.gw;
    for (int d = 0; d < 4; ++d) {
      const int u = x + X[d], v = y + Y[d];
      if (u < 0 || v < 0 || u >= g.gw || v >= g.gh) continue;
      const int j = v * g.gw + u;
      int wt = g.wgt[j];
      const bool near = (u - sx) * (u - sx) + (v - sy) * (v - sy) <= 36;
      if (w.dyn[j]) wt = near ? std::max(wt, 1) + 5 : 0;   // 사람 예측 자리 (시작 가까이는 비싸게 지나감 — 빠져나오기)
      if (wt == 0 && near && g.dist[j] > 0.12f) wt = 6;
      if (wt == 0) continue;
      const int nd = cur + wt;
      if (nd < w.dist[j]) { w.dist[j] = nd; w.par[j] = (uint8_t)d; w.buckets[nd % 8].push_back(j); ++left; }
    }
  }
}

// 한 환경 한 결정: 목표 고르고 경로 짬
void decide(const sim::EnvCfg& c, const Gt& g, const int16_t* L, int S, const uint32_t* trail, const sim::EnvState& s, Oracle& o, Work& w, float t) {
  o.n_dec++;
  // 굵은 격자 (막힘 하나라도 → 막힘, 모두 모름 → 모름, 그 밖 갈 수 있음) — 덜 굳은 벽 찾기에만
  const int f = kF;
  const int cw = cdim(g.gw), ch = cdim(g.gh);
  w.cL.assign(cw * ch, 0);
  const mdp::MapView mv{L, S, g.gw, g.gh};
  for (int cy = 0; cy < ch; ++cy)
    for (int cx = 0; cx < cw; ++cx) {
      bool occ = false, unk = true;
      for (int yy = cy * f; yy < cy * f + f && yy < g.gh; ++yy)
        for (int xx = cx * f; xx < cx * f + f && xx < g.gw; ++xx) {
          const uint8_t k = mdp::cls_at(mv, xx, yy);
          occ = occ || k == mdp::kOcc;
          unk = unk && k == mdp::kUnk;
        }
      w.cL[cy * cw + cx] = occ ? (int16_t)1000 : (unk ? mdp::kUnknownL : (int16_t)-1000);
    }
  // 목표 = 성공 판정과 같은 닫힘 규칙(mdp/frontier.h, sim 의 지나간 길 비트 판)에서 닫힘을 막는 칸 — 그 칸을 품은 굵은 칸으로 묶음
  const int n = g.gw * g.gh, ws = (S + 31) / 32;
  w.fflag.resize(n); w.fqa.resize(n); w.fqb.resize(n); w.flabel.resize(n);
  if (w.fout.size() < (size_t)n) w.fout.resize(n);
  mdp::SearchBuf sb{w.fflag.data(), w.fqa.data(), w.fqb.data(), nullptr, w.flabel.data()};
  sb.vis = trail;
  sb.look = trail + (size_t)S * ws;
  sb.tws = ws;
  sb.close_min = c.close_min;
  sb.rbc = mdp::robot_cells(c.close_radius, c.res);
  const int mx = (int)std::floor((s.x + s.ex - g.ox) / g.res), my = (int)std::floor((s.y + s.ey - g.oy) / g.res);
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const mdp::SearchResult r = mdp::frontier_search(mv, mx, my, rb, 1, sb, w.fout.data(), n);
  std::vector<int> fr;
  {
    std::vector<uint8_t> isc(cw * ch, 0);
    for (int q = 0; q < r.n_reach; ++q) {
      const int k = w.fqa[q], x = k % g.gw, y = k / g.gw;
      if (!(w.fflag[k] & 2) || w.fout[w.flabel[k]].n < c.close_min) continue;
      if (mdp::looked_cell(mv, sb.look, sb.vis, ws, x, y)) continue;
      const int ck = (y / f) * cw + x / f;
      if (!isc[ck]) { isc[ck] = 1; fr.push_back(ck); }
    }
  }
  // 정답 경로 거리 (사람 예측 자리 막음)
  w.dyn.assign(g.gw * g.gh, 0);
  for (const PPred& p : predict_people(c, s))
    for (float tt = 0.f; tt <= 1.51f; tt += 0.5f) {
      const float px = p.x + p.vx * tt, py = p.y + p.vy * tt, rr = p.r + 0.3f;
      const int R = (int)std::ceil(rr / g.res);
      const int pcx = (int)std::floor((px - g.ox) / g.res), pcy = (int)std::floor((py - g.oy) / g.res);
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int u = pcx + dx, v = pcy + dy;
          if (u < 0 || v < 0 || u >= g.gw || v >= g.gh) continue;
          if ((dx * dx + dy * dy) * g.res * g.res <= rr * rr) w.dyn[v * g.gw + u] = 1;
        }
    }
  const int sx = (int)std::floor((s.x - g.ox) / g.res), sy = (int)std::floor((s.y - g.oy) / g.res);
  gt_dijkstra(g, sx, sy, w);
  o.last_nf = (int)fr.size();
  // 남은 굵은 프론티어 가르기: 정답 갈 수 있는 바닥 칸을 품음(몸통이 갈 수 있는 곳) / 안 품음(가구 밑·속, 집 밖 — 몸통은 못 가고 라이다·자세 오차로만 생긴 칸)
  o.last_nf_floor = 0;
  for (int k : fr) {
    bool fl = false;
    for (int yy = (k / cw) * f; yy < (k / cw) * f + f && yy < g.gh && !fl; ++yy)
      for (int xx = (k % cw) * f; xx < (k % cw) * f + f && xx < g.gw && !fl; ++xx) fl = g.stand[yy * g.gw + xx] != 0;
    o.last_nf_floor += fl;
  }
  if (mdp::closed(r) && o.t_closed_first < 0) o.t_closed_first = t;   // 성공 판정과 같은 닫힘을 처음 본 시각
  if (fr.empty()) {   // 닫힘 (또는 아직 설 칸 없음) — 멈춰 기다림
    o.n_closed_dec++;
    o.phase = kClosedWait;
    o.path.clear();
    o.tgt_coarse = -1;
    return;
  }
  // 목표: 굵은 프론티어마다 가운데 0.5 m 안·보이는 설 수 있는 칸 중 가장 가까운 것
  int best_cell = -1, best_c = -1, best_d = kInf, cur_d = kInf, cur_cell = -1;
  // 볼 자리 반경: 본 칸 규칙(라이다 시야 안 close_look)이 굵은 칸을 덮게 close_look − 0.25 m (굵은 칸 반 대각 0.14 m + 장착 앞 0.1 m), 0.5 m 아래로는 안 줄임.
  // 도착하면 그 칸 쪽으로 돌아 3 s 보므로 시야에 듦
  int RR = std::max(10, (int)std::floor((c.close_look - 0.25f) / g.res));
  std::vector<int> unview;   // 볼 자리가 없는 굵은 프론티어
  for (int k : fr) {
    auto it = o.black.find(k);
    if (it != o.black.end() && it->second > t) continue;
    const int ccx = k % cw, ccy = k / cw;
    const float fxw = g.ox + (ccx * f + 0.5f * f) * g.res, fyw = g.oy + (ccy * f + 0.5f * f) * g.res;
    const int fcx = (int)std::floor((fxw - g.ox) / g.res), fcy = (int)std::floor((fyw - g.oy) / g.res);
    int bd = kInf, bc = -1;
    for (int dy = -RR; dy <= RR; ++dy)
      for (int dx = -RR; dx <= RR; ++dx) {
        if (dx * dx + dy * dy > RR * RR) continue;
        const int u = fcx + dx, v = fcy + dy;
        if (u < 0 || v < 0 || u >= g.gw || v >= g.gh) continue;
        const int j = v * g.gw + u;
        if (!g.stand[j] || w.dist[j] >= kInf) continue;
        // 볼 자리: 제자리 회전이 넉넉하고(막힘까지 ≥ 0.25 m — 몸통 반대각 0.194 m + 칸 반 대각) 라이다 높이로 그 굵은 칸이 보이는 칸 먼저,
        // 없으면 설 수 있는 칸 아무 데나(값 + 40 — 본 칸 규칙은 거기 서기만 해도 그 칸을 본 것으로 셈)
        const bool good = g.dist[j] >= 0.25f && gt_sees_block(g, u, v, ccx * f, ccy * f, f);
        const int cost = w.dist[j] + (good ? 0 : 40);
        if (cost >= bd) continue;
        bd = cost;
        bc = j;
      }
    if (bc < 0) { unview.push_back(k); continue; }
    if (k == o.tgt_coarse) { cur_d = bd; cur_cell = bc; }
    if (bd < best_d) { best_d = bd; best_cell = bc; best_c = k; }
  }
  o.last_unview = (int)unview.size();
  o.wall_target = 0;
  if (best_cell < 0 && !unview.empty()) {
    // 볼 수 없는 프론티어(집 밖·가구 속으로 샌 칸 — 지도 자세 오차로 빔이 벽 칸을 지나가 빈 칸이 됨)는 그 둘레 벽이 지도에서 막힘이 되면 번진 곳에서 빠진다.
    // 그래서 그 프론티어에서 굵은 칸 3 칸 안, 정답으로 막힌 칸이 있는데 지도는 아직 막힘이 아닌 굵은 칸(덜 굳은 벽)을 마주 보러 감
    std::vector<uint8_t> nearU(cw * ch, 0);
    for (int k : unview)
      for (int dy = -3; dy <= 3; ++dy)
        for (int dx = -3; dx <= 3; ++dx) {
          const int u = k % cw + dx, v = k / cw + dy;
          if (u >= 0 && v >= 0 && u < cw && v < ch) nearU[v * cw + u] = 1;
        }
    RR = 12;
    for (int k = 0; k < cw * ch; ++k) {
      if (!nearU[k] || w.cL[k] != (int16_t)-1000) continue;   // 굵은 지도에서 갈 수 있음(막힘 아님·모름 아님)
      const int ccx = k % cw, ccy = k / cw;
      bool leaky = false;
      for (int yy = ccy * f; yy < ccy * f + f && yy < g.gh && !leaky; ++yy)
        for (int xx = ccx * f; xx < ccx * f + f && xx < g.gw && !leaky; ++xx) leaky = g.blocked[yy * g.gw + xx] != 0;
      if (!leaky) continue;
      auto it = o.black.find(k);
      if (it != o.black.end() && it->second > t) continue;
      const int fcx = ccx * f + f / 2, fcy = ccy * f + f / 2;
      for (int dy = -RR; dy <= RR; ++dy)
        for (int dx = -RR; dx <= RR; ++dx) {
          if (dx * dx + dy * dy > RR * RR) continue;
          const int u = fcx + dx, v = fcy + dy;
          if (u < 0 || v < 0 || u >= g.gw || v >= g.gh) continue;
          const int j = v * g.gw + u;
          if (!g.stand[j] || g.dist[j] < 0.25f || w.dist[j] >= best_d) continue;
          if (!gt_sees_block(g, u, v, ccx * f, ccy * f, f)) continue;
          if (k == o.tgt_coarse) { cur_d = w.dist[j]; cur_cell = j; }
          best_d = w.dist[j];
          best_cell = j;
          best_c = k;
        }
    }
    if (best_cell >= 0) { o.wall_target = 1; o.n_wall_dec++; }
  }
  if (best_cell < 0) {   // 프론티어가 있는데 정답으로 닿을 곳이 없음 (또는 모두 뺌) — 멈춰 기다림 (지도가 더 안 바뀌면 판정은 시뮬 몫)
    o.n_unreach++;
    o.last_nreach = 0;
    o.phase = kNoTarget;
    o.path.clear();
    o.tgt_coarse = -1;
    return;
  }
  o.last_nreach = 1;
  // 지금 목표를 유지 (훨씬 가까운 게 생기면 바꿈)
  int pick = best_cell, pick_c = best_c, pick_d = best_d;
  if (cur_cell >= 0 && !(best_d + 20 < cur_d * 0.7)) { pick = cur_cell; pick_c = o.tgt_coarse; pick_d = cur_d; }
  if (pick_c != o.tgt_coarse) { o.tgt_t0 = t; o.phase = kGo; o.last_move_t = t; o.mvx = s.x; o.mvy = s.y; }
  if (o.id == g_dbg_env) std::fprintf(stderr, "t %.1f 결정: 굵은 프론티어 %zu, 목표 %d (%d,%d) 비용 %.2f m, 지금 %d, 경로 칸 %d pose %.2f %.2f since_occ %d occ_sum %d\n", t, fr.size(), pick_c, pick_c % cw, pick_c / cw, pick_d * g.res, o.tgt_coarse, w.dist[pick], s.x, s.y, s.since_occ, s.occ_sum);
  o.tgt_coarse = pick_c;
  o.tgt_cost = pick_d * g.res;
  o.tgt_fx = g.ox + ((pick_c % cw) * f + 0.5f * f) * g.res;
  o.tgt_fy = g.oy + ((pick_c / cw) * f + 0.5f * f) * g.res;
  // 경로 (목표 → 시작 뒤로 짚기)
  std::vector<std::pair<float, float>> rev;
  const int X[4] = {-1, 1, 0, 0}, Y[4] = {0, 0, -1, 1};
  int k = pick;
  for (int guard = 0; guard < g.gw * g.gh && k >= 0; ++guard) {
    rev.push_back({g.ox + (k % g.gw + 0.5f) * g.res, g.oy + (k / g.gw + 0.5f) * g.res});
    if (k == sy * g.gw + sx || w.par[k] == 255) break;
    const int d = w.par[k];
    k = (k / g.gw - Y[d]) * g.gw + (k % g.gw - X[d]);
  }
  o.path.assign(rev.rbegin(), rev.rend());
  o.pi = 0;
  if (o.phase != kArrive) o.phase = kGo;
}

// 명령 굴리기: 1 차 지연(τ 0.15 s, 처음 0.08 s 는 지금 속도 — drive.h 지연)
// 보수적 구동 한 걸음(0.1 s): 빨라지는 쪽은 빠르게(v 5 m/s² · ω 30 rad/s²), 느려지는 쪽은 느리게(v 2 m/s² · ω 8 rad/s² — env.yaml floors traction 하한)
float drive_to(float x, float target, float up, float down) {
  const float d = target - x;
  const bool faster = (x >= 0.f && d > 0.f) || (x <= 0.f && d < 0.f);
  const float lim = faster ? up : down;
  return x + std::max(-lim, std::min(lim, d));
}
struct Roll { bool wall_ok; float gap; int k_hit; bool brake_ok; };   // wall_ok: 이 명령 그대로 1 s 안 닿지 않음 · brake_ok: 한 걸음 쓰고 멈춰도 안 닿음
Roll roll(const sim::EnvCfg& c, const Gt& g, const sim::EnvState& s, const std::vector<PPred>& pp, float vc, float wc) {
  // 몸통 여유 1 cm (지금 자세가 이미 여유 안에 닿아 있으면 여유 0 — 벽에 붙은 채로도 돌거나 빠져나오게)
  const float m = body_hit(g, s.x, s.y, s.th, c.half_len + 0.01f, c.half_wid + 0.01f) ? 0.f : 0.01f;
  const float dt = 0.1f, hl = c.half_len + m, hw = c.half_wid + m;
  float x = s.x, y = s.y, th = s.th, v = s.v, w = s.w;
  Roll r{true, 1e9f, 11, true};
  for (const PPred& p : pp) r.gap = std::min(r.gap, rect_gap(x, y, th, c.half_len, c.half_wid, p.x, p.y, p.r));
  for (int k = 1; k <= 20; ++k) {
    // 첫 0.1 s 는 지금 속도(지연 80·60 ms), 그 뒤 drive_to
    if (k > 1) { v = drive_to(v, vc, 0.5f, 0.2f); w = drive_to(w, wc, 3.f, 0.8f); }
    th += w * dt;
    x += v * std::cos(th) * dt;
    y += v * std::sin(th) * dt;
    if (k <= 10 && r.wall_ok && body_hit(g, x, y, th, hl, hw)) { r.wall_ok = false; r.k_hit = k; }
    for (const PPred& p : pp) r.gap = std::min(r.gap, rect_gap(x, y, th, c.half_len, c.half_wid, p.x + p.vx * k * dt, p.y + p.vy * k * dt, p.r));
  }
  // 멈출 수 있나 (DWA admissible): 이 명령을 한 걸음 쓰고 바로 0 을 보내도 1.5 s 안에 안 닿아야 함
  {
    x = s.x; y = s.y; th = s.th; v = s.v; w = s.w;
    for (int k = 1; k <= 15; ++k) {
      if (k == 2 || k == 3) { v = drive_to(v, vc, 0.5f, 0.2f); w = drive_to(w, wc, 3.f, 0.8f); }   // 명령 한 걸음 + 지연 한 걸음
      else if (k > 3) { v = drive_to(v, 0.f, 0.5f, 0.2f); w = drive_to(w, 0.f, 3.f, 0.8f); }
      th += w * dt;
      x += v * std::cos(th) * dt;
      y += v * std::sin(th) * dt;
      if (body_hit(g, x, y, th, hl, hw)) { r.brake_ok = false; break; }
    }
  }
  return r;
}

void control(const sim::EnvCfg& c, const Gt& g, const sim::EnvState& s, Oracle& o, float t, float* v_o, float* w_o) {
  float vn = 0.f, wn = 0.f;
  // 멈춤 감지 (경로가 있는데 8 s 동안 0.3 m 못 감 → 목표 뺌)
  if (std::hypot(s.x - o.mvx, s.y - o.mvy) > 0.3f) { o.mvx = s.x; o.mvy = s.y; o.last_move_t = t; }
  if (o.phase == kGo && !o.path.empty()) {
    const auto& P = o.path;
    // 가장 가까운 점 (앞으로만)
    int bi = o.pi;
    float bd = 1e9f;
    for (int i = o.pi; i < (int)P.size() && i < o.pi + 60; ++i) {
      const float d = std::hypot(P[i].first - s.x, P[i].second - s.y);
      if (d < bd) { bd = d; bi = i; }
    }
    o.pi = bi;
    const float L = 0.45f;
    int j = bi;
    while (j + 1 < (int)P.size() && std::hypot(P[j].first - s.x, P[j].second - s.y) < L) ++j;
    const float gx = P.back().first, gy = P.back().second, dg = std::hypot(gx - s.x, gy - s.y);
    if (dg < 0.15f) {
      o.phase = kArrive;
      o.arrive_t = t;
    } else {
      const float err = dm::wrap_pi(std::atan2(P[j].second - s.y, P[j].first - s.x) - s.th);
      if (std::fabs(err) > 0.7f) {
        wn = std::max(-1.5f, std::min(1.5f, 2.f * err));
      } else {
        const int cx = (int)std::floor((s.x - g.ox) / g.res), cy = (int)std::floor((s.y - g.oy) / g.res);
        const float dw = (cx >= 0 && cy >= 0 && cx < g.gw && cy < g.gh) ? g.dist[cy * g.gw + cx] : 0.f;
        float vmax = dw < 0.3f ? 0.35f : 0.8f;
        vmax = std::min(vmax, 0.25f + dg);
        vn = vmax * std::max(0.3f, std::cos(err));
        const float ld = std::max(0.2f, std::hypot(P[j].first - s.x, P[j].second - s.y));
        wn = std::max(-2.f, std::min(2.f, vn * 2.f * std::sin(err) / ld));
      }
    }
    if (t - o.last_move_t > 8.f && o.phase == kGo) { if (o.id == g_dbg_env) std::fprintf(stderr, "t %.1f 뺌 %s 칸 %d pose %.2f %.2f\n", t, "stuck", o.tgt_coarse, s.x, s.y); o.black[o.tgt_coarse] = t + 30.f; o.n_black++; o.phase = kNoTarget; o.path.clear(); o.tgt_coarse = -1; }
    if (o.phase == kGo && t - o.tgt_t0 > 20.f + 4.f * o.tgt_cost) { if (o.id == g_dbg_env) std::fprintf(stderr, "t %.1f 뺌 %s 칸 %d pose %.2f %.2f\n", t, "timeout", o.tgt_coarse, s.x, s.y); o.black[o.tgt_coarse] = t + 30.f; o.n_black++; o.phase = kNoTarget; o.path.clear(); o.tgt_coarse = -1; }
  }
  if (o.phase == kArrive) {   // 프론티어 쪽으로 돌아 3 s
    const float err = dm::wrap_pi(std::atan2(o.tgt_fy - s.y, o.tgt_fx - s.x) - s.th);
    if (std::fabs(err) > 0.25f) wn = std::max(-1.5f, std::min(1.5f, 2.f * err));
    if (t - o.arrive_t > 3.f) {
      if (o.id == g_dbg_env) std::fprintf(stderr, "t %.1f 뺌 %s 칸 %d pose %.2f %.2f\n", t, "arrived", o.tgt_coarse, s.x, s.y);
      o.black[o.tgt_coarse] = t + 30.f; o.n_black++; o.phase = kNoTarget; o.path.clear(); o.tgt_coarse = -1;
    }
  }
  // 안전: 정답 벽·사람
  const std::vector<PPred> pp = predict_people(c, s);
  const Roll rn = roll(c, g, s, pp, vn, wn);
  if (o.id == g_dbg_env) std::fprintf(stderr, "  t %.1f 경로 명령 (%.2f %.2f) 벽 통과 %d 사람 간격 %.2f phase %d path %zu pi %d v %.2f w %.2f th %.2f\n", t, vn, wn, (int)rn.wall_ok, rn.gap, o.phase, o.path.size(), o.pi, s.v, s.w, s.th);
  if (rn.wall_ok && rn.brake_ok && rn.gap > 0.2f) { *v_o = vn; *w_o = wn; return; }
  const float vs[11] = {vn, 0.6f, 0.3f, 0.1f, 0.05f, 0.f, -0.05f, -0.1f, -0.15f, -0.3f, -0.5f},
              ws[9] = {wn, 0.f, 0.4f, -0.4f, 0.8f, -0.8f, 1.6f, -1.6f, 2.5f};
  float best = -1e9f, bv = 0.f, bw = 0.f;
  bool any = false;
  for (float v : vs)
    for (float w : ws) {
      const Roll r = roll(c, g, s, pp, v, w);
      if (!r.brake_ok) continue;   // 이번 걸음에 써도 멈출 수 있어야 함 (걸음마다 다시 고름)
      const float score = 10.f * std::min(r.gap, 0.35f) - std::fabs(v - vn) - 0.3f * std::fabs(w - wn) - (r.wall_ok ? 0.f : 2.f);
      if (score > best) { best = score; bv = v; bw = w; any = true; }
    }
  if (any && best < -1.f) o.n_wedge++;
  if (!pp.empty() && rn.gap <= 0.2f) o.n_people_override++;
  if (!any) { o.n_stop++; bv = 0.f; bw = 0.f; }
  if (o.id == g_dbg_env) std::fprintf(stderr, "  t %.1f 명령 경로 (%.2f %.2f) 벽 %d 사람 %.2f → (%.2f %.2f)%s\n", t, vn, wn, (int)rn.wall_ok, rn.gap, bv, bw, any ? "" : " 후보 없음");
  *v_o = bv;
  *w_o = bw;
}

// 그림: 정답 막힘 검정, 로봇 지도 모름 회색 · 갈 수 있음 흰색 · 막힘 갈색, 굵은 프론티어 파랑, 목표 굵은 칸 주황, 로봇 초록, 사람 빨강
void dump_ppm(const std::string& path, const sim::EnvCfg& c, const Gt& g, const int16_t* L, int S, const uint32_t* trail, const sim::EnvState& s, const Oracle& o,
              const uint8_t* cam = nullptr) {
  // 그림: 정답 막힘 검정, 로봇 지도 모름 회색 · 갈 수 있음 흰색 · 막힘 갈색, 지나간 칸 연두, 닫힘을 막는 칸 파랑, 목표 굵은 칸 주황, 로봇 초록, 사람 빨강
  const int n = g.gw * g.gh, ws = (S + 31) / 32, cw = cdim(g.gw);
  std::vector<uint8_t> fl(n);
  std::vector<int32_t> qa(n), qb(n), lab(n);
  std::vector<mdp::Frontier> out(n);
  const mdp::MapView mv{L, S, g.gw, g.gh};
  mdp::SearchBuf sb{fl.data(), qa.data(), qb.data(), nullptr, lab.data()};
  sb.vis = trail; sb.look = trail + (size_t)S * ws; sb.tws = ws; sb.close_min = c.close_min; sb.rbc = mdp::robot_cells(c.close_radius, c.res);
  sb.cam = cam;
  const int mx = (int)std::floor((s.x + s.ex - g.ox) / g.res), my = (int)std::floor((s.y + s.ey - g.oy) / g.res);
  const mdp::Robot rb = mdp::obsgen::reach_robot(c);
  const mdp::SearchResult r = mdp::frontier_search(mv, mx, my, rb, 1, sb, out.data(), n);
  if (std::getenv("ORACLE_DUMP_STATS"))
    std::fprintf(stderr, "%s: start %d reach0 %d reach %d cells %d block %d closed %d map_closed %d since_occ %d\n", path.c_str(), r.start, r.n_reach0, r.n_reach, r.n_cells,
                 r.n_block, (int)mdp::closed(r), s.map_closed, s.since_occ);
  std::vector<uint8_t> isb(n, 0);
  for (int q = 0; q < r.n_reach; ++q) {
    const int k = qa[q];
    if ((fl[k] & 2) && out[lab[k]].n >= c.close_min && !mdp::looked_cell(mv, sb.look, sb.vis, ws, k % g.gw, k / g.gw)) isb[k] = 1;
  }
  const int Z = 3;
  FILE* fp = std::fopen((path + ".tmp").c_str(), "wb");
  std::fprintf(fp, "P6\n%d %d\n255\n", g.gw * Z, g.gh * Z);
  const int rcx = (int)std::floor((s.x - g.ox) / g.res), rcy = (int)std::floor((s.y - g.oy) / g.res);
  for (int yy = g.gh * Z - 1; yy >= 0; --yy)
    for (int xx = 0; xx < g.gw * Z; ++xx) {
      const int x = xx / Z, y = yy / Z, k = y * g.gw + x, ck = (y / kF) * cw + x / kF;
      uint8_t p[3] = {128, 128, 128};
      const uint8_t cl = mdp::cls_at(mv, x, y);
      if (cl == mdp::kFree) { p[0] = p[1] = p[2] = 235; }
      else if (cl == mdp::kOcc) { p[0] = 140; p[1] = 80; p[2] = 30; }
      if (g.blocked[k]) { p[0] = p[1] = p[2] = cl == mdp::kOcc ? 60 : 0; }
      if (mdp::plane_bit(trail, ws, x, y)) { p[0] = 150; p[1] = 220; p[2] = 150; }
      else if (mdp::plane_bit(sb.look, ws, x, y) && cl == mdp::kUnk) { p[0] = 150; p[1] = 150; p[2] = 175; }
      if (isb[k]) { p[0] = 30; p[1] = 60; p[2] = 255; }
      if (ck == o.tgt_coarse) { p[0] = 255; p[1] = 150; p[2] = 0; }
      if ((x - rcx) * (x - rcx) + (y - rcy) * (y - rcy) <= 9) { p[0] = 0; p[1] = 200; p[2] = 0; }
      for (int j = 0; j < s.people.n; ++j) {
        const float dx = g.ox + (x + 0.5f) * g.res - s.people.cyl[j].x, dy = g.oy + (y + 0.5f) * g.res - s.people.cyl[j].y;
        if (dx * dx + dy * dy <= s.people.cyl[j].r * s.people.cyl[j].r) { p[0] = 230; p[1] = 0; p[2] = 0; }
      }
      std::fwrite(p, 1, 3, fp);
    }
  std::fclose(fp);
  std::rename((path + ".tmp").c_str(), path.c_str());
}

std::string stem(const std::string& p) {
  std::string b = p.substr(p.find_last_of('/') + 1);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}

struct Res { int env, map, done, coll_kind; float t, cov; std::string name; Oracle o; };

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(k + " 뒤에 값"); return argv[++i]; };
    if (k == "--policy") a.policy = nx();
    else if (k == "--stage") a.stage = std::stoi(nx());
    else if (k == "--people") a.people = nx() == "on";
    else if (k == "--envs") a.envs = std::stoi(nx());
    else if (k == "--per-stage") a.per_stage = std::stoi(nx());
    else if (k == "--seed") a.seed = std::stoull(nx());
    else if (k == "--list") a.list = nx();
    else if (k == "--out") a.out = nx();
    else if (k == "--threads") a.threads = std::max(1, std::stoi(nx()));
    else if (k == "--env") a.env = nx();
    else if (k == "--dump") a.dump = nx();
    else if (k == "--dump-fail") a.dump_fail = nx();
    else if (k == "--dump-env") a.dump_env = std::stoi(nx());
    else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
  }
  if (a.policy != "oracle" && a.policy != "slot0") { std::fprintf(stderr, "--policy oracle|slot0\n"); return 2; }
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  hc.cfg.curriculum = 1;   // 단계 규칙(맵 묶음·오차 배율·사람 수)을 학습과 같게 — 시작 단계만 놓음
  // oracle: 속도 행동(물리 단위). 시뮬 안전층(직전 스캔 앞 길 점 → 앞 명령 0)은 끔 — 정답 격자·사람으로 더 엄격히 스스로 막음 (frontier_run 기준선과 같은 설정).
  // ORACLE_SAFETY=1 이면 켬 (비교용)
  if (a.policy == "oracle") {
    hc.cfg.action_mode = sim::kActVelocity;
    hc.cfg.act_raw = 1;
    const char* sf = std::getenv("ORACLE_SAFETY");
    hc.cfg.safety = (sf && sf[0] == '1') ? hc.cfg.safety : 0;
  }
  else hc.cfg.action_mode = sim::kActPath;
  if (!a.people) hc.cfg.people.spawn_keep = 1e4f;   // 사람 0 명
  const sim::EnvCfg& c = hc.cfg;
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);

  // 학습 목록에서 단계마다 고르게 per_stage 개
  std::vector<std::pair<std::string, int>> items;
  {
    std::ifstream f(a.list);
    if (!f) { std::fprintf(stderr, "목록을 못 엶: %s\n", a.list.c_str()); return 1; }
    const std::string dir = a.list.find('/') == std::string::npos ? std::string() : a.list.substr(0, a.list.find_last_of('/') + 1);
    std::string line;
    std::vector<std::vector<std::string>> by(mdp::cur::kNumStages);
    while (std::getline(f, line)) {
      std::vector<std::string> col;
      std::stringstream ss(line);
      std::string t;
      while (std::getline(ss, t, '\t')) col.push_back(t);
      if (col.size() < 4 || col[0] == "path" || col[0].empty() || col[0][0] == '#') continue;
      const int g = std::stoi(col[3]);
      by[std::max(0, std::min(mdp::cur::kNumStages - 1, g))].push_back(col[0][0] == '/' ? col[0] : dir + col[0]);
    }
    for (int s = 0; s < mdp::cur::kNumStages; ++s) {
      const int n = (int)by[s].size(), k = std::min(n, a.per_stage);
      for (int i = 0; i < k; ++i) items.push_back({by[s][(size_t)((long long)i * n / k)], s});
    }
  }
  std::vector<sim::World> worlds(items.size());
  {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < a.threads; ++t)
      th.emplace_back([&]() {
        for (size_t k; (k = next.fetch_add(1)) < items.size();) {
          worlds[k] = sim::build_world(stem(items[k].first), sim::load_scene(items[k].first), hc);
          worlds[k].stage = items[k].second;
        }
      });
    for (auto& x : th) x.join();
  }
  std::vector<Gt> gts(worlds.size());
  {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < a.threads; ++t)
      th.emplace_back([&]() { for (size_t k; (k = next.fetch_add(1)) < worlds.size();) { gts[k] = make_gt(worlds[k], c.res); gts[k].lidar_z = c.lidar_z; } });
    for (auto& x : th) x.join();
  }
  const int N = a.envs, S = c.size;
  sim::Env env(hc, worlds, lidar, N, a.seed, true);
  {
    std::vector<sim::EnvState> es;
    env.get_state(es);
    for (int i = 0; i < N; ++i) { es[i].stage = a.stage; es[i].cur_hist = 0u; es[i].cur_n = 0; env.set_state(i, es[i]); }
    env.reset();
    env.sync();
  }
  const float dt_act = c.substeps * c.dt;
  const int per_call = a.policy == "slot0" ? c.decide_steps : 1;
  float* d_act = nullptr;
  cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * N);
  std::vector<float> act((size_t)mdp::act::DIM * N, 0.f);
  cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
  std::vector<sim::EnvState> st;
  std::vector<int16_t> L;
  std::vector<uint32_t> T;
  std::vector<uint8_t> CV;
  std::vector<float> CP;
  std::vector<float> rew;
  std::vector<int32_t> done;
  std::vector<sim::EpInfo> ep;
  std::vector<Oracle> orc(N);
  for (int i = 0; i < N; ++i) orc[i].id = i;
  if (const char* d = std::getenv("ORACLE_DEBUG")) g_dbg_env = std::atoi(d);
  std::vector<Work> work(a.threads);
  std::vector<int> fin(N, 0), first_map(N, -1);
  std::vector<Res> res(N);
  int left = N;
  const int max_calls = c.max_steps / per_call + 20;
  env.get_state(st);
  for (int i = 0; i < N; ++i) first_map[i] = st[i].map;
  for (int call = 0; left > 0 && call < max_calls; ++call) {
    if (a.policy == "oracle") {
      env.get_state(st);
      const bool dec = call % c.decide_steps == 0;
      if (dec) { env.get_map(L); env.get_trail(T); }
      const float t = call * dt_act;
      std::atomic<int> next{0};
      std::vector<std::thread> th;
      for (int tt = 0; tt < a.threads; ++tt)
        th.emplace_back([&, tt]() {
          for (int i; (i = next.fetch_add(1)) < N;) {
            float* A = act.data() + (size_t)mdp::act::DIM * i;
            A[0] = A[1] = A[2] = A[3] = 0.f;
            if (fin[i]) continue;
            const sim::EnvState& s = st[i];
            const Gt& g = gts[s.map];
            const uint32_t* Ti = T.data() + (size_t)i * 2 * S * ((S + 31) / 32);
            if (dec) { decide(c, g, L.data() + (size_t)i * S * S, S, Ti, s, orc[i], work[tt], t); orc[i].last_s = s; }
            if (dec && i == a.dump_env && !a.dump.empty()) dump_ppm(a.dump, c, g, L.data() + (size_t)i * S * S, S, Ti, s, orc[i]);
            float v, w;
            control(c, g, s, orc[i], t, &v, &w);
            A[0] = v;
            A[2] = w;
          }
        });
      for (auto& x : th) x.join();
      cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    }
    if (a.policy == "slot0" && !a.dump_fail.empty()) {   // slot0 도 실패 판 그림을 남김 (결정마다 지도·지나간 길·상태)
      env.get_state(st);
      env.get_map(L);
      env.get_trail(T);
      env.get_camlayer(CV, CP);
      for (int i = 0; i < N; ++i) orc[i].last_s = st[i];
    }
    env.step(d_act);
    env.sync();
    env.get_reward(rew, done);
    bool any = false;
    for (int i = 0; i < N; ++i) any = any || (!fin[i] && done[i]);
    if (!any) continue;
    env.get_ep(ep);
    for (int i = 0; i < N; ++i) {
      if (fin[i] || !done[i]) continue;
      fin[i] = 1;
      --left;
      Res& r = res[i];
      r.env = i; r.map = ep[i].map; r.done = done[i]; r.coll_kind = ep[i].coll_kind; r.t = ep[i].t; r.cov = ep[i].coverage;
      r.name = worlds[ep[i].map].name;
      r.o = orc[i];
      if (!a.dump_fail.empty() && (done[i] != mdp::cur::kDeclareOk || ep[i].coverage < 0.8f)) {
        char tag[160];
        std::snprintf(tag, sizeof tag, "%s/s%d_%s_e%d_d%d_%s.ppm", a.dump_fail.c_str(), a.stage, a.people ? "on" : "off", i, done[i], r.name.c_str());
        dump_ppm(tag, c, gts[orc[i].last_s.map], L.data() + (size_t)i * S * S, S, T.data() + (size_t)i * 2 * S * ((S + 31) / 32), orc[i].last_s, orc[i],
                 CV.empty() ? nullptr : CV.data() + (size_t)i * S * S);
      }
    }
    if (call % (600 / per_call) == 0)
      std::fprintf(stderr, "  [%s 단계 %d 사람 %s] %.0f s, 남은 판 %d\n", a.policy.c_str(), a.stage, a.people ? "켬" : "끔", call * per_call * dt_act, left);
  }
  cudaFree(d_act);
  // 끝나지 않은 판 = 끊김
  for (int i = 0; i < N; ++i)
    if (!fin[i]) { res[i].env = i; res[i].map = first_map[i]; res[i].done = mdp::cur::kTruncated; res[i].t = c.max_steps * dt_act; res[i].cov = 0; res[i].name = worlds[first_map[i]].name; res[i].o = orc[i]; }
  // 요약
  double n = N, succ = 0, cw = 0, cp = 0, idle = 0, trunc = 0, fclose = 0, ts = 0, ts2 = 0, cov_s = 0;
  int bad_map = 0;
  for (const Res& r : res) {
    succ += r.done == mdp::cur::kDeclareOk;
    cw += r.done == mdp::cur::kCollision;
    cp += r.done == mdp::cur::kCollisionPerson;
    idle += r.done == mdp::cur::kNoProgress;
    trunc += r.done == mdp::cur::kTruncated;
    if (r.done == mdp::cur::kDeclareOk) { fclose += r.cov < 0.8f; ts += r.t; ts2 += (double)r.t * r.t; cov_s += r.cov; }
    bad_map += r.map != first_map[r.env];
  }
  auto se = [&](double k) { const double p = k / n; return std::sqrt(p * (1 - p) / n); };
  const double tm = succ > 0 ? ts / succ : 0, tsd = succ > 1 ? std::sqrt(std::max(0.0, ts2 / succ - tm * tm)) : 0;
  std::printf("요약\t%s%s\t단계 %d\t사람 %s\tn %d\t성공 %.3f±%.3f\t벽충돌 %.3f±%.3f\t사람충돌 %.3f±%.3f\t진행없음 %.3f±%.3f\t끊김 %.3f\t거짓닫힘 %.3f\t성공까지 %.1f s (sd %.1f)\t성공 덮음 %.3f\t맵 어긋남 %d\n",
              a.policy.c_str(), a.policy == "oracle" ? "(닫힘 규칙)" : "", a.stage, a.people ? "켬" : "끔", N, succ / n, se(succ), cw / n, se(cw), cp / n, se(cp), idle / n, se(idle), trunc / n,
              succ > 0 ? fclose / succ : 0.0, tm, tsd, succ > 0 ? cov_s / succ : 0.0, bad_map);
  if (!a.out.empty()) {
    std::ofstream f(a.out);
    f << "policy\tstage\tpeople\tenv\tmap\tdone\tt\tcov\tcoll_kind\tt_closed_first\tn_dec\tn_unreach\tn_black\tn_closed_dec\tlast_nf\tlast_nreach\tlast_phase\tn_people_override\tn_stop\tlast_unview\tn_wall_dec\tlast_nf_floor\tn_wedge\n";
    for (const Res& r : res)
      f << a.policy << '\t' << a.stage << '\t' << (a.people ? 1 : 0) << '\t' << r.env << '\t' << r.name << '\t' << r.done << '\t' << r.t << '\t' << r.cov << '\t'
        << r.coll_kind << '\t' << r.o.t_closed_first << '\t' << r.o.n_dec << '\t' << r.o.n_unreach << '\t' << r.o.n_black << '\t' << r.o.n_closed_dec << '\t'
        << r.o.last_nf << '\t' << r.o.last_nreach << '\t' << r.o.phase << '\t' << r.o.n_people_override << '\t' << r.o.n_stop << '\t' << r.o.last_unview << '\t' << r.o.n_wall_dec << '\t' << r.o.last_nf_floor << '\t' << r.o.n_wedge << '\n';
  }
  return 0;
}
