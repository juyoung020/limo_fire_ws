// 환경 한 걸음 CPU 참조판: GPU 커널(src/env_kernel.cu)의 블록 하나를 스레드 순서대로 흉내 낸다. 단계 순서는 커널과 한 줄씩 같다
#include <algorithm>
#include <thread>
#include <vector>

#include "env_api.h"
#include "obs_ref.h"

namespace sim {
namespace {

int g_cpu_threads = 1;
CpuTap* g_tap = nullptr;

// 환경 [0, n) 을 스레드 g_cpu_threads 개에 고르게 나눠 f(env) — 환경끼리 독립(상태·지도·관측이 환경마다 따로)이라 결과는 스레드 수와 무관
template <class F>
void for_envs(int n, F f) {
  const int T = std::max(1, std::min(g_cpu_threads, n));
  if (T == 1) {
    for (int env = 0; env < n; ++env) f(env);
    return;
  }
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      for (int env = t; env < n; env += T) f(env);   // 번갈아 나눔: 맵 크기가 섞여 짐이 고름
    });
  for (auto& x : th) x.join();
}

void reset_env(const EnvBatch& b, EnvState& e, int env) {
  for (int tid = 0; tid < kThreads; ++tid) reset_map(b, env, tid);
  reset_state(b, e);
}


// 계층형 결정 (GPU env_plan_kernel 과 같은 규칙·같은 경로): 엄격(지도 막힘 ∪ 약한 막힘) → 못 닿으면 느슨(지도 막힘만).
// 거리 배열 위 Dial 최단 경로(무게 hier_path.h) → 거리 = 내 거리 − 내 무게 인 첫 nb4 이웃으로 뒤로 짚기
void plan_cpu(const EnvBatch& b, EnvState& e, int env, int k) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const mdp::MapView mv{L, c.size, m.gw, m.gh};
  const mdp::Robot rb = path_rb_plan(c, mdp::obsgen::reach_robot(c)), rm = path_rb_mid(c), ro = path_rb_out(c);   // rb: 계획 반경 (path_consistent 면 충돌 검사 발자국 반경, hier_path.h)
  int force = 0;   // 복구 (hier_path.h path_recovery): 다시 계획 · 포기한 목표면 다른 후보(없으면 −1)
  k = path_recovery(b, e, env, k, &force);
  e.p_slot = k;
  const int tgt = k >= 0 ? act_cells(b, env)[k] : -1;
  if (tgt < 0) {
    e.goal_cell = -1;
    path_set_none(e);
    return;
  }
  if (e.kb_x1 < 0) {
    e.goal_cell = tgt;
    path_set_none(e);
    return;
  }
  const size_t n = (size_t)m.gw * m.gh;
  thread_local std::vector<uint8_t> lt, flag;
  thread_local std::vector<int32_t> d2l, d2o, dist, tent, qa;
  if (lt.size() < n) { lt.resize(n); flag.resize(n); d2l.resize(n); d2o.resize(n); dist.resize(n); tent.resize(n); qa.resize(n); }
  int rcx, rcy;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &rcx, &rcy);
  const int x0 = e.kb_x0, x1 = e.kb_x1, y0 = e.kb_y0, y1 = e.kb_y1, R = ro.r;
  const float rr = c.path_obs_range * c.path_obs_range;
  // 막힘 칸: 지도 막힘(occ)·엄격 막힘(lt = occ ∪ 로그 오즈 > 0 이고 로봇 칸에서 obstacle_max_range 안 — path_lethal 과 같은 규칙)
  const int ex0 = std::max(0, x0 - R), ex1 = std::min(m.gw - 1, x1 + R), ey0 = std::max(0, y0 - R), ey1 = std::min(m.gh - 1, y1 + R);
  for (int y = ey0; y <= ey1; ++y)
    for (int x = ex0; x <= ex1; ++x) {
      const int16_t l = L[(size_t)y * c.size + x];
      uint8_t v = mdp::cell_class(l) == mdp::kOcc ? 3 : 0;   // bit0 occ · bit1 lt · bit2 지금 스캔 (아래) — 느슨 막힘 = bit0 · bit2
      if (!v && l != mdp::kUnknownL && l > 0) {
        const float dx = (float)(x - rcx) * c.res, dy = (float)(y - rcy) * c.res;
        if (dx * dx + dy * dy <= rr) v = 2;
      }
      lt[(size_t)y * m.gw + x] = v;
    }
  // 지금 스캔 끝점 (hier_path.h "지금 스캔" (a)): bit2 — 엄격·느슨 둘 다 막힘 (GPU LT·LO 판)
  if (c.path_scan)
    for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
      int x, y;
      if (!path_scan_cell(b, e, env, k, &x, &y)) continue;
      if (x < ex0 || x > ex1 || y < ey0 || y > ey1) continue;
      lt[(size_t)y * m.gw + x] |= 4;
    }
  // 깊이 카메라 장애물 층 (hier_path.h "깊이 카메라 장애물 층"): 지금 스캔과 같은 몫 — 엄격·느슨 둘 다 막힘 (GPU LT·LO 판)
  if (b.cvox)
    for (int y = ey0; y <= ey1; ++y)
      for (int x = ex0; x <= ex1; ++x)
        if (cam_lethal(b, env, x, y)) lt[(size_t)y * m.gw + x] |= 4;
  // 막힘까지 가장 가까운 거리² (칸, 반경 ro 안만) — 범위 안 칸마다 occ·lt 따로. mdp::standable 과 같은 원판 규칙
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) d2l[(size_t)y * m.gw + x] = d2o[(size_t)y * m.gw + x] = 1 << 30;
  for (int y = ey0; y <= ey1; ++y)
    for (int x = ex0; x <= ex1; ++x) {
      const uint8_t v = lt[(size_t)y * m.gw + x];
      if (!v) continue;
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int u = x + dx, w = y + dy;
          if (u < x0 || w < y0 || u > x1 || w > y1) continue;
          const int d2 = dx * dx + dy * dy;
          int32_t& ql = d2l[(size_t)w * m.gw + u];
          ql = std::min(ql, d2);
          if (v & 5) {
            int32_t& qo = d2o[(size_t)w * m.gw + u];
            qo = std::min(qo, d2);
          }
        }
    }
  const auto inbox = [&](int x, int y) { return x >= x0 && y >= y0 && x <= x1 && y <= y1; };
  const auto known = [&](int x, int y) { return mdp::cls_at(mv, x, y) != mdp::kUnk; };
  // mode 1 엄격 · 2 느슨: 설 수 있음, 무게 (GPU: SST·ST 판과 stand_word(LT·OCC) 무게 경계)
  const mdp::Robot rb3 = mdp::obsgen::reach_robot(c);
  const auto pass = [&](int mode, int x, int y) -> bool {
    if (!inbox(x, y) || !known(x, y)) return false;
    const size_t i = (size_t)y * m.gw + x;
    // mode 3: 느슨 막힘 + 관측 BFS 반경 (계획 반경으로 못 짤 때 — hier_path.h "계획·충돌 검사 일치" (e))
    return mode == 1 ? (!lt[i] && d2l[i] > rb.r2) : (!(lt[i] & 5) && d2o[i] > (mode == 3 ? rb3.r2 : rb.r2));
  };
  const auto weight = [&](int mode, int x, int y) -> int {
    if (!pass(mode, x, y)) return 0;
    const size_t i = (size_t)y * m.gw + x;
    const int32_t d = mode == 1 ? d2l[i] : d2o[i];
    return path_weight(true, d > rm.r2, d > ro.r2);
  };
  // 같은 후보 · 경로 그대로?
  const int pm = e.p_mode == 1 ? 1 : 2;
  if (!force && act_same(b, e, env, k) && e.goal_cell >= 0 && e.p_n > 0 && !e.p_trunc && path_clear(e, [&](int x, int y) { return pass(pm, x, y); }))
    return;
  e.goal_cell = tgt;
  path_set_none(e);
  e.p_plans += 1;
  // 한 번 계획 (mode): 시작 = 로봇 칸에서 FIFO 4 이웃 BFS 로 처음 만나는 설 수 있는 칸 (frontier_search nearestCell 과 같은 차례), Dial
  int start = -1, goal = tgt;
  // 목표 칸들 (hier_path.h "목표 허용"): 가까운 목표 = 목표가 설 수 있으면 목표 하나, 아니면 목표에서 path_goal_tol(Smac tolerance) 안 설 수 있는 칸 —
  // 처음 닿는 층에서 멈춤. 못 닿고 BFS 가 끝나면 목표에서 path_goal_far 안 닿은 칸 중 목표에 가장 가까운 칸 (Smac 이 tolerance 안 가장 가까운 노드를 내는 꼴)
  const mdp::Robot rt = path_rb_tol(c), rf = path_rb_far(c);
  const int gx = tgt % m.gw, gy = tgt / m.gw;
  const auto in_goal = [&](int mode, int x, int y) -> bool {
    if (x == gx && y == gy) return pass(mode, x, y);
    if (pass(mode, gx, gy) || rt.r2 <= 0) return false;
    return (x - gx) * (x - gx) + (y - gy) * (y - gy) <= rt.r2 && pass(mode, x, y);
  };
  const auto any_goal = [&](int mode) -> bool {
    for (int dy = -rf.r; dy <= rf.r; ++dy)
      for (int dx = -rf.r; dx <= rf.r; ++dx)
        if (dx * dx + dy * dy <= rf.r2 && pass(mode, gx + dx, gy + dy)) return true;
    return pass(mode, gx, gy);
  };
  // BFS 가 끝난 뒤: 먼 허용 안 닿은 칸 중 (거리², 칸) 최소 (GPU 와 같은 규칙)
  const auto far_goal = [&](int mode) -> bool {
    int64_t best = -1;
    for (int dy = -rf.r; dy <= rf.r; ++dy)
      for (int dx = -rf.r; dx <= rf.r; ++dx) {
        const int x = gx + dx, y = gy + dy;
        if (dx * dx + dy * dy > rf.r2 || !pass(mode, x, y) || dist[(size_t)y * m.gw + x] < 0) continue;
        const int64_t key = ((int64_t)(dx * dx + dy * dy) << 20) | (int64_t)(y * m.gw + x);
        if (best < 0 || key < best) { best = key; goal = y * m.gw + x; }
      }
    return best >= 0;
  };
  const auto run = [&](int mode) -> bool {
    start = -1;
    if (rcx < 0 || rcy < 0 || rcx >= m.gw || rcy >= m.gh) return false;
    std::fill(flag.begin(), flag.begin() + n, 0);
    int head = 0, tail = 0;
    qa[tail++] = rcy * m.gw + rcx;
    flag[rcy * m.gw + rcx] = 1;
    while (head < tail) {
      const int idx = qa[head++], x = idx % m.gw, y = idx / m.gw;
      if (pass(mode, x, y)) { start = idx; break; }
      for (int q = 0; q < 4; ++q) {
        int dx, dy;
        mdp::nb4(q, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= m.gw || v >= m.gh) continue;
        const int j = v * m.gw + u;
        if (flag[j]) continue;
        flag[j] = 1;
        qa[tail++] = j;
      }
    }
    if (start < 0) return false;
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) { dist[(size_t)y * m.gw + x] = -1; tent[(size_t)y * m.gw + x] = 1 << 30; }
    std::vector<int32_t> bk[4];
    bk[0].push_back(start);
    tent[start] = 0;
    for (int t = 0;; ++t) {
      if (bk[0].empty() && bk[1].empty() && bk[2].empty() && bk[3].empty()) return far_goal(mode);
      std::vector<int32_t> cur;
      cur.swap(bk[t & 3]);
      int64_t best = -1;   // 층 t 에서 정해진 목표 칸 중 (목표까지 거리², 칸) 가장 작은 것 — GPU 와 같은 규칙
      for (int v : cur) {
        if (dist[v] >= 0 || tent[v] != t) continue;
        dist[v] = t;
        if (v == tgt) { best = 0; goal = tgt; }
        else if (best != 0 && in_goal(mode, v % m.gw, v / m.gw)) {
          const int ddx = v % m.gw - gx, ddy = v / m.gw - gy;
          const int64_t key = ((int64_t)(ddx * ddx + ddy * ddy) << 20) | v;
          if (best < 0 || key < best) { best = key; goal = v; }
        }
        const int x = v % m.gw, y = v / m.gw;
        for (int q = 0; q < 4; ++q) {
          int dx, dy;
          mdp::nb4(q, &dx, &dy);
          const int w = weight(mode, x + dx, y + dy);
          if (w <= 0) continue;
          const int u = (y + dy) * m.gw + x + dx;
          if (dist[u] >= 0 || t + w >= tent[u]) continue;
          tent[u] = t + w;
          bk[(t + w) & 3].push_back(u);
        }
      }
      if (best >= 0) return true;
    }
  };
  int mode = 0;
  if (any_goal(1) && run(1)) mode = 1;
  else if (run(2)) mode = 2;
  else if (c.path_consistent && rb3.r2 != rb.r2 && run(3)) mode = 3;
  if (!mode) return;   // 설 수 있는 칸이 없거나 못 닿음: 목표만 두고 섬
  const auto wfn = [&](int x, int y) { return weight(mode, x, y); };
  thread_local std::vector<int32_t> list;
  list.clear();
  int v = goal;
  list.push_back(v);
  while (v != start) {
    const int x = v % m.gw, y = v / m.gw, want = dist[v] - wfn(x, y);
    int u = -1;
    for (int q = 0; q < 4 && u < 0; ++q) {
      int dx, dy;
      mdp::nb4(q, &dx, &dy);
      if (wfn(x + dx, y + dy) > 0 && dist[(y + dy) * m.gw + x + dx] == want) u = (y + dy) * m.gw + x + dx;
    }
    if (u < 0 || (int)list.size() >= mdp::kReachCap) return;
    v = u;
    list.push_back(v);
  }
  std::reverse(list.begin(), list.end());
  int trunc = 0;
  e.p_n = path_simplify(list.data(), (int)list.size(), m.gw, wfn, e.wp, mdp::hier::kPathMax, &trunc);
  e.p_i = 0;
  e.p_trunc = trunc;
  e.p_mode = mode;
  if (mode >= 2) e.p_fb += 1;
}

// 환경 하나 행동 하나. sub < 0: 기본, sub ≥ 0: 계층형 묶음 sub 번째 걸음 (GPU env_finish_kernel 의 sub 와 같은 규칙)
void step_env(const EnvBatch& b, const float* actions, int env, int sub) {
  const EnvCfg& c = b.cfg;
  if (sub > 0 && b.done[env] != 0) return;   // 계층형: 이 묶음 앞 걸음에서 판이 끝나 쉼
  if (gdec_idle(b, env, sub)) return;          // 그래프 모드: 결정이 앞 걸음에서 끝남
  Shared sh;
  Shared& g = b.scratch[env];   // 걸음을 넘어 남는 칸 (먼 지도 다시 그릴 곳 — GPU 와 같은 증분)
  mdp::obsgen::far_dirty_clear(g);
  EnvState e = b.st[env];
  e.collided = 0;
  if (sub == 0) gdec_begin(b, e, env);
  apply_action(b, e, env, actions + mdp::act::DIM * env);
  int new_wall = 0, new_floor = 0, d_known = 0, n_new = 0, n_occ = 0;
  sh.scanned = 0;
  const int tick0 = e.tick;
  for (int k = 0; k < c.substeps; ++k) {
    if (!e.collided) physics_step(b, e);
    e.tick += 1;
    if ((tick0 + k + 1) % c.lidar_every != 0) continue;
    scan_begin(b, e, sh);
    if (CpuTap* tap = g_tap) {   // 시험 탭: 이 장을 밖으로 (같은 값을 실물 노드 RobotEnv 에 넣어 비교 — tools/rl_node_check)
      std::vector<float> rr((size_t)sh.n_beams), it((size_t)sh.n_beams);
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j, &rr[(size_t)j], &it[(size_t)j]);
      tap->scan(env, e, sh, rr, it);
    } else {
      for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j);
    }
    int s = 0, f = 0, kn = 0, nn = 0, no = 0;
    for (int tid = 0; tid < kThreads; ++tid) {
      const ApplyCount a = scan_apply(b, e, env, sh, tid, kThreads, g.far_mask);
      s += a.wall;
      f += a.floor;
      kn += a.known;
      nn += a.fresh;
      no += a.occ;
    }
    new_wall += s;
    new_floor += f;
    d_known += kn;
    n_new += nn;
    n_occ += no;
    known_box(b, e, sh);
  }
  map_progress(e, n_occ);   // 진행 없음 시계 — finish_step 판 끝이 씀
  finish_step(b, e, env, new_wall, new_floor, d_known, n_new);
  mdp::obsgen::far_dirty_robot(b, e, g);
  b.reward[env] = e.reward;
  b.done[env] = e.done;
  if (sub >= 0) {   // 계층형: 결정 동안 합
    const float* T = b.reward_terms + (size_t)env * mdp::rew::kTerms;
    float* A = b.tacc + (size_t)env * mdp::rew::kTerms;
    if (sub == 0) {
      b.racc[env] = e.reward;
      for (int k = 0; k < mdp::rew::kTerms; ++k) A[k] = T[k];
    } else {
      b.racc[env] += e.reward;
      for (int k = 0; k < mdp::rew::kTerms; ++k) A[k] += T[k];
    }
  }
  gdec_step_ref(b, e, env, sub);   // 그래프 모드: 결정 끝 판정 (GPU env_finish_kernel 과 같은 자리)
  if (CpuTap* tap = g_tap) tap->step_end(env, e);   // 시험 탭: 걸음 끝 상태 (판 끝 리셋·관측 전)
  if (sh.scanned) ring_advance(e);
  if (sh.scanned)
    for (int tid = 0; tid < kThreads; ++tid) push_bins(b, e, env, sh, tid);
  const bool was_done = e.done != 0;
  if (was_done && sub >= 0) {   // 계층형: 리셋·관측은 묶음 끝에
    b.st[env] = e;
    return;
  }
  if (was_done) reset_env(b, e, env);
  write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, was_done),
            sub < 0 || (b.gdec ? b.gdec[env].reason != kGdRun : sub == c.decide_steps - 1));   // 깊이는 묶음의 마지막 걸음에만
  shape_obs(b, e, env, sh.reach);   // approach·front_cells 보상 (GPU env_reach_kernel 과 같은 때)
  b.st[env] = e;
}

}  // namespace

void set_cpu_threads(int n) { g_cpu_threads = std::max(1, n); }
void set_cpu_tap(CpuTap* t) { g_tap = t; }

void hier_plan_cpu(const EnvBatch& b, EnvState& e, int env, int slot) { plan_cpu(b, e, env, slot); }
// 깊이 카메라 장애물 층 갱신 — include/obs_ref.h camlayer_ref (실물 노드 limo_fire_rl 도 같은 함수)
void hier_camlayer_cpu(const EnvBatch& b, const EnvState& e, int env) { camlayer_ref(b, e, env); }

void env_step_cpu(const EnvBatch& b, const float* actions) {
  const EnvCfg& c = b.cfg;
  if (c.action_mode != kActPath) {
    for_envs(b.n_env, [&](int env) { step_env(b, actions, env, -1); });
    return;
  }
  // 계층형: GPU env_step_gpu 와 같은 차례 — 결정(경로 계획) → decide_steps 걸음 → 묶음 끝
  for_envs(b.n_env, [&](int env) {
    EnvState e = b.st[env];
    hier_camlayer_cpu(b, e, env);   // 깊이 카메라 장애물 층 (cam_on 일 때만) — 계획 앞, 지난 묶음 끝 깊이 장으로
    plan_cpu(b, e, env, slot_of(b, actions[(size_t)mdp::act::DIM * env]));
    b.st[env] = e;
    if (b.gdec) {   // 그래프 모드: 결정 하나 = 노드 하나 이동 — 닿거나 실패·시간 초과까지 (graph_env.h gdec_step)
      for (int s = 0; s < c.graph_max_steps && (s == 0 || b.gdec[env].reason == kGdRun); ++s) step_env(b, actions, env, s);
      // 묶음 끝 (GPU env_graph_kernel gate 1 과 같은 차례): 안 끝난 환경은 다음 결정 관측 + 결정 결과·전작 방식 판 끝, 걸음에서 끝난 환경은 결과만
      if (b.done[env] == 0) graph_ref(b, b.st[env], env, true);
      else graph_finish(b, b.st[env], env, -1, 0);
    } else {
      for (int s = 0; s < c.decide_steps; ++s) step_env(b, actions, env, s);
    }
    b.reward[env] = b.racc[env];
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = b.tacc[(size_t)env * mdp::rew::kTerms + k];
    if (b.done[env] == 0) return;
    Shared sh;
    Shared& g = b.scratch[env];
    e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, true));
    b.st[env] = e;
    graph_ref(b, b.st[env], env);
  });
}

void env_reset_cpu(const EnvBatch& b) {
  for_envs(b.n_env, [&](int env) {
    Shared sh;
    EnvState e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::FarDirty{1, 0.f, 0.f, 0.f, 0.f});
    shape_obs(b, e, env, sh.reach);   // 판 첫 관측: 값만 적음
    b.scratch[env].far_ok = 0;   // GPU 리셋 커널과 같게: 다음 걸음은 다시 전부
    b.st[env] = e;
    graph_ref(b, b.st[env], env);
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  });
}

void env_graph_cpu(const EnvBatch& b) {
  for_envs(b.n_env, [&](int env) { graph_ref(b, b.st[env], env); });
}

}  // namespace sim
