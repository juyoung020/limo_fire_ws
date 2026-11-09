// 환경 한 걸음 CPU 참조판: GPU 커널(src/env_kernel.cu)의 블록 하나를 스레드 순서대로 흉내 낸다. 단계 순서는 커널과 한 줄씩 같다
#include <algorithm>
#include <thread>
#include <vector>

#include "env_api.h"

namespace sim {
namespace {

int g_cpu_threads = 1;

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

// 도달 가능한 프론티어 요약 (mdp/frontier.h reach_summary = 기준선 frontier_search 그대로). GPU 는 env_kernel.cu reach_block — 같은 값.
// 계층형이면 같은 찾기에서 후보(mdp::cand_pick)·주변 프론티어 칸 수(mdp::cand_util)도 → 관측 후보 칸 (GPU env_reach_kernel 과 같은 값)
void reach_obs(const EnvBatch& b, const EnvState& e, int env, Shared& sh) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  mdp::reach_clear(&sh.reach);
  mdp::CandSummary cs;
  mdp::cand_clear(&cs);
  if (e.kb_x1 >= 0) {   // 아는 칸이 있을 때
    const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
    const mdp::MapView mv{L, c.size, m.gw, m.gh};
    int n_tot = 0;   // 지도 전체 프론티어 칸 (모두 건드린 범위 안)
    for (int y = e.kb_y0; y <= e.kb_y1; ++y)
      for (int x = e.kb_x0; x <= e.kb_x1; ++x) n_tot += mdp::frontier_cell(mv, x, y) ? 1 : 0;
    thread_local std::vector<uint8_t> flag;
    thread_local std::vector<int32_t> qa, qb, dist, label;
    thread_local std::vector<mdp::Frontier> out;
    const size_t n = (size_t)m.gw * m.gh;
    if (flag.size() < n) { flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); label.resize(n); }
    if (out.size() < (size_t)mdp::kReachCap + 1) out.resize(mdp::kReachCap + 1);
    int rx, ry;
    cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
    const mdp::SearchBuf sb{flag.data(), qa.data(), qb.data(), dist.data(), b.cand ? label.data() : nullptr};
    // reach_summary 와 같음: 문턱 1 로 모든 덩어리를 out 에, 문턱은 고를 때
    const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, out.data(), (int)out.size());
    const int n_out = r.n_found < (int)out.size() ? r.n_found : (int)out.size();
    mdp::reach_pick(r, out.data(), n_out, mdp::obsgen::reach_min_cells(c), n_tot, &sh.reach);
    if (b.cand) {
      thread_local std::vector<uint64_t> key2;
      if (key2.size() < out.size()) key2.resize(out.size());
      mdp::cand_keys(r, sb, n_out, key2.data());
      mdp::cand_pick(out.data(), key2.data(), n_out, n_tot, e.goal_cell, flag.data(), label.data(), &cs);
      const int R = mdp::cand_util_radius(c.res);
      for (int k = 0; k < cs.n; ++k) cs.c[k].util = mdp::cand_util(mv, cs.c[k].cell % m.gw, cs.c[k].cell / m.gw, R);
    }
  }
  if (b.cand) mdp::obsgen::obs_cand(b, e, env, cs);
}

// depth = false: 깊이 칸을 안 그림 (계층형 묶음 가운데 걸음 — GPU 도 묶음 끝에만 그림, 깊이는 상태만의 함수라 끝 값이 같다)
void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, const mdp::obsgen::FarDirty& fd, bool depth = true) {
  reach_obs(b, e, env, sh);
  for (int tid = 0; tid < kThreads; ++tid) mdp::obsgen::obs_cells(b, e, env, sh, tid, fd);
  if (depth)
    for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) mdp::obsgen::obs_depth(b, e, env, i);
  mdp::obsgen::obs_finish(b, e, env, sh);
}

int slot_of(float a) { return (a >= 0.f && a < (float)mdp::hier::K) ? (int)a : -1; }

// 계층형 결정 (GPU env_plan_kernel 과 같은 규칙·같은 경로): 엄격(지도 막힘 ∪ 약한 막힘) → 못 닿으면 느슨(지도 막힘만).
// 거리 배열 위 Dial 최단 경로(무게 hier_path.h) → 거리 = 내 거리 − 내 무게 인 첫 nb4 이웃으로 뒤로 짚기
void plan_cpu(const EnvBatch& b, EnvState& e, int env, int k) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const mdp::MapView mv{L, c.size, m.gw, m.gh};
  const mdp::Robot rb = mdp::obsgen::reach_robot(c), rm = path_rb_mid(c), ro = path_rb_out(c);
  e.p_slot = k;
  const int tgt = k >= 0 ? b.cand[(size_t)env * mdp::hier::K + k] : -1;
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
      uint8_t v = mdp::cell_class(l) == mdp::kOcc ? 3 : 0;   // bit0 occ · bit1 lt
      if (!v && l != mdp::kUnknownL && l > 0) {
        const float dx = (float)(x - rcx) * c.res, dy = (float)(y - rcy) * c.res;
        if (dx * dx + dy * dy <= rr) v = 2;
      }
      lt[(size_t)y * m.gw + x] = v;
    }
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
          if (v & 1) {
            int32_t& qo = d2o[(size_t)w * m.gw + u];
            qo = std::min(qo, d2);
          }
        }
    }
  const auto inbox = [&](int x, int y) { return x >= x0 && y >= y0 && x <= x1 && y <= y1; };
  const auto known = [&](int x, int y) { return mdp::cls_at(mv, x, y) != mdp::kUnk; };
  // mode 1 엄격 · 2 느슨: 설 수 있음, 무게 (GPU: SST·ST 판과 stand_word(LT·OCC) 무게 경계)
  const auto pass = [&](int mode, int x, int y) -> bool {
    if (!inbox(x, y) || !known(x, y)) return false;
    const size_t i = (size_t)y * m.gw + x;
    return mode == 1 ? (!lt[i] && d2l[i] > rb.r2) : (!(lt[i] & 1) && d2o[i] > rb.r2);
  };
  const auto weight = [&](int mode, int x, int y) -> int {
    if (!pass(mode, x, y)) return 0;
    const size_t i = (size_t)y * m.gw + x;
    const int32_t d = mode == 1 ? d2l[i] : d2o[i];
    return path_weight(true, d > rm.r2, d > ro.r2);
  };
  // 같은 후보 · 경로 그대로?
  const int pm = e.p_mode == 1 ? 1 : 2;
  if (((b.cand_cur[env] >> k) & 1) && e.goal_cell >= 0 && e.p_n > 0 && !e.p_trunc && path_clear(e, [&](int x, int y) { return pass(pm, x, y); }))
    return;
  e.goal_cell = tgt;
  path_set_none(e);
  e.p_plans += 1;
  // 한 번 계획 (mode): 시작 = 로봇 칸에서 FIFO 4 이웃 BFS 로 처음 만나는 설 수 있는 칸 (frontier_search nearestCell 과 같은 차례), Dial
  int start = -1;
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
      if (bk[0].empty() && bk[1].empty() && bk[2].empty() && bk[3].empty()) return false;
      std::vector<int32_t> cur;
      cur.swap(bk[t & 3]);
      for (int v : cur) {
        if (dist[v] >= 0 || tent[v] != t) continue;
        dist[v] = t;
        if (v == tgt) return true;
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
    }
  };
  int mode = 0;
  if (pass(1, tgt % m.gw, tgt / m.gw) && run(1)) mode = 1;
  else if (run(2)) mode = 2;
  if (!mode) return;   // 설 수 있는 칸이 없거나 못 닿음: 목표만 두고 섬
  const auto wfn = [&](int x, int y) { return weight(mode, x, y); };
  thread_local std::vector<int32_t> list;
  list.clear();
  int v = tgt;
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
  if (mode == 2) e.p_fb += 1;
}

// 환경 하나 행동 하나. sub < 0: 기본, sub ≥ 0: 계층형 묶음 sub 번째 걸음 (GPU env_finish_kernel 의 sub 와 같은 규칙)
void step_env(const EnvBatch& b, const float* actions, int env, int sub) {
  const EnvCfg& c = b.cfg;
  if (sub > 0 && b.done[env] != 0) return;   // 계층형: 이 묶음 앞 걸음에서 판이 끝나 쉼
  Shared sh;
  Shared& g = b.scratch[env];   // 걸음을 넘어 남는 칸 (먼 지도 다시 그릴 곳 — GPU 와 같은 증분)
  mdp::obsgen::far_dirty_clear(g);
  EnvState e = b.st[env];
  e.collided = 0;
  apply_action(b, e, env, actions + mdp::act::DIM * env);
  int new_wall = 0, new_floor = 0, d_known = 0, n_new = 0, n_occ = 0;
  sh.scanned = 0;
  const int tick0 = e.tick;
  for (int k = 0; k < c.substeps; ++k) {
    if (!e.collided) physics_step(b, e);
    e.tick += 1;
    if ((tick0 + k + 1) % c.lidar_every != 0) continue;
    scan_begin(b, e, sh);
    for (int j = 0; j < sh.n_beams; ++j) scan_beam(b, e, env, sh, j);
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
  if (sh.scanned) e.lid_head = (e.lid_head + 1) % kLidarRing;
  if (sh.scanned)
    for (int tid = 0; tid < kThreads; ++tid) push_bins(b, e, env, sh, tid);
  const bool was_done = e.done != 0;
  if (was_done && sub >= 0) {   // 계층형: 리셋·관측은 묶음 끝에
    b.st[env] = e;
    return;
  }
  if (was_done) reset_env(b, e, env);
  write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, was_done), sub < 0 || sub == c.decide_steps - 1);
  b.st[env] = e;
}

}  // namespace

void set_cpu_threads(int n) { g_cpu_threads = std::max(1, n); }

void hier_plan_cpu(const EnvBatch& b, EnvState& e, int env, int slot) { plan_cpu(b, e, env, slot); }

void env_step_cpu(const EnvBatch& b, const float* actions) {
  const EnvCfg& c = b.cfg;
  if (c.action_mode != kActPath) {
    for_envs(b.n_env, [&](int env) { step_env(b, actions, env, -1); });
    return;
  }
  // 계층형: GPU env_step_gpu 와 같은 차례 — 결정(경로 계획) → decide_steps 걸음 → 묶음 끝
  for_envs(b.n_env, [&](int env) {
    EnvState e = b.st[env];
    plan_cpu(b, e, env, slot_of(actions[(size_t)mdp::act::DIM * env]));
    b.st[env] = e;
    for (int s = 0; s < c.decide_steps; ++s) step_env(b, actions, env, s);
    b.reward[env] = b.racc[env];
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = b.tacc[(size_t)env * mdp::rew::kTerms + k];
    if (b.done[env] == 0) return;
    Shared sh;
    Shared& g = b.scratch[env];
    e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::far_dirty_take(b, e, g, true));
    b.st[env] = e;
  });
}

void env_reset_cpu(const EnvBatch& b) {
  for_envs(b.n_env, [&](int env) {
    Shared sh;
    EnvState e = b.st[env];
    reset_env(b, e, env);
    write_obs(b, e, env, sh, mdp::obsgen::FarDirty{1, 0.f, 0.f, 0.f, 0.f});
    b.scratch[env].far_ok = 0;   // GPU 리셋 커널과 같게: 다음 걸음은 다시 전부
    b.st[env] = e;
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  });
}

}  // namespace sim
