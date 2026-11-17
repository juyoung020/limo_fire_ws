// 관측 만들기 CPU 참조판 (호스트 전용, CUDA 없음): 프론티어 요약(닫힘)·계층형 후보·깊이 → 관측(write_obs), 깊이 카메라 장애물 층 갱신(camlayer_ref).
// 시뮬 CPU 참조판(src/env_ref.cpp)과 실물 노드(src/limo_fire/limo_fire_explore)가 이 함수들을 그대로 같이 부른다 — 실물 관측이 시뮬 관측과 갈라지지 않게
// 한 벌만 둔다. GPU 판(env_kernel.cu)은 같은 단계 함수로 같은 값을 낸다(tools/env_verify).
#pragma once
#include <cstdint>
#include <vector>

#include "env.h"
#include "frontier_cache.h"   // mdp/ — 걸음 사이 칸 판정 기억 (ReachKeep::cache)

namespace sim {

// 같은 걸음의 프론티어 찾기 결과를 남겨 두는 곳 (실물 노드 RobotEnv — 후보 뽑기 cand::build 가 같은 입력이면 다시 찾지 않고 씀, cand_view.h).
// 찾기 입력(지도 · 지나간 칸 판 · 깊이 층 · 시작 칸 · 걸음)을 같이 적어 두고, 쓰는 쪽이 자기 입력과 같을 때만 쓴다
struct ReachKeep {
  std::vector<uint8_t> flag;
  std::vector<int32_t> qa, qb, dist, label, pre;
  std::vector<mdp::Frontier> out;
  mdp::SearchResult r{};
  int rx = -1, ry = -1, step = -1;
  const int16_t* L = nullptr;
  const uint32_t* trail = nullptr;
  const uint8_t* cam = nullptr;
  bool valid = false;
  // 걸음 사이 칸 판정 기억 (mdp/frontier_cache.h — plan_ok · frontier_cell · 프론티어 칸 수). use_cache 면 reach_obs 가 걸음마다 바뀐 칸 둘레만 고쳐 씀
  bool use_cache = false;
  mdp::FrontierCache cache;
};

// 도달 가능한 프론티어 요약 (mdp/frontier.h reach_summary = 기준선 frontier_search 그대로). GPU 는 env_kernel.cu reach_block — 같은 값.
// 계층형이면 같은 찾기에서 후보(mdp::cand_pick)·주변 프론티어 칸 수(mdp::cand_util)도 → 후보 칸 (GPU env_reach_kernel 과 같은 값).
// cand = false: 계층형 후보를 안 셈 (후보 칸은 빈 칸 — 그 후보를 안 쓰는 실물 위층). keep: 찾기 버퍼·결과를 거기에 남김 (라벨까지)
inline void reach_obs(const EnvBatch& b, const EnvState& e, int env, Shared& sh, bool cand = true, ReachKeep* keep = nullptr) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  mdp::reach_clear(&sh.reach);
  mdp::CandSummary cs;
  mdp::cand_clear(&cs);
  if (keep) keep->valid = false;
  if (e.kb_x1 >= 0) {   // 아는 칸이 있을 때
    const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
    const mdp::MapView mv{L, c.size, m.gw, m.gh};
    thread_local ReachKeep own;
    ReachKeep& K = keep ? *keep : own;
    std::vector<uint8_t>& flag = K.flag;
    std::vector<int32_t>&qa = K.qa, &qb = K.qb, &dist = K.dist, &label = K.label, &pre = K.pre;
    std::vector<mdp::Frontier>& out = K.out;
    const size_t n = (size_t)m.gw * m.gh;
    if (flag.size() < n) { flag.resize(n); qa.resize(n); qb.resize(n); dist.resize(n); label.resize(n); pre.resize((size_t)(m.gw + 2) * m.gh); }
    if (out.size() < (size_t)mdp::kReachCap + 1) out.resize(mdp::kReachCap + 1);
    int rx, ry;
    cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
    const bool hier = b.cand && cand;
    mdp::SearchBuf sb{flag.data(), qa.data(), qb.data(), dist.data(), hier || keep ? label.data() : nullptr};
    {   // 닫힘 규칙 (GPU reach_block 과 같은 값): 지나간 칸 씨앗·본 칸·덩어리 문턱·자격 반경
      const int ws = fmap_row(c), nf = c.size * ws;
      sb.vis = b.trail + (size_t)env * 2 * nf;
      sb.look = sb.vis + nf;
      sb.tws = ws;
      sb.close_min = c.close_min;
      sb.rbc = mdp::robot_cells(c.close_radius, c.res);
      // 깊이 카메라 층 (계층형 cam_on). 합친 탐사 low_explore 2 는 안 씀 — 닫힘은 로봇 지도(라이다)로만 (mdp/frontier.h 머리글 "합친 탐사의 닫힘", 켜야 씀)
      sb.cam = b.cvox && c.low_explore != 2 ? b.cvox + (size_t)env * c.size * c.size : nullptr;
      sb.pre = pre.data();   // 원판 검사 지름길 (값 같음 — GPU reach_block 과 비트 비교는 env_verify)
    }
    int n_tot = 0;   // 지도 전체 프론티어 칸 (모두 건드린 범위 안)
    if (keep && keep->use_cache) {   // 걸음 사이 기억: 바뀐 칸 둘레만 고침 (값 같음 — frontier_cache.h)
      mdp::cache_update(keep->cache, mv, sb.cam, sb.rbc.r < 0 ? mdp::obsgen::reach_robot(c) : sb.rbc, e.kb_x0, e.kb_y0, e.kb_x1, e.kb_y1);
      sb.pc = keep->cache.pc.data();
      sb.fc = keep->cache.fc.data();
      n_tot = (int)keep->cache.n_front;
    } else {
      for (int y = e.kb_y0; y <= e.kb_y1; ++y)
        for (int x = e.kb_x0; x <= e.kb_x1; ++x) n_tot += mdp::frontier_cell(mv, x, y) ? 1 : 0;
    }
    // reach_summary 와 같음: 문턱 1 로 모든 덩어리를 out 에, 문턱은 고를 때
    const mdp::SearchResult r = mdp::frontier_search(mv, rx, ry, mdp::obsgen::reach_robot(c), 1, sb, out.data(), (int)out.size());
    const int n_out = r.n_found < (int)out.size() ? r.n_found : (int)out.size();
    mdp::reach_pick(r, out.data(), n_out, mdp::obsgen::reach_min_cells(c), n_tot, &sh.reach);
    if (keep) {
      keep->r = r; keep->rx = rx; keep->ry = ry; keep->step = e.step;
      keep->L = L; keep->trail = sb.vis; keep->cam = sb.cam;
      keep->valid = true;
    }
    if (hier) {
      thread_local std::vector<uint64_t> key2;
      if (key2.size() < out.size()) key2.resize(out.size());
      mdp::cand_keys(mv, mdp::obsgen::reach_robot(c), r, sb, out.data(), n_out, key2.data());
      if (c.cand_v2) {
        mdp::cand_pick2(out.data(), key2.data(), n_out, n_tot, e.goal_cell, flag.data(), label.data(), m.gw, mdp::cand2_merge_cells(c.res), &cs);
        const int R = mdp::cand_gain_cells(c.map_max_range, c.res);
        for (int k = 0; k < cs.n; ++k)
          for (int a = 0; a < mdp::hier::kGainRays; ++a) cs.c[k].util += mdp::cand_gain_ray(mv, cs.c[k].cell % m.gw, cs.c[k].cell / m.gw, a, R);
      } else {
        mdp::cand_pick(out.data(), key2.data(), n_out, n_tot, e.goal_cell, flag.data(), label.data(), &cs);
        const int R = mdp::cand_util_radius(c.res);
        for (int k = 0; k < cs.n; ++k) cs.c[k].util = mdp::cand_util(mv, cs.c[k].cell % m.gw, cs.c[k].cell / m.gw, R);
      }
    }
  }
  if (b.cand) mdp::obsgen::obs_cand(b, env, cs);
}

// depth = false: 깊이 칸을 안 그림 (계층형 묶음 가운데 걸음 — GPU 도 묶음 끝에만 그림, 깊이는 상태만의 함수라 끝 값이 같다. 실물은 깊이를 영상에서 — RobotEnv::set_depth).
// cand · keep: reach_obs
inline void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, bool depth = true, bool cand = true, ReachKeep* keep = nullptr) {
  trail_mark(b, e, env, 0, 1);   // 닫힘 규칙의 지나간 길 (GPU env_reach_kernel 처음과 같은 때)
  reach_obs(b, e, env, sh, cand, keep);
  e.map_closed = sh.reach.closed;   // 다음 걸음 finish_step 성공 판정 (GPU env_reach_kernel 과 같음)
  if (depth)
    for (int i = 0; i < mdp::obs::DEPTH_SIZE; ++i) mdp::obsgen::obs_depth(b, e, env, i);
}

// 아래 참조판 함수가 HD 템플릿(cam_pixel)에 넘기는 함수 객체 — 람다 대신 DEV operator() (2026-11-16): .cu 번역 단위(env_verify 등)가 이 머리를
// 포함하면 호스트 람다를 HD 템플릿에 넘긴 것이 nvcc 20013 진단이 되는데, 저장소는 그 진단을 오류로 한다 (cmake/cuda_strict.cmake — 장치 쪽 값이 틀리게 나온 일)
struct CamClear {
  uint8_t* V;
  DEV void operator()(int ci, uint8_t bit) const { V[ci] &= (uint8_t)~bit; }
};
struct CamMark {
  uint8_t* V;
  DEV void operator()(int ci, uint8_t bit) const { V[ci] |= bit; }
};

// 깊이 카메라 장애물 층 갱신 (GPU env_camlayer_kernel 과 같은 차례: 모든 픽셀 지우기 → 모든 픽셀 표시 → 몸통 밑 지우기 — 끄기·켜기라 픽셀 차례와 무관)
inline void camlayer_ref(const EnvBatch& b, const EnvState& e, int env) {
  if (!b.cvox) return;
  uint8_t* V = b.cvox + (size_t)env * b.cfg.size * b.cfg.size;
  const int np = cam_npix(b.cfg);   // 층 격자 (cam_lw 0 이면 관측 깊이 64 × 40)
  for (int i = 0; i < np; ++i) cam_pixel(b, e, env, i, 0, CamClear{V});
  for (int i = 0; i < np; ++i) cam_pixel(b, e, env, i, 1, CamMark{V});
  for (int col = 0; col < mdp::obs::DEPTH_W; ++col) cam_column(b, e, env, col);
  cam_compact(b, env);
  const int nfp = cam_fp_cells(b.cfg);
  for (int j = 0; j < nfp; ++j) {
    const int ci = cam_fp_cell(b, e, j);
    if (ci >= 0) V[ci] = 0;
  }
}

}  // namespace sim
