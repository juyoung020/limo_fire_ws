// 목적지만(SIM_LOW_GOALONLY) 아래층 학습의 목표 뽑기 — 경로를 짜지 않고 "로봇 지도에서 로봇과 같은 덩어리인가" 만 (2026-11-15, 순수 RL).
// 갈 수 있는 칸 판 P = 아는 칸 상자(kb — 경로 계획과 같은 낱말 범위) 안에서 계획 반경(path_rb_plan)으로 설 수 있고(mdp::standable) 깊이 카메라 층에 안 막힌 칸.
// 시작 칸 = 로봇 지도 자세 칸에서 맨해튼 고리 차례로 가장 가까운 P 칸 (env_kernel.cu stand_start 와 같은 규칙). 덩어리 = 시작 칸에서 4 이웃으로 P 를 따라 닿는 칸 전부.
// 목표 탐침(low_rl.h lowrl_goal_probe)은 이 덩어리 안 칸만 받는다 → 늘 갈 수 있는 목표, 경로 없음 포기 없음. goal_cell = 뽑은 칸 (경로는 안 만듦 — p_n 0).
// 덩어리는 정답이 하나(알고리즘과 무관)라 CPU(아래 goal_reach_cpu, 큐 BFS)와 GPU(env_kernel.cu goal_reach_gpu, 줄·열 비트 채우기 반복)가 같은 칸 집합을 낸다.
#pragma once
// (low_rl.h 머리에서 include — env.h 의 namespace sim 안)

// 아는 칸 상자의 줄·낱말 범위 (env_plan_kernel 과 같음). kb_x1 < 0 이면 상자 없음 → false
DEV bool goal_box(const EnvState& e, int* by0, int* by1, int* bw0, int* bw1) {
  if (e.kb_x1 < 0) return false;
  *by0 = e.kb_y0;
  *by1 = e.kb_y1;
  *bw0 = e.kb_x0 >> 5;
  *bw1 = e.kb_x1 >> 5;
  return true;
}
// 시작 칸 (stand_start 와 같은 맨해튼 고리 차례): bit(x, y) = 그 칸이 P 인가 (상자 밖이면 false). QA·QB 는 고리 목록 작업 칸
template <class F>
DEV int goal_start(int gw, int gh, int rx, int ry, F bit, int32_t* QA, int32_t* QB, int cap) {
  if (bit(rx, ry)) return ry * gw + rx;
  int32_t* cur = QA;
  int32_t* nxt = QB;
  int nc = 1, d = 0;
  cur[0] = ry * gw + rx;
  while (nc > 0) {
    for (int i = 0; i < nc; ++i)
      if (bit(cur[i] % gw, cur[i] / gw)) return cur[i];
    int nn = 0;
    for (int i = 0; i < nc; ++i) {
      const int x = cur[i] % gw, y = cur[i] / gw;
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        mdp::nb4(k, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
        const int ad = (u > rx ? u - rx : rx - u) + (v > ry ? v - ry : ry - v);
        if (ad != d + 1) continue;
        const int j = v * gw + u;
        bool seen = false;
        for (int q = nn - 1; q >= 0 && !seen; --q) seen = nxt[q] == j;
        if (!seen && nn < cap) nxt[nn++] = j;
      }
    }
    int32_t* t = cur;
    cur = nxt;
    nxt = t;
    nc = nn;
    ++d;
  }
  return -1;
}
// 목표를 줄 때 (경로 계획 자리 — env_plan_kernel · env_ref.cpp low_prepare_env): 계획 앞 손질(path_recovery — 아래층 단독 과제의 결정 시작: 통과 대기 hold·결정 첫 걸음
// d_step·막힘 셈을 지움, 정책 환경은 슬롯 0)을 경로 계획과 같이 부르고, goal_cell = 그 슬롯의 목표 칸, 경로 없음, 계획 수 +1 (기록)
DEV void goal_only_issue(const EnvBatch& b, EnvState& e, int env) {
  int force = 0;
  const int k = path_recovery(b, e, env, 0, &force);
  e.p_slot = k;
  e.goal_cell = k >= 0 ? act_cells(b, env)[k] : -1;
  path_set_none(e);
  e.p_plans += 1;
  if (e.goal_cell >= 0 && b.lowstat && b.lowcmd[env].follow == 2) b.lowstat[env].goals += 1;   // 기록: 받은 목표 수 (경로 판은 첫 경로가 짜일 때 셈 — lowrl_pack)
}

