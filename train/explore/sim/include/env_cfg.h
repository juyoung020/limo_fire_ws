// 학습 환경 설정 읽기 (env.yaml + 그 안에서 가리키는 nav2_params.yaml · sensor_model.yaml) — CUDA 없는 호스트 헤더.
// 학습 환경(env_api.h)과 실물 노드(src/limo_fire/limo_fire_rl)가 같은 함수로 같은 EnvCfg 를 만든다
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "env.h"

namespace sim {

struct EnvHostCfg {
  EnvCfg cfg{};
  float band_lo = 0.f, band_hi = 0.f, spawn_clearance = 0.f;
  float brake_dist = 0.f;   // 규칙 추종기 앞 범퍼 정지 문턱 근거: V_FWD 에서 명령 0 → 멈출 때까지 (m)
};

// env.yaml + 그 안에서 가리키는 nav2_params.yaml
EnvHostCfg load_env_cfg(const std::string& env_yaml);

// 도구의 그래프 모드 덮어쓰기 인자 (설정 파일 action.graph 대신 — 음수 = 설정 파일 값):
//   --graph-node <m> 노드 간격 · --graph-term 0|1 (0 계층형 판 끝·보상, 1 전작 방식) · --graph-view 0|1|2 (방문 노드 유틸리티: 0 전작 · 1·2 우리 수정)
//   --graph-cover 0|1|2 (남은 프론티어 맡기기 — 우리 수정, mdp/graph.h 6b) · --graph-minutil <n> 유틸리티 하한
//   --graph-pass 0|1 (경로 끝 감속 안 함) · --graph-timeout <s> 결정 시간 상한 · --graph-maxdec <n> 판 결정 수 상한 · --graph-dec 0|1 (0 한 결정 = 한 노드, 1 고정 주기 path.period)
struct GraphArgs {
  float node = -1.f, timeout = -1.f, util_r = -1.f;
  int term = -1, view = -1, pass = -1, max_dec = -1, dec = -1, cover = -1, min_util = -1;
};
// a 가 그래프 인자면 값 v 를 읽고 true
inline bool graph_arg(GraphArgs& g, const std::string& a, const char* v) {
  if (a == "--graph-node") g.node = (float)std::atof(v);
  else if (a == "--graph-term") g.term = std::atoi(v);
  else if (a == "--graph-view") g.view = std::atoi(v);
  else if (a == "--graph-cover") g.cover = std::atoi(v);
  else if (a == "--graph-minutil") g.min_util = std::atoi(v);
  else if (a == "--graph-utilr") g.util_r = (float)std::atof(v);   // 유틸리티 범위 (m) — 시험용, 기본은 yaml utility_range_ratio
  else if (a == "--graph-pass") g.pass = std::atoi(v);
  else if (a == "--graph-timeout") g.timeout = (float)std::atof(v);
  else if (a == "--graph-maxdec") g.max_dec = std::atoi(v);
  else if (a == "--graph-dec") g.dec = std::atoi(v);   // 0 = 한 결정 한 노드 · 1 = 고정 주기
  else return false;
  return true;
}
inline void graph_apply(EnvCfg& c, const GraphArgs& g) {
  if (g.node > 0.f) {
    const float st = g.node / c.res;
    c.graph_step = (int)std::lround(st);
    if (std::fabs(st - std::round(st)) > 1e-3f || c.graph_step < 1 || c.graph_step * mdp::graph::GW < c.size)
      throw std::runtime_error("--graph-node: 지도 칸의 배수가 아니거나 노드 격자가 지도를 못 덮음");
  }
  if (g.term >= 0) c.graph_term = g.term;
  if (g.view >= 0) c.graph_view = g.view;
  if (g.cover >= 0) c.graph_cover = g.cover;
  if (g.min_util >= 0) c.graph_min_util = g.min_util;
  if (g.util_r > 0.f) c.graph_util_r = std::max(1, (int)(g.util_r / c.res));
  if (g.pass >= 0) c.graph_pass = g.pass;
  if (g.timeout > 0.f) c.graph_max_steps = std::max(1, (int)std::lround(g.timeout / ((float)c.substeps * c.dt)));
  if (g.max_dec >= 0) c.graph_max_dec = g.max_dec;
  if (g.dec >= 0) c.graph_fixed = g.dec;
}

}  // namespace sim
