// 학습 환경 설정 읽기 (env.yaml + 그 안에서 가리키는 nav2_params.yaml · sensor_model.yaml) — CUDA 없는 호스트 헤더.
// 학습 환경(env_api.h)과 실물 노드(src/limo_fire/limo_fire_rl)가 같은 함수로 같은 EnvCfg 를 만든다
#pragma once
#include <string>

#include "env.h"

namespace sim {

struct EnvHostCfg {
  EnvCfg cfg{};
  float band_lo = 0.f, band_hi = 0.f, spawn_clearance = 0.f;
  float brake_dist = 0.f;   // 안전층 문턱 근거: V_FWD 에서 명령 0 → 멈출 때까지 (m)
};

// env.yaml + 그 안에서 가리키는 nav2_params.yaml
EnvHostCfg load_env_cfg(const std::string& env_yaml);

}  // namespace sim
