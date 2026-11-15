// 아래층 체크포인트의 관측 설정 꼬리 (mdp/ckpt_cfg.h 꼴)에 적는 열쇠 — 정책 입력을 바꾸는 설정(환경변수 SIM_LOW_* 로 고르는 것)과 신경망 구조.
// 평가(low_policy_run · combine_run) · 이어 학습(train --resume) · 실물 노드(limo_fire_rl low_rl)가 환경변수 대신 체크포인트에서 읽는다 —
// 학습 판과 다른 관측으로 정책을 돌리는 실수를 막음. SIM_* 열쇠는 학습 때 실제 값 (env_cfg.cpp 가 읽은 뒤 EnvCfg 값 — 환경변수를 안 줬으면 기본값),
// 그 밖 열쇠(gru_h · d1d · aux_out): 구조 (참고·검사용)
#pragma once
#include <string>

#include "ckpt_cfg.h"   // mdp/
#include "env.h"

namespace sim {

// 정책 입력을 바꾸는 설정 (env_cfg.cpp 의 환경변수 이름 그대로) — 관측(lowrl_pack · lowrl_img · lowrl_wide · 추적) · 직전 행동(후진 없음 자름) · 보조 예측 재료
inline ckcfg::Kv low_obs_cfg_kv(const EnvCfg& c) {
  const auto f = [](float v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.9g", (double)v);
    return std::string(b);
  };
  const auto i = [](int v) { return std::to_string(v); };
  return {{"SIM_LOW_WIDE", i(c.lo_wide)},          {"SIM_LOW_GOALONLY", i(c.lo_goalonly)}, {"SIM_LOW_OBS_MOV", i(c.lo_mov)},
          {"SIM_LOW_MOV4", i(c.lo_mov4)},          {"SIM_LOW_CAM_EVERY", i(c.lo_cam_every)}, {"SIM_LOW_NOBACK", i(c.lg_noback)},
          {"SIM_LOW_AUX", i(c.lo_aux)},            {"SIM_LOW_BODY", i(c.lo_body)},            {"SIM_LOW_TRACK_GATE", f(c.tk_gate)},   {"SIM_LOW_TRACK_SZ", f(c.tk_sz)},
          {"SIM_LOW_TRACK_QA", f(c.tk_qa)},        {"SIM_LOW_TRACK_MISS", i(c.tk_miss)},   {"SIM_LOW_TRACK_CONF", i(c.tk_conf)},
          {"SIM_LOW_TRACK_DISP", f(c.tk_disp)},    {"SIM_LOW_TRACK_SAG", f(c.tk_sag)},     {"SIM_LOW_TRACK_W0", f(c.tk_w0)}};
}

// 학습기가 체크포인트에 붙이는 꼬리: 관측 설정 + 구조
inline ckcfg::Kv low_ckpt_kv(const EnvCfg& c, int gru_h, int d1d, int aux_out) {
  ckcfg::Kv kv = low_obs_cfg_kv(c);
  kv.push_back({"gru_h", std::to_string(gru_h)});
  kv.push_back({"d1d", std::to_string(d1d)});
  kv.push_back({"aux_out", std::to_string(aux_out)});
  return kv;
}

}  // namespace sim
