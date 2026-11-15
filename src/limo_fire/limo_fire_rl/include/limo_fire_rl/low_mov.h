// 아래층 정책 관측의 "움직이는 것" 칸 — 실물 CPU 판 (SIM_LOW_OBS_MOV=1 정책의 kLoMov 6 칸, SIM_LOW_MOV4=1 정책의 kLoMov4 4 × 6 칸).
// 학습 시뮬과 같은 함수를 그대로 부른다: 라이다 덩어리 칼만 추적 sim::lowrl_trk_step · 움직이는 것 고르기 sim::lowrl_trk_movers ·
// 칸 채우기 sim::lowrl_mov_slots (train/explore/sim/include/low_rl.h). -ffp-contract=off 로 빌드하면 같은 입력에서 GPU 와 비트까지 같다
// (sim/tools/env_verify --mov-dump 로 GPU 를 녹화 → sim/tools/low_mov_replay 로 이 판을 다시 돌려 견줌).
// 입력은 RobotEnv 가 이미 들고 있는 라이다 칸 링(EnvBatch::lidar_ring)·링 칸마다 지도 자세(EnvState::lid_pose)·바퀴 앞 속도(drive.wv)·지도 자세.
// 부르는 차례 (시뮬 걸음과 같게): 걸음마다 새 라이다 장을 링에 넣기 "전에" track() (시뮬 lowrl_act 자리 — 직전 걸음 장까지 추적),
// 장을 넣은 뒤 관측 때 slots() (시뮬 lowrl_pack 자리). RobotEnv::set_mov_track(true) 가 이 차례로 부른다.
#pragma once

#include "env.h"   // train/explore/sim/include (low_rl.h 까지)

namespace limo_fire_rl {

class LowMov {
 public:
  LowMov() { reset(); }
  // 판 시작 (시뮬 lowrl_clear 의 추적 칸 — 다른 칸은 실물이 안 씀)
  void reset() { sim::lowrl_clear(R_); }
  // 추적 한 걸음: 링 최신 장(e.lid_head)이 지난번과 다르면 그 장을 덩어리로 나눠 잇는다 (같으면 그대로). b·e 는 환경 하나(env 0)
  void track(const sim::EnvBatch& b, const sim::EnvState& e) {
    sim::EnvBatch t = b;
    t.lowrl = &R_;
    sim::lowrl_trk_step(t, e, 0);
  }
  // 칸 채우기: mov[6] (kLoMov — 가장 가까운 하나), mov4[MOV_SLOTS × MOV_W] (kLoMov4 — nullptr 이면 안 씀). 움직이는 것이 없으면 0
  void slots(const sim::EnvBatch& b, const sim::EnvState& e, float* mov, float* mov4) const {
    for (int k = 0; k < 6; ++k) mov[k] = 0.f;
    if (mov4)
      for (int k = 0; k < mdp::low::MOV_SLOTS * mdp::low::MOV_W; ++k) mov4[k] = 0.f;
    sim::EnvBatch t = b;
    t.lowrl = const_cast<sim::LowRl*>(&R_);   // lowrl_trk_movers 는 읽기만
    sim::LrMover M[sim::kLrMovers];
    const int nm = sim::lowrl_trk_movers(t, 0, M);
    sim::lowrl_mov_slots(M, nm, e.x + e.ex, e.y + e.ey, e.th + e.eth, e.drive.wv, b.cfg.half_len, mov, mov4);
  }
  const sim::LowRl& state() const { return R_; }
  sim::LowRl& state_mut() { return R_; }   // 시험용 (녹화한 GPU 추적 상태를 넣고 다시 돌림)

 private:
  sim::LowRl R_{};
};

}  // namespace limo_fire_rl
