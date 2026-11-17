// 탐사 시뮬 관측·행동 크기 — sim(깊이 관측·라이다 칸)·실물 노드(RobotEnv)가 같이 쓰는 유일한 크기 정의.
// 2026-11-18: 강화학습 정책 관측(가까운·먼 지도 그림, 라이다 4 장 1D, 숫자 64 칸, 후보 특징 칸)과 행동의 끝 선언 칸을 지움 —
// 그만둔 RL(archive/rl) 몫이고 판단 모델 + A* 탐사는 안 쓴다. 남은 것: 깊이 관측(깊이 카메라 장애물 층·충돌 감시의 입력), 라이다 각도 칸(링 — 규칙 추종기),
// 계층형 후보 8 칸(시뮬 기준선 위층 fuel · slot0), 행동 3 칸.
#pragma once

namespace mdp::obs {

// ---- 깊이 (uint8, 0 = 구멍, 1~255 = min_z → obs_max_z — observations.h depth_u8). 배치는 [환경][행][열]
constexpr int DEPTH_C = 1, DEPTH_H = 40, DEPTH_W = 64;
constexpr int DEPTH_SIZE = DEPTH_C * DEPTH_H * DEPTH_W;

// ---- 라이다 각도 칸 (env.h 라이다 링 — 칸마다 가장 가까운 거리)
constexpr int LIDAR_BINS = 248;            // ±110°
constexpr float kLidarMin = -1.91986218f;  // −110° (rad) — 칸 k = [kLidarMin + k·kLidarBin, + kLidarBin)
constexpr float kLidarBin = 3.83972435f / 248.f;

}  // namespace mdp::obs

// ---- 계층형 행동 모드 (2026-11-08): 1 s 마다 프론티어 후보 K 개 중 하나를 고르고, 경로 계획 + 추종기가 몬다.
// 후보 칸은 EnvBatch::cand (시뮬 기준선 위층 slot0 · fuel — scripts/low_explore_ctl.h). 판단 모델 후보는 따로 (scripts/cand_view.h)
namespace mdp::hier {
constexpr int K = 8;                 // 후보 수 (ARiADNE 류의 이웃 후보 k — 잠정)
constexpr int kCandMinCells = 1;     // 후보 덩어리 크기 문턱 (칸) — 0.75 m 미만 조각도 후보 (정체 판의 36 % 가 그런 조각만 남아 실패, 2026-11-08 조사)
constexpr int kCandMinDist = 6;      // 후보 칸은 로봇 칸에서 BFS 6 칸(0.3 m) 이상 — 그 안은 라이다가 못 보는 뒤·발밑(fov 220°)이라 가도 안 풀림. 잠정
                                     // (2026-11-08 규칙 정책: 이 문턱 없이 늘 0 번을 고르면 거리 0 덩어리에 "도착"해 128 판 모두 정체)
constexpr float kUtilRadius = 2.0f;  // m — 잠정 (시선 검사 없는 근사)
// 후보 v2 (EnvCfg::cand_v2 = 1 일 때만, env.yaml action.path.candidates_v2 — mdp/README "계층형 후보 v2"). 끄면 위 규칙 그대로(비트 같음).
// 후보 크기 = 합친 덩어리 칸 수, 쓸모(util) = 예상 관측량 (아래). 후보 차례 = 가까운 순 K − kCand2Far 개 + 큰 순 kCand2Far 개
constexpr float kCand2Merge = 1.0f;  // m — 고른 후보 칸에서 직선 이만큼 안(그리고 길 거리 차 2 배 안)의 다른 덩어리는 그 후보에 합침. 잠정
                                     // (기존 후보는 결정마다 평균 2.6 개가 더 가까운 후보의 1 m 안 — 시험 맵 256 판, frontier_run cand_stat. FUEL 은 군집을 2 m 로 쪼개 그보다 잘게는 안 나눔)
constexpr int kCand2Far = 3;         // 가까운 순이 아니라 큰 순으로 채우는 자리 수 — 잠정 (TARE 가 지평 밖 구역을 전역 순회에 남기는 것, DSVP 의 전역 프론티어와 같은 뜻)
constexpr int kGainRays = 64;        // 예상 관측량 광선 수 (360°)
constexpr float kGainRange = 0.8f;   // 광선 길이 = 이 비 × 지도 사거리(map.max_range) — ARiADNE UTILITY_RANGE 0.8 × SENSOR_RANGE (parameter.py)
constexpr int kPathMax = 64;         // 웨이포인트 상한 (EnvState)
}  // namespace mdp::hier

namespace mdp::act {
constexpr int DIM = 3;     // 환경마다 행동 칸: 속도 행동 (v, vy, ω) — [−1, 1] 정규 단위(act_raw 면 물리 단위), 환경이 한도를 곱하고 자름.
                           // 계층형이면 a[0] = 고른 후보 번호 0 … hier::K − 1 (정수 값), 나머지 0. (2026-11-18 전에는 끝 선언 칸이 더 있어 4)
// 한도: 정규 단위 −1 → 뒤 한도, +1 → 앞 한도
constexpr float V_FWD = 1.0f;    // m/s, 데이터시트 최고 속도 1 m/s (실측 1.11 은 바퀴 한도로)
constexpr float V_BACK = 0.3f;   // m/s, 라이다가 뒤 140° 를 못 봐서 느리게
constexpr float VY_MAX = 1.0f;   // m/s, 잠정 📏 (데이터시트에 옆 속도 없음)
constexpr float W_MAX = 3.14f;   // rad/s, 실측 (env.yaml drive.max)
constexpr bool SAFETY = true;    // 규칙 추종기 앞 범퍼 정지 기본 켬 (계층형 path_step_c, 2026-11-14)
}  // namespace mdp::act
