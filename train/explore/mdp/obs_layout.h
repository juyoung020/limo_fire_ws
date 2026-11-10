// 탐사 RL 관측·행동 배치 — sim(mdp 관측 만들기)·network·ppo·실물 노드가 같이 쓰는 유일한 크기 정의.
// 스펙은 mdp/README.md "관측"·"행동". 칸을 미리 잡아 두고 아직 실측 못 맞춘 값(📏)은 0 으로 둔다 —
// 켜도 입력 크기가 안 바뀌어 체크포인트에서 이어 학습한다. 칸을 옮기거나 지우면 체크포인트가 깨지므로
// 새 값은 예약 칸(RESERVED)에 넣는다.
#pragma once

namespace mdp::obs {

// ---- 2D (uint8, 0~255, 신경망에서 /255). 배치는 [환경][채널][행][열]
constexpr int NEAR_C = 6, NEAR_H = 64, NEAR_W = 64;   // 로봇 중심·로봇 방향 10 cm: 확률, 지나간 곳, 물체 4 (문·비상구·소화기·사람)
constexpr int FAR_C = 7, FAR_H = 32, FAR_W = 32;      // 시작점 기준 25.6 m 창 0.8 m: 확률, 지나간 곳, 현재 위치, 물체 4
constexpr int DEPTH_C = 1, DEPTH_H = 40, DEPTH_W = 64; // 깊이 거리 (0 = 구멍)
constexpr int NEAR_SIZE = NEAR_C * NEAR_H * NEAR_W;
constexpr int FAR_SIZE = FAR_C * FAR_H * FAR_W;
constexpr int DEPTH_SIZE = DEPTH_C * DEPTH_H * DEPTH_W;
constexpr int IMG_SIZE = NEAR_SIZE + FAR_SIZE + DEPTH_SIZE;  // 환경 하나의 uint8 관측 바이트 수
// 프론티어 채널 (2026-11-08, Jonnarth et al. ICML 2024 arXiv 2306.16978 꼴 — 지도 그림에 프론티어를 채널로): 가까운 지도·먼 지도 모두 마지막 물체 예약
// 칸(📏 사람 자리)을 쓴다 — 입력 크기가 안 바뀌어 체크포인트·합성곱 커널이 그대로. 값 255 = 그 칸(먼 지도는 칸 안 점 64 개 중 하나라도)이 로봇 지도의
// 프론티어 칸(mdp/frontier.h frontier_cell: 갈 수 있음 + 4 이웃에 모름), 0 = 아님. 사람 물체 채널이 필요해지면 새 칸을 늘린다
constexpr int NEAR_FRONTIER_CH = NEAR_C - 1, FAR_FRONTIER_CH = FAR_C - 1;

// ---- 1D (float, 정규화된 값). 정책 입력 순서 그대로
constexpr int LIDAR_BINS = 248;            // ±110°, 칸마다 가장 가까운 거리
constexpr float kLidarMin = -1.91986218f;  // −110° (rad) — 칸 k = [kLidarMin + k·kLidarBin, + kLidarBin)
constexpr float kLidarBin = 3.83972435f / 248.f;
constexpr int LIDAR_FRAMES = 4;            // 지금 + 직전 3 장 (0.3 s)
constexpr int LIDAR = LIDAR_BINS * LIDAR_FRAMES;
constexpr int INTENSITY = LIDAR_BINS;      // 지금 장 세기
// 숫자 (오프셋은 VEC 안 위치)
constexpr int POSE = 0;          // x, y, sin θ, cos θ (지도 자세, 시작점 기준) — 4
constexpr int MAP_VEL = 4;       // vx, vy, vz (지도 자세 미분, 로봇 기준) — 3
constexpr int WHEEL_V = 7;       // 1
constexpr int GYRO_Z = 8;        // 1
constexpr int SLIP = 9;          // 1
constexpr int EKF = 10;          // vx, ω — 2
constexpr int ACCEL = 12;        // x, y, z — 3
constexpr int IMU_TILT = 15;     // roll, pitch, 자이로 x, y — 4
constexpr int WHEELS4 = 19;      // 바퀴 4 개 — 4
constexpr int LAST_ACT = 23;     // (v, vy, ω, 선언) × 최근 4 걸음 — 16
constexpr int ELAPSED = 39;      // 1
constexpr int KNOWN_AREA = 40;   // 1
// 프론티어 덩어리 (mdp/frontier.h reach_summary — 기준선 frontier_search 와 같은 정의: 설 수 있는 칸으로 번진 곳 안, 크기 ≥ 0.75 m):
//   시작 칸에서 BFS 거리 짧은 순 1·2·3 번째 덩어리마다 BFS 거리, 그 칸 방향 sin, cos, 덩어리 크기 — 4 칸씩 (없으면 4 칸 모두 0)
constexpr int FRONTIER = 41;     // 1 번째 덩어리 — 4 (2026-11-05 전: 도달·크기 무시한 가장 가까운 프론티어 칸 직선거리·방향·칸 수)
constexpr int PROGRESS = 45;     // 최근 새로 안 칸 수, 진행 없음 시계(새 막힘 칸 ≥ kIdleOccMin 인 마지막 행동 뒤 시간 = 판 끝 시계) — 2 (2026-11-05 전 +1: 마지막 새 칸 뒤 시간)
constexpr int HOME = 47;         // 시작점까지 거리, sin, cos — 3
constexpr int RESERVED50 = 50;   // 0 칸 (2026-11-08 전 연기 모드 칸 — 입력 크기·체크포인트 그대로 두려고 자리만 남김) — 1
constexpr int FRONTIER2 = 51;    // 2 번째 덩어리 — 4
constexpr int FRONTIER3 = 55;    // 3 번째 덩어리 — 4
constexpr int FRONT_N = 59;      // 문턱 넘는 덩어리 수 — 1
constexpr int CLOSED = 60;       // 지도 닫힘 1 / 아님 0 (성공 판정과 같은 닫힘 규칙 — frontier.h, mdp/README "닫힘") — 1
constexpr int FRONT_CELLS = 61;  // 번진 곳 안 프론티어 칸 수 — 1
constexpr int OCC_GAIN = 62;     // 최근 2 s 새 막힘 칸 수 (지도 기준 벽 덮음 증가 대체값) — 1
constexpr int RESERVED = 63;     // 나중 값(레이더 사람 등)용 0 칸
constexpr int VEC = 64;
constexpr int POLICY_1D = LIDAR + INTENSITY + VEC;   // 1304

// ---- 가치에만 (참값, 시뮬 전용) — 정책 1D 뒤에 붙인다
constexpr int PRIV_TRUE_POSE = 0;   // x, y, sin, cos — 4
constexpr int PRIV_POSE_ERR = 4;    // ex, ey, eth — 3
constexpr int PRIV_COVERAGE = 7;    // 진짜 벽 덮음 — 1
constexpr int PRIV_REMAIN = 8;      // 남은 벽 면 칸 (정규화) — 1
constexpr int PRIV_RESERVED = 9;    // 사람 참 위치 등 (9: 가장 가까운 사람 간격 — observations.h)
constexpr int PRIV_IDLE = 10;       // 정답 벽 진척 시계: 마지막 새 벽 면 칸 뒤 시간 / kIdleTime (env.h since_wall, 정답 벽이라 정책엔 안 줌, 판 끝엔 안 씀) — 1
constexpr int PRIV = 32;
constexpr int CRITIC_1D = POLICY_1D + PRIV;

}  // namespace mdp::obs

// ---- 계층형 행동 모드 (2026-11-08, mdp/README "계층형"): 정책이 1 s 마다 프론티어 후보 K 개 중 하나를 고르고, 경로 계획 + 추종기가 몬다.
// 이 모드에서만 1D 관측 뒤에 후보 칸 CAND_1D 를 붙인다(정책 1D = hier::POLICY_1D). 기본(속도 행동) 모드의 배치·크기는 그대로.
namespace mdp::hier {
constexpr int K = 8;                 // 후보 수 (ARiADNE 류의 이웃 후보 k — 잠정)
constexpr int FEAT = 7;              // 후보마다 칸 수
enum : int {
  F_DIST = 0,   // 로봇 칸 → 덩어리 가장 가까운 칸 BFS 길 거리(설 수 있는 칸 4 이웃) × 5 cm / kFrontNorm 25.6 m (자름 2)
  F_SIN = 1,    // 그 칸 방향 (로봇 기준 sin, cos)
  F_COS = 2,
  F_SIZE = 3,   // 덩어리 크기 칸 수 × 5 cm / 10 m (자름 1)
  F_UTIL = 4,   // 주변 프론티어 칸 수: 그 칸 둘레 ±kUtilRadius 네모 안 프론티어 칸(지도 전체, 시선 검사 없음) / kUtilNorm (자름 1)
  F_CUR = 5,    // 지금 목표(직전 결정에서 고른 칸)가 이 덩어리 안이면 1
  F_VALID = 6,  // 후보 있음 1 / 빈 칸 0 (= 행동 가림 — 빈 칸은 확률 0)
};
constexpr int CAND_1D = K * FEAT;                         // 56
constexpr int POLICY_1D = mdp::obs::POLICY_1D + CAND_1D;  // 1360 (앞 1304 칸은 기본 모드와 같은 배치)
constexpr int CRITIC_1D = POLICY_1D + mdp::obs::PRIV;
constexpr int kCandMinCells = 1;     // 후보 덩어리 크기 문턱 (칸) — 0.75 m 미만 조각도 후보 (정체 판의 36 % 가 그런 조각만 남아 실패, 2026-11-08 조사)
constexpr int kCandMinDist = 6;      // 후보 칸은 로봇 칸에서 BFS 6 칸(0.3 m) 이상 — 그 안은 라이다가 못 보는 뒤·발밑(fov 220°)이라 가도 안 풀림. 잠정
                                     // (2026-11-08 규칙 정책: 이 문턱 없이 늘 0 번을 고르면 거리 0 덩어리에 "도착"해 128 판 모두 정체)
constexpr float kUtilRadius = 2.0f;  // m — 잠정 (시선 검사 없는 근사)
constexpr float kUtilNorm = 160.f;   // 칸 — 잠정 (±2 m 창 한 변 80 칸 × 2 줄)
constexpr int kPathMax = 64;         // 웨이포인트 상한 (EnvState)
}  // namespace mdp::hier

namespace mdp::act {
constexpr int CONT = 3;    // v, vy, ω — [−1, 1] 정규 단위, 환경이 한도를 곱하고 자름
constexpr int DECLARE = 1; // 끝 선언 (베르누이 로짓)
constexpr int DIM = CONT + DECLARE;
// 한도 (mdp/README "행동"): 정규 단위 −1 → 뒤 한도, +1 → 앞 한도
constexpr float V_FWD = 1.0f;    // m/s, 데이터시트 최고 속도 1 m/s (실측 1.11 은 바퀴 한도로)
constexpr float V_BACK = 0.3f;   // m/s, 라이다가 뒤 140° 를 못 봐서 느리게
constexpr float VY_MAX = 1.0f;   // m/s, 잠정 📏 (데이터시트에 옆 속도 없음)
constexpr float W_MAX = 3.14f;   // rad/s, 실측 (env.yaml drive.max)
constexpr bool SAFETY = true;    // 안전층 기본 켬 (시뮬·실물 같은 규칙)
// 계층형 모드의 행동: 환경마다 같은 DIM 칸, a[0] = 고른 후보 번호 0 … hier::K − 1 (정수 값), 나머지 0
}  // namespace mdp::act
