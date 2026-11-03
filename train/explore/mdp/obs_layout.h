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
constexpr int FRONTIER = 41;     // 가장 가까운 프론티어 거리, sin, cos, 프론티어 칸 수 — 4
constexpr int PROGRESS = 45;     // 최근 새로 안 칸 수, 마지막 새 칸 뒤 시간 — 2
constexpr int HOME = 47;         // 시작점까지 거리, sin, cos — 3
constexpr int RESERVED = 50;     // 나중 값(레이더 사람 등)용 0 칸
constexpr int VEC = 64;
constexpr int POLICY_1D = LIDAR + INTENSITY + VEC;   // 1304

// ---- 가치에만 (참값, 시뮬 전용) — 정책 1D 뒤에 붙인다
constexpr int PRIV_TRUE_POSE = 0;   // x, y, sin, cos — 4
constexpr int PRIV_POSE_ERR = 4;    // ex, ey, eth — 3
constexpr int PRIV_COVERAGE = 7;    // 진짜 벽 덮음 — 1
constexpr int PRIV_REMAIN = 8;      // 남은 벽 면 칸 (정규화) — 1
constexpr int PRIV_RESERVED = 9;    // 사람 참 위치 등
constexpr int PRIV = 32;
constexpr int CRITIC_1D = POLICY_1D + PRIV;

}  // namespace mdp::obs

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
}  // namespace mdp::act
