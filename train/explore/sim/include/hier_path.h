// 계층형 행동 모드 (EnvCfg::action_mode == kActPath, mdp/README "계층형"): 정책이 1 s 마다 프론티어 후보(mdp::CandSummary) 하나를 고르면
// 로봇 자기 지도(lmap)로 경로를 짜고(결정 때만) 0.1 s 마다 추종기가 (v, ω) 를 낸다. 정답 격자(MapDev::grid·blocked·body_hits)는 쓰지 않는다 —
// 쓰는 것은 로봇 지도 칸 종류(mdp::cell_class)·지도 자세(참값 + 오차)·직전 라이다 장(안전층)뿐이라 실물에서도 같은 값을 셀 수 있다
// (sim/tools/env_verify --hier 가 정답 격자를 바꿔도 경로·명령이 같은지 확인).
//
// 경로 계획 = 설 수 있는 칸(로봇 반경 0.2 m 팽창, mdp::standable — 관측 프론티어 요약과 같은 칸) 위 무게 있는 최단 경로(칸 4 이웃, 들어가는 칸의 무게).
//   무게는 Nav2 Smac 2D 의 칸 비용(nav2_smac_planner node_2d.cpp getTraversalCost: 1 + cost_travel_multiplier · cost / 252)에
//   팽창 비용(nav2_costmap_2d inflation_layer.hpp computeCost: 252 · exp(−k (d − r_inscribed)), r_inscribed 0.2 · k 3 · 팽창 반경 0.45 m,
//   cost_travel_multiplier 2 — nav2_params.yaml)을 넣은 값을 막힘까지 거리 d 세 구간 정수로 근사:
//     d ≤ 0.3 m → 3 (Nav2 2.48~3.0) · 0.3 < d ≤ 0.45 m → 2 (1.94~2.48) · d > 0.45 m → 1 (팽창 밖, 비용 0)
//   0.3 m 경계는 잠정(구간을 나눈 자리). 벽에 붙은 최단 경로 대신 가운데로 가게 해 몸통 쓸림을 줄인다(기준선 A* 와 같은 뜻).
//   GPU 는 칸 비트 판 위 층 BFS(무게 ≤ 3 → 다음 층 버킷 4 개, Dial 알고리즘)로, CPU 참조판은 칸마다 거리 배열로 — 거리 장이 같으므로
//   뒤로 짚기(이웃 차례 mdp::nb4 에서 처음으로 거리 = 내 거리 − 내 무게 인 칸)도 같다. GPU 는 거리를 mod 7 로만 남긴다
//   (이웃 거리는 내 거리 ± 3 안이라 mod 7 이면 앞 칸이 하나로 정해짐).
//   웨이포인트: 칸 경로를 앞에서부터 직선(4 이웃 선, 지나는 칸이 모두 설 수 있고 무게가 건너뛴 경로 칸의 최대 무게 이하)으로 이을 수 있는 데까지
//   늘려 꺾이는 칸만 남김(길이 상한 kSegMax 칸), 최대 mdp::hier::kPathMax 개 — 넘으면 앞 부분만(p_trunc, 결정마다 다시 짬).
//   같은 후보를 다시 고르면(그 덩어리가 지금 목표 칸을 품음 — CandTop::cur) 남은 경로가 아직 설 수 있는 칸 위면 그대로 둔다.
//   약한 막힘 (2026-11-08 사용자 결정): 지도 막힘(p > 0.65, 4 번 맞아야 됨)이 되기 전의 맞음 > 지나감 칸(로그 오즈 > 0)도 로봇에서 장애물 층
//   obstacle_max_range(3 m) 안이면 막힘으로 보고 먼저 짠다(엄격) — Nav2 global costmap 의 obstacle_layer 가 스캔 끝점을 그 거리 안에서 바로 LETHAL 로
//   두고 팽창하는 것과 같은 뜻(기준선 lethal_src·추종기 충돌 검사 path_lethal 과 같은 규칙). 목표가 그 팽창 안이거나 엄격으로 못 닿으면 지도 막힘만으로(느슨)
//   다시 짠다 — Nav2 는 거기서 계획 실패지만, 후보는 지도 막힘 기준(닫힘 정의)으로 고른 칸이라 갈 길을 남긴다. 느슨 경로는 충돌 검사가 약한 막힘을 본다.
//   측정 (규칙 정책 slot0, 시험 맵 64 판씩): 계획의 45~63 % 가 느슨으로 내려가고(목표 프론티어 칸이 막 보인 벽 옆), 충돌은 서점 0.156 → 0.188 ·
//   작은 집 0.141 → 0.156 으로 줄지 않음(64 판 표준오차 ≈ 0.05). 약한 막힘을 지나기는 허용하고 무게만 주는 판도 같은 충돌(0.188 · 0.156)
//
// 추종기 = Nav2 Regulated Pure Pursuit 식 (baselines/frontier frontier_explorer.cpp follow 와 같은 식·같은 nav2_params 값):
//   경로 위 가장 가까운 점에서 lookahead 만큼 앞 점(당근) → 방향 오차 > rotate_to_heading_min_angle 이면 제자리 회전, 아니면
//   곡률 감속(regulated_linear_scaling_min_radius)·장애물 근접 감속(use_cost_regulated_linear_velocity_scaling — 로봇 칸에서 막힘까지 거리,
//   Nav2 기본 cost_scaling_dist 0.6 · gain 1.0, 팽창 반경 안에서만)·다가가기 감속 → velocity_smoother 가속 한도(행동 주기 0.1 s) → 안전층.
//   직진 속도는 기준선 fast 판과 같은 1 m/s (env.yaml action.path). 후보가 없으면 제자리 회전(rotate_to_heading 속도)으로 지도를 넓힌다.
//
// 지금 스캔 (2026-11-09 사용자 결정 — 사람·움직이는 장애물): 정답 사람 자리는 안 쓴다. 쓰는 것은 지금 라이다 장(lidar_ring 의 lid_head 칸, 248 각도 칸마다
// 가장 가까운 거리 — 관측·안전층 front_gap 과 같은 값, 실물 /scan 으로 같은 값을 셀 수 있음)뿐. 근거 측정: explore_h1 780 반복 충돌 0.23 중 사람 0.105,
// 단계 0~3 충돌 0.02 · 0.14 · 0.21 · 0.28 (사람 수 0 · 1 · 2 · 3 과 같이 오름). 지도는 사람이 빈 칸으로 여러 번 본 바닥에 서면 한 장의 맞음으로는
// 로그 오즈가 0 을 못 넘어(빈 칸 갱신이 쌓인 만큼) 약한 막힘에도 안 잡힌다.
//   (a) 계획: 스캔 끝점(센서에서 obstacle_max_range 3 m 안 — nav2_params global obstacle_layer)을 엄격·느슨 둘 다 막힘으로 더한다
//       (기본 all = 모든 점. map_free = 지도가 빈 칸으로 아는 칸에 맞은 점만 — 비교는 path_scan_cell).
//       Nav2 ObstacleLayer marking 과 같은 뜻 (nav2_costmap_2d obstacle_layer.cpp updateBounds: 범위 안 점 칸 → LETHAL_OBSTACLE, 기본 obstacle_max_range 2.5).
//       지우기(clearing, raytrace)는 따로 안 한다 — 다음 결정에 지금 장으로 다시 짜므로 지난 장 점이 남지 않음(층이 결정마다 새로 만들어지는 꼴).
//       팽창은 지도 막힘과 같다 (설 수 있음 = 로봇 반경 0.2 m 밖, 무게 경계 0.3 · 0.45 m). 각도는 칸 가운데 (칸 폭 0.89° → 3 m 에서 4.7 cm < 지도 칸 5 cm).
//   (b) 추종기: Nav2 collision_monitor 의 approach (FootprintApproach) — velocity_smoother 뒤, 안전층 앞. 지금 명령 (v, ω) 로 발자국(RPP 충돌 검사와 같은
//       footprint + padding)을 simulation_time_step 마다 time_before_collision 까지 굴려 발자국 안 스캔 점이 max_points 넘는 첫 시각 t 면 (v, ω) × t / T
//       (nav2_collision_monitor polygon.cpp getCollisionTime · collision_monitor_node.cpp processApproach, 값은 humble collision_monitor_params.yaml).
//       Nav2 와 다른 한 곳: 지금 발자국 안에 이미 든 점은 첫 걸음에 더 깊이 들 때만 센다(그 수 > max_points 면 0 = 멈춤, Nav2 의 0 초 충돌) —
//       Nav2 는 안에 든 점이 있으면 무조건 멈추는데, 그러면 곁에 붙은 사람·벽에서 돌거나 물러나 빠져나오지도 못하고 서 있다가 걸어오는 사람과
//       부딪힌다(시뮬은 사람이 다가와 겹쳐도 충돌). 처음 판(안에 든 점을 아예 안 셈)은 0.05 배까지 늦춘 뒤 패딩 안으로 들면 다시 제 속도로 밀고 들어갔다.
//   (c) 제자리 회전: 보이는 앞 220° 안 점은 (b) 가 발자국을 ω 로 돌려 보고 줄인다. 라이다가 못 보는 뒤 140° 는 지도 약한 막힘으로 RPP 충돌 검사
//       (path_imminent — 제자리 회전 세 곳 모두 거침)가 막고, 지도에 안 남는 사람(빈 칸으로 여러 번 본 바닥)은 직전 4 장(0.4 s)을 그때 자세와 함께
//       기억해 회전 원 안 뒤쪽 점이 있으면 그 걸음은 돌지 않는다(path_rotate_blocked).
//   측정(mdp/README "계층형" — 규칙 정책 256 판씩): 켜면 성공·정체는 뚜렷이 좋아지나 사람 충돌은 표준오차 안. 사람 충돌의 대부분은 로봇을 무시하고 걷는
//   사람이 서 있거나 제자리에서 도는 로봇에 걸어 들어온 것 — 지금 장(정지 장면 가정)만으로는 못 막고 움직임 추정·비켜서기가 필요하다. (c) 는 거의 안 걸림.
//
// 깊이 카메라 장애물 층 (2026-11-10, EnvCfg::cam_on · env.yaml action.path.camera_obstacles) — 하위 층 실패 분석(hier_cf 1792 판)에서 벽 충돌 0.17 의 98 % 가
// 닿은 칸이 로봇 지도에서 빈 칸, 92 % 몸통 앞: 몸통 높이 띠(collision_band 0.03~0.25 m) 중 라이다 면(0.111 m) 밖의 낮은 물체·가구 윗판을 라이다로 짠
// 계획·충돌 검사·안전층 모두 못 봄. Nav2 가 깊이 카메라를 쓰는 방식 그대로 — VoxelLayer (nav2_costmap_2d voxel_layer.cpp updateBounds: 지우기 광선
// raytraceFreespace → markVoxelInMap, 2 차원 칸은 높이 칸 하나라도 켜지면 LETHAL = mark_threshold 0) + 관측 거르기 (observation_buffer.cpp bufferCloud:
// min_obstacle_height ≤ z ≤ max_obstacle_height 인 점만 남겨 표시·지우기 모두 그 점으로). 값: 띠 = collision_band (Nav2 문서도 깊이 센서는 바닥 점이
// 안 남게 min 을 0 보다 조금 위로 권함), obstacle_max_range 2.5 · raytrace_max_range 3.0 = Nav2 기본, 높이 칸 8 개(2.75 cm)는 잠정.
//   입력: 정책 관측과 같은 깊이 바이트 40 × 64 (observations.h depth_u8 — 실물 /camera/depth 를 같은 함수로 줄인 값, 정답 격자 아님). 시뮬은 깊이를 결정마다
//   (1 Hz) 그려서 층도 결정 앞에 한 번 갱신 (실물 29 Hz 보다 드묾 — 보수 쪽). 점 = 픽셀 광선 × z → 숙인 각·장착(env.yaml camera_mount·camera_pitch) →
//   지도 자세(참값 + 오차)로 칸. 0(구멍)·255(3 m 넘음)는 점 없음 (Nav2 inf_is_valid 기본 false).
//   차례: 모든 픽셀 지우기(띠 안 점까지 3 차원 광선이 지나는 높이 칸, 끝 칸 빼고) → 표시 → 몸통 밑 칸 지우기 (ObstacleLayer footprint_clearing_enabled).
//   높이 칸으로 지우므로 낮은 상자 위를 지나는 광선은 상자 높이 칸을 못 지운다 — 가까이 와서 상자가 시야 아래로 빠져도 층에 남는다 (2 차원 층이면 지워짐).
//   로봇 지도(lmap)에 안 넣는 까닭: lmap 은 정책 관측(가까운·먼 지도)·닫힘 성공·프론티어 후보·reward_occ 가 쓰는 Cartographer 꼴 점유 지도라 넣으면
//   관측 분포·닫힘·보상이 바뀐다 (닫힘·후보는 다른 작업 몫). Nav2 도 SLAM 지도(static layer)와 장애물 층을 따로 두고 costmap 에서만 합친다.
//   쓰는 곳: 경로 계획(엄격·느슨 둘 다 막힘 — 지금 스캔과 같은 몫), RPP 충돌 검사(path_lethal), 장애물 근접 감속(path_obstacle_dist),
//   collision_monitor (마지막 깊이 장 열마다 가장 가까운 띠 안 점 cpts — pointcloud_to_laserscan 꼴 scan 출처, 1 s 묵은 장은 Nav2 source_timeout 안).
//   env_verify --hier: GPU == CPU 층·점 비트 같음, 정답 격자를 바꿔도 층이 같음.
//
// 계획·충돌 검사 일치 (2026-11-10, EnvCfg::path_consistent · action.path.consistent_plan): 진행 없음 0.10 의 76 % 가 추종 중 RPP 충돌 검사 멈춤·물러남 반복.
// 진단(hier_cf, f0b48eb): 그 창에서 검사가 걸린 걸음의 41~59 % 는 지금 자세가 이미 패딩 발자국에 걸림(모든 명령 막힘), 44~78 % 는 물러나기도 막힘.
// 까닭: 계획은 로봇 반경 0.2 m 원(칸 가운데끼리)으로 설 수 있는 칸을 고르는데, 검사는 footprint(0.32 × 0.22) + 패딩 0.05 의 사각형(외접 반경 0.26 m)이라
// 계획한 경로 위에 선 자세가 검사에선 충돌. Nav2 도 Smac 2D 는 원(칸 비용 ≥ INSCRIBED)으로 짜고 RPP 는 다각형 발자국 ≥ LETHAL 로 검사해 같은 어긋남이 있다.
//   (a) 계획 반경 = 검사 발자국(패딩 포함) 외접 반경 + 칸 대각 (= 0.264 + 0.071 m): 발자국 점 p 는 중심에서 외접 반경 안 → p 가 든 칸 가운데는 중심 칸
//       가운데에서 외접 반경 + 칸 대각 안 → 그 칸은 막힘이 아님. 그래서 설 수 있는 칸에 중심이 있는 자세는 어느 방향이든 검사를 통과 (경로 위 = 통과).
//   (b) 검사 막힘 = 지난 계획 칸 기준: 느슨 계획(지도 막힘만)을 따라갈 때는 약한 막힘을 안 봄 (옛 검사는 늘 약한 막힘까지 봐서 느슨 경로를 막음).
//   (c) 빠져나오기: 지금 자세가 이미 패딩에 걸렸으면 굴린 자세를 몸통(패딩 없음)으로만 검사, 물러나기 검사도 같음 — 옛 동작은 Nav2 그대로 모든 명령을 막아 섬.
//   (e) 계획 반경으로는 엄격·느슨 모두 못 짜면 (좁은 통로 — 계획 반경 원이 안 들어감) 느슨 막힘 + 옛 반경(관측 BFS 0.2 m)으로 한 번 더 (p_mode 3).
//       그 경로는 검사와 어긋날 수 있어 (c) 빠져나오기·복구가 받는다. 끄면 그 자리에서 섬 (진행 없음).
//   (d) 목표 허용: 후보 칸(관측 BFS 반경 0.2 m, 로봇 지도만)은 더 큰 계획 반경·깊이 카메라 층으로 설 수 없을 수 있다 → 목표 둘레 Smac tolerance(nav2_params
//       GridBased 0.25 m) 안 칸 중 처음 닿는 층의 (목표까지 거리², 칸) 최소, 그것도 못 닿고 BFS 가 끝나면 goal_far_tolerance(잠정 1 m) 안 닿은 칸 중 최소.
//
// 복구 (2026-11-10, EnvCfg::rec_on · action.path.recovery): 같은 목표(rec_bl_r 안)에서 결정 걸음의 절반 넘게 충돌 검사가 막은 결정이 이어지면 Nav2 BT
// navigate_to_pose_w_replanning_and_recovery.xml 꼴로 — 1·2 번째: 남은 경로를 버리고 다시 계획 (ComputePathToPose·FollowPath RecoveryNode),
// 3 번째: BackUp backup_dist 0.30 m @ backup_speed 0.05 m/s (BT 값, 6 s), 4 번째: 그 목표 포기 (NavigateRecovery 재시도 다 씀 → 목표 실패) — rec_bl_steps
// 동안 그 둘레 후보를 고르면 목표 없음(제자리 회전)으로. "막힌 비율"을 정책 후보 칸에 넣는 것은 하지 않음: mdp::hier::FEAT 가 바뀌면 관측 크기가 바뀌어
// 지금 체크포인트(model_h2_2850)를 못 쓰고 다시 학습해야 하며, 포기한 목표는 이미 정책이 고를 때 제자리 회전으로 바뀜.
#pragma once
// namespace sim 안에서 넣는다 (env.h — people_sim.h 와 같게)

constexpr int kPathPlanes = 13;   // GPU 계획 비트 판 수 (EnvBatch::pplanes): 무게 경계 2 · 버킷 4 · 거리 mod 7 3 비트 · 엄격 막힘 · 엄격 설 수 있음 · 느슨 막힘(지도 ∪ 스캔) · 느슨 설 수 있음
enum : int { kPpC3 = 0, kPpC45 = 1, kPpB = 2, kPpD = 6, kPpLT = 9, kPpSST = 10, kPpLO = 11, kPpSLO = 12 };

// 지금 라이다 장 각도 칸 k 의 점 (base_link 기준 m). 맞은 점이 없으면 false. *r = 센서에서 거리
DEV bool scan_point(const EnvBatch& b, const EnvState& e, int env, int k, float* px, float* py, float* r_o) {
  const EnvCfg& c = b.cfg;
  const float r = b.lidar_ring[((size_t)env * kLidarRing + e.lid_head) * mdp::obs::LIDAR_BINS + k];
  if (!(r < c.map_max_range)) return false;
  const float a = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
  float s, co;
  dm::sincosf_d(a, &s, &co);
  *px = c.mount_x + r * co;
  *py = c.mount_y + r * s;
  *r_o = r;
  return true;
}
// (a) 계획에 넣을 스캔 끝점 칸: 센서에서 obstacle_max_range(path_obs_range) 안, 지도 자세로 칸. 맵 밖이면 false.
// path_scan 2 (all, 기본 — Nav2 그대로): 모든 점. 1 (map_free): 지도는 그 칸을 빈 칸으로 아는데(로그 오즈 ≤ 0) 지금 스캔이 맞은 점 = 지도와 어긋나는 점만
// (움직이는 물체) — 막 보인 벽 옆 목표를 느슨으로도 못 짜 서 버리는 판을 줄이려던 판. 2026-11-09 규칙 정책 256 판씩 비교에서 all 이 정답 덮음 ≥ 0.8 인
// 성공 0.32·0.48 vs map_free 0.27·0.45 (서점·작은 집, 표준오차 ≈ 0.03 — 차이 없다고 봄), 대신 성공까지 153·149 s vs 134·145 s → 출처 있는 Nav2 쪽을 기본으로
DEV bool path_scan_cell(const EnvBatch& b, const EnvState& e, int env, int k, int* cx, int* cy) {
  const EnvCfg& c = b.cfg;
  float x, y, r;
  if (!scan_point(b, e, env, k, &x, &y, &r) || r > c.path_obs_range) return false;
  const float pth = e.th + e.eth;
  float s, co;
  dm::sincosf_d(pth, &s, &co);
  const int ci = cell_of(c, b.maps[e.map], (e.x + e.ex) + (co * x - s * y), (e.y + e.ey) + (s * x + co * y), cx, cy);
  if (ci < 0) return false;
  if (c.path_scan == 2) return true;   // 모든 점
  const int16_t l = b.lmap[(size_t)env * c.size * c.size + ci];
  return l != mdp::kUnknownL && l <= 0;
}

// ---------- 깊이 카메라 장애물 층 (2026-11-10, EnvCfg::cam_on — 아래 머리말 "깊이 카메라 장애물 층") ----------
constexpr float kCamNone = 3e38f;   // EnvBatch::cpts 빈 칸
// 관측 깊이 픽셀 i (행 · DEPTH_W + 열, 값 q = obs_img 깊이 바이트 — 정책 입력과 같은 값, observations.h depth_u8) → base_link 점 (m).
// 0 = 구멍 · 255 = obs_max_z(3 m) 넘음 → 점 없음 (Nav2 inf_is_valid 기본 false — 무한 값은 표시·지우기 모두 안 씀). 1~254 는 칸 가운데 거리
DEV bool cam_point_base(const EnvCfg& c, uint8_t q, int i, float* bx, float* by, float* bz) {
  if (q == 0 || q == 255) return false;
  const DepthSpec& D = c.depth.spec;
  const float z = D.min_z + ((float)(q - 1) + 0.5f) * ((D.obs_max_z - D.min_z) / 254.f);
  const float su = (float)D.w / (float)mdp::obs::DEPTH_W, sv = (float)D.h / (float)mdp::obs::DEPTH_H;   // observations.h depth_uv 와 같은 픽셀
  const float u = ((float)(i % mdp::obs::DEPTH_W) + 0.5f) * su - 0.5f, v = ((float)(i / mdp::obs::DEPTH_W) + 0.5f) * sv - 0.5f;
  const float X = z, Y = z * (-(u - D.cx) / D.fx), Z = z * (-(v - D.cy) / D.fy);   // 센서 FLU (sensors.h depth_ray × z)
  *bx = c.cam_x + (c.cam_cp * X + c.cam_sp * Z);                                    // 숙인 각 (observations.h depth_camera 의 yaw 0 회전)
  *by = c.cam_y + Y;
  *bz = c.cam_z + (c.cam_cp * Z - c.cam_sp * X);
  return true;
}
// 높이 → 높이 칸 번호 (띠 밖 −1)
DEV int cam_voxel(const EnvCfg& c, float h) {
  if (!(h >= c.cam_min_h && h <= c.cam_max_h)) return -1;
  const int k = (int)floorf((h - c.cam_min_h) * ((float)c.cam_nz / (c.cam_max_h - c.cam_min_h)));
  return k < c.cam_nz ? k : c.cam_nz - 1;
}
DEV const uint8_t* cam_depth(const EnvBatch& b, int env) { return b.obs_img + (size_t)env * mdp::obs::IMG_SIZE + mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE; }
// 픽셀 하나의 층 갱신 몫. phase 0 = 지우기: 띠 안 점까지 3 차원 광선이 지나는 높이 칸 (끝 칸 빼고, 센서에서 cam_ray_range 안) → op(칸, 비트) 로 끔.
// phase 1 = 표시: 띠 안 점이 센서에서 cam_obs_range 안이면 그 칸 그 높이 비트 → op(칸, 비트) 로 켬. 지도 자세(참값 + 오차)로 칸 — 실물과 같은 값.
// Nav2 voxel_layer.cpp updateBounds (raytraceFreespace → markVoxelInMap) · observation_buffer.cpp bufferCloud(min/max_obstacle_height 로 거른 점만 둘 다 씀)
template <class Op>
DEV void cam_pixel(const EnvBatch& b, const EnvState& e, int env, int i, int phase, const Op& op) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  float bx, by, bz;
  const bool pt = cam_point_base(c, cam_depth(b, env)[i], i, &bx, &by, &bz);
  const int kv = pt ? cam_voxel(c, bz) : -1;
  if (kv < 0) return;
  const float px = e.x + e.ex, py = e.y + e.ey;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  const float ox = px + (co * c.cam_x - s * c.cam_y), oy = py + (s * c.cam_x + co * c.cam_y);   // 센서 (지도)
  const float wx = px + (co * bx - s * by), wy = py + (s * bx + co * by);                       // 점 (지도)
  const float hx = bx - c.cam_x, hy = by - c.cam_y;
  const float lh = sqrtf(hx * hx + hy * hy);   // 센서에서 수평 거리
  int hcx, hcy;
  const int hit = cell_of(c, m, wx, wy, &hcx, &hcy);
  if (phase == 1) {
    if (hit >= 0 && lh <= c.cam_obs_range) op(hit, (uint8_t)(1u << kv));
    return;
  }
  // 지우기: 수평 반 칸 간격 표본 (Nav2 clearVoxelLineInMap 의 3 차원 Bresenham 자리 — 표본이 건너뛴 칸은 안 지움 = 덜 지움 쪽)
  const int n = (int)ceilf(lh / (0.5f * c.res));
  for (int q = 0; q < n; ++q) {
    const float t = (float)q / (float)n;
    if (t * lh > c.cam_ray_range) break;
    const float h = c.cam_z + t * (bz - c.cam_z);
    const int k = cam_voxel(c, h);
    if (k < 0) continue;
    int cx, cy;
    const int ci = cell_of(c, m, ox + t * (wx - ox), oy + t * (wy - oy), &cx, &cy);
    if (ci < 0 || ci == hit) continue;
    op(ci, (uint8_t)(1u << k));
  }
}
// 몸통 밑 칸 (칸 가운데가 지도 자세 몸통 사각형 안 — 패딩 없음) j 번째: 몸통 둘레 상자 (2R+1)² 칸을 차례로. 반환 = 칸 번호 (아니면 −1).
// Nav2 ObstacleLayer footprint_clearing_enabled (기본 true) 자리 — 몸통이 겹친 칸은 실물이면 이미 부딪힌 칸이라 자세 오차로 남은 표시
DEV int cam_fp_cell(const EnvBatch& b, const EnvState& e, int j) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int R = (int)ceilf(sqrtf(c.half_len * c.half_len + c.half_wid * c.half_wid) / c.res) + 1, W = 2 * R + 1;
  if (j >= W * W) return -1;
  const float px = e.x + e.ex, py = e.y + e.ey;
  int rcx, rcy;
  cell_of(c, m, px, py, &rcx, &rcy);
  const int x = rcx - R + j % W, y = rcy - R + j / W;
  if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return -1;
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  const float dx = (m.ox + ((float)x + 0.5f) * c.res) - px, dy = (m.oy + ((float)y + 0.5f) * c.res) - py;
  const float u = co * dx + s * dy, v = co * dy - s * dx;
  return (dm::absf(u) <= c.half_len && dm::absf(v) <= c.half_wid) ? y * c.size + x : -1;
}
DEV int cam_fp_cells(const EnvCfg& c) {
  const int R = (int)ceilf(sqrtf(c.half_len * c.half_len + c.half_wid * c.half_wid) / c.res) + 1;
  return (2 * R + 1) * (2 * R + 1);
}
// collision_monitor 점 출처 (cpts): 깊이 열 col 마다 띠 안·cam_obs_range 안 점 중 센서에서 수평으로 가장 가까운 점 하나 (지도 좌표, 없으면 kCamNone) —
// ROS pointcloud_to_laserscan(min_height·max_height → 각도 칸마다 가장 가까운 점)과 같은 꼴. 64 열 × 1.03° 라 라이다 장(248 칸 × 0.89°)과 점 밀도가 비슷해
// Nav2 max_points 문턱을 그대로 쓴다 (픽셀마다 한 점이면 벽 한 곳이 높이마다 여러 점이 되어 문턱 뜻이 달라지고 2560 점을 걸음마다 굴려 학습이 1.5 배 느려짐)
DEV void cam_column(const EnvBatch& b, const EnvState& e, int env, int col) {
  const EnvCfg& c = b.cfg;
  float best = 3e38f, bx0 = 0.f, by0 = 0.f;
  for (int r = 0; r < mdp::obs::DEPTH_H; ++r) {
    const int i = r * mdp::obs::DEPTH_W + col;
    float bx, by, bz;
    if (!cam_point_base(c, cam_depth(b, env)[i], i, &bx, &by, &bz) || cam_voxel(c, bz) < 0) continue;
    const float hx = bx - c.cam_x, hy = by - c.cam_y, l2 = hx * hx + hy * hy;
    if (l2 > c.cam_obs_range * c.cam_obs_range || !(l2 < best)) continue;
    best = l2;
    bx0 = bx;
    by0 = by;
  }
  float* P = b.cpts + ((size_t)env * mdp::obs::DEPTH_W + col) * 2;
  if (best == 3e38f) { P[0] = kCamNone; P[1] = kCamNone; return; }
  float s, co;
  dm::sincosf_d(e.th + e.eth, &s, &co);
  P[0] = (e.x + e.ex) + (co * bx0 - s * by0);
  P[1] = (e.y + e.ey) + (s * bx0 + co * by0);
}
// 열 점을 차례 그대로 앞으로 모으고 끝에 kCamNone (스레드 하나, 결정마다 한 번)
DEV void cam_compact(const EnvBatch& b, int env) {
  float* P = b.cpts + (size_t)env * mdp::obs::DEPTH_W * 2;
  int n = 0;
  for (int i = 0; i < mdp::obs::DEPTH_W; ++i) {
    if (P[2 * i] == kCamNone) continue;
    P[2 * n] = P[2 * i];
    P[2 * n + 1] = P[2 * i + 1];
    ++n;
  }
  if (n < mdp::obs::DEPTH_W) { P[2 * n] = kCamNone; P[2 * n + 1] = kCamNone; }
}
// 층 칸이 막힘인가 (Nav2 mark_threshold 0: 높이 칸 하나라도 켜짐 → LETHAL)
DEV bool cam_lethal(const EnvBatch& b, int env, int x, int y) {
  return b.cvox && b.cvox[(size_t)env * b.cfg.size * b.cfg.size + (size_t)y * b.cfg.size + x] != 0;
}

// (c) 제자리 회전 검사: 직전 장들(링의 지난 kLidarRing − 1 장, 0.1~0.4 s 전 — 장마다 그때 지도 자세 lid_pose)의 점을 지금 로봇 기준으로 옮겨,
// 지금 라이다가 못 보는 곳(라이다에서 ±110° 밖 — 뒤 140°)에 있고 몸통이 돌며 쓰는 원(발자국 꼭짓점 반경) 안·지금 발자국 밖인 점이 max_points 넘으면 true.
// 보이는 곳은 지금 장이 더 새 값이라 (b) 가 맡는다. 지도로는 안 잡히는 경우(사람이 빈 칸으로 여러 번 본 바닥에 섬 — 위 머리말)를 위한 짧은 기억
DEV bool path_rotate_blocked(const EnvBatch& b, const EnvState& e, int env) {
  const EnvCfg& c = b.cfg;
  const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
  const float rs2 = hl * hl + hw * hw;
  const float fov = -mdp::obs::kLidarMin;   // 110°
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  float ps, pc;
  dm::sincosf_d(pth, &ps, &pc);
  int cnt = 0;
  for (int f = 1; f < kLidarRing; ++f) {
    const int slot = (e.lid_head - f + kLidarRing) % kLidarRing;
    const float* R = b.lidar_ring + ((size_t)env * kLidarRing + slot) * mdp::obs::LIDAR_BINS;
    const float fx = e.lid_pose[3 * slot], fy = e.lid_pose[3 * slot + 1];
    float fs, fc;
    dm::sincosf_d(e.lid_pose[3 * slot + 2], &fs, &fc);
    for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
      const float r = R[k];
      if (!(r < c.map_max_range) || r > 1.f) continue;   // 1 m 밖 점은 몸통 원(≈ 0.27 m)에 못 듦 — 라이다는 몸통 앞 0.1 m
      const float a = mdp::obs::kLidarMin + ((float)k + 0.5f) * mdp::obs::kLidarBin;
      float s, co;
      dm::sincosf_d(a, &s, &co);
      const float bx = c.mount_x + r * co, by = c.mount_y + r * s;            // 그 장 base_link
      const float wx = fx + (fc * bx - fs * by), wy = fy + (fs * bx + fc * by);   // 지도
      const float dx = wx - px, dy = wy - py;
      const float x = pc * dx + ps * dy, y = pc * dy - ps * dx;               // 지금 base_link
      if (x * x + y * y > rs2) continue;
      if (dm::absf(x) <= hl && dm::absf(y) <= hw) continue;
      if (dm::absf(dm::atan2f_d(y - c.mount_y, x - c.mount_x)) <= fov) continue;   // 지금 보이는 곳
      ++cnt;
    }
  }
  return cnt > c.mon_max_points;
}

// (b) collision_monitor approach: 명령 (v, ω) 에 곱할 배율 [0, 1] (1 = 걸림 없음)
DEV float path_monitor_scale(const EnvBatch& b, const EnvState& e, int env, float v, float w) {
  const EnvCfg& c = b.cfg;
  if (!c.mon_on || (v == 0.f && w == 0.f)) return 1.f;
  const int n = (int)(c.mon_t / c.mon_dt + 1e-3f);   // 굴릴 걸음 수 (≤ 64, env_host 가 확인)
  const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
  // 굴린 자세 (지금 base_link 기준, Euler — nav2 projectState 와 같은 차례: 방향을 먼저 돌리고 그 방향으로 나아감)
  float qx[64], qy[64], qc[64], qs[64];
  int cnt[64];
  {
    float x = 0.f, y = 0.f, th = 0.f;
    for (int j = 0; j < n; ++j) {
      th += w * c.mon_dt;
      float s, co;
      dm::sincosf_d(th, &s, &co);
      x += v * co * c.mon_dt;
      y += v * s * c.mon_dt;
      qx[j] = x;
      qy[j] = y;
      qc[j] = co;
      qs[j] = s;
      cnt[j] = 0;
    }
  }
  const float reach = dm::absf(v) * c.mon_t + sqrtf(hl * hl + hw * hw);   // 이보다 먼 점은 어느 자세의 발자국에도 안 듦
  int cnt0 = 0;   // 지금 발자국 안이고 첫 걸음에 더 깊이 드는 점 (위 머리말 "Nav2 와 다른 한 곳")
  const auto count = [&](float px, float py) {
    if (px * px + py * py > reach * reach) return;
    if (dm::absf(px) <= hl && dm::absf(py) <= hw) {   // 지금 발자국 안: 발자국 가장자리까지 깊이가 첫 걸음에 늘면 지금 걸림, 아니면(빠져나감) 안 셈
      const float dx = px - qx[0], dy = py - qy[0];
      const float u = qc[0] * dx + qs[0] * dy, t = qc[0] * dy - qs[0] * dx;
      const float d0 = dm::minf(hl - dm::absf(px), hw - dm::absf(py)), d1 = dm::minf(hl - dm::absf(u), hw - dm::absf(t));
      cnt0 += d1 > d0;
      return;
    }
    for (int j = 0; j < n; ++j) {
      const float dx = px - qx[j], dy = py - qy[j];
      const float u = qc[j] * dx + qs[j] * dy, t = qc[j] * dy - qs[j] * dx;
      if (dm::absf(u) <= hl && dm::absf(t) <= hw) cnt[j] += 1;
    }
  };
  for (int k = 0; k < mdp::obs::LIDAR_BINS; ++k) {
    float px, py, r;
    if (!scan_point(b, e, env, k, &px, &py, &r)) continue;
    count(px, py);
  }
  if (b.cpts) {   // 깊이 카메라 점 출처 (cam_column — 마지막 깊이 장의 열마다 가장 가까운 띠 안 점, 지도 좌표 → 지금 지도 자세 base_link)
    const float mx = e.x + e.ex, my = e.y + e.ey;
    float ms, mc;
    dm::sincosf_d(e.th + e.eth, &ms, &mc);
    const float* P = b.cpts + (size_t)env * mdp::obs::DEPTH_W * 2;
    for (int i = 0; i < mdp::obs::DEPTH_W; ++i) {
      if (P[2 * i] == kCamNone) break;   // cam_compact 가 모은 점 끝
      const float dx = P[2 * i] - mx, dy = P[2 * i + 1] - my;
      count(mc * dx + ms * dy, mc * dy - ms * dx);
    }
  }
  if (cnt0 > c.mon_max_points) return 0.f;   // getCollisionTime 이 0 을 내는 경우 (processApproach → 멈춤)
  for (int j = 0; j < n; ++j)
    if (cnt[j] > c.mon_max_points) return (float)(j + 1) * c.mon_dt / c.mon_t;   // 첫 걸린 시각 / T (processApproach change_ratio)
  return 1.f;
}
constexpr int kSegMax = 40;      // 웨이포인트 사이 경로 칸 수 상한 (2 m) — 잠정

DEV int path_weight(bool st, bool c_mid, bool c_out) { return !st ? 0 : (c_out ? 1 : (c_mid ? 2 : 3)); }
// 무게 경계 반경 (설 수 있는 칸 반경 mdp::obsgen::kRobotRadius 0.2 m 바깥 두 구간) → 로봇 반경 꼴
DEV mdp::Robot path_rb_mid(const EnvCfg& c) { return mdp::robot_cells(c.path_r_mid, c.res); }
DEV mdp::Robot path_rb_out(const EnvCfg& c) { return mdp::robot_cells(c.path_r_out, c.res); }
// 계획 반경 (설 수 있는 칸): path_consistent 면 path_r_plan, 아니면 obs (관측 BFS 반경 mdp::obsgen::reach_robot — 옛 동작)
DEV mdp::Robot path_rb_plan(const EnvCfg& c, mdp::Robot obs) { return c.path_consistent ? mdp::robot_cells(c.path_r_plan, c.res) : obs; }
// 목표 허용 반경 (머리말 "목표 허용" — 계획 반경·깊이 카메라 층 중 하나라도 켜졌을 때만, 아니면 r2 = 0 — 목표 칸 하나 = 옛 동작)
DEV mdp::Robot path_rb_tol(const EnvCfg& c) { return (c.path_consistent || c.cam_on) ? mdp::robot_cells(c.path_goal_tol, c.res) : mdp::Robot{0, 0}; }
DEV mdp::Robot path_rb_far(const EnvCfg& c) { return (c.path_consistent || c.cam_on) ? mdp::robot_cells(c.path_goal_far, c.res) : mdp::Robot{0, 0}; }

// (x0, y0) → (x1, y1) 4 이웃 선(한 걸음에 x 나 y 하나만)의 모든 칸이 0 < 무게 ≤ worst 인가. W(x, y) → 무게 (0 = 못 섬)
template <class W>
DEV bool path_los(int x0, int y0, int x1, int y1, int worst, const W& wfn) {
  // 격자 걷기(칸 가운데 선이 다음에 x 경계를 먼저 넘나 y 경계를 먼저 넘나 — 정수 비교, 같으면 y 먼저)
  const int nx = x1 > x0 ? x1 - x0 : x0 - x1, ny = y1 > y0 ? y1 - y0 : y0 - y1;
  const int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  int x = x0, y = y0;
  {
    const int w = wfn(x, y);
    if (w <= 0 || w > worst) return false;
  }
  for (int ix = 0, iy = 0; ix < nx || iy < ny;) {
    if ((1 + 2 * ix) * ny < (1 + 2 * iy) * nx) { x += sx; ++ix; }
    else { y += sy; ++iy; }
    const int w = wfn(x, y);
    if (w <= 0 || w > worst) return false;
  }
  return true;
}

// 칸 경로 cells[0..n) (y · gw + x, 시작 → 목표) → 웨이포인트 wp[2i] = x, wp[2i+1] = y (시작 칸 포함, 목표 칸으로 끝). 반환 = 웨이포인트 수.
// *trunc = 1 이면 max_wp 를 넘어 앞 부분만 담음(마지막 점 ≠ 목표)
template <class W>
DEV int path_simplify(const int32_t* cells, int n, int gw, const W& wfn, int16_t* wp, int max_wp, int* trunc) {
  *trunc = 0;
  if (n <= 0) return 0;
  int nw = 0;
  const auto put = [&](int cell) {
    wp[2 * nw] = (int16_t)(cell % gw);
    wp[2 * nw + 1] = (int16_t)(cell / gw);
    ++nw;
  };
  put(cells[0]);
  int a = 0, worst = wfn(cells[0] % gw, cells[0] / gw);
  int i = 1;
  while (i < n) {
    const int wi = wfn(cells[i] % gw, cells[i] / gw);
    const int w2 = wi > worst ? wi : worst;
    if (i - a <= kSegMax && path_los(cells[a] % gw, cells[a] / gw, cells[i] % gw, cells[i] / gw, w2, wfn)) {
      worst = w2;
      ++i;
      continue;
    }
    // i − 1 까지가 이어짐 (i − 1 == a 면 이웃 칸 하나도 안 이어지는 경우 — 경로 칸끼리는 4 이웃이라 늘 이어지므로 안 옴)
    if (nw >= max_wp - 1) { *trunc = 1; put(cells[i - 1]); return nw; }
    put(cells[i - 1]);
    a = i - 1;
    const int wa = wfn(cells[a] % gw, cells[a] / gw);
    worst = wa;
  }
  put(cells[n - 1]);
  return nw;
}

// 남은 경로(p_i 구간부터)가 지금 지도에서 모두 설 수 있는 칸 위인가. S(x, y) → 설 수 있음
template <class S>
DEV bool path_clear(const EnvState& e, const S& stand) {
  for (int i = e.p_i < 0 ? 0 : e.p_i; i + 1 < e.p_n; ++i) {
    const int x0 = e.wp[2 * i], y0 = e.wp[2 * i + 1], x1 = e.wp[2 * i + 2], y1 = e.wp[2 * i + 3];
    if (!path_los(x0, y0, x1, y1, 1, [&](int x, int y) { return stand(x, y) ? 1 : 0; })) return false;
  }
  if (e.p_n == 1 && !stand(e.wp[0], e.wp[1])) return false;
  return true;
}

// 복구 (EnvCfg::rec_on, 머리말 "복구"): 결정 앞(경로 계획 전, 스레드 하나). 정책이 고른 후보 번호 k → 실제로 갈 후보 번호 (−1 = 목표 없음 — 제자리 회전).
// 지난 결정이 "목표로 못 나아감"이었는지 셈: (a) 충돌 검사가 결정 걸음의 절반 넘게 멈춤·물러남, (b) 목표는 있는데 경로를 못 짬, (c) 허용 칸(목표 아닌 경로 끝)에
// 도착해 섬 — 같은 목표(rec_bl_r 안)에서 이어지면 1·2 번째 다시 계획(*force = 1), 3 번째 (a) 면 물러나기, 4 번째 포기(rec_bl_steps 동안 그 둘레 후보 안 감).
// 포기한 목표 둘레 후보를 고르면 포기 안 한 후보 중 번호가 가장 작은 것으로 바꿈 (explore_lite 가 실패한 목표를 goal_blacklist 에 넣고 다음 프론티어로 가는 꼴)
// 행동 칸: 계층형은 후보 K 칸(b.cand), 그래프 모드는 현재 노드의 이웃 NB 칸(b.gcand) — 계획·복구는 이 셋으로만 고른 칸을 본다
DEV int act_slots(const EnvBatch& b) { return b.gcand ? mdp::graph::NB : mdp::hier::K; }
DEV const int32_t* act_cells(const EnvBatch& b, int env) { return b.gcand ? b.gcand + (size_t)env * mdp::graph::NB : b.cand + (size_t)env * mdp::hier::K; }
DEV int slot_of(const EnvBatch& b, float a) { return (a >= 0.f && a < (float)act_slots(b)) ? (int)a : -1; }   // 범위 밖·NaN 은 −1 (빈 칸과 같음 — 제자리 회전)
// 고른 칸이 지금 목표와 같은가: 계층형은 그 덩어리가 지금 목표 칸을 품음(cand_cur 비트), 그래프 모드는 같은 노드 칸
DEV bool act_same(const EnvBatch& b, const EnvState& e, int env, int k) {
  if (b.gcand) return k >= 0 && b.gcand[(size_t)env * mdp::graph::NB + k] == e.goal_cell;
  return ((b.cand_cur[env] >> k) & 1) != 0;
}
DEV bool path_near(const EnvCfg& c, const MapDev& m, int a, int b, float r) {
  const int dx = a % m.gw - b % m.gw, dy = a / m.gw - b / m.gw;
  const float q = r / c.res;
  return (float)(dx * dx + dy * dy) <= q * q;
}
DEV bool path_blacklisted(const EnvCfg& c, const MapDev& m, const EnvState& e, int cell) {
  for (int i = 0; i < kRecBl; ++i)
    if (e.p_bl_cell[i] >= 0 && e.step < e.p_bl_until[i] && path_near(c, m, cell, e.p_bl_cell[i], c.rec_bl_r)) return true;
  return false;
}
DEV int path_recovery(const EnvBatch& b, EnvState& e, int env, int k, int* force) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  *force = 0;
  if (!c.rec_on) return k;
  const int32_t* C = act_cells(b, env);
  // 지난 결정이 목표로 못 나아갔나
  bool stuck = e.p_blk_now * 2 > c.decide_steps;   // (a)
  const bool blocked = stuck;
  e.p_blk_now = 0;
  if (e.goal_cell >= 0 && e.p_n == 0) stuck = true;   // (b)
  if (e.goal_cell >= 0 && e.p_n > 0 && !e.p_trunc) {    // (c)
    const int ex = e.wp[2 * (e.p_n - 1)], ey = e.wp[2 * (e.p_n - 1) + 1];
    if (ey * m.gw + ex != e.goal_cell) {
      const float dx = m.ox + ((float)ex + 0.5f) * c.res - (e.x + e.ex), dy = m.oy + ((float)ey + 0.5f) * c.res - (e.y + e.ey);
      stuck = stuck || dx * dx + dy * dy <= c.goal_tol * c.goal_tol;
    }
  }
  const int tgt = k >= 0 ? C[k] : -1;
  const bool same = tgt >= 0 && e.goal_cell >= 0 && path_near(c, m, tgt, e.goal_cell, c.rec_bl_r);
  e.p_blk = (same && stuck) ? e.p_blk + 1 : 0;
  if (e.p_blk >= 4) {   // 포기 (BT NavigateRecovery 가 재시도를 다 쓰면 목표 실패)
    const int i = e.p_gaveup % kRecBl;
    e.p_bl_cell[i] = tgt;
    e.p_bl_until[i] = e.step + c.rec_bl_steps;
    e.p_blk = 0;
    e.p_gaveup += 1;
  } else if (e.p_blk == 3 && blocked) {   // 물러나기 (BT BackUp backup_dist 0.30 · backup_speed 0.05) 뒤 다시 계획
    e.p_rec = c.rec_backup_steps;
    *force = 1;
  } else if (e.p_blk > 0) {   // 다시 계획 (BT ComputePathToPose·FollowPath RecoveryNode)
    *force = 1;
  }
  if (tgt < 0 || !path_blacklisted(c, m, e, tgt)) return k;
  for (int j = 0; j < act_slots(b); ++j)   // 포기한 목표 둘레: 포기 안 한 후보 중 번호가 가장 작은 것
    if (C[j] >= 0 && !path_blacklisted(c, m, e, C[j])) return j;
  return -1;
}

DEV void path_set_none(EnvState& e) {
  e.p_n = 0;
  e.p_i = 0;
  e.p_trunc = 0;
}

// 지도 자세 칸에서 막힘(mdp::kOcc)까지 칸 가운데끼리 거리 (m), r 안에 없으면 r (기준선 obstacle_dist 와 같은 뜻 — 막힘 씨앗은 지도 막힘만)
DEV float path_obstacle_dist(const EnvBatch& b, const EnvState& e, int env, float r) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  int cx, cy;
  cell_of(c, m, e.x + e.ex, e.y + e.ey, &cx, &cy);
  const int R = (int)ceilf(r / c.res);
  int best = 1 << 30;
  for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
      const int u = cx + dx, v = cy + dy;
      if (u < 0 || v < 0 || u >= m.gw || v >= m.gh) continue;
      if (mdp::cell_class(L[v * c.size + u]) != mdp::kOcc && !cam_lethal(b, env, u, v)) continue;   // 깊이 카메라 층도 (Nav2 costmap 은 모든 층 합)
      const int d2 = dx * dx + dy * dy;
      best = d2 < best ? d2 : best;
    }
  if (best == (1 << 30)) return r;
  const float d = sqrtf((float)best) * c.res;
  return d < r ? d : r;
}

// RPP 충돌 검사 (regulated_pure_pursuit_controller.cpp isCollisionImminent, 기준선 frontier_explorer.cpp collision_imminent 와 같은 식):
// 막힘 씨앗 칸 = 지도 막힘(kOcc) 또는 로봇에서 obstacle_max_range 안의 맞음 > 지나감 칸(로그 오즈 > 0 — Nav2 장애물 층이 스캔 끝점을 바로
// 막힘으로 두는 몫, 기준선 lethal_src 와 같음). 발자국 = nav2 local_costmap footprint + footprint_padding 둘레 점(반 칸 간격)
// unk = true: 모름 칸도 막힘으로 (물러나기 — 라이다가 뒤 140° 를 못 봐 뒤 벽이 지도에 없을 수 있어서)
DEV bool path_lethal(const EnvBatch& b, const EnvState& e, int env, int x, int y, int rcx, int rcy, bool unk = false) {
  const EnvCfg& c = b.cfg;
  const int16_t l = b.lmap[(size_t)env * c.size * c.size + (size_t)y * c.size + x];
  if (mdp::cell_class(l) == mdp::kOcc || (unk && l == mdp::kUnknownL)) return true;
  if (cam_lethal(b, env, x, y)) return true;   // 깊이 카메라 층 (거리 제한 없음 — 층이 이미 cam_obs_range 안 점만 표시)
  if (l == mdp::kUnknownL || l <= 0) return false;
  if (c.path_consistent && e.p_mode >= 2 && e.p_n > 0) return false;   // 느슨 계획(지도 막힘만)을 따라가는 중: 약한 막힘은 계획과 같게 안 봄 ("계획·충돌 검사 일치")
  const float dx = (float)(x - rcx) * c.res, dy = (float)(y - rcy) * c.res;
  return dx * dx + dy * dy <= c.path_obs_range * c.path_obs_range;
}
DEV bool path_fp_hits(const EnvBatch& b, const EnvState& e, int env, float x, float y, float th, int rcx, int rcy, bool unk = false, bool body = false) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  float s, co;
  dm::sincosf_d(th, &s, &co);
  const float pad = body ? 0.f : c.path_fp_pad;   // body: 몸통만 (패딩 없음 — "계획·충돌 검사 일치" 빠져나오기)
  const float hl = c.half_len + pad, hw = c.half_wid + pad;
  const float cx[4] = {hl, -hl, -hl, hl}, cy[4] = {hw, hw, -hw, -hw};
  for (int k = 0; k < 4; ++k) {
    const float ax = cx[k], ay = cy[k], bx = cx[(k + 1) & 3], by = cy[(k + 1) & 3];
    const float len = sqrtf((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    int n = (int)ceilf(len / (0.5f * c.res));
    n = n < 1 ? 1 : n;
    for (int q = 0; q <= n; ++q) {
      const float u = ax + (bx - ax) * (float)q / (float)n, v = ay + (by - ay) * (float)q / (float)n;
      int qx, qy;
      if (cell_of(c, m, (x + co * u) - s * v, (y + s * u) + co * v, &qx, &qy) < 0) return true;
      if (path_lethal(b, e, env, qx, qy, rcx, rcy, unk)) return true;
    }
  }
  return false;
}
// check_now = false: 지금 자세는 안 봄 (물러나기 — 이미 발자국이 막힘에 닿아 있어도 빠져나오게)
DEV bool path_imminent(const EnvBatch& b, const EnvState& e, int env, float v, float w, float carrot_dist, bool check_now = true, bool unk = false) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  int rcx, rcy;
  cell_of(c, m, px, py, &rcx, &rcy);
  // 지금 자세가 이미 패딩에 걸림: 옛 동작(Nav2 그대로)은 모든 명령 막힘 → 추종 중 갇힘의 41~59 %. path_consistent 면 굴린 자세를 몸통으로만 검사 (빠져나오기)
  bool body = false;
  if (check_now && path_fp_hits(b, e, env, px, py, pth, rcx, rcy)) {
    if (!c.path_consistent) return true;
    body = true;
  }
  float dt;
  if (dm::absf(v) < 0.01f && dm::absf(w) > 0.01f) {
    const float rmax = sqrtf(c.half_len * c.half_len + c.half_wid * c.half_wid);
    dt = 2.f * sinf((c.res * 0.5f) / rmax) / dm::absf(w);
  } else if (dm::absf(v) >= 0.01f) {
    dt = c.res / dm::absf(v);
  } else {
    return false;
  }
  float x = px, y = py, th = pth;
  for (int i = 1; (float)i * dt < c.path_coll_time; ++i) {
    float s, co;
    dm::sincosf_d(th, &s, &co);
    x += dt * v * co;
    y += dt * v * s;
    th += dt * w;
    if (sqrtf((x - px) * (x - px) + (y - py) * (y - py)) > carrot_dist) break;
    if (path_fp_hits(b, e, env, x, y, th, rcx, rcy, unk, body)) return true;
  }
  return false;
}

// 추종기 한 번 (행동 주기 0.1 s, 스레드 하나): 웨이포인트 따라 (vc, ωc) — velocity_smoother 앞 명령. p_i(지금 구간)를 앞으로만 옮긴다
DEV void path_follow(const EnvBatch& b, EnvState& e, int env, float* vc_o, float* wc_o, float* carrot_o = nullptr) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  *vc_o = 0.f;
  *wc_o = 0.f;
  if (carrot_o) *carrot_o = 1e9f;   // 제자리 회전이면 당근 거리 무한 (기준선과 같음)
  if (e.p_n <= 0) {   // 목표 없음: 후보가 없을 때(판 시작·모두 닫힘)는 제자리 회전, 목표는 있는데 경로를 못 짰으면 섬
    if (e.goal_cell < 0) *wc_o = c.rot_w;
    return;
  }
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  const auto wx = [&](int i) { return m.ox + ((float)e.wp[2 * i] + 0.5f) * c.res; };
  const auto wy = [&](int i) { return m.oy + ((float)e.wp[2 * i + 1] + 0.5f) * c.res; };
  // 가장 가까운 점 (지금 구간부터 앞으로만)
  int bi = e.p_i < 0 ? 0 : e.p_i;
  float qx = wx(bi), qy = wy(bi);
  if (e.p_n >= 2) {
    float bd = 3e38f;
    for (int i = bi; i + 1 < e.p_n; ++i) {
      const float ax = wx(i), ay = wy(i), sx = wx(i + 1) - ax, sy = wy(i + 1) - ay;
      const float l2 = sx * sx + sy * sy;
      float t = l2 > 0.f ? ((px - ax) * sx + (py - ay) * sy) / l2 : 0.f;
      t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
      const float x = ax + t * sx, y = ay + t * sy;
      const float d2 = (x - px) * (x - px) + (y - py) * (y - py);
      if (d2 < bd) { bd = d2; bi = i; qx = x; qy = y; }
    }
  } else {
    bi = 0;
  }
  e.p_i = bi;
  // 당근: 가장 가까운 점에서 경로 따라 lookahead
  float rem = c.lookahead, cx = qx, cy = qy;
  {
    int i = bi;
    float ux = qx, uy = qy;
    bool done = false;
    while (i + 1 < e.p_n) {
      const float ex2 = wx(i + 1), ey2 = wy(i + 1);
      const float d = sqrtf((ex2 - ux) * (ex2 - ux) + (ey2 - uy) * (ey2 - uy));
      if (d >= rem && d > 0.f) {
        cx = ux + (ex2 - ux) * (rem / d);
        cy = uy + (ey2 - uy) * (rem / d);
        done = true;
        break;
      }
      rem -= d;
      ux = ex2;
      uy = ey2;
      ++i;
    }
    if (!done) { cx = wx(e.p_n - 1); cy = wy(e.p_n - 1); }
  }
  const float gx = wx(e.p_n - 1), gy = wy(e.p_n - 1);
  const float d_goal = sqrtf((gx - px) * (gx - px) + (gy - py) * (gy - py));
  const float d_car = sqrtf((cx - px) * (cx - px) + (cy - py) * (cy - py));
  if (carrot_o) *carrot_o = d_car;
  // 도착: 다음 결정까지 제자리 회전 — 라이다가 뒤 140° 를 못 봐서(sensor_model.yaml fov 220°) 목표 덩어리가 로봇 뒤·발밑에 남는 일이 많다
  // (2026-11-08 규칙 정책 시험: 서 있기만 하면 0 번 후보가 거리 0 인 채로 안 풀려 128 판 모두 진행 없음 실패)
  if (d_goal <= c.goal_tol) {   // (모름 칸까지 막힘으로 보면 판 시작의 뒤쪽 모름 때문에 영영 안 돌아 128 판 모두 정체 — 지도 막힘만 봄)
    if (!path_imminent(b, e, env, 0.f, c.rot_w, 1e9f)) *wc_o = c.rot_w;
    return;
  }
  const float err = dm::wrap_pi(dm::atan2f_d(cy - py, cx - px) - pth);
  if (dm::absf(err) > c.rot_min_angle) {
    *wc_o = err > 0.f ? c.rot_w : -c.rot_w;
    return;
  }
  float vc = c.path_v;
  const float ld = dm::maxf(c.min_lookahead, dm::minf(c.lookahead, d_car));
  float se, ce;
  dm::sincosf_d(err, &se, &ce);
  const float curv = 2.f * se / ld;
  float v_curv = vc, v_cost = vc;
  {
    const float ac = dm::absf(curv);
    const float radius = ac > 1e-6f ? 1.f / ac : 1e6f;
    if (radius < c.reg_min_radius) v_curv = vc * radius / c.reg_min_radius;
  }
  if (c.path_cost_reg) {
    const float r_eff = dm::minf(c.path_cs_dist, c.path_r_out);
    const float d = path_obstacle_dist(b, e, env, r_eff);
    if (d < r_eff) v_cost = vc * c.path_cs_gain * d / c.path_cs_dist;
  }
  vc = dm::maxf(dm::minf(v_curv, v_cost), c.reg_min_speed);
  // 경로 끝 감속 (RPP approach scaling). 그래프 모드 graph_pass: 경로 끝 = 다음 노드일 뿐 서는 곳이 아니라 감속하지 않음 (닿으면 바로 다음 결정 — graph_env.h gdec_step)
  if (d_goal < c.approach_dist && !(b.gdec && c.graph_pass)) vc = dm::minf(vc, dm::maxf(c.path_v * d_goal / c.approach_dist, c.approach_min_v));
  float wc = vc * curv;
  wc = dm::maxf(-c.w_max, dm::minf(c.w_max, wc));
  *vc_o = vc;
  *wc_o = wc;
}


// 물러나도 되나: 지금 방향 그대로 뒤로 d·k/4 (k = 1…4) 자세의 발자국이 막힘·모름 칸에 안 닿음 (기준선 recover 의 BackUp 앞 검사 꼴 — 뒤는 라이다가 못 봐서 모름도 막힘)
DEV bool path_backup_ok(const EnvBatch& b, const EnvState& e, int env, float d) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const float px = e.x + e.ex, py = e.y + e.ey, pth = e.th + e.eth;
  int rcx, rcy;
  cell_of(c, m, px, py, &rcx, &rcy);
  float s, co;
  dm::sincosf_d(pth, &s, &co);
  // 지금 자세가 이미 패딩에 걸리면 (path_consistent) 몸통으로만 — path_imminent 와 같은 빠져나오기
  const bool body = c.path_consistent && path_fp_hits(b, e, env, px, py, pth, rcx, rcy, true);
  for (int k = 1; k <= 4; ++k) {
    const float t = d * (float)k * 0.25f;
    if (path_fp_hits(b, e, env, px - co * t, py - s * t, pth, rcx, rcy, true, body)) return false;
  }
  return true;
}
