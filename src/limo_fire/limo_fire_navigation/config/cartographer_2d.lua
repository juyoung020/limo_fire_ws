-- WeGo wego/config/limo_lds_2d.lua 그대로 (2026-10-26 실물에서 지도 만들기 확인)
-- 바꾼 것 (2026-10-28, 같은 주행 녹화를 다시 돌려 비교):
--   motion_filter.max_angle 0.1° → 0.5°: 노드가 덜 생겨 루프 클로저 탐색(CPU 의 77 %)이 줄어 CPU -20 %,
--     궤적 차이 정렬 뒤 RMS 1.5~1.8 cm (같은 설정끼리 0.6 cm, 지도 한 칸 5 cm). 1° 는 -28 % · 1.4~1.5 cm
--   pose_publish_period 5 ms → 20 ms: map→odom TF 가 /tf 의 가장 많은 몫(초당 45)이라 모든 TF 구독자가 덜 깬다
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 2e-2,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.min_range = 0.1
TRAJECTORY_BUILDER_2D.max_range = 8
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 0.1
TRAJECTORY_BUILDER_2D.use_imu_data = true
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true 
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.5)

POSE_GRAPH.constraint_builder.min_score = 0.7
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7

-- POSE_GRAPH.optimize_every_n_nodes = 30

return options