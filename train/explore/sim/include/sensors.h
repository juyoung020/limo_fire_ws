// 센서: 라이다·깊이 카메라 광선(장면 좌표). 값 규칙은 교수님 Webots LIMO(LimoFourDiff.proto + limo_sensors.py)와 같다.
//   라이다: Webots Lidar(비회전) 빔 i 각도 = fov/2 − (i + ½)·fov/N (WbLidar.cpp). limo_sensors 가 순서를 뒤집어 ROS 각도 순으로 낸다 →
//           ROS 순서 j 의 각도 = −fov/2 + (j + ½)·fov/N. 거리 = 유클리드. [min, max) 밖은 0.
//   깊이:   핀홀 내부값(fx, fy, cx, cy — ROS camera_info 와 같은 픽셀 가운데 기준). 값 = 광축 방향 z. 범위 밖 · 유효 영역 밖은 0, mm 반올림.
//           기본은 실물 Orbbec Dabai DC1 (정렬 안 한 640 × 400, camera_info fx = fy = 478.02, cx 323.44, cy 199.34 → 67.6° × 45.4°).
//           교수님 Webots(RangeFinder 644 × 524 를 위 44 · 왼쪽 4 잘라 640 × 480, 가로 fov 1.161085)는
//           fx = fy = 322 / tan(0.5805425) ≈ 491.2, cx 317.5, cy 217.5, 유효 행 22~399 · 열 0~627 로 같은 식이 된다.
// 센서 좌표계는 Webots 장치와 같다: x 앞, y 왼쪽, z 위 (FLU). 센서 자세(Pose)는 세계 좌표.
// 여기는 오차 없는 참값(교수님 시뮬과 같은 값인지 V2 로 확인). 실물 라이다 오차·빔 격자·스캔 중 움직임은 lidar_real.h.
#pragma once
#include <cstdint>

#include "scene.h"

namespace sim {

struct LidarSpec {
  int n = 248;
  float fov = 3.839724f;   // 220°
  float min_range = 0.1f;
  float max_range = 12.0f;
};

struct DepthSpec {
  int w = 640, h = 400;
  float fx = 478.02396f, fy = 478.02396f, cx = 323.43732f, cy = 199.33777f;
  float min_z = 0.3f, max_z = 3.0f;      // Dabai DC1 0.3~3 m
  int valid_row0 = 0, valid_row1 = 400;  // 유효 영역 [row0, row1) × [col0, col1)
  int valid_col0 = 0, valid_col1 = 640;
};

// ROS 순서 j 번째 빔의 거리(m). 못 맞거나 범위 밖이면 0
DEV float lidar_beam(const SceneView& s, const LidarSpec& L, const Pose& T, int j) {
  const float step = L.fov / (float)L.n;
  const float a = -0.5f * L.fov + ((float)j + 0.5f) * step;
  float sa, ca;
  dm::sincosf_d(a, &sa, &ca);
  const V3 d = rotate(T, V3{ca, sa, 0.f});
  const float r = raycast(s, T.p, d, L.max_range, nullptr);
  return (r >= L.min_range && r < L.max_range) ? r : 0.f;
}

// 깊이 영상 (u, v) 픽셀(잘라낸 뒤 좌표)의 mm. 유효 영역·범위 밖은 0
// 깊이 영상 (u, v) 픽셀 광선 방향 (센서 좌표 FLU, 광축 x = 1 — raycast 가 d 단위 거리를 내므로 그 값이 곧 광축 z)
DEV V3 depth_ray(const DepthSpec& D, float u, float v) { return V3{1.f, -(u - D.cx) / D.fx, -(v - D.cy) / D.fy}; }

// 깊이 영상 (u, v) 픽셀의 mm (오차 없는 참값). 유효 영역·범위 밖은 0
DEV uint16_t depth_pixel(const SceneView& s, const DepthSpec& D, const Pose& T, int u, int v) {
  if (v < D.valid_row0 || v >= D.valid_row1 || u < D.valid_col0 || u >= D.valid_col1) return 0;
  const V3 d = rotate(T, depth_ray(D, (float)u, (float)v));
  const float z = raycast(s, T.p, d, D.max_z + 1.f, nullptr);
  if (!(z >= D.min_z && z <= D.max_z)) return 0;
  const float mm = z * 1000.f + 0.5f;
  return (uint16_t)(int)mm;
}

}  // namespace sim
