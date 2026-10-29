// 센서: 라이다·깊이 카메라 광선(장면 좌표). 값 규칙은 교수님 Webots LIMO(LimoFourDiff.proto + limo_sensors.py)와 같다.
//   라이다: Webots Lidar(비회전) 빔 i 각도 = fov/2 − (i + ½)·fov/N (WbLidar.cpp). limo_sensors 가 순서를 뒤집어 ROS 각도 순으로 낸다 →
//           ROS 순서 j 의 각도 = −fov/2 + (j + ½)·fov/N. 거리 = 유클리드. [min, max) 밖은 0.
//   깊이:   Webots RangeFinder 644×524 핀홀(가로 fov, 주점 = 가운데)을 (위 44, 왼쪽 4) 잘라 640×480. 값 = 광축 방향 z.
//           0.3~3.0 m 밖 · 유효 영역(행 22~399, 열 0~627) 밖은 0, mm 로 반올림.
// 센서 좌표계는 Webots 장치와 같다: x 앞, y 왼쪽, z 위 (FLU). 센서 자세(Pose)는 세계 좌표.
// 오차(잡음·빠짐·지연)는 따로(noise.h, 다음 단계) — 여기는 오차 없는 참값.
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
  int render_w = 644, render_h = 524;
  float fov = 1.161085f;   // 가로
  int crop_top = 44, crop_left = 4;
  int w = 640, h = 480;
  float min_z = 0.3f, max_z = 3.0f;      // limo_sensors DEPTH_MIN/MAX (Webots RangeFinder maxRange 3.0)
  int valid_row0 = 22, valid_row1 = 400;  // 유효 영역 [row0, row1) × [col0, col1)
  int valid_col0 = 0, valid_col1 = 628;
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
DEV uint16_t depth_pixel(const SceneView& s, const DepthSpec& D, const Pose& T, int u, int v) {
  if (v < D.valid_row0 || v >= D.valid_row1 || u < D.valid_col0 || u >= D.valid_col1) return 0;
  float sh, ch;
  dm::sincosf_d(0.5f * D.fov, &sh, &ch);
  const float f = 0.5f * (float)D.render_w * ch / sh;   // 초점 거리(픽셀) = (W/2) / tan(fov/2)
  const float px = (float)(u + D.crop_left) + 0.5f - 0.5f * (float)D.render_w;   // 오른쪽 +
  const float py = (float)(v + D.crop_top) + 0.5f - 0.5f * (float)D.render_h;    // 아래 +
  // 광축 x = 1 인 방향: 오른쪽은 −y, 아래는 −z. raycast 가 d 단위 거리를 내므로 그 값이 곧 광축 z
  const V3 d = rotate(T, V3{1.f, -px / f, -py / f});
  const float z = raycast(s, T.p, d, D.max_z + 1.f, nullptr);
  if (!(z >= D.min_z && z <= D.max_z)) return 0;
  const float mm = z * 1000.f + 0.5f;
  return (uint16_t)(int)mm;
}

}  // namespace sim
