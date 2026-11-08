// 광선 결과 뒤 단계 (깊이 픽셀·라이다 빔) — GPU 의 두 광선 경로가 같이 쓴다:
//   소프트웨어 순회(src/env_kernel.cu: 광선 묶음 커널 → 이 함수를 부르는 커널)
//   RT 코어(src/optix_rays.cu: raygen 은 광선을 만들어 쏘기만, 결과 뒤는 위와 같은 커널이 이 함수로)
// 값은 CPU 참조판(obs_depth → depth_pixel_real, scan_beam → lidar_beam_real)과 같은 단계 함수(sensor_model.h·lidar_real.h)라 비트 같다.
// 옆 광선이 있어야 하는 픽셀·빔은 가운데 결과를 dep_z·dep_tri / beam_t·beam_tri 에 두고 큐(dep_jobs / beam_jobs)에 넣어 두 번째 패스에서 마무리한다.
#pragma once
#include "env.h"

namespace sim {

// ---------- 깊이 픽셀 (px = env · DEPTH_SIZE + i) ----------

// 픽셀의 가운데 광선 방향 (depth_dir). 카메라 자세는 마무리·리셋 커널이 환경마다 남긴 것 (Shared::cam = depth_camera)
DEV V3 depth_px_dir(const EnvBatch& b, int px) {
  const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
  const EnvCfg& c = b.cfg;
  float u, v;
  mdp::obsgen::depth_uv(c.depth, i, &u, &v);
  return depth_dir(c.depth.spec, b.scratch[env].cam, u, v);
}

// 가운데 광선 결과 (z, tri) 뒤 (tri = −1: 유효 영역 밖·못 맞음). d = 가운데 광선 방향 (depth_dir — 쏜 쪽이 이미 셈). 값이 정해지면 관측 칸에 쓰고 true,
// 옆 광선(왼쪽 u − edge_px 먼저, 모서리가 아니면 오른쪽 u + edge_px)이 답을 바꿀 수 있으면 false
DEV bool depth_px_center(const EnvBatch& b, int px, float z, int tri, V3 d) {
  const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
  uint8_t* out = mdp::obsgen::depth_out(b, env);
  if (tri == -1) {
    out[i] = 0;
    return true;
  }
  const EnvState& e = b.st[env];
  const DepthModel& M = b.cfg.depth;
  const SceneView sv = with_people(b.maps[e.map].scene, e.people);
  uint64_t seed = mdp::obsgen::depth_seed(e, i);
  const float u_hole = dm::rand01(seed);
  float p_edge, p_flat;
  depth_hole_probs(sv, M, d, z, tri, e.depth_floor, &p_edge, &p_flat);
  const int hole = depth_hole_decided(u_hole, p_edge, p_flat);
  if (hole < 0) return false;
  out[i] = mdp::obsgen::depth_u8(M, hole ? 0.f : depth_value(M, z, seed));
  return true;
}

// 옆 광선 뒤 마무리: d = 가운데 광선 방향, side[0] 왼쪽, side[1] 오른쪽 (왼쪽이 모서리면 오른쪽은 안 읽음 — CPU depth_pixel_real 처럼 안 쏴도 됨)
DEV void depth_px_side(const EnvBatch& b, int px, float z, int tri, V3 d, const float* side) {
  const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
  const EnvState& e = b.st[env];
  const DepthModel& M = b.cfg.depth;
  const SceneView sv = with_people(b.maps[e.map].scene, e.people);
  uint64_t seed = mdp::obsgen::depth_seed(e, i);
  const float u_hole = dm::rand01(seed);
  float p_edge, p_flat;
  depth_hole_probs(sv, M, d, z, tri, e.depth_floor, &p_edge, &p_flat);
  const bool edge = depth_edge(z, side[0]) || depth_edge(z, side[1]);
  const int hole = u_hole < (edge ? p_edge : p_flat) ? 1 : 0;
  mdp::obsgen::depth_out(b, env)[i] = mdp::obsgen::depth_u8(M, hole ? 0.f : depth_value(M, z, seed));
}

// ---------- 라이다 빔 (env, j) — lidar_beam_real 을 광선 사이에서 나눈 것 ----------

// 이번 장이 있고 j < 빔 수인가
DEV bool beam_live(const EnvBatch& b, int env, int j) {
  const Shared& sh = b.scratch[env];
  return sh.pending && j < sh.n_beams;
}

// 가운데 광선 결과 (t, tri) 뒤: 빠짐이거나 옆 광선 없이 끝나면(지도 표시까지) true, 옆 광선 4 개(beam_side_dir k = 0..3)가 있어야 하면 false
DEV bool beam_center_done(const EnvBatch& b, int env, int j, float t, int tri) {
  Shared& sh = b.scratch[env];
  const EnvState& e = b.st[env];
  const LidarNoise& N = e.noise;
  const SceneView sv = with_people(b.maps[e.map].scene, e.people);
  BeamGeom geo;
  beam_center_ray(N, sh.scan, j, geo);
  beam_center_hit(sv, geo, t, tri);
  if (!geo.hit) return true;
  const BeamDraw w = beam_draw(sh.scan, j);
  Material mat;
  const int st = beam_mid(sv, N, sh.scan, geo, w, &mat);
  if (st == 0) return true;
  if (st == 2) return false;
  const float r = beam_finish(N, geo, w, mat, false);
  BeamRay ray;
  beam_ray_out(geo, &ray);
  scan_beam_mark(b, e, env, sh, j, r, ray);
  return true;
}

// 옆 광선 4 개 거리 뒤 마무리 (beam_center_done 이 false 였던 빔)
DEV void beam_side_done(const EnvBatch& b, int env, int j, float t, int tri, const float side[4]) {
  Shared& sh = b.scratch[env];
  const EnvState& e = b.st[env];
  const LidarNoise& N = e.noise;
  const SceneView sv = with_people(b.maps[e.map].scene, e.people);
  BeamGeom geo;
  beam_center_ray(N, sh.scan, j, geo);
  beam_center_hit(sv, geo, t, tri);
  const BeamDraw w = beam_draw(sh.scan, j);
  Material mat;
  beam_mid(sv, N, sh.scan, geo, w, &mat);   // 재질 (가운데 패스에서 2 였음)
  beam_edges_from(N, geo, side);
  const float r = beam_finish(N, geo, w, mat, true);
  BeamRay ray;
  beam_ray_out(geo, &ray);
  scan_beam_mark(b, e, env, sh, j, r, ray);
}

}  // namespace sim
