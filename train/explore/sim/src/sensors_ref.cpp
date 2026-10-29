#include "sensors_api.h"

namespace sim {

void lidar_cpu(const SceneView& s, const LidarSpec& L, const Pose* poses, int n_poses, float* out) {
  for (int p = 0; p < n_poses; ++p)
    for (int j = 0; j < L.n; ++j) out[p * L.n + j] = lidar_beam(s, L, poses[p], j);
}

void depth_cpu(const SceneView& s, const DepthSpec& D, const Pose* poses, int n_poses, uint16_t* out) {
  for (int p = 0; p < n_poses; ++p)
    for (int v = 0; v < D.h; ++v)
      for (int u = 0; u < D.w; ++u) out[(size_t)p * D.w * D.h + (size_t)v * D.w + u] = depth_pixel(s, D, poses[p], u, v);
}

}  // namespace sim
