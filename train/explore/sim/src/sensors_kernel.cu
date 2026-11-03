#include "sensors_api.h"

namespace sim {
namespace {

__global__ void lidar_kernel(SceneView s, LidarSpec L, const Pose* poses, int n_poses, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_poses * L.n) return;
  out[i] = lidar_beam(s, L, poses[i / L.n], i % L.n);
}

__global__ void depth_kernel(SceneView s, DepthSpec D, const Pose* poses, int n_poses, uint16_t* out) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  const int p = blockIdx.z;
  if (u >= D.w || v >= D.h || p >= n_poses) return;
  out[(size_t)p * D.w * D.h + (size_t)v * D.w + u] = depth_pixel(s, D, poses[p], u, v);
}

__global__ void lidar_real_kernel(SceneView s, const LidarNoise* noise, const ScanIn* scans, int n_scans, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_scans * kMaxBeams) return;
  const ScanIn& in = scans[i / kMaxBeams];
  out[i] = lidar_beam_real(s, noise[in.noise], in, i % kMaxBeams);
}

}  // namespace

void lidar_gpu(const SceneView& s, const LidarSpec& L, const Pose* poses, int n_poses, float* out, cudaStream_t st) {
  const int n = n_poses * L.n;
  lidar_kernel<<<(n + 127) / 128, 128, 0, st>>>(s, L, poses, n_poses, out);
}

void depth_gpu(const SceneView& s, const DepthSpec& D, const Pose* poses, int n_poses, uint16_t* out, cudaStream_t st) {
  const dim3 b(16, 8), g((D.w + 15) / 16, (D.h + 7) / 8, n_poses);
  depth_kernel<<<g, b, 0, st>>>(s, D, poses, n_poses, out);
}

void lidar_real_gpu(const SceneView& s, const LidarNoise* noise, const ScanIn* scans, int n_scans, float* out,
                    cudaStream_t st) {
  const int n = n_scans * kMaxBeams;
  lidar_real_kernel<<<(n + 127) / 128, 128, 0, st>>>(s, noise, scans, n_scans, out);
}

}  // namespace sim
