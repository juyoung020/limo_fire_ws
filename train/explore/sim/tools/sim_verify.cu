// V0·V1: 같은 장면·자세에서 GPU 와 CPU 참조판의 라이다·깊이가 비트까지 같은지. 젯슨(sm_87)·데스크톱(sm_120)에서 같은 결과여야 한다.
// sim_verify <장면.scene> [자세 수=256] [--negative]
//   --negative: GPU 쪽 자세 하나를 1 mm 옮겨 일부러 어긋나게 한다 (검사가 틀림을 잡는지)
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "scene_host.h"
#include "sensors_api.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

// 세로축 yaw + 작은 pitch(카메라 숙임) 회전
sim::Pose make_pose(float x, float y, float z, float yaw, float pitch) {
  const double cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
  sim::Pose T;
  const double r[9] = {cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, 0, cp};   // Rz(yaw) · Ry(pitch)
  for (int i = 0; i < 9; ++i) T.r[i] = (float)r[i];
  T.p = sim::V3{x, y, z};
  return T;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "sim_verify <장면.scene> [자세 수] [--negative]\n"); return 2; }
  const sim::Scene s = sim::load_scene(argv[1]);
  int n = 256;
  bool negative = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--negative")) negative = true;
    else n = std::stoi(argv[i]);
  }
  const auto& root = s.nodes[0];
  uint64_t rng = 12345;
  std::vector<sim::Pose> lidar_poses(n), cam_poses(n);
  for (int i = 0; i < n; ++i) {
    const float x = dm::rand_range(rng, root.lo[0] + 0.1f * (root.hi[0] - root.lo[0]), root.hi[0] - 0.1f * (root.hi[0] - root.lo[0]));
    const float y = dm::rand_range(rng, root.lo[1] + 0.1f * (root.hi[1] - root.lo[1]), root.hi[1] - 0.1f * (root.hi[1] - root.lo[1]));
    const float yaw = dm::rand_range(rng, -3.14159f, 3.14159f);
    lidar_poses[i] = make_pose(x, y, 0.111f, yaw, 0.f);
    cam_poses[i] = make_pose(x, y, 0.175f, yaw, dm::rand_range(rng, -0.1f, 0.1f));
  }
  const sim::LidarSpec L;
  const sim::DepthSpec D;

  std::vector<sim::Node> nodes = s.nodes;
  sim::SceneView dev{upload(nodes), upload(s.tris), upload(s.obj)};
  std::vector<sim::Pose> gpu_lidar = lidar_poses, gpu_cam = cam_poses;
  if (negative) { gpu_lidar[n / 2].p.x += 1e-3f; gpu_cam[n / 2].p.x += 1e-3f; }
  sim::Pose* d_lp = upload(gpu_lidar);
  sim::Pose* d_cp = upload(gpu_cam);
  float* d_l;
  uint16_t* d_d;
  CK(cudaMalloc(&d_l, sizeof(float) * n * L.n));
  CK(cudaMalloc(&d_d, sizeof(uint16_t) * (size_t)n * D.w * D.h));

  sim::lidar_gpu(dev, L, d_lp, n, d_l);
  sim::depth_gpu(dev, D, d_cp, n, d_d);
  CK(cudaDeviceSynchronize());
  const int reps = 5;
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) { sim::lidar_gpu(dev, L, d_lp, n, d_l); sim::depth_gpu(dev, D, d_cp, n, d_d); }
  CK(cudaDeviceSynchronize());
  const double gsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;

  std::vector<float> gl(n * L.n), cl(n * L.n);
  std::vector<uint16_t> gd((size_t)n * D.w * D.h), cd(gd.size());
  CK(cudaMemcpy(gl.data(), d_l, gl.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gd.data(), d_d, gd.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  const auto t1 = std::chrono::steady_clock::now();
  sim::lidar_cpu(s.view(), L, lidar_poses.data(), n, cl.data());
  sim::depth_cpu(s.view(), D, cam_poses.data(), n, cd.data());
  const double csec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

  size_t lbad = 0, dbad = 0, lhit = 0, dhit = 0;
  for (size_t i = 0; i < gl.size(); ++i) {
    uint32_t a, b;
    std::memcpy(&a, &gl[i], 4);
    std::memcpy(&b, &cl[i], 4);
    lbad += a != b;
    lhit += cl[i] > 0;
  }
  for (size_t i = 0; i < gd.size(); ++i) { dbad += gd[i] != cd[i]; dhit += cd[i] > 0; }
  int dev_id;
  cudaDeviceProp prop;
  cudaGetDevice(&dev_id);
  cudaGetDeviceProperties(&prop, dev_id);
  std::printf("GPU %s (sm_%d%d), 장면 삼각형 %zu, 자세 %d%s\n", prop.name, prop.major, prop.minor, s.tris.size(), n,
              negative ? " (일부러 어긋나게)" : "");
  std::printf("라이다 빔 %zu 개 (맞음 %.1f %%): 비트 다름 %zu\n", gl.size(), 100.0 * lhit / gl.size(), lbad);
  std::printf("깊이 픽셀 %zu 개 (값 있음 %.1f %%): 다름 %zu\n", gd.size(), 100.0 * dhit / gd.size(), dbad);
  const double rays = (double)n * (L.n + D.w * D.h);
  std::printf("GPU %.2f ms (광선 %.1f M/초), CPU %.2f s\n", gsec * 1e3, rays / gsec / 1e6, csec);
  const bool same = lbad == 0 && dbad == 0;
  std::printf("%s\n", same ? "V0 통과: GPU == CPU (비트 같음)" : "다름");
  return negative ? (same ? 1 : 0) : (same ? 0 : 1);
}
