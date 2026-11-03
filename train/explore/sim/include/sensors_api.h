// 센서 묶음 계산: 자세 여러 개의 라이다·깊이를 한 번에. GPU 커널(src/sensors_kernel.cu)과 CPU 참조판(src/sensors_ref.cpp)이
// 같은 함수(sensors.h)를 불러서 비트까지 같은 값을 낸다 (tools/sim_verify)
#pragma once
#include <cuda_runtime.h>

#include "lidar_real.h"
#include "sensors.h"

namespace sim {

// out[pose · L.n + j] (m, 못 맞음 0)
void lidar_gpu(const SceneView& dev_scene, const LidarSpec& L, const Pose* dev_poses, int n_poses, float* dev_out,
               cudaStream_t st = 0);
void lidar_cpu(const SceneView& scene, const LidarSpec& L, const Pose* poses, int n_poses, float* out);

// out[pose · w · h + v · w + u] (mm, 0 = 없음)
void depth_gpu(const SceneView& dev_scene, const DepthSpec& D, const Pose* dev_poses, int n_poses, uint16_t* dev_out,
               cudaStream_t st = 0);
void depth_cpu(const SceneView& scene, const DepthSpec& D, const Pose* poses, int n_poses, uint16_t* out);

// 실물 라이다(lidar_real.h): out[scan · kMaxBeams + j] (m, 0 = 없음, j ≥ scan_beams 도 0). noise[scans[i].noise]
void lidar_real_gpu(const SceneView& dev_scene, const LidarNoise* dev_noise, const ScanIn* dev_scans, int n_scans,
                    float* dev_out, cudaStream_t st = 0);
void lidar_real_cpu(const SceneView& scene, const LidarNoise* noise, const ScanIn* scans, int n_scans, float* out);

}  // namespace sim
