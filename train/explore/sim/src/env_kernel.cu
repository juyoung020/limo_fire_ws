// 환경 한 걸음 GPU: 단계 순서·값은 src/env_ref.cpp(CPU 참조판)와 같다 (비트 동일 — tools/env_verify). 일을 성격대로 커널로 나눈다:
//   1. 물리 (환경 하나 = 스레드 하나): 행동 → 물리 걸음을 다음 라이다 장까지 → 장 준비 (scan_begin)
//   2. 빔 (빔 하나 = 스레드 하나): 재고 지도 표시·각도 칸 (scan_beam — 장치 메모리 원자 연산, 정수 min/or 라 순서 무관)
//   3. 지도 갱신 (환경 하나 = 블록 하나): scan_apply + 정수 합 → 1 로 돌아가 다음 장 (목표점 행동은 한 행동에 장 여럿)
//   4. 마무리 (환경 하나 = 블록 하나): 남은 물리 → 보상·끝 → 라이다 링 → 끝난 판 리셋 → 관측 (깊이 빼고)
//   5. 깊이 (픽셀 하나 = 스레드 하나)
// 커널 사이 상태는 b.st(EnvState)와 b.scratch(Shared, 환경마다)에 둔다.
#include "env_api.h"
#ifdef SIM_OPTIX
#include "optix_rt.h"
#endif

namespace sim {
namespace {

constexpr int kBeamThreads = 64;
constexpr int kPhysThreads = 64;
constexpr int kDepthThreads = 128;

__device__ void reset_env(const EnvBatch& b, EnvState& e, int env, int tid) {
  reset_map(b, env, tid);
  if (tid == 0) reset_state(b, e);
  __syncthreads();
}

// 관측: 칸마다(모든 스레드) → 합치기·숫자(스레드 0)
__device__ void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, int tid, const mdp::obsgen::FarDirty& fd) {
  if (tid == 0) sh.fr_min = mdp::kNoFrontier;
  __syncthreads();
  mdp::obsgen::obs_cells(b, e, env, sh, tid, fd);
  __syncthreads();
  if (tid == 0) mdp::obsgen::obs_finish(b, e, env, sh);
  __syncthreads();
}

// 물리 걸음을 k_next 부터 다음 라이다 장(그 장 준비까지) 또는 행동 끝까지. 장이 있으면 pending = 1
__device__ void physics_until_scan(const EnvBatch& b, EnvState& e, Shared& sh) {
  const EnvCfg& c = b.cfg;
  sh.pending = 0;
  int k = sh.k_next;
  while (k < c.substeps) {
    if (!e.collided) physics_step(b, e);
    e.tick += 1;
    const bool scan = (sh.tick0 + k + 1) % c.lidar_every == 0;
    ++k;
    if (scan) {
      scan_begin(b, e, sh);
      sh.pending = 1;
      break;
    }
  }
  sh.k_next = k;
}

__global__ void __launch_bounds__(kPhysThreads) env_phys_kernel(EnvBatch b, const float* actions, int first) {
  const int env = blockIdx.x * kPhysThreads + threadIdx.x;
  if (env >= b.n_env) return;
  EnvState& e = b.st[env];
  Shared& sh = b.scratch[env];
  if (first) {
    e.collided = 0;
    apply_action(b, e, env, actions + mdp::act::DIM * env);
    sh.new_wall = sh.new_floor = sh.d_known = sh.n_new = 0;
    sh.dx0 = sh.dy0 = 1 << 30;
    sh.dx1 = sh.dy1 = -1;
    sh.scanned = 0;
    sh.k_next = 0;
    sh.tick0 = e.tick;
  }
  physics_until_scan(b, e, sh);
}

__device__ void block_sum4(int v[4], int tid) {
  __shared__ int acc[4];
  if (tid < 4) acc[tid] = 0;
  __syncthreads();
  for (int q = 0; q < 4; ++q) {
    const int s = __reduce_add_sync(0xffffffffu, v[q]);
    if ((tid & 31) == 0) atomicAdd(&acc[q], s);
  }
  __syncthreads();
  for (int q = 0; q < 4; ++q) v[q] = acc[q];
}

__global__ void __launch_bounds__(kThreads) env_apply_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  Shared& sh = b.scratch[env];
  if (!sh.pending) return;   // 블록 전체가 같은 값
  EnvState& e = b.st[env];
  const ApplyCount n = scan_apply(b, e, env, sh, tid, kThreads);
  int v[4] = {n.wall, n.floor, n.known, n.fresh};
  block_sum4(v, tid);
  if (tid == 0) {
    sh.new_wall += v[0];
    sh.new_floor += v[1];
    sh.d_known += v[2];
    sh.n_new += v[3];
    known_box(b, e, sh);
    if (sh.bx1 >= 0) {   // 먼 지도 다시 그릴 곳
      sh.dx0 = min(sh.dx0, sh.bx0);
      sh.dy0 = min(sh.dy0, sh.by0);
      sh.dx1 = max(sh.dx1, sh.bx1);
      sh.dy1 = max(sh.dy1, sh.by1);
    }
    sh.pending = 0;
  }
}

__global__ void __launch_bounds__(kThreads) env_finish_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  __shared__ Shared sh;   // 관측용 (프론티어 합)
  Shared& g = b.scratch[env];
  if (tid == 0) {
    e = b.st[env];
    // 마지막 장 뒤 남은 물리 걸음 (장 수만큼 1~3 을 돌렸으므로 여기선 장이 없다)
    while (g.k_next < b.cfg.substeps) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
      g.k_next += 1;
    }
    finish_step(b, e, env, g.new_wall, g.new_floor, g.d_known, g.n_new);
    {   // 지나간 곳 표시(finish_step, 로봇 칸 ±2)도 먼 지도 다시 그릴 곳에
      int rx, ry;
      cell_of(b.cfg, b.maps[e.map], e.x + e.ex, e.y + e.ey, &rx, &ry);
      g.dx0 = min(g.dx0, rx - 2);
      g.dy0 = min(g.dy0, ry - 2);
      g.dx1 = max(g.dx1, rx + 2);
      g.dy1 = max(g.dy1, ry + 2);
    }
    b.reward[env] = e.reward;
    b.done[env] = e.done;
    if (g.scanned) e.lid_head = (e.lid_head + 1) % kLidarRing;
  }
  __syncthreads();
  if (g.scanned) push_bins(b, e, env, g, tid);
  __syncthreads();
  const bool was_done = e.done != 0;
  __syncthreads();
  if (was_done) reset_env(b, e, env, tid);
  __shared__ mdp::obsgen::FarDirty fd;
  if (tid == 0) {
    // 먼 지도: 판이 바뀌었거나(리셋·처음·밖에서 상태를 바꿈) 아니면 바뀐 칸 범위 둘레 0.6 m 만
    const bool same = g.far_ok && g.far_map == e.map && g.far_key[0] == e.x0 && g.far_key[1] == e.y0 && g.far_key[2] == e.th0;
    const MapDev& m = b.maps[e.map];
    const float res = b.cfg.res, pad = 0.6f;
    fd.all = (was_done || !same) ? 1 : 0;
    fd.x0 = (m.ox + (float)g.dx0 * res) - pad;
    fd.y0 = (m.oy + (float)g.dy0 * res) - pad;
    fd.x1 = (m.ox + (float)(g.dx1 + 1) * res) + pad;
    fd.y1 = (m.oy + (float)(g.dy1 + 1) * res) + pad;
    if (g.dx1 < g.dx0) fd.x1 = fd.x0 - 1.f;   // 바뀐 칸 없음
    g.far_ok = 1;
    g.far_map = e.map;
    g.far_key[0] = e.x0;
    g.far_key[1] = e.y0;
    g.far_key[2] = e.th0;
  }
  write_obs(b, e, env, sh, tid, fd);
  if (tid == 0) b.st[env] = e;
}

__global__ void env_reset_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  __shared__ Shared sh;
  if (tid == 0) e = b.st[env];
  __syncthreads();
  reset_env(b, e, env, tid);
  const mdp::obsgen::FarDirty all{1, 0.f, 0.f, 0.f, 0.f};
  write_obs(b, e, env, sh, tid, all);
  if (tid == 0) b.scratch[env].far_ok = 0;   // 다음 걸음은 다시 전부 (판 기록은 그때 남김)
  if (tid == 0) {
    b.st[env] = e;
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  }
}

// ---------- 광선 묶음: 스레드마다 광선 하나씩 순회(trav_step)하고, 끝난 스레드는 워프 단위로 다음 광선을 받는다 ----------
// 광선마다 걸음 순서는 raycast 와 같아 결과가 같고, 워프 안 광선 길이 차이로 노는 스레드가 없어진다.
// G: make(b, job, sv, o, d, tmax) → 쏠 광선이면 true (아니면 none(b, job)), store(b, job, t, tri)
constexpr int kRayThreads = 128;
constexpr int kRayChunk = 64, kRayRefill = 16, kRaySteps = 4;   // 워프가 한 번에 받는 광선 수, 이만큼 놀면 채움
enum : int { kCntDepNext = 0, kCntDepSide = 1, kCntDepSideNext = 2, kCntBeamNext = 3, kCntBeamSide = 4, kCntBeamSideNext = 5, kCounters = 16 };

template <class G>
__global__ void __launch_bounds__(kRayThreads) ray_kernel(EnvBatch b, G g, int n_static, const int* n_dev, int* next) {
  const int n = n_dev ? *n_dev : n_static;
  const unsigned lane = threadIdx.x & 31u;
  RayTrav r;
  TravStack k;
  SceneView sv;
  int job = 0;
  bool active = false;
  int pool = 0, pool_end = 0;   // 워프가 받아 둔 광선 번호 [pool, pool_end) — 워프 안 같은 값
  bool exhausted = false;
  while (true) {
    const unsigned need = __ballot_sync(0xffffffffu, !active);
    const int n_need = __popc(need);
    if (n_need >= kRayRefill || n_need == 32) {   // 노는 스레드가 어느 만큼 모이면 한꺼번에 채운다
      if (pool == pool_end && !exhausted) {
        int base = 0;
        if (lane == 0) base = atomicAdd(next, kRayChunk);
        base = __shfl_sync(0xffffffffu, base, 0);
        if (base >= n) exhausted = true;
        else { pool = base; pool_end = min(base + kRayChunk, n); }
      }
      const int avail = pool_end - pool;
      if (!active) {
        const int rank = __popc(need & ((1u << lane) - 1u));
        if (rank < avail) {
          job = pool + rank;
          V3 o, d;
          float tmax;
          if (g.make(b, job, sv, o, d, tmax)) {
            trav_init(r, k, sv, o, d, tmax);
            active = true;
          } else {
            g.none(b, job);
          }
        }
      }
      pool += min(avail, n_need);
    }
    if (!__any_sync(0xffffffffu, active)) {
      if (exhausted && pool == pool_end) break;
      continue;
    }
    for (int s = 0; s < kRaySteps && active; ++s)   // 받기 검사 사이 걸음 몇 개
      if (!trav_step(r, k, sv)) {
        int tri;
        const float t = trav_result(r, &tri);
        g.store(b, job, t, tri);
        active = false;
      }
  }
}

template <class G>
void launch_rays(const EnvBatch& b, G g, int n_static, const int* n_dev, int counter, cudaStream_t st) {
  static int blocks = 0;
  if (!blocks) {
    int dev, sms, per;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, ray_kernel<G>, kRayThreads, 0);
    blocks = sms * (per > 0 ? per : 1);
  }
  cudaMemsetAsync(b.counters + counter, 0, sizeof(int), st);
  ray_kernel<G><<<blocks, kRayThreads, 0, st>>>(b, g, n_static, n_dev, b.counters + counter);
}

#ifdef SIM_OPTIX
// RT 코어 경로 (include/optix_rt.h): 광선 묶음 G 를 준비(사람 원기둥까지 소프트웨어 — trav_init 과 같은 best) → OptiX → 결과 넣기.
// 준비 칸은 환경 · DEPTH_SIZE 개라 그보다 많은 묶음(깊이 옆 광선)은 여러 번에 나눠 쏜다. 안 쏘는 칸은 번호 칸에 kRtNone
constexpr int kRtNone = -2147483647 - 1;
template <class G>
__global__ void __launch_bounds__(kDepthThreads) rt_prep_kernel(EnvBatch b, G g, int off, int cap, int n_static, const int* n_dev) {
  const int i = blockIdx.x * kDepthThreads + threadIdx.x;
  if (i >= cap) return;
  const int job = off + i, n = n_dev ? *n_dev : n_static;
  float4* O = (float4*)b.rt_o;
  float4* D = (float4*)b.rt_d;
  SceneView sv;
  V3 o, d;
  float tmax;
  if (job >= n) return;   // 묶음 밖: 아무것도 안 씀 (OptiX·넣기도 n_dev 로 건너뜀)
  if (!g.make(b, job, sv, o, d, tmax)) {
    D[i] = make_float4(0.f, 0.f, 0.f, __int_as_float(-1));
    if constexpr (G::kDirect) g.none(b, job);
    else b.rt_tri0[i] = kRtNone;   // 안 쏘는 광선 (none)
    return;
  }
  float best = tmax;
  int best_tri = -1;
  for (int k = 0; k < sv.n_cyl; ++k) {   // trav_init 과 같은 사람 먼저
    const float t = cyl_hit(sv.cyl[k], o.x, o.y, o.z, d.x, d.y, d.z);
    if (t < best) { best = t; best_tri = -2 - k; }
  }
  O[i] = make_float4(o.x, o.y, o.z, best);
  D[i] = make_float4(d.x, d.y, d.z, __int_as_float(b.st[g.env_of(b, job)].map));
  b.rt_tri0[i] = best_tri;
}
template <class G>
__global__ void __launch_bounds__(kDepthThreads) rt_store_kernel(EnvBatch b, G g, int off, int cap, const int* n_dev) {
  const int i = blockIdx.x * kDepthThreads + threadIdx.x;
  if (i >= cap || (n_dev && off + i >= *n_dev)) return;
  const int tri = b.rt_tri0[i];
  if (tri == kRtNone) { g.none(b, off + i); return; }
  g.store(b, off + i, b.rt_t[i], tri);
}
template <class G>
void rt_rays(const EnvBatch& b, G g, int n_max, int n_static, const int* n_dev, cudaStream_t st) {
  const int cap = b.n_env * mdp::obs::DEPTH_SIZE;
  for (int off = 0; off < n_max; off += cap) {
    const int c = n_max - off < cap ? n_max - off : cap;
    const int blocks = (c + kDepthThreads - 1) / kDepthThreads;
    rt_prep_kernel<G><<<blocks, kDepthThreads, 0, st>>>(b, g, off, c, n_static, n_dev);
    RtParams p{};
    p.o = (const float4*)b.rt_o;
    p.d = (const float4*)b.rt_d;
    p.tri0 = b.rt_tri0;
    if constexpr (G::kDirect) {   // 한 번에 다 (cap = 환경 · DEPTH_SIZE)
      p.out_t = b.dep_z + off;
      p.out_tri = b.dep_tri + off;
    } else {
      p.out_t = b.rt_t;
      p.out_tri = b.rt_tri0;
    }
    p.n = c;
    p.n_dev = n_dev;
    p.off = off;
    ((OptixRt*)b.rt)->launch(p, st);
    if constexpr (!G::kDirect) rt_store_kernel<G><<<blocks, kDepthThreads, 0, st>>>(b, g, off, c, n_dev);
  }
}
#endif

// 깊이 가운데 광선: job = env · DEPTH_SIZE + 픽셀
struct DepthCenter {
  // RT 코어 경로: 묶음 번호 = 픽셀 번호라 결과를 dep_z·dep_tri 에 바로 (넣기 커널 없음)
  static constexpr bool kDirect = true;
  __device__ int env_of(const EnvBatch&, int job) const { return job / mdp::obs::DEPTH_SIZE; }
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int env = job / mdp::obs::DEPTH_SIZE, i = job % mdp::obs::DEPTH_SIZE;
    const EnvState& e = b.st[env];
    const EnvCfg& c = b.cfg;
    float u, v;
    mdp::obsgen::depth_uv(c.depth, i, &u, &v);
    if (!depth_valid(c.depth.spec, u, v)) return false;
    const Pose cam = mdp::obsgen::depth_camera(c, e);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = cam.p;
    d = depth_dir(c.depth.spec, cam, u, v);
    tmax = depth_tmax(c.depth.spec);
    return true;
  }
  __device__ void none(const EnvBatch& b, int job) const { b.dep_tri[job] = -1; }
  __device__ void store(const EnvBatch& b, int job, float t, int tri) const { b.dep_z[job] = t; b.dep_tri[job] = tri; }
};

// 깊이 옆 광선: job = 큐 칸 → 픽셀 · 2 + (0 왼쪽 u − edge_px, 1 오른쪽 u + edge_px)
struct DepthSide {
  static constexpr bool kDirect = false;
  __device__ int env_of(const EnvBatch& b, int job) const { return (b.dep_jobs[job] >> 1) / mdp::obs::DEPTH_SIZE; }
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int q = b.dep_jobs[job], px = q >> 1;
    const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
    const EnvState& e = b.st[env];
    const EnvCfg& c = b.cfg;
    float u, v;
    mdp::obsgen::depth_uv(c.depth, i, &u, &v);
    const Pose cam = mdp::obsgen::depth_camera(c, e);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = cam.p;
    d = depth_dir(c.depth.spec, cam, (q & 1) ? u + c.depth.edge_px : u - c.depth.edge_px, v);
    tmax = depth_tmax(c.depth.spec);
    return true;
  }
  __device__ void none(const EnvBatch&, int) const {}
  __device__ void store(const EnvBatch& b, int job, float t, int) const { b.dep_side[b.dep_jobs[job]] = t; }
};

// 라이다 빔 가운데 광선: job = env · kMaxBeams + j (이번 장이 있는 환경의 j < n_beams 만)
struct BeamCenter {
  static constexpr bool kDirect = false;
  __device__ int env_of(const EnvBatch&, int job) const { return job / kMaxBeams; }
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int env = job / kMaxBeams, j = job % kMaxBeams;
    const Shared& sh = b.scratch[env];
    if (!sh.pending || j >= sh.n_beams) return false;
    const EnvState& e = b.st[env];
    BeamGeom g;
    beam_center_ray(e.noise, sh.scan, j, g);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = g.o;
    d = g.d;
    tmax = e.noise.max_range;
    return true;
  }
  __device__ void none(const EnvBatch&, int) const {}
  __device__ void store(const EnvBatch& b, int job, float t, int tri) const { b.beam_t[job] = t; b.beam_tri[job] = tri; }
};

// 라이다 옆 광선: job = 큐 칸 → 빔 · 4 + k
struct BeamSide {
  static constexpr bool kDirect = false;
  __device__ int env_of(const EnvBatch& b, int job) const { return (b.beam_jobs[job] >> 2) / kMaxBeams; }
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int q = b.beam_jobs[job], bj = q >> 2;
    const int env = bj / kMaxBeams, j = bj % kMaxBeams;
    const Shared& sh = b.scratch[env];
    const EnvState& e = b.st[env];
    BeamGeom g;
    beam_center_ray(e.noise, sh.scan, j, g);
    sv = with_people(b.maps[e.map].scene, e.people);
    o = g.o;
    d = beam_side_dir(e.noise, sh.scan, g, q & 3);
    tmax = e.noise.max_range;
    return true;
  }
  __device__ void none(const EnvBatch&, int) const {}
  __device__ void store(const EnvBatch& b, int job, float t, int) const { b.beam_side[b.beam_jobs[job]] = t; }
};

// 빔마다 (lidar_beam_real 을 광선 사이에서 나눈 것): final = 0 가운데 광선 뒤 — 끝나면 지도 표시, 옆 광선이 있어야 하면 큐에.
// final = 1 옆 광선 뒤 마무리
__global__ void __launch_bounds__(kBeamThreads) env_beam_kernel(EnvBatch b, int final) {
  const int env = blockIdx.y, j = blockIdx.x * kBeamThreads + threadIdx.x;
  Shared& sh = b.scratch[env];
  if (!sh.pending || j >= sh.n_beams) return;
  const int job = env * kMaxBeams + j;
  const EnvState& e = b.st[env];
  const LidarNoise& N = e.noise;
  const SceneView sv = with_people(b.maps[e.map].scene, e.people);
  BeamGeom geo;
  beam_center_ray(N, sh.scan, j, geo);
  beam_center_hit(sv, geo, b.beam_t[job], b.beam_tri[job]);
  if (!geo.hit) return;
  const BeamDraw w = beam_draw(sh.scan, j);
  Material mat;
  const int st = beam_mid(sv, N, sh.scan, geo, w, &mat);
  if (st == 0) return;
  float r;
  if (!final) {
    if (st == 2) {
      const int q = atomicAdd(b.counters + kCntBeamSide, 4);
      for (int k = 0; k < 4; ++k) b.beam_jobs[q + k] = 4 * job + k;
      return;
    }
    r = beam_finish(N, geo, w, mat, false);
  } else {
    if (st != 2) return;
    beam_edges_from(N, geo, b.beam_side + 4 * job);
    r = beam_finish(N, geo, w, mat, true);
  }
  BeamRay ray;
  beam_ray_out(geo, &ray);
  scan_beam_mark(b, e, env, sh, j, r, ray);
}

void beams_gpu(const EnvBatch& b, cudaStream_t st) {
  const dim3 grid((kMaxBeams + kBeamThreads - 1) / kBeamThreads, b.n_env);
#ifdef SIM_OPTIX
  if (b.rt)
    rt_rays(b, BeamCenter{}, b.n_env * kMaxBeams, b.n_env * kMaxBeams, nullptr, st);
  else
#endif
    launch_rays(b, BeamCenter{}, b.n_env * kMaxBeams, nullptr, kCntBeamNext, st);
  cudaMemsetAsync(b.counters + kCntBeamSide, 0, sizeof(int), st);
  env_beam_kernel<<<grid, kBeamThreads, 0, st>>>(b, 0);
#ifdef SIM_OPTIX
  if (b.rt)
    rt_rays(b, BeamSide{}, 4 * b.n_env * kMaxBeams, 0, b.counters + kCntBeamSide, st);
  else
#endif
    launch_rays(b, BeamSide{}, 0, b.counters + kCntBeamSide, kCntBeamSideNext, st);
  env_beam_kernel<<<grid, kBeamThreads, 0, st>>>(b, 1);
}

// 픽셀마다: 가운데 광선 뒤 구멍이 정해지면 값을 쓰고, 아니면 옆 광선 둘을 큐에 (final = 0) / 옆 광선 뒤 마무리 (final = 1)
__global__ void __launch_bounds__(kDepthThreads) env_depth_pixel_kernel(EnvBatch b, int final) {
  const int px = blockIdx.x * kDepthThreads + threadIdx.x;
  if (px >= b.n_env * mdp::obs::DEPTH_SIZE) return;
  const int env = px / mdp::obs::DEPTH_SIZE, i = px % mdp::obs::DEPTH_SIZE;
  const EnvState& e = b.st[env];
  const EnvCfg& c = b.cfg;
  const DepthModel& M = c.depth;
  uint8_t* out = mdp::obsgen::depth_out(b, env);
  if (e.smoke) {   // 연기 모드: 깊이를 가림 (mdp/smoke.h). 두 단계 모두 0 을 쓰고 옆 광선 일감도 안 만듦 — CPU obs_depth 와 같음
    out[i] = mdp::smoke::depth_px(e.smoke, 0);
    return;
  }
  const int tri = b.dep_tri[px];
  if (tri == -1) {   // 유효 영역 밖·못 맞음
    if (!final) out[i] = 0;
    return;
  }
  const float z = b.dep_z[px];
  float u, v;
  mdp::obsgen::depth_uv(M, i, &u, &v);
  const Pose cam = mdp::obsgen::depth_camera(c, e);
  const SceneView sv = with_people(b.maps[e.map].scene, e.people);
  uint64_t seed = mdp::obsgen::depth_seed(e, i);
  const float u_hole = dm::rand01(seed);
  float p_edge, p_flat;
  depth_hole_probs(sv, M, depth_dir(M.spec, cam, u, v), z, tri, &p_edge, &p_flat);
  int hole = depth_hole_decided(u_hole, p_edge, p_flat);
  if (!final) {
    if (hole < 0) {
      const int q = atomicAdd(b.counters + kCntDepSide, 2);
      b.dep_jobs[q] = 2 * px;
      b.dep_jobs[q + 1] = 2 * px + 1;
      return;
    }
  } else {
    if (hole >= 0) return;   // 앞에서 씀
    const bool edge = depth_edge(z, b.dep_side[2 * px]) || depth_edge(z, b.dep_side[2 * px + 1]);
    hole = u_hole < (edge ? p_edge : p_flat) ? 1 : 0;
  }
  out[i] = mdp::obsgen::depth_u8(M, hole ? 0.f : depth_value(M, z, seed));
}


void depth_gpu(const EnvBatch& b, cudaStream_t st) {
  const int n_px = b.n_env * mdp::obs::DEPTH_SIZE;
  const int px_blocks = (n_px + kDepthThreads - 1) / kDepthThreads;
#ifdef SIM_OPTIX
  if (b.rt)
    rt_rays(b, DepthCenter{}, n_px, n_px, nullptr, st);
  else
#endif
    launch_rays(b, DepthCenter{}, n_px, nullptr, kCntDepNext, st);
  cudaMemsetAsync(b.counters + kCntDepSide, 0, sizeof(int), st);
  env_depth_pixel_kernel<<<px_blocks, kDepthThreads, 0, st>>>(b, 0);
#ifdef SIM_OPTIX
  if (b.rt)
    rt_rays(b, DepthSide{}, 2 * n_px, 0, b.counters + kCntDepSide, st);
  else
#endif
    launch_rays(b, DepthSide{}, 0, b.counters + kCntDepSide, kCntDepSideNext, st);
  env_depth_pixel_kernel<<<px_blocks, kDepthThreads, 0, st>>>(b, 1);
}

}  // namespace

void env_step_gpu(const EnvBatch& b, const float* actions, cudaStream_t st) {
  const EnvCfg& c = b.cfg;
  // 한 행동의 라이다 장 수 ≤ ⌈substeps / lidar_every⌉ — 장마다 물리·빔·지도 갱신 한 바퀴
  const int n_scan = (c.substeps + c.lidar_every - 1) / c.lidar_every;
  const int phys_blocks = (b.n_env + kPhysThreads - 1) / kPhysThreads;
  for (int s = 0; s < n_scan; ++s) {
    env_phys_kernel<<<phys_blocks, kPhysThreads, 0, st>>>(b, actions, s == 0);
    beams_gpu(b, st);
    env_apply_kernel<<<b.n_env, kThreads, 0, st>>>(b);
  }
  env_finish_kernel<<<b.n_env, kThreads, 0, st>>>(b);
  depth_gpu(b, st);
}

void env_reset_gpu(const EnvBatch& b, cudaStream_t st) {
  env_reset_kernel<<<b.n_env, kThreads, 0, st>>>(b);
  depth_gpu(b, st);
}

}  // namespace sim
