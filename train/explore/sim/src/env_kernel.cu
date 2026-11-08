// 환경 한 걸음 GPU: 단계 순서·값은 src/env_ref.cpp(CPU 참조판)와 같다 (비트 동일 — tools/env_verify). 일을 성격대로 커널로 나눈다:
//   1. 물리 (환경 하나 = 스레드 하나): 행동 → 물리 걸음을 다음 라이다 장까지 → 장 준비 (scan_begin)
//   2. 빔 (빔 하나 = 스레드 하나): 재고 지도 표시·각도 칸 (scan_beam — 장치 메모리 원자 연산, 정수 min/or 라 순서 무관)
//   3. 지도 갱신 (환경 하나 = 블록 하나): scan_apply + 정수 합 → 1 로 돌아가 다음 장 (목표점 행동은 한 행동에 장 여럿)
//   4. 마무리 (환경 하나 = 블록 하나): 남은 물리 → 보상·끝 → 라이다 링 → 끝난 판 리셋 → 관측 (깊이 빼고)
//   5. 깊이 (픽셀 하나 = 스레드 하나)
// 2·5 의 광선은 RT 코어(OptiX — src/optix_rays.cu 의 raygen 이 광선을 만들고 결과 뒤 단계까지)나 소프트웨어 순회(아래 광선 묶음 커널)로
// 커널 사이 상태는 b.st(EnvState)와 b.scratch(Shared, 환경마다)에 둔다.
#include "env_api.h"
#include "env_rays.h"
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

// 먼 지도: 다시 그릴 칸을 모아(공유 메모리 목록) 칸마다 8 스레드가 점 줄 하나씩 — 최댓값·OR 을 8 스레드끼리 합친다.
// 칸마다 스레드 하나가 64 점을 다 돌면 다시 그릴 칸이 흩어진 워프가 오래 놀아서 (CPU 참조판은 obs_cells 안에서 칸마다 차례로, 같은 함수)
__device__ void obs_far_block(const EnvBatch& b, const EnvState& e, int env, int tid, const mdp::obsgen::FarDirty& fd, bool statics) {
  using namespace mdp::obs;
  constexpr int kRowThreads = mdp::obsgen::kFarSub;   // 칸 하나 = 8 스레드
  constexpr int kPerRound = kThreads / kRowThreads;
  static_assert(32 % kRowThreads == 0 && kThreads % 32 == 0, "점 줄 스레드 묶음이 워프 안에");
  __shared__ int list[FAR_H * FAR_W];
  __shared__ int n_list;
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const int16_t* L = b.lmap + (size_t)env * c.size * c.size;
  const uint32_t* M = b.marks + (size_t)env * (c.size * c.size / 4);
  uint8_t* F = b.obs_img + (size_t)env * IMG_SIZE + NEAR_SIZE;
  mdp::obsgen::FarFrame ff;
  mdp::obsgen::far_frame(e, e.x + e.ex, e.y + e.ey, &ff);
  if (tid == 0) n_list = 0;
  __syncthreads();
  for (int i = tid; i < FAR_H * FAR_W; i += kThreads)
    if (mdp::obsgen::far_cell_head(e, fd, ff, F, i, statics)) list[atomicAdd(&n_list, 1)] = i;
  __syncthreads();
  const int n = n_list, kr = tid % kRowThreads;
  for (int base = 0; base < n; base += kPerRound) {   // 모든 스레드가 같은 횟수 (셔플이 워프 전체)
    const int t = base + tid / kRowThreads;
    int lmax = kUnknown, vis = 0;
    if (t < n) mdp::obsgen::far_cell_row(c, m, L, M, e, ff, list[t], kr, &lmax, &vis);
    for (int o = 1; o < kRowThreads; o <<= 1) {
      const int l2 = __shfl_xor_sync(0xffffffffu, lmax, o);
      lmax = l2 > lmax ? l2 : lmax;
      vis |= __shfl_xor_sync(0xffffffffu, vis, o);
    }
    if (t < n && kr == 0) mdp::obsgen::far_cell_put(c, F, list[t], lmax, vis);
  }
}

// 프론티어 요약 (observations.h obs_cells 의 프론티어 칸과 같은 값): 칸마다 frontier_cell 을 다시 보는 대신 프론티어 칸 지도(fmap)의 켜진 비트만 —
// 프론티어 칸은 아는 칸이라 모두 건드린 범위(kb) 안이고, 비트는 지도 갱신 때 고쳐 둠 (frontier_fix). sh.part[tid] = 칸 수, sh.fr_min = 가장 가까운 키
__device__ void obs_front_block(const EnvBatch& b, const EnvState& e, int env, Shared& sh, int tid) {
  int n = 0;
  if (e.kb_x1 >= 0) {
    const EnvCfg& c = b.cfg;
    const MapDev& m = b.maps[e.map];
    int rx, ry;
    cell_of(c, m, e.x + e.ex, e.y + e.ey, &rx, &ry);
    const int ws = fmap_row(c), w0 = e.kb_x0 >> 5, nw = (e.kb_x1 >> 5) - w0 + 1, rows = e.kb_y1 - e.kb_y0 + 1;
    const uint32_t* F = b.fmap + (size_t)env * c.size * ws;
    unsigned long long best = mdp::kNoFrontier;
    for (int k = tid; k < nw * rows; k += kThreads) {
      const int y = e.kb_y0 + k / nw, wx = w0 + k % nw;
      uint32_t bits = F[y * ws + wx];
      n += __popc(bits);
      for (; bits; bits &= bits - 1) {
        const int x = wx * 32 + __ffs(bits) - 1;
        const int dx = x - rx, dy = y - ry;
        const unsigned long long key = ((unsigned long long)(uint32_t)(dx * dx + dy * dy) << 32) | (uint32_t)(y * m.gw + x);   // mdp::summary_key
        best = key < best ? key : best;
      }
    }
    if (best != mdp::kNoFrontier) atomicMin(&sh.fr_min, best);
  }
  sh.part[tid] = n;
}

// 관측: 칸마다(모든 스레드) → 합치기·숫자(스레드 0). statics: 늘 0 인 물체 채널(📏)도 씀 — 판이 시작할 때만 (observations.h obs_cells)
__device__ void write_obs(const EnvBatch& b, EnvState& e, int env, Shared& sh, int tid, const mdp::obsgen::FarDirty& fd, bool statics) {
  if (tid == 0) sh.fr_min = mdp::kNoFrontier;
  __syncthreads();
  mdp::obsgen::obs_cells(b, e, env, sh, tid, fd, false, statics);
  obs_far_block(b, e, env, tid, fd, statics);
  obs_front_block(b, e, env, sh, tid);
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
    mdp::obsgen::far_dirty_clear(sh);
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

// 프론티어 칸 지도 고치기 (지도 갱신 뒤): 칸 종류가 바뀐 칸이 있는 낱말(4 칸)과 그 4 이웃(위·아래 줄 4 칸, 왼·오른 끝 칸)의 비트를 frontier_cell 로 다시 —
// 다른 칸은 칸 종류도 이웃도 그대로라 비트가 같다.
// 목록이 넘치면 이번 장 범위 ±1 전체를 다시. 같은 칸을 여러 스레드가 고쳐도 같은 지도에서 같은 값
__device__ void frontier_fix(const EnvBatch& b, const EnvState& e, int env, const Shared& sh, int tid, int n_chg) {
  const EnvCfg& c = b.cfg;
  const MapDev& m = b.maps[e.map];
  const mdp::MapView mv{b.lmap + (size_t)env * c.size * c.size, c.size, m.gw, m.gh};
  const int ws = fmap_row(c);
  uint32_t* F = b.fmap + (size_t)env * c.size * ws;
  const auto fix = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= m.gw || y >= m.gh) return;
    const uint32_t bit = 1u << (x & 31);
    uint32_t* w = F + y * ws + (x >> 5);
    if (mdp::frontier_cell(mv, x, y)) {
      if (!(*w & bit)) atomicOr(w, bit);
    } else if (*w & bit) {
      atomicAnd(w, ~bit);
    }
  };
  if (n_chg <= kClassLog) {
    const int32_t* list = b.fchg + (size_t)env * kClassLog;
    for (int k = tid; k < 14 * n_chg; k += kThreads) {   // 낱말마다 14 칸: 줄 y−1·y·y+1 의 4 칸 + (x0 − 1, y) + (x0 + 4, y)
      const int ci = list[k / 14], x0 = ci % c.size, y = ci / c.size, r = k % 14;
      if (r < 12) fix(x0 + (r & 3), y + r / 4 - 1);
      else fix(r == 12 ? x0 - 1 : x0 + 4, y);
    }
  } else if (sh.bx1 >= 0) {
    const int x0 = sh.bx0 - 1, y0 = sh.by0 - 1, w = sh.bx1 - sh.bx0 + 3, h = sh.by1 - sh.by0 + 3;
    for (int k = tid; k < w * h; k += kThreads) fix(x0 + k % w, y0 + k / w);
  }
}

__global__ void __launch_bounds__(kThreads) env_apply_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  Shared& sh = b.scratch[env];
  if (!sh.pending) return;   // 블록 전체가 같은 값
  EnvState& e = b.st[env];
  __shared__ int n_chg;
  if (tid == 0) n_chg = 0;
  __syncthreads();
  const ClassLog log{b.fchg + (size_t)env * kClassLog, &n_chg};
  const ApplyCount n = scan_apply(b, e, env, sh, tid, kThreads, b.scratch[env].far_mask, &log);   // 로그 오즈가 바뀐 칸 → 먼 지도 칸 표시
  int v[4] = {n.wall, n.floor, n.known, n.fresh};
  block_sum4(v, tid);   // 끝에 __syncthreads — 지도 갱신이 다 보임
  frontier_fix(b, e, env, sh, tid, n_chg);
  if (tid == 0) {
    sh.new_wall += v[0];
    sh.new_floor += v[1];
    sh.d_known += v[2];
    sh.n_new += v[3];
    known_box(b, e, sh);
    sh.pending = 0;
  }
}

// 상태(EnvState)를 블록 스레드 모두가 낱말(4 바이트)씩 옮긴다 — 스레드 하나가 구조체 전체를 옮기는 동안 블록이 기다리지 않게
__device__ void copy_state(EnvState& dst, const EnvState& src, int tid) {
  static_assert(sizeof(EnvState) % 4 == 0, "EnvState 낱말 복사");
  const uint32_t* s = (const uint32_t*)&src;
  uint32_t* d = (uint32_t*)&dst;
  for (int i = tid; i < (int)(sizeof(EnvState) / 4); i += kThreads) d[i] = s[i];
}

__global__ void __launch_bounds__(kThreads) env_finish_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  __shared__ Shared sh;   // 관측용 (프론티어 합)
  Shared& g = b.scratch[env];
  copy_state(e, b.st[env], tid);
  __syncthreads();
  if (tid == 0) {
    // 마지막 장 뒤 남은 물리 걸음 (장 수만큼 1~3 을 돌렸으므로 여기선 장이 없다)
    while (g.k_next < b.cfg.substeps) {
      if (!e.collided) physics_step(b, e);
      e.tick += 1;
      g.k_next += 1;
    }
    finish_step(b, e, env, g.new_wall, g.new_floor, g.d_known, g.n_new);
    mdp::obsgen::far_dirty_robot(b, e, g);   // 지나간 곳 표시도 먼 지도 다시 그릴 곳에
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
    fd = mdp::obsgen::far_dirty_take(b, e, g, was_done);
  }
  write_obs(b, e, env, sh, tid, fd, was_done);   // 끝에 __syncthreads
  copy_state(b.st[env], e, tid);
  if (tid == 0) g.cam = mdp::obsgen::depth_camera(b.cfg, e);
}

__global__ void env_reset_kernel(EnvBatch b) {
  const int env = blockIdx.x, tid = threadIdx.x;
  __shared__ EnvState e;
  __shared__ Shared sh;
  copy_state(e, b.st[env], tid);
  __syncthreads();
  reset_env(b, e, env, tid);
  const mdp::obsgen::FarDirty all{1, 0.f, 0.f, 0.f, 0.f};
  write_obs(b, e, env, sh, tid, all, true);
  if (tid == 0) {
    b.scratch[env].far_ok = 0;   // 다음 걸음은 다시 전부 (판 기록은 그때 남김)
    b.scratch[env].cam = mdp::obsgen::depth_camera(b.cfg, e);
  }
  copy_state(b.st[env], e, tid);
  if (tid == 0) {
    b.reward[env] = 0.f;
    b.done[env] = 0;
    for (int k = 0; k < mdp::rew::kTerms; ++k) b.reward_terms[(size_t)env * mdp::rew::kTerms + k] = 0.f;
  }
}

// ---------- 소프트웨어 광선 묶음: 스레드마다 광선 하나씩 순회(trav_step)하고, 끝난 스레드는 워프 단위로 다음 광선을 받는다 ----------
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

// 깊이 가운데 광선: job = env · DEPTH_SIZE + 픽셀
struct DepthCenter {
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
  __device__ bool make(const EnvBatch& b, int job, SceneView& sv, V3& o, V3& d, float& tmax) const {
    const int env = job / kMaxBeams, j = job % kMaxBeams;
    if (!beam_live(b, env, j)) return false;
    const Shared& sh = b.scratch[env];
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

// 빔마다 가운데 광선 뒤: 끝나면 지도 표시, 옆 광선이 있어야 하면 큐에 — PER = 4 (소프트웨어: 광선마다 칸 하나, 빔 · 4 + k) · 1 (RT 코어: 빔)
template <int PER>
__global__ void __launch_bounds__(kBeamThreads) env_beam_kernel(EnvBatch b) {
  const int env = blockIdx.y, j = blockIdx.x * kBeamThreads + threadIdx.x;
  if (!beam_live(b, env, j)) return;
  const int job = env * kMaxBeams + j;
  if (beam_center_done(b, env, j, b.beam_t[job], b.beam_tri[job])) return;
  const int q = atomicAdd(b.counters + kCntBeamSide, PER);
  if (PER == 1) b.beam_jobs[q] = job;
  else
    for (int k = 0; k < 4; ++k) b.beam_jobs[q + k] = 4 * job + k;
}

// 옆 광선 큐의 빔마다(PER 칸 묶음) 그 빔을 마무리 — 큐에 없는 빔은 첫 패스에서 끝났다
template <int PER>
__global__ void __launch_bounds__(kDepthThreads) env_beam_side_kernel(EnvBatch b) {
  const int g = blockIdx.x * kDepthThreads + threadIdx.x;
  if (PER * g >= b.counters[kCntBeamSide]) return;
  const int job = PER == 1 ? b.beam_jobs[g] : b.beam_jobs[4 * g] >> 2;
  beam_side_done(b, job / kMaxBeams, job % kMaxBeams, b.beam_t[job], b.beam_tri[job], b.beam_side + 4 * job);
}

// 픽셀마다 가운데 광선 뒤: 값이 정해지면 쓰고, 아니면 큐에 — PER = 2 (소프트웨어: 광선마다 칸 하나, 픽셀 · 2 + 왼0/오른1) · 1 (RT 코어: 픽셀)
template <int PER>
__global__ void __launch_bounds__(kDepthThreads) env_depth_pixel_kernel(EnvBatch b) {
  const int px = blockIdx.x * kDepthThreads + threadIdx.x;
  if (px >= b.n_env * mdp::obs::DEPTH_SIZE) return;
  const int tri = b.dep_tri[px];
  if (depth_px_center(b, px, b.dep_z[px], tri, tri == -1 ? V3{} : depth_px_dir(b, px))) return;
  const int q = atomicAdd(b.counters + kCntDepSide, PER);
  if (PER == 1) {
    b.dep_jobs[q] = px;
    return;
  }
  b.dep_jobs[q] = 2 * px;
  b.dep_jobs[q + 1] = 2 * px + 1;
}

// 옆 광선 큐의 픽셀마다(PER 칸 묶음) 그 픽셀을 마무리 — 큐에 없는 픽셀은 첫 패스에서 이미 썼다
template <int PER>
__global__ void __launch_bounds__(kDepthThreads) env_depth_side_kernel(EnvBatch b) {
  const int pair = blockIdx.x * kDepthThreads + threadIdx.x;
  if (PER * pair >= b.counters[kCntDepSide]) return;
  const int px = PER == 1 ? b.dep_jobs[pair] : b.dep_jobs[2 * pair] >> 1;
  depth_px_side(b, px, b.dep_z[px], b.dep_tri[px], depth_px_dir(b, px), b.dep_side + 2 * px);
}

#ifdef SIM_OPTIX
// RT 코어 (src/optix_rays.cu): raygen 이 광선을 만들어 쏜다 (깊이 옆 광선은 마무리까지). 1·3 은 픽셀·빔 전부, 2·4 는 큐 길이(장치, counter 칸)만큼
void rt_launch(const EnvBatch& b, RtProg prog, int n, int counter, cudaStream_t st) {
  RtParams p{};
  p.prog = prog;
  p.batch = b.dev_self;
  p.fast = b.rt_fast;
  p.n = n;
  if (prog == kRtDepthSide || prog == kRtBeamSide) p.n_dev = b.counters + counter;
  ((OptixRt*)b.rt)->launch(p, st);
}
#endif

void beams_gpu(const EnvBatch& b, cudaStream_t st) {
  const int n = b.n_env * kMaxBeams;
#ifdef SIM_OPTIX
  const dim3 grid((kMaxBeams + kBeamThreads - 1) / kBeamThreads, b.n_env);
  if (b.rt) {   // raygen 은 광선만 (결과 뒤는 아래 커널)
    rt_launch(b, kRtBeam, n, 0, st);
    cudaMemsetAsync(b.counters + kCntBeamSide, 0, sizeof(int), st);
    env_beam_kernel<1><<<grid, kBeamThreads, 0, st>>>(b);
    rt_launch(b, kRtBeamSide, n, kCntBeamSide, st);
    env_beam_side_kernel<1><<<(n + kDepthThreads - 1) / kDepthThreads, kDepthThreads, 0, st>>>(b);
    return;
  }
#endif
  launch_rays(b, BeamCenter{}, n, nullptr, kCntBeamNext, st);
  cudaMemsetAsync(b.counters + kCntBeamSide, 0, sizeof(int), st);
  env_beam_kernel<4><<<dim3((kMaxBeams + kBeamThreads - 1) / kBeamThreads, b.n_env), kBeamThreads, 0, st>>>(b);
  launch_rays(b, BeamSide{}, 0, b.counters + kCntBeamSide, kCntBeamSideNext, st);
  env_beam_side_kernel<4><<<(n + kDepthThreads - 1) / kDepthThreads, kDepthThreads, 0, st>>>(b);   // 옆 광선을 쏜 빔만 마무리
}

void depth_gpu(const EnvBatch& b, cudaStream_t st) {
  const int n_px = b.n_env * mdp::obs::DEPTH_SIZE;
  const int px_blocks = (n_px + kDepthThreads - 1) / kDepthThreads;
#ifdef SIM_OPTIX
  if (b.rt) {
    rt_launch(b, kRtDepth, n_px, 0, st);
    cudaMemsetAsync(b.counters + kCntDepSide, 0, sizeof(int), st);
    env_depth_pixel_kernel<1><<<px_blocks, kDepthThreads, 0, st>>>(b);
    rt_launch(b, kRtDepthSide, n_px, kCntDepSide, st);
    env_depth_side_kernel<1><<<px_blocks, kDepthThreads, 0, st>>>(b);
    return;
  }
#endif
  launch_rays(b, DepthCenter{}, n_px, nullptr, kCntDepNext, st);
  cudaMemsetAsync(b.counters + kCntDepSide, 0, sizeof(int), st);
  env_depth_pixel_kernel<2><<<px_blocks, kDepthThreads, 0, st>>>(b);
  launch_rays(b, DepthSide{}, 0, b.counters + kCntDepSide, kCntDepSideNext, st);
  env_depth_side_kernel<2><<<px_blocks, kDepthThreads, 0, st>>>(b);   // 옆 광선을 쏜 픽셀만 마무리
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
