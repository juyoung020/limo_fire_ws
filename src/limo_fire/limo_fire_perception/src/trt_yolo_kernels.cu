#include "trt_yolo_kernels.cuh"

namespace limo_fire_perception::gpu
{

namespace
{

// 마스크 계수 수 상한 (YOLO 분할 모델은 32). 픽셀마다 원형 32 장을 레지스터에 보간해 둔다
constexpr int kMaxNm = 32;

__global__ void letterbox_kernel(const uint8_t * __restrict__ img, void * dst, bool nchw_float, Letterbox lb)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lb.in_w || y >= lb.in_h) {
    return;
  }
  float c[3] = {114.0f, 114.0f, 114.0f};
  const int xi = x - lb.left, yi = y - lb.top;
  if (xi >= 0 && yi >= 0 && xi < lb.nw && yi < lb.nh) {
    // cv::resize INTER_LINEAR 과 같은 픽셀 가운데 기준 쌍선형
    const float sx = fmaxf((xi + 0.5f) * lb.img_w / lb.nw - 0.5f, 0.0f);
    const float sy = fmaxf((yi + 0.5f) * lb.img_h / lb.nh - 0.5f, 0.0f);
    const int x0 = min(static_cast<int>(sx), lb.img_w - 1), y0 = min(static_cast<int>(sy), lb.img_h - 1);
    const int x1 = min(x0 + 1, lb.img_w - 1), y1 = min(y0 + 1, lb.img_h - 1);
    const float ax = sx - x0, ay = sy - y0;
    const uint8_t * p00 = img + (static_cast<size_t>(y0) * lb.img_w + x0) * 3;
    const uint8_t * p01 = img + (static_cast<size_t>(y0) * lb.img_w + x1) * 3;
    const uint8_t * p10 = img + (static_cast<size_t>(y1) * lb.img_w + x0) * 3;
    const uint8_t * p11 = img + (static_cast<size_t>(y1) * lb.img_w + x1) * 3;
    for (int k = 0; k < 3; ++k) {
      c[k] = (1 - ay) * ((1 - ax) * p00[k] + ax * p01[k]) + ay * ((1 - ax) * p10[k] + ax * p11[k]);
    }
  }
  const size_t i = static_cast<size_t>(y) * lb.in_w + x;
  if (nchw_float) {
    const size_t plane = static_cast<size_t>(lb.in_w) * lb.in_h;
    float * d = static_cast<float *>(dst);
    for (int k = 0; k < 3; ++k) {
      d[k * plane + i] = c[k] / 255.0f;
    }
  } else {
    uint8_t * d = static_cast<uint8_t *>(dst) + i * 3;
    for (int k = 0; k < 3; ++k) {
      d[k] = static_cast<uint8_t>(__float2int_rn(c[k]));
    }
  }
}

// 워프 하나: 32 줄씩 문턱 검사 → ballot 으로 순서를 지켜 앞에서부터 채운다
__global__ void decode_kernel(const float * __restrict__ out, int rows, int cols, float conf, Letterbox lb,
  DetBlock * block)
{
  const int lane = threadIdx.x;
  int base = 0;
  for (int r0 = 0; r0 < rows; r0 += 32) {
    const int r = r0 + lane;
    const float * p = out + static_cast<size_t>(r) * cols;
    const bool pass = r < rows && p[4] >= conf;
    const unsigned m = __ballot_sync(0xffffffffu, pass);
    const int idx = base + __popc(m & ((1u << lane) - 1u));
    if (pass && idx < kMaxDets) {
      Det d;
      const float fw = static_cast<float>(lb.img_w), fh = static_cast<float>(lb.img_h);
      d.x1 = fminf(fmaxf((p[0] - lb.pad_x) / lb.scale, 0.0f), fw);
      d.y1 = fminf(fmaxf((p[1] - lb.pad_y) / lb.scale, 0.0f), fh);
      d.x2 = fminf(fmaxf((p[2] - lb.pad_x) / lb.scale, 0.0f), fw);
      d.y2 = fminf(fmaxf((p[3] - lb.pad_y) / lb.scale, 0.0f), fh);
      d.score = p[4];
      d.cls = static_cast<int>(p[5]);
      // Ultralytics crop_mask 와 같이 x1 <= u < x2 인 정수 픽셀 (상자는 이미 이미지 안으로 잘림)
      const int ax0 = static_cast<int>(ceilf(d.x1)), ay0 = static_cast<int>(ceilf(d.y1));
      const int ax1 = static_cast<int>(ceilf(d.x2)), ay1 = static_cast<int>(ceilf(d.y2));
      if (ax1 > ax0 && ay1 > ay0) {
        d.rx = ax0; d.ry = ay0; d.rw = ax1 - ax0; d.rh = ay1 - ay0;
      } else {
        d.rx = d.ry = d.rw = d.rh = 0;
      }
      d.row = r;
      block->dets[idx] = d;
    }
    base += __popc(m);
  }
  if (lane == 0) {
    block->count = min(base, kMaxDets);
  }
}

// 픽셀 하나 = 스레드 하나. 계수는 공유 메모리에.
// Ultralytics process_mask 와 같은 셈: 원형을 입력 크기로 쌍선형(align_corners=False) 확대 → 계수와 내적 → 상자 밖 버림 → 로짓 > 0
// 보간이 선형이라 원형 32 장을 이 픽셀 자리에서 먼저 보간해 두고 물체마다 내적만 한다
__global__ void paint_kernel(const float * __restrict__ out, int cols, const float * __restrict__ proto,
  int nm, int ph, int pw, const DetBlock * __restrict__ block, Letterbox lb, uint8_t * __restrict__ ids)
{
  extern __shared__ float s_coef[];          // [n][nm]
  __shared__ int4 s_rect[kMaxDets];          // x0 y0 x1 y1 (끝 미포함)
  __shared__ int s_area[kMaxDets];
  const int n = block->count;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  for (int i = tid; i < n * nm; i += nthreads) {
    s_coef[i] = out[static_cast<size_t>(block->dets[i / nm].row) * cols + 6 + i % nm];
  }
  for (int i = tid; i < n; i += nthreads) {
    const Det & d = block->dets[i];
    s_rect[i] = make_int4(d.rx, d.ry, d.rx + d.rw, d.ry + d.rh);
    s_area[i] = d.rw * d.rh;
  }
  __syncthreads();

  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= lb.img_w || v >= lb.img_h) {
    return;
  }
  float pv[kMaxNm];
  bool have_pv = false;
  int best = 0, best_area = 0x7fffffff;
  for (int k = 0; k < n; ++k) {
    const int4 r = s_rect[k];
    if (u < r.x || u >= r.z || v < r.y || v >= r.w || s_area[k] >= best_area) {
      continue;
    }
    if (!have_pv) {
      // 원본 픽셀 가운데 → 입력 좌표 → 원형 좌표 (칸 가운데 기준)
      const float gx = fmaxf(((u + 0.5f) * lb.scale + lb.pad_x) * pw / lb.in_w - 0.5f, 0.0f);
      const float gy = fmaxf(((v + 0.5f) * lb.scale + lb.pad_y) * ph / lb.in_h - 0.5f, 0.0f);
      const int x0 = min(static_cast<int>(gx), pw - 1), y0 = min(static_cast<int>(gy), ph - 1);
      const int x1 = min(x0 + 1, pw - 1), y1 = min(y0 + 1, ph - 1);
      const float ax = gx - x0, ay = gy - y0;
      const float w00 = (1 - ay) * (1 - ax), w01 = (1 - ay) * ax, w10 = ay * (1 - ax), w11 = ay * ax;
      const size_t plane = static_cast<size_t>(ph) * pw;
      const size_t o00 = static_cast<size_t>(y0) * pw + x0, o01 = static_cast<size_t>(y0) * pw + x1;
      const size_t o10 = static_cast<size_t>(y1) * pw + x0, o11 = static_cast<size_t>(y1) * pw + x1;
#pragma unroll
      for (int c = 0; c < kMaxNm; ++c) {
        if (c < nm) {
          const float * pc = proto + c * plane;
          pv[c] = w00 * pc[o00] + w01 * pc[o01] + w10 * pc[o10] + w11 * pc[o11];
        }
      }
      have_pv = true;
    }
    const float * coef = s_coef + k * nm;
    float logit = 0.0f;
#pragma unroll
    for (int c = 0; c < kMaxNm; ++c) {
      if (c < nm) {
        logit += coef[c] * pv[c];
      }
    }
    if (logit > 0.0f) {
      best = k + 1;
      best_area = s_area[k];
    }
  }
  ids[static_cast<size_t>(v) * lb.img_w + u] = static_cast<uint8_t>(best);
}

}  // namespace

void letterbox(const uint8_t * img, void * dst, bool nchw_float, const Letterbox & lb, cudaStream_t st)
{
  const dim3 b(32, 8), g((lb.in_w + 31) / 32, (lb.in_h + 7) / 8);
  letterbox_kernel<<<g, b, 0, st>>>(img, dst, nchw_float, lb);
}

void decode_end2end(const float * out, int rows, int cols, float conf, const Letterbox & lb,
  DetBlock * block, cudaStream_t st)
{
  decode_kernel<<<1, 32, 0, st>>>(out, rows, cols, conf, lb, block);
}

void paint_masks(const float * out, int cols, const float * proto, int nm, int ph, int pw,
  const DetBlock * block, const Letterbox & lb, uint8_t * ids, cudaStream_t st)
{
  const dim3 b(32, 8), g((lb.img_w + 31) / 32, (lb.img_h + 7) / 8);
  const size_t shmem = static_cast<size_t>(kMaxDets) * nm * sizeof(float);
  paint_kernel<<<g, b, shmem, st>>>(out, cols, proto, nm, ph, pw, block, lb, ids);
}

}  // namespace limo_fire_perception::gpu
