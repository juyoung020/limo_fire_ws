#include "trt_yolo_kernels.cuh"

#include <cuda_fp16.h>

namespace limo_fire_perception::gpu
{

namespace
{

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

// 원본 픽셀 가운데 → 원형 칸 좌표 (칸 가운데 기준, 음수는 0: PyTorch align_corners=False)
__device__ __forceinline__ float to_proto(int u, float scale, float pad, int proto_n, int in_n)
{
  return fmaxf(((u + 0.5f) * scale + pad) * proto_n / in_n - 0.5f, 0.0f);
}

// 워프 하나: 32 줄씩 문턱 검사 → ballot 으로 순서를 지켜 앞에서부터 채운다
__global__ void decode_kernel(const float * __restrict__ out, int rows, int cols, float conf, Letterbox lb,
  int ph, int pw, DetBlock * block, bool set_cond, cudaGraphConditionalHandle cond)
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
      d.rx = d.ry = d.rw = d.rh = 0;
      d.gx0 = d.gy0 = 0;
      d.gx1 = d.gy1 = -1;
      if (ax1 > ax0 && ay1 > ay0) {
        d.rx = ax0; d.ry = ay0; d.rw = ax1 - ax0; d.rh = ay1 - ay0;
        if (pw > 0) {
          d.gx0 = min(static_cast<int>(to_proto(ax0, lb.scale, lb.pad_x, pw, lb.in_w)), pw - 1);
          d.gy0 = min(static_cast<int>(to_proto(ay0, lb.scale, lb.pad_y, ph, lb.in_h)), ph - 1);
          d.gx1 = min(static_cast<int>(to_proto(ax1 - 1, lb.scale, lb.pad_x, pw, lb.in_w)) + 1, pw - 1);
          d.gy1 = min(static_cast<int>(to_proto(ay1 - 1, lb.scale, lb.pad_y, ph, lb.in_h)) + 1, ph - 1);
        }
      }
      d.row = r;
      block->dets[idx] = d;
    }
    base += __popc(m);
  }
  if (lane == 0) {
    block->count = min(base, kMaxDets);
    if (set_cond) {
      cudaGraphSetConditional(cond, base > 0 ? 1u : 0u);
    }
  }
}

// 블록 = (원형 칸 256 개, 물체 하나). 그 물체 상자에 필요한 칸만 계산
__global__ void mask_logits_kernel(const float * __restrict__ out, int cols, Proto proto,
  const DetBlock * __restrict__ block, float * __restrict__ logits)
{
  __shared__ float s_coef[kMaxNm];
  const int k = blockIdx.y;
  const int n = min(block->count, kMaxMaskDets);
  if (k >= n) {
    return;
  }
  const Det & d = block->dets[k];
  if (threadIdx.x < proto.nm) {
    s_coef[threadIdx.x] = out[static_cast<size_t>(d.row) * cols + 6 + threadIdx.x];
  }
  __syncthreads();
  const int plane = proto.ph * proto.pw;
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= plane) {
    return;
  }
  const int y = cell / proto.pw, x = cell - y * proto.pw;
  if (x < d.gx0 || x > d.gx1 || y < d.gy0 || y > d.gy1) {
    return;
  }
  float sum = 0.0f;
  if (proto.half) {
    const __half * p = static_cast<const __half *>(proto.data) + cell;
    for (int c = 0; c < proto.nm; ++c) {
      sum += s_coef[c] * __half2float(p[static_cast<size_t>(c) * plane]);
    }
  } else {
    const float * p = static_cast<const float *>(proto.data) + cell;
    for (int c = 0; c < proto.nm; ++c) {
      sum += s_coef[c] * p[static_cast<size_t>(c) * plane];
    }
  }
  logits[static_cast<size_t>(k) * plane + cell] = sum;
}

// 픽셀 하나 = 스레드 하나. 이 픽셀을 덮는 물체마다 로짓 4 칸 쌍선형
__global__ void paint_kernel(const DetBlock * __restrict__ block, const float * __restrict__ logits, int ph, int pw,
  Letterbox lb, uint8_t * __restrict__ ids)
{
  __shared__ int4 s_rect[kMaxMaskDets];  // x0 y0 x1 y1 (끝 미포함)
  __shared__ int s_area[kMaxMaskDets];
  const int n = min(block->count, kMaxMaskDets);
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x * blockDim.y) {
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
  const float gx = to_proto(u, lb.scale, lb.pad_x, pw, lb.in_w);
  const float gy = to_proto(v, lb.scale, lb.pad_y, ph, lb.in_h);
  const int x0 = min(static_cast<int>(gx), pw - 1), y0 = min(static_cast<int>(gy), ph - 1);
  const int x1 = min(x0 + 1, pw - 1), y1 = min(y0 + 1, ph - 1);
  const float ax = gx - x0, ay = gy - y0;
  const int o00 = y0 * pw + x0, o01 = y0 * pw + x1, o10 = y1 * pw + x0, o11 = y1 * pw + x1;
  const size_t plane = static_cast<size_t>(ph) * pw;
  int best = 0, best_area = 0x7fffffff;
  for (int k = 0; k < n; ++k) {
    const int4 r = s_rect[k];
    if (u < r.x || u >= r.z || v < r.y || v >= r.w || s_area[k] >= best_area) {
      continue;
    }
    const float * L = logits + k * plane;
    const float logit = (1 - ay) * ((1 - ax) * L[o00] + ax * L[o01]) + ay * ((1 - ax) * L[o10] + ax * L[o11]);
    if (logit > 0.0f) {
      best = k + 1;
      best_area = s_area[k];
    }
  }
  ids[static_cast<size_t>(v) * lb.img_w + u] = static_cast<uint8_t>(best);
}

__global__ void draw_kernel(const uint8_t * __restrict__ img, const uint8_t * __restrict__ ids,
  const DetBlock * __restrict__ block, const uint8_t * __restrict__ colors, int n_pix, uint8_t * __restrict__ vis)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_pix) {
    return;
  }
  const uint8_t * s = img + static_cast<size_t>(i) * 3;
  uint8_t * o = vis + static_cast<size_t>(i) * 3;
  const int id = ids ? ids[i] : 0;
  if (id == 0 || id > block->count) {
    o[0] = s[0]; o[1] = s[1]; o[2] = s[2];
    return;
  }
  const uint8_t * c = colors + (block->dets[id - 1].cls & 255) * 3;
  // cv::addWeighted(원본, 0.55, 색, 0.45) 와 같은 반올림
  for (int k = 0; k < 3; ++k) {
    o[k] = static_cast<uint8_t>(min(255, __float2int_rn(0.55f * s[k] + 0.45f * c[k])));
  }
}

}  // namespace

void letterbox(const uint8_t * img, void * dst, bool nchw_float, const Letterbox & lb, cudaStream_t st)
{
  const dim3 b(32, 8), g((lb.in_w + 31) / 32, (lb.in_h + 7) / 8);
  letterbox_kernel<<<g, b, 0, st>>>(img, dst, nchw_float, lb);
}

void decode_end2end(const float * out, int rows, int cols, float conf, const Letterbox & lb, int ph, int pw,
  DetBlock * block, bool set_cond, cudaGraphConditionalHandle cond, cudaStream_t st)
{
  decode_kernel<<<1, 32, 0, st>>>(out, rows, cols, conf, lb, ph, pw, block, set_cond, cond);
}

void mask_logits(const float * out, int cols, const Proto & proto, const DetBlock * block, float * logits,
  cudaStream_t st)
{
  const int plane = proto.ph * proto.pw;
  const dim3 g((plane + 255) / 256, kMaxMaskDets);
  mask_logits_kernel<<<g, 256, 0, st>>>(out, cols, proto, block, logits);
}

void paint_masks(const DetBlock * block, const float * logits, int ph, int pw, const Letterbox & lb, uint8_t * ids,
  cudaStream_t st)
{
  const dim3 b(32, 8), g((lb.img_w + 31) / 32, (lb.img_h + 7) / 8);
  paint_kernel<<<g, b, 0, st>>>(block, logits, ph, pw, lb, ids);
}

void draw_masks(const uint8_t * img, const uint8_t * ids, const DetBlock * block, const uint8_t * colors,
  int img_w, int img_h, uint8_t * vis, cudaStream_t st)
{
  const int n = img_w * img_h;
  draw_kernel<<<(n + 255) / 256, 256, 0, st>>>(img, ids, block, colors, n, vis);
}

}  // namespace limo_fire_perception::gpu
