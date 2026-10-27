// 검증 마스크 IoU 재료(evalmap.h mask_inter): 블록 하나가 예측 하나
#include "evalmap.h"

namespace {
__global__ void mask_inter_k(const MaskQuery* q, const act_t* c0, const act_t* c1, const act_t* c2, const act_t* proto,
                             int mh, int mw, int nm, int imgsz, const int16_t* gm, int G, int* parea, int* inter) {
  extern __shared__ int sh[];  // [G] 정답별 교집합, [1] 넓이
  const MaskQuery& d = q[blockIdx.x];
  for (int i = threadIdx.x; i <= G; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const act_t* cf = (d.lvl == 0 ? c0 : d.lvl == 1 ? c1 : c2) + int64_t(d.cell) * nm;  // cell 은 영상까지 포함한 칸 번호
  float c[32];
  for (int j = 0; j < nm; ++j) c[j] = to_f(cf[j]);
  // 상자를 원형 마스크 크기로(Ultralytics: bboxes · (mw / 입력 폭)), 칸 r 은 x1 ≤ r < x2
  float sx = (float)mw / imgsz, sy = (float)mh / imgsz;
  float X1 = d.x1 * sx, X2 = d.x2 * sx, Y1 = d.y1 * sy, Y2 = d.y2 * sy;
  int x0 = max(0, (int)ceilf(X1)), x1 = min(mw, (int)ceilf(X2)), y0 = max(0, (int)ceilf(Y1)), y1 = min(mh, (int)ceilf(Y2));
  int cw = x1 - x0, np = cw > 0 && y1 > y0 ? cw * (y1 - y0) : 0;
  int area = 0;
  for (int p = threadIdx.x; p < np; p += blockDim.x) {
    int y = y0 + p / cw, x = x0 + p % cw;
    int64_t pix = (int64_t(d.b) * mh + y) * mw + x;
    const act_t* pv = proto + pix * nm;
    float z = 0.f;
    for (int j = 0; j < nm; ++j) z += c[j] * to_f(pv[j]);
    if (z > 0.f) {
      ++area;
      int v = gm[pix];
      if (v > 0) atomicAdd(&sh[v - 1], 1);
    }
  }
  atomicAdd(&sh[G], area);
  __syncthreads();
  for (int i = threadIdx.x; i < G; i += blockDim.x) inter[int64_t(blockIdx.x) * G + i] = sh[i];
  if (threadIdx.x == 0) parea[blockIdx.x] = sh[G];
}
}  // namespace

void mask_inter(const MaskQuery* q, int nq, const act_t* coef0, const act_t* coef1, const act_t* coef2,
                const act_t* proto, int mh, int mw, int nm, int imgsz, const int16_t* gm, int G, int* parea, int* inter,
                cudaStream_t s) {
  if (nq == 0) return;
  YT_ASSERT(nm <= 32, "nm %d", nm);
  mask_inter_k<<<nq, 256, (G + 1) * sizeof(int), s>>>(q, coef0, coef1, coef2, proto, mh, mw, nm, imgsz, gm, G, parea,
                                                      inter);
}
