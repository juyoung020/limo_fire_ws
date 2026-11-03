// 결정적 축소 커널 (원자 연산 없음, 고정 순서)
#include <algorithm>
#include <map>
#include <vector>

#include "common.cuh"

namespace XNET_NS {

namespace {
constexpr int RT = 256;

// 열 합 1 단계: 블록마다 행 묶음 하나의 부분합. C < 256 이면 스레드 256 개를 (묶음 g, 열 c) 로 나눠 연속 읽기.
// ELU: x 를 제자리에서 ELU'(a) 배 한 뒤 더한다 (역방향 ELU 미분 + 편향 기울기 합을 한 번 읽기로). a = 그 층 출력
template <bool ELU>
__global__ void k_colsum_part(real* x, const real* a_out, bool broken, long R, int C, int ld, long rpb, real* part) {
  __shared__ real sh[256];
  int t = threadIdx.x;
  bool wide = C >= 256;
  int G = wide ? 1 : 256 / C;
  int c = wide ? blockIdx.y * 256 + t : t % C;
  int g = wide ? 0 : t / C;
  bool active = wide ? c < C : g < G;
  long r0 = blockIdx.x * rpb, r1 = r0 + rpb < R ? r0 + rpb : R;
  real a = 0;
  if (active)
    for (long r = r0 + g; r < r1; r += G) {
      real v = x[r * ld + c];
      if (ELU) {
        real av = a_out[r * ld + c];
        v *= (av > (real)0 || broken) ? (real)1 : av + (real)1;
        x[r * ld + c] = v;
      }
      a += v;
    }
  sh[t] = a;
  __syncthreads();
  if (wide) {
    if (c < C) part[(long)blockIdx.x * C + c] = a;
  } else if (g == 0) {
    real s = 0;
    for (int k = 0; k < G; ++k) s += sh[k * C + c];
    part[(long)blockIdx.x * C + c] = s;
  }
}
// 2 단계: 열마다 부분합을 블록 순서대로
__global__ void k_colsum_fin(const real* part, int nb, int C, real* out, bool acc) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real s = 0;
  for (int b = 0; b < nb; ++b) s += part[(long)b * C + c];
  out[c] = acc ? out[c] + s : s;
}


// 블록마다 부분합 (double), 마지막은 host 에서 순서대로
template <bool SQ>
__global__ void k_partial(const real* x, long n, double* part) {
  __shared__ double sh[RT];
  double a = 0;
  for (long i = blockIdx.x * (long)RT + threadIdx.x; i < n; i += (long)gridDim.x * RT) {
    double v = x[i];
    a += SQ ? v * v : v;
  }
  sh[threadIdx.x] = a;
  __syncthreads();
  for (int s = RT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

template <bool SQ>
double reduce(const real* x, long n, cudaStream_t s) {
  constexpr int NB = 512;
  static thread_local std::map<cudaStream_t, double*> parts;  // 스트림마다 따로 (actor·critic 이 두 스트림에서 같이 돈다)
  double*& d_part = parts[s];
  if (!d_part) CK(cudaMalloc(&d_part, NB * sizeof(double)));
  k_partial<SQ><<<NB, RT, 0, s>>>(x, n, d_part);
  std::vector<double> h(NB);
  CK(cudaMemcpyAsync(h.data(), d_part, NB * sizeof(double), cudaMemcpyDeviceToHost, s));
  CK(cudaStreamSynchronize(s));
  double a = 0;
  for (double v : h) a += v;
  return a;
}
}  // namespace

void col_sum(const real* x, long R, int C, int ld, real* out, bool acc, cudaStream_t s, const real* a_out, bool broken) {
  constexpr int MAXB = 256;
  struct Part {
    real* p = nullptr;
    long cap = 0;
  };
  static thread_local std::map<cudaStream_t, Part> parts;  // 스트림마다 따로 (actor·critic 이 두 스트림에서 같이 돈다)
  real*& d_part = parts[s].p;
  long& cap = parts[s].cap;
  int nb = (int)std::min<long>(MAXB, std::max<long>(1, R / 256));
  long rpb = (R + nb - 1) / nb;
  if ((long)nb * C > cap) {
    if (d_part) CK(cudaFree(d_part));
    cap = (long)nb * C;
    CK(cudaMalloc(&d_part, cap * sizeof(real)));
  }
  dim3 grid(nb, C >= 256 ? (C + 255) / 256 : 1);
  if (a_out)
    k_colsum_part<true><<<grid, 256, 0, s>>>(const_cast<real*>(x), a_out, broken, R, C, ld, rpb, d_part);
  else
    k_colsum_part<false><<<grid, 256, 0, s>>>(const_cast<real*>(x), nullptr, false, R, C, ld, rpb, d_part);
  k_colsum_fin<<<nblk(C, 128), 128, 0, s>>>(d_part, nb, C, out, acc);
}
double sum_all(const real* x, long n, cudaStream_t s) { return reduce<false>(x, n, s); }
double sumsq_all(const real* x, long n, cudaStream_t s) { return reduce<true>(x, n, s); }

}  // namespace XNET_NS
