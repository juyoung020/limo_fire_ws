// 그래프 어텐션 신경망 — 희소(유효 노드 행, 키 25 개) 정방향·역방향. 구조·출처는 graph_net.cuh, network/README.md.
#include <cmath>
#include <stdexcept>

#include "actor_critic.h"
#include "graph_net.cuh"

namespace XNET_NS {

bool g_break_graph_grad = false;

namespace {
constexpr real kLnEps = (real)1e-5;   // torch nn.LayerNorm 기본
constexpr real kTanhClip = (real)10;  // model.py SingleHeadAttention.tanh_clipping
constexpr int kMaxDk = 32;            // 머리 하나의 차원 상한 (커널 안 고정 배열)

// ---------------- 색인 ----------------
__global__ void k_g_count(const uint8_t* feat, int F, int N, const int* idx, const int* cur, int n, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long r = idx ? idx[i] : i;
  const uint8_t* f = feat + r * N * F;
  int c = 0;
  for (int j = 0; j < N; ++j) c += f[(long)j * F] != 0;
  const int cu = cur[r];
  cnt[i] = (cu < 0 || cu >= N || f[(long)cu * F] == 0) ? -1 : c;
}
// 샘플마다 유효 노드를 자리 순으로 행 번호 매김
__global__ void k_g_map(const uint8_t* feat, int F, int N, const int* idx, const int* row_start, int n, int* node2row, int* rsamp,
                        int* rnode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long r = idx ? idx[i] : i;
  const uint8_t* f = feat + r * N * F;
  int row = row_start[i];
  for (int j = 0; j < N; ++j) {
    if (f[(long)j * F] != 0) {
      node2row[(long)i * N + j] = row;
      rsamp[row] = i;
      rnode[row] = j;
      ++row;
    } else {
      node2row[(long)i * N + j] = -1;
    }
  }
}
// 행마다: 임베딩 입력 X0 = u8/255 (유효 칸 빼고), 스텐실 이웃 행 gnb (없으면 −1), 간선 비트(이웃이 있는 것만)
__global__ void k_g_rows(const uint8_t* feat, const uint32_t* edge, const int* idx, int F, int gw, int gh, const int* node2row,
                         const int* rsamp, const int* rnode, long M, real* X0, int* gnb, uint32_t* ebits) {
  long row = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (row >= M) return;
  const int i = rsamp[row], j = rnode[row], N = gw * gh;
  const long r = idx ? idx[i] : i;
  const uint8_t* f = feat + (r * N + j) * F;
  for (int c = 1; c < F; ++c) X0[row * (F - 1) + c - 1] = (real)f[c] / (real)255;
  uint32_t e = edge[r * N + j] & 0xFFFFFFu;
  const int y = j / gw, x = j % gw;
  for (int k = 0; k < kGNb; ++k) {
    const int i5 = k >= 12 ? k + 1 : k;
    const int ny = y + i5 / 5 - 2, nx = x + i5 % 5 - 2;
    int r2 = -1;
    if (ny >= 0 && ny < gh && nx >= 0 && nx < gw) r2 = node2row[(long)i * N + ny * gw + nx];
    gnb[row * kGNb + k] = r2;
    if (r2 < 0) e &= ~(1u << k);
  }
  ebits[row] = e;
}
__global__ void k_g_cur(const int* cur, const int* idx, const int* node2row, const int* gnb, const uint32_t* ebits, int n, int N,
                        int* rowcur, int* nbrow) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long r = idx ? idx[i] : i;
  const int rc = node2row[(long)i * N + cur[r]];
  rowcur[i] = rc;
  const uint32_t e = ebits[rc];
  for (int k = 0; k < kGNb; ++k) nbrow[i * kGNb + k] = ((e >> k) & 1u) ? gnb[(long)rc * kGNb + k] : -1;
}

// ---------------- 작은 도구 ----------------
__global__ void k_bias(real* Y, long R, int C, const real* b, bool relu) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= R * C) return;
  real v = Y[t] + b[t % C];
  Y[t] = relu ? (v > (real)0 ? v : (real)0) : v;
}
__global__ void k_relu_back(real* dY, const real* Y, long n) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t < n && !(Y[t] > (real)0)) dY[t] = 0;
}
// dst [n][d] = src [rows[i]][d]
__global__ void k_gather_rows(const real* src, const int* rows, int n, int d, real* dst) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * d) return;
  dst[t] = src[(long)rows[t / d] * d + t % d];
}
// dst [rows[i]][d] += a [i] (+ b [i]) — 샘플마다 행이 달라 겹치지 않는다
__global__ void k_scatter_rows(real* dst, const int* rows, int n, int d, const real* a, long lda, const real* b, long ldb) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * d) return;
  const int i = (int)(t / d), c = (int)(t % d);
  real v = a[(long)i * lda + c];
  if (b) v += b[(long)i * ldb + c];
  dst[(long)rows[i] * d + c] += v;
}
// 이웃 특징: Hn [n][24][d] = X [nbrow] (없으면 0)
__global__ void k_gather_nbr(const real* X, const int* nbrow, int n, int d, real* Hn) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * kGNb * d) return;
  const int r = nbrow[t / d];
  Hn[t] = r < 0 ? (real)0 : X[(long)r * d + t % d];
}
// 한 샘플의 이웃 24 칸은 서로 다른 행, 샘플끼리도 다른 행 → 겹치지 않는다
__global__ void k_scatter_nbr(real* dX, const int* nbrow, int n, int d, const real* dHn) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * kGNb * d) return;
  const int r = nbrow[t / d];
  if (r >= 0) dX[(long)r * d + t % d] += dHn[t];
}
// dst [n][d] = a (+ b), a·b 가 없으면 0
__global__ void k_sum2(const real* a, long lda, const real* b, long ldb, int n, int d, real* dst) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * d) return;
  const int i = (int)(t / d), c = (int)(t % d);
  real v = a ? a[(long)i * lda + c] : (real)0;
  if (b) v += b[(long)i * ldb + c];
  dst[t] = v;
}

// ---------------- 층 정규화 (행마다 스레드 하나) ----------------
// xh = (x − 평균)·rs, rs = 1/√(분산 + eps) (분산은 d 로 나눔 — torch), A = g·xh + b
__global__ void k_ln_fwd(const real* X, long R, int d, const real* g, const real* b, real* xh, real* rs, real* A) {
  long r = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (r >= R) return;
  const real* x = X + r * d;
  real m = 0;
  for (int c = 0; c < d; ++c) m += x[c];
  m /= (real)d;
  real v = 0;
  for (int c = 0; c < d; ++c) v += (x[c] - m) * (x[c] - m);
  const real s = (real)1 / sqrt(v / (real)d + kLnEps);
  rs[r] = s;
  for (int c = 0; c < d; ++c) {
    const real h = (x[c] - m) * s;
    xh[r * d + c] = h;
    A[r * d + c] = g[c] * h + b[c];
  }
}
__global__ void k_ln_out(const real* xh, long R, int d, const real* g, const real* b, real* A) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= R * d) return;
  const int c = (int)(t % d);
  A[t] = g[c] * xh[t] + b[c];
}
// dX (+)= rs·(gy − mean(gy) − xh·mean(gy·xh)), gy = dA·g. T = dA·xh (g 기울기의 열 합 재료)
__global__ void k_ln_bwd(const real* dA, const real* xh, const real* rs, const real* g, long R, int d, real* dX, bool acc, real* T,
                         bool broken) {
  long r = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (r >= R) return;
  real m1 = 0, m2 = 0;
  for (int c = 0; c < d; ++c) {
    const real gy = dA[r * d + c] * g[c];
    m1 += gy;
    m2 += gy * xh[r * d + c];
  }
  m1 /= (real)d;
  m2 /= (real)d;
  if (broken) m2 = 0;
  const real s = rs[r];
  for (int c = 0; c < d; ++c) {
    const real h = xh[r * d + c];
    const real v = s * (dA[r * d + c] * g[c] - m1 - h * m2);
    T[r * d + c] = dA[r * d + c] * h;
    dX[r * d + c] = acc ? dX[r * d + c] + v : v;
  }
}

// ---------------- 인코더 어텐션 (행·머리마다 스레드 하나, 키 = 자신 + 간선 이웃) ----------------
// QKV [R][3d]: 질의 | 키 | 값, 머리 h 는 열 h·dk … 안. W [R][H][25] = 주의 가중치(가린 칸 0), O [R][d] = 머리별 Σ w·v
__global__ void k_attn_fwd(const real* QKV, const int* gnb, const uint32_t* eb, long R, int d, int H, real* W, real* O) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= R * H) return;
  const long i = t / H;
  const int h = (int)(t % H), dk = d / H;
  const real* q = QKV + i * 3 * d + h * dk;
  const real nrm = (real)1 / sqrt((real)dk);
  const uint32_t e = eb[i];
  real u[kGSlot];
  long js[kGSlot];
  real mx = (real)-1e30;
  for (int s = 0; s < kGSlot; ++s) {
    const long j = s == 0 ? i : (((e >> (s - 1)) & 1u) ? gnb[i * kGNb + s - 1] : -1);
    js[s] = j;
    if (j < 0) continue;
    const real* k = QKV + j * 3 * d + d + h * dk;
    real a = 0;
    for (int c = 0; c < dk; ++c) a += q[c] * k[c];
    u[s] = a * nrm;
    mx = fmax(mx, u[s]);
  }
  real sum = 0;
  for (int s = 0; s < kGSlot; ++s)
    if (js[s] >= 0) {
      u[s] = exp(u[s] - mx);
      sum += u[s];
    }
  real o[kMaxDk];
  for (int c = 0; c < dk; ++c) o[c] = 0;
  real* w = W + t * kGSlot;
  for (int s = 0; s < kGSlot; ++s) {
    if (js[s] < 0) {
      w[s] = 0;
      continue;
    }
    const real ws = u[s] / sum;
    w[s] = ws;
    const real* v = QKV + js[s] * 3 * d + 2 * d + h * dk;
    for (int c = 0; c < dk; ++c) o[c] += ws * v[c];
  }
  for (int c = 0; c < dk; ++c) O[i * d + h * dk + c] = o[c];
}
// 역방향 1 (질의 쪽): du_s = w_s (dw_s − Σ w dw)/√dk, dw_s = dO·v_s → DU [R][H][25], dQ = Σ du_s k_s
__global__ void k_attn_bwd1(const real* QKV, const int* gnb, const uint32_t* eb, const real* W, const real* dO, long R, int d, int H,
                            real* DU, real* dQKV) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= R * H) return;
  const long i = t / H;
  const int h = (int)(t % H), dk = d / H;
  const real nrm = (real)1 / sqrt((real)dk);
  const uint32_t e = eb[i];
  const real* w = W + t * kGSlot;
  const real* g = dO + i * d + h * dk;
  real dw[kGSlot];
  long js[kGSlot];
  real mean = 0;
  for (int s = 0; s < kGSlot; ++s) {
    const long j = s == 0 ? i : (((e >> (s - 1)) & 1u) ? gnb[i * kGNb + s - 1] : -1);
    js[s] = j;
    dw[s] = 0;
    if (j < 0) continue;
    const real* v = QKV + j * 3 * d + 2 * d + h * dk;
    real a = 0;
    for (int c = 0; c < dk; ++c) a += g[c] * v[c];
    dw[s] = a;
    mean += w[s] * a;
  }
  real dq[kMaxDk];
  for (int c = 0; c < dk; ++c) dq[c] = 0;
  real* du = DU + t * kGSlot;
  for (int s = 0; s < kGSlot; ++s) {
    if (js[s] < 0) {
      du[s] = 0;
      continue;
    }
    const real x = w[s] * (dw[s] - mean) * nrm;
    du[s] = x;
    const real* k = QKV + js[s] * 3 * d + d + h * dk;
    for (int c = 0; c < dk; ++c) dq[c] += x * k[c];
  }
  for (int c = 0; c < dk; ++c) dQKV[i * 3 * d + h * dk + c] = dq[c];
}
// 역방향 2 (키·값 쪽, 모으기): 행 j 를 키로 본 질의 i = 자신 + 반대 스텐실 이웃(k' 번 이웃 i 가 23 − k' 번으로 j 를 본다) 을 고정 순서로
__global__ void k_attn_bwd2(const real* QKV, const int* gnb, const uint32_t* eb, const real* W, const real* DU, const real* dO, long R,
                            int d, int H, real* dQKV) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= R * H) return;
  const long j = t / H;
  const int h = (int)(t % H), dk = d / H;
  real dkk[kMaxDk], dv[kMaxDk];
  {
    const real du = DU[t * kGSlot], w = W[t * kGSlot];
    const real* q = QKV + j * 3 * d + h * dk;
    const real* g = dO + j * d + h * dk;
    for (int c = 0; c < dk; ++c) {
      dkk[c] = du * q[c];
      dv[c] = w * g[c];
    }
  }
  for (int kp = 0; kp < kGNb; ++kp) {
    const long i = gnb[j * kGNb + kp];
    if (i < 0) continue;
    const int k = kGNb - 1 - kp;
    if (!((eb[i] >> k) & 1u)) continue;
    const long ti = (i * H + h) * kGSlot + 1 + k;
    const real du = DU[ti], w = W[ti];
    const real* q = QKV + i * 3 * d + h * dk;
    const real* g = dO + i * d + h * dk;
    for (int c = 0; c < dk; ++c) {
      dkk[c] += du * q[c];
      dv[c] += w * g[c];
    }
  }
  for (int c = 0; c < dk; ++c) {
    dQKV[j * 3 * d + d + h * dk + c] = dkk[c];
    dQKV[j * 3 * d + 2 * d + h * dk + c] = dv[c];
  }
}

// ---------------- 디코더 교차 어텐션 (샘플·머리마다 스레드 하나, 키 = 그 샘플의 모든 행) ----------------
// Qc [n][d], KV [R][2d] (키 | 값). Wd [R][H] = 주의 가중치, Od [n][d]
__global__ void k_dec_fwd(const real* Qc, const real* KV, const int* row_start, int n, int d, int H, real* Wd, real* Od) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * H) return;
  const int b = t / H, h = t % H, dk = d / H;
  const real nrm = (real)1 / sqrt((real)dk);
  const real* q = Qc + (long)b * d + h * dk;
  const long r0 = row_start[b], r1 = row_start[b + 1];
  real mx = (real)-1e30;
  for (long r = r0; r < r1; ++r) {
    const real* k = KV + r * 2 * d + h * dk;
    real a = 0;
    for (int c = 0; c < dk; ++c) a += q[c] * k[c];
    a *= nrm;
    Wd[r * H + h] = a;
    mx = fmax(mx, a);
  }
  real sum = 0;
  for (long r = r0; r < r1; ++r) {
    const real e = exp(Wd[r * H + h] - mx);
    Wd[r * H + h] = e;
    sum += e;
  }
  real o[kMaxDk];
  for (int c = 0; c < dk; ++c) o[c] = 0;
  for (long r = r0; r < r1; ++r) {
    const real w = Wd[r * H + h] / sum;
    Wd[r * H + h] = w;
    const real* v = KV + r * 2 * d + d + h * dk;
    for (int c = 0; c < dk; ++c) o[c] += w * v[c];
  }
  for (int c = 0; c < dk; ++c) Od[(long)b * d + h * dk + c] = o[c];
}
// dOd [n][d] → dQc [n][d], dKV [R][2d]
__global__ void k_dec_bwd(const real* Qc, const real* KV, const real* Wd, const real* dOd, const int* row_start, int n, int d, int H,
                          real* dQc, real* dKV) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * H) return;
  const int b = t / H, h = t % H, dk = d / H;
  const real nrm = (real)1 / sqrt((real)dk);
  const real* q = Qc + (long)b * d + h * dk;
  const real* g = dOd + (long)b * d + h * dk;
  const long r0 = row_start[b], r1 = row_start[b + 1];
  real mean = 0;
  for (long r = r0; r < r1; ++r) {
    const real* v = KV + r * 2 * d + d + h * dk;
    real a = 0;
    for (int c = 0; c < dk; ++c) a += g[c] * v[c];
    mean += Wd[r * H + h] * a;
  }
  real dq[kMaxDk];
  for (int c = 0; c < dk; ++c) dq[c] = 0;
  for (long r = r0; r < r1; ++r) {
    const real* k = KV + r * 2 * d + h * dk;
    const real* v = k + d;
    const real w = Wd[r * H + h];
    real a = 0;
    for (int c = 0; c < dk; ++c) a += g[c] * v[c];
    const real du = w * (a - mean) * nrm;
    for (int c = 0; c < dk; ++c) {
      dq[c] += du * k[c];
      dKV[r * 2 * d + h * dk + c] = du * q[c];
      dKV[r * 2 * d + d + h * dk + c] = w * g[c];
    }
  }
  for (int c = 0; c < dk; ++c) dQc[(long)b * d + h * dk + c] = dq[c];
}

// ---------------- 포인터 머리 ----------------
__global__ void k_ptr_cat(const real* E, const real* Hc, const real* glob, long ldg, int n, int d, int G, real* cat) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int C = 2 * d + G;
  if (t >= (long)n * C) return;
  const int i = (int)(t / C), c = (int)(t % C);
  cat[t] = c < d ? E[(long)i * d + c] : (c < 2 * d ? Hc[(long)i * d + c - d] : glob[(long)i * ldg + c - 2 * d]);
}
// U [n][24] = tanh(q·k/√d), 로짓 = 10·U
__global__ void k_ptr_score(const real* Qp, const real* Kp, int n, int d, real* U, real* logits) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * kGNb) return;
  const real* q = Qp + (long)(t / kGNb) * d;
  const real* k = Kp + (long)t * d;
  real a = 0;
  for (int c = 0; c < d; ++c) a += q[c] * k[c];
  const real u = tanh(a / sqrt((real)d));
  U[t] = u;
  logits[t] = kTanhClip * u;
}
// dKp [n·24][d] = du·q, du = dl·10·(1 − U²)/√d
__global__ void k_ptr_bwd_k(const real* Qp, const real* U, const real* dl, int n, int d, real* dKp) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * kGNb * d) return;
  const long bk = t / d;
  const real du = dl[bk] * kTanhClip * ((real)1 - U[bk] * U[bk]) / sqrt((real)d);
  dKp[t] = du * Qp[(bk / kGNb) * d + t % d];
}
// dQp [n][d] = Σ_k du_k·Kp_k (k 순서 고정)
__global__ void k_ptr_bwd_q(const real* Kp, const real* U, const real* dl, int n, int d, real* dQp) {
  long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (t >= (long)n * d) return;
  const long b = t / d;
  const int c = (int)(t % d);
  real a = 0;
  for (int k = 0; k < kGNb; ++k) {
    const long bk = b * kGNb + k;
    a += dl[bk] * kTanhClip * ((real)1 - U[bk] * U[bk]) / sqrt((real)d) * Kp[bk * d + c];
  }
  dQp[t] = a;
}

void uni(std::vector<real>& host, long off, long n, float bound, std::mt19937_64& rng) {
  std::uniform_real_distribution<float> U(-bound, bound);
  for (long i = 0; i < n; ++i) host[off + i] = U(rng);
}
void fill(std::vector<real>& host, long off, long n, real v) {
  for (long i = 0; i < n; ++i) host[off + i] = v;
}
long reg1(std::vector<ParamBlock>& reg, long& cur, long n, PKind kind, const std::string& name) {
  const long off = cur;
  reg.push_back({cur, n, kind, name});
  cur += n;
  return off;
}
}  // namespace

// ============================== GraphBody ==============================
void GraphBody::build(const GraphSpec& s, std::vector<ParamBlock>& reg, long& cur, const std::string& pfx) {
  if (s.d % s.heads || s.d / s.heads > kMaxDk || s.feat < 2 || s.layers < 1) throw std::runtime_error("GraphSpec: 차원·머리 수");
  s_ = s;
  fin_ = s.feat - 1;
  const long d = s.d, ff = s.ff;
  emb_w_ = reg1(reg, cur, fin_ * d, PKind::LinW, pfx + ".emb.w");   // 모든 W 는 [입력][출력] (y = x·W)
  emb_b_ = reg1(reg, cur, d, PKind::LinB, pfx + ".emb.b");
  L_.assign(s.layers, Layer{});
  for (int l = 0; l < s.layers; ++l) {
    const std::string nm = pfx + ".enc" + std::to_string(l);
    Layer& L = L_[l];
    L.ln1_g = reg1(reg, cur, d, PKind::LinW, nm + ".ln1.g");
    L.ln1_b = reg1(reg, cur, d, PKind::LinB, nm + ".ln1.b");
    L.wqkv = reg1(reg, cur, d * 3 * d, PKind::LinW, nm + ".wqkv");   // [d][질의 d | 키 d | 값 d], 머리 h = 열 h·dk …
    L.wout = reg1(reg, cur, d * d, PKind::LinW, nm + ".wout");
    L.ln2_g = reg1(reg, cur, d, PKind::LinW, nm + ".ln2.g");
    L.ln2_b = reg1(reg, cur, d, PKind::LinB, nm + ".ln2.b");
    L.w1 = reg1(reg, cur, d * ff, PKind::LinW, nm + ".ff1.w");
    L.b1 = reg1(reg, cur, ff, PKind::LinB, nm + ".ff1.b");
    L.w2 = reg1(reg, cur, ff * d, PKind::LinW, nm + ".ff2.w");
    L.b2 = reg1(reg, cur, d, PKind::LinB, nm + ".ff2.b");
  }
  const std::string nm = pfx + ".dec";
  dln1_g_ = reg1(reg, cur, d, PKind::LinW, nm + ".ln1.g");
  dln1_b_ = reg1(reg, cur, d, PKind::LinB, nm + ".ln1.b");
  dwq_ = reg1(reg, cur, d * d, PKind::LinW, nm + ".wq");
  dwkv_ = reg1(reg, cur, d * 2 * d, PKind::LinW, nm + ".wkv");
  dwout_ = reg1(reg, cur, d * d, PKind::LinW, nm + ".wout");
  dln2_g_ = reg1(reg, cur, d, PKind::LinW, nm + ".ln2.g");
  dln2_b_ = reg1(reg, cur, d, PKind::LinB, nm + ".ln2.b");
  dw1_ = reg1(reg, cur, d * ff, PKind::LinW, nm + ".ff1.w");
  db1_ = reg1(reg, cur, ff, PKind::LinB, nm + ".ff1.b");
  dw2_ = reg1(reg, cur, ff * d, PKind::LinW, nm + ".ff2.w");
  db2_ = reg1(reg, cur, d, PKind::LinB, nm + ".ff2.b");
}

void GraphBody::init(std::mt19937_64& rng, std::vector<real>& host) const {
  const long d = s_.d, ff = s_.ff;
  const float bq = 1.f / std::sqrt((float)(s_.d / s_.heads));   // model.py w_query·w_key·w_value: 마지막 차원 = 머리 차원
  const float bd = 1.f / std::sqrt((float)d), bf = 1.f / std::sqrt((float)ff), be = 1.f / std::sqrt((float)fin_);
  uni(host, emb_w_, fin_ * d, be, rng);
  uni(host, emb_b_, d, be, rng);
  auto ln = [&](long g, long b) {
    fill(host, g, d, (real)1);
    fill(host, b, d, (real)0);
  };
  auto ffi = [&](long w1, long b1, long w2, long b2) {
    uni(host, w1, d * ff, bd, rng);
    uni(host, b1, ff, bd, rng);
    uni(host, w2, ff * d, bf, rng);
    uni(host, b2, d, bf, rng);
  };
  for (const Layer& L : L_) {
    ln(L.ln1_g, L.ln1_b);
    uni(host, L.wqkv, d * 3 * d, bq, rng);
    uni(host, L.wout, d * d, bd, rng);   // model.py w_out: 마지막 차원 = d
    ln(L.ln2_g, L.ln2_b);
    ffi(L.w1, L.b1, L.w2, L.b2);
  }
  ln(dln1_g_, dln1_b_);
  uni(host, dwq_, d * d, bq, rng);
  uni(host, dwkv_, d * 2 * d, bq, rng);
  uni(host, dwout_, d * d, bd, rng);
  ln(dln2_g_, dln2_b_);
  ffi(dw1_, db1_, dw2_, db2_);
}

void GraphBody::bind(real* params, real* grads, long cap_rows, int max_n) {
  P_ = params;
  G_ = grads;
  cap_ = cap_rows;
  max_n_ = max_n;
  ws_bytes_ = 0;
  auto mk = [&](long n) {
    real* p;
    CK(cudaMalloc(&p, n * sizeof(real)));
    ws_bytes_ += n * sizeof(real);
    owned_.push_back(p);
    return p;
  };
  auto mki = [&](long n) {
    int* p;
    CK(cudaMalloc(&p, n * sizeof(int)));
    ws_bytes_ += n * sizeof(int);
    owned_.push_back(p);
    return p;
  };
  const long R = cap_rows, n = max_n, d = s_.d, ff = s_.ff, H = s_.heads, N = (long)s_.gw * s_.gh;
  row_start_ = mki(n + 1);
  node2row_ = mki(n * N);
  rsamp_ = mki(R);
  rnode_ = mki(R);
  gnb_ = mki(R * kGNb);
  rowcur_ = mki(n);
  nbrow_ = mki(n * kGNb);
  ebits_ = (uint32_t*)mki(R);
  X0_ = mk(R * fin_);
  X_ = mk(R * d);
  A_ = mk(R * d);
  dX_ = mk(R * d);
  dA_ = mk(R * d);
  dQKV_ = mk(R * 3 * d);
  DU_ = mk(R * H * kGSlot);
  dHID_ = mk(R * ff);
  T_ = mk(R * d);
  for (Layer& L : L_) {
    L.xh1 = mk(R * d);
    L.rs1 = mk(R);
    L.QKV = mk(R * 3 * d);
    L.W = mk(R * H * kGSlot);
    L.O = mk(R * d);
    L.xh2 = mk(R * d);
    L.rs2 = mk(R);
    L.HID = mk(R * ff);
  }
  dxh_ = mk(R * d);
  drs_ = mk(R);
  KV_ = mk(R * 2 * d);
  Wd_ = mk(R * H);
  for (real** p : {&Ac_, &Qc_, &Od_, &h1_, &xh2d_, &E_, &Hc_, &An_, &gE_, &dh1_, &dAn_, &dOd_, &dQc_, &dAc_, &Tn_, &gHc_}) *p = mk(n * d);
  rs2d_ = mk(n);
  hidd_ = mk(n * ff);
  dhidd_ = mk(n * ff);
  Hn_ = mk(n * kGNb * d);
}

void GraphBody::free_ws() {
  for (void* p : owned_) cudaFree(p);
  owned_.clear();
}

void graph_count(const GraphSpec& s, cudaStream_t st, int n, const GraphIn& in, int* d_cnt, int* cnt_host) {
  k_g_count<<<nblk(n, 64), 64, 0, st>>>(in.feat, s.feat, s.gw * s.gh, in.idx, in.cur, n, d_cnt);
  CK(cudaMemcpyAsync(cnt_host, d_cnt, n * sizeof(int), cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
}


void GraphBody::forward(cublasHandle_t h, cudaStream_t st, int n, const GraphIn& in, const int* cnt) {
  if (n > max_n_) throw std::runtime_error("GraphBody::forward: n > max_n");
  h_start_.resize(n + 1);
  long M = 0;
  for (int i = 0; i < n; ++i) {
    if (cnt[i] < 1) throw std::runtime_error("GraphBody::forward: 현재 노드가 무효이거나 유효 노드가 없는 샘플");
    h_start_[i] = (int)M;
    M += cnt[i];
  }
  h_start_[n] = (int)M;
  if (M > cap_) throw std::runtime_error("GraphBody::forward: 유효 노드 합이 cap_rows 를 넘음");
  M_ = M;
  CK(cudaMemcpyAsync(row_start_, h_start_.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, st));
  const int d = s_.d, H = s_.heads, ff = s_.ff, N = s_.gw * s_.gh;
  const int Mi = (int)M;
  k_g_map<<<nblk(n, 64), 64, 0, st>>>(in.feat, s_.feat, N, in.idx, row_start_, n, node2row_, rsamp_, rnode_);
  k_g_rows<<<nblk(M), 256, 0, st>>>(in.feat, in.edge, in.idx, s_.feat, s_.gw, s_.gh, node2row_, rsamp_, rnode_, M, X0_, gnb_, ebits_);
  k_g_cur<<<nblk(n, 64), 64, 0, st>>>(in.cur, in.idx, node2row_, gnb_, ebits_, n, N, rowcur_, nbrow_);
  // 초기 임베딩
  gemm(h, false, false, Mi, d, fin_, 1, X0_, fin_, P_ + emb_w_, d, 0, X_, d);
  k_bias<<<nblk(M * d), 256, 0, st>>>(X_, M, d, P_ + emb_b_, false);
  // 인코더 (X_ 제자리: 잔차를 GEMM 의 beta = 1 로)
  for (Layer& L : L_) {
    k_ln_fwd<<<nblk(M), 256, 0, st>>>(X_, M, d, P_ + L.ln1_g, P_ + L.ln1_b, L.xh1, L.rs1, A_);
    gemm(h, false, false, Mi, 3 * d, d, 1, A_, d, P_ + L.wqkv, 3 * d, 0, L.QKV, 3 * d);
    k_attn_fwd<<<nblk(M * H), 256, 0, st>>>(L.QKV, gnb_, ebits_, M, d, H, L.W, L.O);
    gemm(h, false, false, Mi, d, d, 1, L.O, d, P_ + L.wout, d, 1, X_, d);
    k_ln_fwd<<<nblk(M), 256, 0, st>>>(X_, M, d, P_ + L.ln2_g, P_ + L.ln2_b, L.xh2, L.rs2, A_);
    gemm(h, false, false, Mi, ff, d, 1, A_, d, P_ + L.w1, ff, 0, L.HID, ff);
    k_bias<<<nblk(M * ff), 256, 0, st>>>(L.HID, M, ff, P_ + L.b1, true);
    gemm(h, false, false, Mi, d, ff, 1, L.HID, ff, P_ + L.w2, d, 1, X_, d);
    k_bias<<<nblk(M * d), 256, 0, st>>>(X_, M, d, P_ + L.b2, false);
  }
  // 디코더: 정규화 1 은 질의(현재 노드)·기억(모든 노드) 공용 → 질의 = 정규화된 현재 노드 행
  k_ln_fwd<<<nblk(M), 256, 0, st>>>(X_, M, d, P_ + dln1_g_, P_ + dln1_b_, dxh_, drs_, A_);
  gemm(h, false, false, Mi, 2 * d, d, 1, A_, d, P_ + dwkv_, 2 * d, 0, KV_, 2 * d);
  k_gather_rows<<<nblk((long)n * d), 256, 0, st>>>(A_, rowcur_, n, d, Ac_);
  k_gather_rows<<<nblk((long)n * d), 256, 0, st>>>(X_, rowcur_, n, d, Hc_);
  gemm(h, false, false, n, d, d, 1, Ac_, d, P_ + dwq_, d, 0, Qc_, d);
  k_dec_fwd<<<nblk((long)n * H, 64), 64, 0, st>>>(Qc_, KV_, row_start_, n, d, H, Wd_, Od_);
  CK(cudaMemcpyAsync(h1_, Hc_, (size_t)n * d * sizeof(real), cudaMemcpyDeviceToDevice, st));
  gemm(h, false, false, n, d, d, 1, Od_, d, P_ + dwout_, d, 1, h1_, d);
  k_ln_fwd<<<nblk(n), 256, 0, st>>>(h1_, n, d, P_ + dln2_g_, P_ + dln2_b_, xh2d_, rs2d_, An_);
  gemm(h, false, false, n, ff, d, 1, An_, d, P_ + dw1_, ff, 0, hidd_, ff);
  k_bias<<<nblk((long)n * ff), 256, 0, st>>>(hidd_, n, ff, P_ + db1_, true);
  CK(cudaMemcpyAsync(E_, h1_, (size_t)n * d * sizeof(real), cudaMemcpyDeviceToDevice, st));
  gemm(h, false, false, n, d, ff, 1, hidd_, ff, P_ + dw2_, d, 1, E_, d);
  k_bias<<<nblk((long)n * d), 256, 0, st>>>(E_, n, d, P_ + db2_, false);
  k_gather_nbr<<<nblk((long)n * kGNb * d), 256, 0, st>>>(X_, nbrow_, n, d, Hn_);
}

void GraphBody::backward(cublasHandle_t h, cudaStream_t st, int n, const real* dE, const real* dHc, long ld, const real* dHn,
                         const real* dE2, const real* dHc2, long ld2) {
  const int d = s_.d, H = s_.heads, ff = s_.ff;
  const long M = M_;
  const int Mi = (int)M;
  const bool brk = g_break_graph_grad;
  // ---- 디코더 ----
  k_sum2<<<nblk((long)n * d), 256, 0, st>>>(dE, ld, dE2, ld2, n, d, gE_);
  k_sum2<<<nblk((long)n * d), 256, 0, st>>>(dHc, ld, dHc2, ld2, n, d, gHc_);
  col_sum(gE_, n, d, d, G_ + db2_, true, st);
  gemm(h, true, false, ff, d, n, 1, hidd_, ff, gE_, d, 1, G_ + dw2_, d);
  gemm(h, false, true, n, ff, d, 1, gE_, d, P_ + dw2_, d, 0, dhidd_, ff);
  k_relu_back<<<nblk((long)n * ff), 256, 0, st>>>(dhidd_, hidd_, (long)n * ff);
  col_sum(dhidd_, n, ff, ff, G_ + db1_, true, st);
  k_ln_out<<<nblk((long)n * d), 256, 0, st>>>(xh2d_, n, d, P_ + dln2_g_, P_ + dln2_b_, An_);
  gemm(h, true, false, d, ff, n, 1, An_, d, dhidd_, ff, 1, G_ + dw1_, ff);
  gemm(h, false, true, n, d, ff, 1, dhidd_, ff, P_ + dw1_, ff, 0, dAn_, d);
  CK(cudaMemcpyAsync(dh1_, gE_, (size_t)n * d * sizeof(real), cudaMemcpyDeviceToDevice, st));   // 잔차
  k_ln_bwd<<<nblk(n), 256, 0, st>>>(dAn_, xh2d_, rs2d_, P_ + dln2_g_, n, d, dh1_, true, Tn_, brk);
  col_sum(Tn_, n, d, d, G_ + dln2_g_, true, st);
  col_sum(dAn_, n, d, d, G_ + dln2_b_, true, st);
  gemm(h, true, false, d, d, n, 1, Od_, d, dh1_, d, 1, G_ + dwout_, d);
  gemm(h, false, true, n, d, d, 1, dh1_, d, P_ + dwout_, d, 0, dOd_, d);
  k_dec_bwd<<<nblk((long)n * H, 64), 64, 0, st>>>(Qc_, KV_, Wd_, dOd_, row_start_, n, d, H, dQc_, dQKV_);   // dQKV_ 앞 [R][2d] 를 dKV 로
  gemm(h, true, false, d, d, n, 1, Ac_, d, dQc_, d, 1, G_ + dwq_, d);
  gemm(h, false, true, n, d, d, 1, dQc_, d, P_ + dwq_, d, 0, dAc_, d);
  k_ln_out<<<nblk(M * d), 256, 0, st>>>(dxh_, M, d, P_ + dln1_g_, P_ + dln1_b_, A_);
  gemm(h, true, false, d, 2 * d, Mi, 1, A_, d, dQKV_, 2 * d, 1, G_ + dwkv_, 2 * d);
  gemm(h, false, true, Mi, d, 2 * d, 1, dQKV_, 2 * d, P_ + dwkv_, 2 * d, 0, dA_, d);
  k_scatter_rows<<<nblk((long)n * d), 256, 0, st>>>(dA_, rowcur_, n, d, dAc_, d, nullptr, 0);   // 질의 = 정규화된 현재 노드 행
  k_ln_bwd<<<nblk(M), 256, 0, st>>>(dA_, dxh_, drs_, P_ + dln1_g_, M, d, dX_, false, T_, brk);
  col_sum(T_, M, d, d, G_ + dln1_g_, true, st);
  col_sum(dA_, M, d, d, G_ + dln1_b_, true, st);
  // 현재 노드 특징: 디코더 잔차(dh1) + 머리가 직접 쓴 Hc, 그리고 이웃 특징
  k_scatter_rows<<<nblk((long)n * d), 256, 0, st>>>(dX_, rowcur_, n, d, dh1_, d, gHc_, d);
  if (dHn) k_scatter_nbr<<<nblk((long)n * kGNb * d), 256, 0, st>>>(dX_, nbrow_, n, d, dHn);
  // ---- 인코더 (뒤에서부터) ----
  for (int l = (int)L_.size() - 1; l >= 0; --l) {
    Layer& L = L_[l];
    col_sum(dX_, M, d, d, G_ + L.b2, true, st);
    gemm(h, true, false, ff, d, Mi, 1, L.HID, ff, dX_, d, 1, G_ + L.w2, d);
    gemm(h, false, true, Mi, ff, d, 1, dX_, d, P_ + L.w2, d, 0, dHID_, ff);
    k_relu_back<<<nblk(M * ff), 256, 0, st>>>(dHID_, L.HID, M * ff);
    col_sum(dHID_, M, ff, ff, G_ + L.b1, true, st);
    k_ln_out<<<nblk(M * d), 256, 0, st>>>(L.xh2, M, d, P_ + L.ln2_g, P_ + L.ln2_b, A_);
    gemm(h, true, false, d, ff, Mi, 1, A_, d, dHID_, ff, 1, G_ + L.w1, ff);
    gemm(h, false, true, Mi, d, ff, 1, dHID_, ff, P_ + L.w1, ff, 0, dA_, d);
    k_ln_bwd<<<nblk(M), 256, 0, st>>>(dA_, L.xh2, L.rs2, P_ + L.ln2_g, M, d, dX_, true, T_, brk);
    col_sum(T_, M, d, d, G_ + L.ln2_g, true, st);
    col_sum(dA_, M, d, d, G_ + L.ln2_b, true, st);
    gemm(h, true, false, d, d, Mi, 1, L.O, d, dX_, d, 1, G_ + L.wout, d);
    gemm(h, false, true, Mi, d, d, 1, dX_, d, P_ + L.wout, d, 0, dA_, d);   // dA_ = dO
    k_attn_bwd1<<<nblk(M * H), 256, 0, st>>>(L.QKV, gnb_, ebits_, L.W, dA_, M, d, H, DU_, dQKV_);
    k_attn_bwd2<<<nblk(M * H), 256, 0, st>>>(L.QKV, gnb_, ebits_, L.W, DU_, dA_, M, d, H, dQKV_);
    k_ln_out<<<nblk(M * d), 256, 0, st>>>(L.xh1, M, d, P_ + L.ln1_g, P_ + L.ln1_b, A_);
    gemm(h, true, false, d, 3 * d, Mi, 1, A_, d, dQKV_, 3 * d, 1, G_ + L.wqkv, 3 * d);
    gemm(h, false, true, Mi, d, 3 * d, 1, dQKV_, 3 * d, P_ + L.wqkv, 3 * d, 0, dA_, d);
    k_ln_bwd<<<nblk(M), 256, 0, st>>>(dA_, L.xh1, L.rs1, P_ + L.ln1_g, M, d, dX_, true, T_, brk);
    col_sum(T_, M, d, d, G_ + L.ln1_g, true, st);
    col_sum(dA_, M, d, d, G_ + L.ln1_b, true, st);
  }
  col_sum(dX_, M, d, d, G_ + emb_b_, true, st);
  gemm(h, true, false, fin_, d, Mi, 1, X0_, fin_, dX_, d, 1, G_ + emb_w_, d);
}

// ============================== PointerHead ==============================
void PointerHead::build(int d_, int glob_, std::vector<ParamBlock>& reg, long& cur, const std::string& pfx) {
  d = d_;
  glob = glob_;
  wce = reg1(reg, cur, (long)(2 * d + glob) * d, PKind::LinW, pfx + ".cur_emb.w");   // [E d | Hc d | glob][d]
  bce = reg1(reg, cur, d, PKind::LinB, pfx + ".cur_emb.b");
  wq = reg1(reg, cur, (long)d * d, PKind::LinW, pfx + ".ptr.wq");
  wk = reg1(reg, cur, (long)d * d, PKind::LinW, pfx + ".ptr.wk");
}

void PointerHead::init(std::mt19937_64& rng, std::vector<real>& host) const {
  const float bc = 1.f / std::sqrt((float)(2 * d)), bp = 1.f / std::sqrt((float)d);
  uni(host, wce, (long)2 * d * d, bc, rng);            // nn.Linear(2d, d) 기본 초기화
  fill(host, wce + (long)2 * d * d, (long)glob * d, (real)0);   // 전역 벡터 열(우리 추가)은 0 으로 시작 → 시작 시점 출력은 논문 구조와 같다
  uni(host, bce, d, bc, rng);
  uni(host, wq, (long)d * d, bp, rng);
  uni(host, wk, (long)d * d, bp, rng);
}

void PointerHead::alloc(int mn) {
  max_n = mn;
  ws_bytes = 0;
  auto mk = [&](long n) {
    real* p;
    CK(cudaMalloc(&p, n * sizeof(real)));
    ws_bytes += n * sizeof(real);
    return p;
  };
  const long n = mn, C = 2 * d + glob;
  cat = mk(n * C);
  dcat = mk(n * C);
  s = mk(n * d);
  ds = mk(n * d);
  Qp = mk(n * d);
  dQp = mk(n * d);
  Kp = mk(n * kGNb * d);
  dKp = mk(n * kGNb * d);
  dHn = mk(n * kGNb * d);
  U = mk(n * kGNb);
}

void PointerHead::release() {
  for (real* p : {cat, dcat, s, ds, Qp, dQp, Kp, dKp, dHn, U}) cudaFree(p);
  cat = dcat = s = ds = Qp = dQp = Kp = dKp = dHn = U = nullptr;
}

void PointerHead::forward(cublasHandle_t h, cudaStream_t st, int n, const real* E, const real* Hc, const real* g, long ldg, const real* Hn,
                          const real* P, real* logits) {
  const int C = 2 * d + glob;
  k_ptr_cat<<<nblk((long)n * C), 256, 0, st>>>(E, Hc, g, ldg, n, d, glob, cat);
  gemm(h, false, false, n, d, C, 1, cat, C, P + wce, d, 0, s, d);
  k_bias<<<nblk((long)n * d), 256, 0, st>>>(s, n, d, P + bce, false);
  gemm(h, false, false, n, d, d, 1, s, d, P + wq, d, 0, Qp, d);
  gemm(h, false, false, n * kGNb, d, d, 1, Hn, d, P + wk, d, 0, Kp, d);
  k_ptr_score<<<nblk((long)n * kGNb), 256, 0, st>>>(Qp, Kp, n, d, U, logits);
}

void PointerHead::backward(cublasHandle_t h, cudaStream_t st, int n, const real* Hn, const real* P, real* G, const real* dl) {
  const int C = 2 * d + glob;
  k_ptr_bwd_k<<<nblk((long)n * kGNb * d), 256, 0, st>>>(Qp, U, dl, n, d, dKp);
  k_ptr_bwd_q<<<nblk((long)n * d), 256, 0, st>>>(Kp, U, dl, n, d, dQp);
  gemm(h, true, false, d, d, n * kGNb, 1, Hn, d, dKp, d, 1, G + wk, d);
  gemm(h, false, true, n * kGNb, d, d, 1, dKp, d, P + wk, d, 0, dHn, d);
  gemm(h, true, false, d, d, n, 1, s, d, dQp, d, 1, G + wq, d);
  gemm(h, false, true, n, d, d, 1, dQp, d, P + wq, d, 0, ds, d);
  col_sum(ds, n, d, d, G + bce, true, st);
  gemm(h, true, false, C, d, n, 1, cat, C, ds, d, 1, G + wce, d);
  gemm(h, false, true, n, C, d, 1, ds, d, P + wce, d, 0, dcat, C);
}

}  // namespace XNET_NS
