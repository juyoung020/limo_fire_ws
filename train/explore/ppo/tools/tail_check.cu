// 합성곱 넷째 층 융합 커널(cnn_tail) vs host fp64 기준: 출력 Y·dF3·dW·db 상대 오차, 덧댄 칸·가지 밖 열을 안 건드리는지, 같은 입력 두 번 = 비트 같음.
//   tail_check   (통과 기준: 상대 < 1e-5 = fp32 누산 크기, 가지 밖 0, dF3 모든 칸 씀, 결정적)
// 모양: 실제 구조의 세 가지(가까운 지도 8×8·먼 지도 4×4·깊이 5×8 → 덧댐) C = cout = 32, 검사용 작은 구조 C = cout = 4.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cnn_tail.cuh"
#include "common.cuh"

using namespace xnet;

namespace {
struct Case {
  const char* name;
  int h, w, C, cout;
};

int run(const Case& cs, int n, std::mt19937& rng, double& wmax) {
  const int oh = (cs.h + 1) / 2, ow = (cs.w + 1) / 2, P4 = oh * ow, K = 4 * cs.C, CO = cs.cout;
  const long off = 40, ldy = off + (long)P4 * CO + 13;  // MLP 입력 [F | 1D] 안의 가지 열처럼: 앞뒤에 다른 열
  const long ld1 = ldy + 3, ld2 = ldy + 7;
  TailShape s{n, cs.h, cs.w, cs.C, oh, ow, CO};
  std::normal_distribution<float> N(0.f, 1.f);
  const float bd = 1.f / std::sqrt((float)K);
  std::uniform_real_distribution<float> U(-bd, bd);
  std::vector<float> f3((size_t)n * cs.h * cs.w * cs.C), wb((size_t)CO * K + CO), g1((size_t)n * ld1), g2((size_t)n * ld2);
  for (auto& v : f3) v = std::fabs(N(rng)) * (rng() % 4 == 0 ? 0.f : 1.f);  // ELU 출력처럼 (0 섞음)
  for (auto& v : wb) v = U(rng);
  for (auto& v : g1) v = 1e-3f * N(rng);
  for (auto& v : g2) v = 1e-3f * N(rng);
  float *dF3in, *dWb, *dY, *dg1, *dg2, *ddF3, *dG, *dpart;
  CK(cudaMalloc(&dF3in, f3.size() * 4));
  CK(cudaMalloc(&dWb, wb.size() * 4));
  CK(cudaMalloc(&dY, (size_t)n * ldy * 4));
  CK(cudaMalloc(&dg1, g1.size() * 4));
  CK(cudaMalloc(&dg2, g2.size() * 4));
  CK(cudaMalloc(&ddF3, f3.size() * 4));
  CK(cudaMalloc(&dG, wb.size() * 4));
  CK(cudaMalloc(&dpart, cnn_tail_part_floats(s) * 4));
  CK(cudaMemcpy(dF3in, f3.data(), f3.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWb, wb.data(), wb.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dg1, g1.data(), g1.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dg2, g2.data(), g2.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dY, 0, (size_t)n * ldy * 4));
  const float kNaN = std::nanf("");
  std::vector<float> nanv(f3.size(), kNaN);  // dF3 를 NaN 으로 채워 두고 모든 칸이 쓰였는지 본다
  std::vector<float> gy((size_t)n * ldy), gdf[2], gg[2];
  for (int rep = 0; rep < 2; ++rep) {
    CK(cudaMemcpy(ddF3, nanv.data(), nanv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dG, 0, wb.size() * 4));
    cnn_tail_forward(s, dF3in, dWb, dY + off, ldy, 0);
    cnn_tail_backward(s, dF3in, dWb, dY + off, ldy, dg1 + off, ld1, dg2 + off, ld2, ddF3, dG, dpart, false, 0);
    gdf[rep].resize(f3.size());
    gg[rep].resize(wb.size());
    CK(cudaMemcpy(gy.data(), dY, gy.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gdf[rep].data(), ddF3, f3.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gg[rep].data(), dG, wb.size() * 4, cudaMemcpyDeviceToHost));
  }
  // z_out: dF3 대신 dZ3 = dF3 · ELU'(F3) — 위 dF3 에 같은 fp32 식을 곱한 것과 비트 같아야 한다
  long zbad = 0;
  {
    CK(cudaMemcpy(ddF3, nanv.data(), nanv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dG, 0, wb.size() * 4));
    cnn_tail_backward(s, dF3in, dWb, dY + off, ldy, dg1 + off, ld1, dg2 + off, ld2, ddF3, dG, dpart, false, 0, /*z_out=*/true);
    std::vector<float> gz(f3.size()), gzg(wb.size());
    CK(cudaMemcpy(gz.data(), ddF3, f3.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gzg.data(), dG, wb.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t e = 0; e < f3.size(); ++e) {
      const float x = f3[e], want = gdf[0][e] * ((x > 0.f) ? 1.f : x + 1.f);
      if (std::memcmp(&want, &gz[e], 4) != 0) ++zbad;
    }
    if (std::memcmp(gzg.data(), gg[0].data(), wb.size() * 4) != 0) ++zbad;
  }
  CK(cudaDeviceSynchronize());
  // host fp64 기준
  std::vector<double> y((size_t)n * P4 * CO), dz(y.size()), df(f3.size(), 0.0), gw(wb.size(), 0.0);
  auto xat = [&](int i, int p, int k) -> double {
    const int child = k / cs.C, c = k % cs.C, py = p / ow, px = p % ow;
    const int yy = 2 * py + (child >> 1), xx = 2 * px + (child & 1);
    if (yy >= cs.h || xx >= cs.w) return 0.0;
    return f3[(((size_t)i * cs.h + yy) * cs.w + xx) * cs.C + c];
  };
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < P4; ++p)
      for (int co = 0; co < CO; ++co) {
        double z = wb[(size_t)CO * K + co];
        for (int k = 0; k < K; ++k) z += xat(i, p, k) * wb[(size_t)co * K + k];
        const double a = z > 0 ? z : std::expm1(z);
        const size_t r = ((size_t)i * P4 + p) * CO + co;
        y[r] = a;
        const size_t col = (size_t)p * CO + co;
        const double g = (double)g1[i * ld1 + off + col] + (double)g2[i * ld2 + off + col];
        dz[r] = g * (a > 0 ? 1.0 : a + 1.0);
      }
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < P4; ++p)
      for (int co = 0; co < CO; ++co) {
        const double d = dz[((size_t)i * P4 + p) * CO + co];
        gw[(size_t)CO * K + co] += d;
        for (int k = 0; k < K; ++k) {
          gw[(size_t)co * K + k] += d * xat(i, p, k);
          const int child = k / cs.C, c = k % cs.C, py = p / ow, px = p % ow;
          const int yy = 2 * py + (child >> 1), xx = 2 * px + (child & 1);
          if (yy < cs.h && xx < cs.w) df[(((size_t)i * cs.h + yy) * cs.w + xx) * cs.C + c] += d * wb[(size_t)co * K + k];
        }
      }
  auto rel = [](double num, double den) { return den > 0 ? std::sqrt(num / den) : std::sqrt(num); };
  double ny = 0, ey = 0, outside = 0;
  for (int i = 0; i < n; ++i)
    for (long j = 0; j < ldy; ++j) {
      const double v = gy[(size_t)i * ldy + j];
      if (j >= off && j < off + (long)P4 * CO) {
        const double r = y[(size_t)i * P4 * CO + (j - off)];
        ny += r * r, ey += (v - r) * (v - r);
      } else {
        outside += std::fabs(v);
      }
    }
  double nd = 0, ed = 0;
  long unwritten = 0;
  for (size_t e = 0; e < df.size(); ++e) {
    if (std::isnan(gdf[0][e])) ++unwritten;
    nd += df[e] * df[e], ed += (gdf[0][e] - df[e]) * (gdf[0][e] - df[e]);
  }
  double nw = 0, ew = 0, nb = 0, eb = 0;
  for (size_t e = 0; e < gw.size(); ++e) {
    const bool bias = e >= (size_t)CO * K;
    const double dd = (gg[0][e] - gw[e]) * (gg[0][e] - gw[e]);
    (bias ? nb : nw) += gw[e] * gw[e];
    (bias ? eb : ew) += dd;
  }
  const bool det = std::memcmp(gdf[0].data(), gdf[1].data(), f3.size() * 4) == 0 &&
                   std::memcmp(gg[0].data(), gg[1].data(), wb.size() * 4) == 0;
  const double ry = rel(ey, ny), rd = rel(ed, nd), rw = rel(ew, nw), rb = rel(eb, nb);
  std::printf("%-6s n %5d: Y %.2e  dF3 %.2e  dW %.2e  db %.2e  | 가지 밖 %g  안 쓴 dF3 %ld  두 번 같음 %s  dZ3 다른 칸 %ld\n", cs.name, n, ry,
              rd, rw, rb, outside, unwritten, det ? "예" : "아니오", zbad);
  for (double v : {ry, rd, rw, rb}) wmax = std::max(wmax, v);
  for (float* p : {dF3in, dWb, dY, dg1, dg2, ddF3, dG, dpart}) cudaFree(p);
  return (ry < 1e-5 && rd < 1e-5 && rw < 1e-5 && rb < 1e-5 && outside == 0 && unwritten == 0 && det && zbad == 0) ? 0 : 1;
}
}  // namespace

int main() {
  const Case cases[] = {{"가까움", 8, 8, 32, 32}, {"먼", 4, 4, 32, 32}, {"깊이", 5, 8, 32, 32}, {"작은", 3, 2, 4, 4}};
  std::mt19937 rng(11);
  int fail = 0;
  double wmax = 0;
  for (const auto& cs : cases)
    for (int n : {1, 5, 37, 4096}) fail |= run(cs, n, rng, wmax);
  std::printf("%s (최대 상대 %.2e)\n", fail ? "실패" : "통과", wmax);
  return fail;
}
