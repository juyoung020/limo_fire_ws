// 처리량 재기: 텍스트 한 번, 영상 인코더 + 검출을 N 번
// 사용: sam3_bench <model.safetensors> <ref.safetensors> [반복 20] [f16|bf16|f32]
#include <cstdio>
#include <string>

#include "sam3.h"

using namespace sam3;

template <typename T>
static void run(const SafeTensors& W, const SafeTensors& R, int n) {
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  const auto& ids = R.get("input_ids");
  const auto& am = R.get("attention_mask");
  int P = (int)ids.shape[0];
  Model<T> m(W, P, s);
  m.encode_text(static_cast<const int*>(ids.data), static_cast<const int*>(am.data), P);
  Buf<float> px;
  px.alloc((size_t)3 * IMG * IMG);
  CK(cudaMemcpy(px.p, R.get("pixel_values").f32(), px.n * 4, cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  float tv = 0, td = 0;
  for (int i = 0; i < n + 2; ++i) {
    cudaEventRecord(e0, s);
    m.encode_image(px.p);
    cudaEventRecord(e1, s);
    m.detect();
    cudaEventRecord(e2, s);
    CK(cudaStreamSynchronize(s));
    float a, b;
    cudaEventElapsedTime(&a, e0, e1);
    cudaEventElapsedTime(&b, e1, e2);
    if (i >= 2) tv += a, td += b;
  }
  std::printf("영상 %.1f ms, 검출(프롬프트 %d) %.1f ms, 합 %.1f ms → %.1f 장/초\n", tv / n, P, td / n, (tv + td) / n,
              1000.f * n / (tv + td));
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  SafeTensors W(argv[1]), R(argv[2]);
  int n = argc > 3 ? std::stoi(argv[3]) : 20;
  std::string t = argc > 4 ? argv[4] : "f16";
  if (t == "bf16")
    run<bf16>(W, R, n);
  else if (t == "f32")
    run<float>(W, R, n);
  else
    run<f16>(W, R, n);
}
