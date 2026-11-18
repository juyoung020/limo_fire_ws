// Embedder 판 "siglip2_trt": SigLIP2 (google/siglip2-base-patch16-224) 그림·글 벡터를 TensorRT 로. ROS 에 의존하지 않는다.
//   그림 엔진: pixel_values float [B, 3, 224, 224] ((x/255 − 0.5)/0.5, RGB) → image_embeds float [B, 768]
//   글 엔진:   token_embeds float [1, 64, 768] → text_embeds float [1, 768] (위치 표 · 인코더 · 머리 — 토큰 표는 엔진 밖)
//   토큰 표:   siglip2_tok_emb.f16 (256000 × 768 fp16 원시) 를 mmap — 요청 토큰 64 줄만 읽어 글 엔진 입력을 만든다.
//              표(fp32 786 MB)를 GPU 엔진에 두면 Orin 8 GB 에 판단 모델과 같이 못 올라가서 (README "설명 글로 물체 찾기")
//   둘 다 L2 로 나눈 벡터를 낸다 (점수는 sem_score.hpp)
// 엔진 만들기는 README "설명 글로 물체 찾기". B 는 엔진의 최적 프로필 안(그림 1~max_batch, 글 1).
// 두 엔진은 동시에 돌지 않으므로(호출이 겹치면 안 됨 — 한 스레드에서 부를 것) TensorRT 작업 메모리 한 덩이를 같이 쓴다.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "limo_fire_perception/embedder.hpp"
#include "limo_fire_perception/siglip_tokenizer.hpp"

namespace limo_fire_perception
{

class TrtSiglip : public Embedder
{
public:
  struct Files
  {
    std::string vision_engine, text_engine, token_table, vocab, merges;
  };
  TrtSiglip(const Files & files, int max_batch);
  ~TrtSiglip() override;
  TrtSiglip(const TrtSiglip &) = delete;
  TrtSiglip & operator=(const TrtSiglip &) = delete;

  std::vector<std::vector<float>> embed_images(const std::vector<cv::Mat> & rgb) override;
  std::vector<float> embed_text(const std::string & text) override;
  int dim() const override {return dim_;}
  int max_batch() const override {return max_batch_;}
  std::string describe() const override;
  struct Net;   // 엔진 하나 (trt_siglip.cpp)

private:
  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  };
  Logger logger_;
  SiglipTokenizer tok_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<Net> vis_, txt_;
  void * scratch_ = nullptr;
  int64_t scratch_bytes_ = 0, separate_bytes_ = 0;
  int64_t gpu_used_bytes_ = 0;   // 켤 때 잰 GPU 메모리 증가 (cudaMemGetInfo 앞·뒤 — 엔진 가중치 + 작업 메모리 + 입출력 버퍼)
  int dim_ = 768, max_batch_ = 4, side_ = 224, text_len_ = 64;
  const uint16_t * table_ = nullptr;   // mmap 한 토큰 표 (fp16)
  size_t table_bytes_ = 0, vocab_ = 0;
};

}  // namespace limo_fire_perception
