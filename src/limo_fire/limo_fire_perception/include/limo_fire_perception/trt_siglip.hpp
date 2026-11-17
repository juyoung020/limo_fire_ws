// SigLIP2 (google/siglip2-base-patch16-224) 그림·글 묶음 벡터를 TensorRT 로. ROS 에 의존하지 않는다.
//   그림 엔진: pixel_values float [B, 3, 224, 224] ((x/255 − 0.5)/0.5, RGB) → image_embeds float [B, 768]
//   글 엔진:   token_embeds float [1, 64, 768] → text_embeds float [1, 768] (위치 표 · 인코더 · 머리 — 토큰 표는 엔진 밖)
//   토큰 표:   siglip2_tok_emb.f16 (256000 × 768 fp16 원시) 를 mmap — 요청 토큰 64 줄만 읽어 글 엔진 입력을 만든다.
//              표(fp32 786 MB)를 GPU 엔진에 두면 Orin 8 GB 에 판단 모델과 같이 못 올라가서 (README "SigLIP2")
//   둘 다 L2 로 나눈 뒤 내적 = cos. 점수 = sigmoid(logit_scale · cos + logit_bias) (SigLIP 시그모이드 손실과 같은 꼴)
// 엔진 만들기는 README "SigLIP2". B 는 엔진의 최적 프로필 안(그림 1~8, 글 1).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>

namespace limo_fire_perception
{

class TrtSiglip
{
public:
  TrtSiglip(const std::string & vision_engine, const std::string & text_engine, const std::string & token_table);
  ~TrtSiglip();
  TrtSiglip(const TrtSiglip &) = delete;
  TrtSiglip & operator=(const TrtSiglip &) = delete;

  // RGB(CV_8UC3) 사진 여러 장 → 장마다 L2 단위 벡터 (dim)
  std::vector<std::vector<float>> embed_images(const std::vector<cv::Mat> & rgb);
  // 토큰 번호 64 개 → L2 단위 벡터
  std::vector<float> embed_text(const std::vector<int64_t> & ids);
  int dim() const {return dim_;}
  int max_batch() const {return max_batch_;}
  struct Net;   // 엔진 하나 (trt_siglip.cpp)

private:
  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  };
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<Net> vis_, txt_;
  int dim_ = 768, max_batch_ = 8, side_ = 224, text_len_ = 64;
  const uint16_t * table_ = nullptr;   // mmap 한 토큰 표 (fp16)
  size_t table_bytes_ = 0, vocab_ = 0;
};

}  // namespace limo_fire_perception
