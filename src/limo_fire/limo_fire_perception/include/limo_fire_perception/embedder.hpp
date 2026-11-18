// 그림·글 → 같은 공간의 L2 단위 벡터 (설명 글로 물체 찾기, object_search). ROS 에 의존하지 않는다.
// 모델 판(backend)은 이 틀 뒤에 숨긴다 — 노드는 이 틀만 부르므로, 양자화한 SigLIP2 같은 새 판은
// 구현 하나 + make_embedder 의 이름 하나를 더하면 되고 노드 코드는 그대로다.
// 점수 셈(sigmoid(scale·cos + bias), 사진 최댓값)은 모델과 따로 sem_score.hpp 에 있다.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace limo_fire_perception
{

class Embedder
{
public:
  virtual ~Embedder() = default;
  // RGB(CV_8UC3) 사진 여러 장 → 장마다 L2 단위 벡터 (dim)
  virtual std::vector<std::vector<float>> embed_images(const std::vector<cv::Mat> & rgb) = 0;
  // 글(틀을 채운 뒤, 예: "a photo of 빨간 컵.") → L2 단위 벡터 (dim). 토큰 나누기는 판 안에서
  virtual std::vector<float> embed_text(const std::string & text) = 0;
  virtual int dim() const = 0;
  // 한 번에 넣을 사진 장 수 상한 (넘으면 판이 나눠 돌림 — 노드는 이만큼씩 모은다)
  virtual int max_batch() const = 0;
  // 켤 때 로그 한 줄 (판 이름 · 잰 메모리)
  virtual std::string describe() const = 0;
};

// 판 고르기: backend 이름(파라미터 backend) + 판마다의 파라미터를 읽는 함수 param(이름, 기본값) — 노드가 declare_parameter 로 넘긴다.
// 파일 파라미터에 이름만 쓰면 model_dir 에서 찾는다.
// 모르는 이름이면 std::runtime_error (아는 이름을 메시지에)
using EmbedderParam = std::function<std::string(const std::string & name, const std::string & default_value)>;
std::unique_ptr<Embedder> make_embedder(const std::string & backend, const std::string & model_dir, const EmbedderParam & param);

}  // namespace limo_fire_perception
