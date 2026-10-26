// TensorRT 로 YOLO 엔진 하나를 돌린다. ROS 에 의존하지 않는다.
// 출력 두 가지를 알아서 구분한다:
//   end2end (YOLO26, NMS 없음): [1, N, 6] = x1 y1 x2 y2 score class   (입력 픽셀 좌표)
//   raw (YOLOv8/11):           [1, 4 + 클래스 수, 앵커 수] = cx cy w h + 클래스별 점수 → 여기서 NMS
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>

namespace limo_fire_perception
{

struct Detection
{
  cv::Rect2f box;  // 원본 이미지 픽셀 좌표 (x, y, w, h)
  float score;
  int class_id;
};

class TrtYolo
{
public:
  TrtYolo(const std::string & engine_path, float conf_threshold, float nms_iou);
  ~TrtYolo();
  TrtYolo(const TrtYolo &) = delete;
  TrtYolo & operator=(const TrtYolo &) = delete;

  // BGR 이미지 하나 → 탐지 목록
  std::vector<Detection> infer(const cv::Mat & bgr);

  int input_width() const {return in_w_;}
  int input_height() const {return in_h_;}
  bool end2end() const {return end2end_;}
  int num_classes() const {return num_classes_;}

private:
  struct Letterbox
  {
    float scale;
    float pad_x;
    float pad_y;
  };

  Letterbox preprocess(const cv::Mat & bgr);
  std::vector<Detection> decode_end2end(const Letterbox & lb, int img_w, int img_h) const;
  std::vector<Detection> decode_raw(const Letterbox & lb, int img_w, int img_h) const;

  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  };

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void * stream_ = nullptr;

  std::string in_name_, out_name_;
  int in_w_ = 640, in_h_ = 640;
  void * d_in_ = nullptr;
  void * d_out_ = nullptr;
  std::vector<float> h_in_, h_out_;
  // 출력 모양: end2end 이면 [N, 6], raw 이면 [4 + 클래스, 앵커]
  bool end2end_ = false;
  int out_rows_ = 0, out_cols_ = 0;
  int num_classes_ = 0;

  float conf_;
  float nms_iou_;
};

}  // namespace limo_fire_perception
