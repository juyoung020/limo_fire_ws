#include "limo_fire_perception/trt_yolo.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace limo_fire_perception
{

namespace
{
void check_cuda(cudaError_t e, const char * what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

size_t volume(const nvinfer1::Dims & d)
{
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    v *= static_cast<size_t>(d.d[i]);
  }
  return v;
}
}  // namespace

void TrtYolo::Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << msg << std::endl;
  }
}

TrtYolo::TrtYolo(const std::string & engine_path, float conf_threshold, float nms_iou)
: conf_(conf_threshold), nms_iou_(nms_iou)
{
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("엔진 파일을 열 수 없음: " + engine_path);
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    throw std::runtime_error("엔진 역직렬화 실패 (이 기기·TensorRT 버전에서 만든 엔진인지 확인): " + engine_path);
  }
  context_.reset(engine_->createExecutionContext());

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error(std::string("입출력은 float32 여야 함 (trtexec 기본값): ") + name);
    }
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      in_name_ = name;
    } else if (out_name_.empty()) {
      out_name_ = name;
    }
  }
  if (in_name_.empty() || out_name_.empty()) {
    throw std::runtime_error("입력·출력 텐서를 찾지 못함");
  }

  nvinfer1::Dims in_dims = engine_->getTensorShape(in_name_.c_str());
  if (in_dims.nbDims != 4) {
    throw std::runtime_error("입력은 [1, 3, H, W] 여야 함");
  }
  if (in_dims.d[2] > 0) {in_h_ = in_dims.d[2];}
  if (in_dims.d[3] > 0) {in_w_ = in_dims.d[3];}
  if (in_dims.d[0] < 0 || in_dims.d[2] < 0 || in_dims.d[3] < 0) {
    context_->setInputShape(in_name_.c_str(), nvinfer1::Dims4{1, 3, in_h_, in_w_});
  }

  nvinfer1::Dims out_dims = context_->getTensorShape(out_name_.c_str());
  if (out_dims.nbDims != 3) {
    throw std::runtime_error("출력은 3 차원이어야 함");
  }
  if (out_dims.d[2] == 6) {
    end2end_ = true;
    out_rows_ = out_dims.d[1];
    out_cols_ = 6;
  } else {
    end2end_ = false;
    out_rows_ = out_dims.d[1];   // 4 + 클래스 수
    out_cols_ = out_dims.d[2];   // 앵커 수
    num_classes_ = out_rows_ - 4;
  }

  h_in_.resize(3 * static_cast<size_t>(in_w_) * in_h_);
  h_out_.resize(volume(out_dims));
  check_cuda(cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_)), "cudaStreamCreate");
  check_cuda(cudaMalloc(&d_in_, h_in_.size() * sizeof(float)), "cudaMalloc in");
  check_cuda(cudaMalloc(&d_out_, h_out_.size() * sizeof(float)), "cudaMalloc out");
  context_->setTensorAddress(in_name_.c_str(), d_in_);
  context_->setTensorAddress(out_name_.c_str(), d_out_);
}

TrtYolo::~TrtYolo()
{
  context_.reset();
  engine_.reset();
  runtime_.reset();
  if (d_in_) {cudaFree(d_in_);}
  if (d_out_) {cudaFree(d_out_);}
  if (stream_) {cudaStreamDestroy(static_cast<cudaStream_t>(stream_));}
}

TrtYolo::Letterbox TrtYolo::preprocess(const cv::Mat & bgr)
{
  // 비율 유지 축소 + 회색(114) 덧대기 — Ultralytics 와 같은 방식
  const float s = std::min(static_cast<float>(in_w_) / bgr.cols, static_cast<float>(in_h_) / bgr.rows);
  const int nw = static_cast<int>(std::round(bgr.cols * s));
  const int nh = static_cast<int>(std::round(bgr.rows * s));
  const float px = (in_w_ - nw) / 2.0f;
  const float py = (in_h_ - nh) / 2.0f;

  cv::Mat resized, canvas(in_h_, in_w_, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  resized.copyTo(canvas(cv::Rect(static_cast<int>(std::round(px - 0.1f)), static_cast<int>(std::round(py - 0.1f)), nw, nh)));

  // HWC BGR uint8 → CHW RGB float [0, 1]
  const size_t plane = static_cast<size_t>(in_w_) * in_h_;
  for (int y = 0; y < in_h_; ++y) {
    const cv::Vec3b * row = canvas.ptr<cv::Vec3b>(y);
    for (int x = 0; x < in_w_; ++x) {
      const size_t i = static_cast<size_t>(y) * in_w_ + x;
      h_in_[i] = row[x][2] / 255.0f;
      h_in_[plane + i] = row[x][1] / 255.0f;
      h_in_[2 * plane + i] = row[x][0] / 255.0f;
    }
  }
  return {s, px, py};
}

std::vector<Detection> TrtYolo::infer(const cv::Mat & bgr)
{
  const Letterbox lb = preprocess(bgr);
  auto st = static_cast<cudaStream_t>(stream_);
  check_cuda(cudaMemcpyAsync(d_in_, h_in_.data(), h_in_.size() * sizeof(float), cudaMemcpyHostToDevice, st), "H2D");
  if (!context_->enqueueV3(st)) {
    throw std::runtime_error("enqueueV3 실패");
  }
  check_cuda(cudaMemcpyAsync(h_out_.data(), d_out_, h_out_.size() * sizeof(float), cudaMemcpyDeviceToHost, st), "D2H");
  check_cuda(cudaStreamSynchronize(st), "sync");
  return end2end_ ? decode_end2end(lb, bgr.cols, bgr.rows) : decode_raw(lb, bgr.cols, bgr.rows);
}

namespace
{
cv::Rect2f unletterbox(float x1, float y1, float x2, float y2, float s, float px, float py, int w, int h)
{
  x1 = std::clamp((x1 - px) / s, 0.0f, static_cast<float>(w));
  x2 = std::clamp((x2 - px) / s, 0.0f, static_cast<float>(w));
  y1 = std::clamp((y1 - py) / s, 0.0f, static_cast<float>(h));
  y2 = std::clamp((y2 - py) / s, 0.0f, static_cast<float>(h));
  return {x1, y1, x2 - x1, y2 - y1};
}
}  // namespace

std::vector<Detection> TrtYolo::decode_end2end(const Letterbox & lb, int img_w, int img_h) const
{
  std::vector<Detection> out;
  for (int r = 0; r < out_rows_; ++r) {
    const float * p = &h_out_[static_cast<size_t>(r) * 6];
    if (p[4] < conf_) {
      continue;
    }
    out.push_back({unletterbox(p[0], p[1], p[2], p[3], lb.scale, lb.pad_x, lb.pad_y, img_w, img_h),
        p[4], static_cast<int>(p[5])});
  }
  return out;
}

std::vector<Detection> TrtYolo::decode_raw(const Letterbox & lb, int img_w, int img_h) const
{
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> classes;
  const int n = out_cols_;
  for (int a = 0; a < n; ++a) {
    int best = 0;
    float best_s = 0.0f;
    for (int c = 0; c < num_classes_; ++c) {
      const float s = h_out_[static_cast<size_t>(4 + c) * n + a];
      if (s > best_s) {best_s = s; best = c;}
    }
    if (best_s < conf_) {
      continue;
    }
    const float cx = h_out_[a], cy = h_out_[n + a], w = h_out_[2 * n + a], h = h_out_[3 * n + a];
    const cv::Rect2f r = unletterbox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, lb.scale, lb.pad_x, lb.pad_y, img_w, img_h);
    boxes.emplace_back(r);
    scores.push_back(best_s);
    classes.push_back(best);
  }
  std::vector<int> keep;
  cv::dnn::NMSBoxesBatched(boxes, scores, classes, conf_, nms_iou_, keep);
  std::vector<Detection> out;
  for (int i : keep) {
    out.push_back({cv::Rect2f(boxes[i]), scores[i], classes[i]});
  }
  return out;
}

}  // namespace limo_fire_perception
