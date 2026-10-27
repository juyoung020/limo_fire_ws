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
    } else if (engine_->getTensorShape(name).nbDims == 4) {
      proto_name_ = name;   // 분할 모델의 원형 마스크 [1, nm, ph, pw]
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
  if (!proto_name_.empty()) {
    const nvinfer1::Dims pd = context_->getTensorShape(proto_name_.c_str());
    nm_ = pd.d[1];
    ph_ = pd.d[2];
    pw_ = pd.d[3];
    if (out_dims.d[2] != 6 + nm_) {
      throw std::runtime_error("분할 모델은 end2end 출력 [1, N, 6 + 마스크 계수] 여야 함");
    }
  }
  if (out_dims.d[2] == 6 + nm_) {
    end2end_ = true;
    out_rows_ = out_dims.d[1];
    out_cols_ = 6 + nm_;
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
  if (!proto_name_.empty()) {
    h_proto_.resize(static_cast<size_t>(nm_) * ph_ * pw_);
    check_cuda(cudaMalloc(&d_proto_, h_proto_.size() * sizeof(float)), "cudaMalloc proto");
    context_->setTensorAddress(proto_name_.c_str(), d_proto_);
  }
}

TrtYolo::~TrtYolo()
{
  context_.reset();
  engine_.reset();
  runtime_.reset();
  if (d_in_) {cudaFree(d_in_);}
  if (d_out_) {cudaFree(d_out_);}
  if (d_proto_) {cudaFree(d_proto_);}
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
  if (d_proto_) {
    check_cuda(cudaMemcpyAsync(h_proto_.data(), d_proto_, h_proto_.size() * sizeof(float), cudaMemcpyDeviceToHost, st),
      "D2H proto");
  }
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
    const float * p = &h_out_[static_cast<size_t>(r) * out_cols_];
    if (p[4] < conf_) {
      continue;
    }
    out.push_back({unletterbox(p[0], p[1], p[2], p[3], lb.scale, lb.pad_x, lb.pad_y, img_w, img_h),
        p[4], static_cast<int>(p[5]), {}, {}});
    if (nm_ > 0) {
      decode_mask(p + 6, lb, img_w, img_h, out.back());
    }
  }
  return out;
}

void TrtYolo::decode_mask(const float * coeffs, const Letterbox & lb, int img_w, int img_h, Detection & det) const
{
  // Ultralytics process_mask 와 같은 순서: 계수 · 원형 → 상자 밖 버림 → 원본 크기로 키움 → 0.5 문턱
  // 상자 부분만 계산한다: 원본 상자 → 입력(640) 좌표 → 원형(160) 좌표
  det.mask_roi = cv::Rect(det.box) & cv::Rect(0, 0, img_w, img_h);
  if (det.mask_roi.empty()) {
    return;
  }
  const float kx = static_cast<float>(pw_) / in_w_, ky = static_cast<float>(ph_) / in_h_;
  const float px0 = (det.mask_roi.x * lb.scale + lb.pad_x) * kx;
  const float py0 = (det.mask_roi.y * lb.scale + lb.pad_y) * ky;
  const float px1 = (det.mask_roi.br().x * lb.scale + lb.pad_x) * kx;
  const float py1 = (det.mask_roi.br().y * lb.scale + lb.pad_y) * ky;
  const int gx0 = std::clamp(static_cast<int>(std::floor(px0)), 0, pw_ - 1);
  const int gy0 = std::clamp(static_cast<int>(std::floor(py0)), 0, ph_ - 1);
  const int gx1 = std::clamp(static_cast<int>(std::ceil(px1)), gx0 + 1, pw_);
  const int gy1 = std::clamp(static_cast<int>(std::ceil(py1)), gy0 + 1, ph_);

  // 원형 칸마다 로짓 = Σ 계수 × 원형
  cv::Mat logit(gy1 - gy0, gx1 - gx0, CV_32F, cv::Scalar(0));
  const size_t plane = static_cast<size_t>(ph_) * pw_;
  for (int k = 0; k < nm_; ++k) {
    const float c = coeffs[k];
    const float * pk = &h_proto_[k * plane];
    for (int y = gy0; y < gy1; ++y) {
      float * row = logit.ptr<float>(y - gy0);
      const float * src = pk + static_cast<size_t>(y) * pw_;
      for (int x = gx0; x < gx1; ++x) {
        row[x - gx0] += c * src[x];
      }
    }
  }
  // 원본 픽셀 (u, v) 의 원형 좌표 = (u·s + pad)·k - 0.5 (칸 가운데 기준). 그 자리를 쌍선형으로 읽는다
  cv::Mat map_x(det.mask_roi.size(), CV_32F), map_y(det.mask_roi.size(), CV_32F);
  for (int v = 0; v < det.mask_roi.height; ++v) {
    const float gy = ((det.mask_roi.y + v + 0.5f) * lb.scale + lb.pad_y) * ky - 0.5f - gy0;
    float * mx = map_x.ptr<float>(v);
    float * my = map_y.ptr<float>(v);
    for (int u = 0; u < det.mask_roi.width; ++u) {
      mx[u] = ((det.mask_roi.x + u + 0.5f) * lb.scale + lb.pad_x) * kx - 0.5f - gx0;
      my[u] = gy;
    }
  }
  cv::Mat up;
  cv::remap(logit, up, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  // 시그모이드 > 0.5 ⇔ 로짓 > 0
  cv::compare(up, 0.0, det.mask, cv::CMP_GT);
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
    out.push_back({cv::Rect2f(boxes[i]), scores[i], classes[i], {}, {}});
  }
  return out;
}

}  // namespace limo_fire_perception
