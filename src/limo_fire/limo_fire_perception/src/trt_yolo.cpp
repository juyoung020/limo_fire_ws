#include "limo_fire_perception/trt_yolo.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/dnn.hpp>

#include "trt_yolo_kernels.cuh"

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

template<typename T>
void free_device(T *& p)
{
  if (p) {cudaFree(p); p = nullptr;}
}

template<typename T>
void free_host(T *& p)
{
  if (p) {cudaFreeHost(p); p = nullptr;}
}
}  // namespace

// 슬롯마다 따로 가지는 것: CPU 쪽 고정(pinned) 메모리, 끝났음을 알리는 이벤트, 그 슬롯 주소로 담은 그래프(결과 영상 없음·있음)
struct TrtYolo::Slot
{
  uint8_t * h_img = nullptr;
  gpu::DetBlock * h_block = nullptr;
  uint8_t * h_ids = nullptr;
  uint8_t * h_vis = nullptr;
  float * h_out = nullptr;
  cudaEvent_t done = nullptr;
  cudaGraphExec_t exec[2] = {nullptr, nullptr};
  bool busy = false;
  bool vis = false;
};

void TrtYolo::Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TensorRT] " << msg << std::endl;
  }
}

TrtYolo::TrtYolo(const std::string & engine_path, float conf_threshold, float nms_iou)
: conf_(conf_threshold), nms_iou_(nms_iou)
{
  // GPU 를 기다리는 동안 CPU 가 바쁜 대기로 코어 하나를 태우지 않고 잠들게 (컨텍스트 만들기 전에만 먹힌다)
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaGetLastError();

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

  nvinfer1::DataType proto_type = nvinfer1::DataType::kFLOAT;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    const auto type = engine_->getTensorDataType(name);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      in_name_ = name;
      if (type == nvinfer1::DataType::kUINT8) {
        in_u8_ = true;
      } else if (type != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(std::string("입력은 uint8 [1,H,W,3] 또는 float32 [1,3,H,W] 여야 함: ") + name);
      }
    } else if (engine_->getTensorShape(name).nbDims == 4) {
      proto_name_ = name;   // 분할 모델의 원형 마스크 [1, nm, ph, pw]
      proto_type = type;
    } else if (out_name_.empty()) {
      if (type != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(std::string("탐지 출력은 float32 여야 함: ") + name);
      }
      out_name_ = name;
    }
  }
  if (in_name_.empty() || out_name_.empty()) {
    throw std::runtime_error("입력·출력 텐서를 찾지 못함");
  }

  const nvinfer1::Dims in_dims = engine_->getTensorShape(in_name_.c_str());
  if (in_dims.nbDims != 4 || (in_u8_ && in_dims.d[3] != 3) || (!in_u8_ && in_dims.d[1] != 3)) {
    throw std::runtime_error(in_u8_ ? "uint8 입력은 [1, H, W, 3] 여야 함" : "float 입력은 [1, 3, H, W] 여야 함");
  }
  const int hd = in_u8_ ? 1 : 2, wd = in_u8_ ? 2 : 3;
  if (in_dims.d[hd] > 0) {in_h_ = in_dims.d[hd];}
  if (in_dims.d[wd] > 0) {in_w_ = in_dims.d[wd];}
  if (in_dims.d[0] < 0 || in_dims.d[hd] < 0 || in_dims.d[wd] < 0) {
    context_->setInputShape(in_name_.c_str(),
      in_u8_ ? nvinfer1::Dims4{1, in_h_, in_w_, 3} : nvinfer1::Dims4{1, 3, in_h_, in_w_});
  }

  const nvinfer1::Dims out_dims = context_->getTensorShape(out_name_.c_str());
  if (out_dims.nbDims != 3) {
    throw std::runtime_error("출력은 3 차원이어야 함");
  }
  if (!proto_name_.empty()) {
    const nvinfer1::Dims pd = context_->getTensorShape(proto_name_.c_str());
    nm_ = pd.d[1];
    ph_ = pd.d[2];
    pw_ = pd.d[3];
    if (proto_type != nvinfer1::DataType::kFLOAT && proto_type != nvinfer1::DataType::kHALF) {
      throw std::runtime_error("원형 마스크는 float32 또는 fp16 이어야 함");
    }
    proto_half_ = proto_type == nvinfer1::DataType::kHALF;
    if (out_dims.d[2] != 6 + nm_) {
      throw std::runtime_error("분할 모델은 end2end 출력 [1, N, 6 + 마스크 계수] 여야 함");
    }
    if (nm_ > gpu::kMaxNm) {
      throw std::runtime_error("마스크 계수는 32 개까지");
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
  out_elems_ = volume(out_dims);

  check_cuda(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t *>(&stream_), cudaStreamNonBlocking),
    "cudaStreamCreate");
  check_cuda(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t *>(&mask_stream_), cudaStreamNonBlocking),
    "cudaStreamCreate");
  const size_t in_bytes = static_cast<size_t>(in_w_) * in_h_ * 3 * (in_u8_ ? 1 : sizeof(float));
  check_cuda(cudaMalloc(&d_in_, in_bytes), "cudaMalloc in");
  check_cuda(cudaMalloc(&d_out_, out_elems_ * sizeof(float)), "cudaMalloc out");
  check_cuda(cudaMalloc(&d_block_, sizeof(gpu::DetBlock)), "cudaMalloc dets");
  check_cuda(cudaMalloc(&d_colors_, 256 * 3), "cudaMalloc colors");
  check_cuda(cudaMemset(d_colors_, 160, 256 * 3), "cudaMemset colors");
  context_->setTensorAddress(in_name_.c_str(), d_in_);
  context_->setTensorAddress(out_name_.c_str(), d_out_);
  if (has_masks()) {
    check_cuda(cudaMalloc(&d_proto_, static_cast<size_t>(nm_) * ph_ * pw_ * (proto_half_ ? 2 : 4)), "cudaMalloc proto");
    context_->setTensorAddress(proto_name_.c_str(), d_proto_);
    check_cuda(cudaMalloc(&d_logits_, static_cast<size_t>(gpu::kMaxMaskDets) * ph_ * pw_ * sizeof(float)),
      "cudaMalloc logits");
  }
  for (int i = 0; i < kSlots; ++i) {
    slots_.push_back(std::make_unique<Slot>());
    check_cuda(cudaEventCreateWithFlags(&slots_.back()->done, cudaEventBlockingSync | cudaEventDisableTiming),
      "cudaEventCreate");
  }
}

TrtYolo::~TrtYolo()
{
  if (stream_) {cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));}
  release_image_buffers();
  for (auto & s : slots_) {
    if (s->done) {cudaEventDestroy(s->done);}
  }
  context_.reset();
  engine_.reset();
  runtime_.reset();
  free_device(d_in_);
  free_device(d_out_);
  free_device(d_proto_);
  free_device(d_block_);
  free_device(d_logits_);
  free_device(d_colors_);
  if (stream_) {cudaStreamDestroy(static_cast<cudaStream_t>(stream_));}
  if (mask_stream_) {cudaStreamDestroy(static_cast<cudaStream_t>(mask_stream_));}
}

void TrtYolo::set_class_colors(const std::vector<cv::Vec3b> & rgb)
{
  if (rgb.empty()) {
    return;
  }
  std::vector<uint8_t> table(256 * 3);
  for (size_t c = 0; c < 256; ++c) {
    const cv::Vec3b & v = rgb[c % rgb.size()];
    table[c * 3] = v[0];
    table[c * 3 + 1] = v[1];
    table[c * 3 + 2] = v[2];
  }
  check_cuda(cudaMemcpy(d_colors_, table.data(), table.size(), cudaMemcpyHostToDevice), "colors");
}

void TrtYolo::release_image_buffers()
{
  for (auto & s : slots_) {
    for (auto & e : s->exec) {
      if (e) {cudaGraphExecDestroy(e); e = nullptr;}
    }
    free_host(s->h_img);
    free_host(s->h_block);
    free_host(s->h_ids);
    free_host(s->h_vis);
    free_host(s->h_out);
  }
  free_device(d_img_);
  free_device(d_ids_);
  free_device(d_vis_);
}

void TrtYolo::configure(int img_w, int img_h)
{
  for (const auto & s : slots_) {
    if (s->busy) {
      throw std::runtime_error("영상 크기가 바뀌었는데 아직 collect 하지 않은 슬롯이 있음");
    }
  }
  auto st = static_cast<cudaStream_t>(stream_);
  check_cuda(cudaStreamSynchronize(st), "sync");
  release_image_buffers();
  img_w_ = img_w;
  img_h_ = img_h;

  // 비율 유지 축소 + 회색(114) 덧대기 — Ultralytics LetterBox 와 같은 방식
  lb_scale_ = std::min(static_cast<float>(in_w_) / img_w, static_cast<float>(in_h_) / img_h);
  lb_nw_ = static_cast<int>(std::round(img_w * lb_scale_));
  lb_nh_ = static_cast<int>(std::round(img_h * lb_scale_));
  lb_pad_x_ = (in_w_ - lb_nw_) / 2.0f;
  lb_pad_y_ = (in_h_ - lb_nh_) / 2.0f;
  lb_left_ = static_cast<int>(std::round(lb_pad_x_ - 0.1f));
  lb_top_ = static_cast<int>(std::round(lb_pad_y_ - 0.1f));
  direct_input_ = in_u8_ && img_w == in_w_ && img_h == in_h_;

  const size_t img_bytes = static_cast<size_t>(img_w) * img_h * 3;
  const size_t ids_bytes = static_cast<size_t>(img_w) * img_h;
  if (!direct_input_) {
    check_cuda(cudaMalloc(&d_img_, img_bytes), "cudaMalloc img");
  }
  if (has_masks()) {
    check_cuda(cudaMalloc(&d_ids_, ids_bytes), "cudaMalloc ids");
    check_cuda(cudaMalloc(&d_vis_, img_bytes), "cudaMalloc vis");
  }
  for (auto & s : slots_) {
    check_cuda(cudaMallocHost(&s->h_img, img_bytes), "cudaMallocHost img");
    if (end2end_) {
      check_cuda(cudaMallocHost(&s->h_block, sizeof(gpu::DetBlock)), "cudaMallocHost dets");
      s->h_block->count = 0;
    } else {
      check_cuda(cudaMallocHost(&s->h_out, out_elems_ * sizeof(float)), "cudaMallocHost out");
    }
    if (has_masks()) {
      check_cuda(cudaMallocHost(&s->h_ids, ids_bytes), "cudaMallocHost ids");
      check_cuda(cudaMallocHost(&s->h_vis, img_bytes), "cudaMallocHost vis");
    }
  }

  // 한 번 그대로 돌려 TensorRT 가 준비를 끝내게 한 뒤(그래프에 담기 전 필요), 슬롯·결과 영상 유무마다 그래프에 담는다.
  // 조건 노드 그래프 → 일반 그래프 → 그래프 없음 순으로 되는 것을 쓴다
  std::memset(slots_[0]->h_img, 114, img_bytes);
  enqueue_front(*slots_[0], false, 0);
  if (has_masks()) {enqueue_masks(stream_);}
  enqueue_back(*slots_[0], has_masks());
  check_cuda(cudaStreamSynchronize(st), "warmup");

  for (int mode : {2, 1}) {
    if (mode == 2 && !(end2end_ && has_masks())) {
      continue;
    }
    bool ok = true;
    for (auto & s : slots_) {
      for (int v = 0; v < (has_masks() ? 2 : 1) && ok; ++v) {
        ok = capture(*s, v == 1, mode == 2, s->exec[v]);
      }
    }
    if (ok) {
      graph_mode_ = mode;
      return;
    }
    for (auto & s : slots_) {
      for (auto & e : s->exec) {
        if (e) {cudaGraphExecDestroy(e); e = nullptr;}
      }
    }
    std::cerr << "[TrtYolo] " << (mode == 2 ? "조건 노드 그래프" : "CUDA 그래프") << "를 못 만들어 다음 방식으로" << std::endl;
  }
  graph_mode_ = 0;
}

bool TrtYolo::capture(Slot & s, bool vis, bool conditional, cudaGraphExec_t & exec)
{
  auto st = static_cast<cudaStream_t>(stream_);
  auto ms = static_cast<cudaStream_t>(mask_stream_);
  cudaGraph_t graph = nullptr;
  if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  bool ok = true;
  try {
    if (conditional) {
      // 탐지 → 조건 노드(물체가 있으면 마스크 커널) → 결과 영상·D2H. 조건 값은 해석 커널이 장치에서 정한다
      cudaStreamCaptureStatus cs;
      cudaGraph_t g = nullptr;
      check_cuda(cudaStreamGetCaptureInfo(st, &cs, nullptr, &g, nullptr, nullptr), "capture info");
      cudaGraphConditionalHandle cond;
      check_cuda(cudaGraphConditionalHandleCreate(&cond, g, 0, cudaGraphCondAssignDefault), "conditional handle");
      enqueue_front(s, true, cond);
      const cudaGraphNode_t * deps = nullptr;
      size_t n_deps = 0;
      check_cuda(cudaStreamGetCaptureInfo(st, &cs, nullptr, &g, &deps, &n_deps), "capture info");
      cudaGraphNodeParams p = {};
      p.type = cudaGraphNodeTypeConditional;
      p.conditional.handle = cond;
      p.conditional.type = cudaGraphCondTypeIf;
      p.conditional.size = 1;
      cudaGraphNode_t node;
      check_cuda(cudaGraphAddNode(&node, g, deps, n_deps, &p), "conditional node");
      check_cuda(cudaStreamUpdateCaptureDependencies(st, &node, 1, cudaStreamSetCaptureDependencies), "deps");
      check_cuda(cudaStreamBeginCaptureToGraph(ms, p.conditional.phGraph_out[0], nullptr, nullptr, 0,
        cudaStreamCaptureModeThreadLocal), "capture body");
      cudaGraph_t body;
      try {
        enqueue_masks(ms);
      } catch (...) {
        cudaStreamEndCapture(ms, &body);
        throw;
      }
      check_cuda(cudaStreamEndCapture(ms, &body), "end body");
    } else {
      enqueue_front(s, false, 0);
      if (has_masks()) {enqueue_masks(stream_);}
    }
    enqueue_back(s, vis);
  } catch (const std::exception & e) {
    std::cerr << "[TrtYolo] 그래프 담기 중 오류: " << e.what() << std::endl;
    ok = false;
  }
  ok = cudaStreamEndCapture(st, &graph) == cudaSuccess && ok;
  ok = ok && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
  if (graph) {cudaGraphDestroy(graph);}
  if (!ok) {
    cudaGetLastError();
    if (exec) {cudaGraphExecDestroy(exec); exec = nullptr;}
  }
  return ok;
}

gpu::Letterbox TrtYolo::letterbox_params() const
{
  return {lb_scale_, lb_pad_x_, lb_pad_y_, img_w_, img_h_, in_w_, in_h_, lb_nw_, lb_nh_, lb_left_, lb_top_};
}

void TrtYolo::enqueue_front(Slot & s, bool set_cond, unsigned long long cond)
{
  auto st = static_cast<cudaStream_t>(stream_);
  const size_t img_bytes = static_cast<size_t>(img_w_) * img_h_ * 3;
  const gpu::Letterbox lb = letterbox_params();
  if (direct_input_) {
    check_cuda(cudaMemcpyAsync(d_in_, s.h_img, img_bytes, cudaMemcpyHostToDevice, st), "H2D");
  } else {
    check_cuda(cudaMemcpyAsync(d_img_, s.h_img, img_bytes, cudaMemcpyHostToDevice, st), "H2D");
    gpu::letterbox(static_cast<const uint8_t *>(d_img_), d_in_, !in_u8_, lb, st);
  }
  if (!context_->enqueueV3(st)) {
    throw std::runtime_error("enqueueV3 실패");
  }
  if (end2end_) {
    auto * block = static_cast<gpu::DetBlock *>(d_block_);
    gpu::decode_end2end(static_cast<const float *>(d_out_), out_rows_, out_cols_, conf_, lb, ph_, pw_, block,
      set_cond, cond, st);
    check_cuda(cudaMemcpyAsync(s.h_block, block, sizeof(gpu::DetBlock), cudaMemcpyDeviceToHost, st), "D2H dets");
    if (has_masks()) {
      check_cuda(cudaMemsetAsync(d_ids_, 0, static_cast<size_t>(img_w_) * img_h_, st), "memset ids");
    }
  } else {
    check_cuda(cudaMemcpyAsync(s.h_out, d_out_, out_elems_ * sizeof(float), cudaMemcpyDeviceToHost, st), "D2H");
  }
  check_cuda(cudaGetLastError(), "kernel launch");
}

void TrtYolo::enqueue_masks(void * stream)
{
  auto st = static_cast<cudaStream_t>(stream);
  const auto * block = static_cast<const gpu::DetBlock *>(d_block_);
  auto * logits = static_cast<float *>(d_logits_);
  gpu::mask_logits(static_cast<const float *>(d_out_), out_cols_, gpu::Proto{d_proto_, proto_half_, nm_, ph_, pw_},
    block, logits, st);
  gpu::paint_masks(block, logits, ph_, pw_, letterbox_params(), static_cast<uint8_t *>(d_ids_), st);
  check_cuda(cudaGetLastError(), "kernel launch");
}

void TrtYolo::enqueue_back(Slot & s, bool vis)
{
  if (!has_masks()) {
    return;
  }
  auto st = static_cast<cudaStream_t>(stream_);
  const size_t n = static_cast<size_t>(img_w_) * img_h_;
  if (vis) {
    const auto * src = static_cast<const uint8_t *>(direct_input_ ? d_in_ : d_img_);
    gpu::draw_masks(src, static_cast<const uint8_t *>(d_ids_), static_cast<const gpu::DetBlock *>(d_block_),
      static_cast<const uint8_t *>(d_colors_), img_w_, img_h_, static_cast<uint8_t *>(d_vis_), st);
    check_cuda(cudaMemcpyAsync(s.h_vis, d_vis_, n * 3, cudaMemcpyDeviceToHost, st), "D2H vis");
  }
  check_cuda(cudaMemcpyAsync(s.h_ids, d_ids_, n, cudaMemcpyDeviceToHost, st), "D2H ids");
  check_cuda(cudaGetLastError(), "kernel launch");
}

int TrtYolo::submit(const cv::Mat & rgb, bool want_vis)
{
  if (rgb.type() != CV_8UC3 || rgb.empty()) {
    throw std::runtime_error("submit: RGB CV_8UC3 영상이어야 함");
  }
  const int idx = next_slot_;
  Slot & s = *slots_[idx];
  if (s.busy) {
    throw std::runtime_error("submit: 이 슬롯의 결과를 아직 collect 하지 않음");
  }
  if (rgb.cols != img_w_ || rgb.rows != img_h_) {
    configure(rgb.cols, rgb.rows);
  }
  const size_t row = static_cast<size_t>(rgb.cols) * 3;
  if (rgb.isContinuous()) {
    std::memcpy(s.h_img, rgb.data, row * rgb.rows);
  } else {
    for (int y = 0; y < rgb.rows; ++y) {
      std::memcpy(s.h_img + row * y, rgb.ptr(y), row);
    }
  }
  s.vis = want_vis && has_masks();
  auto st = static_cast<cudaStream_t>(stream_);
  if (s.exec[s.vis ? 1 : 0]) {
    check_cuda(cudaGraphLaunch(s.exec[s.vis ? 1 : 0], st), "cudaGraphLaunch");
  } else {
    enqueue_front(s, false, 0);
    if (has_masks()) {enqueue_masks(stream_);}
    enqueue_back(s, s.vis);
  }
  check_cuda(cudaEventRecord(s.done, st), "cudaEventRecord");
  s.busy = true;
  next_slot_ = (next_slot_ + 1) % kSlots;
  return idx;
}

YoloResult TrtYolo::collect(int slot)
{
  Slot & s = *slots_.at(slot);
  if (!s.busy) {
    throw std::runtime_error("collect: submit 하지 않은 슬롯");
  }
  check_cuda(cudaEventSynchronize(s.done), "cudaEventSynchronize");
  s.busy = false;

  YoloResult r;
  if (!end2end_) {
    r.dets = decode_raw(s.h_out);
    return r;
  }
  const gpu::DetBlock & b = *s.h_block;
  r.dets.reserve(b.count);
  for (int k = 0; k < b.count; ++k) {
    const gpu::Det & d = b.dets[k];
    r.dets.push_back({cv::Rect2f(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1), d.score, d.cls,
        cv::Rect(d.rx, d.ry, d.rw, d.rh)});
  }
  if (has_masks()) {
    r.ids = cv::Mat(img_h_, img_w_, CV_8UC1, s.h_ids);
    if (s.vis) {
      r.vis = cv::Mat(img_h_, img_w_, CV_8UC3, s.h_vis);
    }
  }
  return r;
}

std::vector<Detection> TrtYolo::decode_raw(const float * out) const
{
  const auto unletterbox = [this](float x1, float y1, float x2, float y2) {
      x1 = std::clamp((x1 - lb_pad_x_) / lb_scale_, 0.0f, static_cast<float>(img_w_));
      x2 = std::clamp((x2 - lb_pad_x_) / lb_scale_, 0.0f, static_cast<float>(img_w_));
      y1 = std::clamp((y1 - lb_pad_y_) / lb_scale_, 0.0f, static_cast<float>(img_h_));
      y2 = std::clamp((y2 - lb_pad_y_) / lb_scale_, 0.0f, static_cast<float>(img_h_));
      return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    };
  std::vector<cv::Rect> boxes;
  std::vector<cv::Rect2f> boxes_f;
  std::vector<float> scores;
  std::vector<int> classes;
  const int n = out_cols_;
  for (int a = 0; a < n; ++a) {
    int best = 0;
    float best_s = 0.0f;
    for (int c = 0; c < num_classes_; ++c) {
      const float s = out[static_cast<size_t>(4 + c) * n + a];
      if (s > best_s) {best_s = s; best = c;}
    }
    if (best_s < conf_) {
      continue;
    }
    const float cx = out[a], cy = out[n + a], w = out[2 * n + a], h = out[3 * n + a];
    boxes_f.push_back(unletterbox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2));
    boxes.emplace_back(boxes_f.back());
    scores.push_back(best_s);
    classes.push_back(best);
  }
  std::vector<int> keep;
  cv::dnn::NMSBoxesBatched(boxes, scores, classes, conf_, nms_iou_, keep);
  std::vector<Detection> dets;
  for (int i : keep) {
    const cv::Rect2f & b = boxes_f[i];
    const cv::Point tl(static_cast<int>(std::ceil(b.x)), static_cast<int>(std::ceil(b.y)));
    const cv::Point br(static_cast<int>(std::ceil(b.br().x)), static_cast<int>(std::ceil(b.br().y)));
    dets.push_back({b, scores[i], classes[i], cv::Rect(tl, br)});
  }
  return dets;
}

}  // namespace limo_fire_perception
