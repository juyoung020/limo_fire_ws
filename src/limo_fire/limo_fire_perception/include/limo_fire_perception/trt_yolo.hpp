// TensorRT 로 YOLO 를 돌린다. ROS 에 의존하지 않는다.
// 전처리·출력 해석·마스크·결과 영상 칠하기를 전부 GPU 에서 하고, 한 장 처리를 CUDA 그래프 하나로 묶어 한 번에 띄운다.
// 입력: uint8 RGB [1, H, W, 3] (전처리를 그래프에 넣은 엔진, 권장) 또는 float [1, 3, H, W] (/255 RGB, 기존 엔진)
// 출력 두 가지를 알아서 구분한다:
//   end2end (YOLO26, NMS 없음): [1, N, 6] = x1 y1 x2 y2 score class   (입력 픽셀 좌표) → GPU 에서 해석
//   raw (YOLOv8/11):           [1, 4 + 클래스 수, 앵커 수] = cx cy w h + 클래스별 점수 → CPU 에서 NMS
// 분할(-seg) end2end 는 [1, N, 6 + 32] (마스크 계수) + 원형 마스크 [1, 32, ph, pw] (fp32·fp16) 로 물체 번호 영상을 GPU 에서 만든다.
// 그래프 안 조건 노드로 물체가 있을 때만 마스크 커널을 돌린다.
// (원형 가지를 엔진 둘로 나눠 건너뛰는 것도 쟀지만, 탐지 있는 장이 0.5 ms 느려져 한 엔진으로 둔다 — README)
//
// 비동기: submit() 은 GPU 에 일을 걸고 바로 돌아오고, collect() 가 그 결과를 기다린다(CPU 는 잠듦, 바쁜 대기 안 함).
// 동시에 걸 수 있는 장 수는 kSlots. 한 장의 결과를 CPU 가 쓰는 동안 다음 장을 GPU 가 돌릴 수 있다.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>

namespace limo_fire_perception
{

namespace gpu {struct Letterbox;}

struct Detection
{
  cv::Rect2f box;  // 원본 이미지 픽셀 좌표 (x, y, w, h)
  float score;
  int class_id;
  cv::Rect mask_roi;  // 상자 안 정수 픽셀 범위 (x1 <= u < x2, 이미지 안). 분할 엔진의 마스크는 이 안에만 있다
};

struct YoloResult
{
  std::vector<Detection> dets;
  // 분할 엔진일 때만: 원본 크기 CV_8UC1, 0 = 배경, i + 1 = dets[i]. 겹치면 상자가 작은 물체가 위.
  // 마스크는 앞 64 개 물체까지 (그 뒤는 상자만)
  cv::Mat ids;
  // submit(…, true) 이고 분할 엔진일 때만: 원본에 물체 마스크를 클래스 색으로 덧칠한 RGB (상자·글자는 안 그림)
  cv::Mat vis;
  // ids·vis 는 고정 메모리를 그대로 보여 주는 것이라 같은 슬롯에 다음 submit() 을 하기 전까지만 유효
};

class TrtYolo
{
public:
  static constexpr int kSlots = 2;

  TrtYolo(const std::string & engine_path, float conf_threshold, float nms_iou);
  ~TrtYolo();
  TrtYolo(const TrtYolo &) = delete;
  TrtYolo & operator=(const TrtYolo &) = delete;

  // 결과 영상에 쓸 클래스 색 (RGB). 클래스 번호 % 색 수
  void set_class_colors(const std::vector<cv::Vec3b> & rgb);

  // RGB(CV_8UC3) 한 장을 GPU 에 건다. want_vis 면 마스크 덧칠한 결과 영상도 만든다. 돌려준 슬롯 번호를 collect() 에 준다
  int submit(const cv::Mat & rgb, bool want_vis = false);
  // 그 슬롯이 끝날 때까지 기다려 결과를 낸다
  YoloResult collect(int slot);
  YoloResult infer(const cv::Mat & rgb, bool want_vis = false) {return collect(submit(rgb, want_vis));}

  int input_width() const {return in_w_;}
  int input_height() const {return in_h_;}
  bool end2end() const {return end2end_;}
  int num_classes() const {return num_classes_;}
  bool has_masks() const {return nm_ > 0;}
  bool uint8_input() const {return in_u8_;}
  // 그래프 꼴: 0 = 그래프 없음, 1 = 그래프(마스크 커널 항상), 2 = 그래프 + 조건 노드(물체 있을 때만 마스크 커널)
  int graph_mode() const {return graph_mode_;}

private:
  struct Slot;

  // 이미지 크기가 정해지거나 바뀌면 버퍼·그래프를 다시 만든다 (모든 슬롯이 비어 있어야 하고, 이전 ids 보기는 무효)
  void configure(int img_w, int img_h);
  void release_image_buffers();
  bool capture(Slot & s, bool vis, bool conditional, cudaGraphExec_t & exec);
  // 한 장 처리. 조건 노드를 쓰면 앞(탐지)·뒤(마스크)를 나눠 담는다
  void enqueue_front(Slot & s, bool set_cond, unsigned long long cond);
  void enqueue_masks(void * stream);
  void enqueue_back(Slot & s, bool vis);
  gpu::Letterbox letterbox_params() const;
  std::vector<Detection> decode_raw(const float * out) const;

  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  };

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  void * stream_ = nullptr;
  void * mask_stream_ = nullptr;  // 조건 노드 몸체를 담을 때 쓰는 두 번째 스트림

  std::string in_name_, out_name_, proto_name_;
  int in_w_ = 640, in_h_ = 640;
  bool in_u8_ = false;
  // 출력 모양: end2end 이면 [N, 6 (+nm)], raw 이면 [4 + 클래스, 앵커]
  bool end2end_ = false;
  int out_rows_ = 0, out_cols_ = 0;
  int num_classes_ = 0;
  int nm_ = 0, ph_ = 0, pw_ = 0;
  bool proto_half_ = false;
  size_t out_elems_ = 0;

  // 장치 버퍼 (모든 슬롯이 같이 쓴다: 한 스트림이라 차례로 돈다)
  void * d_in_ = nullptr;      // 엔진 입력
  void * d_out_ = nullptr;     // 엔진 출력
  void * d_proto_ = nullptr;   // 원형 마스크
  void * d_block_ = nullptr;   // gpu::DetBlock
  void * d_logits_ = nullptr;  // 원형 해상도 로짓 [kMaxMaskDets, ph, pw]
  void * d_colors_ = nullptr;  // 클래스 색 RGB × 256
  void * d_img_ = nullptr;     // 원본 RGB (크기가 엔진 입력과 다르거나 float 입력일 때만)
  void * d_ids_ = nullptr;     // 물체 번호 영상
  void * d_vis_ = nullptr;     // 결과 영상

  // 이미지 크기에 따라 정해지는 것
  int img_w_ = 0, img_h_ = 0;
  bool direct_input_ = false;  // 원본을 엔진 입력에 그대로 복사 (uint8 엔진, 같은 크기)
  float lb_scale_ = 1, lb_pad_x_ = 0, lb_pad_y_ = 0;
  int lb_nw_ = 0, lb_nh_ = 0, lb_left_ = 0, lb_top_ = 0;

  std::vector<std::unique_ptr<Slot>> slots_;
  int next_slot_ = 0;
  int graph_mode_ = 0;

  float conf_;
  float nms_iou_;
};

}  // namespace limo_fire_perception
