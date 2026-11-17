#include "limo_fire_perception/trt_siglip.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace limo_fire_perception
{

namespace
{
void check(cudaError_t e, const char * what)
{
  if (e != cudaSuccess) {throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));}
}
float half_to_float(uint16_t h)
{
  const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t f;
  if (e == 0) {
    if (m == 0) {f = s;} else {   // 비정규 수
      int k = -1;
      uint32_t mm = m;
      do {++k; mm <<= 1;} while (!(mm & 0x400));
      f = s | ((127 - 15 - k) << 23) | ((mm & 0x3ff) << 13);
    }
  } else if (e == 31) {f = s | 0x7f800000u | (m << 13);} else {f = s | ((e + 112) << 23) | (m << 13);}
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}
void l2(std::vector<float> & v)
{
  double s = 0;
  for (float x : v) {s += static_cast<double>(x) * x;}
  const float k = s > 0 ? static_cast<float>(1.0 / std::sqrt(s)) : 0.0f;
  for (float & x : v) {x *= k;}
}
}  // namespace

// 엔진 하나: 입력 하나 · 출력 하나(마지막 축 = dim). 입력 첫 축(B)은 동적일 수 있음
struct TrtSiglip::Net
{
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  std::string in_name, out_name;
  nvinfer1::DataType in_type;
  void * d_in = nullptr;
  float * d_out = nullptr;
  size_t in_bytes = 0, out_bytes = 0;
  cudaStream_t st = nullptr;
  ~Net()
  {
    if (d_in) {cudaFree(d_in);}
    if (d_out) {cudaFree(d_out);}
    if (st) {cudaStreamDestroy(st);}
  }
};

void TrtSiglip::Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {std::cerr << "[TensorRT siglip] " << msg << std::endl;}
}

static std::unique_ptr<TrtSiglip::Net> load_net(nvinfer1::IRuntime & rt, const std::string & path, int max_batch, size_t in_elem, size_t dim)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) {throw std::runtime_error("SigLIP2 엔진 파일을 열 수 없음: " + path);}
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto n = std::make_unique<TrtSiglip::Net>();
  n->engine.reset(rt.deserializeCudaEngine(blob.data(), blob.size()));
  if (!n->engine) {throw std::runtime_error("SigLIP2 엔진 역직렬화 실패: " + path);}
  // 작업 메모리는 TrtSiglip 이 두 엔진에 같이 준다 (setDeviceMemoryV2)
  n->ctx.reset(n->engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!n->ctx) {throw std::runtime_error("SigLIP2 실행 문맥을 못 만듦: " + path);}
  for (int i = 0; i < n->engine->getNbIOTensors(); ++i) {
    const char * name = n->engine->getIOTensorName(i);
    if (n->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      n->in_name = name;
      n->in_type = n->engine->getTensorDataType(name);
    } else {
      n->out_name = name;
    }
  }
  const size_t esz = (n->in_type == nvinfer1::DataType::kFLOAT || n->in_type == nvinfer1::DataType::kINT32) ? 4 : 8;
  n->in_bytes = static_cast<size_t>(max_batch) * in_elem * esz;
  n->out_bytes = static_cast<size_t>(max_batch) * dim * sizeof(float);
  check(cudaMalloc(&n->d_in, n->in_bytes), "malloc in");
  check(cudaMalloc(reinterpret_cast<void **>(&n->d_out), n->out_bytes), "malloc out");
  check(cudaStreamCreate(&n->st), "stream");
  return n;
}

TrtSiglip::TrtSiglip(const std::string & vision_engine, const std::string & text_engine, const std::string & token_table)
{
  const int fd = ::open(token_table.c_str(), O_RDONLY);
  if (fd < 0) {throw std::runtime_error("SigLIP2 토큰 표 없음: " + token_table);}
  struct stat st{};
  fstat(fd, &st);
  table_bytes_ = static_cast<size_t>(st.st_size);
  void * p = mmap(nullptr, table_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) {throw std::runtime_error("SigLIP2 토큰 표 mmap 실패: " + token_table);}
  table_ = static_cast<const uint16_t *>(p);
  vocab_ = table_bytes_ / (2u * static_cast<size_t>(dim_));
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  vis_ = load_net(*runtime_, vision_engine, max_batch_, 3u * side_ * side_, dim_);
  txt_ = load_net(*runtime_, text_engine, 1, static_cast<size_t>(text_len_) * dim_, dim_);
  // 두 엔진은 차례로만 도니 작업 메모리 한 덩이(둘 중 큰 것)를 같이 쓴다
  const int64_t vb = vis_->engine->getDeviceMemorySizeV2(), tb = txt_->engine->getDeviceMemorySizeV2();
  separate_bytes_ = vb + tb;
  scratch_bytes_ = std::max<int64_t>(std::max(vb, tb), 1);
  check(cudaMalloc(&scratch_, static_cast<size_t>(scratch_bytes_)), "malloc scratch");
  vis_->ctx->setDeviceMemoryV2(scratch_, scratch_bytes_);
  txt_->ctx->setDeviceMemoryV2(scratch_, scratch_bytes_);
  if (vis_->in_type != nvinfer1::DataType::kFLOAT) {throw std::runtime_error("SigLIP2 그림 엔진 입력은 float32 여야 함");}
  if (txt_->in_type != nvinfer1::DataType::kFLOAT) {throw std::runtime_error("SigLIP2 글 엔진 입력은 float32 token_embeds [1,64,768] 여야 함");}
}

TrtSiglip::~TrtSiglip()
{
  vis_.reset();   // 작업 메모리를 쓰는 문맥을 먼저 지운다
  txt_.reset();
  if (scratch_) {cudaFree(scratch_);}
  if (table_) {munmap(const_cast<uint16_t *>(table_), table_bytes_);}
}

std::vector<std::vector<float>> TrtSiglip::embed_images(const std::vector<cv::Mat> & rgb)
{
  std::vector<std::vector<float>> out;
  for (size_t b0 = 0; b0 < rgb.size(); b0 += static_cast<size_t>(max_batch_)) {
    const int B = static_cast<int>(std::min(rgb.size() - b0, static_cast<size_t>(max_batch_)));
    std::vector<float> host(static_cast<size_t>(B) * 3 * side_ * side_);
    for (int b = 0; b < B; ++b) {
      cv::Mat r;
      cv::resize(rgb[b0 + b], r, cv::Size(side_, side_), 0, 0, cv::INTER_LINEAR);   // SiglipImageProcessor: 224×224 로 늘림 (비율 안 지킴)
      float * p = host.data() + static_cast<size_t>(b) * 3 * side_ * side_;
      for (int y = 0; y < side_; ++y) {
        const uint8_t * row = r.ptr<uint8_t>(y);
        for (int x = 0; x < side_; ++x) {
          for (int c = 0; c < 3; ++c) {
            p[static_cast<size_t>(c) * side_ * side_ + static_cast<size_t>(y) * side_ + x] = (row[x * 3 + c] / 255.0f - 0.5f) / 0.5f;
          }
        }
      }
    }
    vis_->ctx->setInputShape(vis_->in_name.c_str(), nvinfer1::Dims4{B, 3, side_, side_});
    vis_->ctx->setTensorAddress(vis_->in_name.c_str(), vis_->d_in);
    vis_->ctx->setTensorAddress(vis_->out_name.c_str(), vis_->d_out);
    check(cudaMemcpyAsync(vis_->d_in, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, vis_->st), "h2d img");
    if (!vis_->ctx->enqueueV3(vis_->st)) {throw std::runtime_error("SigLIP2 그림 enqueueV3 실패");}
    std::vector<float> res(static_cast<size_t>(B) * dim_);
    check(cudaMemcpyAsync(res.data(), vis_->d_out, res.size() * sizeof(float), cudaMemcpyDeviceToHost, vis_->st), "d2h img");
    check(cudaStreamSynchronize(vis_->st), "sync img");
    for (int b = 0; b < B; ++b) {
      out.emplace_back(res.begin() + static_cast<long>(b) * dim_, res.begin() + static_cast<long>(b + 1) * dim_);
      l2(out.back());
    }
  }
  return out;
}

std::vector<float> TrtSiglip::embed_text(const std::vector<int64_t> & ids)
{
  if (static_cast<int>(ids.size()) != text_len_) {throw std::runtime_error("SigLIP2 글 토큰은 64 개여야 함");}
  std::vector<float> h(static_cast<size_t>(text_len_) * dim_);
  for (int t = 0; t < text_len_; ++t) {   // 토큰 표 한 줄씩 (mmap — 쓰는 줄만 읽힘)
    const size_t id = static_cast<size_t>(ids[static_cast<size_t>(t)]);
    if (id >= vocab_) {throw std::runtime_error("SigLIP2 토큰 번호가 표 밖");}
    const uint16_t * row = table_ + id * static_cast<size_t>(dim_);
    for (int k = 0; k < dim_; ++k) {h[static_cast<size_t>(t) * dim_ + k] = half_to_float(row[k]);}
  }
  txt_->ctx->setInputShape(txt_->in_name.c_str(), nvinfer1::Dims3{1, text_len_, dim_});
  txt_->ctx->setTensorAddress(txt_->in_name.c_str(), txt_->d_in);
  txt_->ctx->setTensorAddress(txt_->out_name.c_str(), txt_->d_out);
  check(cudaMemcpyAsync(txt_->d_in, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice, txt_->st), "h2d txt");
  if (!txt_->ctx->enqueueV3(txt_->st)) {throw std::runtime_error("SigLIP2 글 enqueueV3 실패");}
  std::vector<float> res(static_cast<size_t>(dim_));
  check(cudaMemcpyAsync(res.data(), txt_->d_out, res.size() * sizeof(float), cudaMemcpyDeviceToHost, txt_->st), "d2h txt");
  check(cudaStreamSynchronize(txt_->st), "sync txt");
  l2(res);
  return res;
}

}  // namespace limo_fire_perception
