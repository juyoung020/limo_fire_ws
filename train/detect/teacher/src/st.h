// safetensors 읽기: 파일을 mmap 하고 헤더(JSON)에서 텐서 이름 → 위치·모양을 찾는다. pickle 을 쓰지 않는다.
#pragma once
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct StTensor {
  std::string dtype;           // "F32", "I32", "U8", "I64" ...
  std::vector<int64_t> shape;
  const void* data = nullptr;  // mmap 안을 가리킨다
  size_t bytes = 0;
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  const float* f32() const {
    if (dtype != "F32") throw std::runtime_error("텐서가 F32 가 아님: " + dtype);
    return static_cast<const float*>(data);
  }
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);
  ~SafeTensors();
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  bool has(const std::string& name) const { return t_.count(name) != 0; }
  const StTensor& get(const std::string& name) const;
  // 모양까지 확인해서 꺼낸다
  const StTensor& get(const std::string& name, const std::vector<int64_t>& shape) const;
  const std::map<std::string, StTensor>& all() const { return t_; }

 private:
  void* map_ = nullptr;
  size_t size_ = 0;
  std::map<std::string, StTensor> t_;
};
