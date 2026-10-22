#include "st.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <nlohmann/json.hpp>

SafeTensors::SafeTensors(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("열 수 없음: " + path);
  struct stat sb;
  fstat(fd, &sb);
  size_ = sb.st_size;
  map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map_ == MAP_FAILED) throw std::runtime_error("mmap 실패: " + path);
  const char* p = static_cast<const char*>(map_);
  uint64_t n;
  std::memcpy(&n, p, 8);
  if (8 + n > size_) throw std::runtime_error("헤더 길이 이상: " + path);
  auto h = nlohmann::json::parse(p + 8, p + 8 + n);
  const char* base = p + 8 + n;
  for (auto& [k, v] : h.items()) {
    if (k == "__metadata__") continue;
    StTensor t;
    t.dtype = v["dtype"].get<std::string>();
    t.shape = v["shape"].get<std::vector<int64_t>>();
    auto off = v["data_offsets"].get<std::vector<size_t>>();
    if (base + off[1] > p + size_) throw std::runtime_error("텐서가 파일 밖: " + k);
    t.data = base + off[0];
    t.bytes = off[1] - off[0];
    t_[k] = std::move(t);
  }
}

SafeTensors::~SafeTensors() {
  if (map_) munmap(map_, size_);
}

const StTensor& SafeTensors::get(const std::string& name) const {
  auto it = t_.find(name);
  if (it == t_.end()) throw std::runtime_error("텐서 없음: " + name);
  return it->second;
}

const StTensor& SafeTensors::get(const std::string& name, const std::vector<int64_t>& shape) const {
  const auto& t = get(name);
  if (t.shape != shape) {
    std::string s = "모양 다름: " + name + " [";
    for (auto x : t.shape) s += std::to_string(x) + ",";
    throw std::runtime_error(s + "]");
  }
  return t;
}
