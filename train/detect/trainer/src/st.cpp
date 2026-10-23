#include "st.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
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

void st_write(const std::string& path, const std::vector<StWrite>& ts) {
  nlohmann::json h = nlohmann::json::object();
  size_t off = 0;
  for (auto& t : ts) {
    size_t n = 1;
    for (auto d : t.shape) n *= (size_t)d;
    h[t.name] = {{"dtype", "F32"}, {"shape", t.shape}, {"data_offsets", {off, off + n * 4}}};
    off += n * 4;
  }
  std::string hs = h.dump();
  while (hs.size() % 8) hs += ' ';
  std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) throw std::runtime_error("쓸 수 없음: " + tmp);
  uint64_t n = hs.size();
  std::fwrite(&n, 8, 1, f);
  std::fwrite(hs.data(), 1, hs.size(), f);
  for (auto& t : ts) {
    size_t k = 1;
    for (auto d : t.shape) k *= (size_t)d;
    std::fwrite(t.data, 4, k, f);
  }
  std::fclose(f);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("이름 바꾸기 실패: " + path);
}
