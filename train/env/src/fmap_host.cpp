// .fmap 읽기 + 표면·시작 칸 표시 + 장치로 올리기
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "fmap.h"

namespace fm {

namespace {

template <class T> T rd(std::ifstream& f) {
  T v;
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!f) throw std::runtime_error("fmap: 파일이 짧다");
  return v;
}

struct One {
  uint32_t w = 0, h = 0;
  float ox = 0, oy = 0;
  std::vector<uint8_t> cell;
  std::vector<float> ext, exit, det;
};

One read_fmap(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("fmap: 열 수 없음 " + path);
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "FMAP", 4) != 0) throw std::runtime_error("fmap: 형식 아님 " + path);
  if (rd<uint32_t>(f) != 1) throw std::runtime_error("fmap: 판 1 만 읽는다 " + path);
  One o;
  o.w = rd<uint32_t>(f);
  o.h = rd<uint32_t>(f);
  float res = rd<float>(f);
  if (std::fabs(res - RES) > 1e-6f) throw std::runtime_error("fmap: 해상도가 0.05 m 가 아님 " + path);
  o.ox = rd<float>(f);
  o.oy = rd<float>(f);
  size_t n = (size_t)o.w * o.h;
  o.cell.resize(n);
  f.read(reinterpret_cast<char*>(o.cell.data()), (std::streamsize)n);
  f.seekg((std::streamoff)(n * 2), std::ios::cur);   // 방 번호(지금은 안 씀)
  uint32_t nr = rd<uint32_t>(f);
  f.seekg((std::streamoff)(nr * 5), std::ios::cur);
  for (std::vector<float>* pts : {&o.ext, &o.exit, &o.det}) {
    uint32_t k = rd<uint32_t>(f);
    for (uint32_t i = 0; i < k; ++i) { pts->push_back(rd<float>(f)); pts->push_back(rd<float>(f)); }
  }
  return o;
}

// 벽·바깥 칸까지 거리(m, 8-이웃 BFS 근사) — fmap lib.rs wall_distance 와 같은 방식
std::vector<float> wall_distance(const One& o) {
  size_t n = o.cell.size();
  std::vector<float> d(n, INFINITY);
  std::deque<uint32_t> q;
  for (uint32_t k = 0; k < n; ++k) if (o.cell[k] != FREE) { d[k] = 0.f; q.push_back(k); }
  const float diag = RES * 1.41421356f;
  while (!q.empty()) {
    uint32_t k = q.front(); q.pop_front();
    int i = (int)(k % o.w), j = (int)(k / o.w);
    for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
      if (!di && !dj) continue;
      int a = i + di, b = j + dj;
      if (a < 0 || b < 0 || a >= (int)o.w || b >= (int)o.h) continue;
      uint32_t m = (uint32_t)b * o.w + (uint32_t)a;
      float nd = d[k] + ((di && dj) ? diag : RES);
      if (nd + 1e-6f < d[m]) { d[m] = nd; q.push_back(m); }
    }
  }
  return d;
}

}  // namespace

Bank load_bank(const std::string& dir, const std::string& source, size_t limit, size_t skip) {
  std::ifstream idx(dir + "/index.tsv");
  if (!idx) throw std::runtime_error("fmap: index.tsv 없음 " + dir);
  std::string line;
  std::getline(idx, line);
  Bank b;
  size_t seen = 0;
  while (b.names.size() < limit && std::getline(idx, line)) {
    std::istringstream ss(line);
    std::string src, id;
    std::getline(ss, src, '\t');
    std::getline(ss, id, '\t');
    if (!source.empty() && src != source) continue;
    if (seen++ < skip) continue;
    One o = read_fmap(dir + "/" + src + "/" + id + ".fmap");
    std::vector<float> wd = wall_distance(o);
    uint32_t n_target = 0, n_free = 0;
    std::vector<uint32_t> frees, starts;
    for (uint32_t k = 0; k < o.cell.size(); ++k) {
      uint8_t c = o.cell[k];
      if (c == FREE) {
        frees.push_back(k);
        if (wd[k] >= START_CLEAR) { starts.push_back(k); c |= START; }
        ++n_free;
      } else {
        int i = (int)(k % o.w), j = (int)(k / o.w);
        bool touch = false;
        for (int dj = -1; dj <= 1 && !touch; ++dj) for (int di = -1; di <= 1 && !touch; ++di) {
          int a = i + di, bb = j + dj;
          if (a < 0 || bb < 0 || a >= (int)o.w || bb >= (int)o.h) continue;
          touch = o.cell[(size_t)bb * o.w + a] == FREE;
        }
        if (touch) { c |= TARGET; ++n_target; }
      }
      o.cell[k] = c;
    }
    if (starts.empty() || frees.empty() || n_target == 0) continue;   // 쓸 수 없는 맵
    b.names.push_back(src + "/" + id);
    b.off.push_back(b.cell.size());
    b.cell.insert(b.cell.end(), o.cell.begin(), o.cell.end());
    b.w.push_back(o.w); b.h.push_back(o.h); b.ox.push_back(o.ox); b.oy.push_back(o.oy);
    b.free_off.push_back((uint32_t)b.idx.size()); b.free_n.push_back((uint32_t)frees.size());
    b.idx.insert(b.idx.end(), frees.begin(), frees.end());
    b.start_off.push_back((uint32_t)b.idx.size()); b.start_n.push_back((uint32_t)starts.size());
    b.idx.insert(b.idx.end(), starts.begin(), starts.end());
    b.n_target.push_back(n_target);
    b.area.push_back((float)n_free * RES * RES);
    auto pack = [](std::vector<float>& dst, std::vector<uint32_t>& cnt, const std::vector<float>& src) {
      uint32_t k = (uint32_t)std::min<size_t>(src.size() / 2, MAX_FAC);
      cnt.push_back(k);
      for (int i = 0; i < 2 * MAX_FAC; ++i) dst.push_back(i < (int)(2 * k) ? src[i] : NAN);
    };
    pack(b.ext_xy, b.n_ext, o.ext);
    pack(b.exit_xy, b.n_exit, o.exit);
    pack(b.det_xy, b.n_det, o.det);
    b.max_cells = std::max<uint32_t>(b.max_cells, o.w * o.h);
  }
  if (b.names.empty()) throw std::runtime_error("fmap: 읽은 맵이 없다 (" + dir + ", " + source + ")");
  return b;
}

View Bank::host_view() const {
  View v;
  v.cell = cell.data(); v.off = off.data(); v.w = w.data(); v.h = h.data(); v.ox = ox.data(); v.oy = oy.data();
  v.free_off = free_off.data(); v.free_n = free_n.data(); v.start_off = start_off.data(); v.start_n = start_n.data();
  v.idx = idx.data(); v.n_target = n_target.data(); v.area = area.data();
  v.n_maps = (int)names.size();
  return v;
}

namespace {
template <class T> const T* put(DeviceBank& d, const std::vector<T>& h) {
  void* p = nullptr;
  if (cudaMalloc(&p, h.size() * sizeof(T)) != cudaSuccess) throw std::runtime_error("fmap: cudaMalloc 실패");
  cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  d.blocks[d.n_blocks++] = p;
  return static_cast<const T*>(p);
}
}  // namespace

DeviceBank upload(const Bank& b) {
  DeviceBank d;
  View& v = d.view;
  v.cell = put(d, b.cell); v.off = put(d, b.off); v.w = put(d, b.w); v.h = put(d, b.h);
  v.ox = put(d, b.ox); v.oy = put(d, b.oy);
  v.free_off = put(d, b.free_off); v.free_n = put(d, b.free_n);
  v.start_off = put(d, b.start_off); v.start_n = put(d, b.start_n);
  v.idx = put(d, b.idx); v.n_target = put(d, b.n_target); v.area = put(d, b.area);
  v.n_maps = (int)b.names.size();
  return d;
}

void free_device(DeviceBank& d) {
  for (int i = 0; i < d.n_blocks; ++i) cudaFree(d.blocks[i]);
  d.n_blocks = 0;
}

}  // namespace fm
