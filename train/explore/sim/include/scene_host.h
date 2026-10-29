// 장면(호스트): 삼각형 목록 → BVH 만들기, .scene 파일 읽기·쓰기. 결정적(같은 삼각형이면 같은 BVH).
// .scene 형식(리틀 엔디언): "LFSC" u32 버전 | u32 삼각형 수 | Tri[] | u16 물체[] | u32 마디 수 | Node[] |
//                          u32 물체 이름 수 | (u32 길이, 바이트)[]
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "scene.h"

namespace sim {

struct Scene {
  std::vector<Tri> tris;
  std::vector<uint16_t> obj;
  std::vector<Node> nodes;
  std::vector<std::string> obj_names;
  SceneView view() const { return SceneView{nodes.data(), tris.data(), obj.data()}; }
};

namespace detail {
inline void grow(float lo[3], float hi[3], V3 p) {
  const float v[3] = {p.x, p.y, p.z};
  for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], v[k]); hi[k] = std::max(hi[k], v[k]); }
}
}  // namespace detail

// 가운데 나누기 BVH(가장 긴 축, 무게중심 중앙값). 잎 ≤ leaf_max, 깊이 ≤ kMaxDepth. 삼각형 순서를 잎 순서로 바꾼다
inline void build_bvh(Scene& s, int leaf_max = 4) {
  const int n = (int)s.tris.size();
  if (n == 0) throw std::runtime_error("빈 장면");
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<V3> cen(n);
  for (int i = 0; i < n; ++i) {
    const Tri& t = s.tris[i];
    cen[i] = V3{(t.a.x + t.b.x + t.c.x) / 3.f, (t.a.y + t.b.y + t.c.y) / 3.f, (t.a.z + t.b.z + t.c.z) / 3.f};
  }
  s.nodes.clear();
  s.nodes.reserve(2 * n);
  struct Job { int node, lo, hi, depth; };
  std::vector<Job> jobs;
  s.nodes.push_back(Node{});
  jobs.push_back({0, 0, n, 0});
  while (!jobs.empty()) {
    const Job j = jobs.back();
    jobs.pop_back();
    Node nd{};
    for (int k = 0; k < 3; ++k) { nd.lo[k] = kInf; nd.hi[k] = -kInf; }
    float clo[3] = {kInf, kInf, kInf}, chi[3] = {-kInf, -kInf, -kInf};
    for (int i = j.lo; i < j.hi; ++i) {
      const Tri& t = s.tris[idx[i]];
      detail::grow(nd.lo, nd.hi, t.a);
      detail::grow(nd.lo, nd.hi, t.b);
      detail::grow(nd.lo, nd.hi, t.c);
      detail::grow(clo, chi, cen[idx[i]]);
    }
    const int cnt = j.hi - j.lo;
    int axis = 0;
    for (int k = 1; k < 3; ++k) if (chi[k] - clo[k] > chi[axis] - clo[axis]) axis = k;
    if (cnt <= leaf_max || j.depth >= kMaxDepth - 1 || chi[axis] - clo[axis] <= 0.f) {
      nd.first = j.lo;
      nd.count = cnt;
      s.nodes[j.node] = nd;
      continue;
    }
    const int mid = j.lo + cnt / 2;
    auto key = [&](int i) { const V3& c = cen[i]; return axis == 0 ? c.x : axis == 1 ? c.y : c.z; };
    std::nth_element(idx.begin() + j.lo, idx.begin() + mid, idx.begin() + j.hi, [&](int a, int b) {
      const float ka = key(a), kb = key(b);
      return ka < kb || (ka == kb && a < b);   // 같은 값은 번호로 — 결정적
    });
    nd.first = (int)s.nodes.size();
    nd.count = 0;
    s.nodes[j.node] = nd;
    s.nodes.push_back(Node{});
    s.nodes.push_back(Node{});
    jobs.push_back({nd.first + 1, mid, j.hi, j.depth + 1});
    jobs.push_back({nd.first, j.lo, mid, j.depth + 1});
  }
  std::vector<Tri> tris(n);
  std::vector<uint16_t> obj(n);
  for (int i = 0; i < n; ++i) { tris[i] = s.tris[idx[i]]; obj[i] = s.obj[idx[i]]; }
  s.tris.swap(tris);
  s.obj.swap(obj);
}

inline void save_scene(const Scene& s, const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("못 씀: " + path);
  auto w = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
  const uint32_t ver = 1, nt = (uint32_t)s.tris.size(), nn = (uint32_t)s.nodes.size(), no = (uint32_t)s.obj_names.size();
  w("LFSC", 4); w(&ver, 4);
  w(&nt, 4); w(s.tris.data(), nt * sizeof(Tri)); w(s.obj.data(), nt * sizeof(uint16_t));
  w(&nn, 4); w(s.nodes.data(), nn * sizeof(Node));
  w(&no, 4);
  for (const auto& name : s.obj_names) { const uint32_t l = (uint32_t)name.size(); w(&l, 4); w(name.data(), l); }
  std::fclose(f);
}

inline Scene load_scene(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("못 읽음: " + path);
  auto r = [&](void* p, size_t n) { if (std::fread(p, 1, n, f) != n) throw std::runtime_error("잘린 파일: " + path); };
  char magic[4];
  uint32_t ver, nt, nn, no;
  r(magic, 4); r(&ver, 4);
  if (std::memcmp(magic, "LFSC", 4) != 0 || ver != 1) throw std::runtime_error(".scene 형식 아님: " + path);
  Scene s;
  r(&nt, 4); s.tris.resize(nt); s.obj.resize(nt);
  r(s.tris.data(), nt * sizeof(Tri)); r(s.obj.data(), nt * sizeof(uint16_t));
  r(&nn, 4); s.nodes.resize(nn); r(s.nodes.data(), nn * sizeof(Node));
  r(&no, 4); s.obj_names.resize(no);
  for (auto& name : s.obj_names) { uint32_t l; r(&l, 4); name.resize(l); r(&name[0], l); }
  std::fclose(f);
  return s;
}

}  // namespace sim
