// HSSD(Habitat Synthetic Scenes Dataset, hssd-hab) 집 → 학습 시뮬 장면(.scene). train/explore/scenes/README.md "HSSD 메시 장면".
//   mesh2scene --hssd <hssd-hab 폴더> --out <출력 폴더> [--seed 1] [--cluster 0.04] [--cluster-hi 0.12] [--env config/env.yaml] <장면 id> …
// 장면 하나 = stages/<id>.glb + scenes/<id>.scene_instance.json 의 물체(번역·회전 [w,x,y,z]·크기, translation_origin asset_local).
// glTF 는 y 위 → 우리 z 위 (x, y, z) → (x, −z, y). 층(바닥 높이)마다 장면 하나 hssd_<id>_l<층 번호> (아래층부터 0).
//  1. 층 찾기: 무대(stage) 삼각형 중 위를 보는 수평면(법선 z > 0.9) 넓이를 5 cm 높이 칸으로 모아, 8 m² 이상이고 서로 1.5 m 넘게 떨어진 봉우리
//  2. 그 층 바닥 기준 −0.05 ~ +2.0 m 만 남기고(삼각형을 그 높이로 자름) 바닥을 z = 0 으로. 바닥 ±5 cm 안에 다 든 삼각형(바닥·문턱·깔개)은 빼고
//     평평한 바닥 판으로 바꾼다 — 작은 턱이 몸통 높이(collision_band 0.03~0.25 m)에 걸려 막힘이 되지 않게
//  3. 갈 수 있는 바닥 R = 바닥 칸(위 수평면을 칸에 찍고 3×3 십자 닫기) 중 막힘 아닌 칸의 가장 큰 4 이웃 덩어리.
//     마당 빼기: 8 방향 중 6 방향 이상이 20 m 안에서 벽 높이(0.3 ~ 2.0 m) 형상에 닿는 바닥 칸만 — 집 둘레 넓은 땅(마당·지형)이 창 고르기를
//     차지하지 않게. 천장으로 가리면 HSSD 위층(지붕 없는 무대가 많음)이 통째로 빠져서 둘러싸임으로 본다
//     R 에 8 이웃으로 닿은 "바닥도 막힘도 아닌 칸"(계단 구멍, 바깥으로 난 문, 메시 없는 유리) = 가상 벽(높이 2.4 m) — 방 사이 문은 양쪽이 바닥이라 열린 채
//  4. R 의 바운딩 상자가 25.5 m(510 칸)를 넘으면 R 칸이 가장 많은 25.5 m 창을 고르고, 창 가장자리 칸을 R 에서 빼 잘린 곳도 가상 벽으로 막는다.
//     모든 삼각형을 창(x·y)과 높이 범위로 잘라 BVH 상자 = 창 (build_world 격자는 BVH 상자로 정해져 떨어진 형상 하나가 맵을 망친다)
//  5. 출발점: build_world 와 같은 서는 칸(몸통 반폭 안에 막힘 없음)의 가장 큰 덩어리 중 0.4 m 안에 막힘(가상 벽 포함) 없는 칸, 씨앗·이름 해시로 하나
//  6. 물체: 무대 = "hssd_stage"(구조물), HSSD 물체 인스턴스마다 "furniture_<번호>"(라이다 재질을 물체마다 뽑음, scene_list 가 가구 판으로 셈),
//     "virtual_wall", "floor". 문(metadata fpmodels main_category = door, 닫힌 문짝이 문틀에 든 물체)은 빼서 방 사이 문을 연다
//  7. --cluster q (기본 0.04 m) / --cluster-hi (기본 0.12 m): 물체마다 꼭짓점을 칸으로 모아(vertex clustering) 삼각형 수를 줄임 — 바닥 위 0.35 m
//     아래 꼭짓점은 q, 위는 q_hi (라이다 0.111 m·몸통 띠는 촘촘히, 위는 깊이 카메라만 봄). 0 = 끔.
//     물체의 몸통 띠(collision_band) 발자국이 한 칸 넘게 바뀌면(가는 의자 다리가 사라짐) q/2, q/4, 그래도 바뀌면 그 물체는 그대로
//  끝에 build_world(학습과 같은 함수)로 닫힘·넓이를 다시 재고, 닫힌 것만 저장. 표준 출력: 층마다 한 줄 TSV (아래 머리줄)
#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "env_api.h"

namespace fs = std::filesystem;

namespace {

constexpr float kRes = 0.05f;
constexpr float kFloorTol = 0.05f;     // 바닥 ±5 cm
constexpr float kKeepTop = 2.0f;       // 바닥 위 2.0 m 까지
constexpr float kWallH = 2.4f;         // 가상 벽 높이 (절차 맵 벽과 같게)
constexpr int kMaxCells = 510;         // 창 한 변 25.5 m (build_world: 칸 + 1 ≤ 512)
constexpr int kMargin = 5;             // 잘리지 않은 집 둘레 여유 칸 (0.25 m)
constexpr float kStartClear = 0.4f;    // 출발점은 막힘에서 0.4 m 이상 (scene_grids 와 같은 규칙)
constexpr float kMinLevelArea = 8.f;   // 층 봉우리 최소 넓이 (m², 5 cm 높이 칸)
constexpr float kLevelSep = 1.5f;      // 층 사이 최소 높이 차 (m)
constexpr float kClusterSplit = 0.35f;   // 이 높이(바닥 기준) 아래 꼭짓점은 --cluster, 위는 --cluster-hi 로 모음 (몸통 띠 0.25 m·라이다 0.111 m 위 여유)
constexpr float kMinNavArea = 3.f;     // R 이 이보다 작으면 버림 (scene_grids 와 같은 3 m²)
constexpr float kWallLo = 0.3f;   // 둘러싸임 검사에 쓰는 벽 높이 아래 끝 (m)
constexpr float kEncloseReach = 20.f;   // 둘러싸임: 8 방향 중 이 거리(m) 안에 벽이 닿는 방향이 kEncloseDirs 개 이상 — 잠정
constexpr int kEncloseDirs = 6;
constexpr float kMaxOpenFrac = 0.5f;   // R 둘레 중 가상 벽 몫이 이보다 크면 버림 (지붕·마당 같은 열린 층) — 잠정

struct T9 {
  float p[9];
};

// ---------- glTF ----------
// 노드 세계 행렬까지 곱한 삼각형 (glTF y 위, 자산 좌표)
bool load_glb(const std::string& path, std::vector<T9>& out, std::string& err) {
  cgltf_options opt{};
  cgltf_data* d = nullptr;
  if (cgltf_parse_file(&opt, path.c_str(), &d) != cgltf_result_success) { err = "glTF 못 읽음: " + path; return false; }
  if (cgltf_load_buffers(&opt, d, path.c_str()) != cgltf_result_success) { cgltf_free(d); err = "glTF 버퍼 못 읽음: " + path; return false; }
  for (cgltf_size ni = 0; ni < d->nodes_count; ++ni) {
    const cgltf_node* nd = &d->nodes[ni];
    if (!nd->mesh) continue;
    float m[16];
    cgltf_node_transform_world(nd, m);   // 열 우선
    for (cgltf_size pi = 0; pi < nd->mesh->primitives_count; ++pi) {
      const cgltf_primitive& pr = nd->mesh->primitives[pi];
      if (pr.type != cgltf_primitive_type_triangles && pr.type != cgltf_primitive_type_triangle_strip && pr.type != cgltf_primitive_type_triangle_fan) continue;
      const cgltf_accessor* pos = nullptr;
      for (cgltf_size a = 0; a < pr.attributes_count; ++a)
        if (pr.attributes[a].type == cgltf_attribute_type_position) pos = pr.attributes[a].data;
      if (!pos) continue;
      std::vector<float> v(pos->count * 3);
      for (cgltf_size i = 0; i < pos->count; ++i) {
        float q[3] = {0, 0, 0};
        cgltf_accessor_read_float(pos, i, q, 3);
        for (int k = 0; k < 3; ++k) v[3 * i + k] = m[k] * q[0] + m[4 + k] * q[1] + m[8 + k] * q[2] + m[12 + k];
      }
      std::vector<uint32_t> idx;
      if (pr.indices) {
        idx.resize(pr.indices->count);
        for (cgltf_size i = 0; i < pr.indices->count; ++i) idx[i] = (uint32_t)cgltf_accessor_read_index(pr.indices, i);
      } else {
        idx.resize(pos->count);
        for (cgltf_size i = 0; i < pos->count; ++i) idx[i] = (uint32_t)i;
      }
      auto emit = [&](uint32_t a, uint32_t b, uint32_t c) {
        if (a >= pos->count || b >= pos->count || c >= pos->count) return;
        T9 t;
        for (int k = 0; k < 3; ++k) { t.p[k] = v[3 * a + k]; t.p[3 + k] = v[3 * b + k]; t.p[6 + k] = v[3 * c + k]; }
        out.push_back(t);
      };
      if (pr.type == cgltf_primitive_type_triangles)
        for (size_t i = 0; i + 2 < idx.size(); i += 3) emit(idx[i], idx[i + 1], idx[i + 2]);
      else if (pr.type == cgltf_primitive_type_triangle_strip)
        for (size_t i = 0; i + 2 < idx.size(); ++i) (i & 1) ? emit(idx[i + 1], idx[i], idx[i + 2]) : emit(idx[i], idx[i + 1], idx[i + 2]);
      else
        for (size_t i = 1; i + 1 < idx.size(); ++i) emit(idx[0], idx[i], idx[i + 1]);
    }
  }
  cgltf_free(d);
  return true;
}

// ---------- 변환 ----------
struct Xf {
  double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // 회전 × 크기 (행 우선)
  double t[3] = {0, 0, 0};
};
Xf make_xf(const YAML::Node& o) {
  Xf x;
  double q[4] = {1, 0, 0, 0}, s[3] = {1, 1, 1};
  if (o["rotation"]) for (int k = 0; k < 4; ++k) q[k] = o["rotation"][k].as<double>();   // Habitat: [w, x, y, z]
  if (o["translation"]) for (int k = 0; k < 3; ++k) x.t[k] = o["translation"][k].as<double>();
  if (o["non_uniform_scale"]) for (int k = 0; k < 3; ++k) s[k] = o["non_uniform_scale"][k].as<double>();
  if (o["uniform_scale"]) for (int k = 0; k < 3; ++k) s[k] *= o["uniform_scale"].as<double>();
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double w = q[0] / n, a = q[1] / n, b = q[2] / n, c = q[3] / n;
  const double r[9] = {1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w),
                       2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w),
                       2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) x.R[3 * i + j] = r[3 * i + j] * s[j];
  return x;
}
// 자산(y 위) → Habitat 세계(y 위) → 우리 세계(z 위): (x, y, z) → (x, −z, y)
void apply(const Xf& x, const T9& in, T9& out) {
  for (int v = 0; v < 3; ++v) {
    const float* p = &in.p[3 * v];
    double w[3];
    for (int i = 0; i < 3; ++i) w[i] = x.R[3 * i] * p[0] + x.R[3 * i + 1] * p[1] + x.R[3 * i + 2] * p[2] + x.t[i];
    out.p[3 * v] = (float)w[0];
    out.p[3 * v + 1] = (float)-w[2];
    out.p[3 * v + 2] = (float)w[1];
  }
}

// 꼭짓점 모으기 (vertex clustering): q 칸에 든 꼭짓점을 평균 하나로, 꼭짓점이 겹친 삼각형·같은 삼각형은 버림
// 꼭짓점 높이가 z_split 이상이면 칸 q_hi (라이다·몸통 높이 위는 거칠게)
void cluster(std::vector<T9>& tris, float q, float q_hi = 0.f, float z_split = 1e30f) {
  if (q <= 0.f || tris.empty()) return;
  if (q_hi < q) q_hi = q;
  std::unordered_map<uint64_t, uint32_t> id;
  std::vector<double> acc;
  std::vector<uint32_t> cnt, tri_id(tris.size() * 3);
  for (size_t i = 0; i < tris.size(); ++i)
    for (int v = 0; v < 3; ++v) {
      const float* p = &tris[i].p[3 * v];
      const bool hi = p[2] >= z_split;
      const float qq = hi ? q_hi : q;
      const int64_t a = (int64_t)std::floor(p[0] / qq), b = (int64_t)std::floor(p[1] / qq), c = (int64_t)std::floor(p[2] / qq);
      const uint64_t key = ((uint64_t)hi << 63) | ((uint64_t)(a & 0x1fffff) << 42) | ((uint64_t)(b & 0x1fffff) << 21) | (uint64_t)(c & 0x1fffff);
      auto it = id.find(key);
      uint32_t k;
      if (it == id.end()) { k = (uint32_t)cnt.size(); id.emplace(key, k); cnt.push_back(0); acc.insert(acc.end(), {0.0, 0.0, 0.0}); }
      else k = it->second;
      for (int j = 0; j < 3; ++j) acc[3 * k + j] += p[j];
      ++cnt[k];
      tri_id[3 * i + v] = k;
    }
  std::set<std::array<uint32_t, 3>> seen;
  std::vector<T9> out;
  for (size_t i = 0; i < tris.size(); ++i) {
    const uint32_t a = tri_id[3 * i], b = tri_id[3 * i + 1], c = tri_id[3 * i + 2];
    if (a == b || b == c || a == c) continue;
    std::array<uint32_t, 3> s{a, b, c};
    std::sort(s.begin(), s.end());
    if (!seen.insert(s).second) continue;
    T9 t;
    const uint32_t k3[3] = {a, b, c};
    for (int v = 0; v < 3; ++v)
      for (int j = 0; j < 3; ++j) t.p[3 * v + j] = (float)(acc[3 * k3[v] + j] / cnt[k3[v]]);
    out.push_back(t);
  }
  tris.swap(out);
}

// ---------- 자르기 (Sutherland–Hodgman, 축에 맞춘 평면) ----------
using Poly = std::vector<std::array<float, 3>>;
void clip_plane(const Poly& in, Poly& out, int axis, float bound, bool keep_ge) {
  out.clear();
  const size_t n = in.size();
  for (size_t i = 0; i < n; ++i) {
    const auto& a = in[i];
    const auto& b = in[(i + 1) % n];
    const bool ina = keep_ge ? a[axis] >= bound : a[axis] <= bound;
    const bool inb = keep_ge ? b[axis] >= bound : b[axis] <= bound;
    if (ina) out.push_back(a);
    if (ina != inb) {
      const float t = (bound - a[axis]) / (b[axis] - a[axis]);
      std::array<float, 3> p;
      for (int k = 0; k < 3; ++k) p[k] = a[k] + (b[k] - a[k]) * t;
      p[axis] = bound;   // 경계 값 그대로 — BVH 상자가 창을 넘지 않게
      out.push_back(p);
    }
  }
}
// 삼각형을 상자 [lo, hi] 로 잘라 부채꼴 삼각형으로
void clip_box(const T9& t, const float lo[3], const float hi[3], std::vector<T9>& out) {
  Poly a{{t.p[0], t.p[1], t.p[2]}, {t.p[3], t.p[4], t.p[5]}, {t.p[6], t.p[7], t.p[8]}}, b;
  bool inside = true;
  for (int v = 0; v < 3; ++v)
    for (int k = 0; k < 3; ++k) inside = inside && t.p[3 * v + k] >= lo[k] && t.p[3 * v + k] <= hi[k];
  if (inside) { out.push_back(t); return; }
  for (int k = 0; k < 3 && !a.empty(); ++k) {
    clip_plane(a, b, k, lo[k], true);
    clip_plane(b, a, k, hi[k], false);
  }
  for (size_t i = 1; i + 1 < a.size(); ++i) {
    T9 r;
    const std::array<float, 3>* v[3] = {&a[0], &a[i], &a[i + 1]};
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) r.p[3 * j + k] = (*v[j])[k];
    const float ux = r.p[3] - r.p[0], uy = r.p[4] - r.p[1], uz = r.p[5] - r.p[2], vx = r.p[6] - r.p[0], vy = r.p[7] - r.p[1], vz = r.p[8] - r.p[2];
    const float cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    if (cx * cx + cy * cy + cz * cz < 1e-14f) continue;   // 넓이 0
    out.push_back(r);
  }
}

float tri_area_nz(const T9& t, float* nz) {
  const double ux = t.p[3] - t.p[0], uy = t.p[4] - t.p[1], uz = t.p[5] - t.p[2], vx = t.p[6] - t.p[0], vy = t.p[7] - t.p[1], vz = t.p[8] - t.p[2];
  const double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
  const double l = std::sqrt(cx * cx + cy * cy + cz * cz);
  *nz = l > 0 ? (float)(cz / l) : 0.f;
  return (float)(0.5 * l);
}
float zmin(const T9& t) { return std::min({t.p[2], t.p[5], t.p[8]}); }
float zmax(const T9& t) { return std::max({t.p[2], t.p[5], t.p[8]}); }

// ---------- 격자 ----------
struct Grid {
  int W = 0, H = 0;
  float ox = 0, oy = 0;
  std::vector<uint8_t> v;
  uint8_t& at(int x, int y) { return v[(size_t)y * W + x]; }
  uint8_t get(int x, int y) const { return (x < 0 || y < 0 || x >= W || y >= H) ? 0 : v[(size_t)y * W + x]; }
};
// build_world 1 단계와 같은 점 찍기 (칸 반 간격). band 가 있으면 그 높이 안 점만
void raster(const T9& t, Grid& g, bool use_band, float blo, float bhi) {
  const sim::V3 a{t.p[0], t.p[1], t.p[2]}, b{t.p[3], t.p[4], t.p[5]}, c{t.p[6], t.p[7], t.p[8]};
  if (use_band) {
    const float zlo = std::min({a.z, b.z, c.z}), zhi = std::max({a.z, b.z, c.z});
    if (zhi < blo || zlo > bhi) return;
  }
  auto len = [](sim::V3 p, sim::V3 q) { return std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z)); };
  const float e = std::max({len(a, b), len(b, c), len(c, a)});
  const int n = std::max(1, (int)std::ceil(e / (0.5f * kRes)));
  for (int i = 0; i <= n; ++i)
    for (int j = 0; i + j <= n; ++j) {
      const float u = (float)i / n, v = (float)j / n;
      const sim::V3 p{a.x + (b.x - a.x) * u + (c.x - a.x) * v, a.y + (b.y - a.y) * u + (c.y - a.y) * v, a.z + (b.z - a.z) * u + (c.z - a.z) * v};
      if (use_band && (p.z < blo || p.z > bhi)) continue;
      const int cx = (int)std::floor((p.x - g.ox) / kRes), cy = (int)std::floor((p.y - g.oy) / kRes);
      if (cx >= 0 && cy >= 0 && cx < g.W && cy < g.H) g.v[(size_t)cy * g.W + cx] = 1;
    }
}
// 몸통 높이 띠(band) 안 점이 찍히는 칸 (원점 0 격자) — 꼭짓점 모으기가 의자 다리 같은 가는 부분을 지웠나 보려고
std::set<std::pair<int, int>> band_cells(const std::vector<T9>& tris, float blo, float bhi) {
  std::set<std::pair<int, int>> out;
  for (const T9& t : tris) {
    if (zmax(t) < blo || zmin(t) > bhi) continue;
    float lo[2] = {1e30f, 1e30f}, hi[2] = {-1e30f, -1e30f};
    for (int v = 0; v < 3; ++v)
      for (int k = 0; k < 2; ++k) { lo[k] = std::min(lo[k], t.p[3 * v + k]); hi[k] = std::max(hi[k], t.p[3 * v + k]); }
    Grid g;
    g.ox = std::floor(lo[0] / kRes) * kRes;
    g.oy = std::floor(lo[1] / kRes) * kRes;
    g.W = (int)std::ceil((hi[0] - g.ox) / kRes) + 1;
    g.H = (int)std::ceil((hi[1] - g.oy) / kRes) + 1;
    g.v.assign((size_t)g.W * g.H, 0);
    raster(t, g, true, blo, bhi);
    const int bx = (int)std::lround(g.ox / kRes), by = (int)std::lround(g.oy / kRes);
    for (int y = 0; y < g.H; ++y)
      for (int x = 0; x < g.W; ++x)
        if (g.v[(size_t)y * g.W + x]) out.insert({bx + x, by + y});
  }
  return out;
}
// a 의 칸마다 b 에 8 이웃(자기 포함) 칸이 있나
bool covered(const std::set<std::pair<int, int>>& a, const std::set<std::pair<int, int>>& b) {
  for (const auto& c : a) {
    bool ok = false;
    for (int dy = -1; dy <= 1 && !ok; ++dy)
      for (int dx = -1; dx <= 1 && !ok; ++dx) ok = b.count({c.first + dx, c.second + dy}) > 0;
    if (!ok) return false;
  }
  return true;
}
// 물체 하나(층 바닥 기준 z)를 q 로 모으되, 몸통 띠 발자국이 한 칸 넘게 바뀌면 q/2, q/4, 그래도 바뀌면 원래대로. 돌려주는 값: 쓴 q
float cluster_guarded(std::vector<T9>& tris, float q, float q_hi, float z_split, float blo, float bhi) {
  if (q <= 0.f) return 0.f;
  const auto before = band_cells(tris, blo, bhi);
  for (float qq : {q, 0.5f * q, 0.25f * q}) {
    std::vector<T9> c = tris;
    cluster(c, qq, q_hi, z_split);
    const auto after = band_cells(c, blo, bhi);
    if (covered(before, after) && covered(after, before)) { tris.swap(c); return qq; }
  }
  return 0.f;
}

// 3×3 십자 닫기 (팽창 → 침식)
void close_cross(Grid& g) {
  Grid d = g;
  for (int y = 0; y < g.H; ++y)
    for (int x = 0; x < g.W; ++x)
      d.at(x, y) = g.get(x, y) | g.get(x + 1, y) | g.get(x - 1, y) | g.get(x, y + 1) | g.get(x, y - 1);
  for (int y = 0; y < g.H; ++y)
    for (int x = 0; x < g.W; ++x) {
      auto dd = [&](int u, int v) { return (u < 0 || v < 0 || u >= g.W || v >= g.H) ? (uint8_t)1 : d.get(u, v); };
      g.at(x, y) = dd(x, y) & dd(x + 1, y) & dd(x - 1, y) & dd(x, y + 1) & dd(x, y - 1);
    }
}
// 둘러싸임 마스크 (1 차 격자): 칸에서 8 방향으로 벽 높이(0.3 ~ 2.0 m) 형상까지 kEncloseReach 안에 닿는 방향이 kEncloseDirs 개 이상이면 1.
// 집 둘레 넓은 땅(마당·지형)은 바깥쪽 방향이 아무것도 안 맞아 빠지고, 지붕 없는 위층 방도 벽에 둘러싸여 남는다. 방향마다 한 번 쓸기(O(8N))
void enclosure_mask(const Grid& wall, Grid& allow) {
  const int W = wall.W, H = wall.H, R = (int)std::lround(kEncloseReach / kRes);
  std::vector<uint8_t> hits((size_t)W * H, 0);
  std::vector<int> d((size_t)W * H);
  const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  for (const auto& dv : dirs) {
    const int dx = dv[0], dy = dv[1];
    // (x, y) 에서 (dx, dy) 쪽 벽까지 칸 수: d(x, y) = 벽이면 0, 아니면 d(x + dx, y + dy) + 1 — (x + dx, y + dy) 를 먼저 계산하도록 거꾸로 돈다
    const int y0 = dy > 0 ? H - 1 : 0, y1 = dy > 0 ? -1 : H, sy = dy > 0 ? -1 : 1;
    const int x0 = dx > 0 ? W - 1 : 0, x1 = dx > 0 ? -1 : W, sx = dx > 0 ? -1 : 1;
    for (int y = y0; y != y1; y += sy)
      for (int x = x0; x != x1; x += sx) {
        const size_t k = (size_t)y * W + x;
        if (wall.v[k]) { d[k] = 0; continue; }
        const int u = x + dx, v = y + dy;
        d[k] = (u < 0 || v < 0 || u >= W || v >= H) ? (1 << 28) : d[(size_t)v * W + u] + 1;
      }
    const int lim = (dx && dy) ? (int)(R / 1.4142f) : R;
    for (size_t k = 0; k < d.size(); ++k) hits[k] += d[k] <= lim;
  }
  allow = wall;
  for (size_t k = 0; k < hits.size(); ++k) allow.v[k] = hits[k] >= kEncloseDirs;
}
// 가장 큰 4 이웃 덩어리 (같은 크기면 먼저 찾은 것)
Grid largest_comp(const Grid& m) {
  Grid out = m;
  std::fill(out.v.begin(), out.v.end(), 0);
  std::vector<int> lab(m.v.size(), -1), st, best;
  size_t best_n = 0;
  int nl = 0;
  for (int k = 0; k < (int)m.v.size(); ++k) {
    if (!m.v[k] || lab[k] >= 0) continue;
    std::vector<int> cells;
    st.assign(1, k);
    lab[k] = nl;
    while (!st.empty()) {
      const int c = st.back();
      st.pop_back();
      cells.push_back(c);
      const int x = c % m.W, y = c / m.W;
      const int nb[4][2] = {{x + 1, y}, {x - 1, y}, {x, y + 1}, {x, y - 1}};
      for (const auto& q : nb) {
        if (q[0] < 0 || q[1] < 0 || q[0] >= m.W || q[1] >= m.H) continue;
        const int j = q[1] * m.W + q[0];
        if (m.v[j] && lab[j] < 0) { lab[j] = nl; st.push_back(j); }
      }
    }
    if (cells.size() > best_n) { best_n = cells.size(); best.swap(cells); }
    ++nl;
  }
  for (int c : best) out.v[c] = 1;
  return out;
}

uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
uint64_t fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
  return h;
}

void add_box(sim::Scene& s, uint16_t obj, sim::V3 lo, sim::V3 hi) {
  const sim::V3 p[8] = {{lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z},
                        {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}};
  const int f[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
  for (const auto& q : f) {
    s.tris.push_back(sim::Tri{p[q[0]], p[q[1]], p[q[2]]});
    s.tris.push_back(sim::Tri{p[q[0]], p[q[2]], p[q[3]]});
    s.obj.push_back(obj);
    s.obj.push_back(obj);
  }
}

// CSV 한 줄 (따옴표 안 쉼표·"" 다룸)
std::vector<std::string> csv_split(const std::string& line) {
  std::vector<std::string> col(1);
  bool q = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (q) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') { col.back() += '"'; ++i; }
      else if (c == '"') q = false;
      else col.back() += c;
    } else if (c == '"') q = true;
    else if (c == ',') col.emplace_back();
    else col.back() += c;
  }
  return col;
}

struct Inst {
  std::string tmpl;
  std::vector<T9> tris;   // 우리 세계 (z 위)
};

struct Ctx {
  fs::path root, out;
  uint64_t seed = 1;
  float cluster_q = 0.04f;   // --cluster (m), 0 = 끔
  float cluster_q_hi = 0.12f;   // --cluster-hi: 바닥 위 kClusterSplit 넘는 꼭짓점의 칸 (m)
  sim::EnvHostCfg hc;
  std::unordered_map<std::string, fs::path> obj_cfg;   // 틀 이름 → object_config.json
  std::set<std::string> drop_ids;                     // 빼는 물체 (문)
};

// 층 하나 → 장면. 돌려주는 값: TSV 한 줄
std::string make_level(const Ctx& C, const std::string& id, int li, float fz, const std::vector<T9>& stage, const std::vector<Inst>& inst) {
  const std::string name = "hssd_" + id + "_l" + std::to_string(li);
  char buf[1024];
  auto reject = [&](const std::string& why) {
    std::snprintf(buf, sizeof(buf), "%s\treject\t%.3f\t%s", name.c_str(), fz, why.c_str());
    return std::string(buf);
  };
  // 높이로 고르고 내림 (−0.05 ~ 2.0 와 겹치는 삼각형), 바닥 ±5 cm 안에 다 든 것은 바닥 칸 표시용으로만
  struct LT { T9 t; int obj; };   // obj: −1 무대, i 인스턴스
  std::vector<LT> L;
  std::vector<T9> floor_tris;
  auto take = [&](const T9& t0, int obj) {
    T9 t = t0;
    t.p[2] -= fz; t.p[5] -= fz; t.p[8] -= fz;
    const float lo = zmin(t), hi = zmax(t);
    if (hi < -kFloorTol || lo > kKeepTop) return;
    if (lo >= -kFloorTol && hi <= kFloorTol) {
      float nz;
      tri_area_nz(t, &nz);
      if (nz > 0.9f) floor_tris.push_back(t);
      return;
    }
    L.push_back({t, obj});
  };
  for (const T9& t : stage) take(t, -1);
  for (size_t i = 0; i < inst.size(); ++i)
    for (const T9& t : inst[i].tris) take(t, (int)i);
  if (floor_tris.empty()) return reject("바닥 삼각형 없음");
  // 물체마다 꼭짓점 모으기 (무대는 그대로 — 삼각형이 적고 벽 면이 곧다)
  int n_q_half = 0, n_q_off = 0;
  if (C.cluster_q > 0.f) {
    std::vector<LT> keep;
    std::map<int, std::vector<T9>> per;
    for (LT& t : L) {
      if (t.obj < 0) keep.push_back(t);
      else per[t.obj].push_back(t.t);
    }
    for (auto& [o, tr] : per) {
      const float q = cluster_guarded(tr, C.cluster_q, C.cluster_q_hi, kClusterSplit, C.hc.band_lo, C.hc.band_hi);
      if (q == 0.f) ++n_q_off;
      else if (q < C.cluster_q) ++n_q_half;
      for (const T9& t : tr) keep.push_back({t, o});
    }
    L.swap(keep);
  }
  // 1 차 격자: 바닥 삼각형 상자 + 1 m
  float flo[2] = {1e30f, 1e30f}, fhi[2] = {-1e30f, -1e30f};
  for (const T9& t : floor_tris)
    for (int v = 0; v < 3; ++v)
      for (int k = 0; k < 2; ++k) { flo[k] = std::min(flo[k], t.p[3 * v + k]); fhi[k] = std::max(fhi[k], t.p[3 * v + k]); }
  Grid g1;
  g1.ox = std::floor((flo[0] - 1.f) / kRes) * kRes;
  g1.oy = std::floor((flo[1] - 1.f) / kRes) * kRes;
  g1.W = (int)std::ceil((fhi[0] + 1.f - g1.ox) / kRes);
  g1.H = (int)std::ceil((fhi[1] + 1.f - g1.oy) / kRes);
  if ((size_t)g1.W * g1.H > 40000000ULL) return reject("바닥 범위가 너무 큼");
  g1.v.assign((size_t)g1.W * g1.H, 0);
  Grid floor1 = g1, occ1 = g1;
  for (const T9& t : floor_tris) raster(t, floor1, false, 0, 0);
  close_cross(floor1);
  Grid allow1;
  float encl = 0.f;   // 바닥 칸 중 둘러싸인 몫
  {
    Grid wall1 = g1;
    for (const LT& t : L) raster(t.t, wall1, true, kWallLo, kKeepTop);   // 무대 + 물체 (창문은 물체라 무대만 쓰면 큰 창 방이 빠짐)
    enclosure_mask(wall1, allow1);
    size_t nf = 0, na = 0;
    for (size_t k = 0; k < floor1.v.size(); ++k) { nf += floor1.v[k]; na += floor1.v[k] && allow1.v[k]; floor1.v[k] &= allow1.v[k]; }
    encl = nf ? (float)na / nf : 0.f;
  }
  for (const LT& t : L) raster(t.t, occ1, true, C.hc.band_lo, C.hc.band_hi);
  Grid free1 = g1;
  for (size_t k = 0; k < g1.v.size(); ++k) free1.v[k] = floor1.v[k] && !occ1.v[k];
  const Grid R1 = largest_comp(free1);
  int bx0 = g1.W, by0 = g1.H, bx1 = -1, by1 = -1;
  size_t nR1 = 0;
  for (int y = 0; y < g1.H; ++y)
    for (int x = 0; x < g1.W; ++x)
      if (R1.get(x, y)) { bx0 = std::min(bx0, x); by0 = std::min(by0, y); bx1 = std::max(bx1, x); by1 = std::max(by1, y); ++nR1; }
  if (nR1 * kRes * kRes < kMinNavArea) return reject("갈 수 있는 바닥 < 3 m²");
  // 창 고르기 (1 차 격자 칸)
  int wx0, wy0, wn[2];
  bool cropped = false;
  {
    int span[2] = {bx1 - bx0 + 1 + 2 * kMargin, by1 - by0 + 1 + 2 * kMargin};
    wx0 = bx0 - kMargin;
    wy0 = by0 - kMargin;
    wn[0] = span[0];
    wn[1] = span[1];
    if (span[0] > kMaxCells || span[1] > kMaxCells) {
      cropped = true;
      // R 칸 누적 합으로 가장 많이 담는 창 (넘는 축만 미끄러뜨림)
      std::vector<int64_t> S((size_t)(g1.W + 1) * (g1.H + 1), 0);
      for (int y = 0; y < g1.H; ++y)
        for (int x = 0; x < g1.W; ++x)
          S[(size_t)(y + 1) * (g1.W + 1) + x + 1] = R1.get(x, y) + S[(size_t)y * (g1.W + 1) + x + 1] + S[(size_t)(y + 1) * (g1.W + 1) + x] - S[(size_t)y * (g1.W + 1) + x];
      auto sum = [&](int x0, int y0, int x1, int y1) {   // [x0, x1) × [y0, y1), 격자 밖은 0
        x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(g1.W, x1); y1 = std::min(g1.H, y1);
        if (x1 <= x0 || y1 <= y0) return (int64_t)0;
        return S[(size_t)y1 * (g1.W + 1) + x1] - S[(size_t)y0 * (g1.W + 1) + x1] - S[(size_t)y1 * (g1.W + 1) + x0] + S[(size_t)y0 * (g1.W + 1) + x0];
      };
      const int nx = std::min(span[0], kMaxCells), ny = std::min(span[1], kMaxCells);
      const int xs0 = span[0] > kMaxCells ? bx0 : wx0, xs1 = span[0] > kMaxCells ? bx1 + 1 - nx : wx0;
      const int ys0 = span[1] > kMaxCells ? by0 : wy0, ys1 = span[1] > kMaxCells ? by1 + 1 - ny : wy0;
      int64_t best = -1;
      for (int y = ys0; y <= ys1; ++y)
        for (int x = xs0; x <= xs1; ++x) {
          const int64_t s = sum(x + 1, y + 1, x + nx - 1, y + ny - 1);   // 가장자리 칸은 벽이 되므로 빼고 셈
          if (s > best) { best = s; wx0 = x; wy0 = y; }
        }
      wn[0] = nx;
      wn[1] = ny;
    }
  }
  // 2 차: 창으로 잘라 최종 형상, build_world 와 같은 격자
  const float O[3] = {g1.ox + wx0 * kRes, g1.oy + wy0 * kRes, -kFloorTol};
  const float Hh[3] = {O[0] + wn[0] * kRes, O[1] + wn[1] * kRes, kKeepTop};
  struct FT { T9 t; int obj; };
  std::vector<FT> F;
  {
    std::vector<T9> tmp;
    for (const LT& t : L) {
      tmp.clear();
      clip_box(t.t, O, Hh, tmp);
      for (const T9& r : tmp) F.push_back({r, t.obj});
    }
  }
  Grid g2;
  g2.ox = O[0];
  g2.oy = O[1];
  g2.W = (int)std::ceil((Hh[0] - O[0]) / kRes) + 1;   // build_world 와 같은 식
  g2.H = (int)std::ceil((Hh[1] - O[1]) / kRes) + 1;
  g2.v.assign((size_t)g2.W * g2.H, 0);
  Grid floor2 = g2, occ2 = g2;
  for (const T9& t : floor_tris) raster(t, floor2, false, 0, 0);
  close_cross(floor2);
  for (int y = 0; y < g2.H; ++y)   // 2 차 격자 칸 = 1 차 격자 칸 + 창 위치 (원점이 칸 단위로 맞음)
    for (int x = 0; x < g2.W; ++x) floor2.at(x, y) &= allow1.get(x + wx0, y + wy0);
  for (const FT& t : F) raster(t.t, occ2, true, C.hc.band_lo, C.hc.band_hi);
  // 창 안 칸 = 0 .. wn−1. 가장자리 고리(0, wn−1)와 그 밖은 R 에서 뺌
  Grid free2 = g2;
  for (int y = 0; y < g2.H; ++y)
    for (int x = 0; x < g2.W; ++x) {
      const bool ring = x <= 0 || y <= 0 || x >= wn[0] - 1 || y >= wn[1] - 1;
      free2.at(x, y) = !ring && floor2.get(x, y) && !occ2.get(x, y);
    }
  const Grid R = largest_comp(free2);
  // 가상 벽 칸: R 이 아니고 막힘도 아닌 칸 중 R 에 8 이웃으로 닿은 칸
  Grid vw = g2;
  int n_vw = 0, n_vw_cut = 0, n_occ_edge = 0;
  for (int y = 0; y < g2.H; ++y)
    for (int x = 0; x < g2.W; ++x) {
      if (R.get(x, y)) continue;
      bool adj = false;
      for (int dy = -1; dy <= 1 && !adj; ++dy)
        for (int dx = -1; dx <= 1 && !adj; ++dx) adj = R.get(x + dx, y + dy);
      if (!adj) continue;
      if (occ2.get(x, y)) { ++n_occ_edge; continue; }
      vw.at(x, y) = 1;
      ++n_vw;
      const bool ring = x <= 0 || y <= 0 || x >= wn[0] - 1 || y >= wn[1] - 1;
      if (ring) ++n_vw_cut;
    }
  const float open_frac = (float)(n_vw - n_vw_cut) / (float)std::max(1, n_vw - n_vw_cut + n_occ_edge);
  size_t nR = std::count(R.v.begin(), R.v.end(), 1);
  if (nR * kRes * kRes < kMinNavArea) return reject("창 안 갈 수 있는 바닥 < 3 m²");
  if (open_frac > kMaxOpenFrac) {
    char b[160];
    std::snprintf(b, sizeof(b), "열린 층 — R 둘레 가상 벽 몫 %.2f > %.2f (지붕·마당)", open_frac, kMaxOpenFrac);
    return reject(b);
  }
  // 출발점: 서는 칸(build_world 2 단계 규칙, 가상 벽 포함)의 가장 큰 덩어리 중 0.4 m 안 막힘 없는 칸
  Grid occ3 = occ2;
  for (size_t k = 0; k < occ3.v.size(); ++k) occ3.v[k] |= vw.v[k];
  auto disc = [](float r) {
    std::vector<std::pair<int, int>> d;
    const int Rr = (int)std::ceil(r / kRes);
    for (int dy = -Rr; dy <= Rr; ++dy)
      for (int dx = -Rr; dx <= Rr; ++dx)
        if ((float)(dx * dx + dy * dy) * kRes * kRes <= r * r) d.push_back({dx, dy});
    return d;
  };
  auto clear = [&](int x, int y, const std::vector<std::pair<int, int>>& d, bool out_free) {
    for (const auto& o : d) {
      const int u = x + o.first, v = y + o.second;
      if (u < 0 || v < 0 || u >= g2.W || v >= g2.H) { if (out_free) continue; return false; }
      if (occ3.v[(size_t)v * g2.W + u]) return false;
    }
    return true;
  };
  const auto d_in = disc(C.hc.cfg.half_wid), d_st = disc(kStartClear);
  Grid stand = g2;
  for (int y = 0; y < g2.H; ++y)
    for (int x = 0; x < g2.W; ++x) stand.at(x, y) = R.get(x, y) && clear(x, y, d_in, true);
  const Grid S = largest_comp(stand);
  std::vector<int> cand;
  for (int y = 0; y < g2.H; ++y)
    for (int x = 0; x < g2.W; ++x)
      if (S.get(x, y) && clear(x, y, d_st, false)) cand.push_back(y * g2.W + x);
  if (cand.empty()) return reject("출발점 후보 없음 (막힘에서 0.4 m 떨어진 서는 칸 없음)");
  const int sc = cand[mix64(C.seed ^ fnv1a(name)) % cand.size()];
  // 장면 만들기 (물체 번호는 남은 것만 차례로)
  sim::Scene s;
  s.start[0] = O[0] + ((float)(sc % g2.W) + 0.5f) * kRes;
  s.start[1] = O[1] + ((float)(sc / g2.W) + 0.5f) * kRes;
  std::map<int, uint16_t> oid;
  auto obj_of = [&](int o) {
    auto it = oid.find(o);
    if (it != oid.end()) return it->second;
    const uint16_t k = (uint16_t)s.obj_names.size();
    s.obj_names.push_back(o < 0 ? std::string("hssd_stage") : "furniture_" + std::to_string(o) + "_" + inst[o].tmpl);
    oid.emplace(o, k);
    return k;
  };
  for (const FT& t : F) {
    if (s.obj_names.size() >= 65000) return reject("물체가 너무 많음");
    const uint16_t k = obj_of(t.obj);
    s.tris.push_back(sim::Tri{{t.t.p[0], t.t.p[1], t.t.p[2]}, {t.t.p[3], t.t.p[4], t.t.p[5]}, {t.t.p[6], t.t.p[7], t.t.p[8]}});
    s.obj.push_back(k);
  }
  const int n_furn = (int)oid.size() - (oid.count(-1) ? 1 : 0);
  if (n_vw > 0) {
    const uint16_t k = (uint16_t)s.obj_names.size();
    s.obj_names.push_back("virtual_wall");
    // 줄마다 이어진 칸을 상자 하나로 (1 mm 안쪽 — 점 찍기가 옆 칸에 안 넘어가게)
    for (int y = 0; y < g2.H; ++y)
      for (int x = 0; x < g2.W;) {
        if (!vw.get(x, y)) { ++x; continue; }
        int x1 = x;
        while (x1 + 1 < g2.W && vw.get(x1 + 1, y)) ++x1;
        add_box(s, k, sim::V3{O[0] + x * kRes + 0.001f, O[1] + y * kRes + 0.001f, 0.f},
                sim::V3{O[0] + (x1 + 1) * kRes - 0.001f, O[1] + (y + 1) * kRes - 0.001f, kWallH});
        x = x1 + 1;
      }
  }
  {
    const uint16_t k = (uint16_t)s.obj_names.size();
    s.obj_names.push_back("floor");
    add_box(s, k, sim::V3{O[0], O[1], -0.01f}, sim::V3{Hh[0], Hh[1], 0.f});
  }
  sim::build_bvh(s);
  sim::compute_obj_size(s);
  const std::string path = (C.out / (name + ".scene")).string();
  sim::World w = sim::build_world(path, s, C.hc, false);
  if (!w.closed) return reject("build_world 닫힘 실패: " + w.leak);
  sim::save_scene(s, path);
  std::snprintf(buf, sizeof(buf), "%s\tok\t%.3f\t%.2f\t%d\t%d\t%d\t%zu\t%d\t%d\t%d\t%.2f\t%d\t%.2f\t%.2f\t%.2f\t%.2f", name.c_str(), fz,
                w.n_floor * kRes * kRes, w.n_wall, w.gw, w.gh, s.tris.size(), n_furn, n_vw, n_vw_cut, open_frac, (int)cropped,
                (float)(nR1 * kRes * kRes), (float)(nR * kRes * kRes), s.start[0], s.start[1]);
  char cc[32];
  std::snprintf(cc, sizeof(cc), "\t%.2f", encl);
  return std::string(buf) + "\t" + std::to_string(n_q_half) + "\t" + std::to_string(n_q_off) + cc;
}

int run_scene(const Ctx& C, const std::string& id) {
  const fs::path sj = C.root / "scenes" / (id + ".scene_instance.json");
  YAML::Node sc;
  try { sc = YAML::LoadFile(sj.string()); } catch (const std::exception& e) {
    std::printf("hssd_%s\treject\t0\t장면 JSON 못 읽음: %s\n", id.c_str(), e.what());
    return 1;
  }
  if (sc["translation_origin"] && sc["translation_origin"].as<std::string>() != "asset_local") {
    std::printf("hssd_%s\treject\t0\ttranslation_origin %s 은 안 다룸\n", id.c_str(), sc["translation_origin"].as<std::string>().c_str());
    return 1;
  }
  // 무대
  std::string stage_name = sc["stage_instance"]["template_name"].as<std::string>();   // "stages/<id>"
  const fs::path stage_cfg = C.root / (stage_name + ".stage_config.json");
  std::vector<T9> stage_raw, stage;
  std::string err;
  {
    const YAML::Node c = YAML::LoadFile(stage_cfg.string());
    if (c["up"] && c["up"][1].as<double>() != 1.0) { std::printf("hssd_%s\treject\t0\t무대 up 이 y 가 아님\n", id.c_str()); return 1; }
    if (!load_glb((stage_cfg.parent_path() / c["render_asset"].as<std::string>()).string(), stage_raw, err)) {
      std::printf("hssd_%s\treject\t0\t%s\n", id.c_str(), err.c_str());
      return 1;
    }
    Xf x0 = sc["stage_instance"]["translation"] ? make_xf(sc["stage_instance"]) : Xf{};
    stage.resize(stage_raw.size());
    for (size_t i = 0; i < stage_raw.size(); ++i) apply(x0, stage_raw[i], stage[i]);
  }
  // 물체
  std::vector<Inst> inst;
  std::map<std::string, std::vector<T9>> cache;
  int n_missing = 0, n_drop = 0, n_obj = 0;
  size_t raw_tris = stage_raw.size();
  for (const auto& o : sc["object_instances"]) {
    ++n_obj;
    const std::string t = o["template_name"].as<std::string>();
    const std::string base = t.substr(t.find_last_of('/') + 1);
    if (C.drop_ids.count(base)) { ++n_drop; continue; }
    auto it = C.obj_cfg.find(base);
    if (it == C.obj_cfg.end()) { ++n_missing; continue; }
    auto ct = cache.find(base);
    if (ct == cache.end()) {
      std::vector<T9> m;
      try {
        const YAML::Node c = YAML::LoadFile(it->second.string());
        if (c["up"] && c["up"][1].as<double>() != 1.0) throw std::runtime_error("up 이 y 가 아님");
        if (!load_glb((it->second.parent_path() / c["render_asset"].as<std::string>()).string(), m, err)) throw std::runtime_error(err);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: 물체 %s 못 읽음 (%s) — 뺌\n", id.c_str(), base.c_str(), e.what());
        m.clear();
        ++n_missing;
      }
      ct = cache.emplace(base, std::move(m)).first;
    }
    if (ct->second.empty()) continue;
    raw_tris += ct->second.size();
    const Xf x = make_xf(o);
    Inst I;
    I.tmpl = base;
    I.tris.resize(ct->second.size());
    for (size_t i = 0; i < ct->second.size(); ++i) apply(x, ct->second[i], I.tris[i]);
    inst.push_back(std::move(I));
  }
  // 층 찾기 (무대의 위를 보는 수평면)
  float z0 = 1e30f, z1 = -1e30f;
  for (const T9& t : stage) { z0 = std::min(z0, zmin(t)); z1 = std::max(z1, zmax(t)); }
  const int nb = std::max(1, (int)std::ceil((z1 - z0) / kRes) + 1);
  std::vector<double> hist(nb, 0.0), hz(nb, 0.0);
  for (const T9& t : stage) {
    float nz;
    const float a = tri_area_nz(t, &nz);
    if (nz <= 0.9f) continue;
    const float cz = (t.p[2] + t.p[5] + t.p[8]) / 3.f;
    const int b = std::min(nb - 1, std::max(0, (int)((cz - z0) / kRes)));
    hist[b] += a;
    hz[b] += a * cz;
  }
  std::vector<int> order(nb);
  for (int i = 0; i < nb; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return hist[a] > hist[b] || (hist[a] == hist[b] && a < b); });
  std::vector<float> levels;
  for (int b : order) {
    if (hist[b] < kMinLevelArea) break;
    const float z = (float)(hz[b] / hist[b]);
    bool far = true;
    for (float l : levels) far = far && std::fabs(z - l) > kLevelSep;
    if (far) levels.push_back(z);
  }
  std::sort(levels.begin(), levels.end());
  std::fprintf(stderr, "%s: 물체 %d (못 찾음 %d, 문 뺌 %d), 원래 삼각형 %zu, 층 %zu\n", id.c_str(), n_obj, n_missing, n_drop, raw_tris, levels.size());
  if (levels.empty()) { std::printf("hssd_%s\treject\t0\t층(넓은 수평 바닥) 없음\n", id.c_str()); return 1; }
  for (size_t li = 0; li < levels.size(); ++li) {
    std::string line;
    try { line = make_level(C, id, (int)li, levels[li], stage, inst); } catch (const std::exception& e) {
      line = "hssd_" + id + "_l" + std::to_string(li) + "\treject\t" + std::to_string(levels[li]) + "\t" + e.what();
    }
    std::printf("%s\t%d\t%d\n", line.c_str(), n_missing, n_drop);
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Ctx C;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml";
  std::vector<std::string> ids;
  std::vector<std::string> drop_cat = {"door"};
  bool header = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--hssd" && i + 1 < argc) C.root = argv[++i];
    else if (a == "--out" && i + 1 < argc) C.out = argv[++i];
    else if (a == "--seed" && i + 1 < argc) C.seed = std::stoull(argv[++i]);
    else if (a == "--cluster" && i + 1 < argc) C.cluster_q = std::stof(argv[++i]);
    else if (a == "--cluster-hi" && i + 1 < argc) C.cluster_q_hi = std::stof(argv[++i]);
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--keep-doors") drop_cat.clear();
    else if (a == "--header") header = true;
    else ids.push_back(a);
  }
  if (header)
    std::printf("name\tstatus\tfloor_z\tfloor_m2\twall_cells\tgw\tgh\ttris\tfurniture\tvirtual_cells\tvirtual_cut\topen_frac\tcropped\tnav_full_m2\tnav_m2\tstart_x\tstart_y\tq_half\tq_off\tenclosed\tmissing\tdoors\n");
  if (C.root.empty() || C.out.empty() || ids.empty()) {
    if (header) return 0;
    std::fprintf(stderr, "mesh2scene --hssd <hssd-hab> --out <폴더> [--seed 1] [--cluster q] [--cluster-hi q] [--keep-doors] [--header] <장면 id> …\n");
    return 2;
  }
  C.hc = sim::load_env_cfg(env_path);
  fs::create_directories(C.out);
  for (const auto& e : fs::recursive_directory_iterator(C.root / "objects")) {
    const std::string f = e.path().filename().string();
    const std::string suf = ".object_config.json";
    if (f.size() > suf.size() && f.compare(f.size() - suf.size(), suf.size(), suf) == 0) C.obj_cfg[f.substr(0, f.size() - suf.size())] = e.path();
  }
  if (!drop_cat.empty()) {
    std::ifstream f(C.root / "metadata" / "fpmodels-with-decomposed.csv");
    std::string line;
    int col = -1;
    while (std::getline(f, line)) {
      const auto c = csv_split(line);
      if (col < 0) { for (size_t k = 0; k < c.size(); ++k) if (c[k] == "main_category") col = (int)k; continue; }
      if ((int)c.size() > col && std::find(drop_cat.begin(), drop_cat.end(), c[col]) != drop_cat.end()) C.drop_ids.insert(c[0]);
    }
    if (col < 0) { std::fprintf(stderr, "metadata/fpmodels-with-decomposed.csv 에 main_category 열이 없음\n"); return 1; }
  }
  int bad = 0;
  for (const auto& id : ids) bad += run_scene(C, id);
  return bad == (int)ids.size() ? 1 : 0;
}
