// Webots 장면 내보내기(.w3d, X3D 꼴 XML) → 삼각형 장면. Webots 가 그리는 형상을 그대로 옮긴다:
//   변환: Solid · Pose · Transform · Robot · Group 의 translation · rotation(축-각) · scale 를 부모부터 곱한다
//   형상: Box · Cylinder(Webots 와 같은 n 각형, 기본 36) · Capsule(기본 12) · Sphere(이코 구) · Plane · Cone · IndexedFaceSet
//   USE='id' 는 그 id 의 마디를 그 자리에 다시 쓴다
// 빼는 것: 이름이 exclude 에 있는 Robot(우리 LIMO — 센서가 자기 몸을 안 본다), Background·Viewpoint 같은 그리지 않는 마디
// 물체 번호: 가장 가까운 이름 있는 조상(Solid·Robot)의 이름
#include "w3d.h"

#include <tinyxml2.h>

#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sim {
namespace {

struct M4 {   // 행 우선 3×4 (회전·크기 + 위치), double 로 쌓고 마지막에 float
  double m[12];
  static M4 id() { return M4{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}}; }
  M4 operator*(const M4& b) const {
    M4 r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        double v = 0;
        for (int k = 0; k < 3; ++k) v += m[i * 4 + k] * b.m[k * 4 + j];
        if (j == 3) v += m[i * 4 + 3];
        r.m[i * 4 + j] = v;
      }
    }
    return r;
  }
  V3 apply(double x, double y, double z) const {
    return V3{(float)(m[0] * x + m[1] * y + m[2] * z + m[3]), (float)(m[4] * x + m[5] * y + m[6] * z + m[7]),
              (float)(m[8] * x + m[9] * y + m[10] * z + m[11])};
  }
};

std::vector<double> nums(const char* s) {
  std::vector<double> v;
  if (!s) return v;
  std::stringstream ss(s);
  for (std::string t; ss >> t;) {
    if (t == "TRUE" || t == "true") v.push_back(1);
    else if (t == "FALSE" || t == "false") v.push_back(0);
    else v.push_back(std::stod(t));
  }
  return v;
}

double num(const tinyxml2::XMLElement* e, const char* a, double fallback) {
  const auto v = nums(e->Attribute(a));
  return v.empty() ? fallback : v[0];
}

bool flag(const tinyxml2::XMLElement* e, const char* a, bool fallback) {
  const char* s = e->Attribute(a);
  if (!s) return fallback;
  return std::string(s) == "true" || std::string(s) == "TRUE";
}

M4 local_transform(const tinyxml2::XMLElement* e) {
  M4 t = M4::id();
  const auto tr = nums(e->Attribute("translation"));
  const auto ro = nums(e->Attribute("rotation"));
  const auto sc = nums(e->Attribute("scale"));
  if (ro.size() == 4) {   // 축-각 → 회전 행렬 (로드리게스)
    double x = ro[0], y = ro[1], z = ro[2];
    const double n = std::sqrt(x * x + y * y + z * z);
    if (n > 0) { x /= n; y /= n; z /= n; }
    const double c = std::cos(ro[3]), s = std::sin(ro[3]), C = 1 - c;
    const double r[9] = {c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C,
                         y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) t.m[i * 4 + j] = r[i * 3 + j];
  }
  if (sc.size() == 3) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) t.m[i * 4 + j] *= sc[j];
  }
  if (tr.size() == 3) { t.m[3] = tr[0]; t.m[7] = tr[1]; t.m[11] = tr[2]; }
  return t;
}

class Builder {
 public:
  Builder(Scene& s, const std::vector<std::string>& exclude) : s_(s), exclude_(exclude) {}

  void index(const tinyxml2::XMLElement* e) {
    for (; e; e = e->NextSiblingElement()) {
      if (const char* id = e->Attribute("id")) ids_[id] = e;
      index(e->FirstChildElement());
    }
  }

  // USE='371' 은 id='n371' 을 가리킨다 (Webots w3d 는 USE 에 앞의 n 을 빼고 쓴다)
  const tinyxml2::XMLElement* resolve(const tinyxml2::XMLElement* e) const {
    const char* use = e->Attribute("USE");
    if (!use) return e;
    auto it = ids_.find(use);
    if (it == ids_.end()) it = ids_.find(std::string("n") + use);
    if (it == ids_.end()) throw std::runtime_error(std::string("USE 대상 없음: ") + use);
    return it->second;
  }

  void walk(const tinyxml2::XMLElement* e, const M4& T, int obj) {
    e = resolve(e);
    const std::string tag = e->Name();
    if (tag == "Background" || tag == "Viewpoint" || tag == "WorldInfo" || tag == "PointLight" || tag == "SpotLight" ||
        tag == "DirectionalLight" || tag == "Fog" || tag == "head" || tag == "CadShape")
      return;
    const char* name = e->Attribute("name");
    if (tag == "Robot" && name) {
      for (const auto& x : exclude_) if (x == name) { ++excluded_; return; }
    }
    if (name && (tag == "Solid" || tag == "Robot")) obj = object(name);
    if (tag == "Shape") { shape(e, T, obj); return; }
    const bool tf = tag == "Solid" || tag == "Pose" || tag == "Transform" || tag == "Robot";
    const M4 T2 = tf ? T * local_transform(e) : T;
    for (const auto* c = e->FirstChildElement(); c; c = c->NextSiblingElement()) walk(c, T2, obj);
  }

  int excluded() const { return excluded_; }

 private:
  int object(const std::string& name) {
    const auto it = obj_index_.find(name);
    if (it != obj_index_.end()) return it->second;
    const int i = (int)s_.obj_names.size();
    if (i >= 0xffff) throw std::runtime_error("물체가 너무 많음");
    s_.obj_names.push_back(name);
    obj_index_[name] = i;
    return i;
  }

  void tri(const M4& T, int obj, double ax, double ay, double az, double bx, double by, double bz, double cx, double cy,
           double cz) {
    s_.tris.push_back(Tri{T.apply(ax, ay, az), T.apply(bx, by, bz), T.apply(cx, cy, cz)});
    s_.obj.push_back((uint16_t)obj);
  }
  void quad(const M4& T, int obj, const double* p0, const double* p1, const double* p2, const double* p3) {
    tri(T, obj, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    tri(T, obj, p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]);
  }

  void shape(const tinyxml2::XMLElement* e, const M4& T, int obj) {
    for (const auto* g = e->FirstChildElement(); g; g = g->NextSiblingElement()) {
      const tinyxml2::XMLElement* geo = resolve(g);
      const std::string t = geo->Name();
      if (t == "Box") box(geo, T, obj);
      else if (t == "Cylinder") cylinder(geo, T, obj);
      else if (t == "Capsule") capsule(geo, T, obj);
      else if (t == "Sphere") sphere(geo, T, obj);
      else if (t == "Plane") plane(geo, T, obj);
      else if (t == "Cone") cone(geo, T, obj);
      else if (t == "IndexedFaceSet") ifs(geo, T, obj);
      else if (t == "PBRAppearance" || t == "Appearance") continue;
      else skipped_[t]++;
    }
  }

  void box(const tinyxml2::XMLElement* g, const M4& T, int obj) {
    const auto sz = nums(g->Attribute("size"));
    const double x = (sz.size() == 3 ? sz[0] : 2) / 2, y = (sz.size() == 3 ? sz[1] : 2) / 2, z = (sz.size() == 3 ? sz[2] : 2) / 2;
    const double p[8][3] = {{-x, -y, -z}, {x, -y, -z}, {x, y, -z}, {-x, y, -z}, {-x, -y, z}, {x, -y, z}, {x, y, z}, {-x, y, z}};
    const int f[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    for (const auto& q : f) quad(T, obj, p[q[0]], p[q[1]], p[q[2]], p[q[3]]);
  }

  // Webots wren createUnitCylinder 와 같은 점: 각 = 2π·i/n, (sin, cos)
  void cylinder(const tinyxml2::XMLElement* g, const M4& T, int obj) {
    const double r = num(g, "radius", 1), h = num(g, "height", 2) / 2;
    const int n = (int)num(g, "subdivision", 36);
    const bool side = flag(g, "side", true), top = flag(g, "top", true), bottom = flag(g, "bottom", true);
    for (int i = 0; i < n; ++i) {
      const double a0 = 2 * M_PI * i / n, a1 = 2 * M_PI * (i + 1) / n;
      const double x0 = r * std::sin(a0), y0 = r * std::cos(a0), x1 = r * std::sin(a1), y1 = r * std::cos(a1);
      if (side) {
        const double p0[3] = {x0, y0, -h}, p1[3] = {x1, y1, -h}, p2[3] = {x1, y1, h}, p3[3] = {x0, y0, h};
        quad(T, obj, p0, p1, p2, p3);
      }
      if (top) tri(T, obj, 0, 0, h, x0, y0, h, x1, y1, h);
      if (bottom) tri(T, obj, 0, 0, -h, x1, y1, -h, x0, y0, -h);
    }
  }

  void cone(const tinyxml2::XMLElement* g, const M4& T, int obj) {
    const double r = num(g, "bottomRadius", 1), h = num(g, "height", 2) / 2;
    const int n = (int)num(g, "subdivision", 36);
    for (int i = 0; i < n; ++i) {
      const double a0 = 2 * M_PI * i / n, a1 = 2 * M_PI * (i + 1) / n;
      const double x0 = r * std::sin(a0), y0 = r * std::cos(a0), x1 = r * std::sin(a1), y1 = r * std::cos(a1);
      if (flag(g, "side", true)) tri(T, obj, x0, y0, -h, x1, y1, -h, 0, 0, h);
      if (flag(g, "bottom", true)) tri(T, obj, 0, 0, -h, x1, y1, -h, x0, y0, -h);
    }
  }

  // 원기둥 몸통 + 양 끝 반구 (위·아래 각 n/4 단). Webots 다각형과의 차이는 r·(1 − cos(π/n)) 이하
  void capsule(const tinyxml2::XMLElement* g, const M4& T, int obj) {
    const double r = num(g, "radius", 1), h = num(g, "height", 2) / 2;
    const int n = std::max(8, (int)num(g, "subdivision", 12)), rings = std::max(2, n / 4);
    auto ring = [&](int k, double* xy) {   // k = 0 (적도) … rings (극)
      const double phi = M_PI / 2 * k / rings;
      xy[0] = r * std::cos(phi);
      xy[1] = r * std::sin(phi);
    };
    for (int i = 0; i < n; ++i) {
      const double a0 = 2 * M_PI * i / n, a1 = 2 * M_PI * (i + 1) / n;
      const double s0 = std::sin(a0), c0 = std::cos(a0), s1 = std::sin(a1), c1 = std::cos(a1);
      if (flag(g, "side", true)) {
        const double p0[3] = {r * s0, r * c0, -h}, p1[3] = {r * s1, r * c1, -h}, p2[3] = {r * s1, r * c1, h}, p3[3] = {r * s0, r * c0, h};
        quad(T, obj, p0, p1, p2, p3);
      }
      for (int k = 0; k < rings; ++k) {
        double q0[2], q1[2];
        ring(k, q0);
        ring(k + 1, q1);
        for (int sgn : {1, -1}) {
          if ((sgn > 0 && !flag(g, "top", true)) || (sgn < 0 && !flag(g, "bottom", true))) continue;
          const double z0 = sgn * (h + q0[1]), z1 = sgn * (h + q1[1]);
          const double p0[3] = {q0[0] * s0, q0[0] * c0, z0}, p1[3] = {q0[0] * s1, q0[0] * c1, z0};
          const double p2[3] = {q1[0] * s1, q1[0] * c1, z1}, p3[3] = {q1[0] * s0, q1[0] * c0, z1};
          quad(T, obj, p0, p1, p2, p3);
        }
      }
    }
  }

  void sphere(const tinyxml2::XMLElement* g, const M4& T, int obj) {   // 위도·경도 24 × 12 (이코 구와의 차이 ≤ 0.9 % r)
    const double r = num(g, "radius", 1);
    const int n = 24, m = 12;
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < m; ++k) {
        auto p = [&](int ii, int kk, double* o) {
          const double a = 2 * M_PI * ii / n, b = M_PI * kk / m - M_PI / 2;
          o[0] = r * std::cos(b) * std::cos(a); o[1] = r * std::cos(b) * std::sin(a); o[2] = r * std::sin(b);
        };
        double p0[3], p1[3], p2[3], p3[3];
        p(i, k, p0); p(i + 1, k, p1); p(i + 1, k + 1, p2); p(i, k + 1, p3);
        quad(T, obj, p0, p1, p2, p3);
      }
    }
  }

  void plane(const tinyxml2::XMLElement* g, const M4& T, int obj) {
    const auto sz = nums(g->Attribute("size"));
    const double x = (sz.size() == 2 ? sz[0] : 1) / 2, y = (sz.size() == 2 ? sz[1] : 1) / 2;
    const double p0[3] = {-x, -y, 0}, p1[3] = {x, -y, 0}, p2[3] = {x, y, 0}, p3[3] = {-x, y, 0};
    quad(T, obj, p0, p1, p2, p3);
  }

  void ifs(const tinyxml2::XMLElement* g, const M4& T, int obj) {
    const tinyxml2::XMLElement* c = g->FirstChildElement("Coordinate");
    if (!c) return;
    c = resolve(c);
    const auto pt = nums(c->Attribute("point"));
    const auto ix = nums(g->Attribute("coordIndex"));
    std::vector<int> face;
    for (double d : ix) {
      const int i = (int)d;
      if (i >= 0) { face.push_back(i); continue; }
      for (size_t k = 1; k + 1 < face.size(); ++k) {   // 부채꼴 (Webots 면은 볼록)
        const double* a = &pt[3 * face[0]];
        const double* b = &pt[3 * face[k]];
        const double* cc = &pt[3 * face[k + 1]];
        tri(T, obj, a[0], a[1], a[2], b[0], b[1], b[2], cc[0], cc[1], cc[2]);
      }
      face.clear();
    }
  }

  Scene& s_;
  std::vector<std::string> exclude_;
  std::map<std::string, const tinyxml2::XMLElement*> ids_;
  std::map<std::string, int> obj_index_;
  int excluded_ = 0;

 public:
  std::map<std::string, int> skipped_;
};

}  // namespace

W3dReport load_w3d(const std::string& path, const std::vector<std::string>& exclude_robots, Scene& out) {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) throw std::runtime_error("w3d 못 읽음: " + path);
  const tinyxml2::XMLElement* scene = doc.FirstChildElement("X3D") ? doc.FirstChildElement("X3D")->FirstChildElement("Scene")
                                                                  : nullptr;
  if (!scene) throw std::runtime_error("X3D Scene 없음: " + path);
  out = Scene{};
  out.obj_names.push_back("(이름 없음)");
  Builder b(out, exclude_robots);
  b.index(scene->FirstChildElement());
  for (const auto* e = scene->FirstChildElement(); e; e = e->NextSiblingElement()) b.walk(e, M4::id(), 0);
  W3dReport r;
  r.excluded_robots = b.excluded();
  r.skipped = b.skipped_;
  return r;
}

}  // namespace sim
