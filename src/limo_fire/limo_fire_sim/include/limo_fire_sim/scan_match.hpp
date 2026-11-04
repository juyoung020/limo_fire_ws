// 라이다 스캔 맞추기 (점 → 선 ICP, Censi 2008 PL-ICP 꼴) — 바퀴와 따로 몸통이 실제로 얼마나 움직였는지(참값) 잴 때 쓴다.
// 메카넘 옆 이동은 롤러가 미끄러져 바퀴 오도메트리(/odom vy, 바퀴 속도로 계산)가 몸 이동을 모른다 → 연속 스캔을 맞춰 몸 이동을 구한다.
// 쓰는 법: Odom lo; 장마다 lo.add(stamp, 빔 점들(base_link, 장 시작 시각 기준으로 몸 움직임 보정), 바퀴로 짐작한 이동) → lo.pose
// 키 장(마지막으로 크게 움직인 장)에 맞춰 표류를 줄인다. 실내(벽 몇 m 안)·짧은 녹화용, 고리 닫기 없음.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace scan_match
{

struct Pt { double x, y; };
struct Pose { double x, y, th; };

inline Pose compose(const Pose & a, const Pose & b)
{
  const double c = std::cos(a.th), s = std::sin(a.th);
  return {a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, std::remainder(a.th + b.th, 2 * M_PI)};
}
inline Pose inverse(const Pose & a)
{
  const double c = std::cos(a.th), s = std::sin(a.th);
  return {-(c * a.x + s * a.y), s * a.x - c * a.y, -a.th};
}
inline Pt apply(const Pose & p, const Pt & q)
{
  const double c = std::cos(p.th), s = std::sin(p.th);
  return {p.x + c * q.x - s * q.y, p.y + s * q.x + c * q.y};
}

// 기준 장(target)의 점과 법선 (이웃 점 둘로 선 방향)
struct Ref {
  std::vector<Pt> p, n;
  std::vector<char> ok;
  void build(const std::vector<Pt> & pts)
  {
    p = pts;
    n.assign(p.size(), {0, 0});
    ok.assign(p.size(), 0);
    for (size_t i = 1; i + 1 < p.size(); ++i) {
      const Pt & a = p[i - 1], & b = p[i + 1];
      const double dx = b.x - a.x, dy = b.y - a.y, l = std::hypot(dx, dy);
      // 이웃이 멀면(모서리·끊김) 선이 아님
      if (l < 1e-6 || l > 0.25 || std::hypot(p[i].x - a.x, p[i].y - a.y) > 0.15 || std::hypot(b.x - p[i].x, b.y - p[i].y) > 0.15) {continue;}
      n[i] = {-dy / l, dx / l};
      ok[i] = 1;
    }
  }
};

// cur 를 ref 에 맞추는 자세 T (ref 좌표에서 cur 장 자세). guess 에서 시작. 반환: 맞은 점 비율, rms (m)
inline bool icp(const Ref & ref, const std::vector<Pt> & cur, Pose & T, double * inlier = nullptr, double * rms = nullptr)
{
  double gate = 0.3;
  int used = 0;
  double err = 0;
  for (int it = 0; it < 40; ++it) {
    // 정규 방정식 3×3 (tx, ty, th) — 작은 각 선형화
    double A[3][3] = {{0}}, b[3] = {0};
    used = 0;
    err = 0;
    const double c = std::cos(T.th), s = std::sin(T.th);
    for (const Pt & q : cur) {
      const Pt w{T.x + c * q.x - s * q.y, T.y + s * q.x + c * q.y};
      size_t best = 0;
      double bd = gate * gate;
      bool found = false;
      for (size_t j = 0; j < ref.p.size(); ++j) {
        if (!ref.ok[j]) {continue;}
        const double dx = ref.p[j].x - w.x, dy = ref.p[j].y - w.y, d = dx * dx + dy * dy;
        if (d < bd) {bd = d; best = j; found = true;}
      }
      if (!found) {continue;}
      const Pt & nn = ref.n[best];
      const double r = (w.x - ref.p[best].x) * nn.x + (w.y - ref.p[best].y) * nn.y;
      // d(w)/d th = (-s qx - c qy, c qx - s qy)
      const double jx = nn.x, jy = nn.y, jt = nn.x * (-s * q.x - c * q.y) + nn.y * (c * q.x - s * q.y);
      const double J[3] = {jx, jy, jt};
      for (int a = 0; a < 3; ++a) {
        b[a] -= J[a] * r;
        for (int k = 0; k < 3; ++k) {A[a][k] += J[a] * J[k];}
      }
      err += r * r;
      ++used;
    }
    if (used < 30) {return false;}
    // 3×3 풀기 (크래머)
    const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (std::fabs(det) < 1e-12) {return false;}
    double x[3];
    for (int k = 0; k < 3; ++k) {
      double M[3][3];
      for (int r = 0; r < 3; ++r) {for (int q = 0; q < 3; ++q) {M[r][q] = q == k ? b[r] : A[r][q];}}
      x[k] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
        M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / det;
    }
    T.x += x[0];
    T.y += x[1];
    T.th = std::remainder(T.th + x[2], 2 * M_PI);
    gate = std::max(0.05, gate * 0.7);
    if (std::fabs(x[0]) + std::fabs(x[1]) < 1e-5 && std::fabs(x[2]) < 1e-5 && gate <= 0.05) {break;}
  }
  if (inlier) {*inlier = double(used) / std::max<size_t>(1, cur.size());}
  if (rms) {*rms = std::sqrt(err / std::max(1, used));}
  return true;
}

// 키 장 방식 라이다 오도메트리
struct Odom {
  Ref key;
  Pose key_pose{0, 0, 0}, pose{0, 0, 0};
  bool have = false;
  int fails = 0;
  // pts: 이번 장 점 (base_link), guess: 지난 장에서 이번 장까지 몸 이동 짐작 (바퀴). 성공하면 true
  bool add(const std::vector<Pt> & pts, const Pose & guess)
  {
    if (!have) {
      key.build(pts);
      have = true;
      return true;
    }
    const Pose pred = compose(pose, guess);
    Pose T = compose(inverse(key_pose), pred);
    double inl = 0, rms = 0;
    if (!icp(key, pts, T, &inl, &rms) || inl < 0.5) {
      ++fails;
      pose = pred;   // 못 맞추면 바퀴로 잇고 키 장을 새로
      key.build(pts);
      key_pose = pose;
      return false;
    }
    pose = compose(key_pose, T);
    if (std::hypot(T.x, T.y) > 0.10 || std::fabs(T.th) > 0.15) {
      key.build(pts);
      key_pose = pose;
    }
    return true;
  }
};

}  // namespace scan_match
