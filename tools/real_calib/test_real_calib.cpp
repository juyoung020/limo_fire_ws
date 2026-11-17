// real_calib 계산 확인: 만든 자료로 알려진 답이 나오는지 (외부 의존 없음, 실패하면 1)
#include <cstdio>
#include <sstream>

#include "real_calib.hpp"

using namespace real_calib;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("실패 %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static bool near(double a, double b, double e) { return std::fabs(a - b) <= e; }

int main() {
  {  // 0~20° 는 늘 메아리, 30° 는 절반, 40° 부터 없음
    std::ostringstream o;
    o << "angle_deg,true_m,range_m\n";
    for (int a : {0, 10, 20}) for (int k = 0; k < 20; ++k) o << a << ",1.0," << 1.0 + 0.01 * ((k % 3) - 1) << "\n";
    for (int k = 0; k < 20; ++k) o << "30,1.0," << (k % 2 ? "1.0" : "inf") << "\n";
    for (int a : {40, 50}) for (int k = 0; k < 20; ++k) o << a << ",1.0,\n";
    std::istringstream in(o.str());
    size_t bad = 0;
    const auto rows = read_echo(in, &bad);
    CHECK(rows.size() == 120 && bad == 0);
    const auto bins = by_angle(rows);
    CHECK(bins.size() == 6);
    CHECK(near(bins[3].rate(), 0.5, 1e-9));
    const auto sh = echo_shape(bins);
    CHECK(near(sh.full_deg, 20.0, 1e-9));
    CHECK(near(sh.none_deg, 40.0, 1e-9));
    const auto db = by_distance(rows, 10.0);
    CHECK(db.size() == 1 && db[0].n == 40);
    CHECK(near(db[0].sd, 0.0082, 0.002));
  }
  {  // 0 도 메아리 없음, 머리글 무시
    std::istringstream in("a,b,c\n0,1.0,0\n0,1.0,1.0\n");
    const auto rows = read_echo(in);
    CHECK(rows.size() == 2 && !rows[0].range && rows[1].range);
  }
  {  // slip: 3 m 가면 odom 이 2.91 m
    std::istringstream in("true_m,odom_m\n3.0,2.912\n3.0,2.912\n");
    const auto s = read_slip(in);
    CHECK(s && near(s->ratio.mean, 1.0302, 0.001));
  }
  {  // drift: 0.5 °/min = 1/120 °/s, 360° 넘김 포함
    std::ostringstream o;
    o.precision(12);
    o << "t,yaw\n";
    for (int t = 0; t <= 600; t += 10) {
      double y = 179.0 + t / 120.0;
      if (y > 180.0) y -= 360.0;
      o << t << "," << y << "\n";
    }
    std::istringstream in(o.str());
    const auto d = read_drift(in);
    CHECK(d && near(d->deg_per_min, 0.5, 1e-6) && d->resid_sd_deg < 1e-6);
  }
  std::printf(fails ? "실패 %d\n" : "통과\n", fails);
  return fails ? 1 : 0;
}
