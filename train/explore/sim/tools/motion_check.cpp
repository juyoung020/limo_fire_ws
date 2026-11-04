// 시뮬 = 현실 (구동): 실물 녹화의 명령(리모컨 스틱 또는 /cmd_vel)을 시뮬 구동 모델(drive.h)에 넣어 바퀴 속도를 실물 /odom 과,
// 몸통 회전을 IMU 자이로와 비교한다. 모델 값을 녹화에서 맞추고(--fit), 설정(env.yaml)이 그 맞춤만큼 잘 맞는지 본다.
//   motion_check <구동.csv>[@시작:끝,시작:끝…] … [--set analysis/calibration.yaml fit|validate] [--env config/env.yaml] [--floor office] [--fit]
//   --set: 보정 녹화 목록에서 그 역할의 녹화(뺄 구간 포함)를 더한다
//   @ 뒤 구간(s, 그 녹화 처음 기준)은 비교에서 뺀다 — 부딪히거나 앞이 막힌 곳처럼 구동 응답이 아닌 곳
// CSV 는 limo_fire_sim motion_export 로 만든다. 녹화마다 경로를 고른다: 리모컨 줄이 많으면 리모컨(섀시에 바로 → env.yaml
// drive.rc: 스틱 → 목표 속도 표와 모터 응답), 아니면 ROS 명령(/cmd_vel → limo_fire_base, 정책·Nav2 가 쓰는 길 → drive).
// 바닥(일정할 때 비·몸통 가속 한도 = 마찰, env.yaml floors)은 같은 바닥의 ROS 명령 녹화로 맞춘다 — 몸통 ω 를 자이로에,
// 몸통 직진 가속을 IMU 가속도에 (일정할 때 직진 비는 IMU 로 못 잰다: 가속도 0).
// 리모컨은 일정하게 돌 때도 바퀴 ω 가 자이로보다 크다(사무실 1.15, ROS 명령은 3 rad/s 에서도 0.99) → drive.rc.slip_w 로 따로 곱한다.
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "calibration.h"
#include "drive.h"

namespace {

struct Row {
  double t;
  int mode;
  float rc_v, rc_w, cmd_v, cmd_w, odom_v, odom_w, gyro;
  float acc;   // IMU 앞쪽 가속도 (5 줄 이동 평균, 진동 빼려고). 없으면 NAN
  bool skip;
  float cmd_vy = 0.f;                 // 메카넘 옆 명령 (없으면 0)
  float lx = NAN, ly = NAN, lth = NAN;   // 라이다 몸 자세 (motion_export lo_*, 녹화 처음 기준). 없으면 NAN
};

struct Rec {
  std::string name;
  std::vector<Row> rows;
  bool rc;
  bool body = false;   // 라이다 몸 자세 열이 있음
};

Rec load(const std::string& arg) {
  Rec r;
  const size_t at = arg.find('@');
  r.name = arg.substr(0, at);
  std::vector<std::pair<double, double>> skips;
  if (at != std::string::npos) {
    std::stringstream ss(arg.substr(at + 1));
    std::string part;
    while (std::getline(ss, part, ',')) {
      const size_t c = part.find(':');
      skips.emplace_back(std::stod(part.substr(0, c)), std::stod(part.substr(c + 1)));
    }
  }
  std::ifstream f(r.name);
  if (!f) throw std::runtime_error("못 읽음: " + r.name);
  std::string line;
  std::getline(f, line);
  // 뒤쪽 열(메카넘 옆 명령·라이다 몸 자세)은 이름으로 찾는다 — 예전 CSV 엔 없음
  int c_vy = -1, c_lx = -1, c_ly = -1, c_lt = -1;
  {
    std::stringstream hs(line);
    std::string h;
    for (int k = 0; std::getline(hs, h, ','); ++k) {
      if (h == "cmd_vy") c_vy = k;
      if (h == "lo_x") c_lx = k;
      if (h == "lo_y") c_ly = k;
      if (h == "lo_yaw") c_lt = k;
    }
  }
  r.body = c_lx >= 0 && c_ly >= 0 && c_lt >= 0;
  size_t n_rc = 0;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    Row w{};
    char c;
    ss >> w.t >> c >> w.mode >> c >> w.rc_v >> c >> w.rc_w >> c >> w.cmd_v >> c >> w.cmd_w >> c >> w.odom_v >> c >> w.odom_w >> c >> w.gyro;
    double stamp = 0;
    w.acc = NAN;
    if (ss >> c >> stamp >> c >> w.acc) {}
    w.lx = w.ly = w.lth = NAN;
    if (c_vy >= 0 || r.body) {
      std::vector<std::string> cell;
      std::stringstream ls(line);
      std::string x;
      while (std::getline(ls, x, ',')) cell.push_back(x);
      auto num = [&](int k) { return k >= 0 && k < (int)cell.size() && cell[k] != "nan" ? std::stof(cell[k]) : NAN; };
      if (c_vy >= 0) w.cmd_vy = num(c_vy);
      if (r.body) { w.lx = num(c_lx); w.ly = num(c_ly); w.lth = num(c_lt); }
    }
    for (const auto& [a, b] : skips) w.skip = w.skip || (w.t >= a && w.t <= b);
    n_rc += w.mode == 3;
    r.rows.push_back(w);
  }
  // 가속도 이동 평균
  std::vector<float> raw(r.rows.size());
  for (size_t i = 0; i < raw.size(); ++i) raw[i] = r.rows[i].acc;
  for (size_t i = 0; i < raw.size(); ++i) {
    float sum = 0;
    int k = 0;
    for (size_t j = i >= 2 ? i - 2 : 0; j <= std::min(raw.size() - 1, i + 2); ++j) if (!std::isnan(raw[j])) { sum += raw[j]; ++k; }
    r.rows[i].acc = k ? sum / k : NAN;
  }
  r.rc = n_rc * 2 > r.rows.size();
  return r;
}

// 한 녹화를 다시 돌린 오차 제곱 합 (바퀴 v · 바퀴 ω · 몸통 ω vs 자이로) 과 실물 제곱 합 (R² 용)
struct Sum {
  double ev = 0, ew = 0, eg = 0, sv = 0, mv = 0, sw = 0, mw = 0;
  size_t n = 0;
  void add(const Sum& o) {
    ev += o.ev; ew += o.ew; eg += o.eg; sv += o.sv; mv += o.mv; sw += o.sw; mw += o.mw; sg += o.sg; mg += o.mg; n += o.n;
    ea += o.ea; sa += o.sa; ma += o.ma; na += o.na;
  }
  double rmse_v() const { return std::sqrt(ev / n); }
  double rmse_w() const { return std::sqrt(ew / n); }
  double rmse_g() const { return std::sqrt(eg / n); }
  double r2_v() const { return 1 - ev / n / (sv / n - (mv / n) * (mv / n)); }
  double r2_w() const { return 1 - ew / n / (sw / n - (mw / n) * (mw / n)); }
  double sg = 0, mg = 0;
  double r2_g() const { return 1 - eg / n / (sg / n - (mg / n) * (mg / n)); }
  double ea = 0, sa = 0, ma = 0;   // 몸통 직진 가속 vs IMU
  size_t na = 0;
  double rmse_a() const { return na ? std::sqrt(ea / na) : NAN; }
  double r2_a() const { return na ? 1 - ea / na / (sa / na - (ma / na) * (ma / na)) : NAN; }
};

Sum replay(const Rec& rec, const sim::DriveModel& m, const sim::Floor& fl) {
  Sum s;
  sim::DriveState st;
  sim::drive_reset(st);
  for (size_t i = 0; i < rec.rows.size(); ++i) {
    const Row& r = rec.rows[i];
    const float dt = i ? (float)(r.t - rec.rows[i - 1].t) : 0.02f;
    const float cv = rec.rc ? sim::rc_target(m.rc_v, r.rc_v) : r.cmd_v;
    const float cw = rec.rc ? sim::rc_target(m.rc_w, r.rc_w) : r.cmd_w;
    float bv, bw;
    const float prev_bv = st.bv;
    sim::drive_step(m, st, cv, cw, dt, fl, &bv, &bw);
    if (r.skip) continue;
    if (!std::isnan(r.acc)) {
      const float a = (bv - prev_bv) / dt;
      s.ea += (a - r.acc) * (a - r.acc);
      s.sa += r.acc * r.acc;
      s.ma += r.acc;
      ++s.na;
    }
    s.ev += (st.wv - r.odom_v) * (st.wv - r.odom_v);
    s.ew += (st.ww - r.odom_w) * (st.ww - r.odom_w);
    s.eg += (bw - r.gyro) * (bw - r.gyro);
    s.sv += r.odom_v * r.odom_v;
    s.mv += r.odom_v;
    s.sw += r.odom_w * r.odom_w;
    s.mw += r.odom_w;
    s.sg += r.gyro * r.gyro;
    s.mg += r.gyro;
    ++s.n;
  }
  return s;
}

// 모터 쪽(바퀴 속도)은 바닥과 무관 — 한 통로(v 또는 ω)만 다시 돌린 오차
double channel_mse(const std::vector<const Rec*>& recs, bool lin, const float* table, int delay, float tau, float acc, float slow,
                   float tau_slow, float tau_slow_down, float vmax) {
  double e = 0;
  size_t n = 0;
  float buf[sim::kDelayMax];
  for (const Rec* rec : recs) {
    for (float& x : buf) x = 0.f;
    int head = 0;
    float yf = 0.f, ys = 0.f;
    for (size_t i = 0; i < rec->rows.size(); ++i) {
      const Row& r = rec->rows[i];
      const float dt = i ? (float)(r.t - rec->rows[i - 1].t) : 0.02f;
      float c = rec->rc ? sim::rc_target(table, lin ? r.rc_v : r.rc_w) : (lin ? r.cmd_v : r.cmd_w);
      c = c > vmax ? vmax : (c < -vmax ? -vmax : c);
      buf[head] = c;
      const float target = buf[(head - delay + sim::kDelayMax) % sim::kDelayMax];
      head = (head + 1) % sim::kDelayMax;
      yf = sim::follow(yf, target, tau, acc, dt);
      ys = sim::follow(ys, target, sim::slow_tau(ys, target, tau_slow, tau_slow_down), acc, dt);
      const float y = (1.f - slow) * yf + slow * ys;
      if (r.skip) continue;
      const float m = lin ? r.odom_v : r.odom_w;
      e += (y - m) * (y - m);
      ++n;
    }
  }
  return e / n;
}

// 모터 응답(지연·빠른/느린 갈래·가속)과 (리모컨이면) 스틱 표를 번갈아 맞춘다
void fit_channel(const std::vector<const Rec*>& recs, bool lin, bool rc, float vmax, float* table, int& delay, float& tau, float& acc,
                 float& slow, float& tau_slow, float tau_slow_down) {
  if (rc) for (int k = 0; k < sim::kRcPoints; ++k) table[k] = vmax * k / (sim::kRcPoints - 1);
  const float taus[] = {0.f, 0.01f, 0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.13f, 0.16f, 0.2f};
  const float accs_v[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 8.f, 12.f, 1000.f};
  const float accs_w[] = {3.f, 6.f, 10.f, 15.f, 20.f, 30.f, 50.f, 80.f, 1000.f};
  const float slows[] = {0.f, 0.05f, 0.1f, 0.15f, 0.2f, 0.3f};
  const float tslows[] = {0.06f, 0.1f, 0.15f, 0.2f, 0.3f};
  const float* accs = lin ? accs_v : accs_w;
  for (int round = 0; round < 4; ++round) {
    double best = 1e18;
    for (int d = 0; d < 8; ++d)
      for (float t : taus)
        for (int ai = 0; ai < 9; ++ai)
          for (float sl : slows)
            for (float ts : tslows) {
              if (sl == 0.f && ts != tslows[0]) continue;
              const double e = channel_mse(recs, lin, table, d, t, accs[ai], sl, ts, tau_slow_down, vmax);
              if (e < best) { best = e; delay = d; tau = t; acc = accs[ai]; slow = sl; tau_slow = ts; }
            }
    if (!rc) break;   // ROS 명령은 표가 없다
    float step = 0.25f * vmax;
    for (int it = 0; it < 8; ++it, step *= 0.5f)
      for (int k = 1; k < sim::kRcPoints; ++k)
        for (float sgn : {1.f, -1.f}) {
          const float old = table[k];
          table[k] = std::min(vmax, std::max(table[k - 1], old + sgn * step));
          const double e = channel_mse(recs, lin, table, delay, tau, acc, slow, tau_slow, tau_slow_down, vmax);
          if (e < best) best = e;
          else table[k] = old;
        }
  }
}

// 바닥만: 실물 바퀴 ω(/odom)를 넣어 몸통 ω 를 자이로와 비교 (모터 쪽 오차와 떼어 바닥을 맞춘다)
double floor_mse(const std::vector<const Rec*>& recs, const sim::Floor& f, size_t* n_out) {
  double e = 0;
  size_t n = 0;
  for (const Rec* rec : recs) {
    float bw = 0.f;
    for (size_t i = 0; i < rec->rows.size(); ++i) {
      const Row& r = rec->rows[i];
      const float dt = i ? (float)(r.t - rec->rows[i - 1].t) : 0.02f;
      bw = sim::follow(bw, r.odom_w / f.slip_w, f.tau_w, f.trac_w, dt);
      if (r.skip) continue;
      e += (bw - r.gyro) * (bw - r.gyro);
      ++n;
    }
  }
  if (n_out) *n_out = n;
  return e / n;
}

// 바닥 직진: 실물 바퀴 v 를 넣어 몸통 가속을 IMU 가속도와 비교 (5 줄 이동 평균끼리)
double floor_mse_v(const std::vector<const Rec*>& recs, const sim::Floor& f, size_t* n_out) {
  double e = 0;
  size_t n = 0;
  for (const Rec* rec : recs) {
    std::vector<float> a(rec->rows.size(), 0.f);
    float bv = 0.f;
    for (size_t i = 0; i < rec->rows.size(); ++i) {
      const Row& r = rec->rows[i];
      const float dt = i ? (float)(r.t - rec->rows[i - 1].t) : 0.02f;
      const float nb = sim::follow(bv, r.odom_v / f.slip_v, f.tau_v, f.trac_v, dt);
      a[i] = (nb - bv) / dt;
      bv = nb;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      const Row& r = rec->rows[i];
      if (r.skip || std::isnan(r.acc)) continue;
      float sum = 0;
      int k = 0;
      for (size_t j = i >= 2 ? i - 2 : 0; j <= std::min(a.size() - 1, i + 2); ++j) { sum += a[j]; ++k; }
      const float d = sum / k - r.acc;
      e += d * d;
      ++n;
    }
  }
  if (n_out) *n_out = n;
  return n ? e / n : 0;
}

// 리모컨 녹화면 바닥 비에 drive.rc.slip_w 를 곱한다
sim::Floor path_floor(const sim::Floor& f, bool rc, float rc_slip) {
  sim::Floor g = f;
  if (rc) g.slip_w = f.slip_w * rc_slip;
  return g;
}

sim::DriveModel load_model(const YAML::Node& top, bool rc) {
  const YAML::Node d = rc ? top["rc"] : top;
  if (!d) throw std::runtime_error("env.yaml drive.rc 없음");
  sim::DriveModel m{};
  m.delay_v = d["delay_steps"][0].as<int>();
  m.delay_w = d["delay_steps"][1].as<int>();
  m.tau_v = d["tau"][0].as<float>();
  m.tau_w = d["tau"][1].as<float>();
  m.acc_v = d["accel"][0].as<float>();
  m.acc_w = d["accel"][1].as<float>();
  m.slow_v = d["slow"][0].as<float>();
  m.slow_w = d["slow"][1].as<float>();
  m.tau_slow_v = d["tau_slow"][0].as<float>();
  m.tau_slow_w = d["tau_slow"][1].as<float>();
  // 줄어들 때 느린 갈래 (drive.tau_slow_down). 리모컨 경로(drive.rc)는 따로 안 맞춰 없으면 tau_slow 그대로
  const YAML::Node down = d["tau_slow_down"];
  if (!down && !rc) throw std::runtime_error("env.yaml drive.tau_slow_down 없음");
  m.tau_slow_down_v = down ? down[0].as<float>() : m.tau_slow_v;
  m.tau_slow_down_w = down ? down[1].as<float>() : m.tau_slow_w;
  m.v_max = top["max"][0].as<float>();
  m.w_max = top["max"][1].as<float>();
  if (rc)
    for (int i = 0; i < sim::kRcPoints; ++i) {
      m.rc_v[i] = d["rc_v"][i].as<float>();
      m.rc_w[i] = d["rc_w"][i].as<float>();
    }
  return m;
}

void print_model(const char* where, const sim::DriveModel& m, bool rc) {
  std::printf("  env.yaml %s:\n    delay_steps: [%d, %d]   # %.0f · %.0f ms\n    tau: [%.2f, %.2f]\n    accel: [%.1f, %.1f]\n"
              "    slow: [%.2f, %.2f]\n    tau_slow: [%.2f, %.2f]\n", where, m.delay_v, m.delay_w, m.delay_v * 20.0, m.delay_w * 20.0,
              m.tau_v, m.tau_w, m.acc_v, m.acc_w, m.slow_v, m.slow_w, m.tau_slow_v, m.tau_slow_w);
  if (rc) {
    std::printf("    rc_v: [");
    for (int i = 0; i < sim::kRcPoints; ++i) std::printf(i ? ", %.3f" : "%.3f", m.rc_v[i]);
    std::printf("]\n    rc_w: [");
    for (int i = 0; i < sim::kRcPoints; ++i) std::printf(i ? ", %.3f" : "%.3f", m.rc_w[i]);
    std::printf("]\n");
  }
}


// ---------- 몸 참값 (--body): 라이다 스캔 맞추기 몸 자세(motion_export lo_*)와 자이로로 명령 → 몸 움직임을 본다 ----------
// vy ≠ 0 명령 구간은 시뮬이 정상 메카넘을 가정해 맞추지 않으므로(analysis/real_limo.md 구동 절) 그 구간과 뒤따르는 정지 구간을 뺀다.

constexpr double kLidarH = 0.1;   // 라이다 몸 속도 가운데 차분 반폭 (s) — 라이다 10 Hz

// vy 명령 구간 + 그 뒤 정지 구간(바퀴가 서는 동안)을 skip 으로. 뺀 줄 수
size_t mask_lateral(Rec& r) {
  size_t n = 0, s = 0;
  const size_t N = r.rows.size();
  auto same = [&](size_t a, size_t b) {
    return r.rows[a].cmd_v == r.rows[b].cmd_v && r.rows[a].cmd_w == r.rows[b].cmd_w && r.rows[a].cmd_vy == r.rows[b].cmd_vy;
  };
  std::vector<std::pair<size_t, size_t>> seg;
  for (size_t i = 1; i <= N; ++i)
    if (i == N || !same(i, s)) { seg.emplace_back(s, i); s = i; }
  for (size_t k = 0; k < seg.size(); ++k) {
    if (r.rows[seg[k].first].cmd_vy == 0.f) continue;
    const size_t e = k + 1 < seg.size() ? seg[k + 1].second : seg[k].second;
    for (size_t i = seg[k].first; i < e; ++i) { n += !r.rows[i].skip; r.rows[i].skip = true; }
  }
  return n;
}

// 줄마다 라이다 몸 속도 (몸 좌표 vx, ω), 앞뒤 kLidarH 가운데 차분. 없으면 NAN
void lidar_vel(const Rec& r, std::vector<float>& bv, std::vector<float>& bw) {
  const size_t N = r.rows.size();
  bv.assign(N, NAN);
  bw.assign(N, NAN);
  size_t lo = 0, hi = 0;
  for (size_t i = 0; i < N; ++i) {
    const double t = r.rows[i].t;
    while (lo < i && r.rows[lo].t < t - kLidarH) ++lo;
    while (hi + 1 < N && r.rows[hi].t < t + kLidarH) ++hi;
    const Row &a = r.rows[lo], &b = r.rows[hi];
    const double dt = b.t - a.t;
    if (dt < kLidarH || std::isnan(a.lx) || std::isnan(b.lx) || std::isnan(r.rows[i].lth)) continue;
    const double dx = (b.lx - a.lx) / dt, dy = (b.ly - a.ly) / dt, th = r.rows[i].lth;
    bv[i] = (float)(std::cos(th) * dx + std::sin(th) * dy);
    bw[i] = (float)((b.lth - a.lth) / dt);
  }
}

// 라이다 시각 어긋남 (줄 단위): 라이다 ω 를 k 줄 늦춘 것이 자이로와 가장 맞는 k (−10~10). 양수 = 라이다가 앞섬
int lidar_lag(const Rec& r, const std::vector<float>& bw) {
  int best = 0;
  double be = 1e18;
  for (int k = -10; k <= 10; ++k) {
    double e = 0;
    size_t n = 0;
    for (size_t i = 10; i + 10 < r.rows.size(); ++i) {
      const float x = bw[i - k];
      if (r.rows[i].skip || std::isnan(x)) continue;
      e += (x - r.rows[i].gyro) * (x - r.rows[i].gyro);
      ++n;
    }
    if (n && e / n < be) { be = e / n; best = k; }
  }
  return best;
}

struct BodyOut {
  std::vector<float> wv, ww, bv, bw;   // 시뮬 바퀴·몸통
};
BodyOut body_replay(const Rec& rec, const sim::DriveModel& m, const sim::Floor& fl) {
  BodyOut o;
  sim::DriveState st;
  sim::drive_reset(st);
  for (size_t i = 0; i < rec.rows.size(); ++i) {
    const Row& r = rec.rows[i];
    const float dt = i ? (float)(r.t - rec.rows[i - 1].t) : 0.02f;
    float bv, bw;
    sim::drive_step(m, st, r.cmd_v, r.cmd_w, dt, fl, &bv, &bw);
    o.wv.push_back(st.wv); o.ww.push_back(st.ww); o.bv.push_back(bv); o.bw.push_back(bw);
  }
  return o;
}

struct BodyLim {
  double r2, sv, sw, xy, yaw, t90;
};
struct R2 {
  double e = 0, s = 0, m = 0;
  size_t n = 0;
  void add(double sim, double real) { e += (sim - real) * (sim - real); s += real * real; m += real; ++n; }
  double r2() const { return n ? 1 - e / n / (s / n - (m / n) * (m / n)) : NAN; }
};

// 녹화 하나: 표를 찍고 통과 여부
bool body_check(const Rec& rec, const sim::DriveModel& m, const sim::Floor& fl, const BodyLim& L, bool table) {
  const BodyOut o = body_replay(rec, m, fl);
  std::vector<float> lv, lw;
  lidar_vel(rec, lv, lw);
  const int lag = lidar_lag(rec, lw);
  const size_t N = rec.rows.size();
  auto lid_v = [&](size_t i) { const long j = (long)i - lag; return j >= 0 && j < (long)N ? lv[j] : NAN; };
  R2 rv, rw, rg;
  for (size_t i = 0; i < N; ++i) {
    const Row& r = rec.rows[i];
    if (r.skip) continue;
    rv.add(o.wv[i], r.odom_v); rw.add(o.ww[i], r.odom_w); rg.add(o.bw[i], r.gyro);
  }
  bool pass = rv.r2() >= L.r2 && rw.r2() >= L.r2 && rg.r2() >= L.r2;
  std::printf("\n### %s (라이다 시각 %+d 줄 = %+.0f ms 당겨 맞춤, 뺀 줄 포함 %zu 줄)\n\n", rec.name.c_str(), lag, lag * 20.0, N);
  std::printf("| R² (≥ %.2f) | 바퀴 v %.3f | 바퀴 ω %.3f | 몸통 ω (자이로) %.3f | %s |\n|---|---|---|---|---|\n", L.r2, rv.r2(), rw.r2(), rg.r2(),
              rv.r2() >= L.r2 && rw.r2() >= L.r2 && rg.r2() >= L.r2 ? "통과" : "**미달**");
  // 명령이 같은 구간
  std::vector<std::pair<size_t, size_t>> seg;
  {
    size_t s = 0;
    for (size_t i = 1; i <= N; ++i)
      if (i == N || rec.rows[i].cmd_v != rec.rows[s].cmd_v || rec.rows[i].cmd_w != rec.rows[s].cmd_w || rec.rows[i].cmd_vy != rec.rows[s].cmd_vy) {
        seg.emplace_back(s, i);
        s = i;
      }
  }
  if (table)
    std::printf("\n| 시각 (s) | 명령 v ω | 몸 vx 라이다 → 시뮬 (차 ≤ %.2f) | 몸 ω 자이로 → 시뮬 (차 ≤ %.2f) | 정지→정지 라이다 (x y θ) · 자이로 θ | 시뮬 | 이동 차 (≤ %.2f m) | 회전 차 vs 자이로 (≤ %.3f) | |\n"
                "|---|---|---|---|---|---|---|---|---|\n", L.sv, L.sw, L.xy, L.yaw);
  double max_sv = 0, max_sw = 0, max_xy = 0, max_yaw = 0;
  int n_seg = 0, n_bad = 0;
  for (size_t k = 0; k < seg.size(); ++k) {
    const auto [s, e] = seg[k];
    const Row& c = rec.rows[s];
    if (c.cmd_v == 0.f && c.cmd_w == 0.f) continue;
    bool sk = false;
    for (size_t i = s; i < e; ++i) sk = sk || rec.rows[i].skip;
    const double t0 = rec.rows[s].t, t1 = rec.rows[e - 1].t;
    if (sk || t1 - t0 < 0.4) continue;
    // 일정 구간: 뒤 절반 (라이다 차분이 다음 구간으로 번지지 않게 끝 kLidarH 뺌)
    double ls = 0, ss = 0, gs = 0, ws = 0;
    int nl = 0, ng = 0;
    for (size_t i = s; i < e; ++i) {
      const double t = rec.rows[i].t;
      if (t < 0.5 * (t0 + t1) || t > t1 - kLidarH) continue;
      gs += rec.rows[i].gyro; ws += o.bw[i]; ++ng;
      const float x = lid_v(i);
      if (!std::isnan(x)) { ls += x; ss += o.bv[i]; ++nl; }
    }
    const double dv = nl ? ss / nl - ls / nl : NAN, dw = ng ? ws / ng - gs / ng : NAN;
    // 정지 → 정지: 구간 시작 직전 줄 ~ 다음 정지 구간 끝
    // 회전 참값은 자이로 적분 (라이다 스캔 맞추기 자세의 회전은 v·ω 조합에서 자이로와 최대 17 % 어긋남 — 자이로 배율은 제자리 회전에서
    // 라이다와 0.99~1.00 으로 확인). 이동 참값은 라이다
    double dxy = NAN, dyaw = NAN, L3[3] = {NAN, NAN, NAN}, S3[3] = {0, 0, 0}, G = 0;
    if (k + 1 < seg.size() && s > 0) {
      const size_t a = s - 1, b = seg[k + 1].second - 1;
      const Row& nx = rec.rows[seg[k + 1].first];
      if (nx.cmd_v == 0.f && nx.cmd_w == 0.f && nx.cmd_vy == 0.f && !std::isnan(rec.rows[a].lx) && !std::isnan(rec.rows[b].lx)) {
        const Row &A = rec.rows[a], &B = rec.rows[b];
        const double ca = std::cos(A.lth), sa = std::sin(A.lth);
        L3[0] = ca * (B.lx - A.lx) + sa * (B.ly - A.ly);
        L3[1] = -sa * (B.lx - A.lx) + ca * (B.ly - A.ly);
        L3[2] = B.lth - A.lth;
        for (size_t i = a + 1; i <= b; ++i) {
          const double dt = rec.rows[i].t - rec.rows[i - 1].t, thm = S3[2] + 0.5 * o.bw[i] * dt;
          S3[0] += o.bv[i] * std::cos(thm) * dt;
          S3[1] += o.bv[i] * std::sin(thm) * dt;
          S3[2] += o.bw[i] * dt;
          G += rec.rows[i].gyro * dt;
        }
        dxy = std::hypot(S3[0] - L3[0], S3[1] - L3[1]);
        dyaw = S3[2] - G;
      }
    }
    const bool ok = !(std::fabs(dv) > L.sv) && !(std::fabs(dw) > L.sw) && !(dxy > L.xy) && !(std::fabs(dyaw) > L.yaw);
    ++n_seg;
    n_bad += !ok;
    if (std::isfinite(dv)) max_sv = std::max(max_sv, std::fabs(dv));
    if (std::isfinite(dw)) max_sw = std::max(max_sw, std::fabs(dw));
    if (std::isfinite(dxy)) max_xy = std::max(max_xy, dxy);
    if (std::isfinite(dyaw)) max_yaw = std::max(max_yaw, std::fabs(dyaw));
    if (table)
      std::printf("| %.1f | %+.2f %+.2f | %+.3f → %+.3f (%+.3f) | %+.3f → %+.3f (%+.3f) | %+.3f %+.3f %+.3f · %+.3f | %+.3f %+.3f %+.3f | %.3f | %+.3f | %s |\n", t0,
                  c.cmd_v, c.cmd_w, nl ? ls / nl : NAN, nl ? ss / nl : NAN, dv, ng ? gs / ng : NAN, ng ? ws / ng : NAN, dw, L3[0], L3[1], L3[2], G, S3[0],
                  S3[1], S3[2], dxy, dyaw, ok ? "통과" : "**넘음**");
    pass = pass && ok;
  }
  // 멈춤: 통로 명령이 0 이 아닌 값 → 0 일 때 바퀴가 직전 값의 10 % 아래로 가는 시간 (명령 바뀐 때부터)
  double t90[2] = {0, 0};
  int n90[2] = {0, 0};
  for (int ch = 0; ch < 2; ++ch)
    for (size_t i = 1; i < N; ++i) {
      const Row &p = rec.rows[i - 1], &r = rec.rows[i];
      const float cp = ch ? p.cmd_w : p.cmd_v, cr = ch ? r.cmd_w : r.cmd_v;
      if (cp == 0.f || cr != 0.f || r.skip || p.skip) continue;
      auto when = [&](auto get) {
        const double x0 = std::fabs(get(i));
        for (size_t j = i; j < std::min(N, i + 100); ++j)
          if (std::fabs(get(j)) < 0.1 * x0) return rec.rows[j].t - r.t;
        return (double)NAN;
      };
      const double tr = when([&](size_t j) { return ch ? rec.rows[j].odom_w : rec.rows[j].odom_v; });
      const double ts = when([&](size_t j) { return ch ? o.ww[j] : o.wv[j]; });
      if (std::isnan(tr) || std::isnan(ts)) continue;
      t90[ch] += std::fabs(ts - tr);
      ++n90[ch];
    }
  for (int ch = 0; ch < 2; ++ch) t90[ch] = n90[ch] ? t90[ch] / n90[ch] : NAN;
  const bool ok90 = !(t90[0] > L.t90) && !(t90[1] > L.t90);
  pass = pass && ok90;
  std::printf("\n| 구간 %d 개 중 넘음 %d | 몸 vx 차 최대 %.3f m/s | 몸 ω 차 최대 %.3f rad/s | 정지→정지 이동 차 최대 %.3f m | 회전 차 최대 %.3f rad | 멈춤 10 %% 시간 차 평균 v %.3f · ω %.3f s (≤ %.2f, %d · %d 번) %s |\n",
              n_seg, n_bad, max_sv, max_sw, max_xy, max_yaw, t90[0], t90[1], L.t90, n90[0], n90[1], ok90 ? "통과" : "**미달**");
  return pass;
}

// 맞추기: 느린 갈래가 줄어들 때 시간상수(모터 — 바퀴 오도메트리) → 바닥(회전: 실물 바퀴 ω → 자이로, 직진 비: 실물 바퀴 v → 라이다 몸 vx)
void body_fit(std::vector<Rec>& recs, sim::DriveModel& m, sim::Floor& fl, bool motor) {
  const float downs[] = {0.f, 0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.12f, 0.15f, 0.2f};
  std::vector<const Rec*> rp;
  for (const auto& r : recs) rp.push_back(&r);
  for (int round = 0; round < (motor ? 2 : 1); ++round)
  for (int ch = 0; ch < 2; ++ch) {
    if (motor) {   // 오르는 쪽 모터 응답도 이 녹화로 (--motor)
      if (ch == 0) fit_channel(rp, true, false, m.v_max, m.rc_v, m.delay_v, m.tau_v, m.acc_v, m.slow_v, m.tau_slow_v, m.tau_slow_down_v);
      else fit_channel(rp, false, false, m.w_max, m.rc_w, m.delay_w, m.tau_w, m.acc_w, m.slow_w, m.tau_slow_w, m.tau_slow_down_w);
    }
    double be = 1e18;
    float bd = 0;
    for (float d : downs) {
      (ch ? m.tau_slow_down_w : m.tau_slow_down_v) = d;
      double e = 0;
      for (const auto& r : recs) {
        const BodyOut o = body_replay(r, m, fl);
        for (size_t i = 0; i < r.rows.size(); ++i)
          if (!r.rows[i].skip) e += ch ? (o.ww[i] - r.rows[i].odom_w) * (o.ww[i] - r.rows[i].odom_w) : (o.wv[i] - r.rows[i].odom_v) * (o.wv[i] - r.rows[i].odom_v);
      }
      if (e < be) { be = e; bd = d; }
    }
    (ch ? m.tau_slow_down_w : m.tau_slow_down_v) = bd;
  }
  if (motor) print_model("drive", m, false);
  std::printf("  env.yaml drive.tau_slow_down: [%.2f, %.2f]\n", m.tau_slow_down_v, m.tau_slow_down_w);
  double best = 1e18;
  sim::Floor bf = fl;
  for (int si = 0; si <= 30; ++si)
    for (float tw : {10.f, 15.f, 20.f, 25.f, 30.f, 35.f, 40.f, 50.f, 1000.f}) {
      const sim::Floor f{fl.slip_v, 0.95f + 0.005f * si, fl.trac_v, tw, fl.tau_v, fl.tau_w};
      const double e = floor_mse(rp, f, nullptr);
      if (e < best) { best = e; bf = f; }
    }
  // 직진 비: 앞뒤 단독 명령(ω·vy 0)의 정지 → 정지 이동을 실물 바퀴 적분과 라이다로 (가만히 선 두 자세라 라이다가 가장 믿을 만함 —
  // 움직이는 중 라이다 속도는 스캔 휨 보정이 남아 ±0.02~0.04 m/s 흔들림)
  double a = 0, b = 0;
  for (const auto& r : recs) {
    const size_t N = r.rows.size();
    size_t s = 0;
    for (size_t i = 1; i <= N; ++i) {
      if (i < N && r.rows[i].cmd_v == r.rows[s].cmd_v && r.rows[i].cmd_w == r.rows[s].cmd_w && r.rows[i].cmd_vy == r.rows[s].cmd_vy) continue;
      const size_t e = i;
      const Row& c = r.rows[s];
      size_t nx_end = e;   // 다음 (정지) 구간 끝
      while (nx_end < N && r.rows[nx_end].cmd_v == 0.f && r.rows[nx_end].cmd_w == 0.f && r.rows[nx_end].cmd_vy == 0.f) ++nx_end;
      bool sk = false;
      for (size_t k = s; k < nx_end; ++k) sk = sk || r.rows[k].skip;
      if (c.cmd_v != 0.f && c.cmd_w == 0.f && c.cmd_vy == 0.f && !sk && s > 0 && nx_end > e && !std::isnan(r.rows[s - 1].lx) && !std::isnan(r.rows[nx_end - 1].lx)) {
        const Row &A = r.rows[s - 1], &B = r.rows[nx_end - 1];
        const double lx = std::cos(A.lth) * (B.lx - A.lx) + std::sin(A.lth) * (B.ly - A.ly);
        double wx = 0;
        for (size_t k = s; k < nx_end; ++k) wx += r.rows[k].odom_v * (r.rows[k].t - r.rows[k - 1].t);
        a += wx * lx;
        b += lx * lx;
      }
      s = i;
    }
  }
  if (b > 0) bf.slip_v = (float)(a / b);
  fl = bf;
  std::printf("  env.yaml floors: 일정할 때 바퀴 / 몸통 [v %.3f, ω %.3f], 몸통 회전 가속 한도 %.0f rad/s² (실물 바퀴 ω → 자이로 RMSE %.3f)\n", fl.slip_v,
              fl.slip_w, fl.trac_w, std::sqrt(best));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", floor_name = "office";
  bool fit = false, body = false, table = false, motor = false;
  for (int i = 1; i < argc; ++i) body = body || std::string(argv[i]) == "--body";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--floor" && i + 1 < argc) floor_name = argv[++i];
    else if (a == "--fit") fit = true;
    else if (a == "--body") {}
    else if (a == "--table") table = true;
    else if (a == "--motor") motor = true;
    else if (a == "--set" && i + 2 < argc) {
      std::string fl_set;
      for (const auto& c : sim::load_calibration(argv[i + 1], argv[i + 2], body ? "body" : "recordings")) {
        args.push_back(sim::with_skips(c.csv, c.skip));
        if (body && !fl_set.empty() && fl_set != c.floor) throw std::runtime_error("--body: 한 번에 한 바닥만 (calibration.yaml body: floor)");
        fl_set = c.floor;
      }
      if (body) floor_name = fl_set;   // 몸 참값 녹화는 그 녹화의 바닥으로
      i += 2;
    } else args.push_back(a);
  }
  if (args.empty()) {
    std::fprintf(stderr, "motion_check <구동.csv>[@시작:끝,…] … [--set analysis/calibration.yaml fit|validate] [--env env.yaml] [--floor office] [--fit]\n"
                         "             [--body [--table]]   몸 참값(라이다 lo_* 열): calibration.yaml body: 목록, 기준 env.yaml drive.check body_*\n");
    return 2;
  }
  std::vector<Rec> recs;
  for (const auto& a : args) {
    recs.push_back(load(a));
    size_t sk = 0;
    for (const auto& r : recs.back().rows) sk += r.skip;
    std::printf("%s: %zu 줄 (%.0f s), 경로 %s, 뺀 줄 %zu\n", recs.back().name.c_str(), recs.back().rows.size(),
                recs.back().rows.size() * 0.02, recs.back().rc ? "리모컨" : "ROS 명령", sk);
  }
  const YAML::Node env = YAML::LoadFile(env_path);
  const YAML::Node top = env["drive"];
  sim::DriveModel cmd_m = load_model(top, false), rc_m = load_model(top, true);
  float rc_slip = top["rc"]["slip_w"].as<float>();
  sim::Floor fl{};
  bool found = false;
  for (const auto& f : env["floors"])
    if (f["name"].as<std::string>() == floor_name) {
      auto mid = [&](const char* key, int k) { return 0.5f * (f[key][k][0].as<float>() + f[key][k][1].as<float>()); };
      fl = sim::Floor{mid("slip", 0), mid("slip", 1), mid("traction", 0), mid("traction", 1), mid("lag", 0), mid("lag", 1)};
      found = true;
    }
  if (!found) throw std::runtime_error("env.yaml floors 에 " + floor_name + " 없음");

  if (body) {
    const YAML::Node ck = top["check"];
    const BodyLim L{ck["body_r2_min"].as<double>(), ck["body_steady_v"].as<double>(), ck["body_steady_w"].as<double>(), ck["rest_xy"].as<double>(),
                    ck["rest_yaw"].as<double>(), ck["stop_t90"].as<double>()};
    for (auto& r : recs) {
      if (!r.body) throw std::runtime_error(r.name + ": 라이다 몸 자세 열(lo_*) 없음 — motion_export 를 새로 돌릴 것");
      if (r.rc) throw std::runtime_error(r.name + ": --body 는 /cmd_vel 명령 녹화만");
      std::printf("%s: 옆(vy) 명령 구간과 뒤 정지 구간 %zu 줄 뺌 (시뮬은 정상 메카넘 가정)\n", r.name.c_str(), mask_lateral(r));
    }
    if (fit) {
      std::printf("\n맞추기 (바닥 %s):\n", floor_name.c_str());
      body_fit(recs, cmd_m, fl, motor);
    }
    std::printf("\n몸 참값 비교 (%s, 바닥 %s: 비 v %.3f · ω %.3f)\n", fit ? "방금 맞춘 값" : "설정 값", floor_name.c_str(), fl.slip_v, fl.slip_w);
    bool pass = true;
    for (const auto& r : recs) pass = body_check(r, cmd_m, fl, L, table) && pass;
    std::printf("\n%s\n", pass ? "몸 참값 구동 시뮬 = 실물: 통과" : "몸 참값 구동 시뮬 = 실물: 넘음");
    return pass ? 0 : 1;
  }

  if (fit) {
    std::printf("\n맞추기:\n");
    for (int path = 0; path < 2; ++path) {
      std::vector<const Rec*> group;
      for (const auto& r : recs) if (r.rc == (path == 1)) group.push_back(&r);
      if (group.empty()) continue;
      sim::DriveModel& m = path == 1 ? rc_m : cmd_m;
      if (path == 1) {   // 최고 속도는 끝까지 밀어 본 리모컨 녹화에서
        std::vector<float> av, aw;
        for (const Rec* r : group) for (const auto& w : r->rows) if (!w.skip) { av.push_back(std::fabs(w.odom_v)); aw.push_back(std::fabs(w.odom_w)); }
        std::sort(av.begin(), av.end());
        std::sort(aw.begin(), aw.end());
        cmd_m.v_max = rc_m.v_max = av[av.size() * 995 / 1000];
        cmd_m.w_max = rc_m.w_max = aw[aw.size() * 995 / 1000];
      }
      fit_channel(group, true, path == 1, m.v_max, m.rc_v, m.delay_v, m.tau_v, m.acc_v, m.slow_v, m.tau_slow_v, m.tau_slow_down_v);
      fit_channel(group, false, path == 1, m.w_max, m.rc_w, m.delay_w, m.tau_w, m.acc_w, m.slow_w, m.tau_slow_w, m.tau_slow_down_w);
      print_model(path == 1 ? "drive.rc" : "drive", m, path == 1);
    }
    std::printf("    (drive.max: [%.2f, %.2f])\n", cmd_m.v_max, cmd_m.w_max);
    // 바닥: 실물 바퀴 ω → 몸통 ω 를 자이로에 (ROS 명령 녹화로, 없으면 모든 녹화). 직진 쪽은 몸통 v 참값이 없어 못 맞춤
    bool any_cmd = false;
    for (const auto& r : recs) any_cmd = any_cmd || !r.rc;
    std::vector<const Rec*> floor_recs;
    for (const auto& r : recs) if (!r.rc || !any_cmd) floor_recs.push_back(&r);
    double best = 1e18;
    size_t n_all = 0;
    sim::Floor bf = fl;
    for (int si = 0; si <= 36; ++si)
      for (float tw : {3.f, 5.f, 8.f, 10.f, 15.f, 20.f, 30.f, 50.f, 1000.f})
        for (float lw : {0.f, 0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.13f, 0.16f, 0.2f}) {
          const sim::Floor f{fl.slip_v, 0.94f + 0.01f * si, fl.trac_v, tw, fl.tau_v, lw};
          size_t n;
          const double e = floor_mse(floor_recs, f, &n);
          if (e < best) { best = e; bf = f; n_all = n; }
        }
    std::printf("  env.yaml floors %s: 일정할 때 바퀴 ω / 몸통 ω %.2f, 몸통 회전 가속 한도 %.0f rad/s², 몸통 회전 지연 %.2f s"
                " (실물 바퀴로 자이로 RMSE %.3f)\n", floor_name.c_str(), bf.slip_w, bf.trac_w, bf.tau_w, std::sqrt(best));
    {
      double bv_e = 1e18;
      for (float tv : {1.5f, 2.f, 2.5f, 3.f, 3.5f, 4.f, 4.5f, 5.f, 5.5f, 6.f, 7.f, 8.f, 10.f, 100.f})
        for (float lv : {0.f, 0.02f, 0.04f, 0.06f, 0.08f, 0.1f}) {
          sim::Floor f = bf;
          f.trac_v = tv;
          f.tau_v = lv;
          size_t na;
          const double e = floor_mse_v(floor_recs, f, &na);
          if (na && e < bv_e) { bv_e = e; bf.trac_v = tv; bf.tau_v = lv; }
        }
      std::printf("  env.yaml floors %s: 몸통 직진 가속 한도 %.1f m/s², 몸통 직진 지연 %.2f s (실물 바퀴로 IMU 가속도 RMSE %.2f m/s²)\n",
                  floor_name.c_str(), bf.trac_v, bf.tau_v, std::sqrt(bv_e));
    }
    (void)n_all;
    // 리모컨 쪽 곱: 그 바닥으로 리모컨 녹화의 실물 바퀴 → 자이로에
    float best_rc = rc_slip;
    std::vector<const Rec*> rc_recs;
    for (const auto& r : recs) if (r.rc) rc_recs.push_back(&r);
    if (any_cmd && !rc_recs.empty()) {
      double be = 1e18;
      for (int k = 0; k <= 40; ++k) {
        const float rs = 0.9f + 0.01f * k;
        const double e = floor_mse(rc_recs, path_floor(bf, true, rs), nullptr);
        if (e < be) { be = e; best_rc = rs; }
      }
      std::printf("  env.yaml drive.rc.slip_w: %.2f (리모컨일 때 바닥 비에 곱함, 실물 바퀴로 자이로 RMSE %.3f)\n", best_rc, std::sqrt(be));
    }
    std::printf("\n| 녹화 (맞춘 값) | 경로 | 바퀴 v RMSE | 바퀴 ω RMSE | 몸통 ω vs 자이로 |\n|---|---|---|---|---|\n");
    for (const auto& r : recs) {
      const Sum s = replay(r, r.rc ? rc_m : cmd_m, path_floor(bf, r.rc, best_rc));
      std::printf("| %s | %s | %.3f (R² %.3f) | %.3f (R² %.3f) | %.3f |\n", r.name.c_str(), r.rc ? "리모컨" : "ROS 명령", s.rmse_v(),
                  s.r2_v(), s.rmse_w(), s.r2_w(), s.rmse_g());
    }
    cmd_m = load_model(top, false);
    rc_m = load_model(top, true);
  }
  std::printf("\n| 녹화 (설정 %s, 바닥 %s) | 경로 | 바퀴 v RMSE | 바퀴 ω RMSE | 몸통 ω vs 자이로 RMSE | 몸통 가속 vs IMU RMSE |\n|---|---|---|---|---|---|\n",
              env_path.c_str(), floor_name.c_str());
  bool pass = true;
  const double r2_min = top["check"]["r2_min"].as<double>();
  for (const auto& r : recs) {
    const Sum s = replay(r, r.rc ? rc_m : cmd_m, path_floor(fl, r.rc, rc_slip));
    std::printf("| %s | %s | %.3f (R² %.3f) | %.3f (R² %.3f) | %.3f (R² %.3f) | %.2f m/s² (R² %.2f) |\n", r.name.c_str(),
                r.rc ? "리모컨" : "ROS 명령", s.rmse_v(), s.r2_v(), s.rmse_w(), s.r2_w(), s.rmse_g(), s.r2_g(), s.rmse_a(), s.r2_a());
    pass = pass && s.r2_v() >= r2_min && s.r2_w() >= r2_min && s.r2_g() >= r2_min;
  }
  std::printf("\n%s (녹화마다 바퀴 v·ω · 몸통 ω(자이로) R² ≥ %.2f)\n", pass ? "구동 시뮬 = 실물: 통과" : "구동 시뮬 = 실물: 넘음", r2_min);
  return pass ? 0 : 1;
}
