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
};

struct Rec {
  std::string name;
  std::vector<Row> rows;
  bool rc;
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
  size_t n_rc = 0;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    Row w{};
    char c;
    ss >> w.t >> c >> w.mode >> c >> w.rc_v >> c >> w.rc_w >> c >> w.cmd_v >> c >> w.cmd_w >> c >> w.odom_v >> c >> w.odom_w >> c >> w.gyro;
    double stamp = 0;
    w.acc = NAN;
    if (ss >> c >> stamp >> c >> w.acc) {}
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
                   float tau_slow, float vmax) {
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
      ys = sim::follow(ys, target, tau_slow, acc, dt);
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
                 float& slow, float& tau_slow) {
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
              const double e = channel_mse(recs, lin, table, d, t, accs[ai], sl, ts, vmax);
              if (e < best) { best = e; delay = d; tau = t; acc = accs[ai]; slow = sl; tau_slow = ts; }
            }
    if (!rc) break;   // ROS 명령은 표가 없다
    float step = 0.25f * vmax;
    for (int it = 0; it < 8; ++it, step *= 0.5f)
      for (int k = 1; k < sim::kRcPoints; ++k)
        for (float sgn : {1.f, -1.f}) {
          const float old = table[k];
          table[k] = std::min(vmax, std::max(table[k - 1], old + sgn * step));
          const double e = channel_mse(recs, lin, table, delay, tau, acc, slow, tau_slow, vmax);
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

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", floor_name = "office";
  bool fit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--floor" && i + 1 < argc) floor_name = argv[++i];
    else if (a == "--fit") fit = true;
    else if (a == "--set" && i + 2 < argc) {
      for (const auto& c : sim::load_calibration(argv[i + 1], argv[i + 2])) args.push_back(sim::with_skips(c.csv, c.skip));
      i += 2;
    } else args.push_back(a);
  }
  if (args.empty()) { std::fprintf(stderr, "motion_check <구동.csv>[@시작:끝,…] … [--env env.yaml] [--floor office] [--fit]\n"); return 2; }
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
      fit_channel(group, true, path == 1, m.v_max, m.rc_v, m.delay_v, m.tau_v, m.acc_v, m.slow_v, m.tau_slow_v);
      fit_channel(group, false, path == 1, m.w_max, m.rc_w, m.delay_w, m.tau_w, m.acc_w, m.slow_w, m.tau_slow_w);
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
