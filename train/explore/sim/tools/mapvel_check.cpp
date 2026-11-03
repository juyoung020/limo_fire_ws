// 시뮬 = 현실 (지도 속도): 관측 "지도 속도" = 지도 자세(Cartographer map → base_link)를 0.1 s 로 미분해 로봇 기준으로 돌린 값.
// 실물은 스캔 매칭 보정이 계단처럼 튀어 바퀴 v·자이로 ω 와 어긋난다. 학습 시뮬 지도 자세 오차 모델(pose_error.h — 학습 환경이 쓰는
// 그 함수)에 실물 움직임(바퀴 v, 자이로 ω, 지도 방향)을 넣어 같은 잔차 통계를 내고 실물과 비교한다. --fit 은 맞춤 녹화로 값을 정한다.
//   mapvel_check [--set analysis/calibration.yaml] [--env config/env.yaml] [--model config/sensor_model.yaml] [--fit]
// 녹화는 calibration.yaml mapvel: 목록 (CSV 는 motion_export — map_x·map_y·map_yaw·mo_* 열).
// 잔차 (0.1 s 창마다): 앞 = 지도 vx − 바퀴 v 평균, 옆 = 지도 vy (차동 구동이라 참값 0), 회전 = 지도 ω − 자이로 평균.
// 통계: 잔차 σ 정지·느림·빠름, 움직일 때 큰 잔차 비율(앞·옆 > 0.1 m/s, 회전 > 0.3 rad/s), 0.1 s 자기상관. 허용 차이는 sensor_model.yaml check.mapvel_*.
// 실물 잔차에는 보정 말고도 바퀴 헛돎·토픽 시각 어긋남이 섞인다 — 시뮬은 이것까지 보정 잡음 하나로 낸다.
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pose_error.h"

namespace {

constexpr float kDt = 0.02f;   // 학습 환경 물리 걸음 = 오도메트리 줄 간격
constexpr int kWin = 5;        // 0.1 s 창 (정책 한 걸음)
const char* kName[3] = {"앞 (m/s)", "옆 (m/s)", "회전 (rad/s)"};
const float kTail[3] = {0.1f, 0.1f, 0.3f};
double g_err2[3] = {0, 0, 0};   // 시뮬 오차 제곱 합 (위치, yaw, 줄 수) — 지도 자세 오차 크기가 그럴듯한지 같이 본다

struct Row {
  double t;
  float v, w, mx, my, myaw;
  bool map_ok, corr, skip;   // corr: 이 줄에 map → odom 이 바뀜
};

struct Rec {
  std::string name, kind;
  std::vector<Row> rows;
};

Rec load(const std::string& path, const std::string& name, const std::string& kind, const std::vector<std::pair<double, double>>& skip) {
  Rec rec{name, kind, {}};
  std::ifstream f(path);
  if (!f) throw std::runtime_error("못 읽음: " + path + " (motion_export 로 만든다)");
  std::string line;
  std::getline(f, line);
  std::vector<std::string> head;
  {
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) head.push_back(c);
  }
  auto col = [&](const char* k) {
    for (size_t i = 0; i < head.size(); ++i) if (head[i] == k) return (int)i;
    throw std::runtime_error(path + ": 열 " + k + " 없음 (새 motion_export 로 다시 만든다)");
  };
  const int ct = col("t"), cv = col("odom_v"), cw = col("odom_w"), cg = col("gyro_w"), cx = col("map_x"), cy = col("map_y"), cth = col("map_yaw"),
            mox = col("mo_x"), moy = col("mo_y"), moth = col("mo_yaw");
  double px = NAN, py = NAN, pth = NAN;
  while (std::getline(f, line)) {
    std::vector<double> x;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) x.push_back(c == "nan" ? NAN : std::stod(c));
    Row r{};
    r.t = x[ct];
    r.v = (float)x[cv];
    r.w = (r.v == 0.f && x[cw] == 0.0) ? 0.f : (float)x[cg];   // 바퀴가 멈췄으면 정지 (IMU 와 같은 판정)
    r.mx = (float)x[cx];
    r.my = (float)x[cy];
    r.myaw = (float)x[cth];
    r.map_ok = std::isfinite(x[cx]);
    r.corr = std::isfinite(px) && std::isfinite(x[mox]) &&
             (std::fabs(x[mox] - px) > 1e-5 || std::fabs(x[moy] - py) > 1e-5 || std::fabs(std::remainder(x[moth] - pth, 2 * M_PI)) > 1e-5);
    px = x[mox];
    py = x[moy];
    pth = x[moth];
    for (const auto& [a, b] : skip) r.skip = r.skip || (r.t >= a && r.t <= b);
    rec.rows.push_back(r);
  }
  return rec;
}

// 창 하나의 잔차. pose(k) = 지도 자세 (실물) 또는 오차 (시뮬, 참값 = 바퀴로 만든 자세라 잔차 = 오차 미분)
struct Win {
  float r[3];
  float v, w;
  int bin;
  size_t k;
};

int bin_of(float v, float w) { return !sim::imu_moving(v, w) ? 0 : (std::fabs(v) < 0.3f && std::fabs(w) < 1.f ? 1 : 2); }

// 실물: 지도 자세 미분 − 바퀴·자이로
std::vector<Win> real_windows(const Rec& rec) {
  std::vector<Win> out;
  const auto& R = rec.rows;
  for (size_t k = kWin; k < R.size(); k += kWin) {
    const size_t j = k - kWin;
    bool ok = R[k].map_ok && R[j].map_ok;
    for (size_t q = j; q <= k && ok; ++q) ok = !R[q].skip;
    const double dt = R[k].t - R[j].t;
    if (!ok || dt < 0.09 || dt > 0.11) continue;
    const double dth = std::remainder((double)R[k].myaw - R[j].myaw, 2 * M_PI), m = R[j].myaw + 0.5 * dth;
    const double dx = R[k].mx - R[j].mx, dy = R[k].my - R[j].my;
    double v = 0, w = 0;
    for (size_t q = j + 1; q <= k; ++q) { v += R[q].v / kWin; w += R[q].w / kWin; }
    Win wd{};
    wd.r[0] = (float)((std::cos(m) * dx + std::sin(m) * dy) / dt - v);
    wd.r[1] = (float)((-std::sin(m) * dx + std::cos(m) * dy) / dt);
    wd.r[2] = (float)(dth / dt - w);
    wd.v = (float)v;
    wd.w = (float)w;
    wd.bin = bin_of(wd.v, wd.w);
    wd.k = k;
    out.push_back(wd);
  }
  return out;
}

// 시뮬: 같은 줄마다 pose_err_step (참값 방향 = 실물 지도 방향), 같은 창의 오차 변화 / 창 시간
std::vector<Win> sim_windows(const sim::PoseErrModel& M, const Rec& rec, const std::vector<Win>& real, uint64_t seed) {
  const auto& R = rec.rows;
  std::vector<std::array<float, 3>> e(R.size());
  float ee[3] = {0, 0, 0}, dd[3] = {0, 0, 0}, th = 0;
  uint64_t rng = seed;
  for (size_t i = 0; i < R.size(); ++i) {
    if (R[i].map_ok) th = R[i].myaw;
    sim::pose_err_step(M, ee, dd, th, R[i].v, R[i].w, rng);
    e[i] = {ee[0], ee[1], ee[2]};
    g_err2[0] += (double)ee[0] * ee[0] + (double)ee[1] * ee[1];
    g_err2[1] += (double)ee[2] * ee[2];
    g_err2[2] += 1;
  }
  std::vector<Win> out = real;
  for (auto& wd : out) {
    const size_t k = wd.k, j = k - kWin;
    const double dt = R[k].t - R[j].t, m = R[j].myaw;
    const double dx = e[k][0] - e[j][0], dy = e[k][1] - e[j][1];
    wd.r[0] = (float)((std::cos(m) * dx + std::sin(m) * dy) / dt);
    wd.r[1] = (float)((-std::sin(m) * dx + std::cos(m) * dy) / dt);
    wd.r[2] = (float)((e[k][2] - e[j][2]) / dt);
  }
  return out;
}

struct Stats {
  double sd[3][3], tail[3], acf[3];
  long n[3];
};

Stats stats(const std::vector<std::vector<Win>>& all) {
  Stats s{};
  double sq[3][3] = {}, tl[3] = {}, num[3] = {}, den[3] = {};
  long nm = 0;
  for (const auto& ws : all) {
    for (size_t i = 0; i < ws.size(); ++i) {
      const Win& w = ws[i];
      ++s.n[w.bin];
      for (int c = 0; c < 3; ++c) sq[c][w.bin] += (double)w.r[c] * w.r[c];
      if (w.bin > 0) {
        ++nm;
        for (int c = 0; c < 3; ++c) tl[c] += std::fabs(w.r[c]) > kTail[c];
        if (i + 1 < ws.size() && ws[i + 1].k == w.k + kWin && ws[i + 1].bin > 0)
          for (int c = 0; c < 3; ++c) { num[c] += (double)w.r[c] * ws[i + 1].r[c]; den[c] += (double)w.r[c] * w.r[c]; }
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < 3; ++b) s.sd[c][b] = s.n[b] ? std::sqrt(sq[c][b] / s.n[b]) : NAN;
    s.tail[c] = nm ? tl[c] / nm : NAN;
    s.acf[c] = den[c] > 0 ? num[c] / den[c] : NAN;
  }
  return s;
}

sim::PoseErrModel load_model(const std::string& env_yaml) {
  const YAML::Node p = YAML::LoadFile(env_yaml)["pose_error"]["mapvel_model"];
  sim::PoseErrModel m{};
  m.p_move = p["p"][0].as<float>();
  m.p_rest = p["p"][1].as<float>();
  m.tau = p["tau"].as<float>();
  m.tau_d = p["drift"][0].as<float>();
  for (int k = 0; k < 3; ++k) m.drift_f[k] = p["drift"][1 + k].as<float>();
  const char* keys[3] = {"fwd", "side", "yaw"};
  for (int k = 0; k < 3; ++k) {
    const YAML::Node c = p["noise"][keys[k]];
    m.s0[k] = c[0].as<float>();
    m.q1[k] = c[1].as<float>();
    m.qv[k] = c[2].as<float>();
    m.qw[k] = c[3].as<float>();
  }
  sim::pose_err_set_dt(m, kDt);
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  std::string set_path = SIM_ANALYSIS_DIR "/calibration.yaml", env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml";
  bool do_fit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--set" && i + 1 < argc) set_path = argv[++i];
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--fit") do_fit = true;
    else { std::fprintf(stderr, "mapvel_check [--set ..] [--env ..] [--model ..] [--fit]\n"); return 2; }
  }
  const std::string role = do_fit ? "fit" : "validate";
  const YAML::Node y = YAML::LoadFile(set_path);
  const std::string dir = set_path.substr(0, set_path.find_last_of('/') + 1);
  const std::string data = dir + y["data"].as<std::string>() + "/";
  std::vector<Rec> recs;
  for (const auto& r : y["mapvel"]) {
    if (r["role"].as<std::string>() != role) continue;
    std::vector<std::pair<double, double>> skip;
    if (r["skip"]) for (const auto& s : r["skip"]) skip.emplace_back(s[0].as<double>(), s[1].as<double>());
    const std::string bag = r["bag"].as<std::string>();
    recs.push_back(load(data + bag + ".csv", bag, r["kind"].as<std::string>(), skip));
  }
  if (recs.empty()) { std::fprintf(stderr, "mapvel 목록에 %s 녹화가 없음\n", role.c_str()); return 1; }
  std::vector<std::vector<Win>> real;
  for (const auto& r : recs) {
    real.push_back(real_windows(r));
    long nc = 0;
    for (const auto& w : r.rows) nc += w.corr;
    std::printf("%s %s (%s): 창 %zu, map → odom 바뀐 줄 %ld / %zu\n", role.c_str(), r.name.c_str(), r.kind.c_str(), real.back().size(), nc, r.rows.size());
  }

  sim::PoseErrModel M;
  if (do_fit) {
    M = sim::PoseErrModel{};
    M.tau = YAML::LoadFile(env_path)["pose_error"]["mapvel_model"]["tau"].as<float>();
    // 보정 빈도: 줄마다 map → odom 이 바뀐 비율 (움직임·정지)
    double c[2] = {0, 0}, n[2] = {0, 0};
    for (const auto& r : recs)
      for (const auto& w : r.rows) {
        if (w.skip || !w.map_ok) continue;
        const int mv = sim::imu_moving(w.v, w.w);
        c[mv] += w.corr;
        n[mv] += 1;
      }
    M.p_move = (float)(c[1] / n[1]);
    M.p_rest = (float)(c[0] / n[0]);
    // 잡음: 창 잔차 분산 = 창 안 보정 수(평균 kWin·p)·σ² / W² → σ² = 잔차²·W² / (kWin·p). 정지는 평균, 움직임은 [1, |v|, |ω|] 최소제곱
    const double W = kWin * kDt;
    for (int ch = 0; ch < 3; ++ch) {
      double s0 = 0, n0 = 0, A[3][3] = {}, b[3] = {};
      for (const auto& ws : real)
        for (const auto& w : ws) {
          if (w.bin == 0) { s0 += (double)w.r[ch] * w.r[ch] * W * W / (kWin * M.p_rest); n0 += 1; continue; }
          const double yv = (double)w.r[ch] * w.r[ch] * W * W / (kWin * M.p_move);
          const double x[3] = {1.0, std::fabs(w.v), std::fabs(w.w)};
          for (int p = 0; p < 3; ++p) { b[p] += x[p] * yv; for (int q = 0; q < 3; ++q) A[p][q] += x[p] * x[q]; }
        }
      M.s0[ch] = (float)std::sqrt(s0 / n0);
      double cf[3] = {0, 0, 0};
      for (int mask = 1; mask > 0;) {   // 상수 하나만: 실측 잔차가 느림·빠름에서 비슷 (느림 0.16 · 빠름 0.16 m/s) — 속도 항을 넣으면 느림을 절반으로 줄여 버림
        double a[3][3], bb[3];
        for (int p = 0; p < 3; ++p) { bb[p] = (mask >> p & 1) ? b[p] : 0; for (int q = 0; q < 3; ++q) a[p][q] = ((mask >> p & 1) && (mask >> q & 1)) ? A[p][q] : (p == q); }
        for (int p = 0; p < 3; ++p)
          for (int q = p + 1; q < 3; ++q) {
            const double f = a[q][p] / a[p][p];
            for (int j = p; j < 3; ++j) a[q][j] -= f * a[p][j];
            bb[q] -= f * bb[p];
          }
        for (int p = 2; p >= 0; --p) {
          double s = bb[p];
          for (int j = p + 1; j < 3; ++j) s -= a[p][j] * cf[j];
          cf[p] = s / a[p][p];
        }
        int neg = -1;
        for (int p = 0; p < 3; ++p) if ((mask >> p & 1) && cf[p] < 0) neg = p;
        if (neg < 0) break;
        mask &= ~(1 << neg);
        cf[neg] = 0;
      }
      M.q1[ch] = (float)cf[0];
      M.qv[ch] = (float)cf[1];
      M.qw[ch] = (float)cf[2];
    }
    // 이어지는 몫: 움직일 때 잔차 0.1 s 자기상관 ρ1 이 0.3~0.5 인데 0.2 s 는 ≈ 0 (짧게 이어짐 — 지수 꼴이 아님) → 상관 시간 tau_d·몫 f 를
    // 격자에서 시뮬로 골라 ρ1 을 맞춘다 (tau_d 는 세 통로 같이, f 는 통로마다)
    const auto acf1 = [&](const std::vector<std::vector<Win>>& all, int ch) {
      double num = 0, den = 0;
      for (const auto& ws : all)
        for (size_t i = 0; i + 1 < ws.size(); ++i)
          if (ws[i].bin > 0 && ws[i + 1].bin > 0 && ws[i + 1].k == ws[i].k + kWin) { num += (double)ws[i].r[ch] * ws[i + 1].r[ch]; den += (double)ws[i].r[ch] * ws[i].r[ch]; }
      return den > 0 ? num / den : 0.0;
    };
    double target[3];
    for (int ch = 0; ch < 3; ++ch) target[ch] = acf1(real, ch);
    const float taus[] = {0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.15f, 0.2f, 0.3f};
    double best = 1e30;
    sim::PoseErrModel bestM = M;
    for (float td : taus) {
      sim::PoseErrModel T = M;
      T.tau_d = td;
      double loss = 0;
      for (int ch = 0; ch < 3; ++ch) {
        double bl = 1e30;
        float bf = 0;
        for (int fi = 0; fi <= 19; ++fi) {
          const float f = 0.05f * fi;
          for (int c2 = 0; c2 < 3; ++c2) T.drift_f[c2] = f;
          sim::pose_err_set_dt(T, kDt);
          std::vector<std::vector<Win>> sw;
          for (size_t q = 0; q < recs.size(); ++q) sw.push_back(sim_windows(T, recs[q], real[q], 7 + 13 * q));
          const double d = acf1(sw, ch) - target[ch];
          if (d * d < bl) { bl = d * d; bf = f; }
        }
        loss += bl;
        T.drift_f[ch] = bf;
        bestM.drift_f[ch] = bestM.drift_f[ch];
        if (ch == 0) {}
        T.tau_d = td;
        // 통로마다 고른 f 를 기억
        (ch == 0 ? M.drift_f[0] : ch == 1 ? M.drift_f[1] : M.drift_f[2]) = bf;
      }
      if (loss < best) { best = loss; bestM = M; bestM.tau_d = td; }
    }
    M = bestM;
    std::printf("실측 0.1 s 자기상관 (앞·옆·회전): %.3f · %.3f · %.3f → tau_d %.2f s\n", target[0], target[1], target[2], M.tau_d);
    sim::pose_err_set_dt(M, kDt);
    std::printf("\n# env.yaml pose_error.mapvel_model 에 넣을 값 (mapvel_check --fit)\n  p: [%.4f, %.4f]\n  drift: [%.3f, %.3f, %.3f, %.3f]\n  noise:\n", M.p_move, M.p_rest, M.tau_d, M.drift_f[0], M.drift_f[1], M.drift_f[2]);
    const char* keys[3] = {"fwd", "side", "yaw"};
    for (int k = 0; k < 3; ++k) std::printf("    %s: [%.5f, %.4e, %.4e, %.4e]\n", keys[k], M.s0[k], M.q1[k], M.qv[k], M.qw[k]);
  } else {
    M = load_model(env_path);
  }
  std::vector<std::vector<Win>> simw;
  for (size_t q = 0; q < recs.size(); ++q) simw.push_back(sim_windows(M, recs[q], real[q], 2026 + 31 * q));
  const Stats R = stats(real), S = stats(simw);
  const YAML::Node tol = YAML::LoadFile(model_path)["check"];
  const double t_sd = tol["mapvel_sigma_ratio"].as<double>(), t_tail = tol["mapvel_tail"].as<double>(), t_acf = tol["mapvel_acf"].as<double>();
  std::printf("\n창 수: 정지 %ld · 느림 %ld · 빠름 %ld\n\n| 잔차 | 항목 | 시뮬 | 실물 | 차이 | 허용 | |\n|---|---|---|---|---|---|---|\n", R.n[0], R.n[1], R.n[2]);
  bool pass = true;
  auto row = [&](int c, const std::string& item, double s, double r, bool ratio, double t) {
    if (!std::isfinite(r)) return;
    const double d = ratio ? std::fabs(s / r - 1) : std::fabs(s - r);
    const bool ok = std::isfinite(d) && d <= t;
    pass = pass && ok;
    std::printf("| %s | %s | %.4f | %.4f | %s%.3f | %.2f | %s |\n", kName[c], item.c_str(), s, r, ratio ? "×" : "", d, t, ok ? "통과" : "**넘음**");
  };
  const char* bn[3] = {"σ 정지", "σ 느림", "σ 빠름"};
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < 3; ++b) if (R.n[b] > 30) row(c, bn[b], S.sd[c][b], R.sd[c][b], true, t_sd);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "움직일 때 |잔차| > %.1f 비율", kTail[c]);
    row(c, buf, S.tail[c], R.tail[c], false, t_tail);
    row(c, "0.1 s 자기상관 (움직임)", S.acf[c], R.acf[c], false, t_acf);
  }
  std::printf("\n시뮬 지도 자세 오차 RMS: 위치 %.3f m, yaw %.3f rad (실물 녹화 전체 map → odom 보정 0.07 m · 0.05 rad — 이보다 크면 지도가 망가짐)\n", std::sqrt(g_err2[0] / g_err2[2]), std::sqrt(g_err2[1] / g_err2[2]));
  std::printf("\n%s\n", pass ? "지도 속도 통과: 시뮬 잔차 통계가 실측 허용 안" : "지도 속도 넘음");
  return pass ? 0 : 1;
}
