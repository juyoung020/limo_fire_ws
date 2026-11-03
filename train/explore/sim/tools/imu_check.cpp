// 시뮬 = 현실 (IMU): 실물 녹화의 몸통 움직임(바퀴 // 잡음: s0 = 정지 σ. 움직일 때 (q1, qv, qw) = 잔차² 을 [1, |v|, |ω|] 로 최소제곱 (분산의 치우침 없는 맞춤). 음수 계수는 빼고 다시, 자이로 ω, v 변화 a)을 학습 시뮬 IMU 모델(imu.h — 학습 환경이 쓰는 그 함수)에
// 넣어 가속도 x·y·z, 자이로 x·y, roll·pitch 를 만들고, 실물 /imu 와 같은 통계 코드로 비교한다. --fit 은 맞춤 녹화로 모델 값을 정해
// sensor_model.yaml imu 에 넣을 줄을 낸다.
//   imu_check [--set analysis/calibration.yaml] [--role fit|validate] [--model config/sensor_model.yaml] [--fit]
// 녹화는 calibration.yaml imu: 목록 (kind rest = 정지 녹화, drive = 주행). CSV 는 limo_fire_sim motion_export (IMU 원값·REP-103 roll·pitch 열).
// 통계 (통로마다, 결정적 몫(imu_mean)을 뺀 잔차):
//   σ 정지 (바퀴 v = ω = 0) · 느림 (|v| < 0.3 m/s, |ω| < 1 rad/s) · 빠름 (그 밖), 0.1 s 자기상관(움직이는 줄), 원값 σ(움직이는 줄)
// 실물 결정적 몫은 녹화마다 정지 평균으로 정한 바닥 기울기·자이로 바이어스를 쓰고, 시뮬도 같은 값을 넣는다(잡음 모델만 비교).
// 허용 차이는 sensor_model.yaml check.imu_*.
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "imu.h"
#include "sensor_model.h"

namespace {

constexpr float kDt = 0.02f;   // 학습 환경 물리 걸음 = 오도메트리 줄 간격
constexpr int kLag = 5;        // 0.1 s
const char* kName[sim::kImuCh] = {"ax", "ay", "az", "gx", "gy", "roll", "pitch"};

struct Row {
  double t;
  float v, w, a;
  float m[sim::kImuCh];
  bool fresh, skip, ok;
};

struct Rec {
  std::string name, kind;
  std::vector<Row> rows;
  float roll0 = 0, pitch0 = 0, bgx = 0, bgy = 0;
};

bool rest(const Row& r) { return !sim::imu_moving(r.v, r.w); }   // 시뮬 모델과 같은 판정 (정지 줄은 아래에서 ω 도 0)
int bin_of(const Row& r) { return rest(r) ? 0 : (std::fabs(r.v) < 0.3f && std::fabs(r.w) < 1.f ? 1 : 2); }

Rec load(const std::string& path, const std::string& name, const std::string& kind, const std::vector<std::pair<double, double>>& skip) {
  Rec rec;
  rec.name = name;
  rec.kind = kind;
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
  const int ct = col("t"), cv = col("odom_v"), cw = col("odom_w"), cg = col("gyro_w"), cn = col("imu_n");
  int cm[sim::kImuCh];
  for (int k = 0; k < sim::kImuCh; ++k) cm[k] = col(kName[k]);
  long last_n = -1;
  std::vector<float> wheel_w;
  while (std::getline(f, line)) {
    std::vector<double> x;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) x.push_back(c == "nan" ? NAN : std::stod(c));
    Row r{};
    r.t = x[ct];
    r.v = (float)x[cv];
    r.w = (float)x[cg];
    wheel_w.push_back((float)x[cw]);
    r.ok = true;
    for (int k = 0; k < sim::kImuCh; ++k) { r.m[k] = (float)x[cm[k]]; r.ok = r.ok && std::isfinite(r.m[k]); }
    const long n = (long)x[cn];
    r.fresh = n != last_n;   // 오도메트리 50 Hz · IMU 100 Hz: 같은 IMU 메시지를 두 줄이 같이 보면 뒤 줄은 통계에서 뺀다
    last_n = n;
    for (const auto& [a, b] : skip) r.skip = r.skip || (r.t >= a && r.t <= b);
    rec.rows.push_back(r);
  }
  // 몸통 직진 가속: 바퀴 v 를 ±0.1 s 로 미분 (바퀴 v 1 mm/s 단위 계단을 누름). 정지 판정은 바퀴 v·ω
  const size_t n = rec.rows.size();
  for (size_t i = 0; i < n; ++i) {
    const size_t lo = i >= 5 ? i - 5 : 0, hi = std::min(n - 1, i + 5);
    const double dt = rec.rows[hi].t - rec.rows[lo].t;
    rec.rows[i].a = dt > 0 ? (float)((rec.rows[hi].v - rec.rows[lo].v) / dt) : 0.f;
    if (rec.rows[i].v == 0.f && wheel_w[i] == 0.f) rec.rows[i].w = 0.f;   // 정지 줄은 ω 도 0 (rest())
  }
  // 바닥 기울기·자이로 바이어스: 정지 줄 평균 (없으면 전체)
  double s[4] = {0, 0, 0, 0};
  int k = 0;
  for (int pass = 0; pass < 2 && k == 0; ++pass)
    for (const Row& r : rec.rows)
      if (r.ok && r.fresh && (pass == 1 || rest(r))) { s[0] += r.m[5]; s[1] += r.m[6]; s[2] += r.m[3]; s[3] += r.m[4]; ++k; }
  rec.roll0 = s[0] / k;
  rec.pitch0 = s[1] / k;
  rec.bgx = s[2] / k;
  rec.bgy = s[3] / k;
  return rec;
}

sim::ImuState state_of(const Rec& r) {
  sim::ImuState s{};
  s.roll0 = r.roll0;
  s.pitch0 = r.pitch0;
  s.bgx = r.bgx;
  s.bgy = r.bgy;
  return s;
}

// 잔차 통계 (통로마다)
struct Stats {
  double sd[sim::kImuCh][3];
  long n[3];
  double acf[sim::kImuCh];
  double raw_sd[sim::kImuCh];
};

// meas[rec][row][ch] 를 recs 의 결정적 몫과 비교
Stats stats(const sim::ImuModel& m, const std::vector<Rec>& recs, const std::vector<std::vector<std::array<float, sim::kImuCh>>>& meas) {
  Stats st{};
  double sq[sim::kImuCh][3] = {}, num[sim::kImuCh] = {}, den[sim::kImuCh] = {}, rs[sim::kImuCh] = {}, rq[sim::kImuCh] = {};
  long nm = 0;
  for (size_t q = 0; q < recs.size(); ++q) {
    const Rec& rec = recs[q];
    const sim::ImuState s = state_of(rec);
    const size_t n = rec.rows.size();
    std::vector<std::array<float, sim::kImuCh>> res(n);
    for (size_t i = 0; i < n; ++i) {
      float mean[sim::kImuCh];
      sim::imu_mean(m, s, rec.rows[i].v, rec.rows[i].w, rec.rows[i].a, mean);
      for (int k = 0; k < sim::kImuCh; ++k) res[i][k] = meas[q][i][k] - mean[k];
    }
    auto use = [&](size_t i) { return rec.rows[i].ok && rec.rows[i].fresh && !rec.rows[i].skip; };
    for (size_t i = 0; i < n; ++i) {
      if (!use(i)) continue;
      const int b = bin_of(rec.rows[i]);
      ++st.n[b];
      for (int k = 0; k < sim::kImuCh; ++k) sq[k][b] += (double)res[i][k] * res[i][k];
      if (b > 0) {
        ++nm;
        for (int k = 0; k < sim::kImuCh; ++k) { rs[k] += meas[q][i][k]; rq[k] += (double)meas[q][i][k] * meas[q][i][k]; }
        if (i + kLag < n && use(i + kLag) && bin_of(rec.rows[i + kLag]) > 0)
          for (int k = 0; k < sim::kImuCh; ++k) { num[k] += (double)res[i][k] * res[i + kLag][k]; den[k] += (double)res[i][k] * res[i][k]; }
      }
    }
  }
  for (int k = 0; k < sim::kImuCh; ++k) {
    for (int b = 0; b < 3; ++b) st.sd[k][b] = st.n[b] ? std::sqrt(sq[k][b] / st.n[b]) : NAN;
    st.acf[k] = den[k] > 0 ? num[k] / den[k] : NAN;
    st.raw_sd[k] = nm ? std::sqrt(std::max(0.0, rq[k] / nm - (rs[k] / nm) * (rs[k] / nm))) : NAN;
  }
  return st;
}

std::vector<std::vector<std::array<float, sim::kImuCh>>> real_meas(const std::vector<Rec>& recs) {
  std::vector<std::vector<std::array<float, sim::kImuCh>>> out(recs.size());
  for (size_t q = 0; q < recs.size(); ++q)
    for (const Row& r : recs[q].rows) {
      std::array<float, sim::kImuCh> a;
      for (int k = 0; k < sim::kImuCh; ++k) a[k] = r.m[k];
      out[q].push_back(a);
    }
  return out;
}

// 시뮬: 같은 움직임을 imu_step 에 (기울기·바이어스는 그 녹화 실측값, 잡음만 모델)
std::vector<std::vector<std::array<float, sim::kImuCh>>> sim_meas(const sim::ImuModel& m, const std::vector<Rec>& recs, uint64_t seed) {
  std::vector<std::vector<std::array<float, sim::kImuCh>>> out(recs.size());
  for (size_t q = 0; q < recs.size(); ++q) {
    uint64_t rng = seed + q * 7919;
    sim::ImuState s;
    sim::imu_reset(m, s, rng);
    const sim::ImuState fixed = state_of(recs[q]);
    s.roll0 = fixed.roll0; s.pitch0 = fixed.pitch0; s.bgx = fixed.bgx; s.bgy = fixed.bgy;
    for (const Row& r : recs[q].rows) {
      sim::imu_step(m, s, r.v, r.w, r.a, rng);
      std::array<float, sim::kImuCh> a;
      for (int k = 0; k < sim::kImuCh; ++k) a[k] = s.out[k];
      out[q].push_back(a);
    }
  }
  return out;
}

// 단순 최소제곱 y = b·x (원점 지남)
struct Ls { double xy = 0, xx = 0; void add(double x, double y) { xy += x * y; xx += x * x; } double b() const { return xx > 0 ? xy / xx : 0; } };

sim::ImuModel fit(const std::vector<Rec>& recs) {
  sim::ImuModel m{};
  // g: 정지 줄 az 평균, 기울기 σ·자이로 바이어스: 정지 녹화들의 녹화별 값
  double gz = 0, tilt2 = 0, bmax = 0;
  long ng = 0;
  int nt = 0;
  for (const Rec& r : recs) {
    for (const Row& w : r.rows) if (w.ok && w.fresh && rest(w)) { gz += w.m[2]; ++ng; }
    if (r.kind == "rest") {
      tilt2 += (double)r.roll0 * r.roll0 + (double)r.pitch0 * r.pitch0;
      nt += 2;
      bmax = std::max({bmax, (double)std::fabs(r.bgx), (double)std::fabs(r.bgy)});
    }
  }
  m.g = gz / ng;
  m.tilt_sigma = std::sqrt(tilt2 / nt);
  m.gyro_bias = std::max(bmax, 1e-4);
  // 기울기 응답 → 가속도 배율 (주행 녹화 움직이는 줄)
  Ls lp, lr, lx, ly;
  for (const Rec& r : recs) {
    // 움직이는 줄은 맞춤 녹화 전부에서 (정지 녹화의 주행 부분 포함)
    for (const Row& w : r.rows) {
      if (!w.ok || !w.fresh || w.skip || rest(w)) continue;
      lp.add(w.a, w.m[6] - r.pitch0);
      lr.add(w.v * w.w, w.m[5] - r.roll0);
    }
  }
  m.b_pitch = lp.b();
  m.b_roll = lr.b();
  for (const Rec& r : recs) {
    // 움직이는 줄은 맞춤 녹화 전부에서 (정지 녹화의 주행 부분 포함)
    for (const Row& w : r.rows) {
      if (!w.ok || !w.fresh || w.skip || rest(w)) continue;
      lx.add(w.a, w.m[0] + m.g * (r.pitch0 + m.b_pitch * w.a));
      ly.add(w.v * w.w, w.m[1] - m.g * (r.roll0 + m.b_roll * w.v * w.w));
    }
  }
  m.k_ax = lx.b();
  m.k_ay = ly.b();
  // 잡음: s0 = 정지 σ. 움직일 때 (s1, sv, sw) = √(π/2)|잔차| 를 [1, |v|, |ω|] 로 최소제곱 (E|N(0, σ)| = σ·√(2/π)). 음수 계수는 빼고 다시
  for (int k = 0; k < sim::kImuCh; ++k) m.ch[k] = sim::ImuChannel{0, 0, 0, 0, 0, 0};
  std::vector<double> s0q(sim::kImuCh, 0);
  long n0 = 0;
  for (const Rec& r : recs) {
    const sim::ImuState s = state_of(r);
    for (const Row& w : r.rows) {
      if (!w.ok || !w.fresh || w.skip || !rest(w)) continue;
      float mean[sim::kImuCh];
      sim::imu_mean(m, s, w.v, w.w, w.a, mean);
      for (int k = 0; k < sim::kImuCh; ++k) s0q[k] += (double)(w.m[k] - mean[k]) * (w.m[k] - mean[k]);
      ++n0;
    }
  }
  for (int k = 0; k < sim::kImuCh; ++k) m.ch[k].s0 = (float)std::sqrt(s0q[k] / n0);
  for (int k = 0; k < sim::kImuCh; ++k) {
    // 정규방정식 3 × 3, 쓰는 열 mask (음수 계수가 나오면 그 열을 빼고 다시)
    double c[3] = {0, 0, 0};
    for (int mask = 7; mask > 0; ) {
      double A[3][3] = {}, b[3] = {};
      for (const Rec& r : recs) {
        // 움직이는 줄은 맞춤 녹화 전부에서 (정지 녹화의 주행 부분 포함)
        const sim::ImuState s = state_of(r);
        for (const Row& w : r.rows) {
          if (!w.ok || !w.fresh || w.skip || rest(w)) continue;
          float mean[sim::kImuCh];
          sim::imu_mean(m, s, w.v, w.w, w.a, mean);
          const double y = (double)(w.m[k] - mean[k]) * (w.m[k] - mean[k]);
          const double x[3] = {1.0, std::fabs(w.v), std::fabs(w.w)};
          for (int p = 0; p < 3; ++p) { b[p] += x[p] * y; for (int q = 0; q < 3; ++q) A[p][q] += x[p] * x[q]; }
        }
      }
      for (int p = 0; p < 3; ++p) if (!(mask >> p & 1)) { for (int q = 0; q < 3; ++q) A[p][q] = A[q][p] = 0; A[p][p] = 1; b[p] = 0; }
      // 가우스 소거
      for (int p = 0; p < 3; ++p) {
        for (int q = p + 1; q < 3; ++q) {
          const double f = A[q][p] / A[p][p];
          for (int j = p; j < 3; ++j) A[q][j] -= f * A[p][j];
          b[q] -= f * b[p];
        }
      }
      for (int p = 2; p >= 0; --p) {
        double s = b[p];
        for (int j = p + 1; j < 3; ++j) s -= A[p][j] * c[j];
        c[p] = s / A[p][p];
      }
      int neg = -1;
      for (int p = 0; p < 3; ++p) if ((mask >> p & 1) && c[p] < 0) neg = p;
      if (neg < 0) break;
      mask &= ~(1 << neg);
      c[neg] = 0;
    }
    m.ch[k].q1 = (float)c[0];
    m.ch[k].qv = (float)c[1];
    m.ch[k].qw = (float)c[2];
  }
  // 상관 시간: 움직이는 줄 잔차의 0.1 s 자기상관 ρ (확인 통계와 같은 정의) → tau = −0.1 / ln ρ
  for (int k = 0; k < sim::kImuCh; ++k) {
    double num = 0, den = 0;
    for (const Rec& r : recs) {
      // 움직이는 줄은 맞춤 녹화 전부에서 (정지 녹화의 주행 부분 포함)
      const sim::ImuState s = state_of(r);
      const size_t n = r.rows.size();
      std::vector<double> z(n, NAN);
      for (size_t i = 0; i < n; ++i) {
        const Row& w = r.rows[i];
        if (!w.ok || !w.fresh || w.skip || rest(w)) continue;
        float mean[sim::kImuCh];
        sim::imu_mean(m, s, w.v, w.w, w.a, mean);
        z[i] = w.m[k] - mean[k];
      }
      for (size_t i = 0; i + kLag < n; ++i)
        if (std::isfinite(z[i]) && std::isfinite(z[i + kLag])) { num += z[i] * z[i + kLag]; den += z[i] * z[i]; }
    }
    const double rho = den > 0 ? num / den : 0;
    m.ch[k].tau = rho > 0.05 ? (float)(-kLag * kDt / std::log(std::min(rho, 0.995))) : 0.f;
  }
  sim::imu_set_dt(m, kDt);
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  std::string set_path = SIM_ANALYSIS_DIR "/calibration.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", role = "validate";
  bool do_fit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--set" && i + 1 < argc) set_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--role" && i + 1 < argc) role = argv[++i];
    else if (a == "--fit") { do_fit = true; role = "fit"; }
    else { std::fprintf(stderr, "imu_check [--set calibration.yaml] [--role fit|validate] [--model sensor_model.yaml] [--fit]\n"); return 2; }
  }
  const YAML::Node y = YAML::LoadFile(set_path);
  const std::string dir = set_path.substr(0, set_path.find_last_of('/') + 1);
  const std::string data = dir + y["data"].as<std::string>() + "/";
  std::vector<Rec> recs;
  for (const auto& r : y["imu"]) {
    if (r["role"].as<std::string>() != role) continue;
    std::vector<std::pair<double, double>> skip;
    if (r["skip"]) for (const auto& s : r["skip"]) skip.emplace_back(s[0].as<double>(), s[1].as<double>());
    const std::string bag = r["bag"].as<std::string>();
    recs.push_back(load(data + bag + ".csv", bag, r["kind"].as<std::string>(), skip));
    std::printf("%s %s (%s): 줄 %zu, 바닥 기울기 roll %.4f pitch %.4f rad, 자이로 바이어스 x %.5f y %.5f rad/s\n", role.c_str(), bag.c_str(),
                recs.back().kind.c_str(), recs.back().rows.size(), recs.back().roll0, recs.back().pitch0, recs.back().bgx, recs.back().bgy);
  }
  if (recs.empty()) { std::fprintf(stderr, "%s: imu 목록에 role %s 녹화가 없음\n", set_path.c_str(), role.c_str()); return 1; }

  sim::ImuModel m;
  if (do_fit) {
    m = fit(recs);
    std::printf("\n# sensor_model.yaml imu 에 넣을 값 (imu_check --fit, 녹화 위)\nimu:\n  g: %.4f\n  k: [%.3f, %.3f]\n  tilt: [%.5f, %.5f]\n  tilt_sigma: %.4f\n  gyro_bias: %.5f\n  noise:\n",
                m.g, m.k_ax, m.k_ay, m.b_pitch, m.b_roll, m.tilt_sigma, m.gyro_bias);
    for (int k = 0; k < sim::kImuCh; ++k)
      std::printf("    %s: [%.5f, %.4e, %.4e, %.4e, %.3f]\n", kName[k], m.ch[k].s0, m.ch[k].q1, m.ch[k].qv, m.ch[k].qw, m.ch[k].tau);
  } else {
    m = sim::load_imu_model(model_path, kDt);
  }
  const Stats R = stats(m, recs, real_meas(recs));
  const Stats S = stats(m, recs, sim_meas(m, recs, 2026));
  const YAML::Node tol = YAML::LoadFile(model_path)["check"];
  const double t_sd = tol["imu_sigma_ratio"].as<double>(), t_acf = tol["imu_acf"].as<double>(), t_raw = tol["imu_raw_ratio"].as<double>();
  std::printf("\n줄 수: 정지 %ld · 느림 %ld · 빠름 %ld (실물 같은 줄을 시뮬로)\n\n| 통로 | 항목 | 시뮬 | 실물 | 차이 | 허용 | |\n|---|---|---|---|---|---|---|\n",
              R.n[0], R.n[1], R.n[2]);
  bool pass = true;
  const char* bn[3] = {"잔차 σ 정지", "잔차 σ 느림", "잔차 σ 빠름"};
  auto row = [&](int k, const char* item, double s, double r, bool ratio, double t) {
    const double d = ratio ? std::fabs(s / r - 1) : std::fabs(s - r);
    if (!std::isfinite(r)) return;
    const bool ok = std::isfinite(d) && d <= t;
    pass = pass && ok;
    std::printf("| %s | %s | %.5f | %.5f | %s%.3f | %.2f | %s |\n", kName[k], item, s, r, ratio ? "×" : "", d, t, ok ? "통과" : "**넘음**");
  };
  for (int k = 0; k < sim::kImuCh; ++k) {
    for (int b = 0; b < 3; ++b) if (R.n[b] > 50) row(k, bn[b], S.sd[k][b], R.sd[k][b], true, t_sd);
    if (R.n[1] + R.n[2] > 50) {
      row(k, "0.1 s 자기상관 (움직임)", S.acf[k], R.acf[k], false, t_acf);
      row(k, "원값 σ (움직임)", S.raw_sd[k], R.raw_sd[k], true, t_raw);
    }
  }
  std::printf("\n%s\n", pass ? "IMU 통과: 시뮬 잡음·진동 통계가 실측 허용 안" : "IMU 넘음");
  return pass ? 0 : 1;
}
