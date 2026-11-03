// fake_run — 가짜 탐사 PPO 실행 폴더 만들기(뷰어 시험용). 진짜 러너와 같은 길로 쓰려고 include/runfolder.h 만 쓴다.
//
// 곡선(progress)·판 기록(episodes, s_eval)·평가 표(evals)·사건(events)·재생 파일(.xrp)을 만든다.
// 재생 판은 작은 2D 시뮬로 실제로 돌린다: 방 여럿인 정답 맵, 라이다 248 빔(±110°, 8 m), 관측 지도 쌓기, 2.5 m 안에서 본 빈 칸 = 덮음,
// 관측 지도의 가장 가까운 프론티어를 따라가는 정책 + 학습 진행에 따라 줄어드는 잡음(충돌·이른 선언). 숫자는 전부 가짜다.
// run.json 에 "synthetic": true 를 적어 뷰어가 "가짜 시험 자료" 띠를 띄운다.
//
//   fake_run --out <root>/<이름> [--iters 300] [--seed 1] [--group fake_base] [--live 0.2] [--resume <이터>] [--replay-every 50] [--eval-every 50]
//     --live S     : 업데이트마다 S 초 쉰다(학습 중인 실행 흉내 — 뷰어 "Training" 불, SSE 시험). 없으면 한 번에 다 쓴다.
//     --resume I   : 같은 폴더를 이터 I 체크포인트에서 이어 쓴다(I 이상 줄을 지움 — 되감기 시험).
#include "runfolder.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <random>
#include <thread>

namespace {

constexpr double DT = 0.1;          // 10 Hz (mdp/README 행동 주기)
constexpr double V_MAX = 1.0, VY_MAX = 1.0, W_MAX = 3.14;
constexpr int LIDAR_N = 248;        // ±110°, 248 칸 (mdp/README)
constexpr double LIDAR_FOV = 220.0 * M_PI / 180.0, LIDAR_RMAX = 8.0;
constexpr double COVER_R = 2.5;     // 가짜 덮음 정의: 2.5 m 안에서 본 빈 칸 (진짜 정의는 mdp 에서 정할 것)
constexpr double ROBOT_R = 0.2;
constexpr double SUCCESS_COV = 0.90;  // 가짜 성공 기준(선언 때 덮음)
constexpr double T_SAFE = 120.0;    // 가짜 안전 시간(s)
const std::vector<std::string> TERMS = {"coverage", "time", "slip", "collision", "declare"};
const std::vector<std::string> STAGES = {"small", "medium", "large"};

double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double wrap(double a) { while (a > M_PI) a -= 2 * M_PI; while (a < -M_PI) a += 2 * M_PI; return a; }
double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

// ---------------------------------------------------------------------------------------------- 정답 맵
struct Map {
  rf::Grid g;
  std::vector<uint8_t> c;  // rf::gt
  std::string name;
  double sx = 0, sy = 0;   // 시작 자세
  int free_cells = 0;
  uint8_t at(int i, int j) const { return (i < 0 || j < 0 || i >= g.w || j >= g.h) ? rf::gt::OUTSIDE : c[size_t(j) * g.w + i]; }
  bool wall_or_out(double x, double y) const {
    int i = int(std::floor((x - g.ox) / g.res)), j = int(std::floor((y - g.oy) / g.res));
    return at(i, j) != rf::gt::FREE;
  }
};

// 방 nx × ny 개(한 변 room m), 사이 벽에 문, 상자 장애물 몇 개. 바깥은 OUTSIDE.
Map make_map(int nx, int ny, double room, std::mt19937& rng, const std::string& name) {
  Map m;
  m.name = name;
  m.g.res = 0.1;
  const int rc = int(room / m.g.res), pad = 4;
  m.g.w = nx * rc + 1 + 2 * pad;
  m.g.h = ny * rc + 1 + 2 * pad;
  m.g.ox = -pad * m.g.res;
  m.g.oy = -pad * m.g.res;
  m.c.assign(size_t(m.g.w) * m.g.h, rf::gt::OUTSIDE);
  auto set = [&](int i, int j, uint8_t v) { if (i >= 0 && j >= 0 && i < m.g.w && j < m.g.h) m.c[size_t(j) * m.g.w + i] = v; };
  for (int j = 0; j <= ny * rc; j++)
    for (int i = 0; i <= nx * rc; i++) set(i + pad, j + pad, (i % rc == 0 || j % rc == 0) ? rf::gt::WALL : rf::gt::FREE);
  std::uniform_int_distribution<int> door(rc / 4, rc - rc / 4 - 9);
  // 문(폭 0.9 m): 옆 방 사이마다, 가끔 벽을 통째로 없앰(큰 방)
  std::uniform_real_distribution<double> u(0, 1);
  for (int by = 0; by < ny; by++)
    for (int bx = 0; bx < nx; bx++) {
      if (bx + 1 < nx) {
        int x = (bx + 1) * rc, d = door(rng);
        bool open = u(rng) < 0.15;
        for (int j = by * rc + 1; j < (by + 1) * rc; j++) if (open || (j - by * rc >= d && j - by * rc < d + 9)) set(x + pad, j + pad, rf::gt::FREE);
      }
      if (by + 1 < ny) {
        int y = (by + 1) * rc, d = door(rng);
        bool open = u(rng) < 0.15;
        for (int i = bx * rc + 1; i < (bx + 1) * rc; i++) if (open || (i - bx * rc >= d && i - bx * rc < d + 9)) set(i + pad, y + pad, rf::gt::FREE);
      }
      // 상자 하나(가구), 방 가운데는 비움
      int bw = 4 + int(u(rng) * 6), bh = 4 + int(u(rng) * 6);
      int x0 = bx * rc + 3 + int(u(rng) * (rc - bw - 6)), y0 = by * rc + 3 + int(u(rng) * (rc - bh - 6));
      if (u(rng) < 0.8)
        for (int j = y0; j < y0 + bh; j++) for (int i = x0; i < x0 + bw; i++) {
          double cx = (bx + 0.5) * rc, cy = (by + 0.5) * rc;
          if (std::hypot(i - cx, j - cy) > rc * 0.2) set(i + pad, j + pad, rf::gt::WALL);
        }
    }
  m.sx = 0.5 * room;
  m.sy = 0.5 * room;
  for (auto v : m.c) m.free_cells += v == rf::gt::FREE;
  return m;
}

// ---------------------------------------------------------------------------------------------- 판 하나 시뮬
struct EpResult {
  std::string outcome;
  int steps = 0;
  double t = 0, ret = 0, coverage = 0, cov_at_declare = NAN, t95 = NAN, dist = 0, known_m2 = 0, area_m2 = 0;
  std::map<std::string, double> r;
};

struct Sim {
  const Map& m;
  std::mt19937& rng;
  double skill;  // 0 = 학습 처음, 1 = 다 배움
  std::vector<int8_t> obs;
  std::vector<uint8_t> cov;
  std::vector<float> ranges;
  int covered_n = 0;
  Sim(const Map& mm, std::mt19937& r, double s) : m(mm), rng(r), skill(s), obs(mm.c.size(), -1), cov(mm.c.size(), 0), ranges(LIDAR_N) {}

  int idx(double x, double y) const {
    int i = int(std::floor((x - m.g.ox) / m.g.res)), j = int(std::floor((y - m.g.oy) / m.g.res));
    return (i < 0 || j < 0 || i >= m.g.w || j >= m.g.h) ? -1 : j * m.g.w + i;
  }
  // 라이다: 빔마다 반 칸씩 걸어 벽에서 멈춤. 지도에 빈 칸·벽 쌓기, 3 m 안 빈 칸 덮음. 새로 덮인 칸 수.
  int sense(double x, double y, double yaw, std::normal_distribution<double>& nz) {
    int fresh = 0;
    for (int k = 0; k < LIDAR_N; k++) {
      double a = yaw - LIDAR_FOV / 2 + k * LIDAR_FOV / (LIDAR_N - 1), ca = std::cos(a), sa = std::sin(a);
      double r = 0;
      int last = -1;
      ranges[k] = INFINITY;
      for (r = 0.05; r < LIDAR_RMAX; r += 0.05) {
        int c = idx(x + r * ca, y + r * sa);
        if (c < 0 || m.c[c] == rf::gt::OUTSIDE) break;
        if (c == last) continue;
        last = c;
        if (m.c[c] == rf::gt::WALL) {
          obs[c] = int8_t(std::min(100, (obs[c] < 0 ? 50 : obs[c]) + 40));
          ranges[k] = float(r + 0.01 * nz(rng));
          break;
        }
        obs[c] = int8_t(std::max(0, (obs[c] < 0 ? 50 : obs[c]) - 25));
        if (r <= COVER_R && !cov[c]) { cov[c] = 1; covered_n++; fresh++; }
      }
    }
    return fresh;
  }
  bool blocked(int i, int j) const {  // 관측 지도 기준 막힘(로봇 반지름만큼 부풀림)
    for (int dj = -3; dj <= 3; dj++)
      for (int di = -3; di <= 3; di++) {
        int ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= m.g.w || jj >= m.g.h) return true;
        if (obs[size_t(jj) * m.g.w + ii] >= 60) return true;
      }
    return false;
  }
  // 가장 가까운 목표(안 덮인 아는 빈 칸, 또는 모르는 칸 옆 프론티어)까지 BFS 길 → 몇 칸 앞 경유점. 없으면 false.
  bool waypoint(double x, double y, double& wx, double& wy) {
    const int W = m.g.w, H = m.g.h;
    int s = idx(x, y);
    if (s < 0) return false;
    std::vector<int> prev(size_t(W) * H, -2);
    std::deque<int> q{s};
    prev[s] = -1;
    int goal = -1;
    while (!q.empty() && goal < 0) {
      int c = q.front();
      q.pop_front();
      if (c != s && !cov[c]) { goal = c; break; }  // 아직 안 덮인 아는 빈 칸
      int i = c % W, j = c / W;
      static const int d4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (auto& d : d4) {
        int ii = i + d[0], jj = j + d[1];
        if (ii < 0 || jj < 0 || ii >= W || jj >= H) continue;
        int n = jj * W + ii;
        if (prev[n] != -2) continue;
        if (obs[n] < 0) { if (c != s) goal = c; continue; }  // c 는 프론티어(모르는 칸 옆 아는 빈 칸)
        if (obs[n] >= 50 || (blocked(ii, jj) && c != s)) continue;
        prev[n] = c;
        q.push_back(n);
      }
    }
    if (goal < 0) return false;
    std::vector<int> path;
    for (int c = goal; c >= 0; c = prev[c]) path.push_back(c);
    std::reverse(path.begin(), path.end());
    int w = path[std::min<size_t>(path.size() - 1, 6)];
    wx = m.g.ox + (w % W + 0.5) * m.g.res;
    wy = m.g.oy + (w / W + 0.5) * m.g.res;
    return true;
  }
};

EpResult simulate(const Map& m, std::mt19937& rng, double skill, rf::ReplayWriter* rw) {
  Sim s(m, rng, skill);
  std::normal_distribution<double> nz(0, 1);
  std::uniform_real_distribution<double> u(0, 1);
  EpResult e;
  for (auto& t : TERMS) e.r[t] = 0;
  int reachable = 0;
  {  // 정답 맵에서 시작점에서 닿는 빈 칸(덮음 분모)
    std::vector<char> seen(m.c.size(), 0);
    std::deque<int> q{s.idx(m.sx, m.sy)};
    seen[q.front()] = 1;
    while (!q.empty()) {
      int c = q.front(); q.pop_front(); reachable++;
      int i = c % m.g.w, j = c / m.g.w;
      for (int d = 0; d < 4; d++) {
        int ii = i + (d == 0) - (d == 1), jj = j + (d == 2) - (d == 3);
        int n = jj * m.g.w + ii;
        if (m.at(ii, jj) == rf::gt::FREE && !seen[n]) { seen[n] = 1; q.push_back(n); }
      }
    }
  }
  e.area_m2 = reachable * m.g.res * m.g.res;
  const double t_ref = reachable * m.g.res * m.g.res / 0.5;  // 기준 덮는 속도 0.5 m²/s (가짜)
  double x = m.sx, y = m.sy, yaw = u(rng) * 2 * M_PI - M_PI, ex = 0, ey = 0, eth = 0, ret = 0;
  const double noise = 0.9 * (1 - skill) + 0.05, hazard = 0.006 * (1 - skill) * (1 - skill), speed = 0.45 + 0.55 * skill;
  double v = 0, vy = 0, w = 0, wx = x, wy = y, last_plan = -1;
  bool have_wp = false;
  for (int step = 0;; step++) {
    const double t = step * DT;
    int fresh = s.sense(x, y, yaw, nz);
    double cov = double(s.covered_n) / std::max(1, reachable);
    // 정책(가짜): 경유점으로 돌고 나아감 + 잡음
    if (t - last_plan >= 0.5 || !have_wp || std::hypot(wx - x, wy - y) < 0.3) { have_wp = s.waypoint(x, y, wx, wy); last_plan = t; }
    double logit = -12.0 + 10.0 * skill * clampd((cov - 0.8) / 0.2, 0, 1);  // 걸음마다 뽑으므로 아주 낮게 시작(mdp/README: 로짓 0 이면 평균 2 걸음에 끝남)
    if (!have_wp) logit = 3.0;                       // 프론티어 없음 → 선언
    if (u(rng) < hazard) logit = 2.0;                // 이른 선언(학습 초기)
    double p_end = sigm(logit);
    bool declare = u(rng) < p_end;
    if (have_wp) {
      double err = wrap(std::atan2(wy - y, wx - x) - yaw), d = std::hypot(wx - x, wy - y);
      w = clampd(2.5 * err + noise * 1.2 * nz(rng), -W_MAX, W_MAX);
      v = clampd(speed * std::cos(err) * std::min(1.0, d + 0.3) + noise * 0.25 * nz(rng), -0.3, V_MAX);
      if (std::fabs(err) > 1.2) v = 0.05;
      vy = clampd(0.5 * skill * std::sin(err) * std::min(1.0, d) + noise * 0.2 * nz(rng), -VY_MAX, VY_MAX);
    } else {
      v = 0; vy = 0; w = 0.5 * nz(rng) * noise;
    }
    // 이번 걸음 보상(가짜 식 — 진짜는 mdp/rewards.h)
    std::map<std::string, double> r;
    r["coverage"] = 2.0 * fresh * m.g.res * m.g.res;
    r["time"] = -DT / t_ref * 10.0;
    r["slip"] = -0.02 * std::fabs(nz(rng)) * noise;
    r["collision"] = 0;
    r["declare"] = 0;
    uint32_t evb = 0;
    if (std::isnan(e.t95) && cov >= 0.95) { e.t95 = t; evb |= rf::ev::REACHED95; }
    // 배운 피하기(가짜): 실력만큼의 확률로, 다음 자세가 관측 지도의 벽에 닿으면 멈추고 제자리 회전
    if (u(rng) < skill) {
      double px = x + (v * std::cos(yaw) - vy * std::sin(yaw)) * DT * 2, py = y + (v * std::sin(yaw) + vy * std::cos(yaw)) * DT * 2;
      bool near = false;
      for (int k = 0; k < 12 && !near; k++) { int c = s.idx(px + (ROBOT_R + 0.08) * std::cos(k * M_PI / 6), py + (ROBOT_R + 0.08) * std::sin(k * M_PI / 6)); near = c < 0 || s.obs[c] >= 60; }
      if (near) { v = 0; vy = 0; w = clampd(w, -1.5, 1.5); }
    }
    // 움직이고 충돌 확인
    double nx = x + (v * std::cos(yaw) - vy * std::sin(yaw)) * DT, ny = y + (v * std::sin(yaw) + vy * std::cos(yaw)) * DT;
    bool hit = false;
    for (int k = 0; k < 12 && !hit; k++) hit = m.wall_or_out(nx + ROBOT_R * std::cos(k * M_PI / 6), ny + ROBOT_R * std::sin(k * M_PI / 6));
    std::string end;
    if (declare) {
      end = cov >= SUCCESS_COV ? rf::outcome::SUCCESS : rf::outcome::DECLARED_SHORT;
      r["declare"] = 15.0 * sigm((cov - SUCCESS_COV) / 0.02) - 5.0;
      e.cov_at_declare = cov;
      evb |= rf::ev::DECLARE;
    } else if (hit) {
      end = rf::outcome::COLLISION;
      r["collision"] = -10.0;
      evb |= rf::ev::COLLISION;
    } else if (t + DT >= T_SAFE) {
      end = rf::outcome::TRUNCATED;
      evb |= rf::ev::TRUNCATED;
    }
    double rs = 0;
    for (auto& kv : r) { e.r[kv.first] += kv.second; rs += kv.second; }
    ret += rs;
    if (rw) {
      std::vector<float> row;
      row.reserve(32);
      ex += 0.002 * nz(rng); ey += 0.002 * nz(rng); eth += 0.001 * nz(rng);
      for (double val : {t, x, y, yaw, x + ex, y + ey, yaw + eth, v * (1 + 0.03 * nz(rng)), vy, w, v, vy, w, p_end, double(declare)}) row.push_back(float(val));
      row.push_back(float(2.0 * reachable * m.g.res * m.g.res * (1 - cov) * (0.3 + 0.7 * skill) + (1.5 - skill) * nz(rng)));  // value(가짜): 남은 덮음 보상 어림
      for (auto& tn : TERMS) row.push_back(float(r[tn]));
      double known = 0;
      for (auto o : s.obs) known += (o >= 0 && o < 50);
      e.known_m2 = known * m.g.res * m.g.res;
      for (double val : {ret, cov, e.known_m2, double(evb)}) row.push_back(float(val));
      rw->frame(row.data());
      rw->obs(s.obs.data());
      rw->covered(s.cov.data());
      rw->scan(s.ranges.data());
    }
    if (!end.empty()) {
      e.outcome = end;
      e.steps = step + 1;
      e.t = (step + 1) * DT;
      e.coverage = cov;
      e.ret = ret;
      break;
    }
    e.dist += std::hypot(nx - x, ny - y);
    x = nx; y = ny; yaw = wrap(yaw + w * DT);
  }
  return e;
}

std::vector<std::string> replay_cols() {
  std::vector<std::string> c = {"t", "x", "y", "yaw", "sx", "sy", "syaw", "v", "vy", "w", "a_v", "a_vy", "a_w", "p_end", "a_end", "value"};
  for (auto& t : TERMS) c.push_back("r_" + t);
  for (auto n : {"ret", "coverage", "known_m2", "ev"}) c.push_back(n);
  return c;
}

rf::Json episode_line(const EpResult& e, long long ep, double it, double steps, int env, const std::string& map, const std::string& stage) {
  rf::Json j = rf::Json::object();
  j.set("iter", it).set("env_steps", steps).set("env", env).set("ep", ep).set("map", map).set("stage", stage).set("outcome", e.outcome)
      .set("success", e.outcome == rf::outcome::SUCCESS).set("declared", e.outcome == rf::outcome::SUCCESS || e.outcome == rf::outcome::DECLARED_SHORT)
      .set("collided", e.outcome == rf::outcome::COLLISION).set("truncated", e.outcome == rf::outcome::TRUNCATED)
      .set("t", e.t).set("steps", e.steps).set("ret", e.ret).set("coverage", e.coverage).set("path_len", e.dist).set("known_m2", e.known_m2).set("area_m2", e.area_m2);
  rf::Json r = rf::Json::object();
  double s = 0;
  for (auto& kv : e.r) { r.set(kv.first, kv.second); s += kv.second; }
  j.set("r", r);
  j.set("ret", s);
  if (std::isfinite(e.cov_at_declare)) j.set("coverage_at_declare", e.cov_at_declare);
  if (std::isfinite(e.t95)) j.set("t95", e.t95);
  return j;
}

// 정해진 결과로 판 줄만(재생 없이) — 표본 줄 수를 채운다
EpResult sample_episode(std::mt19937& rng, double L, double area) {
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> nz(0, 1);
  EpResult e;
  double ps = 0.85 * L, pc = 0.35 * (1 - L) + 0.03, pd = 0.4 * std::exp(-5 * L) + 0.03, x = u(rng);
  e.outcome = x < ps ? rf::outcome::SUCCESS : x < ps + pc ? rf::outcome::COLLISION : x < ps + pc + pd ? rf::outcome::DECLARED_SHORT : rf::outcome::TRUNCATED;
  e.coverage = clampd(e.outcome == rf::outcome::SUCCESS ? 0.93 + 0.05 * u(rng) : 0.2 + 0.6 * u(rng) * (0.4 + 0.6 * L), 0, 1);
  e.t = e.outcome == rf::outcome::TRUNCATED ? T_SAFE : clampd((110 - 50 * L) * (0.5 + 0.6 * u(rng)), 2, T_SAFE);
  e.steps = int(e.t / DT);
  e.dist = e.t * (0.3 + 0.4 * L) * (0.8 + 0.4 * u(rng));
  if (e.coverage >= 0.95) e.t95 = e.t * (0.8 + 0.15 * u(rng));
  if (e.outcome == rf::outcome::SUCCESS || e.outcome == rf::outcome::DECLARED_SHORT) e.cov_at_declare = e.coverage;
  e.known_m2 = area * e.coverage * 1.05;
  e.area_m2 = area;
  e.r["coverage"] = 2.0 * area * e.coverage;
  e.r["time"] = -e.t / (area / 0.5) * 10.0;
  e.r["slip"] = -0.01 * e.steps * (1.1 - L) * 0.1;
  e.r["collision"] = e.outcome == rf::outcome::COLLISION ? -10.0 : 0.0;
  e.r["declare"] = std::isfinite(e.cov_at_declare) ? 15.0 * sigm((e.coverage - SUCCESS_COV) / 0.02) - 5.0 : 0.0;
  for (auto& kv : e.r) e.ret += kv.second;
  (void)nz;
  return e;
}

struct Args {
  std::string out, group = "fake_base";
  int iters = 300, seed = 1, replay_every = 50, eval_every = 50;
  double live = 0, resume = -1;
};

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) { std::cerr << k << " needs a value\n"; std::exit(2); } return argv[++i]; };
    if (k == "--out") a.out = nx();
    else if (k == "--iters") a.iters = std::stoi(nx());
    else if (k == "--seed") a.seed = std::stoi(nx());
    else if (k == "--group") a.group = nx();
    else if (k == "--live") a.live = std::stod(nx());
    else if (k == "--resume") a.resume = std::stod(nx());
    else if (k == "--replay-every") a.replay_every = std::stoi(nx());
    else if (k == "--eval-every") a.eval_every = std::stoi(nx());
    else { std::cerr << "usage: fake_run --out <root>/<name> [--iters 300] [--seed 1] [--group g] [--live S] [--resume ITER] [--replay-every 50] [--eval-every 50]\n"; return 2; }
  }
  if (a.out.empty()) { std::cerr << "fake_run: --out is required\n"; return 2; }
  std::mt19937 rng(uint32_t(a.seed * 7919 + (a.resume >= 0 ? int(a.resume) : 0)));
  std::normal_distribution<double> nz(0, 1);
  std::uniform_real_distribution<double> u(0, 1);
  const int n_envs = 4096, T = 24, log_envs = 8;
  const double gamma = 0.995;

  rf::Json meta = rf::Json::object();
  rf::Json stages = rf::Json::array();
  for (size_t k = 0; k < STAGES.size(); k++) { rf::Json s = rf::Json::object(); s.set("name", STAGES[k]).set("rooms", std::to_string(k + 2) + "x" + std::to_string(k + 1)); stages.push(s); }
  rf::Json cur = rf::Json::object();
  cur.set("stages", stages);
  rf::Json refs = rf::Json::object();
  refs.set("train/approx_kl", 0.02).set("train/grad_norm", 1.0).set("rollout/success_rate", 0.8);
  rf::Json limits = rf::Json::object();
  limits.set("v", V_MAX).set("vy", VY_MAX).set("w", W_MAX);
  meta.set("synthetic", true).set("trainer", "fake_run (runfolder.h)").set("group", a.group).set("seed", a.seed).set("n_envs", n_envs)
      .set("rollout_T", T).set("log_envs", log_envs).set("ctrl_hz", 10).set("gamma", gamma).set("target_kl", 0.02).set("lr", 3e-4)
      .set("reward_terms", rf::Json::array(TERMS)).set("curriculum", cur).set("refs", refs).set("action_limits", limits)
      .set("success_coverage", SUCCESS_COV).set("t_safe_s", T_SAFE).set("coverage_def", "fake: free cell seen within 2.5 m")
      .set("area_bins", rf::Json::array(std::vector<double>{40, 120}));

  rf::RunFolder::Options opt;
  opt.every_s = a.live > 0 ? 1.0 : 0.0;
  if (a.resume >= 0) { opt.resume = true; opt.resume_iter = a.resume; opt.resume_steps = a.resume * n_envs * T; }
  rf::RunFolder run(a.out, meta, opt);
  std::cerr << "fake_run: " << a.out << " pid " << getpid() << (a.resume >= 0 ? " (resume)" : "") << "\n";

  const int it0 = a.resume >= 0 ? int(a.resume) : 0;
  int stage = -1;
  std::map<int, Map> maps;
  auto map_for = [&](int st) -> Map& {
    auto it = maps.find(st);
    if (it == maps.end()) {
      std::mt19937 mr(uint32_t(1000 + st * 31 + a.seed));
      it = maps.emplace(st, make_map(2 + st, 1 + st, 5.0, mr, "rooms_" + STAGES[st])).first;
    }
    return it->second;
  };
  auto run_replay = [&](double L, double it, double steps, const std::string& stream, int env) {
    int st = std::max(0, stage);
    Map& m = map_for(st);
    std::vector<std::string> cols = replay_cols();
    rf::ReplayWriter rw(cols, DT, m.g, m.c.data());
    rw.set_lidar(LIDAR_N, float(-LIDAR_FOV / 2), float(LIDAR_FOV / (LIDAR_N - 1)), float(LIDAR_RMAX));
    rw.set_limits(float(V_MAX), float(VY_MAX), float(W_MAX));
    rw.set_reward_terms(TERMS);
    EpResult e = simulate(m, rng, clampd(L + 0.15 * nz(rng), 0, 1), &rw);
    long long ep = run.new_episode_id();
    rf::Json line = episode_line(e, ep, it, steps, env, m.name, STAGES[st]);
    std::string path = run.replay_path(ep, e.outcome, stream);
    line.set("replay", rf::basename_of(path));
    size_t bytes = rw.finish(path, line);
    run.episode(line, stream);
    return std::make_pair(e, bytes);
  };

  for (int it = it0; it < a.iters; it++) {
    const double p = double(it) / std::max(1, a.iters - 1), L = sigm((p - 0.4) * 9);
    const double steps = double(it + 1) * n_envs * T;
    int st = p < 0.3 ? 0 : p < 0.65 ? 1 : 2;
    if (st != stage) {
      if (stage >= 0) run.event("curriculum: stage " + std::to_string(stage) + " -> " + std::to_string(st) + " (" + STAGES[st] + ")");
      stage = st;
      run.flush_row();
    }
    Map& m = map_for(stage);
    const double area = m.free_cells * m.g.res * m.g.res;
    const double ep_len = clampd((110 - 50 * L) * 10 * (0.6 + 0.4 * L), 30, T_SAFE * 10);
    const double n_eps = std::max(0.0, n_envs * T / ep_len * (1 + 0.1 * nz(rng)));
    auto& r = run.row();
    r.last(rf::key::ITER, it);
    r.last(rf::key::STEPS, steps);
    if (a.live <= 0) r.last(rf::key::ELAPSED, 2.3 * (it + 1));  // 한 번에 쓸 때는 가짜 경과 시간
    r.mean(rf::key::FPS, 42000 * (1 + 0.03 * nz(rng)));
    r.mean(rf::key::ROLLOUT_MS, 1600 * (1 + 0.05 * nz(rng)));
    r.mean(rf::key::UPDATE_MS, 650 * (1 + 0.05 * nz(rng)));
    r.mean(rf::key::ITER_MS, 2300 * (1 + 0.04 * nz(rng)));
    r.sum(rf::key::N_EPISODES, std::round(n_eps));
    const double w = std::round(n_eps);
    double sr = clampd(0.85 * L + 0.03 * nz(rng), 0, 1), cr = clampd(0.35 * (1 - L) + 0.03 + 0.02 * nz(rng), 0, 1);
    double dr = clampd(0.4 * std::exp(-5 * p) + 0.03 + 0.01 * nz(rng), 0, 1), tr = clampd(1 - sr - cr - dr, 0, 1);
    double covm = clampd(0.3 + 0.65 * L + 0.02 * nz(rng), 0, 1);
    r.mean_w(rf::key::SUCCESS_RATE, sr, w);
    r.mean_w(rf::key::COLLISION_RATE, cr, w);
    r.mean_w(rf::key::DECLARE_SHORT_RATE, dr, w);
    r.mean_w(rf::key::DECLARE_RATE, sr + dr, w);
    r.mean_w(rf::key::TRUNCATION_RATE, tr, w);
    r.mean_w(rf::key::COVERAGE_MEAN, covm, w);
    r.mean_w(rf::key::COVERAGE_AT_DECLARE, clampd(0.55 + 0.4 * L + 0.02 * nz(rng), 0, 1), w * (sr + dr));
    double reach = clampd(0.9 * L, 0, 1);
    r.mean_w(rf::key::T95_MEAN, (110 - 55 * L) * (1 + 0.04 * nz(rng)), w * reach);  // 도달한 판이 없으면 키가 빠진다
    r.mean_w(rf::key::T95_REACHED_RATE, reach, w);
    r.mean_w(rf::key::PATH_LEN_MEAN, (25 + 30 * L) * (1 + 0.05 * nz(rng)), w);
    r.mean_w(rf::key::KNOWN_AREA_MEAN, area * covm * 1.05, w);
    r.mean_w(rf::key::EP_LEN_MEAN, ep_len * (1 + 0.04 * nz(rng)), w);
    // 보상 항: 판당 합 평균 — 리턴은 항의 합과 같게
    std::map<std::string, double> rt;
    rt["coverage"] = 2.0 * area * covm;
    rt["time"] = -ep_len * DT / (area / 0.5) * 10.0;
    rt["slip"] = -0.1 * (1.1 - L) * (1 + 0.1 * nz(rng));
    rt["collision"] = -10.0 * cr;
    rt["declare"] = sr * 9.0 + dr * -4.5;
    double ret = 0;
    for (auto& kv : rt) { r.mean_w(rf::reward_key(kv.first), kv.second, w); ret += kv.second; }
    r.mean_w(rf::key::EP_REW_MEAN, ret, w);
    r.mean_w(rf::key::STEP_REWARD_MEAN, ret / ep_len, w);
    // PPO 건강
    double log_std = -0.0 - 1.1 * p + 0.02 * nz(rng);
    r.mean(rf::key::LOG_STD, log_std);
    r.mean(rf::key::STD, std::exp(log_std));
    r.mean(rf::std_key("v"), std::exp(log_std + 0.1));
    r.mean(rf::std_key("vy"), std::exp(log_std + 0.2));
    r.mean(rf::std_key("w"), std::exp(log_std - 0.1));
    r.mean(rf::key::ENTROPY_LOSS, -(3 * (0.5 * std::log(2 * M_PI * M_E) + log_std)));
    r.mean(rf::entropy_key("declare"), 0.01 + 0.05 * p);
    r.mean(rf::key::APPROX_KL, std::fabs(0.012 + 0.004 * nz(rng)) * (it > a.iters * 0.8 && a.seed == 2 ? 2.5 : 1));
    r.mean(rf::key::CLIP_FRACTION, clampd(0.12 + 0.02 * nz(rng), 0, 1));
    r.mean(rf::key::EXPLAINED_VAR, clampd(0.2 + 0.7 * L + 0.03 * nz(rng), -1, 1));
    r.mean(rf::key::VALUE_LOSS, (2.0 - 1.5 * L) * (1 + 0.1 * nz(rng)));
    r.mean(rf::key::PG_LOSS, -0.01 + 0.005 * nz(rng));
    r.mean(rf::key::LOSS, (1.0 - 0.75 * L) * (1 + 0.1 * nz(rng)));
    r.mean(rf::key::GRAD_NORM, std::fabs(0.6 + 0.15 * nz(rng)));
    r.mean(rf::key::LEARNING_RATE, 3e-4);
    r.mean(rf::key::ADV_STD, 1.0 + 0.05 * nz(rng));
    r.mean(rf::key::VALUE_MEAN, ret * 0.8);
    r.mean(rf::key::DECLARE_LOGIT, -7 + 3.5 * L);
    r.mean(rf::key::DECLARE_PROB, sigm(-7 + 3.5 * L));
    r.last(rf::key::CURRICULUM_STAGE, stage);
    r.mean_w("curriculum/success_rate/" + STAGES[stage], sr, w);
    r.mean(rf::key::GPU_MEM_MB, 6200 + 50 * nz(rng));
    r.sum(rf::dropped_key("eps"), 0);
    // 표본 판 줄(log_envs 개 환경의 끝난 판 전부): 기대값 log_envs × T / ep_len
    double lam = log_envs * T / ep_len;
    int k = std::poisson_distribution<int>(lam)(rng);
    for (int q = 0; q < k; q++) {
      EpResult e = sample_episode(rng, L, area);
      run.episode(episode_line(e, run.new_episode_id(), it, steps, int(u(rng) * log_envs), m.name, STAGES[stage]));
    }
    // 재생 판(본 줄기): 시뮬로 실제로 돌림
    if (a.replay_every > 0 && (it % a.replay_every == 0 || it == a.iters - 1))
      for (int q = 0; q < 2; q++) run_replay(L, it, steps, "", q);
    // 평가: 판 4 개를 시뮬로 돌려 s_eval 에 쓰고 표·eval/* 키
    if (a.eval_every > 0 && it > 0 && (it % a.eval_every == 0 || it == a.iters - 1)) {
      double n = 0, s_ = 0, c_ = 0, tr_ = 0, ds = 0, cv = 0, t95s = 0, t95n = 0, pl = 0, rew = 0, len = 0;
      for (int q = 0; q < 4; q++) {
        auto [e, bytes] = run_replay(L, it, steps, "eval", q);
        (void)bytes;
        n++; s_ += e.outcome == rf::outcome::SUCCESS; c_ += e.outcome == rf::outcome::COLLISION; tr_ += e.outcome == rf::outcome::TRUNCATED;
        ds += e.outcome == rf::outcome::DECLARED_SHORT; cv += e.coverage; pl += e.dist; rew += e.ret; len += e.steps;
        if (std::isfinite(e.t95)) { t95s += e.t95; t95n++; }
      }
      const double frontier_t95 = 70.0;  // 가짜 프론티어 기준선
      r.last(rf::key::EVAL_N, n);
      r.last(rf::key::EVAL_SUCCESS, s_ / n);
      r.last(rf::key::EVAL_COLLISION, c_ / n);
      r.last(rf::key::EVAL_TRUNCATION, tr_ / n);
      r.last(rf::key::EVAL_DECLARE_SHORT, ds / n);
      r.last(rf::key::EVAL_COVERAGE, cv / n);
      r.last(rf::key::EVAL_PATH_LEN, pl / n);
      r.last(rf::key::EVAL_REWARD, rew / n);
      r.last(rf::key::EVAL_LEN, len / n);
      r.last(rf::key::EVAL_T95_REACHED, t95n / n);
      if (t95n > 0) r.last(rf::key::EVAL_T95, t95s / t95n);
      r.last(rf::key::EVAL_FRONTIER_T95, frontier_t95);
      char nm[32];
      std::snprintf(nm, sizeof nm, "it%06d", it);
      rf::Json rows = rf::Json::array();
      rf::Json row = rf::Json::object();
      row.set("eval", "train_maps").set("split", "eval").set("n", n).set("success", s_ / n).set("collision", c_ / n).set("truncation", tr_ / n)
          .set("declare_short", ds / n).set("coverage", cv / n).set("ref_t95", frontier_t95).set("pass", s_ / n >= 0.75);
      if (t95n > 0) row.set("t95", t95s / t95n);
      rows.push(row);
      rf::Json ev = rf::Json::object();
      ev.set("iter", it).set("ckpt", std::string("ckpt_") + nm + ".bin").set("rows", rows);
      run.eval(nm, ev);
      run.event(std::string("eval ") + nm + ": success " + std::to_string(int(100 * s_ / n)) + " %");
    }
    if (it > 0 && it % 100 == 0) run.event("checkpoint ckpt_it" + std::to_string(it) + ".bin");
    run.end_update();
    if (a.live > 0) std::this_thread::sleep_for(std::chrono::duration<double>(a.live));
  }
  rf::Json extra = rf::Json::object();
  extra.set("final_iteration", a.iters - 1);
  run.finish(extra);
  std::cerr << "fake_run: done, " << run.rows_written() << " progress rows\n";
  return 0;
}
