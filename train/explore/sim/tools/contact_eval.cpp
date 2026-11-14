// 닿음 센서 일치도·충돌 진단 (low_policy_run --trace 의 TSV 를 읽음, 호스트만). mdp/README "닿음 — 센서".
//   contact_eval 자취.tsv [자취.tsv …] [--pre 1.0] [--post 2.0] [--merge 2.0]
// 판마다 정답 충돌(시뮬 몸통 겹침 gt_touch) 첫 걸음 t_gt 와 센서 닿음 첫 걸음 t_det 를 견준다 (SIM_CON_END=0 으로 돌린 자취 — 닿아도 판이 안 끝나 둘 다 보임):
//   잡음   : t_gt − pre ≤ t_det ≤ t_gt + post  (늦음 = t_det − t_gt, 음수면 미리)
//   놓침   : 정답 충돌은 있는데 그 창 안에 센서 닿음이 없음
//   헛닿음 : t_det < t_gt − pre (또는 정답 충돌 없는 판) — 학습에서 닿지 않았는데 판이 끝났을 걸음. 100 m 당 = 판 첫 닿음까지 간 거리로 나눔
// 센서는 따로·묶음 둘 다: 라이다(s_lid < 문턱) · 깊이 층(s_cam < 문턱) · 멈춤(s_stall ≥ 걸음) · 지도(s_map < 문턱), 문턱 여러 벌.
// 정답 첫 충돌마다(= 오늘 판 끝 충돌과 같은 걸음 — 그 앞 궤적은 판 끝 규칙과 무관) 진단: 닿은 곳(앞·옆·뒤, 정답 격자 가장 가까운 칸 몸통 좌표),
// 움직임(직진·제자리 회전·멈춤), 속도·명령, 안전층이 고쳤나, 그 앞 2 s 동안 라이다·깊이 층 최소 간격(센서가 봤나), 사람.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Row {
  int t = 0, end = 0;
  float x, y, th, v, w, a_v, a_w, cmd_v, cmd_w, wv, gyro, s_lid, s_map, s_cam, s_pgap, gt_gap, gt_u, gt_v, gt_pgap, goal_d;
  int safe, s_stall, s_kind, contact, gt_touch, gt_person;
};
struct Ep {
  std::string scene;
  int people = 0;
  std::vector<Row> r;
  int end = 0;
};

constexpr float kDt = 0.1f, kHalfLen = 0.16f, kHalfWid = 0.11f;   // 걸음 (s), 몸통 반 길이·반 폭 (env.yaml body.footprint 0.32 × 0.22)

// 판의 첫 센서 닿음 걸음 (없으면 −1): 문턱으로 다시 정한 검출기
struct Det {
  std::string name;
  float lid = -1.f, cam = -1.f, map = -1.f;   // 음수 = 안 씀
  int stall = 0;                              // 0 = 안 씀
  bool fire(const Row& q) const {
    return (lid >= 0.f && q.s_lid < lid) || (cam >= 0.f && q.s_cam < cam) || (map >= 0.f && q.s_map < map) || (stall > 0 && q.s_stall >= stall);
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> files;
  float pre = 1.0f, post = 2.0f;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--pre" && i + 1 < argc) pre = std::strtof(argv[++i], nullptr);
    else if (k == "--post" && i + 1 < argc) post = std::strtof(argv[++i], nullptr);
    else files.push_back(k);
  }
  if (files.empty()) { std::fprintf(stderr, "contact_eval 자취.tsv [...] [--pre 1.0] [--post 2.0]\n"); return 2; }
  std::vector<Ep> eps;
  for (const auto& fn : files) {
    std::ifstream in(fn);
    if (!in) { std::fprintf(stderr, "못 엶: %s\n", fn.c_str()); return 2; }
    std::string line;
    std::getline(in, line);   // 머리
    std::map<std::tuple<std::string, int, int, int>, int> open;   // (장면, 사람, 환경, 판) → eps 번호
    while (std::getline(in, line)) {
      std::stringstream ss(line);
      std::string scene;
      int people, env, ep, t, end;
      std::getline(ss, scene, '\t');
      ss >> people >> env >> ep >> t >> end;
      const auto key = std::make_tuple(scene, people, env, ep);
      auto it = open.find(key);
      if (it == open.end()) {
        it = open.emplace(key, (int)eps.size()).first;
        eps.push_back(Ep{scene, people, {}, 0});
      }
      Ep& E = eps[it->second];
      if (end != 0 || ss.eof()) { E.end = end; continue; }
      Row q;
      q.t = t;
      ss >> q.x >> q.y >> q.th >> q.v >> q.w >> q.a_v >> q.a_w >> q.cmd_v >> q.cmd_w >> q.safe >> q.wv >> q.gyro >> q.s_lid >> q.s_map >> q.s_cam >> q.s_pgap >>
          q.s_stall >> q.s_kind >> q.contact >> q.gt_touch >> q.gt_person >> q.gt_gap >> q.gt_u >> q.gt_v >> q.gt_pgap >> q.goal_d;
      if (!ss) continue;
      E.r.push_back(q);
    }
  }
  const int npre = (int)std::lround(pre / kDt), npost = (int)std::lround(post / kDt);
  // 판마다 정답 첫 충돌 걸음 (자취 안 번호), 그때까지 간 거리
  struct EpRes { int i_gt = -1; double dist_all = 0; std::vector<double> cum; bool skip = false; };
  int n_skip = 0;
  std::vector<EpRes> er(eps.size());
  long n_steps = 0;
  double dist_total = 0;
  for (size_t k = 0; k < eps.size(); ++k) {
    const Ep& E = eps[k];
    double d = 0;
    er[k].cum.resize(E.r.size());
    for (size_t j = 0; j < E.r.size(); ++j) {
      if (j > 0) d += std::hypot(E.r[j].x - E.r[j - 1].x, E.r[j].y - E.r[j - 1].y);
      er[k].cum[j] = d;
      if (er[k].i_gt < 0 && E.r[j].gt_touch) er[k].i_gt = (int)j;
    }
    // 예열(규칙 추종기)에서 이미 닿아 있던 판은 뺀다: 정책 첫 걸음에 정답 충돌이 있거나 멈춤 시계가 차 있음 (SIM_CON_END=0 이라 예열 충돌이 판을 안 끝냄)
    if (!E.r.empty() && (E.r[0].gt_touch || E.r[0].s_stall > 1)) { er[k].skip = true; ++n_skip; continue; }
    er[k].dist_all = d;
    n_steps += (long)E.r.size();
    dist_total += d;
  }
  int n_gt = 0, n_gt_p = 0;
  for (auto& x : er) n_gt += !x.skip && x.i_gt >= 0;
  for (size_t k = 0; k < eps.size(); ++k) n_gt_p += !er[k].skip && er[k].i_gt >= 0 && eps[k].r[(size_t)er[k].i_gt].gt_person;
  std::printf("자취 %zu 개 · 판 %zu (예열에서 이미 닿아 뺀 판 %d) · 정책 걸음 %ld · 거리 %.1f m · 정답 첫 충돌 판 %d (사람 %d) — 창: 앞 %.1f s · 뒤 %.1f s\n", files.size(),
              eps.size(), n_skip, n_steps, dist_total, n_gt, n_gt_p, pre, post);

  // ---- 1. 검출기별 일치도 ----
  std::vector<Det> dets;
  for (float th : {0.0f, 0.01f, 0.02f, 0.03f, 0.05f}) { Det d; d.lid = th; char b[64]; std::snprintf(b, sizeof b, "라이다 < %.2f", th); d.name = b; dets.push_back(d); }
  for (float th : {0.0f, 0.02f, 0.05f}) { Det d; d.cam = th; char b[64]; std::snprintf(b, sizeof b, "깊이층 < %.2f", th); d.name = b; dets.push_back(d); }
  for (float th : {-0.02f + 1e-6f, 0.0f, 0.02f}) { Det d; d.map = th; char b[64]; std::snprintf(b, sizeof b, "지도 < %.2f", th); d.name = b; dets.push_back(d); }
  for (int n : {2, 3, 5}) { Det d; d.stall = n; d.name = "멈춤 ≥ " + std::to_string(n) + " 걸음"; dets.push_back(d); }
  { Det d; d.lid = 0.02f; d.cam = 0.02f; d.stall = 3; d.name = "묶음: 라이다 0.02 | 깊이 0.02 | 멈춤 3 (처음 잡은 값)"; dets.push_back(d); }
  { Det d; d.lid = 0.02f; d.cam = 0.02f; d.stall = 3; d.map = 0.0f; d.name = "묶음 + 지도 0.00"; dets.push_back(d); }
  { Det d; d.lid = 0.03f; d.cam = 0.02f; d.stall = 2; d.name = "묶음: 라이다 0.03 | 깊이 0.02 | 멈춤 2"; dets.push_back(d); }
  { Det d; d.lid = 0.01f; d.cam = 0.0f; d.stall = 3; d.name = "묶음: 라이다 0.01 | 깊이 0.00 | 멈춤 3"; dets.push_back(d); }
  { Det d; d.lid = 0.02f; d.stall = 2; d.name = "묶음: 라이다 0.02 | 멈춤 2"; dets.push_back(d); }
  { Det d; d.lid = 0.05f; d.cam = -0.02f; d.stall = 2; d.name = "묶음: 라이다 0.05 | 깊이 −0.02 | 멈춤 2"; dets.push_back(d); }
  { Det d; d.lid = 0.03f; d.cam = -0.02f; d.stall = 2; d.name = "묶음: 라이다 0.03 | 깊이 −0.02 | 멈춤 2"; dets.push_back(d); }
  { Det d; d.lid = 0.02f; d.cam = 0.0f; d.stall = 2; d.name = "묶음: 라이다 0.02 | 깊이 0.00 | 멈춤 2"; dets.push_back(d); }
  { Det d; d.lid = 0.02f; d.cam = -0.02f; d.stall = 2; d.name = "묶음: 라이다 0.02 | 깊이 −0.02 | 멈춤 2"; dets.push_back(d); }
  { Det d; d.lid = 0.01f; d.cam = -0.02f; d.stall = 2; d.name = "묶음: 라이다 0.01 | 깊이 −0.02 | 멈춤 2 (채택)"; dets.push_back(d); }
  std::printf("\n헛닿음 중 \"먼\" = 그 걸음 정답 간격(몸통~정답 격자 칸, 사람 포함) > 3 cm — 그 아래는 스치기 직전이라 일찍 끝낸 셈\n");
  std::printf("| 검출기 | 잡음 / 정답 충돌 판 | 놓침 | 늦음 중앙 (s) | 늦음 최대 (s) | 헛닿음 판 (먼) | 헛닿음 /100 m (먼) |\n|---|---|---|---|---|---|---|\n");
  for (const Det& D : dets) {
    int hit = 0, miss = 0, fp = 0, fp_far = 0;
    double fp_dist = 0;
    std::vector<float> delay;
    for (size_t k = 0; k < eps.size(); ++k) {
      if (er[k].skip) continue;
      const Ep& E = eps[k];
      int i_det = -1;
      for (size_t j = 0; j < E.r.size() && i_det < 0; ++j) if (D.fire(E.r[j])) i_det = (int)j;
      const int ig = er[k].i_gt;
      // 이 검출기로 판이 끝났을 걸음까지 간 거리 (헛닿음 분모) — 정답 충돌·센서 닿음 중 이른 것, 둘 다 없으면 판 끝
      int i_end = (int)E.r.size() - 1;
      if (i_det >= 0) i_end = i_det;
      if (ig >= 0 && ig < i_end) i_end = ig;
      if (i_end >= 0) fp_dist += er[k].cum[(size_t)i_end];
      if (i_det >= 0 && (ig < 0 || E.r[(size_t)i_det].t < E.r[(size_t)ig].t - npre)) {
        ++fp;
        const Row& q = E.r[(size_t)i_det];
        fp_far += std::fmin(q.gt_gap - 0.025f, q.gt_pgap) > 0.03f;
        continue;
      }
      if (ig < 0) continue;
      if (i_det >= 0 && E.r[(size_t)i_det].t <= E.r[(size_t)ig].t + npost) { ++hit; delay.push_back((float)(E.r[(size_t)i_det].t - E.r[(size_t)ig].t) * kDt); }
      else ++miss;
    }
    std::sort(delay.begin(), delay.end());
    const float med = delay.empty() ? NAN : delay[delay.size() / 2], mx = delay.empty() ? NAN : delay.back();
    std::printf("| %s | %d / %d | %d | %+.1f | %+.1f | %d (%d) | %.3f (%.3f) |\n", D.name.c_str(), hit, n_gt, miss, med, mx, fp, fp_far, fp_dist > 0 ? 100.0 * fp / fp_dist : 0.0,
                fp_dist > 0 ? 100.0 * fp_far / fp_dist : 0.0);
  }

  // ---- 2. 정답 첫 충돌 진단 ----
  std::printf("\n정답 첫 충돌마다 (진단): 닿은 곳 = 정답 격자 가장 가까운 칸의 몸통 좌표 (앞 u > %.2f · 뒤 u < −%.2f · 그 밖 옆), 사람이면 사람\n", kHalfLen - 0.03f, kHalfLen - 0.03f);
  std::printf("| 장면 | 사람 | 판 시각 (s) | 닿은 곳 | 움직임 | v · ω (참) | 명령 v · ω | 정책 v · ω | 추종기 앞 정지 (2 s 안 걸음 — 정책 걸음엔 0) | 앞 2 s 라이다 최소 | 앞 2 s 깊이층 최소 | 닿은 걸음 라이다 · 깊이 · 멈춤 | 기본 묶음 늦음 (s) | 목표까지 (m) |\n");
  std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
  int cat[6] = {};   // 0 앞 1 옆 2 뒤 3 사람
  int mov[3] = {};   // 0 직진 1 제자리 회전 2 멈춤
  int seen_lid = 0, seen_cam_only = 0, seen_none = 0, by_safe = 0;
  const Det base = dets.back();   // 채택 (env_cfg.cpp 기본): 라이다 0.01 | 깊이 −0.02 | 멈춤 2
  for (size_t k = 0; k < eps.size(); ++k) {
    const int ig = er[k].i_gt;
    if (ig < 0 || er[k].skip) continue;
    const Ep& E = eps[k];
    const Row& q = E.r[(size_t)ig];
    const char* where;
    int ci;
    if (q.gt_person) { where = "사람"; ci = 3; }
    else if (q.gt_u > kHalfLen - 0.03f) { where = "앞"; ci = 0; }
    else if (q.gt_u < -(kHalfLen - 0.03f)) { where = "뒤"; ci = 2; }
    else { where = q.gt_v > 0 ? "옆(왼)" : "옆(오른)"; ci = 1; }
    cat[ci] += 1;
    const char* m;
    int mi;
    if (std::fabs(q.cmd_v) >= 0.05f) { m = "직진·곡선"; mi = 0; }
    else if (std::fabs(q.cmd_w) >= 0.2f) { m = "제자리 회전"; mi = 1; }
    else { m = "멈춤"; mi = 2; }
    mov[mi] += 1;
    float lmin = 9.f, cmin = 9.f;
    int safe_n = 0;
    for (int j = ig - 1; j >= 0 && E.r[(size_t)j].t >= q.t - 20; --j) {
      lmin = std::fmin(lmin, E.r[(size_t)j].s_lid);
      cmin = std::fmin(cmin, E.r[(size_t)j].s_cam);
      safe_n += E.r[(size_t)j].safe;
    }
    if (lmin < 0.05f || q.s_lid < 0.05f) ++seen_lid;
    else if (cmin < 0.05f || q.s_cam < 0.05f) ++seen_cam_only;
    else ++seen_none;
    by_safe += safe_n > 0;
    int i_det = -1;
    for (size_t j = 0; j < E.r.size() && i_det < 0; ++j) if (base.fire(E.r[j])) i_det = (int)j;
    char dl[32];
    if (i_det >= 0) std::snprintf(dl, sizeof dl, "%+.1f", (float)(E.r[(size_t)i_det].t - q.t) * kDt);
    else std::snprintf(dl, sizeof dl, "놓침");
    std::printf("| %s | %d | %.1f | %s | %s | %.2f · %.2f | %.2f · %.2f | %.2f · %.2f | %d | %.3f | %.3f | %.3f · %.3f · %d | %s | %.1f |\n", E.scene.c_str(), E.people,
                q.t * kDt, where, m, q.v, q.w, q.cmd_v, q.cmd_w, q.a_v, q.a_w, safe_n, lmin, cmin, q.s_lid, q.s_cam, q.s_stall, dl, q.goal_d);
  }
  std::printf("\n요약: 닿은 곳 앞 %d · 옆 %d · 뒤 %d · 사람 %d | 움직임 직진·곡선 %d · 제자리 회전 %d · 멈춤 %d | 앞 2 s 라이다가 5 cm 안을 봄 %d · 깊이층만 %d · 둘 다 못 봄 %d | "
              "앞 2 s 추종기 앞 정지 걸음 있음 %d\n",
              cat[0], cat[1], cat[2], cat[3], mov[0], mov[1], mov[2], seen_lid, seen_cam_only, seen_none, by_safe);
  // 낮은 물체 노출: 깊이 층이 라이다보다 5 cm 넘게 가까운 걸음 비율 (라이다 면 아래 물체가 몸통 0.25 m 안)
  long low = 0, n = 0;
  for (size_t k = 0; k < eps.size(); ++k) {
    if (er[k].skip) continue;
    const Ep& E = eps[k];
    for (const Row& q : E.r) { ++n; low += q.s_cam < 0.25f && q.s_cam + 0.05f < q.s_lid; }
  }
  std::printf("낮은 물체 노출: 깊이 층 간격 < 0.25 m 이고 라이다보다 5 cm 넘게 가까운 걸음 %.2f %% (%ld / %ld)\n", n ? 100.0 * low / n : 0.0, low, n);

  // ---- 3. 아래층 보상 항: 옛(정답) vs 새(센서) 같은 걸음에서 (low_rl.h lowrl_finish — 상수는 그 파일 기본값) ----
  // 근접 = kLrProxK × (1 − d/0.2) [d < 0.2] + lr_mprox_k × (|v| + 0.194 |ω|) × (1 − d/0.2) [d < 0.2]: 옛 d = min(정답 사람 간격, 정답 격자 간격 − 반 칸),
  // v·ω 는 참 몸통 · 새 d = 센서 간격(min 라이다·지도·깊이 층), v·ω 는 바퀴 오도메트리·자이로. 충돌 = −10: 옛 정답 겹침 · 새 센서 닿음(기본 묶음)
  {
    constexpr float kProxD = 0.2f, kProxK = -0.05f, kMproxK = -0.3f, kRc = 0.194f, kColl = -10.f;
    auto prox = [&](float d, float v, float w) { return d < kProxD ? (kProxK + kMproxK * (std::fabs(v) + kRc * std::fabs(w))) * (1.f - std::fmax(d, 0.f) / kProxD) : 0.f; };
    double so = 0, sn = 0, soo = 0, snn = 0, son = 0, mad = 0, co = 0, cn = 0, cmad = 0;
    long m = 0;
    for (size_t k = 0; k < eps.size(); ++k) {
      if (er[k].skip) continue;
      for (const Row& q : eps[k].r) {
        const float d_old = std::fmin(q.gt_pgap, q.gt_gap - 0.025f);
        const float d_new = std::fmin(std::fmin(q.s_lid, q.s_map), q.s_cam);
        const double o = prox(d_old, q.v, q.w), nn = prox(d_new, q.wv, q.gyro);
        so += o; sn += nn; soo += o * o; snn += nn * nn; son += o * nn; mad += std::fabs(o - nn);
        const double c0 = q.gt_touch ? kColl : 0.0, c1 = base.fire(q) ? kColl : 0.0;
        co += c0; cn += c1; cmad += std::fabs(c0 - c1);
        ++m;
      }
    }
    if (m > 0) {
      const double cov = son / m - (so / m) * (sn / m), vo = soo / m - (so / m) * (so / m), vn = snn / m - (sn / m) * (sn / m);
      std::printf("\n아래층 보상 항, 옛(정답) vs 새(센서) — 같은 %ld 걸음:\n", m);
      std::printf("  근접(proximity): 걸음 평균 옛 %+.5f · 새 %+.5f, 상관 %.3f, |차| 평균 %.5f\n", so / m, sn / m, vo > 0 && vn > 0 ? cov / std::sqrt(vo * vn) : NAN, mad / m);
      std::printf("  충돌(collision): 걸음 평균 옛 %+.5f · 새 %+.5f, |차| 평균 %.5f (닿아도 판이 안 끝나는 자취라 닿은 뒤 걸음도 셈)\n", co / m, cn / m, cmad / m);
    }
  }
  return 0;
}
