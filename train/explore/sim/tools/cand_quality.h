// 위층 목표 후보 품질 재기 (low_explore_check --cand · low_explore_replay --cand 가 같이 씀) — 환경 hier 후보 8 개(옛)와 scripts/cand_view.h 후보(새)를 같은 잣대로.
// 한 결정마다 cand::build 바로 뒤에 부른다 (그 Work 의 갈 수 있는 칸·로봇 거리·막힘 거리 지도를 그대로 씀). 잣대:
//   길 있음      : 로봇 지도의 설 수 있는 칸(몸통 반경 close_radius + 깊이 층)으로 로봇에서 닿음 (계획기와 같은 칸)
//   간격         : 후보 칸에서 막힘(지도 막힘 + 깊이 층)까지 ≥ close_radius + margin
//   로봇 곁      : 후보 칸이 로봇에서 vp_min(0.75 m) 안 — 몸·뒤 사각지대 몫, 가자마자 "통과"
//   예상 넓이    : 광선(32 · gain_m)이 닿는 로봇 지도 모름 넓이 > 0
//   참 넓이      : (시뮬만) 같은 광선을 정답 격자로 — 정답에서 안 막힌 동안 로봇 지도 모름 칸 넓이. 예상 대비 비 = 실제로 보일 몫
//   방향 퍼짐    : 후보 방위가 차지한 45° 칸 수 (0 ~ 8)
//   정답 갈 수 있음: (시뮬만) 정답 격자 막힘을 몸통 반경만큼 키운 밖으로 로봇 참 자리에서 닿음 — 로봇 지도로는 갈 수 있어 보여도 실제로는 막힌 후보(빔 틈 너머)
//   0 후보       : 후보가 없는데 지도가 안 닫힘 (닫힘을 막는 프론티어가 남음)
//   정답 견줌    : (시뮬만) 로봇 지도와 정답 격자 둘 다로 갈 수 있는 칸 0.1 m 격자 + 두 무리 후보 중 로봇에서 vp_min 밖 전부에서 (정답으로 못 가는 후보는 0) 참 넓이 / (가는 시간 + t0) 최댓값 = 정답 최선. 후보 가운데 최선 · 0 번 후보의 같은 값 비
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cand_view.h"

namespace candq {

struct Fam {   // 한 후보 무리 (옛 · 새)
  long n = 0, reach = 0, clear = 0, near = 0, gain0 = 0, dec = 0, sectors = 0, none = 0, failed = 0, gt_reach = 0, gt_n = 0, gt_stand = 0, gt_lidar = 0;
  double pred = 0, truth = 0, best_ratio = 0, slot0_ratio = 0;
  long n_or = 0;
};
struct Acc {
  long dec = 0, open = 0;
  Fam f[2];   // 0 옛 hier · 1 새
};

struct Cell { int cell; int id; int anc; };

// 한 결정 재기. hier = 환경 후보 칸 K 개 (−1 빈 칸), gt = 정답 격자 (없으면 nullptr — 실물 녹화), closed = 지금 지도 닫힘
inline void eval(const cand::In& in, const cand::Opt& o, const cand::Work& w, const cand::Out& out, const int32_t* hier, const sim::World* gt, bool closed, Acc& acc,
                 const cand::Mem* mem = nullptr) {
  const sim::EnvCfg& c = *in.c;
  const int gw = in.W->gw, gh = in.W->gh;
  const mdp::MapView mv{in.L, c.size, gw, gh};
  const sim::EnvState& s = *in.s;
  const double res = c.res, rxf = (s.x + s.ex - in.W->ox) * c.inv_res, ryf = (s.y + s.ey - in.W->oy) * c.inv_res, th = s.th + s.eth;
  const int Rg = std::max(1, (int)std::lround(o.gain_m * c.inv_res)), step = std::max(1, mdp::hier::kGainRays / std::max(1, o.rays));
  const double area_k = (double)step * 2.0 * M_PI / mdp::hier::kGainRays * res * res;
  const cand::Work& W = w;
  const auto pass = [&](int idx) { return (W.fw.flag[idx] & 1) && W.fw.dist[idx] >= 0 && W.fw.cd[idx] >= 0; };
  // 예상 · 참 넓이: cand::Opt::coarse_m > 0 이면 굵은 칸(대충 앎으로 넘어가는 굵은 칸 수 × 넓이 — cand_view.h 와 같은 셈), 아니면 칸 모름 넓이
  const float qcoarse = o.coarse_m > 0.f ? o.coarse_m : 0.5f;   // 잣대는 늘 굵은 칸 (후보를 칸 단위로 만들어도 같은 잣대로 견줌)
  const int B = std::max(1, (int)std::lround(qcoarse * c.inv_res));
  const int bw = (gw + B - 1) / B, bh = (gh + B - 1) / B;
  static std::vector<int32_t> bkn, btot;
  static std::vector<double> bnew;
  static std::vector<uint32_t> seen;
  static uint32_t sid = 0;
  if (B) {
    bkn.assign((size_t)bw * bh, 0); btot.assign((size_t)bw * bh, 0); bnew.assign((size_t)bw * bh, 0.0);
    for (int y = 0; y < gh; ++y)
      for (int x = 0; x < gw; ++x) { const int q = (y / B) * bw + x / B; btot[q] += 1; bkn[q] += mdp::cls_at(mv, x, y) != mdp::kUnk; }
  }
  if ((int)seen.size() < gw * gh) seen.assign((size_t)gw * gh, 0);
  const double ray_k = (double)step * 2.0 * M_PI / mdp::hier::kGainRays;
  static const int16_t T[64] = {16364,16207,15893,15426,14811,14053,13160,12140,11003,9760,8423,7005,5520,3981,2404,804,-804,-2404,-3981,-5520,-7005,-8423,-9760,-11003,-12140,-13160,-14053,-14811,-15426,-15893,-16207,-16364,-16364,-16207,-15893,-15426,-14811,-14053,-13160,-12140,-11003,-9760,-8423,-7005,-5520,-3981,-2404,-804,804,2404,3981,5520,7005,8423,9760,11003,12140,13160,14053,14811,15426,15893,16207,16364};
  // 광선 하나 (gain_ray_walk 와 같은 표·같은 칸) — stop_gt 면 정답 막힘에서, 아니면 로봇 지도 막힘에서 멈춤
  const auto walk = [&](int vp, int a, bool stop_gt, const std::function<void(int, int, int)>& fn) {
    const int cx = vp % gw, cy = vp / gw, C = T[a], S = T[(a + 48) & 63];
    const auto rnd = [](int v) { v += 8192; return v >= 0 ? v >> 14 : -((-v + 16383) >> 14); };
    for (int t = 1; t <= Rg; ++t) {
      const int x = cx + rnd(t * C), y = cy + rnd(t * S);
      if (x < 0 || y < 0 || x >= gw || y >= gh) break;
      if (stop_gt ? (gt->grid[(size_t)y * gw + x] & sim::kCellBlocked) != 0 : mdp::cls_at(mv, x, y) == mdp::kOcc) break;
      if (mdp::cls_at(mv, x, y) == mdp::kUnk) fn(x, y, t);
    }
  };
  const auto area_of = [&](int vp, bool stop_gt) {
    ++sid;
    if (!B) {
      long g = 0;
      for (int a = 0; a < mdp::hier::kGainRays; a += step) walk(vp, a, stop_gt, [&](int, int, int t) { g += t; });
      return g * area_k;
    }
    std::vector<int> touch;
    for (int a = 0; a < mdp::hier::kGainRays; a += step)
      walk(vp, a, stop_gt, [&](int x, int y, int t) {
        const int j = y * gw + x;
        if (seen[j] == sid) return;
        seen[j] = sid;
        const int q = (y / B) * bw + x / B;
        if (bnew[q] == 0.0) touch.push_back(q);
        bnew[q] += t * ray_k;
      });
    long g = 0;
    for (int q : touch) {
      const double need = o.known_frac * btot[q];
      if (bkn[q] < need && bkn[q] + bnew[q] >= need) ++g;
      bnew[q] = 0.0;
    }
    return g * (double)B * B * res * res;
  };
  const auto pred = [&](int vp) { return area_of(vp, false); };
  const auto truth = [&](int vp) { return area_of(vp, true); };   // 정답 격자로 광선 — 정답 막힘에서 멈춤, 로봇 지도 모름 칸만
  const auto t_of = [&](int idx) { return W.fw.cd[idx] >= 0 ? W.fw.cd[idx] * 0.2 * res / o.v_eff : 1e6; };   // 정답 견줌용 (회전 뺌 — 두 무리 같은 잣대)
  // 정답 격자로 정말 갈 수 있나 (시뮬만): 정답 막힘을 몸통 반경(close_radius)만큼 키운 밖의 칸을 로봇 참 자리에서 4 이웃으로 번짐
  static std::vector<uint8_t> gst, gfl;
  static std::vector<int32_t> gq;
  static const sim::World* gst_w = nullptr;
  if (gt) {
    const int n = gw * gh;
    if (gst_w != gt || (int)gst.size() != n) {   // 설 수 있음 (장면마다 한 번)
      gst.assign(n, 0);
      const mdp::Robot rb = mdp::robot_cells(c.close_radius, c.res);
      for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x) {
          bool ok = !(gt->grid[(size_t)y * gw + x] & sim::kCellBlocked);
          for (int dy = -rb.r; dy <= rb.r && ok; ++dy)
            for (int dx = -rb.r; dx <= rb.r && ok; ++dx) {
              if (dx * dx + dy * dy > rb.r2) continue;
              const int u = x + dx, v = y + dy;
              if (u >= 0 && v >= 0 && u < gw && v < gh && (gt->grid[(size_t)v * gw + u] & sim::kCellBlocked)) ok = false;
            }
          gst[(size_t)y * gw + x] = ok;
        }
      gst_w = gt;
    }
    gfl.assign(n, 0);
    gq.resize(n);
    int tx = (int)std::floor((s.x - in.W->ox) * c.inv_res), ty = (int)std::floor((s.y - in.W->oy) * c.inv_res);
    int h = 0, t = 0;
    // 참 자리에서 가장 가까운 설 수 있는 칸 (몸통이 벽에 붙어 있으면)
    for (int r = 0; r <= 6 && t == 0; ++r)
      for (int dy = -r; dy <= r && t == 0; ++dy)
        for (int dx = -r; dx <= r && t == 0; ++dx) {
          const int u = tx + dx, v = ty + dy;
          if (u >= 0 && v >= 0 && u < gw && v < gh && gst[(size_t)v * gw + u]) { gq[t++] = v * gw + u; gfl[(size_t)v * gw + u] = 1; }
        }
    while (h < t) {
      const int idx = gq[h++], x = idx % gw, y = idx / gw;
      for (int k = 0; k < 4; ++k) {
        int dx, dy;
        mdp::nb4(k, &dx, &dy);
        const int u = x + dx, v = y + dy;
        if (u < 0 || v < 0 || u >= gw || v >= gh) continue;
        const int j = v * gw + u;
        if (gfl[j] || !gst[j]) continue;
        gfl[j] = 1;
        gq[t++] = j;
      }
    }
  }
  // 정답 최선 (시뮬만)
  double best = 0;
  int best_cell = -1;
  const char* dbg = std::getenv("CANDQ_DEBUG");
  const bool verbose = dbg && std::atol(dbg) == acc.dec;
  if (gt) {
    const auto try_cell = [&](int idx) {
      const int x = idx % gw, y = idx / gw;
      if (!pass(idx) || !gfl[idx] || std::hypot(x + 0.5 - rxf, y + 0.5 - ryf) * res < o.vp_min) return;   // 정답으로 갈 수 있는 칸만 · 로봇 곁(가자마자 통과)은 뺌
      const double sc = truth(idx) / (t_of(idx) + o.t0);
      if (sc > best) { best = sc; best_cell = idx; }
    };
    for (int y = 0; y < gh; y += 2)
      for (int x = 0; x < gw; x += 2) try_cell(y * gw + x);
    for (int k = 0; k < mdp::hier::K; ++k) if (hier[k] >= 0) try_cell(hier[k]);   // 두 무리 후보도 넣어 비가 1 을 안 넘게
    for (const cand::Cand& x : out.c) try_cell(x.vp);
  }
  acc.dec += 1;
  acc.open += !closed;
  for (int fi = 0; fi < 2; ++fi) {
    std::vector<int> cells;
    if (fi == 0) { for (int k = 0; k < mdp::hier::K; ++k) if (hier[k] >= 0) cells.push_back(hier[k]); }
    else for (const cand::Cand& x : out.c) cells.push_back(x.vp);
    Fam& F = acc.f[fi];
    F.dec += 1;
    if (cells.empty() && !closed) F.none += 1;
    uint8_t sec = 0;
    double bc = 0, s0 = 0;
    for (size_t i = 0; i < cells.size(); ++i) {
      const int cell = cells[i];
      F.n += 1;
      F.reach += pass(cell);
      F.clear += W.dto[cell] * res / 3.0 >= c.close_radius + o.margin;
      F.near += std::hypot(cell % gw + 0.5 - rxf, cell / gw + 0.5 - ryf) * res < o.vp_min;
      if (mem) {   // 실패 자리 fail_m 안 후보 (다시 나온 실패 자리)
        const double wx = in.W->ox + (cell % gw + 0.5) * res, wy = in.W->oy + (cell / gw + 0.5) * res;
        for (const auto& f : mem->fails) if (std::hypot(f.x - wx, f.y - wy) < in.fail_m) { F.failed += 1; break; }
      }
      const double p = pred(cell);
      F.gain0 += p <= 0;
      F.pred += p;
      const double b = std::atan2(cell / gw + 0.5 - ryf, cell % gw + 0.5 - rxf) - th, bw = std::atan2(std::sin(b), std::cos(b));
      sec |= (uint8_t)(1u << ((int)std::floor((bw + M_PI) / (M_PI / 4)) & 7));
      if (gt) {
        F.gt_n += 1;
        F.gt_reach += gfl[cell];
        if (!gfl[cell]) {   // 왜 못 가나: 시점 자리가 정답에서 설 수 없음(가구 밑·곁 — 라이다 아래 막힘) · 설 수는 있는데 끊긴 곳(틈 너머)
          F.gt_stand += gst[cell];
        }
        const double tr = truth(cell);
        F.truth += tr;
        const double sc = pass(cell) && gfl[cell] && std::hypot(cell % gw + 0.5 - rxf, cell / gw + 0.5 - ryf) * res >= o.vp_min ? tr / (t_of(cell) + o.t0) : 0.0;   // 로봇 곁은 0 (정답 쪽과 같은 잣대)
        bc = std::max(bc, sc);
        if (i == 0) s0 = sc;
        if (verbose) std::printf("    [%s %zu] 칸 (%d,%d) 로봇에서 %.2f m 길 %.2f m · 예상 %.2f · 참 %.2f m² · 참/시간 %.3f\n", fi ? "새" : "옛", i, cell % gw, cell / gw,
                                 std::hypot(cell % gw + 0.5 - rxf, cell / gw + 0.5 - ryf) * res, W.fw.cd[cell] * 0.2 * res, p, tr, sc);
      }
    }
    if (verbose && fi == 1 && best_cell >= 0)
      std::printf("    [정답 최선] 칸 (%d,%d) 로봇에서 %.2f m · 참 %.2f m² · 참/시간 %.3f · 막힘까지 %.2f m\n", best_cell % gw, best_cell / gw,
                  std::hypot(best_cell % gw + 0.5 - rxf, best_cell / gw + 0.5 - ryf) * res, truth(best_cell), best, W.dto[best_cell] * res / 3.0);
    for (int b = 0; b < 8; ++b) F.sectors += (sec >> b) & 1;
    if (std::getenv("CANDQ_TRACE") && gt && best > 0) std::printf("    결정 %ld %s 최선 비 %.3f 0번 %.3f (후보 %zu)\n", acc.dec, fi ? "새" : "옛", bc / best, s0 / best, cells.size());
    if (gt && best > 0 && !cells.empty()) { F.n_or += 1; F.best_ratio += std::min(1.0, bc / best); F.slot0_ratio += std::min(1.0, s0 / best); }
  }
}

inline void print(const Acc& a, const char* tag) {
  static const char* nm[2] = {"옛 hier 8", "새 cand"};
  std::printf("후보 품질 (%s, 결정 %ld · 안 닫힌 결정 %ld):\n", tag, a.dec, a.open);
  for (int i = 0; i < 2; ++i) {
    const Fam& F = a.f[i];
    const double n = F.n ? (double)F.n : 1.0;
    std::printf("  %-9s 결정당 %.2f 개 · 길 있음 %.3f · 간격 %.3f · 로봇 곁 %.3f · 예상 넓이 0 %.3f · 예상 넓이 평균 %.2f m²", nm[i], (double)F.n / std::max(1L, F.dec), F.reach / n, F.clear / n,
                F.near / n, F.gain0 / n, F.pred / n);
    if (F.gt_n) std::printf(" · 정답 지도로 갈 수 있음 %.3f (못 가는 것 중 자리는 설 수 있는데 끊김 %.3f)", (double)F.gt_reach / F.gt_n,
                            F.gt_n > F.gt_reach ? (double)F.gt_stand / (F.gt_n - F.gt_reach) : 0.0);
    if (F.n_or) std::printf(" · 참 넓이 평균 %.2f m² (예상의 %.2f) · 정답 최선 대비 — 후보 중 최선 %.3f · 0 번 %.3f", F.truth / n, F.pred > 0 ? F.truth / F.pred : 0.0, F.best_ratio / F.n_or, F.slot0_ratio / F.n_or);
    std::printf(" · 방향 45° 칸 %.2f · 후보 0 인데 안 닫힘 %ld · 실패 자리 1 m 안 %.3f\n", (double)F.sectors / std::max(1L, F.dec), F.none, F.failed / n);
  }
}

// 그림 재료: 로봇 지도 PPM + 후보 표 (옛 칸 · 새 시점·대표 칸·번호) — 그리기는 scratchpad 스크립트가 (글자·둘을 나란히)
inline void dump(const std::string& pre, const cand::In& in, const cand::Out& out, const int32_t* hier) {
  const sim::EnvCfg& c = *in.c;
  const int gw = in.W->gw, gh = in.W->gh;
  const mdp::MapView mv{in.L, c.size, gw, gh};
  std::vector<uint8_t> im((size_t)gw * gh * 3);
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) {
      const uint8_t k = mdp::cls_at(mv, x, y);
      uint8_t v = k == mdp::kUnk ? 150 : k == mdp::kOcc ? 0 : 255;
      uint8_t* q = &im[((size_t)(gh - 1 - y) * gw + x) * 3];
      q[0] = q[1] = q[2] = v;
      if (k == mdp::kFree && mdp::frontier_cell(mv, x, y)) { q[0] = 60; q[1] = 120; q[2] = 255; }
      if (in.cam && in.cam[y * c.size + x] && k != mdp::kOcc) { q[0] = 255; q[1] = 170; q[2] = 60; }
    }
  if (FILE* f = std::fopen((pre + ".ppm").c_str(), "wb")) {
    std::fprintf(f, "P6\n%d %d\n255\n", gw, gh);
    std::fwrite(im.data(), 1, im.size(), f);
    std::fclose(f);
  }
  if (FILE* f = std::fopen((pre + ".tsv").c_str(), "w")) {
    const sim::EnvState& s = *in.s;
    std::fprintf(f, "robot\t%.3f\t%.3f\t%.4f\t%d\n", (s.x + s.ex - in.W->ox) * c.inv_res, gh - (s.y + s.ey - in.W->oy) * c.inv_res, s.th + s.eth, gh);
    for (int k = 0; k < mdp::hier::K; ++k) if (hier[k] >= 0) std::fprintf(f, "old\t%d\t%d\t%d\n", k, hier[k] % gw, gh - 1 - hier[k] / gw);
    for (size_t i = 0; i < out.c.size(); ++i) {
      const cand::Cand& x = out.c[i];
      std::fprintf(f, "new\t%zu\t%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%d\n", i, x.id, x.vp % gw, gh - 1 - x.vp / gw, x.anc % gw, gh - 1 - x.anc / gw, x.gain_m2, x.t_s, x.kept);
    }
    std::fclose(f);
  }
}


// 단계별 그림 재료 (README "후보 뽑기") — build 와 같은 단계 함수를 차례로 부르며 중간 결과를 TSV 로. mem 은 복사본을 줌 (build 와 같은 시점의 기억)
//   frontier x y · cluster k x y · sample k x y tier why(0 통과 · 1 못 감·피함 · 2 로봇 곁 · 3 간격 · 4 안 보임) · best k x y score · rest k x y pocket
//   id k id age kept · pick order k x y marg score · cover order x y   (y 는 그림 위가 0)
inline void dump_stages(const std::string& pre, const cand::In& in, const cand::Opt& o, cand::Mem mem) {
  cand::Work w;
  cand::Out out;
  w.fit(in.W->gw * in.W->gh);
  cand::detail::Run R(in, o, mem, w, out);
  FILE* f = std::fopen((pre + ".tsv").c_str(), "w");
  if (!f) return;
  const int gw = R.gw, gh = R.gh;
  const auto P = [&](int idx) { return std::to_string(idx % gw) + "\t" + std::to_string(gh - 1 - idx / gw); };
  std::fprintf(f, "robot\t%.3f\t%.3f\t%.4f\t%d\n", R.rxf, gh - R.ryf, R.th, gh);
  if (!R.frontier()) { std::fclose(f); return; }
  R.dist_maps();
  for (int q = 0; q < R.nr0; ++q) { const int idx = R.fw.qa[q]; if (R.fw.flag[idx] & 2) std::fprintf(f, "frontier\t%s\n", P(idx).c_str()); }
  const std::vector<fu::Node> nodes = R.clusters();
  for (size_t k = 0; k < nodes.size(); ++k) for (int idx : nodes[k].cells) std::fprintf(f, "cluster\t%zu\t%s\n", k, P(idx).c_str());
  R.robot_dist();
  R.coarse_stats();
  std::vector<cand::detail::Run::Live> live;
  std::vector<uint8_t> used(mem.prev.size(), 0);
  for (int k = 0; k < (int)nodes.size(); ++k) {
    const fu::Node& nd = nodes[k];
    // 3 · 4: 뽑은 칸마다 거른 까닭 (첫 간격 단계 — best_view 와 같은 칸 차례)
    const double need = R.c.close_radius + o.margin;
    const auto why = [&](int cell) {
      if (cell < 0 || cell >= R.n || !R.pass(cell) || in.avoided(cell) || R.fw.dist[cell] < mdp::hier::kCandMinDist) return 1;
      if (std::hypot(cell % gw + 0.5 - R.rxf, cell / gw + 0.5 - R.ryf) * R.res < R.dmin) return 2;
      if (R.dto_m(cell) < need) return 3;
      if (cell != nd.anc && !R.los(cell, nd.anc)) return 4;
      return 0;
    };
    std::fprintf(f, "sample\t%d\t%s\t0\t%d\n", k, P(nd.anc).c_str(), why(nd.anc));
    for (int ri = 0; ri < 3; ++ri)
      for (int a = 0; a < o.ring_n; ++a) {
        const double rr = o.ring_m[ri], ang = 2 * M_PI * (a + 0.5 * ri) / o.ring_n;
        const int x = (int)std::floor(nd.cx + rr / R.res * std::cos(ang)), y = (int)std::floor(nd.cy + rr / R.res * std::sin(ang));
        if (x >= 0 && y >= 0 && x < gw && y < gh) std::fprintf(f, "sample\t%d\t%s\t0\t%d\n", k, P(y * gw + x).c_str(), why(y * gw + x));
      }
    cand::detail::Run::Live L;
    const bool pk = R.pocket(nd);
    if (pk || !R.best_view(k, nd, L)) { std::fprintf(f, "rest\t%d\t%s\t%d\n", k, P(nd.anc).c_str(), (int)pk); continue; }
    std::fprintf(f, "best\t%d\t%s\t%.4f\n", k, P(L.vp).c_str(), L.g / (L.t + o.t0));
    R.match(nd, L, used);
    std::fprintf(f, "id\t%d\t%d\t%d\t%d\t%s\n", k, L.id, L.age, L.kept, P(L.vp).c_str());
    live.push_back(L);
  }
  const auto picks = R.select(live);
  for (size_t i = 0; i < picks.size(); ++i) {
    const auto& L = live[picks[i].i];
    std::fprintf(f, "pick\t%zu\t%d\t%s\t%.3f\t%.4f\t%d\n", i, L.node, P(L.vp).c_str(), picks[i].m * R.area_k, picks[i].sc, L.id);
  }
  // 덮음: 고른 차례마다 그 시점 광선이 지난 모름 칸 (앞 후보가 덮은 칸은 앞 차례)
  std::vector<int> owner((size_t)R.n, -1);
  for (size_t i = 0; i < picks.size(); ++i) {
    const int vp = live[picks[i].i].vp;
    for (int a = 0; a < mdp::hier::kGainRays; a += R.step)
      mdp::gain_ray_walk(R.mv, vp % gw, vp / gw, a, R.Rg, [&](int x, int y, int, uint8_t cl) {
        if (cl != mdp::kUnk || owner[y * gw + x] >= 0) return;
        owner[y * gw + x] = (int)i;
        std::fprintf(f, "cover\t%zu\t%s\n", i, P(y * gw + x).c_str());
      });
  }
  std::fclose(f);
}

}  // namespace candq
