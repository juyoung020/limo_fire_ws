// dec_rollout — 판단 모델 위층(upper:=decider, 아래층 lower:=rule)의 결과 기반 라벨을 만드는 갈래 굴리기 (upper_lora/README "결과 라벨").
// 실물 노드와 같은 위층 코드(scripts/low_explore_ctl.h Ctl · scripts/low_decider.h Upper)를 CPU 시뮬 환경 하나로 돌린다 (low_explore_check 의 시뮬 쪽과 같은 차례).
//   기준 판: 판단 모델 자리에 "기준 응답기"(--base slot0: 늘 첫 글자 A = 피하지 않는 첫 후보)를 두고, 요청은 주기(--period)마다, 답은 지연(--latency) 뒤 걸음에.
//   갈래: 기준 판에서 보낸 요청 r 가운데 고른 것(--every)마다, 그 요청의 답을 글자 k 로 바꾼 판을 처음부터 다시 돌린다 (같은 씨앗이라 r 까지 같음).
//         r 을 보낸 걸음에 환경 난수를 섞어(--reps 번, 0 번은 그대로) 그 뒤 우연만 다르게. 그 뒤는 기준 응답기 그대로. 판 끝 또는 --horizon 초까지.
// 출력 (--out 앞머리):
//   <앞머리>.jsonl  요청 r 마다 한 줄: 장면 · 씨앗 · 요청 번호 · 걸음 · 자세 · 후보(글자 차례: 슬롯 · 칸 · 특징 F[FEAT]) · 기준 판 결과 · 갈래 결과 [글자][반복] = {끝 코드, 끝 시각 s, 아는 칸 수}
//   <앞머리>_r<번호>.req.json  그 요청 본문 (지금 꼴 — decider_input both: 지도 사진 + 후보 글)
//   <앞머리>_r<번호>.map.bin   원시 지도 (아는 칸 범위만: i32 x0 y0 w h · i16 [h][w]) — 후보 꼴이 바뀌면 이것과 jsonl 로 요청을 다시 그림
//   dec_rollout <장면.scene> --seed 1 --out p [--people 0] [--period 3.0] [--latency 2.6] [--every 3] [--reps 4] [--horizon 600] [--max-req 0] [--base slot0]
#include <algorithm>
#include <array>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "env_api.h"
#include "low_decider.h"
#include "low_explore_ctl.h"

namespace {
uint64_t mix(uint64_t x) {   // splitmix64 (combine_run 과 같음)
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
// 기준 응답기: 글자 A (slot0 차례). 요청 forced 의 답만 글자 letter. lat 걸음 뒤
struct BaseModel : lxd::Transport {
  int lat = 26, now = 0;
  long forced = -1;
  int letter = 0;
  struct P { int ready; lxd::Answer a; };
  std::vector<P> q;
  std::vector<lxd::Req> sent;
  void send(const lxd::Req& r) override {
    lxd::Answer a;
    a.id = r.id;
    a.ok = true;
    a.choice = std::string(1, (char)('A' + (r.id == forced ? letter : 0)));
    q.push_back({now + lat, a});
    sent.push_back(r);
  }
  bool poll(lxd::Answer* a) override {
    if (q.empty() || q.front().ready > now) return false;
    *a = q.front().a;
    q.erase(q.begin());
    return true;
  }
};
using clk = std::chrono::steady_clock;
double g_tm[6] = {0, 0, 0, 0, 0, 0};   // 걸음 안 시간 (s): 꺼내기 · 위층(요청 그리기 포함) · 준비(계획) · 걸음(센서·지도) · 걸음 뒤 · 걸음 수
inline double since(clk::time_point& a) { const auto b = clk::now(); const double d = std::chrono::duration<double>(b - a).count(); a = b; return d; }
struct Out {
  int done = 0;
  double t = 0;
  long known = 0;
};
struct Snap {   // 요청을 보낸 때의 원시 상태 (후보 꼴이 바뀌면 이것으로 요청을 다시 그림)
  long id;
  int step;
  float x, y, th;
  std::vector<cand::Cand> cands;   // 글자 차례 (Ctl::cands — 기억 필드 포함)
  std::vector<int> cell;           // 글자 차례 서는 자리 (vp)
  cand::Mem mem;                   // 판 기억 (지나온 길 trail · 목표 goals · 실패 자리 fails)
  std::string body;                // 지금 꼴 요청 (decider_input both)
  int x0, y0, w, h;
  std::vector<int16_t> map;        // 지금 지도 (아는 칸 범위)
  std::vector<int> past_age;       // 지난 지도의 걸음 차 (같은 범위로 자름 — 그림 꼴 B)
  std::vector<std::vector<int16_t>> past;
  std::vector<std::array<float, 4>> kf;
  std::vector<lxd::Dec> decs;   // 판 동안 위층 결정 전부 (렌더 때 결정 기록 K 를 바꿀 수 있게)   // 판 시작부터 10 걸음마다 (걸음 · 지도 x · y · th) — 시뮬은 장면으로 그때 카메라를 다시 그림 (후보별 지난 장면)
};
struct Cfg {
  sim::EnvHostCfg hc;
  std::vector<sim::World> worlds;
  sim::LidarModel lidar;
  uint64_t seed = 1;
  int people = 0;
  lxc::Params lp;
  lxd::Params dp;
  int lat = 26;
};
using Hist = std::deque<std::pair<int, std::vector<int16_t>>>;   // (걸음, 지도) — 10 걸음마다, 최근 31 개 (30 s)
void hist_push(Hist& h, int step, const std::vector<int16_t>& lm) {
  if (step % 10 != 0) return;
  h.emplace_back(step, lm);
  if (h.size() > 31) h.pop_front();
}
Snap make_snap(const Cfg& C, const sim::World& w, const sim::EnvState& e, const std::vector<int16_t>& lm, const lxc::Ctl& cs, const lxd::Req& r, const Hist& hist,
               const std::vector<std::array<float, 4>>& kf, const lxd::Upper& us) {
  Snap s;
  s.kf = kf;
  s.decs = us.decisions();
  s.id = r.id; s.step = e.step; s.x = e.x + e.ex; s.y = e.y + e.ey; s.th = e.th + e.eth;
  s.body = r.body;
  s.cands = cs.cands().c;
  s.cell = r.cells;
  s.mem = cs.cand_mem();
  const int S = C.hc.cfg.size;
  int x0 = 1 << 30, y0 = 1 << 30, x1 = -1, y1 = -1;
  for (int y = 0; y < w.gh; ++y)
    for (int x = 0; x < w.gw; ++x)
      if (lm[(size_t)y * S + x] != sim::kUnknown) { x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y); }
  if (x1 < 0) { x0 = y0 = 0; x1 = y1 = -1; }
  s.x0 = x0; s.y0 = y0; s.w = x1 - x0 + 1; s.h = y1 - y0 + 1;
  const auto crop = [&](const std::vector<int16_t>& m) {
    std::vector<int16_t> o;
    o.reserve((size_t)s.w * s.h);
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) o.push_back(m[(size_t)y * S + x]);
    return o;
  };
  s.map = crop(lm);
  for (int ago : {50, 150, 300}) {   // −5 · −15 · −30 s 쯤 (10 걸음 칸에서 가장 가까운 것)
    const std::pair<int, std::vector<int16_t>>* best = nullptr;
    for (const auto& q : hist) if (q.first <= e.step - ago && (!best || q.first > best->first)) best = &q;
    if (!best && !hist.empty()) best = &hist.front();
    if (!best) continue;
    s.past_age.push_back(e.step - best->first);
    s.past.push_back(crop(best->second));
  }
  return s;
}
long count_known(const std::vector<int16_t>& lm) {
  long n = 0;
  for (int16_t v : lm) n += v != sim::kUnknown;
  return n;
}
// 판 하나. forced ≥ 0 이면 그 요청 답 = letter, 그 요청을 보낸 걸음에 reseed(> 0)로 난수 섞음. horizon_steps (> 0) 는 그 요청 뒤 걸음 한도.
// snaps 를 주면 요청마다 원시 상태를 담음
Out run(const Cfg& C, long forced, int letter, uint64_t reseed, int horizon_steps, std::vector<Snap>* snaps) {
  sim::Env cpu(C.hc, C.worlds, C.lidar, 1, C.seed, false);
  if (C.people > 0) cpu.set_people_max(std::vector<int>(1, C.people));
  cpu.reset();
  std::vector<sim::EnvState> st;
  lxc::Ctl cs(C.lp);
  lxd::Upper us(C.dp);
  BaseModel bm;
  bm.lat = C.lat;
  bm.forced = forced;
  bm.letter = letter;
  cpu.get_state(st);
  const sim::World& w = C.worlds[(size_t)st[0].map];
  cs.set_geom(w.gw, C.hc.cfg.res);
  std::vector<int16_t> lm;
  std::vector<uint32_t> tr;
  std::vector<uint8_t> vox, img;
  std::vector<float> pts, o1d, rew;
  std::vector<int32_t> cand, ccur, dn;
  std::vector<float> act((size_t)mdp::act::DIM, 0.f);
  const double dt = (double)C.hc.cfg.substeps * C.hc.cfg.dt;
  int branch_step = -1;
  Hist hist;
  std::vector<std::array<float, 4>> kfp;
  Out o;
  for (int t = 0;; ++t) {
    auto tp = clk::now();
    cpu.get_state(st);
    bm.now = t;
    cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts); cpu.get_cand(cand); cpu.get_cand_cur(ccur); cpu.get_obs(img, o1d);
    if (snaps) { hist_push(hist, st[0].step, lm); if (st[0].step % 10 == 0) kfp.push_back({(float)st[0].step, st[0].x + st[0].ex, st[0].y + st[0].ey, st[0].th + st[0].eth}); }
    const lxc::View v{&C.hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
    g_tm[0] += since(tp);
    const size_t ns = bm.sent.size();
    sim::LowCmd c1;
    if (us.step(v, o1d.data() + mdp::obs::POLICY_1D, cs, bm, &c1)) cpu.set_low_cmd(std::vector<sim::LowCmd>{c1});
    if (bm.sent.size() > ns) {   // 이 걸음에 요청을 보냄
      const lxd::Req& r = bm.sent.back();
      if (snaps) snaps->push_back(make_snap(C, w, st[0], lm, cs, r, hist, kfp, us));
      if (r.id == forced) {
        branch_step = t;
        if (reseed > 0) {   // 그 뒤 우연만 다르게 (combine_run --branch-reseed 와 같은 섞기)
          cpu.get_state(st);
          st[0].rng = mix(st[0].rng ^ mix(reseed * 0x9e3779b97f4a7c15ull));
          cpu.set_state(0, st[0]);
        }
      }
    }
    g_tm[1] += since(tp);
    std::fill(act.begin(), act.end(), 0.f);
    act[0] = (float)cs.slot();   // 아래층 규칙 추종기: 결정 때 정한 행동 칸 (low_explore_check 와 같음)
    cpu.low_prepare(act.data());
    g_tm[2] += since(tp);
    cpu.step(act.data());
    g_tm[3] += since(tp);
    g_tm[5] += 1;
    cpu.get_reward(rew, dn);
    cpu.get_state(st);
    if (dn[0] != 0) {
      std::vector<sim::EpInfo> ep;
      cpu.get_ep(ep);
      o.done = dn[0];
      o.t = ep[0].t;
      cpu.get_map(lm);   // 리셋 뒤 지도일 수 있어 아는 칸은 직전 값을 씀
      return o;
    }
    std::vector<sim::LowRl> R;
    std::vector<sim::LowEp> L;
    cpu.get_low_rl(R); cpu.get_low(L);
    cs.after_step(st[0], R[0], L[0]);
    o.known = count_known(lm);
    g_tm[4] += since(tp);
    if (std::getenv("DEC_PROF") && (long)g_tm[5] % 100 == 0)
      std::printf("  걸음 %.0f · ms/걸음: 꺼내기 %.2f · 위층 %.2f · 준비 %.2f · 걸음 %.2f · 걸음 뒤 %.2f\n", g_tm[5], 1e3 * g_tm[0] / g_tm[5], 1e3 * g_tm[1] / g_tm[5], 1e3 * g_tm[2] / g_tm[5],
                  1e3 * g_tm[3] / g_tm[5], 1e3 * g_tm[4] / g_tm[5]), std::fflush(stdout);
    if (horizon_steps > 0 && branch_step >= 0 && t - branch_step >= horizon_steps) {
      o.done = 0;   // 한도 (끝나지 않음)
      o.t = (t + 1) * dt;
      return o;
    }
  }
}
// 스냅숏 확인: 기준 판을 a 걸음까지, 저장, n 걸음 더 (걸음마다 상태 해시), 복원(위층 객체도 값 복사), 같은 n 걸음 → 해시가 다 같은가
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
int snap_test(const Cfg& C, int a, int n) {
  sim::Env cpu(C.hc, C.worlds, C.lidar, 1, C.seed, false);
  if (C.people > 0) cpu.set_people_max(std::vector<int>(1, C.people));
  cpu.reset();
  std::vector<sim::EnvState> st;
  lxc::Ctl cs(C.lp);
  lxd::Upper us(C.dp);
  BaseModel bm;
  bm.lat = C.lat;
  cpu.get_state(st);
  const sim::World& w = C.worlds[(size_t)st[0].map];
  cs.set_geom(w.gw, C.hc.cfg.res);
  std::vector<int16_t> lm;
  std::vector<uint32_t> tr;
  std::vector<uint8_t> vox, img;
  std::vector<float> pts, o1d, rew;
  std::vector<int32_t> cand, ccur, dn;
  std::vector<float> act((size_t)mdp::act::DIM, 0.f);
  const auto one = [&](int t) {
    cpu.get_state(st);
    bm.now = t;
    cpu.get_map(lm); cpu.get_trail(tr); cpu.get_camlayer(vox, pts); cpu.get_cand(cand); cpu.get_cand_cur(ccur); cpu.get_obs(img, o1d);
    const lxc::View v{&C.hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
    sim::LowCmd c1;
    if (us.step(v, o1d.data() + mdp::obs::POLICY_1D, cs, bm, &c1)) cpu.set_low_cmd(std::vector<sim::LowCmd>{c1});
    std::fill(act.begin(), act.end(), 0.f);
    act[0] = (float)cs.slot();
    cpu.low_prepare(act.data());
    cpu.step(act.data());
    cpu.get_reward(rew, dn);
    cpu.get_state(st);
    std::vector<sim::LowRl> R;
    std::vector<sim::LowEp> L;
    cpu.get_low_rl(R); cpu.get_low(L);
    if (!dn[0]) cs.after_step(st[0], R[0], L[0]);
    cpu.get_map(lm);
    return fnv(lm.data(), lm.size() * 2, fnv(&st[0], sizeof(sim::EnvState)));
  };
  for (int t = 0; t < a; ++t) one(t);
  const std::vector<uint8_t> blob = cpu.env_blob(0);
  const lxc::Ctl cs0 = cs;
  const lxd::Upper us0 = us;
  const BaseModel bm0 = bm;
  std::vector<uint64_t> h1, h2;
  for (int t = a; t < a + n; ++t) h1.push_back(one(t));
  cpu.set_env_blob(0, blob);
  cs = cs0; us = us0; bm = bm0;
  for (int t = a; t < a + n; ++t) h2.push_back(one(t));
  int bad = 0;
  for (int k = 0; k < n; ++k) bad += h1[(size_t)k] != h2[(size_t)k];
  // 다른 Env 로 옮겨도 같은가 (일꾼 스레드마다 Env 를 따로 둘 때)
  sim::Env e2(C.hc, C.worlds, C.lidar, 1, C.seed + 99, false);
  if (C.people > 0) e2.set_people_max(std::vector<int>(1, C.people));
  e2.reset();
  e2.set_env_blob(0, blob);
  const int bad2 = e2.env_blob(0) != blob;
  std::printf("스냅숏 확인: 걸음 %d 에서 저장 · %d 걸음 다시 — 다른 걸음 %d · 다른 Env 로 옮김 %s · 몫 %.1f MB\n", a, n, bad, bad2 ? "다름" : "같음", blob.size() / 1e6);
  return bad == 0 && bad2 == 0 ? 0 : 1;
}

// ---- GPU 길 (--gpu N): 기준 판을 GPU 환경 하나로 돌리며 고른 요청마다 상태(Env::env_blob)와 위층 객체를 떠 두고,
// 갈래는 GPU 환경 N 칸에 그 상태를 넣어(set_env_blob) 한꺼번에 굴린다. 갈래의 위층은 칸마다 호스트에서 (글 요청만 — 기준 응답기는 그림을 안 봄).
struct Cap {
  long id;
  int t;                       // 기준 판 걸음 번호 (요청을 보낸 걸음 — 저장은 위층 뒤, 준비·걸음 앞)
  std::vector<uint8_t> blob;
  lxc::Ctl cs;
  lxd::Upper us;
  BaseModel bm;
  size_t snap;                 // snaps 번호
};
struct GpuAct {
  float* d = nullptr;
  int n = 0;
  explicit GpuAct(int n_) : n(n_) { cudaMalloc(&d, sizeof(float) * mdp::act::DIM * n); }
  ~GpuAct() { cudaFree(d); }
  void up(const std::vector<float>& a) { cudaMemcpy(d, a.data(), sizeof(float) * a.size(), cudaMemcpyHostToDevice); }
};
Out base_gpu(const Cfg& C, int every, std::vector<Snap>* snaps, std::vector<Cap>* caps) {
  sim::Env g(C.hc, C.worlds, C.lidar, 1, C.seed, true);
  if (C.people > 0) g.set_people_max(std::vector<int>(1, C.people));
  g.reset();
  std::vector<sim::EnvState> st;
  lxc::Ctl cs(C.lp);
  lxd::Upper us(C.dp);
  BaseModel bm;
  bm.lat = C.lat;
  g.get_state(st);
  const sim::World& w = C.worlds[(size_t)st[0].map];
  cs.set_geom(w.gw, C.hc.cfg.res);
  std::vector<int16_t> lm;
  std::vector<uint32_t> tr;
  std::vector<uint8_t> vox, img;
  std::vector<float> pts, o1d, rew;
  std::vector<int32_t> cand, ccur, dn;
  std::vector<float> act((size_t)mdp::act::DIM, 0.f);
  GpuAct da(1);
  Out o;
  long nreq = 0;
  Hist hist;
  std::vector<std::array<float, 4>> kfp;
  for (int t = 0;; ++t) {
    g.get_state(st);
    bm.now = t;
    g.get_map(lm); g.get_trail(tr); g.get_camlayer(vox, pts); g.get_cand(cand); g.get_cand_cur(ccur); g.get_obs(img, o1d);
    hist_push(hist, st[0].step, lm);
    if (st[0].step % 10 == 0) kfp.push_back({(float)st[0].step, st[0].x + st[0].ex, st[0].y + st[0].ey, st[0].th + st[0].eth});
    const lxc::View v{&C.hc.cfg, &w, lm.data(), tr.data(), vox.empty() ? nullptr : vox.data(), &st[0], cand.data(), (uint32_t)ccur[0]};
    const size_t ns = bm.sent.size();
    sim::LowCmd c1;
    if (us.step(v, o1d.data() + mdp::obs::POLICY_1D, cs, bm, &c1)) g.set_low_cmd(std::vector<sim::LowCmd>{c1});
    if (bm.sent.size() > ns) {
      const lxd::Req& r = bm.sent.back();
      if (nreq++ % every == 0 && r.cells.size() >= 2) {
        snaps->push_back(make_snap(C, w, st[0], lm, cs, r, hist, kfp, us));
        caps->push_back(Cap{r.id, t, g.env_blob(0), cs, us, bm, snaps->size() - 1});
      }
    }
    std::fill(act.begin(), act.end(), 0.f);
    act[0] = (float)cs.slot();
    da.up(act);
    g.low_prepare(da.d);
    g.step(da.d);
    g.get_reward(rew, dn);
    g.get_state(st);
    if (dn[0] != 0) {
      std::vector<sim::EpInfo> ep;
      g.get_ep(ep);
      o.done = dn[0];
      o.t = ep[0].t;
      return o;
    }
    std::vector<sim::LowRl> R;
    std::vector<sim::LowEp> L;
    g.get_low_rl(R); g.get_low(L);
    cs.after_step(st[0], R[0], L[0]);
  }
}
// 갈래 묶음: jobs = (cap 번호, 글자, 반복). 칸 N 개씩 한꺼번에. 결과 out[job]
void branch_gpu(const Cfg& C, const std::vector<Cap>& caps, const std::vector<std::array<int, 3>>& jobs, int horizon_steps, int nslot, std::vector<Out>* out) {
  out->assign(jobs.size(), Out());
  const int K = mdp::hier::K, AW = mdp::act::DIM;
  const double dt = (double)C.hc.cfg.substeps * C.hc.cfg.dt;
  for (size_t j0 = 0; j0 < jobs.size(); j0 += (size_t)nslot) {
    const int n = (int)std::min(jobs.size() - j0, (size_t)nslot);
    sim::Env g(C.hc, C.worlds, C.lidar, n, C.seed, true);
    if (C.people > 0) g.set_people_max(std::vector<int>((size_t)n, C.people));
    g.reset();
    std::vector<sim::EnvState> st;
    struct Sl { lxc::Ctl cs; lxd::Upper us; BaseModel bm; int t; int t_branch; bool live; };
    std::vector<Sl> sl;
    for (int i = 0; i < n; ++i) {
      const auto& jb = jobs[j0 + (size_t)i];
      const Cap& cp = caps[(size_t)jb[0]];
      g.set_env_blob(i, cp.blob);
      Sl x{cp.cs, cp.us, cp.bm, cp.t, cp.t, true};
      x.us = cp.us;
      x.us.set_input(dec::kText);
      x.bm.q.back().a.choice = std::string(1, (char)('A' + jb[1]));   // 그 요청의 답 = 글자
      sl.push_back(std::move(x));
    }
    g.get_state(st);
    for (int i = 0; i < n; ++i) {   // 갈래 뒤 우연 (반복 0 은 그대로)
      const int rp = jobs[j0 + (size_t)i][2];
      if (rp > 0) { st[(size_t)i].rng = mix(st[(size_t)i].rng ^ mix((uint64_t)rp * 0x9e3779b97f4a7c15ull)); g.set_state(i, st[(size_t)i]); }
    }
    const sim::World& w = C.worlds[0];
    std::vector<int32_t> cand, ccur, dn;
    std::vector<float> o1d((size_t)n * mdp::hier::POLICY_1D), rew, act((size_t)AW * n, 0.f);
    std::vector<sim::LowCmd> cmd((size_t)n, sim::LowCmd{-1, 0});
    std::vector<sim::LowRl> R;
    std::vector<sim::LowEp> L;
    std::vector<sim::EpInfo> ep;
    std::vector<int16_t> lm;
    std::vector<uint32_t> trs;
    std::vector<uint8_t> vx;
    GpuAct da(n);
    for (int i = 0; i < n; ++i) cmd[(size_t)i] = sl[(size_t)i].cs.cmd();
    g.set_low_cmd(cmd);
    int live = n;
    bool first = true;
    while (live > 0) {
      g.get_state(st); g.get_cand(cand); g.get_cand_cur(ccur);
      cudaMemcpy(o1d.data(), g.batch().obs_1d, sizeof(float) * o1d.size(), cudaMemcpyDeviceToHost);
      bool chg = false;
      for (int i = 0; i < n; ++i) {
        Sl& x = sl[(size_t)i];
        if (!x.live) continue;
        if (first) continue;   // 첫 걸음: 저장한 때 위층은 이미 돌았음 (준비·걸음부터)
        x.bm.now = x.t;
        // 후보를 만들 걸음(위층 결정 차례 · 답이 옴 · 보낼 때)에만 이 칸의 지도 · 지나간 길 · 깊이 층을 꺼냄 (cand::build 가 읽음)
        const bool ans = !x.bm.q.empty() && x.bm.q.front().ready <= x.t;
        const bool snd = !x.us.waiting() && (x.us.last_send() < 0 || st[(size_t)i].step - x.us.last_send() >= C.dp.period_steps);
        const bool need = x.cs.due(st[(size_t)i]) || ans || snd;
        if (need) {
          const sim::EnvBatch& B = g.batch();
          const size_t cells = (size_t)C.hc.cfg.size * C.hc.cfg.size, nf = (size_t)C.hc.cfg.size * ((C.hc.cfg.size + 31) / 32);
          lm.resize(cells); trs.resize(2 * nf); vx.resize(cells);
          cudaMemcpy(lm.data(), B.lmap + (size_t)i * cells, cells * 2, cudaMemcpyDeviceToHost);
          cudaMemcpy(trs.data(), B.trail + (size_t)i * 2 * nf, 2 * nf * 4, cudaMemcpyDeviceToHost);
          if (B.cvox) cudaMemcpy(vx.data(), B.cvox + (size_t)i * cells, cells, cudaMemcpyDeviceToHost);
        }
        const lxc::View v{&C.hc.cfg, &w, need ? lm.data() : nullptr, need ? trs.data() : nullptr, need && g.batch().cvox ? vx.data() : nullptr, &st[(size_t)i],
                          &cand[(size_t)i * K], (uint32_t)ccur[(size_t)i]};
        sim::LowCmd c1;
        if (x.us.step(v, o1d.data() + (size_t)i * mdp::hier::POLICY_1D + mdp::obs::POLICY_1D, x.cs, x.bm, &c1)) { cmd[(size_t)i] = c1; chg = true; }
      }
      first = false;
      if (chg) g.set_low_cmd(cmd);
      for (int i = 0; i < n; ++i) act[(size_t)i * AW] = (float)sl[(size_t)i].cs.slot();
      da.up(act);
      g.low_prepare(da.d);
      g.step(da.d);
      g.get_reward(rew, dn);
      g.get_state(st);
      g.get_low_rl(R); g.get_low(L);
      bool need_ep = false;
      for (int i = 0; i < n; ++i) need_ep = need_ep || (sl[(size_t)i].live && dn[(size_t)i]);
      if (need_ep) g.get_ep(ep);
      for (int i = 0; i < n; ++i) {
        Sl& x = sl[(size_t)i];
        if (!x.live) continue;
        Out& o = (*out)[j0 + (size_t)i];
        if (dn[(size_t)i]) { o.done = dn[(size_t)i]; o.t = ep[(size_t)i].t; x.live = false; --live; continue; }
        x.cs.after_step(st[(size_t)i], R[(size_t)i], L[(size_t)i]);
        x.t += 1;
        if (horizon_steps > 0 && x.t - x.t_branch >= horizon_steps) {
          o.done = 0; o.t = x.t * dt;
          g.get_map_env(i, lm);
          o.known = count_known(lm);
          x.live = false; --live;
        }
      }
    }
  }
}
std::string jesc(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o;
}
}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::string scene, out, base = "slot0";
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml";
  Cfg C;
  double period = 3.0, latency = 2.6, horizon = 600.0;
  int every = 3, reps = 4, max_req = 0, snap_a = -1, gpu_n = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " 뒤에 값"); return argv[++i]; };
    if (a == "--seed") C.seed = std::stoull(nx());
    else if (a == "--out") out = nx();
    else if (a == "--people") C.people = std::stoi(nx());
    else if (a == "--period") period = std::stod(nx());
    else if (a == "--latency") latency = std::stod(nx());
    else if (a == "--every") every = std::stoi(nx());
    else if (a == "--reps") reps = std::stoi(nx());
    else if (a == "--horizon") horizon = std::stod(nx());
    else if (a == "--max-req") max_req = std::stoi(nx());
    else if (a == "--base") base = nx();
    else if (a == "--env") env_path = nx();
    else if (a == "--snap-test") snap_a = std::stoi(nx());
    else if (a == "--gpu") gpu_n = std::stoi(nx());
    else scene = a;
  }
  if (scene.empty() || out.empty() || base != "slot0") {
    std::fprintf(stderr, "dec_rollout <장면.scene> --seed 1 --out p [--people 0] [--period 3.0] [--latency 2.6] [--every 3] [--reps 4] [--horizon 600] [--max-req 0] [--base slot0]\n");
    return 2;
  }
  try {
    C.hc = sim::load_env_cfg(env_path);
    sim::EnvCfg& c = C.hc.cfg;   // low_explore_check 와 같은 설정 (combine_run.cpp 합친 탐사)
    c.action_mode = sim::kActPath;
    c.curriculum = 0;
    c.low_stat = c.low_task = c.low_rl = 1;
    c.low_auto = 0;
    c.low_act_w = 0;
    c.low_explore = 1;
    c.lo_aux = 0;
    C.lidar = sim::load_lidar_model(model_path);
    C.worlds.push_back(sim::build_world(scene, sim::load_scene(scene), C.hc));
    sim::set_cpu_threads(1);
    C.lp = lxc::Params::from_cfg(c);
    C.lp.use_fuel = false;
    C.lp.use_cand = true;   // 후보 = scripts/cand_view.h (실물 노드 upper:=decider decider_cand true 와 같음)
    C.dp.cand = true;
    const double dt = (double)c.substeps * c.dt;
    C.dp.period_steps = std::max(1, (int)std::lround(period / dt));
    C.lat = std::max(0, (int)std::lround(latency / dt));
    const int hz = horizon > 0 ? (int)std::lround(horizon / dt) : 0;
    if (snap_a >= 0) return snap_test(C, snap_a, 400);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Snap> snaps;
    std::vector<Cap> caps;
    const Out b = gpu_n > 0 ? base_gpu(C, every, &snaps, &caps) : run(C, -1, 0, 0, 0, &snaps);
    const double base_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("기준 판 (%s, %s): 끝 %d · %.1f s · 고른 요청 %zu 개 · 걸린 시간 %.1f s\n", base.c_str(), gpu_n > 0 ? "GPU" : "CPU", b.done, b.t, snaps.size(), base_s);
    std::fflush(stdout);
    // 고른 요청 (GPU 길은 기준 판에서 이미 every 마다 골라 둠)
    std::vector<size_t> pick;
    for (size_t k = 0; k < snaps.size(); ++k) {
      if (gpu_n <= 0 && ((long)k % every != 0 || snaps[k].cell.size() < 2)) continue;
      if (max_req > 0 && (int)pick.size() >= max_req) break;
      pick.push_back(k);
    }
    std::vector<std::vector<std::vector<Out>>> br(snaps.size());
    const auto t1 = std::chrono::steady_clock::now();
    if (gpu_n > 0) {
      std::vector<std::array<int, 3>> jobs;
      for (size_t k : pick)
        for (size_t L = 0; L < snaps[k].cell.size(); ++L)
          for (int rp = 0; rp < reps; ++rp) jobs.push_back({(int)k, (int)L, rp});
      std::vector<Out> res;
      branch_gpu(C, caps, jobs, hz, gpu_n, &res);
      for (size_t q = 0; q < jobs.size(); ++q) {
        auto& t = br[(size_t)jobs[q][0]];
        if (t.empty()) t.assign(snaps[(size_t)jobs[q][0]].cell.size(), std::vector<Out>((size_t)reps));
        t[(size_t)jobs[q][1]][(size_t)jobs[q][2]] = res[q];
      }
      std::printf("갈래 %zu 판 (GPU 칸 %d): %.1f s\n", jobs.size(), gpu_n, std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    } else {
      for (size_t k : pick) {
        const Snap& s = snaps[k];
        br[k].assign(s.cell.size(), std::vector<Out>((size_t)reps));
        for (size_t L = 0; L < s.cell.size(); ++L)
          for (int rp = 0; rp < reps; ++rp) br[k][L][(size_t)rp] = run(C, s.id, (int)L, (uint64_t)rp, hz, nullptr);
      }
      std::printf("갈래 (CPU): %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    }
    std::FILE* f = std::fopen((out + ".jsonl").c_str(), "w");
    if (!f) throw std::runtime_error("--out 파일을 못 엶");
    const int n_done = (int)pick.size();
    for (size_t k : pick) {
      const Snap& s = snaps[k];
      const char* stem = out.c_str();
      char p[1024];
      std::snprintf(p, sizeof p, "%s_r%ld.req.json", stem, s.id);
      if (std::FILE* g = std::fopen(p, "w")) { std::fwrite(s.body.data(), 1, s.body.size(), g); std::fclose(g); }
      std::snprintf(p, sizeof p, "%s_r%ld.map.bin", stem, s.id);
      if (std::FILE* g = std::fopen(p, "wb")) {   // i32 x0 y0 w h n_past · i32 past_age[n_past] · i16 지금 [h][w] · i16 지난 [n_past][h][w]
        const int32_t hd[5] = {s.x0, s.y0, s.w, s.h, (int32_t)s.past.size()};
        std::fwrite(hd, 4, 5, g);
        for (int a : s.past_age) { const int32_t v = a; std::fwrite(&v, 4, 1, g); }
        std::fwrite(s.map.data(), 2, s.map.size(), g);
        for (const auto& m : s.past) std::fwrite(m.data(), 2, m.size(), g);
        std::fclose(g);
      }
      std::fprintf(f, "{\"scene\":\"%s\",\"seed\":%llu,\"people\":%d,\"req\":%ld,\"step\":%d,\"x\":%.4f,\"y\":%.4f,\"th\":%.5f,\"res\":%.4f,\"ox\":%.4f,\"oy\":%.4f,\"gw\":%d,"
                      "\"cand_ver\":\"cand-v1-b8d942b\",\"period_steps\":%d,\"lat_steps\":%d,\"horizon_s\":%.1f,\"base\":{\"done\":%d,\"t\":%.2f},\"mem\":{\"trail\":[",
                   jesc(scene).c_str(), (unsigned long long)C.seed, C.people, s.id, s.step, s.x, s.y, s.th, c.res, C.worlds[0].ox, C.worlds[0].oy, C.worlds[0].gw, C.dp.period_steps, C.lat,
                   horizon, b.done, b.t);
      for (size_t q = 0; q < s.mem.trail.size(); ++q) { const auto& t = s.mem.trail[q]; std::fprintf(f, "%s[%.3f,%.3f,%.4f,%d]", q ? "," : "", t.x, t.y, t.th, t.step); }
      std::fprintf(f, "],\"goals\":[");
      for (size_t q = 0; q < s.mem.goals.size(); ++q) { const auto& t = s.mem.goals[q]; std::fprintf(f, "%s[%d,%.3f,%.3f,%d,%d,%d]", q ? "," : "", t.id, t.x, t.y, t.start, t.end, t.why); }
      std::fprintf(f, "],\"fails\":[");
      for (size_t q = 0; q < s.mem.fails.size(); ++q) { const auto& t = s.mem.fails[q]; std::fprintf(f, "%s[%.3f,%.3f,%d,%d]", q ? "," : "", t.x, t.y, t.why, t.step); }
      std::fprintf(f, "]},\"kf\":[");
      for (size_t q = 0; q < s.kf.size(); ++q) std::fprintf(f, "%s[%d,%.3f,%.3f,%.4f]", q ? "," : "", (int)s.kf[q][0], s.kf[q][1], s.kf[q][2], s.kf[q][3]);
      std::fprintf(f, "],\"decisions\":[");
      for (size_t q = 0; q < s.decs.size(); ++q) { const auto& d = s.decs[q]; std::fprintf(f, "%s[%d,%d,%.3f,%.3f,%d,%d]", q ? "," : "", d.step, d.id, d.x, d.y, d.kept, d.src); }
      std::fprintf(f, "],\"cand\":[");
      for (size_t L = 0; L < s.cands.size(); ++L) {
        const cand::Cand& q = s.cands[L];
        std::fprintf(f, "%s{\"id\":%d,\"age\":%d,\"vp\":%d,\"anc\":%d,\"len_m\":%.3f,\"turn_rad\":%.4f,\"t_s\":%.3f,\"bearing\":%.4f,\"gain_m2\":%.3f,\"marg_m2\":%.3f,\"score\":%.5f,"
                        "\"open_m\":%.3f,\"clear_m\":%.3f,\"path_clear_m\":%.3f,\"same_room\":%d,\"room_left\":%d,\"cur\":%d,\"kept\":%d,\"chosen\":%d,\"aborts\":%d,\"last_fail\":%d,"
                        "\"since_fail_s\":%.2f,\"fail_near\":%d,\"in_avoid\":%d,\"visited\":%d,\"out\":[",
                     L ? "," : "", q.id, q.age, q.vp, q.anc, q.len_m, q.turn_rad, q.t_s, q.bearing, q.gain_m2, q.marg_m2, q.score, q.open_m, q.clear_m, q.path_clear_m, q.same_room,
                     q.room_left, q.cur, q.kept, q.chosen, q.aborts, q.last_fail, q.since_fail_s, q.fail_near, q.in_avoid, q.visited);
        for (int rp = 0; rp < reps; ++rp)
          if (L < br[k].size()) std::fprintf(f, "%s[%d,%.2f,%ld]", rp ? "," : "", br[k][L][(size_t)rp].done, br[k][L][(size_t)rp].t, br[k][L][(size_t)rp].known);
        std::fprintf(f, "]}");
      }
      std::fprintf(f, "]}\n");
    }
    std::fclose(f);
    std::printf("끝: 요청 %d 개 · 모두 %.1f s\n", n_done, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "dec_rollout: %s\n", ex.what());
    return 1;
  }
}
