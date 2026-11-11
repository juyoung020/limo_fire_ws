// V0·V1 (환경): 같은 맵·씨앗·행동열이면 GPU 커널과 CPU 참조판의 상태·보상(항마다)·끝·판 끝 정보·지도·덮음 표시·관측(2D·1D·가치 전용)이
// 걸음마다 비트까지 같은지. 그리고 관측 값 범위(NaN 없음, 정규화 범위 — mdp/observations.h 머리 주석).
// env_verify <장면.scene> [장면 …] [--list 목록] [--envs 8] [--steps 80] [--declare 0.01] [--negative] [--env config/env.yaml]
//            [--model config/sensor_model.yaml] [--dump 지도.pgm]   (env 0 의 마지막 지도: 0 모름 128, 빈 칸 흼, 막힘 검음)
//   --declare p: 걸음마다 끝 선언을 p 확률로 (선언 성공·실패 끝도 비교에 들어가게)
//   --negative: GPU 쪽 행동 하나를 바꿔 일부러 어긋나게 한다 (검사가 틀림을 잡는지)
//   --people K: 환경마다 사람 상한 K (판마다 0~K 명, people.h). --negative-person: 가운데 걸음에 GPU 쪽 사람 하나를 1 cm 옮긴다
//   --slot0: (--hier 와) 늘 후보 0 번 — 복구·목표 허용 길까지 비교
//   --graph: 그래프 모드(--hier 위에, mdp/graph.h — 행동 = 현재 노드의 이웃 번호, 열에 여덟은 유효 이웃 중 무작위). 노드 특징·간선·현재 노드·전역 숫자·이웃 칸이
//            판 시작·결정마다 GPU == CPU 인지, 정답 격자를 바꾼 묶음에서 그래프 관측이 같은지 (--negative 는 유효 이웃 번호 하나를 다른 유효 이웃으로)
//   --hier: 계층형 행동 모드(kActPath — 결정 하나 = 속도 행동 10 걸음 묶음, 후보 번호 무작위). 덧붙여 정답 격자를 쓰지 않는지:
//           CPU 묶음의 정답 격자(MapDev::grid)를 모두 막힘·모두 빈 칸으로 바꾼 사본에서 경로 계획·추종기 명령이 같은지 (--negative 는 후보 번호 하나를 바꿈)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"
#include "obs_ref.h"

namespace {
// 1D 관측 칸마다 허용 범위 (observations.h 정규화 근거)
void range_1d(int i, float* lo, float* hi) {
  using namespace mdp::obs;
  *lo = -1.f;
  *hi = 1.f;
  if (i >= POLICY_1D) {   // 계층형 후보 칸 (mdp::hier)
    const int f = (i - POLICY_1D) % mdp::hier::FEAT;
    if (f == mdp::hier::F_DIST) { *lo = 0.f; *hi = 2.f; }
    else if (f != mdp::hier::F_SIN && f != mdp::hier::F_COS) { *lo = 0.f; *hi = 1.f; }
    return;
  }
  if (i < LIDAR) { *lo = 0.f; return; }
  if (i < LIDAR + INTENSITY) { *lo = 0.f; *hi = 1.5f; return; }   // 세기 / 100

  const int k = i - LIDAR - INTENSITY;
  if (k < POSE + 2) { *lo = -2.f; *hi = 2.f; }                       // 시작점 기준 위치 / 12.8 m (지도 25.6 m 안)
  else if (k == SLIP || k == GYRO_Z || k == EKF + 1) { *lo = -2.f; *hi = 2.f; }   // 헛돎은 순간 3.14 넘을 수 있음
  else if (k >= WHEELS4 && k < WHEELS4 + 4) { *lo = -1.5f; *hi = 1.5f; }          // 바퀴 한도 1.11 m/s — v·vy·ω 가 따로 응답해 바뀌는 중엔 합이 잠깐 넘음
  else if (k == WHEEL_V || k == EKF) { *lo = -1.2f; *hi = 1.2f; }
  else if (k == ELAPSED || k == KNOWN_AREA || k == PROGRESS + 1 || k == FRONT_N || k == CLOSED || k == FRONT_CELLS) { *lo = 0.f; *hi = 1.f; }
  else if (k == FRONTIER + 3 || k == FRONTIER2 + 3 || k == FRONTIER3 + 3) { *lo = 0.f; *hi = 1.f; }   // 덩어리 크기
  else if (k == FRONTIER || k == FRONTIER2 || k == FRONTIER3) { *lo = 0.f; *hi = 2.f; }               // BFS 길 거리 (자름 2)
  else if (k == PROGRESS || k == OCC_GAIN) { *lo = 0.f; *hi = 2.f; }
  else if (k == HOME) { *lo = 0.f; *hi = 2.9f; }
  else if (k >= LAST_ACT && k < LAST_ACT + 16 && (k - LAST_ACT) % 4 == 3) { *lo = 0.f; *hi = 1.f; }
  else if (k >= ACCEL && k < IMU_TILT + 4) { *lo = -3.f; *hi = 3.f; }   // IMU (observations.h 정규화 — 대개 ±0.6, 잡음 꼬리 여유)
  else if (k >= RESERVED || (k >= MAP_VEL && k < MAP_VEL + 3)) { *lo = 0.f; *hi = 0.f; }   // 📏·예약 = 0
}
}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);   // GPU == CPU·옛 빌드 비교는 정확한 길로 (RT 코어 빠른 길은 tools/rt_fastcheck 로 통계 확인)
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", dump, list;
  int n_env = 8, n_steps = 80;
  float p_declare = 0.01f;
  bool negative = false;
  bool negative_person = false;
  bool graph = false;
  int graph_open = -1;    // --graph-open 0/1: 유틸리티가 닫힘을 막는 프론티어 칸만 세는지
  int graph_stand = -1;   // --graph-stand 0/1: 노드·간선 칸 기준을 설정 파일 대신 (0 빈 칸 · 1 설 수 있는 칸)
  int people = 0;
  bool spin = false;
  bool slot0 = false;   // --slot0 (계층형): 늘 후보 0 번 — 같은 목표가 이어져 복구(hier_path.h "복구")·목표 허용 길을 GPU == CPU 로 지나가게
  bool hier = false;   // --spin: 행동을 제자리 돌기(ω 0.5 정규)로 — 작은 방에서 지도 닫힘 성공(판 끝 1) 길을 GPU == CPU 로 지나가게
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--declare" && i + 1 < argc) p_declare = std::stof(argv[++i]);
    else if (a == "--negative") negative = true;
    else if (a == "--negative-person") negative = negative_person = true;
    else if (a == "--people" && i + 1 < argc) people = std::stoi(argv[++i]);
    else if (a == "--spin") spin = true;
    else if (a == "--slot0") slot0 = true;
    else if (a == "--hier") hier = true;
    else if (a == "--graph") hier = graph = true;
    else if (a == "--graph-stand") graph_stand = std::atoi(argv[++i]);
    else if (a == "--graph-open") graph_open = std::atoi(argv[++i]);
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--list" && i + 1 < argc) list = argv[++i];
    else if (a == "--threads" && i + 1 < argc) sim::set_cpu_threads(std::stoi(argv[++i]));   // CPU 참조판 스레드 수
    else scenes.push_back(a);
  }
  if (scenes.empty() && list.empty()) { std::fprintf(stderr, "env_verify <장면.scene> [장면 …] [--list 목록] [--envs 8] [--steps 80] [--negative]\n"); return 2; }
  // 커리큘럼 오르내림 규칙 (curriculums.h on_episode_end): 최근 8 판 끝 덮음 평균 ≥ 0.8 → 올림, < 0.4 → 내림, 그 사이 → 그대로,
  // 7 판까지는 안 바뀜, 최고·최저 단계에서 멈춤
  {
    auto run = [](int stage, std::vector<float> covs, int max_stage) {
      int32_t s = stage, n = 0;
      uint32_t h = 0;
      for (float c : covs) mdp::cur::on_episode_end(&s, &h, &n, c, max_stage);
      return s;
    };
    const std::vector<float> hi(8, 0.85f), lo(8, 0.3f), mid(8, 0.6f), edge(8, 0.8f), seven(7, 1.0f);
    std::vector<float> mixed = {1.f, 1.f, 1.f, 1.f, 0.6f, 0.6f, 0.6f, 0.6f};   // 평균 0.8 → 올림
    const bool ok = run(1, hi, 3) == 2 && run(1, lo, 3) == 0 && run(1, mid, 3) == 1 && run(1, edge, 3) == 2 && run(1, mixed, 3) == 2 &&
                    run(1, seven, 3) == 1 && run(3, hi, 3) == 3 && run(0, lo, 3) == 0;
    std::printf("커리큘럼 덮음 규칙(옛, curriculum.rule: coverage) %s (8 판 덮음 평균 ≥ 0.8 → 올림, < 0.4 → 내림, 그 사이·7 판까지 → 그대로, 끝 단계에서 멈춤)\n", ok ? "통과" : "실패");
    if (!ok) return 1;
    // 맨 위 단계 무작위 (실험 top_random): 맨 위에서 올림 조건 → rnd % (max + 1), 끄면 그대로, 맨 위 아니면 보통 올림, 내림은 그대로
    auto run_r = [](int stage, std::vector<float> covs, int max_stage, bool tr, uint32_t rnd) {
      int32_t s = stage, n = 0;
      uint32_t h = 0;
      for (float c : covs) mdp::cur::on_episode_end(&s, &h, &n, c, max_stage, tr, rnd);
      return s;
    };
    const bool ok2 = run_r(3, hi, 3, true, 5u) == 1 && run_r(3, hi, 3, true, 7u) == 3 && run_r(3, hi, 3, false, 5u) == 3 &&
                     run_r(1, hi, 3, true, 5u) == 2 && run_r(3, lo, 3, true, 5u) == 2 && run_r(3, mid, 3, true, 5u) == 3;
    std::printf("커리큘럼 맨 위 무작위 규칙 %s\n", ok2 ? "통과" : "실패");
    if (!ok2) return 1;
  }
  sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  // 성공 규칙 (curriculums.h on_episode_end_success, env.yaml curriculum.success): 덱(deque)으로 따로 짠 참조 규칙과 무작위 판 열 4 만 판에서
  // 단계·창이 판마다 같은지 (창 크기·문턱 여러 벌 + 지금 설정 값), 그리고 손으로 정한 경우
  {
    struct Ref {   // 참조: 최근 W 판 성공을 덱에, 문턱은 정수 비교(k·1000 ≥ round(up·1000)·W)
      int W, up_m, down_m, max_stage;
      bool tr;
      int stage = 0;
      std::vector<int> win;
      void step(bool succ, uint32_t rnd) {
        win.push_back(succ ? 1 : 0);
        if ((int)win.size() > W) win.erase(win.begin());
        if ((int)win.size() < W) return;
        int k = 0;
        for (int b : win) k += b;
        const int s0 = stage;
        if (k * 1000 >= up_m * W) stage = stage < max_stage ? stage + 1 : (tr ? (int)(rnd % (uint32_t)(max_stage + 1)) : stage);
        else if (k * 1000 < down_m * W && stage > 0) stage -= 1;
        if (stage != s0) win.clear();
      }
    };
    struct P { int W; float up, down; };
    const sim::EnvCfg& c0 = hc.cfg;
    const P ps[] = {{24, 0.8f, 0.5f}, {24, 0.9f, 0.5f}, {8, 0.75f, 0.25f}, {32, 0.7f, 0.4f}, {1, 1.f, 1.f},
                    {c0.cur_window > 32 ? 32 : c0.cur_window, c0.cur_up, c0.cur_down}};
    bool ok = true;
    long n_up = 0, n_down = 0, n_top = 0;
    uint64_t rng = 12345;
    for (const P& q : ps)
      for (int tr = 0; tr < 2; ++tr) {
        Ref ref{q.W, (int)(q.up * 1000.f + 0.5f), (int)(q.down * 1000.f + 0.5f), 3, tr != 0};
        int32_t st = 0, n = 0;
        uint32_t h = 0;
        for (int e = 0; e < 40000 && ok; ++e) {
          const double p_succ = 0.3 + 0.65 * ((e / 997) % 3) / 2.0;   // 성공 확률을 0.3 · 0.625 · 0.95 로 바꿔 가며 (오르내림 모두)
          const bool succ = dm::rand01(rng) < p_succ;
          const uint32_t rnd = (uint32_t)(dm::splitmix64(rng) >> 32);
          const int s_before = st;
          mdp::cur::on_episode_end_success(&st, &h, &n, succ, 3, tr != 0, rnd, q.W, q.up, q.down);
          ref.step(succ, rnd);
          n_up += st > s_before;
          n_down += st < s_before;
          n_top += s_before == 3 && st != 3 && tr;
          if (st != ref.stage || n != (int)ref.win.size()) ok = false;
        }
      }
    // 손으로: 창 24 · 올림 0.8(20/24 아님 — 19.2 → 20 판 성공) · 내림 0.5(12 판)
    auto run = [](int stage, const std::vector<int>& seq, bool tr = false, uint32_t rnd = 0u, int32_t* n_out = nullptr) {
      int32_t s = stage, n = 0;
      uint32_t h = 0;
      for (int b : seq) mdp::cur::on_episode_end_success(&s, &h, &n, b != 0, 3, tr, rnd, 24, 0.8f, 0.5f);
      if (n_out) *n_out = n;
      return s;
    };
    auto seq = [](int succ, int fail) { std::vector<int> v(succ, 1); v.insert(v.end(), fail, 0); return v; };
    int32_t n_after = -1;
    const bool hand = run(1, seq(23, 0)) == 1 &&                     // 창이 덜 참 → 그대로
                      run(1, seq(24, 0), false, 0u, &n_after) == 2 && n_after == 0 &&   // 다 성공 → 올림, 창 비움
                      run(1, seq(20, 4)) == 2 && run(1, seq(19, 5)) == 1 &&            // 20/24 = 0.833 올림 · 19/24 = 0.79 그대로
                      run(1, seq(12, 12)) == 1 && run(1, seq(11, 13)) == 0 &&          // 12/24 = 0.5 그대로 · 11/24 내림
                      run(0, seq(0, 24)) == 0 &&                                       // 맨 아래에서 멈춤
                      run(3, seq(24, 0), false) == 3 && run(3, seq(24, 0), true, 5u) == 1 &&   // 맨 위: 그대로 · 무작위(5 % 4)
                      run(1, [] { std::vector<int> v(19, 1); v.insert(v.end(), 5, 0); v.push_back(1); return v; }()) == 1 &&   // 창이 미끄러져 첫 성공이 빠짐: 19/24 그대로
                      run(1, [] { std::vector<int> v(5, 0); v.insert(v.end(), 19, 1); v.push_back(1); return v; }()) == 2;    // 첫 실패가 빠짐: 20/24 올림
    ok = ok && hand && n_up > 0 && n_down > 0 && n_top > 0;
    std::printf("커리큘럼 성공 규칙 %s (최근 N 판 성공 비율 ≥ 올림 문턱 → 올림, < 내림 문턱 → 내림, 단계 바뀌면 창 비움, 미끄러지는 창 — 참조 구현과 판 %d 벌 × 4 만 판 같음 "
                "(올림 %ld · 내림 %ld · 맨 위 무작위 %ld), 손 경우 %s; 지금 설정: %s 규칙, 창 %d · 올림 %.2f · 내림 %.2f)\n",
                ok ? "통과" : "실패", (int)(sizeof(ps) / sizeof(ps[0])) * 2, n_up, n_down, n_top, hand ? "같음" : "다름",
                c0.cur_rule == mdp::cur::kCurSuccess ? "성공" : "덮음", c0.cur_window, c0.cur_up, c0.cur_down);
    if (!ok) return 1;
  }
  if (hier) hc.cfg.action_mode = sim::kActPath;
  if (graph) hc.cfg.graph_on = 1;
  if (graph_stand >= 0) hc.cfg.graph_stand = graph_stand;
  if (graph_open >= 0) hc.cfg.graph_open = graph_open;
  if (people > 0) hc.cfg.curriculum = 0;   // 커리큘럼이 켜져 있으면 단계가 사람 수를 정함 — --people 로 정한 수를 쓰려고 끔
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  std::vector<sim::World> worlds;
  try {
    if (!list.empty()) {
      std::vector<std::string> skipped;
      worlds = sim::load_scene_list(list, hc, &skipped);
      for (auto& s : skipped) std::printf("버림: %s\n", s.c_str());
    }
    for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "env_verify: %s\n", ex.what());
    return 1;
  }
  int n_print = 0;
  for (const auto& w : worlds)
    if (++n_print <= 6 || n_print == (int)worlds.size()) std::printf("맵 %s: %d×%d 칸, 벽 면 칸(덮음 분모) %d (%.2f m²), 갈 수 있는 바닥 %d (%.1f m², 단계 %d), T_ref %.0f s, 시작 후보 %zu, %s\n",
                w.name.c_str(), w.gw, w.gh, w.n_wall, w.n_wall * hc.cfg.res * hc.cfg.res, w.n_floor, w.n_floor * hc.cfg.res * hc.cfg.res,
                (w.stage >= 0 ? w.stage : mdp::cur::stage_of_area(w.n_floor * hc.cfg.res * hc.cfg.res)), mdp::rew::t_ref(w.n_wall), w.spawn.size(), w.closed ? "닫힘" : w.leak.c_str());
  std::printf("안전층 %s: 문턱 %.3f m (앞 범퍼에서, 제동 거리 %.3f m + 판단 지연 0.2 s × %.1f m/s)\n", hc.cfg.safety ? "켬" : "끔", hc.cfg.safety_dist,
              hc.brake_dist, mdp::act::V_FWD);
  sim::Env gpu(hc, worlds, lidar, n_env, 7, true), cpu(hc, worlds, lidar, n_env, 7, false);
  gpu.set_people_max(std::vector<int>(n_env, people));
  cpu.set_people_max(std::vector<int>(n_env, people));
  gpu.reset();
  cpu.reset();
  gpu.sync();
  constexpr int A = mdp::act::DIM, K = mdp::rew::kTerms;
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * A * n_env);
  uint64_t rng = 99;
  size_t bad_state = 0, bad_reward = 0, bad_obs = 0, bad_range = 0, n_nan = 0, safety_zero = 0;
  int first_bad = -1, ends[mdp::cur::kNumDone] = {}, first_range = -1;
  double reward_sum = 0, term_sum[K] = {};
  long people_sum = 0, hit_person = 0;
  std::vector<sim::EnvState> sg, sc;
  std::vector<float> rg, rc, tg, tc, og1, oc1, pg, pc;
  std::vector<int32_t> dg, dc;
  std::vector<sim::EpInfo> eg, ec;
  std::vector<uint8_t> ig, ic;
  // 그래프 모드: 관측 버퍼 GPU == CPU (판 시작 관측부터), 통계
  std::vector<uint8_t> gfg, gfc;
  std::vector<uint32_t> geg, gec;
  std::vector<int32_t> gcg, gcc, gng, gnc;
  std::vector<float> ggg, ggc;
  size_t bad_graph = 0, g_obs = 0, g_valid = 0, g_edges = 0, g_util = 0, g_guide = 0, g_noedge = 0, g_moved = 0;
  int neg_done = 0;
  const auto graph_cmp = [&](int t) {
    if (!graph) return;
    gpu.get_graph(gfg, geg, gcg, ggg, gng);
    cpu.get_graph(gfc, gec, gcc, ggc, gnc);
    size_t bg = 0;
    for (size_t i = 0; i < gfc.size(); ++i) bg += gfg[i] != gfc[i];
    for (size_t i = 0; i < gec.size(); ++i) bg += geg[i] != gec[i];
    for (size_t i = 0; i < gcc.size(); ++i) bg += gcg[i] != gcc[i];
    for (size_t i = 0; i < gnc.size(); ++i) bg += gng[i] != gnc[i];
    for (size_t i = 0; i < ggc.size(); ++i) bg += std::memcmp(&ggg[i], &ggc[i], 4) != 0;
    for (float v : ggc) n_nan += !std::isfinite(v);
    bad_graph += bg;
    if (bg && first_bad < 0) first_bad = t;
    for (int i = 0; i < n_env; ++i) {
      ++g_obs;
      for (int k = 0; k < mdp::graph::N; ++k) {
        const uint8_t* F = &gfc[((size_t)i * mdp::graph::N + k) * mdp::graph::FEAT];
        g_valid += F[mdp::graph::F_VALID] != 0;
        g_util += F[mdp::graph::F_UTIL] != 0;
        g_guide += F[mdp::graph::F_LGUIDE] != 0;
        g_edges += __builtin_popcount(gec[(size_t)i * mdp::graph::N + k]);
      }
      g_noedge += gec[(size_t)i * mdp::graph::N + gcc[i]] == 0u;
    }
  };
  graph_cmp(-1);
  std::vector<int32_t> cur_prev = gcc;
  for (int t = 0; t < n_steps; ++t) {
    std::vector<float> act((size_t)A * n_env);
    for (int i = 0; i < n_env; ++i) sim::random_action(hc.cfg, rng, &act[(size_t)A * i], p_declare);
    if (spin)
      for (int i = 0; i < n_env; ++i) { float* q = &act[(size_t)A * i]; q[0] = 0.f; q[1] = 0.f; q[2] = 0.5f; }
    if (slot0)
      for (int i = 0; i < n_env; ++i) act[(size_t)A * i] = 0.f;
    if (graph)   // 열에 여덟은 유효 이웃 중에서 (무작위 24 칸이면 거의 빈 칸이라 안 움직임), 나머지는 아무 번호 (빈 칸 길도 지나게)
      for (int i = 0; i < n_env; ++i) {
        int v[mdp::graph::NB], nv = 0;
        for (int k = 0; k < mdp::graph::NB; ++k)
          if (gnc[(size_t)i * mdp::graph::NB + k] >= 0) v[nv++] = k;
        if (nv > 0 && dm::splitmix64(rng) % 10 < 8) act[(size_t)A * i] = (float)v[dm::splitmix64(rng) % (uint64_t)nv];
      }
    std::vector<float> gact = act;
    if (graph && negative && !negative_person && t >= n_steps / 2 && !neg_done)   // 유효 이웃이 둘 이상인 첫 환경에서 GPU 쪽만 다른 유효 이웃으로
      for (int i = 0; i < n_env && !neg_done; ++i) {
        int v[mdp::graph::NB], nv = 0;
        for (int k = 0; k < mdp::graph::NB; ++k)
          if (gnc[(size_t)i * mdp::graph::NB + k] >= 0) v[nv++] = k;
        if (nv < 2) continue;
        const int a0 = (int)act[(size_t)A * i];
        gact[(size_t)A * i] = (float)(v[0] == a0 ? v[1] : v[0]);
        neg_done = 1;
        std::printf("일부러 어긋나게: 걸음 %d 환경 %d 의 GPU 쪽 이웃 번호 %d → %d\n", t, i, (int)act[(size_t)A * i], (int)gact[(size_t)A * i]);
      }
    if (negative_person && t == n_steps / 2) {   // 사람이 있는 첫 환경의 사람 0 을 1 cm
      gpu.get_state(sg);
      for (int i = 0; i < n_env; ++i)
        if (sg[i].people.n > 0) { sg[i].people.cyl[0].x += 0.01f; gpu.set_state(i, sg[i]); break; }
    }
    if (!graph && negative && !negative_person && t == n_steps / 2) gact[0] = hc.cfg.action_mode == sim::kActPath ? (float)(((int)gact[0] + 1) % mdp::hier::K) : hc.cfg.action_mode == sim::kActVelocity ? (gact[0] > 0.f ? -gact[0] : 0.5f) : (float)(((int)gact[0] + 1) % (hc.cfg.n_dir * hc.cfg.n_dist));
    cudaMemcpy(d_act, gact.data(), sizeof(float) * A * n_env, cudaMemcpyHostToDevice);
    gpu.step(d_act);
    cpu.step(act.data());
    gpu.sync();
    gpu.get_state(sg);
    cpu.get_state(sc);
    gpu.get_reward(rg, dg);
    cpu.get_reward(rc, dc);
    gpu.get_terms(tg);
    cpu.get_terms(tc);
    gpu.get_ep(eg);
    cpu.get_ep(ec);
    gpu.get_obs(ig, og1, pg);
    cpu.get_obs(ic, oc1, pc);
    graph_cmp(t);
    if (graph) {
      for (int i = 0; i < n_env; ++i) g_moved += dc[i] == 0 && gcc[i] != cur_prev[i];
      cur_prev = gcc;
    }
    for (int i = 0; i < n_env; ++i) {
      const bool s_ok = std::memcmp(&sg[i], &sc[i], sizeof(sim::EnvState)) == 0;
      bool r_ok = std::memcmp(&rg[i], &rc[i], 4) == 0 && dg[i] == dc[i] && std::memcmp(&tg[(size_t)K * i], &tc[(size_t)K * i], 4 * K) == 0;
      if (dc[i] != 0) r_ok = r_ok && std::memcmp(&eg[i], &ec[i], sizeof(sim::EpInfo)) == 0;
      bad_state += !s_ok;
      bad_reward += !r_ok;
      if ((!s_ok || !r_ok) && first_bad < 0) first_bad = t;
      ends[dc[i]]++;
      reward_sum += rc[i];
      for (int k = 0; k < K; ++k) term_sum[k] += tc[(size_t)K * i + k];
      // 안전층이 앞 명령을 0 으로 막은 걸음 (정규 v > 0 을 줬는데 명령 v == 0)
      if (hc.cfg.safety && act[(size_t)A * i] > 0.f && sc[i].cmd_v == 0.f && dc[i] == 0) ++safety_zero;
      people_sum += sc[i].people.n;
      hit_person += sc[i].people.hit_person;
    }
    size_t bo = 0;
    for (size_t i = 0; i < ig.size(); ++i) bo += ig[i] != ic[i];
    for (size_t i = 0; i < og1.size(); ++i) bo += std::memcmp(&og1[i], &oc1[i], 4) != 0;
    for (size_t i = 0; i < pg.size(); ++i) bo += std::memcmp(&pg[i], &pc[i], 4) != 0;
    bad_obs += bo;
    if (bo && first_bad < 0) first_bad = t;
    // 범위
    for (size_t i = 0; i < oc1.size(); ++i) {
      const float v = oc1[i];
      float lo, hi;
      range_1d((int)(i % gpu.batch().o1d), &lo, &hi);
      if (!std::isfinite(v)) ++n_nan;
      else if (v < lo || v > hi) {
        if (first_range < 0) {
          first_range = t;
          std::printf("범위 밖: 걸음 %d 환경 %zu 칸 %zu 값 %g (허용 %g~%g)\n", t, i / gpu.batch().o1d, i % gpu.batch().o1d, v, lo, hi);
        }
        ++bad_range;
      }
    }
    for (size_t i = 0; i < pc.size(); ++i) n_nan += !std::isfinite(pc[i]);
    for (float v : tc) n_nan += !std::isfinite(v);
  }
  std::vector<int16_t> mg, mc;
  gpu.get_map(mg);
  cpu.get_map(mc);
  size_t bad_map = 0, known = 0;
  for (size_t i = 0; i < mg.size(); ++i) { bad_map += mg[i] != mc[i]; known += mc[i] != sim::kUnknown; }
  // 깊이 카메라 장애물 층 (계층형 cam_on): 높이 칸 비트·마지막 장 점이 GPU == CPU
  size_t bad_cam = 0, cam_cells = 0, cam_only = 0;
  {
    std::vector<uint8_t> vg, vc;
    std::vector<float> pg2, pc2;
    gpu.get_camlayer(vg, pg2);
    cpu.get_camlayer(vc, pc2);
    bad_cam += vg.size() != vc.size() || pg2.size() != pc2.size();
    for (size_t i = 0; i < vg.size() && i < vc.size(); ++i) {
      bad_cam += vg[i] != vc[i];
      cam_cells += vc[i] != 0;
      cam_only += vc[i] != 0 && i < mc.size() && mdp::cell_class(mc[i]) != mdp::kOcc;   // 로봇 지도는 막힘이 아닌데 카메라 층만 막힘
    }
    for (size_t i = 0; i < pg2.size() && i < pc2.size(); ++i) bad_cam += std::memcmp(&pg2[i], &pc2[i], 4) != 0;
    if (!vc.empty())
      std::printf("깊이 카메라 장애물 층: 막힘 칸 %zu (그중 로봇 지도 막힘 아님 %zu), GPU != CPU 칸 %zu\n", cam_cells, cam_only, bad_cam);
  }
  std::vector<uint8_t> seen_g, seen_c;
  gpu.get_seen(seen_g);
  cpu.get_seen(seen_c);
  size_t bad_seen = 0, seen_wall = 0;
  for (size_t i = 0; i < seen_g.size(); ++i) { bad_seen += seen_g[i] != seen_c[i]; seen_wall += seen_c[i] & 1; }
  // 닫힘 규칙의 지나간 길 비트 판 (지나간 칸 · 본 칸 — env.h trail_mark)
  std::vector<uint32_t> trail_g, trail_c;
  gpu.get_trail(trail_g);
  cpu.get_trail(trail_c);
  size_t bad_trail = 0, trail_bits = 0;
  for (size_t i = 0; i < trail_g.size(); ++i) { bad_trail += trail_g[i] != trail_c[i]; trail_bits += __builtin_popcount(trail_c[i]); }
  std::printf("닫힘 규칙 지나간 길 비트 판: 켜진 칸 %zu, 다른 낱말 %zu\n", trail_bits, bad_trail);
  // 마지막 관측 채널마다 0 아닌 값 수 (env 전체)
  size_t nz_near[mdp::obs::NEAR_C] = {}, nz_far[mdp::obs::FAR_C] = {}, depth_valid = 0;
  for (int i = 0; i < n_env; ++i) {
    const uint8_t* I = ic.data() + (size_t)i * mdp::obs::IMG_SIZE;
    for (int ch = 0; ch < mdp::obs::NEAR_C; ++ch)
      for (int k = 0; k < mdp::obs::NEAR_H * mdp::obs::NEAR_W; ++k) nz_near[ch] += I[ch * mdp::obs::NEAR_H * mdp::obs::NEAR_W + k] != 0;
    for (int ch = 0; ch < mdp::obs::FAR_C; ++ch)
      for (int k = 0; k < mdp::obs::FAR_H * mdp::obs::FAR_W; ++k) nz_far[ch] += I[mdp::obs::NEAR_SIZE + ch * mdp::obs::FAR_H * mdp::obs::FAR_W + k] != 0;
    for (int k = 0; k < mdp::obs::DEPTH_SIZE; ++k) depth_valid += I[mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE + k] != 0;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::printf("GPU %s (sm_%d%d), 환경 %d × 행동 %d, 선언 확률 %.3f%s\n", prop.name, prop.major, prop.minor, n_env, n_steps, p_declare,
              negative ? " (일부러 어긋나게)" : "");
  std::printf("판 끝: 선언 성공 %d · 벽 충돌 %d · 끊김 %d · 선언 실패 %d · 사람 충돌 %d · 진행 없음 %d, 행동당 평균 보상 %.4f, 안전층이 앞 명령을 막은 걸음 %zu, 지도에서 아는 칸 %zu, 지금 판들이 본 벽 면 칸 %zu\n",
              ends[1], ends[2], ends[3], ends[4], ends[5], ends[6], reward_sum / (n_env * n_steps), safety_zero, known, seen_wall);
  std::printf("보상 항 합(전체):");
  for (int k = 0; k < K; ++k) std::printf(" %s %.3f", mdp::rew::term_name(k), term_sum[k]);
  std::printf("\n관측 0 아닌 칸 — 가까운 지도:");
  for (int ch = 0; ch < mdp::obs::NEAR_C; ++ch) std::printf(" %zu", nz_near[ch]);
  std::printf(" · 먼 지도:");
  for (int ch = 0; ch < mdp::obs::FAR_C; ++ch) std::printf(" %zu", nz_far[ch]);
  std::printf(" · 깊이 값 있음 %.1f %%\n", 100.0 * depth_valid / std::max<size_t>(1, (size_t)n_env * mdp::obs::DEPTH_SIZE));
  {
    int hist[mdp::cur::kNumStages] = {};
    for (const auto& e : sc) hist[e.stage]++;
    std::printf("환경 단계 분포 (커리큘럼 %s):", hc.cfg.curriculum ? "켬" : "끔");
    for (int k = 0; k < mdp::cur::kNumStages; ++k) std::printf(" %d:%d", k, hist[k]);
    std::printf(", 단계별 맵 수:");
    for (int k = 0; k < mdp::cur::kNumStages; ++k) std::printf(" %d", gpu.batch().stage_off[k + 1] - gpu.batch().stage_off[k]);
    std::printf("\n");
  }
  {
    // 마지막 1D 관측: 세기 칸 중 0 아닌 비율·평균, IMU 칸(가속도 3·기울기 2·자이로 x·y 2) 평균 |값|
    using namespace mdp::obs;
    size_t nz = 0;
    double isum = 0, imu[7] = {};
    for (int i = 0; i < n_env; ++i) {
      const float* O = oc1.data() + (size_t)i * gpu.batch().o1d;
      for (int k = 0; k < INTENSITY; ++k) { nz += O[LIDAR + k] != 0.f; isum += O[LIDAR + k]; }
      for (int k = 0; k < 7; ++k) imu[k] += std::fabs(O[LIDAR + INTENSITY + ACCEL + k]);
    }
    std::printf("세기 칸 0 아님 %.1f %% (평균 %.3f), IMU 평균 |값| ax %.3f ay %.3f az−g %.3f roll %.3f pitch %.3f gx %.3f gy %.3f\n",
                100.0 * nz / ((size_t)n_env * INTENSITY), isum / std::max<size_t>(1, nz), imu[0] / n_env, imu[1] / n_env, imu[2] / n_env,
                imu[3] / n_env, imu[4] / n_env, imu[5] / n_env, imu[6] / n_env);
  }
  std::printf("관측 범위: NaN·inf %zu, 범위 밖 %zu (처음 걸음 %d)\n", n_nan, bad_range, first_range);
  std::printf("상태 다름 %zu, 보상·항·끝·판 정보 다름 %zu, 관측 다름 %zu (처음 걸음 %d), 지도 칸 다름 %zu, 덮음 표시 다름 %zu\n", bad_state, bad_reward,
              bad_obs, first_bad, bad_map, bad_seen);
  std::printf("사람: 환경당 평균 %.2f 명 (상한 %d), 사람 충돌 걸음 %ld\n", (double)people_sum / (n_env * n_steps), people, hit_person);
  if (!dump.empty()) {
    const int S = hc.cfg.size;
    FILE* f = std::fopen(dump.c_str(), "wb");
    std::fprintf(f, "P5\n%d %d\n255\n", S, S);
    for (int y = S - 1; y >= 0; --y)
      for (int x = 0; x < S; ++x) {
        const int16_t v = mc[(size_t)y * S + x];
        const uint8_t p = v == sim::kUnknown ? 128 : (uint8_t)(255 - (v - hc.cfg.l_min) * 255 / (hc.cfg.l_max - hc.cfg.l_min));
        std::fputc(p, f);
      }
    std::fclose(f);
  }
  bool gt_ok = true;
  if (hier) {
    // 계층형 마지막 관측: 후보 수·지금 목표 표시, 경로 있는 환경 수
    using mdp::hier::FEAT; using mdp::hier::F_VALID; using mdp::hier::F_CUR;
    int nc = 0, ncur = 0, npath = 0, ngoal = 0, nmon = 0;
    for (int i = 0; i < n_env; ++i) {
      const float* O = oc1.data() + (size_t)i * gpu.batch().o1d + mdp::obs::POLICY_1D;
      for (int k = 0; k < mdp::hier::K; ++k) { nc += O[k * FEAT + F_VALID] > 0.5f; ncur += O[k * FEAT + F_CUR] > 0.5f; }
      npath += sc[i].p_n > 0;
      ngoal += sc[i].goal_cell >= 0;
      nmon += sc[i].p_mon;
    }
    std::printf("계층형: 환경당 후보 %.2f 개, 지금 목표 표시 %d, 목표 있음 %d · 경로 있음 %d (환경 %d), 이번 판 충돌 감시가 줄인 걸음 합 %d\n", (double)nc / n_env, ncur, ngoal, npath, n_env, nmon);
    // 정답 격자를 안 쓰나: CPU 묶음 사본의 맵 격자를 모두 막힘(0xff)·모두 0 으로 바꿔 경로 계획·추종기 명령을 다시 — 같아야 함
    const sim::EnvBatch& B = cpu.batch();
    std::vector<sim::MapDev> m_blk(B.maps, B.maps + B.n_maps), m_free(B.maps, B.maps + B.n_maps);
    size_t gmax = 0;
    for (int k = 0; k < B.n_maps; ++k) gmax = std::max(gmax, (size_t)B.maps[k].gw * B.maps[k].gh);
    std::vector<uint8_t> g_blk(gmax, 0xff), g_free(gmax, 0);
    for (int k = 0; k < B.n_maps; ++k) { m_blk[k].grid = g_blk.data(); m_free[k].grid = g_free.data(); }
    sim::EnvBatch Bb = B, Bf = B;
    Bb.maps = m_blk.data();
    Bf.maps = m_free.data();
    int n_cmp = 0, n_diff = 0, n_planned = 0, n_mon = 0, n_tol = 0;
    // 깊이 카메라 장애물 층도: 정답 격자를 바꾼 묶음에서 다시 갱신해도 층·점이 같은지 (층은 깊이 영상(장면 삼각형 광선)·지도 자세만 씀)
    int n_cam_diff = 0;
    if (B.cvox) {
      const size_t cs = (size_t)B.cfg.size * B.cfg.size, ps = (size_t)mdp::obs::DEPTH_W * 2;
      for (int i = 0; i < n_env; ++i) {
        std::vector<uint8_t> v0(B.cvox + cs * i, B.cvox + cs * (i + 1));
        std::vector<float> p0(B.cpts + ps * i, B.cpts + ps * (i + 1));
        std::vector<uint8_t> r[3];
        std::vector<float> q[3];
        const sim::EnvBatch* bs[3] = {&B, &Bb, &Bf};
        for (int k = 0; k < 3; ++k) {
          std::copy(v0.begin(), v0.end(), B.cvox + cs * i);
          std::copy(p0.begin(), p0.end(), B.cpts + ps * i);
          sim::hier_camlayer_cpu(*bs[k], sc[i], i);
          r[k].assign(B.cvox + cs * i, B.cvox + cs * (i + 1));
          q[k].assign(B.cpts + ps * i, B.cpts + ps * (i + 1));
        }
        n_cam_diff += r[0] != r[1] || r[0] != r[2] || std::memcmp(q[0].data(), q[1].data(), 4 * ps) != 0 || std::memcmp(q[0].data(), q[2].data(), 4 * ps) != 0;
        // 아래 계획·명령 비교는 갱신한 층 위에서 (세 묶음이 같은 층을 씀 — 층 포인터 공유)
      }
      std::printf("정답 격자 안 씀 (깊이 카메라 층): 환경 %d 개 층을 정답 격자 모두 막힘·모두 빈 칸 묶음으로 다시 갱신해 달라진 것 %d\n", n_env, n_cam_diff);
    }
    for (int i = 0; i < n_env; ++i)
      for (int k = 0; k < sim::act_slots(B); ++k) {
        sim::EnvState e0 = sc[i], e1 = sc[i], e2 = sc[i];
        e0.p_n = e1.p_n = e2.p_n = 0;   // 늘 새로 짜게
        sim::hier_plan_cpu(B, e0, i, k);
        sim::hier_plan_cpu(Bb, e1, i, k);
        sim::hier_plan_cpu(Bf, e2, i, k);
        float v0, w0, v1, w1, v2, w2;
        sim::path_follow(B, e0, i, &v0, &w0);
        sim::path_follow(Bb, e1, i, &v1, &w1);
        sim::path_follow(Bf, e2, i, &v2, &w2);
        // 충돌 감시(collision_monitor approach — 지금 스캔만)도 같은 명령에서 같은 배율
        const float k0 = sim::path_monitor_scale(B, e0, i, v0, w0), k1 = sim::path_monitor_scale(Bb, e1, i, v1, w1), k2 = sim::path_monitor_scale(Bf, e2, i, v2, w2);
        ++n_cmp;
        n_planned += e0.p_n > 0;
        if (e0.p_n > 0 && e0.goal_cell >= 0) n_tol += (int)e0.wp[2 * (e0.p_n - 1)] + (int)e0.wp[2 * (e0.p_n - 1) + 1] * B.maps[e0.map].gw != e0.goal_cell && !e0.p_trunc;
        n_mon += k0 < 1.f;
        const bool same_p = std::memcmp(&e0, &e1, sizeof(e0)) == 0 && std::memcmp(&e0, &e2, sizeof(e0)) == 0;
        if (!same_p || v0 != v1 || v0 != v2 || w0 != w1 || w0 != w2 || k0 != k1 || k0 != k2) ++n_diff;
      }
    gt_ok = n_diff == 0 && n_planned > 0 && n_cam_diff == 0;
    if (graph) {   // 그래프 관측도: 정답 격자를 바꾼 묶음에서 다시 만들어 같은지 (버퍼는 세 묶음이 같이 씀 — 만들 때마다 꺼내 비교)
      int n_gdiff = 0;
      const sim::EnvBatch* bs[3] = {&B, &Bb, &Bf};
      std::vector<uint8_t> f0;
      std::vector<uint32_t> e0;
      std::vector<int32_t> c0, n0;
      std::vector<float> q0;
      for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < n_env; ++i) sim::graph_ref(*bs[k], sc[i], i);
        cpu.get_graph(gfc, gec, gcc, ggc, gnc);
        if (k == 0) { f0 = gfc; e0 = gec; c0 = gcc; n0 = gnc; q0 = ggc; }
        else n_gdiff += f0 != gfc || e0 != gec || c0 != gcc || n0 != gnc || std::memcmp(q0.data(), ggc.data(), 4 * q0.size()) != 0;
      }
      const bool again = f0 == gfg && e0 == geg && c0 == gcg && n0 == gng;   // 다시 만든 것 = 걸음 끝 GPU 관측 (같은 상태에서 같은 값)
      gt_ok = gt_ok && n_gdiff == 0 && again;
      std::printf("정답 격자 안 씀 (그래프 관측): 정답 격자 모두 막힘·모두 빈 칸 묶음에서 다시 만들어 달라진 묶음 %d, 다시 만든 것 == 걸음 끝 GPU 관측 %s — %s\n", n_gdiff, again ? "같음" : "다름",
                  n_gdiff == 0 && again ? "통과" : "실패");
    }
    std::printf("정답 격자 안 씀: 환경 × 후보 %d 개 중 경로 짠 것 %d (충돌 감시가 줄인 것 %d), 정답 격자를 모두 막힘·모두 빈 칸으로 바꿔 경로·명령·감시 배율이 달라진 것 %d — %s\n",
                n_cmp, n_planned, n_mon, n_diff, gt_ok ? "통과" : "실패");
    {
      long stops = 0, gave = 0;
      for (int i = 0; i < n_env; ++i) { stops += sc[i].p_stops; gave += sc[i].p_gaveup; }
      std::printf("계획·충돌 검사 일치 %d (계획 반경 %.3f m, 목표 허용 %.2f m — 위 경로 중 목표 둘레 칸에서 끝난 것 %d) · 깊이 카메라 층 %d · 복구 %d (지금 판들 충돌 검사 멈춤 걸음 %ld, 목표 포기 %ld)\n",
                  B.cfg.path_consistent, B.cfg.path_r_plan, B.cfg.path_goal_tol, n_tol, B.cfg.cam_on, B.cfg.rec_on, stops, gave);
    }
    // 지금 스캔 막힘·감시가 켜져 있나 (SIM_HIER_AVOID=0 이면 끔)
    std::printf("사람 피하기: 지금 스캔 막힘 %d · 충돌 감시 %d (T %.1f s, 걸음 %.2f s, 점 > %d)\n", B.cfg.path_scan, B.cfg.mon_on, B.cfg.mon_t, B.cfg.mon_dt, B.cfg.mon_max_points);
  }
  if (graph)
    std::printf("그래프 모드 (노드·간선 칸 기준 %s): 관측 %zu 개(환경 × (판 시작 + 결정)) GPU 와 다른 값 %zu · 관측당 유효 노드 %.1f · 간선 %.1f · 유틸리티 노드 %.2f · 길잡이 노드 %.2f, "
                "현재 노드에 간선 없는 관측 %zu, 결정 사이 현재 노드가 바뀐 수 %zu\n", hc.cfg.graph_stand ? "설 수 있는 칸" : "빈 칸", g_obs, bad_graph, (double)g_valid / g_obs, (double)g_edges / 2 / g_obs,
                (double)g_util / g_obs, (double)g_guide / g_obs, g_noedge, g_moved);
  const bool same = bad_state == 0 && bad_reward == 0 && bad_map == 0 && bad_seen == 0 && bad_trail == 0 && bad_obs == 0 && bad_cam == 0 && bad_graph == 0;
  const bool range_ok = n_nan == 0 && bad_range == 0;
  std::printf("%s\n", same ? "환경 V0 통과: GPU == CPU (비트 같음)" : "다름");
  std::printf("%s\n", range_ok ? "관측 범위 통과" : "관측 범위 실패");
  return negative ? (same ? 1 : 0) : (same && range_ok && gt_ok ? 0 : 1);
}
