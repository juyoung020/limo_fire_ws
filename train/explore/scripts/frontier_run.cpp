// frontier_run — 프론티어 탐사 기준선(baselines/frontier)을 GPU 시뮬에서 장면 여럿 × 판 여럿 돌리고, 학습 뷰어 실행 폴더로 쓴다.
// 판 하나 = 시뮬 환경 하나(장면마다 환경 --episodes 개를 한꺼번에). 시뮬은 env_api.h 로만 쓴다(get_state·get_map·step).
//
//   frontier_run --scenes a.scene [b.scene …] [--modes nav2,fast] [--episodes 4] [--root train/explore/runs] [--name frontier]
//                [--seed 1] [--env sim/config/env.yaml] [--model sim/config/sensor_model.yaml] [--frontier baselines/frontier/config/frontier.yaml]
//                [--obs-every 5] [--cpu] [--eval-seed]
//   --modes 에 slot0 · slot0_nosafe 를 넣으면 계층형 규칙 정책(늘 0 번 후보, 아래 run_hier)을 같은 판으로
//   --people K: 계층형 판마다 사람 0 ~ K 명 (시험 맵 넓이 = 커리큘럼 단계 2 → 2). 끝 지표는 벽 충돌·사람 충돌을 나누고 충돌 때 명령(앞·제자리 회전·뒤·멈춤)도 셈
//   --eval-seed: 장면 i 의 시뮬 씨앗을 학습 러너 평가와 같은 규칙(train.cpp evaluate: seed · 7919 + 17 + i)으로 — --scenes 를 train --test 와
//                같은 차례로 주면 RL 평가·scripts/hybrid_eval 과 같은 시작점(환경 번호 같으면 같은 시작 칸·방향·바닥). 기본은 옛 규칙(seed · 1000003 + i · 7919)
//   → <root>/<name>_<판>/ (run.json kind "baseline", progress.jsonl, episodes.jsonl, evals/frontier.json, replays/*.xrp)
//
// 판 지표(episodes.jsonl): coverage(끝 덮은 비율), t95(95 % 도달 시각), tail_s(95 % 도달 → 판 끝), floor_coverage, path_len(참 자세 이동 거리),
// collided, outcome·end_reason. 덮음 = 시뮬 정의(sim/include/env.h): 본 벽 면 칸 / 출발점에서 갈 수 있는 바닥에 맞닿은 벽 면 칸 (run.json 에 적음).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"
#include "runfolder.h"

namespace {

constexpr int kLidarN = 248;   // 재생 라이다: ±110°, 248 칸 (mdp/README 라이다 관측과 같은 꼴)
constexpr float kLidarFov = 220.f * (float)M_PI / 180.f, kLidarMax = 8.f;
constexpr float kSuccessCov = 0.95f;   // 성공 기준(덮음) — mdp/README 성공 기준·env.yaml success_coverage 와 같은 값

struct Args {
  std::vector<std::string> scenes, modes{"nav2", "fast"};
  int episodes = 4, obs_every = 5;
  int people = 0;   // 계층형(slot0) 판마다 사람 0 ~ 이 수 (people.h, Env::set_people_max) — 0 이면 사람 없음(옛 판과 같은 씨앗)
  uint64_t seed = 1;
  std::string root = "train/explore/runs", name = "frontier";
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml", frontier = FRONTIER_CONFIG;
  bool cpu = false, eval_seed = false;
  int graph_open = -1;    // 그래프 모드 유틸리티가 안 본 프론티어 칸만 세는지 (0·1, 설정 파일 대신)
  int graph_stand = -1;   // 그래프 모드 노드·간선 칸 기준을 설정 파일 대신 (0 빈 칸 · 1 설 수 있는 칸)
};

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, c)) if (!t.empty()) out.push_back(t);
  return out;
}
std::string stem(const std::string& p) {
  std::string b = rf::basename_of(p);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}

// 판 하나 기록
struct Episode {
  int env = 0;
  bool active = true, declared = false, collided = false, truncated = false, success = false;   // success: 닫힘 성공 규칙이면 시뮬이 지도 닫힘으로 끝낸 판
  std::string reason;
  float t = 0, coverage = 0, floor_cov = 0, t95 = -1, path = 0, known_m2 = 0, cov_at_declare = -1;
  int steps = 0;
  float last_x = 0, last_y = 0;
  bool have_last = false;
  rf::ReplayWriter* rw = nullptr;
};

rf::Json params_json(const fb::Params& p) {
  rf::Json j = rf::Json::object();
  j.set("v_des", p.v_des).set("v_max", p.v_max).set("w_max", p.w_max).set("a_max", p.a_max).set("alpha_max", p.alpha_max)
      .set("lookahead", p.lookahead).set("min_lookahead", p.min_lookahead).set("rot_w", p.rot_w).set("rot_min_angle", p.rot_min_angle)
      .set("approach_dist", p.approach_dist).set("approach_min_v", p.approach_min_v).set("goal_tol", p.goal_tol)
      .set("robot_radius", p.robot_radius).set("inflation_radius", p.inflation_radius).set("cost_scaling", p.cost_scaling)
      .set("planner_tolerance", p.planner_tolerance).set("travel_mult", p.travel_mult).set("allow_unknown", p.allow_unknown).set("obstacle_range", p.obstacle_range).set("footprint_half", rf::Json::array(std::vector<double>{p.fp_half_len, p.fp_half_wid})).set("collision_time", p.collision_time).set("bt_retries", p.bt_retries)
      .set("replan_period", p.replan_period).set("inscribed_escape_cost", p.inscribed_escape_cost)
      .set("planner_period", p.planner_period).set("progress_timeout", p.progress_timeout).set("potential_scale", p.potential_scale)
      .set("gain_scale", p.gain_scale).set("min_frontier_size", p.min_frontier_size).set("blacklist_tol_cells", p.blacklist_tol)
      .set("wait_timeout", p.wait_timeout).set("t_safe", p.t_safe).set("res", p.res).set("ctrl_dt", p.ctrl_dt)
      .set("regulated", p.regulated).set("reg_min_radius", p.reg_min_radius).set("reg_min_speed", p.reg_min_speed)
      .set("vel_lookahead", p.vel_lookahead).set("cost_regulated", p.cost_regulated).set("cost_scaling_dist", p.cost_scaling_dist)
      .set("cost_scaling_gain", p.cost_scaling_gain).set("occ_logodds_min", (int)mdp::kOccL).set("occ_p", mdp::kOccP);
  return j;
}

int run_mode(const Args& a, const std::string& mode) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  const fb::Params P = fb::load_params(a.frontier, a.env, mode);
  // 시뮬 판 끝: 시뮬은 덮음으로 저절로 끝내지 않는다(선언 행동만 — 기준선은 선언을 시뮬에 안 보내고 스스로 끝냄), 안전 시간은 기준선 t_safe 보다 넉넉히.
  // 행동은 물리 단위(기준선 v, ω), 안전층·커리큘럼 끔 (기준선은 Nav2 감속을 스스로 함, 맵은 고르게·오차 배율 1)
  hc.cfg.act_raw = 1;
  hc.cfg.safety = 0;
  hc.cfg.curriculum = 0;
  hc.cfg.max_steps = (int)std::ceil(P.t_safe / P.ctrl_dt) + 100;
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;

  std::vector<sim::World> worlds;
  rf::Json scene_list = rf::Json::array();
  for (auto& s : a.scenes) {
    worlds.push_back(sim::build_world(stem(s), sim::load_scene(s), hc));
    auto& w = worlds.back();
    rf::Json sj = rf::Json::object();
    sj.set("name", w.name).set("file", s).set("gw", w.gw).set("gh", w.gh).set("n_wall", w.n_wall).set("wall_m2", w.n_wall * c.res * c.res)
        .set("area_m2", w.n_floor * c.res * c.res).set("closed", w.closed).set("n_spawn", (unsigned long long)w.spawn.size());
    scene_list.push(sj);
    std::fprintf(stderr, "장면 %s: %d×%d 칸, 벽 면 칸 %d (%.2f m²), 갈 수 있는 바닥 %.1f m², 시작 후보 %zu, 닫힘\n", w.name.c_str(), w.gw, w.gh,
                 w.n_wall, w.n_wall * c.res * c.res, w.n_floor * c.res * c.res, w.spawn.size());
  }

  rf::Json limits = rf::Json::object();
  limits.set("v", P.v_max).set("vy", 0.0).set("w", P.w_max);
  rf::Json meta = rf::Json::object();
  meta.set("kind", "baseline").set("baseline", "frontier (explore_lite + Nav2 Smac2D/RPP 식)").set("mode", mode).set("group", "frontier_" + mode)
      .set("seed", (unsigned long long)a.seed).set("n_envs", a.episodes).set("ctrl_hz", 1.0 / P.ctrl_dt).set("success_coverage", kSuccessCov)
      .set("t_safe_s", P.t_safe).set("action_limits", limits).set("scenes", scene_list).set("params", params_json(P))
      .set("coverage_def", "벽 면 칸 덮음: 출발점에서 갈 수 있는 바닥에 4 이웃으로 맞닿은 막힌 칸 중, 라이다 빔 끝점(참 광선·잡음 섞인 거리, 8 m 안)이 "
                            "한 번이라도 떨어진(끝점 칸 → 센서 쪽 앞 칸 → 뒤 칸) 칸의 비율 (sim env.h wall_credit)")
      .set("floor_coverage_def", "빔이 한 번이라도 지나간(지도 자세 Bresenham) 갈 수 있는 바닥 칸 비율 — 보조 지표")
      .set("end_def", "지도 닫힘(mdp/frontier.h closed): 모름 = 안 건드린 칸, 막힘 = p > 0.65, 막힘을 robot_radius 만큼 팽창하고 설 수 있는 칸 flood fill 이 "
                       "프론티어(모름에 닿은 갈 수 있는 칸)에 안 닿음")
      .set("sim_overrides", "시뮬은 선언 행동으로만 끝냄(기준선은 선언을 안 보냄), 물리 단위 행동·안전층 끔·커리큘럼 끔, max_steps = t_safe + 10 s — 판 끝은 기준선 선언·충돌·t_safe")
      .set("known_area_def", "known_m2: 지도에서 갈 수 있는 칸(mdp/frontier.h cell_class == kFree: 건드렸고 막힘 아님) 넓이")
      .set("replay_lidar", "참 자세에서 정적 장면에 다시 쏜 깨끗한 광선(248 빔 ±110°) — 시뮬 내부의 오차 스캔은 env_api 로 안 나옴")
      .set("sources", "m-explore 712bdd41 explore/src/frontier_search.cpp·explore.cpp·launch/explore.launch; navigation2 humble 3c3db59d "
                      "inflation_layer.hpp:148-162, nav2_smac_planner node_2d.cpp:64-80, regulated_pure_pursuit_controller.cpp:442-454; "
                      "cartographer_ros c138034d ros_map.cc:44-45");
  rf::RunFolder::Options opt;
  opt.every_s = 0;
  const std::string dir = a.root + "/" + a.name + "_" + mode;
  rf::RunFolder run(dir, meta, opt);
  std::fprintf(stderr, "[%s] 실행 폴더 %s (v_des %.2f, v_max %.2f m/s)\n", mode.c_str(), dir.c_str(), P.v_des, P.v_max);

  const int S = c.size, N = a.episodes;
  double total_steps = 0;
  rf::Json eval_rows = rf::Json::array();
  std::vector<double> all_t95;
  for (size_t wi = 0; wi < worlds.size(); ++wi) {
    const sim::World& W = worlds[wi];
    std::vector<sim::World> one{W};
    sim::Env env(hc, one, lidar, N, a.eval_seed ? a.seed * 7919ull + 17 + wi : a.seed * 1000003ull + wi * 7919ull, !a.cpu);
    env.reset();
    env.sync();
    // 재생용 정답 맵·광선
    std::vector<uint8_t> gt((size_t)W.gw * W.gh);
    for (size_t k = 0; k < gt.size(); ++k) gt[k] = (W.grid[k] & 1) ? rf::gt::WALL : ((W.grid[k] & 2) ? rf::gt::FREE : rf::gt::OUTSIDE);
    const sim::SceneView sv = W.scene.view();
    rf::Grid g;
    g.w = W.gw; g.h = W.gh; g.res = c.res; g.ox = W.ox; g.oy = W.oy;
    const std::vector<std::string> cols = {"t", "x", "y", "yaw", "sx", "sy", "syaw", "v", "w", "a_v", "a_w", "coverage", "known_m2", "ev"};
    std::vector<fb::Explorer> ex;
    std::vector<Episode> eps(N);
    std::vector<std::unique_ptr<rf::ReplayWriter>> rws;
    for (int i = 0; i < N; ++i) {
      ex.emplace_back(P, W.gw, W.gh, S, W.ox, W.oy);
      eps[i].env = i;
      rws.emplace_back(new rf::ReplayWriter(cols, P.ctrl_dt, g, gt.data()));
      rws.back()->set_lidar(kLidarN, -0.5f * kLidarFov, kLidarFov / (kLidarN - 1), kLidarMax);
      rws.back()->set_limits(P.v_max, 0.f, P.w_max);
      eps[i].rw = rws.back().get();
    }
    float* d_act = nullptr;
    if (!a.cpu) cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * N);
    std::vector<sim::EnvState> st;
    std::vector<int16_t> L;
    std::vector<uint8_t> seen;
    std::vector<float> reward, act((size_t)mdp::act::DIM * N);
    std::vector<int32_t> done;
    std::vector<int8_t> obs((size_t)W.gw * W.gh);
    std::vector<uint8_t> cov((size_t)W.gw * W.gh);
    std::vector<float> ranges(kLidarN);
    int n_active = N, step = 0;
    auto finish = [&](Episode& e, const std::string& outcome) {
      e.active = false;
      --n_active;
      rf::Json line = rf::Json::object();
      const long long id = run.new_episode_id();
      line.set("iter", (int)wi).set("env_steps", total_steps).set("env", e.env).set("ep", id).set("map", W.name).set("stage", mode)
          .set("outcome", outcome).set("success", outcome == rf::outcome::SUCCESS).set("declared", e.declared).set("collided", e.collided)
          .set("truncated", e.truncated).set("t", e.t).set("steps", e.steps).set("coverage", e.coverage).set("path_len", e.path)
          .set("known_m2", e.known_m2).set("area_m2", W.n_floor * c.res * c.res).set("n_wall", W.n_wall)
          .set("floor_coverage", e.floor_cov).set("end_reason", e.reason).set("mode", mode);
      if (e.t95 >= 0) line.set("t95", e.t95).set("tail_s", e.t - e.t95);   // 95 % 도달 → 판 끝(선언·충돌·안전 시간)
      if (e.declared) line.set("coverage_at_declare", e.cov_at_declare);
      const fb::Stats& s = ex[e.env].stats();
      rf::Json bs = rf::Json::object();
      bs.set("plans", s.n_plans).set("goals", s.n_goals).set("reached", s.n_reached).set("aborted", s.n_aborted).set("timeouts", s.n_timeout)
          .set("astar", s.n_astar).set("blocked_replans", s.n_blocked_replan).set("blacklist", (unsigned long long)ex[e.env].blacklist_size())
          .set("collision_stops", s.n_collision_stops).set("recoveries", s.n_recoveries).set("last_frontiers", s.n_frontiers).set("last_frontiers_small", s.n_frontier_small);
      line.set("baseline", bs);
      const std::string path = run.replay_path(id, outcome);
      line.set("replay", rf::basename_of(path));
      const size_t bytes = e.rw->finish(path, line);
      run.episode(line);
      std::fprintf(stderr, "  [%s %s env %d] %s (%s) t %.1f s, 덮음 %.3f, t95 %s, 이동 %.1f m, 목표 %d·도착 %d·실패 %d·시간초과 %d, 재생 %.1f MB\n",
                   mode.c_str(), W.name.c_str(), e.env, outcome.c_str(), e.reason.c_str(), e.t, e.coverage,
                   e.t95 >= 0 ? std::to_string(e.t95).c_str() : "—", e.path, s.n_goals, s.n_reached, s.n_aborted, s.n_timeout, bytes / 1e6);
    };
    while (n_active > 0) {
      env.get_state(st);
      env.get_map(L);
      env.get_seen(seen);
      std::fill(act.begin(), act.end(), 0.f);
      for (int i = 0; i < N; ++i) {
        Episode& e = eps[i];
        if (!e.active) continue;
        const sim::EnvState& s = st[i];
        const int16_t* Li = L.data() + (size_t)i * S * S;
        const float t = (float)s.step * P.ctrl_dt;
        const float px = s.x + s.ex, py = s.y + s.ey, pth = s.th + s.eth;
        // 지표
        if (e.have_last) e.path += std::hypot(s.x - e.last_x, s.y - e.last_y);
        e.last_x = s.x; e.last_y = s.y; e.have_last = true;
        e.t = t;
        e.steps = s.step;
        e.coverage = s.coverage;
        e.floor_cov = s.floor_cov;
        uint32_t evb = 0;
        if (e.t95 < 0 && s.coverage >= kSuccessCov) { e.t95 = t; evb |= rf::ev::REACHED95; }
        int n_free = 0;
        for (int y = 0; y < W.gh; ++y)
          for (int x = 0; x < W.gw; ++x) {
            const int16_t l = Li[(size_t)y * S + x];
            const size_t k = (size_t)y * W.gw + x;
            n_free += mdp::cell_class(l) == mdp::kFree;
            cov[k] = seen[(size_t)i * S * S + (size_t)y * S + x] & 1;   // 본 벽 면 칸 (덮음 분자)
            if (step % a.obs_every == 0)
              obs[k] = l == sim::kUnknown ? (int8_t)-1 : (int8_t)std::lround(100.0 / (1.0 + std::exp(-(double)l / 1000.0)));
          }
        e.known_m2 = n_free * c.res * c.res;
        // 기준선 행동
        const fb::Action ac = ex[i].step(Li, px, py, pth, t);
        const bool end = ex[i].phase() == fb::Phase::Done;
        if (end) {
          e.reason = ex[i].end_reason();
          if (ac.declare) { e.declared = true; e.cov_at_declare = s.coverage; evb |= rf::ev::DECLARE; }
          else { e.truncated = true; evb |= rf::ev::TRUNCATED; }
        }
        act[mdp::act::DIM * i] = end ? 0.f : ac.v;
        act[mdp::act::DIM * i + 1] = 0.f;
        act[mdp::act::DIM * i + 2] = end ? 0.f : ac.w;
        act[mdp::act::DIM * i + 3] = 0.f;
        // 재생 프레임
        const float row[] = {t, s.x, s.y, s.th, px, py, pth, s.drive.wv, s.drive.ww, act[mdp::act::DIM * i], act[mdp::act::DIM * i + 2], s.coverage, e.known_m2, (float)evb};
        e.rw->frame(row);
        if (step % a.obs_every == 0 || end) {
          if (step % a.obs_every != 0)
            for (int y = 0; y < W.gh; ++y)
              for (int x = 0; x < W.gw; ++x) {
                const int16_t l = Li[(size_t)y * S + x];
                obs[(size_t)y * W.gw + x] = l == sim::kUnknown ? (int8_t)-1 : (int8_t)std::lround(100.0 / (1.0 + std::exp(-(double)l / 1000.0)));
              }
          e.rw->obs(obs.data());
        }
        e.rw->covered(cov.data());
        {
          float sn, cs;
          sincosf(s.th, &sn, &cs);
          const sim::V3 o{s.x + cs * c.mount_x - sn * c.mount_y, s.y + sn * c.mount_x + cs * c.mount_y, c.lidar_z};
          for (int k = 0; k < kLidarN; ++k) {
            const float aa = s.th - 0.5f * kLidarFov + k * (kLidarFov / (kLidarN - 1));
            uint16_t obj;
            const float r = sim::raycast(sv, o, sim::V3{std::cos(aa), std::sin(aa), 0.f}, kLidarMax, &obj);
            ranges[k] = r;
          }
          e.rw->scan(ranges.data());
        }
        if (end) {
          if (c.success_closed) {
            // 닫힘 성공 규칙: 성공은 시뮬이 지도 닫힘으로 판을 끝낼 때만. 기준선이 스스로 끝냄(남은 덩어리 모두 포기 등) = 실패("포기", 진행 없음으로 셈)
            e.declared = false;
            finish(e, rf::outcome::NO_PROGRESS);
          } else {
            e.success = e.declared && s.coverage >= kSuccessCov;
            finish(e, e.declared ? (s.coverage >= kSuccessCov ? rf::outcome::SUCCESS : rf::outcome::DECLARED_SHORT) : rf::outcome::TRUNCATED);
          }
        }
      }
      if (n_active == 0) break;
      if (a.cpu) env.step(act.data());
      else {
        cudaMemcpy(d_act, act.data(), sizeof(float) * mdp::act::DIM * N, cudaMemcpyHostToDevice);
        env.step(d_act);
      }
      env.sync();
      env.get_reward(reward, done);
      total_steps += N;
      ++step;
      for (int i = 0; i < N; ++i) {
        Episode& e = eps[i];
        if (!e.active || done[i] == 0) continue;
        // 시뮬이 끝낸 판(충돌). 끝난 판은 같은 커널에서 바로 리셋되므로 지표는 직전 걸음 값
        if (done[i] == 2 && std::getenv("FRONTIER_DEBUG")) {
          const sim::EnvState& s = st[i];   // 부딪히기 직전 상태
          const int16_t* Li = L.data() + (size_t)i * S * S;
          float bd = 1e9f; int bk = -1;
          for (int k = 0; k < W.gw * W.gh; ++k) if (W.grid[k] & 1) { const float dx = W.ox + (k % W.gw + 0.5f) * c.res - s.x, dy = W.oy + (k / W.gw + 0.5f) * c.res - s.y; const float d = std::hypot(dx, dy); if (d < bd) { bd = d; bk = k; } }
          std::fprintf(stderr, "    충돌 직전: 참 (%.2f %.2f %.2f) 오차 (%.3f %.3f %.3f) 행동 (%.2f %.2f) 바퀴 (%.2f %.2f) 목표 (%.2f %.2f) 가장 가까운 정답 벽 %.3f m, 그 칸 지도 l %d\n", s.x, s.y, s.th, s.ex, s.ey, s.eth,
                       act[mdp::act::DIM * i], act[mdp::act::DIM * i + 2], s.drive.wv, s.drive.ww, ex[i].goal_x(), ex[i].goal_y(), bd, (int)Li[(size_t)(bk / W.gw) * S + bk % W.gw]);
        }
        if (done[i] == 2) { e.collided = true; e.reason = "collision"; finish(e, rf::outcome::COLLISION); }
        else if (done[i] == mdp::cur::kDeclareOk && c.success_closed) {   // 지도 닫힘 = 성공 (직전 걸음 덮음)
          e.success = true;
          e.reason = "map_closed";
          finish(e, rf::outcome::SUCCESS);
        }
        else if (done[i] == mdp::cur::kNoProgress) { e.reason = "no_progress"; finish(e, rf::outcome::NO_PROGRESS); }
        else { e.truncated = true; e.reason = "sim_done_" + std::to_string(done[i]); finish(e, rf::outcome::TRUNCATED); }
      }
      if (step % 600 == 0) {
        int a_n = 0;
        double cv = 0;
        for (int i = 0; i < N; ++i) if (eps[i].active) { ++a_n; cv += eps[i].coverage; }
        std::fprintf(stderr, "  [%s %s] %.0f s, 진행 중 %d 판, 평균 덮음 %.3f\n", mode.c_str(), W.name.c_str(), step * P.ctrl_dt, a_n, a_n ? cv / a_n : 0.0);
      }
    }
    if (d_act) cudaFree(d_act);
    // 이 장면 집계 → progress 한 줄 + 평가 표 한 줄
    double n = N, succ = 0, coll = 0, trunc = 0, decl = 0, ds = 0, cv = 0, cad = 0, t95s = 0, t95n = 0, tails = 0, fcv = 0, pl = 0, km = 0, len = 0;
    double s_cov = 0, s_t = 0, s_low = 0, s_low8 = 0, gaveup = 0;   // 성공 판의 덮음·시간, 성공인데 덮음 < 0.95·< 0.8, 기준선 포기
    for (auto& e : eps) {
      if (e.success) { s_cov += e.coverage; s_t += e.t; s_low += e.coverage < kSuccessCov; s_low8 += e.coverage < 0.8f; }
      gaveup += !e.success && !e.collided && e.reason != "no_progress" && e.reason.rfind("sim_done", 0) != 0 && !e.truncated && e.reason != "map_closed";
      succ += e.success;
      ds += e.declared && e.coverage < kSuccessCov;
      coll += e.collided;
      trunc += e.truncated;
      decl += e.declared;
      cv += e.coverage;
      if (e.declared) cad += e.cov_at_declare;
      if (e.t95 >= 0) { t95s += e.t95; t95n++; tails += e.t - e.t95; all_t95.push_back(e.t95); }
      fcv += e.floor_cov;
      pl += e.path;
      km += e.known_m2;
      len += e.steps;
    }
    auto& r = run.row();
    r.last(rf::key::ITER, (double)wi);
    r.last(rf::key::STEPS, total_steps);
    r.sum(rf::key::N_EPISODES, n);
    r.mean_w(rf::key::SUCCESS_RATE, succ / n, n);
    r.mean_w(rf::key::COLLISION_RATE, coll / n, n);
    r.mean_w(rf::key::TRUNCATION_RATE, trunc / n, n);
    r.mean_w(rf::key::DECLARE_RATE, decl / n, n);
    r.mean_w(rf::key::DECLARE_SHORT_RATE, ds / n, n);
    r.mean_w(rf::key::COVERAGE_MEAN, cv / n, n);
    if (decl > 0) r.mean_w(rf::key::COVERAGE_AT_DECLARE, cad / decl, decl);
    if (t95n > 0) r.mean_w(rf::key::T95_MEAN, t95s / t95n, t95n);
    r.mean_w(rf::key::T95_REACHED_RATE, t95n / n, n);
    r.mean_w(rf::key::PATH_LEN_MEAN, pl / n, n);
    r.mean_w(rf::key::KNOWN_AREA_MEAN, km / n, n);
    r.mean_w(rf::key::EP_LEN_MEAN, len / n, n);
    if (t95n > 0) r.last(rf::key::EVAL_FRONTIER_T95, t95s / t95n);   // Training 탭 t95 그림의 "frontier baseline" 선
    run.end_update();
    rf::Json row = rf::Json::object();
    row.set("eval", W.name).set("split", mode).set("n", n).set("success", succ / n).set("collision", coll / n).set("truncation", trunc / n)
        .set("cov_at_success", succ > 0 ? s_cov / succ : 0.0).set("t_success", succ > 0 ? s_t / succ : 0.0).set("closed_low_gt", s_low / n).set("closed_lt80", s_low8 / n).set("gave_up", gaveup / n).set("declare_short", ds / n).set("coverage", cv / n).set("t95_reached", t95n / n).set("path_len", pl / n)
        .set("pass", coll == 0 && t95n == n);
    row.set("floor_coverage", fcv / n);
    if (t95n > 0) row.set("t95", t95s / t95n).set("ref_t95", t95s / t95n).set("tail_s", tails / t95n);
    eval_rows.push(row);
    run.event("scene " + W.name + " done: " + std::to_string((int)t95n) + "/" + std::to_string(N) + " reached 95 %");
  }
  rf::Json ev = rf::Json::object();
  ev.set("iter", (int)worlds.size() - 1).set("ckpt", "frontier_" + mode).set("rows", eval_rows);
  run.eval("frontier", ev);
  rf::Json extra = rf::Json::object();
  if (!all_t95.empty()) {
    double s = 0;
    for (double v : all_t95) s += v;
    extra.set("frontier_t95_mean", s / all_t95.size());
  }
  run.finish(extra);
  return 0;
}

// 계층형 규칙 정책 (mdp/README "계층형" 단계 1 비교): 시뮬 계층형 모드(kActPath — 결정 1 s 마다 후보 하나 → 로봇 지도 경로 + RPP 식 추종)를
// 규칙으로 몬다. 판 = 기준선 판과 같은 씨앗 규칙(장면 차례·환경 번호가 같으면 같은 출발). 판 끝은 시뮬(닫힘 성공·충돌·진행 없음·안전 시간).
//   slot0        : 늘 0 번 후보(BFS 가장 가까운 덩어리), 안전층 켬 (정책이 실물에서 쓰는 설정)
//   slot0_nosafe : 같은데 안전층 끔 (기준선 frontier_run 과 같은 설정)
// 그래프 모드(mdp/graph.h — 행동 = 현재 노드의 이웃 번호) 규칙 정책 둘. 판 조건·판 끝·하위 주행은 slot0 과 같다.
//   graph_guide  : 간선 길이로 가장 가까운 유틸리티 노드로 가는 최단 경로의 첫 이웃 (= 지역 길잡이 위 이웃). 유틸리티 노드가 없으면 무작위 유효 이웃
//   graph_max    : 같은데 목표가 유틸리티가 가장 큰 노드 (같으면 가까운 쪽) — 가까운 노드부터 훑는 graph_guide 와 견줄 것
//   graph_front  : 견줌용 — 그래프 관측 밖 정보(계층형 0 번 후보 칸 = slot0 이 가는 곳)를 써서 그 칸에 가장 가까운 노드 쪽 첫 이웃.
//                  "이웃 노드 고르기 + 같은 하위 주행" 으로 slot0 만큼 되는지(행동 공간·주행 몫)를 관측 몫과 갈라 보려는 것
//   graph_random : 무작위 유효 이웃 (바닥선). 유효 이웃이 없으면 0 번 — 시뮬이 제자리 회전
int run_hier(const Args& a, const std::string& mode) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  const fb::Params P = fb::load_params(a.frontier, a.env, "fast");   // 안전 시간만 기준선과 같게
  hc.cfg.action_mode = sim::kActPath;
  hc.cfg.curriculum = 0;
  if (mode == "slot0_nosafe") hc.cfg.safety = 0;
  const bool graph = mode.rfind("graph_", 0) == 0, by_front = mode == "graph_front", guide = mode == "graph_guide" || mode == "graph_max" || by_front, by_max = mode == "graph_max";
  if (graph) hc.cfg.graph_on = 1;
  if (graph && a.graph_stand >= 0) hc.cfg.graph_stand = a.graph_stand;
  if (graph && a.graph_open >= 0) hc.cfg.graph_open = a.graph_open;
  const float dec_s = (float)(hc.cfg.substeps * hc.cfg.decide_steps) * hc.cfg.dt;
  hc.cfg.max_steps = (int)std::ceil(P.t_safe / (hc.cfg.substeps * hc.cfg.dt)) + 100;
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  rf::Json meta = rf::Json::object();
  meta.set("kind", "baseline").set("baseline", "hierarchical rule policy (path mode, slot 0)").set("mode", mode).set("group", "hier_" + mode)
      .set("seed", (unsigned long long)a.seed).set("n_envs", a.episodes).set("ctrl_hz", 1.0 / dec_s).set("safety", c.safety)
      .set("path_v", c.path_v).set("t_safe_s", P.t_safe);
  rf::RunFolder::Options opt;
  opt.every_s = 0;
  const std::string dir = a.root + "/" + a.name + "_" + mode;
  rf::RunFolder run(dir, meta, opt);
  const int N = a.episodes;
  rf::Json eval_rows = rf::Json::array();
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    const sim::World& W = one[0];
    sim::Env env(hc, one, lidar, N, a.seed * 1000003ull + wi * 7919ull, !a.cpu);
    env.reset();
    env.sync();
    if (a.people > 0) {   // 사람 상한은 판 시작 인자 — 놓고 다시 리셋 (people 0 이면 안 건드려 옛 판과 같은 씨앗)
      env.set_people_max(std::vector<int>(N, a.people));
      env.reset();
      env.sync();
    }
    float* d_act = nullptr;
    std::vector<float> act((size_t)mdp::act::DIM * N, 0.f);   // 늘 0 번
    if (!a.cpu) {
      cudaMalloc(&d_act, sizeof(float) * act.size());
      cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    }
    std::vector<int> fin(N, 0);
    std::vector<sim::EpInfo> ep, first(N);
    std::vector<float> rew;
    std::vector<int32_t> done;
    std::vector<double> ret(N, 0.0);
    std::vector<sim::EnvState> st;
    std::vector<int> reached(N, 0), dec(N, 0);   // 목표 칸 0.25 m 안에 서 있던 결정 수 (도착했는데 덩어리가 안 풀림 — "가도 안 풀리는 프론티어")
    std::vector<int> n_plans(N, 0), n_fb(N, 0);   // 판 동안 경로 계획 수 · 그중 느슨(지도 막힘만)으로 짠 수 (약한 막힘을 피해 못 닿을 때)
    std::vector<int> n_mon(N, 0);   // 판 동안 충돌 감시(collision_monitor approach)가 명령을 줄인 걸음 수
    // 그래프 규칙 정책 + 실패 원인 집계
    std::vector<uint8_t> gf;
    std::vector<uint32_t> ge;
    std::vector<int32_t> gc, gn;
    std::vector<float> gg;
    std::vector<int> n_noutil(N, 0), n_nonb(N, 0), n_offguide(N, 0), n_same(N, 0), n_back(N, 0), last_util(N, 0), last_valid(N, 0), cur1(N, -1), cur2(N, -1);
    std::vector<double> dist_m(N, 0.0);
    std::vector<int32_t> hcand;   // graph_front: 계층형 후보 칸
    std::vector<float> px(N), py(N);
    uint64_t prng = a.seed * 0x9E3779B97F4A7C15ull + wi * 104729ull + 17;
    env.get_state(st);
    for (int i = 0; i < N; ++i) { px[i] = st[i].x; py[i] = st[i].y; }
    const auto choose = [&]() {
      using namespace mdp::graph;
      env.get_graph(gf, ge, gc, gg, gn);
      if (by_front) env.get_cand(hcand);
      std::vector<int> dist(mdp::graph::N);
      for (int i = 0; i < N; ++i) {
        if (fin[i]) continue;
        const uint8_t* F = &gf[(size_t)i * mdp::graph::N * FEAT];
        const uint32_t* E = &ge[(size_t)i * mdp::graph::N];
        const int cur = gc[i];
        int v[NB], nv = 0, nu = 0, nval = 0;
        for (int k = 0; k < NB; ++k) if (gn[(size_t)i * NB + k] >= 0) v[nv++] = k;
        for (int j = 0; j < mdp::graph::N; ++j) { nu += F[j * FEAT + F_UTIL] != 0; nval += F[j * FEAT + F_VALID] != 0; }
        last_util[i] = nu;
        last_valid[i] = nval;
        n_same[i] += cur == cur1[i];
        n_back[i] += cur != cur1[i] && cur == cur2[i];   // 두 결정 전 노드로 되돌아옴 (왔다 갔다)
        cur2[i] = cur1[i];
        cur1[i] = cur;
        int k_sel = 0;
        if (nv == 0) n_nonb[i] += 1;
        else if (!guide || (nu == 0 && !by_front)) {
          if (guide) n_noutil[i] += 1;
          k_sel = v[dm::splitmix64(prng) % (uint64_t)nv];
        } else {
          for (int j = 0; j < mdp::graph::N; ++j) dist[j] = mdp::graphgen::kInf;
          dist[cur] = 0;
          for (bool ch = true; ch;) {
            ch = false;
            for (int j = 0; j < mdp::graph::N; ++j) {
              if (!E[j]) continue;
              const int nd = mdp::graphgen::relax(dist.data(), E[j], j);
              if (nd < dist[j]) { dist[j] = nd; ch = true; }
            }
          }
          int best = -1;
          const int tc = by_front ? hcand[(size_t)i * mdp::hier::K] : -1;   // 0 번 후보 칸 (slot0 이 가는 곳)
          if (by_front && tc >= 0) {   // 그 칸에 가장 가까운 노드 (현재 노드에서 닿는 것, 같으면 그래프 거리 짧은 쪽). 현재 노드가 가장 가까우면 이웃 중 가장 가까운 것
            const int step = (int)(NODE_RES / c.res + 0.5f), k0 = v[0];
            int sdy, sdx;
            mdp::graphgen::stencil(k0, &sdy, &sdx);
            const int cx = gn[(size_t)i * NB + k0] % W.gw - sdx * step, cy = gn[(size_t)i * NB + k0] / W.gw - sdy * step;
            const int tx = tc % W.gw, ty = tc / W.gw;
            long long bd = 0;
            for (int pass = 0; pass < 2 && (best < 0 || best == cur); ++pass)
              for (int j = 0; j < mdp::graph::N; ++j) {
                if (j == cur || dist[j] >= mdp::graphgen::kInf || (pass == 1 && dist[j] > 2828)) continue;
                const long long ddx = cx + (j % GW - cur % GW) * step - tx, ddy = cy + (j / GW - cur / GW) * step - ty, d2 = ddx * ddx + ddy * ddy;
                if (pass == 0) {
                  const long long c2 = (long long)(cx - tx) * (cx - tx) + (long long)(cy - ty) * (cy - ty);
                  if (d2 >= c2) continue;   // 현재 노드보다 가까운 것만
                }
                if (best < 0 || best == cur || d2 < bd || (d2 == bd && dist[j] < dist[best])) { best = j; bd = d2; }
              }
          } else
          for (int j = 0; j < mdp::graph::N; ++j)
            if (F[j * FEAT + F_UTIL] != 0 && dist[j] < mdp::graphgen::kInf &&
                (best < 0 || (by_max ? (F[j * FEAT + F_UTIL] > F[best * FEAT + F_UTIL] || (F[j * FEAT + F_UTIL] == F[best * FEAT + F_UTIL] && dist[j] < dist[best])) : dist[j] < dist[best])))
              best = j;
          if (best < 0 || best == cur) { n_noutil[i] += 1; k_sel = v[dm::splitmix64(prng) % (uint64_t)nv]; }
          else {
            int j = best;
            for (int p = mdp::graphgen::pred_of(dist.data(), E[j], j); p >= 0 && p != cur; p = mdp::graphgen::pred_of(dist.data(), E[j], j)) j = p;
            k_sel = -1;
            for (int k = 0; k < NB; ++k) if (((E[cur] >> k) & 1u) && mdp::graphgen::node_nb(cur, k) == j) k_sel = k;
            if (k_sel < 0) throw std::runtime_error("graph_guide: 첫 이웃이 현재 노드의 이웃이 아님");
            if (!by_front) n_offguide[i] += F[j * FEAT + F_LGUIDE] == 0;   // 고른 이웃이 지역 길잡이 표시 밖 (0 이어야 — 관측의 길잡이와 같은 경로)
          }
        }
        act[(size_t)mdp::act::DIM * i] = (float)k_sel;
      }
      if (!a.cpu) cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    };
    int left = N;
    for (int d = 0; left > 0 && d < c.max_steps / c.decide_steps + 10; ++d) {
      if (graph) choose();
      env.step(a.cpu ? act.data() : d_act);
      env.sync();
      env.get_reward(rew, done);
      env.get_ep(ep);
      env.get_state(st);
      env.get_state(st);
      for (int i = 0; i < N; ++i) {
        if (fin[i]) continue;
        ret[i] += rew[i];
        dec[i] += 1;
        if (!done[i]) {
          const sim::EnvState& s = st[i];
          dist_m[i] += std::hypot(s.x - px[i], s.y - py[i]);   // 결정 사이 직선 거리의 합 (끝난 결정의 마지막 1 s 는 빠짐)
          px[i] = s.x;
          py[i] = s.y;
          n_plans[i] = s.p_plans;
          n_fb[i] = s.p_fb;
          n_mon[i] = s.p_mon;
          if (s.p_n > 0) {
            const float gx = W.ox + (s.wp[2 * (s.p_n - 1)] + 0.5f) * c.res, gy = W.oy + (s.wp[2 * (s.p_n - 1) + 1] + 0.5f) * c.res;
            if (std::hypot(gx - (s.x + s.ex), gy - (s.y + s.ey)) <= c.goal_tol) reached[i] += 1;
          }
          continue;
        }
        fin[i] = 1;
        first[i] = ep[i];
        first[i].done = done[i];
        --left;
      }
    }
    if (d_act) cudaFree(d_act);
    double n = N, succ = 0, coll = 0, idle = 0, trunc = 0, ts = 0, cov_s = 0, rsum = 0, stand = 0, pl = 0, fb = 0;
    double d_succ = 0, d_all = 0, g_noutil = 0, g_nonb = 0, g_off = 0, g_same = 0, g_back = 0, f_noutil = 0, f_util = 0, f_nonb_ep = 0;
    double c_wall = 0, c_person = 0, t_all = 0, mon = 0, steps_all = 0, ck[5] = {0, 0, 0, 0, 0};   // 벽·사람 충돌, 판 길이 합, 충돌 때 명령 (EpInfo::coll_kind)
    for (int i = 0; i < N; ++i) { pl += n_plans[i]; fb += n_fb[i]; }
    for (int i = 0; i < N; ++i) {
      const sim::EpInfo& f = first[i];
      const int dd = fin[i] ? f.done : mdp::cur::kTruncated;
      const char* oc = dd == mdp::cur::kDeclareOk ? rf::outcome::SUCCESS
                       : (dd == mdp::cur::kCollision || dd == mdp::cur::kCollisionPerson) ? rf::outcome::COLLISION
                       : dd == mdp::cur::kNoProgress ? rf::outcome::NO_PROGRESS : rf::outcome::TRUNCATED;
      succ += dd == mdp::cur::kDeclareOk;
      coll += dd == mdp::cur::kCollision || dd == mdp::cur::kCollisionPerson;
      c_wall += dd == mdp::cur::kCollision;
      c_person += dd == mdp::cur::kCollisionPerson;
      if (fin[i] && (dd == mdp::cur::kCollision || dd == mdp::cur::kCollisionPerson)) ck[std::min(4, std::max(0, f.coll_kind))] += 1;
      t_all += fin[i] ? f.t : dec[i] * dec_s;
      mon += n_mon[i];
      steps_all += dec[i] * c.decide_steps;
      idle += dd == mdp::cur::kNoProgress;
      trunc += !fin[i] || dd == mdp::cur::kTruncated;
      if (dd == mdp::cur::kDeclareOk) { ts += f.t; cov_s += f.coverage; d_succ += dist_m[i]; }
      d_all += dist_m[i];
      g_noutil += n_noutil[i];
      g_nonb += n_nonb[i];
      g_off += n_offguide[i];
      g_same += n_same[i];
      g_back += n_back[i];
      const bool stuck = dd == mdp::cur::kNoProgress || !fin[i] || dd == mdp::cur::kTruncated;
      f_noutil += graph && stuck && last_util[i] == 0;   // 못 끝낸 판: 마지막 결정 때 유틸리티 노드가 하나도 없었음 (닫힘은 안 됐는데 갈 곳 표시가 없음)
      f_util += graph && stuck && last_util[i] > 0;      // 못 끝낸 판: 유틸리티 노드는 있었음 (못 닿거나 가도 안 풀림)
      f_nonb_ep += graph && n_nonb[i] > 0;
      rsum += ret[i];
      stand += dec[i] > 0 ? (double)reached[i] / dec[i] : 0.0;
      rf::Json line = rf::Json::object();
      line.set("iter", (int)wi).set("env", i).set("ep", run.new_episode_id()).set("map", W.name).set("stage", mode).set("outcome", oc)
          .set("success", dd == mdp::cur::kDeclareOk).set("collided", dd == mdp::cur::kCollision || dd == mdp::cur::kCollisionPerson)
          .set("collided_person", dd == mdp::cur::kCollisionPerson).set("coll_kind", fin[i] ? f.coll_kind : 0)
          .set("t", f.t).set("coverage", f.coverage).set("dist_m", dist_m[i]).set("g_noutil", n_noutil[i]).set("g_nonb", n_nonb[i]).set("g_last_util", last_util[i]).set("reward", ret[i]).set("decisions", dec[i]).set("at_goal_frac", dec[i] ? (double)reached[i] / dec[i] : 0.0)
          .set("mode", mode);
      run.episode(line);
    }
    rf::Json row = rf::Json::object();
    row.set("eval", W.name).set("split", mode).set("n", n).set("success", succ / n).set("collision", coll / n).set("no_progress", idle / n)
        .set("truncation", trunc / n).set("t_success", succ > 0 ? ts / succ : 0.0).set("cov_at_success", succ > 0 ? cov_s / succ : 0.0)
        .set("reward", rsum / n).set("at_goal_frac", stand / n).set("loose_plan_frac", pl > 0 ? fb / pl : 0.0)
        .set("collision_wall", c_wall / n).set("collision_person", c_person / n).set("t_mean", t_all / n).set("monitor_frac", steps_all > 0 ? mon / steps_all : 0.0)
        .set("dist_success_m", succ > 0 ? d_succ / succ : 0.0).set("dist_mean_m", d_all / n)
        .set("coll_fwd", ck[1] / n).set("coll_rotate", ck[2] / n).set("coll_back", ck[3] / n).set("coll_stopped", ck[4] / n);
    eval_rows.push(row);
    std::fprintf(stderr, "[%s %s] 성공 %.3f 충돌 %.3f 진행 없음 %.3f 끊김 %.3f, 성공까지 %.1f s (성공 판 정답 덮음 %.3f), 판 합 보상 %.2f, 목표에 서 있던 결정 비율 %.3f, 느슨 계획 비율 %.3f\n", mode.c_str(),
                 W.name.c_str(), succ / n, coll / n, idle / n, trunc / n, succ > 0 ? ts / succ : 0.0, succ > 0 ? cov_s / succ : 0.0, rsum / n, stand / n, pl > 0 ? fb / pl : 0.0);
    std::fprintf(stderr, "[%s %s] 사람 %d: 벽 충돌 %.3f 사람 충돌 %.3f · 충돌 때 명령 앞 %.3f 제자리 회전 %.3f 뒤 %.3f 멈춤 %.3f · 판 평균 길이 %.1f s · 충돌 감시가 줄인 걸음 비율 %.3f\n", mode.c_str(),
                 W.name.c_str(), a.people, c_wall / n, c_person / n, ck[1] / n, ck[2] / n, ck[3] / n, ck[4] / n, t_all / n, steps_all > 0 ? mon / steps_all : 0.0);
    double dec_all = steps_all / c.decide_steps;
    std::fprintf(stderr, "[%s %s] 이동 거리: 성공 판 평균 %.1f m · 모든 판 평균 %.1f m\n", mode.c_str(), W.name.c_str(), succ > 0 ? d_succ / succ : 0.0, d_all / n);
    if (graph)
      std::fprintf(stderr, "[%s %s] 그래프 (칸 기준 %s · 프론티어 %s): 결정 %.0f 개 중 유틸리티 노드 없어 무작위 %.4f · 유효 이웃 없음 %.4f (그런 판 %.3f) · 현재 노드 그대로 %.3f · 두 결정 전 노드로 되돌아옴 %.3f · 고른 이웃이 길잡이 표시 밖 %.0f 개 | 못 끝낸 판(진행 없음·끊김) 중 마지막에 유틸리티 노드 없음 %.3f · 있음 %.3f (판 비율)\n",
                   mode.c_str(), W.name.c_str(), c.graph_stand ? "설 수 있는 칸" : "빈 칸", c.graph_open ? "막는 칸만" : "모두", dec_all, g_noutil / dec_all, g_nonb / dec_all, f_nonb_ep / n, g_same / dec_all, g_back / dec_all, g_off, f_noutil / n, f_util / n);
  }
  rf::Json ev = rf::Json::object();
  ev.set("iter", (int)a.scenes.size() - 1).set("ckpt", "hier_" + mode).set("rows", eval_rows);
  run.eval("frontier", ev);
  run.finish(rf::Json::object());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " 뒤에 값");
      return argv[++i];
    };
    if (k == "--scenes") { while (i + 1 < argc && argv[i + 1][0] != '-') a.scenes.push_back(argv[++i]); }
    else if (k == "--modes") a.modes = split(nx(), ',');
    else if (k == "--episodes") a.episodes = std::stoi(nx());
    else if (k == "--seed") a.seed = std::stoull(nx());
    else if (k == "--root") a.root = nx();
    else if (k == "--name") a.name = nx();
    else if (k == "--env") a.env = nx();
    else if (k == "--model") a.model = nx();
    else if (k == "--frontier") a.frontier = nx();
    else if (k == "--obs-every") a.obs_every = std::max(1, std::stoi(nx()));
    else if (k == "--people") a.people = std::stoi(nx());
    else if (k == "--graph-stand") a.graph_stand = std::stoi(nx());
    else if (k == "--graph-open") a.graph_open = std::stoi(nx());
    else if (k == "--cpu") a.cpu = true;
    else if (k == "--eval-seed") a.eval_seed = true;
    else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
  }
  if (a.scenes.empty()) {
    std::fprintf(stderr, "frontier_run --scenes a.scene [b.scene …] [--modes nav2,fast] [--episodes 4] [--root runs] [--name frontier] [--seed 1] [--obs-every 5] [--cpu] [--eval-seed]\n");
    return 2;
  }
  try {
    for (auto& m : a.modes) {
      if (m == "slot0" || m == "slot0_nosafe" || m == "graph_guide" || m == "graph_max" || m == "graph_front" || m == "graph_random") run_hier(a, m);
      else run_mode(a, m);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "frontier_run: %s\n", e.what());
    return 1;
  }
  return 0;
}
