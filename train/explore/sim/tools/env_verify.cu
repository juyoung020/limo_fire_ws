// V0·V1 (환경): 같은 맵·씨앗·행동열이면 GPU 커널과 CPU 참조판의 상태·보상(항마다)·끝·판 끝 정보·지도·덮음 표시·관측(2D·1D·가치 전용)이
// 걸음마다 비트까지 같은지. 그리고 관측 값 범위(NaN 없음, 정규화 범위 — mdp/observations.h 머리 주석).
// env_verify <장면.scene> [장면 …] [--list 목록] [--envs 8] [--steps 80] [--declare 0.01] [--negative] [--env config/env.yaml]
//            [--model config/sensor_model.yaml] [--dump 지도.pgm]   (env 0 의 마지막 지도: 0 모름 128, 빈 칸 흼, 막힘 검음)
//   --declare p: 걸음마다 끝 선언을 p 확률로 (선언 성공·실패 끝도 비교에 들어가게)
//   --negative: GPU 쪽 행동 하나를 바꿔 일부러 어긋나게 한다 (검사가 틀림을 잡는지)
//   --people K: 환경마다 사람 상한 K (판마다 0~K 명, people.h). --negative-person: 가운데 걸음에 GPU 쪽 사람 하나를 1 cm 옮긴다
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"

namespace {
// 1D 관측 칸마다 허용 범위 (observations.h 정규화 근거)
void range_1d(int i, float* lo, float* hi) {
  using namespace mdp::obs;
  *lo = -1.f;
  *hi = 1.f;
  if (i < LIDAR) { *lo = 0.f; return; }
  if (i < LIDAR + INTENSITY) { *lo = 0.f; *hi = 1.5f; return; }   // 세기 / 100

  const int k = i - LIDAR - INTENSITY;
  if (k < POSE + 2) { *lo = -2.f; *hi = 2.f; }                       // 시작점 기준 위치 / 12.8 m (지도 25.6 m 안)
  else if (k == SLIP || k == GYRO_Z || k == EKF + 1) { *lo = -2.f; *hi = 2.f; }   // 헛돎은 순간 3.14 넘을 수 있음
  else if (k >= WHEELS4 && k < WHEELS4 + 4) { *lo = -1.5f; *hi = 1.5f; }          // 바퀴 한도 1.11 m/s — v·vy·ω 가 따로 응답해 바뀌는 중엔 합이 잠깐 넘음
  else if (k == WHEEL_V || k == EKF) { *lo = -1.2f; *hi = 1.2f; }
  else if (k == ELAPSED || k == KNOWN_AREA || k == FRONTIER + 3 || k == PROGRESS + 1) { *lo = 0.f; *hi = 1.f; }
  else if (k == FRONTIER) { *lo = 0.f; *hi = 1.5f; }
  else if (k == PROGRESS) { *lo = 0.f; *hi = 2.f; }
  else if (k == HOME) { *lo = 0.f; *hi = 2.9f; }
  else if (k >= LAST_ACT && k < LAST_ACT + 16 && (k - LAST_ACT) % 4 == 3) { *lo = 0.f; *hi = 1.f; }
  else if (k >= ACCEL && k < IMU_TILT + 4) { *lo = -3.f; *hi = 3.f; }   // IMU (observations.h 정규화 — 대개 ±0.6, 잡음 꼬리 여유)
  else if (k >= RESERVED || (k >= MAP_VEL && k < MAP_VEL + 3)) { *lo = 0.f; *hi = 0.f; }   // 📏·예약 = 0
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", dump, list;
  int n_env = 8, n_steps = 80;
  float p_declare = 0.01f;
  bool negative = false;
  bool negative_person = false;
  int people = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--declare" && i + 1 < argc) p_declare = std::stof(argv[++i]);
    else if (a == "--negative") negative = true;
    else if (a == "--negative-person") negative = negative_person = true;
    else if (a == "--people" && i + 1 < argc) people = std::stoi(argv[++i]);
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--list" && i + 1 < argc) list = argv[++i];
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
    std::printf("커리큘럼 규칙 %s (8 판 덮음 평균 ≥ 0.8 → 올림, < 0.4 → 내림, 그 사이·7 판까지 → 그대로, 끝 단계에서 멈춤)\n", ok ? "통과" : "실패");
    if (!ok) return 1;
  }
  sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
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
  for (int t = 0; t < n_steps; ++t) {
    std::vector<float> act((size_t)A * n_env);
    for (int i = 0; i < n_env; ++i) sim::random_action(hc.cfg, rng, &act[(size_t)A * i], p_declare);
    std::vector<float> gact = act;
    if (negative_person && t == n_steps / 2) {   // 사람이 있는 첫 환경의 사람 0 을 1 cm
      gpu.get_state(sg);
      for (int i = 0; i < n_env; ++i)
        if (sg[i].people.n > 0) { sg[i].people.cyl[0].x += 0.01f; gpu.set_state(i, sg[i]); break; }
    }
    if (negative && !negative_person && t == n_steps / 2) gact[0] = hc.cfg.action_mode == sim::kActVelocity ? (gact[0] > 0.f ? -gact[0] : 0.5f) : (float)(((int)gact[0] + 1) % (hc.cfg.n_dir * hc.cfg.n_dist));
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
      range_1d((int)(i % mdp::obs::POLICY_1D), &lo, &hi);
      if (!std::isfinite(v)) ++n_nan;
      else if (v < lo || v > hi) {
        if (first_range < 0) {
          first_range = t;
          std::printf("범위 밖: 걸음 %d 환경 %zu 칸 %zu 값 %g (허용 %g~%g)\n", t, i / mdp::obs::POLICY_1D, i % mdp::obs::POLICY_1D, v, lo, hi);
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
  std::vector<uint8_t> seen_g, seen_c;
  gpu.get_seen(seen_g);
  cpu.get_seen(seen_c);
  size_t bad_seen = 0, seen_wall = 0;
  for (size_t i = 0; i < seen_g.size(); ++i) { bad_seen += seen_g[i] != seen_c[i]; seen_wall += seen_c[i] & 1; }
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
      const float* O = oc1.data() + (size_t)i * POLICY_1D;
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
  const bool same = bad_state == 0 && bad_reward == 0 && bad_map == 0 && bad_seen == 0 && bad_obs == 0;
  const bool range_ok = n_nan == 0 && bad_range == 0;
  std::printf("%s\n", same ? "환경 V0 통과: GPU == CPU (비트 같음)" : "다름");
  std::printf("%s\n", range_ok ? "관측 범위 통과" : "관측 범위 실패");
  return negative ? (same ? 1 : 0) : (same && range_ok ? 0 : 1);
}
