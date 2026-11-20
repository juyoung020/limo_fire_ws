// env.yaml 읽기 (load_env_cfg · load_people_cfg) — CUDA 없이 빌드되는 호스트 코드. 학습 환경(env_host.cpp)과 실물 노드(limo_fire_explore)가 같이 쓴다
#include "env_cfg.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace sim {

namespace {

int16_t log_odds(float p) { return (int16_t)std::lround(1000.0 * std::log(p / (1.0 - p))); }

YAML::Node need(const YAML::Node& n, const char* key, const std::string& where) {
  const YAML::Node v = n[key];
  if (!v) throw std::runtime_error(where + ": " + key + " 없음");
  return v;
}

}  // namespace

EnvHostCfg load_env_cfg(const std::string& path) {
  const YAML::Node y = YAML::LoadFile(path);
  EnvHostCfg h{};   // 0 으로 (안 쓰는 칸 gone_rl_* 도 — 녹화 머리가 이 바이트를 담음)
  EnvCfg& c = h.cfg;
  const YAML::Node t = need(y, "time", path);
  c.dt = t["dt"].as<float>();
  c.lidar_every = t["lidar_every"].as<int>();
  // Nav2 값은 Nav2 설정 파일에서 (같은 값을 두 곳에 두지 않는다)
  const std::string dir = path.substr(0, path.find_last_of('/') + 1);
  const std::string nav_path = dir + need(y, "nav2_params", path).as<std::string>();
  const YAML::Node nav = YAML::LoadFile(nav_path);
  const YAML::Node ctl = nav["controller_server"]["ros__parameters"];
  const YAML::Node rpp = need(ctl, "FollowPath", nav_path);
  c.v_des = rpp["desired_linear_vel"].as<float>();
  c.lookahead = rpp["lookahead_dist"].as<float>();
  c.min_lookahead = rpp["min_lookahead_dist"].as<float>();
  c.rot_w = rpp["rotate_to_heading_angular_vel"].as<float>();
  c.rot_min_angle = rpp["rotate_to_heading_min_angle"].as<float>();
  c.approach_dist = rpp["approach_velocity_scaling_dist"].as<float>();
  c.approach_min_v = rpp["min_approach_linear_velocity"].as<float>();
  c.goal_tol = need(ctl, "general_goal_checker", nav_path)["xy_goal_tolerance"].as<float>();
  c.reg_min_radius = rpp["regulated_linear_scaling_min_radius"].as<float>();   // 계층형 추종기 곡률 감속 (hier_path.h)
  c.reg_min_speed = rpp["regulated_linear_scaling_min_speed"].as<float>();
  c.path_r_out = need(need(nav["global_costmap"]["global_costmap"]["ros__parameters"], "inflation_layer", nav_path), "inflation_radius", nav_path).as<float>();
  {   // 계층형 추종기 RPP 충돌 검사 (hier_path.h path_imminent) — 기준선 frontier_explorer.cpp load_params 와 같은 칸에서
    const YAML::Node gcm = nav["global_costmap"]["global_costmap"]["ros__parameters"];
    c.path_obs_range = need(need(gcm, "obstacle_layer", nav_path), "scan", nav_path)["obstacle_max_range"].as<float>();
    const YAML::Node lcm = nav["local_costmap"]["local_costmap"]["ros__parameters"];
    c.path_fp_pad = lcm["footprint_padding"] ? lcm["footprint_padding"].as<float>() : 0.f;
    c.path_coll_time = need(rpp, "max_allowed_time_to_collision_up_to_carrot", nav_path).as<float>();
  }
  const YAML::Node vs = nav["velocity_smoother"]["ros__parameters"];
  c.v_max = vs["max_velocity"][0].as<float>();
  c.w_max = vs["max_velocity"][2].as<float>();
  c.a_max = vs["max_accel"][0].as<float>();
  c.alpha_max = vs["max_accel"][2].as<float>();

  const YAML::Node body = need(y, "body", path);
  c.half_len = 0.5f * body["footprint"][0].as<float>();
  c.half_wid = 0.5f * body["footprint"][1].as<float>();
  c.mount_x = body["lidar_mount"][0].as<float>();
  c.mount_y = body["lidar_mount"][1].as<float>();
  c.lidar_z = body["lidar_mount"][2].as<float>();
  h.band_lo = body["collision_band"][0].as<float>();
  h.band_hi = body["collision_band"][1].as<float>();
  h.spawn_clearance = body["spawn_clearance"].as<float>();

  const YAML::Node m = need(y, "map", path);
  c.res = m["resolution"].as<float>();
  c.inv_res = 1.f / c.res;
  c.size = m["size"].as<int>();
  if (c.size % 4 != 0) throw std::runtime_error(path + ": map.size 는 4 의 배수");
  c.l_hit = log_odds(m["hit"].as<float>());
  c.l_miss = log_odds(m["miss"].as<float>());
  c.l_min = log_odds(m["clamp"][0].as<float>());
  c.l_max = log_odds(m["clamp"][1].as<float>());
  c.map_max_range = m["max_range"].as<float>();
  c.missing_ray = m["missing_ray"].as<float>();

  const YAML::Node pe = need(y, "pose_error", path);
  c.pe_xy = pe["sigma_xy"].as<float>();
  c.pe_yaw = pe["sigma_yaw"].as<float>();
  c.pe_tau = pe["tau"].as<float>();

  const YAML::Node a = need(y, "action", path);
  const std::string mode = need(a, "mode", path).as<std::string>();
  if (mode != "velocity" && mode != "path") throw std::runtime_error(path + ": action.mode 는 velocity · path (목표점 행동 goal 은 2026-11-18 지움)");
  c.action_mode = mode == "velocity" ? kActVelocity : kActPath;
  const YAML::Node av = need(a, "velocity", path);
  const float period = av["period"].as<float>();   // 계층형도 0.1 s 속도 행동 주기로 시뮬 걸음 (결정은 그 묶음)
  c.substeps = (int)std::lround(period / c.dt);
  if (c.substeps < 1 || std::fabs(c.substeps * c.dt - period) > 1e-4f)
    throw std::runtime_error(path + ": action period 는 time.dt 의 배수");
  // 계층형 (include/hier_path.h): 결정 주기 = 속도 행동 주기의 정수 배, 추종기 직진 속도·장애물 근접 감속 (기준선 fast 판과 같은 값), 경로 무게 경계.
  // 학습기·도구는 --hier 로 action_mode 를 kActPath 로 바꾼다 (이 파일의 기본 mode 는 velocity 그대로 — 기준선이 velocity 를 요구)
  {
    const YAML::Node ap = need(a, "path", path);
    const float dp = ap["period"].as<float>(), pv = av["period"].as<float>();
    c.decide_steps = (int)std::lround(dp / pv);
    if (c.decide_steps < 1 || std::fabs(c.decide_steps * pv - dp) > 1e-4f) throw std::runtime_error(path + ": action.path.period 는 velocity.period 의 배수");
    c.path_v = ap["v"].as<float>();
    c.face_tol = ap["face_tol"] ? ap["face_tol"].as<float>() : 0.15f;
    c.face_w_min = ap["face_w_min"] ? ap["face_w_min"].as<float>() : 0.4f;
    // 돌며 가기 (env.yaml action.path.curved_follow): 켜면 제자리 회전 문턱 · 회전 ω · ω 한도를 nav2_params(기준선 Nav2 판 0.785 · 1.0 · 1.0) 대신 아래 값으로.
    // 끄면 nav2_params 그대로 = 2026-11-17 전 추종기 (turn_gain 맞춤도 꺼짐 — hier_path.h path_follow 가 문턱으로 가름)
    if (ap["curved_follow"] && ap["curved_follow"].as<bool>()) {
      c.rot_min_angle = need(ap, "rotate_to_heading_min_angle", path).as<float>();
      c.rot_w = need(ap, "rotate_to_heading_angular_vel", path).as<float>();
      c.w_max = need(ap, "max_angular_vel", path).as<float>();
      if (!(c.rot_min_angle > 0.8f) || c.w_max > need(y, "drive", path)["max"][1].as<float>())
        throw std::runtime_error(path + ": curved_follow 는 rotate_to_heading_min_angle > 0.8 · max_angular_vel ≤ drive.max ω");
    }
    c.cand_v2 = ap["candidates_v2"] && ap["candidates_v2"].as<bool>() ? 1 : 0;
    if (const char* g = std::getenv("SIM_CAND_V2")) c.cand_v2 = g[0] == '1' ? 1 : 0;   // 앞뒤 비교
    {   // 아래층 계측·단독 과제 (include/low_task.h). 스위치는 도구가 켠다 — 여기선 값만
      const YAML::Node fo = ap["follow"];
      const auto num = [&](const char* k, float d) { return fo && fo[k] ? fo[k].as<float>() : d; };
      c.low_stat = c.low_task = 0;
      c.low_rl = c.low_period = c.low_explore = 0;   // 아래층 걸음 길 (include/low_rl.h) — 도구가 켠다
      c.low_pass_r = num("pass_radius", 0.5f);
      c.low_idle_d = num("stall_dist", 0.3f);
      c.low_idle_steps = (int)std::lround(num("stall_time", 10.f) / pv);
    }
    // 경로를 다시 살피는 주기 (s — low_rl.h lowrl_period, 목표가 바뀌면 바로): 결정 주기와 뗌. 없으면 0 = 결정 주기 (옛 동작)
    if (ap["replan_period"]) {
      c.low_period = (int)std::lround(ap["replan_period"].as<float>() / pv);
      if (c.low_period < 1) throw std::runtime_error(path + ": action.path.replan_period 는 행동 주기 이상");
    }
    c.path_cost_reg = ap["use_cost_regulated_linear_velocity_scaling"].as<bool>() ? 1 : 0;
    c.path_cs_dist = ap["cost_scaling_dist"].as<float>();
    c.path_cs_gain = ap["cost_scaling_gain"].as<float>();
    c.path_r_mid = ap["weight_mid_dist"].as<float>();
    if (const char* g = std::getenv("SIM_WEIGHT_MID")) c.path_r_mid = std::stof(g);   // 앞뒤 비교
    c.path_backup_v = ap["backup_speed"].as<float>();
    c.path_fp_pad += ap["extra_padding"].as<float>();   // nav2 footprint_padding 위에 더 (잠정)
    // 사람·움직이는 장애물 피하기 (hier_path.h "지금 스캔"): 지금 스캔 막힘 · collision_monitor approach. SIM_HIER_AVOID=0 이면 둘 다 끔 (앞뒤 비교)
    {
      const std::string sm = need(ap, "scan_obstacles", path).as<std::string>();
      if (sm != "off" && sm != "map_free" && sm != "all") throw std::runtime_error(path + ": action.path.scan_obstacles 는 off · map_free · all");
      c.path_scan = sm == "off" ? 0 : (sm == "map_free" ? 1 : 2);
    }
    if (const char* g = std::getenv("SIM_SCAN_OBSTACLES")) c.path_scan = std::atoi(g);   // 실험용 (0 · 1 · 2) — 앞뒤 비교
    const YAML::Node cm = need(ap, "collision_monitor", path);
    c.mon_on = cm["enabled"].as<bool>() ? 1 : 0;
    c.mon_t = cm["time_before_collision"].as<float>();
    c.mon_dt = cm["simulation_time_step"].as<float>();
    c.mon_max_points = cm["max_points"].as<int>();
    if (!(c.mon_dt > 0.f) || c.mon_t < c.mon_dt || c.mon_t / c.mon_dt > 64.f) throw std::runtime_error(path + ": collision_monitor 시간 (0 < 걸음 ≤ 시간, 걸음 수 ≤ 64)");
    if (const char* g = std::getenv("SIM_COLLISION_MONITOR")) c.mon_on = g[0] == '1' ? 1 : 0;   // 실험용 — 앞뒤 비교
    if (const char* g = std::getenv("SIM_HIER_AVOID"); g && g[0] == '0') c.path_scan = c.mon_on = 0;
    // 깊이 카메라 장애물 층 (hier_path.h "깊이 카메라 장애물 층"): 절이 없으면 끔 (옛 설정 그대로). SIM_CAM_OBSTACLES=0/1 이면 그 값 (앞뒤 비교)
    c.cam_on = c.cam_nz = 0;
    c.cam_every = 0;
    c.cam_lw = c.cam_lh = 0;
    c.cam_min_h = c.cam_max_h = c.cam_obs_range = c.cam_ray_range = 0.f;
    if (const YAML::Node co = ap["camera_obstacles"]) {
      c.cam_on = co["enabled"].as<bool>() ? 1 : 0;
      c.cam_every = co["update_every_step"] && co["update_every_step"].as<bool>() ? 1 : 0;
      if (const YAML::Node lg = co["layer_grid"]) {   // 층 전용 픽셀 격자 [열, 행] — 없으면 깊이 관측 그대로
        if (lg.size() != 2) throw std::runtime_error(path + ": camera_obstacles.layer_grid 는 [열, 행]");
        c.cam_lw = lg[0].as<int>();
        c.cam_lh = lg[1].as<int>();
        if (c.cam_lw < mdp::obs::DEPTH_W || c.cam_lw % mdp::obs::DEPTH_W != 0 || c.cam_lh < 1 || c.cam_lw * c.cam_lh > (1 << 16))
          throw std::runtime_error(path + ": camera_obstacles.layer_grid 열은 64 의 배수, 픽셀 65536 이하");
      }
      c.cam_min_h = co["min_obstacle_height"].as<float>();
      c.cam_max_h = co["max_obstacle_height"].as<float>();
      c.cam_nz = co["z_voxels"].as<int>();
      c.cam_obs_range = co["obstacle_max_range"].as<float>();
      c.cam_ray_range = co["raytrace_max_range"].as<float>();
      if (c.cam_nz < 1 || c.cam_nz > 8 || !(c.cam_max_h > c.cam_min_h))
        throw std::runtime_error(path + ": camera_obstacles (z_voxels 1~8, max_obstacle_height > min_obstacle_height)");
    }
    if (const char* g = std::getenv("SIM_CAM_OBSTACLES")) c.cam_on = (g[0] == '1' && c.cam_nz > 0) ? 1 : 0;
    // 계획·충돌 검사 일치 (hier_path.h "계획·충돌 검사 일치"): 절이 없으면 끔 (옛 동작). SIM_PATH_CONSISTENT=0/1 (앞뒤 비교)
    c.path_consistent = ap["consistent_plan"] && ap["consistent_plan"].as<bool>() ? 1 : 0;
    if (const char* g = std::getenv("SIM_PATH_CONSISTENT")) c.path_consistent = g[0] == '1' ? 1 : 0;
    c.path_fallback = 3;   // 엄격으로 못 짜면 느슨 → 작은 반경 (hier_path.h (e)). 진단용으로 SIM_PATH_FALLBACK=0·1 이면 그 단계를 안 씀 (경로 없음 = 목표만 두고 섬)
    if (const char* g = std::getenv("SIM_PATH_FALLBACK")) c.path_fallback = std::atoi(g);
    {
      // 계획 반경 = 충돌 검사 발자국(footprint + 패딩) 외접 반경 + 칸 대각(칸 가운데끼리 거리로 재는 반경 → 발자국 점이 든 칸까지 · 로봇이 칸 안 어디든 — 칸 반 대각 × 2)
      // 여유 plan_margin (env.yaml action.path.plan_margin, 없으면 칸 대각 — 옛 값). SIM_PLAN_MARGIN 으로 덮음 (앞뒤 비교)
      const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
      float pm = ap["plan_margin"] ? ap["plan_margin"].as<float>() : c.res * std::sqrt(2.f);
      if (const char* g = std::getenv("SIM_PLAN_MARGIN")) pm = std::stof(g);
      if (pm < 0.f || pm > 0.2f) throw std::runtime_error(path + ": action.path.plan_margin 은 0 ~ 0.2 m");
      c.path_r_plan = std::sqrt(hl * hl + hw * hw) + pm;
      c.path_goal_tol = need(need(nav["planner_server"]["ros__parameters"], "GridBased", nav_path), "tolerance", nav_path).as<float>();   // Smac tolerance
      c.path_goal_far = ap["goal_far_tolerance"] ? ap["goal_far_tolerance"].as<float>() : c.path_goal_tol;
      if (c.path_goal_far < c.path_goal_tol || c.path_goal_far > 1.5f) throw std::runtime_error(path + ": goal_far_tolerance 는 tolerance ~ 1.5 m");
      if (c.path_consistent && (c.path_r_plan >= c.path_r_out || c.path_r_plan < 0.2f))
        throw std::runtime_error(path + ": consistent_plan 계획 반경 " + std::to_string(c.path_r_plan) + " 은 0.2 이상 팽창 반경 미만이어야 함");
    }
    // 복구 (hier_path.h "복구"): 절이 없으면 끔. 물러나기 걸음 = BT BackUp backup_dist / backup_speed / 행동 주기. SIM_PATH_RECOVERY=0/1 (앞뒤 비교)
    c.rec_on = c.rec_backup_steps = c.rec_bl_steps = 0;
    c.rec_bl_r = 0.f;
    c.rec_win = c.rec_nobk = c.imm_rot = c.rec_abort = 0;
    {   // '대충 다 앎' 끝 기다림: 0 · 없으면 close_idle_s (옛 동작)
      const float rw = ap["rough_done_wait"] ? ap["rough_done_wait"].as<float>() : 0.f;
      if (rw < 0.f) throw std::runtime_error(path + ": action.path.rough_done_wait 는 0 이상 (s, 0 = close_idle_s)");
      c.rough_wait = rw > 0.f ? std::max(1, (int)std::lround(rw / pv)) : 0;
    }
    c.spin_free = ap["turn_while_driving"] && ap["turn_while_driving"].as<bool>() ? 1 : 0;   // 없으면 제자리 회전 (옛 동작)
    c.spin_v = c.spin_free ? need(ap, "turn_drive_v", path).as<float>() : 0.f;
    if (c.spin_free && !(c.spin_v > 0.f && c.spin_v <= c.path_v)) throw std::runtime_error(path + ": action.path.turn_drive_v 는 (0, v]");
    if (const YAML::Node rc = ap["recovery"]) {
      c.rec_on = rc["enabled"].as<bool>() ? 1 : 0;
      c.rec_backup_steps = (int)std::lround(rc["backup_dist"].as<float>() / c.path_backup_v / pv);
      c.rec_bl_steps = (int)std::lround(rc["blacklist_time"].as<float>() / pv);
      c.rec_bl_r = rc["same_goal_radius"].as<float>();
      c.rec_win = (int)std::lround((rc["blocked_window"] ? rc["blocked_window"].as<float>() : dp) / pv);   // 없으면 결정 주기 (옛 동작)
      if (c.rec_win < 1) throw std::runtime_error(path + ": recovery.blocked_window 는 행동 주기 이상");
      c.rec_nobk = rc["backup"] && !rc["backup"].as<bool>() ? 1 : 0;              // 없으면 물러남 (옛 동작)
      c.imm_rot = rc["blocked_turn"] && rc["blocked_turn"].as<bool>() ? 1 : 0;   // 없으면 섬
      const float ga = rc["goal_abort"] ? rc["goal_abort"].as<float>() : 0.f;     // 없으면 끔
      if (ga < 0.f) throw std::runtime_error(path + ": recovery.goal_abort 는 0 이상 (s, 0 = 끔)");
      c.rec_abort = ga > 0.f ? std::max(1, (int)std::lround(ga / pv)) : 0;
    }
    if (const char* g = std::getenv("SIM_PATH_RECOVERY")) c.rec_on = (g[0] == '1' && c.rec_bl_r > 0.f) ? 1 : 0;
    // 아래층 지역 계획기 (include/local_planner.h): 절이 없거나 enabled false 면 끔 (RPP 추종기 그대로 — 비트 같음). SIM_LOCAL_PLANNER=0/1 (앞뒤 비교)
    c.lp = LpCfg{};
    if (const YAML::Node lp = ap["local_planner"]) {
      LpCfg& L = c.lp;
      L.on = lp["enabled"].as<bool>() ? 1 : 0;
      L.n_steps = (int)std::lround(lp["horizon"].as<float>() / pv);
      L.k_switch = (int)std::lround(lp["switch_time"].as<float>() / pv);
      L.block_steps = (int)std::lround(lp["block_time"].as<float>() / pv);
      L.rot_on = lp["rotate_in_place"].as<bool>() ? 1 : 0;
      L.mon_real_w = lp["monitor_real_w"].as<bool>() ? 1 : 0;
      L.bumper_scaled = lp["bumper_scaled"].as<bool>() ? 1 : 0;
      const auto list = [&](const char* k, float* dst, int cap) {
        const YAML::Node n = lp[k];
        if (!n || !n.IsSequence() || n.size() < 1 || (int)n.size() > cap)
          throw std::runtime_error(path + ": local_planner." + k + " 는 1~" + std::to_string(cap) + " 개 목록");
        for (size_t i = 0; i < n.size(); ++i) dst[i] = n[i].as<float>();
        return (int)n.size();
      };
      L.n_v = list("speeds", L.v, kLpV);
      L.n_k1 = list("curvatures_first", L.k1, kLpK1);
      L.n_k2 = list("curvatures_second", L.k2, kLpK2);
      L.lag_mix = lp["lag_mix"].as<float>();
      L.ring = lp["scan_memory"].as<bool>() ? 1 : 0;
      L.ring_range = lp["scan_memory_range"].as<float>();
      if (const YAML::Node pp = lp["predict_moving"]) {
        L.ppl = pp["enabled"].as<bool>() ? 1 : 0;
        L.ppl_r = pp["clear"].as<float>();
        L.ppl_hard_t = pp["hard_time"].as<float>();
        L.ppl_vmin = pp["speed_min"].as<float>();
        L.ppl_vmax = pp["speed_max"].as<float>();
        L.w_ppl = pp["weight"].as<float>();
      }
      L.slow_dist = lp["slow_dist"].as<float>();
      L.slow_min = lp["slow_min"].as<float>();
      L.margin = lp["margin"].as<float>();
      L.center_min = lp["center_clear"].as<float>();
      L.r_soft = lp["soft_dist"].as<float>();
      L.fwd_off = lp["window_forward"].as<float>();
      const YAML::Node wt = lp["weights"];
      L.w_prog = wt["progress"].as<float>();
      L.w_v = wt["speed"].as<float>();
      L.w_dev = wt["deviation"].as<float>();
      L.dev_free = wt["deviation_free"].as<float>();
      L.w_head = wt["heading"].as<float>();
      L.w_clr = wt["clearance"].as<float>();
      L.w_hys = wt["hysteresis"].as<float>();
      L.w_rot = wt["rotate"].as<float>();
      L.w_flip = wt["flip"].as<float>();
      if (L.n_steps < 1 || L.n_steps > kLpN || L.k_switch < 0 || L.block_steps < 0 || !(L.r_soft > 0.f) || L.lag_mix < 0.f || L.lag_mix > 1.f)
        throw std::runtime_error(path + ": local_planner (horizon 1~" + std::to_string(kLpN) + " 걸음, soft_dist > 0, lag_mix 0~1)");
    }
    if (const char* g = std::getenv("SIM_LOCAL_PLANNER")) c.lp.on = (g[0] == '1' && c.lp.n_steps > 0) ? 1 : 0;
  }

  // 섀시 구동 (drive.h) · 바닥 재질
  const YAML::Node dr = need(y, "drive", path);
  DriveModel& drv = c.drive;
  drv.delay_v = dr["delay_steps"][0].as<int>();
  drv.delay_w = dr["delay_steps"][1].as<int>();
  if (drv.delay_v >= kDelayMax || drv.delay_w >= kDelayMax) throw std::runtime_error(path + ": drive.delay_steps 가 너무 큼");
  drv.tau_v = dr["tau"][0].as<float>();
  drv.tau_w = dr["tau"][1].as<float>();
  drv.acc_v = dr["accel"][0].as<float>();
  drv.acc_w = dr["accel"][1].as<float>();
  drv.slow_v = dr["slow"][0].as<float>();
  drv.slow_w = dr["slow"][1].as<float>();
  drv.tau_slow_v = dr["tau_slow"][0].as<float>();
  drv.tau_slow_w = dr["tau_slow"][1].as<float>();
  drv.tau_slow_down_v = need(dr, "tau_slow_down", path)[0].as<float>();   // 줄어들 때 (멈춤·감속)
  drv.tau_slow_down_w = dr["tau_slow_down"][1].as<float>();
  drv.v_max = dr["max"][0].as<float>();
  drv.w_max = dr["max"][1].as<float>();
  drv.n_tg = 0;
  if (dr["turn_gain"]) {   // 없으면 끔 (옛 동작)
    const YAML::Node tv = dr["turn_gain"]["v"], tg = dr["turn_gain"]["gain"];
    if (!tv || !tg || tv.size() != tg.size() || tv.size() < 2 || tv.size() > (size_t)kTurnGainPts) throw std::runtime_error(path + ": drive.turn_gain 은 v · gain 같은 수 2~8 개");
    drv.n_tg = (int32_t)tv.size();
    for (int i = 0; i < drv.n_tg; ++i) {
      drv.tg_v[i] = tv[i].as<float>();
      drv.tg_g[i] = tg[i].as<float>();
      if (i && !(drv.tg_v[i] > drv.tg_v[i - 1])) throw std::runtime_error(path + ": drive.turn_gain.v 는 커지는 차례");
    }
  }
  // rc_v·rc_w 는 리모컨 녹화를 다시 돌릴 때만(motion_check, drive.rc) — 학습 환경은 /cmd_vel 경로
  // 물리 단위 행동(act_raw) 한도 = 섀시 최고 속도. 정규 단위 행동 한도는 mdp/obs_layout.h mdp::act
  c.act_v_max = drv.v_max;
  c.act_w_max = mdp::act::W_MAX < drv.w_max ? mdp::act::W_MAX : drv.w_max;
  c.act_raw = 0;
  c.act_v_fwd = mdp::act::V_FWD;
  c.act_v_back = mdp::act::V_BACK;
  c.act_vy_max = mdp::act::VY_MAX;
  c.wheel_k = mdp::obsgen::kWheelK;
  c.wheel_max = drv.v_max;   // 바퀴 하나 최고 선속도 = 실측 1.11 m/s
  const YAML::Node fl = need(y, "floors", path);
  c.n_floors = (int)fl.size();
  if (c.n_floors < 1 || c.n_floors > kMaxFloors) throw std::runtime_error(path + ": floors 는 1~8 개");
  float wsum = 0;
  for (const auto& f : fl) wsum += f["weight"].as<float>();
  float cum = 0;
  for (int i = 0; i < c.n_floors; ++i) {
    cum += fl[i]["weight"].as<float>() / wsum;
    c.floor_cum[i] = cum;
    for (int k = 0; k < 2; ++k) {
      Floor& f = k == 0 ? c.floor_lo[i] : c.floor_hi[i];
      f.slip_v = fl[i]["slip"][0][k].as<float>();
      f.slip_w = fl[i]["slip"][1][k].as<float>();
      f.trac_v = fl[i]["traction"][0][k].as<float>();
      f.trac_w = fl[i]["traction"][1][k].as<float>();
      f.tau_v = fl[i]["lag"][0][k].as<float>();
      f.tau_w = fl[i]["lag"][1][k].as<float>();
    }
  }
  c.floor_cum[c.n_floors - 1] = 1.f;

  const YAML::Node imu = need(y, "imu", path);
  c.gyro_sigma = imu["gyro_sigma"].as<float>();
  c.gyro_bias = imu["gyro_bias"].as<float>();
  // 판 끝 조건·관측 크기는 mdp/ 코드 안 (episode.h·obs_layout.h)
  c.max_steps = (int)std::lround(mdp::cur::kSafeTime / (c.substeps * c.dt));
  c.depth_w = mdp::obs::DEPTH_W;
  c.depth_h = mdp::obs::DEPTH_H;
  c.depth = load_depth_model(dir + need(y, "sensor_model", path).as<std::string>());
  c.imu = load_imu_model(dir + need(y, "sensor_model", path).as<std::string>(), c.dt);   // IMU 가속도·기울기·자이로 x·y (imu.h)
  c.cam_x = body["camera_mount"][0].as<float>();
  c.cam_y = body["camera_mount"][1].as<float>();
  c.cam_z = body["camera_mount"][2].as<float>();
  const double cam_pitch = body["camera_pitch"].as<double>();
  c.cam_cp = (float)std::cos(cam_pitch);
  c.cam_sp = (float)std::sin(cam_pitch);
  // 규칙 추종기 앞 범퍼 정지 문턱 (mdp/README "안전층" — 정책 명령엔 안 씀): 앞 한도 V_FWD 로 가다가 명령 0 → 구동 모델(drive.h)로 멈출 때까지 간 거리
  //   (지연·가속 한도·느린 갈래 모두 들어감, 바닥은 env.yaml floors 중 몸통 가속 한도가 가장 낮은 것)
  //   + 판단이 늦는 몫: 라이다 지연 0.1 s(sensor_model.yaml latency) + 행동 한 걸음(앞 장으로 판단) × V_FWD
  {
    Floor f = c.floor_lo[0];
    for (int i = 0; i < c.n_floors; ++i)
      if (c.floor_lo[i].trac_v < f.trac_v) f = c.floor_lo[i];
    f.slip_v = 1.f;
    f.slip_w = 1.f;
    DriveState ds;
    drive_reset(ds);
    float bv = 0, bw = 0;
    for (int i = 0; i < 500; ++i) drive_step(drv, ds, mdp::act::V_FWD, 0.f, c.dt, f, &bv, &bw);
    float d = 0;
    for (int i = 0; i < 2000 && bv > 1e-3f; ++i) {
      drive_step(drv, ds, 0.f, 0.f, c.dt, f, &bv, &bw);
      d += bv * c.dt;
    }
    h.brake_dist = d;
    c.safety = mdp::act::SAFETY ? 1 : 0;
    c.safety_dist = d + mdp::act::V_FWD * (0.1f + (float)c.substeps * c.dt);
    // 몸통 쓸림 안전층 (실험, env.h sweep_scale) — 기본 끔. SIM_SWEEP_GUARD=1 로 켬(학습·평가·env_verify 모두 같은 규칙).
    // 내다보는 시간 0.2 s = 행동 2 걸음(이번 걸음 + 다음 판단 전), 여유 0.02 m (지도 칸 5 cm 의 반 아래): 잠정
    {
      const char* g = std::getenv("SIM_SWEEP_GUARD");
      c.sweep_guard = (g && g[0] == '1') ? 1 : 0;
      c.sweep_h = 0.2f;
      c.sweep_margin = 0.02f;
    }
    {
      // 닿음 (2026-11-14, contact.h — 센서로 정한 부딪힘이 충돌 보상·판 끝. 시뮬 정답 충돌은 물리·기록만). 값은 low_e0 정책 자취의 정답 충돌과 견줘 골랐다
      // (mdp/README "닿음 — 센서": 학습 맵 8 + 시험 맵 2 × 64 판, 사람 2, 안전층 켬/끔 — 켬 57/66 · 끔 310/337 판을 1 s 앞 ~ 2 s 뒤 안에 잡음,
      //  닿기 1 s 넘게 앞서 판을 끝냈을 헛닿음 0.050 · 0.027 /100 m. 처음 잡은 값 라이다 0.02 | 깊이 0.02 | 멈춤 3 은 깊이 층 헛닿음이 0.39~0.52 /100 m)
      //   con_lid 0.01 m: 라이다 거리 σ 4.2 mm (sensor_model.yaml lidar sigma) 의 2 배 남짓 — 0.02·0.03·0.05 는 헛닿음이 1.2~4 배
      //   con_cam −0.02 m: 깊이 층 칸이 몸통 안으로 2 cm 넘게 든 때만. 층은 몸통 밑을 지우고(hier_path.h) 깊이 최소 거리 0.3 m(sensor_model.yaml depth range)
      //         안은 못 봐서, 0 이상 문턱은 잡음 없이 헛닿음만 늘었다(0.00: 0.16~0.35 /100 m). 낮은 물체에 박는 것은 멈춤이 잡는다 — 간격(근접 항)에는 그대로 씀
      //   멈춤: 명령 |v| ≥ 0.05 m/s 또는 |ω| ≥ 0.2 rad/s 인데 바퀴 오도메트리 |v| < 0.01 m/s (단위 1 mm/s 의 10 배 — imu.h kImuMoveV),
      //         |ω|·자이로 < 0.03 rad/s (자이로 σ 0.003 rad/s 의 10 배 — env.yaml gyro_sigma) 인 걸음이 2 번(0.2 s) 이어짐 — 구동 지연 80 ms·시간상수 40 ms
      //         (env.yaml drive) 뒤라 출발 걸음은 안 걸림(헛닿음 0 ~ 0.034 /100 m). 시뮬은 부딪히면 바퀴를 멈추므로(physics_step drive_reset) 잘 맞는다 —
      //         실물 모터가 벽을 밀 때 바퀴가 서는지 헛도는지는 📏 (헛돌면 오도메트리는 움직이고 자이로·IMU 만 0)
      //   con_map: 로봇 지도 막힘 칸 간격 — 닿음에 안 넣음(기록만): 0.00 이 헛닿음 0.14 /100 m, 정답 충돌 잡음은 멈춤보다 적음
      const auto envf = [](const char* k, float d) { const char* s = std::getenv(k); return s && s[0] ? (float)std::atof(s) : d; };
      c.con_lid = envf("SIM_CON_LID", 0.01f);
      c.con_cam = envf("SIM_CON_CAM", -0.02f);
      c.con_map = envf("SIM_CON_MAP", 0.0f);
      c.con_cmd_v = 0.05f;
      c.con_cmd_w = 0.2f;
      c.con_odo_v = 0.01f;
      c.con_odo_w = 0.03f;
      c.con_stall = (int)envf("SIM_CON_STALL", 2.f);
      c.con_mask = (int)envf("SIM_CON_MASK", (float)(kConLidar | kConCam | kConStall));
      c.con_end = (int)envf("SIM_CON_END", 1.f);
      // 닿은 뒤 바퀴·IMU (2026-11-16, 실물 bag 1103_M_drive3 — env.h EnvCfg::con_slip): 끄면 옛 모델 (지문 그대로)
      c.con_slip = (int)envf("SIM_CON_SLIP", 0.f);
      c.con_slip_k = (float)std::exp(-(double)c.dt / (double)envf("SIM_CON_SLIP_TAU", 0.25f));   // 55.6~55.9 s 바퀴 v 0.84 → 0.55 → 0.37 (0.1 s 마다 ×0.67) — 잠정
      c.con_imp_k = envf("SIM_CON_IMP_K", 90.f);    // 정면 약 1.0 m/s → 91.9 m/s² · 0.3 m/s → 26.4 (잠정)
      // 바퀴·자이로 어긋남 (kConYaw — 기본 마스크에 없음): 실물 bag 7 개 676 s 에서 1.5 rad/s · 2 걸음이면 아는 충돌 둘만, 시뮬은 보통 주행에서도
      // 1.5 를 자주 넘어(새 접촉 모델 진단: 헛것 19 /100 m — 시뮬 회전 미끄러짐이 실물보다 큼) 판정에 못 씀. 값만 재고 기록
      c.con_yaw = envf("SIM_CON_YAW", 1.5f);
      c.con_yaw_k = (int)envf("SIM_CON_YAW_K", 2.f);
      c.con_acc = envf("SIM_CON_ACC", 15.f);         // 실물 보통 세게 몰기 최대 10.8, 99.9 % ≤ 7.5 m/s² (imu_peaks.py)
      // 규칙 추종기 앞 범퍼 정지 끄기 (진단 — 기준선을 앞 정지 없이 잴 때). 정책 명령을 고치는 층은 없음 (2026-11-14 에 없앰)
      if (const char* g = std::getenv("SIM_SAFETY")) c.safety = g[0] == '0' ? 0 : c.safety;
    }
    {
      // 판 끝·성공 (2026-11-08, 닫힘 정의는 2026-11-10 고침 — mdp/README "판 끝"·"닫힘"): 성공 = 로봇 지도 닫힘(아래 닫힘 규칙) + 지도 진행 없음
      // 시계 ≥ 10 s, 1 s(10 걸음)마다 판정. (옛 규칙 선언 + 정답 덮음 ≥ 0.95 는 2026-11-14 에 없앰 — 실물이 못 재는 값)
      // 옛 굵은 격자(20 cm) 닫힘은 로봇이 문틀에 서면 번지기가 둘레 주머니에 갇혀 거짓 닫힘(oracle 성공의 7~14 % 가 정답 덮음 < 0.8)이라 바꿈
      c.close_every = 10;
      c.close_idle_s = 10.f;
      // 닫힘 규칙 (2026-11-10, mdp/README "닫힘" — 값의 근거는 그 절의 oracle 측정): 로봇 시작 칸 + 지나간 칸에서 번진 프론티어 덩어리 중
      // close_min 칸 이상 덩어리의, 옆 모름 칸이 라이다 시야 안 close_look m 안에 든 적이 없는 칸이 하나라도 있으면 안 닫힘.
      // 번지기 몸통 반경 close_radius = 계층형 계획기가 길을 찾는 가장 작은 반경 — consistent_plan 이면 path_r_plan(발자국+패딩 외접 + 칸 대각)으로
      // 먼저 짜고 못 짜면 관측 BFS 반경 mdp::obsgen::kRobotRadius(= nav2 robot_radius)로 한 번 더 짜므로(hier_path.h (e)) 둘 중 작은 것.
      // 이 반경 + 깊이 카메라 층(계획기와 같은 막힘)으로 못 가는 곳의 프론티어는 닫힘·후보에서 빠짐. 계획 반경 0.293 m 를 그대로 쓰면 좁은 문 너머가
      // 빠져 oracle 거짓 닫힘이 0.044 → 0.211 (학습 단계 1 사람 끔 256 판, mdp/README "닫힘").
      // SIM_CLOSE_MIN·SIM_CLOSE_LOOK·SIM_CLOSE_RADIUS 는 측정 비교용 (학습·확인은 기본값)
      c.close_radius = c.path_consistent ? std::min(c.path_r_plan, mdp::obsgen::kRobotRadius) : mdp::obsgen::kRobotRadius;
      c.close_min = 4;
      c.close_look = 1.5f;
      if (const char* v = std::getenv("SIM_CLOSE_MIN")) c.close_min = std::atoi(v);
      if (const char* v = std::getenv("SIM_CLOSE_LOOK")) c.close_look = (float)std::atof(v);
      if (const char* v = std::getenv("SIM_CLOSE_RADIUS")) c.close_radius = (float)std::atof(v);
      if (mdp::robot_cells(c.close_radius, c.res).r > 12) throw std::runtime_error("닫힘 반경 close_radius 는 0.6 m 이하");   // GPU stand_word 낱말 밀기 < 32
    }
  }
  c.people = load_people_cfg(path);
  return h;
}

PeopleCfg load_people_cfg(const std::string& path) {
  const YAML::Node y = YAML::LoadFile(path)["people"];
  if (!y) throw std::runtime_error(path + ": people 없음");
  PeopleCfg p{};
  auto pair = [&](const char* k, float* lo, float* hi) {
    if (!y[k] || y[k].size() != 2) throw std::runtime_error(path + ": people." + k + " 는 [최소, 최대]");
    *lo = y[k][0].as<float>();
    *hi = y[k][1].as<float>();
    if (*hi < *lo) throw std::runtime_error(path + ": people." + k + " 최소 > 최대");
  };
  pair("radius", &p.r_lo, &p.r_hi);
  pair("height", &p.h_lo, &p.h_hi);
  pair("speed", &p.v_lo, &p.v_hi);
  p.p_stand = y["p_stand"].as<float>();
  p.p_ignore = y["p_ignore"].as<float>();
  p.p_approach = y["p_approach"] ? y["p_approach"].as<float>() : 0.f;   // 없으면 0 (옛 설정 그대로)
  p.p_cross = y["p_cross"] ? y["p_cross"].as<float>() : 0.f;
  // 행동 비율 환경변수 (학습·판정 세트가 사람 행동만 바꿀 때 — mdp/README "아래층 RL — 사람"): SIM_PEOPLE_P_STAND · _IGNORE · _APPROACH · _CROSS
  for (auto [k, v] : {std::pair<const char*, float*>{"SIM_PEOPLE_P_STAND", &p.p_stand}, {"SIM_PEOPLE_P_IGNORE", &p.p_ignore},
                      {"SIM_PEOPLE_P_APPROACH", &p.p_approach}, {"SIM_PEOPLE_P_CROSS", &p.p_cross}})
    if (const char* g = std::getenv(k); g && g[0]) *v = (float)std::atof(g);
  if (p.p_approach < 0.f || p.p_cross < 0.f || p.p_approach + p.p_cross > 1.f) throw std::runtime_error(path + ": people.p_approach + p_cross 는 [0, 1]");
  p.yield_dist = y["yield_dist"].as<float>();
  p.yield_speed = y["yield_speed"].as<float>();
  p.spawn_keep = y["spawn_keep"].as<float>();
  p.pause = y["pause"].as<float>();
  if (p.r_lo <= 0.f || p.r_hi > 0.5f) throw std::runtime_error(path + ": people.radius 는 (0, 0.5] m");
  if (p.v_hi > 2.0f) throw std::runtime_error(path + ": people.speed 최대가 너무 큼");
  return p;
}

}  // namespace sim
