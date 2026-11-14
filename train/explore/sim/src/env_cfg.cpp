// env.yaml 읽기 (load_env_cfg · load_people_cfg) — CUDA 없이 빌드되는 호스트 코드. 학습 환경(env_host.cpp)과 실물 노드(limo_fire_rl)가 같이 쓴다
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
  EnvHostCfg h;
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
  if (mode != "velocity" && mode != "goal" && mode != "path") throw std::runtime_error(path + ": action.mode 는 velocity · goal · path");
  c.action_mode = mode == "velocity" ? kActVelocity : (mode == "path" ? kActPath : kActGoal);
  const YAML::Node av = need(a, "velocity", path), ag = need(a, "goal", path);
  c.n_dir = ag["n_dir"].as<int>();
  c.n_dist = (int)ag["dist"].size();
  if (c.n_dist < 1 || c.n_dist > 4) throw std::runtime_error(path + ": action.goal.dist 는 1~4 개");
  for (int i = 0; i < c.n_dist; ++i) c.dist[i] = ag["dist"][i].as<float>();
  c.goal_margin = ag["goal_margin"].as<float>();
  const float period = (c.action_mode != kActGoal ? av : ag)["period"].as<float>();   // 계층형도 0.1 s 속도 행동 주기로 시뮬 걸음 (결정은 그 묶음)
  c.substeps = (int)std::lround(period / c.dt);
  {   // 그래프 모드 (mdp/graph.h, action.graph) — 켜는 것은 도구·학습기(--graph → graph_on = 1, action_mode = path)
    const YAML::Node gr = a["graph"];
    const auto num = [&](const char* k, float d) { return gr && gr[k] ? gr[k].as<float>() : d; };
    c.graph_on = 0;
    c.graph_stand = gr && gr["standable"] ? (gr["standable"].as<bool>() ? 1 : 0) : 1;
    c.graph_util_r = (int)(num("utility_range_ratio", 0.8f) * c.map_max_range / c.res);
    c.graph_min_util = (int)num("min_utility", 2.f);
    c.graph_open = gr && gr["blocking_only"] ? (gr["blocking_only"].as<bool>() ? 1 : 0) : 1;
    c.graph_fvox = (int)std::lround(num("frontier_cell", 0.4f) / c.res);
    if (c.graph_fvox < 1 || ((c.size + c.graph_fvox - 1) / c.graph_fvox) * ((c.size + c.graph_fvox - 1) / c.graph_fvox) > mdp::graphgen::kVoxMax)
      throw std::runtime_error(path + ": action.graph.frontier_cell 이 너무 작음 (복셀 수 상한)");
    c.graph_util_norm = std::max(1, (int)(c.map_max_range * 3.14f / ((float)c.graph_fvox * c.res)));   // 전작 env.py: SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE
    const float st = num("node_resolution", mdp::graph::NODE_RES) / c.res;
    c.graph_step = (int)std::lround(st);
    if (std::fabs(st - std::round(st)) > 1e-3f || c.graph_step < 1 || c.graph_step * mdp::graph::GW < c.size)
      throw std::runtime_error(path + ": 그래프 노드 격자(node_resolution × GW)가 지도(size × res)를 못 덮거나 node_resolution 이 map.resolution 의 배수가 아님");
    const std::string term = gr && gr["termination"] ? gr["termination"].as<std::string>() : "ariadne";
    if (term != "ariadne" && term != "hier") throw std::runtime_error(path + ": action.graph.termination 은 ariadne | hier");
    c.graph_term = term == "ariadne" ? 1 : 0;
    c.graph_max_steps = std::max(1, (int)std::lround(num("decision_timeout", 15.f) / period));
    c.graph_max_dec = (int)num("max_decisions", 0.f);
    const std::string gdm = gr && gr["decision"] ? gr["decision"].as<std::string>() : "node";
    if (gdm != "node" && gdm != "period") throw std::runtime_error(path + ": action.graph.decision 은 node | period");
    c.graph_fixed = gdm == "period" ? 1 : 0;
    c.graph_pass = gr && gr["pass_through"] ? (gr["pass_through"].as<bool>() ? 1 : 0) : 1;
    c.graph_view = gr && gr["visited_utility"] ? gr["visited_utility"].as<int>() : 0;
    if (c.graph_view < 0 || c.graph_view > 2) throw std::runtime_error("env.yaml action.graph.visited_utility 는 0 · 1 · 2");
    c.graph_cover = gr && gr["leftover_utility"] ? gr["leftover_utility"].as<int>() : 1;
    if (c.graph_cover < 0 || c.graph_cover > 2) throw std::runtime_error("env.yaml action.graph.leftover_utility 는 0 · 1 · 2");
    c.graph_rew_dist = num("reward_dist_scale", 1.f) / c.map_max_range;   // 전작 env.py:82 dist / UPDATING_MAP_SIZE * 5 = dist / SENSOR_RANGE
    c.graph_rew_done = num("reward_done", 20.f);                           // 전작 worker.py:58
    c.graph_arrive = num("arrive_radius", c.goal_tol);                     // 기본 = Nav2 xy_goal_tolerance (위에서 읽음)
  }
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
    c.cand_v2 = ap["candidates_v2"] && ap["candidates_v2"].as<bool>() ? 1 : 0;
    if (const char* g = std::getenv("SIM_CAND_V2")) c.cand_v2 = g[0] == '1' ? 1 : 0;   // 앞뒤 비교
    {   // 아래층 계측·단독 과제 (include/low_task.h). 스위치는 도구가 켠다 — 여기선 값만
      const YAML::Node fo = ap["follow"];
      const auto num = [&](const char* k, float d) { return fo && fo[k] ? fo[k].as<float>() : d; };
      c.low_stat = c.low_task = 0;
      c.low_rl = c.low_auto = c.low_period = c.low_act_w = c.low_explore = c.low_solo = 0;   // 아래층 RL (include/low_rl.h) — 도구·학습기가 켠다
      c.low_pass_r = num("pass_radius", 0.5f);
      c.low_idle_d = num("stall_dist", 0.3f);
      c.low_idle_steps = (int)std::lround(num("stall_time", 10.f) / pv);
      c.low_time_steps = (int)std::lround(num("time_limit", 60.f) / pv);
      c.low_goals = (int)num("max_goals", 5.f);
      c.low_front = 0.5f * num("front_fov_deg", 220.f) * 3.14159265f / 180.f;
    }
    c.path_cost_reg = ap["use_cost_regulated_linear_velocity_scaling"].as<bool>() ? 1 : 0;
    c.path_cs_dist = ap["cost_scaling_dist"].as<float>();
    c.path_cs_gain = ap["cost_scaling_gain"].as<float>();
    c.path_r_mid = ap["weight_mid_dist"].as<float>();
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
    c.cam_min_h = c.cam_max_h = c.cam_obs_range = c.cam_ray_range = 0.f;
    if (const YAML::Node co = ap["camera_obstacles"]) {
      c.cam_on = co["enabled"].as<bool>() ? 1 : 0;
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
    {
      // 계획 반경 = 충돌 검사 발자국(footprint + 패딩) 외접 반경 + 칸 대각(칸 가운데끼리 거리로 재는 반경 → 발자국 점이 든 칸까지 · 로봇이 칸 안 어디든 — 칸 반 대각 × 2)
      const float hl = c.half_len + c.path_fp_pad, hw = c.half_wid + c.path_fp_pad;
      c.path_r_plan = std::sqrt(hl * hl + hw * hw) + c.res * std::sqrt(2.f);
      c.path_goal_tol = need(need(nav["planner_server"]["ros__parameters"], "GridBased", nav_path), "tolerance", nav_path).as<float>();   // Smac tolerance
      c.path_goal_far = ap["goal_far_tolerance"] ? ap["goal_far_tolerance"].as<float>() : c.path_goal_tol;
      if (c.path_goal_far < c.path_goal_tol || c.path_goal_far > 1.5f) throw std::runtime_error(path + ": goal_far_tolerance 는 tolerance ~ 1.5 m");
      if (c.path_consistent && (c.path_r_plan >= c.path_r_out || c.path_r_plan < 0.2f))
        throw std::runtime_error(path + ": consistent_plan 계획 반경 " + std::to_string(c.path_r_plan) + " 은 0.2 이상 팽창 반경 미만이어야 함");
    }
    // 복구 (hier_path.h "복구"): 절이 없으면 끔. 물러나기 걸음 = BT BackUp backup_dist / backup_speed / 행동 주기. SIM_PATH_RECOVERY=0/1 (앞뒤 비교)
    c.rec_on = c.rec_backup_steps = c.rec_bl_steps = 0;
    c.rec_bl_r = 0.f;
    if (const YAML::Node rc = ap["recovery"]) {
      c.rec_on = rc["enabled"].as<bool>() ? 1 : 0;
      c.rec_backup_steps = (int)std::lround(rc["backup_dist"].as<float>() / c.path_backup_v / pv);
      c.rec_bl_steps = (int)std::lround(rc["blacklist_time"].as<float>() / pv);
      c.rec_bl_r = rc["same_goal_radius"].as<float>();
    }
    if (const char* g = std::getenv("SIM_PATH_RECOVERY")) c.rec_on = (g[0] == '1' && c.rec_bl_r > 0.f) ? 1 : 0;
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
  c.curriculum = 1;
  c.max_delay = 1;
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
  // 판 끝·보상·관측 크기는 mdp/ 코드 안 (rewards.h·curriculums.h·obs_layout.h)
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
  // 안전층 문턱 (mdp/README "안전층"): 앞 한도 V_FWD 로 가다가 명령 0 → 구동 모델(drive.h)로 멈출 때까지 간 거리
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
      // 아래층 RL 벽 충돌 대책 (low_rl.h, mdp/README "아래층 RL — 벽 충돌"). 정책 환경(LowCmd::follow == 2)에만 듣는다.
      // 환경변수는 견줌 실험용 — 학습·평가·env_verify 가 같은 값을 읽는다. SIM_LOW_GUARD=0 SIM_LOW_NOBACK=0 SIM_LR_MPROX_K=0 이 2026-11-12 동작
      const auto envf = [](const char* k, float d) { const char* s = std::getenv(k); return s && s[0] ? (float)std::atof(s) : d; };
      c.lg_on = (int)envf("SIM_LOW_GUARD", (float)kLrGuardOn);
      c.lg_h = envf("SIM_LOW_GUARD_H", kLrGuardH);
      c.lg_margin = envf("SIM_LOW_GUARD_MARGIN", kLrGuardMargin);
      c.lg_noback = (int)envf("SIM_LOW_NOBACK", (float)kLrNoBack);
      c.lg_touch = (int)envf("SIM_LOW_GUARD_TOUCH", (float)kLrGuardTouch);
      c.lr_mprox_k = envf("SIM_LR_MPROX_K", kLrMproxK);
      c.lr_mprox_d = envf("SIM_LR_MPROX_D", kLrMproxD);
      // 사람 충돌·속도 대책 (mdp/README "아래층 RL — 사람 충돌·속도"). 보상만 바꾼다 — 정책 행동·관측은 그대로
      c.lr_ppl_k = envf("SIM_LR_PPL_K", kLrPplK);
      c.lr_ppl_t = envf("SIM_LR_PPL_T", kLrPplT);
      c.lr_ppl_m = envf("SIM_LR_PPL_M", kLrPplM);
      c.lr_pcol_k = envf("SIM_LR_PCOL_K", kLrPersonCollisionK);
      c.lr_w_k = envf("SIM_LR_W_K", kLrTurnK);
      c.ps_on = (int)envf("SIM_LOW_YIELD", (float)kLrYieldOn);
      c.ps_h = envf("SIM_LOW_YIELD_H", kLrYieldH);
      c.ps_gate = envf("SIM_LOW_YIELD_GATE", kLrYieldGate);
      c.ps_margin = envf("SIM_LOW_YIELD_MARGIN", kLrYieldMargin);
      c.ps_hold = (int)envf("SIM_LOW_YIELD_HOLD", (float)kLrYieldHold);
      c.ps_mem = (int)envf("SIM_LOW_YIELD_MEM", (float)kLrYieldMem);
      c.ps_trk = (int)envf("SIM_LOW_TRACK", (float)kLrTrkOn);
      c.tk_gate = envf("SIM_LOW_TRACK_GATE", kLrTrkGate);
      c.tk_sz = envf("SIM_LOW_TRACK_SZ", kLrTrkSz);
      c.tk_qa = envf("SIM_LOW_TRACK_QA", kLrTrkQa);
      c.tk_miss = (int)envf("SIM_LOW_TRACK_MISS", (float)kLrTrkMiss);
      c.tk_conf = (int)envf("SIM_LOW_TRACK_CONF", (float)kLrTrkConf);
      c.tk_disp = envf("SIM_LOW_TRACK_DISP", kLrTrkDisp);
      c.lo_mov = (int)envf("SIM_LOW_OBS_MOV", 0.f);
      c.tk_sag = envf("SIM_LOW_TRACK_SAG", kLrClusSag);
      c.tk_w0 = envf("SIM_LOW_TRACK_W0", kLrClusRound0);
      c.lr_w0 = (c.wheel_max - c.act_v_fwd) / c.wheel_k;   // 이 아래 ω 는 바퀴 한도가 v 를 안 깎는다 (1.11 − 1.0) / 0.2 = 0.55 rad/s
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
      // 안전층 (앞 범퍼 정지) 끄기 — 진단 전용 (안전층 끈 충돌 재기). 몸통 지킴이·비켜 주기는 SIM_LOW_GUARD=0·SIM_LOW_YIELD=0
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
    {
      const char* g = std::getenv("SIM_CUR_TOPRAND");   // 커리큘럼 맨 위 단계 무작위 (curriculums.h on_episode_end) — 기본 켬, SIM_CUR_TOPRAND=0 으로 끔
      c.cur_top_random = (g && g[0] == '0') ? 0 : 1;      // explore_v8_cur A/B: 맨 위 단계에 고여 중간 크기 맵을 잊던 것
    }
    {
      // 오르내림 규칙 (env.yaml curriculum, curriculums.h): 성공 규칙만 (옛 덮음 규칙은 정답 덮음이라 2026-11-14 에 없앰)
      const YAML::Node cu = need(y, "curriculum", path);
      {
        const std::string rule = need(cu, "rule", path).as<std::string>();
        if (rule != "success") throw std::runtime_error(path + ": curriculum.rule 은 success (정답 덮음 규칙 coverage 는 없앰)");
        const YAML::Node r = need(cu, rule.c_str(), path);
        c.cur_window = need(r, "window", path).as<int>();
        c.cur_up = need(r, "up", path).as<float>();
        c.cur_down = need(r, "down", path).as<float>();
        if (c.cur_window < 1 || c.cur_window > mdp::cur::kCurWindowMax)
          throw std::runtime_error(path + ": curriculum." + rule + ".window 는 1 ~ " + std::to_string(mdp::cur::kCurWindowMax));
        if (!(c.cur_down <= c.cur_up)) throw std::runtime_error(path + ": curriculum." + rule + " down ≤ up 이어야 함");
      }
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
