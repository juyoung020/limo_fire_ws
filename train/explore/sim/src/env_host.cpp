#include "env_api.h"
#ifdef SIM_OPTIX
#include "optix_rt.h"
#endif

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>

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
  if (mode != "velocity" && mode != "goal") throw std::runtime_error(path + ": action.mode 는 velocity 또는 goal");
  c.action_mode = mode == "velocity" ? kActVelocity : kActGoal;
  const YAML::Node av = need(a, "velocity", path), ag = need(a, "goal", path);
  c.n_dir = ag["n_dir"].as<int>();
  c.n_dist = (int)ag["dist"].size();
  if (c.n_dist < 1 || c.n_dist > 4) throw std::runtime_error(path + ": action.goal.dist 는 1~4 개");
  for (int i = 0; i < c.n_dist; ++i) c.dist[i] = ag["dist"][i].as<float>();
  c.goal_margin = ag["goal_margin"].as<float>();
  const float period = (c.action_mode == kActVelocity ? av : ag)["period"].as<float>();
  c.substeps = (int)std::lround(period / c.dt);
  if (c.substeps < 1 || std::fabs(c.substeps * c.dt - period) > 1e-4f)
    throw std::runtime_error(path + ": action period 는 time.dt 의 배수");

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
  c.max_steps = (int)std::lround(mdp::rew::kSafeTime / (c.substeps * c.dt));
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
  }
  c.people = load_people_cfg(path);
  return h;
}

// 장면 → 칸 격자 (칸 = 지도와 같은 res). 정적 장면만 본다 — 움직이는 물체(사람)는 여기 안 들어가 분모에 안 셈.
//   1. 막힘: 몸통 높이(collision_band)에 걸린 삼각형을 칸 반 간격 점으로 찍은 칸 (충돌 격자)
//   2. 서는 칸: 막힘이 몸통 반폭(half_wid, 내접 반지름) 안에 없는 빈 칸 — 몸통 폭보다 좁은 틈은 못 지난다(필요조건).
//      칸 가운데끼리 거리로 잰다(칸 하나 = res).
//   3. 갈 수 있는 서는 칸: 장면 출발점(Scene::start)에서 서는 칸을 4 이웃 flood fill. 맵 테두리 칸에 닿으면 집이 새는 것 —
//      출발점에서 그 테두리 칸까지 길과, 길에서 가장 좁은 곳(막힘까지 가장 가까운 칸)을 leak 에 적는다
//   4. 갈 수 있는 바닥(kCellFloor): 갈 수 있는 서는 칸에서 몸통이 쓸고 지나갈 수 있는 빈 칸 — 외접 반지름(hypot(half_len, half_wid))
//      안이면서 빈 칸으로만 이어진 칸(빈 칸 BFS 로 번지되 출처 서는 칸에서 외접 반지름 안까지 — 얇은 벽 너머로 안 넘어감)
//   5. 벽 면 칸(kCellWall, 덮음 분모): 갈 수 있는 바닥에 4 이웃으로 맞닿은 막힌 칸. 벽 두께 안쪽·집 밖은 바닥이 안 닿아 저절로 빠진다
//   6. 시작 칸: 갈 수 있는 서는 칸 중 spawn_clearance 반경 안이 모두 빈 칸
World build_world(const std::string& name, Scene scene, const EnvHostCfg& hc, bool require_closed) {
  const EnvCfg& c = hc.cfg;
  World w;
  w.name = name;
  const Node& root = scene.nodes[0];
  w.ox = root.lo[0];
  w.oy = root.lo[1];
  w.gw = (int)std::ceil((root.hi[0] - root.lo[0]) / c.res) + 1;
  w.gh = (int)std::ceil((root.hi[1] - root.lo[1]) / c.res) + 1;
  if (w.gw > c.size || w.gh > c.size)
    throw std::runtime_error(name + ": 맵 " + std::to_string(w.gw) + "×" + std::to_string(w.gh) + " 칸이 지도 크기보다 큼");
  const int W = w.gw, H = w.gh, N = W * H;
  std::vector<uint8_t> occ((size_t)N, 0);
  // 1. 몸통 높이에 걸린 삼각형: 칸 반 간격으로 점을 찍어 그 칸을 막음
  for (const Tri& t : scene.tris) {
    const float zlo = std::min({t.a.z, t.b.z, t.c.z}), zhi = std::max({t.a.z, t.b.z, t.c.z});
    if (zhi < hc.band_lo || zlo > hc.band_hi) continue;
    auto len = [](V3 p, V3 q) { return std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z)); };
    const float e = std::max({len(t.a, t.b), len(t.b, t.c), len(t.c, t.a)});
    const int n = std::max(1, (int)std::ceil(e / (0.5f * c.res)));
    for (int i = 0; i <= n; ++i)
      for (int j = 0; i + j <= n; ++j) {
        const float u = (float)i / n, v = (float)j / n;
        const V3 p{t.a.x + (t.b.x - t.a.x) * u + (t.c.x - t.a.x) * v, t.a.y + (t.b.y - t.a.y) * u + (t.c.y - t.a.y) * v,
                   t.a.z + (t.b.z - t.a.z) * u + (t.c.z - t.a.z) * v};
        if (p.z < hc.band_lo || p.z > hc.band_hi) continue;
        const int cx = (int)std::floor((p.x - w.ox) / c.res), cy = (int)std::floor((p.y - w.oy) / c.res);
        if (cx >= 0 && cy >= 0 && cx < W && cy < H) occ[(size_t)cy * W + cx] = 1;
      }
  }
  // 반지름 r(m) 안 칸 오프셋 (칸 가운데끼리)
  auto disc = [&](float r) {
    std::vector<std::pair<int, int>> d;
    const int R = (int)std::ceil(r / c.res);
    for (int dy = -R; dy <= R; ++dy)
      for (int dx = -R; dx <= R; ++dx)
        if ((float)(dx * dx + dy * dy) * c.res * c.res <= r * r) d.push_back({dx, dy});
    return d;
  };
  // out_free: 맵 밖을 빈 곳으로 (서는 칸 — 맵 밖은 열린 곳이라 테두리까지 번져야 새는 것을 잡는다). 아니면 막힘으로 (시작 칸)
  auto clear_disc = [&](int x, int y, const std::vector<std::pair<int, int>>& d, bool out_free) {
    for (const auto& o : d) {
      const int u = x + o.first, v = y + o.second;
      if (u < 0 || v < 0 || u >= W || v >= H) {
        if (out_free) continue;
        return false;
      }
      if (occ[(size_t)v * W + u]) return false;
    }
    return true;
  };
  // 2. 서는 칸
  const auto d_in = disc(c.half_wid);
  std::vector<uint8_t> stand((size_t)N, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) stand[(size_t)y * W + x] = !occ[(size_t)y * W + x] && clear_disc(x, y, d_in, true);
  // 3. 출발점에서 flood fill. 출발점 칸이 서는 칸이 아니면(벽에 붙어 시작) 몸통 반 길이 안에서 가장 가까운 서는 칸
  const int sx0 = (int)std::floor((scene.start[0] - w.ox) / c.res), sy0 = (int)std::floor((scene.start[1] - w.oy) / c.res);
  int start = -1;
  {
    int bd = 1 << 30;
    const int R = (int)std::ceil(c.half_len / c.res);
    for (int dy = -R; dy <= R; ++dy)
      for (int dx = -R; dx <= R; ++dx) {
        const int u = sx0 + dx, v = sy0 + dy;
        if (u < 0 || v < 0 || u >= W || v >= H || !stand[(size_t)v * W + u]) continue;
        if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; start = v * W + u; }
      }
  }
  if (start < 0) {
    char b[200];
    std::snprintf(b, sizeof(b), ": 출발점 (%.2f, %.2f) 근처에 로봇이 설 칸이 없음 (장면 출발점 확인)", scene.start[0], scene.start[1]);
    throw std::runtime_error(name + b);
  }
  std::vector<int32_t> parent((size_t)N, -2), queue;
  queue.reserve(N);
  queue.push_back(start);
  parent[start] = -1;
  int border = -1;
  for (size_t h = 0; h < queue.size(); ++h) {
    const int k = queue[h], x = k % W, y = k / W;
    if (border < 0 && (x == 0 || y == 0 || x == W - 1 || y == H - 1)) border = k;
    const int nb[4][2] = {{x + 1, y}, {x - 1, y}, {x, y + 1}, {x, y - 1}};
    for (const auto& q : nb) {
      if (q[0] < 0 || q[1] < 0 || q[0] >= W || q[1] >= H) continue;
      const int j = q[1] * W + q[0];
      if (stand[j] && parent[j] == -2) { parent[j] = k; queue.push_back(j); }
    }
  }
  auto at = [&](int k, float* x, float* y) { *x = w.ox + ((float)(k % W) + 0.5f) * c.res; *y = w.oy + ((float)(k / W) + 0.5f) * c.res; };
  if (border >= 0) {
    // 길에서 가장 좁은 곳: 막힌 칸까지 가장 가까운 거리(2 m 안에서 찾음)가 가장 작은 칸 — 틈(문·벽 구멍).
    // 같은 폭이 이어지면(뚫린 통로) 출발점 쪽에서 처음 그 폭이 되는 칸 = 집에서 빠져나가는 어귀
    std::vector<int> path;
    for (int k = border; k >= 0; k = parent[k]) path.push_back(k);
    std::reverse(path.begin(), path.end());
    std::vector<float> clr(path.size());
    const int R = (int)std::ceil(2.f / c.res);
    float nd = 1e9f;
    for (size_t i = 0; i < path.size(); ++i) {
      const int x = path[i] % W, y = path[i] / W;
      int b2 = R * R + 1;
      for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
          const int u = x + dx, v = y + dy;
          if (u >= 0 && v >= 0 && u < W && v < H && occ[(size_t)v * W + u] && dx * dx + dy * dy < b2) b2 = dx * dx + dy * dy;
        }
      clr[i] = std::sqrt((float)b2) * c.res;
      nd = std::min(nd, clr[i]);
    }
    float nx = 0, ny = 0;
    for (size_t i = 0; i < path.size(); ++i)
      if (clr[i] <= nd + 0.5f * c.res) { at(path[i], &nx, &ny); nd = clr[i]; break; }
    float bx, by, s0x, s0y;
    at(border, &bx, &by);
    at(start, &s0x, &s0y);
    char b[400];
    std::snprintf(b, sizeof(b),
                  "집이 닫히지 않음: 출발점 (%.2f, %.2f) 에서 갈 수 있는 바닥이 맵 테두리 (%.2f, %.2f) 에 닿음. 그 길에서 가장 좁은 곳 (%.2f, %.2f) — 벽까지 %.2f m",
                  s0x, s0y, bx, by, nx, ny, nd);
    w.closed = false;
    w.leak = b;
    if (require_closed) throw std::runtime_error(name + ": " + w.leak);
  }
  // 4. 갈 수 있는 바닥: 서는 칸에서 빈 칸으로만 번지되 출처 서는 칸에서 외접 반지름 안까지 (BFS 차례 = 가까운 출처가 먼저 닿음)
  const float r_out = std::sqrt(c.half_len * c.half_len + c.half_wid * c.half_wid);
  const float r_out2 = (r_out / c.res) * (r_out / c.res);
  std::vector<int32_t> src((size_t)N, -1);
  std::vector<int32_t> q2(queue.begin(), queue.end());
  for (int k : queue) src[k] = k;
  for (size_t h = 0; h < q2.size(); ++h) {
    const int k = q2[h], x = k % W, y = k / W, s = src[k], sxk = s % W, syk = s / W;
    const int nb[4][2] = {{x + 1, y}, {x - 1, y}, {x, y + 1}, {x, y - 1}};
    for (const auto& q : nb) {
      if (q[0] < 0 || q[1] < 0 || q[0] >= W || q[1] >= H) continue;
      const int j = q[1] * W + q[0];
      if (occ[j] || src[j] >= 0) continue;
      const int ddx = q[0] - sxk, ddy = q[1] - syk;
      if ((float)(ddx * ddx + ddy * ddy) > r_out2) continue;
      src[j] = s;
      q2.push_back(j);
    }
  }
  // 5·6. 격자 비트, 분모, 시작 칸
  const auto d_spawn = disc(hc.spawn_clearance);
  w.grid.assign((size_t)N, 0);
  for (int k = 0; k < N; ++k) {
    if (occ[k]) w.grid[k] |= kCellBlocked;
    if (src[k] >= 0) { w.grid[k] |= kCellFloor; ++w.n_floor; }
  }
  for (int k = 0; k < N; ++k) {
    if (!occ[k]) continue;
    const int x = k % W, y = k / W;
    const int nb[4][2] = {{x + 1, y}, {x - 1, y}, {x, y + 1}, {x, y - 1}};
    for (const auto& q : nb)
      if (q[0] >= 0 && q[1] >= 0 && q[0] < W && q[1] < H && (w.grid[q[1] * W + q[0]] & kCellFloor)) { w.grid[k] |= kCellWall; ++w.n_wall; break; }
  }
  for (int k : queue)
    if (clear_disc(k % W, k / W, d_spawn, false)) w.spawn.push_back((int32_t)k);
  if (w.spawn.empty()) throw std::runtime_error(name + ": 시작 칸 없음 (spawn_clearance 반경 안이 모두 빈 칸인 갈 수 있는 칸이 없음)");
  if (w.n_wall == 0) throw std::runtime_error(name + ": 벽 면 칸 없음 (덮음 분모 0)");
  std::sort(w.spawn.begin(), w.spawn.end());
  w.clear = people_clearance(w.grid, W, H, c.res);   // 움직이는 사람이 설 수 있는 곳
  w.scene = std::move(scene);
  return w;
}

template <class T>
T* Env::alloc(size_t n) {
  if (gpu_) {
    void* p = nullptr;
    if (cudaMalloc(&p, std::max<size_t>(1, n * sizeof(T))) != cudaSuccess) throw std::runtime_error("cudaMalloc 실패");
    cudaMemset(p, 0, n * sizeof(T));
    owned_.push_back(p);
    return (T*)p;
  }
  host_.emplace_back(std::max<size_t>(1, n * sizeof(T)), 0);
  return (T*)host_.back().data();
}

template <class T>
T* Env::upload(const std::vector<T>& v) {
  T* p = alloc<T>(v.size());
  if (gpu_) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  else std::copy(v.begin(), v.end(), p);
  return p;
}

Env::Env(const EnvHostCfg& hc, const std::vector<World>& worlds, const LidarModel& lidar, int n_env, uint64_t seed, bool gpu)
    : gpu_(gpu) {
  b_.cfg = hc.cfg;
  b_.lidar = lidar;
  b_.n_env = n_env;
  b_.n_maps = (int)worlds.size();
  std::vector<MapDev> maps;
  for (const World& w : worlds) {
    MapDev m{};
    // BVH 상자를 RT 코어 경로(optix_rt.cpp)와 같게 키움: 1 mm + 상대 1e−5. 상자 반올림으로 ray_tri 가 맞히는 삼각형을 놓치지 않아 답이 (t, 번호) 최소로 정해진다
    std::vector<Node> nodes = w.scene.nodes;
    for (Node& nd : nodes)
      for (int q = 0; q < 3; ++q) {
        const float pad = 1e-3f + 1e-5f * std::max(std::fabs(nd.lo[q]), std::fabs(nd.hi[q]));
        nd.lo[q] -= pad;
        nd.hi[q] += pad;
      }
    m.scene = SceneView{upload(nodes), upload(w.scene.tris), upload(w.scene.obj), upload(w.scene.obj_size)};
    m.grid = upload(w.grid);
    m.spawn = upload(w.spawn);
    m.clear = upload(w.clear);
    m.gw = w.gw;
    m.gh = w.gh;
    m.n_spawn = (int)w.spawn.size();
    m.n_wall = w.n_wall;
    m.n_floor = w.n_floor;
    m.ox = w.ox;
    m.oy = w.oy;
    m.t_ref = mdp::rew::t_ref(w.n_wall);
    m.stage = w.stage >= 0 ? w.stage : mdp::cur::stage_of_area(w.n_floor * hc.cfg.res * hc.cfg.res);
    maps.push_back(m);
  }
  b_.maps = upload(maps);
  // 단계별 맵 묶음: 그 단계 맵이 없으면 가장 가까운 아래 단계(없으면 위 단계) 맵을 쓴다. 맵이 있는 가장 높은 단계 = max_stage
  {
    std::vector<int32_t> list;
    b_.max_stage = 0;
    for (const MapDev& m : maps) b_.max_stage = std::max(b_.max_stage, m.stage);
    for (int s = 0; s < mdp::cur::kNumStages; ++s) {
      b_.stage_off[s] = (int)list.size();
      int pick = -1;
      for (int d = 0; d < mdp::cur::kNumStages && pick < 0; ++d)
        for (int t : {s - d, s + d})
          if (pick < 0 && t >= 0 && t < mdp::cur::kNumStages)
            for (const MapDev& m : maps)
              if (m.stage == t) { pick = t; break; }
      for (int i = 0; i < (int)maps.size(); ++i)
        if (maps[i].stage == pick) list.push_back(i);
    }
    b_.stage_off[mdp::cur::kNumStages] = (int)list.size();
    b_.stage_maps = upload(list);
  }
  std::vector<EnvState> st(n_env);
  uint64_t r = seed;
  for (auto& e : st) {
    e = EnvState{};
    e.rng = dm::splitmix64(r);
  }
  b_.st = upload(st);
  const size_t cells = (size_t)hc.cfg.size * hc.cfg.size;
  b_.lmap = alloc<int16_t>(cells * n_env);
  b_.marks = alloc<uint32_t>(cells / 4 * n_env);
  b_.obs_img = alloc<uint8_t>((size_t)mdp::obs::IMG_SIZE * n_env);
  b_.obs_1d = alloc<float>((size_t)mdp::obs::POLICY_1D * n_env);
  b_.priv = alloc<float>((size_t)mdp::obs::PRIV * n_env);
  b_.lidar_ring = alloc<float>((size_t)kLidarRing * mdp::obs::LIDAR_BINS * n_env);
  b_.vec_prev = alloc<float>((size_t)mdp::obs::VEC * n_env);
  b_.lidar_int = alloc<float>((size_t)mdp::obs::LIDAR_BINS * n_env);
  b_.reward = alloc<float>(n_env);
  b_.reward_terms = alloc<float>((size_t)mdp::rew::kTerms * n_env);
  b_.done = alloc<int32_t>(n_env);
  b_.ep = alloc<EpInfo>(n_env);
  b_.scratch = alloc<Shared>(n_env);
  if (gpu) {
    const size_t px = (size_t)mdp::obs::DEPTH_SIZE * n_env;
    b_.dep_z = alloc<float>(px);
    b_.dep_tri = alloc<int32_t>(px);
    b_.dep_side = alloc<float>(2 * px);
    b_.dep_jobs = alloc<int32_t>(2 * px);
    b_.counters = alloc<int32_t>(16);
    const size_t nb = (size_t)kMaxBeams * n_env;
    b_.beam_t = alloc<float>(nb);
    b_.beam_tri = alloc<int32_t>(nb);
    b_.beam_side = alloc<float>(4 * nb);
    b_.beam_jobs = alloc<int32_t>(4 * nb);
#ifdef SIM_OPTIX
    // RT 코어: 맵마다 GAS. 환경 변수 SIM_NO_OPTIX 가 있으면 소프트웨어 순회 (A/B·확인용)
    if (!std::getenv("SIM_NO_OPTIX") && OptixRt::available()) {
      std::vector<const Scene*> sc;
      std::vector<const void*> dt;
      for (size_t i = 0; i < worlds.size(); ++i) {
        sc.push_back(&worlds[i].scene);
        dt.push_back(maps[i].scene.tris);
      }
      rt_ = new OptixRt(sc, dt);
      b_.rt = rt_;
      b_.rt_o = alloc<float4>(px);
      b_.rt_d = alloc<float4>(px);
      b_.rt_tri0 = alloc<int32_t>(px);
      b_.rt_t = alloc<float>(px);
    }
#endif
  }
}

Env::~Env() {
#ifdef SIM_OPTIX
  delete (OptixRt*)rt_;
#endif
  for (void* p : owned_) cudaFree(p);
}

void Env::reset() {
  if (gpu_) env_reset_gpu(b_);
  else env_reset_cpu(b_);
}

void Env::step(const float* actions, cudaStream_t st) {
  if (gpu_) env_step_gpu(b_, actions, st);
  else env_step_cpu(b_, actions);
}

StepOut Env::outputs() const {
  return StepOut{b_.obs_img, b_.obs_1d, b_.priv, b_.reward, b_.reward_terms, b_.done, b_.ep};
}

void Env::sync() const {
  if (gpu_ && cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
}

namespace {
template <class T>
void fetch(bool gpu, const T* p, size_t n, std::vector<T>& out) {
  out.resize(n);
  if (gpu) cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
  else std::copy(p, p + n, out.begin());
}
}  // namespace

void Env::get_state(std::vector<EnvState>& out) const { fetch(gpu_, b_.st, b_.n_env, out); }
void Env::set_state(int env, const EnvState& s) {
  if (gpu_) cudaMemcpy(b_.st + env, &s, sizeof(EnvState), cudaMemcpyHostToDevice);
  else b_.st[env] = s;
}
void Env::set_smoke(const std::vector<int>& on) {
  if ((int)on.size() != b_.n_env) throw std::runtime_error("set_smoke: 환경 수와 다름");
  std::vector<EnvState> st;
  get_state(st);
  for (int i = 0; i < b_.n_env; ++i) {
    if (on[i] != 0 && on[i] != 1) throw std::runtime_error("set_smoke: 0 또는 1");
    if (st[i].smoke == on[i]) continue;
    st[i].smoke = on[i];
    st[i].smoke_steps = 0;
    set_state(i, st[i]);
  }
}
void Env::get_map(std::vector<int16_t>& out) const {
  fetch(gpu_, b_.lmap, (size_t)b_.cfg.size * b_.cfg.size * b_.n_env, out);
}
void Env::get_seen(std::vector<uint8_t>& out) const {
  std::vector<uint32_t> m;
  const size_t cells = (size_t)b_.cfg.size * b_.cfg.size * b_.n_env;
  fetch(gpu_, b_.marks, cells / 4, m);
  out.resize(cells);
  for (size_t i = 0; i < cells; ++i) out[i] = (uint8_t)((m[i >> 2] >> (8 * (i & 3) + 3)) & 3u);
}
void Env::get_obs(std::vector<uint8_t>& img, std::vector<float>& o1d, std::vector<float>& priv) const {
  fetch(gpu_, b_.obs_img, (size_t)mdp::obs::IMG_SIZE * b_.n_env, img);
  fetch(gpu_, b_.obs_1d, (size_t)mdp::obs::POLICY_1D * b_.n_env, o1d);
  fetch(gpu_, b_.priv, (size_t)mdp::obs::PRIV * b_.n_env, priv);
}
void Env::get_terms(std::vector<float>& terms) const { fetch(gpu_, b_.reward_terms, (size_t)mdp::rew::kTerms * b_.n_env, terms); }
void Env::get_ep(std::vector<EpInfo>& ep) const { fetch(gpu_, b_.ep, (size_t)b_.n_env, ep); }

// 장면 묶음 목록 (env_api.h 위 설명). 등급(4 번째 열)이 있으면 커리큘럼 단계로.
std::vector<World> load_scene_list(const std::string& list_file, const EnvHostCfg& hc, std::vector<std::string>* skipped) {
  std::ifstream f(list_file);
  if (!f) throw std::runtime_error(list_file + ": 장면 목록을 못 엶");
  const std::string dir = list_file.find('/') == std::string::npos ? std::string() : list_file.substr(0, list_file.find_last_of('/') + 1);
  struct Item {
    std::string path;
    int stage = -1;
  };
  std::vector<Item> items;
  std::string line;
  while (std::getline(f, line)) {
    const size_t h = line.find('#');
    if (h != std::string::npos) line = line.substr(0, h);
    std::vector<std::string> col;
    {
      size_t p = 0;
      while (true) {
        const size_t q = line.find('\t', p);
        std::string c = line.substr(p, q == std::string::npos ? std::string::npos : q - p);
        const size_t x = c.find_first_not_of(" \r"), y = c.find_last_not_of(" \r");
        col.push_back(x == std::string::npos ? std::string() : c.substr(x, y - x + 1));
        if (q == std::string::npos) break;
        p = q + 1;
      }
    }
    if (col.empty() || col[0].empty() || col[0] == "path") continue;
    std::string p = col[0];
    if (p[0] != '/') p = dir + p;
    items.push_back({p, (col.size() > 3 && !col[3].empty()) ? std::stoi(col[3]) : -1});
  }
  // 장면마다 읽기·BVH·격자 만들기는 서로 독립 → 호스트 스레드로 나눠 만든다 (맵 520 개에 한 줄로 ~25 s 걸리던 시작 시간).
  // 결과·건너뜀 목록은 목록 순서 그대로 모은다 (스레드 수와 무관하게 같은 결과).
  std::vector<std::unique_ptr<World>> built(items.size());
  std::vector<std::string> errs(items.size());
  {
    std::atomic<size_t> next{0};
    auto work = [&]() {
      for (size_t k; (k = next.fetch_add(1)) < items.size();) {
        try {
          built[k].reset(new World(build_world(items[k].path, load_scene(items[k].path), hc)));
          if (items[k].stage >= 0) built[k]->stage = std::max(0, std::min(mdp::cur::kNumStages - 1, items[k].stage));
        } catch (const std::exception& ex) {
          errs[k] = ex.what();
          if (errs[k].empty()) errs[k] = "장면 오류";
        }
      }
    };
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
  }
  std::vector<World> out;
  for (size_t k = 0; k < items.size(); ++k) {
    if (built[k]) {
      out.push_back(std::move(*built[k]));
    } else {
      if (!skipped) throw std::runtime_error(errs[k]);
      skipped->push_back(errs[k]);
    }
  }
  if (out.empty()) throw std::runtime_error(list_file + ": 쓸 수 있는 장면이 없음");
  return out;
}
void Env::get_reward(std::vector<float>& reward, std::vector<int32_t>& done) const {
  fetch(gpu_, b_.reward, b_.n_env, reward);
  fetch(gpu_, b_.done, b_.n_env, done);
}

}  // namespace sim
