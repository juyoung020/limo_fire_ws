#include "frontier_explorer.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <stdexcept>

#include "env_api.h"

namespace fb {

namespace {
YAML::Node need(const YAML::Node& n, const char* k, const std::string& where) {
  const YAML::Node v = n[k];
  if (!v) throw std::runtime_error(where + ": " + k + " 없음");
  return v;
}
float wrap(float a) {
  while (a > (float)M_PI) a -= 2.f * (float)M_PI;
  while (a < -(float)M_PI) a += 2.f * (float)M_PI;
  return a;
}
float approach(float cur, float target, float step) {
  const float d = target - cur;
  return cur + (d > step ? step : (d < -step ? -step : d));
}
}  // namespace

Params load_params(const std::string& fy, const std::string& env_yaml, const std::string& mode) {
  Params p{};
  // RPP·velocity_smoother·goal checker: 시뮬과 같은 읽기(sim::load_env_cfg → nav2_params.yaml)
  const sim::EnvHostCfg hc = sim::load_env_cfg(env_yaml);
  const sim::EnvCfg& c = hc.cfg;
  p.v_des = c.v_des; p.v_max = c.v_max; p.w_max = c.w_max; p.a_max = c.a_max; p.alpha_max = c.alpha_max;
  p.lookahead = c.lookahead; p.min_lookahead = c.min_lookahead; p.rot_w = c.rot_w; p.rot_min_angle = c.rot_min_angle;
  p.approach_dist = c.approach_dist; p.approach_min_v = c.approach_min_v; p.goal_tol = c.goal_tol;
  p.res = c.res;
  p.ctrl_dt = (float)c.substeps * c.dt;
  if (c.action_mode != sim::kActVelocity) throw std::runtime_error(env_yaml + ": 기준선은 action.mode velocity 가 필요");
  // 팽창·계획기: 같은 nav2_params.yaml
  const YAML::Node ey = YAML::LoadFile(env_yaml);
  const std::string dir = env_yaml.substr(0, env_yaml.find_last_of('/') + 1);
  const std::string nav_path = dir + need(ey, "nav2_params", env_yaml).as<std::string>();
  const YAML::Node nav = YAML::LoadFile(nav_path);
  const YAML::Node gc = nav["global_costmap"]["global_costmap"]["ros__parameters"];
  p.robot_radius = need(gc, "robot_radius", nav_path).as<float>();
  const YAML::Node infl = need(gc, "inflation_layer", nav_path);
  p.inflation_radius = infl["inflation_radius"].as<float>();
  p.cost_scaling = infl["cost_scaling_factor"].as<float>();
  const YAML::Node gb = need(nav["planner_server"]["ros__parameters"], "GridBased", nav_path);
  p.planner_tolerance = gb["tolerance"].as<float>();
  p.travel_mult = gb["cost_travel_multiplier"].as<float>();
  p.allow_unknown = gb["allow_unknown"].as<bool>();
  // 장애물 층(obstacle_layer): 라이다 끝점을 이 거리 안에서 바로 막힘으로 (확률 지도와 따로)
  p.obstacle_range = need(need(gc, "obstacle_layer", nav_path), "scan", nav_path)["obstacle_max_range"].as<float>();
  // 발자국: local_costmap footprint "[[x, y], …]" 문자열 + footprint_padding
  {
    const YAML::Node lc = nav["local_costmap"]["local_costmap"]["ros__parameters"];
    const YAML::Node fpn = YAML::Load(need(lc, "footprint", nav_path).as<std::string>());
    float mx = 0, my = 0;
    for (const auto& q : fpn) { mx = std::max(mx, std::fabs(q[0].as<float>())); my = std::max(my, std::fabs(q[1].as<float>())); }
    const float pad = lc["footprint_padding"] ? lc["footprint_padding"].as<float>() : 0.f;
    p.fp_half_len = mx + pad;
    p.fp_half_wid = my + pad;
  }
  p.collision_time = need(need(nav["controller_server"]["ros__parameters"], "FollowPath", nav_path), "max_allowed_time_to_collision_up_to_carrot", nav_path).as<float>();
  // explore_lite·우리 몫
  const YAML::Node f = YAML::LoadFile(fy);
  const YAML::Node el = need(f, "explore_lite", fy);
  p.planner_period = 1.f / el["planner_frequency"].as<float>();
  p.progress_timeout = el["progress_timeout"].as<float>();
  p.potential_scale = el["potential_scale"].as<float>();
  p.gain_scale = el["gain_scale"].as<float>();
  p.min_frontier_size = el["min_frontier_size"].as<float>();
  p.blacklist_tol = el["blacklist_tolerance"].as<int>();
  const YAML::Node n2 = need(f, "nav2", fy);
  p.replan_period = n2["replan_period"].as<float>();
  p.inscribed_escape_cost = n2["inscribed_escape_cost"].as<float>();
  p.bt_retries = n2["bt_retries"].as<int>();
  p.spin_dist = n2["spin_dist"].as<float>();
  p.wait_duration = n2["wait_duration"].as<float>();
  p.backup_dist = n2["backup_dist"].as<float>();
  p.backup_speed = n2["backup_speed"].as<float>();
  p.behavior_time_allowance = n2["behavior_time_allowance"].as<float>();
  p.wait_timeout = need(f, "start", fy)["wait_timeout"].as<float>();
  p.t_safe = need(f, "episode", fy)["t_safe"].as<float>();
  const YAML::Node md = need(f, "modes", fy)[mode];
  if (!md) throw std::runtime_error(fy + ": modes." + mode + " 없음");
  p.mode = mode;
  const YAML::Node v = need(md, "v", fy);
  if (v.as<std::string>() != "nav2") {
    p.v_des = v.as<float>();
    p.v_max = v.as<float>();
  }
  return p;
}

Explorer::Explorer(const Params& p, int gw, int gh, int stride, float ox, float oy)
    : p_(p), w_(gw), h_(gh), stride_(stride), ox_(ox), oy_(oy) {
  const size_t n = (size_t)gw * gh;
  flag_.resize(n);
  cost_.resize(n);
  qa_.resize(n);
  qb_.resize(n);
  fr_.resize(4096);
  g_.resize(n);
  parent_.resize(n);
  closed_.resize(n);
  // Nav2 InflationLayer::computeCost (inflation_layer.hpp:148-162): 막힌 칸 254, 내접 반경 안 253, 밖은 252·exp(−k(d − r_in))
  const int R = (int)std::ceil(p.inflation_radius / p.res);
  inscribed_cells_ = (int)std::ceil(p.robot_radius / p.res);
  for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
      const float d = std::sqrt((float)(dx * dx + dy * dy)) * p.res;
      if (d > p.inflation_radius) continue;
      uint8_t c;
      if (dx == 0 && dy == 0) c = 254;
      else if (d <= p.robot_radius) c = 253;
      else c = (uint8_t)(252.0 * std::exp(-1.0 * p.cost_scaling * (d - p.robot_radius)));
      if (c > 0) kernel_.push_back({dx, dy, c});
    }
}

int Explorer::cell(float x, float y, int* cx, int* cy) const {
  *cx = (int)std::floor((x - ox_) / p_.res);
  *cy = (int)std::floor((y - oy_) / p_.res);
  return (*cx >= 0 && *cy >= 0 && *cx < w_ && *cy < h_) ? *cy * w_ + *cx : -1;
}

bool Explorer::on_blacklist(float x, float y) const {
  const float tol = (float)p_.blacklist_tol * p_.res;   // explore.cpp:249-260
  for (auto& b : blacklist_)
    if (std::fabs(x - b.first) < tol && std::fabs(y - b.second) < tol) return true;
  return false;
}

// 전역 비용 지도: 정적(지도) 칸 종류 + 팽창. 모름은 255(NO_INFORMATION) 그대로, 단 내접 반경 안이면 253
// (Nav2 InflationLayer::updateCosts: 옛 값이 NO_INFORMATION 이면 cost ≥ INSCRIBED 일 때만 덮어씀 — inflate_unknown 기본 false)
void Explorer::build_costmap(const int16_t* L) {
  const mdp::MapView m = view(L);
  for (int y = 0; y < h_; ++y)
    for (int x = 0; x < w_; ++x) {
      const uint8_t c = mdp::cls_at(m, x, y);
      cost_[(size_t)y * w_ + x] = c == mdp::kUnk ? 255 : 0;
    }
  for (int y = 0; y < h_; ++y)
    for (int x = 0; x < w_; ++x) {
      if (!lethal_src(m, x, y)) continue;
      for (const Off& o : kernel_) {
        const int u = x + o.dx, v = y + o.dy;
        if (u < 0 || v < 0 || u >= w_ || v >= h_) continue;
        uint8_t& d = cost_[(size_t)v * w_ + u];
        if (d == 255) { if (o.cost >= 253) d = o.cost; }
        else if (o.cost > d) d = o.cost;
      }
    }
}

// Smac 2D 와 같은 비용의 A*: 8 이웃, 한 칸 = (1 + travel_mult · cost / 252) × (대각 √2) (node_2d.cpp:64-80),
// 못 지나는 칸 = cost ≥ 253, 모름(255)은 allow_unknown 이면 지남 (Smac GridCollisionChecker). 시작 칸이 내접 반경 안이면 탈출 비용(위 yaml)
// 목표: 목표 칸이 못 지나는 칸이면 tolerance 안에서 가장 가까운 지날 수 있는 칸 (Smac tolerance). 경로 = 칸 가운데 점들
bool Explorer::astar(float px, float py, float gx, float gy) {
  st_.n_astar++;
  path_.clear();
  path_c_.clear();
  path_i_ = 0;
  int sx, sy, tx, ty;
  if (cell(px, py, &sx, &sy) < 0) return false;
  cell(gx, gy, &tx, &ty);
  const bool escape = cost_[(size_t)sy * w_ + sx] >= 253;
  auto passable = [&](uint8_t c) { return c < 253 || (c == 255 && p_.allow_unknown) || (escape && c == 253); };
  auto ncost = [&](uint8_t c) { return c == 253 ? p_.inscribed_escape_cost : (float)c / 252.f; };
  // 목표 칸 고르기
  const int tol = (int)std::floor(p_.planner_tolerance / p_.res);
  int best = -1, bd = 1 << 30;
  for (int dy = -tol; dy <= tol; ++dy)
    for (int dx = -tol; dx <= tol; ++dx) {
      const int u = tx + dx, v = ty + dy, d2 = dx * dx + dy * dy;
      if (u < 0 || v < 0 || u >= w_ || v >= h_ || d2 > tol * tol) continue;
      const uint8_t c = cost_[(size_t)v * w_ + u];
      if (!(c < 253 || (c == 255 && p_.allow_unknown))) continue;
      if (d2 < bd) { bd = d2; best = v * w_ + u; }
    }
  if (best < 0) return false;
  const int goal = best, gxc = goal % w_, gyc = goal / w_;
  const int start = sy * w_ + sx;
  std::fill(g_.begin(), g_.end(), 1e30f);
  std::fill(closed_.begin(), closed_.end(), 0);
  using QE = std::pair<float, int32_t>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> open;
  auto h = [&](int k) {
    const int dx = std::abs(k % w_ - gxc), dy = std::abs(k / w_ - gyc);
    return (float)std::max(dx, dy) + 0.41421356f * (float)std::min(dx, dy);
  };
  g_[start] = 0;
  parent_[start] = -1;
  open.push({h(start), start});
  bool found = false;
  while (!open.empty()) {
    const int k = open.top().second;
    open.pop();
    if (closed_[k]) continue;
    closed_[k] = 1;
    if (k == goal) { found = true; break; }
    const int x = k % w_, y = k / w_;
    for (int q = 0; q < 8; ++q) {
      int dx, dy;
      mdp::nb8(q, &dx, &dy);
      const int u = x + dx, v = y + dy;
      if (u < 0 || v < 0 || u >= w_ || v >= h_) continue;
      const int j = v * w_ + u;
      if (closed_[j]) continue;
      const uint8_t c = cost_[j];
      if (!passable(c)) continue;
      const float step = (dx != 0 && dy != 0 ? 1.41421356f : 1.f) * (1.f + p_.travel_mult * ncost(c));
      const float ng = g_[k] + step;
      if (ng < g_[j]) {
        g_[j] = ng;
        parent_[j] = k;
        open.push({ng + h(j), j});
      }
    }
  }
  if (!found) return false;
  std::vector<int> rev;
  for (int k = goal; k >= 0; k = parent_[k]) rev.push_back(k);
  for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
    path_.push_back(ox_ + ((float)(*it % w_) + 0.5f) * p_.res);
    path_.push_back(oy_ + ((float)(*it / w_) + 0.5f) * p_.res);
    path_c_.push_back(cost_[*it]);
  }
  return true;
}

// 앞 경로(지금 칸부터 lookahead × 4 까지)가 새로 막힌 칸(막힘 또는 내접 반경 안)에 들었나 — RPP use_collision_detection 과
// 비용 지도 갱신 때 BT 가 경로를 다시 짜는 몫. 지도만 본다(참값 안 씀)
bool Explorer::path_blocked(const int16_t* L) const {
  const mdp::MapView m = view(L);
  const float horizon = 4.f * p_.lookahead;
  float acc = 0;
  for (size_t i = path_i_; i + 1 < path_.size() / 2 && acc < horizon; ++i) {
    const float x0 = path_[2 * i], y0 = path_[2 * i + 1], x1 = path_[2 * i + 2], y1 = path_[2 * i + 3];
    acc += std::hypot(x1 - x0, y1 - y0);
    if (path_c_[i + 1] == 253) continue;   // 짤 때 이미 내접 반경 안이던 점(탈출 경로)은 막힘으로 안 봄
    int cx, cy;
    if (cell(x1, y1, &cx, &cy) < 0) return true;
    const int R = inscribed_cells_;
    for (int dy = -R; dy <= R; ++dy)
      for (int dx = -R; dx <= R; ++dx) {
        const int u = cx + dx, v = cy + dy;
        if (u < 0 || v < 0 || u >= w_ || v >= h_) continue;
        if ((float)(dx * dx + dy * dy) * p_.res * p_.res > p_.robot_radius * p_.robot_radius) continue;
        if (lethal_src(m, u, v)) return true;
      }
  }
  return false;
}

// explore_lite Explore::makePlan (explore.cpp:179-245) + 목표 보내기(Nav2 계획 실패 = ABORTED → reachedGoal 의 블랙리스트, 264-281)
bool Explorer::make_plan(const int16_t* L, float px, float py, double t) {
  for (int guard = 0; guard < 10000; ++guard) {
    st_.n_plans++;
    int rx, ry;
    cell(px, py, &rx, &ry);
    const int min_cells = (int)std::ceil(p_.min_frontier_size / p_.res - 1e-4f);
    const mdp::SearchResult r = mdp::frontier_search(view(L), rx, ry, min_cells, mdp::SearchBuf{flag_.data(), qa_.data(), qb_.data()},
                                                     fr_.data(), (int)fr_.size());
    st_.n_frontiers = r.n_found;
    st_.n_frontier_small = r.n_small;
    const int n = std::min<int>(r.n_found, (int)fr_.size());
    if (n == 0) { reason_ = "no_frontier"; return false; }   // explore.cpp:190-193
    // 비용 (frontier_search.cpp:191-196): potential × min_distance(m) × res − gain × size × res, 오름차순
    struct C { float cost, cx, cy, dist; int i; };
    std::vector<C> cs;
    for (int i = 0; i < n; ++i) {
      const mdp::Frontier& f = fr_[i];
      const float dist = std::sqrt((float)f.min_d2) * p_.res;
      const float cx = ox_ + ((float)f.sx / (float)f.n + 0.5f) * p_.res, cy = oy_ + ((float)f.sy / (float)f.n + 0.5f) * p_.res;
      cs.push_back({p_.potential_scale * dist * p_.res - p_.gain_scale * (float)f.n * p_.res, cx, cy, dist, i});
    }
    std::stable_sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.cost < b.cost; });
    auto it = std::find_if(cs.begin(), cs.end(), [&](const C& c) { return !on_blacklist(c.cx, c.cy); });
    if (it == cs.end()) { reason_ = "all_blacklisted"; return false; }   // explore.cpp:200-209
    if (std::getenv("FRONTIER_DEBUG"))
      std::fprintf(stderr, "    plan t %.1f robot (%.2f %.2f) frontiers %d small %d -> goal (%.2f %.2f) n %d dist %.2f cost %.3f bl %zu\n", t, px, py, n,
                   r.n_small, it->cx, it->cy, fr_[it->i].n, it->dist, it->cost, blacklist_.size());
    const bool same = std::hypot(it->cx - prev_gx_, it->cy - prev_gy_) < 0.01f;   // explore.cpp:42-49, 213
    prev_gx_ = it->cx;
    prev_gy_ = it->cy;
    if (!same || prev_dist_ > it->dist) { last_progress_ = t; prev_dist_ = it->dist; }
    if (t - last_progress_ > p_.progress_timeout) {   // explore.cpp:221-226
      blacklist_.push_back({it->cx, it->cy});
      st_.n_timeout++;
      continue;
    }
    // explore.cpp:229-231: 같은 목표면 아무것도 안 보냄 — 이미 도착한 목표라도(그러면 진척 없음 시간이 지나 블랙리스트될 때까지 선다)
    if (same) return true;
    // 새 목표 → Nav2: 경로를 짜 보고 못 짜면 ABORTED
    st_.n_goals++;
    gx_ = it->cx;
    gy_ = it->cy;
    has_goal_ = true;
    build_costmap(L);
    if (!astar(px, py, gx_, gy_)) {
      st_.n_aborted++;
      blacklist_.push_back({gx_, gy_});
      has_goal_ = false;
      continue;
    }
    next_replan_t_ = t + p_.replan_period;
    return true;
  }
  reason_ = "plan_loop";
  return false;
}

Action Explorer::smooth(float vc, float wc) {
  // velocity_smoother 가속 한도 (시뮬 physics_step 과 같은 식, 주기만 행동 주기)
  vs_ = approach(vs_, std::max(-p_.v_max, std::min(p_.v_max, vc)), p_.a_max * p_.ctrl_dt);
  ws_ = approach(ws_, wc, p_.alpha_max * p_.ctrl_dt);
  Action a;
  a.v = vs_;
  a.w = ws_;
  return a;
}

// 발자국(local_costmap footprint + footprint_padding) 둘레가 막힘 씨앗 칸에 닿나 — Nav2 FootprintCollisionChecker::footprintCostAtPose 는
// 둘레 선분 칸의 최댓값, RPP inCollision 은 그 값이 LETHAL 이면 충돌(regulated_pure_pursuit_controller.cpp:555-580)
bool Explorer::footprint_hits(const int16_t* L, float x, float y, float th) const {
  const mdp::MapView m = view(L);
  const float c = std::cos(th), s = std::sin(th);
  const float hl = p_.fp_half_len, hw = p_.fp_half_wid;
  const float cx[4] = {hl, -hl, -hl, hl}, cy[4] = {hw, hw, -hw, -hw};
  for (int e = 0; e < 4; ++e) {
    const float ax = cx[e], ay = cy[e], bx = cx[(e + 1) % 4], by = cy[(e + 1) % 4];
    const int n = std::max(1, (int)std::ceil(std::hypot(bx - ax, by - ay) / (0.5f * p_.res)));
    for (int k = 0; k <= n; ++k) {
      const float u = ax + (bx - ax) * k / n, v = ay + (by - ay) * k / n;
      int qx, qy;
      if (cell(x + c * u - s * v, y + s * u + c * v, &qx, &qy) < 0) return true;
      if (lethal_src(m, qx, qy)) return true;
    }
  }
  return false;
}

// RPP isCollisionImminent (regulated_pure_pursuit_controller.cpp:475-550): 지금 자세와, 명령 (v, ω) 를 한 칸씩 앞으로 굴린 자세들을
// max_allowed_time_to_collision_up_to_carrot 까지(당근을 넘으면 그만) 발자국으로 검사
bool Explorer::collision_imminent(const int16_t* L, float px, float py, float pth, float v, float w, float carrot_dist) const {
  if (footprint_hits(L, px, py, pth)) return true;
  float dt;
  if (std::fabs(v) < 0.01f && std::fabs(w) > 0.01f) {
    const float rmax = std::hypot(p_.fp_half_len, p_.fp_half_wid);
    dt = 2.f * std::sin((p_.res / 2.f) / rmax) / std::fabs(w);
  } else if (std::fabs(v) >= 0.01f) {
    dt = p_.res / std::fabs(v);
  } else {
    return false;
  }
  float x = px, y = py, th = pth;
  for (int i = 1; (float)i * dt < p_.collision_time; ++i) {
    x += dt * v * std::cos(th);
    y += dt * v * std::sin(th);
    th += dt * w;
    if (std::hypot(x - px, y - py) > carrot_dist) break;
    if (footprint_hits(L, x, y, th)) return true;
  }
  return false;
}

// RPP: 경로에서 가장 가까운 점부터 lookahead 이상 떨어진 첫 점(regulated_pure_pursuit_controller.cpp:442-454), 없으면 끝 점.
// 속도·곡률은 시뮬 goal 모드 추종기(env.h physics_step)와 같은 식. (vc, wc) 는 velocity_smoother 앞 명령
void Explorer::follow(float px, float py, float pth, float* vc_out, float* wc_out, float* carrot_dist) {
  const size_t np = path_.size() / 2;
  // 가장 가까운 점으로 앞자르기 (앞으로만)
  float bd = 1e30f;
  size_t bi = path_i_;
  for (size_t i = path_i_; i < np; ++i) {
    const float d = std::hypot(path_[2 * i] - px, path_[2 * i + 1] - py);
    if (d < bd) { bd = d; bi = i; }
    if (d > bd + 2.f) break;
  }
  path_i_ = bi;
  size_t ci = np - 1;
  for (size_t i = path_i_; i < np; ++i)
    if (std::hypot(path_[2 * i] - px, path_[2 * i + 1] - py) >= p_.lookahead) { ci = i; break; }
  const float cx = path_[2 * ci], cy = path_[2 * ci + 1];
  const float ex = path_[2 * (np - 1)], ey = path_[2 * (np - 1) + 1];
  const float d_goal = std::hypot(ex - px, ey - py);
  const float d_car = std::hypot(cx - px, cy - py);
  float vc = 0, wc = 0;
  if (d_goal > p_.goal_tol) {
    const float err = wrap(std::atan2(cy - py, cx - px) - pth);
    if (std::fabs(err) > p_.rot_min_angle) {
      wc = err > 0 ? p_.rot_w : -p_.rot_w;
    } else {
      vc = p_.v_des;
      if (d_goal < p_.approach_dist) vc = std::max(p_.v_des * d_goal / p_.approach_dist, p_.approach_min_v);
      const float ld = std::max(p_.min_lookahead, std::min(p_.lookahead, d_car));
      wc = vc * (2.f * std::sin(err) / ld);
      wc = std::max(-p_.w_max, std::min(p_.w_max, wc));
    }
  }
  *vc_out = vc;
  *wc_out = wc;
  *carrot_dist = d_car;
}

// Nav2 BT 복구(navigate_to_pose_w_replanning_and_recovery.xml): 경로 추종이 실패하면 RoundRobin 으로 하나씩 —
// 비용 지도 비우기(우리는 다음 걸음 다시 짜기) · Spin 1.57 rad · Wait 5 s · BackUp 0.30 m @ 0.05 m/s. 실패가 number_of_retries(6) 를 넘으면 목표 ABORTED.
// 돌기·후진은 Nav2 처럼 발자국으로 앞을 보고(simulate_ahead_time 2 s 는 짧게 줄여 행동 한 칸) 부딪힐 것 같으면 그 복구를 그만둔다
Action Explorer::recover(const int16_t* L, float px, float py, float pth, double t) {
  const float dt = p_.ctrl_dt;
  bool stop = t >= rec_end_t_;
  float vc = 0, wc = 0;
  if (!stop && rec_ == Rec::Spin) {
    rec_done_ += std::fabs(wrap(pth - rec_last_th_));
    rec_last_th_ = pth;
    if (rec_done_ >= p_.spin_dist || collision_imminent(L, px, py, pth, 0.f, p_.rot_w, 1e9f)) stop = true;
    else wc = p_.rot_w;
  } else if (!stop && rec_ == Rec::Backup) {
    rec_done_ = std::hypot(px - rec_x0_, py - rec_y0_);
    if (rec_done_ >= p_.backup_dist || footprint_hits(L, px - std::cos(pth) * p_.backup_speed * dt * 2.f, py - std::sin(pth) * p_.backup_speed * dt * 2.f, pth))
      stop = true;
    else vc = -p_.backup_speed;
  }
  if (stop) {
    rec_ = Rec::None;
    next_replan_t_ = t;   // 복구 뒤 다시 짬
  }
  // 복구 동작은 velocity_smoother 를 거침 (Nav2 behavior_server 출력도 smoother 를 지남)
  return smooth(vc, wc);
}

Action Explorer::declare_end() {
  phase_ = Phase::Done;
  Action a;
  a.declare = true;
  return a;
}

Action Explorer::step(const int16_t* L, float px, float py, float pth, double t) {
  cell(px, py, &rcx_, &rcy_);
  if (phase_ == Phase::Done) return Action{};
  if (t >= p_.t_safe) { phase_ = Phase::Done; reason_ = "t_safe"; return Action{}; }
  if (phase_ == Phase::Wait) {
    if (t_wait0_ < 0) t_wait0_ = t;
    // 아는 빈 칸에서 시작해 크기 문턱을 넘는 프론티어가 하나라도 보일 때까지 (빈 칸 문턱이 엄격해 처음 몇 초는 빈 칸 덩어리가 작다)
    const int min_cells = (int)std::ceil(p_.min_frontier_size / p_.res - 1e-4f);
    const mdp::SearchResult r = mdp::frontier_search(view(L), rcx_, rcy_, min_cells, mdp::SearchBuf{flag_.data(), qa_.data(), qb_.data()},
                                                     fr_.data(), (int)fr_.size());
    if (r.start_free && r.n_found > 0) {
      phase_ = Phase::Go;
      plan_now_ = true;
    } else if (t - t_wait0_ > p_.wait_timeout) {
      reason_ = "no_frontier_at_start";
      return declare_end();
    } else {
      return smooth(0.f, collision_imminent(L, px, py, pth, 0.f, p_.rot_w, 1e9f) ? 0.f : p_.rot_w);
    }
  }
  // explore_lite 타이머 + 도착 콜백
  if (has_goal_ && std::hypot(gx_ - px, gy_ - py) <= p_.goal_tol) {   // Nav2 goal checker → SUCCEEDED → 바로 다시 고름
    st_.n_reached++;
    has_goal_ = false;
    rec_ = Rec::None;
    plan_now_ = true;
  }
  if (plan_now_ || t >= next_plan_t_) {
    plan_now_ = false;
    next_plan_t_ = t + p_.planner_period;
    const bool had = has_goal_;
    const float ogx = gx_, ogy = gy_;
    if (!make_plan(L, px, py, t)) return declare_end();
    if (!had || ogx != gx_ || ogy != gy_) { fails_ = 0; rec_ = Rec::None; }   // 새 목표 = 새 BT 실행
  }
  if (!has_goal_) return smooth(0.f, 0.f);
  if (rec_ != Rec::None) return recover(L, px, py, pth, t);
  auto abort_goal = [&]() -> bool {   // Nav2 ABORTED → explore_lite 블랙리스트 + 바로 다시 고름 (explore.cpp:264-281)
    st_.n_aborted++;
    blacklist_.push_back({gx_, gy_});
    has_goal_ = false;
    fails_ = 0;
    rec_ = Rec::None;
    if (!make_plan(L, px, py, t)) return false;
    next_plan_t_ = t + p_.planner_period;
    return true;
  };
  // Nav2: 주기마다 또는 앞이 막히면 다시 짬. 못 짜면 ABORTED
  const bool blocked = !path_.empty() && path_blocked(L);
  if (blocked) st_.n_blocked_replan++;
  if (blocked || t >= next_replan_t_ || path_.empty()) {
    next_replan_t_ = t + p_.replan_period;
    build_costmap(L);
    if (!astar(px, py, gx_, gy_)) {
      if (!abort_goal()) return declare_end();
      if (!has_goal_) return smooth(0.f, 0.f);
    }
  }
  float vc, wc, cd;
  follow(px, py, pth, &vc, &wc, &cd);
  if (collision_imminent(L, px, py, pth, vc, wc, cd)) {
    // 추종 실패 → 복구 하나 (RoundRobin) 또는 재시도 소진이면 ABORTED
    st_.n_collision_stops++;
    if (++fails_ > p_.bt_retries) {
      if (!abort_goal()) return declare_end();
      return smooth(0.f, 0.f);
    }
    const Rec order[4] = {Rec::Clear, Rec::Spin, Rec::Wait, Rec::Backup};
    rec_ = order[rr_ % 4];
    rr_++;
    st_.n_recoveries++;
    rec_done_ = 0;
    rec_last_th_ = pth;
    rec_x0_ = px;
    rec_y0_ = py;
    const float allow = p_.behavior_time_allowance;
    rec_end_t_ = t + (rec_ == Rec::Wait ? p_.wait_duration : (rec_ == Rec::Clear ? 0.0 : allow));
    if (rec_ == Rec::Clear) { rec_ = Rec::None; next_replan_t_ = t; }
    return smooth(0.f, 0.f);
  }
  return smooth(vc, wc);
}

}  // namespace fb
