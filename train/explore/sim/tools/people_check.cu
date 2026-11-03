// 움직이는 사람 시험 (people.h):
//   1. 보임: 서 있는 로봇 앞 1.5 m 를 가로지르는 사람 한 명 — 라이다 빔이 사람을 맞고(스캔에 나타남), 깊이 관측·지도가 사람 없을 때와 달라지나.
//      같은 씨앗 CPU 환경 둘(사람 있음·없음)을 같은 행동(정지)으로 돌려 비교한다.
//   2. 벽 안 뚫음: GPU 환경 여럿 × 긴 시간(사람 상한 8, 반은 정지·반은 무작위 행동) 동안 걸음마다 모든 사람을 정답 격자로 따로 검사 —
//      원기둥이 막힌 칸(정사각형)과 겹친 횟수, 갈 수 있는 바닥 밖 횟수, 사람끼리 겹침, 판 시작에 출발점 1.5 m 안, 속도 한도 넘음.
// people_check <장면.scene> [장면 …] [--envs 256] [--steps 3000]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "env_api.h"

namespace {

// 로봇 앞 1.5 m 를 왼쪽 1.2 m → 오른쪽 1.2 m 로 가로지를 길이 비었나 (사람 r 0.25, 라이다에서 그 길까지 장면이 안 가림)
bool crossing_ok(const sim::EnvBatch& b, const sim::EnvState& e, float r) {
  const sim::MapDev& m = b.maps[e.map];
  const float s = std::sin(e.th), c = std::cos(e.th);
  const float lx = e.x + c * b.cfg.mount_x, ly = e.y + s * b.cfg.mount_x;
  for (float left = -1.4f; left <= 1.4f; left += 0.05f) {
    const float x = e.x + c * 1.5f - s * left, y = e.y + s * 1.5f + c * left;
    if (!sim::person_free(b.cfg, m, x, y, r)) return false;
    const float dx = x - lx, dy = y - ly, d = std::sqrt(dx * dx + dy * dy);
    const float t = sim::raycast(m.scene, sim::V3{lx, ly, b.cfg.lidar_z}, sim::V3{dx / d, dy / d, 0.f}, d, nullptr);
    if (t < d) return false;
  }
  return true;
}

int test_visible(const sim::EnvHostCfg& hc, const std::vector<sim::World>& worlds, const sim::LidarModel& lidar) {
  const int n = 64;
  sim::Env with(hc, worlds, lidar, n, 11, false), without(hc, worlds, lidar, n, 11, false);
  with.reset();
  without.reset();
  std::vector<sim::EnvState> st;
  with.get_state(st);
  int pick = -1;
  for (int i = 0; i < n && pick < 0; ++i)
    if (crossing_ok(with.batch(), st[i], 0.25f)) pick = i;
  if (pick < 0) { std::printf("보임 시험: 가로지를 빈 자리를 못 찾음\n"); return 1; }
  sim::EnvState e = st[pick];
  const float s = std::sin(e.th), c = std::cos(e.th);
  sim::People& P = e.people;
  P.n = 1;
  P.cyl[0] = sim::Cyl{e.x + c * 1.5f - s * 1.2f, e.y + s * 1.5f + c * 1.2f, 0.25f, 1.7f};
  P.gx[0] = e.x + c * 1.5f + s * 1.2f;
  P.gy[0] = e.y + s * 1.5f - c * 1.2f;
  P.speed[0] = 1.0f;
  P.wait[0] = 0.f;
  P.mode[0] = sim::kPersonWalk | sim::kPersonIgnore;
  with.set_state(pick, e);
  std::vector<float> act((size_t)mdp::act::DIM * n, 0.f);   // 로봇은 서 있음
  const sim::EnvBatch& b = with.batch();
  const sim::MapDev& m = b.maps[e.map];
  std::printf("보임 시험: 환경 %d, 로봇 (%.2f, %.2f) 방향 %.2f, 사람 r 0.25 · 1.0 m/s 로 로봇 앞 1.5 m 를 왼쪽 → 오른쪽\n", pick, e.x, e.y, e.th);
  int scans_with_person = 0, total = 0, max_beams = 0;
  long depth_diff = 0, map_diff = 0;
  for (int t = 0; t < 30; ++t) {
    with.step(act.data());
    without.step(act.data());
    std::vector<sim::EnvState> sw;
    with.get_state(sw);
    const sim::EnvState& ew = sw[pick];
    // 지금 자세에서 스캔 한 장 (env.h scan_begin · scan_beam 과 같은 함수): 사람을 맞은 빔 수
    sim::Shared sh;
    sim::scan_begin(b, ew, sh);
    int beams = 0, valid = 0;
    for (int j = 0; j < sh.n_beams; ++j) {
      sim::BeamRay ray;
      const float r = sim::lidar_beam_real(sim::with_people(m.scene, ew.people), ew.noise, sh.scan, j, &ray);
      if (r > 0.f) { ++valid; beams += ray.obj == sim::kObjPerson; }
    }
    std::vector<uint8_t> d1, d2;
    std::vector<float> n1, n2, q1, q2;
    with.get_obs(d1, n1, q1);
    without.get_obs(d2, n2, q2);
    const size_t D = (size_t)mdp::obs::DEPTH_SIZE, off = (size_t)mdp::obs::NEAR_SIZE + mdp::obs::FAR_SIZE;   // 깊이 = 2D 관측 끝 칸
    long dd = 0;
    for (size_t k = 0; k < D; ++k) dd += d1[pick * mdp::obs::IMG_SIZE + off + k] != d2[pick * mdp::obs::IMG_SIZE + off + k];
    const float px = ew.people.cyl[0].x - ew.x, py = ew.people.cyl[0].y - ew.y;
    if (t % 3 == 0)
      std::printf("  t %.1f s: 사람 로봇 기준 (앞 %.2f, 왼 %+.2f) m, 이 장 사람 맞은 빔 %d / 유효 %d, 깊이 다른 칸 %ld / %zu, 몸통 간격 %.2f m, 판 끝 %d\n",
                  0.1 * (t + 1), c * px + s * py, -s * px + c * py, beams, valid, dd, D, ew.people.near_d, ew.done);
    scans_with_person += beams > 0;
    max_beams = beams > max_beams ? beams : max_beams;
    depth_diff += dd;
    ++total;
  }
  std::vector<int16_t> m1, m2;
  with.get_map(m1);
  without.get_map(m2);
  const size_t S2 = (size_t)hc.cfg.size * hc.cfg.size;
  for (size_t k = 0; k < S2; ++k) {
    const int16_t a = m1[pick * S2 + k], z = m2[pick * S2 + k];
    map_diff += a != sim::kUnknown && a > 0 && (z == sim::kUnknown || z <= 0);
  }
  std::printf("보임: 사람이 스캔에 나타난 장 %d / %d (한 장 최대 %d 빔), 깊이 관측 다른 칸 합 %ld, 사람 때문에 막힘으로 바뀐 지도 칸 %ld\n", scans_with_person,
              total, max_beams, depth_diff, map_diff);
  const bool ok = scans_with_person >= total / 2 && depth_diff > 0 && map_diff > 0;
  std::printf("%s\n", ok ? "보임 시험 통과" : "보임 시험 실패");
  return ok ? 0 : 1;
}

int test_walls(const sim::EnvHostCfg& hc, const std::vector<sim::World>& worlds, const sim::LidarModel& lidar, int n_env, int n_steps) {
  sim::Env env(hc, worlds, lidar, n_env, 3, true);
  env.set_people_max(std::vector<int>(n_env, sim::kMaxPeople));
  env.reset();
  const sim::EnvCfg& c = hc.cfg;
  float* d_act;
  cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * n_env);
  uint64_t rng = 17;
  std::vector<sim::EnvState> prev, st;
  env.get_state(prev);
  long person_steps = 0, in_blocked = 0, off_floor = 0, overlap = 0, spawn_near = 0, too_fast = 0, moved = 0, walker_steps = 0, resets = 0,
       hit_person = 0, hit_any = 0;
  double max_speed = 0;
  const float dt_act = c.substeps * c.dt;
  for (int t = 0; t < n_steps; ++t) {
    std::vector<float> act((size_t)mdp::act::DIM * n_env, 0.f);
    for (int i = 1; i < n_env; i += 2) sim::random_action(c, rng, &act[(size_t)mdp::act::DIM * i]);   // 홀수 환경은 무작위 행동 (비켜섬·충돌도 보게)
    cudaMemcpy(d_act, act.data(), sizeof(float) * mdp::act::DIM * n_env, cudaMemcpyHostToDevice);
    env.step(d_act);
    env.get_state(st);
    std::vector<float> rw;
    std::vector<int32_t> dn;
    env.get_reward(rw, dn);
    for (int i = 0; i < n_env; ++i) {
      const sim::EnvState& e = st[i];
      const sim::World& w = worlds[e.map];
      const sim::People& P = e.people;
      hit_person += P.hit_person;
      hit_any += dn[i] == 2;
      const bool fresh = dn[i] != 0;   // 이번에 리셋된 판
      resets += fresh;
      for (int k = 0; k < P.n; ++k) {
        const sim::Cyl& C = P.cyl[k];
        ++person_steps;
        // 정답 격자로 따로: 원기둥 ~ 막힌 칸 정사각형 거리 < r 이면 겹침
        const int x0 = (int)std::floor((C.x - C.r - w.ox) / c.res), x1 = (int)std::floor((C.x + C.r - w.ox) / c.res);
        const int y0 = (int)std::floor((C.y - C.r - w.oy) / c.res), y1 = (int)std::floor((C.y + C.r - w.oy) / c.res);
        bool hit = false;
        for (int y = y0; y <= y1 && !hit; ++y)
          for (int x = x0; x <= x1 && !hit; ++x) {
            if (x < 0 || y < 0 || x >= w.gw || y >= w.gh) { hit = true; break; }
            if (!(w.grid[(size_t)y * w.gw + x] & sim::kCellBlocked)) continue;
            const float lx = w.ox + x * c.res, ly = w.oy + y * c.res;
            const float qx = std::fmax(std::fmax(lx - C.x, C.x - (lx + c.res)), 0.f), qy = std::fmax(std::fmax(ly - C.y, C.y - (ly + c.res)), 0.f);
            hit = qx * qx + qy * qy < C.r * C.r;
          }
        in_blocked += hit;
        const int cx = (int)std::floor((C.x - w.ox) / c.res), cy = (int)std::floor((C.y - w.oy) / c.res);
        off_floor += !(cx >= 0 && cy >= 0 && cx < w.gw && cy < w.gh && (w.grid[(size_t)cy * w.gw + cx] & sim::kCellFloor));
        for (int j = k + 1; j < P.n; ++j) {
          const float dx = C.x - P.cyl[j].x, dy = C.y - P.cyl[j].y, rr = C.r + P.cyl[j].r;
          overlap += dx * dx + dy * dy < rr * rr - 1e-6f;
        }
        if (fresh && e.step == 0) {
          const float dx = C.x - e.x, dy = C.y - e.y;
          spawn_near += dx * dx + dy * dy < 1.5f * 1.5f;
        } else if (!fresh && k < prev[i].people.n && (P.mode[k] & sim::kPersonWalk)) {
          const float dx = C.x - prev[i].people.cyl[k].x, dy = C.y - prev[i].people.cyl[k].y;
          const double v = std::sqrt(dx * dx + dy * dy) / dt_act;
          ++walker_steps;
          moved += v > 1e-3;
          max_speed = std::max(max_speed, v);
          too_fast += v > 1.3 + 1e-3;
        }
      }
    }
    prev = st;
  }
  std::printf("벽 시험: 환경 %d × 행동 %d (%.0f s), 사람-걸음 %ld, 판 리셋 %ld\n", n_env, n_steps, n_steps * dt_act, person_steps, resets);
  std::printf("  막힌 칸과 겹침 %ld, 바닥 밖 %ld, 사람끼리 겹침 %ld, 판 시작에 출발점 1.5 m 안 %ld, 1.3 m/s 넘음 %ld (최대 %.3f m/s)\n", in_blocked, off_floor,
              overlap, spawn_near, too_fast, max_speed);
  std::printf("  걷는 사람이 움직인 걸음 %.1f %%, 충돌 판 끝 %ld 중 사람 충돌 %ld\n", 100.0 * moved / std::max(1L, walker_steps), hit_any, hit_person);
  const bool ok = in_blocked == 0 && off_floor == 0 && overlap == 0 && spawn_near == 0 && too_fast == 0 && person_steps > 0 && moved > 0;
  std::printf("%s\n", ok ? "벽 시험 통과" : "벽 시험 실패");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> scenes;
  int n_env = 256, n_steps = 3000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else scenes.push_back(a);
  }
  if (scenes.empty()) { std::fprintf(stderr, "people_check <장면.scene> [장면 …] [--envs 256] [--steps 3000]\n"); return 2; }
  sim::EnvHostCfg hc = sim::load_env_cfg(SIM_CONFIG_DIR "/env.yaml");
  hc.cfg.curriculum = 0;   // 사람 수를 set_people_max 로 정함 (커리큘럼이 켜져 있으면 단계가 정함)
  const sim::LidarModel lidar = sim::load_lidar_model(SIM_CONFIG_DIR "/sensor_model.yaml");
  std::vector<sim::World> worlds;
  for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
  const int a = test_visible(hc, worlds, lidar);
  const int b = test_walls(hc, worlds, lidar, n_env, n_steps);
  return a | b;
}
