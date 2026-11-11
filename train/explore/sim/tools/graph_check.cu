// 그래프 모드 관측(mdp/graph.h, mdp/obs_layout.h mdp::graph) 검사 도구. GPU 환경을 그래프 모드로 돌리며 관측마다(판 시작 + 결정마다) 불변식을 본다:
//   1 간선 대칭 (i 의 비트 k ⇔ 이웃 j 의 비트 23 − k), 격자 밖 이웃 없음, 위 8 비트 0
//   2 무효 노드는 특징 8 칸·간선 모두 0
//   3 현재 노드는 범위 안·유효, 유틸리티 0 (방문함)
//   4 유효 노드는 모두 현재 노드에서 간선으로 닿음
//   5 지역 길잡이 = 유틸리티 노드마다 현재 노드까지 최단 경로(호스트에서 다시 풂)의 노드들과 정확히 같음 (현재 노드 뺌) — 닿는 유틸리티 노드마다 길잡이 경로
//   6 특징 값: 유효 0/255, 길잡이 0/255, 전역 길잡이·남는 칸 0, dx·dy = 128 + 4 × (열·행 차)
//   7 이웃 칸(cand) ≥ 0 ⇔ 현재 노드의 간선 비트, 이웃 칸끼리 간격 = 스텐실 × 노드 간격
//   8 전역 숫자 유한, 8 ~ 15 = 0
// --twice: 같은 씨앗으로 두 번 돌려 관측·상태 해시가 같은지 (결정성). --png: 환경 하나의 한 결정 관측을 그림으로 (지도 + 프론티어 + 노드·간선·유틸리티·길잡이 + 지나온 길).
// 행동: --policy guide(가장 가까운 유틸리티 노드 쪽 첫 이웃) · random(무작위 유효 이웃) · mix(환경 번호 짝·홀로 둘 다, 기본)
// graph_check <장면.scene> [장면 …] [--envs 64] [--steps 300] [--people 2] [--seed 7] [--policy mix] [--graph-stand 0|1] [--graph-open 0|1] [--twice] [--png 파일 --png-env 0 --png-at 120]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "env_api.h"

namespace {
uint64_t fnv(const void* p, size_t n, uint64_t h) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
template <class T>
uint64_t fnv_v(const std::vector<T>& v, uint64_t h) { return fnv(v.data(), v.size() * sizeof(T), h); }

// PNG (압축 없는 deflate 블록 — 바깥 라이브러리 없이)
uint32_t crc32_of(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
  for (size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return c;
}
void put32(std::vector<uint8_t>& o, uint32_t v) { for (int s = 24; s >= 0; s -= 8) o.push_back((uint8_t)(v >> s)); }
void chunk(std::vector<uint8_t>& o, const char* tag, const std::vector<uint8_t>& d) {
  put32(o, (uint32_t)d.size());
  const size_t at = o.size();
  o.insert(o.end(), tag, tag + 4);
  o.insert(o.end(), d.begin(), d.end());
  put32(o, ~crc32_of(&o[at], o.size() - at));
}
bool write_png(const std::string& file, int w, int h, const std::vector<uint8_t>& rgb) {
  std::vector<uint8_t> raw;
  raw.reserve((size_t)h * (3 * w + 1));
  for (int y = 0; y < h; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), &rgb[(size_t)y * w * 3], &rgb[(size_t)y * w * 3] + 3 * w);
  }
  std::vector<uint8_t> z{0x78, 0x01};
  uint32_t a = 1, b = 0;
  for (uint8_t v : raw) { a = (a + v) % 65521u; b = (b + a) % 65521u; }
  for (size_t at = 0; at < raw.size();) {
    const size_t n = std::min<size_t>(65535, raw.size() - at);
    z.push_back(at + n == raw.size() ? 1 : 0);
    z.push_back((uint8_t)(n & 255)); z.push_back((uint8_t)(n >> 8));
    z.push_back((uint8_t)(~n & 255)); z.push_back((uint8_t)((~n >> 8) & 255));
    z.insert(z.end(), &raw[at], &raw[at] + n);
    at += n;
  }
  put32(z, (b << 16) | a);
  std::vector<uint8_t> o{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, ihdr;
  put32(ihdr, (uint32_t)w); put32(ihdr, (uint32_t)h);
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
  chunk(o, "IHDR", ihdr);
  chunk(o, "IDAT", z);
  chunk(o, "IEND", {});
  FILE* f = std::fopen(file.c_str(), "wb");
  if (!f) return false;
  std::fwrite(o.data(), 1, o.size(), f);
  std::fclose(f);
  return true;
}

struct Canvas {
  int w, h;
  std::vector<uint8_t> px;
  void set(int x, int y, uint32_t c) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    uint8_t* p = &px[((size_t)y * w + x) * 3];
    p[0] = (uint8_t)(c >> 16); p[1] = (uint8_t)(c >> 8); p[2] = (uint8_t)c;
  }
  void rect(int x0, int y0, int x1, int y1, uint32_t c) { for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) set(x, y, c); }
  void line(float x0, float y0, float x1, float y1, uint32_t c, int th = 1) {
    const int n = (int)std::ceil(std::max(std::fabs(x1 - x0), std::fabs(y1 - y0))) + 1;
    for (int i = 0; i <= n; ++i) {
      const float t = (float)i / n;
      const int x = (int)std::lround(x0 + (x1 - x0) * t), y = (int)std::lround(y0 + (y1 - y0) * t);
      rect(x - th / 2, y - th / 2, x - th / 2 + th, y - th / 2 + th, c);
    }
  }
  void disk(float cx, float cy, float r, uint32_t c, bool ring = false) {
    for (int y = (int)(cy - r - 1); y <= (int)(cy + r + 1); ++y)
      for (int x = (int)(cx - r - 1); x <= (int)(cx + r + 1); ++x) {
        const float d = std::hypot(x - cx, y - cy);
        if (ring ? (d <= r && d > r - 2.f) : d <= r) set(x, y, c);
      }
  }
};

struct Stats {
  size_t obs = 0, viol[9] = {}, valid = 0, edges = 0, util = 0, guide = 0, no_nb = 0, util_obs0 = 0;
  uint64_t hash = 1469598103934665603ull;
  int ends[mdp::cur::kNumDone] = {};
};
}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);
  std::vector<std::string> scenes;
  std::string env_path = SIM_CONFIG_DIR "/env.yaml", model_path = SIM_CONFIG_DIR "/sensor_model.yaml", policy = "mix", png;
  int n_env = 64, n_steps = 300, people = 2, graph_stand = -1, graph_open = -1, png_env = 0, png_at = 120;
  uint64_t seed = 7;
  bool twice = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--envs" && i + 1 < argc) n_env = std::stoi(argv[++i]);
    else if (a == "--steps" && i + 1 < argc) n_steps = std::stoi(argv[++i]);
    else if (a == "--people" && i + 1 < argc) people = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    else if (a == "--policy" && i + 1 < argc) policy = argv[++i];
    else if (a == "--graph-stand" && i + 1 < argc) graph_stand = std::stoi(argv[++i]);
    else if (a == "--graph-open" && i + 1 < argc) graph_open = std::stoi(argv[++i]);
    else if (a == "--twice") twice = true;
    else if (a == "--png" && i + 1 < argc) png = argv[++i];
    else if (a == "--png-env" && i + 1 < argc) png_env = std::stoi(argv[++i]);
    else if (a == "--png-at" && i + 1 < argc) png_at = std::stoi(argv[++i]);
    else if (a == "--env" && i + 1 < argc) env_path = argv[++i];
    else if (a[0] == '-') { std::fprintf(stderr, "모르는 인자 %s\n", a.c_str()); return 2; }
    else scenes.push_back(a);
  }
  if (scenes.empty()) {
    std::fprintf(stderr, "graph_check <장면.scene> [장면 …] [--envs 64] [--steps 300] [--people 2] [--seed 7] [--policy mix|guide|random] [--graph-stand 0|1] [--graph-open 0|1] [--twice] [--png 파일 --png-env 0 --png-at 120]\n");
    return 2;
  }
  sim::EnvHostCfg hc = sim::load_env_cfg(env_path);
  hc.cfg.action_mode = sim::kActPath;
  hc.cfg.graph_on = 1;
  hc.cfg.curriculum = 0;
  if (graph_stand >= 0) hc.cfg.graph_stand = graph_stand;
  if (graph_open >= 0) hc.cfg.graph_open = graph_open;
  const sim::EnvCfg& c = hc.cfg;
  const sim::LidarModel lidar = sim::load_lidar_model(model_path);
  std::vector<sim::World> worlds;
  try {
    for (const auto& p : scenes) worlds.push_back(sim::build_world(p, sim::load_scene(p), hc));
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "graph_check: %s\n", ex.what());
    return 1;
  }
  using namespace mdp::graph;
  namespace gg = mdp::graphgen;
  constexpr int NN = mdp::graph::N, A = mdp::act::DIM;
  const int step_cells = (int)(NODE_RES * c.inv_res + 0.5f);

  const auto run = [&](Stats& S, bool draw) {
    sim::Env env(hc, worlds, lidar, n_env, seed, true);
    env.set_people_max(std::vector<int>(n_env, people));
    env.reset();
    env.sync();
    float* d_act;
    cudaMalloc(&d_act, sizeof(float) * A * n_env);
    std::vector<float> act((size_t)A * n_env, 0.f), glob, rew;
    std::vector<uint8_t> feat;
    std::vector<uint32_t> edge;
    std::vector<int32_t> cur, cand, done;
    std::vector<sim::EnvState> st;
    std::vector<int> dist(NN), expect(NN), queue(NN);
    std::vector<float> trail;   // 그릴 환경이 이 판에 지나온 자리 (지도 자세)
    uint64_t rng = seed * 77ull + 5;
    for (int t = -1; t < n_steps; ++t) {
      if (t >= 0) {
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
        env.step(d_act);
        env.sync();
        env.get_reward(rew, done);
        for (int i = 0; i < n_env; ++i) S.ends[done[i]] += done[i] != 0;
        if (done[png_env]) trail.clear();
      }
      env.get_graph(feat, edge, cur, glob, cand);
      env.get_state(st);
      S.hash = fnv_v(feat, fnv_v(edge, fnv_v(cur, fnv_v(glob, fnv_v(cand, S.hash)))));
      for (const auto& s : st) S.hash = fnv(&s.x, 3 * sizeof(float), S.hash);
      trail.push_back(st[png_env].x + st[png_env].ex);
      trail.push_back(st[png_env].y + st[png_env].ey);
      for (int i = 0; i < n_env; ++i) {
        const uint8_t* F = &feat[(size_t)i * NN * FEAT];
        const uint32_t* E = &edge[(size_t)i * NN];
        const int32_t* C = &cand[(size_t)i * NB];
        const float* G = &glob[(size_t)i * GLOB];
        const int cu = cur[i];
        ++S.obs;
        size_t v[9] = {};
        v[3] += cu < 0 || cu >= NN;
        if (cu < 0 || cu >= NN) { S.viol[3] += 1; continue; }
        v[3] += F[cu * FEAT + F_VALID] != 255 || F[cu * FEAT + F_UTIL] != 0;
        int n_util = 0;
        for (int j = 0; j < NN; ++j) {
          const uint8_t* f = &F[j * FEAT];
          const bool valid = f[F_VALID] != 0;
          v[1] += (E[j] >> NB) != 0u;
          for (int k = 0; k < NB; ++k) {
            if (!((E[j] >> k) & 1u)) continue;
            const int nb = gg::node_nb(j, k);
            v[1] += nb < 0 || !((E[nb] >> (NB - 1 - k)) & 1u);
            ++S.edges;
          }
          if (!valid) {
            for (int q = 0; q < FEAT; ++q) v[2] += f[q] != 0;
            v[2] += E[j] != 0u;
            continue;
          }
          ++S.valid;
          S.util += f[F_UTIL] != 0;
          n_util += f[F_UTIL] != 0;
          S.guide += f[F_LGUIDE] != 0;
          v[6] += f[F_VALID] != 255 || (f[F_LGUIDE] != 0 && f[F_LGUIDE] != 255) || f[F_GGUIDE] != 0 || f[F_RSV0] != 0 || f[F_RSV1] != 0;
          v[6] += f[F_DX] != 128 + 4 * (j % GW - cu % GW) || f[F_DY] != 128 + 4 * (j / GW - cu / GW);
        }
        S.util_obs0 += n_util == 0;
        // 4·5: 최단 거리를 호스트에서 다시 (넓이 우선 + 풀기)
        for (int j = 0; j < NN; ++j) { dist[j] = gg::kInf; expect[j] = 0; }
        dist[cu] = 0;
        for (bool ch = true; ch;) {
          ch = false;
          for (int j = 0; j < NN; ++j) {
            if (!E[j]) continue;
            const int nd = gg::relax(dist.data(), E[j], j);
            if (nd < dist[j]) { dist[j] = nd; ch = true; }
          }
        }
        for (int j = 0; j < NN; ++j) v[4] += F[j * FEAT + F_VALID] != 0 && dist[j] >= gg::kInf;
        for (int j = 0; j < NN; ++j) {
          if (F[j * FEAT + F_UTIL] == 0 || dist[j] >= gg::kInf) continue;
          for (int q = j; q >= 0 && q != cu && !expect[q]; q = gg::pred_of(dist.data(), E[q], q)) expect[q] = 1;
        }
        for (int j = 0; j < NN; ++j) v[5] += (F[j * FEAT + F_LGUIDE] != 0) != (expect[j] != 0);
        // 7
        int k0 = -1;
        for (int k = 0; k < NB; ++k) {
          v[7] += (C[k] >= 0) != (((E[cu] >> k) & 1u) != 0u);
          if (C[k] < 0) continue;
          if (k0 < 0) k0 = k;
          int dy0, dx0, dy1, dx1;
          gg::stencil(k0, &dy0, &dx0);
          gg::stencil(k, &dy1, &dx1);
          const int gw = worlds[st[i].map].gw;
          v[7] += (C[k] % gw - C[k0] % gw) != (dx1 - dx0) * step_cells || (C[k] / gw - C[k0] / gw) != (dy1 - dy0) * step_cells;
        }
        S.no_nb += k0 < 0;
        for (int q = 0; q < GLOB; ++q) v[8] += !std::isfinite(G[q]) || (q >= 8 && G[q] != 0.f);
        for (int q = 1; q < 9; ++q) S.viol[q] += v[q] != 0;
        // 다음 행동
        int vk[NB], nv = 0;
        for (int k = 0; k < NB; ++k) if (C[k] >= 0) vk[nv++] = k;
        int k_sel = (int)(dm::splitmix64(rng) % (uint64_t)NB);   // 유효 이웃이 없으면 아무 번호 (제자리 회전)
        const bool use_guide = policy == "guide" || (policy == "mix" && (i & 1));
        if (nv > 0) {
          k_sel = vk[dm::splitmix64(rng) % (uint64_t)nv];
          int best = -1;
          for (int j = 0; use_guide && j < NN; ++j)
            if (F[j * FEAT + F_UTIL] != 0 && dist[j] < gg::kInf && (best < 0 || dist[j] < dist[best])) best = j;
          if (best >= 0) {
            int j = best;
            for (int p = gg::pred_of(dist.data(), E[j], j); p >= 0 && p != cu; p = gg::pred_of(dist.data(), E[j], j)) j = p;
            for (int k = 0; k < NB; ++k) if (((E[cu] >> k) & 1u) && gg::node_nb(cu, k) == j) k_sel = k;
          }
        }
        act[(size_t)A * i] = (float)k_sel;
      }
      if (draw && !png.empty() && t == png_at) {
        const int i = png_env;
        const sim::World& W = worlds[st[i].map];
        std::vector<int16_t> lmap;
        env.get_map(lmap);
        const int16_t* L = &lmap[(size_t)i * c.size * c.size];
        const mdp::MapView mv{L, c.size, W.gw, W.gh};
        const int sx = (int)std::floor((st[i].x0 - W.ox) * c.inv_res), sy = (int)std::floor((st[i].y0 - W.oy) * c.inv_res);
        const int ox = ((sx % step_cells) + step_cells) % step_cells, oy = ((sy % step_cells) + step_cells) % step_cells;
        const int Z = 4;   // 칸당 화소
        Canvas cv{W.gw * Z, W.gh * Z, std::vector<uint8_t>((size_t)W.gw * Z * W.gh * Z * 3, 0)};
        const auto py = [&](float cy) { return (float)(W.gh * Z) - cy * Z; };   // y 는 위로
        for (int y = 0; y < W.gh; ++y)
          for (int x = 0; x < W.gw; ++x) {
            const uint8_t cl = mdp::cls_at(mv, x, y);
            uint32_t col = cl == mdp::kUnk ? 0x50545Au : cl == mdp::kOcc ? 0x000000u : 0xF4F4F0u;
            if (cl == mdp::kFree && mdp::frontier_cell(mv, x, y)) col = 0xFF8A00u;
            cv.rect(x * Z, (W.gh - 1 - y) * Z, x * Z + Z, (W.gh - y) * Z, col);
          }
        const uint8_t* F = &feat[(size_t)i * NN * FEAT];
        const uint32_t* E = &edge[(size_t)i * NN];
        const auto nx = [&](int j) { return (ox + (j % GW) * step_cells + 0.5f) * Z; };
        const auto ny = [&](int j) { return py(oy + (j / GW) * step_cells + 0.5f); };
        for (int j = 0; j < NN; ++j)
          for (int k = NB / 2; k < NB; ++k)
            if ((E[j] >> k) & 1u) cv.line(nx(j), ny(j), nx(gg::node_nb(j, k)), ny(gg::node_nb(j, k)), 0x9FB4D8u);
        for (size_t q = 2; q < trail.size(); q += 2)
          cv.line((trail[q - 2] - W.ox) * c.inv_res * Z, py((trail[q - 1] - W.oy) * c.inv_res), (trail[q] - W.ox) * c.inv_res * Z, py((trail[q + 1] - W.oy) * c.inv_res), 0x00A0C8u, 2);
        for (int j = 0; j < NN; ++j) {
          const uint8_t* f = &F[j * FEAT];
          if (!f[F_VALID]) continue;
          if (f[F_LGUIDE]) cv.disk(nx(j), ny(j), 11.f, 0xD0209Cu, true);
          if (f[F_UTIL]) cv.disk(nx(j), ny(j), 3.f + 6.f * f[F_UTIL] / 255.f, 0x18A048u);
          else cv.disk(nx(j), ny(j), 3.f, 0x4060C0u);
        }
        cv.disk(nx(cur[i]), ny(cur[i]), 8.f, 0xE02020u, true);
        cv.disk(nx(cur[i]), ny(cur[i]), 7.f, 0xE02020u, true);
        const float rx = (st[i].x + st[i].ex - W.ox) * c.inv_res * Z, ry = py((st[i].y + st[i].ey - W.oy) * c.inv_res);
        cv.disk(rx, ry, 5.f, 0xFFD400u);
        cv.line(rx, ry, rx + 14.f * std::cos(st[i].th + st[i].eth), ry - 14.f * std::sin(st[i].th + st[i].eth), 0xFFD400u, 2);
        const bool ok = write_png(png, cv.w, cv.h, cv.px);
        std::printf("그림 %s: %s, 환경 %d 결정 %d (맵 %s %d×%d 칸, 칸당 %d 화소) — 회색 모름·흰색 빈 칸·검정 막힘·주황 프론티어 칸, 파란 점 유효 노드·초록 점 유틸리티 노드(클수록 큼)·"
                    "자주 고리 지역 길잡이·빨간 고리 현재 노드·노란 점 로봇(선 = 앞)·청록 선 이 판에 지나온 길\n", png.c_str(), ok ? "씀" : "못 씀", i, t, W.name.c_str(), W.gw, W.gh, Z);
      }
    }
    cudaFree(d_act);
  };

  Stats S1;
  run(S1, true);
  size_t bad = 0;
  for (int q = 1; q < 9; ++q) bad += S1.viol[q];
  std::printf("그래프 불변식 (칸 기준 %s · 프론티어 %s, 정책 %s, 환경 %d × 결정 %d, 사람 %d): 관측 %zu 개 중 위반 — 1 간선 대칭 %zu · 2 무효 노드 0 %zu · 3 현재 노드 %zu · 4 유효 노드가 현재 노드에서 닿음 %zu · "
              "5 길잡이 = 최단 경로 %zu · 6 특징 값 %zu · 7 이웃 칸 %zu · 8 전역 숫자 %zu\n", c.graph_stand ? "설 수 있는 칸" : "빈 칸", c.graph_open ? "막는 칸만" : "모두", policy.c_str(), n_env, n_steps, people, S1.obs,
              S1.viol[1], S1.viol[2], S1.viol[3], S1.viol[4], S1.viol[5], S1.viol[6], S1.viol[7], S1.viol[8]);
  std::printf("관측당 유효 노드 %.1f · 간선 %.1f · 유틸리티 노드 %.2f · 길잡이 노드 %.2f, 유효 이웃 없는 관측 %zu · 유틸리티 노드 없는 관측 %zu, 끝난 판 %d (성공 %d · 충돌 %d · 진행 없음 %d)\n",
              (double)S1.valid / S1.obs, (double)S1.edges / 2 / S1.obs, (double)S1.util / S1.obs, (double)S1.guide / S1.obs, S1.no_nb, S1.util_obs0,
              S1.ends[1] + S1.ends[2] + S1.ends[3] + S1.ends[4] + S1.ends[5] + S1.ends[6], S1.ends[mdp::cur::kDeclareOk], S1.ends[mdp::cur::kCollision] + S1.ends[mdp::cur::kCollisionPerson],
              S1.ends[mdp::cur::kNoProgress]);
  bool same = true;
  if (twice) {
    Stats S2;
    run(S2, false);
    same = S1.hash == S2.hash;
    std::printf("결정성: 같은 씨앗 두 번 — 관측·자세 해시 %016llx / %016llx %s\n", (unsigned long long)S1.hash, (unsigned long long)S2.hash, same ? "같음" : "다름");
  }
  const bool ok = bad == 0 && same && S1.valid > 0 && S1.util > 0 && S1.guide > 0;
  std::printf("그래프 검사 %s\n", ok ? "통과" : "실패");
  return ok ? 0 : 1;
}
