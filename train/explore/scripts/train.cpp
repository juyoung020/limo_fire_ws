// train — 탐사 RL 학습 러너 (S6). 시뮬(sim/include/env_api.h) × 신경망·PPO(network/·ppo/) → 학습 뷰어 실행 폴더(viewer/include/runfolder.h).
//
//   train [--envs 2048] [--iters 0] [--minutes 0] [--name explore] [--seed 1] [--lr-max 1e-3] [--mlp 2] [--chunk N] [--eval-every 50] [--ckpt-every 50]
//         [--replay-every 25] [--eval-episodes 32] [--resume] [--root train/explore/runs] [--list train/explore/scenes/train/list.tsv]
//         [--test a.scene,b.scene] [--env sim/config/env.yaml] [--model sim/config/sensor_model.yaml]
//   --iters·--minutes 가 0 이면 그 한계 없음(둘 다 0 이면 Ctrl-C 까지가 아니라 --iters 1000 으로 둠).
//
// 반복 하나 = 24 걸음 롤아웃(act → env.step → store) → update. done 매핑: mdp::cur::terminal(1·2·4·5) → 1 종료, 3 → 3 끊김.
// 실행 폴더 <root>/<name>/: run.json(kind explore_ppo), progress.jsonl(반복마다), episodes.jsonl(표본 환경의 판), evals/test_it<반복>.json,
//   replays/*.xrp(환경 0·1 의 판을 --replay-every 반복마다 한 판씩), ckpt/model_<반복>.bin·model_latest.bin.
// 평가: --eval-every 반복마다 시험 맵마다 별도 Env(환경 --eval-episodes 개, 커리큘럼 끔)로 결정적 정책(act_inference), 환경마다 첫 판.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "curriculums.h"
#include "env_api.h"
#include "ppo.h"
#include "rewards.h"
#include "runfolder.h"
#include "sensor_model.h"

namespace {

constexpr int kLidarN = 248;
constexpr float kLidarFov = 220.f * (float)M_PI / 180.f, kLidarMax = 8.f;
constexpr int kSampleEnvs = 64;    // episodes.jsonl 에 쓰는 표본 환경 수
constexpr int kReplayEnvs = 2;     // --replay-every 반복마다 표본 환경에서 새로 시작하는 판 2 개를 녹화
constexpr int kReplayMax = 8;      // 동시에 녹화 중인 판 상한 (긴 판 — 끊김 12 000 걸음 — 이 녹화 칸을 오래 잡아도 새 녹화가 멈추지 않게)
constexpr int kReplayMapEvery = 10;  // 재생에 로봇이 만든 지도·덮은 벽 칸을 이 걸음마다 넣음 (그 환경 칸만 장치에서 꺼냄)

volatile std::sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

struct Args {
  int envs = 2048, iters = 0, eval_every = 50, ckpt_every = 50, replay_every = 25, eval_eps = 32;
  double minutes = 0;
  uint64_t seed = 1;
  float lr_max = 1e-3f;
  int mlp = 2, chunk = 0;  // chunk 0 = PPOConfig 기본
  bool resume = false;
  std::string name = "explore", root = "train/explore/runs", list = "train/explore/scenes/train/list.tsv";
  std::vector<std::string> test = {"train/explore/scenes/test/bookstore.scene", "train/explore/scenes/test/small-house.scene"};
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml";
};

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, c))
    if (!t.empty()) out.push_back(t);
  return out;
}
std::string stem(const std::string& p) {
  std::string b = rf::basename_of(p);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}
bool file_exists(const std::string& p) {
  struct stat s;
  return stat(p.c_str(), &s) == 0;
}

__global__ void k_map_done(const int32_t* d, int n, int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = d[i] == mdp::cur::kRunning ? 0 : (mdp::cur::terminal(d[i]) ? 1 : 3);
}

// 평가 장부 (장치): 환경마다 첫 판만 — 아직 안 끝났으면 보상을 더하고(double, 걸음 순서대로), 끝나면 판 정보를 적는다.
// 호스트가 걸음마다 동기화하던 것을 GPU 에서 해 걸음 사이 동기화를 없앤다 (결과는 같다).
__global__ void k_eval_book(const int32_t* done, const float* rew, const sim::EpInfo* ep, int n, int* first,
                            sim::EpInfo* fe, double* er) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || first[i]) return;
  er[i] += rew[i];
  if (done[i]) {
    first[i] = 1;
    fe[i] = ep[i];
    fe[i].done = done[i];
  }
}

const char* outcome_of(int32_t d) {
  switch (d) {
    case mdp::cur::kDeclareOk: return rf::outcome::SUCCESS;
    case mdp::cur::kDeclareShort: return rf::outcome::DECLARED_SHORT;
    case mdp::cur::kCollision:
    case mdp::cur::kCollisionPerson: return rf::outcome::COLLISION;
    case mdp::cur::kNoProgress: return rf::outcome::NO_PROGRESS;
    default: return rf::outcome::TRUNCATED;
  }
}

// 재생 한 판 (환경 하나)
struct Rec {
  int map = -1, env = -1;
  std::unique_ptr<rf::ReplayWriter> w;
};

// URDF(xacro) 에서 로봇 치수 읽기 — 재생 화면용 (run.json robot.urdf). 고정값 대신 원본 파일을 그대로 읽는다.
// limo_pro_base.xacro: base_x/y_size(차체 상자), wheelbase·track·wheel_radius·wheel_length(바퀴 4 개), limo_laser origin(라이다)
// limo_fire.urdf.xacro: camera_xyz (platform real). 못 읽으면 robot.urdf_error 에 까닭을 적고 화면은 sim 몸통만 그린다.
rf::Json read_urdf_robot(const std::string& dir, std::string* err) {
  auto slurp = [](const std::string& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("못 엶 " + p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  };
  try {
    const std::string base = slurp(dir + "/limo_pro_base.xacro"), top = slurp(dir + "/limo_fire.urdf.xacro");
    auto prop = [&](const std::string& name) {
      std::smatch m;
      if (!std::regex_search(base, m, std::regex("<xacro:property\\s+name=\"" + name + "\"\\s+value=\"([-0-9.eE]+)\"")))
        throw std::runtime_error("xacro 속성 없음 " + name);
      return std::stod(m[1].str());
    };
    auto xyz3 = [](const std::string& s) {
      std::stringstream ss(s);
      double x = 0, y = 0, z = 0;
      ss >> x >> y >> z;
      return std::vector<double>{x, y, z};
    };
    const double bx = prop("base_x_size"), by = prop("base_y_size"), wb = prop("wheelbase"), tr = prop("track"), wr = prop("wheel_radius"),
                 wl = prop("wheel_length");
    std::smatch m;
    if (!std::regex_search(base, m, std::regex("<xacro:limo_laser[^>]*>\\s*<origin\\s+xyz=\"([^\"]+)\"")))
      throw std::runtime_error("limo_laser origin 없음");
    const auto lz = xyz3(m[1].str());
    if (!std::regex_search(top, m, std::regex("real='([^']+)'"))) throw std::runtime_error("camera_xyz real 없음");
    const auto cz = xyz3(m[1].str());
    rf::Json wheels = rf::Json::array();
    for (int sx : {1, -1})
      for (int sy : {1, -1}) wheels.push(rf::Json::array(std::vector<double>{sx * wb / 2, sy * tr / 2}));
    rf::Json u = rf::Json::object();
    u.set("source", dir).set("base_box", rf::Json::array(std::vector<double>{bx, by})).set("wheels", wheels)
        .set("wheel_diameter", 2 * wr).set("wheel_width", wl).set("laser", rf::Json::array(std::vector<double>{lz[0], lz[1]}))
        .set("camera", rf::Json::array(std::vector<double>{cz[0], cz[1]}));
    return u;
  } catch (const std::exception& e) {
    *err = e.what();
    return rf::Json();
  }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string k = argv[i];
      auto nx = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(k + " 뒤에 값");
        return argv[++i];
      };
      if (k == "--envs") a.envs = std::stoi(nx());
      else if (k == "--iters") a.iters = std::stoi(nx());
      else if (k == "--minutes") a.minutes = std::stod(nx());
      else if (k == "--name") a.name = nx();
      else if (k == "--seed") a.seed = std::stoull(nx());
      else if (k == "--lr-max") a.lr_max = std::stof(nx());
      else if (k == "--mlp") a.mlp = std::stoi(nx());
      else if (k == "--chunk") a.chunk = std::stoi(nx());
      else if (k == "--eval-every") a.eval_every = std::stoi(nx());
      else if (k == "--ckpt-every") a.ckpt_every = std::stoi(nx());
      else if (k == "--replay-every") a.replay_every = std::stoi(nx());
      else if (k == "--eval-episodes") a.eval_eps = std::stoi(nx());
      else if (k == "--resume") a.resume = true;
      else if (k == "--root") a.root = nx();
      else if (k == "--list") a.list = nx();
      else if (k == "--test") a.test = split(nx(), ',');
      else if (k == "--env") a.env = nx();
      else if (k == "--model") a.model = nx();
      else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (a.iters <= 0 && a.minutes <= 0) a.iters = 1000;
  std::signal(SIGINT, on_sig);
  std::signal(SIGTERM, on_sig);
  try {
    // ---- 시뮬
    sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
    const sim::LidarModel lidar = sim::load_lidar_model(a.model);
    const sim::EnvCfg& c = hc.cfg;
    if (c.action_mode != sim::kActVelocity || c.act_raw) throw std::runtime_error("env.yaml: 속도 행동·정규 단위여야 함");
    const float dt_act = c.substeps * c.dt;
    std::vector<std::string> skipped;
    std::vector<sim::World> worlds = sim::load_scene_list(a.list, hc, &skipped);
    if (worlds.empty()) throw std::runtime_error("학습 맵이 없음: " + a.list);
    std::fprintf(stderr, "학습 맵 %zu 개 (안 닫혀 뺀 것 %zu)\n", worlds.size(), skipped.size());
    const int N = a.envs;
    sim::Env env(hc, worlds, lidar, N, a.seed, true);
    env.reset();
    env.sync();

    // 시험 맵: 커리큘럼 끔 (맵 하나, 오차 배율 1)
    sim::EnvHostCfg hc_eval = hc;
    hc_eval.cfg.curriculum = 0;
    std::vector<sim::World> tests;
    for (auto& t : a.test) tests.push_back(sim::build_world(stem(t), sim::load_scene(t), hc_eval));

    // ---- PPO
    xppo::PPOConfig pc;
    pc.num_envs = N;
    pc.lr_max = a.lr_max;
    pc.mlp = a.mlp;
    if (a.chunk > 0) pc.chunk = a.chunk;
    xppo::PPO ppo(pc, xnet::explore_spec(), a.seed);
    const std::string dir = a.root + "/" + a.name;
    const std::string ckdir = dir + "/ckpt";
    if (a.resume) {
      const std::string p = ckdir + "/model_latest.bin";
      if (!file_exists(p)) throw std::runtime_error("--resume: 체크포인트 없음 " + p);
      ppo.load(p);
      std::fprintf(stderr, "이어 학습: %s 반복 %d, 학습률 %.2e\n", p.c_str(), ppo.iteration(), ppo.learning_rate());
    }

    // ---- 실행 폴더
    rf::Json scene_meta = rf::Json::array();
    int max_stage = 0;
    for (auto& w : worlds) max_stage = std::max(max_stage, w.stage);
    rf::Json limits = rf::Json::object();
    limits.set("v", c.act_v_fwd).set("vy", c.act_vy_max).set("w", c.act_w_max);
    rf::Json terms = rf::Json::array();
    for (int k = 0; k < mdp::rew::kTerms; ++k) terms.push(mdp::rew::term_name(k));
    rf::Json tests_meta = rf::Json::array();
    for (auto& w : tests) tests_meta.push(w.name);
    // 로봇 모양 (재생 화면): sim 충돌 몸통·센서 자리(env.yaml body) + URDF 치수. 둘은 다를 수 있다 — 화면은 sim 몸통 실선, URDF 차체를 그 위에
    rf::Json robot = rf::Json::object();
    {
      const sim::DepthSpec& D = c.depth.spec;
      const double a0 = std::atan((D.cx - D.valid_col0) / D.fx), a1 = std::atan((D.valid_col1 - D.cx) / D.fx);   // 왼쪽(+)·오른쪽 반각
      rf::Json simb = rf::Json::object();
      simb.set("footprint", rf::Json::array(std::vector<double>{2.0 * c.half_len, 2.0 * c.half_wid}))
          .set("lidar", rf::Json::array(std::vector<double>{c.mount_x, c.mount_y})).set("lidar_fov", (double)kLidarFov).set("lidar_range", (double)kLidarMax)
          .set("camera", rf::Json::array(std::vector<double>{c.cam_x, c.cam_y})).set("camera_hfov", rf::Json::array(std::vector<double>{a0, a1}))
          .set("camera_range", (double)D.max_z);
      robot.set("sim", simb);
      std::string err;
      rf::Json u = read_urdf_robot(LIMO_URDF_DIR, &err);
      if (err.empty()) robot.set("urdf", u);
      else robot.set("urdf_error", err);
    }
    rf::Json meta = rf::Json::object();
    meta.set("robot", robot);
    meta.set("kind", "explore_ppo").set("group", a.name).set("seed", (unsigned long long)a.seed).set("n_envs", N)
        .set("ctrl_hz", 1.0 / dt_act).set("success_coverage", mdp::rew::kSuccessCov).set("action_limits", limits)
        .set("reward_terms", terms).set("n_train_maps", (unsigned long long)worlds.size()).set("test_maps", tests_meta)
        .set("scene_list", a.list).set("steps_per_env", pc.steps_per_env).set("lr_max", a.lr_max)
        .set("n_params", (long long)ppo.ac().n_params)
        .set("ppo", rf::Json::object().set("epochs", pc.epochs).set("minibatches", pc.minibatches).set("clip", pc.clip)
                        .set("gamma", pc.gamma).set("lam", pc.lam).set("entropy_coef", pc.entropy_coef).set("lr", pc.lr)
                        .set("desired_kl", pc.desired_kl).set("max_grad_norm", pc.max_grad_norm));
    rf::RunFolder::Options opt;
    opt.every_s = 0;
    if (a.resume) {
      opt.resume = true;
      opt.resume_iter = ppo.iteration();
    }
    rf::RunFolder run(dir, meta, opt);
    rf::mkdirs(ckdir);
    std::fprintf(stderr, "실행 폴더 %s, 환경 %d, 변수 %ld\n", dir.c_str(), N, ppo.ac().n_params);

    // ---- 버퍼
    float* d_act;
    int* d_done;
    cudaMalloc(&d_act, sizeof(float) * mdp::act::DIM * N);
    cudaMalloc(&d_done, sizeof(int) * N);
    std::vector<float> h_terms((size_t)N * mdp::rew::kTerms), h_rew(N);
    std::vector<int32_t> h_done(N);
    std::vector<sim::EpInfo> h_ep(N);
    std::vector<double> ep_terms((size_t)N * mdp::rew::kTerms, 0.0), ep_rew(N, 0.0);
    std::vector<int> ep_steps(N, 0);
    double total_steps = (double)ppo.iteration() * pc.steps_per_env * N;

    // 재생
    std::vector<Rec> rec;   // 녹화 중인 판들
    int rec_want = 0;       // 새로 시작할 녹화 수 (--replay-every 반복마다 kReplayEnvs 개, 표본 환경에서 새 판이 시작될 때 붙음)
    std::vector<sim::EnvState> st;
    std::vector<float> h_actR(mdp::act::DIM * kSampleEnvs), h_logit(kSampleEnvs), h_val(kSampleEnvs), ranges(kLidarN);
    // 재생 열 (viewer/trainfmt/src/xrp.rs): 자세·잰 속도 · 정책 행동(물리 단위)·선언 확률·뽑은 선언 · 가치 · 보상 항 · 누적 리턴 · 덮음 · 사건 · 사람 원기둥
    std::vector<std::string> cols = {"t", "x", "y", "yaw", "sx", "sy", "syaw", "v", "w", "a_v", "a_vy", "a_w", "p_end", "a_end",
                                     "value", "ret", "coverage", "ev"};
    for (int k = 0; k < mdp::rew::kTerms; ++k) cols.push_back(std::string("r_") + mdp::rew::term_name(k));
    for (int k = 0; k < sim::kMaxPeople; ++k)
      for (const char* q : {"_x", "_y", "_r"}) cols.push_back("h" + std::to_string(k) + q);
    std::vector<std::string> term_names;
    for (int k = 0; k < mdp::rew::kTerms; ++k) term_names.push_back(mdp::rew::term_name(k));
    std::vector<float> row(cols.size());
    std::vector<int16_t> h_lmap((size_t)c.size * c.size);
    std::vector<uint32_t> h_marks((size_t)c.size * c.size / 4);
    std::vector<int8_t> rgrid;
    std::vector<uint8_t> rcov;
    auto start_rec = [&](int env_i, int map) {
      Rec R;
      const sim::World& W = worlds[map];
      std::vector<uint8_t> gt((size_t)W.gw * W.gh);
      for (size_t k = 0; k < gt.size(); ++k) gt[k] = (W.grid[k] & 1) ? rf::gt::WALL : ((W.grid[k] & 2) ? rf::gt::FREE : rf::gt::OUTSIDE);
      rf::Grid g;
      g.w = W.gw; g.h = W.gh; g.res = c.res; g.ox = W.ox; g.oy = W.oy;
      R.w.reset(new rf::ReplayWriter(cols, dt_act, g, gt.data()));
      R.w->set_lidar(kLidarN, -0.5f * kLidarFov, kLidarFov / (kLidarN - 1), kLidarMax);
      R.w->set_limits(c.act_v_fwd, c.act_vy_max, c.act_w_max);
      R.w->set_reward_terms(term_names);
      R.map = map;
      R.env = env_i;
      rec.push_back(std::move(R));
    };
    auto clip1 = [](float x) { return x < -1.f ? -1.f : (x > 1.f ? 1.f : x); };
    auto rec_frame = [&](Rec& R, uint32_t evb) {
      const int ei = R.env;
      const sim::EnvState& s = st[ei];
      const sim::World& W = worlds[R.map];
      const float* a = &h_actR[mdp::act::DIM * ei];
      const float nv = clip1(a[0]);   // 정책 행동은 정규 단위 샘플(자르기 전) → 환경이 하는 대로 잘라 물리 단위로 (sim apply_action)
      std::fill(row.begin(), row.end(), std::numeric_limits<float>::quiet_NaN());
      auto put = [&](const char* name, float v) { row[R.w->col(name)] = v; };
      put("t", s.step * dt_act), put("x", s.x), put("y", s.y), put("yaw", s.th);
      put("sx", s.x + s.ex), put("sy", s.y + s.ey), put("syaw", s.th + s.eth), put("v", s.drive.wv), put("w", s.drive.ww);
      put("a_v", nv >= 0.f ? nv * c.act_v_fwd : nv * c.act_v_back), put("a_vy", clip1(a[1]) * c.act_vy_max), put("a_w", clip1(a[2]) * c.act_w_max);
      put("p_end", 1.f / (1.f + std::exp(-h_logit[ei]))), put("a_end", a[3]), put("value", h_val[ei]), put("ret", (float)ep_rew[ei]);
      put("coverage", s.coverage), put("ev", (float)evb);
      for (int k = 0; k < mdp::rew::kTerms; ++k) row[R.w->col(std::string("r_") + mdp::rew::term_name(k))] = h_terms[(size_t)ei * mdp::rew::kTerms + k];
      for (int k = 0; k < s.people.n && k < sim::kMaxPeople; ++k) {
        const std::string h = "h" + std::to_string(k);
        row[R.w->col(h + "_x")] = s.people.cyl[k].x, row[R.w->col(h + "_y")] = s.people.cyl[k].y, row[R.w->col(h + "_r")] = s.people.cyl[k].r;
      }
      R.w->frame(row.data());
      if ((R.w->n_frames() - 1) % kReplayMapEvery == 0) {
        // 로봇이 만든 지도(로그 오즈 × 1000 → 점유 %, 모름 −1)·본 벽 면 칸 — 이 환경 칸만 장치에서 꺼냄 (지도 격자 원점 = 맵 격자 원점)
        const sim::EnvBatch& b = env.batch();
        const size_t cells = (size_t)c.size * c.size;
        cudaMemcpy(h_lmap.data(), b.lmap + (size_t)ei * cells, sizeof(int16_t) * cells, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_marks.data(), b.marks + (size_t)ei * (cells / 4), sizeof(uint32_t) * (cells / 4), cudaMemcpyDeviceToHost);
        rgrid.assign((size_t)W.gw * W.gh, -1);
        rcov.assign((size_t)W.gw * W.gh, 0);
        for (int j = 0; j < W.gh && j < c.size; ++j)
          for (int i = 0; i < W.gw && i < c.size; ++i) {
            const size_t ci = (size_t)j * c.size + i, k = (size_t)j * W.gw + i;
            const int16_t l = h_lmap[ci];
            if (l != sim::kUnknown) rgrid[k] = (int8_t)std::lround(100.0 / (1.0 + std::exp(-l / 1000.0)));
            rcov[k] = (uint8_t)((h_marks[ci >> 2] >> (8 * (ci & 3) + 3)) & 1u);   // env.h marks bit3 = 본 벽 면 칸 (Env::get_seen bit0)
          }
        R.w->obs(rgrid.data());
        R.w->covered(rcov.data());
      }
      const sim::SceneView sv = W.scene.view();
      float sn, cs;
      sincosf(s.th, &sn, &cs);
      const sim::V3 o{s.x + cs * c.mount_x - sn * c.mount_y, s.y + sn * c.mount_x + cs * c.mount_y, c.lidar_z};
      for (int k = 0; k < kLidarN; ++k) {
        const float aa = s.th - 0.5f * kLidarFov + k * (kLidarFov / (kLidarN - 1));
        uint16_t obj;
        ranges[k] = sim::raycast(sv, o, sim::V3{std::cos(aa), std::sin(aa), 0.f}, kLidarMax, &obj);
      }
      R.w->scan(ranges.data());
    };

    // ---- 평가
    auto evaluate = [&](int it) {
      auto t0 = std::chrono::steady_clock::now();
      const int E = std::min(a.eval_eps, N);
      rf::Json rows = rf::Json::array();
      double s_n = 0, s_succ = 0, s_coll = 0, s_trunc = 0, s_short = 0, s_cov = 0, s_t95 = 0, s_t95n = 0, s_len = 0, s_cad = 0, s_decl = 0, s_rew = 0;
      // 시험 맵들을 한 걸음 루프에서 같이 돈다 (맵마다 따로 Env, 같은 씨앗 규칙). 환경 수가 작아(32) 시뮬 커널이 GPU 를 못 채우므로
      // 맵마다 스트림을 따로 두어 겹친다: 정책(act_inference, PPO 스트림 — 작업 버퍼 공유라 차례로) → 사건 → 맵 스트림에서 env.step·장부.
      // 첫 판 기록은 장치 장부(k_eval_book), 호스트는 kCheck 걸음마다 한 번만 남은 환경 수를 본다 —
      // 다 끝난 뒤 최대 kCheck−1 걸음 더 돌 수 있으나 첫 판 기록은 그대로.
      constexpr int kCheck = 32;
      const size_t NT = tests.size();
      std::vector<std::unique_ptr<sim::Env>> evs;
      for (size_t ti = 0; ti < NT; ++ti) {
        std::vector<sim::World> one{tests[ti]};
        evs.emplace_back(new sim::Env(hc_eval, one, lidar, E, a.seed * 7919ull + 17 + ti, true));
        evs.back()->reset();
      }
      int* d_first;
      sim::EpInfo* d_fe;
      double* d_er;
      float* d_eact;
      cudaMalloc(&d_first, sizeof(int) * E * NT);
      cudaMalloc(&d_fe, sizeof(sim::EpInfo) * E * NT);
      cudaMalloc(&d_er, sizeof(double) * E * NT);
      cudaMalloc(&d_eact, sizeof(float) * mdp::act::DIM * E * NT);
      cudaMemset(d_first, 0, sizeof(int) * E * NT);
      cudaMemset(d_fe, 0, sizeof(sim::EpInfo) * E * NT);
      cudaMemset(d_er, 0, sizeof(double) * E * NT);
      std::vector<int> live(NT, 1), hfirst((size_t)E * NT);
      cudaDeviceSynchronize();  // 리셋(기본 스트림)이 끝난 뒤 맵 스트림(비차단)이 읽게
      const cudaStream_t pst = ppo.ac().stream;
      std::vector<cudaStream_t> mst(NT);
      std::vector<cudaEvent_t> ev_act(NT), ev_obs(NT);
      for (size_t ti = 0; ti < NT; ++ti) {
        cudaStreamCreateWithFlags(&mst[ti], cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ev_act[ti], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_obs[ti], cudaEventDisableTiming);
        cudaEventRecord(ev_obs[ti], mst[ti]);
      }
      for (int step = 0; step < c.max_steps + 10; ++step) {
        bool any_live = false;
        for (size_t ti = 0; ti < NT; ++ti) {
          if (!live[ti]) continue;
          any_live = true;
          float* act = d_eact + (size_t)mdp::act::DIM * E * ti;
          const sim::StepOut o = evs[ti]->outputs();
          cudaStreamWaitEvent(pst, ev_obs[ti], 0);   // 이 맵의 지난 걸음 관측이 다 나온 뒤
          ppo.act_inference(o.obs_img, o.obs_1d, act, E);
          cudaEventRecord(ev_act[ti], pst);
          cudaStreamWaitEvent(mst[ti], ev_act[ti], 0);
          evs[ti]->step(act, mst[ti]);
          const sim::StepOut o2 = evs[ti]->outputs();
          k_eval_book<<<(E + 127) / 128, 128, 0, mst[ti]>>>(o2.done, o2.reward, o2.ep, E, d_first + (size_t)E * ti,
                                                             d_fe + (size_t)E * ti, d_er + (size_t)E * ti);
          cudaEventRecord(ev_obs[ti], mst[ti]);
        }
        if (!any_live) break;
        if ((step + 1) % kCheck == 0 || step + 1 == c.max_steps + 10) {
          cudaDeviceSynchronize();  // 맵 스트림(비차단)까지
          cudaMemcpy(hfirst.data(), d_first, sizeof(int) * E * NT, cudaMemcpyDeviceToHost);
          for (size_t ti = 0; ti < NT; ++ti) {
            int left = 0;
            for (int i = 0; i < E; ++i) left += !hfirst[(size_t)E * ti + i];
            if (left == 0) live[ti] = 0;
          }
        }
      }
      cudaDeviceSynchronize();
      for (size_t ti = 0; ti < NT; ++ti) {
        cudaStreamDestroy(mst[ti]);
        cudaEventDestroy(ev_act[ti]);
        cudaEventDestroy(ev_obs[ti]);
      }
      std::vector<sim::EpInfo> hfe((size_t)E * NT);
      std::vector<double> her((size_t)E * NT);
      cudaMemcpy(hfirst.data(), d_first, sizeof(int) * E * NT, cudaMemcpyDeviceToHost);
      cudaMemcpy(hfe.data(), d_fe, sizeof(sim::EpInfo) * E * NT, cudaMemcpyDeviceToHost);
      cudaMemcpy(her.data(), d_er, sizeof(double) * E * NT, cudaMemcpyDeviceToHost);
      cudaFree(d_first);
      cudaFree(d_fe);
      cudaFree(d_er);
      cudaFree(d_eact);
      evs.clear();
      for (size_t ti = 0; ti < NT; ++ti) {
        const int* first = hfirst.data() + (size_t)E * ti;
        const sim::EpInfo* fe = hfe.data() + (size_t)E * ti;
        const double* er = her.data() + (size_t)E * ti;
        double n = E, succ = 0, coll = 0, trunc = 0, shrt = 0, cov = 0, t95 = 0, t95n = 0, len = 0, cad = 0, decl = 0, rws = 0;
        for (int i = 0; i < E; ++i) {
          const auto& f = fe[i];
          if (!first[i]) { trunc += 1; continue; }
          succ += f.done == mdp::cur::kDeclareOk;
          coll += f.done == mdp::cur::kCollision || f.done == mdp::cur::kCollisionPerson;
          trunc += f.done == mdp::cur::kTruncated;
          shrt += f.done == mdp::cur::kDeclareShort;
          cov += f.coverage;
          len += f.t;
          rws += er[i];
          if (f.done == mdp::cur::kDeclareOk || f.done == mdp::cur::kDeclareShort) { cad += f.coverage; decl += 1; }
          if (f.t95 >= 0) { t95 += f.t95; t95n += 1; }
        }
        rf::Json row = rf::Json::object();
        row.set("eval", tests[ti].name).set("split", "test").set("n", n).set("success", succ / n).set("collision", coll / n)
            .set("truncation", trunc / n).set("declare_short", shrt / n).set("coverage", cov / n).set("t95_reached", t95n / n)
            .set("pass", coll == 0 && succ == n);
        if (t95n > 0) row.set("t95", t95 / t95n);
        rows.push(row);
        s_n += n; s_succ += succ; s_coll += coll; s_trunc += trunc; s_short += shrt; s_cov += cov; s_t95 += t95; s_t95n += t95n;
        s_len += len; s_cad += cad; s_decl += decl; s_rew += rws;
        std::fprintf(stderr, "  평가 %s: 성공 %.0f/%d 충돌 %.0f 끊김 %.0f 선언 미달 %.0f 끝 덮음 %.3f t95 %s\n", tests[ti].name.c_str(), succ, E, coll,
                     trunc, shrt, cov / n, t95n > 0 ? std::to_string(t95 / t95n).c_str() : "—");
      }
      rf::Json evj = rf::Json::object();
      evj.set("iter", it).set("ckpt", "model_" + std::to_string(it)).set("rows", rows);
      run.eval("test_it" + std::to_string(it), evj);
      auto& r = run.row();
      r.last(rf::key::EVAL_N, s_n);
      r.last(rf::key::EVAL_SUCCESS, s_succ / s_n);
      r.last(rf::key::EVAL_COLLISION, s_coll / s_n);
      r.last(rf::key::EVAL_TRUNCATION, s_trunc / s_n);
      r.last(rf::key::EVAL_DECLARE_SHORT, s_short / s_n);
      r.last(rf::key::EVAL_COVERAGE, s_cov / s_n);
      r.last(rf::key::EVAL_T95_REACHED, s_t95n / s_n);
      r.last(rf::key::EVAL_LEN, s_len / s_n);
      r.last(rf::key::EVAL_REWARD, s_rew / s_n);
      if (s_t95n > 0) r.last(rf::key::EVAL_T95, s_t95 / s_t95n);
      if (s_decl > 0) r.last(rf::key::EVAL_COVERAGE_AT_DECLARE, s_cad / s_decl);
      run.event("eval at iteration " + std::to_string(it) + ": success " + std::to_string(s_succ / s_n));
      std::fprintf(stderr, "  평가 %.1f s\n", ms_since(t0) / 1000);
    };

    auto save = [&](bool numbered) {
      ppo.save(ckdir + "/model_latest.bin");
      if (numbered) ppo.save(ckdir + "/model_" + std::to_string(ppo.iteration()) + ".bin");
    };

    // ---- 학습
    const auto t_start = std::chrono::steady_clock::now();
    const int it0 = ppo.iteration();
    int it_done = 0;
    while (!g_stop) {
      const int it = ppo.iteration();
      if (a.iters > 0 && it - it0 >= a.iters) break;
      if (a.minutes > 0 && ms_since(t_start) / 60000.0 >= a.minutes) break;
      const auto t_it = std::chrono::steady_clock::now();
      if (a.replay_every > 0 && it % a.replay_every == 0) rec_want = std::min(kReplayEnvs, kReplayMax - (int)rec.size());
      const bool need_state = rec_want > 0 || !rec.empty();
      double t_act = 0, t_sim = 0, t_host = 0;
      // 반복 통계
      double n_ep = 0, n_succ = 0, n_coll = 0, n_pcoll = 0, n_trunc = 0, n_short = 0, n_idle = 0, cov = 0, cad = 0, n_decl = 0, t95 = 0, t95n = 0, stage = 0,
             len = 0, rew_ep = 0, step_rew = 0;
      std::vector<double> term_sum(mdp::rew::kTerms, 0.0);
      for (int t = 0; t < pc.steps_per_env; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        const sim::StepOut o = env.outputs();
        ppo.act(o.obs_img, o.obs_1d, o.priv, d_act);
        cudaDeviceSynchronize();
        t_act += ms_since(t0);
        if (need_state) {
          cudaMemcpy(h_actR.data(), d_act, sizeof(float) * h_actR.size(), cudaMemcpyDeviceToHost);
          ppo.last_act_info(kSampleEnvs, h_logit.data(), h_val.data());
        }
        t0 = std::chrono::steady_clock::now();
        env.step(d_act);
        env.sync();
        t_sim += ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        const sim::StepOut o2 = env.outputs();
        k_map_done<<<(N + 255) / 256, 256>>>(o2.done, N, d_done);
        ppo.store(o2.reward, d_done);
        cudaMemcpy(h_terms.data(), o2.reward_terms, sizeof(float) * h_terms.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rew.data(), o2.reward, sizeof(float) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_done.data(), o2.done, sizeof(int32_t) * N, cudaMemcpyDeviceToHost);
        bool any_done = false;
        for (int i = 0; i < N; ++i) {
          for (int k = 0; k < mdp::rew::kTerms; ++k) ep_terms[(size_t)i * mdp::rew::kTerms + k] += h_terms[(size_t)i * mdp::rew::kTerms + k];
          ep_rew[i] += h_rew[i];
          step_rew += h_rew[i];
          ep_steps[i] += 1;
          any_done |= h_done[i] != 0;
        }
        if (any_done) cudaMemcpy(h_ep.data(), o2.ep, sizeof(sim::EpInfo) * N, cudaMemcpyDeviceToHost);
        if (need_state) env.get_state(st);
        for (auto& R : rec) {
          uint32_t evb = 0;
          const int32_t d = h_done[R.env];
          if (d == mdp::cur::kCollision || d == mdp::cur::kCollisionPerson) evb |= rf::ev::COLLISION;
          if (d == mdp::cur::kDeclareOk || d == mdp::cur::kDeclareShort) evb |= rf::ev::DECLARE;
          if (d == mdp::cur::kTruncated) evb |= rf::ev::TRUNCATED;
          if (!d) rec_frame(R, evb);  // 끝난 걸음의 상태는 이미 리셋됨 — 끝 사건은 아래 판 줄에
        }
        for (int i = 0; i < N; ++i) {
          const int32_t d = h_done[i];
          if (!d) continue;
          const sim::EpInfo& f = h_ep[i];
          n_ep += 1;
          n_succ += d == mdp::cur::kDeclareOk;
          n_coll += d == mdp::cur::kCollision || d == mdp::cur::kCollisionPerson;
          n_pcoll += d == mdp::cur::kCollisionPerson;
          n_trunc += d == mdp::cur::kTruncated;
          n_short += d == mdp::cur::kDeclareShort;
          n_idle += d == mdp::cur::kNoProgress;
          cov += f.coverage;
          if (d == mdp::cur::kDeclareOk || d == mdp::cur::kDeclareShort) { cad += f.coverage; n_decl += 1; }
          if (f.t95 >= 0) { t95 += f.t95; t95n += 1; }
          stage += f.stage;
          len += ep_steps[i];
          rew_ep += ep_rew[i];
          for (int k = 0; k < mdp::rew::kTerms; ++k) term_sum[k] += ep_terms[(size_t)i * mdp::rew::kTerms + k];
          if (i < kSampleEnvs) {
            rf::Json line = rf::Json::object();
            const long long id = run.new_episode_id();
            const std::string oc = outcome_of(d);
            const int mi = (f.map >= 0 && f.map < (int)worlds.size()) ? f.map : 0;
            line.set("iter", it).set("env_steps", total_steps + (double)(t + 1) * N).set("env", i).set("ep", id).set("map", stem(worlds[mi].name))
                .set("stage", f.stage).set("area_m2", worlds[mi].n_floor * c.res * c.res).set("outcome", oc)
                .set("success", d == mdp::cur::kDeclareOk).set("declared", d == mdp::cur::kDeclareOk || d == mdp::cur::kDeclareShort)
                .set("collided", d == mdp::cur::kCollision || d == mdp::cur::kCollisionPerson).set("person_collision", d == mdp::cur::kCollisionPerson)
                .set("truncated", d == mdp::cur::kTruncated).set("done", d).set("t", f.t).set("steps", ep_steps[i])
                .set("coverage", f.coverage).set("reward", ep_rew[i]);
            if (f.t95 >= 0) line.set("t95", f.t95);
            if (d == mdp::cur::kDeclareOk || d == mdp::cur::kDeclareShort) line.set("coverage_at_declare", f.coverage);
            rf::Json rt = rf::Json::object();
            for (int k = 0; k < mdp::rew::kTerms; ++k) rt.set(mdp::rew::term_name(k), ep_terms[(size_t)i * mdp::rew::kTerms + k]);
            line.set("reward_terms", rt);
            for (size_t r = 0; r < rec.size(); ++r)
              if (rec[r].env == i) {
                const std::string path = run.replay_path(id, oc);
                line.set("replay", rf::basename_of(path));
                if (rec[r].w->n_frames() > 0) rec[r].w->finish(path, line);
                else line.erase("replay");
                rec.erase(rec.begin() + r);
                break;
              }
            run.episode(line);
          }
          for (int k = 0; k < mdp::rew::kTerms; ++k) ep_terms[(size_t)i * mdp::rew::kTerms + k] = 0;
          ep_rew[i] = 0;
          ep_steps[i] = 0;
          // 새 판 시작: 재생 원하면 이 판부터
          if (i < kSampleEnvs && need_state && rec_want > 0 && (int)rec.size() < kReplayMax) {
            bool busy = false;
            for (auto& rs : rec) busy |= rs.env == i;
            if (!busy) {
              start_rec(i, st[i].map);
              --rec_want;
            }
          }
        }
        t_host += ms_since(t0);
      }
      total_steps += (double)pc.steps_per_env * N;
      auto tu = std::chrono::steady_clock::now();
      const sim::StepOut o = env.outputs();
      const xppo::UpdateStats us = ppo.update(o.obs_img, o.obs_1d, o.priv);
      const double t_upd = ms_since(tu);
      const double t_iter = ms_since(t_it);
      if (!std::isfinite(us.value) || !std::isfinite(us.surrogate) || !std::isfinite(us.kl)) {
        run.event("NaN at iteration " + std::to_string(it) + " — stop");
        std::fprintf(stderr, "NaN: 반복 %d 에서 멈춤\n", it);
        run.finish();
        return 3;
      }
      float ls[3];
      cudaMemcpy(ls, ppo.ac().log_std(), sizeof(ls), cudaMemcpyDeviceToHost);
      size_t fr, tot;
      cudaMemGetInfo(&fr, &tot);
      auto& r = run.row();
      r.last(rf::key::ITER, it);
      r.last(rf::key::STEPS, total_steps);
      r.last(rf::key::FPS, pc.steps_per_env * N / (t_iter / 1000));
      r.last(rf::key::ITER_MS, t_iter);
      r.last(rf::key::ROLLOUT_MS, t_act + t_sim + t_host);
      r.last(rf::key::UPDATE_MS, t_upd);
      r.last("time/sim_ms", t_sim);
      r.last("time/act_ms", t_act);
      r.last("time/host_ms", t_host);
      r.sum(rf::key::N_EPISODES, n_ep);
      r.last(rf::key::STEP_REWARD_MEAN, step_rew / ((double)pc.steps_per_env * N));
      if (n_ep > 0) {
        r.mean_w(rf::key::EP_REW_MEAN, rew_ep / n_ep, n_ep);
        r.mean_w(rf::key::EP_LEN_MEAN, len / n_ep, n_ep);
        r.mean_w(rf::key::SUCCESS_RATE, n_succ / n_ep, n_ep);
        r.mean_w(rf::key::COLLISION_RATE, n_coll / n_ep, n_ep);
        r.mean_w(rf::key::PERSON_COLLISION_RATE, n_pcoll / n_ep, n_ep);
        r.mean_w(rf::key::TRUNCATION_RATE, n_trunc / n_ep, n_ep);
        r.mean_w(rf::key::DECLARE_RATE, n_decl / n_ep, n_ep);
        r.mean_w(rf::key::DECLARE_SHORT_RATE, n_short / n_ep, n_ep);
        r.mean_w(rf::key::NO_PROGRESS_RATE, n_idle / n_ep, n_ep);
        r.mean_w(rf::key::COVERAGE_MEAN, cov / n_ep, n_ep);
        r.mean_w(rf::key::T95_REACHED_RATE, t95n / n_ep, n_ep);
        r.mean_w(rf::key::CURRICULUM_STAGE, stage / n_ep, n_ep);
        for (int k = 0; k < mdp::rew::kTerms; ++k) r.mean_w(rf::reward_key(mdp::rew::term_name(k)), term_sum[k] / n_ep, n_ep);
        if (n_decl > 0) r.mean_w(rf::key::COVERAGE_AT_DECLARE, cad / n_decl, n_decl);
        if (t95n > 0) r.mean_w(rf::key::T95_MEAN, t95 / t95n, t95n);
      }
      r.last(rf::key::PG_LOSS, us.surrogate);
      r.last(rf::key::VALUE_LOSS, us.value);
      r.last(rf::key::ENTROPY_LOSS, -us.entropy);
      r.last(rf::key::APPROX_KL, us.kl);
      r.last(rf::key::GRAD_NORM, us.grad_norm);
      r.last(rf::key::LEARNING_RATE, us.lr);
      r.last(rf::key::VALUE_MEAN, us.mean_value);
      r.last(rf::key::STD, (std::exp(ls[0]) + std::exp(ls[1]) + std::exp(ls[2])) / 3.0);
      r.last(rf::key::LOG_STD, (ls[0] + ls[1] + ls[2]) / 3.0);
      r.last(rf::key::GPU_MEM_MB, (tot - fr) / 1048576.0);
      const int it_now = ppo.iteration();  // update 뒤 = it + 1
      std::fprintf(stderr,
                   "반복 %d: 판 %.0f 성공 %.3f 충돌 %.3f 끊김 %.3f 덮음 %.3f 단계 %.2f 걸음보상 %.4f | value %.4f kl %.4f lr %.2e σ %.3f | "
                   "act %.0f sim %.0f host %.0f update %.0f ms (%.0f 걸음/s)\n",
                   it, n_ep, n_ep ? n_succ / n_ep : 0, n_ep ? n_coll / n_ep : 0, n_ep ? n_trunc / n_ep : 0, n_ep ? cov / n_ep : 0,
                   n_ep ? stage / n_ep : 0, step_rew / ((double)pc.steps_per_env * N), us.value, us.kl, us.lr, std::exp(ls[0]), t_act,
                   t_sim, t_host, t_upd, pc.steps_per_env * N / (t_iter / 1000));
      ++it_done;
      if (a.eval_every > 0 && it_now % a.eval_every == 0) evaluate(it_now);
      run.end_update();
      if (a.ckpt_every > 0 && it_now % a.ckpt_every == 0) save(true);
      else if (it_now % 10 == 0) save(false);
    }
    save(true);
    run.event("stop at iteration " + std::to_string(ppo.iteration()) + (g_stop ? " (signal)" : ""));
    run.finish();
    std::fprintf(stderr, "끝: 반복 %d, 체크포인트 %s\n", ppo.iteration(), (ckdir + "/model_latest.bin").c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "train: %s\n", e.what());
    return 1;
  }
  return 0;
}
