// train — 탐사 RL 학습 러너 (S6). 시뮬(sim/include/env_api.h) × 신경망·PPO(network/·ppo/) → 학습 뷰어 실행 폴더(viewer/include/runfolder.h).
//
//   train [--envs 2048] [--iters 0] [--minutes 0] [--name explore] [--seed 1] [--lr-max 1e-3] [--std-max 1] [--mlp 2] [--chunk N] [--eval-every 50] [--ckpt-every 50]
//         [--replay-every 25] [--eval-episodes 32] [--resume] [--root train/explore/runs] [--list train/explore/scenes/train/list.tsv]
//         [--test a.scene,b.scene] [--env sim/config/env.yaml] [--model sim/config/sensor_model.yaml]
//         [--slot-cm-coef 1e-3]  계층형 슬롯 로짓 공통 성분 벌점 (0 = 끔, ppo.h PPOConfig::slot_cm_coef)
//         [--hier]  계층형: 1 s 마다 프론티어 후보 K 개 중 하나(가린 범주 분포) → 시뮬 경로 계획·추종 (mdp/README "계층형"). 반복 = 결정 24 개
//         [--graph] 그래프 모드 (HEADER 식, mdp/README "그래프 모드"): 결정마다 현재 노드의 이웃 노드 하나(가린 범주 분포, 최대 24) → 시뮬이 그 노드까지 주행. 결정 길이는 가변(약 2 s).
//                   신경망 = 어텐션 인코더 6 층 + 디코더 + 포인터 (network/README "그래프 어텐션 정책"). 환경 수·롤아웃 길이·에포크·γ 기본값은 ppo.h graph_ppo_config() 한곳.
//                   시뮬 쪽 덮어쓰기는 --graph-node·--graph-term·… (sim/include/env_cfg.h GraphArgs). --graph-share 1 = 가치 머리가 정책 몸통을 같이 씀(기본 따로)
//         [--steps T] [--epochs E] [--minibatches M] [--gamma g] [--lam l]  PPO 값 덮어쓰기 (안 주면 PPOConfig 기본, --graph 면 graph_ppo_config)
//         [--low]   아래층 RL: 판 = 아래층 단독 과제(목표 하나씩 → 규칙 경로 계획 → 통과), 정책이 걸음(0.1 s)마다 (v, ω). 신경망 = 가까운 지도·라이다·깊이 CNN + MLP
//                   (network/README "아래층"), 환경 기본 1024 · γ 기본 0.99 · [--people 2] 판마다 사람 0 ~ 이 수. 진행 로그 low/*·reward/*, 평가 = 시험 맵 60 s (mdp/README "아래층 RL")
//         [--low-explore] (--low 와) 탐사 판 학습: 판 = 탐사 판(위층 0 번 후보 규칙을 환경이 1 Hz 로 돌려 목표를 줌 — EnvCfg::low_auto 2 + low_explore, low_rl.h
//                   lowrl_goal_explore. 평가 도구 combine_run 과 같은 규칙). [--low-solo P] 환경의 P % 는 단독 과제 그대로 (섞어 학습 — 단독 과제 성능을 지킴).
//                   진행 로그 lowx/*, 평가에 탐사 판(사람 0 · 결정적 · 환경마다 첫 판 · 상한 --low-eval-steps 3000 걸음)이 더해짐 (mdp/README "아래층 RL — 탐사 판 학습")
//         [--critic-warmup N]  이번 실행의 처음 N 반복은 가치만 맞춘다 (PPO::set_critic_only — 정책·학습률 그대로). 가치 입력이 바뀐 체크포인트
//                   (ppo/tools/ckpt_drop_priv)에서 --resume 할 때 쓴다 (ppo/README "가치 참값 입력 뺌"). 로그 train/value0 = 걸음 전 가치 손실
//         [--people-ramp N]  (--low) 이번 실행 처음부터 N 반복에 걸쳐 판마다 사람 상한을 1 → --people 로 곧게 올린다 (사람 행동 비율은 SIM_PEOPLE_P_* — env_cfg.cpp)
//         [--map-prio]  (--low) 맵 뽑기를 최근 닿음 비율이 높은 맵 쪽으로 (확률 ∝ 0.5 + 닿음으로 끝난 판 비율, 지수 이동 합). 충돌 시간 벌점은 SIM_LR_TTC_K (0 = 끔)
//   --iters·--minutes 가 0 이면 그 한계 없음(둘 다 0 이면 Ctrl-C 까지가 아니라 --iters 1000 으로 둠).
//
// 반복 하나 = 24 걸음 롤아웃(act → env.step → store) → update. done 매핑: mdp::cur::terminal(1·2·4·5) → 1 종료, 3 → 3 끊김.
// 실행 폴더 <root>/<name>/: run.json(kind explore_ppo), progress.jsonl(반복마다), episodes.jsonl(표본 환경의 판), evals/test_it<반복>.json,
//   replays/*.xrp(환경 0·1 의 판을 --replay-every 반복마다 한 판씩), ckpt/model_<반복>.bin·model_latest.bin.
// 평가: --eval-every 반복마다 시험 맵마다 별도 Env(환경 --eval-episodes 개, 커리큘럼 끔)로 결정적 정책(act_inference), 환경마다 첫 판.
//   같은 판을 정책 분포 샘플로 한 번 더(evals/test_sample_it*.json, progress eval_sample/*) — 학습 성공과 평가 성공의 차이 보기.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
  int envs = 0, iters = 0, eval_every = 50, ckpt_every = 50, replay_every = 25, eval_eps = 32;
  double minutes = 0;
  // envs 0 = 기본 (2048, --graph 면 graph_ppo_config 의 num_envs)
  uint64_t seed = 1;
  float lr_max = -1;        // < 0 = 기본: 1e-3 (그래프 모드는 graph_ppo_config 값)
  float bound_coef = -1;  // < 0 = PPOConfig 기본. 평균 행동 경계 손실 계수 (0 = 끔, explore_v7 과 같음)
  float entropy = -1, desired_kl = -1, std_min = -1;   // < 0 = PPOConfig 기본 (엔트로피 계수 · KL 적응 목표 · σ 하한)
  float slot_cm_coef = -1;   // < 0 = PPOConfig 기본 (계층형 슬롯 로짓 공통 성분 벌점, ppo.h)
  float std_max = 0;  // 0 = PPOConfig 기본(1). 정책 σ 상한 — 크게(예 1e6) 주면 사실상 상한 없음, 하한 std_min 은 그대로
  int mlp = 2, chunk = 0;  // chunk 0 = PPOConfig 기본 (미니배치 하나)
  bool resume = false;
  bool hier = false;   // --hier: 계층형 행동 (1 s 마다 프론티어 후보 하나 — sim kActPath, PPO 가린 범주 분포)
  int critic_warmup = 0;   // --critic-warmup: 처음 이만큼 반복은 가치만 맞춤
  bool low = false;    // --low: 아래층 RL (단독 과제를 정책이 10 Hz 로 — run_low)
  int low_people = 2;  // --people: 아래층 RL 판마다 사람 0 ~ 이 수
  bool map_prio = false;  // --map-prio (--low): 맵 뽑기를 최근 닿음 비율이 높은 맵 쪽으로 — 확률 ∝ 0.5 + 맵의 닿음으로 끝난 판 비율 (판 수 지수 이동 합, 반감 ~14 반복)
  int people_ramp = 0;  // --people-ramp N: 이어 학습 처음부터 N 반복에 걸쳐 판마다 사람 상한을 1 → --people 로 곧게 올림 (0 = 처음부터 --people)
  bool low_explore = false;   // --low-explore: 탐사 판 학습 (위층 규칙을 환경이 돌림)
  int low_solo = 0;           // --low-solo: 그때 단독 과제로 도는 환경 몫 (%)
  int low_eval_steps = 3000;  // --low-eval-steps: 탐사 판 평가의 걸음 상한
  bool lean = true;           // --no-lean: 아래층 단독 과제에서도 프론티어 요약 커널을 돌림 (Env::set_lean_low — 결과는 같고 느림)
  bool step_sync = false;     // --step-sync: 아래층 롤아웃 걸음마다 호스트가 GPU 를 기다림 (옛 방식 — act·sim 시간을 호스트 시계로 가름). 끄면 CUDA 이벤트로 잼
  bool graph = false;  // --graph: 그래프 모드 (이웃 노드 하나 — sim kActPath + graph_on, PPO 그래프 어텐션 정책). 결정 하나 = env.step 한 번, 걸음 수는 가변
  int graph_share = -1;   // --graph-share 0|1 (음수 = ACSpec 기본: 가치 몸통 따로)
  sim::GraphArgs gargs;   // --graph-node 등 (sim/include/env_cfg.h)
  int steps = 0, epochs = 0, minibatches = 0;   // 0 = 기본 (PPOConfig, --graph 면 graph_ppo_config)
  float gamma = -1, lam = -1;                   // < 0 = 기본
  std::string eval_ckpt;
  std::string convert_in, convert_out;   // --convert-shared 입력 출력: 형식 2(공유 인코더) → 형식 3(인코더 나눔) 체크포인트로 바꿔 쓰고 끝  // 비어 있지 않으면 이 체크포인트를 불러 평가만 하고 끝 (학습 안 함)
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

// 그래프 모드 진단: 정책이 고른 이웃 번호 a[0] 이 가려진 칸(cand < 0)이면 cnt[0]++, 유효 이웃이 하나도 없는 환경이면 cnt[1]++ (그때는 PPO 가 0 번을 내고
// 정책 손실에서 뺀다 — 가린 선택으로 안 셈). cnt[0] 은 늘 0 이어야 한다
__global__ void k_graph_pick_check(const float* act, const int32_t* cand, int n, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int32_t* cd = cand + (size_t)i * mdp::graph::NB;
  bool any = false;
  for (int k = 0; k < mdp::graph::NB; ++k) any |= cd[k] >= 0;
  if (!any) { atomicAdd(cnt + 1, 1); return; }
  const float a = act[(size_t)i * mdp::act::DIM];
  const int k = (int)a;
  if (!(a >= 0.f) || k >= mdp::graph::NB || (float)k != a || cd[k] < 0) atomicAdd(cnt + 0, 1);
}

// 평가 장부 (장치): 환경마다 첫 판만 — 아직 안 끝났으면 보상을 더하고(double, 걸음 순서대로), 끝나면 판 정보를 적는다.
// 호스트가 걸음마다 동기화하던 것을 GPU 에서 해 걸음 사이 동기화를 없앤다 (결과는 같다).
// 결정적 평가 행동(= 정책 평균 μ, 자르기 전) 통계: 아직 첫 판 중인 환경만. s[0] 걸음 수, s[1..3] μ 합(v, vy, ω),
// s[4..6] |μ| > 1 걸음 수(잘리는 영역), s[7] μ_v > 1.5 걸음 수
__global__ void k_mu_stats(const float* act, const int* first, int n, double* s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || first[i]) return;
  const float* m = act + (size_t)i * mdp::act::DIM;
  atomicAdd(s + 0, 1.0);
  for (int j = 0; j < 3; ++j) {
    atomicAdd(s + 1 + j, (double)m[j]);
    if (fabsf(m[j]) > 1.f) atomicAdd(s + 4 + j, 1.0);
  }
  if (m[0] > 1.5f) atomicAdd(s + 7, 1.0);
}

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

// ================= 아래층 RL (--low) =================
// 판 = 아래층 단독 과제(sim/include/low_task.h: 목표 → 규칙 경로 계획 → 통과 반경 · 시간 제한 · 정체)를 정책이 10 Hz 로 몬다 (sim/include/low_rl.h).
// 목표는 시뮬이 뽑는다(low_auto). 걸음 하나 = 행동 하나 (v, ω) — 신경망 = explore_spec_low (가까운 지도·라이다 그림·깊이 CNN + MLP, 연속 가우시안).
// done 매핑: 충돌(벽·사람) → 1 종료, 그 밖의 끝(시간 제한·정체·목표 수 채움) → 3 끊김(가치 이어 붙임).
__global__ void k_map_done_low(const int32_t* d, int n, int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = d[i] == mdp::cur::kRunning ? 0 : ((d[i] == mdp::cur::kCollision || d[i] == mdp::cur::kCollisionPerson) ? 1 : 3);
}

const char* const kLowTermName[sim::kLrTerms] = {"progress", "time", "reach", "collision", "proximity", "smooth", "backward", "ttc"};

struct LowSum {   // LowRlStat 여러 환경 합
  double term[sim::kLrTerms] = {}, steps = 0, episodes = 0, goals = 0, reached = 0, hit_wall = 0, hit_person = 0, stall = 0, timeout = 0, giveup = 0,
         back = 0, reach_steps = 0, dist = 0, reach_len = 0, x_eps = 0, x_ok = 0, x_wall = 0, x_person = 0, x_idle = 0, x_ok_steps = 0, x_rule = 0,
         gt_wall = 0, gt_person = 0;   // 판 동안 시뮬 정답 충돌이 있었던 판 (기록 전용 — hit_* 는 센서 닿음으로 끝난 판)
  void add(const sim::LowRlStat& s) {
    for (int k = 0; k < sim::kLrTerms; ++k) term[k] += s.term[k];
    steps += s.steps; episodes += s.episodes; goals += s.goals; reached += s.reached; hit_wall += s.hit_wall; hit_person += s.hit_person;
    stall += s.stall; timeout += s.timeout; giveup += s.giveup; back += s.back_steps; reach_steps += s.reach_steps;
    dist += s.dist; reach_len += s.reach_len;
    x_eps += s.x_eps; x_ok += s.x_ok; x_wall += s.x_wall; x_person += s.x_person; x_idle += s.x_idle; x_ok_steps += s.x_ok_steps; x_rule += s.x_rule_steps;
    gt_wall += s.gt_wall; gt_person += s.gt_person;
  }
  // 도달률 = 통과 ÷ 경로가 짜인 목표 (아직 가는 중인 목표도 분모 — 긴 구간에서는 시작·끝 수가 같아져 치우침이 사라진다)
  double reach_rate() const { return goals > 0 ? reached / goals : 0; }
  // 과제 구간 평균 속도 = 통과한 목표들의 처음 계획 길이 합 ÷ 걸린 시간 합 (기준선 low_run 의 "속도" 와 같은 정의)
  double speed(double dt) const { return reach_steps > 0 ? reach_len / (reach_steps * dt) : 0; }
  double wall100() const { return dist > 0 ? hit_wall / dist * 100 : 0; }
  double person100() const { return dist > 0 ? hit_person / dist * 100 : 0; }
  double gt_wall100() const { return dist > 0 ? gt_wall / dist * 100 : 0; }
  double gt_person100() const { return dist > 0 ? gt_person / dist * 100 : 0; }
};

int run_low(Args a) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  hc.cfg.action_mode = sim::kActPath;
  hc.cfg.curriculum = 0;   // 모든 맵 고르게 (단계 커리큘럼은 탐사 판의 것 — 아래층은 목표 거리 등급이 그 자리, low_rl.h lowrl_goal_probe)
  hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = hc.cfg.low_auto = 1;
  hc.cfg.low_act_w = mdp::low::ACT;
  const sim::EnvHostCfg hc_solo = hc;   // 단독 과제 (평가 — 탐사 판 학습에서도 같은 곡선을 봄)
  if (a.low_explore) {   // 탐사 판 학습 (low_rl.h lowrl_goal_explore)
    hc.cfg.low_auto = 2;
    hc.cfg.low_explore = 1;
    hc.cfg.low_solo = a.low_solo;
  }
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const double dt_act = c.substeps * c.dt;
  std::vector<std::string> skipped;
  std::vector<sim::World> worlds = sim::load_scene_list(a.list, hc, &skipped);
  if (worlds.empty()) throw std::runtime_error("학습 맵이 없음: " + a.list);
  std::fprintf(stderr, "학습 맵 %zu 개 (안 닫혀 뺀 것 %zu)\n", worlds.size(), skipped.size());
  xppo::PPOConfig pc;
  pc.num_envs = a.envs > 0 ? a.envs : 1024;   // 잠정: RTX 3060 에서 위층 학습(약 5 GB)과 같이 돌 크기 (scripts/README "아래층 RL")
  pc.gamma = 0.99f;                            // 잠정: 걸음 0.1 s × 100 = 10 s 지평 (과제 하나 3 ~ 15 m ≈ 4 ~ 20 s). rsl_rl PPO 기본값과 같음
  pc.lr_max = a.lr_max > 0 ? a.lr_max : 1e-3f;
  pc.desired_kl = 0.01f;                      // 잠정: rsl_rl 기본값. PPOConfig 기본 0.003 은 30 반복 시운전에서 학습률이 하한(1e-5)과 4e-4 사이를 오갔고 0.01 은 1e-3 에 머묾 (곡선은 비슷)
  pc.mlp = a.mlp;
  if (a.std_max > 0) pc.std_max = a.std_max;
  if (a.entropy >= 0) pc.entropy_coef = a.entropy;
  if (a.desired_kl >= 0) pc.desired_kl = a.desired_kl;
  if (a.std_min > 0) pc.std_min = a.std_min;
  if (a.bound_coef >= 0) pc.bound_coef = a.bound_coef;
  if (a.chunk > 0) pc.chunk = a.chunk;
  if (a.steps > 0) pc.steps_per_env = a.steps;
  if (a.epochs > 0) pc.epochs = a.epochs;
  if (a.minibatches > 0) pc.minibatches = a.minibatches;
  if (a.gamma >= 0) pc.gamma = a.gamma;
  if (a.lam >= 0) pc.lam = a.lam;
  const int N = pc.num_envs, AW = mdp::low::ACT;
  sim::Env env(hc, worlds, lidar, N, a.seed, true);
  env.set_people_max(std::vector<int>(N, a.low_people));   // 판마다 사람 0 ~ 이 수 (people.h)
  env.set_lean_low(a.lean && !a.low_explore);   // 단독 과제: 안 읽히는 프론티어 요약을 건너뜀 (탐사 판 학습은 후보·닫힘을 쓰므로 그대로)
  env.reset();
  env.sync();
  std::vector<sim::World> tests;
  for (auto& t : a.test) tests.push_back(sim::build_world(stem(t), sim::load_scene(t), hc));
  const xnet::ACSpec spec = xnet::explore_spec_low();
  xppo::PPO ppo(pc, spec, a.seed);
  const std::string dir = a.root + "/" + a.name, ckdir = dir + "/ckpt";
  if (a.resume) {
    const std::string p = ckdir + "/model_latest.bin";
    if (!file_exists(p)) throw std::runtime_error("--resume: 체크포인트 없음 " + p);
    ppo.load(p);
    std::fprintf(stderr, "이어 학습: %s 반복 %d, 학습률 %.2e\n", p.c_str(), ppo.iteration(), ppo.learning_rate());
  }
  rf::Json terms = rf::Json::array();
  for (int k = 0; k < sim::kLrTerms; ++k) terms.push(kLowTermName[k]);
  rf::Json tests_meta = rf::Json::array();
  for (auto& w : tests) tests_meta.push(w.name);
  rf::Json limits = rf::Json::object();
  limits.set("v", c.act_v_fwd).set("v_back", c.act_v_back).set("w", c.act_w_max);
  rf::Json meta = rf::Json::object();
  meta.set("kind", "explore_ppo").set("group", a.name).set("seed", (unsigned long long)a.seed).set("n_envs", N).set("ctrl_hz", 1.0 / dt_act).set("low", true)
      .set("action_limits", limits).set("reward_terms", terms).set("n_train_maps", (unsigned long long)worlds.size()).set("test_maps", tests_meta)
      .set("scene_list", a.list).set("steps_per_env", pc.steps_per_env).set("people_max", a.low_people).set("n_params", (long long)ppo.ac().n_params)
      .set("ppo", rf::Json::object().set("epochs", pc.epochs).set("minibatches", pc.minibatches).set("clip", pc.clip).set("gamma", pc.gamma).set("lam", pc.lam)
                      .set("entropy_coef", pc.entropy_coef).set("lr_max", pc.lr_max).set("desired_kl", pc.desired_kl).set("std_min", pc.std_min)
                      .set("std_max", pc.std_max).set("bound_coef", pc.bound_coef));
  rf::RunFolder::Options opt;
  opt.every_s = 0;
  if (a.resume) { opt.resume = true; opt.resume_iter = ppo.iteration(); }
  rf::RunFolder run(dir, meta, opt);
  rf::mkdirs(ckdir);
  std::fprintf(stderr, "아래층 RL: 실행 폴더 %s, 환경 %d, 변수 %ld, 롤아웃 %d 걸음, γ %.4f, 사람 0 ~ %d\n", dir.c_str(), N, ppo.ac().n_params, pc.steps_per_env,
               pc.gamma, a.low_people);

  float* d_act;
  int* d_done;
  cudaMalloc(&d_act, sizeof(float) * AW * N);
  cudaMalloc(&d_done, sizeof(int) * N);
  std::vector<sim::LowRlStat> h_stat;
  double total_steps = (double)ppo.iteration() * pc.steps_per_env * N;
  std::vector<cudaEvent_t> ev_roll(2 * pc.steps_per_env + 1);   // 롤아웃 걸음 경계 (--step-sync 가 아닐 때)
  for (auto& e : ev_roll) cudaEventCreate(&e);

  // 주기 평가: 시험 맵마다 환경 E 개(앞 절반 사람 0 · 뒤 절반 사람 0 ~ 2), 자동 목표로 kEvalSteps 걸음. sample = 정책 분포 샘플(학습 롤아웃과 같은 행동 방식).
  // 기준선과 같은 조건의 측정(경로 길이 구간별 목표)은 따로 low_policy_run — 이 평가는 학습 곡선용
  constexpr int kEvalSteps = 600;   // 60 s
  auto put_low = [&](rf::Json& j, const LowSum& s) {
    j.set("goals", s.goals).set("reach_rate", s.reach_rate()).set("speed", s.speed(dt_act)).set("wall_per_100m", s.wall100()).set("person_per_100m", s.person100()).set("gt_wall_per_100m", s.gt_wall100()).set("gt_person_per_100m", s.gt_person100())
        .set("stall", s.stall).set("timeout", s.timeout).set("giveup", s.giveup)
        .set("back_frac", s.steps > 0 ? s.back / s.steps : 0.0).set("dist", s.dist);
  };
  auto evaluate = [&](int it, bool sample) {
    const auto t0 = std::chrono::steady_clock::now();
    const int E = std::max(2, std::min(a.eval_eps, N));
    float* d_eact;
    cudaMalloc(&d_eact, sizeof(float) * AW * E);
    rf::Json rows = rf::Json::array();
    LowSum all;
    for (size_t ti = 0; ti < tests.size(); ++ti) {
      std::vector<sim::World> one{tests[ti]};
      sim::Env ev(hc_solo, one, lidar, E, a.seed * 7919ull + 17 + ti, true);
      std::vector<int> pm(E);
      for (int i = 0; i < E; ++i) pm[i] = i < E / 2 ? 0 : 2;
      ev.set_people_max(pm);
      ev.set_lean_low(a.lean);   // 단독 과제 평가 (hc_solo)
      ev.reset();
      ev.sync();
      for (int step = 0; step < kEvalSteps; ++step) {
        const sim::StepOut o = ev.outputs();
        ppo.act_inference(o.low_img, o.low_1d, d_eact, E, sample ? (int64_t)step * (int64_t)tests.size() + (int64_t)ti : -1);
        cudaDeviceSynchronize();
        ev.step(d_eact);
        ev.sync();
      }
      ev.take_low_stat(h_stat);
      LowSum s[2];
      for (int i = 0; i < E; ++i) s[i < E / 2 ? 0 : 1].add(h_stat[i]);
      for (int p = 0; p < 2; ++p) {
        rf::Json row = rf::Json::object();
        row.set("eval", tests[ti].name).set("split", sample ? "test_sample" : "test").set("people", p ? 2 : 0).set("n", (double)(p ? E - E / 2 : E / 2));
        put_low(row, s[p]);
        rows.push(row);
        std::fprintf(stderr, "  평가%s %s 사람 %d: 목표 %.0f 도달률 %.3f 속도 %.3f m/s 벽 %.3f·사람 %.3f /100 m 정체 %.0f 시간끝 %.0f\n", sample ? "(샘플)" : "",
                     tests[ti].name.c_str(), p ? 2 : 0, s[p].goals, s[p].reach_rate(), s[p].speed(dt_act), s[p].wall100(), s[p].person100(), s[p].stall,
                     s[p].timeout);
      }
      for (int i = 0; i < E; ++i) all.add(h_stat[i]);
    }
    cudaFree(d_eact);
    rf::Json evj = rf::Json::object();
    evj.set("iter", it).set("ckpt", "model_" + std::to_string(it)).set("rows", rows);
    run.eval((sample ? "test_sample_it" : "test_it") + std::to_string(it), evj);
    auto& r = run.row();
    const std::string pre = sample ? "eval_sample/" : "eval/";
    r.last(pre + "reach_rate", all.reach_rate());
    r.last(pre + "speed", all.speed(dt_act));
    r.last(pre + "wall_per_100m", all.wall100());
    r.last(pre + "person_per_100m", all.person100());
    r.last(pre + "gt_wall_per_100m", all.gt_wall100());   // 시뮬 정답 충돌 (기록 전용)
    r.last(pre + "gt_person_per_100m", all.gt_person100());
    r.last(pre + "goals", all.goals);
    std::fprintf(stderr, "  평가%s 반복 %d: 도달률 %.3f 속도 %.3f m/s (%.1f s)\n", sample ? "(샘플)" : "", it, all.reach_rate(), all.speed(dt_act), ms_since(t0) / 1000);
  };
  // 탐사 판 평가 (--low-explore): 시험 맵마다 환경 E 개 · 사람 0 · 결정적 · 환경마다 첫 판 (combine_run --policy 와 같은 규칙. 광선 길은 학습과 같은 빠른 길이고
  // 걸음 상한이 있어 값이 똑같지는 않음 — 학습 곡선용, 최종 값은 combine_run)
  auto evaluate_x = [&](int it) {
    const auto t0 = std::chrono::steady_clock::now();
    const int E = std::max(2, std::min(a.eval_eps, N));
    sim::EnvHostCfg hx = hc;
    hx.cfg.low_solo = 0;
    float* d_eact;
    cudaMalloc(&d_eact, sizeof(float) * AW * E);
    rf::Json rows = rf::Json::array();
    double a_ok = 0, a_wall = 0, a_n = 0, a_t = 0;
    for (size_t ti = 0; ti < tests.size(); ++ti) {
      std::vector<sim::World> one{tests[ti]};
      sim::Env ev(hx, one, lidar, E, a.seed * 7919ull + 17 + ti, true);
      ev.reset();
      ev.sync();
      std::vector<int> fin(E, 0), fd(E, 0), fs(E, 0);
      std::vector<float> rw;
      std::vector<int32_t> dn;
      int left = E;
      for (int step = 0; step < a.low_eval_steps && left > 0; ++step) {
        const sim::StepOut o = ev.outputs();
        ppo.act_inference(o.low_img, o.low_1d, d_eact, E, -1);
        cudaDeviceSynchronize();
        ev.step(d_eact);
        ev.sync();
        ev.get_reward(rw, dn);
        for (int i = 0; i < E; ++i)
          if (!fin[i] && dn[i]) { fin[i] = 1; fd[i] = dn[i]; fs[i] = step + 1; --left; }
      }
      double ok = 0, wall = 0, person = 0, idle = 0, open = 0, ts = 0;
      for (int i = 0; i < E; ++i) {
        ok += fd[i] == mdp::cur::kDeclareOk;
        wall += fd[i] == mdp::cur::kCollision;
        person += fd[i] == mdp::cur::kCollisionPerson;
        idle += fd[i] == mdp::cur::kNoProgress;
        open += !fin[i];
        if (fd[i] == mdp::cur::kDeclareOk) ts += fs[i] * dt_act;
      }
      rf::Json row = rf::Json::object();
      row.set("eval", tests[ti].name).set("split", "explore").set("people", 0).set("n", (double)E).set("success", ok / E).set("wall", wall / E).set("person", person / E)
          .set("idle", idle / E).set("open", open / E).set("t_close", ok > 0 ? ts / ok : 0.0);
      rows.push(row);
      std::fprintf(stderr, "  탐사 판 평가 %s: 판 %d 성공 %.3f 벽 충돌 %.3f 진행 없음 %.3f 안 끝남 %.3f 닫힘까지 %.1f s\n", tests[ti].name.c_str(), E, ok / E, wall / E, idle / E,
                   open / E, ok > 0 ? ts / ok : 0.0);
      auto& r = run.row();
      r.last("evalx/" + tests[ti].name + "/success", ok / E);
      r.last("evalx/" + tests[ti].name + "/wall", wall / E);
      r.last("evalx/" + tests[ti].name + "/t_close", ok > 0 ? ts / ok : 0.0);
      a_ok += ok; a_wall += wall; a_n += E; a_t += ts;
    }
    cudaFree(d_eact);
    rf::Json evj = rf::Json::object();
    evj.set("iter", it).set("ckpt", "model_" + std::to_string(it)).set("rows", rows);
    run.eval("explore_it" + std::to_string(it), evj);
    auto& r = run.row();
    r.last("evalx/success", a_n > 0 ? a_ok / a_n : 0.0);
    r.last("evalx/wall", a_n > 0 ? a_wall / a_n : 0.0);
    r.last("evalx/t_close", a_ok > 0 ? a_t / a_ok : 0.0);
    std::fprintf(stderr, "  탐사 판 평가 반복 %d: 성공 %.3f 벽 충돌 %.3f 닫힘까지 %.1f s (%.1f s)\n", it, a_n > 0 ? a_ok / a_n : 0.0, a_n > 0 ? a_wall / a_n : 0.0,
                 a_ok > 0 ? a_t / a_ok : 0.0, ms_since(t0) / 1000);
  };
  auto save = [&](bool numbered) {
    ppo.save(ckdir + "/model_latest.bin");
    if (numbered) ppo.save(ckdir + "/model_" + std::to_string(ppo.iteration()) + ".bin");
  };
  if (!a.eval_ckpt.empty()) {
    ppo.load(a.eval_ckpt);
    evaluate(ppo.iteration(), false);
    evaluate(ppo.iteration(), true);
    if (a.low_explore) evaluate_x(ppo.iteration());
    run.end_update();
    run.finish();
    return 0;
  }

  const auto t_start = std::chrono::steady_clock::now();
  const int it0 = ppo.iteration();
  int k_people = a.low_people;
  std::vector<double> mp_eps(worlds.size(), 0.0), mp_hit(worlds.size(), 0.0);   // --map-prio: 맵마다 판 수·닿음 판 수 (지수 이동 합)
  double mp_wmax = 1.0;
  while (!g_stop) {
    const int it = ppo.iteration();
    if (a.map_prio) {   // 맵 뽑기 우선순위 (우선 재생 꼴 — Jiang et al. "Prioritized Level Replay", ICML 2021 의 점수를 닿음 비율로)
      std::vector<int32_t> ne, nh;
      env.take_map_stats(ne, nh);
      std::vector<float> cdf(worlds.size());
      double acc = 0.0, wmax = 0.0;
      for (size_t m = 0; m < worlds.size(); ++m) {
        mp_eps[m] = 0.95 * mp_eps[m] + ne[m];
        mp_hit[m] = 0.95 * mp_hit[m] + nh[m];
        const double w = 0.5 + (mp_eps[m] > 0.5 ? mp_hit[m] / mp_eps[m] : 0.0);
        wmax = std::max(wmax, w);
        acc += w;
        cdf[m] = (float)acc;
      }
      mp_wmax = wmax;
      env.set_map_cdf(cdf);
    }
    if (a.people_ramp > 0) {   // 사람 상한 커리큘럼: 1 → --people (다음 판부터 들음 — Env::set_people_max)
      const int k = std::min(a.low_people, 1 + (int)((double)(a.low_people - 1) * (it - it0) / a.people_ramp + 0.5));
      if (k != k_people || it == it0) {
        k_people = k;
        env.set_people_max(std::vector<int>(N, k));
        run.event("사람 상한 " + std::to_string(k) + " (반복 " + std::to_string(it) + ")");
      }
    }
    if (a.iters > 0 && it - it0 >= a.iters) break;
    if (a.minutes > 0 && ms_since(t_start) / 60000.0 >= a.minutes) break;
    const auto t_it = std::chrono::steady_clock::now();
    if (a.eval_every > 0 && (it == 0 || (a.low_explore && it == it0))) { evaluate(it, false); if (a.low_explore) evaluate_x(it); else evaluate(it, true); }   // 배우기 전 정책 (비교 기준 — 탐사 판 학습은 이어 학습의 시작점도)
    double t_act = 0, t_sim = 0, step_rew = 0;
    if (a.step_sync) {   // 옛 방식: 걸음마다 세 번 기다림 (호스트 시계로 act·sim 을 가름)
      for (int t = 0; t < pc.steps_per_env; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        const sim::StepOut o = env.outputs();
        ppo.act(o.low_img, o.low_1d, d_act);
        cudaDeviceSynchronize();
        t_act += ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        env.step(d_act);
        env.sync();
        const sim::StepOut o2 = env.outputs();
        k_map_done_low<<<(N + 255) / 256, 256>>>(o2.done, N, d_done);
        ppo.store(o2.reward, d_done);
        cudaDeviceSynchronize();
        t_sim += ms_since(t0);
      }
    } else {
      // 롤아웃 동안 호스트는 안 기다린다: 정책(ppo 스트림 — 막는 스트림)과 시뮬(기본 스트림 0)은 기본 스트림의 암묵 동기로 차례가 지켜진다.
      // act·sim 시간은 기본 스트림 이벤트 사이 GPU 시간 (걸음마다 둘씩, 반복 끝에 한 번 기다려 읽음)
      for (int t = 0; t < pc.steps_per_env; ++t) {
        cudaEventRecord(ev_roll[2 * t], 0);
        const sim::StepOut o = env.outputs();
        ppo.act(o.low_img, o.low_1d, d_act);
        cudaEventRecord(ev_roll[2 * t + 1], 0);
        env.step(d_act);
        const sim::StepOut o2 = env.outputs();
        k_map_done_low<<<(N + 255) / 256, 256>>>(o2.done, N, d_done);
        ppo.store(o2.reward, d_done);
      }
      cudaEventRecord(ev_roll[2 * pc.steps_per_env], 0);
      cudaEventSynchronize(ev_roll[2 * pc.steps_per_env]);
      for (int t = 0; t < pc.steps_per_env; ++t) {
        float ma = 0, ms = 0;
        cudaEventElapsedTime(&ma, ev_roll[2 * t], ev_roll[2 * t + 1]);
        cudaEventElapsedTime(&ms, ev_roll[2 * t + 1], ev_roll[2 * t + 2]);
        t_act += ma;
        t_sim += ms;
      }
    }
    const double iter_steps = (double)pc.steps_per_env * N;
    total_steps += iter_steps;
    const auto tu = std::chrono::steady_clock::now();
    const sim::StepOut o = env.outputs();
    ppo.set_critic_only(it - it0 < a.critic_warmup);
    const xppo::UpdateStats us = ppo.update(o.low_img, o.low_1d);
    const double t_upd = ms_since(tu), t_iter = ms_since(t_it);
    if (!std::isfinite(us.value) || !std::isfinite(us.surrogate) || !std::isfinite(us.kl)) {
      run.event("NaN at iteration " + std::to_string(it) + " — stop");
      std::fprintf(stderr, "NaN: 반복 %d 에서 멈춤\n", it);
      run.finish();
      return 3;
    }
    env.take_low_stat(h_stat);
    LowSum s;
    for (int i = 0; i < N; ++i) s.add(h_stat[i]);
    for (int k = 0; k < sim::kLrTerms; ++k) step_rew += s.term[k];
    float ls[2] = {0.f, 0.f};
    cudaMemcpy(ls, ppo.ac().log_std(), sizeof(ls), cudaMemcpyDeviceToHost);
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    const double ns = s.steps > 0 ? s.steps : 1;
    auto& r = run.row();
    r.last(rf::key::ITER, it);
    r.last(rf::key::STEPS, total_steps);
    r.last(rf::key::FPS, iter_steps / (t_iter / 1000));
    r.last(rf::key::ITER_MS, t_iter);
    r.last(rf::key::ROLLOUT_MS, t_act + t_sim);
    r.last(rf::key::UPDATE_MS, t_upd);
    r.last("time/act_ms", t_act);
    r.last("time/sim_ms", t_sim);
    r.sum(rf::key::N_EPISODES, s.episodes);
    r.last(rf::key::STEP_REWARD_MEAN, step_rew / ns);
    if (s.episodes > 0) {
      r.mean_w(rf::key::COLLISION_RATE, (s.hit_wall + s.hit_person) / s.episodes, s.episodes);
      r.mean_w(rf::key::PERSON_COLLISION_RATE, s.hit_person / s.episodes, s.episodes);
    }
    for (int k = 0; k < sim::kLrTerms; ++k) r.last(rf::reward_key(kLowTermName[k]), s.term[k] / ns);   // 걸음당 평균
    r.last("low/goals", s.goals);
    r.last("low/reach_rate", s.reach_rate());
    r.last("low/speed", s.speed(dt_act));
    r.last("low/reach_len_mean", s.reached > 0 ? s.reach_len / s.reached : 0.0);
    r.last("low/wall_per_100m", s.wall100());
    r.last("low/people_max", k_people);
    if (a.map_prio) r.last("low/map_prio_wmax", mp_wmax);   // 판마다 사람 0 ~ 이 수 (--people-ramp 면 올라감)
    r.last("low/person_per_100m", s.person100());
    r.last("low/gt_wall_per_100m", s.gt_wall100());   // 시뮬 정답 충돌 (기록 전용)
    r.last("low/gt_person_per_100m", s.gt_person100());
    r.last("low/stall_per_goal", s.goals > 0 ? s.stall / s.goals : 0.0);
    r.last("low/timeout_per_goal", s.goals > 0 ? s.timeout / s.goals : 0.0);
    r.last("low/giveup_per_goal", s.goals > 0 ? s.giveup / s.goals : 0.0);
    r.last("low/back_frac", s.back / ns);
    r.last("low/policy_frac", s.steps / iter_steps);
    r.last("low/mean_speed_all", s.dist / (ns * dt_act));
    if (a.low_explore) {   // 탐사 판 환경의 끝난 판 (학습 롤아웃 — 정책 분포 샘플 · 사람 0 ~ --people)
      r.last("lowx/episodes", s.x_eps);
      r.last("lowx/success", s.x_eps > 0 ? s.x_ok / s.x_eps : 0.0);
      r.last("lowx/wall", s.x_eps > 0 ? s.x_wall / s.x_eps : 0.0);
      r.last("lowx/person", s.x_eps > 0 ? s.x_person / s.x_eps : 0.0);
      r.last("lowx/idle", s.x_eps > 0 ? s.x_idle / s.x_eps : 0.0);
      r.last("lowx/t_close", s.x_ok > 0 ? s.x_ok_steps / s.x_ok * dt_act : 0.0);
      r.last("lowx/rule_frac", s.x_rule / iter_steps);
      std::fprintf(stderr, "  탐사 판(학습): 끝난 판 %.0f 성공 %.0f 벽 %.0f 사람 %.0f 진행 없음 %.0f 닫힘까지 %.1f s 규칙 걸음 %.3f\n", s.x_eps, s.x_ok, s.x_wall, s.x_person, s.x_idle,
                   s.x_ok > 0 ? s.x_ok_steps / s.x_ok * dt_act : 0.0, s.x_rule / iter_steps);
    }
    r.last(rf::key::PG_LOSS, us.surrogate);
    r.last(rf::key::VALUE_LOSS, us.value);
    r.last(rf::key::ENTROPY_LOSS, -us.entropy);
    r.last(rf::key::APPROX_KL, us.kl);
    r.last("train/kl0", us.kl0);
    r.last("train/value0", us.value0);
    r.last(rf::key::GRAD_NORM, us.grad_norm);
    r.last(rf::key::LEARNING_RATE, us.lr);
    r.last(rf::key::VALUE_MEAN, us.mean_value);
    r.last(rf::key::STD, (std::exp(ls[0]) + std::exp(ls[1])) / 2.0);
    r.last(rf::key::LOG_STD, (ls[0] + ls[1]) / 2.0);
    r.last("train/std_v", std::exp(ls[0]));
    r.last("train/std_w", std::exp(ls[1]));
    r.last(rf::key::GPU_MEM_MB, (tot - fr) / 1048576.0);
    const int it_now = ppo.iteration();
    std::fprintf(stderr,
                 "반복 %d: 목표 %.0f 도달률 %.3f 속도 %.3f m/s 벽 %.3f·사람 %.3f /100 m 정체 %.0f 시간끝 %.0f 후진 %.3f 걸음보상 %.4f | value %.4f kl %.4f lr %.2e "
                 "σ %.3f %.3f | act %.0f sim %.0f update %.0f ms (%.0f 걸음/s)\n",
                 it, s.goals, s.reach_rate(), s.speed(dt_act), s.wall100(), s.person100(), s.stall, s.timeout, s.back / ns, step_rew / ns, us.value,
                 us.kl, us.lr, std::exp(ls[0]), std::exp(ls[1]), t_act, t_sim, t_upd, iter_steps / (t_iter / 1000));
    if (a.eval_every > 0 && it_now % a.eval_every == 0) { evaluate(it_now, false); if (a.low_explore) evaluate_x(it_now); else evaluate(it_now, true); }
    run.end_update();
    if (a.ckpt_every > 0 && it_now % a.ckpt_every == 0) save(true);
    else if (it_now % 10 == 0) save(false);
  }
  save(a.ckpt_every > 0);
  run.event("stop at iteration " + std::to_string(ppo.iteration()) + (g_stop ? " (signal)" : ""));
  run.finish();
  std::fprintf(stderr, "끝: 반복 %d, 체크포인트 %s\n", ppo.iteration(), (ckdir + "/model_latest.bin").c_str());
  cudaFree(d_act);
  cudaFree(d_done);
  return 0;
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
      else if (k == "--std-max") a.std_max = std::stof(nx());
      else if (k == "--entropy") a.entropy = std::stof(nx());
      else if (k == "--desired-kl") a.desired_kl = std::stof(nx());
      else if (k == "--slot-cm-coef") a.slot_cm_coef = std::stof(nx());
      else if (k == "--std-min") a.std_min = std::stof(nx());
      else if (k == "--bound-coef") a.bound_coef = std::stof(nx());
      else if (k == "--mlp") a.mlp = std::stoi(nx());
      else if (k == "--chunk") a.chunk = std::stoi(nx());
      else if (k == "--eval-every") a.eval_every = std::stoi(nx());
      else if (k == "--ckpt-every") a.ckpt_every = std::stoi(nx());
      else if (k == "--replay-every") a.replay_every = std::stoi(nx());
      else if (k == "--eval-episodes") a.eval_eps = std::stoi(nx());
      else if (k == "--resume") a.resume = true;
      else if (k == "--critic-warmup") a.critic_warmup = std::stoi(nx());
      else if (k == "--hier") a.hier = true;
      else if (k == "--graph") a.graph = true;
      else if (k == "--low") a.low = true;
      else if (k == "--people") a.low_people = std::stoi(nx());
      else if (k == "--people-ramp") a.people_ramp = std::stoi(nx());
      else if (k == "--map-prio") a.map_prio = true;
      else if (k == "--low-explore") a.low_explore = true;
      else if (k == "--low-solo") a.low_solo = std::stoi(nx());
      else if (k == "--low-eval-steps") a.low_eval_steps = std::stoi(nx());
      else if (k == "--step-sync") a.step_sync = true;
      else if (k == "--no-lean") a.lean = false;
      else if (k == "--graph-share") a.graph_share = std::stoi(nx());
      else if (k.rfind("--graph-", 0) == 0) { const std::string v = nx(); if (!sim::graph_arg(a.gargs, k, v.c_str())) throw std::runtime_error("모르는 인자 " + k); }
      else if (k == "--steps") a.steps = std::stoi(nx());
      else if (k == "--epochs") a.epochs = std::stoi(nx());
      else if (k == "--minibatches") a.minibatches = std::stoi(nx());
      else if (k == "--gamma") a.gamma = std::stof(nx());
      else if (k == "--lam") a.lam = std::stof(nx());
      else if (k == "--eval-ckpt") a.eval_ckpt = nx();
      else if (k == "--convert-shared") { a.convert_in = nx(); a.convert_out = nx(); }
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
    if (a.low) {
      if (a.hier || a.graph) throw std::runtime_error("--low 는 --hier·--graph 와 같이 못 씀");
      return run_low(a);
    }
    // ---- 시뮬
    sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
    if (a.hier && a.graph) throw std::runtime_error("--hier 와 --graph 는 같이 못 씀");
    const bool cat = a.hier || a.graph;   // 범주 행동 (σ 없음, 결정 하나 = env.step 한 번)
    if (cat) {   // 계층형: 결정 하나 = 속도 행동 decide_steps 걸음 (env.step 한 번). 재생 녹화는 아직 속도 행동 열만 있어 끔
      hc.cfg.action_mode = sim::kActPath;
      if (a.graph) {   // 그래프 모드: 계층형 경로 추종 위에 얹음 — 결정 하나 = 이웃 노드 하나 (걸음 수 가변, env.get_gdec)
        hc.cfg.graph_on = 1;
        sim::graph_apply(hc.cfg, a.gargs);
      }
      if (a.replay_every > 0) std::fprintf(stderr, "%s: 재생 녹화는 아직 없음 — --replay-every 무시\n", a.graph ? "--graph" : "--hier");
      a.replay_every = 0;
    }
    const sim::LidarModel lidar = sim::load_lidar_model(a.model);
    const sim::EnvCfg& c = hc.cfg;
    if ((c.action_mode != sim::kActVelocity && c.action_mode != sim::kActPath) || c.act_raw) throw std::runtime_error("env.yaml: 속도 행동·정규 단위여야 함");
    const float dt_act = c.substeps * c.dt;   // 시뮬 걸음 (상태 step 단위)
    const int dec = a.hier ? c.decide_steps : 1;   // 정책 걸음 하나 = 시뮬 걸음 dec 개
    std::vector<std::string> skipped;
    std::vector<sim::World> worlds = sim::load_scene_list(a.list, hc, &skipped);
    if (worlds.empty()) throw std::runtime_error("학습 맵이 없음: " + a.list);
    std::fprintf(stderr, "학습 맵 %zu 개 (안 닫혀 뺀 것 %zu)\n", worlds.size(), skipped.size());
    // ---- PPO 설정 (그래프 모드는 ppo.h graph_ppo_config 한곳에서 — 환경 수·롤아웃 길이·에포크·γ)
    xppo::PPOConfig pc = a.graph ? xppo::graph_ppo_config() : xppo::PPOConfig();
    if (a.envs > 0) pc.num_envs = a.envs;
    const int N = pc.num_envs;
    sim::Env env(hc, worlds, lidar, N, a.seed, true);
    env.reset();
    env.sync();

    // 시험 맵: 커리큘럼 끔 (맵 하나, 오차 배율 1)
    sim::EnvHostCfg hc_eval = hc;
    hc_eval.cfg.curriculum = 0;
    std::vector<sim::World> tests;
    for (auto& t : a.test) tests.push_back(sim::build_world(stem(t), sim::load_scene(t), hc_eval));

    // ---- PPO
    if (a.lr_max > 0) pc.lr_max = a.lr_max;
    else if (!a.graph) pc.lr_max = 1e-3f;
    pc.mlp = a.mlp;
    if (a.std_max > 0) pc.std_max = a.std_max;
    if (a.entropy >= 0) pc.entropy_coef = a.entropy;
    if (a.desired_kl >= 0) pc.desired_kl = a.desired_kl;
    if (a.slot_cm_coef >= 0) pc.slot_cm_coef = a.slot_cm_coef;
    if (a.std_min > 0) pc.std_min = a.std_min;
    if (a.bound_coef >= 0) pc.bound_coef = a.bound_coef;
    if (a.chunk > 0) pc.chunk = a.chunk;
    if (a.hier) pc.gamma = (float)std::pow((double)pc.gamma, (double)dec);   // γ_결정 = 0.999^10 (같은 100 s 지평)
    if (a.steps > 0) pc.steps_per_env = a.steps;
    if (a.epochs > 0) pc.epochs = a.epochs;
    if (a.minibatches > 0) pc.minibatches = a.minibatches;
    if (a.gamma >= 0) pc.gamma = a.gamma;   // 결정 하나마다 (그래프·계층형), 직접 제어는 걸음마다
    if (a.lam >= 0) pc.lam = a.lam;
    xnet::ACSpec spec = a.graph ? xnet::explore_spec_graph() : a.hier ? xnet::explore_spec_hier() : xnet::explore_spec();
    if (a.graph && a.graph_share >= 0) spec.graph_share = a.graph_share != 0;
    xppo::PPO ppo(pc, spec, a.seed);
    if (!a.convert_in.empty()) {
      ppo.load_convert_shared(a.convert_in);
      ppo.save(a.convert_out);
      std::fprintf(stderr, "변환: %s (형식 2, 공유 인코더) → %s (형식 3, 인코더 나눔·프론티어 채널 첫 층 가중치 0), 반복 %d\n", a.convert_in.c_str(),
                   a.convert_out.c_str(), ppo.iteration());
      return 0;
    }
    const std::string dir = a.root + "/" + a.name;
    const std::string ckdir = dir + "/ckpt";
    if (a.resume) {
      const std::string p = ckdir + "/model_latest.bin";
      if (!file_exists(p)) throw std::runtime_error("--resume: 체크포인트 없음 " + p);
      ppo.load(p);
      std::fprintf(stderr, "이어 학습: %s 반복 %d, 학습률 %.2e\n", p.c_str(), ppo.iteration(), ppo.learning_rate());
      // 커리큘럼 단계 되살리기: 단계는 환경 상태라 체크포인트에 없다 — 그냥 이으면 모든 환경이 단계 0 에서 다시 오른다
      // (explore_v7 반복 2608·8850 재개 때 단계 평균 2.9 → 0 → ~1500 반복 동안 다시 오름, value 손실 튐).
      // ckpt/stages_latest.bin(환경마다 int32, 저장 때 같이 씀)이 있으면 그대로, 없으면 episodes.jsonl 끝 2000 판의 단계 분포를
      // 환경 번호 순서로 나눠 준다(옛 실행 폴더용 근사). 오르내림 창은 아래 cur_window_latest.bin.
      if (hc.cfg.curriculum) {
        std::vector<int32_t> stg;
        {
          std::ifstream f(ckdir + "/stages_latest.bin", std::ios::binary);
          if (f) {
            stg.resize(N);
            f.read(reinterpret_cast<char*>(stg.data()), sizeof(int32_t) * N);
            if (f.gcount() != (std::streamsize)(sizeof(int32_t) * N)) stg.clear();
          }
        }
        const char* how = "stages_latest.bin";
        if (stg.empty()) {
          std::ifstream f(dir + "/episodes.jsonl");
          std::vector<int> last;
          std::string line;
          const std::regex re("\"stage\":(-?\\d+)");
          while (std::getline(f, line)) {
            std::smatch m;
            if (std::regex_search(line, m, re)) last.push_back(std::stoi(m[1].str()));
          }
          if (!last.empty()) {
            const size_t k0 = last.size() > 2000 ? last.size() - 2000 : 0;
            std::vector<int> h(mdp::cur::kNumStages, 0);
            for (size_t k = k0; k < last.size(); ++k) h[std::max(0, std::min(mdp::cur::kNumStages - 1, last[k]))]++;
            const int tot = (int)(last.size() - k0);
            stg.resize(N);
            for (int i = 0; i < N; ++i) {   // 환경 i → 누적 분포에서 (i + 0.5)/N 자리의 단계
              const double q = (i + 0.5) / N * tot;
              int s = 0, acc = h[0];
              while (s + 1 < mdp::cur::kNumStages && acc <= q) acc += h[++s];
              stg[i] = s;
            }
            how = "episodes.jsonl 끝 판 분포";
          }
        }
        // 오르내림 창(cur_hist·cur_n): ckpt/cur_window_latest.bin(머리 "CURW" · 판 1 · 규칙 · 창 · 환경 수, 그 뒤 환경마다 uint32 hist · int32 n)이
        // 지금 설정(규칙·창 크기·환경 수)과 맞으면 그대로. 없거나 안 맞으면(옛 실행 폴더·규칙 바꿈) 창을 비운다 — 그러면 환경마다 창이 다시 찰 때까지
        // (성공 규칙 window 판) 단계가 안 바뀐다(단계 자체는 위대로 되살림)
        std::vector<uint32_t> wh;
        std::vector<int32_t> wn;
        const char* how_w = "없음 — 창 비움";
        if (!stg.empty()) {
          std::ifstream f(ckdir + "/cur_window_latest.bin", std::ios::binary);
          int32_t hd[5] = {0, 0, 0, 0, 0};
          if (f && f.read(reinterpret_cast<char*>(hd), sizeof(hd))) {
            if (std::memcmp(hd, "CURW", 4) == 0 && hd[1] == 1 && hd[2] == 1 && hd[3] == hc.cfg.cur_window && hd[4] == N) {
              wh.resize(N);
              wn.resize(N);
              f.read(reinterpret_cast<char*>(wh.data()), sizeof(uint32_t) * N);
              f.read(reinterpret_cast<char*>(wn.data()), sizeof(int32_t) * N);
              if (!f) { wh.clear(); wn.clear(); how_w = "파일 짧음 — 창 비움"; }
              else how_w = "cur_window_latest.bin";
            } else {
              how_w = "규칙·창·환경 수가 달라 — 창 비움";
            }
          }
        }
        if (!stg.empty()) {
          std::vector<sim::EnvState> es;
          env.get_state(es);
          double mean = 0;
          for (int i = 0; i < N; ++i) {
            es[i].stage = stg[i];
            es[i].cur_hist = wh.empty() ? 0u : wh[i];
            es[i].cur_n = wn.empty() ? 0 : wn[i];
            env.set_state(i, es[i]);
            mean += stg[i];
          }
          env.reset();
          env.sync();
          std::fprintf(stderr, "커리큘럼 단계 되살림 (%s): 평균 %.2f · 오르내림 창 (%s)\n", how, mean / N, how_w);
        }
      }
    }

    if (!a.eval_ckpt.empty()) {
      ppo.load(a.eval_ckpt);
      std::vector<float> ls(mdp::act::CONT, 0.f);
      if (!cat) cudaMemcpy(ls.data(), ppo.ac().log_std(), sizeof(float) * ls.size(), cudaMemcpyDeviceToHost);
      std::fprintf(stderr, "평가만: %s 반복 %d, σ (v, vy, ω) = %.3f %.3f %.3f\n", a.eval_ckpt.c_str(), ppo.iteration(), std::exp(ls[0]),
                   std::exp(ls[1]), std::exp(ls[2]));
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
          .set("camera_range", (double)D.obs_max_z);   // 관측이 보는 거리 (센서는 더 멀리도 냄 — sensor_model.yaml depth)
      robot.set("sim", simb);
      std::string err;
      rf::Json u = read_urdf_robot(LIMO_URDF_DIR, &err);
      if (err.empty()) robot.set("urdf", u);
      else robot.set("urdf_error", err);
    }
    rf::Json meta = rf::Json::object();
    meta.set("robot", robot);
    meta.set("kind", "explore_ppo").set("group", a.name).set("seed", (unsigned long long)a.seed).set("n_envs", N)
        .set("ctrl_hz", 1.0 / (dt_act * dec)).set("hier", a.hier).set("graph", a.graph).set("decide_steps", dec).set("success_coverage", mdp::cur::kSuccessCov).set("action_limits", limits)
        .set("reward_terms", terms).set("n_train_maps", (unsigned long long)worlds.size()).set("test_maps", tests_meta)
        .set("sweep_guard", c.sweep_guard).set("cur_top_random", c.cur_top_random).set("con_mask", c.con_mask).set("con_end", c.con_end).set("scene_list", a.list).set("steps_per_env", pc.steps_per_env).set("lr_max", a.lr_max)
        .set("n_params", (long long)ppo.ac().n_params)
        .set("ppo", rf::Json::object().set("epochs", pc.epochs).set("minibatches", pc.minibatches).set("clip", pc.clip)
                        .set("gamma", pc.gamma).set("lam", pc.lam).set("entropy_coef", pc.entropy_coef).set("lr", pc.lr)
                        .set("desired_kl", pc.desired_kl).set("max_grad_norm", pc.max_grad_norm).set("slot_cm_coef", pc.slot_cm_coef)
                        .set("std_min", pc.std_min).set("std_max", pc.std_max)
                        .set("bound_coef", pc.bound_coef).set("bound_soft", pc.bound_soft));
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
    // 그래프 모드는 결정마다 걸음 수가 달라 이어 학습 때 지난 걸음 수를 모른다 — 결정당 21 걸음(측정 평균 2.0 ~ 2.3 s, mdp/README)으로 어림 (잠정, 가로축 표시용)
    double total_steps = (double)ppo.iteration() * pc.steps_per_env * N * (a.graph ? 21 : dec);   // 시뮬 걸음 (0.1 s) 단위
    std::vector<sim::GraphDec> h_gd;   // 그래프 모드: 방금 끝난 결정들
    int* d_pick = nullptr;             // 그래프 모드 진단 [2]: 가린 선택 수 · 이웃 없는 결정 수 (k_graph_pick_check)
    if (a.graph) cudaMalloc(&d_pick, sizeof(int) * 2);

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
    // sample = true: 같은 시험 판(같은 씨앗)을 정책 분포 샘플로 — 학습 롤아웃과 같은 행동 방식(evals/test_sample_it*.json)
    auto evaluate = [&](int it, bool sample) {
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
      double* d_mus;
      constexpr int kMu = 8;
      cudaMalloc(&d_mus, sizeof(double) * kMu * NT);
      cudaMemset(d_mus, 0, sizeof(double) * kMu * NT);
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
      // 그래프 모드는 결정 길이가 가변(걸음 1 개 이상) — 결정 수 상한은 걸음 상한과 같게 (판이 다 끝나면 아래에서 먼저 나감)
      const int eval_steps = a.graph ? c.max_steps + 10 : c.max_steps / dec + 10;
      for (int step = 0; step < eval_steps; ++step) {
        bool any_live = false;
        for (size_t ti = 0; ti < NT; ++ti) {
          if (!live[ti]) continue;
          any_live = true;
          float* act = d_eact + (size_t)mdp::act::DIM * E * ti;
          const sim::StepOut o = evs[ti]->outputs();
          cudaStreamWaitEvent(pst, ev_obs[ti], 0);   // 이 맵의 지난 걸음 관측이 다 나온 뒤
          const int64_t sc = sample ? (int64_t)step * (int64_t)NT + (int64_t)ti : -1;
          if (a.graph) {
            const sim::GraphOut g = evs[ti]->graph();
            ppo.act_inference(xppo::GraphObs{g.feat, g.edge, g.cur, g.glob}, act, E, sc);
          } else {
            ppo.act_inference(o.obs_img, o.obs_1d, act, E, sc);
          }
          if (!sample && !cat) k_mu_stats<<<(E + 127) / 128, 128, 0, pst>>>(act, d_first + (size_t)E * ti, E, d_mus + (size_t)kMu * ti);
          if (!sample && std::getenv("EVAL_ACT_DUMP") && step < 300) {   // 변환 확인용: 결정적 평가 행동(평균) 걸음마다
            static FILE* fa = std::fopen(std::getenv("EVAL_ACT_DUMP"), "w");
            std::vector<float> ha((size_t)mdp::act::DIM * E);
            cudaStreamSynchronize(pst);
            cudaMemcpy(ha.data(), act, sizeof(float) * ha.size(), cudaMemcpyDeviceToHost);
            for (int i = 0; i < E; ++i) std::fprintf(fa, "%zu %d %d %a %a %a %a\n", ti, step, i, ha[4 * i], ha[4 * i + 1], ha[4 * i + 2], ha[4 * i + 3]);
            std::fflush(fa);
          }
          cudaEventRecord(ev_act[ti], pst);
          cudaStreamWaitEvent(mst[ti], ev_act[ti], 0);
          evs[ti]->step(act, mst[ti]);
          const sim::StepOut o2 = evs[ti]->outputs();
          k_eval_book<<<(E + 127) / 128, 128, 0, mst[ti]>>>(o2.done, o2.reward, o2.ep, E, d_first + (size_t)E * ti,
                                                             d_fe + (size_t)E * ti, d_er + (size_t)E * ti);
          cudaEventRecord(ev_obs[ti], mst[ti]);
        }
        if (!any_live) break;
        if ((step + 1) % kCheck == 0 || step + 1 == eval_steps) {
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
      std::vector<double> hmu((size_t)kMu * NT);
      cudaMemcpy(hmu.data(), d_mus, sizeof(double) * kMu * NT, cudaMemcpyDeviceToHost);
      cudaFree(d_mus);
      if (!sample && !cat)
        for (size_t ti = 0; ti < NT; ++ti) {
          const double* q = hmu.data() + (size_t)kMu * ti;
          const double nn = q[0] > 0 ? q[0] : 1;
          std::fprintf(stderr, "  평가 μ %s: 걸음 %.0f, 평균 μ (v, vy, ω) %.3f %.3f %.3f, |μ| > 1 비율 %.3f %.3f %.3f, μ_v > 1.5 비율 %.3f\n",
                       tests[ti].name.c_str(), q[0], q[1] / nn, q[2] / nn, q[3] / nn, q[4] / nn, q[5] / nn, q[6] / nn, q[7] / nn);
        }
      evs.clear();
      for (size_t ti = 0; ti < NT; ++ti) {
        const int* first = hfirst.data() + (size_t)E * ti;
        const sim::EpInfo* fe = hfe.data() + (size_t)E * ti;
        const double* er = her.data() + (size_t)E * ti;
        double n = E, succ = 0, coll = 0, trunc = 0, shrt = 0, cov = 0, t95 = 0, t95n = 0, len = 0, cad = 0, decl = 0, rws = 0;
        double idle = 0, tcl = 0;   // 진행 없음 끝 · 성공(닫힘)까지 시간 합
        for (int i = 0; i < E; ++i) {
          const auto& f = fe[i];
          if (!first[i]) { trunc += 1; continue; }
          succ += f.done == mdp::cur::kDeclareOk;
          idle += f.done == mdp::cur::kNoProgress;
          if (f.done == mdp::cur::kDeclareOk) tcl += f.t;
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
        row.set("eval", tests[ti].name).set("split", sample ? "test_sample" : "test").set("n", n).set("success", succ / n).set("collision", coll / n)
            .set("truncation", trunc / n).set("no_progress", idle / n).set("pass", coll == 0 && succ == n);
        if (succ > 0) row.set("t_success", tcl / succ);   // 닫힘(성공)까지 시간 (s)
        rows.push(row);
        s_n += n; s_succ += succ; s_coll += coll; s_trunc += trunc; s_short += shrt; s_cov += cov; s_t95 += t95; s_t95n += t95n;
        s_len += len; s_cad += cad; s_decl += decl; s_rew += rws;
        std::fprintf(stderr, "  평가%s %s: 성공(닫힘) %.0f/%d 충돌 %.0f 진행 없음 %.0f 끊김 %.0f 닫힘까지 %.1f s\n", sample ? "(샘플)" : "", tests[ti].name.c_str(), succ, E,
                     coll, idle, trunc, succ > 0 ? tcl / succ : 0.0);
      }
      rf::Json evj = rf::Json::object();
      evj.set("iter", it).set("ckpt", "model_" + std::to_string(it)).set("rows", rows);
      run.eval((sample ? "test_sample_it" : "test_it") + std::to_string(it), evj);
      auto& r = run.row();
      if (sample) {
        r.last("eval_sample/success_rate", s_succ / s_n);
        r.last("eval_sample/coverage_mean", s_cov / s_n);
        r.last("eval_sample/collision_rate", s_coll / s_n);
        r.last("eval_sample/declare_short_rate", s_short / s_n);
        std::fprintf(stderr, "  평가(샘플) %.1f s\n", ms_since(t0) / 1000);
        return;
      }
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
      if (hc.cfg.curriculum) {   // 커리큘럼 단계 (이어 학습 때 되살림 — 위 "커리큘럼 단계 되살리기")
        std::vector<sim::EnvState> es;
        env.get_state(es);
        std::vector<int32_t> stg(N);
        for (int i = 0; i < N; ++i) stg[i] = es[i].stage;
        const std::string p = ckdir + "/stages_latest.bin";
        std::ofstream f(p + ".tmp", std::ios::binary);
        f.write(reinterpret_cast<const char*>(stg.data()), sizeof(int32_t) * N);
        f.close();
        std::rename((p + ".tmp").c_str(), p.c_str());
        // 오르내림 창 (위 "되살리기" — 규칙·창 크기를 머리에 적어 바뀌면 안 씀)
        std::vector<uint32_t> wh(N);
        std::vector<int32_t> wn(N);
        for (int i = 0; i < N; ++i) { wh[i] = es[i].cur_hist; wn[i] = es[i].cur_n; }
        const int32_t hd[5] = {0x57525543 /* "CURW" */, 1, 1 /* 규칙 번호: 성공 (옛 덮음 0 은 없앰) */, hc.cfg.cur_window, N};
        const std::string pw = ckdir + "/cur_window_latest.bin";
        std::ofstream fw(pw + ".tmp", std::ios::binary);
        fw.write(reinterpret_cast<const char*>(hd), sizeof(hd));
        fw.write(reinterpret_cast<const char*>(wh.data()), sizeof(uint32_t) * N);
        fw.write(reinterpret_cast<const char*>(wn.data()), sizeof(int32_t) * N);
        fw.close();
        std::rename((pw + ".tmp").c_str(), pw.c_str());
      }
    };

    if (!a.eval_ckpt.empty()) {
      evaluate(ppo.iteration(), false);
      evaluate(ppo.iteration(), true);
      cudaFree(d_pick);
      cudaFree(d_act);
      cudaFree(d_done);
      return 0;
    }

    // ---- 학습
    const auto t_start = std::chrono::steady_clock::now();
    const int it0 = ppo.iteration();
    int it_done = 0;
    int lr_lock_run = 0;   // 학습률 하한 갇힘이 이어진 반복 수 (아래 감시)
    while (!g_stop) {
      const int it = ppo.iteration();
      if (a.iters > 0 && it - it0 >= a.iters) break;
      if (a.minutes > 0 && ms_since(t_start) / 60000.0 >= a.minutes) break;
      const auto t_it = std::chrono::steady_clock::now();
      if (a.replay_every > 0 && it % a.replay_every == 0) rec_want = std::min(kReplayEnvs, kReplayMax - (int)rec.size());
      const bool need_state = rec_want > 0 || !rec.empty();
      double t_act = 0, t_sim = 0, t_host = 0;
      double dec_steps = 0;   // 그래프 모드: 이번 반복 결정들의 시뮬 걸음 합
      if (a.graph) cudaMemset(d_pick, 0, sizeof(int) * 2);
      // 반복 통계
      double n_ep = 0, n_succ = 0, n_coll = 0, n_pcoll = 0, n_trunc = 0, n_short = 0, n_idle = 0, cov = 0, cad = 0, n_decl = 0, t95 = 0, t95n = 0, stage = 0,
             len = 0, rew_ep = 0, step_rew = 0;
      double t_succ_sum = 0;   // 성공(닫힘) 판의 시간 합
      std::vector<double> term_sum(mdp::rew::kTerms, 0.0);
      for (int t = 0; t < pc.steps_per_env; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        const sim::StepOut o = env.outputs();
        if (a.graph) {
          const sim::GraphOut g = env.graph();
          ppo.act(xppo::GraphObs{g.feat, g.edge, g.cur, g.glob}, d_act);
          k_graph_pick_check<<<(N + 255) / 256, 256>>>(d_act, g.cand, N, d_pick);
        } else {
          ppo.act(o.obs_img, o.obs_1d, d_act);
        }
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
        if (a.graph) {   // 결정 길이 (가변) — 시뮬 걸음 수 장부
          env.get_gdec(h_gd);
          double s = 0;
          for (int i = 0; i < N; ++i) s += h_gd[i].steps;
          dec_steps += s;
          total_steps += s;
        }
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
          if (d == mdp::cur::kDeclareOk) t_succ_sum += f.t;
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
            line.set("iter", it).set("env_steps", a.graph ? total_steps : total_steps + (double)(t + 1) * N * dec).set("env", i).set("ep", id).set("map", stem(worlds[mi].name))
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
      const double iter_steps = a.graph ? dec_steps : (double)pc.steps_per_env * N * dec;   // 이번 반복의 시뮬 걸음
      if (!a.graph) total_steps += iter_steps;
      auto tu = std::chrono::steady_clock::now();
      const sim::StepOut o = env.outputs();
      xppo::UpdateStats us;
      ppo.set_critic_only(it - it0 < a.critic_warmup);
      if (a.graph) {
        const sim::GraphOut g = env.graph();
        us = ppo.update(xppo::GraphObs{g.feat, g.edge, g.cur, g.glob});
      } else {
        us = ppo.update(o.obs_img, o.obs_1d);
      }
      const double t_upd = ms_since(tu);
      const double t_iter = ms_since(t_it);
      if (!std::isfinite(us.value) || !std::isfinite(us.surrogate) || !std::isfinite(us.kl)) {
        run.event("NaN at iteration " + std::to_string(it) + " — stop");
        std::fprintf(stderr, "NaN: 반복 %d 에서 멈춤\n", it);
        run.finish();
        return 3;
      }
      // 학습률 하한 갇힘 감시 (ppo/README "슬롯 로짓 공통 성분"): 적응 학습률은 KL 이 학습률 탓이라 가정한다. 하한(lr_min)에서도 반복 평균 KL 이 목표의
      // 2 배를 넘는 반복이 kLockIters 번 이어지면 KL 은 학습률로 줄지 않는 것(수치 오차·과민한 망)이라 그대로 갇힌다 — explore_h2 반복 2889 → 3799 (kl 0.05 → 14),
      // 공통 성분 벌점 끈 처음부터 학습 lr 상한 1e-2 반복 54 → (kl 0.12, 걸음 전 kl0 1e-3). 이어 가면 model_latest 만 망가지므로 멈춘다
      {
        constexpr int kLockIters = 10;
        const bool locked = pc.desired_kl > 0 && us.lr <= pc.lr_min * 1.0001f && us.kl > 2.f * pc.desired_kl;
        lr_lock_run = locked ? lr_lock_run + 1 : 0;
        if (lr_lock_run >= kLockIters) {
          char msg[240];
          std::snprintf(msg, sizeof msg, "lr locked at floor %.1e with kl %.3g > 2*desired_kl for %d iterations at iteration %d (kl0 %.3g) - stop",
                        pc.lr_min, us.kl, kLockIters, it, us.kl0);
          run.event(msg);
          std::fprintf(stderr, "학습률 하한 갇힘: 반복 %d 까지 %d 반복 동안 lr %.1e 에서 kl %.3g > 목표 2 배 (걸음 전 kl0 %.3g, 로짓 공통 성분 %.3g) — 멈춤\n", it,
                       kLockIters, pc.lr_min, us.kl, us.kl0, us.logit_cm);
          run.finish();
          return 4;
        }
      }
      float ls[3] = {0.f, 0.f, 0.f};   // 계층형은 σ 없음
      if (!cat) cudaMemcpy(ls, ppo.ac().log_std(), sizeof(ls), cudaMemcpyDeviceToHost);
      size_t fr, tot;
      cudaMemGetInfo(&fr, &tot);
      auto& r = run.row();
      r.last(rf::key::ITER, it);
      r.last(rf::key::STEPS, total_steps);
      r.last(rf::key::FPS, iter_steps / (t_iter / 1000));   // 시뮬 걸음/s (계층형은 결정/s × dec, 그래프는 결정마다 잰 걸음 수 합)
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
        if (n_succ > 0) r.mean_w("rollout/t_success_mean", t_succ_sum / n_succ, n_succ);
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
      r.last("train/kl0", us.kl0);
      r.last("train/value0", us.value0);
      if (cat) {
        r.last("train/logit_cm", us.logit_cm);
        r.last("train/logit_spread", us.logit_spread);
        r.last("train/logit_absmax", us.logit_absmax);
      }
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
                   t_sim, t_host, t_upd, iter_steps / (t_iter / 1000));
      if (a.graph) {   // 그래프 모드 줄: 결정 속도·길이, 정책 엔트로피, 가린 선택(0 이어야), 유효 노드 수
        const double nd = (double)pc.steps_per_env * N;
        int hp[2] = {0, 0};
        cudaMemcpy(hp, d_pick, sizeof(hp), cudaMemcpyDeviceToHost);
        r.last("time/decisions_per_s", nd / (t_iter / 1000));
        r.last("rollout/decision_steps_mean", dec_steps / nd);
        r.last("rollout/masked_picks", hp[0]);
        r.last("rollout/no_neighbor", hp[1]);
        r.last("rollout/graph_nodes_mean", ppo.graph_nodes_mean());
        std::fprintf(stderr,
                     "  그래프: 결정 %.0f (%.1f 결정/s, 결정마다 %.1f 걸음) 판 %.0f 판보상 %.3f 엔트로피 %.4f value손실 %.4f 가린 선택 %d 이웃 없음 %d 유효 노드 %.1f, 반복 %.1f s\n",
                     nd, nd / (t_iter / 1000), dec_steps / nd, n_ep, n_ep ? rew_ep / n_ep : 0.0, us.entropy, us.value, hp[0], hp[1], ppo.graph_nodes_mean(),
                     t_iter / 1000);
        if (hp[0] != 0) {
          run.event("masked action picked at iteration " + std::to_string(it) + " - stop");
          std::fprintf(stderr, "가린 선택 %d 개: 반복 %d 에서 멈춤 (정책 가림과 시뮬 후보가 어긋남)\n", hp[0], it);
          run.finish();
          return 5;
        }
      }
      ++it_done;
      if (a.eval_every > 0 && it_now % a.eval_every == 0) { evaluate(it_now, false); evaluate(it_now, true); }
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
