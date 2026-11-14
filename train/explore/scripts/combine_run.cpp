// combine_run — 두 층을 한 판에서 합쳐 탐사를 끝까지 돌린다: 위층 = 가장 가까운 후보 규칙(slot0, 1 Hz), 아래층 = 학습한 RL 정책(10 Hz) 또는 규칙 추종기.
// 시뮬은 아래층 RL 걸음(EnvCfg::low_rl, step 한 번 = 0.1 s) + 합친 탐사 스위치(EnvCfg::low_explore — 탐사 판 끝 규칙 그대로, 정체는 막힘 신호만)로 돈다.
//
//   combine_run --scenes a.scene [b.scene …] [--policy <아래층 체크포인트>] [--episodes 256] [--seed 1] [--people 0] [--eval-seed] [--out <접두사>]
//               [--no-avoid] [--no-redecide] [--no-stick] [--hold <걸음>] [--stall-s <초>] [--rule-near <m>] [--env …] [--model …] [--frontier …]
//               [--upper slot0|random|decider|cand_tsp] [--rule-stick x] [--rule-turn m] [--decider-url http://127.0.0.1:<포트>] [--decider-input text|image|both|all] [--decider-prompt 1|2|3]
//               [--decider-mode choice|prune] [--decider-veto-thr 0.5] [--decider-ask-m 4] [--decider-cam-h 480] [--decider-cam-div 2] [--gt] [--gt-base a0.scene …]
//               [--decider-timeout-ms 10000] [--decider-parallel 16] [--decider-px 384] [--decider-shuffle] [--decider-dump <폴더>] [--decider-dump-n 6]
//               [--decider-finish] [--decider-config <yaml>] [--decider-mission "<글>"] [--decider-left-hold N] [--decider-score-margin x] [--decider-score-finish x]   끝 선언 · 점수 방식 (아래)
//               [--video <접두사>] [--video-env 0] [--end-hold 3]   환경 하나의 판을 mp4 로 (위에서 본 지도 + 걸음마다 새로 그린 카메라 그림·모델에 보낸 그림 + 판단 모델·아래층 RL 출력 글 — decider.h Video). 판이 끝나면 리셋 없이 명령 0 으로 --end-hold 초 더 찍음 (0 = 끔)
//   --policy 를 주면 아래층 RL(LowCmd::follow 2 — 목표 칸 = 고른 후보 칸), 안 주면 같은 걸음 루프에서 규칙 추종기(follow 0 — 후보 번호를 그대로, 대조).
//   --auto: 위층 규칙을 호스트 대신 환경이 돌린다 (EnvCfg::low_auto 2 — 탐사 판 학습 train --low --low-explore 와 같은 길, --policy 와 같이). 판 결과가 호스트 규칙과
//           같아야 한다 (학습 때 받는 목표 = 평가 때 받는 목표). 위층 걸음 통계(결정 수 등)는 호스트 규칙 것만 셈
//   판·씨앗·판 길이 상한은 frontier_run --modes slot0 와 같다 (장면 i 의 씨앗 seed · 1000003 + i · 7919, --eval-seed 면 seed · 7919 + 17 + i).
//
// 위층 규칙 (환경마다, 결정 = 10 걸음마다 · 아래 "바로 다시 고름" 때):
//   · 후보 0 번부터 차례로, 피하는 칸(아래) 둘레가 아닌 첫 후보. 없으면 목표 없음(규칙 추종기의 제자리 회전 — 지도 닫힘을 기다림)
//   · 고른 후보의 덩어리가 지금 목표 칸을 품으면(Env::get_cand_cur — hier_path.h act_same 과 같은 뜻) 목표 칸을 그대로 둔다 (--no-stick 으로 끔)
//   · 막힘 신호(sim::LowRl::stall — 10 s 동안 0.3 m 를 못 벗어남, env.yaml follow.stall_*)가 서면 그때 가던 칸(stall_tgt)을 피하는 칸에 넣고 바로 다시 고른다.
//     피하는 반경·시간은 규칙 추종기 복구와 같은 값(env.yaml recovery.same_goal_radius · blacklist_time — 둘 다 잠정). explore_lite 의 progress_timeout →
//     frontier_blacklist 와 같은 꼴 (--no-avoid 로 끔)
//   · 아래층 RL 만: 목표 통과 반경에 들었거나(pass_radius) 경로 끝이 목표 밖이거나 경로를 못 짰으면 다음 걸음에 바로 다시 고른다 (--no-redecide 로 끔).
//     경로 끝이 목표 밖·경로 없음은 그 칸도 피하는 칸에 넣는다 (정책 환경은 규칙 추종기의 복구가 없어서)
//
// 위층 고르기 (--upper, 위 규칙의 첫 줄만 바꿈 — 피하는 칸·목표 이어 가기·바로 다시 고름은 그대로):
//   slot0   : 유효 후보(빈 슬롯·피하는 칸이 아닌 후보) 가운데 번호가 가장 작은 것 — 기준선
//   random  : 유효 후보 가운데 고르게 아무거나 — 하한
//   cand_tsp: 후보를 모두 도는 가장 짧은 차례(로봇 지도 BFS 거리)의 첫 후보 — scripts/cand_rules.h, frontier_run --modes cand_tsp 와 같은 규칙(GT 안 씀).
//             --rule-stick · --rule-turn 도 frontier_run 과 같은 뜻(기본 0). 후보 v2 는 환경 변수 SIM_CAND_V2=1 (env.yaml candidates_v2). 고른 후보가 피하는 칸이면 slot0 차례
//   decider : 바깥 판단 모델 서버(decider.h — llama-server /v1/systemone)에 choice 질문 하나로 물음 (--decider-url 을 주면 이 방식). 유효 후보가 2 개 이상인 결정만 묻고,
//             같은 걸음의 모든 환경 요청을 한꺼번에 보낸다. 서버 오류·시간 끝·답 꼴 오류면 그 결정은 slot0 로 떨어지고 그 수를 센다 (많으면 모델 결과가 아니다).
//             판단에 걸린 시간은 시뮬 시계에 안 들어간다. --decider-shuffle: 글자를 후보 번호와 무관하게 섞어 붙임 (글자 자리 쏠림 확인용).
//             --decider-dump: 요청(JSON)·사진(PNG)·답을 장면마다 몇 개 남김
//             --decider-input all: 로봇 카메라 자리에서 본 장면 그림(진짜 RGB 아님 — decider.h) + 지도 사진 + 글을 한 요청에. 내부값은 sensor_model.yaml depth,
//             달린 자리·숙인 각은 env.yaml body, 그림은 --decider-cam-div 배 줄임. 카메라 그림은 참 자세에서 그린다 (실물 카메라가 그렇듯)
//             --decider-mode prune: 규칙 차례(slot0)는 그대로 두고, 앞 --decider-ask-m 후보마다 "그 길이 막혔나"(noul)만 물어 확률이 --decider-veto-thr 이상인
//             후보를 건너뛴다. 물은 후보가 다 막혔다면 안 물은 다음 후보(없으면 규칙의 첫 후보). 모델은 그 밖에 차례를 못 바꾼다
//             --decider-mode score: 요청 하나에 후보마다 score 질문 하나("그 후보로 가면 지도가 얼마나 더 채워지나", 단계 5 개 — 글·단계는 scripts/config/decider.yaml).
//             기대 점수(Σ 단계 × 확률)가 가장 높은 후보로 간다. 최고에서 score_margin 안이면 비긴 것: 지금 목표가 그 안에 있으면 그대로, 아니면 가까운 후보.
//             <out>_<장면>_score.tsv 에 결정·후보마다 단계 확률과 기대 점수, <out>_<장면>_gain.tsv 에 고른 목표로 가는 동안 로봇 지도에서 새로 알게 된 칸 수와 걸린 시간
//             (도착 · 포기 · 바꿈 · 판 끝까지 — 로봇 지도만 쓰는 값, 뒤에 학습 정답으로)
//   --decider-finish (끝 선언, 기본 끔 — 끄면 모든 동작·출력이 그대로. --decider-mode choice · score 만, prune 은 안 됨):
//             choice: 고르기 질문은 그대로 후보만(FINISH 선택지 없음). 같은 요청에 score 질문 하나("바닥이 얼마나 남았나", 단계 5 개 — 글·단계는 설정 파일)를 더해
//             "남은 게 없음"(가장 그럴듯한 단계가 0, 또는 left_finish_below 를 주면 기대 점수가 그 아래)이 left_hold(기본 2, 잠정) 결정 이어지면 끝을 선언한다.
//             후보가 2 개보다 적은 결정에는 score 질문만 보낸다. <out>_<장면>_left.tsv 에 결정마다 choice 확률 · 단계 확률 · 기대 점수와 그때부터 판 끝까지 로봇 지도가 는 칸 수.
//             score: 모든 후보의 가장 그럴듯한 단계가 0(또는 score_finish_below 를 주면 모든 기대 점수가 그 아래)이거나 후보가 하나도 없으면 끝을 선언한다.
//             임무 문장은 scripts/config/decider.yaml mission (--decider-config · --decider-mission 으로 바꿈).
//             choice 의 상태 글에는 로봇이 스스로 아는 값만: 지금 후보 수 · 포기한 칸이라 뺀 후보 수 · 라이다 지도 닫힘 ·
//             지도가 마지막으로 늘어난 뒤 시간 · 가다가 포기한 목표 수. 모델이 끝을 선언하면 그 걸음에 판이 끝난다 (EnvCfg::finish_declare). 판정은 로봇 지도만으로:
//             선언한 걸음의 지도가 닫힘(mdp/frontier.h 닫힘 규칙 — EnvState::map_closed)이면 성공 코드 1, 아니면 이른 선언 코드 4. 정답 덮음은 판정에 안 씀.
//             같은 판정을 ROS 점유 격자만으로 하는 함수(decider.h grid_open_frontier)의 결과도 tsv 에 같이 적는다. 닫힘 성공 규칙(닫힘 + 10 s 지도 안 늘어남)은 그대로 살아 있어
//             모델이 선언을 안 해도 판은 끝난다 — 어느 쪽이 끝냈는지는 tsv ended_by_model 칸
//   --gt [--gt-base …]: 정답 격자로 "막힌 후보"를 센다 (셈 전용, 고르기·판 결과는 그대로): 로봇 참 자리에서 후보 칸까지 참 길 거리가, 낮은 장애물이 없는 바탕 장면
//             (--gt-base, --scenes 와 같은 차례)에서보다 3 m · 1.5 배 넘게 길거나 못 가면 막힘 (잠정). 바탕을 안 주면 늘 0
//
// 출력: 장면마다 성공률·실패 종류·닫힘까지 시간·평균 속도·이동 거리 + 원인 분석용 걸음 통계. --out 을 주면 <접두사>_<장면>.tsv 에 판마다 한 줄
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "actor_critic.h"
#include "cand_rules.h"
#include "decider.h"
#include "env_api.h"
#include "frontier.h"
#include "frontier_explorer.h"
#include "ppo.h"

namespace {

struct Args {
  std::vector<std::string> scenes;
  int episodes = 256, people = 0;
  uint64_t seed = 1;
  std::string policy, out;
  std::string env = SIM_CONFIG_DIR "/env.yaml", model = SIM_CONFIG_DIR "/sensor_model.yaml", frontier = FRONTIER_CONFIG;
  bool eval_seed = false, avoid = true, redecide = true, stick = true;
  double rule_near = 0.0;   // > 0 이면 아래층 RL 판에서도 목표까지 직선 거리가 이 값 아래인 결정은 규칙 추종기에 맡김 (실험용)
  double stall_s = 0.0;   // > 0 이면 정체 판정 시간(env.yaml follow.stall_time)을 이 값으로 (실험용)
  bool autox = false;   // --auto: 위층 규칙을 환경이 돌림 (EnvCfg::low_auto 2 — 탐사 판 학습과 같은 길). 호스트 규칙과 결과가 같은지 재는 용
  int hold = 0;   // 아래층 RL: 목표 칸을 정한 뒤 이 걸음 수 동안은 주기 결정에서 안 바꿈 (0 = 끔 — 실험용)
  std::string upper = "slot0";   // 위층 고르기: slot0 · random · decider · cand_tsp
  cr::Opt rule;                  // cand_tsp: --rule-stick · --rule-turn (scripts/cand_rules.h, frontier_run 과 같은 뜻)
  std::string dec_url, dec_input = "text", dec_dump;
  int dec_prompt = 1, dec_timeout_ms = 10000, dec_parallel = 16, dec_px = 384, dec_dump_n = 6;
  bool dec_shuffle = false;
  std::string dec_mode = "choice";   // choice: 후보 하나를 고름 · prune: 규칙 차례 그대로, 막혔다는 후보만 건너뜀 (입력 all 만)
  double dec_veto_thr = 0.5;   // prune: "막혔다" 확률이 이 값 이상이면 건너뜀 (잠정)
  int dec_ask_m = 4;           // prune: 규칙 차례 앞에서 이 수만큼만 물음 (잠정)
  int dec_cam_h = 480, dec_cam_div = 2;   // 카메라 그림: 실물 컬러 영상 높이(docs/hardware.md 640 × 480 — 폭·내부값은 sensor_model.yaml depth) · 줄이는 배수 (잠정)
  bool dec_finish = false;     // --decider-finish: 판 끝 선언을 판단 모델의 선택지로 (입력 all 만). 끄면 모든 동작·출력이 그대로
  int dec_left_hold = 0;   // --decider-left-hold: 설정 파일 값을 덮어씀 (0 = 파일 값)
  double dec_score_margin = -1.0, dec_score_finish = -2.0;   // --decider-score-margin · --decider-score-finish: 설정 파일 값을 덮어씀 (안 주면 파일 값)
  std::string dec_config = DECIDER_CONFIG, dec_mission;   // 판단 모델 설정 파일 (임무 문장) · --decider-mission 으로 덮어씀
  bool gt = false;             // 정답 격자로 "막힌 후보" 셈 (고르기에는 안 씀)
  std::vector<std::string> gt_base;   // 장면마다 낮은 장애물 없는 바탕 장면 (--scenes 와 같은 차례). 없으면 그 장면 자체 (막힌 후보 0)
  double gt_near = 0.5, gt_add = 3.0, gt_mul = 1.5;   // 후보 칸 둘레 (m) · 바탕보다 이만큼(m) 그리고 이 배수 넘게 길면 "막힘" (잠정)
  std::string video;           // --video <접두사>: 환경 하나(--video-env)의 판을 <접두사>_<장면>.mp4 로 녹화 (decider.h Video — ffmpeg 필요)
  double end_hold = 3.0;       // --end-hold <s>: 녹화할 때 판이 끝난 뒤 리셋 없이 명령 0 으로 세워 두고 더 찍는 시간 (0 = 끔 — 옛 방식). 잠정. --video 없으면 안 씀
  int video_env = 0;
};

std::string stem(const std::string& p) {
  const auto s = p.find_last_of('/');
  std::string b = s == std::string::npos ? p : p.substr(s + 1);
  const auto k = b.rfind('.');
  return k == std::string::npos ? b : b.substr(0, k);
}

enum Upper { kSlot0, kRandom, kDecider, kTsp };
uint64_t mix(uint64_t x) {   // splitmix64
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

constexpr int kBl = 8;   // 환경마다 피하는 칸 수 (고리)
struct Avoid {
  int cell[kBl];
  int until[kBl];
  int n = 0;
  Avoid() { for (int i = 0; i < kBl; ++i) { cell[i] = -1; until[i] = 0; } }
};

// ---- 정답 격자 길 거리 (--gt, 셈 전용) ----
// 갈 수 있는 칸 = 갈 수 있는 바닥(kCellFloor)이고 막힌 칸까지 거리가 로봇 반폭 이상. 8 이웃 다익스트라 (m)
std::vector<uint8_t> gt_trav(const sim::World& W, float half_wid) {
  std::vector<uint8_t> t(W.grid.size(), 0);
  for (size_t k = 0; k < t.size(); ++k) t[k] = (W.grid[k] & sim::kCellFloor) && (float)W.clear[k] >= half_wid * 100.f;
  return t;
}
int gt_cell(const std::vector<uint8_t>& trav, int gw, int gh, int cx, int cy) {   // (cx, cy) 에서 가장 가까운 갈 수 있는 칸 (0.5 m 안), 없으면 −1
  int best = -1, bd = 1 << 30;
  for (int dy = -10; dy <= 10; ++dy)
    for (int dx = -10; dx <= 10; ++dx) {
      const int x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= gw || y >= gh || !trav[(size_t)y * gw + x]) continue;
      if (dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; best = y * gw + x; }
    }
  return best;
}
void gt_dijkstra(const std::vector<uint8_t>& trav, int gw, int gh, int src, float res, std::vector<float>& d) {
  d.assign(trav.size(), 3e38f);
  if (src < 0) return;
  typedef std::pair<float, int> E;
  std::priority_queue<E, std::vector<E>, std::greater<E>> pq;
  d[src] = 0.f;
  pq.push(E(0.f, src));
  while (!pq.empty()) {
    const E e = pq.top();
    pq.pop();
    if (e.first > d[e.second]) continue;
    const int x = e.second % gw, y = e.second / gw;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy) continue;
        const int nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
        const int k = ny * gw + nx;
        if (!trav[k]) continue;
        if (dx && dy && (!trav[y * gw + nx] || !trav[ny * gw + x])) continue;   // 모서리 사이로 안 빠짐
        const float nd = e.first + res * (dx && dy ? 1.41421356f : 1.f);
        if (nd < d[k]) { d[k] = nd; pq.push(E(nd, k)); }
      }
  }
}
float gt_near(const std::vector<float>& d, int gw, int gh, int cell, float rad) {   // 칸 둘레 rad 칸 안에서 가장 짧은 길 거리
  float best = 3e38f;
  const int cx = cell % gw, cy = cell / gw, R = (int)std::ceil(rad);
  for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
      const int x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= gw || y >= gh || (float)(dx * dx + dy * dy) > rad * rad) continue;
      best = std::min(best, d[(size_t)y * gw + x]);
    }
  return best;
}

int run(const Args& a) {
  sim::EnvHostCfg hc = sim::load_env_cfg(a.env);
  const fb::Params P = fb::load_params(a.frontier, a.env, "fast");   // frontier_run run_hier 와 같은 판 길이 상한
  hc.cfg.action_mode = sim::kActPath;
  hc.cfg.curriculum = 0;
  hc.cfg.max_steps = (int)std::ceil(P.t_safe / (hc.cfg.substeps * hc.cfg.dt)) + 100;
  hc.cfg.low_stat = hc.cfg.low_task = hc.cfg.low_rl = 1;
  hc.cfg.low_auto = 0;      // 목표는 호스트(위층 규칙)가 — Env::low_prepare
  hc.cfg.low_act_w = 0;     // 행동 줄 폭 mdp::act::DIM — 규칙 추종 환경이 후보 번호를 읽음
  if (a.stall_s > 0.0) hc.cfg.low_idle_steps = std::max(1, (int)std::lround(a.stall_s / (hc.cfg.substeps * hc.cfg.dt)));
  hc.cfg.low_explore = 1;   // 합친 탐사: 탐사 판 끝 규칙 그대로 · 정체는 막힘 신호만
  hc.cfg.end_hold = !a.video.empty() && !a.autox && a.end_hold > 0.0 ? 1 : 0;   // 녹화: 판이 끝나면 리셋 없이 멈춤 (env.h EnvCfg::end_hold). 녹화 안 하면 0 — 환경 동작 그대로
  hc.cfg.finish_declare = a.dec_finish ? 1 : 0;   // 판단 모델의 끝 선언 (env.h EnvCfg::finish_declare) — 끄면 환경 동작 그대로
  if (a.autox) hc.cfg.low_auto = 2;   // 위층 규칙을 환경이 (low_rl.h lowrl_goal_explore) — 준비 단계는 step·reset 이 끝에서 부름
  const sim::LidarModel lidar = sim::load_lidar_model(a.model);
  const sim::EnvCfg& c = hc.cfg;
  const int N = a.episodes, AW = mdp::act::DIM, PW = mdp::low::ACT, K = mdp::hier::K, DS = c.decide_steps;
  const double dt = (double)c.substeps * c.dt;
  const bool use_pol = !a.policy.empty();
  const int upper = a.upper == "slot0" ? kSlot0 : a.upper == "random" ? kRandom : a.upper == "decider" ? kDecider : a.upper == "cand_tsp" ? kTsp : -1;
  const int dec_in = dec::input_of(a.dec_input);
  const int dec_mode = a.dec_mode == "choice" ? dec::kChoice : a.dec_mode == "prune" ? dec::kPrune : a.dec_mode == "score" ? dec::kScore : -1;
  const bool all = dec_in == dec::kAll, prune = dec_mode == dec::kPrune, score = dec_mode == dec::kScore;
  const int n_thr = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  if (upper < 0) throw std::runtime_error("--upper 는 slot0 · random · decider · cand_tsp");
  if (upper != kSlot0 && a.autox) throw std::runtime_error("--auto 는 slot0 규칙만");
  if (upper == kDecider && (a.dec_url.empty() || dec_in < 0 || a.dec_prompt < 1 || a.dec_prompt > 3)) throw std::runtime_error("decider: --decider-url · --decider-input text|image|both · --decider-prompt 1~3");
  if (upper == kDecider && (dec_mode < 0 || ((prune || score) && !all) || (all && a.dec_prompt > 2) || a.dec_ask_m < 1 || a.dec_ask_m > 8)) throw std::runtime_error("decider: --decider-mode choice|prune|score (prune · score 는 --decider-input all 만) · 입력 all 의 프롬프트는 1~2 · --decider-ask-m 1~8");
  if (a.dec_finish && (upper != kDecider || !all || prune || a.autox)) throw std::runtime_error("--decider-finish 는 --upper decider · --decider-input all · --decider-mode choice|score 와 같이");
  dec::ScoreCfg sc;
  if (score || a.dec_finish) {
    sc = dec::load_score_cfg(a.dec_config);
    if (a.dec_left_hold > 0) sc.left_hold = a.dec_left_hold;
    if (!a.dec_mission.empty()) sc.mission = a.dec_mission;
    if (a.dec_score_margin >= 0.0) sc.margin = (float)a.dec_score_margin;
    if (a.dec_score_finish > -2.0) sc.finish_below = (float)a.dec_score_finish;
  }
  const int SL = (int)sc.levels.size(), LL = (int)sc.left_levels.size();
  if (!a.gt_base.empty() && a.gt_base.size() != a.scenes.size()) throw std::runtime_error("--gt-base 는 --scenes 와 같은 수");
  std::unique_ptr<dec::Client> client;
  if (upper == kDecider) client.reset(new dec::Client(a.dec_url, a.dec_timeout_ms, a.dec_parallel));
  const float bl_r = c.rec_bl_r > 0.f ? c.rec_bl_r : 0.5f;
  const int bl_steps = c.rec_bl_steps > 0 ? c.rec_bl_steps : (int)std::lround(10.0 / dt);
  std::unique_ptr<xppo::PPO> ppo;
  if (use_pol) {
    xppo::PPOConfig pc;
    pc.num_envs = N;
    ppo.reset(new xppo::PPO(pc, xnet::explore_spec_low(), 1));
    ppo->load(a.policy);
    std::printf("아래층: RL 정책 %s (반복 %d) — 결정적(평균 행동)\n", a.policy.c_str(), ppo->iteration());
  } else {
    std::printf("아래층: 규칙 추종기 (같은 걸음 루프 — 대조)\n");
  }
  if (upper == kDecider)
    std::printf("위층 고르기: 판단 모델 서버 %s · 입력 %s · 프롬프트 %d · 글자 섞기 %s · 시간 끝 %d ms · 동시 %d 개 · 사진 %d px — 판단 시간은 시뮬 시계에 안 들어감\n", a.dec_url.c_str(), a.dec_input.c_str(),
                a.dec_prompt, a.dec_shuffle ? "켬" : "끔", a.dec_timeout_ms, a.dec_parallel, a.dec_px);
  else std::printf("위층 고르기: %s\n", upper == kRandom ? "유효 후보 가운데 아무거나 (random)" : upper == kTsp ? "후보를 모두 도는 가장 짧은 차례의 첫 후보 (cand_tsp, 로봇 지도만)" : "유효 후보 가운데 번호가 가장 작은 것 (slot0)");
  std::printf("위층: 0 번 후보 규칙 · 결정 %d 걸음마다 · 막힘 신호 피하기 %s (반경 %.2f m · %.0f s) · 바로 다시 고름 %s · 목표 이어 가기 %s · 목표 붙들기 %d 걸음 | 정체 = %.0f s 동안 %.2f m · 통과 반경 %.2f m · 판 상한 %.0f s · 사람 %d\n",
              DS, a.avoid ? "켬" : "끔", bl_r, bl_steps * dt, a.redecide ? "켬" : "끔", a.stick ? "켬" : "끔", a.hold, c.low_idle_steps * dt, c.low_idle_d, c.low_pass_r, c.max_steps * dt, a.people);
  float *d_act, *d_pol;
  cudaMalloc(&d_act, sizeof(float) * AW * N);
  cudaMalloc(&d_pol, sizeof(float) * PW * N);
  for (size_t wi = 0; wi < a.scenes.size(); ++wi) {
    std::vector<sim::World> one{sim::build_world(stem(a.scenes[wi]), sim::load_scene(a.scenes[wi]), hc)};
    const sim::World& W = one[0];
    const uint64_t seed = a.eval_seed ? a.seed * 7919ull + 17 + wi : a.seed * 1000003ull + wi * 7919ull;
    sim::Env env(hc, one, lidar, N, seed, true);
    env.reset();
    env.sync();
    if (a.people > 0) {   // frontier_run 과 같은 차례 (people 0 이면 안 건드림)
      env.set_people_max(std::vector<int>(N, a.people));
      env.reset();
      env.sync();
    }
    std::vector<float> act((size_t)AW * N, 0.f), pol((size_t)PW * N, 0.f), rew;
    std::vector<sim::LowCmd> cmd(N, sim::LowCmd{-1, 0});
    std::vector<int32_t> cand, cur, done;
    std::vector<sim::EnvState> st;
    std::vector<sim::EpInfo> ep, first(N);
    std::vector<sim::LowEp> low;
    std::vector<sim::LowRl> rl;
    std::vector<Avoid> av(N);
    std::vector<int> fin(N, 0), redo(N, 1), slot(N, -1), sig_prev(N, 0), nsteps(N, 0), last_dec(N, 0), goal_prev(N, -1), sw_step(N, 0), end_since(N, -1), end_safe(N, 0);
    std::vector<float> end_v(N, 0.f), end_d(N, 0.f), end_w(N, 0.f);
    constexpr int kH = 50;   // 끝 직전 5 s 의 자리·후진 (충돌 원인 분석)
    std::vector<float> hx((size_t)N * kH, 0.f), hy((size_t)N * kH, 0.f);
    std::vector<unsigned char> hb((size_t)N * kH, 0);
    std::vector<double> dist(N, 0.0), chord(N, 0.0), px(N), py(N), cx(N), cy(N);
    // 판마다 걸음 통계 (원인 분석)
    std::vector<int> fc_step(N, -1), t_n(N, 0), t_sum(N, 0), t_cell(N, -1), t_start(N, 0), t_prev(N, -1);   // 처음 닫힌 걸음 · 60° 이상 전환 (수 · 걸음 합 · 재는 중인 목표 칸 · 시작 걸음 · 직전 목표 칸)
    std::vector<int> n_dec(N, 0), n_sw(N, 0), n_stall(N, 0), n_reach(N, 0), n_off(N, 0), n_nopath(N, 0), n_safe(N, 0), n_back(N, 0), n_slow(N, 0), n_nogoal(N, 0), n_avoided(N, 0), n_early(N, 0), n_pathless(N, 0);
    std::vector<double> tgt_d(N, 0.0), w_abs(N, 0.0);
    std::vector<int> n_rule(N, 0);
    long dbin[4] = {0, 0, 0, 0};   // 결정 때 목표까지 직선 거리: < 1 · 1 ~ 3 · 3 ~ 15 · ≥ 15 m
    env.get_state(st);
    for (int i = 0; i < N; ++i) { px[i] = cx[i] = st[i].x; py[i] = cy[i] = st[i].y; }
    const auto avoided = [&](int i, int cell, int step) {
      if (!a.avoid || cell < 0) return false;
      const float q = bl_r / c.res;
      for (int k = 0; k < kBl; ++k) {
        if (av[i].cell[k] < 0 || step >= av[i].until[k]) continue;
        const int dx = cell % W.gw - av[i].cell[k] % W.gw, dy = cell / W.gw - av[i].cell[k] / W.gw;
        if ((float)(dx * dx + dy * dy) <= q * q) return true;
      }
      return false;
    };
    const auto avoid_add = [&](int i, int cell, int step) {
      if (!a.avoid || cell < 0) return;
      av[i].cell[av[i].n % kBl] = cell;
      av[i].until[av[i].n % kBl] = step + bl_steps;
      av[i].n += 1;
    };
    // 위층 고르기 (--upper): 걸음마다 결정할 환경의 유효 후보를 모으고(need · vslot · nv), 방식대로 고른 후보 번호를 pick 에 둔다
    std::vector<int> need(N, 0), nv(N, 0), vslot((size_t)N * K, 0), pick(N, -1), ask, askg, n_asked(N, 0);
    std::vector<uint32_t> veto(N, 0), gtb(N, 0);   // 유효 후보 차례 비트: 모델이 막혔다 함 · 정답으로 막힘
    std::vector<int> asked(N, 0);                  // prune: 이번 결정에서 물은 후보 수
    // 끝 선언 (--decider-finish): askq = 이번 걸음에 모델에 묻는 환경 (끄면 ask 와 같음 — 유효 후보 2 개 이상, 켜면 결정하는 모든 환경)
    std::vector<int> askq, nskip(N, 0), decl(N, 0), end_by(N, 0), d_block(N, -1), d_clus(N, -1), d_closed(N, 0), n_finq(N, 0);
    std::vector<float> d_idle(N, -1.f), d_p(N, -1.f);
    std::vector<int8_t> ogrid;
    std::vector<int16_t> lmap;
    std::vector<int16_t> r_map; std::vector<uint32_t> r_trail; std::vector<uint8_t> r_cam; std::vector<float> r_pts; cr::Work r_work;   // cand_tsp 만
    long u_finq = 0, u_finyes = 0;
    double u_pfin = 0.0;
    // 점수 방식 기록: 결정마다 후보마다 단계 확률·기대 점수(<out>_<장면>_score.tsv), 고른 후보로 가는 동안 로봇 지도가 실제로 는 칸 수(<out>_<장면>_gain.tsv — 뒤에 학습 정답으로 쓸 값, 로봇 지도만)
    struct Seg { bool on = false; int dec = 0, step0 = 0, cell = -1, slot = -1; char key = '?'; float path = 0, e = 0; std::vector<float> p; long known0 = 0; };
    std::vector<Seg> seg(N), sel(N);
    std::vector<long> kn(N, 0);
    std::vector<int> kn_step(N, -1), seg_why(N, 0);   // seg_why: 1 = 도착, 2 = 포기(막힘 신호 · 경로 없음 · 경로 벗어남)
    std::FILE *fsc = nullptr, *fgn = nullptr;
    long u_sq = 0, u_tie = 0, u_seg = 0, u_fin_nocand = 0, u_lvl[10] = {0}, u_llvl[10] = {0};
    std::vector<std::vector<int>> h_try(N), h_ab(N);   // 이 판에서 목표로 삼은 칸 · 가다가 포기한 칸 (전부 — 피하는 칸 고리 Avoid 는 8 개 · 10 s 뒤 지워져 횟수를 못 셈)
    std::vector<float> vlv;                           // 영상: "남은 탐사" 단계 확률
    std::vector<int> fstreak(N, 0);                 // choice 끝 선언: "남은 게 없음" 답이 이어진 결정 수
    std::vector<std::vector<std::string>> lrow(N);   // <out>_<장면>_left.tsv 줄 (판 끝에 "그때부터 판 끝까지 는 칸 수"를 붙여 씀)
    std::vector<std::vector<long>> lkn(N);
    std::FILE* flf = nullptr;
    if (a.dec_finish && !score && !a.out.empty()) {
      flf = std::fopen((a.out + "_" + W.name + "_left.tsv").c_str(), "w");
      if (!flf) throw std::runtime_error("--out 끝 선언 기록 파일을 못 엶");
      std::fputs("env\tdec\tstep\tt_s\tn_cand\tpicked", flf);
      for (int k = 0; k < K; ++k) std::fprintf(flf, "\tp_%c", 'A' + k);
      for (int l = 0; l < LL; ++l) std::fprintf(flf, "\tleft_p%d", l);
      std::fputs("\tleft_expected\tnothing_left\tstreak\tdeclared\tmap_closed\tknown_now\tknown_end\tcells_gained_until_end\n", flf);
    }
    double u_sexp = 0.0, u_sbest = 0.0;
    if (score && !a.out.empty()) {
      fsc = std::fopen((a.out + "_" + W.name + "_score.tsv").c_str(), "w");
      fgn = std::fopen((a.out + "_" + W.name + "_gain.tsv").c_str(), "w");
      if (!fsc || !fgn) throw std::runtime_error("--out 점수 기록 파일을 못 엶");
      std::fputs("env\tdec\tstep\tt_s\tn_cand\tkey\tslot\tcell_x\tcell_y\tpath_m\tis_cur", fsc);
      std::fputs("env\tdec\tstep_start\tstep_end\telapsed_s\tkey\tslot\tcell_x\tcell_y\tpath_m", fgn);
      for (int l = 0; l < SL; ++l) { std::fprintf(fsc, "\tp%d", l); std::fprintf(fgn, "\tp%d", l); }
      std::fputs("\texpected\tchosen\tfinish\n", fsc);
      std::fputs("\texpected\tknown_start\tknown_end\tcells_gained\tend\n", fgn);
    }
    const auto known = [&](int i, int step) {   // 로봇 지도에서 아는 칸 수 (모름이 아닌 칸) — 로봇 지도만 씀
      if (kn_step[i] == step) return kn[i];
      env.get_map_env(i, lmap);
      long n = 0;
      for (int y = 0; y < W.gh; ++y)
        for (int x = 0; x < W.gw; ++x) n += lmap[(size_t)y * c.size + x] != sim::kUnknown;
      kn[i] = n;
      kn_step[i] = step;
      return n;
    };
    const auto seg_close = [&](int i, int step, long kn_now, const char* why) {
      Seg& g = seg[i];
      if (!g.on) return;
      g.on = false;
      u_seg += 1;
      if (!fgn) return;
      std::fprintf(fgn, "%d\t%d\t%d\t%d\t%.1f\t%c\t%d\t%d\t%d\t%.1f", i, g.dec, g.step0, step, (step - g.step0) * dt, g.key, g.slot, g.cell % W.gw, g.cell / W.gw, g.path);
      for (float p : g.p) std::fprintf(fgn, "\t%.4f", p);
      std::fprintf(fgn, "\t%.3f\t%ld\t%ld\t%ld\t%s\n", g.e, g.known0, kn_now, kn_now - g.known0, why);
    };
    const auto declare = [&](int i, float p) {   // 끝 선언: 이 걸음에 선언 (환경이 로봇 지도 닫힘으로 판정 — env.h finish_step). 채점 값은 전부 로봇 지도에서
      decl[i] = 1;
      u_finyes += 1;
      d_p[i] = p;
      d_idle[i] = (float)(st[i].since_occ * dt);
      d_closed[i] = st[i].map_closed;
      // 채점: 선언한 걸음의 로봇 지도를 ROS 점유 격자 꼴로 바꿔 격자만으로 남은 프론티어를 셈 (decider.h grid_open_frontier — 정답 지도 안 씀)
      env.get_map_env(i, lmap);
      ogrid.assign((size_t)W.gw * W.gh, (int8_t)-1);
      for (int y = 0; y < W.gh; ++y)
        for (int x = 0; x < W.gw; ++x) {
          const int16_t l = lmap[(size_t)y * c.size + x];
          if (l != sim::kUnknown) ogrid[(size_t)y * W.gw + x] = l >= mdp::kOccL ? 100 : 0;
        }
      const dec::GridOpen go = dec::grid_open_frontier(ogrid.data(), W.gw, W.gh, (int)((st[i].x + st[i].ex - W.ox) / c.res), (int)((st[i].y + st[i].ey - W.oy) / c.res), c.close_radius / c.res, c.close_min);
      d_block[i] = go.start >= 0 ? go.n_block : -1;
      d_clus[i] = go.n_clusters;
    };
    long u_q = 0, u_veto = 0, u_allveto = 0, g_n = 0, g_first = 0, g_pick = 0, g_cand = 0, g_cand_blk = 0, g_vt_blk = 0, g_vt_clear = 0, g_miss = 0, g_ok_clear = 0;
    double u_pblk = 0.0;
    std::vector<uint8_t> trav, trav0;
    if (a.gt) {
      trav = gt_trav(W, c.half_wid);
      trav0 = trav;
      if (!a.gt_base.empty()) {
        const sim::World B = sim::build_world(stem(a.gt_base[wi]), sim::load_scene(a.gt_base[wi]), hc);
        if (B.gw != W.gw || B.gh != W.gh || B.ox != W.ox || B.oy != W.oy) throw std::runtime_error("--gt-base 장면의 격자가 다름");
        trav0 = gt_trav(B, c.half_wid);
      }
    }
    dec::CamSpec cam;   // 실물 카메라 내부값(sensor_model.yaml depth) · 달린 자리(env.yaml body) 를 줄인 그림에 맞춤
    {
      const sim::DepthSpec& Dp = c.depth.spec;
      const float q = (float)a.dec_cam_div;
      cam.w = Dp.w / a.dec_cam_div; cam.h = a.dec_cam_h / a.dec_cam_div;
      cam.fx = Dp.fx / q; cam.fy = Dp.fy / q; cam.cx = (Dp.cx + 0.5f) / q - 0.5f; cam.cy = (Dp.cy + 0.5f) / q - 0.5f;
      cam.mx = c.cam_x; cam.my = c.cam_y; cam.mz = c.cam_z; cam.cp = c.cam_cp; cam.sp = c.cam_sp;
    }
    const sim::SceneView sv = W.scene.view();
    const dec::RayFn ray = [&sv](const float o_[3], const float d_[3], float* t_, float n_[3], int* obj_) {
      uint16_t ho = 0;
      int tri = -1;
      const float r = sim::raycast(sv, sim::V3{o_[0], o_[1], o_[2]}, sim::V3{d_[0], d_[1], d_[2]}, 1000.f, &ho, &tri);
      if (tri < 0 || !(r < 1000.f)) return false;
      const sim::Tri& T = sv.tris[tri];
      const float e1[3] = {T.b.x - T.a.x, T.b.y - T.a.y, T.b.z - T.a.z}, e2[3] = {T.c.x - T.a.x, T.c.y - T.a.y, T.c.z - T.a.z};
      float nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
      const float l = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (l > 0.f) { nx /= l; ny /= l; nz /= l; }
      n_[0] = nx; n_[1] = ny; n_[2] = nz;
      *t_ = r;
      *obj_ = (int)ho;
      return true;
    };
    std::vector<float> o1d;
    std::vector<double> lat;   // 요청 지연 (ms, 성공한 것)
    long u_ask = 0, u_fall = 0, u_fall_curl = 0, u_fall_http = 0, u_fall_shape = 0, u_first = 0, u_mismatch = 0, u_batches = 0;
    long u_rank[mdp::hier::K] = {0}, u_slot[mdp::hier::K] = {0}, u_nv = 0;
    double u_wall = 0.0, u_conf = 0.0, u_p = 0.0;
    int n_dump = 0;
    std::string first_err;
    int left = N;
    std::vector<sim::LowCmd> ecmd;
    // 녹화 (--video): 환경 ve 의 판을 걸음마다 한 장. 카메라 칸은 걸음마다 지금 자세에서 새로 그리고, 모델에 실제로 보낸 그림은 작은 그림으로 따로 둠. 결정 때 후보·모델 답을 잡아 다음 결정까지 보여 줌. 셈·고르기에는 안 씀
    const int ve = a.video_env;
    bool vrec = false, vcap = false, vok = false;
    std::unique_ptr<dec::Video> vid;
    std::vector<dec::VidCand> vcand;
    std::vector<uint8_t> vcam, vsent, vlow;   // vcam = 지금 자세 그림 (걸음마다), vsent = 모델에 마지막으로 보낸 그림 그대로
    std::vector<float> vtrail, vprob, vdist;
    std::vector<int> vslots;
    std::vector<int16_t> vmap;
    std::string vdec = "no decision yet", vlast, verr;
    int vnq = 0, vflash = 0, vhold = -1;   // vhold: 판 끝 뒤 남은 멈춤 걸음 (−1 = 판 진행 중)
    bool vend_coll = false;
    std::string vend;
    double vsent_t = -1.0;
    float vfin_p = -2.f;   // 끝 선언 확률 (−2 = 안 물음, −1 = 서버가 확률을 안 줌)
    bool vfin = false;
    double vms = 0.0;
    const char* const kG = "{\\c&H3CD228&}";   // ASS 색: 초록 · 자홍 · 주황 · 흰색
    const char* const kM = "{\\c&HDC28FF&}";
    const char* const kO = "{\\c&H1E8CFF&}";
    const char* const kW = "{\\c&HFFFFFF&}";
    if (!a.video.empty() && !a.autox) {
      if (ve < 0 || ve >= N) throw std::runtime_error("--video-env 는 0 … 판 수 − 1");
      vid.reset(new dec::Video(a.video + "_" + W.name, 1.0 / dt, "Lidar map (white=free black=wall grey=unknown)  " + W.name,
                               upper == kDecider && all ? "Live camera (synthetic, not real RGB); inset = sent to model" : "Live camera (synthetic render); nothing sent to a model"));
      vrec = true;
      if (!a.gt_base.empty()) {
        vlow.resize(trav.size());
        for (size_t k = 0; k < trav.size(); ++k) vlow[k] = trav0[k] && !trav[k];
      }
    }
    for (long t = 0; (left > 0 && t < (long)c.max_steps + 10 * DS) || (vrec && vhold >= 0); ++t) {   // 뒤 조건: 녹화의 판 끝 뒤 멈춤 장면 (vhold 걸음 뒤 vrec 가 꺼짐)
      env.get_cand(cand);
      env.get_cand_cur(cur);
      if (a.autox) env.get_low_cmd(ecmd);
      ask.clear();
      askg.clear();
      askq.clear();
      for (int i = 0; i < N && !a.autox; ++i) {   // 결정할 환경과 유효 후보
        need[i] = 0;
        if (fin[i]) continue;
        const sim::EnvState& s = st[i];
        if (!(s.step % DS == 0 || redo[i])) continue;
        need[i] = 1;
        nv[i] = 0;
        nskip[i] = 0;
        decl[i] = 0;
        for (int j = 0; j < K; ++j) {
          const int cell = cand[(size_t)i * K + j];
          if (cell < 0) continue;
          if (avoided(i, cell, s.step)) { if (nv[i] == 0) n_avoided[i] += 1; nskip[i] += 1; continue; }   // 셈은 첫 유효 후보 앞의 것만 (slot0 규칙이 건너뛴 수)
          vslot[(size_t)i * K + nv[i]++] = j;
        }
        pick[i] = nv[i] > 0 ? vslot[(size_t)i * K] : -1;
        if (nv[i] >= 2) askg.push_back(i);
        if (upper != kSlot0 && nv[i] >= 2) ask.push_back(i);
        if (upper == kDecider && (nv[i] >= 2 || (a.dec_finish && !score) || (score && nv[i] >= 1))) askq.push_back(i);   // choice 끝 선언은 후보가 0 · 1 개여도 "남은 탐사"를 물음
        sel[i].on = false;
        if (score && a.dec_finish && nv[i] == 0 && nskip[i] == 0 && !h_try[i].empty()) {   // 이 판에서 목표를 한 번이라도 잡은 뒤에만 (판 첫 걸음에는 후보가 아직 없음)
          u_fin_nocand += 1; declare(i, -1.f); }   // 점수 방식의 끝: 후보가 하나도 없음 (피하는 칸이라 뺀 후보만 남은 때는 선언 안 함)
      }
      if (upper == kTsp && !ask.empty()) {   // cand_tsp (scripts/cand_rules.h): 규칙이 고른 후보가 피하는 칸이면 slot0 차례 그대로
        env.get_map(r_map); env.get_trail(r_trail); env.get_camlayer(r_cam, r_pts);
        const size_t tws = (size_t)(c.size + 31) / 32, per_trail = 2 * c.size * tws, per_map = (size_t)c.size * c.size;
        for (int i : ask) {
          const cr::In in{&c, &W, nullptr, &r_map[i * per_map], &r_trail[i * per_trail], r_cam.empty() ? nullptr : &r_cam[i * per_map], &st[i], &cand[(size_t)i * K]};
          const int k = cr::choose(cr::kTsp, in, a.rule, r_work);
          for (int q = 0; q < nv[i]; ++q) if (vslot[(size_t)i * K + q] == k) pick[i] = k;
        }
      } else if (upper == kRandom) {
        for (int i : ask) pick[i] = vslot[(size_t)i * K + mix(seed * 0x9e3779b97f4a7c15ull + (uint64_t)i * 1000003ull + (uint64_t)n_dec[i]) % (uint64_t)nv[i]];
      } else if (upper == kDecider && !askq.empty()) {
        const sim::StepOut o = env.outputs();
        o1d.resize((size_t)N * o.o1d);
        cudaMemcpy(o1d.data(), o.obs_1d, sizeof(float) * o1d.size(), cudaMemcpyDeviceToHost);
        std::vector<std::string> bodies(askq.size());
        std::vector<std::vector<dec::Cand>> cs(askq.size());
        std::vector<std::vector<uint8_t>> pngs(askq.size()), cams(askq.size());
        std::vector<std::vector<float>> wxy(askq.size());
        std::vector<int> dump(askq.size(), 0), nq(askq.size(), 0);
        std::vector<dec::FinishInfo> fi(askq.size());
        for (size_t q = 0; q < askq.size(); ++q) {
          const int i = askq[q];
          const sim::EnvState& s = st[i];
          const float* F = o1d.data() + (size_t)i * o.o1d + mdp::obs::POLICY_1D;
          std::vector<int> cells;
          uint64_t r = mix(seed * 31ull + (uint64_t)i * 7919ull + (uint64_t)n_dec[i] * 104729ull);
          int perm[mdp::hier::K];
          for (int v = 0; v < nv[i]; ++v) perm[v] = v;
          if (a.dec_shuffle && !prune)
            for (int v = nv[i] - 1; v > 0; --v) { r = mix(r); std::swap(perm[v], perm[r % (uint64_t)(v + 1)]); }
          for (int v = 0; v < nv[i]; ++v) {   // 글자 차례로: 글자 v 는 유효 후보 perm[v]
            const int j = vslot[(size_t)i * K + perm[v]];
            const float* f = F + j * mdp::hier::FEAT;
            if (f[mdp::hier::F_VALID] < 0.5f) u_mismatch += 1;   // 후보 특징과 후보 칸이 어긋남 (없어야 함)
            dec::Cand cd;
            cd.key = (char)('A' + v);
            cd.slot = j;
            cd.path_m = f[mdp::hier::F_DIST] * mdp::obsgen::kFrontNorm;
            cd.bearing = std::atan2(f[mdp::hier::F_SIN], f[mdp::hier::F_COS]);
            cd.size_m = f[mdp::hier::F_SIZE] * mdp::obsgen::kFrontSizeNorm;
            cd.util = c.cand_v2 ? f[mdp::hier::F_UTIL] : f[mdp::hier::F_UTIL] * mdp::hier::kUtilNorm;
            cd.cur = f[mdp::hier::F_CUR] > 0.5f;
            const int cell = cand[(size_t)i * K + j];
            if (a.dec_finish && !score) {   // 후보마다 시도 · 포기 횟수 (후보 칸에서 피하는 반경 안의 기록)
              const float qr = bl_r / c.res;
              const auto near = [&](int h) { const int dx = cell % W.gw - h % W.gw, dy = cell / W.gw - h / W.gw; return (float)(dx * dx + dy * dy) <= qr * qr; };
              cd.tried = cd.abandoned = 0;
              for (int h : h_try[i]) cd.tried += near(h);
              for (int h : h_ab[i]) cd.abandoned += near(h);
            }
            cs[q].push_back(cd);
            cells.push_back(cell % W.gw);
            cells.push_back(cell / W.gw);
            wxy[q].push_back(W.ox + ((float)(cell % W.gw) + 0.5f) * c.res);
            wxy[q].push_back(W.oy + ((float)(cell / W.gw) + 0.5f) * c.res);
          }
          if (dec_in != dec::kText) {
            env.get_map_env(i, lmap);
            int kb[4] = {s.kb_x0, s.kb_y0, s.kb_x1, s.kb_y1};
            pngs[q] = dec::render_png(lmap.data(), c.size, W.gw, W.gh, sim::kUnknown, mdp::kOccL, kb, (s.x + s.ex - W.ox) / c.res, (s.y + s.ey - W.oy) / c.res, s.th + s.eth, cs[q], cells, a.dec_px);
          }
          nq[q] = prune && nv[i] >= 2 ? std::min(nv[i], a.dec_ask_m) : 0;   // 후보 하나 이하는 막힘을 안 물음 (끝 선언만 물을 때)
          if (a.dec_finish) {
            fi[q].n_valid = nv[i];
            fi[q].n_skipped = nskip[i];
            fi[q].closed = s.map_closed != 0;
            fi[q].idle_s = (float)(s.since_occ * dt);
            fi[q].n_abandoned = av[i].n;
            fi[q].t_s = (float)(s.step * dt);
            fi[q].cfg = &sc;
          }
        }
        if (all) {   // 카메라 그림: 참 자세에서 장면에 광선 (CPU, 요청마다 따로 — 스레드로 나눔)
          std::atomic<size_t> nxt{0};
          const auto work = [&]() {
            for (size_t q; (q = nxt.fetch_add(1)) < askq.size();) cams[q] = dec::render_cam_png(cam, st[askq[q]].x, st[askq[q]].y, st[askq[q]].th, ray, cs[q], wxy[q], vrec && askq[q] == ve ? &vsent : nullptr);
          };
          std::vector<std::thread> th;
          for (int k = 0; k < n_thr; ++k) th.emplace_back(work);
          for (auto& t_ : th) t_.join();
        }
        for (size_t q = 0; q < askq.size(); ++q) {
          const int i = askq[q];
          bodies[q] = score ? dec::build_body_score(cs[q], cams[q], pngs[q], c.cand_v2 != 0, sc) : all ? dec::build_body_all(dec_mode, a.dec_prompt, cs[q], cams[q], pngs[q], c.cand_v2 != 0, nq[q], a.dec_finish ? &fi[q] : nullptr) : dec::build_body(dec_in, a.dec_prompt, cs[q], pngs[q], c.cand_v2 != 0);
          dump[q] = !a.dec_dump.empty() && n_dump < a.dec_dump_n && i < 3 && n_dec[i] % 40 == 5;
          n_dump += dump[q];
        }
        const auto w0 = std::chrono::steady_clock::now();
        const std::vector<dec::Reply> rep = score ? client->ask(bodies, cs, {}, false, false, SL) : prune ? client->ask(bodies, cs, nq, a.dec_prompt == 2, a.dec_finish) : client->ask(bodies, cs, {}, false, a.dec_finish, a.dec_finish ? LL : 0);
        u_wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        u_batches += 1;
        for (size_t q = 0; q < askq.size(); ++q) {
          const int i = askq[q];
          const dec::Reply& r = rep[q];
          if (r.ok && score) {   // 기대 점수가 가장 높은 후보. 최고에서 margin 안이면 비긴 것: 지금 목표 먼저, 다음은 가까운 후보
            lat.push_back(r.ms);
            const int nc = (int)cs[q].size();
            float best = -1.f;
            for (int k = 0; k < nc; ++k) best = std::max(best, r.exp[k]);
            int pk = -1, ntie = 0;
            for (int k = 0; k < nc; ++k) {
              if (r.exp[k] < best - sc.margin) continue;
              ntie += 1;
              if (pk < 0 || (cs[q][k].cur && !cs[q][pk].cur) || (cs[q][k].cur == cs[q][pk].cur && cs[q][k].path_m < cs[q][pk].path_m)) pk = k;
            }
            pick[i] = cs[q][pk].slot;
            u_tie += ntie > 1;
            u_sbest += best;
            bool fz = true;   // 끝: 모든 후보에서 얻을 게 없음
            for (int k = 0; k < nc; ++k) {
              int am = 0;
              for (int l = 1; l < SL; ++l)
                if (r.lv[(size_t)k * SL + l] > r.lv[(size_t)k * SL + am]) am = l;
              u_lvl[am] += 1;
              u_sq += 1;
              u_sexp += r.exp[k];
              fz = fz && (sc.finish_below >= 0.f ? r.exp[k] < sc.finish_below : am == 0);
            }
            if (a.dec_finish) {
              u_finq += 1;
              n_finq[i] += 1;
              if (fz) declare(i, best);
            }
            Seg& g = sel[i];
            g.on = true; g.dec = n_dec[i]; g.step0 = st[i].step; g.slot = cs[q][pk].slot; g.cell = cand[(size_t)i * K + g.slot]; g.key = cs[q][pk].key; g.path = cs[q][pk].path_m; g.e = r.exp[pk];
            g.p.assign(r.lv.begin() + (size_t)pk * SL, r.lv.begin() + (size_t)(pk + 1) * SL);
            if (fsc)
              for (int k = 0; k < nc; ++k) {
                const int cell = cand[(size_t)i * K + cs[q][k].slot];
                std::fprintf(fsc, "%d\t%d\t%d\t%.1f\t%d\t%c\t%d\t%d\t%d\t%.1f\t%d", i, n_dec[i], st[i].step, st[i].step * dt, nc, cs[q][k].key, cs[q][k].slot, cell % W.gw, cell / W.gw, cs[q][k].path_m, (int)cs[q][k].cur);
                for (int l = 0; l < SL; ++l) std::fprintf(fsc, "\t%.4f", r.lv[(size_t)k * SL + l]);
                std::fprintf(fsc, "\t%.3f\t%d\t%d\n", r.exp[k], k == pk, (int)(a.dec_finish && fz));
              }
          } else if (r.ok && prune && nq[q] == 0) {
   // 끝 선언만 물은 결정 (후보 하나 이하): 고르기는 규칙 그대로
            lat.push_back(r.ms);
          } else if (r.ok && prune) {   // 규칙 차례 그대로, 막혔다는 후보만 건너뜀
            lat.push_back(r.ms);
            int k = 0;
            while (k < nq[q] && r.blocked[k] >= (float)a.dec_veto_thr) ++k;
            for (int v = 0; v < nq[q]; ++v) {
              u_q += 1;
              u_pblk += r.blocked[v];
              if (r.blocked[v] >= (float)a.dec_veto_thr) { veto[i] |= 1u << v; u_veto += 1; }
            }
            asked[i] = nq[q];
            if (k >= nq[q]) { u_allveto += 1; k = nv[i] > nq[q] ? nq[q] : 0; }   // 물은 후보가 다 막혔다: 안 물은 다음 후보, 없으면 규칙의 첫 후보
            pick[i] = vslot[(size_t)i * K + k];
          } else if (r.ok) {
            if (r.choice >= 0) pick[i] = cs[q][r.choice].slot;   // FINISH·CONTINUE 를 골랐으면 (끝 선언) 고르기는 규칙 그대로
            lat.push_back(r.ms);
            u_conf += r.conf;
            u_p += r.p;
          } else {   // 0 번(첫 유효 후보)으로 떨어짐
            u_fall += 1;
            if (r.err.rfind("curl", 0) == 0) u_fall_curl += 1;
            else if (r.err.rfind("http", 0) == 0) u_fall_http += 1;
            else u_fall_shape += 1;
            if (first_err.empty()) first_err = r.err + " | " + r.raw.substr(0, 200);
          }
          if (r.ok && a.dec_finish && !score) {   // 끝 선언 (choice 의 FINISH 선택지): 모델이 고르면 이 걸음에 선언
            u_finq += 1;
            n_finq[i] += 1;
            int am = 0;
            for (int l = 1; l < LL; ++l)
              if (r.lv[l] > r.lv[am]) am = l;
            const float e = r.exp[0];
            const bool none = sc.left_finish_below >= 0.f ? e < sc.left_finish_below : am == 0;   // "남은 게 없음"
            u_pfin += e;
            u_llvl[am] += 1;
            fstreak[i] = none ? fstreak[i] + 1 : 0;
            if (fstreak[i] >= sc.left_hold) declare(i, e);
            if (flf) {   // 결정 기록 (그때부터 판 끝까지 는 칸 수는 판 끝에 붙임)
              char b[512];
              int n = std::snprintf(b, sizeof b, "%d\t%d\t%d\t%.1f\t%d\t%c", i, n_dec[i], st[i].step, st[i].step * dt, (int)cs[q].size(), r.choice >= 0 ? cs[q][r.choice].key : '-');
              for (int k = 0; k < K; ++k) n += std::snprintf(b + n, sizeof b - n, "\t%.4f", k < (int)r.probs.size() ? r.probs[k] : -1.f);
              for (int l = 0; l < LL; ++l) n += std::snprintf(b + n, sizeof b - n, "\t%.4f", r.lv[l]);
              const long kn_now = known(i, st[i].step);
              std::snprintf(b + n, sizeof b - n, "\t%.3f\t%d\t%d\t%d\t%d\t%ld", e, (int)none, fstreak[i], decl[i], st[i].map_closed, kn_now);
              lrow[i].push_back(b);
              lkn[i].push_back(kn_now);
            }
          }
          if (vrec && i == ve) {   // 녹화: 보낸 후보(글자 그대로)와 답
            vcap = true;
            vok = r.ok;
            vms = r.ms;
            verr = r.err;
            vnq = nq[q];
            vfin_p = a.dec_finish && r.ok && !score ? r.exp[0] : -2.f;
            vfin = decl[i] != 0;
            vlv = a.dec_finish && r.ok && !score ? r.lv : std::vector<float>();
            vcand.clear(); vprob.clear(); vdist.clear(); vslots.clear();
            for (size_t k = 0; k < cs[q].size(); ++k) {
              const int cell = cand[(size_t)i * K + cs[q][k].slot];
              dec::VidCand vc;
              vc.key = cs[q][k].key;
              vc.cx = cell % W.gw;
              vc.cy = cell / W.gw;
              const float pb = !r.ok ? -1.f : score ? r.exp[k] : prune ? ((int)k < nq[q] ? r.blocked[k] : -1.f) : (k < r.probs.size() ? r.probs[k] : -1.f);
              vc.veto = r.ok && prune && (int)k < nq[q] && pb >= (float)a.dec_veto_thr;
              vcand.push_back(vc);
              vprob.push_back(pb);
              vdist.push_back(cs[q][k].path_m);
              vslots.push_back(cs[q][k].slot);
            }
          }
          if (dump[q]) {
            char nm[256];
            std::snprintf(nm, sizeof nm, "%s/%s_env%d_dec%d", a.dec_dump.c_str(), W.name.c_str(), i, n_dec[i]);
            dec::write_file(std::string(nm) + ".request.json", bodies[q].data(), bodies[q].size());
            if (!pngs[q].empty()) dec::write_file(std::string(nm) + (all ? ".map.png" : ".png"), pngs[q].data(), pngs[q].size());
            if (!cams[q].empty()) dec::write_file(std::string(nm) + ".cam.png", cams[q].data(), cams[q].size());
            const std::string ans = r.ok ? r.raw : "FALLBACK: " + r.err + "\n" + r.raw;
            dec::write_file(std::string(nm) + ".reply.json", ans.data(), ans.size());
          }
        }
      }
      if (a.gt) {   // 정답 격자로 본 "막힌 후보" (셈만 — 고르기에는 안 씀): 낮은 장애물 없는 바탕 장면보다 참 길이 훨씬 길거나 못 감
        std::atomic<size_t> nxt{0};
        const auto work = [&]() {
          std::vector<float> d0, d1;
          for (size_t q; (q = nxt.fetch_add(1)) < askg.size();) {
            const int i = askg[q];
            const int rc = gt_cell(trav, W.gw, W.gh, (int)((st[i].x - W.ox) / c.res), (int)((st[i].y - W.oy) / c.res)), rb = gt_cell(trav0, W.gw, W.gh, (int)((st[i].x - W.ox) / c.res), (int)((st[i].y - W.oy) / c.res));
            gt_dijkstra(trav, W.gw, W.gh, rc, c.res, d1);
            gt_dijkstra(trav0, W.gw, W.gh, rb, c.res, d0);
            gtb[i] = 0;
            for (int v = 0; v < nv[i]; ++v) {
              const int cell = cand[(size_t)i * K + vslot[(size_t)i * K + v]];
              const float t1 = gt_near(d1, W.gw, W.gh, cell, a.gt_near / c.res), t0 = gt_near(d0, W.gw, W.gh, cell, a.gt_near / c.res);
              if (t0 < 1e30f && (t1 >= 1e30f || (t1 > t0 + (float)a.gt_add && t1 > t0 * (float)a.gt_mul))) gtb[i] |= 1u << v;
            }
          }
        };
        std::vector<std::thread> th;
        for (int k = 0; k < n_thr; ++k) th.emplace_back(work);
        for (auto& t_ : th) t_.join();
        for (int i : askg) {
          int rank = 0;
          while (rank < nv[i] && vslot[(size_t)i * K + rank] != pick[i]) ++rank;
          g_n += 1;
          g_first += gtb[i] & 1u;
          g_pick += (gtb[i] >> rank) & 1u;
          for (int v = 0; v < nv[i]; ++v) { g_cand += 1; g_cand_blk += (gtb[i] >> v) & 1u; }
          for (int v = 0; v < asked[i]; ++v) {
            const bool vt = (veto[i] >> v) & 1u, gb = (gtb[i] >> v) & 1u;
            g_vt_blk += vt && gb;
            g_vt_clear += vt && !gb;
            g_miss += !vt && gb;
            g_ok_clear += !vt && !gb;
          }
        }
      }
      if (vrec && need[ve] && !fin[ve]) {   // 녹화: 이번 결정을 글과 표시로
        const int i = ve;
        const bool sent = vcap;
        if (sent && upper == kDecider && all) { vsent_t = st[i].step * dt; vflash = 5; }   // 보낸 그림 표시: 0.5 s 동안 초록 테두리
        if (!vcap) {   // 모델에 안 물은 결정 (규칙 · 후보 1 개 이하): 유효 후보 차례로
          vcand.clear(); vprob.clear(); vdist.clear(); vslots.clear();
          std::vector<dec::Cand> cs1;
          std::vector<float> w1;
          for (int v = 0; v < nv[i]; ++v) {
            const int j = vslot[(size_t)i * K + v], cell = cand[(size_t)i * K + j];
            dec::VidCand vc;
            vc.key = (char)('A' + v);
            vc.cx = cell % W.gw;
            vc.cy = cell / W.gw;
            vcand.push_back(vc);
            vprob.push_back(-1.f);
            vdist.push_back(-1.f);
            vslots.push_back(j);
            dec::Cand cd;
            cd.key = vc.key;
            cd.slot = j;
            cs1.push_back(cd);
            w1.push_back(W.ox + ((float)vc.cx + 0.5f) * c.res);
            w1.push_back(W.oy + ((float)vc.cy + 0.5f) * c.res);
          }
        }
        char b[256];
        std::snprintf(b, sizeof b, "decision #%d at t=%.1f s:\n  ", n_dec[i] + 1, st[i].step * dt);
        vdec = b;
        if (sent && vok) { std::snprintf(b, sizeof b, "decision model (%s), %.0f ms", score ? "score: expected map gain 0-4" : prune ? "prune" : "choice", vms); vdec += b; }
        else if (sent) vdec += std::string(kM) + "model FAILED (" + verr.substr(0, 24) + ") -> rule slot0" + kW;
        else if (upper == kDecider) vdec += "fewer than 2 candidates, model not asked";
        else vdec += upper == kRandom ? "random pick (no model)" : "rule slot0 = first valid candidate (no model)";
        for (size_t k = 0; k < vcand.size(); ++k) {
          int v = 0;
          while (v < nv[i] && vslot[(size_t)i * K + v] != vslots[k]) ++v;
          vcand[k].chosen = vslots[k] == pick[i];
          vcand[k].gt_blk = a.gt && nv[i] >= 2 && v < nv[i] && ((gtb[i] >> v) & 1u);
          vdec += std::string("\n ") + (vcand[k].chosen ? kG : vcand[k].veto ? kM : kW) + vcand[k].key;
          if (sent && vok && prune) {
            if ((int)k < vnq) { std::snprintf(b, sizeof b, "  P(blocked)=%.2f %s", vprob[k], vcand[k].veto ? "VETO" : "ok  "); vdec += b; }
            else vdec += "  (not asked)          ";
          } else if (sent && vok && vprob[k] >= 0.f) { std::snprintf(b, sizeof b, score ? "  gain=%.2f" : "  p=%.2f", vprob[k]); vdec += b; }
          if (vdist[k] >= 0.f) { std::snprintf(b, sizeof b, "  path %4.1f m", vdist[k]); vdec += b; }
          if (vcand[k].chosen) vdec += "  <- CHOSEN";
          vdec += kW;
          if (vcand[k].gt_blk) vdec += std::string("  ") + kO + "[truth: blocked]" + kW;
        }
        if (vcand.empty()) vdec += "\n (no candidate: turn in place)";
        if (a.dec_finish && sent && vok && score) {
          std::snprintf(b, sizeof b, "\n %sFINISH rule (no candidate promises gain): %s%s", vfin ? kG : kW, vfin ? "YES -> DECLARE END" : "no", kW);
          vdec += b;
        } else if (a.dec_finish && sent && vok) {   // 끝 선언 (choice): 모델이 받은 상태와 "남은 탐사" 답
          std::snprintf(b, sizeof b, "\n told: %d cand, %d skipped, map %s, no growth %.0f s, %d abandoned", nv[i], nskip[i], st[i].map_closed ? "closed" : "open", st[i].since_occ * dt, av[i].n);
          vdec += b;
          vdec += std::string("\n ") + (vfin ? kG : kW) + "still mappable from reachable places? p(0 nothing.." + std::to_string(LL - 1) + ")=";
          for (float p : vlv) { std::snprintf(b, sizeof b, " %.2f", p); vdec += b; }
          std::snprintf(b, sizeof b, "\n expected %.2f  'nothing' x%d/%d  %s%s", vfin_p, fstreak[i], sc.left_hold, vfin ? "-> DECLARE END" : "-> go on", kW);
          vdec += b;
        }
        vcap = false;
      }
      for (int i : askg) { veto[i] = 0; asked[i] = 0; }
      for (int i : ask) {   // 고른 후보의 자리 (유효 후보 가운데 몇 째 · 후보 번호)
        int rank = 0;
        while (rank < nv[i] && vslot[(size_t)i * K + rank] != pick[i]) ++rank;
        u_ask += 1;
        n_asked[i] += 1;
        u_nv += nv[i];
        u_rank[std::min(rank, K - 1)] += 1;
        u_slot[pick[i]] += 1;
        u_first += rank == 0;
      }
      for (int i = 0; i < N && !a.autox; ++i) {
        if (fin[i]) { cmd[i] = sim::LowCmd{-1, 0}; act[(size_t)AW * i] = 0.f; continue; }   // 끝난 환경: 규칙 탐사로 흘려 둠 (안 셈)
        const sim::EnvState& s = st[i];
        if (!need[i]) continue;
        if (s.step % DS != 0) n_early[i] += 1;
        const bool forced = redo[i] != 0 && n_dec[i] > 0;
        redo[i] = 0;
        last_dec[i] = s.step;
        const int k = pick[i];
        if (score && seg[i].on) known(i, s.step);   // 판 끝에 쓸 "아는 칸 수"를 결정마다 새로
        n_dec[i] += 1;
        slot[i] = k;
        int tgt = k >= 0 ? cand[(size_t)i * K + k] : -1;
        if (use_pol) {
          const bool live = cmd[i].follow == 2 && cmd[i].target >= 0 && cmd[i].target == s.goal_cell && s.p_n > 0 && !avoided(i, cmd[i].target, s.step);
          if (a.hold > 0 && !forced && live && s.step - sw_step[i] < a.hold) tgt = cmd[i].target;
          else if (a.stick && k >= 0 && ((cur[i] >> k) & 1) && cmd[i].follow == 2 && cmd[i].target >= 0 && cmd[i].target == s.goal_cell && s.p_n > 0 && !avoided(i, cmd[i].target, s.step)) tgt = cmd[i].target;
          const double dn = tgt >= 0 ? std::hypot(W.ox + ((double)(tgt % W.gw) + 0.5) * c.res - (s.x + s.ex), W.oy + ((double)(tgt / W.gw) + 0.5) * c.res - (s.y + s.ey)) : 0.0;
          if (tgt >= 0 && dn < a.rule_near) {   // 가까운 목표: 규칙 추종기 (후보 번호 그대로)
            cmd[i] = sim::LowCmd{-1, 0};
            act[(size_t)AW * i] = (float)k;
            n_rule[i] += 1;
          } else if (tgt >= 0) {
            if (tgt != cmd[i].target || cmd[i].follow != 2) { sw_step[i] = s.step; h_try[i].push_back(tgt); }
            if (score && (tgt != cmd[i].target || cmd[i].follow != 2 || !seg[i].on)) {   // 목표 칸이 바뀜: 앞 구간을 닫고(실제로 는 칸 수) 새 구간을 엶
              const long kn_now = known(i, s.step);
              seg_close(i, s.step, kn_now, seg_why[i] == 1 ? "arrived" : seg_why[i] == 2 ? "abandoned" : "switched");
              seg_why[i] = 0;
              if (sel[i].on && sel[i].slot == k) { seg[i] = sel[i]; seg[i].cell = tgt; seg[i].known0 = kn_now; }
            }
            cmd[i] = sim::LowCmd{tgt, 2};
          } else {   // 후보 없음: 규칙 추종기의 "목표 없음"(제자리 회전)
            cmd[i] = sim::LowCmd{-1, 0};
            act[(size_t)AW * i] = -1.f;
          }
        } else {
          cmd[i] = sim::LowCmd{-1, 0};
          act[(size_t)AW * i] = (float)k;
        }
        if (decl[i]) {   // 판단 모델이 끝을 선언: 이 걸음에 환경이 판을 끝냄 (low_task.h kLowFinish) — 이 걸음은 규칙 추종기의 "목표 없음"
          cmd[i] = sim::LowCmd{sim::kLowFinish, 0};
          act[(size_t)AW * i] = -1.f;
        }
        if (tgt < 0) n_nogoal[i] += 1;
        else {
          const double d = std::hypot(W.ox + ((double)(tgt % W.gw) + 0.5) * c.res - (s.x + s.ex), W.oy + ((double)(tgt / W.gw) + 0.5) * c.res - (s.y + s.ey));
          tgt_d[i] += d;
          dbin[d < 1.0 ? 0 : d < 3.0 ? 1 : d < 15.0 ? 2 : 3] += 1;
        }
      }
      if (a.autox) {
        cmd = ecmd;   // 환경이 고른 목표 (아래 걸음 통계·정책 행동 넣기가 같은 값을 보게)
      } else {
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
        env.set_low_cmd(cmd);
        env.low_prepare(d_act);
        env.sync();
      }
      if (use_pol && left > 0) {   // 모든 판이 끝났으면 (녹화의 판 끝 뒤 멈춤) 정책을 돌리지 않음
        const sim::StepOut o = env.outputs();
        ppo->act_inference(o.low_img, o.low_1d, d_pol, N, -1);
        cudaDeviceSynchronize();
        cudaMemcpy(pol.data(), d_pol, sizeof(float) * pol.size(), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
          if (cmd[i].follow != 2) continue;
          act[(size_t)AW * i] = pol[(size_t)PW * i];
          act[(size_t)AW * i + 1] = pol[(size_t)PW * i + 1];
        }
        cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
      }
      env.step(d_act);
      env.sync();
      env.get_reward(rew, done);
      env.get_ep(ep);
      env.get_state(st);
      env.get_low(low);
      env.get_low_rl(rl);
      if (vrec) {   // 녹화: 이 걸음 한 장. 판이 끝나면 (--end-hold > 0) 리셋 없이 멈춘 로봇을 그 시간만큼 더 찍고 닫음 — 위층 결정 없음 · 정책 안 돌림 · 명령 0 (EnvCfg::end_hold)
        static const char* const kEnd[] = {"running", "SUCCESS (map closed, declared)", "WALL COLLISION", "TIME LIMIT", "declared too early", "person collision", "NO PROGRESS"};
        if (!fin[ve] && done[ve]) {
          const int dn = done[ve];
          char b[200];
          const char* why = dn >= 0 && dn <= 6 ? kEnd[dn] : "other";
          if (a.dec_finish && decl[ve] && dn == 1) why = "model said FINISH, robot map closed: SUCCESS";
          else if (a.dec_finish && decl[ve] && dn == 4) why = "model said FINISH TOO EARLY (map not closed)";
          else if (a.dec_finish && dn == 1) why = "SUCCESS by fallback rule (closed + no growth)";
          std::snprintf(b, sizeof b, "%sEPISODE END at t=%.1f s: %s%s\n", dn == 1 ? kG : kM, (nsteps[ve] + 1) * dt, why, kW);
          vend = b;
          vend_coll = dn == mdp::cur::kCollision || dn == mdp::cur::kCollisionPerson;
          vhold = c.end_hold ? (int)std::lround(a.end_hold / dt) : 0;
          if (vhold <= 0) {   // 멈춤 끔: 환경이 이미 리셋됐으므로 마지막 장만 3 s 더
            vid->hold(3.0, vlast + "\n" + vend);
            vid->close();
            vrec = false;
          }
        }
        if (vrec) {
          const sim::EnvState& s = st[ve];
          env.get_map_env(ve, vmap);
          vtrail.push_back((s.x + s.ex - W.ox) / c.res);
          vtrail.push_back((s.y + s.ey - W.oy) / c.res);
          dec::VidFrame vf;
          vf.lmap = vmap.data(); vf.size = c.size; vf.gw = W.gw; vf.gh = W.gh;
          vf.unknown_l = sim::kUnknown; vf.occ_l = mdp::kOccL;
          vf.low = vlow.empty() ? nullptr : vlow.data();
          vf.rob_cx = (s.x + s.ex - W.ox) / c.res; vf.rob_cy = (s.y + s.ey - W.oy) / c.res; vf.rob_th = s.th + s.eth;
          vf.trail = &vtrail;
          vf.cand = &vcand;
          if (vhold < 0) {   // 끝난 뒤에는 목표·길 표시 없음
            if (s.goal_cell >= 0) { vf.goal_cx = s.goal_cell % W.gw; vf.goal_cy = s.goal_cell / W.gw; }
            vf.wp = s.wp; vf.wp_n = s.p_n; vf.wp_i = s.p_i;
          }
          {   // 카메라 칸: 걸음마다 지금 자세에서 새로 그림 (후보 표시 없음). 모델에 보낸 그림은 작은 그림으로 따로
            std::vector<dec::Cand> nc;
            dec::render_cam_png(cam, s.x, s.y, s.th, ray, nc, std::vector<float>(), &vcam);
          }
          vf.cam = &vcam; vf.cam_w = cam.w; vf.cam_h = cam.h;
          if (vsent_t >= 0.0 && !vsent.empty()) {
            char sb[96];
            std::snprintf(sb, sizeof sb, "%ssent to model at t=%.1f s", vflash > 0 ? kG : kW, vsent_t);
            vf.sent = &vsent; vf.sent_flash = vflash > 0; vf.sent_label = sb;
            if (vflash > 0) --vflash;
          }
          char b[512];
          std::snprintf(b, sizeof b, "t=%6.1f s   upper=%s%s%s   seed %llu env %d\n", s.step * dt, a.upper.c_str(), upper == kDecider ? "/" : "", upper == kDecider ? a.dec_mode.c_str() : "", (unsigned long long)a.seed, ve);
          vf.text = b;
          if (vhold >= 0) {   // 끝난 뒤: 상태에 적힌 명령·속도를 그대로 (명령은 환경이 0 으로 — physics_step)
            vf.text += vend;
            vf.text += "upper: finished (no more decisions)\n";
            if (vhold == (int)std::lround(a.end_hold / dt)) std::snprintf(b, sizeof b, "%s: switching OFF (last cmd v=%+.2f m/s w=%+.2f rad/s)\n", use_pol ? "RL lower" : "lower", s.cmd_v, s.cmd_w);   // 끝난 바로 그 걸음: 명령은 아직 그 걸음 것
            else std::snprintf(b, sizeof b, "%s: OFF, cmd v=%+.2f m/s w=%+.2f rad/s\n", use_pol ? "RL lower" : "lower", s.cmd_v, s.cmd_w);
            vf.text += b;
            std::snprintf(b, sizeof b, "robot: v=%+.2f m/s w=%+.2f rad/s   GT wall cov %.2f (sim only)\n", s.v, s.w, s.coverage);
            vf.text += b;
            if (vend_coll) vf.text += "(sim does not integrate motion after a collision)\n";
            vf.text += "last " + vdec;
          } else {
            if (use_pol && cmd[ve].follow == 2) std::snprintf(b, sizeof b, "RL lower: a=(%+.2f,%+.2f) -> cmd v=%+.2f m/s w=%+.2f rad/s\n", pol[(size_t)PW * ve], pol[(size_t)PW * ve + 1], s.cmd_v, s.cmd_w);
            else std::snprintf(b, sizeof b, "lower: rule follower -> cmd v=%+.2f m/s w=%+.2f rad/s\n", s.cmd_v, s.cmd_w);
            vf.text += b;
            std::snprintf(b, sizeof b, "robot: v=%+.2f m/s w=%+.2f rad/s   GT wall cov %.2f (sim only)%s\n", s.v, s.w, s.coverage, s.map_closed ? "  map closed" : "");
            vf.text += b;
            vf.text += vdec;
          }
          vlast = vf.text;
          vid->frame(vf);
          if (vhold >= 0 && --vhold < 0) {   // 멈춤 시간 끝
            vid->close();
            vrec = false;
          }
        }
      }
      for (int i = 0; i < N; ++i) {
        if (fin[i]) continue;
        nsteps[i] += 1;
        if (done[i]) {
          fin[i] = 1;
          first[i] = ep[i];
          if (score) seg_close(i, st[i].step, kn[i], "episode_end");
          h_try[i].clear();
          h_ab[i].clear();
          fstreak[i] = 0;
          if (flf) {   // 끝 선언 기록: 판 끝의 아는 칸 수(마지막 결정 때 센 값)로 "그때부터 는 칸 수"를 붙임
            for (size_t r = 0; r < lrow[i].size(); ++r) std::fprintf(flf, "%s\t%ld\t%ld\n", lrow[i][r].c_str(), kn[i], kn[i] - lkn[i][r]);
            lrow[i].clear();
            lkn[i].clear();
          }   // 판 끝: 리셋 뒤 지도일 수 있어 마지막으로 센 값(1 s 안쪽)을 씀
          end_by[i] = decl[i] && (done[i] == mdp::cur::kDeclareOk || done[i] == mdp::cur::kDeclareShort) ? 1 : 0;   // 1 = 모델의 끝 선언으로 끝남 (그 밖의 성공은 닫힘 규칙)
          first[i].done = done[i];
          --left;
          continue;
        }
        const sim::EnvState& s = st[i];
        dist[i] += std::hypot(s.x - px[i], s.y - py[i]);
        px[i] = s.x;
        py[i] = s.y;
        if (s.step % DS == 0) {   // frontier_run 과 같은 셈: 결정(1 s) 사이 직선 거리의 합
          chord[i] += std::hypot(s.x - cx[i], s.y - cy[i]);
          cx[i] = s.x;
          cy[i] = s.y;
        }
        if (s.goal_cell != goal_prev[i]) {
          if (goal_prev[i] >= 0 && s.goal_cell >= 0) n_sw[i] += 1;
          if (!use_pol) sw_step[i] = s.step;
          goal_prev[i] = s.goal_cell;
        }
        if (s.map_closed && fc_step[i] < 0) fc_step[i] = s.step;
        {   // 60° 이상 전환: 목표 칸이 바뀐 걸음에 새 목표 쪽 방위가 60° 이상이면 재기 시작, 방위가 30° 안에 들면 끝 (그 전에 목표가 또 바뀌면 버림)
          const auto bearing = [&](int cell) {
            const double dx = W.ox + ((double)(cell % W.gw) + 0.5) * c.res - s.x, dy = W.oy + ((double)(cell / W.gw) + 0.5) * c.res - s.y;
            return std::fabs(std::remainder(std::atan2(dy, dx) - (double)s.th, 6.283185307179586)) * 57.29577951308232;
          };
          if (t_cell[i] >= 0) {
            if (s.goal_cell != t_cell[i]) t_cell[i] = -1;
            else if (bearing(t_cell[i]) < 30.0) { t_n[i] += 1; t_sum[i] += s.step - t_start[i]; t_cell[i] = -1; }
          }
          if (s.goal_cell != t_prev[i]) {
            if (t_prev[i] >= 0 && s.goal_cell >= 0 && bearing(s.goal_cell) >= 60.0) { t_cell[i] = s.goal_cell; t_start[i] = s.step; }
            t_prev[i] = s.goal_cell;
          }
        }
        end_v[i] = s.v;
        end_w[i] = s.w;
        hx[(size_t)i * kH + nsteps[i] % kH] = s.x;
        hy[(size_t)i * kH + nsteps[i] % kH] = s.y;
        hb[(size_t)i * kH + nsteps[i] % kH] = s.cmd_v < 0.f;
        end_since[i] = s.step - sw_step[i];
        end_safe[i] = rl[i].safe != 0;
        end_d[i] = s.goal_cell >= 0 ? (float)std::hypot(W.ox + ((double)(s.goal_cell % W.gw) + 0.5) * c.res - s.x, W.oy + ((double)(s.goal_cell / W.gw) + 0.5) * c.res - s.y) : -1.f;
        n_safe[i] += rl[i].safe != 0;
        n_back[i] += s.cmd_v < 0.f;
        n_slow[i] += std::fabs(s.v) < 0.1f;
        w_abs[i] += std::fabs(s.w);
        if (use_pol && cmd[i].follow == 2 && (s.goal_cell != cmd[i].target || s.p_n <= 0)) n_pathless[i] += 1;
        // 막힘 신호 → 그 칸을 피하고 바로 다시 고름
        if (rl[i].stall && !sig_prev[i]) {
          n_stall[i] += 1;
          avoid_add(i, rl[i].stall_tgt, s.step);
          if (rl[i].stall_tgt >= 0) h_ab[i].push_back(rl[i].stall_tgt);
          if (a.avoid) redo[i] = 1;
          seg_why[i] = 2;
        }
        sig_prev[i] = rl[i].stall;
        if (use_pol && cmd[i].follow == 2) {
          const sim::LowEp& L = low[i];
          const bool reached = L.reach_step == s.step && L.reached > 0;
          const bool off = !reached && L.hold != 0;
          const bool nopath = !reached && !off && s.step > last_dec[i] && (s.goal_cell != cmd[i].target || s.p_n <= 0);
          if (reached) n_reach[i] += 1;
          if (reached) seg_why[i] = 1;
          else if (off || nopath) seg_why[i] = 2;
          if (a.redecide && (reached || off || nopath)) {
            if (off) { n_off[i] += 1; avoid_add(i, cmd[i].target, s.step); }
            if (nopath) { n_nopath[i] += 1; avoid_add(i, cmd[i].target, s.step); }
            if ((off || nopath) && cmd[i].target >= 0) h_ab[i].push_back(cmd[i].target);
            redo[i] = 1;
          }
        }
      }
    }
    // 장면 요약
    double n = N, succ = 0, cw = 0, cp = 0, idle = 0, trunc = 0, ts = 0, t_all = 0, d_all = 0, ch_all = 0, d_succ = 0, cov_s = 0;
    double s_dec = 0, s_sw = 0, s_stall = 0, s_reach = 0, s_off = 0, s_nopath = 0, s_safe = 0, s_back = 0, s_slow = 0, s_nogoal = 0, s_avoided = 0, s_early = 0, s_steps = 0, s_tgt = 0, s_w = 0, s_pathless = 0, ep_stall = 0, s_rule = 0;
    double fc_s = 0, s_tn = 0, s_tsum = 0;
    std::FILE* f = nullptr;
    if (!a.out.empty()) {
      f = std::fopen((a.out + "_" + W.name + ".tsv").c_str(), "w");
      if (!f) throw std::runtime_error("--out 파일을 못 엶");
      std::fputs("env\tdone\tt\tcoverage\tdist_m\tchord_m\tdecisions\tswitches\tstalls\treached\toff\tnopath\tsafe_steps\tback_steps\tslow_steps\tnogoal_dec\tend_v\tend_steps_since_goal_change\tend_goal_dist\tend_safe\tend_w\tend_5s_span_m\tend_5s_back_steps\tfirst_closed_t\tturn60_n\tturn60_steps", f);
      if (a.dec_finish) std::fputs("\tended_by_model\tfinish_questions\tdecl_p\tdecl_idle_s\tdecl_map_closed\tdecl_grid_open_cells\tdecl_grid_clusters", f);   // 끝 선언을 켰을 때만 (끄면 파일이 그대로)
      std::fputs("\n", f);
    }
    for (int i = 0; i < N; ++i) {
      const int dd = fin[i] ? first[i].done : mdp::cur::kTruncated;
      const double ti = fin[i] ? first[i].t : nsteps[i] * dt;
      succ += dd == mdp::cur::kDeclareOk;
      cw += dd == mdp::cur::kCollision;
      cp += dd == mdp::cur::kCollisionPerson;
      idle += dd == mdp::cur::kNoProgress;
      trunc += dd == mdp::cur::kTruncated;
      t_all += ti;
      d_all += dist[i];
      ch_all += chord[i];
      if (dd == mdp::cur::kDeclareOk) { ts += first[i].t; d_succ += dist[i]; cov_s += first[i].coverage; }
      s_dec += n_dec[i]; s_sw += n_sw[i]; s_stall += n_stall[i]; s_reach += n_reach[i]; s_off += n_off[i]; s_nopath += n_nopath[i]; s_safe += n_safe[i];
      s_back += n_back[i]; s_slow += n_slow[i]; s_nogoal += n_nogoal[i]; s_avoided += n_avoided[i]; s_early += n_early[i]; s_steps += nsteps[i]; s_tgt += tgt_d[i]; s_w += w_abs[i];
      s_pathless += n_pathless[i];
      s_rule += n_rule[i];
      ep_stall += n_stall[i] > 0;
      float span = 0.f;
      int nb = 0;
      if (nsteps[i] > kH) {
        const size_t o = (size_t)i * kH;
        const int last = (nsteps[i] - 1) % kH;   // 끝난 걸음 바로 앞
        for (int q = 0; q < kH; ++q) {
          span = std::max(span, (float)std::hypot(hx[o + q] - hx[o + last], hy[o + q] - hy[o + last]));
          nb += hb[o + q];
        }
      }
      if (f)
        std::fprintf(f, "%d\t%d\t%.1f\t%.4f\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%d\t%.2f\t%d\t%.2f\t%.2f\t%d\t%.1f\t%d\t%d", i, dd, ti, fin[i] ? first[i].coverage : 0.f, dist[i], chord[i], n_dec[i], n_sw[i], n_stall[i], n_reach[i], n_off[i],
                     n_nopath[i], n_safe[i], n_back[i], n_slow[i], n_nogoal[i], end_v[i], end_since[i], end_d[i], end_safe[i], end_w[i], span, nb, fc_step[i] >= 0 ? fc_step[i] * dt : -1.0, t_n[i], t_sum[i]);
      if (f && a.dec_finish) std::fprintf(f, "\t%d\t%d\t%.3f\t%.1f\t%d\t%d\t%d", end_by[i], n_finq[i], d_p[i], d_idle[i], d_closed[i], d_block[i], d_clus[i]);
      if (f) std::fputs("\n", f);
      if (dd == mdp::cur::kDeclareOk) fc_s += fc_step[i] >= 0 ? fc_step[i] * dt : first[i].t;
      s_tn += t_n[i]; s_tsum += t_sum[i];
    }
    if (f) std::fclose(f);
    const char* tag = use_pol ? "rl" : "rule";
    std::printf("[%s %s] 판 %d: 성공 %.3f · 벽 충돌 %.3f · 사람 충돌 %.3f · 진행 없음 %.3f · 끊김 %.3f | 닫힘까지 %.1f s (성공 판 평균, 정답 덮음 %.3f) · 판 평균 길이 %.1f s\n", tag, W.name.c_str(), N, succ / n,
                cw / n, cp / n, idle / n, trunc / n, succ > 0 ? ts / succ : 0.0, succ > 0 ? cov_s / succ : 0.0, t_all / n);
    std::printf("[%s %s] 평균 속도 %.3f m/s (걸음 거리 합 / 시간) · %.3f m/s (1 s 사이 직선 거리 합 / 시간 — frontier_run 과 같은 셈) · 이동 거리 모든 판 %.1f m · 성공 판 %.1f m\n", tag, W.name.c_str(),
                t_all > 0 ? d_all / t_all : 0.0, t_all > 0 ? ch_all / t_all : 0.0, d_all / n, succ > 0 ? d_succ / succ : 0.0);
    const double nd = std::max(1.0, s_dec), ng = std::max(1.0, s_dec - s_nogoal), nst = std::max(1.0, s_steps);
    std::printf("[%s %s] 위층: 판당 결정 %.1f 개 (그중 10 걸음 주기 밖 %.3f) · 목표 없는 결정 %.3f · 규칙 추종기에 맡긴 결정 %.3f · 목표 칸 바뀜 판당 %.1f 번 (%.1f s 에 한 번) · 피하는 칸이라 건너뛴 후보 %.0f 개 | 결정 때 목표까지 직선 거리 평균 %.2f m — "
                "1 m 안 %.3f · 1~3 m %.3f · 3~15 m %.3f · 15 m 밖 %.3f\n", tag, W.name.c_str(), s_dec / n, s_early / nd, s_nogoal / nd, s_rule / nd, s_sw / n, s_sw > 0 ? t_all / s_sw : 0.0, s_avoided, s_tgt / ng,
                dbin[0] / ng, dbin[1] / ng, dbin[2] / ng, dbin[3] / ng);
    std::printf("[%s %s] 아래층: 막힘 신호 %.0f 번 (선 판 비율 %.3f) · 통과 %.0f · 경로 끝이 목표 밖 %.0f · 경로 없음 %.0f | 걸음 비율 — 안전층이 고침 %.3f · 뒤로 %.3f · 0.1 m/s 아래 %.3f · 경로 없이 몬 걸음 %.3f · 평균 |ω| %.2f rad/s\n",
                tag, W.name.c_str(), s_stall, ep_stall / n, s_reach, s_off, s_nopath, s_safe / nst, s_back / nst, s_slow / nst, s_pathless / nst, s_w / nst);
    std::printf("[%s %s] 더: 지도가 처음 닫힌 시각 %.1f s (성공 판 평균) · 뒤로 걸음 비율 %.4f · 안전층·지킴이가 고친 걸음 %.2f s/판 · 60° 이상 전환 판당 %.1f 번 · 평균 %.2f s (목표 쪽 방위 30° 안까지; 그 전에 목표가 바뀐 것은 뺌)\n",
                tag, W.name.c_str(), succ > 0 ? fc_s / succ : 0.0, s_back / nst, s_safe * dt / n, s_tn / n, s_tn > 0 ? s_tsum * dt / s_tn : 0.0);
    std::printf("[%s %s] 벽 충돌 %.2f 건 / 100 m (벽 충돌 판 %.0f · 모든 판 이동 거리 합 %.0f m)\n", tag, W.name.c_str(), d_all > 0 ? cw / d_all * 100.0 : 0.0, cw, d_all);
    if (upper != kSlot0) {
      std::sort(lat.begin(), lat.end());
      const double na = std::max(1L, u_ask), nok = std::max<size_t>(1, lat.size());
      std::printf("[%s %s] 위층 고르기 %s: 고른 결정(유효 후보 2 개 이상) %ld 개 (모든 결정의 %.3f, 유효 후보 평균 %.2f 개) · slot0 로 떨어짐 %ld (연결·시간 끝 %ld · HTTP %ld · 답 꼴 %ld) · 특징 어긋남 %ld\n", tag, W.name.c_str(),
                  a.upper.c_str(), u_ask, u_ask / nd, u_nv / na, u_fall, u_fall_curl, u_fall_http, u_fall_shape, u_mismatch);
      std::printf("[%s %s] 고른 후보가 유효 후보 가운데 몇 째인가 (0 = slot0 규칙과 같은 것): ", tag, W.name.c_str());
      for (int k = 0; k < K; ++k) std::printf("%d: %.3f%s", k, u_rank[k] / na, k + 1 < K ? " · " : "\n");
      std::printf("[%s %s] 고른 후보 번호 분포: ", tag, W.name.c_str());
      for (int k = 0; k < K; ++k) std::printf("%d: %.3f%s", k, u_slot[k] / na, k + 1 < K ? " · " : "\n");
      if (upper == kDecider) {
        std::printf("[%s %s] 판단 요청 지연 (성공 %zu 개, 동시 %d 개씩 보낸 상태 — 시뮬 시계에 안 들어감): 중앙값 %.1f ms · p95 %.1f ms · 최대 %.1f ms | 묶음 %ld 번 · 기다린 시간 합 %.1f s · 평균 confidence %.3f · 고른 선택지 확률 평균 %.3f\n",
                    tag, W.name.c_str(), lat.size(), a.dec_parallel, lat.empty() ? 0.0 : lat[lat.size() / 2], lat.empty() ? 0.0 : lat[std::min(lat.size() - 1, (size_t)(lat.size() * 0.95))],
                    lat.empty() ? 0.0 : lat.back(), u_batches, u_wall, u_conf / nok, u_p / nok);
        if (!first_err.empty()) std::printf("[%s %s] 첫 실패: %s\n", tag, W.name.c_str(), first_err.c_str());
      }
    }
    if (score) {
      std::printf("[%s %s] 점수 방식(score, 단계 %d 개 · 비김 폭 %.2f): 후보 점수 물음 %ld 개 · 기대 점수 평균 %.3f · 결정마다 최고 기대 점수 평균 %.3f · 비긴 결정 %ld 개 · 목표 구간 %ld 개 · 가장 그럴듯한 단계 분포", tag,
                  W.name.c_str(), SL, sc.margin, u_sq, u_sq > 0 ? u_sexp / u_sq : 0.0, lat.empty() ? 0.0 : u_sbest / lat.size(), u_tie, u_seg);
      for (int l = 0; l < SL; ++l) std::printf(" %d: %.3f", l, u_sq > 0 ? (double)u_lvl[l] / u_sq : 0.0);
      std::printf("\n");
      if (a.dec_finish) std::printf("[%s %s] 점수 방식의 끝 선언 규칙: %s · 후보가 하나도 없어 선언한 결정 %ld 개\n", tag, W.name.c_str(), sc.finish_below >= 0.f ? "모든 후보 기대 점수 < score_finish_below" : "모든 후보의 가장 그럴듯한 단계가 0", u_fin_nocand);
      if (fsc) std::fclose(fsc);
      if (fgn) std::fclose(fgn);
    }
    if (a.dec_finish && !score) {
      std::printf("[%s %s] choice 끝 선언: \"남은 탐사\" 질문의 가장 그럴듯한 단계 분포 (0 = 남은 게 없음, %d 결정 이어지면 선언)", tag, W.name.c_str(), sc.left_hold);
      for (int l = 0; l < LL; ++l) std::printf(" %d: %.3f", l, u_finq > 0 ? (double)u_llvl[l] / u_finq : 0.0);
      std::printf("\n");
      if (flf) std::fclose(flf);
    }
    if (upper == kDecider && prune)

      std::printf("[%s %s] 가지치기(prune): 물은 후보 %ld 개 (결정당 앞 %d 개까지) · 막혔다 함 %ld (%.3f, 문턱 %.2f) · \"막혔다\" 확률 평균 %.3f · 물은 후보가 다 막혔다 한 결정 %ld\n", tag, W.name.c_str(), u_q, a.dec_ask_m, u_veto,
                  u_q > 0 ? (double)u_veto / u_q : 0.0, a.dec_veto_thr, u_q > 0 ? u_pblk / u_q : 0.0, u_allveto);
    if (a.gt) {
      const double gn = std::max(1L, g_n);
      std::printf("[%s %s] 정답 격자 (유효 후보 2 개 이상 결정 %ld 개, 바탕 장면 %s): 규칙의 첫 후보가 막힘 %.4f · 실제 고른 후보가 막힘 %.4f · 모든 유효 후보 가운데 막힘 %.4f (%ld / %ld) — 막힘 = 낮은 장애물 없는 바탕보다 참 길이 %.1f m · %.1f 배 넘게 길거나 못 감\n",
                  tag, W.name.c_str(), g_n, a.gt_base.empty() ? "없음(자기 자신)" : stem(a.gt_base[wi]).c_str(), g_first / gn, g_pick / gn, g_cand > 0 ? (double)g_cand_blk / g_cand : 0.0, g_cand_blk, g_cand, a.gt_add, a.gt_mul);
      if (prune)
        std::printf("[%s %s] 가지치기 대 정답: 막혔다 함·정답 막힘 %ld · 막혔다 함·정답 안 막힘 %ld · 안 막혔다 함·정답 막힘 %ld · 안 막혔다 함·정답 안 막힘 %ld | 막혔다 한 것 가운데 맞은 비율 %.3f · 정답 막힘 가운데 잡은 비율 %.3f\n",
                    tag, W.name.c_str(), g_vt_blk, g_vt_clear, g_miss, g_ok_clear, g_vt_blk + g_vt_clear > 0 ? (double)g_vt_blk / (g_vt_blk + g_vt_clear) : 0.0, g_vt_blk + g_miss > 0 ? (double)g_vt_blk / (g_vt_blk + g_miss) : 0.0);
    }
    if (a.dec_finish) {   // 끝 선언 요약. 판정은 로봇 지도만 — 정답 덮음은 맨 끝 진단 칸에만
      int nm = 0, m_ok = 0, m_early = 0, g_ok = 0, g_early = 0, n_rule_end = 0;
      double idle_m = 0, saved = 0, cov_m = 0, t_m = 0;
      for (int i = 0; i < N; ++i) {
        const int dd = fin[i] ? first[i].done : mdp::cur::kTruncated;
        if (!end_by[i]) { n_rule_end += dd == mdp::cur::kDeclareOk; continue; }
        nm += 1;
        m_ok += dd == mdp::cur::kDeclareOk;
        m_early += dd == mdp::cur::kDeclareShort;
        g_ok += d_block[i] == 0;
        g_early += d_block[i] != 0;
        idle_m += d_idle[i];
        t_m += first[i].t;
        cov_m += first[i].coverage;
        if (dd == mdp::cur::kDeclareOk) saved += std::max(0.0, (double)c.close_idle_s - d_idle[i]);
      }
      std::printf("[%s %s] 끝 선언(--decider-finish): 물음 %ld 개 · 끝 선언 %ld 개 (choice 방식: \"남은 탐사\" 기대 점수 평균 %.3f) | 모델 선언으로 끝난 판 %d — 로봇 지도 닫힘 규칙(frontier.h)으로 맞음 %d · 이른 선언(코드 4) %d ‖ 점유 격자만(grid_open_frontier)으로 맞음 %d · 이름 %d | "
                  "닫힘 규칙(닫힘 + %.0f s)으로 끝난 성공 판 %d\n", tag, W.name.c_str(), u_finq, u_finyes, u_finq > 0 ? u_pfin / u_finq : 0.0, nm, m_ok, m_early, g_ok, g_early, c.close_idle_s, n_rule_end);
      std::printf("[%s %s] 끝 선언 판 평균: 끝 시각 %.1f s · 선언 때 지도 안 늘어난 시간 %.1f s · 규칙 대기 대비 아낀 시간 추정 %.1f s (맞은 선언 판 평균 = %.0f s − 선언 때 시계, 0 아래는 0 — 추정) · [시뮬 전용 진단, 판정에 안 씀] 정답 벽 덮음 %.3f\n",
                  tag, W.name.c_str(), nm > 0 ? t_m / nm : 0.0, nm > 0 ? idle_m / nm : 0.0, m_ok > 0 ? saved / m_ok : 0.0, c.close_idle_s, nm > 0 ? cov_m / nm : 0.0);
    }
    {
      double qa = 0, qs = 0;
      for (int i = 0; i < N; ++i) { qa += n_asked[i]; if (fin[i] && first[i].done == mdp::cur::kDeclareOk) qs += n_asked[i]; }
      if (upper != kSlot0) std::printf("[%s %s] 판당 물은 결정 %.1f 개 (성공 판 평균 %.1f 개)\n", tag, W.name.c_str(), qa / n, succ > 0 ? qs / succ : 0.0);
    }
    std::fflush(stdout);
  }
  cudaFree(d_act);
  cudaFree(d_pol);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("SIM_EXACT", "1", 0);   // frontier_run 과 같은 광선 길 (정확한 길)
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " 뒤에 값");
      return argv[++i];
    };
    try {
      if (k == "--scenes") { while (i + 1 < argc && argv[i + 1][0] != '-') a.scenes.push_back(argv[++i]); }
      else if (k == "--policy") a.policy = nx();
      else if (k == "--episodes") a.episodes = std::stoi(nx());
      else if (k == "--seed") a.seed = std::stoull(nx());
      else if (k == "--people") a.people = std::stoi(nx());
      else if (k == "--out") a.out = nx();
      else if (k == "--env") a.env = nx();
      else if (k == "--model") a.model = nx();
      else if (k == "--frontier") a.frontier = nx();
      else if (k == "--eval-seed") a.eval_seed = true;
      else if (k == "--no-avoid") a.avoid = false;
      else if (k == "--no-redecide") a.redecide = false;
      else if (k == "--no-stick") a.stick = false;
      else if (k == "--hold") a.hold = std::stoi(nx());
      else if (k == "--stall-s") a.stall_s = std::stod(nx());
      else if (k == "--rule-near") a.rule_near = std::stod(nx());
      else if (k == "--auto") a.autox = true;
      else if (k == "--upper") a.upper = nx();
      else if (k == "--rule-stick") a.rule.stick = std::stof(nx());
      else if (k == "--rule-turn") a.rule.turn_m = std::stof(nx());
      else if (k == "--decider-url") { a.dec_url = nx(); a.upper = "decider"; }
      else if (k == "--decider-input") a.dec_input = nx();
      else if (k == "--decider-prompt") a.dec_prompt = std::stoi(nx());
      else if (k == "--decider-timeout-ms") a.dec_timeout_ms = std::stoi(nx());
      else if (k == "--decider-parallel") a.dec_parallel = std::stoi(nx());
      else if (k == "--decider-px") a.dec_px = std::stoi(nx());
      else if (k == "--decider-shuffle") a.dec_shuffle = true;
      else if (k == "--decider-dump") a.dec_dump = nx();
      else if (k == "--decider-dump-n") a.dec_dump_n = std::stoi(nx());
      else if (k == "--decider-mode") a.dec_mode = nx();
      else if (k == "--decider-veto-thr") a.dec_veto_thr = std::stod(nx());
      else if (k == "--decider-ask-m") a.dec_ask_m = std::stoi(nx());
      else if (k == "--decider-cam-h") a.dec_cam_h = std::stoi(nx());
      else if (k == "--decider-cam-div") a.dec_cam_div = std::stoi(nx());
      else if (k == "--decider-finish") a.dec_finish = true;
      else if (k == "--decider-config") a.dec_config = nx();
      else if (k == "--decider-left-hold") a.dec_left_hold = std::stoi(nx());
      else if (k == "--decider-score-margin") a.dec_score_margin = std::stod(nx());
      else if (k == "--decider-score-finish") a.dec_score_finish = std::stod(nx());
      else if (k == "--decider-mission") a.dec_mission = nx();
      else if (k == "--video") a.video = nx();
      else if (k == "--video-env") a.video_env = std::stoi(nx());
      else if (k == "--end-hold") a.end_hold = std::stod(nx());
      else if (k == "--gt") a.gt = true;
      else if (k == "--gt-base") { while (i + 1 < argc && argv[i + 1][0] != '-') a.gt_base.push_back(argv[++i]); a.gt = true; }
      else { std::fprintf(stderr, "모르는 인자 %s\n", k.c_str()); return 2; }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "combine_run: %s\n", e.what());
      return 2;
    }
  }
  if (a.scenes.empty()) {
    std::fprintf(stderr, "combine_run --scenes a.scene [b.scene …] [--policy <아래층 체크포인트>] [--episodes 256] [--seed 1] [--people 0] [--eval-seed] [--out <접두사>] [--no-avoid] [--no-redecide] [--no-stick]\n");
    return 2;
  }
  try {
    return run(a);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "combine_run: %s\n", e.what());
    return 1;
  }
}
