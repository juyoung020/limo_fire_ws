// GPU 학습 환경 호스트 쪽: 설정 읽기, 맵(장면 → 정답 격자) 만들기, 환경 묶음 메모리와 한 걸음 부르기.
// 같은 Env 를 GPU(학습)와 CPU(참조, 비트 비교)로 돌릴 수 있다 (tools/env_verify).
#pragma once
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "env.h"
#include "env_cfg.h"   // EnvHostCfg · load_env_cfg (CUDA 없는 호스트 코드)
#include "scene_host.h"

namespace sim {

// 맵 하나: 장면 + 칸 격자 + 시작 칸
struct World {
  std::string name;
  Scene scene;
  std::vector<uint8_t> grid;            // kCell* 비트 (env.h)
  std::vector<int32_t> spawn;
  std::vector<uint8_t> clear;           // 칸마다 막힌 칸까지 거리 (cm) — 움직이는 사람 (people_host.cpp people_clearance)
  int gw = 0, gh = 0;
  int n_wall = 0, n_floor = 0;          // 덮음 분모 (벽 면 칸), 바닥 덮음 분모 (갈 수 있는 바닥 칸)
  bool closed = true;                   // 갈 수 있는 바닥이 맵 테두리에 안 닿음 (집이 닫힘)
  std::string leak;                     // 닫히지 않았으면 어디서 새는지 (좌표)
  float ox = 0, oy = 0;
  int stage = -1;                       // 커리큘럼 단계 (장면 목록의 등급 열). −1 이면 넓이로 (mdp::cur::stage_of_area)
};

// 장면 → 칸 격자 (env_host.cpp 위 설명). 집이 닫히지 않으면(갈 수 있는 바닥이 맵 테두리에 닿음) require_closed 면 예외,
// 아니면 closed = false·leak 에 적고 돌려줌 (덮음을 안 쓰는 도구 — 실물 녹화 다시 돌리기처럼 일부만 그린 지도)
World build_world(const std::string& name, Scene scene, const EnvHostCfg& hc, bool require_closed = true);

// 장면 묶음 목록 파일 (env_host.cpp) — 줄마다 장면 하나: 경로(목록 파일 기준 상대·절대) [탭 넓이 탭 벽 칸 탭 등급 …] (S2 list.tsv 꼴,
// 첫 열이 path 인 머리줄·# 주석·빈 줄 무시). skipped 를 주면 안 닫힌 장면은 버리고 이유를 넣음
std::vector<World> load_scene_list(const std::string& list_file, const EnvHostCfg& hc, std::vector<std::string>* skipped = nullptr);

// 러너가 읽는 출력 — 모두 GPU 장치 포인터 (CPU 판이면 호스트), step 이 끝나면(스트림 순서) 유효. 호스트 왕복 없음
struct StepOut {
  const uint8_t* obs_img;       // [N][mdp::obs::IMG_SIZE]
  const float* obs_1d;          // [N][o1d] — o1d = mdp::obs::POLICY_1D (기본), mdp::hier::POLICY_1D (계층형: 뒤에 후보 칸)
  const float* priv;            // [N][mdp::obs::PRIV]
  const float* reward;          // [N]
  const float* reward_terms;    // [N][mdp::rew::kTerms]
  const int32_t* done;          // [N] mdp::cur::Done — 종료 여부는 mdp::cur::terminal(done), 끊김 = 3
  const EpInfo* ep;             // [N] done ≠ 0 인 환경의 끝난 판 정보 (덮음·t95·길이·충돌·단계·맵)
  int o1d;                      // 관측 1D 한 환경 칸 수
};

// 그래프 모드 관측 (cfg.graph_on — mdp/obs_layout.h mdp::graph, mdp/graph.h). StepOut 처럼 장치 포인터(CPU 판이면 호스트), 그래프 모드가 아니면 nullptr
struct GraphOut {
  const uint8_t* feat;          // [N][graph::N][graph::FEAT]
  const uint32_t* edge;         // [N][graph::N]  비트 k = 스텐실 k 번 이웃과 간선
  const int32_t* cur;           // [N]  현재 노드 (늘 유효)
  const float* glob;            // [N][graph::GLOB]
  const int32_t* cand;          // [N][graph::NB]  현재 노드의 이웃 칸 (y · gw + x, −1 간선 없음) — 행동 a[0] = k 의 목표. 행동 마스크 = cand ≥ 0 = edge[cur] 비트
};

class Env {
 public:
  // gpu: true 면 장치 메모리·커널, false 면 CPU 참조판
  Env(const EnvHostCfg& hc, const std::vector<World>& worlds, const LidarModel& lidar, int n_env, uint64_t seed, bool gpu);
  ~Env();
  Env(const Env&) = delete;
  Env& operator=(const Env&) = delete;

  void reset();                         // 모든 환경 새 판 + 관측
  // 환경마다 float mdp::act::DIM 개 (v, vy, ω, 끝 선언) — 정규 단위 [−1, 1], 선언은 0/1 (> 0.5 이면 선언).
  // cfg.act_raw 면 물리 단위. 목표점 행동이면 [번호, 0, 0, 선언]. 계층형(kActPath)이면 [후보 번호, 0, 0, 0] — 한 번 부르면 결정 하나
  // (속도 행동 decide_steps 걸음 묶음: 보상·항은 묶음 합, done 은 끝난 걸음의 코드 — 판이 묶음 가운데서 끝나면 그 환경은 남은 걸음을 쉬고 묶음 끝에 리셋).
  // 계층형 후보 칸은 관측 1D 뒤 mdp::hier::CAND_1D 칸. GPU 면 장치 포인터, CPU 면 호스트 포인터
  void step(const float* actions, cudaStream_t st = 0);
  StepOut outputs() const;
  GraphOut graph() const;
  void graph_refresh();   // 그래프 관측을 지금 상태에서 한 번 더 냄 (값은 그대로 — 시간 재기용, env_bench --graph)
  void sync() const;

  // 호스트로 꺼내기 (비교·뷰어용)
  void get_state(std::vector<EnvState>& out) const;
  void set_state(int env, const EnvState& s);   // 시험용: 한 환경 상태를 바꿔 넣음 (실물 녹화 다시 돌리기 등)
  void get_map(std::vector<int16_t>& out) const;
  // 판 동안 덮은 칸 [env · size² + cy · size + cx]: bit0 본 벽 면 칸(덮음 분자), bit1 빔이 지나간 바닥 칸 (env.h marks bit3·4)
  void get_seen(std::vector<uint8_t>& out) const;
  // 깊이 카메라 장애물 층 (계층형 cam_on — 아니면 빈 벡터): 칸마다 높이 칸 비트 [env · size²], 마지막 깊이 장 열마다 띠 안 점 [env · DEPTH_W · 2]
  void get_camlayer(std::vector<uint8_t>& vox, std::vector<float>& pts) const;
  // 닫힘 규칙의 지나간 길 비트 판 [env · 2 · size · fmap_row] (env.h EnvBatch::trail: 0 지나간 칸 · 1 본 칸) — 도구가 닫힘을 호스트에서 다시 셀 때
  void get_trail(std::vector<uint32_t>& out) const;
  void get_obs(std::vector<uint8_t>& img, std::vector<float>& o1d, std::vector<float>& priv) const;
  void get_cand(std::vector<int32_t>& out) const;   // 계층형 후보 목표 칸 [N][mdp::hier::K] (−1 빈 슬롯) — 도구용
  void get_graph(std::vector<uint8_t>& feat, std::vector<uint32_t>& edge, std::vector<int32_t>& cur, std::vector<float>& glob, std::vector<int32_t>& cand) const;
  void get_gdec(std::vector<GraphDec>& out) const;   // 그래프 모드: 방금 끝난 결정의 결과 [N] (env.h GraphDec — 걸음 수·이동 거리·끝난 사유·전작 방식 보상 조각)
  // 아래층 계측·단독 과제 (include/low_task.h — cfg.low_stat·low_task 일 때만): 판 누적 [N] (끄면 빈 벡터) · 환경마다 목표 칸·모드 넣기 (결정 앞) ·
  // 환경 하나의 로봇 지도 [size²] (도구가 목표를 뽑을 때)
  void get_low(std::vector<LowEp>& out) const;
  void set_low_cmd(const std::vector<LowCmd>& cmd);
  void get_map_env(int env, std::vector<int16_t>& out) const;
  void get_terms(std::vector<float>& terms) const;
  void get_ep(std::vector<EpInfo>& ep) const;
  void get_reward(std::vector<float>& reward, std::vector<int32_t>& done) const;
  // 움직이는 사람 (src/people_host.cpp): 환경마다 다음 판 사람 수 상한 (판 시작 인자, 0 ~ kMaxPeople) · 이 판 사람들 (뷰어 재생)
  void set_people_max(const std::vector<int>& k);
  void get_people(std::vector<People>& out) const;

  const EnvBatch& batch() const { return b_; }
  int n_env() const { return b_.n_env; }

 private:
  bool gpu_;
  EnvBatch b_{};
  std::vector<void*> owned_;            // 장치 메모리
  std::vector<std::vector<uint8_t>> host_;   // CPU 판 메모리
  void* rt_ = nullptr;                  // OptixRt (SIM_OPTIX 빌드·GPU 판, RT 코어가 있으면)
  template <class T>
  T* alloc(size_t n);
  template <class T>
  T* upload(const std::vector<T>& v);
};

// 시험용 무작위 행동 하나 (out[0..3]). 속도 행동은 정규 단위로 고르게(조금 넘게 — 자르기도 시험), 선언은 드물게(p_declare),
// act_raw 면 물리 단위 한도 안. 목표점 행동은 번호
inline void random_action(const EnvCfg& c, uint64_t& rng, float* out, float p_declare = 0.f) {
  if (c.action_mode == kActVelocity) {
    if (c.act_raw) {
      out[0] = dm::rand_range(rng, -0.3f * c.act_v_max, c.act_v_max);
      out[1] = 0.f;
      out[2] = dm::rand_range(rng, -c.act_w_max, c.act_w_max);
    } else {
      out[0] = dm::rand_range(rng, -1.2f, 1.2f);
      out[1] = dm::rand_range(rng, -1.2f, 1.2f);
      out[2] = dm::rand_range(rng, -1.2f, 1.2f);
    }
  } else if (c.action_mode == kActPath) {   // 후보 번호 (빈 칸이면 환경이 제자리 회전)
    out[0] = (float)(dm::splitmix64(rng) % (uint64_t)(c.graph_on ? mdp::graph::NB : mdp::hier::K));
    out[1] = out[2] = 0.f;
  } else {
    out[0] = (float)(dm::splitmix64(rng) % (uint64_t)(c.n_dir * c.n_dist));
    out[1] = out[2] = 0.f;
  }
  out[3] = dm::rand01(rng) < p_declare ? 1.f : 0.f;
}

// 물리 단위 (v, vy, ω) → 정규 단위 (기준선·도구가 정규 단위 환경을 몰 때)
inline void to_normalized(const EnvCfg& c, float v, float vy, float w, float declare, float* out) {
  out[0] = v >= 0.f ? v / c.act_v_fwd : v / c.act_v_back;
  out[1] = vy / c.act_vy_max;
  out[2] = w / c.act_w_max;
  out[3] = declare;
}

// 한 걸음 (src/env_kernel.cu, src/env_ref.cpp)
void env_step_gpu(const EnvBatch& b, const float* actions, cudaStream_t st = 0);   // actions: [N][mdp::act::DIM]
void env_step_cpu(const EnvBatch& b, const float* actions);
// 시험용 (tools/rl_node_check): CPU 참조판 걸음 안의 라이다 장(지도 쪽 준비 뒤 빔마다 거리·세기)과 걸음 끝 상태(판 끝 판정 뒤, 링·관측 전)를
// 밖으로 넘긴다 — 실물 노드(limo_fire_rl RobotEnv)에 같은 센서 값을 넣어 관측이 같은지 본다. 탭은 스레드 1 개(set_cpu_threads(1))일 때만 쓴다
struct CpuTap {
  virtual ~CpuTap() = default;
  virtual void scan(int env, const EnvState& e, const Shared& sh, const std::vector<float>& r, const std::vector<float>& inten) = 0;
  virtual void step_end(int env, const EnvState& e) = 0;
};
void set_cpu_tap(CpuTap* t);   // nullptr = 끔 (기본)
// CPU 참조판 스레드 수 (기본 1). 환경을 스레드에 나눠 돌린다 — 결과는 스레드 수와 무관하게 같다
void set_cpu_threads(int n);
// 시험용 (env_verify --hier): 계층형 결정 하나의 경로 계획을 CPU 참조판 그대로 (상태 e 의 경로 칸만 바꿈). b 는 CPU 묶음
void hier_plan_cpu(const EnvBatch& b, EnvState& e, int env, int slot);
// 시험용: 계층형 결정 앞 깊이 카메라 장애물 층 갱신 (CPU 참조판 그대로 — b.cvox·b.cpts 의 환경 env 칸만 바꿈, cam_on 이 아니면 아무것도 안 함)
void hier_camlayer_cpu(const EnvBatch& b, const EnvState& e, int env);
void env_reset_gpu(const EnvBatch& b, cudaStream_t st = 0);
void env_graph_gpu(const EnvBatch& b, cudaStream_t st = 0);   // 그래프 관측만 (graph_on)
void env_graph_cpu(const EnvBatch& b);
void env_reset_cpu(const EnvBatch& b);

}  // namespace sim
