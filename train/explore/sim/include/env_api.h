// 탐사 시뮬 호스트 쪽 (Env): 설정 읽기, 맵(장면 → 정답 격자) 만들기, 환경 묶음 메모리와 한 걸음 부르기, 상태 꺼내기·저장·복원.
// 같은 Env 를 GPU(많은 환경 — combine_run · dec_rollout)와 CPU(참조 — low_explore_check · room_run, 비트 비교)로 돌릴 수 있다 (tools/env_verify).
// 합친 탐사 한 걸음: set_low_cmd(위층 목표) → low_prepare(계획) → step → get_state·get_low_rl(막힘 신호)·get_ep (scripts/combine_run.cpp)
#pragma once
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "env.h"
#include "env_cfg.h"   // EnvHostCfg · load_env_cfg (CUDA 없는 호스트 코드)
#include "scene_host.h"
#include "world.h"     // World (CUDA 없는 호스트 코드도 씀 — scripts/fuel_upper.h · 실물 노드)

namespace sim {


// 장면 → 칸 격자 (env_host.cpp 위 설명). 집이 닫히지 않으면(갈 수 있는 바닥이 맵 테두리에 닿음) require_closed 면 예외,
// 아니면 closed = false·leak 에 적고 돌려줌 (덮음을 안 쓰는 도구 — 실물 녹화 다시 돌리기처럼 일부만 그린 지도)
World build_world(const std::string& name, Scene scene, const EnvHostCfg& hc, bool require_closed = true);
// 바탕 + 덧붙인 형상 (낮은 상자 변형 — 집 메시를 맵끼리 나눔): 격자는 둘을 합친 삼각형으로, 장면은 바탕 · extra 로 나눠 둔다
World build_world_overlay(const std::string& name, const std::string& base_key, Scene base, Scene extra, const EnvHostCfg& hc, bool require_closed = true);
// 경로 하나 → 맵 ("바탕.scene+덧.scene" 이면 build_world_overlay, 아니면 build_world). name 이 비면 경로
World load_world(const std::string& path, const EnvHostCfg& hc, const std::string& name = "", bool require_closed = true);

// 장면 묶음 목록 파일 (env_host.cpp) — 줄마다 장면 하나: 경로(목록 파일 기준 상대·절대) [탭 넓이 탭 벽 칸 탭 등급 …] (S2 list.tsv 꼴,
// 첫 열이 path 인 머리줄·# 주석·빈 줄 무시). skipped 를 주면 안 닫힌 장면은 버리고 이유를 넣음
std::vector<World> load_scene_list(const std::string& list_file, const EnvHostCfg& hc, std::vector<std::string>* skipped = nullptr);

// 러너가 읽는 출력 — 모두 GPU 장치 포인터 (CPU 판이면 호스트), step 이 끝나면(스트림 순서) 유효. 호스트 왕복 없음
struct StepOut {
  const uint8_t* obs_img;       // [N][mdp::obs::IMG_SIZE]
  const float* obs_1d;          // [N][o1d] — o1d = mdp::obs::POLICY_1D (기본), mdp::hier::POLICY_1D (계층형: 뒤에 후보 칸)
  const int32_t* done;          // [N] mdp::cur::Done — 종료 여부는 mdp::cur::terminal(done), 끊김 = 3
  const EpInfo* ep;             // [N] done ≠ 0 인 환경의 끝난 판 정보 (덮음·t95·길이·충돌·단계·맵)
  int o1d;                      // 관측 1D 한 환경 칸 수
};

// GPU 광선 길 (include/optix_rt.h). CPU 참조판과 비트까지 같은 것은 kRayRtFast 만 빼고 모두 — 빠른 길은 하드웨어 삼각형 교차라
// 모서리·동점에서 다른 삼각형(또는 맞음·빗나감)을 골라 통계로만 같다. GPU 결과를 --cpu·옛 빌드와 줄 단위로 견주는 도구는 정확한 길로 돈다
enum RayPath : int { kRayCpu = 0, kRaySoft = 1, kRayRtExact = 2, kRayRtFast = 3 };
inline const char* ray_path_name(int p) {
  return p == kRayCpu ? "cpu" : (p == kRaySoft ? "gpu_soft" : (p == kRayRtExact ? "rt_exact" : "rt_fast"));
}

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
  void get_obs(std::vector<uint8_t>& img, std::vector<float>& o1d) const;
  void get_cand(std::vector<int32_t>& out) const;   // 계층형 후보 목표 칸 [N][mdp::hier::K] (−1 빈 슬롯) — 도구용
  void get_cand_cur(std::vector<int32_t>& out) const;   // 후보마다 "그 덩어리가 지금 목표 칸을 품음" 비트 [N] (hier_path.h act_same — 합친 탐사 도구가 목표를 이어 갈 때)
  // 아래층 계측·단독 과제 (include/low_task.h — cfg.low_stat·low_task 일 때만): 판 누적 [N] (끄면 빈 벡터) · 환경마다 목표 칸·모드 넣기 (결정 앞) ·
  // 환경 하나의 로봇 지도 [size²] (도구가 목표를 뽑을 때)
  void get_low(std::vector<LowEp>& out) const;
  void set_low_cmd(const std::vector<LowCmd>& cmd);
  void get_map_env(int env, std::vector<int16_t>& out) const;
  // 아래층 걸음 길 (cfg.low_rl — include/low_task.h, low_rl.h): step 한 번 = 걸음 하나(0.1 s). 호스트가 걸음마다
  // set_low_cmd → low_prepare(다음 step 에 줄 행동 줄 — 규칙 추종 환경의 후보 번호, 없으면 nullptr) → step
  void low_prepare(const float* actions = nullptr, cudaStream_t st = 0);
  void get_low_cmd(std::vector<LowCmd>& out) const;
  void get_low_rl(std::vector<LowRl>& out) const;
  void get_ep(std::vector<EpInfo>& ep) const;
  void get_done(std::vector<int32_t>& done) const;   // 판 끝 코드 [N] (mdp::cur::Done — 끝난 판은 리셋됨)
  // 움직이는 사람 (src/people_host.cpp): 환경마다 다음 판 사람 수 상한 (판 시작 인자, 0 ~ kMaxPeople) · 이 판 사람들 (뷰어 재생)
  void set_people_max(const std::vector<int>& k);
  void get_people(std::vector<People>& out) const;
  // 환경 하나의 실행 상태 저장·복원 (2026-11-17, 판단 모델 결과 라벨의 갈래 굴리기 — upper_lora/rollout): 상태(사람 포함)·지도·지나간 길·관측·후보·아래층·작업 판.
  // 같은 설정(EnvHostCfg · 장면 · n_env 무관)의 Env 끼리 주고받을 수 있음. CPU · GPU 같은 꼴 (GPU 는 장치 ↔ 호스트 복사). 맵·장면 같은 정적 업로드는 안 담음
  std::vector<uint8_t> env_blob(int env) const;
  void set_env_blob(int env, const std::vector<uint8_t>& blob);
  void copy_env(int src, const std::vector<int>& dst);   // 한 환경 상태를 다른 환경 칸들로 (장치 안 복사)

  const EnvBatch& batch() const { return b_; }
  int n_env() const { return b_.n_env; }
  int ray_path() const { return !gpu_ ? kRayCpu : (!b_.rt ? kRaySoft : (b_.rt_fast ? kRayRtFast : kRayRtExact)); }
  bool bit_exact() const { return ray_path() != kRayRtFast; }   // CPU 참조판과 비트까지 같다고 약속하는 길인가

 private:
  bool gpu_;
  EnvBatch b_{};
  std::vector<void*> owned_;            // 장치 메모리
  std::vector<std::vector<uint8_t>> host_;   // CPU 판 메모리
  std::vector<std::pair<void*, size_t>> per_env_;   // 환경마다 따로인 버퍼 (시작 · 환경 하나 몫 바이트) — env_blob · copy_env
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
    out[0] = (float)(dm::splitmix64(rng) % (uint64_t)mdp::hier::K);
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
// 아래층 준비 단계 (include/low_rl.h 머리말 — 호스트가 LowCmd 를 넣은 뒤 부른다. Env::low_prepare)
void env_low_prepare_gpu(const EnvBatch& b, const float* actions, cudaStream_t st = 0);
void env_low_prepare_cpu(const EnvBatch& b, const float* actions);
// 시험용 (tools/low_explore_check · room_run): CPU 참조판 걸음 안의 라이다 장(지도 쪽 준비 뒤 빔마다 거리·세기)과 걸음 끝 상태(판 끝 판정 뒤, 링·관측 전)를
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
void env_reset_cpu(const EnvBatch& b);

}  // namespace sim
