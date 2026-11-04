# network — 신경망 (2D 관측마다 CNN + MLP, GPU 정방향·역방향)

C++/CUDA 만 쓴다(MLP 는 cuBLAS GEMM, 합성곱 가지는 직접 짠 융합 커널 — cuDNN 은 이 PC 에 시스템 설치가 없어 안 씀).
빌드·검증 도구는 `ppo/CMakeLists.txt` 가 같이 만든다(`ppo/README.md`).

| 파일 | 내용 |
|---|---|
| `common.cuh` · `common.cu` | 오류 검사, 행 우선 GEMM(cuBLAS), 결정적 축소(열 합·전체 합 — 원자 연산 없음). `real` 형(fp32, 검사용 fp64 한 벌) |
| `gemm_fused.cuh` · `gemm_fused.cu` | MLP 용 직접 짠 GEMM + 에필로그 융합 (bf16 텐서 코어 또는 fp32 정확 모드) — 아래 "MLP GEMM" |
| `actor_critic.h` · `actor_critic.cu` | `Net`(CNN 가지들 + MLP), `Normalizer`(1D 실행 평균·분산), `ActorCritic`(actor·critic·log σ 를 변수 버퍼 하나에), `explore_spec()` |
| `cnn_fused.cuh` · `cnn_fused.cu` · `cnn_fused_tc.cuh` | 합성곱 가지 하나(3 층 + 편향 + ELU)를 커널 하나로 — 정방향·역방향, fp32 판(정확 경로)과 텐서 코어 판(bf16 입력·fp32 누산, 학습 기본) |
| `distribution.cuh` | 행동 분포: 가우시안 3(v, vy, ω) + 베르누이 1(끝 선언) — log π·엔트로피·KL (`__device__`) |

입력·출력 크기는 `mdp/obs_layout.h` 만 보고 맞춘다(`explore_spec()`).

## 구조

```
                     공유 인코더 (actor·critic 이 같이 씀)
가까운 지도 u8 6×64×64 ─/255→ conv 2×2 s2 16 → 32 → 32 → 32, ELU → 32×4×4 = 512 ┐
먼 지도     u8 7×32×32 ─/255→ conv 2×2 s2 16 → 32 → 32 → 32, ELU → 32×2×2 = 128 ├ F (1024) ─┬→ [F | 정책 1D 1304] → actor MLP 512-256-128 ELU → v·vy·ω 평균 3 + 선언 로짓 1
깊이        u8 1×40×64 ─/255→ conv 2×2 s2 16 → 32 → 32 → 32, ELU → 32×3×4 = 384 ┘            │                                       (log σ 3 개는 상태 무관 변수)
                                    (넷째 층 앞에서 5×8 격자를 0 행 덧대 6×8)            └→ [F | 참값까지 1D 1336] → critic MLP 512-256-128 ELU → 가치 1
```

- 학습 변수 약 276 만 개(11.0 MB, 옛 구조 565 만 개의 반): 인코더 약 6.5 만, actor·critic MLP 첫 층이 2328·2360 → 512.
- **공유 인코더(구조 버전 2, 2026-11-04)**: 합성곱 가지를 actor·critic 이 같이 쓴다 — 한 번만 계산하고, 두 MLP 의 특징 기울기를 더해(고정 순서) 인코더로 흘린다(손실은 지금과 같은 합 손실).
  Stable-Baselines3 의 PPO `CnnPolicy` 기본이 같은 꼴이다("The CNN is shared between actor and critic for A2C/PPO (on-policy algorithms) to reduce computation", SB3 문서 Policy Networks).
  주의: rsl_rl(2.3.1)의 `ActorCritic` 은 actor·critic 을 따로 둔다(공유 아님), 상태 입력 연구(Andrychowicz et al., "What Matters in On-Policy RL", 2021)는 따로 둔 쪽이 낫다고 보고 —
  그래서 학습 곡선을 옛 구조와 같은 조건으로 비교해 확인했다(ppo/README "구조 버전 2").
- **합성곱 4 층**: 해상도 1/16(64→4, 32→2, 40×64→3×4) — 평탄화 3 840 → 1 024, MLP 첫 층 입력 5 144 → 2 328 (actor). 넷째 층 수용 영역 16×16 칸(가까운 지도 1.6 m, 먼 지도 12.8 m).
  깊이 40 행은 16 의 배수가 아니라 셋째 층 뒤 5 행을 0 행 하나로 덧대 6 행으로(덧댄 자리는 입력 0 — 합성곱 0 덧댐과 같은 뜻).
  앞 세 층은 융합 커널(아래 "계산 방식"), 넷째 층은 덧댄 im2col + GEMM + 편향·ELU(`Net::tail_forward/backward`).
- 체크포인트 형식 버전 2 (층 이름도 `enc.cnn*`) — 버전 1(옛 구조) 체크포인트는 불러오기를 거부한다.
- **MLP [512, 256, 128] · ELU**: legged_gym 기본값 그대로 — `legged_gym/envs/base/legged_robot_config.py:207-209`
  (`actor_hidden_dims = critic_hidden_dims = [512, 256, 128]`, `activation = 'elu'`, [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) master, 2026-11-04 확인).
  actor·critic 따로인 것도 rsl_rl `modules/actor_critic.py:42-55`(2.3.1) 와 같다.
- **σ**: 상태 무관 변수. rsl_rl `actor_critic.py:69-72` 의 `noise_std_type="log"` 꼴(`log_std` 변수), 초기 σ = 1
  (`init_noise_std = 1.0`, legged_gym `legged_robot_config.py:206`). tanh 없음 — log π 는 자르기 전 값(mdp/README "행동").
- **선언 로짓 편향 초기값 −7** (actor 마지막 층 편향의 마지막 원소): 걸음마다 뽑으므로 0 이면 판이 평균 2 걸음에 끝난다(mdp/README "행동", 범위 −6~−8 중간).
- **초기화**: torch `nn.Linear`·`nn.Conv2d` 기본(가중치·편향 U(±1/√fan_in) — kaiming_uniform a=√5 와 같은 경계), rsl_rl 은 따로 초기화하지 않는다.
- **CNN: 2×2 stride 2 합성곱 (버전 1 은 3 층·망마다 따로, 버전 2 는 위 공유 4 층), 채널 16-32-32(-32)**
  - 버전 1 의 층 수 3: 해상도를 1/8 로 줄여 평탄화(3840)를 1D(1304)와 비슷한 규모로 뒀다. 버전 2 는 MLP 첫 층(반복 시간의 큰 몫)을 줄이려 1/16. 공간 배치는 평탄화로 MLP 까지 그대로 간다(전역 풀링 안 함 — 지도에서 "어디에" 가 중요).
  - 2×2 stride 2(겹치지 않는 조각): ConvNeXt 의 내려받기 층과 같은 꼴(Liu et al., "A ConvNet for the 2020s", arXiv:2201.03545 — 단계 사이 2×2 s2 합성곱).
    처음엔 3×3 s2 pad 1 로 짰으나 im2col 이 입력을 9/4 배로 불려 메모리 대역이 병목이었다(장난감 환경 1024 개 반복 9.0 s → 2×2 계층 배치 5.3 s). 2×2 로 바꾸면 조각이 안 겹쳐
    아래 "계층 배치"로 복사 없이 GEMM 하나가 된다. 3 층 뒤 수용 영역은 8×8 칸(가까운 지도 0.8 m, 먼 지도 6.4 m); 그 위는 MLP 가 섞는다.
  - 채널 16-32-32: 과제 예시값이자 작은 입력(64×64 이하)용으로 흔한 크기. 첫 층 출력이 활성값 메모리의 대부분(가까운 지도 16×32×32)이라 첫 층을 16 으로 둔다.
  - 잠정: 본 학습 곡선을 보고 바꿀 수 있다(바꾸면 체크포인트가 깨진다 — 층 이름·크기를 체크포인트가 확인한다).
- **정규화**: 1D 관측만, rsl_rl `modules/normalizer.py` EmpiricalNormalization 과 같은 식(eps 1e-2, `(x−μ)/(σ+eps)`, 배치 평균·분산을 개수 비율로 섞음, until 없음).
  통계는 CRITIC_1D 차원 하나로 두고 actor 는 앞 POLICY_1D 차원을 쓴다 — 차원마다 따로 세는 통계라 actor·critic 정규화기를 따로 둔 것과 수학적으로 같다.
  롤아웃 중 `act` 마다 갱신, 평가(`act_inference`)·마지막 가치(`compute_returns`)에서는 안 바꾼다. 체크포인트에 들어간다.
- 2D 는 u8 로 받아 첫 커널에서 /255.

### 그림 (Netron)

예전 구조 버전 2 의 그림·ONNX 는 지금 구조와 많이 달라 `archive/explore_net_v2/` 로 옮겼다(2026-11-04). 지금 구조로 다시 만들려면
(Python 없이, protobuf 를 와이어 형식으로 직접 씀 — `network/tools/onnx_export.cpp`):

```bash
cmake --build <빌드> --target onnx_export                                    # ppo/CMakeLists.txt
<빌드>/onnx_export --out /tmp/net.onnx                                         # 모양만 (구조는 explore_spec() 그대로)
<빌드>/onnx_export --out /tmp/net.onnx --ckpt train/explore/runs/<실험>/ckpt/model_latest.bin   # 가중치·정규화 통계까지
# 그림: netron.app 에 끌어 놓고 전체 그래프 화면을 PNG 로 (브라우저 안에서 읽음, 서버로 안 올림)
```

합성곱 가중치는 체크포인트의 `[cout][자식 dy·2+dx][cin]` 을 ONNX `[cout][cin][kh][kw]` 로 바꿔 넣는다(계층 배치 합성곱 = 보통의 2×2 s2 합성곱).
평탄화 순서가 `[픽셀(래스터)][채널]` 이라 Flatten 앞에 NHWC Transpose 를 둔다. 주의: ONNX 쪽 출력을 우리 CUDA 정방향과 수치로 맞춰 본 적은 없다(그림·구조 확인용).

## 계산 방식

- **합성곱 = GEMM 하나 (계층 배치)**: 픽셀을 "부모 조각 순서"(쿼드트리)로, 채널을 맨 뒤에 둔다. 그러면 층 입력 `[n·P_out][4·cin]` 이 그대로 im2col 행렬이고
  출력 `[n·P_out][cout]` 이 다음 층 입력 배치와 같다 → 정방향 `Y = X·Wᵀ`, 역방향 `dW += dZᵀ·X`, `dX = dZ·W` 가 모두 복사 없는 GEMM. u8 입력만 처음 한 번 이 배치로 바꾼다(`k_reorder_u8`).
  합성곱 가중치 배치는 `[cout][자식 2×2][cin]` (체크포인트도 이 순서).
- MLP 는 `[샘플][특징]`. **MLP GEMM** (`gemm_fused.cu`, `Net::mlp_mode` = `PPOConfig::mlp`):
  - 정방향 `Y = ELU(X·Wᵀ + b)` 를 커널 하나로(편향·ELU 가 GEMM 에필로그). 역방향 `dW += dYᵀ·X` 커널이 같은 K 루프에서 편향 기울기 Σ dY 도 더하고,
    `dX = (dY·W) ⊙ ELU'(H)` 커널이 ELU 미분을 에필로그로 곱한다 → MLP 의 편향+ELU·ELU 미분·열 합 커널이 없다. 첫 층 dX 는 CNN 열(3840)만 — 1D 입력 기울기는 안 쓴다.
  - 모드 2(기본) = bf16 텐서 코어(WMMA 16×16×16, 입력을 공유 메모리에 넣을 때 bf16 반올림, fp32 누산·fp32 에필로그). RTX 3060 은 TF32 텐서 코어가 fp32 SIMT 와 최고 속도가 같고(12.7 TFLOPS)
    fp16/bf16·fp32 누산만 두 배(25.5 TFLOPS)라 bf16 을 쓴다. 모드 1 = 같은 타일·같은 에필로그의 fp32 FMA(정확 모드, `gradcheck` 기본). 모드 0 = 옛 cuBLAS sgemm + 따로 커널.
  - 타일: 128×64 블록이 4 물결(SM 28 × 4 = 112) 이상이면 128×64(워프 4 개, 워프당 64×32), 아니면 64×64 — `gemm_bench` 실측으로 정함.
  - 블록이 모자라고 K 가 긴 GEMM 은 split-K (블록 × 조각 ≥ 224, 조각마다 K 타일 4 개 이상, 최대 16): 날값을 작업 공간 [조각][M][N] 에 쓰고 마무리 커널이 조각 순서대로 더한 뒤 에필로그(원자 연산 없음).
    첫 층 정방향(덩이 1024 일 때 128×64 블록 64 개)과 작은 층 dW(K = 덩이 샘플 수 4096, 블록 2~32 개 — 편향 기울기 조각도 같이 모아 더함)에 걸린다.
    `gemm_bench --n 4096`: 512→256 dW 0.179 → 0.103 ms, 256→128 0.169 → 0.042, 128→4 0.179 → 0.022 (편향 기울기 vs host 합 상대 오차 ≤ 3e-7).
  - 가중치 bf16 사본(`ActorCritic::d_params_h`, Adam·불러오기·초기화 뒤 `params_changed()` 가 새로): bf16 정방향·dX 가 W 를 bf16 으로 바로 읽는다(읽는 바이트·레지스터 반).
    반올림이 로더의 즉석 변환과 같아 결과는 비트 단위로 같다(`toy_train` 해시 그대로). `gemm_bench --n 4096`: 첫 층 dX 1.50 → 1.445 ms, 세 층 합 4.75 → 4.66 ms.
    B 레지스터가 반이 되어 큰 정방향·dX(128×128 블록 112 개 이상, K 또는 N ≥ 1024)는 128×128 타일: 첫 층 정방향 1.29 → 1.20 ms(18.0 TFLOPS), dX 1.43 → 1.31 ms(16.5), 세 층 합 4.66 → 4.42 ms.
  - 큰 dW 는 cp.async 2 단(BK 16, 공유 41 KB → SM 당 블록 2 개): fp32 를 레지스터 없이 공유 메모리로 받고 안에서 bf16 으로 바꿔 WMMA. 첫 층 dW 1.46 → 1.29 ms(16.6 TFLOPS), 결과 비트 같음.
  - 큰 dX 도 cp.async(A=dY fp32 2 단 + 변환, B=W bf16 사본은 그대로 3 칸): 첫 층 dX 1.31 → 1.18 ms(18.3 TFLOPS), 결과 비트 같음. 정방향은 1.20 → 1.22 ms 로 이득 없어 그대로. 세 층 합 4.42 → 4.10 ms.
  - 결정성: 타일·K 순서 고정, 편향 기울기는 스레드별 부분합을 고정 순서로 합침(원자 연산 없음) → 같은 입력이면 비트 단위로 같음.
- **합성곱 = 가지마다 융합 커널 하나** (`cnn_fused.cu`): 2×2 stride 2 를 3 번 거친 출력 픽셀 하나는 입력 8×8 조각 하나에만 기댄다(조각끼리 안 겹침).
  그래서 블록이 조각 몇 개(타일)를 맡아 u8 입력을 바로 읽고(/255 는 첫 층 누산 뒤에 곱함) 층 0·1·2 + 편향 + ELU 를 공유 메모리·레지스터 안에서 끝내고 평탄화 결과만 MLP 입력(X0)에 쓴다.
  조각 안은 "부모 조각 순서"(쿼드트리)·채널 맨 뒤 배치라 층 입력 `[행][4·cin]` 이 그대로 im2col 행렬 — 공유 메모리 안에서 복사 없이 다음 층 GEMM 의 입력이 된다.
  - 역방향: 같은 타일을 다시 정방향해(층 0·1 활성값을 전역에 안 둠 — 재계산이 전역 왕복보다 싸다; 층 2 출력은 X0 에 있는 것을 씀) dW·db 를 블록이 레지스터에 쌓는다.
    블록은 상주(격자 = SM 수 × 상주 블록 수, 기기마다 고정)하고 타일을 돌아가며 맡아, 끝에 블록별 부분합 → 블록 순서대로 더함(원자 연산 없음 → 결정적). 첫 층 입력 기울기는 안 구함(u8).
  - **텐서 코어 판(학습 기본, `PPOConfig::cnn_bf16 = true`)**: 활성값·가중치를 공유 메모리에 bf16 으로, `ldmatrix`(전치가 필요한 곱은 `.trans`) + `mma.sync m16n8k16` (fp32 누산).
    u8 0~255 는 bf16 에 정확히 들어간다. 정밀도: fp64 기준과 상대 오차 합성곱 층 ~1e-2·MLP 층 ~1e-3 (fp32 판은 ~1e-6) — 장난감 학습 곡선은 fp32 판과 같다(ppo/README "속도").
  - **fp32 판** (`cnn_bf16 = false`): 같은 구조를 fp32 FMA 로. 기울기 검사(`gradcheck`)가 이 경로를 본다. 채널이 16 의 배수가 아닌 모양(검사용 작은 구조)도 이 판.
  - 지원 안 하는 모양(층 수 ≠ 3 등)·fp64 검사 빌드는 기준 경로(합성곱 = cuBLAS GEMM, 입력을 계층 배치로 바꾸는 `k_reorder_u8`)로 돈다 — fp64 기준이 융합 커널과 독립인 구현이라 검사가 서로를 본다.
  합성곱 가중치 배치는 `[cout][자식 2×2][cin]` (체크포인트도 이 순서, 융합 커널도 그대로 읽는다 — 형식 안 바뀜).
- MLP 는 `[샘플][특징]`, cuBLAS sgemm(fp32, TF32 안 씀).
- **actor ∥ critic 두 스트림**: `ActorCritic::stream2`·`cublas2` 에서 critic 정방향·역방향을 actor 와 동시에(PPO 가 fork/join 이벤트로 잇는다).
  둘은 쓰는 버퍼·기울기 칸이 겹치지 않고 원자 연산이 없어 결과는 한 스트림과 비트 단위로 같다(`toy_train` 해시 같음). 열 합·전체 합의 작업 공간은 스트림마다 따로.
- **결정성**: 원자 연산 없음. 편향 기울기·통계는 고정 순서 두 단계 축소, cuBLAS 는 한 스트림. → 같은 씨앗이면 비트 단위로 같은 결과(`toy_train --check-determinism`).
- **덩이(chunk)**: 신경망 작업 공간은 `max_chunk` 샘플(기본 4096, `PPOConfig::chunk`) 만큼만 잡고, 그보다 큰 배치는 나눠 기울기를 더한다(같은 식 — 더하는 순서만 달라 마지막 비트가 다를 수 있음, 덩이가 같으면 비트 같음).
  작업 공간: actor·critic 각각 0.05 GB (덩이 1024 일 때, 융합 커널 — 합성곱 중간 활성값이 전역에 없음; 기준 경로는 0.60 GB).
- 역방향은 직전 정방향과 같은 `n`·입력으로 부른다(융합 커널은 u8 입력을 다시 읽고 X0 의 층 2 출력을 쓴다).

## 검증 (도구는 `ppo/tools`, 출력은 `ppo/README.md`)

- `gradcheck`: fp64 한 벌(같은 소스, `XNET_FP64`)로 중앙 차분 vs 해석 기울기, fp32 해석 기울기 vs fp64 해석 기울기 — 모든 층 종류(u8 합성곱 첫 층, 합성곱, 선형, ELU, log σ, PPO 손실 커널, 덩이 누적).
- fp32 만으로 직접 차분하면 손실 반올림 잡음(~1e-7/h)이 작은 기울기 층(합성곱 편향 등)에서 1e-3 을 넘어(h 를 키우면 PPO 자르기·ELU 꺾임에 걸림) 기준으로 못 쓴다 — 그래서 차분은 fp64 로 한다.
