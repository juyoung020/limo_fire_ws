# scripts — 실행 파일

| 파일 | 내용 |
|---|---|
| `train.cpp` | 학습 러너: 장면 목록으로 환경 N 개 → 24 걸음 롤아웃 → PPO update 반복, 뷰어 실행 폴더(`run.json kind: explore_ppo`)·체크포인트·시험 맵 평가. 빌드·옵션은 아래 |
| `play.cpp` | 학습한 정책 돌려 보기 |
| `frontier_run.cpp` | 프론티어 탐사 기준선을 장면 × 판으로 돌려 뷰어 실행 폴더(`run.json kind: baseline`)로 쓴다. 빌드·옵션은 [baselines/frontier](../baselines/frontier/README.md) · `--modes slot0`(·`slot0_nosafe`) 는 계층형 규칙 정책(늘 0 번 후보)을 같은 판으로 |
| `oracle_run.cpp` | **측정 도구 — 정답을 씀.** 학습 맵 단계마다 정답 격자 경로·참 자세·정답 사람 위치로 모는 oracle(`--policy oracle`, 목표는 로봇 지도 굵은 프론티어, 성공 정의 그대로)과 slot0 규칙 정책을 같은 판으로 돌려 성공 상한을 잰다(`--stage 0..3 --people on\|off --out 판.tsv`). 시뮬은 env_api 로만 몰고 정답은 러너 안에서만 — 학습 환경·관측과 무관. 빌드는 baselines/frontier, 결과는 [mdp/README](../mdp/README.md) "커리큘럼" |
| `low_run.cpp` | 아래층 기준선(mdp/README "아래층 계측·단독 과제") — 빌드는 baselines/frontier 의 `low_run`. 여기 `CMakeLists.txt` 의 `low_policy_run` 은 같은 파일에 정책 걸음 루프를 넣은 것: `low_policy_run --mode follow --policy <체크포인트> --scenes …` 가 `train --low` 정책을 기준선과 같은 단독 과제(예열 탐사 → 경로 3~15 m 목표, 사람 0·2 차례로)로 재고 ROADMAP "2 단계 통과 조건" 줄마다 통과·불통과를 낸다. `--policy` 대신 `--step` 이면 같은 걸음 루프에서 규칙 추종기(대조) |
| `combine_run.cpp` | 두 층을 한 판에서 합쳐 탐사를 끝까지: 위층 0 번 후보 규칙(1 Hz) + 아래층 RL 정책(10 Hz, `--policy <체크포인트>`) 또는 규칙 추종기(`--policy` 없음 — 같은 걸음 루프의 대조). 막힘 신호(`LowRl::stall`)를 받아 그 칸을 한동안 피한다. `combine_run --scenes a.scene b.scene [--policy …] --episodes 256 --seed 1 --people 0 [--rule-near 2]` — 결과·원인 분석은 mdp/README "아래층 RL — 위층 규칙과 합친 탐사". 위층을 바깥 판단 모델에 맡기는 실험(`--upper decider`)과 한 판 녹화(`--video`)는 아래 "combine_run — 위층을 판단 모델에 맡기기" |
| `decider.h` · `decider.cpp` | 판단 모델 서버(llama.cpp `/v1/systemone`)에 묻는 쪽: 요청 글·사진 만들기, 여러 요청 한꺼번에 보내기, 점유 격자만으로 남은 프론티어 세기(`grid_open_frontier`). 설정 글은 `config/decider.yaml` |

## train (`train.cpp`, 빌드 `CMakeLists.txt`)

```bash
cmake -S train/explore/scripts -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build <빌드> -j8 --target train
# 저장소 뿌리에서 (기본 경로가 train/explore/… 상대)
<빌드>/train --name explore_s1 --seed 1 --minutes 120 --eval-every 50 --ckpt-every 50
<빌드>/train --name explore_s1 --resume --minutes 60      # ckpt/model_latest.bin 에서 이어 (실행 폴더도 그 반복으로 되감음)
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--envs` | 2048 (`--graph` 는 512) | 학습 환경 수 (2048 에서 GPU 5.6 GB) |
| `--iters` · `--minutes` | 0 · 0 | 반복 수·시간 한계 (둘 다 0 이면 1000 반복). 끝·SIGINT·SIGTERM 이면 체크포인트 저장 후 끝 |
| `--name` · `--root` | explore · train/explore/runs | 실행 폴더 `<root>/<name>` |
| `--seed` | 1 | 시뮬·정책 씨앗 |
| `--lr-max` | 1e-3 (`--graph` 는 1e-4) | KL 적응 학습률 상한 (ppo/README "검증 2": rsl_rl 기본 1e-2 는 장난감에서 critic 이 터짐) |
| `--eval-every` · `--eval-episodes` | 50 · 32 | 시험 맵(`--test`, 기본 scenes/test 의 bookstore·small-house)마다 별도 Env, 결정적 정책, 환경마다 첫 판 |
| `--ckpt-every` | 50 | `ckpt/model_<반복>.bin`, 그리고 10 반복마다 `ckpt/model_latest.bin` |
| `--replay-every` | 25 | 이 반복마다 표본 환경(0~63)에서 새로 시작하는 판 2 개를 재생 파일로 (동시 녹화 최대 8 판 — 끊김 판(12 000 걸음)이 녹화 칸을 오래 잡아도 새 녹화가 계속 붙음) |
| `--list` · `--env` · `--model` | scenes/train/list.tsv · sim/config/env.yaml · sensor_model.yaml | 학습 맵 목록·시뮬 설정 |
| `--hier` | 끔 | 계층형(mdp/README "계층형"): 1 s 마다 프론티어 후보 8 개 중 하나(가린 범주 분포, `xnet::explore_spec_hier`) → 시뮬이 경로 계획·추종. 반복 = 결정 24 개(= 시뮬 240 걸음), γ = 0.999¹⁰, 걸음/s 는 시뮬 걸음 단위. 재생 녹화는 아직 없음(`--replay-every` 무시) · 체크포인트는 기본 구조와 안 섞임(층 이름이 다름) |
| `--graph` | 끔 | 그래프 모드(network/README "그래프 어텐션 정책", HEADER 식): 관측 = 노드 그래프(32×32 격자·특징 5·간선 비트), 행동 = 이웃 노드 하나(최대 24, 가린 범주 분포, `xnet::explore_spec_graph`). 반복 = 환경마다 결정 16 개, 결정 하나의 시뮬 걸음 수는 가변(실측 평균 33 걸음)이라 걸음/s 는 `get_gdec` 의 실제 걸음 합으로 센다. PPO 값은 `xppo::graph_ppo_config()` 한곳(γ 0.95 결정당·환경 512·에포크 3·학습률 1e-5 시작, 범위 1e-6~1e-4 — ppo/README "그래프 모드 학습 속도"). `--graph-<이름> <값>` 은 시뮬 그래프 설정(`sim::graph_arg`), `--graph-share 1` 은 가치 몸통 공유(갱신 약 2 배 빠름, 구조가 바뀌므로 기본 끔). 재생 녹화는 아직 없음. `--hier` 와 같이 못 씀 |
| `--low` | 끔 | 아래층 RL(mdp/README "아래층 RL", `xnet::explore_spec_low`): 단독 과제를 정책이 10 Hz 로 — 행동 (v, ω), 환경이 목표를 스스로 뽑음. 환경 기본 1024(학습 중 GPU 메모리 약 4.4 GB, 반복 하나 24 걸음 약 2.2 s) · γ 0.99 · 목표 KL 0.01(둘 다 rsl_rl 기본값, 잠정) · `--people 2`(판마다 사람 0 ~ 이 수) · 커리큘럼 없음(모든 맵 고르게). progress.jsonl 은 `low/reach_rate`·`low/speed`(통과한 목표의 계획 길이 ÷ 걸린 시간)·100 m 당 충돌·안전층 걸음 비율과 `reward/<항>`(걸음당). 평가는 시험 맵마다 60 s(앞 절반 사람 0 · 뒤 절반 0~2, 결정적 + 샘플) — 학습 곡선용이고 기준선과 견주는 값은 `low_policy_run` |
| `--steps` · `--epochs` · `--minibatches` · `--gamma` · `--lam` | 모드 기본 | 롤아웃 길이(환경마다 결정/걸음 수)·에포크·미니배치·할인율·GAE λ 덮어쓰기 (`--envs × --steps` 가 `--minibatches` 로 나눠져야 함) |

- done 매핑: `mdp::cur::terminal` (1 선언 성공·2 벽 충돌·4 선언 실패·5 사람 충돌) → PPO 1 종료, 3 → 끊김(가치로 메움).
- progress.jsonl (반복마다): `reward/<항>`(끝난 판의 항별 합 평균), `rollout/success_rate`·`collision_rate`(벽+사람)·`person_collision_rate`·`declare_short_rate`(선언 실패)·`truncation_rate`·`coverage_mean`·`coverage_at_declare_mean`·`t95_mean`·`t95_reached_rate`·`ep_len_mean`(걸음)·`ep_rew_mean`, `curriculum/stage`(끝난 판 단계 평균), `train/*`(learning_rate·approx_kl·value_loss·policy_gradient_loss·entropy_loss·grad_norm·std), `time/*`(fps·iter_ms·rollout_ms·update_ms + sim_ms·act_ms·host_ms), `gpu/mem_used_mb`, 평가 반복에는 `eval/*`(success_rate·collision_rate·t95_mean·coverage_mean … — frontier 실행의 `eval/frontier_t95_mean` 과 나란히).
- episodes.jsonl: 표본 환경 0~63 의 판마다 한 줄(맵·단계·결과·덮음·t95·항별 보상). evals/test_it<반복>.json: 시험 맵별 성공·충돌·끊김·선언 미달·끝 덮음·t95.
- 재생 파일은 걸음마다 참 자세·지도 자세·잰 속도, 정책 행동 v·vy·ω(물리 단위, 환경이 자르는 대로)·선언 확률·뽑은 선언, 가치 V(s), 보상 항 7 개(`r_<항>`)와 누적 리턴,
  덮음, 사건, 움직이는 사람 원기둥(`h<k>_x/_y/_r`), 참 자세에서 다시 쏜 깨끗한 라이다(frontier_run 과 같은 꼴).
  로봇이 만든 지도(로그 오즈 → 점유 %)와 본 벽 면 칸(덮음 겹)은 10 걸음마다 그 환경 칸만 장치에서 꺼내 넣는다(녹화 중인 판만 — 2048 환경 전체를 안 꺼냄).
- run.json `robot`: 시작할 때 sim 충돌 몸통·센서 자리(env.yaml body, 깊이 카메라 수평 시야는 sensor_model 의 fx·유효 열)와 URDF(`src/limo_fire/limo_fire_description/urdf` 의 xacro 를 그대로 읽음 — 차체 상자·바퀴 4 개·라이다·카메라 자리)를 적는다.
  뷰어 재생은 이것으로 위에서 본 로봇을 그린다. 2026-11-04 값: sim 몸통 0.32 × 0.22 m(nav2 footprint) ↔ URDF 차체 상자 0.13 × 0.12 m(단순 충돌 상자) — 다르다, 화면은 sim 몸통 실선 + URDF 상자 점선.
  바퀴 (±0.10, ±0.065) 지름 0.09·폭 0.045, 라이다 (0.103, 0) 둘이 같음, 카메라 수평 자리 (0.15, −0.005) 둘이 같음(높이·숙임은 sim 이 실측 0.1881 m·6.6°, URDF 0.205 m·0° — 위에서 볼 땐 안 보임).
- 이어 학습은 PPO 체크포인트(가중치·Adam·정규화·반복)만 잇고 시뮬은 새 판으로 시작한다(커리큘럼 단계도 0 부터).


## combine_run — 위층을 판단 모델에 맡기기 (`--upper decider`, 실험 중)

두 층 탐사(위층 1 Hz 후보 고르기 + 아래층 RL 10 Hz)에서 위층의 "어느 후보로 갈까"를 바깥 판단 모델 서버에 묻는다(`decider.h` · `decider.cpp`).
모델은 글을 만들지 않고 선택지·예/아니오·점수 단계의 확률만 낸다. 미세조정 없이(제로샷) 붙인 것이고, **지금까지 규칙(늘 가장 가까운 후보)보다 나아진 것은 없다** — 아래 "결과".

### 준비: 모델과 서버 (저장소에 없음)

- **모델 가중치는 저장소에 넣지 않는다.** LiquidAI d1-3B 의 GGUF 두 파일을 각자 받는다: <https://huggingface.co/LiquidAI/d1-3B-GGUF> 의
  `d1-3B-Q8_0.gguf`(2.9 GB) · `mmproj-d1-3B-Q8_0.gguf`(0.6 GB, 사진 입력용). 이 PC 는 `~/models/decision/d1-3b/` 에 뒀다.
- **라이선스**: LFM Open License v1.0 (받은 폴더의 `LICENSE`). 쓰기 전에 본문을 직접 읽는다 — 조건을 여기 옮겨 적지 않는다.
- **서버**: llama.cpp(<https://github.com/ggml-org/llama.cpp>) 의 `llama-server`. `/v1/systemone` 길이 있는 판이어야 한다(이 PC 는 커밋 `31953bc`, CUDA 빌드).

```bash
llama-server -m ~/models/decision/d1-3b/d1-3B-Q8_0.gguf --mmproj ~/models/decision/d1-3b/mmproj-d1-3B-Q8_0.gguf \
  --host 127.0.0.1 --port 8171 -ngl 99 -c 32768 -np 16
```

`-np 16` 은 한꺼번에 받는 요청 수(`--decider-parallel` 과 맞춤), `-c` 는 그 16 칸이 나눠 쓰는 문맥 길이. RTX 3060 12 GB 에서 서버 약 4.0 GB, 시뮬과 합쳐 4.7 ~ 5.2 GB.
빌드에는 libcurl · zlib · nlohmann_json 개발 패키지(`libcurl4-openssl-dev` · `zlib1g-dev` · `nlohmann-json3-dev`), 녹화에는 `ffmpeg` 가 있어야 한다.

### 돌리기

```bash
cmake -S train/explore/scripts -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build <빌드> -j8 --target combine_run scene_lowobs
cd train/explore/sim
T=../scenes/test; CK=../runs/<아래층 실험>/ckpt/model_latest.bin
# 기준선 (위층 규칙)
<빌드>/combine_run --people 0 --policy $CK --scenes $T/bookstore.scene $T/small-house.scene --episodes 24 --seed 1 --out <접두사>
# 판단 모델: 막힌 후보만 건너뛰기 (prune) · 후보 하나 고르기 (choice)
<빌드>/combine_run --people 0 --policy $CK --scenes $T/bookstore.scene $T/small-house.scene --episodes 24 --seed 1 \
  --decider-url http://127.0.0.1:8171 --decider-input all --decider-mode prune --decider-ask-m 2 --decider-timeout-ms 120000 --out <접두사>
# 한 판을 영상으로 (환경 0, 위에서 본 지도 + 카메라 그림 + 두 층의 출력 글)
<빌드>/combine_run --people 0 --policy $CK --scenes $T/small-house.scene --episodes 1 --seed 1 \
  --decider-url http://127.0.0.1:8171 --decider-input all --decider-mode prune --decider-ask-m 2 --decider-timeout-ms 120000 --video <접두사>
```

모델에 가는 것(`--decider-input all`): 로봇 카메라 자리에서 본 장면 그림, 위에서 본 로봇 지도 사진(후보에 글자 A~H), 후보 표(경로 거리·방향·폭·예상 관측량).
카메라 그림은 장면 형상을 참 자세에서 그린 것이지 진짜 컬러 영상이 아니다. 서버 오류·시간 끝·답 꼴 오류면 그 결정은 규칙으로 떨어지고 그 수를 출력에 적는다.
**판단에 걸린 시간은 시뮬 시계에 안 들어간다**(로봇이 기다리지 않는 것으로 셈) — 닫힘까지 시간은 지연 0 가정이다.

### 옵션

공통(위층 방식과 무관):

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--scenes a.scene …` | (필수) | 장면 파일들. 장면마다 따로 돌리고 따로 적는다 |
| `--policy <체크포인트>` | 없음 | 아래층 RL 정책. 안 주면 같은 걸음 루프에서 규칙 추종기(대조) |
| `--episodes` · `--seed` · `--people` | 256 · 1 · 0 | 판 수(= 환경 수, 환경마다 첫 판) · 씨앗 · 사람 수 |
| `--eval-seed` | 끔 | 장면 씨앗을 학습기 평가와 같은 식으로 |
| `--out <접두사>` | 없음 | `<접두사>_<장면>.tsv` 에 판마다 한 줄 |
| `--env` · `--model` · `--frontier` | sim/config 의 env.yaml · sensor_model.yaml, 프론티어 기준선 설정 | 시뮬·센서·후보 설정 파일 |
| `--no-avoid` | 끔 | 막힘 신호가 선 칸을 피하지 않음 |
| `--no-redecide` | 끔 | 도착·경로 실패 때 바로 다시 고르지 않음 |
| `--no-stick` | 끔 | 고른 후보 덩어리가 지금 목표를 품어도 목표 칸을 새로 잡음 |
| `--hold <걸음>` | 0 | 목표를 정한 뒤 이 걸음 수 동안 주기 결정에서 안 바꿈 (실험용) |
| `--stall-s <초>` | 0 | 막힘 판정 시간을 이 값으로 (0 = env.yaml 값, 실험용) |
| `--rule-near <m>` | 0 | 목표까지 이 거리 안이면 그 결정은 규칙 추종기에 맡김 (실험용) |
| `--auto` | 끔 | 위층 규칙을 호스트 대신 환경이 돌림(학습 때와 같은 길). slot0 규칙만 — `--policy` 와 같이, 판단 모델과는 같이 못 씀 |
| `--gt` · `--gt-base a0.scene …` | 끔 | **셈 전용 — 정답 지도를 씀.** 후보까지 참 길이 바탕 장면(낮은 장애물 없는 원본)보다 3 m · 1.5 배 넘게 길거나 없으면 "막힌 후보"로 센다(문턱 잠정). 고르기·판 결과에는 안 쓴다 |
| `--video <접두사>` · `--video-env` · `--end-hold <초>` | 없음 · 0 · 3 | 환경 하나의 판을 `<접두사>_<장면>.mp4` 로(시뮬 시계 실시간). 판이 끝나면 리셋 없이 명령 0 으로 `--end-hold` 초 더 찍는다(0 = 끔, 값 잠정) |

위층 방식:

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--upper slot0\|random\|decider\|cand_tsp` | slot0 | slot0 = 유효 후보 가운데 번호가 가장 작은 것(기준선) · random = 아무거나(하한) · decider = 판단 모델 · cand_tsp = 후보를 모두 도는 가장 짧은 차례의 첫 후보(`cand_rules.h`, 로봇 지도만 — 고른 후보가 피하는 칸이면 slot0) |
| `--rule-stick <x>` · `--rule-turn <m>` | 0 · 0 | cand_tsp: frontier_run 과 같은 뜻(지금 목표보다 1 + x 배 넘게 좋을 때만 바꿈 · 방향 1 rad 어긋남마다 더하는 거리). 후보 v2 는 환경 변수 `SIM_CAND_V2=1` |
| `--decider-url <주소>` | 없음 | 서버 주소. 주면 `--upper decider` |
| `--decider-input text\|image\|both\|all` | text | 모델에 주는 것: 후보 표 글 · 지도 사진 · 둘 다 · 카메라 그림까지(all). `prune` · `--decider-finish` 는 all 만 |
| `--decider-mode choice\|prune\|score` | choice | choice = 후보 하나를 고름 · prune = 규칙 차례는 그대로 두고 "그 길이 막혔나"만 물어 막혔다는 후보를 건너뜀 · score = 후보마다 "가면 지도가 얼마나 더 채워지나"를 5 단계로 물어 기대 점수가 가장 높은 후보로 (아래 "점수 방식") |
| `--decider-prompt 1\|2\|3` | 1 | 질문 글 판(실험용, `decider.cpp` 에 글자 그대로). 입력 all 은 1 · 2: 2 는 "카메라에 보이는 장애물" 을 따로 일러 줌. 3 은 글 입력에서 상태를 JSON 으로 |
| `--decider-ask-m` | 4 | prune: 규칙 차례 앞에서 이 수만큼만 묻는다 (잠정) |
| `--decider-veto-thr` | 0.5 | prune: "막혔다" 확률이 이 값 이상이면 건너뜀 (잠정) |
| `--decider-finish` | 끔 | 판 끝을 모델이 선언 — 아래 "끝 선언". choice · score 만 (prune 과 같이 주면 오류) |
| `--decider-config <yaml>` | `scripts/config/decider.yaml` | 임무 문장 · 점수 질문 글과 단계 · 문턱을 읽는 설정 파일. score 방식이나 `--decider-finish` 일 때만 읽는다 |
| `--decider-mission "<글>"` | 없음 | 설정 파일의 임무 문장을 실행할 때 덮어씀 |
| `--decider-left-hold <N>` | 설정 파일 `left_hold` (2, 잠정) | choice 끝 선언: "더 넣을 게 없다" 답이 이 수만큼 이어진 결정에서 선언 |
| `--decider-score-margin <x>` | 설정 파일 `score_margin` (0.25, 잠정) | score: 최고 기대 점수에서 이 안이면 비긴 것으로 (지금 목표 · 가까운 후보 차례) |
| `--decider-score-finish <x>` | 설정 파일 `score_finish_below` (−1) | score 끝 선언: 모든 후보의 기대 점수가 이 값 아래면 끝. 음수면 "모든 후보의 가장 그럴듯한 단계가 0" |
| `--decider-timeout-ms` | 10000 | 요청 하나의 시간 한계. 넘으면 규칙으로 떨어짐. 환경 여러 개를 한꺼번에 물으면 줄 서는 시간이 들어가므로 넉넉히 |
| `--decider-parallel` | 16 | 한꺼번에 보내는 요청 수 (서버 `-np` 와 맞춤). 1 이면 하나씩 — 지연을 잴 때 |
| `--decider-px` | 384 | 지도 사진 한 변 픽셀 |
| `--decider-cam-h` · `--decider-cam-div` | 480 · 2 | 카메라 그림 높이(실물 컬러 영상 640 × 480) · 줄이는 배수 (잠정) |
| `--decider-shuffle` | 끔 | 후보 글자를 번호와 무관하게 섞음 (글자 자리 쏠림 확인용) |
| `--decider-dump <폴더>` · `--decider-dump-n` | 없음 · 6 | 요청(JSON) · 사진(PNG) · 답을 장면마다 이 수만큼 남김 |

### 점수 방식 (`--decider-mode score`)

요청 하나에 후보마다 score 질문 하나를 넣는다("그 후보로 가면 지도가 얼마나 더 채워지나" — 0 전혀 · 1 조금 · 2 꽤 · 3 많이 · 4 새 구역이 열림).
질문 글과 단계는 `scripts/config/decider.yaml` 의 `score_question` · `score_level_0…` 에서 읽는다(단계 수는 2 ~ 10, 서버 한계).
기대 점수(Σ 단계 × 확률)가 가장 높은 후보로 가고, 최고에서 `score_margin` 안이면 비긴 것으로 보아 지금 목표가 그 안에 있으면 그대로, 아니면 경로가 가까운 후보로 간다.

뒤에 학습 정답으로 쓸 기록 두 파일을 남긴다(둘 다 로봇 지도만으로 만든 값):

- `<접두사>_<장면>_score.tsv` — 결정 · 후보마다 단계 확률 `p0…`, 기대 점수, 고른 후보인지, 끝 규칙에 걸렸는지.
- `<접두사>_<장면>_gain.tsv` — 고른 목표마다: 그 목표로 정한 걸음부터 도착(`arrived`) · 포기(`abandoned`) · 다른 목표로 바꿈(`switched`) · 판 끝(`episode_end`)까지
  로봇 지도에서 새로 알게 된 칸 수(`cells_gained`)와 걸린 시간(`elapsed_s`), 그때 모델이 낸 단계 확률.

### 끝 선언 (`--decider-finish`, 기본 끔)

끄면 판은 규칙으로만 끝난다: 로봇 지도가 닫히고 10 초 동안 지도가 안 늘면 성공. 켜면 모델이 그보다 먼저 끝을 선언할 수 있고, 선언한 걸음에 판이 끝난다
(위층 멈춤 · 아래층 정책 꺼짐 · 명령 0). 규칙은 그대로 살아 있어 모델이 선언하지 않아도 판은 끝난다. 옵션을 끄면 요청 글과 출력은 본 측정 때와 글자까지 같다.

- **choice 방식(주 설계)**: 고르기 질문은 그대로 후보 A~H 만 고른다(FINISH 선택지 없음). **같은 요청**에 score 질문 하나를 더한다 —
  "로봇이 갈 수 있는 곳에서 지도에 더 넣을 수 있는 게 얼마나 남았나(갈 수 없는 모르는 곳은 세지 않음)", 0 더 없음 … 4 새 구역 하나.
  글과 단계는 설정 파일 `left_question` · `left_level_0…`. 가장 그럴듯한 단계가 0 인 답이 `left_hold`(기본 2, 잠정) 결정 이어지면 끝을 선언한다
  (`left_finish_below` 를 0 이상으로 주면 "기대 점수가 그 값 아래"로 바뀜). 후보가 2 개보다 적은 결정에는 score 질문만 보낸다.
  `<접두사>_<장면>_left.tsv` 에 결정마다 choice 확률, 단계 확률, 기대 점수, 그때 로봇 지도의 아는 칸 수와 **그때부터 판 끝까지 로봇 지도가 는 칸 수**(뒤에 이 점수의 학습 정답)를 남긴다.
- **score 방식**: 모든 후보의 가장 그럴듯한 단계가 0 이거나 후보가 하나도 없으면(가다 포기해 잠깐 뺀 후보도 없을 때) 끝을 선언한다.
- **임무 문장은 설정에서**: `mission`(기본 "map the whole floor as fast as possible"). `--decider-mission` 으로 바꿀 수 있다.
- **모델이 받는 것 — 전부 로봇이 스스로 아는 값**: 지금 후보 수, 가다 포기해 뺀 후보 수, 로봇 지도가 닫혔는지, 지도가 마지막으로 늘어난 뒤 시간(초), 지금까지 포기한 목표 수, 판 시간,
  그리고 후보마다 이 판에서 그 자리를 목표로 삼은 횟수와 가다가 포기한 횟수(진행 없음 시간 초과 · 경로 잃음). 정답 지도에서 나온 값은 넣지 않는다.
- **채점도 로봇 지도만으로**(실물에서 같은 식으로 잴 수 있어야 하므로): 선언한 걸음에 로봇 지도가 닫혀 있으면 "맞음"(판 끝 코드 1), 갈 수 있는 프론티어가 남아 있으면 "이른 선언"(코드 4).
  닫힘 정의는 `mdp/frontier.h` 머리글(출발 칸에서 설 수 있는 칸으로 번져 닿는, 4 칸 이상 프론티어 덩어리가 없음 + 라이다가 가까이서 이미 본 틈은 안 침).
  이 정의는 로봇이 쌓는 "가 본 칸 · 가까이서 본 칸" 겹도 쓰므로, **점유 격자 한 장만으로** 같은 판정을 하는 함수(`decider.h` `grid_open_frontier` — ROS `nav_msgs/OccupancyGrid`
  꼴: −1 모름, 65 이상 막힘)도 같이 돌려 tsv 에 적는다. 둘이 다르면 다른 대로 적는다.
- **정답 덮음은 시뮬 전용 진단**으로만 출력 한 줄에 따로 적는다("시뮬 전용 진단" 이라 표시). 판정·모델 입력에 안 쓴다.
- tsv 에 붙는 칸(켰을 때만): `ended_by_model`(1 = 모델 선언으로 끝남, 0 = 규칙 · 충돌 · 진행 없음 등) · `finish_questions` · `decl_p`(선언 때 "남은 탐사" 기대 점수, score 방식은 후보 기대 점수 가운데 최고, 후보가 없어 끝내면 −1)
  · `decl_idle_s`(선언 때 지도 안 늘어난 시간) · `decl_map_closed`(frontier.h 닫힘) · `decl_grid_open_cells` · `decl_grid_clusters`(격자만으로 남은 프론티어 칸 · 덩어리).

### 결과 (2026-11-14, 제로샷 — 미세조정 없음)

조건: 아래층은 같은 RL 체크포인트, 사람 0, 시험 맵 bookstore · small-house 와 그 낮은 장애물 변형, 씨앗 1 · 2 · 3, 씨앗마다 24 판, d1-3B Q8, RTX 3060.
"돕는다"의 기준은 측정 전에 정해 두었고(변형 장면마다 실패 판이 규칙보다 적고 닫힘 시간이 규칙의 씨앗 범위 안, 원본 장면마다 성공률 · 닫힘 시간이 규칙의 씨앗 범위 안 — 넷 다),
결과를 본 뒤 바꾸지 않았다. **두 방식 모두 기준을 못 채웠다 — 규칙(slot0)보다 낫다고 할 수 없다.**

세 씨앗 합 · 평균 (실패 판 = 벽 충돌 판 + 진행 없음 판, 72 판 가운데 / 닫힘까지 시간은 성공 판 평균의 세 씨앗 평균, 판단 지연 0 가정):

| 장면 | slot0 규칙 | choice | prune (앞 2 후보) |
|---|---|---|---|
| bookstore | 실패 2 · 127.7 s | 실패 2 · 124.8 s | 실패 3 · 128.0 s |
| small-house | 실패 3 · 95.4 s | 실패 2 · 100.8 s | 실패 0 · 99.9 s |
| bookstore-lowobs | 실패 16 · 141.0 s | 실패 12 · 137.3 s | 실패 8 · 143.3 s |
| small-house-lowobs | 실패 2 · 121.2 s | 실패 12 · 119.8 s | 실패 7 · 120.6 s |

- choice 는 small-house-lowobs 에서 실패가 규칙보다 많고(12 대 2), 원본 small-house 에서 닫힘 시간이 규칙의 씨앗 범위를 넘었다(100.8 s 대 한도 100.0 s).
- prune 은 small-house-lowobs 에서 실패가 규칙보다 많다(7 대 2). "막혔다"는 답은 잡음에 가깝다: 낮은 장애물이 없는 원본에서도 장면 · 씨앗마다 800 ~ 1000 번 막혔다 했고(정답 막힘 0),
  변형에서 "막혔다 한 것 가운데 정답 막힘"(0.030 ~ 0.083)은 물은 후보 전체의 정답 막힘 비율(0.026 ~ 0.081)과 거의 같고, small-house-lowobs 두 씨앗에서는 오히려 낮다.
- 프롬프트 판 2 는 씨앗 1 · 변형 두 장면에서만 돌렸다: choice 2 는 성공 0.750 · 0.917(판 1 은 0.708 · 0.750), prune 2 는 거의 모든 후보를 막혔다 해 성공 0.458 · 0.458 로 떨어졌다.
- 지연: 요청을 하나씩 보내면 choice 283 ms, prune(질문 2 개) 322 ms (중앙값). 측정에서는 환경 16 개를 한꺼번에 물어 요청마다 2.3 ~ 3.7 s 였다. 서버가 못 답해 규칙으로 떨어진 결정은 0.
  판단 시간은 시뮬 시계에 안 넣었다 — 실물에서는 그만큼 늦은 결정이 된다.
- 시험 장면의 한계: 낮은 상자 자리는 손으로 고른 값(잠정)이고, 아래층 RL 이 낮은 상자에 더 자주 부딪혀 위층 고르기와 무관한 실패가 섞인다. 판 수 24 는 씨앗 사이 흔들림이 크다.

**끝 선언 (2026-11-14, 제로샷)** — choice + "남은 탐사" score(주 설계)를 씨앗 1 의 두 판(small-house-lowobs: 47.1 s 에 벽 충돌 · 원본 small-house: 95.0 s 에 규칙으로 성공)과
씨앗 4 의 세 장면 × 12 판(성공 bookstore 0.917 · small-house 1.000 · small-house-lowobs 0.917)에서 돌렸다.

- **모델 끝 선언은 한 번도 없었다** — 모든 판을 규칙(또는 충돌)이 끝냈다. 물은 결정 4492 개 모두에서 가장 그럴듯한 단계가 2 또는 3 이었고 0("더 없음")은 0 번,
  기대 점수는 2.4 ~ 2.6 에서 거의 안 움직였다. 지도가 다 차 판 끝까지 몇 칸밖에 안 느는 마지막 결정들에서도 2.4 안팎이다.
- 결정마다 점수와 "그때부터 판 끝까지 로봇 지도가 는 칸 수"의 상관: 씨앗 4 세 장면 0.07 · 0.16 · 0.18, 씨앗 1 두 판 −0.04 · −0.40.
  지금 꼴로는 끝 판단에 쓸 수 없다 — 이 기록(`_left.tsv`)을 정답으로 가르치는 것이 다음 일이다. 문턱은 시험 장면을 보고 바꾸지 않았다.
- 요청 하나의 지연(환경 1 개, 하나씩): choice 만 283 ms, choice + score 질문 하나 367 ms.
- 씨앗 4 의 대조(slot0 · choice 만 · 후보마다 점수 방식 · `--upper cand_tsp`)는 **측정 대기** — GPU 를 아래층 학습에 먼저 쓰느라 멈춤.

### 시험 장면: 라이다에 안 보이는 낮은 장애물 (`scene_lowobs`)

판단 모델이 도울 수 있는 경우(지도에는 길이 뚫려 보이는데 실제로는 막힌 곳)를 만들려고, 시험 맵에 높이 0.09 m 턱·상자를 더한 변형 장면을 쓴다.
라이다 평면(0.111 m) 아래라 로봇 지도에 안 찍히지만 몸통과는 부딪히고 카메라에는 보인다. 만드는 법은 [scenes/README](../scenes/README.md) "낮은 장애물 변형".
