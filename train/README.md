# train

GPU 학습 환경(C++/CUDA/Rust). ROS 패키지가 아니라 빌드에서 빠진다(`COLCON_IGNORE`).

| 폴더 | 내용 | 상태 |
|---|---|---|
| `maps/` | 맵 변환기 `fmap` (Rust): HouseExpo·ProcTHOR·덕트호스 생성기 → 공통 점유 격자 `.fmap` + 소화기·비상구·화재 감지기 배치. 탐사·조준 공용 | 됨 |
| `explore/sim/` | 탐사 RL — GPU 학습 시뮬(CUDA): 실물 LIMO 녹화로 맞춘 구동·라이다·IMU·깊이, 확률 지도, 움직이는 사람, 보상, 로봇 중심 관측. CPU 참조판과 비트 동일 | 됨 |
| `explore/mdp/` · `network/` · `ppo/` · `scripts/` | 탐사 RL — 관측·보상·커리큘럼, CNN 혼합 정책·가치, PPO, 실행기 ([구조](explore/README.md)) | 됨 — 두 층 구조로 가는 중([계획](explore/ROADMAP.md)) |
| `aim/` | 조준 RL | 나중 |
| `detect/` | YOLO 탐지(불·연기·소화기·비상구·사람·문), C++/CUDA 학습기 | 할 일 |

## 맵 만들기

```bash
cd train/maps && cargo build --release
M=../../sim/maps
./target/release/fmap_build --out out \
  --houseexpo $M/houseexpo/HouseExpo/json \
  --procthor $M/procthor_10k/train.jsonl.gz --procthor $M/procthor_10k/val.jsonl.gz --procthor $M/procthor_10k/test.jsonl.gz \
  --duct 5000 --png-every 2000
```
32 코어에서 52,032 개 약 22 초, 12 GB(`out/`, git 에 넣지 않음). `out/index.tsv` 에 맵마다 넓이·방·소화기·비상구·감지기 수.

## 소방 시설 배치 규칙 (`src/lib.rs` `place_facilities`)

| 시설 | 규칙 |
|---|---|
| 소화기 | 벽에 붙은 칸에 둔다. 모든 빈 칸이 보행거리 20 m 안(화재안전기준 소형소화기)이 될 때까지 가장 먼 곳 근처에 하나씩 + 33 m² 이상인 방에는 방마다 하나 |
| 비상구 | ProcTHOR 는 바깥 벽 문, 덕트호스는 외곽 출입구. 없으면 외벽 앞 칸 중 보행거리로 서로 가장 먼 1 곳(100 m² 이상이면 2 곳) |
| 화재 감지기 | 방마다 방 중심(천장). 방 정보가 없으면 6 m 격자 |
| 불 | 여기서 놓지 않는다. GPU 환경이 판마다 도달 영역 안 무작위 칸에 만든다 |

로봇 중심이 벽에서 0.15 m 이상 떨어진 빈 칸의 가장 큰 연결 영역만 도달 영역(FREE)으로 남긴다. 광선은 바깥(0)과 벽(2)에서 모두 막힌다.

## 탐사 학습 시뮬 (`explore/sim/`) — 새 판, 실물에 맞춘 센서

환경 하나 = CUDA 블록 하나, 커널 한 번 = 행동 하나(속도 행동 0.1 s · 목표점 행동 0.5 s). 그 안에서 (목표점이면 Nav2 RPP 추종기) → 실물로 맞춘 섀시 구동(지연·가속·바닥 미끄러짐)·몸통 충돌·실물 오차 라이다(`lidar_real.h`)·Cartographer 식 확률 지도·보상·리셋을 다 한다. 설정 `config/env.yaml`, 센서 오차 `config/sensor_model.yaml`, 근거 `analysis/`.

```bash
cd train/explore/sim && cmake -S . -B build && cmake --build build -j
./build/env_verify <장면.scene> …            # GPU == CPU 비트 동일 (상태·보상·지도·관측)
./build/env_verify <장면.scene> … --negative # 음성 대조
./build/env_bench  <장면.scene> … --envs 1024
```

장면은 Webots 월드를 띄워 `/Ros2Supervisor/animation_start_recording` 로 `.w3d` 를 받고 `w3d2scene` 으로 바꾼다. 장면 파일(버전 2)엔 출발점이 들어 있어(`w3d2scene` 은 뺀 로봇 자리, `map2scene` 은 지도 원점 또는 `--start x y`) 덮음 분모를 거기서 정한다 — 옛 버전 1 장면은 다시 만든다.
구동은 실물 녹화로 맞추고 확인한다: `motion_export`(limo_fire_sim) → `./build/motion_check <구동.csv> [--fit]`.
GPU 환경 전체를 실물 녹화로 다시 돌린다: `bag_export <녹화> <x.lfrec>`(지도 `.map.yaml` 도) → `./build/map2scene x.lfrec.map.yaml x.scene` → `./build/replay_check x.scene <구동.csv> x.lfrec`.
보정 녹화는 [analysis/calibration.yaml](explore/sim/analysis/calibration.yaml) 에 역할(fit·validate)과 뺄 구간을 적고 `--set analysis/calibration.yaml fit|validate` 로 돌린다 (`motion_check --fit`, `replay_check --fit-dropout`).
옛 초안 환경(`.fmap` 맵 읽기 포함)은 `archive/explore_env`.
