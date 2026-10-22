# train

GPU 학습 환경(C++/CUDA/Rust). ROS 패키지가 아니라 빌드에서 빠진다(`COLCON_IGNORE`).

| 폴더 | 내용 | 상태 |
|---|---|---|
| `maps/` | 맵 변환기 `fmap` (Rust): HouseExpo·ProcTHOR·덕트호스 생성기 → 공통 점유 격자 `.fmap` + 소화기·비상구·화재 감지기 배치 | 됨 |
| `env/` | LIMO GPU 환경 커널(CUDA): 차동 구동 + 목표점 추종기, 라이다 248 빔 DDA, 덮음 보상, 판마다 불 무작위 생성. CPU 참조판과 비트 동일 | 됨(관측 만들기는 PPO 와 함께) |
| `ppo/` | PPO 학습기 (object-memory-robot `training/RL` 에서 옮김) | 할 일 |
| `viewer/` | 학습 뷰어 trainview (Rust) | 할 일 |

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

## 환경 (`env/`)

```bash
cd train/env && cmake -S . -B build && cmake --build build -j
./build/fenv_verify ../maps/out 300 512 400            # GPU == CPU 비트 동일이어야 통과
./build/fenv_verify ../maps/out 300 512 400 --negative # 일부러 어긋나게 해서 잡히는지(음성 대조)
./build/fenv_bench  ../maps/out 1000 16384 200         # 처리량
```

| 항목 | 값 (`include/fenv.h` 의 `K`) |
|---|---|
| 행동 | 48 = 로봇 기준 방향 16 × 거리 {0.5, 1, 2} m. 0.5 s 동안 추종기가 목표점으로(벽 너머면 벽 앞에서 자름) |
| 구동 | v ≤ 0.5 m/s, ω ≤ 1.5 rad/s, 가속 1.0 m/s² · 3.0 rad/s² (가정), 충돌 원 0.13 m |
| 라이다 | 248 빔 / 220° / 12 m (교수님 Webots 모델과 같음), 칸 DDA |
| 덮기 | 빈 곳과 맞닿은 벽 칸(표면)이 3 m 안에서 3 번 맞으면 덮음. 95 % 면 성공 |
| 불 | 판마다 1 개(80 %) 또는 2 개, 도달 영역 무작위 칸, 시작에서 2 m 이상. 작은 불 r 0.05–0.15 m(70 %), 큰 불 0.3–0.8 m. 카메라 66.5° · 5 m 안 · 가리지 않으면 "봄" |
| 보상 | 덮은 비율 × 10, 성공 +5, 행동마다 −0.005, 재방문 −0.01, 제자리 −0.01, 충돌 −2(끝), 불 처음 봄 +1, 목표점 잘림 −0.005 |
| 시간 | 판당 행동 수 = 도달 넓이(m²) × 4, 200–2000 |

검증 결과(2026-10-23, RTX 5070 Ti, 맵 300 개 = 세 소스 고루): 512 환경 × 400 행동 **비트 동일**, 음성 대조는 행동 200·아는 지도 14,786 칸 불일치로 잡힘.
무작위 행동 기준 끝난 판 2,049: 성공 31 / 충돌 1,903 / 시간초과 115, 평균 덮은 비율 0.22, 불 본 비율 0.34.
처리량: 16,384 환경 3.5e5 env-step/s(시뮬 0.5 s 씩 → 실시간의 약 17 만 배), 아는 지도 0.65 GB.
