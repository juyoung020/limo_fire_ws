# train

GPU 학습 환경(C++/CUDA/Rust). ROS 패키지가 아니라 빌드에서 빠진다(`COLCON_IGNORE`).

| 폴더 | 내용 | 상태 |
|---|---|---|
| `maps/` | 맵 변환기 `fmap` (Rust): HouseExpo·ProcTHOR·덕트호스 생성기 → 공통 점유 격자 `.fmap` + 소화기·비상구·화재 감지기 배치 | 됨 |
| `env/` | LIMO GPU 환경 커널(CUDA): 차동 구동, 라이다·depth 광선, 덮음 보상, 판마다 불 무작위 생성 | 할 일 |
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
