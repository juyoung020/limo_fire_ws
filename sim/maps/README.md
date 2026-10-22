# maps

GPU 학습 환경·검증에 쓰는 실내 맵. 받은 날짜 2026-10-22.

| 폴더 | 내용 | 규모 | 형식 | 라이선스 | 저장소 |
|---|---|---|---|---|---|
| `aws_robomaker/hospital` | 병원(복도·병실·로비) — 점검 시설 시험 맵 | 1 | Gazebo SDF + 메시 | MIT-0 | 포함 |
| `aws_robomaker/small-warehouse` | 물류창고(선반·통로) | 1 | Gazebo SDF + 메시 | MIT-0 | 포함 |
| `aws_robomaker/bookstore` | 서점(서가·테이블·의자) | 1 | Gazebo SDF + 메시 | MIT-0 | 포함 |
| `aws_robomaker/small-house` | 방 여러 개인 집 | 1 | Gazebo SDF + 메시 | MIT-0 | 포함 |
| `procthor_10k` | 절차적으로 만든 집 | 10,000 + val 1,000 + test 1,000 | `*.jsonl.gz` (방·벽·문·물체) | Apache-2.0 | 포함 |
| `houseexpo` | 실제 평면도(SUNCG 기반) | 35,126 집, 252,550 방 | `HouseExpo/json/*.json` (벽 다각형 `verts`, 미터) | 코드 MIT, 데이터는 SUNCG 유래 | **제외**(.gitignore) |

## 쓰는 곳

| 용도 | 맵 |
|---|---|
| 학습 | houseexpo + procthor_10k + 덕트호스 절차 생성기(만들 예정) |
| 처음 보는 맵 시험 | aws_robomaker 4 종 — 학습에 넣지 않는다 |
| 시연 리허설 | 덕트호스 시연장(만들 예정) |

## 출처

- AWS RoboMaker 월드: `github.com/aws-robotics/aws-robomaker-<이름>-world` 의 `ros2` 브랜치. AWS RoboMaker 는 2025-09 서비스 종료, 저장소는 보관 상태.
- ProcTHOR-10K: https://github.com/allenai/procthor-10k (Git LFS)
- HouseExpo: https://github.com/TeaganLi/HouseExpo — SUNCG 는 2019 년 법적 분쟁으로 내려간 데이터라 다시 배포하지 않는다. 받기:
  ```bash
  git clone --depth 1 https://github.com/TeaganLi/HouseExpo.git sim/maps/houseexpo
  tar -xzf sim/maps/houseexpo/HouseExpo/json.tar.gz -C sim/maps/houseexpo/HouseExpo
  ```
