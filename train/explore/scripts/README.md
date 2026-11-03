# scripts — 실행 파일

| 파일 | 내용 |
|---|---|
| `train.cpp` | 학습. 명령줄 옵션: `--num_envs` `--max_iterations` `--experiment_name` `--seed` `--resume` `--load_run` `--checkpoint` |
| `play.cpp` | 학습한 정책 돌려 보기 |
| `frontier_run.cpp` | 프론티어 탐사 기준선을 장면 × 판으로 돌려 뷰어 실행 폴더(`run.json kind: baseline`)로 쓴다. 빌드·옵션은 [baselines/frontier](../baselines/frontier/README.md) |
