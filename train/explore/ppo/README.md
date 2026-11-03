# ppo — 어떻게 배우나

| 파일 | 내용 |
|---|---|
| `rollout.cu` | 롤아웃 버퍼·GAE |
| `ppo.h` · `ppo.cu` | PPO 손실·갱신·Adam + 하이퍼파라미터 |
| `runner.cpp` | 학습 루프·체크포인트 저장/이어하기·로그 (`runs/<실험>/`). 로그는 `viewer/include/runfolder.h` 로 써서 학습 뷰어(`viewer/README.md`)가 바로 읽는다 |
