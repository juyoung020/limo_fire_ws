# real_calib — 실물 측정 → 시뮬 파라미터 후보

ROS 없이 빌드되는 C++ 도구. 측정 csv 를 읽어 후보를 보여 줄 뿐 설정 파일은 고치지 않는다 — 후보는 날짜·녹화 이름과 함께 config 주석에 옮긴다
([real_robot_checklist.md](../../docs/real_robot_checklist.md) §3). 맞추기(fit)와 확인(validate) 측정은 나눈다(`train/explore/sim/analysis/calibration.yaml`).

```bash
cmake -S tools/real_calib -B /tmp/real_calib_build && cmake --build /tmp/real_calib_build && ctest --test-dir /tmp/real_calib_build
/tmp/real_calib_build/real_calib echo  echo.csv  [--max-angle 10]
/tmp/real_calib_build/real_calib slip  slip.csv
/tmp/real_calib_build/real_calib drift imu.csv
```

| 모드 | csv 열 | 나오는 것 | 비교할 시뮬 가정 |
|---|---|---|---|
| `echo` | `angle_deg,true_m,range_m[,material]` (range 가 비었거나 inf·nan·0 이면 메아리 없음) | 각도별 메아리 비율·편향·σ, 항상 메아리인 마지막 각도, 거의 없어지는 첫 각도, 정면 거리 잡음 | `hard` 22.5° · `soft` 25~55° · 잡음 1 % |
| `slip` | `true_m,odom_m` (줄자 이동, `/odom` 이동) | 비율 평균·σ | 실제 이동 ×1.03 |
| `drift` | `t_s,yaw_deg` | IMU yaw 드리프트 °/min, 직선 벗어남 | `wall.q_yaw`, `drift.yaw_per_rad` |

시험(`test_real_calib`)은 만든 자료로 알려진 답을 확인한다(메아리 모양·잡음·비율·360° 넘김). 실물 자료로는 아직 안 돌려 봤다.
