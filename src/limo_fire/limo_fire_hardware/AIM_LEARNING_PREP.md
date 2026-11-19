# 조준 보정 학습 준비 — 실행 조준에는 미연결

현재 기준 조준은 `approach_aim.py`의 표적 지도 좌표·장착 치수·탄도 계산과 로봇 회전/상하 서보 제어다. 이 문서의 학습 도구는 **오프라인 발사 기록 분석 전용**이다. `real_camera_aim.launch.py`, ROS 토픽, 펌웨어는 학습 결과를 읽지 않는다. 데이터가 없어 학습된 모델이나 성능 수치를 제시하지 않는다.

## 기록할 값

`python3 -m limo_fire_hardware.aim_learning init shots.csv`로 빈 CSV 헤더를 만든다. 한 발당 한 행을 기록한다. `run_id`와 `shot_id`의 조합은 중복될 수 없다.

| 열 | 뜻 |
| --- | --- |
| `recorded_at_utc`, `source` | UTC 발사 시각과 `real_camera` 또는 `mujoco`. 두 출처는 별도로 학습한다. |
| `ball_color`, `launch_speed_mps` | `blue`/`yellow` 및 그 발사에서 설정한 속도. 현재 근삿값은 각각 3.9/4.2 m/s이며 실제 값은 재측정 대상이다. |
| `target_id`, `range_m`, `height_delta_m` | 실제 탐지 ID, 발사축에서 표적까지 수평 거리, 표적 중심과 발사축 높이 차. `map`/TF와 현장 줄자값을 대조한다. |
| `baseline_yaw_deg`, `baseline_pitch_deg` | 기존 계산이 낸 로봇 방향과 상하각. yaw는 `map` 기준 왼쪽 회전 +, pitch는 위쪽 +. |
| `delta_yaw_deg`, `delta_pitch_deg` | 시험 때 기준각에 적용한 보정. 왼쪽/위쪽 +. 보정하지 않은 기준 시험은 둘 다 `0`. |
| `impact_error_right_m`, `impact_error_up_m` | 원하는 착탄점에서 실제 착탄점까지의 오차. 발사기에서 표적을 볼 때 오른쪽 +, 위쪽 +. 두 값은 함께 측정하거나 함께 비운다. |
| `contact`, `toppled`, `notes` | 관찰 결과 `true`/`false`/빈칸, 현장 조건. `toppled`는 표적 전도 시험에서만 쓴다. |

착탄점을 확인하지 못한 발사도 CSV에 남길 수 있지만 학습에는 쓰이지 않는다. 화면에서 추정한 착탄점을 실제 측정으로 표기하지 않는다. `source=real_camera`는 작성자의 기록이므로 카메라 로그·영상과 대조해야 한다.

## 오프라인 검증·후보 학습

```bash
source /home/wego/limo_fire_ws/env.sh
source /home/wego/launcher_aim_friday_20261120/install/local_setup.bash
python3 -m limo_fire_hardware.aim_learning init /home/wego/aim_shots.csv
# 현장 기록을 CSV에 입력한 뒤:
python3 -m limo_fire_hardware.aim_learning validate /home/wego/aim_shots.csv
python3 -m limo_fire_hardware.aim_learning fit /home/wego/aim_shots.csv /home/wego/aim_candidate.json --source real_camera --min-samples 3
```

동일 명령을 두 번 실행해 기존 CSV/모델을 덮어쓰지 않도록 생성은 새 파일에만 허용한다. MuJoCo 기록은 별도 파일 또는 `--source mujoco`로 후보를 만들고 실물 데이터와 섞지 않는다. 공 색·발사 속도 0.2m/s·거리 0.25m·높이 차 0.10m·기준 pitch 1°·보정각 0.5° 구간에서 **같은 보정 동작을 실제로 실행한 발사**만 묶는다. 한 발로 끝나는 문제라 할인율 0인 contextual bandit의 `Q(상황, 보정) = -평균 착탄 오차(m)`를 저장한다. 시도하지 않은 보정에는 값을 만들지 않는다. 한 구간·동작에 측정 3발 미만이면 후보 셀에서 제외한다. 이는 최소 데이터 필터일 뿐 정확도 검증 기준이 아니다.

JSON에는 입력 CSV의 SHA-256, 사용 발수, 출처, 구간·보정·표본 수와 `runtime_enabled: false`가 들어간다. ROS 조준 노드는 이 JSON을 읽지 않는다. 실제 적용을 요청받으면 별도 보류 실험으로 기준 계산과 후보를 비교하고, 조준각/주행 범위 제한·표적 신선도·카메라 거리 확인을 통과한 경우에만 작은 보정값을 쓰도록 연결한다. 착탄 오차가 없는 `CONTACT`/`TOPPLED`만으로 위치 보정을 학습했다고 주장하지 않는다.

## 적용 전 필요한 현장 자료

1. 실물 표적 탐지, `map` TF, 발사축 실측 치수와 ESP32 조준각 확인.
2. 발사 거리·공 색별 기준 조준 발사와 착탄점 오차. 영상 또는 표적 눈금과 CSV 행을 같은 `run_id`/`shot_id`로 대응시킨다.
3. 학습에 사용하지 않은 회차에서 기준 조준과 후보 보정의 착탄 오차를 비교한다. 보정이 나빠지면 적용하지 않는다.

실제 발사 및 강화학습 보정의 런타임 적용은 이 준비 작업에 포함되지 않는다.
