# limo_fire_hardware — 탁구공 발사대 접근·좌우 조준·발사 터미널

표적 탐지 → 정지 거리 후보 검사 → Nav2 접근 → 차체 좌우 정렬 → 정지·재탐지 → (선택) 장전+발사 한 번.
**발사대 각은 기구로 15° 고정**이다(2026-11-20 각도 서보를 뗌). 그래서 상하 조준은 없고, 맞추는 것은 **정지 거리와 좌우**뿐이다.

- 실물 데모 실행 순서: [REAL_FIRE_TEST.md](REAL_FIRE_TEST.md)
- 장착값 확인·무장전 조준 시험: [REAL_AIM_TEST.md](REAL_AIM_TEST.md)
- 발사대 시리얼은 **ESP32 다리 하나만** 연다 — 센서와 같은 펌웨어([firmware/limo_fire_esp32](../../../firmware/limo_fire_esp32/README.md)), 같은 포트, 921600 baud.
  발사대 ROS 인터페이스(액션 `/launcher/launch`, 서비스 `/launcher/enable`, 상태 `/launcher/state`)는 [limo_fire_esp32_bridge](../limo_fire_esp32_bridge/README.md).

예전 Python 패키지(`approach_aim.py`·`pitch_serial_bridge.py`·`launcher_control_terminal.py`·`mapper_observation_bridge.py`·`aim_learning.py`)는 2026-11-20 C++ 로 옮기며 지웠다.
상하각 명령·탄도 계산·각도 피드백·조준 학습 준비(`aim_learning`)는 각도 서보가 없어져 뺐다.

## 실행 파일

| 실행 파일 (노드) | 하는 일 |
|---|---|
| `fire_observation_bridge` (`/fire_observation_bridge`) | 물체 지도 `/perception/objects` 에서 **최근 0.5 s 안에 다시 본** `fire` 하나만 `/fire/observations`(`limo_fire_msgs/FireObservation`)로. 저장된 물체를 다시 낸 것은 새 관측으로 치지 않는다(칸 `covariance[35]` = 마지막으로 본 뒤 지난 초). 불이 여럿이면 `target_id` 로 하나를 고르게 하고 넘기지 않는다 |
| `approach_aim` (`/launcher/approach_aim`) | 요청 한 번에 한 바퀴: 접근·좌우 정렬·(auto_fire) 발사 액션 한 번 |
| `launcher_terminal` | 사람이 따로 연 창에서 `a` `r` `c` `f` `s` `x` `?` `q` |

### `/launcher/approach_aim` 입출력

| 종류 | 이름 | 형 | 뜻 |
|---|---|---|---|
| 서비스 | `~/start` | `std_srvs/Trigger` | 한 바퀴 시작. 조건이 안 되면 `success=false` + 까닭 |
| 서비스 | `~/cancel` | `std_srvs/Trigger` | 취소 — 보낸 Nav2 목표·발사 동작(다리가 `x`)을 취소하고 정지를 확인 |
| 토픽(냄) | `~/state` | `std_msgs/String`, TRANSIENT_LOCAL | `idle` `planning` `approaching` `wait_stop` `reacquiring` `aligning` `yaw_aligning` `wait_shot` / 끝: `preview_ready` `aim_ready` `shot_done` `shot_rejected` `shot_unconfirmed` `failed` `cancelled` `stop_unconfirmed` |
| 토픽(냄) | `~/status` | `std_msgs/String`, TRANSIENT_LOCAL | 같은 때 한글 한 줄(`<state>: <까닭>`) |
| 토픽(냄) | `/cmd_vel` | `geometry_msgs/Twist` | `yaw_alignment_mode: yaw_rate` 일 때만 제자리 회전 `angular.z` |
| 액션(씀) | `/compute_path_to_pose` · `/navigate_to_pose` | Nav2 | 후보 경로 계획·접근·(nav2 모드) 좌우 정렬 |
| 액션(씀) | `/launcher/launch` | `limo_fire_msgs/action/Launch` | `auto_fire_enabled` 일 때 `SHOT` 한 번 |
| 토픽(받음) | `/fire/observations` `/odometry/filtered` `/global_costmap/costmap` `/limo_status` `/launcher/state` · TF `map→base_link` `base_link→launcher_pivot` | | 표적·정지·costmap·주행 모드·발사대 상태·위치·장착 |

`shot_done` 은 펌웨어가 장전·발사를 마쳤다는 응답이다. 공의 비행·명중은 재지 않는다.

## 빌드

```bash
source env.sh
colcon build --packages-select limo_fire_msgs limo_fire_esp32_bridge limo_fire_hardware limo_fire_description limo_fire_bringup
source install/setup.bash
```

`limo_msgs`(LIMO 상태 메시지)는 로봇의 WeGo 드라이버 작업공간에서 온다. 없는 PC 에서는 `src/external/wego_limo_pro/limo_ros2/limo_msgs` 도 같이 빌드한다.

## 준비할 값

- 장착: URDF `limo_fire_description/urdf/limo_fire.urdf.xacro` 의 `launcher_xyz`·`yaw` 한 곳. 조준은 TF `base_link → launcher_pivot` 으로 읽고, 없으면 시작을 거절한다. 2026-11-20 결정: 회전축 높이는 실측(바닥 21.0 cm), x·y·yaw 는 CAD 값.
  실물 설정 파일에서 `mount_calibrated: true` (저장소 기본 `false`).
- `desired_distance_m`: 회전축에서 표적까지 **수평 정지 거리**. 각이 15° 고정이라 이 거리가 탄착을 정한다. 기본 1.2 m 는 **잠정 예시값** — 실물 사거리 표로 정한다
  (공기저항 없는 계산으로는 3.9 m/s·15°·같은 높이에서 약 0.78 m. 실제 공은 공기저항·높이 차·옆 휨으로 다르다).
- `robot_radius_m + footprint_margin_m` 은 차체 외곽을 포함해야 한다.
- 발사 직후 공 속도(사용자 제공 근삿값, 2026-11-18, 재측정 안 함): 파란 공 약 3.9 m/s, 노란 공 약 4.2 m/s. 계산에는 쓰지 않고 거리 고를 때 참고만.

기하 계산은 평면 주행을 가정한다. TF 의 차체 roll/pitch 가 `max_base_tilt_deg` 를 넘으면 멈춘다.

## 흐름과 안전 규칙

1. **시작 조건**: 지도 좌표·신뢰도 0.5 이상·1 s 안의 표적, 최근 costmap, 위치 TF, 발사대 TF, 정지한 최근 오도메트리, 경로 계획 서버.
   움직임(`motion_enabled`)이면 + `mount_calibrated`, Nav2 이동 서버, (yaw_rate 모드) 확인된 차동·메카넘 모드, (auto_fire) 발사대 액션 서버와 발사대 상태 `ready`·`off`.
2. **후보**: 표적 둘레 정지 거리 고리 위의 자리(0°, ±15°, ±30°, ±60°)를 차례로, 차체 외곽 정사각과 회전축→표적 시선이 costmap 빈 칸인지 본다(표적 앞 `sight_target_margin_m` 는 뺌). Nav2 가 그 자리까지 경로를 만드는 첫 후보를 쓴다.
3. `motion_enabled=false` → `preview_ready` (후보 자리·yaw·회전축 값을 글로). 움직이지 않는다.
4. 접근 → 정지 확인(`stop_dwell_s`) → **다시 본 표적**으로 거리·좌우 오차 → 정렬: `nav2`(두 번째 자세 목표) 또는 `yaw_rate`(`/cmd_vel` 최대 0.25 rad/s, 3° 안이면 0, 45° 넘거나 8 s 안에 못 맞추면 0 속도 후 실패) → 정지 → 오차 확인.
5. `auto_fire_enabled=false` → `aim_ready`. 켜져 있으면 발사대 상태를 다시 보고 `SHOT` 한 번 → `shot_done` / `shot_rejected`(다리가 거절) / `shot_unconfirmed`(고장·시간 초과).
- 진행 중 다른 표적 ID 는 받지 않고, 고른 표적이 처음 자리에서 `target_shift_limit_m` 넘게 움직이면 취소한다.
- 취소·실패는 `cancelling` → 보낸 목표가 모두 끝나고 정지를 확인하면 `cancelled`/`failed`. 확인 못 하면 `stop_unconfirmed` 이고 그동안 새 시작을 받지 않는다.
- 끌 때(Ctrl+C) 보낸 Nav2·발사 목표를 취소하고 0 속도를 보낸 뒤 끝난다(최대 5 s 확인).
- 설정은 실행 중 바꿀 수 없다(read-only 파라미터) — 바꾸려면 다시 켠다.

`yaw_rate` 는 `ros2 topic echo /limo_status --once` 의 `motion_mode` 가 차동 `0`·메카넘 `2` 일 때만 쓴다. Ackermann `1` 은 `angular.z` 가 조향각으로 들어가 제자리 회전이 안 된다(`limo_fire_base` `/cmd_vel` 처리).

## 터미널 `launcher_terminal`

```bash
ros2 run limo_fire_hardware launcher_terminal --log /home/wego/launcher-commands.jsonl
bash tools/hardware_check/open_ros_launcher_terminal.sh   # GUI 새 창
```

| 키 | 하는 일 |
|---|---|
| `a` | 표적 찾기 → 접근·조준·장전+발사 임무 한 번. 조준 노드 `auto_fire_enabled` 가 꺼져 있으면 거절(파라미터를 읽어 확인) |
| `r` | 공 다시 넣음 확인. 자동 임무가 발사를 요청했을 수 있으면(`wait_shot` 이후, 결과 못 받음, Ctrl+C) **`r` 전에는 다음 `a` 를 거절** — 미리 쳐 둔 `a` 도 |
| `c` · `f` · `s` | 장전 · 발사 · 장전+발사 (액션 `/launcher/launch`, 진행 단계가 보임). 자동 임무 중이면 거절 |
| `x` | 레버 끔(`/launcher/enable false`). 발사대 고장 상태도 이것으로 푼다 |
| `?` · `q` | 상태 · 끝 |

- `a` 를 눌렀을 때 1 s 안에 같은 표적(같은 ID, 0.25 m 안)을 서로 다른 영상 시각으로 세 번 봤으면 바로 시작한다. 아니면 `/explore/frontier_explorer/enable` 로 탐사를 켜고(상태가 `idle`·`done` 일 때만, 이미 `running` 이면 거절), 표적을 찾으면 끄고 `idle` + 차체 정지 0.5 s 를 확인한 뒤 시작한다. `--search-timeout-s`(기본 180 s) 안에 못 찾거나 탐사가 끝나면 발사하지 않는다. `--no-search` 는 탐사 없이 지금 보이는 표적만.
- Ctrl+C: 탐사 중이면 탐사를 끄고, 임무 중이면 `/launcher/approach_aim/cancel` 을, 수동 동작 중이면 동작 취소(다리가 `x`)를 보낸 뒤 끝난다. 응답을 못 받을 수 있으니 로봇 정지는 현장에서 확인한다.

## 소프트웨어 시험

```bash
firmware/limo_fire_esp32/test/run.sh       # 펌웨어 호스트 시험 + /tmp/launcher_host_sim 빌드 (발사대 실제 경로 시험에 필요)
colcon test --packages-select limo_fire_esp32_bridge limo_fire_hardware && colcon test-result --verbose
```

- `test_aim_geometry`: 정지 거리 후보·좌우 오차·제자리 회전 속도·costmap 외곽·시선 검사·물체 지도에서 불 고르기 (예전 Python 시험에서 상하·탄도 빼고 옮김).
- `test_approach_aim_ros`: 실행 파일 `approach_aim`·`launcher_terminal` 을 띄우고 시험 안의 가짜 로봇(Nav2 액션 서버·TF·costmap·오도메트리·표적·LIMO 상태)과 맞물림 —
  미리보기는 움직이지 않음, Nav2 접근+정렬 → `aim_ready`, 12° 틀어진 뒤 제자리 회전, 확인 안 된 주행 모드 거절, 이동 중 취소 → Nav2 취소·`cancelled`,
  터미널 `a`·미리 친 `a` 거절·`r` 뒤 `a` = 발사 정확히 두 번, 자동 발사 꺼짐이면 거절,
  **터미널 `a` → 조준 → 실제 `esp32_bridge` → 펌웨어 원본(가상 시리얼)**: 레버 PWM 이 장전(≈833 µs)·발사(≈1889 µs) 자리로, 각도 서보 핀은 켜지지 않음.
- 실물 로봇·서보·공은 없다. 차체는 Nav2 목표 자리로 순간 이동한다.

## 확인한 것

- 2026-11-20 (이 PC, Humble, 실물 없음): `colcon build` 경고 0, `colcon test` gtest limo_fire_esp32_bridge 37 개 · limo_fire_hardware 30 개(계산 22 · ROS 실행 파일 8) 모두 통과, 세 번 되풀이해도 같음. 펌웨어 `test/run.sh` 통과.
- **실물에서는 아직 안 돌려 봤다**: C++ 판으로 ESP32 업로드·레버 동작·카메라 표적·Nav2 주행·발사 모두 미확인.
- 예전 Python 판의 실물 준비 기록(2026-11-19~20: Wego 센서 수신, YOLO 약 30 장/s, 표적 없음·`map` TF 없음으로 탐지 0, `wego` 사용자 `dialout` 아님, CP2102 `ttyUSB0` 은 라이다)은 [REAL_FIRE_TEST.md](REAL_FIRE_TEST.md) "예전 기록" 에 남겼다.
