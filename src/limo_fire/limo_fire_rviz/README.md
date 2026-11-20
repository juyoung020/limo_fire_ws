# limo_fire_rviz — 운용 화면 패널

운용 화면(`limo_fire_description/rviz/limo_fire.rviz`)에 기본으로 붙는 RViz2 패널 플러그인 — 탐사 패널과 연기 모드 패널.

## 탐사 패널 (`limo_fire_rviz/ExplorePanel`, 화면 왼쪽 아래 "Explore")

| 부분 | 하는 일 |
|---|---|
| 대상 | 합친 탐사 `/explore/explorer` (위층 판단 모델 + 아래층 A* 경로 추종) — [limo_fire_explore](../limo_fire_explore/README.md) |
| 버튼 **Auto explore OFF (누르면 시작)** / **ON (누르면 멈춤, 초록)** | `/explore/enable` (std_srvs/SetBool) 에 true / false. 끄면 로봇이 선다 |
| 상태 | `running`(탐사 중) · `done`(끝 — 다시 누르면 처음부터) · `idle`(멈춤) · `paused`(리모컨·닿음) · `노드 없음`(서비스가 안 보임 — 버튼 꺼짐) |
| 아래 회색 줄 | `…/status` 한 줄 (목표·프론티어 수·지도 넓이, 끝나면 걸린 시간) |

- 상태는 `/explore/state`(transient local)에서 온다. 옛 고르기(Nav2 프론티어 탐사 · 탐사 RL)는 지웠다 — 탐사는 하나다.
- 노드는 패널이 띄우지 않는다. `app.launch.xml`(limo_fire_explore)이 띄운다 — 실물 `real.launch.xml`, 시뮬 `sim.launch.xml slam:=true`.
- 패널이 안 보이면: 메뉴 Panels → Add New Panel → `limo_fire_rviz/ExplorePanel`. 목록에도 없으면 이 패키지를 빌드하고 `source install/setup.bash` 한 터미널에서 RViz 를 띄웠는지 본다.

## 연기 모드 패널 (`limo_fire_rviz/SmokePanel`, "Smoke")

| 부분 | 하는 일 |
|---|---|
| 버튼 **연기 모드 OFF (누르면 켬)** / **ON (누르면 평상시로, 회색)** | `/smoke_mode/set` (std_srvs/SetBool) 에 true / false — [limo_fire_navigation 연기 모드](../limo_fire_navigation/README.md#연기-모드) |
| 상태 | 평상시(라이다로 AMCL) · 연기 모드(바퀴 + IMU 로만) · `노드 없음`(서비스가 안 보임 — 버튼 꺼짐) |
| 센서 줄 (켰을 때) | 라이다는 위치 추정에 안 쓰고 장애물 지도엔 저장 지도 벽(가상 스캔)만, 카메라는 주행에 안 씀 |
| 아래 회색 줄 | 연기 속 추정 불확실성 1σ(위치 m · 방향 °, `/smoke_mode/pose`), 노드가 거절하면 그 까닭 |

- 상태는 `/smoke_mode/active`(transient local)에서 온다. 화면에 "Smoke Pose" 로 연기 속 추정 자세와 불확실성 타원도 보인다.
- 저장 지도(`real.launch.xml` · `sim.launch.xml` 기본)에서만 켜진다. 지도 만들기(`slam:=true`) 중이거나 아직 2D Pose Estimate 로 위치를 안 잡았으면 노드가 거절하고 까닭이 회색 줄에 뜬다.
- 라이다·카메라 드라이버를 끄는 것이 아니라 주행(위치 추정·장애물 지도)이 그 값을 안 받게 끊는다 — 나올 때 AMCL 이 라이다로 다시 맞추고, `LaserScan` 표시(빨강)로 실제 벽과 추정이 얼마나 벌어졌는지 볼 수 있다.

실물에서 써 보기 (저장 지도):

1. `ros2 launch limo_fire_bringup real.launch.xml` (지도 기본 `lab_1116b`, 다른 지도는 `map:=…`)
2. RViz **2D Pose Estimate** 로 처음 위치 → 초록 입자가 모이고 라이다(빨강)가 지도 벽에 겹칠 때까지
3. **Smoke** 패널 버튼 → `연기 모드 ON` (속도 한도 `smoke_mode.yaml speed_limit`)
4. 리모컨 SWB 위(오토) → **Nav2 Goal** 로 목표 — 한 번에 움직이는 거리는 정한 범위 안
5. 도착하면 빨강 스캔이 지도 벽에서 벌어진 만큼이 바퀴 + IMU 오차. 버튼을 다시 눌러 평상시로 → AMCL 이 다시 맞춤

```bash
colcon build --packages-select limo_fire_rviz   # Qt5 · rviz_common (ROS 2 Humble 데스크톱 설치에 들어 있음)
```

확인 (2026-11-11, 이 PC): 빌드 경고 없음, 화면 없는 X(Xvfb)에서 `rviz2 -d limo_fire.rviz` 가 패널을 불러옴(플러그인 오류 없음),
덕트호스 미로 시뮬에서 상태가 idle → running → idle → running → done 으로 바뀜. 젯슨에서는 아직 빌드 안 해 봄.

연기 모드 패널 확인 (2026-11-18, 젯슨 Humble): 빌드 경고 없음, 플러그인 등록·기호 확인. 화면에 띄운 확인은 아직 — 로봇 화면에서 할 차례.
