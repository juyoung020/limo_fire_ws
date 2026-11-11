# limo_fire_rviz — 운용 화면 패널

운용 화면(`limo_fire_description/rviz/limo_fire.rviz`)에 기본으로 붙는 RViz2 패널 플러그인.

## 탐사 패널 (`limo_fire_rviz/ExplorePanel`, 화면 왼쪽 아래 "Explore")

| 부분 | 하는 일 |
|---|---|
| 고르기 | 프론티어(`/explore_frontier`) · RL(`/explore_rl`) — [limo_fire_rl](../limo_fire_rl/README.md) 의 두 노드 |
| 버튼 **Auto explore OFF (누르면 시작)** / **ON (누르면 멈춤, 초록)** | `<노드>/enable` (std_srvs/SetBool) 에 true / false. 끄면 노드가 Nav2 목표를 취소해 로봇이 선다 |
| 상태 | `running`(탐사 중) · `done`(끝 — 다시 누르면 처음부터) · `idle`(멈춤) · `노드 없음`(서비스가 안 보임 — 버튼 꺼짐) |
| 아래 회색 줄 | 노드의 `<노드>/status` 한 줄 (목표·프론티어 수·지도 넓이, 끝나면 걸린 시간) |

- 상태는 프론티어 노드의 `~/state`(transient local)에서 온다. RL 노드는 이 토픽이 없어 `~/done` 과 패널이 마지막으로 보낸 요청으로 보인다 — 명령줄로 끄고 켜면 패널이 모를 수 있다.
- 노드는 패널이 띄우지 않는다. 시뮬은 `sim.launch.xml slam:=true`(기본으로 같이 뜸), 실물은 `explore_frontier.launch.xml`.
- 패널이 안 보이면: 메뉴 Panels → Add New Panel → `limo_fire_rviz/ExplorePanel`. 목록에도 없으면 이 패키지를 빌드하고 `source install/setup.bash` 한 터미널에서 RViz 를 띄웠는지 본다.

```bash
colcon build --packages-select limo_fire_rviz   # Qt5 · rviz_common (ROS 2 Humble 데스크톱 설치에 들어 있음)
```

확인 (2026-11-11, 이 PC): 빌드 경고 없음, 화면 없는 X(Xvfb)에서 `rviz2 -d limo_fire.rviz` 가 패널을 불러옴(플러그인 오류 없음),
덕트호스 미로 시뮬에서 상태가 idle → running → idle → running → done 으로 바뀜. 젯슨에서는 아직 빌드 안 해 봄.
