# sim/webots — 교수님 Webots 맵 + YOLO 물체

교수님 `webots_ros2_limo` 의 `limo_world.wbt`(5 × 10 m 방)를 복사해서 YOLO 7 클래스 물체를 놓았다. 교수님 코드는 고치지 않고, 월드 파일만 절대경로로 넘겨 띄운다.

```bash
source ~/limo_fire_ws/env.sh
ros2 launch webots_ros2_limo robot_launch.py world:=$HOME/limo_fire_ws/sim/webots/worlds/limo_fire_world.wbt
# SLAM: ... slam:=true   /   Nav2: ... nav:=true map:=<지도.yaml>
```

처음 열 때 Webots 기본 물체(벽·가구·소화기·사람·계단)를 GitHub 에서 받으므로 인터넷이 필요하다.

## 놓은 것

정확한 위치·크기는 [objects.yaml](objects.yaml)(YOLO 벤치마크 정답). 로봇은 (6.36, 0) 에서 +x(동쪽)를 보고 시작한다.

| 클래스 | 개수 | 모델 |
|---|---|---|
| 0 fire | 3 | 바닥 불(0.7 × 0.4 m, 주황 불빛) · 탁자 위 촛불 · 작은 불 — `protos/FireBillboard.proto`, `protos/Candle.proto` |
| 1 smoke | 1 | 바닥 불 위 연기 — `FireBillboard` (texture `smoke`) |
| 2 fire_extinguisher | 3 | Webots `FireExtinguisher` |
| 3 exit_sign | 3 | 피난구 유도등 — `protos/ExitSign.proto` (ISO 7010 E001 그림, 스스로 빛남) |
| 4 person | 2 | Webots `Pedestrian` (서 있음, 컨트롤러 없음) |
| 5 door | 1 | 원래 맵의 서쪽 벽 문 |
| 6 stairs | 1 | Webots `StraightStairs` 4 단 |

- 불·연기는 사진을 투명 배경으로 잘라 십자 판 두 장에 붙인 것이다(Webots 에는 불·연기 모델이 없다). 판이라 아주 가까이·옆에서 보면 얇게 보인다.
- 문은 원래 맵의 것 하나뿐이고, 원래 있던 가구(소파·냉장고·화분·`DirectionPanel` 표지판 등)는 그대로 둬서 헷갈리게 하는 배경 역할을 한다.
- Nav2 지도(`resource/limo_world_map.yaml`)에는 새 물체가 없다. 그대로 써도 AMCL 은 돌지만, 정확히 하려면 `slam:=true` 로 다시 만든다.

## 텍스처 출처 (`textures/`)

전부 재배포 가능한 것만 썼다. 검은 배경을 밝기로 투명하게 바꿨다(불: 색조 유지·밝기 최대, 연기: 회색).

| 파일 | 원본 | 라이선스 |
|---|---|---|
| `flame_bonfire.png` | StockSnap "Fire Bonfire" `0G7N1L3G2F` (https://stocksnap.io/photo/0G7N1L3G2F) | CC0 |
| `flame_single.png` | StockSnap "Flame Fire" `EAYRCBDWQ3` (https://stocksnap.io/photo/EAYRCBDWQ3) | CC0 |
| `smoke.png` | rawpixel "Smoke black background" (px7779, Openverse 검색) | CC0 |
| `exit_sign_e001.png` | Wikimedia Commons "ISO 7010 E001.svg" | Public domain |

## 확인 안 한 것

로봇(Jetson, arm64)에는 Webots 가 없어서 **아직 한 번도 띄워 보지 않았다.** 처음 띄울 때 볼 것:

- [ ] 월드가 오류 없이 열리는지 (특히 우리 proto 3 개의 템플릿)
- [ ] 불·연기 투명 부분이 투명하게 보이는지, 불이 스스로 빛나는지
- [ ] 유도등 그림이 방 쪽을 보는지(뒤집혔으면 `ExitSign.proto` 의 앞면 방향)
- [ ] 계단이 벽을 뚫지 않는지 — `StraightStairs` 원점이 어디인지 보고 `objects.yaml` 의 `stairs` 고치기
- [ ] 라이다가 불·연기 판을 장애물로 잡는지 (잡으면 Nav2 가 피해 간다)
- [ ] `objects.yaml` 의 `approx: true` 크기를 Webots 에서 재서 고치기
