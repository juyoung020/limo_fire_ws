# sim/webots — 교수님 Webots 맵 + YOLO 물체

교수님 `webots_ros2_limo` 의 `limo_world.wbt`(5 × 10 m 방)를 복사해서 YOLO 7 클래스 물체를 놓았다. 교수님 코드는 고치지 않고, 월드 파일만 절대경로로 넘겨 띄운다.

```bash
source ~/limo_fire_ws/env.sh
ros2 launch webots_ros2_limo robot_launch.py world:=$HOME/limo_fire_ws/sim/webots/worlds/limo_fire_world.wbt
# SLAM: ... slam:=true   /   Nav2: ... nav:=true map:=<지도.yaml>
```

Webots 기본 물체(벽·가구·소화기·사람·계단)는 `webots://projects/...` 로 Webots 설치본 안의 것을 쓴다(인터넷 필요 없음). 교수님 원본 월드는 GitHub 주소를 쓰는데, Jetson 에서 빌드한 Webots(Qt 6.2)는 HTTP/2 다운로드가 17 % 에서 멈춰서 바꿨다.

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

## Jetson(ARM)에서 돌리기

Webots 공식 배포는 x86 뿐이라 로봇(Jetson Orin Nano, arm64)에서는 R2025a 소스를 직접 빌드했다(2026-10-26). 시뮬 시간은 실시간의 약 0.9 배, 카메라 약 20 Hz.

```bash
git clone --depth 1 --branch R2025a --recurse-submodules --shallow-submodules https://github.com/cyberbotics/webots.git ~/webots_build/webots
sudo apt install qt6-base-dev qt6-base-dev-tools qt6-base-private-dev qt6-declarative-dev qt6-declarative-private-dev \
  libqt6opengl6-dev qt6-qpa-plugins libassimp-dev libois-dev libttspico-dev libopenal-dev libfreeimage3 libzip-dev \
  libssh-gcrypt-dev swig libglu1-mesa-dev libxcb-cursor0 ninja-build \
  ros-humble-webots-ros2-driver ros-humble-webots-ros2-control ros-humble-diff-drive-controller \
  ros-humble-joint-state-broadcaster ros-humble-controller-manager ros-humble-mecanum-drive-controller
```

1. Qt WebSockets 6.2.4 는 Ubuntu 22.04 arm64 패키지가 없어서 소스(`qtwebsockets-everywhere-src-6.2.4`)를 cmake 로 빌드해 `/usr` 에 설치.
2. Webots 가 내려받는 x86 전용 의존(Qt 6.5.3·OIS·assimp·pico) 대신 시스템 것을 연결: `include/qt/<모듈>` → `/usr/include/aarch64-linux-gnu/qt6/<모듈>`, `bin/qt/{moc,lrelease,lupdate}` → `/usr/lib/qt6/...`, `lib/webots/{libOIS,libassimp,libpico}.so` → 시스템 라이브러리, `lib/webots/qt/plugins` → `/usr/lib/aarch64-linux-gnu/qt6/plugins`, `dependencies/Makefile.linux` 의 `PACKAGES = open-al`.
3. 소스 두 군데: `src/webots/Makefile` 의 `QT_CORE_INCLUDE` 에 `-isystem $(WEBOTS_PATH)/include/qt` 추가(시스템 Qt 는 `<QtCore/...>` 형태), `include/libpico/*.h` 는 복사본에서 인자 이름 `this` → `this_` (C++ 예약어).
4. `make webots_target TARGET=release -j6` 와 `make -C src/controller release`. 로봇 창용 `qt_utils` 는 실패하지만 우리 시나리오(LIMO 는 `<extern>` ROS 컨트롤러)에는 필요 없다.
5. `sudo ln -s ~/webots_build/webots /usr/local/webots` (`env.sh` 의 `WEBOTS_HOME`).

## 확인한 것 (2026-10-26, Jetson)

- [x] 월드가 오류 없이 열린다 — 우리 proto 3 개 포함, ROS 컨트롤러 연결, `/camera/color/image_raw` 20 Hz · `/scan` 9.4 Hz · `/odom` 46 Hz · `/imu` 99 Hz
- [x] 로봇 카메라에 소화기·사람·계단·유도등·작은 불이 보인다
- [x] 소화기가 물리 때문에 넘어져서 `enablePhysics FALSE` 로 고정

## 아직 볼 것

- [ ] 바닥 불(`fire_floor`)이 카메라에 잘 안 보임 — 위치·크기 다시 보기
- [ ] 불·연기 투명 부분이 투명하게 보이는지 (작은 불은 투명하게 보임), 연기 보이는지
- [ ] 유도등 그림이 방 쪽을 보는지(뒤집혔으면 `ExitSign.proto` 의 앞면 방향)
- [ ] 계단이 벽을 뚫지 않는지 — `StraightStairs` 원점이 어디인지 보고 `objects.yaml` 의 `stairs` 고치기
- [ ] 라이다가 불·연기 판을 장애물로 잡는지 (잡으면 Nav2 가 피해 간다)
- [ ] `objects.yaml` 의 `approx: true` 크기를 Webots 에서 재서 고치기
