# sim/webots — Webots 월드 + YOLO 정답 물체

시뮬은 실물과 같은 launch(지도 만들기·Nav2·인식)를 쓰고, 로봇 켜기만 Webots 로 바꾼다(`limo_fire_bringup/launch/sim_robot.launch.xml`).
LIMO 모델·센서 플러그인은 교수님 `webots_ros2_limo` 를 고치지 않고 쓴다.

```bash
source ~/limo_fire_ws/env.sh
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze                    # 저장 지도 + AMCL + Nav2
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze slam:=true         # Cartographer 로 지도 만들며 Nav2
ros2 launch limo_fire_bringup save_map.launch.xml name:=sim/limo_fire_maze            # 지도 저장 (slam 중에)
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze perception:=true   # + YOLO·물체 지도
```

## 맵

| 이름 | 크기 | 모양 |
|---|---|---|
| `limo_fire_maze` | 8 × 6 m | 시연장 추정 모양. 덕트호스(지름 0.15 m) 두 줄로 만든 ㄹ자 통로 셋 |
| `limo_fire_office` | 12 × 8 m | 복도 + 방 셋(문 열림) + 계단실, 동쪽 끝 비상구 |
| `limo_fire_warehouse` | 10 × 10 m | 팔레트 두 줄 사이 통로, 드럼통·상자, 닫힌 문 둘 |
| `limo_fire_world` | 5 × 10 m | 교수님 `limo_world.wbt`(아파트 방) 복사본 |

맵마다 YOLO 7 클래스(불·연기·소화기·유도등·사람·문·계단)를 모두 놓았다.

### 맵 더하기
같은 이름으로 파일만 넣으면 된다. launch 는 고치지 않는다.

1. `worlds/<이름>.wbt` — 다른 월드를 복사해 시작. 끝에 있는 `Ros2Supervisor` 로봇(`/clock` 을 냄)은 지우지 않는다. 정답 물체에는 정답 색을 준다 (아래 "인식 정답").
2. `worlds/<이름>.yaml` — 월드 데이터 파일 (아래 형식)
3. `ros2 launch limo_fire_bringup sim.launch.xml world:=<이름> slam:=true` 로 순찰점을 돌고 `save_map.launch.xml name:=sim/<이름>` → `limo_fire_navigation/maps/sim/<이름>.yaml`

### 월드 데이터 파일 (`worlds/<이름>.yaml`)
좌표는 전부 Webots 세계 좌표(m, rad).

```yaml
robot_start: {x, y, yaw}      # 월드의 LimoFourDiff 자리와 같게. 시뮬 지도의 원점(map 0,0,0)이 이 자리라 AMCL 이 처음 위치를 따로 받지 않는다
patrol: [[x, y], ...]          # 순찰점(차례대로). 지도 만들기·Nav2 확인에 쓴다
objects:                       # YOLO 정답 물체. 벤치마크에서 카메라에 투영해 2D 정답 상자를 만든다
  - {name, cls, center: [x, y, z], size: [x 폭, y 폭, 높이], yaw, approx?, note?}
```
- `cls` 는 train/detect 와 같다: 0 fire · 1 smoke · 2 fire_extinguisher · 3 exit_sign · 4 person · 5 door · 6 stairs
- `size` 는 회전 전 물체 자기 축 기준. `approx: true` 는 Webots 모델 크기를 눈대중으로 적은 값

### 인식 정답

인식 벤치마크([docs/benchmarks](../../docs/benchmarks/README.md))의 정답은 카메라 Recognition 분할 영상에서 나온다. 그래서 월드의 정답 물체마다 정답 색을 준다.

- 색 코드: `recognitionColors [ (클래스+1)/255  (월드 데이터 파일 objects 의 몇 번째, 1 부터)/255  1 ]`. 클래스 번호는 train/detect 와 같다.
- 우리 proto(`FireBillboard` · `ExitSign` · `Candle`(불꽃에만))는 `recognitionColors` 필드에 바로 쓴다.
- Webots 기본 모델(`FireExtinguisher` · `StraightStairs` · `Door`)은 그 필드가 없어 `Solid { name "gt_<이름>" recognitionColors [...] children [ <모델> ] }` 로 감싼다 (안쪽 모양 전체가 이 색으로 분할된다).
- `Pedestrian` 은 셔츠 색을 recognitionColors 로 쓰므로 `shirtColor` 에 색 코드를 넣는다(셔츠가 파랗게 보임).
- 로봇은 `LimoFire`(`protos/LimoFire.proto`, 교수님 `LimoFourDiff` 에 카메라 Recognition 만 더한 판).
- 물체를 더하거나 순서를 바꾸면 월드 데이터 파일의 순서와 색 코드 두 번째 값을 같이 맞춘다.

### 월드를 만들 때 알아둘 것
- Webots 기본 물체는 `webots://projects/...` 로 설치본 안의 것을 쓴다(인터넷 필요 없음). 교수님 원본 월드는 GitHub 주소를 쓰는데, Jetson 에서 빌드한 Webots(Qt 6.2)는 HTTP/2 다운로드가 17 % 에서 멈춘다.
- `Door` 는 `position` 이 양수일 때 로컬 +x 쪽으로 열리고 음수는 경첩 한계로 안 열린다. 방 안쪽으로 열리게 회전을 고르고, 끝까지(1.57) 열어 문짝을 벽에 붙인다 — 비스듬히 열린 문짝에 로봇이 부딪혀 SLAM 이 틀어졌다.
- 덕트호스(`protos/DuctHose.proto`)는 지름이 라이다 높이(0.111 m)보다 커야 라이다에 잡힌다.
- `StraightStairs` 는 원점에서 +x 로 올라간다(원점은 첫 단 앞 가운데).

## `limo_fire_world` 에 놓은 것

정확한 위치·크기는 [worlds/limo_fire_world.yaml](worlds/limo_fire_world.yaml). 로봇은 (6.36, 0) 에서 +x(동쪽)를 보고 시작한다.

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
3. 소스 세 군데 — [webots_R2025a_arm64.patch](webots_R2025a_arm64.patch) (`git apply`):
   - `src/webots/Makefile` 의 `QT_CORE_INCLUDE` 에 `-isystem $(WEBOTS_PATH)/include/qt` 추가(시스템 Qt 는 `<QtCore/...>` 형태)
   - `dependencies/Makefile.linux` 의 `PACKAGES = open-al`
   - **`src/wren/Camera.cpp`: `glm::perspective` 를 double 로 부르면 aarch64 에서 쓰레기 행렬(예: [0][0]=1, [1][1]=0)이 나온다.** 이 행렬은 거리 렌더링(RangeFinder·Lidar)에만 쓰여서 컬러 카메라는 멀쩡하고 라이다·깊이만 전부 `inf`(ROS 로는 0)가 됐다. 같은 공식(perspectiveRH_ZO)을 직접 계산하게 바꿨다.
   - 그 밖에 `include/libpico/*.h` 는 복사본에서 인자 이름 `this` → `this_` (C++ 예약어)
4. `make webots_target TARGET=release -j6` 와 `make -C src/controller release`. 로봇 창용 `qt_utils` 는 실패하지만 우리 시나리오(LIMO 는 `<extern>` ROS 컨트롤러)에는 필요 없다.
5. `sudo ln -s ~/webots_build/webots /usr/local/webots` (`env.sh` 의 `WEBOTS_HOME`).

## 확인한 것 (2026-10-26, Jetson)

- [x] 월드가 오류 없이 열린다 — 우리 proto 3 개 포함, ROS 컨트롤러 연결, `/camera/color/image_raw` 20 Hz · `/scan` 9.4 Hz · `/odom` 46 Hz · `/imu` 99 Hz
- [x] 로봇 카메라에 소화기·사람·계단·유도등·작은 불이 보인다
- [x] 소화기가 물리 때문에 넘어져서 `enablePhysics FALSE` 로 고정
- [x] 라이다·깊이 카메라 정상(위 Camera.cpp 수정 뒤): `/scan` 248 빔 0.82~6.09 m, 교수님 `slam:=true` 로 Cartographer 지도 생성
- [x] YOLO26s TensorRT(`limo_fire_perception`) 로 계단 84 % · 소화기 88 % · 문 84 % · 유도등 86 % 검출

## 아직 볼 것

- [ ] 바닥 불(`fire_floor`)이 카메라에 잘 안 보임 — 위치·크기 다시 보기
- [ ] 불·연기 투명 부분이 투명하게 보이는지 (작은 불은 투명하게 보임), 연기 보이는지
- [ ] 유도등 그림이 방 쪽을 보는지(뒤집혔으면 `ExitSign.proto` 의 앞면 방향)
- [ ] 계단이 벽을 뚫지 않는지 — `StraightStairs` 원점이 어디인지 보고 `worlds/limo_fire_world.yaml` 의 `stairs` 고치기
- [ ] 라이다가 불·연기 판을 장애물로 잡는지 (잡으면 Nav2 가 피해 간다)
- [ ] 월드 데이터 파일의 `approx: true` 크기를 Webots 에서 재서 고치기
