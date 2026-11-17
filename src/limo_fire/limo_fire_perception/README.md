# limo_fire_perception

카메라 영상에서 불·연기·소화기·유도등·사람·문·계단을 찾는 C++ TensorRT YOLO 노드. 실물 Orbbec 카메라와 Webots 시뮬 카메라에 똑같이 쓴다.

```bash
ros2 launch limo_fire_perception yolo.launch.xml          # 탐지 + 물체 지도 + 운용 화면 (이미 화면을 띄웠으면 rviz:=false)
```

- `yolo_detector`·`object_mapper` 는 컴포넌트(`limo_fire_perception::YoloDetectorNode`, `::ObjectMapperNode`). 같은 이름의 실행 파일도 있다
- 기본은 둘을 한 스레드 컨테이너 `/perception_container` 한 프로세스에 올리고 프로세스 안 전달(intra-process)로 탐지·마스크를 복사 없이 넘긴다.
  예전에 한 스레드 컨테이너가 5 초에 18~20 장을 버린 까닭은 매퍼가 콜백 안에서 TF 를 최대 0.2 s 기다린 것 — 이제 기다리지 않고, YOLO 는 영상을 자기 작업 스레드에서 직접 받는다 (아래 "스레드·메모리")
- 카메라 드라이버(Orbbec) 컨테이너에는 올리지 않는다: 드라이버가 transient_local 발행자를 만들어 프로세스 안 전달을 켤 수 없고
  (켜면 rclcpp 가 노드 생성을 거부), 같은 프로세스여도 직렬화는 그대로라 이득이 작은데 YOLO 가 죽으면 카메라도 같이 죽는다

- 입력: `/camera/color/image_raw` (rgb8·bgr8·rgba8·bgra8, 크기 무관)
- 출력: `/yolo_detector/detections` (`vision_msgs/Detection2DArray`), 상자 그린 영상 `/yolo_detector/image` (rgb8)
- 분할(-seg) 엔진이면 물체 번호 영상 `/yolo_detector/mask` (mono8, 값 = `detections` 안 순서 + 1, 0 = 배경, 겹치면 상자가 작은 물체가 위)도 같은 시각으로 낸다 (물체가 있는 장만). 마스크는 앞 64 개 물체까지.
  탐지 영상에는 마스크를 반투명하게 칠한다 (GPU, 구독자가 있을 때만)
- 파라미터: `config/yolo_fire.yaml` (엔진 경로·문턱·클래스 이름)
- 출력 형식을 알아서 구분: YOLO26 end2end `[1, 300, 6]`(NMS 없음), YOLO26-seg end2end `[1, 300, 38]` + 원형 마스크 `[1, 32, ph, pw]`, YOLOv8/11 raw `[1, 4+클래스, 앵커]`(+NMS)
- 입력 형식도 알아서 구분: uint8 RGB `[1, H, W, 3]` (전처리를 엔진에 넣은 것, 기본) 또는 float `[1, 3, H, W]` (/255 RGB, Ultralytics 기본 내보내기)
- `cv_bridge` 는 쓰지 않는다: ROS 의 cv_bridge 는 Ubuntu OpenCV 4.5, Jetson 은 OpenCV 4.8 이라 한 프로세스에 섞이면 위험

### 한 장 처리 (전부 GPU, CUDA 그래프 하나)

```
카메라 영상 ─memcpy→ 고정(pinned) 메모리 ─H2D→ [letterbox 커널: 크기·형식이 엔진과 다를 때만]
  → TensorRT → [end2end 해석 커널: 문턱·상자 되돌리기, 조건 값 = 물체 있음] ─D2H→ 탐지 목록
  → IF 조건 노드 { [로짓 커널: 원형 해상도에서 계수 · 원형] → [칠하기 커널: 4 칸 쌍선형 > 0 → 물체 번호 영상] }
  → [결과 영상 커널: 원본에 마스크 반투명 덧칠 + 테두리, 구독자 있을 때만] ─D2H→ 마스크 + 결과 영상
```

- 영상 크기가 정해지면 위 전체를 슬롯(2 개)마다 CUDA 그래프로 담아(결과 영상 있는 그래프는 구독자가 처음 생길 때), 한 장에 `cudaGraphLaunch` 한 번만 한다 (TensorRT 를 그냥 띄우면 CPU 가 일을 거는 데만 4.6 ms)
- 조건 노드(CUDA 12.4+ `cudaGraphConditionalHandle`): 해석 커널이 장치에서 조건 값을 정해, 물체가 없는 장은 마스크 커널을 건너뛴다 (CPU 왕복 없음). 안 되면 일반 그래프 → 그래프 없음 순으로 물러난다
- GPU 를 기다릴 때 CPU 는 잠든다 (`cudaEventBlockingSync`, 바쁜 대기 없음)
- 영상은 작업 스레드가 WaitSet 으로 직접 꺼내(실행기 콜백 아님) 가장 최근 것만 쓰고, 결과를 받자마자 다음 영상을 GPU 에 건 뒤 앞 장을 발행한다. 밀리면 오래된 영상은 버린다
- 마스크: 원형 해상도(120×160)에서 계수와 내적(로짓) → 입력 크기로 쌍선형(align_corners=False) → 상자(`x1 <= u < x2`) 밖 버림 → 로짓 > 0.
  Ultralytics `process_mask(upsample=True)` 와 같은 순서라, 같은 엔진 출력으로 만든 Ultralytics 결과와 물체 번호 영상이 픽셀까지 같다 (시뮬 영상 4 장, 물체 1~3 개, 307,200 중 다른 픽셀 0)
- 원형 마스크(2.4 MB)는 CPU 로 가져오지 않는다
- 결과 영상의 마스크 덧칠·테두리(바깥 흰색 1 px + 안쪽 클래스 색 1 px)도 GPU 커널(0.1 ms). CPU 는 상자·글자만 그린다. 덧칠만 하면 소화기처럼 클래스 색과 같은 색 물체는 모양이 안 보였다

원형(proto) 가지를 엔진 둘로 나눠 물체가 없을 때 건너뛰는 것도 재 봤다 (2026-10-28, Webots 멈춤, trtexec 중앙값):
한 엔진 9.53 ms, 나누면 물체 없는 장 7.97 ms · 물체 있는 장 7.97 + 2.08 = 10.05 ms. 물체 없는 장이 25 % 넘어야 평균으로 이기고,
불·소화기가 보이는(빨리 판단해야 하는) 장은 0.5 ms 느려져 한 엔진으로 둔다.

## 엔진 만들기 (기기마다 한 번)

기본 엔진은 탐지 + 분할 모델, 입력 480×640 uint8 RGB `models/yolo26s_seg_fire_u8_480.engine` (Release [detect-y26s-seg](https://github.com/juyoung020/limo_fire_ws/releases/tag/detect-y26s-seg) `yolo26s-seg_fire7.safetensors`).
카메라(실물 Orbbec·Webots)가 640×480 rgb8 이라 그대로 엔진에 들어간다: 덧대기(letterbox) 없음, 전처리(형 변환·/255·HWC→CHW)는 엔진 안 첫 세 노드.

1. 아래 "상자만 내는 이전 모델" 순서로 `.pt` 를 만들되 다음만 바꾼다: `DetectionModel('yolo26s.yaml', …)` → `SegmentationModel('yolo26s-seg.yaml', …)`, `dm.args['task'] = 'segment'`, 파일 이름 `yolo26s-seg_fire7.*`. 마지막 `export(...)` 줄은 빼고 2 로.
2. 480×640 으로 내보내고 uint8 입력 노드를 앞에 붙인 뒤 엔진을 만든다:

```bash
cd ~/limo_fire_ws/train/detect/data/runs/y26s_seg
env -u PYTHONPATH ~/venvs/yolo_export/bin/python - <<'PY'
import onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper
from ultralytics import YOLO
YOLO('yolo26s-seg_fire7.pt', task='segment').export(format='onnx', imgsz=(480, 640), opset=17, simplify=True, nms=False)
m = onnx.load('yolo26s-seg_fire7.onnx'); g = m.graph; old = g.input[0]
_, c, h, w = [d.dim_value for d in old.type.tensor_type.shape.dim]
g.initializer.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), 'k255'))
for i, n in enumerate([helper.make_node('Cast', ['rgb_u8'], ['rgb_f'], to=TensorProto.FLOAT),
                       helper.make_node('Transpose', ['rgb_f'], ['rgb_chw'], perm=[0, 3, 1, 2]),
                       helper.make_node('Div', ['rgb_chw', 'k255'], [old.name])]):
    g.node.insert(i, n)
g.input.remove(old); g.input.insert(0, helper.make_tensor_value_info('rgb_u8', TensorProto.UINT8, [1, h, w, 3]))
onnx.checker.check_model(m); onnx.save(m, 'yolo26s-seg_fire7_u8_480.onnx')   # 출력 (1,300,38) + (1,32,120,160)
PY
/usr/src/tensorrt/bin/trtexec --onnx=yolo26s-seg_fire7_u8_480.onnx --fp16 --builderOptimizationLevel=5 \
  --saveEngine=$HOME/limo_fire_ws/src/limo_fire/limo_fire_perception/models/yolo26s_seg_fire_u8_480.engine
```

`--builderOptimizationLevel=5`(기본 3)는 전술을 더 찾아 9.8 → 9.5 ms (정밀도는 같음). 빌드는 Webots 를 `kill -STOP` 으로 멈춰도 약 27 분 걸린다.
이전 엔진(640×640 float: `yolo26s_seg_fire.engine`, 상자만 내는 `yolo26s_fire.engine`)도 노드가 그대로 돌린다 (덧대기·/255 는 GPU 커널이 한다).

### 속도 (2026-10-28, Jetson 15W, Webots 멈춤, 노드 없이 TrtYolo 만 같은 영상 300 장)

| 엔진 | GPU 계산 (trtexec, CUDA 그래프) | 한 장 전체 (영상 넣기 → 탐지·마스크 받기) | CPU 시간/장 |
|---|---|---|---|
| **seg uint8 480×640 최적화 5 (기본)** | 9.5 ms | **10.3 ms** (물체 없는 장 10.2, 결과 영상 +0.1~0.17) | 0.40 ms |
| seg uint8 480×640 최적화 3 | 9.8 ms | 10.6 ms | 0.41 ms |
| seg float 640×640 (이전) | 13.1 ms | 14.1 ms | 0.39 ms |
| 상자만 float 640×640 | — | 9.9 ms | 0.35 ms |

카메라처럼 33 ms 마다 한 장씩 넣으면 장 사이에 GPU 클럭이 내려가 21 ms 가 된다. `jetson_clocks` 로 클럭을 고정하면 10.8 ms (`limo_fire_bringup` robot.launch.xml 이 켠다).

INT8·희소성도 재 봤다 (정확도 검증 전, 속도만): Q/DQ INT8(합성곱·MatMul, 원형 가지 FP16) 8.8 ms(-10 %), 2:4 희소 FP16 은 TensorRT 가 희소 전술을 한 층도 고르지 않아 9.6 ms.
이 모델은 배치 1 에 층이 252 개라 층마다 평균 40 µs 이고 텐서 코어 사용률이 25 % 라(Nsight Compute), MAC 을 늘리는 것보다 커널 수·메모리 왕복이 시간을 정한다.

바꾸기 전 노드(CPU 전처리 12 ms + 매번 TensorRT 띄우기 + 원형 마스크 D2H + CPU 마스크)는 Webots 없이 약 30 ms 로 추정했었다.
Webots 와 같이 돌 때는 Webots 가 GPU 를 같이 써서 노드 25 ms 안팎 (같은 때 trtexec 만 돌려도 22 ms). 실물에서는 위 표 값이 나와야 한다.
두 장 겹치기(다음 장을 먼저 걸고 앞 장 발행)는 0.2 ms 정도만 줄인다: GPU 가 병목이고 CPU 일은 0.4 ms 뿐이라서.

상자만 내는 이전 모델 (Release detect-y26s-clean):

가중치는 Release [detect-y26s-clean](https://github.com/juyoung020/limo_fire_ws/releases/tag/detect-y26s-clean) 의 `yolo26s_fire7.safetensors`(우리 학습기 형식). Ultralytics 공식 YOLO26s 에 넣어 ONNX 로 내보낸 뒤 trtexec 로 엔진을 만든다. WeGo `dl_env` 를 건드리지 않게 따로 만든 venv 에서 한다.

```bash
~/miniconda3/envs/dl_env/bin/python -m venv --system-site-packages ~/venvs/yolo_export
env -u PYTHONPATH ~/venvs/yolo_export/bin/pip install "ultralytics>=8.4" safetensors onnx onnxslim "numpy<2"
cd ~/limo_fire_ws/train/detect/data/runs/y26s_clean     # safetensors 를 여기에
env -u PYTHONPATH ~/venvs/yolo_export/bin/python - <<'PY'
import torch
from safetensors.torch import load_file
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
dm = DetectionModel('yolo26s.yaml', nc=7, verbose=False)
r = dm.load_state_dict(load_file('yolo26s_fire7.safetensors'), strict=False)
assert not r.unexpected_keys and all(k.endswith('num_batches_tracked') for k in r.missing_keys)
dm.names = {0: 'fire', 1: 'smoke', 2: 'fire_extinguisher', 3: 'exit_sign', 4: 'person', 5: 'door', 6: 'stairs'}
dm.args = vars(get_cfg()); dm.model[-1].end2end = True; dm.eval()
torch.save({'model': dm, 'train_args': {}}, 'yolo26s_fire7.pt')
YOLO('yolo26s_fire7.pt').export(format='onnx', imgsz=640, opset=17, simplify=True, nms=False)  # 출력 (1, 300, 6)
PY
/usr/src/tensorrt/bin/trtexec --onnx=yolo26s_fire7.onnx --fp16 \
  --saveEngine=$HOME/limo_fire_ws/src/limo_fire/limo_fire_perception/models/yolo26s_fire.engine
```

`PYTHONPATH` 를 비우는 이유: `.bashrc` 가 `dl_env`(Ultralytics 8.3, YOLO26 모름)를 맨 앞에 둔다.

## 확인 (2026-10-26, Jetson Orin Nano, Webots 시뮬 카메라)

- trtexec FP16 GPU 시간 중앙값 10.4 ms (Webots 와 같이 돌 때), 노드 전체(전처리 포함) 약 34 ms
- 시뮬 방에서 제자리 회전: 계단 84 % · 소화기 88 % · 문 84 % · 유도등 86 % 검출. 반쯤 잘린 사람, 작은 불(사진 판)은 못 잡음

## 지도에 물체 올리기 (`object_mapper`)

탐지 상자 + 컬러에 정렬된 깊이 영상 + TF(`map ← camera_color_optical_frame`) → 물체의 지도 좌표 → `/object_mapper/markers` (MarkerArray, 클래스별 색 기둥 + 이름).
분할 모델이면 같은 시각의 `/yolo_detector/mask` 에서 그 물체 픽셀만 골라, 픽셀 무게중심 방향 + 그 픽셀 깊이의 가운데 값으로 위치를 잰다(배경·뒤 벽이 안 섞임). 불은 마스크 픽셀 수로 앞면 넓이(m²)도 재서 현황판에 보인다.
마스크가 없으면(탐지 전용 엔진) 상자 가운데 50 % 깊이의 하위 30 % 값을 거리로 쓴다(얇은 소화기 뒤 벽 무시). 같은 클래스가 0.8 m 안이면 같은 물체로 합치며, 5 번 이상 본 것만 올린다. 값은 `config/yolo_fire.yaml`.
`map` 프레임이 있어야 하므로 SLAM(Cartographer) 또는 AMCL 과 같이 돌린다.
다른 노드가 읽는 물체 목록은 `/object_mapper/objects` (`vision_msgs/Detection3DArray`, map frame, 지도에 올라간 것만): `id` 는 `fire-1` 꼴, 결과 클래스·최고 점수, 위치, `bbox.size.y` = 앞면 넓이의 제곱근(m, 모르면 0). 임무 BT([limo_fire_bt](../limo_fire_bt/README.md))가 불 위치·크기·사람 수를 여기서 읽는다. 마커는 표시용이라 읽지 않는다.

운용 화면 (`limo_fire_description` 의 `viewer.launch.xml` · `rviz/limo_fire.rviz`, 지도 만들기·Nav2 와 같은 화면):
- 왼쪽: YOLO 탐지 영상. 오른쪽: 지도 + 물체 + 로봇(축) + Nav2 경로(전역·지역), costmap 은 Displays 에서 켠다
- 도구: 2D Pose Estimate, 2D Goal Pose(Nav2 목표), Publish Point(물체 고르기)
- 물체 모양 보기: 툴바 **Publish Point** 를 누르고 지도 위 아이콘을 누르면, 그 물체가 가장 크고 확실하게 찍힌 장면(분할 마스크 덧칠·테두리가 있는 탐지 영상을 물체 둘레만 자른 것)이 왼쪽 아래 **Selected** 패널에 뜬다. 고른 아이콘은 흰 고리, 현황판 맨 아래에 종류·번호·신뢰도·본 횟수. 빈 곳을 누르면 선택 해제 (`/clicked_point` → `/object_mapper/selected`)
- 왼쪽 위 현황판(`/object_mapper/summary`, `rviz_2d_overlay_plugins`): 화재 경고, 종류별 개수, 로봇에서 가장 가까운 것과 거리
- 물체마다 원형 아이콘 배지(`icons/`, Material Symbols) + 번호 라벨(`EXT-1` …). 불은 위험 반경(`fire_zone_radius`)이 깜빡인다
- 표시 켜고 끄기: 메뉴 Panels → Displays. 지도 물체 모두 지우기: `ros2 service call /object_mapper/clear std_srvs/srv/Trigger`
- 필요한 apt 패키지: `ros-humble-rviz-2d-overlay-plugins`, `ros-humble-nav2-rviz-plugins`

```bash
# Webots: 로봇 + 저장 지도 + Nav2 + 인식 + 운용 화면 한 번에
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze perception:=true
```

### 확인 (2026-10-27, Webots 방 한 바퀴 + 웨이포인트 4 곳에서 제자리 회전)

| 정답 물체 | 지도 위치 오차 |
|---|---|
| 불(바닥) · 불(작은 것) | 0.06 · 0.09 m |
| 소화기 동 · 서 · 남서 | 0.06 · 0.19 · 0.41 m |
| 문 | 0.31 m |
| 계단 | 0.61 m (계단 원점이 한쪽 끝이라 가운데와 차이) |

틀린 3 개는 위치가 아니라 YOLO 클래스 오류: 사람(person_1) → 소화기, 탁자 → 계단, 불 근처 → 유도등. 사람 2 명·벽 높이 유도등 2 개·촛불·연기는 못 잡음.

### 분할 모델로 다시 확인 (2026-10-27, 같은 방·같은 웨이포인트, 마스크로 위치 재기)

| 정답 물체 | 상자로 잴 때 | 마스크로 잴 때 |
|---|---|---|
| 불(바닥) · 불(작은 것) | 0.06 · 0.09 m | 0.17 · 0.07 m |
| 소화기 동 · 서 · 남서 | 0.06 · 0.19 · 0.41 m | 0.14 · 0.02 · 0.33 m |
| 문 | 0.31 m | 0.19 m |
| 계단 | 0.61 m | 0.52 m |

틀린 것: 사람(person_1) → 소화기(위치는 0.14 m), 북쪽 벽 → 문 1 개. 탁자 → 계단, 불 근처 → 유도등 오류는 없어짐. 사람·유도등·촛불·연기는 여전히 못 잡음.
노드 전체 추론(전처리·마스크 포함) 약 47 ms (Webots 와 같이 돌 때).

### GPU 로 옮긴 뒤 다시 확인 (2026-10-28, uint8 480×640 엔진, 같은 방·같은 웨이포인트)

| 정답 물체 | 지도 위치 오차 (이전 마스크 → 지금) |
|---|---|
| 불(바닥) · 불(작은 것) | 0.17 · 0.07 → 0.19 · 0.12 m |
| 소화기 동 · 서 · 남서 | 0.14 · 0.02 · 0.33 → 0.14 · 0.03 · 0.28 m |
| 문 | 0.19 → 0.26 m |
| 계단 | 0.52 → 0.44 m |

틀린 것: 이전과 같은 사람(person_1) → 소화기, 북쪽 벽 → 문에 더해 작은 불 자리에 연기 1 개. 노드 25 ms 안팎 (Webots 와 같이 돌 때).

## 모델 고르기(COCO 80 종 · 화재 7 종) · 지도 위 물체 모양 · 물체 지도 파일 (2026-11-17)

모델은 설정 파일로 고른다: `yolo.launch.xml config:=yolo_coco.yaml`(기본, Ultralytics 공식 YOLO26s-seg COCO 80 종 — 의자·컵·노트북 …) ·
`config:=yolo_fire.yaml`(우리 화재 7 종 분할). 노드·물체 지도·표시는 같고 엔진 경로·클래스 이름·색(`class_colors`)만 다르다.
다른 탐지기를 같은 물체 지도에 더 붙일 수도 있다 (`extra_detections_topics` · `extra_class_names` — 기본은 비움).

- **물체 모양(발자국)**: 분할 모델이면 물체 마스크 픽셀(`footprint_stride` 칸마다)의 깊이로 3D 점을 만들어 지도 칸(`cell_size` 0.05 m)에 내린다.
  물체 깊이 가운데 값 ± 0.5 m 밖(배경)은 버리고, 여러 장에서 본 칸을 쌓아 `footprint_min_count` 번 넘게 본 칸만 클래스 색으로 칠한다 (`/object_mapper/markers`, CUBE_LIST).
  이름표는 마스크 무게중심(가운데 점)에 둔다. 상자만 내는 엔진이면 잰 너비 지름의 원.
- **밑점**: 마스크 3D 점 가운데 가장 낮은 10 % 의 평균(map). `/object_mapper/objects` 의 `results[0].pose` 에 담는다 (불이면 불 밑동). `bbox.center` 는 가운데 점, `bbox.size.x` 는 너비(m).
  깊이가 비는 픽셀(불꽃은 깊이가 자주 빔)은 버리고, 남은 점이 10 개 안 되면 밑점을 갱신하지 않는다.
- **지도 갱신 (없어진 물체 빼기)**: 지도에 올라간 물체가 이 장에서 보여야 하는데(카메라 시야 가장자리 10 % 안쪽 · 3 m 안 · 그 픽셀 깊이가 물체보다 0.3 m 넘게 앞이 아님)
  못 잡으면 못 본 수를 늘리고 다시 보면 0, `neg_misses`(20) 번이면 지도에서 뺀다(마커·목록·저장에서 빠지고, 같은 자리 탐지는 새 물체로). 사람·동물(`dynamic_classes`)은
  `dynamic_ttl` 안 보이면 뺀다. 1116_fuel_rl_1 녹화 재생: 8 이면 의자 하나가 잘못 빠졌고 20 이면 의자 5 개가 남고 떠난 사람 2 명만 빠짐. 옮긴 물체 다시 등록은 아직 실물로 못 봄.
- **사람·동물은 실시간** (`dynamic_*`): 위치·발자국은 가장 최근 장 것(평균·누적 안 함 — 예전엔 걷는 사람을 0.6 m 쯤 늦게 따라가고 지나온 자리가 발자국으로 남았다),
  같은 사람 찾기는 merge 반경 + 1.5 m/s × 못 본 시간(2 m 까지), 시야 안에서 5 장 못 보면 · 3 s 못 보면 뺌, 그 안에 다시 보이면 같은 번호로 살림.
  `/object_mapper/objects`·마커는 10 Hz (벽시계 타이머). 1116_fuel_rl_1 결정적 재생(사람 측정 111·110 번): 측정 다음 0.1 s 표시에서 가장 가까운 사람까지 중앙 0.06 → 0.04 m · 90 % 0.27 → 0.17 m,
  사람 발자국 칸 최대 269 → 57, 떠난 사람 마커는 마지막으로 본 뒤 10.1 → 3.1 s 에 지워짐, 지운 뒤 RViz 에 남는 마커 0 (DELETE 를 RViz 처럼 쌓아 확인).
- **옮긴 정지 물체** (`move_dist`·`move_frames`): 처음 자리에서 max(0.5 m, 너비/2) 넘게 떨어진 측정이 10 장 이어지면 그 자리로 옮기고 발자국을 새로 쌓는다
  (예전엔 반경 안 이동은 평균으로 천천히 끌려가고 옛 자리 발자국이 남음). 반경 밖 이동은 위 지도 갱신(옛 자리 빼기 + 새 물체). 1116 재생에서 0.5 m 는 잘못 옮김 0 (0.4 m 는 붙은 의자 1 개를 잘못 옮김) — 실제로 옮긴 녹화로는 아직 못 봄.
- **물체 지도 파일**: 한 줄에 물체 하나 — 클래스 · 번호 · 가운데 x y z · 방향 · 본 횟수 · 최고 점수 · 넓이 · 너비 · 마지막 본 시각 · 발자국 칸(`gx,gy,횟수`). map 좌표.
  `objects_file` 이 있으면 켤 때 읽고(저장한 지도로 다시 돌 때), `~/save` 는 그 파일에, `~/save_to`(std_msgs/String 경로)는 준 경로에 쓴다.
  `limo_fire_bringup save_map.launch.xml` 이 지도와 같은 이름으로 `<이름>_objects.txt` 를 옆에 저장한다. 다시 돌 때 `objects_file:=…/<이름>_objects.txt`.
- 탐사 중 보기: `ros2 launch limo_fire_rl explore_fuel_rl.launch.xml … perception:=true [perception_config:=yolo_coco.yaml]` — 탐사 RViz 가 같은 `/object_mapper/markers` 를 그린다.

### COCO 80 종 분할 엔진 만들기 (로봇에서 한 번)

Ultralytics 공식 `yolo26s-seg.pt`(COCO 80 종, 인터넷에서 처음 한 번 받음)를 화재 엔진과 같은 꼴(480×640 uint8 RGB 입력, end2end)로 내보낸다. 위 "엔진 만들기" 2 와 같은 코드에서 파일 이름만 바꾼다:

```bash
mkdir -p ~/yolo_coco && cd ~/yolo_coco
env -u PYTHONPATH ~/venvs/yolo_export/bin/python - <<'PY'
import onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper
from ultralytics import YOLO
YOLO('yolo26s-seg.pt').export(format='onnx', imgsz=(480, 640), opset=17, simplify=True, nms=False)
m = onnx.load('yolo26s-seg.onnx'); g = m.graph; old = g.input[0]
_, c, h, w = [d.dim_value for d in old.type.tensor_type.shape.dim]
g.initializer.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), 'k255'))
for i, n in enumerate([helper.make_node('Cast', ['rgb_u8'], ['rgb_f'], to=TensorProto.FLOAT),
                       helper.make_node('Transpose', ['rgb_f'], ['rgb_chw'], perm=[0, 3, 1, 2]),
                       helper.make_node('Div', ['rgb_chw', 'k255'], [old.name])]):
    g.node.insert(i, n)
g.input.remove(old); g.input.insert(0, helper.make_tensor_value_info('rgb_u8', TensorProto.UINT8, [1, h, w, 3]))
onnx.checker.check_model(m); onnx.save(m, 'yolo26s-seg_coco_u8_480.onnx')
PY
/usr/src/tensorrt/bin/trtexec --onnx=yolo26s-seg_coco_u8_480.onnx --fp16 \
  --saveEngine=$WS/src/limo_fire/limo_fire_perception/models/yolo26s_seg_coco_u8_480.engine   # WS = 로봇의 이 저장소 폴더, 뒤에 colcon build
```

`config/yolo_coco.yaml` 의 `engine_path` 는 파일 이름만 쓴다 — `models/` 의 엔진이 `colcon build` 때 설치 폴더로 복사되고 노드가 거기서 찾는다.

## 바닥 광선 거리 (가상 깊이, `~/instances`)

불꽃처럼 깊이 카메라가 비거나 틀리는 물체를 위해, 마스크 밑동(맨 아래 5 % 줄의 가운데 열) 픽셀을 지나는 카메라 광선이 바닥면(base_link 높이 − 0.145 m, `limo_pro_base.xacro`)과 만나는 거리를 잰다.
`prefer_floor_classes`(기본 fire · smoke)는 마스크 깊이와 `fuse_tol_m + fuse_tol_frac × 거리` 넘게 다르면 바닥 광선 거리를, 다른 클래스는 마스크 깊이를 쓰고, 마스크 깊이가 없으면 바닥 광선 거리를 쓴다.
물체마다 `~/instances`(그 장, Detection3DArray): `id` = "클래스|쓴 출처(mask·box·floor)|mask 거리|floor 거리", `bbox.center` = 쓴 자리, `results[0].pose` = 밑점, `bbox.size.x`·`z` = 너비·높이(마스크 크기 × 거리).
바닥면은 지금 고정(로봇 높이) — 깊이로 바닥면을 맞추는 일(바닥 위 낮은 물체 작업)이 끝나면 그 면으로 바꿀 자리.

1116_fuel_rl_1 녹화(실험실, 바닥에 선 의자 · 사람)에서 마스크 깊이와 견준 바닥 광선 거리 (`depth_check_csv`, 61 개):

| 마스크 깊이 | 수 | 바닥 광선 − 마스크 (가운데) | 차 크기 (가운데) |
|---|---|---|---|
| 0~1 m | 2 | −0.07 m | 0.19 m |
| 1~1.5 m | 4 | −0.23 m | 0.24 m |
| 1.5~2 m | 40 | −0.20 m | 0.22 m |
| 2~3 m | 9 | −0.23 m | 0.34 m |
| 3~5 m | 6 | −2.62 m | 2.71 m |

바닥 광선은 물체 앞 밑동, 마스크 깊이는 물체 픽셀 가운데 값이라 3 m 안에서는 약 0.2 m 짧게 나오는 것이 맞는 방향이다(의자·사람의 앞뒤 깊이). 3 m 넘으면 밑동이 바닥이 아니거나(가려짐) 각이 얕아 크게 틀린다 — 3 m 안에서만 믿는다.
아직 불로는 재지 않았다 (불 녹화 없음).

## 스레드·메모리 (2026-11-17)

로봇에서 판단 모델(llama-server d1-3B, 약 4.1 GB + 요청 버퍼)이 인식 + RViz 와 같이 돌 때 GPU 메모리 부족(NvMap error 12)으로 죽어 인식 쪽을 줄였다.

- **한 스레드**: 컨테이너는 `component_container`(실행기 스레드 1). `yolo_detector` 는 영상 구독을 실행기에 올리지 않고 작업 스레드가 WaitSet 으로 직접 꺼낸다
  (GPU 를 기다리는 스레드 하나는 그대로) — 다른 콜백이 길어도 영상 받기가 안 밀린다. 받기 메시지는 몇 개를 돌려 써서 장마다 921 KB 할당이 없다.
  `object_mapper` 는 받기 콜백에서 일하지 않는다(예전엔 콜백 안에서 처리 + TF 를 최대 0.2 s 기다림 — 1116 실시간 재생에서 2500 장 중 71 번 33 ms 넘게, 최대 232 ms 막음):
  탐지·깊이 짝은 작업 줄(`max_pending` 10 장, 넘치면 가장 오래된 장을 버려 최신 쪽만 남김)에 넣기만 하고, `process_period`(20 ms, 노드 시계) 타이머가
  그 시각 TF 가 있는 장부터 처리한다(`tf_wait` 지나도 없으면 버림). 물체의 "마지막 본 시각"은 처리한 때가 아니라 받은 때. TF 리스너도 따로 스레드 없이 이 노드 콜백.
  깊이 영상(다른 프로세스 → 역직렬화)은 받기 메시지를 돌려 쓰는 메모리 전략으로 받아 장마다 영상 버퍼를 새로 할당하지 않는다.
- **재기**: `object_mapper` 가 `stats_period`(10 s)마다 한 줄 — 탐지·깊이 받은 수, 짝(ApproximateTime, `max_sync_dt` 0.05 s) 수와 짝 못 지은 탐지 비율,
  줄 지연(받음 → 처리 시작, 평균·최대), 장당 처리 시간, 탐지 시각부터 처리 끝까지 나이, 버린 장(TF 없음·줄 넘침).
- **보는 쪽이 있을 때만**: 결과 영상(GPU 덧칠 + 상자)은 `~/image` 구독자가 있을 때만 만들고, 그 GPU 버퍼(장치 1 장 + 고정 메모리 2 장 = 2.8 MB)와 결과 영상 그래프 2 개는
  처음 필요할 때 만들어 `vis_release_s`(5 s) 동안 구독자가 없으면 푼다. 예전엔 `object_mapper` 가 대표 사진용으로 늘 구독해서 매 장 만들었다 —
  이제 매퍼는 `~/selected` 를 보는 쪽(RViz Selected 칸)이 있을 때만 결과 영상을 구독한다
  (10 장 921 KB = 9.2 MB 를 안 함). 그래서 RViz 를 안 켠 동안 찍힌 물체는 대표 사진이 없다(다시 보이면 생김).
- **물체 지도 픽셀 반복**: 마스크를 상자 안에서 한 번만 훑어 위치·거리·밑동·발자국이 같이 쓰고(예전 세 번), 물체가 없는 줄은 SIMD 로 건너뛰며(분기 없는 OR 모음 — 컴파일러가 NEON·SSE 로),
  깊이 열 번호는 표로, 점마다 나누던 것은 역수 곱, 작업 버퍼는 다시 쓴다. 빌드 종류를 안 주면 Release(예전엔 colcon 기본이라 -O0), Orin 이면 `-mcpu=cortex-a78ae`.

확인 (PC, x86 i7·GCC 11, TensorRT 없음 — GPU 부분은 TensorRT 10.3 헤더로 문법만):

| 무엇 | 전 (2072d35) | 후 |
|---|---|---|
| 물체 지도 파일 (1116_fuel_rl_1 결정적 재생: 녹화 수신 시각·한 스레드·같은 순서·TF 0.25 s 미리, `dynamic_classes:=[none]`·`move_frames:=0` — 옛 동작) | 기준 (발자국 칸만 (gx, gy) 로 정렬해 비교) | **바이트 같음**. 점마다 나눗셈 → 역수 곱으로 바꾼 판과 되돌린 판도 바이트 같음 |
| 역수 곱 차이 (모든 640×480 픽셀 × 깊이 0.2~5 m 1 mm 간격, 이 녹화 fx 489.4) | — | 좌표 33 % 가 마지막 비트 다름 · 최대 4.4e-16 m · 0.05 m 칸 번호 바뀜 0 / 14.7 억 |
| 탐지 한 개 픽셀 일 (마스크 훑기 + 위치 + 밑동 + 발자국, Release, 104 장 × 5, 3 번 중 최소) | 338 µs (230 + 49 + 60) | 175 µs (131 + 11 + 3 + 30) |
| 마스크 없을 때 상자 깊이 | 95 µs | 25 µs |
| 장당 매퍼 시간 · 할당 (ROS 전달 포함, 2687 장) | 0.151 ms · 41.7 번 · 23 KB | 0.113 ms · 32.4 번 · 3 KB (+ 실물은 깊이 영상 614 KB 역직렬화 할당도 없음 — 받기 메시지 돌려 씀) |
| 짝 못 지은 탐지 (결정적 재생) | — | 2687 중 58 = 2.2 % · 줄 넘침 0 · TF 없음 0 |
| 가짜 YOLO + 매퍼 한 프로세스, 실시간 재생 (PC 부하 평균 15 — 입력이 12~18 장/초로 밀림) RSS · 스레드 | 한 스레드 85 MB · 16 / 멀티 112 MB · 27 | 한 스레드 52 MB · 15 |
| 같은 시험: 매퍼 10 s 통계 (새 판만) | — | 짝 못 지은 탐지 23~49 % (깊이·영상이 부하로 듬성듬성 와서 — 로봇에서 다시 잴 것), 줄 지연 평균 16~20 · 최대 117 ms (`--clock` 40 Hz 라 노드 시계 타이머가 25 ms 단위), 처리 평균 0.3~0.5 ms, 버림 TF 2 장 |

가짜 YOLO 는 TensorRT 대신 11 ms 잠자고 상자·마스크 하나를 내는 대역 — 실행기·영상 받기 구조만 본다. 옛 판 RSS 가 큰 것은 결과·원본 영상을 늘 구독해 10 장씩 들고 있던 몫이 크다.
NEON: `row_has`·깊이 열 표 반복은 x86 GCC 11 `-O3` 에서 "loop vectorized using 16 byte vectors" 를 확인했고, aarch64 는 이 PC 에 교차 컴파일러가 없어 못 봄(같은 GCC 자동 벡터화라 NEON 16 바이트가 기대값).

Orin 메모리 어림 (기본: yolo_coco + 매퍼, search 끔, 로봇에서 RViz 안 켬):
- GPU(NvMap): 결과 영상 버퍼 2.8 MB + 그래프 2 개(어림 1~4 MB) ≈ **4~7 MB** 덜 씀. 엔진 가중치(~21 MB)·TensorRT 작업 메모리·CUDA 문맥은 그대로 —
  켤 때 로그 `GPU 메모리(대략): 엔진 … 실행 문맥 … (TensorRT 작업 메모리 …)` 로 잰다. 지연 로딩은 CUDA 12.2+ 리눅스(535 드라이버~)에서 이미 기본이라([CUDA 12.2 릴리스 노트](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html)) JetPack 6 에선 보통 그대로 — 환경에서 EAGER 로 꺼져 있던 경우만 준다(TensorRT 가 `CUDA lazy loading is not enabled` 경고를 냈었는지 로봇에서 확인).
- CPU(RSS 436 MB 였음): 매퍼 영상 줄 18 MB + 스레드(멀티 → 한 스레드 컨테이너·TF 스레드) ≈ **20~35 MB** 덜 씀
  (PC 대역 시험 85 → 52 MB). 깊이 영상(614 KB × 30 장/초)·카메라 영상(921 KB)을 받기 메시지로 돌려 써 할당·페이지 폴트가 줄어든다(RSS 보다는 CPU 몫).
- 판단 모델 OOM 을 막기엔 작다. 큰 것은 RViz 를 로봇 밖(PC)에서 띄우기, 엔진 작업 메모리 줄이기(아래) 쪽이다.
- 엔진 작업 메모리 줄이기 (로봇에서 잴 것): 같은 ONNX 로 `trtexec … --memPoolSize=workspace:256M` 엔진을 만들어 노드 로그의 TensorRT 작업 메모리와 `추론 … ms` 를 지금 엔진과 견준다.
  빌드 작업 공간 기본은 GPU 메모리 전체라, 작업 메모리 = 활성 메모리 + 고른 전술의 스크래치(그 한도 안)다(같은 TensorRT 문서).

## 인식 벤치마크 (`perception_benchmark`)

Webots 정답(같은 카메라의 Recognition 분할, `limo_fire_sim`)과 장마다 비교해 상자·마스크 P·R·F1·mAP50·mAP50-95·IoU, 크기별 재현율, 속도, 물체 지도 P·R·F1·위치 오차를 표로 낸다.
짝짓기·AP 는 학습기와 같은 `include/limo_fire_perception/det_metrics.hpp`. 실행·지표 정의·결과는 [docs/benchmarks](../../../docs/benchmarks/README.md).

```bash
ros2 launch limo_fire_bringup sim.launch.xml world:=limo_fire_maze benchmark:=true
```
