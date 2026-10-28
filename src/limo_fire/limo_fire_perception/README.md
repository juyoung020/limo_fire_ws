# limo_fire_perception

카메라 영상에서 불·연기·소화기·유도등·사람·문·계단을 찾는 C++ TensorRT YOLO 노드. 실물 Orbbec 카메라와 Webots 시뮬 카메라에 똑같이 쓴다.

```bash
ros2 launch limo_fire_perception yolo.launch.xml          # 탐지 + rqt_image_view
```

- `yolo_detector`·`object_mapper` 는 컴포넌트(`limo_fire_perception::YoloDetectorNode`, `::ObjectMapperNode`). 같은 이름의 실행 파일도 있다
- 기본은 둘을 멀티스레드 컨테이너 `/perception_container` 한 프로세스에 올리고 프로세스 안 전달(intra-process)로 탐지·마스크를 복사 없이 넘긴다.
  한 스레드 컨테이너는 매퍼 콜백 뒤에 영상 콜백이 밀려 5 초에 18~20 장을 버렸고, 멀티스레드는 0 장
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
  → [결과 영상 커널: 원본에 마스크 반투명 덧칠, 구독자 있을 때만] ─D2H→ 마스크 + 결과 영상
```

- 영상 크기가 정해지면 위 전체를 슬롯(2 개) × 결과 영상 유무마다 CUDA 그래프로 담아, 한 장에 `cudaGraphLaunch` 한 번만 한다 (TensorRT 를 그냥 띄우면 CPU 가 일을 거는 데만 4.6 ms)
- 조건 노드(CUDA 12.4+ `cudaGraphConditionalHandle`): 해석 커널이 장치에서 조건 값을 정해, 물체가 없는 장은 마스크 커널을 건너뛴다 (CPU 왕복 없음). 안 되면 일반 그래프 → 그래프 없음 순으로 물러난다
- GPU 를 기다릴 때 CPU 는 잠든다 (`cudaEventBlockingSync`, 바쁜 대기 없음)
- 구독 콜백은 가장 최근 영상만 남기고 바로 돌아오고, 작업 스레드가 결과를 받자마자 다음 영상을 GPU 에 건 뒤 앞 장을 발행한다. 밀리면 오래된 영상은 버린다
- 마스크: 원형 해상도(120×160)에서 계수와 내적(로짓) → 입력 크기로 쌍선형(align_corners=False) → 상자(`x1 <= u < x2`) 밖 버림 → 로짓 > 0.
  Ultralytics `process_mask(upsample=True)` 와 같은 순서라, 같은 엔진 출력으로 만든 Ultralytics 결과와 물체 번호 영상이 픽셀까지 같다 (시뮬 영상 4 장, 물체 1~3 개, 307,200 중 다른 픽셀 0)
- 원형 마스크(2.4 MB)는 CPU 로 가져오지 않는다
- 결과 영상의 마스크 덧칠도 GPU 커널(0.1 ms). CPU 는 상자·글자만 그린다

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

`rviz:=true` 화면 (`limo_fire_description/rviz/object_map.rviz`):
- 왼쪽: YOLO 탐지 영상. 오른쪽: 지도 + 물체 + 로봇(축)
- 왼쪽 위 현황판(`/object_mapper/summary`, `rviz_2d_overlay_plugins`): 화재 경고, 종류별 개수, 로봇에서 가장 가까운 것과 거리
- 물체마다 원형 아이콘 배지(`icons/`, Material Symbols) + 번호 라벨(`EXT-1` …). 불은 위험 반경(`fire_zone_radius`)이 깜빡인다
- 표시 켜고 끄기: 메뉴 Panels → Displays. 지도 물체 모두 지우기: `ros2 service call /object_mapper/clear std_srvs/srv/Trigger`
- 필요한 apt 패키지: `ros-humble-rviz-2d-overlay-plugins`

```bash
# Webots (slam:=true 로 켠 뒤)
ros2 launch limo_fire_perception yolo.launch.xml use_sim_time:=true viewer:=false rviz:=true
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
