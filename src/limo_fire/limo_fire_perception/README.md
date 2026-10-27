# limo_fire_perception

카메라 영상에서 불·연기·소화기·유도등·사람·문·계단을 찾는 C++ TensorRT YOLO 노드. 실물 Orbbec 카메라와 Webots 시뮬 카메라에 똑같이 쓴다.

```bash
ros2 launch limo_fire_perception yolo.launch.xml          # 탐지 + rqt_image_view
```

- 입력: `/camera/color/image_raw` (rgb8·bgr8)
- 출력: `/yolo_detector/detections` (`vision_msgs/Detection2DArray`), 상자 그린 영상 `/yolo_detector/image`
- 분할(-seg) 엔진이면 물체 번호 영상 `/yolo_detector/mask` (mono8, 값 = `detections` 안 순서 + 1, 0 = 배경, 겹치면 작은 물체가 위)도 같은 시각으로 내고, 탐지 영상에 마스크를 반투명하게 칠한다
- 파라미터: `config/yolo_fire.yaml` (엔진 경로·문턱·클래스 이름)
- 출력 형식을 알아서 구분: YOLO26 end2end `[1, 300, 6]`(NMS 없음), YOLO26-seg end2end `[1, 300, 38]` + 원형 마스크 `[1, 32, 160, 160]`, YOLOv8/11 raw `[1, 4+클래스, 앵커]`(+NMS)
- `cv_bridge` 는 쓰지 않는다: ROS 의 cv_bridge 는 Ubuntu OpenCV 4.5, Jetson 은 OpenCV 4.8 이라 한 프로세스에 섞이면 위험

## 엔진 만들기 (기기마다 한 번)

기본 엔진은 탐지 + 분할 모델 `models/yolo26s_seg_fire.engine` (Release [detect-y26s-seg](https://github.com/juyoung020/limo_fire_ws/releases/tag/detect-y26s-seg) `yolo26s-seg_fire7.safetensors`). 아래 순서에서 다음만 바꾼다:
`DetectionModel('yolo26s.yaml', …)` → `SegmentationModel('yolo26s-seg.yaml', …)`, `dm.args['task'] = 'segment'`, `YOLO(…, task='segment')`, 파일 이름 `yolo26s-seg_fire7.*`, `--saveEngine=…/models/yolo26s_seg_fire.engine`.
출력은 `(1, 300, 38)` + `(1, 32, 160, 160)`. trtexec FP16 GPU 시간 중앙값 14.4 ms (Webots 와 같이 돌 때, 탐지만 하는 모델은 10.4 ms).
마스크 계산은 Ultralytics `process_mask` + `scale_masks` 와 같다 (같은 ONNX 출력에서 마스크 IoU 1.000 확인).

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
