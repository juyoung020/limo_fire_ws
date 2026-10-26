# limo_fire_perception

카메라 영상에서 불·연기·소화기·유도등·사람·문·계단을 찾는 C++ TensorRT YOLO 노드. 실물 Orbbec 카메라와 Webots 시뮬 카메라에 똑같이 쓴다.

```bash
ros2 launch limo_fire_perception yolo.launch.xml          # 탐지 + rqt_image_view
```

- 입력: `/camera/color/image_raw` (rgb8·bgr8)
- 출력: `/yolo_detector/detections` (`vision_msgs/Detection2DArray`), 상자 그린 영상 `/yolo_detector/image`
- 파라미터: `config/yolo_fire.yaml` (엔진 경로·문턱·클래스 이름)
- 출력 형식 두 가지를 알아서 구분: YOLO26 end2end `[1, 300, 6]`(NMS 없음), YOLOv8/11 raw `[1, 4+클래스, 앵커]`(+NMS)
- `cv_bridge` 는 쓰지 않는다: ROS 의 cv_bridge 는 Ubuntu OpenCV 4.5, Jetson 은 OpenCV 4.8 이라 한 프로세스에 섞이면 위험

## 엔진 만들기 (기기마다 한 번)

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
상자 가운데 50 % 깊이의 하위 30 % 값을 거리로 쓰고(얇은 소화기 뒤 벽 무시), 같은 클래스가 0.8 m 안이면 같은 물체로 합치며, 5 번 이상 본 것만 올린다. 값은 `config/yolo_fire.yaml`.
`map` 프레임이 있어야 하므로 SLAM(Cartographer) 또는 AMCL 과 같이 돌린다.

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
