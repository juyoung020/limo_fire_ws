# wego_limo_pro — 실물 LIMO Pro 에 깔려 있던 WeGo 코드

2026-10-26 실물 로봇(Jetson Orin Nano)의 `~/wego_ws/src` 를 그대로 복사했다. 고치지 않는다.
GitHub [WeGo-Robotics/limo_pro](https://github.com/WeGo-Robotics/limo_pro) `7ef355c` 와 다르다(EKF 설정, 카메라 틸트 TF, launch, 차선 인식 코드 등) — 실제로 돌아가는 쪽이 이것이라 이걸로 바꿨다.

| 폴더 | 내용 | 비고 |
|---|---|---|
| `limo_ros2` | `limo_base`(바퀴·IMU 시리얼 드라이버), `limo_description`(URDF), `limo_msgs` | |
| `ydlidar_ros2_driver` | YDLIDAR T-mini Plus 드라이버 | |
| `OrbbecSDK_ROS2` | Orbbec Dabai DC1 드라이버(`orbbec_camera`) | [orbbec/OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2) `0b2b33d` 에 WeGo 가 손댄 것 |
| `robot_localization` | EKF (`params/limo_ekf.yaml`) | |
| `wego`, `wego_2d_nav`, `wego_msgs` | 대시보드가 부르는 launch·Nav2·Cartographer 설정 | 우리 판: `limo_fire_bringup`, `limo_fire_navigation` |
| `limo_ros2_application` | 차선 따라가기 예제 | 우리 판: `lane_tracking.launch.xml` |
| `dl_ros2_application`, `dl_ros2_msgs` | 딥러닝 주행·YOLO·사진 찍기·라벨러·TensorRT 변환 예제 | 우리 판: `dl_tracking.launch.xml` |
| `lift_ctr` | 리프트 제어 (설명 없음) | |

## 넣지 않은 것 (로봇에만 있음)

| 로봇 경로 | 이유 |
|---|---|
| `~/wego_ws/src/navigation2` | Nav2 humble `1.1.19-4-gc8c8aba2` 원본에서 `nav2_system_tests` 만 지운 것. [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2) 에서 받으면 같다 |
| `~/wego_ws/src/dataset` | WeGo 예제 학습 사진 4,858 장 + YOLO 라벨 (221 MB) |
| `~/wego_ws/src/{dl_ros2_application,limo_ros2_application}/weight` | 학습 가중치 `.pt`·`.onnx`·`.h5`·`.engine` (116 MB). `.engine` 은 기기마다 다시 만들어야 한다 |

실물에서는 `~/wego_ws` 에 빌드된 것을 `env.sh` 가 밑에 깐다. 이 폴더는 우리 워크스페이스 전체 빌드에서도 같이 빌드되니, 평소에는 `colcon build --packages-select <우리 패키지>` 로 필요한 것만 빌드한다.
