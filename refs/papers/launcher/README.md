# 발사기 조준 선행 연구 모음 (탁구공 → 불 모양 표적)

LIMO 위 팬·틸트 발사기(팬 MG996R, 틸트 DS3218)로 YOLO 불 표적을 맞혀 쓰러뜨리는 문제에 참고할 논문들이다.
링크는 모두 직접 확인했다. PDF는 합법 공개본(arXiv, MDPI, 학회지, 저자·대학 저장소)만 받았고, 공개본이 없으면 "PDF 없음"으로 표시했다.
요약은 각 PDF의 초록과 방법·실험 절을 읽고 썼다. PDF가 없는 논문은 출판사·색인 페이지의 초록만 보고 썼으므로 그렇게 표시했다.

## 한눈에 보기

| 파일 | 논문(저자, 연도, 학회/저널) | 링크 | 한 줄 요지 |
| --- | --- | --- | --- |
| `2024_kim_firefighting_robot_system.pdf` | Kim 외, 2024, 로봇학회 논문지(JKROS) 19(3) | [DOI](https://doi.org/10.7746/jkros.2024.19.3.287) | Jetson Nano+YOLO+4자유도 팔 펌프. 거리에 따라 각도 비율을 바꾸는 경험식과 ±15° 좌우 쓸기 |
| `2016_wu_firefighting_video_closed_loop.pdf` | Wu 외, 2016, Int. J. Safety and Security Eng. 6(2) | [DOI](https://doi.org/10.2495/SAFE-V6-N2-254-269) | 노즐에 붙인 단안 카메라로 불과 물 착지점을 함께 보고, 오차로 팬·틸트를 계속 고침. 고각 보정은 구간별 선형 경험식 |
| `2015_mcneil_fire_suppression_feedback_control.pdf` | McNeil, 2015, Virginia Tech 박사논문 (본문 2장 = Fire Technology 2016) | [저장소](https://vtechworks.lib.vt.edu/handle/10919/64784), [논문 DOI](https://doi.org/10.1007/s10694-016-0564-8) | IR 스테레오로 불 위치를 잡고, 물줄기 각도를 영상으로 추정해 칼만+PI(불감대 ±1°)로 노즐을 보정 |
| `2019_zhu_jet_trajectory_nearfield_vision.pdf` | Zhu, Li, Lin, Zhao, 2019, Sensors 19(3):690 | [DOI](https://doi.org/10.3390/s19030690) | 노즐 근처 카메라로 물줄기 앞부분만 찍어 기울기 특징 3개로 사거리를 회귀 예측(정확도 ≥94%) |
| `2021_lin_two_stage_water_jet_landing.pdf` | Lin, Ji, He, Chen, 2021, Sensors 21(8):2704 | [DOI](https://doi.org/10.3390/s21082704) | 1단계: 거리→피치 다항식 오프라인 적합. 2단계: 카메라로 본 착지 오차로 계수를 역전파로 온라인 갱신 |
| (PDF 없음(유료)) | Pan, Li, Zhu 외, 2024, Control Engineering Practice 144:105816 | [DOI](https://doi.org/10.1016/j.conengprac.2023.105816) | 물줄기 지연 모델을 넣은 시각 예측제어(제목·참고문헌만 확인, 초록 미공개) |
| `2025_pozharkova_fire_monitor_jet_forecast.pdf` | Pozharkova, Chentsov, 2025, Robotics 14(12):188 | [DOI](https://doi.org/10.3390/robotics14120188) | 바람 속 물줄기 궤적을 CFD로 만든 데이터로 MLP 학습, 실측 RMSD 0.48 m, 추론 9 ms |
| `2025_quenzel_aerial_fire_monitor_aiming.pdf` | Quenzel 외, 2025, IEEE SSRR 2025 | [arXiv](https://arxiv.org/abs/2511.14504) | 드론이 불 위치를 삼각측량하고 사다리차 방수포가 탄도 모델로 조준. 관련연구 정리가 좋음 |
| `2019_zeng_tossingbot.pdf` | Zeng 외, 2019/2020, RSS 2019 → IEEE T-RO 2020 | [arXiv](https://arxiv.org/abs/1903.11239), [DOI](https://doi.org/10.1109/TRO.2020.2988642) | 탄도식으로 구한 발사 속도에 신경망 잔차 δ를 더함(Residual Physics). 실물 투척 정확도 84.7% |
| (PDF 자동 다운로드 실패) | Wirkus, de Gea Fernández, Kassahun, 2012, ERLARS 2012 워크숍 | [DFKI 페이지](https://robotik.dfki-bremen.de/en/research/publications/6406) | 단순 시뮬 모델로 투척을 최적화하고, 실물 데이터로 학습한 신경망이 시뮬 오차를 보정(초록 기준) |
| `2022_kasaei_throwing_moving_basket.pdf` | H. Kasaei, M. Kasaei, 2022, arXiv 프리프린트 | [arXiv](https://arxiv.org/abs/2210.00609) | Gazebo에서 SAC/DDPG로 던지기 파라미터 4개를 학습해 실물에 그대로 이식. 실물 성공률 85~90% |
| `2013_kormushev_rl_robotics_archery.pdf` | Kormushev, Calinon, Caldwell, 2013, Robotics 2(3):122 | [DOI](https://doi.org/10.3390/robotics2030122) | iCub 활쏘기: 화살이 꽂힌 2D 오차 벡터로 회귀(ARCHER)해 실물 8번 만에 과녁 중앙 명중 |
| `2023_dittrich_aimy_launcher.pdf` | Dittrich 외, 2023, ICRA 2023 | [arXiv](https://arxiv.org/abs/2210.06048) | 바퀴 3개 탁구공 발사기(오픈소스). 3761개 궤적으로 "목표 착지점→제어값" 역모델 신경망 학습 |
| `2018_schneider_tabletennis_trajectories.pdf` | Schneider 외, 2018, Applied Sciences 8(12):2595 | [DOI](https://doi.org/10.3390/app8122595) | 탁구공 운동방정식(중력+항력+마그누스)을 CUDA로 5억 개 적분. 속도 의존 C_D·C_L 적합식 사용 |
| `2026_conti_tabletennis_physics_models.pdf` | Conti 외(Sony AI), 2026, arXiv(RA-L 투고) | [arXiv](https://arxiv.org/abs/2606.28805) | C_D·C_M을 레이놀즈수·회전비 함수로 표로 만들어 착지 오차 중앙값 59% 감소. C_D ≈ 0.37~0.55 |
| `2023_qin_robomaster_ep_yolo_aiming.pdf` | Qin, Xu, 2023, arXiv 프리프린트 | [arXiv](https://arxiv.org/abs/2312.05055) | RoboMaster EP 짐벌: YOLOv5+칼만+PID 추적. 거리별 탄 낙차를 실측해 회귀(KNN, R²=0.987)로 보정 |
| (PDF 없음(유료)) | Sun 외, 2019, IEEE RCAR 2019, pp. 907–912 | [DOI](https://doi.org/10.1109/RCAR47638.2019.9044108) | 몸체 고정 카메라 ↔ 팬틸트 포신 캘리브레이션: 포신 레이저를 평면 거울로 반사시켜 회전행렬을 구함(초록 기준) |

---

## A. 소방 로봇·방수포의 조준

### Kim 외 2024 — 초기 화재 진압을 위한 자율주행 소방 로봇 시스템 설계 (JKROS)
- **어떻게 하나**
  - Jetson Nano 4GB, RPLidar A1, FHD 카메라, IR 불꽃 센서를 쓴다. ROS Melodic에서 Cartographer로 지도를 만들고 AMCL과 DWA로 순찰한다.
  - 불 탐지는 Kaggle 사진 2000장으로 전이학습한 YOLOv4-tiny다. 박스 중심 좌표를 4자유도 팔(D-H 역기구학)에 넘기고, 펌프(6 L/min)로 물을 뿌린다.
  - Yaw축이 목표를 벗어나는 문제는 "거리가 멀수록 각도 비율을 조절"하는 Map 함수식 이득 보정으로 풀었다. 작은 불은 좌표 오차를 메우려고 좌우 15°씩 반복해 쓸었다.
  - 50회 시험에서 탐지 성공률 98%, 1.2 m 거리 불을 평균 21초에 껐다. 밝은 환경에서는 성공률이 50% 미만으로 떨어졌다.
- **우리에게**: 구성이 우리와 거의 같다(Jetson+라이다+YOLO+작은 조준기). 우리 baseline인 "거리별 보정표"가 이 논문의 Map 함수 보정과 같은 계열이다. 조준 정확도는 따로 재지 않았으므로 성능 기준선으로는 쓰기 어렵다.

### Wu 외 2016 — Firefighting Robot with Video Full-Closed Loop Control
- **어떻게 하나**
  - 2관절(수평 회전 + 수직 피치) 방수 로봇의 2번째 링크에 단안 카메라를 노즐과 평행하게 달았다. 4.2–4.5 µm 협대역 필터로 불을 찾고, 필터를 기계식으로 빼서 물줄기를 본다.
  - 불의 지면 좌표는 설치 높이 H와 핀홀 모델, 관절각으로 구한다(지면 평면 가정 단안 측위).
  - 고각 보정 Δθ는 이론식 대신 실험으로 만든 **구간별 선형식**이다. 35°~65°를 네 구간(마지막 60°~65°)으로 나눠 기울기 k1~k4를 둔다(설치 높이·수압마다 보정, 65° 넘으면 안 씀).
  - 그 뒤 영상에서 물줄기 화소를 줄 단위로 추적하고, 직선을 적합해 착지점을 찾는다. 착지점과 불 중심의 차이로 팬·틸트를 계속 고친다.
  - 25 m 옥상에서 반경 60 m 안의 불을 대상으로 시험했다. 위치 오차는 1~2 m였고, 약 50회 시험에서 평균 1분 53초에 껐다. 실패 3회는 물줄기가 불판을 엎은 경우였다.
- **우리에게**: "카메라를 발사기와 같이 팬·틸트에 올리는" 안의 원형이다. 우리는 공이 한 발씩 나가므로, 착지점 피드백을 **발사 사이마다 한 번씩 보정**하는 형태로 바꿔야 한다. 고각 보정을 구간별 경험식으로 만든 방식은 우리 보정표와 같다.

### McNeil 2015 — Autonomous Fire Suppression Using Feedback Control for a Firefighting Robot (박사논문, 2장은 Fire Technology 2016)
- **어떻게 하나**
  - FLIR A35 IR 카메라 두 대(기준선 0.20 m)로 스테레오 깊이를 구한다. 2–5 m 거리에서 오차는 약 9.3%다. 노즐은 카메라와 따로 Dynamixel 팬·틸트 위에 있다.
  - 첫 조준(2장)은 물이 직선으로 날아간다고 가정한다. 이후 불 크기 비율로 판단한다. 비율이 0.9 이상이면 물이 짧다고 보고 피치를 0.5° 올리고, 그 아래면 불 중심 이동 방향으로 0.5°씩 움직인다. 불이 10% 아래로 줄면 좌우로 쓴다.
  - 5장에서는 IR 영상의 물 분무를 베이즈 분류로 찾는다. 스테레오 점들에 궤적 모델을 맞춰(황금분할 탐색) 실제 yaw·pitch를 추정하고, 칼만 필터를 거쳐 PI 제어(Kp 0.75, Ki 0.075, 불감대 ±1°)를 한다.
  - 2장 실험에서 2.8–5.5 m 거리의 40–50 kW 목재 불을 맑은 시야 3.72 s, 저시정 4.49 s에 진압했다. 5장에서는 초기 편향 최대 5°를 넣고도 진압했고(5장 요약은 정상 11.15 s·편향 16.93 s, 결론부는 15.7 s로 원문 안에서 값이 섞임), 원격조종보다 3배 이상 빨랐다.
- **우리에게**: "모델로 첫발 → 실제 탄착을 보고 칼만+PI로 고침 → 작은 오차는 불감대로 무시"라는 구조를 그대로 빌릴 수 있다. 이것이 RL이 실패할 때 받아 줄 BT Fallback의 뼈대가 된다. 카메라와 노즐이 분리된 구성이라 카메라–노즐 변환도 다룬다.

### Zhu, Li, Lin, Zhao 2019 — Real-Time Monitoring of Jet Trajectory during Jetting Based on Near-Field Computer Vision (Sensors)
- **어떻게 하나**
  - 방수포에서 40 cm 떨어진 웹캠(1280×960, 6 fps)으로 물줄기 **앞부분만** 찍는다. 원근 변환, 영상 강조, 분할을 거쳐 궤적 중심선을 뽑는다.
  - 특징 3개(시작점 기울기, 끝점 기울기, 전체 기울기)를 입력으로 하는 다중 선형회귀로 사거리를 예측한다.
  - 피치 10–35°, 펌프 5–40 Hz(유속 5.4–31.1 m/s)에서 150세트로 학습하고 30세트로 검증했다. 예측 정확도는 94% 이상, 처리 시간은 0.88 s 미만이었다.
- **우리에게**: 비행 초반만 보고 착지점을 미리 맞히는 발상은, 카메라 시야가 좁아 표적까지 공을 끝까지 못 볼 때 쓸모가 있다. 다만 우리 공은 30 fps에서 몇 프레임만 잡히므로 실제 적용은 어렵다. 같은 연구실(CUMT)이 Pan 2024의 저자들이다.

### Lin, Ji, He, Chen 2021 — Two-Stage Water Jet Landing Point Prediction Model for Intelligent Water Shooting Robot (Sensors)
- **어떻게 하나**
  - 2자유도 물대포, 광전 카메라 두 대(거리 측정), 흔들림 보정용 자이로로 구성된다. Yaw는 기하식으로 바로 구한다.
  - 1단계: 바람 없음, 펌프 일정 조건에서 피치를 −9°~32°까지 1° 간격으로 바꾸며 각 20회 평균 사거리를 쟀다. MATLAB polyfit으로 "거리→피치" 다항식을 만든다.
  - 2단계: 카메라로 착지점과 표적의 차이를 재고, 그 오차로 다항식 계수를 **역전파로 온라인 갱신**한다. 바람, 펌프 출력 변화, 기어 백래시를 실시간으로 흡수한다.
  - 시뮬레이션 피치 예측 오차는 0.1° 이내였다. 실물에서는 착지점이 표적에서 0.5 m 벗어났을 때 0.3 s 만에 고쳤다. GA-BP보다 파라미터는 1/4, 순전파 시간은 1/7이었다.
- **우리에게**: 우리 "거리별 틸트 보정표"를 다항식으로 두고, 빗맞은 발마다 계수를 조금씩 고치는 방식으로 그대로 옮길 수 있다. 실물 학습량이 매우 적다는 점이 RL보다 유리하다. 1단계 데이터 수집 절차(각도 1° 간격, 각 20회)도 실험 계획에 참고할 만하다.

### Pan 외 2024 — Visual Predictive Control of Fire Monitor with Time Delay Model of Fire Extinguishing Jet (Control Engineering Practice 144) — PDF 없음(유료)
- **어떻게 하나** (초록이 공개되지 않아 제목·서지·참고문헌 목록만 확인했다)
  - 제목대로 물줄기 비행에 걸리는 **시간 지연**을 모델에 넣고, 영상 기반 예측제어(MPC 계열)로 방수포를 조준하는 연구로 보인다.
  - 참고문헌에는 영상 서보 기초(Chaumette), 데드존·포화가 있는 MPC, IBVS-MPC, Lin 2021 착지점 모델, McNeil의 시각 서보 소화 논문이 들어 있다.
  - 저자는 CUMT 그룹(Zhu 2019와 같은 연구실)이다.
- **우리에게**: 팀원 시뮬레이터(`PAPER_CONTROL.md`의 C단계)가 "지연을 예측에 넣는다"는 동기를 이 논문에서 가져왔다. 세부 수식은 원문을 확보해야 확인할 수 있다.

### Pozharkova, Chentsov 2025 — Real-Time Forecasting of a Fire-Extinguishing Agent Jet Trajectory from a Robotic Fire Monitor Under Disturbances (Robotics)
- **어떻게 하나**
  - 실측은 카메라 3대(소리로 동기화)로 물줄기 위 경계를 사진측량해 3D로 복원했다. 대상은 LS-P20U 방수포, 피치 30/44.5/57°, 풍속 0–3.3 m/s다.
  - 학습 데이터는 실험으로 검증한 CFD(k-ε/k-ω, VOF 등)로 735개 시나리오(시나리오당 100점)를 만들어 확보했다. 입력은 피치, 유량, 풍향, 풍속이다.
  - scikit-learn MLP로 궤적을 예측한다. 실측 대비 RMSD는 0.48 m, 최대 오차는 0.71 m 이하, 저가 CPU 추론 시간은 0.009 s였다.
- **우리에게**: "실물로 몇 점만 재서 시뮬 모델을 검증하고, 시뮬로 대량 데이터를 만들어 작은 신경망을 학습"하는 흐름이 MuJoCo+RL 계획과 같다. 우리는 CFD 대신 항력식이 있는 공 모델이면 충분하다.

### Quenzel 외 2025 — Aerial Assistance System for Automated Firefighting during Turntable Ladder Operations (SSRR 2025)
- **어떻게 하나**
  - 드론(DJI M3T)의 열화상·컬러 영상으로 불을 찾고, 두 위치를 오가며 삼각측량해 GNSS 좌표를 낸다. 키프레임 사이 편차는 0.435 m였다.
  - 사다리차 방수포 제어기(Raspberry Pi 4 + ESP32, ROS 2 Jazzy, CAN)가 GNSS 자세와 압력 센서 값으로 **탄도 궤적**을 계산해 팬·틸트를 맞춘다.
  - 각도 오차는 평균 0.45° 미만이었다(45° 탄도 기준 최악 0.46–1.2 m). 속도를 높이면 오버슈트가 커졌다. 물줄기 분할은 했지만 피드백 제어는 향후 과제로 남겼다.
  - 관련연구 절이 정리를 잘 해 두었다. 노즐 정렬 카메라는 화소 오차로 yaw 피드백, 고정 궤적 모델(McNeil), 착지점으로 모델 갱신(Lin)으로 나뉜다.
- **우리에게**: 표적 좌표를 다른 센서(우리는 depth)로 받아 탄도식으로 조준하는 "개루프 탄도" 구성의 실물 예다. 서보 각도 분해능 0.5°가 곧 거리 오차가 된다는 계산도 우리 서보(MG996R) 한계를 따질 때 참고할 수 있다.

## B. 던지기·발사 로봇과 학습

### Zeng 외 2019/2020 — TossingBot: Learning to Throw Arbitrary Objects with Residual Physics (RSS 2019, T-RO 2020)
- **어떻게 하나**
  - UR5와 RG2 그리퍼를 쓰고, 위에서 내려다보는 RGB-D로 상자 안 물체를 본다. 잡기와 던지기를 함께 학습한다(FCN).
  - 던지기 속도는 탄도식(공기저항 없음, 45° 방출)으로 구한 v̂에 신경망이 예측한 **잔차 δ**를 더한 값이다(v = v̂ + δ). 잔차가 공기역학과 잡은 위치의 영향을 흡수한다.
  - 정답 라벨은 자기지도로 만든다. 던진 뒤 실제 착지 상자를 카메라로 보고, 그 착지점에 맞았을 탄도 속도와 실제 속도의 차이를 δ 라벨로 쓴다(Huber 손실).
  - 실물에서 시간당 608회 잡기, 투척 정확도 84.7%(514 MPPH)였다. 실물에서는 Physics-only보다 Residual이 뚜렷이 좋았다. 학습 물체에는 탁구공도 있었다.
- **우리에게**: "탄도(+항력) 해 + 작은 잔차"는 우리에게 가장 직접 맞는 구조다. RL 정책이 틸트 전체를 내지 않고 **물리식 틸트에 대한 보정량만** 내게 하면 학습량이 크게 준다. 빗맞은 발도 "그 착지점이 정답이었을 입력"으로 라벨을 다시 붙여 버리지 않고 쓰는 방법을 빌릴 만하다.

### Wirkus, de Gea Fernández, Kassahun 2012 — Realizing Target-Directed Throwing With a Real Robot Using Machine Learning Techniques (ERLARS 2012) — PDF 자동 다운로드 실패
- **어떻게 하나** (DFKI 페이지의 초록만 확인했다. 공개 PDF 링크가 있지만 봇 차단 때문에 받지 못했다)
  - 인간형 팔 로봇이 임의 위치의 통에 공을 던진다. 시뮬레이션 모델 위의 비용함수를 진화 알고리즘으로 최적화해 조준과 동작을 만든다.
  - 시뮬과 실물의 오차는 실물 투척 데이터로 학습한 신경망이 보정한다. 저자들은 "단순 시뮬 모델도 학습 보정을 붙이면 좋은 성능이 나온다"고 결론짓는다.
- **우리에게**: TossingBot보다 앞선 "시뮬 + 학습 잔차"의 예다. MuJoCo 모델이 실물과 어긋나도 보정 신경망으로 메울 수 있다는 근거가 된다.

### H. Kasaei, M. Kasaei 2022 — Throwing Objects into A Moving Basket While Avoiding Obstacles (arXiv)
- **어떻게 하나**
  - 손으로 가르친 던지기 궤적 하나를 커널로 두고, RL이 고르는 것은 4개 값뿐이다(어깨 시작각, 끝각, 실행 시간, 놓는 시점).
  - 상태는 19차원이다(관절값, 장애물 위치, 바구니 위치·속도, 거리, 시간). 보상은 성공 +1, 장애물 충돌 −10, 실패 시 −거리다. 한 번 던지면 에피소드가 끝난다.
  - Gazebo(ODE)에서 SAC/DDPG(stable-baselines3, 은닉층 256×2)로 학습하고, 실물(듀얼암, Asus Xtion RGB-D)에 미세조정 없이 이식했다.
  - 학습한 물체로 장애물 없이 던졌을 때 SAC 94%, DDPG 91%, BC 86%였다. 실물은 90%(18/20), 처음 보는 물체는 85%(17/20)였다.
- **우리에게**: "한 발 = 한 에피소드, 행동은 각도 몇 개"라는 작은 RL 정식화는 우리 팬·틸트 문제와 모양이 같다. 시뮬과 실물의 동역학이 비슷하면 그대로 옮겨도 된다는 사례지만, 탁구공 항력은 시뮬에 꼭 넣어야 한다.

### Kormushev, Calinon, Caldwell 2013 — Reinforcement Learning in Robotics: Applications and Real-World Challenges (Robotics) — 활쏘기 절
- **어떻게 하나**
  - iCub 인간형 로봇이 2.2 m 앞, 지름 50 cm 과녁에 활을 쏜다. 학습하는 값은 양손 상대 위치 3개다.
  - 매 발 뒤 카메라(GMM 색 분할)로 화살 끝과 과녁 중심의 **2D 오차 벡터**를 잰다.
  - ARCHER는 이 벡터 보상을 쓴다. 가까웠던 N발의 (파라미터, 오차) 쌍으로 국소 선형회귀(유사역행렬)를 해서 다음 파라미터를 고른다.
  - 시뮬에서 스칼라 보상 RL(PoWER)은 19발, ARCHER는 5발 만에 중앙에 맞혔다. 실물 iCub은 10발 미만(원문 "less than 10 rollouts")에 중앙으로 수렴했다.
- **우리에게**: 탄착 오차를 "거리 하나"로 뭉개지 말고 (좌우, 위아래) 벡터로 쓰면, 몇 발 만에 수렴하는 회귀형 보정이 된다. 실물 탄 수가 적은 시연 상황에서 RL의 현실적 대안이다.

### Dittrich 외 2023 — AIMY: An Open-source Table Tennis Ball Launcher (ICRA 2023)
- **어떻게 하나**
  - 발포 바퀴 3개(120° 배치, BLDC)로 공을 쏘는 오픈소스 발사기다. 방위는 서보, 고각은 리니어 모터로 바꾼다. 최대 15.4 m/s, 회전 192 1/s를 낸다.
  - 정확도는 착지점 표준편차로 평가했다. 바퀴 램프업 시간이 0.5 s 미만이거나 공 압착이 약하면 산포가 커졌다. 상용 발사기 두 종보다 대체로 산포가 작았다.
  - 표적 사격은 해석 모델 대신 3761개 실측 궤적으로 "원하는 착지점 → 바퀴 속도·각도" 역모델 MLP(은닉층 3개)를 학습했다. 결과는 "대략 맞음" 수준이었다.
- **우리에게**: 우리 발사기도 바퀴형이면 공에 회전이 걸려 마그누스 힘이 생긴다. 바퀴 속도 안정화 시간과 공 압착이 탄착 산포를 좌우한다는 실측 결과도 참고할 만하다. 발사 반복성(같은 각도에서의 산포)을 먼저 재야 조준 알고리즘의 한계를 알 수 있다.

## C. 탁구공 공기역학(항력·마그누스)

### Schneider 외 2018 — Statistical Analysis of Table-Tennis Ball Trajectories (Applied Sciences)
- **어떻게 하나**
  - 운동방정식은 m·a = 중력 + 항력 −½·C_D·ρ·A·|v|·v + 마그누스 양력(C_L)이다. C_D와 C_L은 회전 속도/병진 속도 비의 함수로, 실험 데이터에 맞춘 유리함수다.
  - CUDA GPU에서 오일러 적분(dt = 1e-4 s)으로 초기조건 5억 개를 풀었다. RK4와의 차이는 최대 약 1 mm인데 속도는 4.37배 빨라서 오일러를 택했다.
  - 공 규격은 40 mm·2.7 g(현행), 38 mm·2.5 g, 44 mm·2.3 g을 비교했다.
- **우리에게**: 시뮬 공을 실제 탁구공(40 mm, 2.7 g)으로 바꿀 때 쓸 항력·마그누스 식과 적분 방식(속도 의존 힘이라 Verlet 대신 Euler/RK)이 그대로 나와 있다. GPU 일괄 적분 구조는 우리 CUDA RL 환경에 바로 옮길 수 있다.

### Conti 외(Sony AI) 2026 — Physics Models for Sim-to-Real Transfer in Professional-Level Robot Table Tennis (arXiv, RA-L 투고)
- **어떻게 하나**
  - 카메라 9대(200 Hz)로 프로·엘리트 경기 277게임의 궤적을 모았다. 관측과 모델의 RMSE가 20 mm를 넘는 구간은 버렸다.
  - 항력·마그누스 계수를 상수가 아니라 레이놀즈수 Re = 2vr/ν와 회전비 Sp = rω/v의 함수로 둔다. 궤적마다 최적 계수를 맞춘 뒤 (Re, Sp)에 대해 회귀해 표(breakpoint)로 만든다. 표의 C_D는 대략 0.37~0.55다.
  - 테이블·라켓 접촉에는 잔차(신경망) 모델을 더했다. 착지 오차 중앙값이 표준 기준선 대비 59% 줄었고, 이 모델로 RL 정책을 학습해 실제 로봇을 프로 선수와 대전시켰다.
- **우리에게**: 회전이 거의 없는 우리 속도대(3–7 m/s)에서는 C_D 상수 하나(약 0.4~0.5)로 시작해도 된다. 우리 계산으로는 r = 0.02 m, m = 2.7 g, C_D ≈ 0.45일 때 종단속도가 약 8.8 m/s다. 5 m/s에서 항력 가속도는 약 0.3 g, 7 m/s에서 약 0.6 g이라 **항력을 빼면 먼 표적에서 크게 짧게 떨어진다**. 지금 시뮬 공(0.10 kg)은 상대 항력이 이보다 훨씬 작다.

## D. 팬·틸트 터렛(이동 로봇 위)

### Qin, Xu 2023 — Design and Implementation of Automatic Assisted Aiming System for Robomaster EP Based on YOLOv5 (arXiv)
- **어떻게 하나**
  - 메카넘 바퀴 RoboMaster EP 위 짐벌이 물젤탄을 쏜다. 카메라(OV2710, 640×480, 120 fps)는 발사축과 정렬돼 있다. YOLOv5로 장갑판을 찾고, DeepSORT와 헝가리안 알고리즘으로 표적을 고른다.
  - YOLO 갱신 사이에는 칼만 필터로 위치를 예측하고, 피드포워드 PID와 FIR 필터로 짐벌을 움직인다.
  - 거리는 박스 픽셀 높이로 추정한다(D = f·H/h). 카메라와 발사구 위치가 달라 물리식 대신, **명중한 순간의 화면 y 오프셋과 IR 거리센서 값을 거리별로 모아 회귀**했다. 다항식, SVR, KNN을 비교해 KNN이 R² 0.987로 가장 좋았다.
- **우리에게**: 스펙의 baseline("화면 오차 PID + 거리별 보정표")과 거의 같은 구성이 실제로 돌아간 예다. 우리는 거리를 depth로 바로 받으므로 더 간단하다. 칼만으로 YOLO 지연을 메우는 부분도 스펙과 같다.

### Sun 외 2019 — A Visual Calibration Method of Monocular Camera and Pan-Tilt for Mobile Robots (IEEE RCAR 2019) — PDF 없음(유료)
- **어떻게 하나** (OpenAlex와 HIT 저장소의 초록만 확인했다)
  - RoboMaster 로봇의 몸체 고정 단안 카메라로 표적 3D 자세를 구하지만, 사격은 팬·틸트 좌표계에서 해야 한다.
  - 포신에 레이저 포인터를 달고 평면 거울로 반사시킨다. 카메라 보정 기법으로 카메라–팬틸트 사이 회전행렬을 빠르게 구했고, 대회에서 1위를 했다.
- **우리에게**: 스펙의 미정 사항 "카메라를 몸체에 고정하면 카메라–노즐 캘리브레이션이 필요"에 해당하는 방법이다. 발사기에 싼 레이저 포인터를 달아 카메라 영상에서 점을 찾는 방식은 거울 없이도 응용할 수 있다.

---

## 방식 갈래 정리

| 갈래 | 대표 | 필요한 것 | 장점 | 약점 |
| --- | --- | --- | --- | --- |
| 기하(직선) 조준 | McNeil 2장 첫 조준, 시뮬 `geometric` | 표적 3D 위치 | 가장 단순, 가까우면 충분 | 거리가 멀어지면 중력 낙차만큼 짧게 떨어짐 |
| 탄도(+항력) 모델 | Quenzel, Schneider, Conti, 시뮬 `ballistic` | 발사 속도, C_D, 질량·지름 | 처음 보는 거리에도 일반화, 학습 불필요 | 발사 속도 산포·회전·서보 오차는 못 잡음. 탁구공은 항력을 빼면 크게 틀림 |
| 경험식·보정표·회귀 | Kim, Wu(구간 선형), Qin(KNN), Zhu 2019 | 거리별 실측 데이터 | 실물 오차를 그대로 담음, 구현 쉬움 | 데이터를 모은 조건 밖에서는 약함, 환경이 바뀌면 다시 측정 |
| 탄착 피드백(시각 서보) | Wu, McNeil(칼만+PI+불감대), Lin(계수 온라인 갱신), Kormushev(ARCHER) | 탄착/궤적을 보는 카메라 | 체계 오차(백래시, 속도 편차)를 발마다 제거 | 첫 발은 못 고침. 공이 작고 빨라 탄착 검출이 어려움 |
| 학습 잔차 | TossingBot, Wirkus | 물리 모델 + 시행 데이터 | 물리식의 일반화 + 학습의 정밀도, 적은 데이터 | 잔차 라벨을 만들려면 착지점 측정이 필요 |
| RL | Kasaei(SAC/DDPG, sim→real), Conti(시뮬로 정책 학습) | 실물과 비슷한 시뮬(항력 포함) | 지연·이동 표적·다목표까지 한 정책에 담음 | 시뮬과 실물 차이에 민감, 실물 탄 수가 적으면 검증이 어려움 |

추천 조합: 항력을 넣은 탄도 해를 기본으로 두고, RL이나 회귀는 그 위의 틸트·팬 **잔차만** 학습하게 한다. Fallback은 Lin/McNeil식 탄착 피드백 보정(빗맞은 오차 벡터로 다음 발 보정)으로 한다.
