# 연기 모드 선행 연구

`docs/spec.md`의 "연기 모드"(EKF + 초음파 AMCL + 연기 반사를 거른 라이다 + Nav2)와 "인원 집계"에 참고할 논문을 모았다.
PDF는 합법적으로 공개된 것(arXiv, 저자·기관 저장소, 오픈액세스 학회지)만 받았다. 공개본이 없는 논문은 링크만 적고 "PDF 없음(유료)"로 표시했다.
요약은 받은 PDF의 초록·방법·결과를 읽고 썼다. PDF가 없는 논문은 공개된 초록만 보고 썼으며, 해당 절에 "(초록 기준)"이라고 적었다. (수집일 2026-11-04)

## 목록

| 파일 | 논문(저자, 연도, 학회/저널) | 링크 | 한 줄 요지 |
| --- | --- | --- | --- |
| `2016_starr_rangefinding-fire-smoke-thesis.pdf` | Starr, *Rangefinding in Fire Smoke Environments*, 2016, Virginia Tech 박사논문 (2장 = Starr & Lattimer 2014 Fire Technology 논문의 채택본) | [VTechWorks](https://vtechworks.lib.vt.edu/handle/10919/73780) | 실제 화재 연기에서 센서 11종 측정. 라이다는 가시거리 4 m부터 틀어지고, 깊이 카메라(Kinect)는 8 m부터 실패. 레이더·열화상은 영향 없음. 초음파는 연기에는 강하지만 고온에서 10–20% 거리 오차 |
| PDF 없음(유료) | Starr & Lattimer, *Evaluation of Navigation Sensors in Fire Smoke Environments*, 2014, Fire Technology 50:1459 | [DOI 10.1007/s10694-013-0356-3](https://doi.org/10.1007/s10694-013-0356-3) | 위 박사논문 2장과 같은 내용 |
| PDF 없음(유료) | Kim, Starr & Lattimer, *Firefighting Robot Stereo Infrared Vision and Radar Sensor Fusion for Imaging through Smoke*, 2015, Fire Technology 51:823 | [DOI 10.1007/s10694-014-0413-6](https://doi.org/10.1007/s10694-014-0413-6) | 스테레오 열화상과 FMCW 레이더를 융합해 거리 오차를 1–19%에서 1–2%로 줄임 |
| `2010_peynot_marulan-datasets-ijrr.pdf`, `2009_peynot_marulan-sensor-data-integrity-report.pdf` | Peynot, Scheding & Terho, *The Marulan Data Sets*, 2010, IJRR 29(13), 기술보고서 ACFR-TR-2009-002 포함 | [DOI 10.1177/0278364910384638](https://doi.org/10.1177/0278364910384638), [ACFR 페이지](https://sdi.acfr.usyd.edu.au/) | 먼지·연기·비 속에서 레이저 4대, 94 GHz 레이더, 컬러·IR 카메라를 동기화해 기록한 공개 데이터셋. 레이저는 연기를 장애물처럼 보지만 레이더는 거의 그대로임 |
| `2018_bijelic_lidar-fog-benchmark.pdf` | Bijelic, Gruber & Ritter, *A Benchmark for Lidar Sensors in Fog: Is Detection Breaking Down?*, 2018, IEEE IV | [arXiv 1912.03251](https://arxiv.org/abs/1912.03251), [DOI 10.1109/IVS.2018.8500543](https://doi.org/10.1109/IVS.2018.8500543) | 안개 챔버에서 라이다 4종 비교. 가시거리가 40 m 미만이면 25 m 밖은 못 봄. 근거리 잡음이 생기고, 안개 속에서는 last echo가 strongest echo보다 나음 |
| `2021_bystrov_radar-smoke.pdf` | Bystrov et al., *Experimental Evaluation of 79 and 300 GHz Radar Performance in Fire Environments*, 2021, Sensors 21(2):439 | [DOI 10.3390/s21020439](https://doi.org/10.3390/s21020439), [PMC7827190](https://pmc.ncbi.nlm.nih.gov/articles/PMC7827190/) | 실제 소방 훈련장에서 79 GHz 레이더의 연기 감쇠는 0.09 dB/m. 같은 짙은 연기에서 Velodyne 라이다는 뒷벽을 못 봄 |
| `2019_stanislas_airborne-particle-lidar-classification.pdf` | Stanislas et al., *Airborne Particle Classification in LiDAR Point Clouds Using Deep Learning*, 2019, FSR | [EasyChair 프리프린트](https://easychair.org/publications/preprint/TCpW), [DOI 10.1007/978-981-15-9460-1_28](https://doi.org/10.1007/978-981-15-9460-1_28) | 먼지·안개 점을 점 단위로 분류(F1 94%). 기하 + 멀티에코 특징이 핵심. 가짜 장애물의 96%를 제거함 |
| PDF 없음(유료) | Shamsudin et al., *Fog removal using laser beam penetration, laser intensity, and geometrical features for 3D measurements in fog-filled room*, 2016, Advanced Robotics 30(11-12) | [DOI 10.1080/01691864.2016.1164620](https://doi.org/10.1080/01691864.2016.1164620) | 실내 안개방(가시거리 6 m·2 m)에서 intensity·기하 특징만으로는 오판이 생김. 빔 투과 특징을 더하고 SVM으로 분류함 |
| `2016_fritsche_radar-lidar-fusion-low-visibility.pdf` | Fritsche, Kueppers, Briese & Wagner, *Radar and LiDAR Sensorfusion in Low Visibility Environments*, 2016, ICINCO | [DOI 10.5220/0005960200300036](https://doi.org/10.5220/0005960200300036) | 같은 방향의 라이다 거리가 레이더보다 짧으면 그 빔을 레이더 값으로 바꾼다. 바뀐 빔의 비율(LRR)로 연기를 감지함 |
| PDF 없음(유료) | Fritsche, Zeise, Hemme & Wagner, *Fusion of radar, LiDAR and thermal information for hazard detection in low visibility environments*, 2017, IEEE SSRR | [DOI 10.1109/SSRR.2017.8088146](https://doi.org/10.1109/SSRR.2017.8088146) | 라이다와 레이더를 융합한 스캔에 열화상 온도를 투영해 핫스팟을 찾음. 연기 찬 방에서 원격조종으로 시연 |
| `2013_gerardocastro_laser-radar-gp-fusion.pdf` | Gerardo-Castro, Peynot & Ramos, *Laser-Radar Data Fusion with Gaussian Process Implicit Surfaces*, 2013, FSR | [QUT ePrints](https://eprints.qut.edu.au/67603/), [DOI 10.1007/978-3-319-07488-7_20](https://doi.org/10.1007/978-3-319-07488-7_20) | 레이저와 레이더로 각각 표면을 추정한 뒤 합치는 것이 원시 데이터를 바로 합치는 것보다 정확함. 일관성 검사의 바탕이 된 논문 |
| PDF 없음(유료) | Gerardo-Castro, Peynot, Ramos & Fitch, *Robust multiple-sensing-modality data fusion using Gaussian Process Implicit Surfaces*, 2014, FUSION | [UTS 저장소(비공개)](https://opus.cloud1.lib.uts.edu.au/handle/10453/121410) | 센서별 표면의 국소 일관성 검사로 레이저와 레이더가 서로 다른 것을 보는 부분을 떼어냄. 먼지·연기 실험 |
| PDF 없음(유료) | Brunner, Peynot & Vidal-Calleja, *Combining multiple sensor modalities for a localisation robust to smoke*, 2011, IROS | [DOI 10.1109/IROS.2011.6094614](https://doi.org/10.1109/IROS.2011.6094614) | IR 카메라만으로도 연기 속 위치 추정이 됨. 영상 품질 지표로 연기 낀 프레임을 버림 |
| `2019_khattak_thermal-inertial-odometry.pdf` | Khattak, Papachristos & Alexis, *Keyframe-based Direct Thermal-Inertial Odometry*, 2019, ICRA | [arXiv 1903.00798](https://arxiv.org/abs/1903.00798), [DOI 10.1109/ICRA.2019.8793927](https://doi.org/10.1109/ICRA.2019.8793927) | 14비트 원시 열화상 + IMU 최적화 오도메트리. 어둡고 먼지 낀 광산에서 50 m 비행 시 오차 약 0.54 m |
| `2020_lu_millimap-mmwave-mapping-smoke.pdf` | Lu et al., *See Through Smoke: Robust Indoor Mapping with Low-cost mmWave Radar* (milliMap), 2020, ACM MobiSys | [arXiv 1911.00398](https://arxiv.org/abs/1911.00398), [DOI 10.1145/3386901.3388945](https://doi.org/10.1145/3386901.3388945) | 단일칩 mmWave 레이더(AWR1443) + 바퀴 오도메트리로 실내 지도를 만듦. 라이다 지도로 학습한 cGAN 사용. 연기 속 지도 오차 0.2 m 미만, 라이다는 옅은 연기에서도 망가짐 |
| `2020_lu_milliego-radar-imu-odometry.pdf` | Lu et al., *milliEgo: Single-chip mmWave Radar Aided Egomotion Estimation via Deep Sensor Fusion*, 2020, ACM SenSys | [arXiv 2006.02266](https://arxiv.org/abs/2006.02266), [DOI 10.1145/3384419.3430776](https://doi.org/10.1145/3384419.3430776) | mmWave 레이더 + IMU 딥러닝 오도메트리(누적 오차 1.3%). 희소한 레이더 점으로는 ICP가 실패함 |
| `2018_heilig_ultrasonic-adaptive-mcl.pdf` | Heilig, Mamaev, Hein & Malov, *Adaptive particle filter for localization problem in service robotics*, 2018, MATEC Web of Conf. 161 | [DOI 10.1051/matecconf/201816101004](https://doi.org/10.1051/matecconf/201816101004) | HC-SR04 4개로 하는 ROS MCL. 원뿔당 광선 3개로 측정값을 예측하고 적응형 리샘플링을 씀. 복도에서 위치가 모호해지는 한계가 있음 |
| PDF 없음(유료) | Guadagnino, Della Corte & Grisetti, *Low-cost Sonar Navigation System*, 2019, IROS | [DOI 10.1109/IROS40897.2019.8967877](https://doi.org/10.1109/IROS40897.2019.8967877) | 좋은 센서를 단 로봇 한 대가 지도를 만들고, 나머지는 소나만으로 MCL을 함. 소나 외부 파라미터 보정 포함 |
| `2022_zhou_graph-slam-sparse-sensing.pdf` | Zhou, Hu, Liu & Khan, *Efficient 2D Graph SLAM for Sparse Sensing*, 2022, IROS | [arXiv 2312.02353](https://arxiv.org/abs/2312.02353), [DOI 10.1109/IROS47612.2022.9981200](https://doi.org/10.1109/IROS47612.2022.9981200) | 거리값 4개짜리 스캔 30개를 오도메트리로 묶어(multiscan) 선분을 뽑음. GMapping은 빔 30개 미만에서 실패함 |
| `2023_truong_human-detection-fire-ir-thermal.pdf` | Truong et al., *Real-time Human Detection in Fire Scenarios using Infrared and Thermal Imaging Fusion*, 2023, arXiv | [arXiv 2307.04223](https://arxiv.org/abs/2307.04223) | FLIR Lepton 3.5 + IR 카메라를 YOLOv4-tiny에서 특징 융합. Jetson Nano에서 15 FPS, 연기방 데이터 기준 mAP@0.5 95.9% |
| PDF 없음(유료) | Gelfert, *Body Part Detection in Smoky Environments with Thermal Camera Using Deep Learning*, 2022, ICCAS | [DOI 10.23919/ICCAS55662.2022.10003742](https://doi.org/10.23919/ICCAS55662.2022.10003742) | 저해상도 열화상을 단 원격조종 로봇이 짙은 연기 속 손을 90% 이상 탐지 |
| `2024_xu_slam-perturbation-benchmark.pdf` | Xu et al., *Customizable Perturbation Synthesis for Robust SLAM Benchmarking*, 2024, arXiv | [arXiv 2402.08125](https://arxiv.org/abs/2402.08125) | RGB-D SLAM용 센서 열화 합성 도구. 영상·깊이 노이즈와 누락, 궤적 흔들림, 센서 간 시간 어긋남을 강도별로 정적·동적 주입함 |

---

## 1. 연기 속 센서 측정

### Starr 2016 박사논문 (2장 = Starr & Lattimer 2014)
**어떻게 하나**
- 실험 설비: 1.22×1.83×1.22 m 소형 시험틀에 실제 화재 연기층(약 1.3 m 두께)을 만들고 레이저 소광계로 가시거리를 쟀다. 조건은 두 가지다. 짙고 차가운 연기(가시거리 1 m 미만)와 옅고 뜨거운 연기(가시거리 5 m 초과).
- 센서: Hokuyo UTM-30LX, 멀티에코 IBEO Lux, 소나 2종(42/50 kHz), Kinect 깊이, 가시광 카메라, 열화상, 야간투시, 레이더 등.
- 라이다: 가시거리 4 m에서 거리가 틀어지기 시작한다. 1 m 이하에서는 연기층 바닥까지의 거리를 돌려준다. 멀티에코 라이다도 짙은 연기에서는 같았다. 대형 시험에서는 5 m부터 감쇠됐다.
- Kinect 깊이: 가시거리 8 m에서 이미 실패했다. 뜨거운 연기에서는 불꽃의 근적외선 때문에 측정이 안 됐다.
- 소나: 연기 자체에는 영향을 받지 않는다. 다만 고온에서 음속이 바뀌어 경계 위치가 10–20% 짧게 나온다(온도 보정 가능). 레이더는 목표 3.2/4.2 m에서 전 구간 오차가 5 mm 미만이었다.
- 7장 라이다 전처리: 맑은 날의 intensity-거리 그래프에 곡선을 맞춰 임계값으로 쓴다. 곡선 아래 점(강하게 감쇠된 반사)은 "빈 공간" 정보로만 쓰고 점유로는 쓰지 않는다.

**우리에게**
- 초음파는 연기에 강하다는 근거가 된다. 다만 불 근처 고온 구간에서는 거리가 짧게 나오므로 AMCL 측정 노이즈를 거리에 비례해 크게 두는 계획이 맞다.
- Orbbec 깊이 카메라는 근적외선 방식이라 Kinect처럼 라이다보다 먼저 죽을 가능성이 크다. 연기 모드에서는 장애물 지도에서 빼야 한다.
- 연기로 의심되는 라이다 반사는 Nav2 obstacle layer에서 `marking: false`, `clearing: true` 소스로만 쓰자. 연기 반사가 장애물로 찍히지 않고 빈 공간만 지우게 하는 "free space 전용" 처리를 그대로 옮길 수 있다.

### Kim, Starr & Lattimer 2015 (초록 기준, PDF 없음)
**어떻게 하나**
- 스테레오 열화상(LWIR)으로 3D 장면을 얻고, FMCW 레이더로 물체 거리를 정확히 잰다.
- 레이더 검출을 3D 영상의 물체와 짝지어 원거리 오차를 고치고, 스테레오 오매칭으로 생긴 유령 물체를 지운다.
- 대형 화재 시험에서 거리 오차가 1–19%에서 1–2%로 줄었다.

**우리에게**
- 고가 장비 조합이라 그대로 따라 할 수는 없다. 다만 "정밀하지만 연기에 약한 센서"와 "거칠지만 연기에 강한 센서"를 짝지어 서로 검증한다는 구조는 우리 라이다와 초음파 조합에 그대로 들어맞는다.

### Peynot et al. 2010 Marulan 데이터셋 (IJRR + 기술보고서)
**어떻게 하나**
- 장비: 무인 지상차량에 SICK LMS291/221 레이저 4대, 94 GHz FMCW 레이더, 컬러 카메라, IR 카메라(Raytheon Thermal-eye)를 달았다.
- 조건: 압축공기로 먼지를 일으키고, 연막탄(약 1분)과 스프링클러 비를 쓴 정지·주행 데이터다. 원시 데이터는 약 400 GB다.
- 보고서 5.1절: 레이저는 먼지·연기 구름을 실제 장애물처럼 본다. 로봇과 실제 물체 사이에 무작위 점이 생긴다. SICK는 첫 반사만 주기 때문에 더 심하다.
- 레이더는 노이즈가 조금 늘고 반사 세기가 약해지는 것 말고는 거의 영향이 없다. IR 카메라도 가시광보다 덜 영향을 받는다.

**우리에게**
- 2D 라이다에서 연기는 "벽보다 앞쪽 근거리의 흩어진 점"으로 나타난다. 이 관찰이 `laser_filters`로 근거리 반사를 거르는 계획의 근거가 된다.
- 데이터셋이 공개돼 있다. 주입 노드의 노이즈 모양을 실제 연기 스캔과 비교하는 데 쓸 수 있다.

### Bijelic et al. 2018 안개 속 라이다 벤치마크
**어떻게 하나**
- CEREMA 안개 챔버에서 투과율계로 가시거리 V를 맞추고 Velodyne HDL64-S2/S3D, Ibeo Lux/Lux HD 4종을 시험했다.
- 반사율 5/50/90% 표적까지의 최대 탐지거리를 쟀다. V < 40 m이면 25 m보다 훨씬 짧아진다.
- 안개가 짙을수록 점이 줄고, 아주 가까운 거리에 잡음(clutter)이 생긴다. last echo가 strongest echo보다 실제 물체와 더 잘 겹친다.
- 레이저 출력과 노이즈 임계값을 수동으로 조정하면 몇 m 나아지지만, 짙은 안개에서 믿을 만한 수준은 아니다(905 nm 눈 안전 출력 한계).

**우리에게**
- 차량용 장거리 결과지만, 근거리 잡음이 생긴다는 점과 멀티에코가 도움 된다는 점은 실내 연기에도 통한다.
- 우리 2D 라이다는 단일 에코라 에코 비교는 못 쓴다. 근거리 거리 필터와 intensity 필터가 현실적이다.

### Bystrov et al. 2021 화재 속 79/300 GHz 레이더 (`2021_bystrov_radar-smoke.pdf`)
**어떻게 하나**
- 영국 소방 훈련장의 실제 화재(강한 불꽃, 약 400 °C 짙은 연기, 수증기와 입자 혼합)에서 실험했다. 회전대에 레이더 2대를 달았고 Velodyne 32채널과 스테레오 카메라를 비교 대상으로 썼다.
- 측정 감쇠는 79 GHz 기준 불꽃 0.02 dB/m, 짙은 연기 0.09 dB/m, 수증기 0.45 dB/m였다. 모든 경우 8 m 떨어진 뒷벽이 보였다.
- 짙은 연기에서 라이다는 뒷벽을 보지 못했다.

**우리에게**
- 레이더가 연기를 통과한다는 정량 근거다. 다만 LD2450(24 GHz)은 정지 물체를 걸러내므로 위치 보정에는 못 쓰고, 사람 탐지 전용이라는 spec 판단이 맞다.
- 논문이 설명하는 Rayleigh 산란 논리로 보면, 주파수가 낮은 24 GHz는 연기 영향이 더 작을 것으로 추정된다. 이는 직접 측정한 값이 아니다.

## 2. 라이다에서 연기·먼지 반사를 찾아 거르기

### Stanislas et al. 2019 공중 입자 분류
**어떻게 하나**
- 데이터: Husky에 Velodyne HDL-32E(듀얼 에코)를 달고 실외 19개 장면을 찍었다. 먼지와 안개가 있을 때와 없을 때를 정지 상태로 찍어 배경 차분으로 라벨을 만들었다(16,247 스캔).
- 방법은 두 가지다. 복셀 분류와 2D 라이다 이미지 위 U-Net 점 분류다. 특징은 기하, intensity, 멀티에코를 조합했다.
- 점 분류가 기하 + 멀티에코로 F1 94%를 냈다. 입자는 intensity가 낮고, 첫 에코는 입자에서, 둘째 에코는 뒤 물체에서 돌아오는 경향이 있다.
- 입자 때문에 ICP 정합 오차가 2.5 cm에서 17 cm로 커지는데, 거르면 정답 수준으로 돌아온다. 가짜 장애물 셀은 96%를 막았고 진짜 장애물은 0.4%만 잘못 지웠다.

**우리에게**
- 연기 점은 장애물 지도만이 아니라 위치 정합(AMCL)도 망친다. 그래서 필터는 AMCL 입력 앞에도 걸어야 한다.
- 딥러닝과 멀티에코 3D 라이다가 필요하므로 우리 2D 라이다와 Jetson에는 과하다. 쓸 만한 것은 "낮은 intensity + 근거리" 규칙 정도다.

### Shamsudin et al. 2016 안개 제거 (초록 기준, PDF 없음)
**어떻게 하나**
- 재난 대응 로봇용으로 안개 찬 실내(가시거리 6 m와 2 m)에서 3D 라이다 점을 '안개'와 '물체'로 나눴다.
- intensity와 기하 특징만으로는 오판이 생겨 빔 투과 특징을 더했고, SVM과 KNN으로 분류했다. SVM의 F-measure가 90% 이상이었다.
- 거리 보정으로 오차를 최대 4.6% 줄였다. 전체 recall은 90.1%, F-measure는 79.4%였다.

**우리에게**
- intensity와 거리만 쓰는 필터는 짙은 연기에서 오판한다는 경고다. 필터가 점을 너무 많이 지우면(남은 유효 빔 비율이 낮으면) 라이다를 AMCL에서 아예 빼는 규칙을 BT에 넣자.

## 3. 저시정 환경의 다중 센서 융합과 센서 대체

### Fritsche et al. 2016 레이더-라이다 융합 (ICINCO)
**어떻게 하나**
- 장비: 80 GHz 회전 레이더(MPR, 거리 분해능 3.75 cm, 0.2–15 m, 최대 2.5 Hz)와 Velodyne VLP-16을 Pioneer에 달고 ROS gmapping을 돌렸다. 마당에서 안개 기계를 10초 켜서 시험했다.
- 레이더 이상치를 LOF 필터로 줄인 뒤 같은 각도의 두 거리를 비교한다. 차이가 β 이내면 라이다 값을 쓰고, 라이다가 더 짧으면(앞에 에어로졸이 있으면) 레이더 값으로 바꾼다. 레이더 범위 밖은 라이다를 쓴다.
- 바뀐 빔의 비율 LRR(LiDAR-Radar-Ratio)이 안개가 있을 때 올라가서, 느린 연기감지기보다 빠른 연기 지표가 된다.
- 한계: 레이더에 안 보이는 물체(생울타리)에 다가가도 LRR이 올라가서 연기와 구분되지 않는다.

**우리에게**
- 레이더 자리에 초음파를 넣으면 거의 그대로 쓸 수 있다. 초음파 3개 방향의 라이다 빔과 초음파 거리를 비교해 "라이다 ≪ 초음파"인 빔의 비율로 BT의 연기 감지 트리거를 만들자.
- 다만 초음파도 비스듬한 면이나 흡음재를 놓칠 수 있다. 이 한계를 생각해 몇 프레임 연속 조건을 둬야 한다.

### Fritsche et al. 2017 레이더·라이다·열화상 위험 감지 (초록 기준, PDF 없음)
**어떻게 하나**
- 위 방식으로 라이다와 레이더를 먼저 융합한 뒤, 열화상 온도를 거리 스캔에 투영해 "온도가 붙은 스캔"을 만들고 핫스팟을 찾는다.
- 연기 찬 방에서 로봇을 원격조종하며 시연했다.

**우리에게**
- MLX90640 온도를 라이다 각도에 투영하면, 화재 위치 보고와 연기 속 뜨거운 구역 회피에 쓸 수 있다.

### Gerardo-Castro et al. 2013 레이저-레이더 GPIS 융합 (+ 2014 후속, 초록 기준)
**어떻게 하나**
- 2013년 논문: SICK LMS-291과 94 GHz FMCW 레이더 데이터로 Gaussian Process Implicit Surface를 만들어 물체 표면을 복원했다. 실외 주행 데이터를 썼다.
- RMSE는 레이저 0.069 m, 레이더 0.125 m였다. 센서별 표면을 먼저 추정한 뒤 합치는 방식이 원시 점을 바로 합치는 방식보다 나았다. 특히 두 센서가 다르게 반응하는 곳(차 유리)에서 차이가 났다.
- 일관성 검사는 이 논문에서 후속 과제로 남겼다. 2014년 FUSION 논문(초록 기준)은 국소 일관성 검사로 두 센서가 같은 표적을 본 부분만 융합하고, 먼지·연기 구름 속 실외 로봇으로 검증했다.

**우리에게**
- "합치기 전에 두 센서가 같은 것을 보는지 검사하라"는 원칙이 핵심이다. GP는 우리에게 과하므로, 라이다와 초음파 거리 비교 같은 단순한 일관성 검사로 충분하다.

### Brunner, Peynot & Vidal-Calleja 2011 (초록 기준, PDF 없음)
**어떻게 하나**
- 가시광 카메라와 IR 카메라로 visual SLAM 위치 추정을 했다. IR만으로도 연기 속에서 꽤 정확했다.
- 두 영상을 합치면 더 좋았다. 영상 품질 지표로 연기 낀 가시광 프레임을 감지해 버리면 성능 저하를 미리 막을 수 있었다. dGPS/INS 기준과 비교했다.

**우리에게**
- "품질 지표로 나쁜 데이터를 버린다"는 방식은 라이다에도 적용된다(위 LRR, 유효 빔 비율).
- 다만 MLX90640(32×24)은 visual SLAM에 쓰기에는 해상도가 너무 낮다.

### Khattak et al. 2019 열화상-관성 오도메트리
**어떻게 하나**
- 장비: DJI Matrice 100에 FLIR Tau 2(640×512, 30 fps)와 IMU를 달았다. 14비트 원시 열화상으로 직접 정렬하고(8비트 재스케일·AGC 없이), 키프레임 슬라이딩 윈도우에서 재투영 오차와 IMU 오차를 함께 최소화했다.
- VICON 기준 위치 RMSE는 x 0.27 m, y 0.09 m, z 0.05 m였다. 같은 데이터에서 OKVIS, ROVIO, DSO는 발산하거나 실패했다.
- 어둡고 먼지 낀 지하 광산에서 50 m 이동 후 오차는 0.54–0.60 m였다.

**우리에게**
- 고해상도 열화상이 있어야 하는 방법이라 MLX90640으로는 불가능하다. 지상 로봇에는 바퀴 오도메트리 + IMU EKF가 더 싸고 안정적이다. 지금 계획을 유지하는 근거가 된다.

### Lu et al. 2020 milliMap (MobiSys)
**어떻게 하나**
- 장비: TurtleBot 2에 TI AWR1443 단일칩 레이더(77–81 GHz, 거리 분해능 약 4 cm, 방위 시야 120°), VLP-16(학습 정답용), 바퀴 오도메트리를 달았다.
- 희소하고 다중경로 잡음이 많은 레이더 점을 6×6 m 패치 지도로 쌓고, 라이다 지도로 지도학습한 조건부 GAN과 직선 벽 사전정보로 조밀하게 복원했다. 로봇 자세는 바퀴·관성 오도메트리로 얻었다.
- 연기 기계로 12×1.5 m 복도를 옅게·중간·짙게 채웠다. 라이다는 옅은 연기에서도 없는 물체를 만들거나 있는 물체를 놓쳤다. 깊이·RGB 카메라도 실패했다. milliMap은 세 경우 모두 복도를 복원했다(지도 오차 0.2 m 미만, 의미 분류 약 90%).

**우리에게**
- 연기 속에서는 "오도메트리로 자세를 잡고 연기에 강한 거리 센서로 보정"하는 구조가 통한다는 근거다. 우리 EKF + 초음파 구조와 같다.
- LD2450은 점구름이 아니라 표적 3개만 내보내므로 이 방법을 쓸 수 없다. 레이더로 지도를 다시 만들려면 IWR/AWR 계열 점구름 레이더가 필요하다(향후 확장).

### Lu et al. 2020 milliEgo (SenSys)
**어떻게 하나**
- AWR1843 레이더 점구름과 IMU(Xsens MTi-1)를 각각 CNN/RNN으로 처리하고, 2단계 attention으로 융합해 6자유도 자세를 회귀한다.
- 정답은 HDL-32E + gmapping이다. 17개 층 데이터를 썼다.
- 희소한 레이더 점(프레임당 약 100점, 최대 75%가 이상치)으로는 ICP와 TEASER가 실패했다. milliEgo는 3D ATE 0.81 m(약 1.3%)를 냈다.
- 유리 복도, 울퉁불퉁한 바닥, 눈부심, 어두운 방에서 시험했다. 연기 속 실험은 없다.

**우리에게**
- 저가 레이더의 희소한 점으로는 고전적 스캔 정합이 안 된다. 같은 이유로 초음파 3점 스캔도 정합 대상이 아니라 "기존 지도와 비교하는 측정"으로만 써야 한다(AMCL이 그런 방식이다).

## 4. 저가 초음파·희소 거리 센서 위치 추정

### Heilig et al. 2018 초음파 적응형 MCL
**어떻게 하나**
- 장비: 장난감 차를 개조한 로봇에 HC-SR04 4개, 엔코더, MPU-9150, Raspberry Pi 3를 달았다. 위치 추정 노드는 PC 또는 Jetson TX2에서 ROS로 돌렸다.
- 입자마다 지도 비트맵 위에서 소나 원뿔의 양끝과 중심선, 광선 3개를 쏴서 가장 가까운 거리를 예측값으로 삼는다(실제 소나가 가장 가까운 물체만 보는 것을 흉내 냄). 가우시안 우도를 쓰고, 조상 입자의 우도를 반영하며, 체계적 리샘플링을 한다.
- 최고 입자의 우도가 낮으면 입자 수와 리샘플링 노이즈를 늘린다.
- 평가는 대부분 Gazebo 시뮬레이션 10회였다. 수렴 유지 시 σ 0.14 m, 1 m 납치 후 4초 안에 회복했다. 긴 복도 전역 위치 추정은 10회 중 3회만 정확했다(5회는 비슷한 여러 곳에 수렴).

**우리에게**
- "15° 원뿔 폭에 점 여러 개 + 큰 노이즈" 설계와 거의 같은 방식이 ROS에서 동작한 사례다.
- 복도에서는 초음파만으로 전역 위치를 못 찾는다. AMCL은 EKF 위치 근처에서 국소 보정만 하게 하자(전역 초기화 금지, 입자 퍼짐 제한).

### Guadagnino, Della Corte & Grisetti 2019 (초록 기준, PDF 없음)
**어떻게 하나**
- 고급 센서를 단 로봇 한 대가 보정과 지도 작성을 맡고, 나머지 로봇은 저가 소나만으로 위치 추정과 주행을 한다.
- 소나 외부 파라미터(장착 위치·각도) 보정 절차를 새로 제안했고, 고전적 MCL을 소나 모델에 맞게 효율적으로 고쳤다. 코드를 공개했다.

**우리에게**
- 평소 라이다로 만든 지도 위에서 연기 때 초음파로 위치를 잡는다는 우리 구조와 같다.
- 초음파 3개의 장착 각도와 위치를 지도 기준으로 따로 보정하는 단계가 필요하다는 점을 일정에 넣자.

### Zhou et al. 2022 희소 센싱 그래프 SLAM
**어떻게 하나**
- 저전력 나노 드론(Crazyflie, 거리값 4개, 10 Hz, 최대 4 m)용이다. 연속된 4점 스캔 30개(120점)를 오도메트리로 한 좌표계에 모아 "multiscan"을 만들고, 선분을 뽑아 랜드마크 그래프로 오도메트리를 보정한다.
- 이후 상관 스캔-지도 매칭(3×3 커널)으로 루프를 닫는다.
- Intel Lab 데이터에서 빔 수를 4–60개로 줄여 보니 GMapping은 30개 미만에서 실패했고, 이 방법은 4개로도 쓸 만한 지도를 만들었다.
- multiscan이 너무 작으면 선분이 부족하고, 너무 크면 오도메트리 오차가 쌓인다. 중간 크기가 가장 좋다.

**우리에게**
- 3점 스캔을 그대로 AMCL에 넣으면 정보가 너무 적다. 최근 1–2초의 초음파 측정을 EKF 자세로 묶은 누적 스캔을 AMCL에 넣는 쪽을 검토하자. 누적 길이는 바퀴 오도메트리 오차가 작게 유지되는 범위로 잡는다.

## 5. 연기 속 사람 탐지 (선택)

### Truong et al. 2023 IR + 열화상 융합 사람 탐지
**어떻게 하나**
- 장비: Jetson Nano, IMX219 IR 카메라(적외선 조명), FLIR Lepton 3.5(PureThermal).
- 가열한 체스보드로 두 카메라를 보정하고 호모그래피로 정렬한 뒤, YOLOv4-tiny 넥에서 두 영상 특징을 합친다.
- 닫힌 방에 흰 연기를 풀고 사람 자세와 거리를 바꿔 3,000장을 모았다. mAP@0.5는 95.9%로 단일 카메라 YOLOv4-tiny나 YOLOv5s보다 3–38% 높았다. 속도는 15 FPS였다.

**우리에게**
- Jetson급에서 실시간으로 돌아간다는 근거다. 다만 MLX90640(32×24)은 Lepton(160×120)보다 해상도가 훨씬 낮아 DNN 탐지는 어렵다.
- LD2450 표적이 있는 방향의 열화상 온도로 "사람 같은 열원"인지 확인하는 보조 용도가 현실적이다.

### Gelfert 2022 저해상도 열화상 신체 부위 탐지 (초록 기준, PDF 없음)
**어떻게 하나**
- 원격조종 로봇에 저해상도 열화상을 달고 딥러닝으로 손을 탐지했다. 짙은 연기 실내에서 실시간 탐지율이 90% 이상이었다.
- 같은 저자의 2023 RiTA 논문(광학 + 저해상도 열화상)은 짙은 연기에서 평균 75% 이상이었고, 2022 ICMERR 논문은 8×8 열센서로 3D 열지도를 만들었다.

**우리에게**
- 아주 낮은 해상도의 열센서로도 연기 속 사람 단서를 얻을 수 있다는 사례다. MLX90640 활용 근거로 쓸 수 있다.

## 6. 센서 열화 주입으로 강건성 시험 (선택)

### Xu et al. 2024 SLAM 교란 합성 벤치마크
**어떻게 하나**
- 시뮬레이션(Replica 기반)의 깨끗한 RGB-D 데이터를 대상으로 한다. 영상 오염 16종(안개, 노이즈, 블러 등), 깊이 오염 4종(가우시안 노이즈, 무작위 누락, 경계 침식, 거리 잘림), 빠른 움직임, 궤적 흔들림, 센서 간 시간·공간 어긋남을 저·중·고 강도로 합성했다.
- 강도는 정적(수열 전체 고정)과 동적(프레임마다 변함) 두 방식이다.
- 결과: 환경 효과(안개·눈)가 가장 큰 오차를 냈다. 깊이에서는 누락보다 가우시안 노이즈가 더 해로웠다. ORB-SLAM3는 깊이 누락에서 추적을 자주 잃었다.

**우리에게**
- 우리 "센서 열화 주입 노드"의 항목표로 쓸 수 있다. 끊김과 노이즈를 둘 다 넣고, 강도를 시간에 따라 바꾸고, 토픽 간 지연도 넣자.
- 연기는 가우시안 노이즈보다 "근거리 가짜 반사 + 먼 거리 소실"(Starr, Marulan, Bijelic)에 가깝다. 이 모양을 라이다 주입의 기본 모드로 두는 것이 좋다.

---

## 방식 갈래 정리

**연기에 살아남는 센서**
- 레이더(24–300 GHz)와 장파장 열화상(LWIR, 7–15 µm)은 짙은 연기에서도 거의 영향이 없다(Starr: 레이더 오차 5 mm 미만, Bystrov: 79 GHz 0.09 dB/m).
- 초음파도 연기 입자에는 강하다. 다만 고온 기체에서 음속이 바뀌어 거리가 10–20% 짧게 나온다(Starr).
- 905 nm 라이다는 가시거리 4–5 m부터 틀어지고 1 m 이하에서는 연기층을 벽으로 본다. 근적외선 패턴 방식 깊이 카메라는 라이다보다 먼저(8 m) 실패한다. 멀티에코 라이다도 짙은 연기에서는 별 도움이 안 된다.

**세 갈래 방식**
1. **필터링(라이다를 계속 쓰기)**: intensity·거리 임계값(Starr), 기하 + intensity + 빔 투과 특징으로 SVM 분류(Shamsudin), 멀티에코 딥러닝(Stanislas). 옅은 연기에서는 효과가 있지만 짙은 연기에서는 남는 정보가 없다. 우리 `laser_filters` 근거리 필터는 이 갈래이고, 옅은 연기 구간용이다.
2. **융합(서로 검증)**: 같은 방향에서 라이다가 더 짧으면 다른 센서 값으로 바꾸고 그 비율을 연기 지표로 쓴다(Fritsche의 LRR). 합치기 전에 일관성을 검사한다(Gerardo-Castro). 정밀 센서와 강건 센서를 짝짓는다(Kim). 우리에게는 라이다와 초음파의 빔 비교가 이 갈래이고, BT의 연기 감지 트리거로 쓸 수 있다.
3. **센서 대체(연기에 약한 센서를 빼기)**: 오도메트리 + 레이더 지도(milliMap), 레이더 + IMU(milliEgo), 열화상 + IMU(Khattak), IR 카메라(Brunner), 소나 MCL(Heilig, Guadagnino). 우리 짙은 연기 모드(EKF + 초음파 AMCL)가 이 갈래다. 초음파는 정보가 적어서 전역 위치 추정은 안 되고(Heilig 복도 3/10), 국소 보정과 누적 스캔(Zhou)으로 써야 한다.

**시험 방법**
- 실제 연기·화재를 쓴 연구: Starr(실화재, 소광계로 가시거리 측정), Bystrov(소방 훈련장), Marulan(연막탄, 먼지 송풍), milliMap·Fritsche(연기·안개 기계), Shamsudin·Bijelic(안개방·안개 챔버).
- 데이터에 열화를 주입한 연구: Xu(합성 교란), Zhou(빔 수를 줄여 희소 센싱 흉내), Stanislas(정지 장면에 자세 오차를 넣어 ICP 시험).
- 실제 연기 실험은 가시거리를 함께 재야 비교할 수 있다(투과율계, 소광계).
- 우리 시연처럼 토픽에 노이즈와 끊김을 넣는 방식은 Xu의 접근과 같다. 다만 라이다 주입은 실측 연기 모양(근거리 가짜 점, 먼 거리 소실, 가시거리에 따른 단계)을 흉내 내도록 만들어야 결과가 실제 연기에 가깝다.
