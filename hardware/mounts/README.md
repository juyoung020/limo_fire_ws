# 센서·전장 고정대 (3D 출력)

LIMO 에 붙는 곳은 LIMO 에 원래 있는 M3 나사 구멍에, 출력물끼리는 출력 M5 볼트로 잠근다. 나사도 전부 출력물. 원본 모델·검사 스크립트는 3d-modeling 저장소 `LIMO센서/`.

![조립](limo_mounts.png)

| 파일 | 부품 | 개수 | 붙는 곳 / 조이는 나사 |
|---|---|---|---|
| `print_sonar.stl` | 초음파 받침 (HC-SR04 ×2, 좌우 ±20°·위 5°) | 1 | 차체 밑면 앞 구멍 4, `print_m3x8` ×4 |
| `print_sonar_tab.stl` | 초음파 바깥 누름쇠(보드 바깥 끝을 누름) | 2 | 기둥에 `print_sonar_bolt` (M5×10) ×2 |
| `print_sonar_ctab.stl` | 초음파 가운데 누름쇠(두 보드 안쪽 끝) | 1 | 위에서 `print_sonar_cbolt` (M5×10, 일자 홈) ×1 |
| `print_arm.stl` | 열화상(카메라 위)·레이더(카메라 오른쪽) 팔 — 옆으로 눕혀 출력 | 1 | 지붕 앞 구멍 2, `print_m3x5` ×2 |
| `print_cap_thermal.stl`, `print_cap_radar.stl` | 열화상·레이더 보드 뚜껑 | 각 1 | `print_arm_bolt` (M5×12) ×2 |
| `print_pod_right.stl` | 오른쪽 전장 판: 배선판(뒤), ESP32-S3(앞, USB-C 앞으로) | 1 | 오른쪽 옆판 구멍 4, `print_m3x5` ×4 |
| `print_pod_left.stl` | 왼쪽 전장 판: XL4015(뒤), LM2596(앞) | 1 | 왼쪽 옆판 구멍 4, `print_m3x5` ×4 |
| `print_pod_right_lid.stl`, `print_pod_left_lid.stl` | 전장 덮개 상자(안쪽 기둥이 보드를 눌러 고정) | 각 1 | `print_pod_bolt` (M5) 3 + 4 |
| `print_m3x8.stl`, `print_m3x5.stl` | LIMO 구멍용 출력 M3(표준 나사산, 눕혀 출력) | 4+2, 10+2 | 손으로 살짝 조임 |

- 검사: LIMO 메시 안에 들어간 곳 0, 발사대(5~30°, 공 통 조임 고리 포함)·센서 실물 부피와 겹침 0, 공이 날아가는 길과 20.8 mm 이상, 라이다·카메라 시야 밖, 출력 볼트는 암나사와 맞물림.
- 초음파 가운데 누름쇠: 보드를 넣고 누름쇠를 기둥 위 3 mm 뒤에 내려놓은 뒤 앞으로 밀어 넣고 위에서 볼트.
- 카메라 각도는 팔을 달기 전에 맞춘다(팔이 카메라 위 1.9 mm 로 지나감).
