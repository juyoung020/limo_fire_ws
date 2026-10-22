# LIMO Pro 조별 프로젝트 (로봇프로그래밍 0533)

AgileX LIMO Pro 기반 화재 대응 로봇: 열화상으로 불 위치를 찾고, 팬/틸트 서보로 노즐을 조준해 물을 분사, 24GHz 레이더로 연기 속 사람 감지.

## 폴더 (`~/limo_fire_ws` = ROS 2 워크스페이스)
```
limo_fire_ws/
├── src/                 ROS 2 패키지 (colcon build는 여기서)
│   └── wego_limo_pro/   WeGo LIMO Pro 드라이버·URDF·내비 (github.com/WeGo-Robotics/limo_pro)
├── sim/                 Isaac Sim 4.5 확장·USD (WeGo LimoIsaacSIM)      [COLCON_IGNORE]
├── firmware/            ESP32 코드
├── tools/               fetch_third_party.sh (외부 저장소 받기)
├── docs/                아이디어·스펙·부품 문서
├── refs/                참고 자료: papers/ 논문, code/ 참고 코드, datasheets/ 부품 데이터시트   [COLCON_IGNORE]
└── archive/             안 쓰는 것 (교수님 Webots 시뮬 등)              [COLCON_IGNORE]
```
처음 받은 뒤: `./tools/fetch_third_party.sh` (WeGo 패키지·Isaac Sim 확장 클론, USD 받는 법 안내)

빌드: `cd ~/limo_fire_ws && colcon build` (conda는 PATH에서 빼고)

`src/wego_limo_pro`, `sim/wego_LimoIsaacSIM`, USD 파일, 빌드 결과물은 저장소에 올리지 않는다(.gitignore).

## 시뮬레이션
Isaac Sim 4.5 + `sim/wego_LimoIsaacSIM` (설치 전). 예전 Webots 시뮬은 `archive/limo_ws_webots_prof`.
