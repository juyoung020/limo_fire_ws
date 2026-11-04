# archive

안 쓰는 것을 보관한다. 빌드에서 빠진다(`COLCON_IGNORE`).

- `robot-agent-history-*.bundle`: 저장소를 비우기 전 옛 커밋 기록 백업(저장소에는 올리지 않음). 되살리기: `git clone <bundle> 폴더`
- `explore_env/`: 탐사 RL 첫 초안 GPU 환경(원래 `train/explore/env`, 2026-10-23). 목표점 행동 48·칸 DDA 라이다·덮음 보상, `train/maps` 의 `.fmap` 맵 읽기(`src/fmap_host.cpp`)가 여기 있다. 실물 녹화로 맞춘 `train/explore/sim` 이 대체했다. 빌드는 `../sim/include` 의 `detmath.h` 를 가리키던 채라 옮긴 뒤에는 그대로 안 된다.
- `limo_base/`: WeGo/AgileX 섀시 드라이버(원래 `src/external/wego_limo_pro/limo_ros2/limo_base`, 2026-10-26 로봇 복사본). 그대로 가져와 받고 버리던 리모컨(0x241)·모터 상태(0x251~0x254)·바퀴 거리(0x311)만 더 낸 우리 판이 `src/limo_fire/limo_fire_base` (`diff -r archive/limo_base src/limo_fire/limo_fire_base` 로 다른 줄을 본다). 로봇의 `~/wego_ws` 빌드(대시보드용)에는 그대로 있다.
