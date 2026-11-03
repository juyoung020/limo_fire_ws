# archive

안 쓰는 것을 보관한다. 빌드에서 빠진다(`COLCON_IGNORE`).

- `robot-agent-history-*.bundle`: 저장소를 비우기 전 옛 커밋 기록 백업(저장소에는 올리지 않음). 되살리기: `git clone <bundle> 폴더`
- `limo_base/`: WeGo/AgileX 바닥판 드라이버(원래 `src/external/wego_limo_pro/limo_ros2/limo_base`, 2026-10-26 로봇 복사본). 그대로 가져와 받고 버리던 리모컨(0x241)·모터 상태(0x251~0x254)·바퀴 거리(0x311)만 더 낸 우리 판이 `src/limo_fire/limo_fire_base` (`diff -r archive/limo_base src/limo_fire/limo_fire_base` 로 다른 줄을 본다). 로봇의 `~/wego_ws` 빌드(대시보드용)에는 그대로 있다.
