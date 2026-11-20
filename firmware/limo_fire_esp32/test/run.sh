#!/bin/sh
# 펌웨어의 호스트 시험(보드 없이): 프레임 조립·체크섬·레이더 부호, 발사대 명령(막지 않고 움직임·응답 순서)
set -e
cd "$(dirname "$0")/.."
g++ -std=c++17 -Wall -Wextra -Iinclude test/test_sensor_frames.cpp -o /tmp/test_sensor_frames && /tmp/test_sensor_frames
g++ -std=c++17 -Wall -Wextra -Iinclude test/test_launcher.cpp -o /tmp/test_launcher && /tmp/test_launcher
# 발사 다리 시험용(src/main.cpp 를 PC 에서 가상 시리얼에) — /tmp/launcher_host_sim 을 limo_fire_esp32_bridge·limo_fire_hardware 의 colcon test 가 씀(LAUNCHER_HOST_SIM 로 바꿀 수 있음)
g++ -std=c++17 -O1 -Wall -Wextra -Itest/host_sim -Iinclude test/host_sim/host_main.cpp -o /tmp/launcher_host_sim -lutil && echo "호스트 시뮬 빌드: /tmp/launcher_host_sim"
