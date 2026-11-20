#!/usr/bin/env bash
set -euo pipefail

# 발사대 터미널(limo_fire_hardware launcher_terminal)을 새 GUI 창으로 연다.
# USB 포트는 ESP32 다리(esp32_bridge, robot.launch.xml esp32:=true)가 혼자 연다 — 터미널은 ROS 로만 말한다.
# LIMO_FIRE_WS: 작업공간(기본 /home/wego/limo_fire_ws), LIMO_FIRE_AIM_OVERLAY: 그 위에 얹을 시험 오버레이(있으면)
if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo 'gnome-terminal 이 없습니다. 지금 터미널에서: ros2 run limo_fire_hardware launcher_terminal' >&2
  exit 1
fi
gnome-terminal -- bash -lc 'source "${LIMO_FIRE_WS:-/home/wego/limo_fire_ws}/env.sh"; source "${LIMO_FIRE_WS:-/home/wego/limo_fire_ws}/install/local_setup.bash"; if [[ -n "${LIMO_FIRE_AIM_OVERLAY:-}" ]]; then source "$LIMO_FIRE_AIM_OVERLAY/install/local_setup.bash"; fi; ros2 run limo_fire_hardware launcher_terminal --log "${HOME}/launcher-commands.jsonl"; exec bash'
