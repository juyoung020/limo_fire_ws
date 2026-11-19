#!/usr/bin/env bash
set -euo pipefail

# Open a separate control window. The running pitch_serial_bridge owns the USB port.
if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo 'gnome-terminal is unavailable. Run: ros2 run limo_fire_hardware launcher_control_terminal' >&2
  exit 1
fi
gnome-terminal -- bash -lc 'source /opt/ros/humble/setup.bash; source "${LIMO_FIRE_WS:-/home/wego/limo_fire_ws}/install/local_setup.bash"; if [[ -n "${LIMO_FIRE_AIM_OVERLAY:-}" ]]; then source "$LIMO_FIRE_AIM_OVERLAY/install/local_setup.bash"; fi; ros2 run limo_fire_hardware launcher_control_terminal; exec bash'
