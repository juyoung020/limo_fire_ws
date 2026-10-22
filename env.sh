# 워크스페이스 환경: source ~/limo_fire_ws/env.sh
# conda Python 이 ROS 2 Humble(rclpy, python3.10)과 충돌하므로 PATH 에서 뺀다
conda deactivate 2>/dev/null
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd:)
source /opt/ros/humble/setup.bash
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$WS/install/local_setup.bash" ] && source "$WS/install/local_setup.bash"
export WEBOTS_HOME=/usr/local/webots
