# 워크스페이스 환경: source ~/limo_fire_ws/env.sh
# conda Python 이 ROS 2 Humble(rclpy, python3.10)과 충돌하므로 PATH 에서 뺀다
conda deactivate 2>/dev/null
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd:)
source /opt/ros/humble/setup.bash
# 실물 LIMO: WeGo 가 빌드한 드라이버(limo_base·orbbec_camera·ydlidar·robot_localization)를 밑에 깔고,
# WeGo 대시보드 스크립트와 같은 DDS 설정(이 컴퓨터 안에서만)으로 맞춘다
if [ -f "$HOME/wego_ws/install/setup.bash" ]; then
  source "$HOME/wego_ws/install/setup.bash"
  export ROS_LOCALHOST_ONLY=1
fi
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$WS/install/local_setup.bash" ] && source "$WS/install/local_setup.bash"
export WEBOTS_HOME=/usr/local/webots
