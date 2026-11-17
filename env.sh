# 워크스페이스 환경: source ~/limo_fire_ws/env.sh
# conda Python 이 ROS 2 Humble(rclpy, python3.10)과 충돌하므로 PATH 에서 뺀다
conda deactivate 2>/dev/null
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd:)
# 팀 기준은 Humble(로봇 Jetson). Humble 이 없는 Jazzy 개발 PC 는 Jazzy 로 (docs/simulation.md "ROS 2 Jazzy PC 에서 돌리기")
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
elif [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
  # Jazzy 용 webots_ros2 를 따로 빌드해 둔 곳
  [ -f "$HOME/webots_ros2_ws/install/setup.bash" ] && source "$HOME/webots_ros2_ws/install/setup.bash"
fi
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# DDS(Fast DDS) 설정: 공유 메모리 칸을 영상 몇 장 크기(4 MB)로 — 기본 512 KB 는 640×480 영상 한 장보다 작아 장이 빠진다
# (limo_fire_bringup/config/fastdds.xml 머리말, docs/runtime_io.md "DDS 설정"). 보내는 쪽·받는 쪽 모두 이 셸에서 띄울 것
export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/src/limo_fire/limo_fire_bringup/config/fastdds.xml"
# 실물 LIMO: WeGo 가 빌드한 드라이버(limo_base·orbbec_camera·ydlidar·robot_localization)를 밑에 깔고,
# WeGo 대시보드 스크립트와 같은 범위(이 컴퓨터 안에서만)로 맞춘다 — ROS_LOCALHOST_ONLY=1 은 rmw 가 위 설정에 기본 전송을 더 붙여 섞이므로
# 같은 범위를 XML 로 (UDP 를 127.0.0.1 로만 묶은 판, ROS_LOCALHOST_ONLY=1 인 WeGo 노드와 서로 통함)
if [ -f "$HOME/wego_ws/install/setup.bash" ]; then
  source "$HOME/wego_ws/install/setup.bash"
  export FASTRTPS_DEFAULT_PROFILES_FILE="$WS/src/limo_fire/limo_fire_bringup/config/fastdds_localhost.xml"
  unset ROS_LOCALHOST_ONLY
fi
[ -f "$WS/install/local_setup.bash" ] && source "$WS/install/local_setup.bash"
export WEBOTS_HOME=/usr/local/webots
