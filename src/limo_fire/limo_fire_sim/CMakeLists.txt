cmake_minimum_required(VERSION 3.8)
project(limo_fire_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# Webots 정답 카메라 플러그인만 webots_ros2_driver 가 필요하다. 없는 PC(실물 녹화 분석만 하는 데스크톱)에서도 녹화 도구는 빌드한다
find_package(webots_ros2_driver QUIET)
find_package(nav_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(limo_msgs REQUIRED)

# 정답 카메라: Webots Recognition(분할) → /gt/detections, /gt/mask (webots_ros2_driver 플러그인)
if(webots_ros2_driver_FOUND)
  add_library(gt_camera SHARED src/gt_camera.cpp src/ground_truth_pose.cpp)
  target_compile_features(gt_camera PUBLIC cxx_std_17)
  ament_target_dependencies(gt_camera rclcpp pluginlib sensor_msgs vision_msgs geometry_msgs webots_ros2_driver)
  pluginlib_export_plugin_description_file(webots_ros2_driver limo_fire_sim_plugins.xml)
  install(TARGETS gt_camera LIBRARY DESTINATION lib)
else()
  message(WARNING "webots_ros2_driver 없음 — 정답 카메라 플러그인(gt_camera)은 빼고 녹화 도구만 빌드")
endif()

# 라이다 오차 통계(V3): 녹화 → YAML. 같은 통계 코드(scan_stats.hpp)를 학습 시뮬 noise_check 도 쓴다
add_executable(scan_stats_bag src/scan_stats_bag.cpp)
target_compile_features(scan_stats_bag PUBLIC cxx_std_17)
target_include_directories(scan_stats_bag PRIVATE include)
ament_target_dependencies(scan_stats_bag rclcpp rosbag2_cpp sensor_msgs nav_msgs)

# 실물 녹화 → 다시 돌리기 파일(.lfrec): 학습 시뮬 replay_check 가 같은 자세에서 시뮬 라이다를 실물과 비교
add_executable(bag_export src/bag_export.cpp)
target_compile_features(bag_export PUBLIC cxx_std_17)
ament_target_dependencies(bag_export rclcpp rosbag2_cpp sensor_msgs nav_msgs tf2_msgs)

# 실물 녹화 → 구동 응답 표(CSV): 학습 시뮬 motion_check 가 시뮬 구동 모델을 실물 명령·속도에 맞춘다 (몸 이동 참값은 라이다 스캔 맞추기 scan_match.hpp)
add_executable(motion_export src/motion_export.cpp)
target_compile_features(motion_export PUBLIC cxx_std_17)
target_include_directories(motion_export PRIVATE include)
ament_target_dependencies(motion_export rclcpp rosbag2_cpp sensor_msgs nav_msgs geometry_msgs limo_msgs tf2_msgs)

# 깊이 오차 통계: 녹화 → YAML. 같은 통계 코드(depth_stats.hpp)를 학습 시뮬 depth_check 도 쓴다
add_executable(depth_stats_bag src/depth_stats_bag.cpp)
target_compile_features(depth_stats_bag PUBLIC cxx_std_17)
target_include_directories(depth_stats_bag PRIVATE include)
ament_target_dependencies(depth_stats_bag rclcpp rosbag2_cpp sensor_msgs nav_msgs)

# 라이다 세기 통계: 녹화 → YAML (intensity_stats.hpp, 빔 기하는 scan_stats.hpp)
add_executable(intensity_stats_bag src/intensity_stats_bag.cpp)
target_compile_features(intensity_stats_bag PUBLIC cxx_std_17)
target_include_directories(intensity_stats_bag PRIVATE include)
ament_target_dependencies(intensity_stats_bag rclcpp rosbag2_cpp sensor_msgs nav_msgs)

# 시뮬 초음파 → 실물 ESP32 다리 꼴 (메아리 없음 +inf · 놓침 · 고장 · 지연·시각 앞섬, sonar_sim.hpp). sim_robot.launch.xml 이 늘 띄움
add_executable(sonar_sim_node src/sonar_sim_node.cpp)
target_compile_features(sonar_sim_node PUBLIC cxx_std_17)
target_include_directories(sonar_sim_node PRIVATE include)
ament_target_dependencies(sonar_sim_node rclcpp sensor_msgs)

# 초음파 오차 통계: 실물 녹화(벽 앞 정지 구간) → YAML. sonar_sim_real.yaml 값을 정한다 (data_plan.md 초음파 절)
add_executable(sonar_stats_bag src/sonar_stats_bag.cpp)
target_compile_features(sonar_stats_bag PUBLIC cxx_std_17)
target_include_directories(sonar_stats_bag PRIVATE include)
ament_target_dependencies(sonar_stats_bag rclcpp rosbag2_cpp sensor_msgs nav_msgs tf2_msgs)

install(TARGETS scan_stats_bag bag_export motion_export depth_stats_bag intensity_stats_bag sonar_sim_node sonar_stats_bag RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY resource DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_sonar_sim test/test_sonar_sim.cpp test/test_sonar_stats.cpp)
  target_include_directories(test_sonar_sim PRIVATE include)
endif()

ament_package()
