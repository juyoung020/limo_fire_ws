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
find_package(webots_ros2_driver REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(limo_msgs REQUIRED)

# 정답 카메라: Webots Recognition(분할) → /gt/detections, /gt/mask (webots_ros2_driver 플러그인)
add_library(gt_camera SHARED src/gt_camera.cpp)
target_compile_features(gt_camera PUBLIC cxx_std_17)
ament_target_dependencies(gt_camera rclcpp pluginlib sensor_msgs vision_msgs webots_ros2_driver)
pluginlib_export_plugin_description_file(webots_ros2_driver limo_fire_sim_plugins.xml)

# 라이다 오차 통계(V3): 녹화 → YAML. 같은 통계 코드(scan_stats.hpp)를 학습 시뮬 noise_check 도 쓴다
add_executable(scan_stats_bag src/scan_stats_bag.cpp)
target_compile_features(scan_stats_bag PUBLIC cxx_std_17)
target_include_directories(scan_stats_bag PRIVATE include)
ament_target_dependencies(scan_stats_bag rclcpp rosbag2_cpp sensor_msgs nav_msgs)

# 실물 녹화 → 다시 돌리기 파일(.lfrec): 학습 시뮬 replay_check 가 같은 자세에서 시뮬 라이다를 실물과 비교
add_executable(bag_export src/bag_export.cpp)
target_compile_features(bag_export PUBLIC cxx_std_17)
ament_target_dependencies(bag_export rclcpp rosbag2_cpp sensor_msgs nav_msgs tf2_msgs)

# 실물 녹화 → 구동 응답 표(CSV): 학습 시뮬 motion_check 가 시뮬 구동 모델을 실물 명령·속도에 맞춘다
add_executable(motion_export src/motion_export.cpp)
target_compile_features(motion_export PUBLIC cxx_std_17)
ament_target_dependencies(motion_export rclcpp rosbag2_cpp sensor_msgs nav_msgs geometry_msgs limo_msgs)

install(TARGETS gt_camera LIBRARY DESTINATION lib)
install(TARGETS scan_stats_bag bag_export motion_export RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY resource DESTINATION share/${PROJECT_NAME})

ament_package()
