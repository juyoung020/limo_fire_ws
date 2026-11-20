cmake_minimum_required(VERSION 3.8)
project(limo_fire_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(limo_fire_msgs REQUIRED)
find_package(limo_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# 물체 지도 → /fire/observations
add_executable(fire_observation_bridge src/fire_observation_bridge_node.cpp)
target_include_directories(fire_observation_bridge PRIVATE include)
ament_target_dependencies(fire_observation_bridge rclcpp limo_fire_msgs vision_msgs)

# 접근·좌우 조준·(선택) 한 발
add_executable(approach_aim src/approach_aim_node.cpp)
target_include_directories(approach_aim PRIVATE include)
ament_target_dependencies(approach_aim rclcpp rclcpp_action limo_fire_msgs limo_msgs nav2_msgs nav_msgs geometry_msgs
  std_msgs std_srvs tf2 tf2_ros)

# 사람이 쓰는 발사대 터미널
add_executable(launcher_terminal src/launcher_terminal.cpp)
target_include_directories(launcher_terminal PRIVATE include)
ament_target_dependencies(launcher_terminal rclcpp rclcpp_action limo_fire_msgs nav_msgs std_msgs std_srvs)

install(TARGETS fire_observation_bridge approach_aim launcher_terminal DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_aim_geometry test/test_aim_geometry.cpp)
  target_include_directories(test_aim_geometry PRIVATE include)
  # 실행 파일끼리: 가짜 Nav2·표적·차체(시험 안) + approach_aim + 발사 터미널 (+ 펌웨어 호스트 시뮬 + esp32_bridge 가 있으면 진짜 발사대 경로)
  find_package(ament_index_cpp REQUIRED)
  ament_add_gtest(test_approach_aim_ros test/test_approach_aim_ros.cpp TIMEOUT 240)
  target_include_directories(test_approach_aim_ros PRIVATE include)
  ament_target_dependencies(test_approach_aim_ros rclcpp rclcpp_action limo_fire_msgs limo_msgs nav2_msgs nav_msgs
    geometry_msgs std_msgs std_srvs tf2_ros ament_index_cpp)
  target_compile_definitions(test_approach_aim_ros PRIVATE
    APPROACH_AIM_EXE="$<TARGET_FILE:approach_aim>"
    LAUNCHER_TERMINAL_EXE="$<TARGET_FILE:launcher_terminal>"
    CONFIG_YAML="${CMAKE_CURRENT_SOURCE_DIR}/config/approach_aim.yaml")
  add_dependencies(test_approach_aim_ros approach_aim launcher_terminal)
endif()

ament_package()
