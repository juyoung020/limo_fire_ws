cmake_minimum_required(VERSION 3.8)
project(limo_fire_esp32_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(limo_fire_msgs REQUIRED)

# ESP32 USB 시리얼 ↔ ROS (초음파 Range · 열화상 Image·Temperature · 레이더 PoseArray · 발사대 액션·서비스·상태 · /diagnostics). 줄 형식은 include/limo_fire_esp32_bridge/protocol.hpp = 펌웨어 include/protocol.h
add_executable(esp32_bridge src/esp32_bridge_node.cpp)
target_include_directories(esp32_bridge PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
ament_target_dependencies(esp32_bridge rclcpp rclcpp_action sensor_msgs geometry_msgs std_msgs std_srvs diagnostic_updater limo_fire_msgs)

# 가짜 ESP32 (가상 시리얼 pty 에 펌웨어와 같은 줄) — 실물 없이 다리를 시험. ROS 의존 없음
add_executable(fake_esp32 tools/fake_esp32.cpp)
target_include_directories(fake_esp32 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

install(TARGETS esp32_bridge fake_esp32 DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/fake_bridge_check.sh DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_protocol test/test_protocol.cpp)
  target_include_directories(test_protocol PRIVATE include)
  ament_add_gtest(test_fake_stream test/test_fake_stream.cpp)
  target_include_directories(test_fake_stream PRIVATE include)
  ament_add_gtest(test_launcher_logic test/test_launcher_logic.cpp)
  target_include_directories(test_launcher_logic PRIVATE include)
  # 다리 실행 파일 ↔ 펌웨어 원본(firmware/limo_fire_esp32 를 PC 에서 가상 시리얼로 돌린 /tmp/launcher_host_sim, test/run.sh 가 빌드).
  # 없으면 건너뜀(GTEST_SKIP). 액션·서비스·상태 토픽을 실제 펌웨어 타이밍으로 확인
  ament_add_gtest(test_bridge_host_sim test/test_bridge_host_sim.cpp TIMEOUT 120)
  target_compile_definitions(test_bridge_host_sim PRIVATE ESP32_BRIDGE_EXE="$<TARGET_FILE:esp32_bridge>")
  add_dependencies(test_bridge_host_sim esp32_bridge)
  target_include_directories(test_bridge_host_sim PRIVATE include)
  ament_target_dependencies(test_bridge_host_sim rclcpp rclcpp_action std_msgs std_srvs sensor_msgs limo_fire_msgs diagnostic_msgs)
endif()

ament_package()
