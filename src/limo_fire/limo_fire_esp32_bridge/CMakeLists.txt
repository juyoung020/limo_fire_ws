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

# ESP32 USB 시리얼 → ROS (초음파 Range · 열화상 Image · 레이더 PoseArray). 줄 형식은 include/limo_fire_esp32_bridge/protocol.hpp = 펌웨어 include/protocol.h
add_executable(esp32_bridge src/esp32_bridge_node.cpp)
target_include_directories(esp32_bridge PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
ament_target_dependencies(esp32_bridge rclcpp sensor_msgs geometry_msgs std_msgs)

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
endif()

ament_package()
