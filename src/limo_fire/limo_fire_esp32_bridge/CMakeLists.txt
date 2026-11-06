cmake_minimum_required(VERSION 3.8)
project(limo_fire_esp32_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# ESP32 USB 시리얼 → ROS (지금은 초음파 Range). 줄 형식은 include/limo_fire_esp32_bridge/protocol.hpp = 펌웨어 include/protocol.h
add_executable(esp32_bridge src/esp32_bridge_node.cpp)
target_include_directories(esp32_bridge PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
ament_target_dependencies(esp32_bridge rclcpp sensor_msgs)

install(TARGETS esp32_bridge DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_protocol test/test_protocol.cpp)
  target_include_directories(test_protocol PRIVATE include)
endif()

ament_package()
