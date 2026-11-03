cmake_minimum_required(VERSION 3.8)
project(limo_fire_base)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(limo_msgs REQUIRED)

# WeGo/AgileX limo_base(archive/limo_base) 를 그대로 가져와 받고 버리던 프레임(리모컨·모터·바퀴 거리)만 더 낸다.
# 원본과 다른 줄은 "limo_fire" 표시 — diff -r archive/limo_base src/limo_fire/limo_fire_base 로 본다
add_executable(limo_fire_base src/limo_base_node.cpp src/limo_driver.cpp src/serial_port.cpp)
target_compile_features(limo_fire_base PUBLIC cxx_std_14)
target_include_directories(limo_fire_base PRIVATE include)
ament_target_dependencies(limo_fire_base rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs tf2 tf2_ros tf2_geometry_msgs limo_msgs)

install(TARGETS limo_fire_base RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
