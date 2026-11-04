cmake_minimum_required(VERSION 3.8)
project(limo_fire_navigation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# 연기 모드: map → odom 발행 + 라이다 길목 + 속도 한도 (README "연기 모드")
add_executable(smoke_mode src/smoke_mode_node.cpp)
target_include_directories(smoke_mode PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(smoke_mode rclcpp geometry_msgs nav_msgs nav2_msgs sensor_msgs std_msgs std_srvs
  tf2 tf2_ros tf2_geometry_msgs)

install(TARGETS smoke_mode DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config maps DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_map_raycast test/test_map_raycast.cpp)
  target_include_directories(test_map_raycast PRIVATE include)
endif()

ament_package()
