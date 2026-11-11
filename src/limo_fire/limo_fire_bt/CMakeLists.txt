cmake_minimum_required(VERSION 3.8)
project(limo_fire_bt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

# 트리 XML 의 노드들 (조건·주행·임무 기록·부품) + 함께 쓰는 ROS 연결
add_library(${PROJECT_NAME}_nodes SHARED
  src/ros_context.cpp
  src/ros_helpers.cpp
  src/condition_nodes.cpp
  src/motion_nodes.cpp
  src/mission_nodes.cpp
  src/hardware_nodes.cpp
)
target_include_directories(${PROJECT_NAME}_nodes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_nodes yaml-cpp)
ament_target_dependencies(${PROJECT_NAME}_nodes
  behaviortree_cpp geometry_msgs nav2_msgs nav_msgs rclcpp rclcpp_action sensor_msgs std_msgs std_srvs
  tf2_geometry_msgs tf2_ros vision_msgs)

# 임무 노드: 트리를 읽어 돌린다
add_executable(mission src/mission_node.cpp)
target_link_libraries(mission ${PROJECT_NAME}_nodes)
ament_target_dependencies(mission behaviortree_cpp rclcpp)

install(TARGETS ${PROJECT_NAME}_nodes LIBRARY DESTINATION lib)
install(TARGETS mission RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY behavior_trees config launch DESTINATION share/${PROJECT_NAME})

ament_package()
