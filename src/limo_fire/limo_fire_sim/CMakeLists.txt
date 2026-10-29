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

# 정답 카메라: Webots Recognition(분할) → /gt/detections, /gt/mask (webots_ros2_driver 플러그인)
add_library(gt_camera SHARED src/gt_camera.cpp)
target_compile_features(gt_camera PUBLIC cxx_std_17)
ament_target_dependencies(gt_camera rclcpp pluginlib sensor_msgs vision_msgs webots_ros2_driver)
pluginlib_export_plugin_description_file(webots_ros2_driver limo_fire_sim_plugins.xml)

install(TARGETS gt_camera LIBRARY DESTINATION lib)
install(DIRECTORY resource DESTINATION share/${PROJECT_NAME})

ament_package()
