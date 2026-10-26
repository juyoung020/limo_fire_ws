cmake_minimum_required(VERSION 3.8)
project(limo_fire_perception LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(CUDAToolkit REQUIRED)

# TensorRT (JetPack: /usr/include/aarch64-linux-gnu, /usr/lib/aarch64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(NVINFER_LIB nvinfer)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT(NvInfer.h, libnvinfer) 를 찾지 못함")
endif()

add_library(trt_yolo src/trt_yolo.cpp)
target_include_directories(trt_yolo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_yolo PUBLIC ${NVINFER_LIB} CUDA::cudart opencv_core opencv_imgproc opencv_dnn)

add_executable(yolo_detector src/yolo_detector_node.cpp)
target_link_libraries(yolo_detector trt_yolo)
ament_target_dependencies(yolo_detector rclcpp sensor_msgs vision_msgs)

# 탐지 상자 + 깊이 → 지도 위 물체 위치 (TensorRT·OpenCV 안 씀)
add_executable(object_mapper src/object_mapper_node.cpp)
ament_target_dependencies(object_mapper rclcpp sensor_msgs vision_msgs visualization_msgs geometry_msgs
  message_filters tf2_ros tf2_geometry_msgs)

install(TARGETS trt_yolo yolo_detector object_mapper
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
