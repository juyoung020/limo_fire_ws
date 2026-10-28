cmake_minimum_required(VERSION 3.8)
# GPU 커널(전처리·출력 해석·마스크)용 nvcc. JetPack 은 PATH 에 없을 수 있다
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
endif()
project(limo_fire_perception LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rviz_2d_overlay_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
find_package(CUDAToolkit REQUIRED)

# TensorRT (JetPack: /usr/include/aarch64-linux-gnu, /usr/lib/aarch64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(NVINFER_LIB nvinfer)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT(NvInfer.h, libnvinfer) 를 찾지 못함")
endif()

add_library(trt_yolo STATIC src/trt_yolo.cpp src/trt_yolo_kernels.cu)
set_target_properties(trt_yolo PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(trt_yolo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_yolo PUBLIC ${NVINFER_LIB} CUDA::cudart opencv_core opencv_imgproc opencv_dnn)

# 노드는 컴포넌트(공유 라이브러리)로 만든다: 카메라 드라이버·서로와 한 프로세스에 올리면 메시지를 복사 없이 주고받는다.
# 같은 이름의 실행 파일(yolo_detector, object_mapper)도 함께 만들어져 따로 띄울 수도 있다
add_library(yolo_detector_component SHARED src/yolo_detector_node.cpp)
target_link_libraries(yolo_detector_component trt_yolo)
ament_target_dependencies(yolo_detector_component rclcpp rclcpp_components sensor_msgs vision_msgs)
rclcpp_components_register_node(yolo_detector_component
  PLUGIN "limo_fire_perception::YoloDetectorNode" EXECUTABLE yolo_detector)

# 탐지 상자 + 깊이 → 지도 위 물체 위치 (TensorRT·OpenCV 안 씀)
add_library(object_mapper_component SHARED src/object_mapper_node.cpp)
ament_target_dependencies(object_mapper_component rclcpp rclcpp_components sensor_msgs vision_msgs
  visualization_msgs geometry_msgs message_filters tf2_ros tf2_geometry_msgs std_srvs rviz_2d_overlay_msgs)
rclcpp_components_register_node(object_mapper_component
  PLUGIN "limo_fire_perception::ObjectMapperNode" EXECUTABLE object_mapper)

install(TARGETS trt_yolo yolo_detector_component object_mapper_component
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY launch config icons DESTINATION share/${PROJECT_NAME})

ament_package()
