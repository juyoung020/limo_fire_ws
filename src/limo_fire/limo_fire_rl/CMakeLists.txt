cmake_minimum_required(VERSION 3.8)
project(limo_fire_rl)

# 학습 시뮬과 같은 관측 코드(train/explore/sim·mdp 헤더)를 그대로 컴파일한다 — 같은 입력이면 시뮬 CPU 참조판과 비트까지 같게:
# -ffp-contract=off (곱셈·덧셈 합치기 끔 — aarch64 g++ 는 기본으로 합쳐 값이 달라짐, sim/CMakeLists.txt 와 같은 플래그). CUDA 는 안 씀
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-ffp-contract=off)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

set(EXPLORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../train/explore)
get_filename_component(EXPLORE_DIR ${EXPLORE_DIR} ABSOLUTE)
if(NOT EXISTS ${EXPLORE_DIR}/sim/include/obs_ref.h)
  message(FATAL_ERROR "train/explore 소스가 없음 (${EXPLORE_DIR}) — 저장소 전체를 받은 워크스페이스에서 빌드")
endif()

# ROS 없는 부분: RobotEnv(관측) + 설정 읽기 (시뮬 소스 그대로)
add_library(limo_fire_rl_core STATIC src/robot_env.cpp ${EXPLORE_DIR}/sim/src/env_cfg.cpp ${EXPLORE_DIR}/sim/src/sensor_model.cpp)
target_include_directories(limo_fire_rl_core PUBLIC include ${EXPLORE_DIR}/sim/include ${EXPLORE_DIR}/mdp ${EXPLORE_DIR}/network)
target_link_libraries(limo_fire_rl_core PUBLIC yaml-cpp)
set_target_properties(limo_fire_rl_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(explore_rl src/explore_rl_node.cpp)
target_link_libraries(explore_rl limo_fire_rl_core)
# 기본 env.yaml = 이 저장소의 학습 설정 (그 안에서 nav2_params.yaml·sensor_model.yaml 을 상대 경로로 읽음 — 학습과 같은 값)
target_compile_definitions(explore_rl PRIVATE LIMO_FIRE_RL_ENV_YAML="${EXPLORE_DIR}/sim/config/env.yaml")
ament_target_dependencies(explore_rl rclcpp rclcpp_action geometry_msgs nav_msgs nav2_msgs sensor_msgs std_msgs std_srvs tf2 tf2_ros visualization_msgs)

# 프론티어 탐사 노드: 탐사 기준선과 같은 찾기(mdp/frontier.h)·고르기(baselines/frontier/include/explore_lite.h), 값도 기준선 설정 그대로
find_package(ament_index_cpp REQUIRED)
add_executable(explore_frontier src/explore_frontier_node.cpp)
target_include_directories(explore_frontier PRIVATE ${EXPLORE_DIR}/sim/include ${EXPLORE_DIR}/mdp ${EXPLORE_DIR}/baselines/frontier/include)
target_link_libraries(explore_frontier yaml-cpp)
target_compile_definitions(explore_frontier PRIVATE LIMO_FIRE_FRONTIER_YAML="${EXPLORE_DIR}/baselines/frontier/config/frontier.yaml")
ament_target_dependencies(explore_frontier rclcpp rclcpp_action nav2_msgs nav_msgs std_msgs std_srvs tf2 tf2_ros visualization_msgs ament_index_cpp)

install(TARGETS explore_rl explore_frontier DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
