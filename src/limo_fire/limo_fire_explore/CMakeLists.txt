cmake_minimum_required(VERSION 3.8)
project(limo_fire_explore)

# 시뮬과 같은 지도·관측 코드(train/explore/sim·mdp 헤더)를 그대로 컴파일한다 — 같은 입력이면 시뮬 CPU 참조판과 비트까지 같게:
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
find_package(vision_msgs REQUIRED)
find_package(rviz_2d_overlay_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

set(EXPLORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../train/explore)
get_filename_component(EXPLORE_DIR ${EXPLORE_DIR} ABSOLUTE)
if(NOT EXISTS ${EXPLORE_DIR}/sim/include/obs_ref.h)
  message(FATAL_ERROR "train/explore 소스가 없음 (${EXPLORE_DIR}) — 저장소 전체를 받은 워크스페이스에서 빌드")
endif()

# ROS 없는 부분: RobotEnv(관측) + 설정 읽기 (시뮬 소스 그대로)
add_library(limo_fire_explore_core STATIC src/robot_env.cpp ${EXPLORE_DIR}/sim/src/env_cfg.cpp ${EXPLORE_DIR}/sim/src/sensor_model.cpp)
target_include_directories(limo_fire_explore_core PUBLIC include ${EXPLORE_DIR}/sim/include ${EXPLORE_DIR}/mdp)
target_link_libraries(limo_fire_explore_core PUBLIC yaml-cpp)
set_target_properties(limo_fire_explore_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 합친 탐사 노드 explorer: 위층 판단 모델 + 아래층 규칙 추종기(A* + 경로 추종) — 시뮬 low_explore_check 와 같은 코드 (README "합친 탐사")
# 위층 판단 모델: 지도 사진은 시뮬과 같은 scripts/decider.cpp (PNG 는 zlib), 요청 본문은 scripts/low_decider.h (JSON 은 nlohmann_json). HTTP 는 노드의 비차단 소켓 (decider_client.h)
# 컴포넌트(공유 라이브러리)로만 만든다: app.launch.xml 이 인식 노드와 한 프로세스 · 한 스레드 컨테이너에 올린다
find_package(rclcpp_components REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)   # 판단 모델 그림을 RViz 용 sensor_msgs/Image 로 풀 때 (decider_image)
add_library(explorer_component SHARED src/explorer_node.cpp ${EXPLORE_DIR}/scripts/decider.cpp)
target_link_libraries(explorer_component ZLIB::ZLIB PNG::PNG limo_fire_explore_core)
target_include_directories(explorer_component PRIVATE ${EXPLORE_DIR}/sim/tools ${EXPLORE_DIR}/scripts)   # low_node_rec.h (녹화 꼴) · low_explore_ctl.h (위층 — combine_run 과 같은 코드)
target_compile_definitions(explorer_component PRIVATE LIMO_FIRE_EXPLORE_ENV_YAML="${EXPLORE_DIR}/sim/config/env.yaml")
ament_target_dependencies(explorer_component rclcpp rclcpp_components geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf2 tf2_ros visualization_msgs rviz_2d_overlay_msgs vision_msgs)
rclcpp_components_register_nodes(explorer_component "limo_fire_explore::ExplorerNode")

# 프론티어 탐사 노드: 탐사 기준선과 같은 찾기(mdp/frontier.h)·고르기(baselines/frontier/include/explore_lite.h), 값도 기준선 설정 그대로
find_package(ament_index_cpp REQUIRED)
add_executable(frontier_explorer src/frontier_explorer_node.cpp)
target_include_directories(frontier_explorer PRIVATE ${EXPLORE_DIR}/sim/include ${EXPLORE_DIR}/mdp ${EXPLORE_DIR}/baselines/frontier/include)
target_link_libraries(frontier_explorer yaml-cpp)
target_compile_definitions(frontier_explorer PRIVATE LIMO_FIRE_FRONTIER_YAML="${EXPLORE_DIR}/baselines/frontier/config/frontier.yaml")
ament_target_dependencies(frontier_explorer rclcpp rclcpp_action nav2_msgs nav_msgs std_msgs std_srvs tf2 tf2_ros visualization_msgs ament_index_cpp)

install(TARGETS frontier_explorer DESTINATION lib/${PROJECT_NAME})
install(TARGETS explorer_component ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
# include/limo_fire_explore/class_alias.h (지시 글 → 클래스) 는 저장 지도 object_goto(limo_fire_navigation) 도 씀
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  # 리모컨 받기 시험 (rc_watch.h — 다시 구독 판단 · QoS 호환 · 받는 노드 다시 만들기): colcon test --packages-select limo_fire_explore
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_rc_watch test/test_rc_watch.cpp TIMEOUT 60)
  target_include_directories(test_rc_watch PRIVATE include)
  ament_target_dependencies(test_rc_watch rclcpp sensor_msgs)
endif()

ament_package()
