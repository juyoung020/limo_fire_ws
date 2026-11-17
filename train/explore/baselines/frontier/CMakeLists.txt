cmake_minimum_required(VERSION 3.22)
# 프론티어 탐사 기준선(explore_lite + Nav2 식)을 시뮬로 모는 라이브러리 — 시뮬 라이브러리(limo_sim)와 공용 헤더 mdp/frontier.h 를 쓴다.
# 정답 상한 oracle_run 은 2026-11-17 에 archive/rl 로 (커리큘럼 문턱 측정). 실물 노드 limo_fire_explore frontier_explorer 는 explore_lite.h 만 씀.
#   export PATH=/usr/local/cuda-13.0/bin:$PATH
#   cmake -S train/explore/baselines/frontier -B <빌드> -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build <빌드> -j
project(limo_fire_frontier LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/cuda_strict.cmake)   # 장치 코드의 호스트 함수 부름 = 오류 (저장소 cmake/cuda_strict.cmake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(EXPLORE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_subdirectory(${EXPLORE}/sim ${CMAKE_CURRENT_BINARY_DIR}/sim)

add_library(frontier_explorer STATIC src/frontier_explorer.cpp)
target_include_directories(frontier_explorer PUBLIC include ${EXPLORE}/mdp)
target_link_libraries(frontier_explorer PUBLIC limo_sim)
target_compile_options(frontier_explorer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
target_compile_definitions(frontier_explorer PUBLIC FRONTIER_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/frontier.yaml")

# 실행 도구: 장면 × 판을 돌려 학습 뷰어 실행 폴더로 (기준선 nav2 · fast)
add_executable(frontier_run ${EXPLORE}/scripts/frontier_run.cpp)
target_include_directories(frontier_run PRIVATE ${EXPLORE}/viewer/include)
target_link_libraries(frontier_run frontier_explorer)
target_compile_options(frontier_run PRIVATE -Wall -Wextra)

# mdp/frontier.h CPU == GPU 비트 비교
add_executable(frontier_check tools/frontier_check.cu)
target_link_libraries(frontier_check frontier_explorer)
