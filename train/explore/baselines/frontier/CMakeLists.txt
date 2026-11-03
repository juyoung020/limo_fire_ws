cmake_minimum_required(VERSION 3.22)
# 프론티어 탐사 기준선: 시뮬 라이브러리(limo_sim)를 그대로 가져다 쓰고(시뮬 코드는 안 고침), 공용 헤더 mdp/frontier.h,
# 실행 폴더 쓰개 viewer/include/runfolder.h 를 쓴다.
#   export PATH=/usr/local/cuda-13.0/bin:$PATH
#   cmake -S train/explore/baselines/frontier -B <빌드> -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build <빌드> -j
project(limo_fire_frontier LANGUAGES CXX CUDA)
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

# 실행 도구: 장면 × 판을 돌려 학습 뷰어 실행 폴더로
add_executable(frontier_run ${EXPLORE}/scripts/frontier_run.cpp)
target_include_directories(frontier_run PRIVATE ${EXPLORE}/viewer/include)
target_link_libraries(frontier_run frontier_explorer)
target_compile_options(frontier_run PRIVATE -Wall -Wextra)

# mdp/frontier.h CPU == GPU 비트 비교
add_executable(frontier_check tools/frontier_check.cu)
target_link_libraries(frontier_check frontier_explorer)
