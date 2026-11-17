# 두 층 합친 탐사 측정(combine_run): 시뮬(limo_sim) + 프론티어 기준선(baselines/frontier) + 위층 slot0 · fuel(fuel_upper.h).
#   cmake -S train/explore/scripts -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake --build <빌드> -j8 --target combine_run
# 탐사 강화학습(학습기 train · 평가 low_policy_run · hybrid_eval)은 2026-11-16 에 그만둠 — archive/rl/README.md
cmake_minimum_required(VERSION 3.22)
# GPU 아키텍처 기본값은 project() 앞에서 정한다 — project() 가 CUDA 를 켜며 nvcc 기본값(75)을 넣어 버려, 뒤에서 "없으면" 을 보면 늘 있다.
# -DCMAKE_CUDA_ARCHITECTURES=… 나 환경 변수 CUDAARCHS 를 주면 그 값을 쓴다.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060
endif()
project(limo_fire_explore_scripts LANGUAGES CXX CUDA)
include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/cuda_strict.cmake)   # 장치 코드의 호스트 함수 부름 = 오류 (저장소 cmake/cuda_strict.cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(EXPLORE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_subdirectory(${EXPLORE}/sim ${CMAKE_CURRENT_BINARY_DIR}/sim)

set(FRONTIER_DIR ${EXPLORE}/baselines/frontier)
add_library(frontier_explorer_h STATIC ${FRONTIER_DIR}/src/frontier_explorer.cpp)
target_include_directories(frontier_explorer_h PUBLIC ${FRONTIER_DIR}/include ${EXPLORE}/mdp)
target_link_libraries(frontier_explorer_h PUBLIC limo_sim)
target_compile_definitions(frontier_explorer_h PUBLIC FRONTIER_CONFIG="${FRONTIER_DIR}/config/frontier.yaml")

# 두 층 합친 탐사 측정(combine_run): 위층(slot0 · fuel) + 아래층 규칙 추종기로 탐사를 끝까지 — 평가 전용. 판단 모델 위층은 sim/tools/low_explore_check · upper_lora/rollout/dec_rollout (decider.cpp 를 그쪽이 직접 빌드)
add_executable(combine_run combine_run.cpp)
set_source_files_properties(combine_run.cpp PROPERTIES LANGUAGE CUDA)   # 예전 빌드와 같은 번역(nvcc 호스트 경로) — 결과 비트 같게
target_link_libraries(combine_run frontier_explorer_h)
