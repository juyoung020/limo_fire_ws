# 탐사 RL 학습 러너(train): 시뮬(limo_sim) + 신경망·PPO(explore_ppo) + 실행 폴더 쓰개(viewer/include/runfolder.h).
#   cmake -S train/explore/scripts -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake --build <빌드> -j8 --target train
cmake_minimum_required(VERSION 3.22)
# GPU 아키텍처 기본값은 project() 앞에서 정한다 — project() 가 CUDA 를 켜며 nvcc 기본값(75)을 넣어 버려, 뒤에서 "없으면" 을 보면 늘 있다.
# -DCMAKE_CUDA_ARCHITECTURES=… 나 환경 변수 CUDAARCHS 를 주면 그 값을 쓴다.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060
endif()
project(limo_fire_explore_train LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(EXPLORE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_subdirectory(${EXPLORE}/sim ${CMAKE_CURRENT_BINARY_DIR}/sim)
add_subdirectory(${EXPLORE}/ppo ${CMAKE_CURRENT_BINARY_DIR}/ppo)

add_executable(train train.cpp)
set_source_files_properties(train.cpp PROPERTIES LANGUAGE CUDA)   # done 매핑 커널 하나
target_include_directories(train PRIVATE ${EXPLORE}/viewer/include)
target_link_libraries(train limo_sim explore_ppo)
# 재생 화면의 로봇 모양: 러너가 시작할 때 URDF(xacro) 치수를 읽어 run.json robot 에 적는다
target_compile_definitions(train PRIVATE LIMO_URDF_DIR="${EXPLORE}/../../src/limo_fire/limo_fire_description/urdf")
