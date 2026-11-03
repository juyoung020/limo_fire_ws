# 탐사 RL 신경망(network/) + PPO(ppo/) 라이브러리와 검증 도구.
#   cmake -S train/explore/ppo -B <빌드> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc && cmake --build <빌드> -j
cmake_minimum_required(VERSION 3.22)
project(explore_ppo LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

set(NET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../network)
add_library(explore_net STATIC ${NET_DIR}/common.cu ${NET_DIR}/actor_critic.cu ${NET_DIR}/cnn_fused.cu ${NET_DIR}/gemm_fused.cu)
target_include_directories(explore_net PUBLIC ${NET_DIR})
target_link_libraries(explore_net PUBLIC CUDA::cudart CUDA::cublas)

add_library(explore_ppo STATIC ppo.cu)
target_include_directories(explore_ppo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(explore_ppo PUBLIC explore_net CUDA::curand)

# fp64 한 벌: 기울기 검사의 기준 (이름공간 xnet64·xppo64)
add_library(explore_net64 STATIC ${NET_DIR}/common.cu ${NET_DIR}/actor_critic.cu)
target_include_directories(explore_net64 PUBLIC ${NET_DIR})
target_compile_definitions(explore_net64 PUBLIC XNET_FP64)
target_link_libraries(explore_net64 PUBLIC CUDA::cudart CUDA::cublas)
add_library(explore_ppo64 STATIC ppo.cu tools/gradcheck64.cu)
target_include_directories(explore_ppo64 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} tools)
target_link_libraries(explore_ppo64 PUBLIC explore_net64)

add_library(gc32 STATIC tools/gradcheck32.cu)
target_include_directories(gc32 PRIVATE tools)
target_link_libraries(gc32 PUBLIC explore_ppo)
add_executable(gradcheck tools/gradcheck.cu)
target_link_libraries(gradcheck gc32 explore_ppo64)

foreach(t toy_train bench gemm_bench)
  add_executable(${t} tools/${t}.cu)
  target_link_libraries(${t} explore_ppo)
endforeach()
# 합성곱 융합 커널 정확도(텐서 코어 vs fp32)·속도
foreach(t cnn_check cnn_bench)
  add_executable(${t} tools/${t}.cu)
  target_link_libraries(${t} explore_net)
endforeach()
