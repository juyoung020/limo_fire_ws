cmake_minimum_required(VERSION 3.22)
# 탐사 RL 학습 시뮬레이터: 삼각형 장면 + BVH 광선 추적 센서. CPU 참조판과 비트까지 같으려고 fmad 를 끄고 나눗셈·제곱근을 정확하게 한다.
# 빌드 대상: 젯슨 Orin(sm_87) 검증, 데스크톱 RTX 5070 Ti(sm_120) 학습. 따로 안 주면 이 기기에 맞춘다
project(limo_fire_sim LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ffp-contract=off")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false -prec-div=true -prec-sqrt=true -ftz=false")
find_package(CUDAToolkit REQUIRED)
# tinyxml2: CMake 설정이 있으면 그것, 없으면(우분투 22.04 젯슨) 라이브러리를 직접 찾는다
find_package(tinyxml2 CONFIG QUIET)
if(NOT TARGET tinyxml2::tinyxml2)
  find_library(TINYXML2_LIB tinyxml2 REQUIRED)
  find_path(TINYXML2_INC tinyxml2.h REQUIRED)
  add_library(tinyxml2::tinyxml2 UNKNOWN IMPORTED)
  set_target_properties(tinyxml2::tinyxml2 PROPERTIES IMPORTED_LOCATION ${TINYXML2_LIB} INTERFACE_INCLUDE_DIRECTORIES ${TINYXML2_INC})
endif()

find_package(yaml-cpp REQUIRED)

add_library(limo_sim STATIC src/sensors_kernel.cu src/sensors_ref.cpp src/w3d.cpp src/sensor_model.cpp
            src/env_kernel.cu src/env_ref.cpp src/env_host.cpp src/people_host.cpp)
target_include_directories(limo_sim PUBLIC include ${CMAKE_CURRENT_SOURCE_DIR}/../mdp)
find_package(Threads REQUIRED)
target_link_libraries(limo_sim PUBLIC CUDA::cudart tinyxml2::tinyxml2 yaml-cpp Threads::Threads)
# 기본 모델·실측 통계 파일 위치
target_compile_definitions(limo_sim PUBLIC SIM_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
                                           SIM_ANALYSIS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/analysis")

add_executable(w3d2scene tools/w3d2scene.cpp)
target_link_libraries(w3d2scene limo_sim)
add_executable(sim_verify tools/sim_verify.cu)
target_link_libraries(sim_verify limo_sim)
# V3: 라이다 오차 분포 vs 실측. 통계 코드는 실물 녹화 도구(limo_fire_sim scan_stats_bag)와 같은 헤더
add_executable(noise_check tools/noise_check.cu)
target_include_directories(noise_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/limo_fire/limo_fire_sim/include)
target_link_libraries(noise_check limo_sim)
# 라이다 먼 거리 빠짐 vs 실물 정지 묶음 (analysis/real_far_stats.yaml)
add_executable(far_check tools/far_check.cu)
target_link_libraries(far_check limo_sim)
# 학습 환경(env.h): GPU == CPU 비트 비교, 속도
add_executable(env_verify tools/env_verify.cu)
target_link_libraries(env_verify limo_sim)
add_executable(env_bench tools/env_bench.cu)
target_link_libraries(env_bench limo_sim)
# 시뮬 = 현실 (구동): 실물 녹화 명령으로 구동 모델을 맞추고 확인 (CSV 는 limo_fire_sim motion_export)
add_executable(motion_check tools/motion_check.cpp)
target_link_libraries(motion_check limo_sim)
# 실물 지도 → 장면, GPU 환경 전체를 실물 녹화로 다시 돌려 비교
add_executable(map2scene tools/map2scene.cpp)
target_link_libraries(map2scene limo_sim)
add_executable(replay_check tools/replay_check.cu)
target_link_libraries(replay_check limo_sim)
# 시뮬 = 현실 (IMU): 실물 움직임을 시뮬 IMU 모델(imu.h)에 넣어 가속도·자이로 x·y·roll·pitch 통계 비교·맞춤
# 시뮬 = 현실 (깊이): 실물 깊이 통계(limo_fire_sim depth_stats_bag)로 깊이 모델을 맞추고 확인
add_executable(depth_check tools/depth_check.cpp)
target_link_libraries(depth_check limo_sim)
# 시뮬 = 현실 (지도 속도): 실물 움직임을 지도 자세 오차 모델(pose_error.h)에 넣어 0.1 s 미분 잔차 통계 비교·맞춤
add_executable(mapvel_check tools/mapvel_check.cpp)
target_link_libraries(mapvel_check limo_sim)
# 시뮬 = 현실 (라이다 세기): 실물 정지 세기 통계(limo_fire_sim intensity_stats_bag)로 세기 모델 맞춤·확인
add_executable(intensity_check tools/intensity_check.cpp)
target_link_libraries(intensity_check limo_sim)
add_executable(imu_check tools/imu_check.cpp)
target_link_libraries(imu_check limo_sim)
# 학습 맵 묶음: build_world 닫힘 검사 + 넓이·벽 면 칸 수·크기 등급 목록 (train/explore/scenes/README.md)
add_executable(scene_list tools/scene_list.cpp)
target_link_libraries(scene_list limo_sim)
# 움직이는 사람: 라이다·깊이·지도에 보이나, 긴 시간 돌려 벽을 안 뚫나 (people.h)
add_executable(people_check tools/people_check.cu)
target_link_libraries(people_check limo_sim)
# 옛 빌드 == 새 빌드: 같은 씨앗 출력 해시 (커널 최적화가 시뮬 결과를 안 바꿨나)
add_executable(env_dump tools/env_dump.cu)
target_link_libraries(env_dump limo_sim)
# RT 코어 광선 순회 (OptiX). 헤더(../third_party/optix-dev, README)가 있으면 켠다 — 없으면 소프트웨어 순회만 (같은 답)
set(OPTIX_INC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/optix-dev/include)
# raygen 이 광선 결과 뒤 단계(include/env_rays.h)까지 하므로 시뮬·mdp 헤더 전부에 기댄다
file(GLOB SIM_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/../mdp/*.h)
option(SIM_OPTIX "RT 코어(OptiX)로 깊이 광선" ON)
if(SIM_OPTIX AND EXISTS ${OPTIX_INC}/optix.h)
  # PTX 로 (optix-ir 이 아니라): PTX 는 곱·합이 mul.rn·add.rn 으로 적혀 OptiX 컴파일러가 fma 로 합치지 않는다 — raygen 의 광선 만들기·결과 뒤 단계가 CPU 와 비트 같게
  # PTX 아키텍처는 89 가 상한: nvcc 13 은 sm_90 이상이면 -ptx 뒤에 ptxas 검사를 돌려 OptiX 내장 함수(_optix_*)를 모른다고 거부한다.
  # OptiX 가 실행 때 PTX 를 그 GPU(예: RTX 5060 sm_120)로 다시 컴파일하므로 낮은 PTX 로 충분하다.
  set(OPTIX_PTX_ARCH ${CMAKE_CUDA_ARCHITECTURES})
  if(OPTIX_PTX_ARCH GREATER 89)
    set(OPTIX_PTX_ARCH 89)
  endif()
  set(OPTIX_PTX ${CMAKE_CURRENT_BINARY_DIR}/optix_rays.ptx)
  add_custom_command(OUTPUT ${OPTIX_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_${OPTIX_PTX_ARCH} --fmad=false -prec-div=true -prec-sqrt=true -ftz=false -std=c++17
            -I${OPTIX_INC} -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/../mdp ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_rays.cu -o ${OPTIX_PTX}
    DEPENDS src/optix_rays.cu ${SIM_HEADERS})
  add_custom_target(optix_rays_ptx ALL DEPENDS ${OPTIX_PTX})
  target_sources(limo_sim PRIVATE src/optix_rt.cpp)
  add_dependencies(limo_sim optix_rays_ptx)
  target_include_directories(limo_sim PRIVATE ${OPTIX_INC})
  target_compile_definitions(limo_sim PUBLIC SIM_OPTIX=1 SIM_OPTIX_PTX="${OPTIX_PTX}")
  target_link_libraries(limo_sim PUBLIC ${CMAKE_DL_LIBS})
  message(STATUS "SIM_OPTIX: RT 코어 깊이 광선 켬 (${OPTIX_INC})")
  # 빠른 길(하드웨어 삼각형) vs 정확한 길 통계 동등성
  add_executable(rt_fastcheck tools/rt_fastcheck.cu)
  target_link_libraries(rt_fastcheck limo_sim)
endif()
# CPU 참조판은 이 기기에 맞춰 최적화(-O3 -march=native)하되 곱·합을 합치지 않아(-ffp-contract=off) GPU 와 비트 같음 그대로 — env_bench --cpu 속도 비교용
set_source_files_properties(src/env_ref.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native;-ffp-contract=off")
