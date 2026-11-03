# C++ 쪽: fake_run(가짜 실행 생성기). runfolder.h 는 헤더 하나라 러너는 include 경로만 더하면 된다.
#   cmake -S train/explore/viewer -B train/explore/viewer/build -DCMAKE_BUILD_TYPE=Release && cmake --build train/explore/viewer/build -j4
cmake_minimum_required(VERSION 3.16)
project(trainview_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_executable(fake_run tools/fake_run.cpp)
target_include_directories(fake_run PRIVATE include)
target_compile_options(fake_run PRIVATE -Wall -Wextra)
