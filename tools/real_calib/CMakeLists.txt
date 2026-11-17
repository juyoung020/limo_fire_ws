cmake_minimum_required(VERSION 3.10)
project(real_calib CXX)
# ROS 없이 빌드: cmake -S tools/real_calib -B /tmp/real_calib_build && cmake --build /tmp/real_calib_build && ctest --test-dir /tmp/real_calib_build
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)
add_executable(real_calib real_calib.cpp)
add_executable(test_real_calib test_real_calib.cpp)
enable_testing()
add_test(NAME real_calib COMMAND test_real_calib)
