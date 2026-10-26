cmake_minimum_required(VERSION 3.8)
project(limo_fire_navigation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config maps DESTINATION share/${PROJECT_NAME})

ament_package()
