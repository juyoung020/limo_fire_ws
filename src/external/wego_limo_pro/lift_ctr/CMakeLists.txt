cmake_minimum_required(VERSION 3.8)
project(lift_ctr)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

include_directories(./include/)

add_executable(lift_ctr ./src/lift_ctr_node.cpp ./include/lift_ctr/lift_ctr.hpp ./include/lift_ctr.cpp)
target_include_directories(lift_ctr PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(lift_ctr ${Boost_LIBRARIES})
ament_target_dependencies(lift_ctr rclcpp std_msgs)

install(TARGETS
  lift_ctr
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
