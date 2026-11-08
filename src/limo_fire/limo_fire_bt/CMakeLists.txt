cmake_minimum_required(VERSION 3.8)
project(limo_fire_bt)

find_package(ament_cmake REQUIRED)

# 트리 XML 만 설치한다. BT 노드 플러그인(C++)은 docs/bt_sim_plan.md 순서대로 더한다
install(DIRECTORY behavior_trees DESTINATION share/${PROJECT_NAME})

ament_package()
