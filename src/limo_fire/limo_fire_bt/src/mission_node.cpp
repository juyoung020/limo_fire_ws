// 임무 노드: 트리 XML 을 읽어 tick_hz 로 돌린다. 주행은 Nav2 navigate_to_pose 를 부른다 (Nav2 bt_navigator 와 따로).
// 설정: config/limo_fire_bt.yaml (실물), 시뮬은 그 위에 config/limo_fire_bt_sim.yaml
// Groot2: groot_port 로 지금 켜진 노드를 본다. btlog_dir 이 비어 있지 않으면 판마다 .btlog 를 남긴다
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <thread>
#include <memory>
#include <string>

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/xml_parsing.h>
#include <behaviortree_cpp/loggers/bt_file_logger_v2.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>
#include <rclcpp/rclcpp.hpp>

#include "limo_fire_bt/nodes.hpp"
#include "limo_fire_bt/status_overlay.hpp"

namespace
{
void declare_params(rclcpp::Node & n)
{
  n.declare_parameter<std::string>("tree_file", "");
  n.declare_parameter<std::string>("patrol_file", "");
  n.declare_parameter<double>("tick_hz", 10.0);
  n.declare_parameter<int>("groot_port", 1667);
  n.declare_parameter<std::string>("btlog_dir", "");
  n.declare_parameter<double>("server_timeout_s", 5.0);
  n.declare_parameter<bool>("rc.required", true);
  n.declare_parameter<double>("rc.timeout_s", 0.5);
  n.declare_parameter<int>("rc.switch_index", 1);
  n.declare_parameter<int>("rc.auto_value", 2);
  n.declare_parameter<std::string>("classes.fire", "fire");
  n.declare_parameter<std::string>("classes.smoke", "smoke");
  n.declare_parameter<std::string>("classes.person", "person");
  n.declare_parameter<double>("hit_window_s", 1.0);
  n.declare_parameter<double>("approx_distance_m", 2.0);
  n.declare_parameter<int>("frontier.free_max", 25);
  n.declare_parameter<bool>("call119.enabled", false);
  n.declare_parameter<std::string>("call119.topic", "/mission/call119");
  n.declare_parameter<std::string>("record_dir", "/tmp/limo_fire_incidents");
  n.declare_parameter<std::string>("nozzle.command_topic", "/nozzle/command");
  n.declare_parameter<std::string>("nozzle.pan_joint", "nozzle_pan_joint");
  n.declare_parameter<std::string>("nozzle.tilt_joint", "nozzle_tilt_joint");
  n.declare_parameter<double>("nozzle.tolerance_rad", 0.05);
  n.declare_parameter<double>("nozzle.timeout_s", 3.0);
  n.declare_parameter<std::string>("speaker.service", "/speaker/evacuation");
  n.declare_parameter<double>("people.timeout_s", 1.0);
  n.declare_parameter<std::string>("overlay.topic", "/mission/overlay");
  n.declare_parameter<double>("overlay.period_s", 0.5);
  n.declare_parameter<int>("overlay.events", 3);
  n.declare_parameter<int>("overlay.width", 420);
  n.declare_parameter<int>("overlay.height", 260);
}

volatile std::sig_atomic_t g_stop = 0;
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, [](int) {g_stop = 1;});
  std::signal(SIGTERM, [](int) {g_stop = 1;});

  auto node = std::make_shared<rclcpp::Node>("mission");
  declare_params(*node);
  auto ctx = std::make_shared<limo_fire_bt::RosContext>(node);
  const auto log = node->get_logger();

  BT::BehaviorTreeFactory factory;
  limo_fire_bt::register_all_nodes(factory, ctx);
  // --model: Groot2·트리 XML 의 <TreeNodesModel> 을 코드에서 뽑아 낸다 (포트가 코드와 어긋나지 않게)
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--model") {
      std::cout << BT::writeTreeNodesModelXML(factory, false) << std::endl;
      rclcpp::shutdown();
      return 0;
    }
  }

  const auto tree_file = node->get_parameter("tree_file").as_string();
  if (tree_file.empty() || !std::filesystem::exists(tree_file)) {
    RCLCPP_FATAL(log, "tree_file 없음: \"%s\"", tree_file.c_str());
    return 1;
  }
  BT::Tree tree;
  try {
    factory.registerBehaviorTreeFromFile(tree_file);
    // ROS 설정 → 트리가 읽는 블랙보드 값
    auto bb = BT::Blackboard::create();
    bb->set("patrol_file", node->get_parameter("patrol_file").as_string());
    tree = factory.createTree("MainTree", bb);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(log, "트리를 못 만듦: %s", e.what());
    return 1;
  }

  // Groot2 는 보기용: 포트가 이미 쓰이고 있어도(다른 mission·시뮬) 임무는 계속한다
  std::unique_ptr<BT::Groot2Publisher> groot;
  const auto groot_port = static_cast<unsigned>(node->get_parameter("groot_port").as_int());
  try {
    groot = std::make_unique<BT::Groot2Publisher>(tree, groot_port);
  } catch (const std::exception & e) {
    RCLCPP_WARN(log, "Groot2 포트 %u 를 못 엶 (%s) — Groot2 없이 계속. groot_port 를 바꾸면 됨", groot_port, e.what());
  }
  std::unique_ptr<BT::FileLogger2> file_log;
  if (const auto dir = node->get_parameter("btlog_dir").as_string(); !dir.empty()) {
    std::filesystem::create_directories(dir);
    char stamp[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
    const auto path = std::filesystem::path(dir) / (std::string("mission_") + stamp + ".btlog");
    file_log = std::make_unique<BT::FileLogger2>(tree, path);
    RCLCPP_INFO(log, "BT 기록: %s", path.c_str());
  }

  limo_fire_bt::StatusOverlay overlay(ctx, tree);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  const auto period = std::chrono::duration<double>(1.0 / node->get_parameter("tick_hz").as_double());
  RCLCPP_INFO(log, "임무 시작: %s", tree_file.c_str());

  BT::NodeStatus status = BT::NodeStatus::RUNNING;
  const bool sim_time = node->get_parameter("use_sim_time").as_bool();
  auto clock_warn_at = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!g_stop && rclcpp::ok() && status == BT::NodeStatus::RUNNING) {
    const auto next = std::chrono::steady_clock::now() + period;
    exec.spin_some();
    // 시뮬 시계(/clock)가 들어오기 전에는 틱하지 않는다: 시각 0 에서 시작한 기다림(깊이 10 s 등)이
    // 첫 /clock 에 수십 초를 건너뛰어 바로 시간 초과가 된다. 안 들어오면 말없이 멈추지 않게 알린다
    if (sim_time && node->now().nanoseconds() == 0) {
      if (std::chrono::steady_clock::now() > clock_warn_at) {
        RCLCPP_WARN(log, "use_sim_time 인데 /clock 이 안 들어옴 — 시뮬(Webots)을 먼저 띄우거나 sim:=false 로 실행");
        clock_warn_at = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      }
      std::this_thread::sleep_until(next);
      continue;
    }
    try {
      status = tree.tickOnce();
      overlay.update();
    } catch (const std::exception & e) {
      RCLCPP_FATAL(log, "트리 오류: %s", e.what());
      status = BT::NodeStatus::FAILURE;
      break;
    }
    // 틱 사이에도 콜백(액션 응답 등)을 받는다. Sleep 등이 깨우면 바로 다음 틱
    while (std::chrono::steady_clock::now() < next && !g_stop) {
      exec.spin_some(std::chrono::milliseconds(5));
      if (tree.sleep(std::chrono::milliseconds(0))) {break;}
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  // 멈출 때: 진행 중인 노드를 멈춰 Nav2 목표 취소·펌프·연기 모드·탐사를 끈다. 보낸 요청이 나가도록 잠깐 돌린다
  tree.haltTree();
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  while (std::chrono::steady_clock::now() < until) {exec.spin_some(std::chrono::milliseconds(20));}
  RCLCPP_INFO(log, "임무 끝: %s", BT::toStr(status).c_str());
  rclcpp::shutdown();
  return status == BT::NodeStatus::FAILURE ? 1 : 0;
}
