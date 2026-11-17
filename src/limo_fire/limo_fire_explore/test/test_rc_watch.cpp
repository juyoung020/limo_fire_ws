// explorer 리모컨 받기 (include/limo_fire_explore/rc_watch.h) 시험: 다시 구독 판단 · QoS 가 RELIABLE/BEST_EFFORT 발행자와 맞는지 ·
// 받는 노드를 여러 번 새로 만들어도(노드 다시 띄우기) 받는지
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>

#include "limo_fire_explore/rc_watch.h"

using limo_fire_explore::RcWatch;
using clk = std::chrono::steady_clock;
using namespace std::chrono_literals;

TEST(RcWatch, ResubscribeDecision) {
  RcWatch w;
  const auto t0 = clk::now();
  w.subscribed(t0);
  EXPECT_FALSE(w.want_resub(t0 + 1s, 1, 2.0));    // 아직 기다림
  EXPECT_TRUE(w.want_resub(t0 + 3s, 1, 2.0));     // 발행자는 있는데 하나도 못 받음
  EXPECT_FALSE(w.want_resub(t0 + 3s, 0, 2.0));    // 발행자 없음 (섀시 드라이버 꺼짐) — 다시 구독 안 함
  w.got(t0 + 2500ms);
  EXPECT_FALSE(w.want_resub(t0 + 3s, 1, 2.0));    // 받는 중
  EXPECT_NEAR(w.age_s(t0 + 3s), 0.5, 1e-6);
  EXPECT_TRUE(w.want_resub(t0 + 5s, 1, 2.0));     // 받다가 끊김
  w.subscribed(t0 + 5s);
  EXPECT_EQ(w.resubs(), 1);
  EXPECT_FALSE(w.want_resub(t0 + 6s, 1, 2.0));
}

namespace {
// pub_qos 로 40 Hz 내는 발행 노드 하나 + 받는 노드를 n 번 새로 만들어 각각 2 s 안에 받는지
int receive_rounds(const rclcpp::QoS& pub_qos, int rounds) {
  auto pn = std::make_shared<rclcpp::Node>("rc_pub_test");
  auto pub = pn->create_publisher<sensor_msgs::msg::Joy>("/rc_state_test", pub_qos);
  std::atomic<bool> run{true};
  std::thread th([&] {
    while (run) {
      sensor_msgs::msg::Joy j;
      j.buttons = {0, 2};
      pub->publish(j);
      std::this_thread::sleep_for(25ms);
    }
  });
  int ok = 0;
  for (int r = 0; r < rounds; ++r) {
    auto sn = std::make_shared<rclcpp::Node>("rc_sub_test");
    std::atomic<int> got{0};
    auto sub = sn->create_subscription<sensor_msgs::msg::Joy>("/rc_state_test", limo_fire_explore::rc_qos(),
                                                              [&](sensor_msgs::msg::Joy::ConstSharedPtr m) { got += m->buttons.size() > 1 && m->buttons[1] == 2; });
    rclcpp::executors::SingleThreadedExecutor ex;
    ex.add_node(sn);
    const auto t0 = clk::now();
    while (got == 0 && clk::now() - t0 < 2s) ex.spin_some(10ms);
    ok += got > 0;
  }
  run = false;
  th.join();
  return ok;
}
}  // namespace

TEST(RcQos, ReliablePublisherAndRestarts) { EXPECT_EQ(receive_rounds(rclcpp::QoS(10).reliable(), 5), 5); }
TEST(RcQos, BestEffortPublisher) { EXPECT_EQ(receive_rounds(rclcpp::SensorDataQoS(), 3), 3); }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int r = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return r;
}
