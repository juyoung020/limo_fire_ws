// shared_image.hpp 시험: 한 프로세스의 두 노드가 같은 토픽을 받으면 DDS 구독은 하나, 같은 포인터를 받음.
// 구독을 만든 노드가 빠지면 남은 노드에 다시 만들어 계속 받음, 다 빠지면 구독 0.
// colcon test --packages-select limo_fire_perception
#include <chrono>
#include <functional>
#include <memory>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "limo_fire_perception/shared_image.hpp"

using limo_fire_perception::SharedImageSubscription;
using Image = sensor_msgs::msg::Image;

class SharedImageTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    a = std::make_shared<rclcpp::Node>("a");
    b = std::make_shared<rclcpp::Node>("b");
    cam = std::make_shared<rclcpp::Node>("cam");
    pub = cam->create_publisher<Image>("/shared_image_test", 10);
    for (auto & n : {a, b, cam}) {exec.add_node(n);}
  }
  // cond 가 참이 될 때까지 영상을 내며 돌림 (최대 5 s)
  bool spin_until(const std::function<bool()> & cond)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < end) {
      if (cond()) {return true;}
      pub->publish(Image());
      exec.spin_some(std::chrono::milliseconds(20));
    }
    return cond();
  }
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::Node::SharedPtr a, b, cam;
  rclcpp::Publisher<Image>::SharedPtr pub;
};

TEST_F(SharedImageTest, OneSubscriptionSamePointerAndHandover)
{
  Image::ConstSharedPtr got_a, got_b;
  auto ha = SharedImageSubscription::create(a.get(), "/shared_image_test", [&](const Image::ConstSharedPtr & m) {got_a = m;});
  auto hb = SharedImageSubscription::create(b.get(), "shared_image_test", [&](const Image::ConstSharedPtr & m) {got_b = m;});
  ASSERT_TRUE(spin_until([&] {return got_a && got_b && got_a == got_b;}));
  EXPECT_EQ(pub->get_subscription_count(), 1u);   // 두 노드에 구독 하나
  EXPECT_EQ(got_a.get(), got_b.get());            // 같은 메시지 (역직렬화 한 번)

  ha.reset();   // 구독을 만든 노드가 빠짐 → b 에 다시 만듦
  got_b.reset();
  ASSERT_TRUE(spin_until([&] {return got_b != nullptr;}));
  EXPECT_EQ(pub->get_subscription_count(), 1u);
  EXPECT_EQ(b->count_subscribers("/shared_image_test"), 1u);
  EXPECT_EQ(a->count_subscribers("/shared_image_test"), 1u);

  hb.reset();   // 다 빠짐
  ASSERT_TRUE(spin_until([&] {return pub->get_subscription_count() == 0u;}));
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int r = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return r;
}
