#include "limo_base/limo_ackerman_driver.hpp"

using namespace WeGo;

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LimoAckermanDriver>());
  rclcpp::shutdown();
  return 0;
}
