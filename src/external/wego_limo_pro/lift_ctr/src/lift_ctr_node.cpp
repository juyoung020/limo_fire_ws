#include "lift_ctr/lift_ctr.hpp"

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LiftControl>());
    rclcpp::shutdown();
    return 0;
}