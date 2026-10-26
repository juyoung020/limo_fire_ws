#pragma once
#include <memory>
#include <chrono>
#include <functional>
#include <string>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/string.hpp"

#include <boost/asio.hpp>

class LiftControl : public rclcpp::Node
{
public:
    LiftControl();

private:
    void lift_callback(const std_msgs::msg::UInt8 & msg);
    void timer_callback();
    
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr lift_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lift_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    boost::asio::io_service io_;
    boost::asio::serial_port serial_{io_};
};