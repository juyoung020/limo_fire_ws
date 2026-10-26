#ifndef LIMO_ACKERMAN_DRIVER_H
#define LIMO_ACKERMAN_DRIVER_H

#include <string>
#include <memory>
#include <iostream>
#include <functional>
#include <cmath>
#include <atomic>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "limo_msgs/msg/limo_status.hpp"


#include "limo_base/serial_port.h"
#include "limo_base/limo_protocol.h"

namespace WeGo{
class LimoAckermanDriver : public rclcpp::Node
{
    public:
        LimoAckermanDriver();
        virtual ~LimoAckermanDriver() = default;

    private:
        // parameter
        std::string odom_frame_;
        std::string base_frame_;
        bool pub_odom_tf_ = false;

        // limo serial
        std::shared_ptr<AgileX::SerialPort> port_;
        std::atomic<bool> keep_running_;
        std::shared_ptr<std::thread> read_data_thread_;

        // set publisher
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
        rclcpp::Publisher<limo_msgs::msg::LimoStatus>::SharedPtr status_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;

        // set subscriber
        rclcpp::Subscription<ackermann_msgs::msg::AckermannDrive>::SharedPtr motion_cmd_sub_;

        // limo status check
        uint8_t motion_mode_;
        AgileX::ImuData imu_data_;


        // limo hardare
        static constexpr double max_steering_angle_ = 0.42;
        static constexpr double max_inner_angle_ = 0.48869;  // 28 degree
        static constexpr double track_ = 0.172;           // m (left right wheel distance)
        static constexpr double wheelbase_ = 0.2;         // m (front rear wheel distance)
        static constexpr double left_angle_scale_ = 2.47;
        static constexpr double right_angle_scale_ = 2.47;

        // for imu
        double present_theta_,last_theta_,delta_theta_,real_theta_,rad;

        // for odometry
        double position_x_ = 0.0;
        double position_y_ = 0.0;
        double theta_ = 0.0;
        
    private:
        void ackermanCmdCallback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg);
        void setMotionCommand(double linear_vel, double steer_angle, double lateral_vel, double angular_vel);
        void sendFrame(const AgileX::LimoFrame& frame);
        void connect(std::string dev_name, uint32_t bouadrate);
        void readData();
        void processRxData(uint8_t data);
        void parseFrame(const AgileX::LimoFrame& frame);
        void enableCommandedMode();
        double degToRad(double deg);
        void processErrorCode(uint16_t error_code);
        void publishLimoState(double stamp, uint8_t vehicle_state, uint8_t control_mode,
                        double battery_voltage, uint16_t error_code, int8_t motion_mode);
        void publishIMUData(double stamp);
        void publishOdometry(double stamp, double linear_velocity,
                         double angular_velocity, double lateral_velocity,
                         double steering_angle);
        double convertInnerAngleToCentral(double inner_angle);
};

}

#endif // LIMO_ACKERMAN_DRIVER_H
