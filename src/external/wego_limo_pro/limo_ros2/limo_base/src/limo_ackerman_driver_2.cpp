#include "limo_base/limo_ackerman_driver.hpp"

int flag=0; 

namespace WeGo{
    LimoAckermanDriver::LimoAckermanDriver()
        : rclcpp::Node("limo_ackerman_node")
    {
        // seting the parameter
        std::string port_name = "ttyTHS1";
        port_name = this->declare_parameter<std::string>("port_name", port_name);
        odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
        base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
        pub_odom_tf_ = this->declare_parameter<bool>("pub_odom_tf", false);

        std::cout << "Loading parameters: " << std::endl;
        std::cout << "- port name: " << port_name << std::endl;
        std::cout << "- odom frame name: " << odom_frame_ << std::endl;
        std::cout << "- base frame name: " << base_frame_ << std::endl;
        std::cout << "- odom topic name: " << pub_odom_tf_ << std::endl;

        // set publisher
        tf_broadcaster_=std::make_shared<tf2_ros::TransformBroadcaster>(*this);
        odom_publisher_=this->create_publisher<nav_msgs::msg::Odometry>("/odom",50);
        status_publisher_ = this->create_publisher<limo_msgs::msg::LimoStatus>("/limo_status",50);
        imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("/imu",10);

        // set subscriber
        motion_cmd_sub_= this->create_subscription<ackermann_msgs::msg::AckermannDrive>(
        "ack_cmd",10,std::bind(&LimoAckermanDriver::ackermanCmdCallback,this,std::placeholders::_1));

        // connect to the serial port
        if (port_name.find("tty") != port_name.npos){ 
            port_name = "/dev/" + port_name;
            keep_running_=true;
            this->connect(port_name, B460800);
            this->enableCommandedMode();
            RCLCPP_INFO(this->get_logger(),"Open the serial port:'%s'",port_name.c_str());
        }
    }

    void LimoAckermanDriver::ackermanCmdCallback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg)
    {
        switch (motion_mode_) {
            case AgileX::MODE_ACKERMANN: {
                double steering_angle = msg->steering_angle / 1.215;
                double linear_vel = msg->speed;

                steering_angle = std::min(std::max(steering_angle, -max_steering_angle_), max_steering_angle_);
                
                setMotionCommand(linear_vel, 0, 0, steering_angle);
                break;       
            }
            default:{
                RCLCPP_WARN(this->get_logger(),"Invalid motion mode you should change limo to ackermann mode!");
                break;
            }
        }
    }

    void LimoAckermanDriver::setMotionCommand(double linear_vel, double angular_vel,
                                  double lateral_velocity, double steering_angle) {
        AgileX::LimoFrame frame;
        frame.id = MSG_MOTION_COMMAND_ID;
        int16_t linear_cmd = linear_vel * 1000;
        int16_t angular_cmd = angular_vel * 1000;
        int16_t lateral_cmd = lateral_velocity * 1000;
        int16_t steering_cmd = steering_angle * 1000;

        frame.data[0] = static_cast<uint8_t>(linear_cmd >> 8);
        frame.data[1] = static_cast<uint8_t>(linear_cmd & 0x00ff);
        frame.data[2] = static_cast<uint8_t>(angular_cmd >> 8);
        frame.data[3] = static_cast<uint8_t>(angular_cmd & 0x00ff);
        frame.data[4] = static_cast<uint8_t>(lateral_cmd >> 8);
        frame.data[5] = static_cast<uint8_t>(lateral_cmd & 0x00ff);
        frame.data[6] = static_cast<uint8_t>(steering_cmd >> 8);
        frame.data[7] = static_cast<uint8_t>(steering_cmd & 0x00ff);
        sendFrame(frame);
    }

    void LimoAckermanDriver::sendFrame(const AgileX::LimoFrame& frame) {
        uint32_t checksum = 0;
        uint8_t frame_len = 0x0e;
        uint8_t data[14] = {0x55, frame_len};

        data[2] = static_cast<uint8_t>(frame.id >> 8);
        data[3] = static_cast<uint8_t>(frame.id & 0xff);
        for (size_t i = 0; i < 8; ++i) {
            data[i + 4] = frame.data[i];
            checksum += frame.data[i];
        }
        data[frame_len - 1] = static_cast<uint8_t>(checksum & 0xff);

        port_->writeData(data, frame_len);
    }

    void LimoAckermanDriver::connect(std::string dev_name, uint32_t bouadrate) {
        RCLCPP_INFO(this->get_logger(),"connet the serial port:'%s'",dev_name.c_str());
        port_ = std::shared_ptr<AgileX::SerialPort>(new AgileX::SerialPort(dev_name, bouadrate));

        if (port_->openPort() == 0) {
            read_data_thread_ = std::shared_ptr<std::thread>(
                new std::thread(std::bind(&LimoAckermanDriver::readData, this)));
        }
        else {
            RCLCPP_ERROR(this->get_logger(),"Failed to open: '%s'",port_->getDevPath().c_str());
            port_->closePort();
            exit(-1);
        }
    }

    void LimoAckermanDriver::readData() {
        uint8_t rx_data = 0;
        while (rclcpp::ok()) {
            auto len = port_->readByte(&rx_data);
            if (len < 1)
                continue;
            processRxData(rx_data);
        }
    }

    void LimoAckermanDriver::processRxData(uint8_t data) {
        static AgileX::LimoFrame frame;
        static int data_num = 0;
        static uint8_t checksum = 0;
        static uint8_t state = AgileX::LIMO_WAIT_HEADER;
        switch (state) {
            case AgileX::LIMO_WAIT_HEADER:
             {

                if (data == FRAME_HEADER) {
                    frame.stamp = this->get_clock()->now().seconds();
                    state = AgileX::LIMO_WAIT_LENGTH;
                }
                break;
            }
            case AgileX::LIMO_WAIT_LENGTH:
            {
                if (data == FRAME_LENGTH) {
                    state = AgileX::LIMO_WAIT_ID_HIGH;
                }
                else {
                    state = AgileX::LIMO_WAIT_HEADER;
                }
                break;
            }
            case AgileX::LIMO_WAIT_ID_HIGH:
            {
                frame.id = static_cast<uint16_t>(data) << 8;
                state = AgileX::LIMO_WAIT_ID_LOW;
                break;
            }
            case AgileX::LIMO_WAIT_ID_LOW:
            {
                frame.id |= static_cast<uint16_t>(data);
                state = AgileX::LIMO_WAIT_DATA;
                data_num = 0;
                break;
            }
            case AgileX::LIMO_WAIT_DATA:
            {
                if (data_num < 8) {
                    frame.data[data_num++] = data;
                    checksum += data;
                }
                else {
                    frame.count = data;
                    state = AgileX::LIMO_CHECK;
                    data_num = 0;
                }
                break;
            }
            case AgileX::LIMO_CHECK:
            {
                if (data == checksum) {
                    parseFrame(frame);
                }
                else {
                    RCLCPP_WARN(this->get_logger(),"Invalid frame! Check sum failed! ");
                }

                state = AgileX::LIMO_WAIT_HEADER;
                checksum = 0;
                memset(&frame.data[0], 0, 8);
                break;
            }
            default:
                break;
        }
    }

    void LimoAckermanDriver::parseFrame(const AgileX::LimoFrame& frame) {
        switch (frame.id) {
            case MSG_MOTION_STATE_ID: {
                double linear_velocity = static_cast<int16_t>((frame.data[1] & 0xff) | (frame.data[0] << 8)) / 1000.0;
                double angular_velocity = static_cast<int16_t>((frame.data[3] & 0xff) | (frame.data[2] << 8)) / 1000.0;
                double lateral_velocity = static_cast<int16_t>((frame.data[5] & 0xff) | (frame.data[4] << 8)) / 1000.0;
                double steering_angle = static_cast<int16_t>((frame.data[7] & 0xff) | (frame.data[6] << 8)) / 1000.0;
                if (steering_angle > 0) {
                    steering_angle *= left_angle_scale_;
                }
                else {
                    steering_angle *= right_angle_scale_;
                }
                publishOdometry(frame.stamp, linear_velocity, angular_velocity,
                                lateral_velocity, steering_angle);
                // RCLCPP_INFO(this->get_logger(),"MSG_MOTION_STATE_ID :");
                break;
            }
            case MSG_SYSTEM_STATE_ID: {
                uint8_t vehicle_state = frame.data[0];
                uint8_t control_mode = frame.data[1];
                double battervyoltage = ((frame.data[3] & 0xff) | (frame.data[2] << 8)) * 0.1;
                uint16_t error_code = ((frame.data[5] & 0xff) | (frame.data[4] << 8));

                motion_mode_ = frame.data[6];
                
                processErrorCode(error_code);
                publishLimoState(frame.stamp, vehicle_state, control_mode,
                                 battervyoltage, error_code, motion_mode_);
                // RCLCPP_INFO(this->get_logger(),"MSG_SYSTEM_STATE_ID :");
                break;
            }
            case MSG_ACTUATOR1_HS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR2_HS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR3_HS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR4_HS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR1_LS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR2_LS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR3_LS_STATE_ID: {
                break;
            }
            case MSG_ACTUATOR4_LS_STATE_ID: {
                break;
            }
            /****************** sensor frame *****************/
            case MSG_ODOMETRY_ID: {
                // int32_t left_wheel_odom = (frame.data[3] & 0xff) | (frame.data[2] << 8) |
                //                           (frame.data[1] << 16)  | (frame.data[0] << 24);
                // int32_t right_wheel_odom = (frame.data[7] & 0xff) | (frame.data[6] << 8) |
                //                            (frame.data[5] << 16)  | (frame.data[4] << 24);
                // // RCLCPP_INFO(this->get_logger(),"MSG_SYSTEM_STATE_ID :");

                break;
            }
            case MSG_IMU_ACCEL_ID: { // accelerate
                imu_data_.accel_x = static_cast<int16_t>((frame.data[1] & 0xff) | (frame.data[0] << 8)) / 100.0;
                imu_data_.accel_y = static_cast<int16_t>((frame.data[3] & 0xff) | (frame.data[2] << 8)) / 100.0;
                imu_data_.accel_z = static_cast<int16_t>((frame.data[5] & 0xff) | (frame.data[4] << 8)) / 100.0;
                // RCLCPP_INFO(this->get_logger(),"MSG_IMU_ACCEL_ID :");
                break;
            }
            case MSG_IMU_GYRO_ID: {
                imu_data_.gyro_x = degToRad(static_cast<int16_t>((frame.data[1] & 0xff) |
                                            (frame.data[0] << 8)) / 100.0);
                imu_data_.gyro_y = degToRad(static_cast<int16_t>((frame.data[3] & 0xff) |
                                            (frame.data[2] << 8)) / 100.0);
                imu_data_.gyro_z = degToRad(static_cast<int16_t>((frame.data[5] & 0xff) |
                                            (frame.data[4] << 8)) / 100.0);
                // RCLCPP_INFO(this->get_logger(),"MSG_IMU_GYRO_ID :");
                break;
            }
            case MSG_IMU_EULER_ID: {
                imu_data_.yaw = static_cast<int16_t>((frame.data[1] & 0xff) | (frame.data[0] << 8)) / 100.0;
                imu_data_.pitch = static_cast<int16_t>((frame.data[3] & 0xff) | (frame.data[2] << 8)) / 100.0;
                imu_data_.roll = static_cast<int16_t>((frame.data[5] & 0xff) | (frame.data[4] << 8)) / 100.0;
                publishIMUData(frame.stamp);
                // RCLCPP_INFO(this->get_logger(),"MSG_IMU_EULER_ID :");
                break;
            }
            default:
                break;
        }
    }

    void LimoAckermanDriver::enableCommandedMode() {
        AgileX::LimoFrame frame;
        frame.id = MSG_CTRL_MODE_CONFIG_ID;
        frame.data[0] = 0x01;
        frame.data[1] = 0;
        frame.data[2] = 0;
        frame.data[3] = 0;
        frame.data[4] = 0;
        frame.data[5] = 0;
        frame.data[6] = 0;
        frame.data[7] = 0;

        sendFrame(frame);
        RCLCPP_INFO(this->get_logger(),"enableCommandedMode :");
    }

    double LimoAckermanDriver::degToRad(double deg) {
        return deg / 180.0 * M_PI;
    }

    void LimoAckermanDriver::processErrorCode(uint16_t error_code) {
        if (error_code & 0x0001) {
            std::cout << "LIMO: Low battery!:" << std::endl;
        }
        if (error_code & 0x0002) {
            std::cout << "LIMO: Low battery!:" << std::endl;
        }
        if (error_code & 0x0004) {
            std::cout << "LIMO: Remote control lost connect!" << std::endl;
        }
        if (error_code & 0x0008) {
            std::cout << "LIMO: Motor driver 1 error!" << std::endl;
        }
        if (error_code & 0x0010) {
            std::cout << "LIMO: Motor driver 2 error!" << std::endl;
        }
        if (error_code & 0x0020) {
            std::cout << "LIMO: Motor driver 3 error!" << std::endl;
        }
        if (error_code & 0x0040) {
            std::cout << "LIMO: Motor driver 4 error!" << std::endl;
        }
        if (error_code & 0x0100) {
            std::cout << "LIMO: Drive status error!" << std::endl;
        }
    }

    void LimoAckermanDriver::publishIMUData(double stamp) {
        sensor_msgs::msg::Imu imu_msg;

        geometry_msgs::msg::TransformStamped t;  

        imu_msg.header.stamp = rclcpp::Time(RCL_S_TO_NS(stamp));
        imu_msg.header.frame_id = "imu_link";

        imu_msg.linear_acceleration.x = imu_data_.accel_x;
        imu_msg.linear_acceleration.y = imu_data_.accel_y;
        imu_msg.linear_acceleration.z = imu_data_.accel_z;

        imu_msg.angular_velocity.x = imu_data_.gyro_x;
        imu_msg.angular_velocity.y = imu_data_.gyro_y;
        imu_msg.angular_velocity.z = imu_data_.gyro_z;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, degToRad(imu_data_.yaw));

        if (flag==0)
        {
            double present_theta_ =imu_data_.yaw;
            double last_theta_ = imu_data_.yaw;
            flag=1;    

        }
        //ROS_INFO("flag:%d",flag);
        present_theta_ = imu_data_.yaw;
        delta_theta_ = present_theta_ - last_theta_;
        if(delta_theta_< 0.1 && delta_theta_> -0.1) delta_theta_=0;
        real_theta_ = real_theta_ + delta_theta_;
        last_theta_ = present_theta_;
        //ROS_INFO("present_theta_:%f;delta_theta_:%f;real_theta_:%f;last_theta_:%f",present_theta_,delta_theta_,real_theta_,last_theta_);

        imu_msg.orientation.x = q.x();
        imu_msg.orientation.y = q.y();
        imu_msg.orientation.z = q.z();
        imu_msg.orientation.w = q.w();

        imu_msg.linear_acceleration_covariance[0] = 1.0f;
        imu_msg.linear_acceleration_covariance[4] = 1.0f;
        imu_msg.linear_acceleration_covariance[8] = 1.0f;

        imu_msg.angular_velocity_covariance[0] = 1e-6;
        imu_msg.angular_velocity_covariance[4] = 1e-6;
        imu_msg.angular_velocity_covariance[8] = 1e-6;

        imu_msg.orientation_covariance[0] = 1e-6;
        imu_msg.orientation_covariance[4] = 1e-6;
        imu_msg.orientation_covariance[8] = 1e-6;

        imu_publisher_->publish(imu_msg);
    }

    void LimoAckermanDriver::publishLimoState(double stamp, uint8_t vehicle_state, uint8_t control_mode,
                                  double battery_voltage, uint16_t error_code, int8_t motion_mode) {

        limo_msgs::msg::LimoStatus status_msg;
        status_msg.header.stamp = rclcpp::Time(stamp);
        status_msg.vehicle_state = vehicle_state;
        status_msg.control_mode = control_mode;
        status_msg.battery_voltage = battery_voltage;
        status_msg.error_code = error_code;
        status_msg.motion_mode = motion_mode;

        status_publisher_->publish(status_msg);
    }

    void LimoAckermanDriver::publishOdometry(double stamp, double linear_velocity,
                                 double angular_velocity, double lateral_velocity,
                                 double steering_angle) {
        steering_angle /= 2;
        static double last_stamp = stamp;
        double dt = stamp - last_stamp;
        last_stamp = stamp;
        double omega = 0;

        switch (motion_mode_) {
            case AgileX::MODE_ACKERMANN: {
                double R = (steering_angle == 0) ? std::numeric_limits<double>::infinity() : wheelbase_ / tan(steering_angle);
                // RCLCPP_WARN(this->get_logger(),"steering_angle %f", steering_angle);
                omega = (R == std::numeric_limits<double>::infinity()) ? 0 : linear_velocity / R;
                break;
            }
            default:{
                RCLCPP_WARN(this->get_logger(),"Invalid motion mode you should change limo to ackermann mode!");
                return;
                break;
            }
        }
        theta_ += omega * dt;
        position_x_ += linear_velocity * cos(theta_) * dt;
        position_y_ += linear_velocity * sin(theta_) * dt;

        tf2::Quaternion limo_yaw;
        limo_yaw.setRPY(0,0,theta_);
        geometry_msgs::msg::Quaternion odom_quat = tf2::toMsg(limo_yaw);
    
        if (pub_odom_tf_) {
            rclcpp::Time now = this->get_clock()->now();
            geometry_msgs::msg::TransformStamped tf_msg;

            tf_msg.header.stamp = now;
            tf_msg.header.frame_id = odom_frame_;
            tf_msg.child_frame_id = base_frame_;

            tf_msg.transform.translation.x = position_x_;
            tf_msg.transform.translation.y = position_y_;
            tf_msg.transform.translation.z = 0.0;
            tf_msg.transform.rotation = odom_quat;
            tf_broadcaster_->sendTransform(tf_msg);
        }

        rclcpp::Time now = this->get_clock()->now();
        nav_msgs::msg::Odometry odom_msg;

        odom_msg.header.stamp = now;
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.child_frame_id = base_frame_;

        odom_msg.pose.pose.position.x = position_x_;
        odom_msg.pose.pose.position.y = position_y_;
        odom_msg.pose.pose.position.z = 0.0;
        odom_msg.pose.pose.orientation = odom_quat;

        odom_msg.twist.twist.linear.x = linear_velocity;
        odom_msg.twist.twist.linear.y = 0.0;
        odom_msg.twist.twist.angular.z = omega;

        odom_msg.pose.covariance[0] = 0.1;
        odom_msg.pose.covariance[7] = 0.1;
        odom_msg.pose.covariance[14] = 0.1;
        odom_msg.pose.covariance[21] = 1.0;
        odom_msg.pose.covariance[28] = 1.0;
        odom_msg.pose.covariance[35] = 1.0;

        odom_publisher_->publish(odom_msg);
    }
}


