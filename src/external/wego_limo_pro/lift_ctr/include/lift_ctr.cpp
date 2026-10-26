#include "lift_ctr/lift_ctr.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

LiftControl::LiftControl()
: Node("lift_ctr")
{
    //set publisher and subscriber
    lift_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
        "lift_cmd", 10, std::bind(&LiftControl::lift_callback, this, _1));
    
    lift_pub_ = this->create_publisher<std_msgs::msg::String>("lift_status", 10);
    
    timer_ = this->create_wall_timer(
        500ms, std::bind(&LiftControl::timer_callback, this));
    
    //set serial port parameter
    this->declare_parameter<std::string>("port", "/dev/ttyUSB0");
    this->declare_parameter<int>("baud_rate", 115200);

    std::string port;
    int baud_rate;
    this->get_parameter("port", port);
    this->get_parameter("baud_rate", baud_rate);

    //open the serial and set
    try {
        serial_.open(port);
        serial_.set_option(boost::asio::serial_port_base::baud_rate(baud_rate));
        serial_.set_option(boost::asio::serial_port_base::character_size(8));
        serial_.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one));
        serial_.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none));
        serial_.set_option(boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none));
    } catch (const boost::system::system_error& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open serial port: %s", e.what());
    }
}

void LiftControl::lift_callback(const std_msgs::msg::UInt8 & msg)
{
    // set data
    auto input_num = msg.data;
    std::string data;
    if(input_num == 1){
        data = "#UP";
    }else if(input_num == 2){
        data = "#DOWN";
    }else{
        data = "#HOLD";
    }

    // send to serial port
    try {
        boost::asio::write(serial_, boost::asio::buffer(data));
    } catch (const boost::system::system_error& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to write to serial port: %s", e.what());
    }
}

void LiftControl::timer_callback()
{
    // read serial until line end
    boost::asio::streambuf buf;
    boost::asio::read_until(serial_, buf, "\n");
    std::istream is(&buf);
    std::string line;
    std::getline(is, line);

    // publish status
    std_msgs::msg::String msg;
    msg.data = line;
    lift_pub_->publish(msg);
}