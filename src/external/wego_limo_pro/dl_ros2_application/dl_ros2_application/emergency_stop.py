import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math

class EmergencyStop(Node):
    def __init__(self):
        super().__init__('emergency_stop')
        self.sub_ = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            rclpy.qos.qos_profile_sensor_data)
        
        self.pub_ = self.create_publisher(Bool, 'emergency_stop', 10)

    def laser_callback(self, msg):
        emergency_stop_flag = Bool()
        emergency_stop_flag.data = False
        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            if 0.01 < x < 0.2 and -0.1 < y < 0.1:
                emergency_stop_flag.data = True
                break
        
        self.pub_.publish(emergency_stop_flag)

def main(args=None):
    rclpy.init(args=args)
    node = EmergencyStop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
