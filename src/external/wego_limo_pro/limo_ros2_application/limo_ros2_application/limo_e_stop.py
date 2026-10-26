import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool 
from sensor_msgs.msg import LaserScan
from math import pi, cos, sin

class LimoEStop(Node):
    def __init__(self):
        super().__init__('limo_e_stop')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            rclpy.qos.qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning

        self.publisher_ = self.create_publisher(Bool, 'e_stop', 10)
        
    def laser_callback(self, msg):
        estop = Bool()
        estop.data = False

        for i, data in enumerate(msg.ranges):
            current_angle = msg.angle_min + msg.angle_increment*i
            cx = data * cos(current_angle)
            cy = data * sin(current_angle)
            if 0.01 < cx <0.2 and -0.1 < cy < 0.1:
                estop.data = True
                break

        self.publisher_.publish(estop)

def main(args=None):
    rclpy.init(args=args)

    limo_e_stop = LimoEStop()
    rclpy.spin(limo_e_stop)

    limo_e_stop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()