import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Bool


class LimoControl(Node):

    def __init__(self):
        super().__init__('limo_control')
        
        # Setting for publisher of cmd velocity
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # Setting for subscriber of e_stop and lane_detection and slow down
        self.e_stop_subscription = self.create_subscription(
            Bool,
            'e_stop',
            self.e_stop_callback,
            10)
        self.distance_subscription = self.create_subscription(
            Int32,
            'distance_y',
            self.distance_callback,
            10)
        self.slow_down_subscription = self.create_subscription(
            Bool,
            'slow_down',
            self.slow_down_callback,
            10)
        
        # prevent from warning
        self.e_stop_subscription
        self.distance_subscription
        self.slow_down_subscription
        
        # flag and input value of twisting
        self.e_stop_flag = False
        self.gap = 0
        self.slow_down_flag = False

        # parameter for default speed, p_gain for twist, slow down ratio
        self.declare_parameter('default_speed', 0.2)
        self.declare_parameter('p_gain', 0.007)
        self.declare_parameter('slow_down_ratio', 0.5)

        self.default_speed =self.get_parameter('default_speed')
        self.p_gain =self.get_parameter('p_gain')
        self.slow_down_ratio =self.get_parameter('slow_down_ratio')

    def e_stop_callback(self, msg):
        self.e_stop_flag = msg.data
    
    def distance_callback(self, msg):
        self.gap = msg.data

    def slow_down_callback(self, msg):
        self.slow_down_flag = msg.data

    def timer_callback(self):
        # set the limo speed
        msg = Twist()
        msg.linear.x = self.default_speed.value
        msg.angular.z = self.gap * self.p_gain.value

        # if e_stop called
        if self.e_stop_flag :
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        
        # if slow down called
        if self.slow_down_flag:
            msg.linear.x = msg.linear.x * self.slow_down_ratio.value
        
        self.publisher_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)

    limo_control = LimoControl()

    rclpy.spin(limo_control)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    limo_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()