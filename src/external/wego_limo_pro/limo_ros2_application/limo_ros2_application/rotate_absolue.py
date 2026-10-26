import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from limo_ros2_application.nav_utils import euler_from_quaternion, normalize_angle

class RotateAbsolute(Node):
    def __init__(self):
        super().__init__('rotate_absolute')

        self.current_yaw_ = 0.0
        self.goal_yaw_ = 999.0

        # set subscription
        self.subscription = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10)
        self.subscription  # prevent unused variable warning

        # set publisher
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    
    def set_goal(self, goal):
        self.goal_yaw_ = goal

    def imu_callback(self, msg):
        (roll, pitch, yaw) = euler_from_quaternion(msg.orientation.x,
                                                   msg.orientation.y,
                                                   msg.orientation.z, 
                                                   msg.orientation.w)
        self.current_yaw_ = yaw
        self.get_logger().info('roll: %f, pitch: %f, yaw: %f' % (roll, pitch, yaw))

    def timer_callback(self):
        cmd = Twist()
        
        gap = self.goal_yaw_ - self.current_yaw_
        gap = normalize_angle(gap)

        if abs(gap) < 0.05:
            self.get_logger().info('Goal Reached: ')
            rclpy.shutdown()
            self.destroy_node()
        
        if gap > 0:
            cmd.angular.z = 0.3
        else:
            cmd.angular.z = -0.3
        
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)

    rotate_absolute = RotateAbsolute()

    rotate_absolute.get_logger().info('Please set goal(-3.14 < x < 3.14): ')
    goal=float(input())

    while not (-3.141592< goal <3.141592):
        rotate_absolute.get_logger().info('Wrong input please rewrite(-3.14 < x < 3.14): ')
        goal=float(input())
    
    rotate_absolute.set_goal(goal)
    rclpy.spin(rotate_absolute)

if __name__ == '__main__':
    main()