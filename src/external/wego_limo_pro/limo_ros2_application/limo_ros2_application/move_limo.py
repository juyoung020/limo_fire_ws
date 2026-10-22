import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from math import pi, sin


class MoveLimo(Node):

    def __init__(self):
        super().__init__('move_limo')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = sin(2*pi*self.i/2)
        self.publisher_.publish(msg)
        self.i += self.timer_period


def main(args=None):
    rclpy.init(args=args)

    move_limo = MoveLimo()

    rclpy.spin(move_limo)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    move_limo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()