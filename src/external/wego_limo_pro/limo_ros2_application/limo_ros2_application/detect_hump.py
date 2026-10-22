import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool 
from sensor_msgs.msg import Imu
from builtin_interfaces.msg import Duration

import time

class DetectHump(Node):
    def __init__(self):
        super().__init__('detect_hump')
        self.subscription = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10)
        self.subscription  # prevent unused variable warning

        # parameter for time duration to slow down & thresh hold for pithing velocity 
        self.declare_parameter('hump_threshold', -0.2)
        self.declare_parameter('slow_down_duration', 2)

        self.threshold=self.get_parameter('hump_threshold')
        self.slow_down_duration = Duration()
        self.slow_down_duration.sec = self.get_parameter('slow_down_duration').value

        # flag for the slowing down
        self.publisher_ = self.create_publisher(Bool, 'slow_down', 10)

        self.slow_down_flag = Bool()
        self.slow_down_flag.data = False

        # if i set past time initial value as -2 * slow down duration 
        # the initial hump detection will be always true
        self.past_time = -2 * self.slow_down_duration.sec
    
    def imu_callback(self, msg):
        current_time = time.time() 
        # when the time duration is bigger than the slow down duration
        if int(current_time - self.past_time) > self.slow_down_duration.sec:
            #for now it is not hump 
            self.slow_down_flag.data = False

            # when imu anglular velocity y get over threshold value
            if msg.angular_velocity.y < self.threshold.value:
                self.past_time = current_time # update the time => the hump is starting
                self.slow_down_flag.data = True
        else: # when it is under the slowing down duration condition
            self.slow_down_flag.data = True
        
        self.publisher_.publish(self.slow_down_flag)

def main(args=None):
    rclpy.init(args=args)

    detect_hump = DetectHump()

    rclpy.spin(detect_hump)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detect_hump.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()