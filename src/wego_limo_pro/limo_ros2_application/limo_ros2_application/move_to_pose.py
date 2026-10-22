import rclpy
from rclpy.node import Node

# for limo control
from geometry_msgs.msg import Twist

# for tf listener
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# for some calculation
import numpy as np
import math

# for visulaization
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class MoveToPose(Node):
    def __init__(self):
        super().__init__('move_to_pose')

        # set the TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # set gain for distnace and orinentation contorl
        self.declare_parameter('Kp_rho', 0.2) # distance to goal
        self.declare_parameter('Kp_alpha', 1.7) # heading to goal
        self.declare_parameter('Kp_beta', 1.0) # the offset between goal orientation and current
        self.Kp_rho = self.get_parameter('Kp_rho')
        self.Kp_alpha = self.get_parameter('Kp_alpha')
        self.Kp_beta = self.get_parameter('Kp_beta')

        # set goal tolerance
        self.declare_parameter('xy_tolerance', 0.05) # distance tolerance
        self.declare_parameter('yaw_tolerance', 0.15) # orientation tolerance
        self.xy_tolerance = self.get_parameter('xy_tolerance')
        self.yaw_tolerance = self.get_parameter('yaw_tolerance')

        # set absolute max velocity
        self.declare_parameter('max_linear', 0.6) # max linear velocity
        self.declare_parameter('max_angular', 1.0) # max anglur velocity
        self.max_linear = self.get_parameter('max_linear')
        self.max_angular = self.get_parameter('max_angular')

        # set publisher for control
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_limo)

        # flag for goal reach
        self.goal_reached = False

        # to visualize the path
        self.path_pub = self.create_publisher(Path, 'traj', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            'odometry/filtered',
            self.odom_callback,
            10)

        self.odom_sub # prevent from warning

        # set initial value for path
        self.path = Path()
        self.path.header.frame_id = 'odom'

    def control_limo(self):
        # listen the TF
        try:
            t_ = self.tf_buffer.lookup_transform(
                'base_link',
                'goal_pose',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException as ex:
            print(ex)
            return
        
        # calculate input variable
        x_diff_ = t_.transform.translation.x
        y_diff_ = t_.transform.translation.y

        yaw_ = self.get_yaw_from_quaternion(t_.transform.rotation)
        yaw_diff_ = self.normalize_angle(yaw_) # rotation diff between two frame

        rho_ = np.hypot(y_diff_, x_diff_) #calculate distance
        alpha_ =  self.normalize_angle(np.arctan2(y_diff_, x_diff_)) #calculate heading
        beta_ = self.normalize_angle(yaw_diff_ - alpha_) # error of heading and rattation diff
        
        # check if goal reached the goal
        if rho_ < self.xy_tolerance.value and abs(yaw_diff_) < self.yaw_tolerance.value:
            self.goal_reached = True

            # if goal reach publish path
            self.path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path)
            return

        # calculate linear velocity and angular velocity
        v_ = self.Kp_rho.value * rho_

        # calculate angular velocity
        if rho_ > self.xy_tolerance.value:    
            w_ = self.Kp_alpha.value * alpha_ - self.Kp_beta.value * beta_
        else: # if distance is reached than just align the orientation
            w_ = self.Kp_beta.value * self.normalize_angle(yaw_diff_)

        # is goal on back side? than velocity is minus
        if alpha_ > np.pi / 2 or alpha_ < -np.pi / 2:
            v_ = -v_

        # set the ceiling and floor for control data
        msg_ =Twist()
        msg_.linear.x = min(max(v_, -self.max_linear.value), self.max_linear.value) 
        msg_.angular.z = min(max(w_, -self.max_angular.value), self.max_angular.value)

        # publish command velocity
        self.cmd_pub.publish(msg_)
    
    # calculate yaw from quaternion value
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    # set angle to -pi ~ pi
    def normalize_angle(slef, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # get the odometry and append position to path
    def odom_callback(self, msg):
        if not self.goal_reached:
            point_ = PoseStamped()
            point_.header = msg.header
            point_.pose = msg.pose.pose
            self.path.poses.append(point_)

def main(args=None):
    rclpy.init(args=args)
    move_to_pose = MoveToPose()

    rclpy.spin(move_to_pose)

    move_to_pose.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()