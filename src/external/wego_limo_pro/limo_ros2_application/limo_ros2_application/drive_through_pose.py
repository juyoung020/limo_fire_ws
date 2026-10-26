# basic
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# module for navigation
from nav2_msgs.action import NavigateThroughPoses
from limo_ros2_application.nav_utils import NavPose, NavigateStatus
from geometry_msgs.msg import PoseStamped

# for math
import numpy as np

class DriveThroughPose(Node):
    def __init__(self):
        super().__init__('drive_through_pose')

        # set action client
        self.action_client_ = ActionClient(
            self,
            NavigateThroughPoses,
            'navigate_through_poses')

        # set waypoint
        self.goals_msg = NavigateThroughPoses.Goal()
        self.make_points() 
        
    def send_goal(self):
        # send goal
        self.action_client_.wait_for_server()
        self._send_goal_future = self.action_client_.send_goal_async(self.goals_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        # get respownse if the request was accepted
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
        
        
    def get_result_callback(self, future):
        # when the navigation done just shutdown
        rclpy.shutdown()

    def make_points(self):
        # set 1st waypoint
        tmp_point_ = NavPose()
        tmp_point_.set_pose(0.0, 0.0, 0.0)
        waypoint_ = PoseStamped()
        waypoint_.header.frame_id = 'map'
        waypoint_.header.stamp = self.get_clock().now().to_msg()
        waypoint_.pose = tmp_point_.get_pose()
        self.goals_msg.poses.append(waypoint_)
    
        # set 2nd waypoint
        tmp_point_.set_relative(2.0, 0.0, 0.0)
        waypoint_ = PoseStamped()
        waypoint_.header.frame_id = 'map'
        waypoint_.header.stamp = self.get_clock().now().to_msg()
        waypoint_.pose = tmp_point_.get_pose()
        self.goals_msg.poses.append(waypoint_)
    
        # set 3rd waypoint
        tmp_point_.set_relative(0.0, 1.0, np.pi/2)
        waypoint_ = PoseStamped()
        waypoint_.header.frame_id = 'map'
        waypoint_.header.stamp = self.get_clock().now().to_msg()
        waypoint_.pose = tmp_point_.get_pose()
        self.goals_msg.poses.append(waypoint_)
    
        # set 4th waypoint
        tmp_point_.set_relative(1.0, 0.0, -np.pi/2)
        waypoint_ = PoseStamped()
        waypoint_.header.frame_id = 'map'
        waypoint_.header.stamp = self.get_clock().now().to_msg()
        waypoint_.pose = tmp_point_.get_pose()
        self.goals_msg.poses.append(waypoint_)
    
def main(args=None):
    rclpy.init(args=args)
    drive_through_pose = DriveThroughPose()
    
    drive_through_pose.send_goal()
    
    rclpy.spin(drive_through_pose)

    drive_through_pose.destroy_node()

if __name__ == '__main__':
    main()