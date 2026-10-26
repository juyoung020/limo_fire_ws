# basic
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# module for navigation
from nav2_msgs.action import NavigateToPose
from limo_ros2_application.nav_utils import NavPose, NavigateStatus

class PatrolLimo(Node):
    def __init__(self):
        super().__init__('patrol_limo')

        # set action client
        self.action_client_ = ActionClient(
            self,
            NavigateToPose, 
            'navigate_to_pose')
        
        # set the patrol point
        pos_1 = NavPose()
        pos_2 = NavPose()
        pos_3 = NavPose()
        pos_4 = NavPose()
        pos_5 = NavPose()
        pos_6 = NavPose()
        pos_1.set_pose(0.885, -0.205, -1.57) # x, y, theta
        pos_2.set_pose(0.855, -2.432, -3.14) # x, y, theta
        pos_3.set_pose(-0.112, -2.438, 1.57)
        pos_4.set_pose(-0.115, -1.042, -3.14)
        pos_5.set_pose(-1.394114, -1.0083698, 1.57)
        pos_6.set_pose(-1.2995, -0.1265, 0.0)

        # set the goal list
        self.goal_list = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]
        self.index = 0

        # set timer, evry 1 sec run callback
        self.timer = self.create_timer(2.0, self.timer_callback)

        # to check navigation status first set default
        self.limo_state = NavigateStatus(0)

    def timer_callback(self):
        # send goal when the limo is not navigating
        if not self.limo_state == NavigateStatus.ACTIVE:
            self.send_goal()
            
    def send_goal(self):
        # send goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose = self.goal_list[self.index].get_pose()
        self.action_client_.wait_for_server()
        self._send_goal_future = self.action_client_.send_goal_async(goal_msg)
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
        
        # if accepted then update limo state and goal index
        self.limo_state = NavigateStatus.ACTIVE
        self.index = (self.index + 1) % len(self.goal_list)
    
    def get_result_callback(self, future):
        # after reach goal and get result than update limo state
        self.limo_state = NavigateStatus.GOAL
        
def main(args=None):
    rclpy.init(args=args)
    patrol_limo = PatrolLimo()
    
    rclpy.spin(patrol_limo)

    patrol_limo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
