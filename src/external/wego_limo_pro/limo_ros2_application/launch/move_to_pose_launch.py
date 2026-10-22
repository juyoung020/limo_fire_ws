from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    limo_application_share_dir = get_package_share_directory('limo_ros2_application')
    rviz_config_path = os.path.join(limo_application_share_dir, 'rviz', 'move_to_pose.rviz')
    parameters_file_dir = os.path.join(limo_application_share_dir, 'params')
    move_to_pose_parameters = os.path.join(parameters_file_dir, 'move_to_pose.yaml')

    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['--x', '1.0',
                        '--y', '1.0',
                        '--z', '0',
                        '--yaw', '3.14',
                        '--pitch', '0',
                        '--roll', '0',
                        '--frame-id', 'odom',
                        '--child-frame-id', 'goal_pose']
        ),

        Node(
            package='limo_ros2_application',
            executable='move_to_pose',
            parameters=[move_to_pose_parameters]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        )
    ])