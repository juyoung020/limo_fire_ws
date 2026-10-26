import launch

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import DeclareLaunchArgument

from launch.substitutions import LaunchConfiguration

def generate_launch_description():
     degree = LaunchConfiguration('degree')

     degree_launch_arg = DeclareLaunchArgument(
          'degree',
          default_value='0.0'
     )

     return launch.LaunchDescription([
          degree_launch_arg,
          Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='base_link_to_camera_mount',
               arguments = ['--x', '0.2', 
                            '--y', '0.1',
                            '--z', '0.06', 
                            '--yaw', '0', 
                            '--pitch', '0', 
                            '--roll', '0', 
                            '--frame-id', 'base_link', 
                            '--child-frame-id', 'camera_mount']
          ),
          Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='camera_mount_to_roate',
               arguments = ['--x', '0.00', 
                            '--y', '0.00',
                            '--z', '0.00', 
                            '--yaw', '0', 
                            '--pitch', degree, 
                            '--roll', '0', 
                            '--frame-id', 'camera_mount', 
                            '--child-frame-id', 'camera_rotate']
          ),
          Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='rotate_to_camera_link',
               arguments = ['--x', '0.03', 
                            '--y', '-0.05',
                            '--z', '0.00', 
                            '--yaw', '0', 
                            '--pitch', '0', 
                            '--roll', '0', 
                            '--frame-id', 'camera_rotate', 
                            '--child-frame-id', 'camera_link']
          )
     ]) 