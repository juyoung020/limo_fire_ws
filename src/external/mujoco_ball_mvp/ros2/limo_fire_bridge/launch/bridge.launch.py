from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mujoco_root', description='Path to simulation/mujoco_ball_mvp'),
        DeclareLaunchArgument('aim', default_value='surface'),
        DeclareLaunchArgument('speed', default_value='5.5'),
        Node(package='limo_fire_bridge', executable='fire_bridge', output='screen', parameters=[{
            'mujoco_root': LaunchConfiguration('mujoco_root'), 'aim': LaunchConfiguration('aim'),
            'speed': LaunchConfiguration('speed'), 'use_sim_time': True,
        }]),
    ])
