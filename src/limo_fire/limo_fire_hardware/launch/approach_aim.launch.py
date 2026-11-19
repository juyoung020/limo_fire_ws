"""Real approach/aim controller. Robot, localization and perception run separately."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('limo_fire_hardware'), 'config', 'approach_aim.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config),
        DeclareLaunchArgument('motion_enabled', default_value='false'),
        DeclareLaunchArgument('serial_enabled', default_value='false'),
        DeclareLaunchArgument('serial_port', default_value=''),
        DeclareLaunchArgument('yaw_alignment_mode', default_value='nav2'),
        DeclareLaunchArgument('yaw_rate_drive_mode_verified', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='limo_fire_hardware', executable='approach_aim', name='launcher_approach_aim',
             output='screen', parameters=[LaunchConfiguration('config'), {
                 'motion_enabled': ParameterValue(LaunchConfiguration('motion_enabled'), value_type=bool),
                 'yaw_alignment_mode': ParameterValue(LaunchConfiguration('yaw_alignment_mode'), value_type=str),
                 'yaw_rate_drive_mode_verified': ParameterValue(
                     LaunchConfiguration('yaw_rate_drive_mode_verified'), value_type=bool),
                 'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
             }]),
        Node(package='limo_fire_hardware', executable='pitch_serial_bridge', name='pitch_serial_bridge',
             output='screen', parameters=[LaunchConfiguration('config'), {
                 'enabled': ParameterValue(LaunchConfiguration('serial_enabled'), value_type=bool),
                 'serial_port': ParameterValue(LaunchConfiguration('serial_port'), value_type=str),
                 'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
             }]),
    ])
