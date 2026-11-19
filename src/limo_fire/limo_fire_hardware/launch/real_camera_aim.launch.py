"""RGB-D mapping -> fresh fire observation -> preview/approach/aim.

Launch robot.launch.xml and navigation.launch.xml separately first. This launch
never publishes a synthetic target and defaults to motion/serial disabled.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    hardware = get_package_share_directory('limo_fire_hardware')
    perception = get_package_share_directory('limo_fire_perception')
    juyoung_launch = os.path.join(perception, 'launch', 'perception.launch.xml')
    juyoung_pipeline = os.path.exists(juyoung_launch)
    perception_launch = juyoung_launch if juyoung_pipeline else os.path.join(
        perception, 'launch', 'yolo.launch.xml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(hardware, 'config', 'approach_aim.yaml')),
        DeclareLaunchArgument('motion_enabled', default_value='false'),
        DeclareLaunchArgument('serial_enabled', default_value='false'),
        DeclareLaunchArgument('serial_port', default_value=''),
        DeclareLaunchArgument('firing_enabled', default_value='false'),
        DeclareLaunchArgument('auto_fire_enabled', default_value='false'),
        DeclareLaunchArgument('fire_on_pitch_ack', default_value='false'),
        DeclareLaunchArgument('yaw_alignment_mode', default_value='nav2'),
        DeclareLaunchArgument('yaw_rate_drive_mode_verified', default_value='false'),
        DeclareLaunchArgument('target_id', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # Scoped: the perception 'config' (a YOLO file name) must not overwrite this
        # launch's 'config' (the measured aim YAML) for the include below.
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            AnyLaunchDescriptionSource(perception_launch),
            launch_arguments={
                'config': 'yolo_fire.yaml', 'mapper': 'true', 'rviz': 'false',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                **({'objects_file': ''} if juyoung_pipeline else {}),
            }.items())]),
        *([Node(package='limo_fire_hardware', executable='mapper_observation_bridge',
                name='mapper_observation_bridge', output='screen', parameters=[{
                    'target_id': ParameterValue(LaunchConfiguration('target_id'), value_type=str),
                    'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                }])] if juyoung_pipeline else []),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(hardware, 'launch', 'approach_aim.launch.py')),
            launch_arguments={name: LaunchConfiguration(name) for name in (
                'config', 'motion_enabled', 'serial_enabled', 'serial_port',
                'firing_enabled', 'auto_fire_enabled', 'fire_on_pitch_ack',
                'yaw_alignment_mode', 'yaw_rate_drive_mode_verified', 'use_sim_time')}.items()),
    ])
