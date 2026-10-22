#!/usr/bin/env python

# Nav2 + AMCL + RViz for the Webots AgileX LIMO.
#
# Wraps `nav2_bringup/bringup_launch.py` with this package's map, Nav2
# parameters and RViz view, so navigation starts with a single command.
#
# This deliberately does not use `turtlebot3_navigation2/navigation2.launch.py`:
# that file is the same bringup_launch.py plus an RViz node you cannot switch
# off, and it reads os.environ['TURTLEBOT3_MODEL'] at import time, raising
# KeyError when the variable is unset -- even though params_file overrides
# everything the variable would select.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_limo')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    default_map = os.path.join(package_dir, 'resource', 'limo_world_map.yaml')
    # resource/diff/nav2_params.yaml or resource/mecanum/nav2_params.yaml
    default_params = PathJoinSubstitution(
        [package_dir, 'resource', LaunchConfiguration('drive'), 'nav2_params.yaml'])
    default_rviz = os.path.join(nav2_bringup_dir, 'rviz',
                                'nav2_default_view.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    nav2_map = LaunchConfiguration('map', default=default_map)
    params_file = LaunchConfiguration('params_file', default=default_params)
    rviz_config = LaunchConfiguration('rviz_config', default=default_rviz)

    return LaunchDescription([
        DeclareLaunchArgument(
            'drive', default_value='diff', choices=['diff', 'mecanum'],
            description='Drive type of the LIMO; selects the default params_file'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use the Webots /clock as ROS 2 time'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Open RViz with the Nav2 view'),
        DeclareLaunchArgument(
            'map', default_value=default_map,
            description='Map yaml for AMCL to localise against'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Nav2 parameter file'),
        DeclareLaunchArgument(
            'rviz_config', default_value=default_rviz,
            description='RViz configuration file'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')),
            launch_arguments=[
                ('map', nav2_map),
                ('params_file', params_file),
                ('use_sim_time', use_sim_time),
                # Pin bringup's own 'slam' argument. Launch configurations
                # propagate into included launch files, so robot_launch.py's
                # `slam:=false` would otherwise land here -- and bringup
                # evaluates it as a Python literal (`not false` -> NameError).
                ('slam', 'False'),
            ]),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(use_rviz)),
    ])
