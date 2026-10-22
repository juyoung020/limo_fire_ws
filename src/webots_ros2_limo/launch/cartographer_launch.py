#!/usr/bin/env python

# Cartographer SLAM + RViz for the Webots AgileX LIMO.
#
# Replaces `turtlebot3_cartographer/cartographer.launch.py` so that this
# package depends on plain `cartographer_ros` instead of a robot-specific
# wrapper. Start it in a second terminal once `robot_launch.py` is up, or let
# `robot_launch.py slam:=true` include it for you.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_limo')
    default_config_dir = os.path.join(package_dir, 'resource')
    default_rviz = os.path.join(package_dir, 'rviz', 'limo_cartographer.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    cartographer_config_dir = LaunchConfiguration(
        'cartographer_config_dir', default=default_config_dir)
    configuration_basename = LaunchConfiguration(
        'configuration_basename', default='limo_lds_2d.lua')
    resolution = LaunchConfiguration('resolution', default='0.05')
    publish_period_sec = LaunchConfiguration('publish_period_sec', default='1.0')
    rviz_config = LaunchConfiguration('rviz_config', default=default_rviz)

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use the Webots /clock as ROS 2 time'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Open RViz with the SLAM view'),
        DeclareLaunchArgument(
            'cartographer_config_dir', default_value=default_config_dir,
            description='Directory holding the cartographer lua configuration'),
        DeclareLaunchArgument(
            'configuration_basename', default_value='limo_lds_2d.lua',
            description='Name of the cartographer lua configuration file'),
        DeclareLaunchArgument(
            'resolution', default_value='0.05',
            description='Grid cell size of the published occupancy grid'),
        DeclareLaunchArgument(
            'publish_period_sec', default_value='1.0',
            description='OccupancyGrid publishing period'),
        DeclareLaunchArgument(
            'rviz_config', default_value=default_rviz,
            description='RViz configuration file'),

        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-configuration_directory', cartographer_config_dir,
                       '-configuration_basename', configuration_basename]),

        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-resolution', resolution,
                       '-publish_period_sec', publish_period_sec]),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(use_rviz)),
    ])
