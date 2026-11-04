"""Run a map-aligned MuJoCo viewer and its ROS target/pose bridge."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    home = EnvironmentVariable('HOME')
    root = LaunchConfiguration('mujoco_root')
    port = LaunchConfiguration('port')
    map_yaml = LaunchConfiguration('map_yaml')
    python = LaunchConfiguration('python')

    viewer = ExecuteProcess(
        cmd=[
            python,
            PathJoinSubstitution([root, 'launch.py']),
            '--map-yaml', map_yaml,
            '--pose-sync-port', port,
            '--aim', 'surface',
        ],
        cwd=root,
        additional_env={'PYTHONPATH': '', 'PYTHONHOME': ''},
        output='screen',
    )
    bridge = Node(
        package='limo_fire_mujoco',
        executable='pose_udp',
        name='limo_mujoco_pose_sync',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'udp_port': port,
            'target_frame': 'map',
            'fire_namespace': 'fire',
            'markers_topic': '/object_mapper/markers',
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'mujoco_root', default_value=PathJoinSubstitution([home, 'robot-programming/simulation/mujoco_ball_mvp'])),
        DeclareLaunchArgument(
            'python', default_value=PathJoinSubstitution([home, 'robot-programming/simulation/mujoco_ball_mvp/.venv/bin/python'])),
        DeclareLaunchArgument('map_yaml'),
        DeclareLaunchArgument('port', default_value='8765'),
        bridge,
        viewer,
    ])
