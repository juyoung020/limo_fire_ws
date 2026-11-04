from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('port', default_value='8765'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('markers_topic', default_value='/object_mapper/markers'),
        DeclareLaunchArgument('target_topic', default_value='/fire/detected_target_pose'),
        DeclareLaunchArgument('targets_topic', default_value='/fire/detected_targets'),
        DeclareLaunchArgument('target_frame', default_value='map'),
        DeclareLaunchArgument('fire_namespace', default_value='fire'),
        DeclareLaunchArgument('target_timeout', default_value='0.75'),
    ]
    bridge = Node(
        package='limo_fire_mujoco',
        executable='pose_udp',
        name='limo_mujoco_pose_sync',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'udp_port': LaunchConfiguration('port'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'markers_topic': LaunchConfiguration('markers_topic'),
                'target_topic': LaunchConfiguration('target_topic'),
                'targets_topic': LaunchConfiguration('targets_topic'),
                'target_frame': LaunchConfiguration('target_frame'),
                'fire_namespace': LaunchConfiguration('fire_namespace'),
                'target_timeout': LaunchConfiguration('target_timeout'),
            }
        ],
    )
    return LaunchDescription(arguments + [bridge])
