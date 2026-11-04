from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port',default_value='8765'),
        DeclareLaunchArgument('odom_topic',default_value='/odom'),
        DeclareLaunchArgument('target_topic',default_value='/fire/detected_target_pose'),
        DeclareLaunchArgument('targets_topic',default_value='/fire/detected_targets'),
        Node(package='limo_fire_bridge',executable='fire_detector',output='screen',parameters=[{
            'use_sim_time':True,'target_topic':LaunchConfiguration('target_topic'),
            'targets_topic':LaunchConfiguration('targets_topic')
        }]),
        Node(package='limo_fire_bridge',executable='pose_udp',output='screen',parameters=[{
            'udp_port':LaunchConfiguration('port'),'odom_topic':LaunchConfiguration('odom_topic'),
            'target_topic':LaunchConfiguration('target_topic'),
            'targets_topic':LaunchConfiguration('targets_topic'),'use_sim_time':True
        }]),
    ])
