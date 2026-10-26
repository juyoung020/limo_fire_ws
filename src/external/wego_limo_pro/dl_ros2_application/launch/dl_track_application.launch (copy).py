import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share_dir = get_package_share_directory('dl_ros2_application')
    params_file_path = os.path.join(pkg_share_dir, 'params', 'params.yaml')
    track_weight_file_path = os.path.join(pkg_share_dir, 'weight', 'dl_drive.engine')
    object_weight_file_path = os.path.join(pkg_share_dir, 'weight', 'yolo11n.engine')

    return LaunchDescription([
        Node(
            package='dl_ros2_application',
            executable='emergency_stop',
            name='emergency_stop_node',
            output='screen',
        ),

        # Node(
        #     package='dl_ros2_application',
        #     executable='tf_track_detect',
        #     name='tf_track_detect_node',
        #     output='screen',
        #     parameters=[
        #         {'model_weight_path': track_weight_file_path}
        #     ]
        # ),

        Node(
            package='dl_ros2_application',
            executable='trt_track_detect',
            name='trt_track_detect_node',
            output='screen',
            parameters=[
                {'engine_file_path': track_weight_file_path}
            ]
        ),

        Node(
            package='dl_ros2_application',
            executable='yolo_object_detect',
            name='yolo_object_detect_node',
            output='screen',
            parameters=[
                {'model_weight_path': object_weight_file_path}
            ]
        ),

        Node(
            package='dl_ros2_application',
            executable='dl_control',
            name='dl_control_node',
            output='screen',
            parameters=[
                params_file_path
            ]
        ),
    ])
