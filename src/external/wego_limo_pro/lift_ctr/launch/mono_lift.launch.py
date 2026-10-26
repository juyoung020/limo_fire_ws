from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    limo_application_share_dir = get_package_share_directory('lift_ctr')
    parameters_file_dir = os.path.join(limo_application_share_dir, 'params')
    mono_lift_param = os.path.join(parameters_file_dir, 'mono_lift.yaml')

    return LaunchDescription([
        Node(
            package='lift_ctr',
            executable='lift_ctr',
            parameters=[mono_lift_param],
        )
    ])