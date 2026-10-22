#!/usr/bin/env python

# Launch Webots AgileX LIMO driver.
#
# Structure mirrors the official Cyberbotics example
# `webots_ros2/webots_ros2_turtlebot/launch/robot_launch.py`:
#   * `slam:=true`  -> includes turtlebot3_cartographer (cartographer SLAM + RViz)
#   * `nav:=true`   -> includes turtlebot3_navigation2 (Nav2/AMCL + RViz) with a saved map
# Both are optional and only activate when the corresponding package is installed.
#
# `drive:=diff` (default) is the four-wheel differential LIMO, `drive:=mecanum`
# the omnidirectional one. The drive selects the world (robot model), the
# ros2_control controller and the Nav2 parameters in `resource/<drive>/`.

import os
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch import LaunchDescription
from launch_ros.actions import Node
import launch
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection

# Per drive: default world, ros2_control controller and the topic on which
# that controller listens for velocity commands (remapped to /cmd_vel).
DRIVES = {
    'diff': {
        'world': 'limo_world.wbt',
        'controller': 'diffdrive_controller',
        'cmd_vel_topic': '/diffdrive_controller/cmd_vel_unstamped',
    },
    'mecanum': {
        'world': 'limo_world_mecanum.wbt',
        'controller': 'mecanum_drive_controller',
        'cmd_vel_topic': '/mecanum_drive_controller/reference_unstamped',
    },
}


def launch_setup(context):
    package_dir = get_package_share_directory('webots_ros2_limo')
    drive = LaunchConfiguration('drive').perform(context)
    if drive not in DRIVES:
        raise RuntimeError(f"drive must be one of {list(DRIVES)}, got '{drive}'")
    world = LaunchConfiguration('world').perform(context) or DRIVES[drive]['world']
    mode = LaunchConfiguration('mode')
    use_nav = LaunchConfiguration('nav', default=False)
    use_slam = LaunchConfiguration('slam', default=False)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    nav2_map = LaunchConfiguration(
        'map', default=os.path.join(package_dir, 'resource', 'limo_world_map.yaml'))

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', world),
        mode=mode,
        ros2_supervisor=True
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '<robot name=""><link name=""/></robot>'
        }],
    )

    # No base_link -> base_footprint TF: the real LIMO has no base_footprint, and a
    # config that relies on it (e.g. Nav2 AMCL's default base_frame_id) would work
    # here but fail on the robot.

    # Camera mount TFs: same frames and chain as the real robot (wego/camera_tilt_launch.py,
    # degree:=0), but base_link -> camera_mount is set so that camera_link lands on the Webots
    # camera, i.e. the camera block of the LIMO body mesh: camera_link = (0.096, 0.015, 0.03).
    # (The robot's own values, (0.2, 0.1, 0.06), would put the camera 7 cm in front of the
    # bumper and 5 cm to the left of where it is mounted.)
    camera_mount_tfs = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=name,
            output='screen',
            arguments=['--x', x, '--y', y, '--z', z, '--frame-id', parent, '--child-frame-id', child],
        )
        for name, x, y, z, parent, child in [
            ('base_link_to_camera_mount', '0.066', '0.065', '0.03', 'base_link', 'camera_mount'),
            ('camera_mount_to_roate', '0.0', '0.0', '0.0', 'camera_mount', 'camera_rotate'),
            ('rotate_to_camera_link', '0.03', '-0.05', '0.0', 'camera_rotate', 'camera_link'),
        ]
    ]

    # ROS control spawners
    controller_manager_timeout = ['--controller-manager-timeout', '50']
    drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[DRIVES[drive]['controller']] + controller_manager_timeout,
    )
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['joint_state_broadcaster'] + controller_manager_timeout,
    )
    ros_control_spawners = [drive_controller_spawner, joint_state_broadcaster_spawner]

    robot_description_path = os.path.join(package_dir, 'resource', 'limo.urdf')
    ros2_control_params = os.path.join(package_dir, 'resource', drive, 'ros2control.yaml')
    mappings = [
        (DRIVES[drive]['cmd_vel_topic'], '/cmd_vel'),
    ]
    limo_driver = WebotsController(
        robot_name='LIMO',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time': use_sim_time,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=mappings,
        respawn=True
    )

    # /odom and the odom -> base_link TF come from the limo_sensors plugin (see
    # resource/limo.urdf), not from the drive controller's own odometry.

    # Optional stacks. Both launch files live in this package and each brings
    # its own RViz, so these shortcut arguments run exactly what the README
    # asks you to type in a second terminal.
    navigation_nodes = []
    if 'nav2_bringup' in get_packages_with_prefixes():
        limo_navigation = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                package_dir, 'launch', 'nav2_launch.py')),
            launch_arguments=[
                ('drive', drive),
                ('map', nav2_map),
                ('use_sim_time', use_sim_time),
            ],
            condition=launch.conditions.IfCondition(use_nav))
        navigation_nodes.append(limo_navigation)

    if 'cartographer_ros' in get_packages_with_prefixes():
        limo_slam = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                package_dir, 'launch', 'cartographer_launch.py')),
            launch_arguments=[
                ('use_sim_time', use_sim_time),
            ],
            condition=launch.conditions.IfCondition(use_slam))
        navigation_nodes.append(limo_slam)

    # Wait for the simulation to be ready to start navigation nodes.
    #
    # NOTE: starting the simulator and the SLAM/Nav stack from this single
    # command loses a startup race in roughly 1 launch in 5 — Nav2's
    # global_costmap can latch the robot at the map origin and never recover.
    # The upstream webots_ros2_turtlebot example behaves the same way, so the
    # wiring here is left identical to it. The README documents starting the
    # two stacks in separate terminals, which avoids the race entirely.
    waiting_nodes = WaitForControllerConnection(
        target_driver=limo_driver,
        nodes_to_start=navigation_nodes + ros_control_spawners
    )

    return [
        webots,
        webots._supervisor,

        robot_state_publisher,
        *camera_mount_tfs,

        limo_driver,
        waiting_nodes,

        # This action will kill all nodes once the Webots simulation has exited
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[
                    launch.actions.EmitEvent(event=launch.events.Shutdown())
                ],
            )
        ),
    ]


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_limo')
    return LaunchDescription([
        DeclareLaunchArgument(
            'drive',
            default_value='diff',
            choices=list(DRIVES),
            description='Drive type of the LIMO'
        ),
        DeclareLaunchArgument(
            'world',
            default_value='',
            description='World file from `webots_ros2_limo/worlds`. Empty: the '
                        'default world of the chosen drive'
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='realtime',
            description='Webots startup mode'
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='false',
            description='Also start cartographer SLAM + RViz (cartographer_launch.py)'
        ),
        DeclareLaunchArgument(
            'nav',
            default_value='false',
            description='Also start Nav2/AMCL + RViz (nav2_launch.py)'
        ),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(package_dir, 'resource', 'limo_world_map.yaml'),
            description='Map yaml used when nav:=true'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Webots /clock as ROS 2 time'
        ),
        OpaqueFunction(function=launch_setup),
    ])
