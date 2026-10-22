import os
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, LoadComposableNodes
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ComposableNode, ParameterFile
from nav2_common.launch import RewrittenYaml
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    wego_share_dir = get_package_share_directory('wego_2d_nav')

    # setting for map and parameter
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
 
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(wego_share_dir, 'maps', 'map.yaml'),
        description='Full path to map yaml file to load')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(wego_share_dir, 'params', 'diff_navigation_params.yaml'),
        description='Full path to parameter yaml file to load')

    
    # remapping the topic and set lifecycle node
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    lifecycle_nodes = ['map_server', 'amcl']

    load_composable_nodes = GroupAction(
        actions=[
            LoadComposableNodes(
                target_container='nav2_container',
                composable_node_descriptions=[
                    ComposableNode( # loading the map to map server
                        package='nav2_map_server',
                        plugin='nav2_map_server::MapServer',
                        name='map_server',
                        parameters=[{'yaml_filename': map_yaml_file}],
                        remappings=remappings,
                    ),
                    ComposableNode( # amcl localization
                        package='nav2_amcl',
                        plugin='nav2_amcl::AmclNode',
                        name='amcl',
                        parameters=[ParameterFile(params_file)],
                    ),
                    ComposableNode( # For lifecycle
                        package='nav2_lifecycle_manager',
                        plugin='nav2_lifecycle_manager::LifecycleManager',
                        name='lifecycle_manager_localization',
                        parameters=[
                            {'autostart': True,
                            'node_names': lifecycle_nodes}
                        ],
                    ),
                ],
            ),
        ],
    )
    
    return LaunchDescription([
        declare_map_yaml_cmd,
        declare_params_file_cmd,
        load_composable_nodes,
    ])