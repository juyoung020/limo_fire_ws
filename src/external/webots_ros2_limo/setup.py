from setuptools import find_packages, setup

package_name = 'webots_ros2_limo'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/protos', [
    'protos/LimoFourDiff.proto',
    'protos/LimoMecanum.proto',
]))
data_files.append(('share/' + package_name + '/protos/meshes', [
    'protos/meshes/limo_base.dae',
    'protos/meshes/limo_wheel.dae',
]))
data_files.append(('share/' + package_name + '/worlds', [
    'worlds/limo_world.wbt',
    'worlds/limo_world_mecanum.wbt',
]))
data_files.append(('share/' + package_name + '/resource', [
    'resource/limo.urdf',
    'resource/limo_lds_2d.lua',
    'resource/limo_world_map.yaml',
    'resource/limo_world_map.pgm',
]))
for drive in ('diff', 'mecanum'):
    data_files.append(('share/' + package_name + '/resource/' + drive, [
        'resource/' + drive + '/ros2control.yaml',
        'resource/' + drive + '/nav2_params.yaml',
    ]))
data_files.append(('share/' + package_name + '/launch', [
    'launch/robot_launch.py',
    'launch/cartographer_launch.py',
    'launch/nav2_launch.py',
]))
data_files.append(('share/' + package_name + '/rviz', ['rviz/limo_cartographer.rviz']))
data_files.append(('share/' + package_name, ['package.xml']))


setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='inmojang',
    maintainer_email='inmo.jang@kau.ac.kr',
    description='AgileX LIMO (four-wheel diff / mecanum) Webots simulation with webots_ros2_driver + ros2_control + Nav2.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
        ],
    },
)
