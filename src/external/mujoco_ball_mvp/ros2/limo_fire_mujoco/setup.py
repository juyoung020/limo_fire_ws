from setuptools import setup

package_name = 'limo_fire_mujoco'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/pose_sync.launch.py',
            'launch/live_map.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'pose_udp = limo_fire_mujoco.pose_udp:main',
            'auto_approach_fire = limo_fire_mujoco.auto_approach_fire:main',
        ],
    },
)
