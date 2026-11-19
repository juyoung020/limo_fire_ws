from glob import glob

from setuptools import setup

package_name = 'limo_fire_hardware'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'REAL_AIM_TEST.md',
                                  'REAL_FIRE_TEST.md', 'AIM_LEARNING_PREP.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'approach_aim = limo_fire_hardware.approach_aim:main',
        'pitch_serial_bridge = limo_fire_hardware.pitch_serial_bridge:main',
        'mapper_observation_bridge = limo_fire_hardware.mapper_observation_bridge:main',
        'launcher_control_terminal = limo_fire_hardware.launcher_control_terminal:main',
    ]},
)
