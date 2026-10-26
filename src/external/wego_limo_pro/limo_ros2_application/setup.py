from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'limo_ros2_application'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params', '*.yaml*'))),
        (os.path.join('share', package_name, 'weight'), glob(os.path.join('weight', '*.h5*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*rviz*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='changmin@wego-robotics.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_limo = limo_ros2_application.move_limo:main',
            'limo_e_stop = limo_ros2_application.limo_e_stop:main',
            'detect_line = limo_ros2_application.detect_line:main',
            'limo_control = limo_ros2_application.limo_control:main',
            'lets_take_a_picture = limo_ros2_application.lets_take_a_picture:main',
            'move_to_pose = limo_ros2_application.move_to_pose:main',
            'patrol_limo = limo_ros2_application.patrol_limo:main',
            'drive_through_pose = limo_ros2_application.drive_through_pose:main',
            'dl_detect_line = limo_ros2_application.dl_detect_line:main',
            'detect_object = limo_ros2_application.detect_object:main',
            'rotate_absolute = limo_ros2_application.rotate_absolue:main',
        ],
    },
)
