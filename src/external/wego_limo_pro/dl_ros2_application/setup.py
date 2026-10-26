from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dl_ros2_application'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'script'), glob(os.path.join('script', '*.sh*'))),
        (os.path.join('share', package_name, 'weight'), glob(os.path.join('weight', '*.h5*'))),   
        (os.path.join('share', package_name, 'weight'), glob(os.path.join('weight', '*.pt*'))),        
        (os.path.join('share', package_name, 'weight'), glob(os.path.join('weight', '*.engine*'))),  
        (os.path.join('share', package_name, 'params'), glob(os.path.join('params', '*.yaml*'))),      
        (os.path.join('share', package_name, 'weight'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'rclpy', 'opencv-python', 'dl_ros2_msgs'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='wego@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'take_a_picture = dl_ros2_application.take_a_picture:main',
            'emergency_stop = dl_ros2_application.emergency_stop:main',
            'tf_track_detect = dl_ros2_application.tf_track_detect:main',
            'trt_track_detect = dl_ros2_application.trt_track_detect:main',
            'yolo_labeler = dl_ros2_application.yolo_labeler:main',
            'yolo_object_detect = dl_ros2_application.yolo_object_detect:main',
            'dl_control = dl_ros2_application.dl_control:main',        
            'publisher_node = dl_ros2_application.publisher_node:main',
            'tf_to_trt = dl_ros2_application.tf_to_trt_runner:main',
            'yolo_to_trt = dl_ros2_application.yolo_export_runner:main',
        ],
    },
)
