import rclpy
from rclpy.node import Node

import os

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from ament_index_python.packages import get_package_share_directory
from wego_msgs.srv import Chalkak


class LetsTakeAPicture(Node):

    def __init__(self):
        super().__init__('lets_take_a_picture')

        # CV bridge
        self.br = CvBridge()

        # service server for taking picuture
        self.srv = self.create_service(Chalkak, 'say_kimchi', self.take_picture)

        # subscribe camera data
        self.subscription = self.create_subscription(
                            Image,
                            '/camera/color/image_raw', 
                            self.image_callback, 
                            rclpy.qos.qos_profile_sensor_data)
        self.subscription # to prevent from warning

        # to syncronize subscriber and service server
        self.sub_flag = False

        # for saving the image
        package_directory = get_package_share_directory('limo_ros2_application')
        self.picture_directory = package_directory + '/picture/'
        self.picture_index = 1


    def take_picture(self, request, response):
        if self.sub_flag and request.kimchi: # if all goes well save picture at package + picture directory
            response.picture = f'{self.picture_directory}{self.picture_index}_pic.png'
            cv2.imwrite(response.picture,self.image_)
            self.picture_index = self.picture_index + 1
        else:
            response.picture = 'Did you execute the camera driver?'
        return response

    def image_callback(self, msg):
        self.image_ = self.br.imgmsg_to_cv2(msg, 'bgr8') # subscribe image and change to cv type
        self.sub_flag = True
        

def main(args=None):
    rclpy.init(args=args)

    lets_take_a_picture = LetsTakeAPicture()

    rclpy.spin(lets_take_a_picture)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lets_take_a_picture.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()