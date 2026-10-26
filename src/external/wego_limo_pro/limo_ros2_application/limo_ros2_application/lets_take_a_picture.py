import rclpy
from rclpy.node import Node

import os

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from wego_msgs.srv import Chalkak
from datetime import datetime


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
        self.dataset_directory = '/home/wego/resources/dataset' 

    def take_picture(self, request, response):
        if self.sub_flag and request.kimchi: 
            # if all goes well save picture at directory + picture name
            #get the current time
            current_time_sec = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            
            # set the file name and directory
            response.picture = os.path.join(
                self.dataset_directory,
                f'img_{current_time_sec}.jpg')

            # write the image
            cv2.imwrite(response.picture,self.image_)
        else:
            # you did something wrong
            response.picture = 'Did you execute the camera driver?'
        return response

    def image_callback(self, msg):
        # subscribe image and change to cv type
        cv_img = self.br.imgmsg_to_cv2(msg, 'bgr8') 
        
        # resize image
        self.image_ = cv2.resize(cv_img, (300, 300))

        # set flag true
        self.sub_flag = True
        

def main(args=None):
    rclpy.init(args=args)

    lets_take_a_picture = LetsTakeAPicture()

    rclpy.spin(lets_take_a_picture)

    lets_take_a_picture.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()