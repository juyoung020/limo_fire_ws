import rclpy
from rclpy.node import Node 

import cv2
import numpy as np
from cv_bridge import CvBridge
from tensorflow import keras

from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from ament_index_python.packages import get_package_share_directory
import os

class DetectLine(Node):
    def __init__(self):
        # CV bridge
        self.br = CvBridge()

        # set node name
        super().__init__('dl_detect_line')

        #set default weigh file path
        package_directory = get_package_share_directory('limo_ros2_application')
        folder_name = 'weight'
        file_name = 'dl_drive.h5'
        self.weight = os.path.join(package_directory, folder_name, file_name)
        
        # initialize the deep learning model
        self.initModel()

        # Subscribe camera data
        self.subscription = self.create_subscription(
                            Image,
                            '/camera/color/image_raw', 
                            self.image_callback, 
                            rclpy.qos.qos_profile_sensor_data)
        self.subscription # to prevent from warning

        # Publish result (offset between reference distance and real distance)
        self.dis_publisher = self.create_publisher(Float32, 'distance_y', 10)

        # Publish Image for debugging
        self.debug_publisher = self.create_publisher(Image, 'debug_image', 10)
        self.timer_ = self.create_timer(0.1, self.timer_callback)

        # set the reference lane
        self.declare_parameter('reference_distance', 150.0)
        self.reference_distance = self.get_parameter('reference_distance')

        # to syncronize subscriber and publisher
        self.sub_flag = False

    
    def initModel(self):
        input1 = keras.layers.Input(
            shape=(
                110,
                300,
                3,
            )
        )

        conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(input1)
        norm1 = keras.layers.BatchNormalization()(conv1)
        pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(norm1)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(pool1)
        norm2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm2)
        norm3 = keras.layers.BatchNormalization()(conv3)
        add1 = keras.layers.Add()([norm2, norm3])
        conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add1)
        norm4 = keras.layers.BatchNormalization()(conv4)
        conv5 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm4)
        norm5 = keras.layers.BatchNormalization()(conv5)
        add2 = keras.layers.Add()([norm4, norm5])
        conv6 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add2)
        norm6 = keras.layers.BatchNormalization()(conv6)
        conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="swish")(norm6)
        norm7 = keras.layers.BatchNormalization()(conv7)
        add3 = keras.layers.Add()([norm6, norm7])
        conv8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(add3)
        norm7 = keras.layers.BatchNormalization()(conv8)
        conv9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(norm7)
        norm8 = keras.layers.BatchNormalization()(conv9)
        flat1 = keras.layers.Flatten()(norm8)
        dense1 = keras.layers.Dense(128, activation="swish")(flat1)
        norm9 = keras.layers.BatchNormalization()(dense1)
        dense2 = keras.layers.Dense(64, activation="swish")(norm9)
        norm10 = keras.layers.BatchNormalization()(dense2)
        dense3 = keras.layers.Dense(64, activation="swish")(norm10)
        norm11 = keras.layers.BatchNormalization()(dense3)
        dense4 = keras.layers.Dense(2, activation="tanh")(norm11)
        self.model = keras.models.Model(inputs=input1, outputs=dense4)
        self.model.load_weights(self.weight)
    
    def timer_callback(self):
        if self.sub_flag:      
            self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.debug_image,'bgr8'))
          
    def image_callback(self, msg):
        # convert opencv Mat type to image msg type
        image_ = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # resize the image
        resize_img = cv2.resize(image_, (300, 300), cv2.INTER_LINEAR)

        # corp_image
        crop_img = resize_img[190:300, :]

        #get the x, y (0~1, 0~1)
        x, y = self.model(np.array([crop_img]).astype(np.float32)).numpy()[0]

        # map 0~1 to 0~300 
        cx = int(300 * x)

        # calculate the gap
        distance_to_ref = self.reference_distance.value -cx

        # draw circle
        self.debug_image = cv2.circle(resize_img, (cx, 190), 10,(255, 0, 0), -1)

        # Publishing the offset between current distance with lane and reference distance
        dis = Float32()
        dis.data = distance_to_ref
        self.dis_publisher.publish(dis)

        self.sub_flag = True

        
def main(args=None):
    rclpy.init(args=args)

    detect_line = DetectLine()
    rclpy.spin(detect_line)

    detect_line.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()