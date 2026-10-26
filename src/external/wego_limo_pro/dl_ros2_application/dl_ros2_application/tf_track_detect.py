import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from tensorflow import keras
from sensor_msgs.msg import CompressedImage, Image
from dl_ros2_msgs.msg import Control
from ament_index_python.packages import get_package_share_directory
import os

class TFTrackDetect(Node):
    def __init__(self):
        super().__init__('tf_track_detect_node')
        self.br = CvBridge()
        
        self.declare_parameter('model_weight_path', '')
        self.weight_path = self.get_parameter('model_weight_path').get_parameter_value().string_value
        
        if not self.weight_path or not os.path.exists(self.weight_path):
            self.get_logger().error(f"Model weight file not found at: {self.weight_path}")
            raise FileNotFoundError("Model weight file not found.")
        
        self.initModel()
        
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        
        self.control_publisher = self.create_publisher(Control, '/control', 10)

        self.debug_publisher = self.create_publisher(Image, 'tf_image', 10)
        self.debug_compressed_publisher = self.create_publisher(CompressedImage, 'tf_image/compressed', 10)

    def initModel(self):
        input1 = keras.layers.Input(shape=(110, 300, 3))
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
        norm8 = keras.layers.BatchNormalization()(conv8)
        conv9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="swish")(norm8)
        norm9 = keras.layers.BatchNormalization()(conv9)
        
        flat1 = keras.layers.Flatten()(norm9)
        dense1 = keras.layers.Dense(128, activation="swish")(flat1)
        norm10 = keras.layers.BatchNormalization()(dense1)
        dense2 = keras.layers.Dense(64, activation="swish")(norm10)
        norm11 = keras.layers.BatchNormalization()(dense2)
        dense3 = keras.layers.Dense(64, activation="swish")(norm11)
        norm12 = keras.layers.BatchNormalization()(dense3)
        dense4 = keras.layers.Dense(2, activation="tanh")(norm12)
        
        self.model = keras.models.Model(inputs=input1, outputs=dense4)
        
        try:
            self.model.load_weights(self.weight_path)
            self.get_logger().info(f"Model loaded successfully from {self.weight_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model weights: {e}")
            raise

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_ = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image_ is None:
                self.get_logger().warn("Failed to decode compressed image")
                return
            
            resize_img = cv2.resize(image_, (300, 300), cv2.INTER_LINEAR)
            crop_img = resize_img[190:300, :]
            
            output = self.model(np.array([crop_img]).astype(np.float32)).numpy()[0]
            x_1, x_2 = output
            
            cx_1 = int(300 * x_1)
            cx_2 = int(300 * x_2)
            ref = 150
            
            gap = Control()
            gap.normal_mode = int(ref - cx_1)
            gap.turn_mode = int(ref - cx_2)

            self.control_publisher.publish(gap)
            
            debug_image = cv2.putText(resize_img, 'Normal Mode', (cx_1, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            debug_image = cv2.line(debug_image, (ref, 175), (cx_1, 175), (255, 0, 0), 3)
            debug_image = cv2.circle(debug_image, (ref, 175), 5, (0, 255, 0), -1)
            debug_image = cv2.circle(debug_image, (cx_1, 175), 5, (255, 0, 0), -1)
            
            debug_image = cv2.putText(debug_image, 'Turn Mode', (cx_2, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            debug_image = cv2.line(debug_image, (ref, 205), (cx_2, 205), (0, 0, 255), 3)
            debug_image = cv2.circle(debug_image, (ref, 205), 5, (0, 255, 0), -1)
            debug_image = cv2.circle(debug_image, (cx_2, 205), 5, (0, 0, 255), -1)
            
            self.debug_publisher.publish(self.br.cv2_to_imgmsg(debug_image, 'bgr8'))

            debug_compressed_msg = self.br.cv2_to_compressed_imgmsg(debug_image)
            self.debug_compressed_publisher.publish(debug_compressed_msg)

            # self.get_logger().info(f"Published error: normal={gap.normal_mode}, turn={gap.turn_mode}")
        
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    tf_track_detect_node = None
    try:
        tf_track_detect_node = TFTrackDetect()
        rclpy.spin(tf_track_detect_node)
    except Exception as e:
        print(f"Error running node: {e}")
    finally:
        if tf_track_detect_node is not None:
            tf_track_detect_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
