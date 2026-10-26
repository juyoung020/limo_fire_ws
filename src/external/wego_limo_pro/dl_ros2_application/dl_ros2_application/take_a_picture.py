import rclpy
from rclpy.node import Node
from dl_ros2_msgs.srv import PhotoShoot
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

class TakeAPicture(Node):
    def __init__(self):
        super().__init__('take_a_picture_node')
        self.br = CvBridge()
        self.cv_image = None
        self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        self.create_service(PhotoShoot, 'photoshoot', self.save_image)
        self.focus_dir = os.path.abspath(os.path.join('dataset', 'dl_images'))
        self.yolo_train_dir = os.path.abspath(os.path.join('dataset', 'yolo_images', 'images', 'train'))
        self.yolo_val_dir = os.path.abspath(os.path.join('dataset', 'yolo_images', 'images', 'val'))
        for path in [self.focus_dir, self.yolo_train_dir, self.yolo_val_dir]:
            os.makedirs(path, exist_ok=True)

    def image_callback(self, msg):
        self.cv_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

    def save_image(self, request, response):
        if self.cv_image is None:
            response.photoshoot = False
            return response
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f'img_{timestamp}.jpg'

        try:
            if request.chalkak:
                image = cv2.resize(self.cv_image, (300, 300))
                save_path = os.path.join(self.focus_dir, filename)
                cv2.imwrite(save_path, image)
            else:
                for path in [self.yolo_train_dir, self.yolo_val_dir]:
                    save_path = os.path.join(path, filename)
                    cv2.imwrite(save_path, self.cv_image)

            response.photoshoot = True
        except Exception:
            response.photoshoot = False

        return response

def main(args=None):
    rclpy.init(args=args)
    node = TakeAPicture()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
