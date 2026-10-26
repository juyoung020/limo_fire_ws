import rclpy
from rclpy.node import Node 

import cv2
import numpy as np
from cv_bridge import CvBridge
from limo_ros2_application.yolov8_det_vid import YoLov8TRT

from sensor_msgs.msg import Image

class DetectObject(Node):
    def __init__(self):
        # CV bridge
        self.br = CvBridge()

        # set node name
        super().__init__('detect_object')

        #set file path
        engine_file_path = '/home/wego/third_party_library/AI/yolov8_trt_ros2/build/yolov8n.engine'
        library_file_path = '/home/wego/third_party_library/AI/yolov8_trt_ros2/build/libmyplugins.so'

        # set the model for Yolov8
        self.model = YoLov8TRT(library_file_path, engine_file_path)

        # Subscribe camera data
        self.subscription = self.create_subscription(
                            Image,
                            '/camera/color/image_raw', 
                            self.image_callback, 
                            rclpy.qos.qos_profile_sensor_data)
        self.subscription # to prevent from warning

        # Publish Image for yolo
        self.yolo_publisher = self.create_publisher(Image, 'yolo_image', 10)


    def image_callback(self, msg):
        # convert opencv Mat type to image msg type
        image = np.empty(shape=[1])
        image_ = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # get data and save
        result_boxes, result_scores, result_classid = self.model.Inference(image_)

        # plot all object
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            self.plot_one_box(box, image_, label="{}:{:.2f}".format(self.model.categories[int(result_classid[i])], result_scores[i]),)
        
        # publish image
        self.yolo_publisher.publish(self.br.cv2_to_imgmsg(image_,'bgr8'))
    

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )  # line/font thickness
        color = (255, 0 ,0)#color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,)


def main(args=None):
    rclpy.init(args=args)

    detect_object = DetectObject()
    rclpy.spin(detect_object)

    detect_object.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()