import rclpy
from rclpy.node import Node 

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Int32

class DetectLine(Node):
    def __init__(self):
        # CV bridge
        self.br = CvBridge()

        # Subscribe camera data
        super().__init__('detect_line')
        self.subscription = self.create_subscription(
                            Image,
                            '/camera/color/image_raw', 
                            self.image_callback, 
                            rclpy.qos.qos_profile_sensor_data)
        self.subscription # to prevent from warning

        # Publish result (offset between reference distance and real distance)
        self.dis_publisher = self.create_publisher(Int32, 'distance_y', 10)

        # Publish Image for debugging
        self.debug_publisher = self.create_publisher(Image, 'debug_image', 10)
        self.timer_ = self.create_timer(0.1, self.timer_callback)

        # Parameters (For Masking Lane, For the reference distance of lane, ROI)
        self.declare_parameter('roi_x_l', 0)
        self.declare_parameter('roi_x_h', 320) 
        self.declare_parameter('roi_y_l', 400)
        self.declare_parameter('roi_y_h', 480)
        
        self.roi_x_l=self.get_parameter('roi_x_l')
        self.roi_x_h=self.get_parameter('roi_x_h')
        self.roi_y_l=self.get_parameter('roi_y_l')
        self.roi_y_h=self.get_parameter('roi_y_h')

        self.declare_parameter('lane_h_l', 0)
        self.declare_parameter('lane_l_l', 90)
        self.declare_parameter('lane_s_l', 100)
        self.declare_parameter('lane_h_h', 60)
        self.declare_parameter('lane_l_h', 220)
        self.declare_parameter('lane_s_h', 255)

        lane_h_l=self.get_parameter('lane_h_l')
        lane_l_l=self.get_parameter('lane_l_l')
        lane_s_l=self.get_parameter('lane_s_l')
        lane_h_h=self.get_parameter('lane_h_h')
        lane_l_h=self.get_parameter('lane_l_h')
        lane_s_h=self.get_parameter('lane_s_h')
        
        self.yellow_lane_low = np.array([lane_h_l.value,
                                         lane_l_l.value,
                                         lane_s_l.value])
        self.yellow_lane_high = np.array([lane_h_h.value,
                                          lane_l_h.value, 
                                          lane_s_h.value])
        
        self.declare_parameter('reference_distance', 170)
        self.reference_distance = self.get_parameter('reference_distance')

        # Parameter For debugging
        # 0: ROI
        # 1: Masking
        # 2: Moment of lane
        self.declare_parameter('debug_image_num', 2) #default 2
        self.debug_sequence = self.get_parameter('debug_image_num')

        # to syncronize subscriber and publisher
        self.sub_flag = False
    
    def timer_callback(self):
        if self.sub_flag:     
            if self.debug_sequence.value == 0:
                self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.roi_,'bgr8'))
            elif self.debug_sequence.value == 1:
                self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.mask_yellow,'mono8'))
            else:    
                self.debug_publisher.publish(self.br.cv2_to_imgmsg(self.image_,'bgr8'))
          
    def image_callback(self, msg):
        # convert opencv Mat type to image msg type
        self.image_ = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # Make Region of Interest
        self.roi_ = self.image_[self.roi_y_l.value:self.roi_y_h.value,
                                self.roi_x_l.value:self.roi_x_h.value]

        # Masking the lane
        hls = cv2.cvtColor(self.roi_, cv2.COLOR_BGR2HLS)
        self.mask_yellow = cv2.inRange(hls, self.yellow_lane_low,
                                            self.yellow_lane_high)
        
        # Calculating the Moment of the lane
        M = cv2.moments(self.mask_yellow)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cy = self.roi_y_l.value + cy
                        
            self.image_ = cv2.line(self.image_, 
                        (self.reference_distance.value, 0),
                        (self.reference_distance.value, 480),
                        (0, 255, 0), 
                        5)
            self.image_ = cv2.circle(self.image_, (cx, cy), 10,(255, 0, 0), -1)

            distance_to_ref = self.reference_distance.value -cx
        else: # When limo cannot find lane publish 0 data
            distance_to_ref = 0

        # Publishing the offset between current distance with lane and reference distance
        dis = Int32()
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