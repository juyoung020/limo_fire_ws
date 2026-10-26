import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sensor_msgs.msg import CompressedImage, Image
from dl_ros2_msgs.msg import Control
from ament_index_python.packages import get_package_share_directory
import os

class TensorRTInference:
    def __init__(self, engine_path, logger):
        self.engine_path = engine_path
        self.logger = logger
        self.engine = None
        self.context = None
        self.h_input = None
        self.d_input = None
        self.h_output = None
        self.d_output = None
        self.bindings = None
        self.stream = None
        self.init_tensorrt()

    def init_tensorrt(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

            input_name = self.engine.get_tensor_name(0)
            output_name = self.engine.get_tensor_name(1)
            input_shape = self.engine.get_tensor_shape(input_name)
            output_shape = self.engine.get_tensor_shape(output_name)
            
            self.h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.stream = cuda.Stream()
            
            self.logger.info(f"TensorRT engine initialized with input shape: {input_shape}")
            self.logger.info(f"TensorRT engine initialized with output shape: {output_shape}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorRT: {e}")
            raise

    def infer(self, input_data):
        try:
            np.copyto(self.h_input, input_data.ravel())
            cuda.memcpy_htod(self.d_input, self.h_input)
            
            self.context.execute_v2(self.bindings)
            
            cuda.memcpy_dtoh(self.h_output, self.d_output)
            return self.h_output.reshape((1, 2))
        except Exception as e:
            self.logger.error(f"TensorRT inference failed: {e}")
            raise

class TRTTrackDetect(Node):
    def __init__(self):
        super().__init__('trt_track_detect_node')
        self.br = CvBridge()
        
        self.declare_parameter('engine_file_path', '')
        engine_path = self.get_parameter('engine_file_path').get_parameter_value().string_value
        
        if not engine_path or not os.path.exists(engine_path):
            self.get_logger().error(f"Engine file not found at: {engine_path}")
            raise FileNotFoundError("Engine file not found.")

        self.trt_inference = TensorRTInference(engine_path, self.get_logger())
        self.get_logger().info("TensorRT inference engine initialized.")

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        
        self.control_publisher = self.create_publisher(Control, '/control', 10)

        self.debug_publisher = self.create_publisher(Image, 'trt_image', 10)
        self.debug_compressed_publisher = self.create_publisher(CompressedImage, 'trt_image/compressed', 10)

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_ = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image_ is None:
                self.get_logger().warn("Failed to decode compressed image")
                return

            resize_img = cv2.resize(image_, (300, 300), interpolation=cv2.INTER_LINEAR)
            crop_img = resize_img[190:300, :].astype(np.float32)
            input_data = np.expand_dims(crop_img, axis=0)
            
            output = self.trt_inference.infer(input_data)
            x_1, x_2 = output[0]

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
            self.get_logger().warn(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    trt_track_detect_node = None
    try:
        trt_track_detect_node = TRTTrackDetect()
        rclpy.spin(trt_track_detect_node)
    except Exception as e:
        print(f"Error running node: {e}")
    finally:
        if trt_track_detect_node is not None:
            trt_track_detect_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
