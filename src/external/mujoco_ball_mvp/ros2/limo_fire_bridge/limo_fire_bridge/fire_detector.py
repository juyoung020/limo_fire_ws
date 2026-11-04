"""Detect the red Webots fire-box face and estimate its odom-frame position."""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class FireDetector(Node):
    def __init__(self):
        super().__init__('webots_fire_detector')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_topic', '/fire/detected_target_pose')
        self.declare_parameter('targets_topic', '/fire/detected_targets')
        self.declare_parameter('debug_topic', '/fire/detection_image')
        self.declare_parameter('target_frame', 'odom')
        self.declare_parameter('target_depth_offset', 0.15)
        self.declare_parameter('min_component_area', 250)
        self.declare_parameter('max_depth_age', 0.15)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.color = None
        self.color_stamp = None
        self.depth = None
        self.depth_stamp = None
        self.fx = self.fy = self.cx = self.cy = None
        self._last_detection_log = self.get_clock().now()
        self._last_missing_log = self.get_clock().now()

        self.target_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter('target_topic').value), 10)
        self.targets_pub = self.create_publisher(
            PoseArray, str(self.get_parameter('targets_topic').value), 10)
        self.debug_pub = self.create_publisher(
            Image, str(self.get_parameter('debug_topic').value), 1)
        camera_qos = QoSProfile(depth=1)
        self.create_subscription(
            Image, str(self.get_parameter('depth_topic').value), self.on_depth, camera_qos)
        self.create_subscription(
            CameraInfo, str(self.get_parameter('camera_info_topic').value), self.on_camera_info, camera_qos)
        self.create_subscription(
            Image, str(self.get_parameter('color_topic').value), self.on_color, camera_qos)
        self.get_logger().info(
            'Detecting the Webots fire box from red RGB pixels and registered depth; '
            'publishing all detected fire boxes on /fire/detected_targets and the best candidate on /fire/detected_target_pose')

    @staticmethod
    def stamp_seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_camera_info(self, msg):
        self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
        self.cx, self.cy = float(msg.k[2]), float(msg.k[5])

    def on_depth(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_stamp = self.stamp_seconds(msg.header.stamp)
        except (ValueError, RuntimeError) as exc:
            self.get_logger().warning(f'Cannot decode depth image: {exc}')

    def on_color(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (ValueError, RuntimeError) as exc:
            self.get_logger().warning(f'Cannot decode color image: {exc}')
            return
        self.color = image
        self.color_stamp = self.stamp_seconds(msg.header.stamp)
        if (self.depth is None or self.depth_stamp is None or
                abs(self.color_stamp - self.depth_stamp) > float(self.get_parameter('max_depth_age').value)):
            self.publish_debug(image, msg, None)
            return
        if self.fx is None or self.fy is None or self.fx <= 0 or self.fy <= 0:
            self.publish_debug(image, msg, None)
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # The Webots target box has a saturated red finish. Keep the hue band
        # narrow so brown cartons and warm posters do not become fire targets.
        red = cv2.inRange(hsv, np.array((0, 120, 55), np.uint8), np.array((8, 255, 255), np.uint8))
        red |= cv2.inRange(hsv, np.array((172, 120, 55), np.uint8), np.array((179, 255, 255), np.uint8))
        kernel = np.ones((3, 3), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(red, 8)
        min_area = int(self.get_parameter('min_component_area').value)
        candidates = []
        height, width = image.shape[:2]
        for label in range(1, count):
            x, y, w, h, area = map(int, stats[label])
            if area < min_area or w < 8 or h < 30:
                continue
            aspect = h / max(w, 1)
            if not 0.7 <= aspect <= 4.0:
                continue
            # Prefer tall box-shaped components near the image center.
            u, v = map(float, centroids[label])
            center_penalty = ((u - width / 2) / (width * 0.55)) ** 2 + ((v - height / 2) / (height * 0.65)) ** 2
            score = area / (1.0 + center_penalty)
            candidates.append((score, label, x, y, w, h, area, u, v))

        if not candidates:
            self.publish_debug(image, msg, None)
            self.log_missing()
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                str(self.get_parameter('target_frame').value), msg.header.frame_id,
                Time(), timeout=Duration(seconds=0.05))
        except TransformException as exc:
            self.log_missing(f'camera-to-odom TF unavailable: {exc}')
            return

        depth = np.asarray(self.depth)
        accepted = []
        for score, label, x, y, w, h, area, u, v in sorted(candidates, reverse=True):
            component_depth = depth[labels == label]
            valid_depth = component_depth[np.isfinite(component_depth) & (component_depth > 0)]
            if valid_depth.size < max(8, area // 20):
                continue
            z_raw = float(np.median(valid_depth))
            if depth.dtype == np.uint16:
                z_raw *= 0.001
            if not 0.25 <= z_raw <= 4.0:
                continue
            width_m=w*z_raw/self.fx
            height_m=h*z_raw/self.fy
            if not 0.10 <= width_m <= 0.65 or not 0.15 <= height_m <= 0.90:
                continue

            # Registered depth estimates each box independently. The measured
            # red face is about half a box depth in front of its center.
            z = z_raw + float(self.get_parameter('target_depth_offset').value)
            point = PointStamped()
            point.header = msg.header
            point.point.x = (u - self.cx) * z / self.fx
            point.point.y = (v - self.cy) * z / self.fy
            point.point.z = z
            point_odom = do_transform_point(point, transform)
            pose = PoseStamped().pose
            pose.position = point_odom.point
            pose.orientation.w = 1.0

            # A single box can produce nearby components on its front and side;
            # merge those while retaining spatially separate boxes.
            if any(math.hypot(pose.position.x-old.pose.position.x,
                              pose.position.y-old.pose.position.y) < 0.28
                   for _, old, _ in accepted):
                continue
            accepted.append((score, pose, (x, y, w, h, u, v, z_raw)))

        if not accepted:
            self.publish_debug(image, msg, None)
            self.log_missing('red regions failed depth/target-size checks')
            return

        target_frame = str(self.get_parameter('target_frame').value)
        array = PoseArray()
        array.header.stamp = transform.header.stamp
        array.header.frame_id = target_frame
        array.poses = [pose for _, pose, _ in accepted]
        self.targets_pub.publish(array)

        best_pose = PoseStamped()
        best_pose.header = array.header
        best_pose.pose = accepted[0][1]
        self.target_pub.publish(best_pose)
        self.publish_debug(image, msg, [candidate for _, _, candidate in accepted])
        now = self.get_clock().now()
        if (now - self._last_detection_log).nanoseconds > 1_000_000_000:
            self.get_logger().info(
                f'Fire boxes detected: {len(accepted)}; best odom=('
                f'{best_pose.pose.position.x:.2f}, {best_pose.pose.position.y:.2f}) m')
            self._last_detection_log = now

    def publish_debug(self, image, source_msg, candidate):
        debug = image.copy()
        if candidate is not None:
            candidates = candidate if isinstance(candidate, list) else [candidate]
            for index, item in enumerate(candidates, start=1):
                x, y, w, h, u, v, depth = item
                color = (0, 255, 0) if depth is not None else (0, 165, 255)
                cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
                cv2.drawMarker(debug, (int(u), int(v)), color, cv2.MARKER_CROSS, 14, 2)
                label = f'fire {index}' if depth is not None else 'red candidate: no depth'
                if depth is not None:
                    label += f' {depth:.2f}m'
                cv2.putText(debug, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2, cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        msg.header = source_msg.header
        self.debug_pub.publish(msg)

    def log_missing(self, reason='no matching red fire box in camera view'):
        now = self.get_clock().now()
        if (now - self._last_missing_log).nanoseconds > 2_000_000_000:
            self.get_logger().info(f'Fire target not detected: {reason}')
            self._last_missing_log = now


def main(args=None):
    rclpy.init(args=args)
    node = FireDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
