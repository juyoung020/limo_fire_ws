"""Webots plugin that publishes the LIMO sensor suite exactly like the real robot.

Every value below was measured on a real LIMO (Jetson Orin Nano, `ros2 launch wego
teleop_launch.py`, 2026-10-06), so that code written against the simulator runs
unchanged on the robot:

  /scan  <- ydlidar_ros2_driver_node : LaserScan, BEST_EFFORT, 10 Hz, frame laser_link,
            +-110 deg, 248 beams, range_min 0.01 / range_max 64 (driver values),
            invalid beams = 0.0 (not inf), intensities filled.
  /imu   <- /limo/limo_base           : Imu, RELIABLE, 100 Hz, frame imu_link, gravity
            included, covariances 1e-6 / 1e-6 / 1.0.
  /odom  <- /limo/limo_base           : Odometry, RELIABLE, 50 Hz, frame odom, child base_link,
            published together with the odom -> base_link TF (on the robot that TF comes
            from its EKF). Distance from the wheel encoders, heading from the IMU.
  /camera/... <- /camera/camera       : Orbbec DaBai DC1 (OrbbecSDK_ROS2) with hardware
            depth-to-color alignment, so color, depth and IR are all stamped with
            camera_color_optical_frame and share the color intrinsics.

Camera streams are rendered only while somebody subscribes to them, which keeps the
simulation light when nobody looks at the camera.

Message payloads are assigned as array.array: rclpy validates a `bytes` value element by
element (~90 ms for one 640x480 RGB frame), while array.array is copied directly.
"""

import array
import math
import struct

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, LaserScan, PointCloud2, PointField
from tf2_ros import TransformBroadcaster

# ---- YDLIDAR (ydlidar_ros2_driver/params/ydlidar.yaml on the robot) ----
SCAN_PERIOD = 0.1
SCAN_ANGLE_MIN = -1.919862151145935       # -110 deg
SCAN_ANGLE_MAX = 1.919862151145935        # +110 deg
SCAN_ANGLE_INCREMENT = 0.015514037571847439  # 2*pi / 405 beams per turn
SCAN_RANGE_MIN = 0.01
SCAN_RANGE_MAX = 64.0
SCAN_INTENSITY = 47.0  # The real driver reports ~30-50 for valid returns, 0 otherwise.

# ---- IMU (limo_base) ----
IMU_PERIOD = 0.01
IMU_ORIENTATION_COV = [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6]
IMU_GYRO_COV = [1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6]
IMU_ACCEL_COV = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# ---- Odometry (limo_base) ----
ODOM_PERIOD = 0.02  # 50 Hz
WHEEL_RADIUS = 0.045
WHEEL_SENSORS = ('front_left_wheel_sensor', 'rear_left_wheel_sensor',
                 'front_right_wheel_sensor', 'rear_right_wheel_sensor')
MECANUM_MODEL = 'LimoMecanum'  # Robot.model in protos/LimoMecanum.proto

# ---- Orbbec DaBai DC1 (camera_info read from the robot) ----
CAMERA_FRAME = 'camera_color_optical_frame'
FX = FY = 490.8955383300781
CX, CY = 317.5404357910156, 217.3939208984375
COLOR_W, COLOR_H = 640, 480
IR_W, IR_H = 640, 400
# Webots renders an ideal pinhole with the principal point in the image center, so each
# stream is rendered slightly larger and cropped to put (CX, CY) where the robot has it.
# See the Camera/RangeFinder sizes in LimoFourDiff.proto.
COLOR_CROP = (44, 4)   # (top, left) rows/cols removed from a 644x524 render -> cx 317.5, cy 217.5
IR_CROP = (0, 4)       # from a 644x436 render -> cx 317.5, cy 217.5
DEPTH_MIN, DEPTH_MAX = 0.3, 3.0  # DaBai DC1 depth range [m]
# Part of the 640x480 color image the depth-to-color aligned stream covers (measured on the
# robot: rows 22..399, right edge at column 627).
DEPTH_ROWS = (22, 400)
DEPTH_COLS = (0, 628)
CLOUD_MIN, CLOUD_MAX = 0.02, 10.0  # OrbbecSDK_ROS2 point cloud validity window [m]
# camera_link -> stream frames, as published on /tf by the Orbbec driver (10 Hz).
CAMERA_TFS = [
    ('camera_link', 'camera_color_frame', (0.00015, -0.03, 0.00002), (0.00087, 0.00020, 0.00048, 1.0)),
    ('camera_link', 'camera_depth_frame', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ('camera_link', 'camera_ir_frame', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ('camera_color_frame', 'camera_color_optical_frame', (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5)),
    ('camera_depth_frame', 'camera_depth_optical_frame', (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5)),
    ('camera_ir_frame', 'camera_ir_optical_frame', (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5)),
]
CAMERA_PERIOD_MS = 30
CAMERA_TF_PERIOD = 0.1
SUBSCRIBER_CHECK_PERIOD = 0.5


def _stamp(msg, t, frame_id):
    msg.header.stamp.sec = int(t)
    msg.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1000000000
    msg.header.frame_id = frame_id
    return msg


def _yaw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _camera_info(width, height, d):
    info = CameraInfo()
    info.width, info.height = width, height
    info.distortion_model = 'rational_polynomial'
    info.d = list(d)
    info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def _bgra(camera, crop, width, height):
    raw = np.frombuffer(camera.getImage(), np.uint8).reshape(camera.getHeight(), camera.getWidth(), 4)
    top, left = crop
    return raw[top:top + height, left:left + width]


class LimoSensors:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        step = int(self.__robot.getBasicTimeStep())

        rclpy.init(args=None)
        # Same node names as on the robot.
        self.__lidar_node = rclpy.create_node('ydlidar_ros2_driver_node')
        self.__base_node = rclpy.create_node('limo_base', namespace='limo')
        self.__camera_node = rclpy.create_node('camera', namespace='camera')

        # LiDAR: the Webots device spans exactly SCAN_ANGLE_MIN..MAX with one ray per beam.
        self.__lidar = self.__robot.getDevice('laser')
        self.__lidar.enable(int(SCAN_PERIOD * 1000))
        self.__scan_pub = self.__lidar_node.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.__scan = LaserScan()
        self.__scan.angle_min = SCAN_ANGLE_MIN
        self.__scan.angle_max = SCAN_ANGLE_MAX
        self.__scan.angle_increment = SCAN_ANGLE_INCREMENT
        self.__scan.scan_time = SCAN_PERIOD
        self.__scan.time_increment = SCAN_PERIOD / round(2 * math.pi / SCAN_ANGLE_INCREMENT)
        self.__scan.range_min = SCAN_RANGE_MIN
        self.__scan.range_max = SCAN_RANGE_MAX
        self.__lidar_min = self.__lidar.getMinRange()
        self.__lidar_max = self.__lidar.getMaxRange()

        # IMU
        self.__inertial = self.__robot.getDevice('imu inertial')
        self.__gyro = self.__robot.getDevice('imu gyro')
        self.__accel = self.__robot.getDevice('imu accelerometer')
        for device in (self.__inertial, self.__gyro, self.__accel):
            device.enable(max(step, int(IMU_PERIOD * 1000)))
        self.__imu_pub = self.__base_node.create_publisher(Imu, '/imu', 10)

        # Odometry. ros2_control reads the same wheel sensors, at the basic time step.
        self.__wheels = [self.__robot.getDevice(name) for name in WHEEL_SENSORS]
        for wheel in self.__wheels:
            if wheel.getSamplingPeriod() == 0:
                wheel.enable(step)
        self.__mecanum = self.__robot.getModel() == MECANUM_MODEL
        self.__base_node.get_logger().info(
            'Odometry kinematics: ' + ('mecanum' if self.__mecanum else 'four-wheel differential'))
        self.__odom_pub = self.__base_node.create_publisher(Odometry, '/odom', 10)
        self.__odom_tf = TransformBroadcaster(self.__base_node)
        self.__odom_prev = None  # (time, wheel positions, IMU yaw) of the last update
        self.__yaw0 = None       # IMU yaw at start: /odom starts at heading 0
        self.__x = self.__y = 0.0

        # Camera
        self.__color_cam = self.__robot.getDevice('rgb_camera')
        self.__depth_cam = self.__robot.getDevice('depth_camera')
        self.__ir_cam = self.__robot.getDevice('ir_camera')
        cn = self.__camera_node
        self.__pubs = {
            'color': cn.create_publisher(Image, '/camera/color/image_raw', 10),
            'color_info': cn.create_publisher(CameraInfo, '/camera/color/camera_info', 10),
            'color_jpeg': cn.create_publisher(CompressedImage, '/camera/color/image_raw/compressed', 10),
            'depth': cn.create_publisher(Image, '/camera/depth/image_raw', 10),
            'depth_info': cn.create_publisher(CameraInfo, '/camera/depth/camera_info', 10),
            'depth_jpeg': cn.create_publisher(CompressedImage, '/camera/depth/image_raw/compressed', 10),
            'depth_png': cn.create_publisher(CompressedImage, '/camera/depth/image_raw/compressedDepth', 10),
            'ir': cn.create_publisher(Image, '/camera/ir/image_raw', 10),
            'ir_info': cn.create_publisher(CameraInfo, '/camera/ir/camera_info', 10),
            'ir_jpeg': cn.create_publisher(CompressedImage, '/camera/ir/image_raw/compressed', 10),
            'points': cn.create_publisher(PointCloud2, '/camera/depth/points', 10),
            'colored_points': cn.create_publisher(PointCloud2, '/camera/depth_registered/points', 10),
        }
        self.__color_info = _camera_info(
            COLOR_W, COLOR_H,
            [0.06985840946435928, -0.09333103150129318, 0.0004893741570413113, 0.00035896344343200326,
             0.0, 0.0, 0.0, 0.0])
        self.__depth_info = _camera_info(COLOR_W, COLOR_H, [0.0] * 8)
        self.__ir_info = _camera_info(IR_W, IR_H, [0.0] * 8)
        self.__wanted = {}
        self.__enabled = {'color': False, 'depth': False, 'ir': False}
        self.__ready = {}  # stream -> sim time of its first frame after enable()
        self.__tf = TransformBroadcaster(cn)

        # Pixel rays for depth -> point cloud, and the image area the stereo pair covers.
        u, v = np.meshgrid(np.arange(COLOR_W, dtype=np.float32), np.arange(COLOR_H, dtype=np.float32))
        self.__ray_x = (u - CX) / FX
        self.__ray_y = (v - CY) / FY
        self.__depth_area = np.zeros((COLOR_H, COLOR_W), bool)
        self.__depth_area[DEPTH_ROWS[0]:DEPTH_ROWS[1], DEPTH_COLS[0]:DEPTH_COLS[1]] = True

        self.__next = {'scan': 0.0, 'imu': 0.0, 'odom': 0.0, 'camera': 0.0, 'tf': 0.0, 'subs': 0.0}

    # ------------------------------------------------------------------ helpers
    def __due(self, key, t, period):
        if t + 1e-6 < self.__next[key]:
            return False
        self.__next[key] = max(self.__next[key] + period, t + period / 2)
        return True

    def __update_subscribers(self, t):
        count = {k: p.get_subscription_count() for k, p in self.__pubs.items()}
        self.__wanted = {k: c > 0 for k, c in count.items()}
        w = self.__wanted
        need = {
            'color': w['color'] or w['color_info'] or w['color_jpeg'] or w['colored_points'],
            'depth': (w['depth'] or w['depth_info'] or w['depth_jpeg'] or w['depth_png']
                      or w['points'] or w['colored_points']),
            'ir': w['ir'] or w['ir_info'] or w['ir_jpeg'],
        }
        devices = {'color': self.__color_cam, 'depth': self.__depth_cam, 'ir': self.__ir_cam}
        for name, on in need.items():
            if on != self.__enabled[name]:
                if on:
                    devices[name].enable(CAMERA_PERIOD_MS)
                    self.__ready[name] = t + CAMERA_PERIOD_MS / 1000.0
                else:
                    devices[name].disable()
                self.__enabled[name] = on

    # ------------------------------------------------------------------ sensors
    def __publish_scan(self, t):
        # Webots orders the rays from left (+fov/2) to right; ROS wants increasing angle.
        ranges = np.array(self.__lidar.getRangeImage(), dtype=np.float32)[::-1]
        valid = np.isfinite(ranges) & (ranges >= self.__lidar_min) & (ranges < self.__lidar_max)
        ranges = np.where(valid, ranges, 0.0).astype(np.float32)
        _stamp(self.__scan, t, 'laser_link')
        self.__scan.ranges = array.array('f', ranges.tobytes())
        self.__scan.intensities = array.array('f', np.where(valid, SCAN_INTENSITY, 0.0).astype(np.float32).tobytes())
        self.__scan_pub.publish(self.__scan)

    def __publish_imu(self, t):
        msg = _stamp(Imu(), t, 'imu_link')
        q = self.__inertial.getQuaternion()
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        g = self.__gyro.getValues()
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = g
        a = self.__accel.getValues()
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a
        msg.orientation_covariance = IMU_ORIENTATION_COV
        msg.angular_velocity_covariance = IMU_GYRO_COV
        msg.linear_acceleration_covariance = IMU_ACCEL_COV
        self.__imu_pub.publish(msg)

    def __publish_odom(self, t):
        """Wheel + IMU odometry, as computed by the former limo_odometry node.

        Heading comes from the IMU, not the encoders: the LIMO's wheels slip sideways
        whenever it turns, so the encoder yaw is only right for one turning speed (in this
        simulation it over-reports by 6 % at 1.0 rad/s on the spot and by 14 % while
        driving). The real LIMO firmware takes its heading from the IMU for the same reason.
        Reading the devices here, at the exact simulation time, also keeps the stamps and
        the velocities free of /clock jitter.
        """
        positions = [wheel.getValue() for wheel in self.__wheels]  # fl, rl, fr, rr [rad]
        yaw = _yaw(self.__inertial.getQuaternion())
        if not all(math.isfinite(v) for v in positions + [yaw]):
            return  # devices not sampled yet
        if self.__odom_prev is None:
            self.__odom_prev = (t, positions, yaw)
            self.__yaw0 = yaw
            return
        t0, positions0, yaw0 = self.__odom_prev
        self.__odom_prev = (t, positions, yaw)

        # Rim travel of each wheel; body displacement in base_link. Forward is the mean of
        # all wheels; mecanum wheels also move the body sideways (same forward kinematics as
        # ros2_controllers' mecanum_drive_controller).
        fl, rl, fr, rr = ((p - p0) * WHEEL_RADIUS for p, p0 in zip(positions, positions0))
        dx = (fl + fr + rl + rr) / 4.0
        dy = (-fl + fr + rl - rr) / 4.0 if self.__mecanum else 0.0

        # Integrate along the mean heading over the interval.
        prev_heading = _wrap(yaw0 - self.__yaw0)
        heading = _wrap(yaw - self.__yaw0)
        mid = prev_heading + 0.5 * _wrap(heading - prev_heading)
        self.__x += dx * math.cos(mid) - dy * math.sin(mid)
        self.__y += dx * math.sin(mid) + dy * math.cos(mid)

        odom = _stamp(Odometry(), t, 'odom')
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.__x
        odom.pose.pose.position.y = self.__y
        q = odom.pose.pose.orientation
        q.z, q.w = math.sin(0.5 * heading), math.cos(0.5 * heading)
        odom.twist.twist.linear.x = dx / (t - t0)
        odom.twist.twist.linear.y = dy / (t - t0)
        odom.twist.twist.angular.z = self.__gyro.getValues()[2]
        for i, v in ((0, 1e-3), (7, 1e-3), (14, 1e6), (21, 1e6), (28, 1e6), (35, 1e-3)):
            odom.pose.covariance[i] = v
            odom.twist.covariance[i] = v
        self.__odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self.__x
        tf.transform.translation.y = self.__y
        tf.transform.rotation = odom.pose.pose.orientation
        self.__odom_tf.sendTransform(tf)

    def __publish_camera_tf(self, t):
        transforms = []
        for parent, child, xyz, quat in CAMERA_TFS:
            tf = _stamp(TransformStamped(), t, parent)
            tf.child_frame_id = child
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = xyz
            r = tf.transform.rotation
            r.x, r.y, r.z, r.w = quat
            transforms.append(tf)
        self.__tf.sendTransform(transforms)

    def __image(self, t, data, encoding, step):
        msg = _stamp(Image(), t, CAMERA_FRAME)
        msg.height, msg.width = data.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = step
        msg.data = array.array('B', data.tobytes())
        return msg

    def __compressed(self, t, fmt, data):
        msg = _stamp(CompressedImage(), t, CAMERA_FRAME)
        msg.format = fmt
        msg.data = array.array('B', data)
        return msg

    def __publish_camera(self, t):
        w, p = self.__wanted, self.__pubs
        live = {k: on and t + 1e-6 >= self.__ready[k] for k, on in self.__enabled.items()}
        rgb = None
        if live['color']:
            rgb = np.ascontiguousarray(_bgra(self.__color_cam, COLOR_CROP, COLOR_W, COLOR_H)[:, :, 2::-1])
            if w['color']:
                p['color'].publish(self.__image(t, rgb, 'rgb8', COLOR_W * 3))
            if w['color_info']:
                p['color_info'].publish(_stamp(self.__color_info, t, CAMERA_FRAME))
            if w['color_jpeg']:
                ok, jpg = cv2.imencode('.jpg', rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                p['color_jpeg'].publish(self.__compressed(t, 'rgb8; jpeg compressed bgr8', jpg.tobytes()))

        if live['depth']:
            ptr = self.__depth_cam.getRangeImage(data_type='buffer')
            full = np.ctypeslib.as_array(ptr, (self.__depth_cam.getHeight(), self.__depth_cam.getWidth()))
            top, left = COLOR_CROP
            meters = np.array(full[top:top + COLOR_H, left:left + COLOR_W], dtype=np.float32)
            valid = np.isfinite(meters) & (meters >= DEPTH_MIN) & (meters <= DEPTH_MAX)
            valid &= self.__depth_area
            # Webots returns the distance along the optical axis, i.e. the z of each pixel.
            mm = np.where(valid, np.round(meters * 1000.0), 0).astype(np.uint16)
            if w['depth']:
                p['depth'].publish(self.__image(t, mm, '16UC1', COLOR_W * 2))
            if w['depth_info']:
                p['depth_info'].publish(_stamp(self.__depth_info, t, CAMERA_FRAME))
            if w['depth_jpeg']:
                # The real driver advertises this topic but always sends an empty payload
                # (image_transport cannot JPEG-encode 16UC1).
                p['depth_jpeg'].publish(self.__compressed(t, '16UC1; jpeg compressed mono8', b''))
            if w['depth_png']:
                ok, png = cv2.imencode('.png', mm)
                header = struct.pack('<iff', 0, 0.0, 0.0)
                p['depth_png'].publish(self.__compressed(t, '16UC1; compressedDepth', header + png.tobytes()))
            if w['points'] or (w['colored_points'] and rgb is not None):
                z = mm.astype(np.float32) / 1000.0
                keep = (z >= CLOUD_MIN) & (z <= CLOUD_MAX)
                xyz = np.stack((self.__ray_x * z, self.__ray_y * z, z), axis=-1)[keep]
                if len(xyz):
                    if w['points']:
                        # Both clouds use the color frame on the robot (depth is aligned to color).
                        p['points'].publish(self.__cloud(t, CAMERA_FRAME, xyz, None))
                    if w['colored_points'] and rgb is not None:
                        p['colored_points'].publish(self.__cloud(t, CAMERA_FRAME, xyz, rgb[keep]))

        if live['ir']:
            bgra = _bgra(self.__ir_cam, IR_CROP, IR_W, IR_H)
            gray = cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2GRAY)
            ir = ((gray.astype(np.uint32) * 1023 + 127) // 255).astype(np.uint16)  # Y10 -> 0..1023
            if w['ir']:
                p['ir'].publish(self.__image(t, ir, 'mono16', IR_W * 2))
            if w['ir_info']:
                p['ir_info'].publish(_stamp(self.__ir_info, t, CAMERA_FRAME))
            if w['ir_jpeg']:
                ok, jpg = cv2.imencode('.jpg', (ir >> 8).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
                p['ir_jpeg'].publish(self.__compressed(t, 'mono16; jpeg compressed mono8', jpg.tobytes()))

    def __cloud(self, t, frame, xyz, rgb):
        """Unordered cloud with the field layout of OrbbecSDK_ROS2."""
        msg = _stamp(PointCloud2(), t, frame)
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
        if rgb is None:
            dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                              'offsets': [0, 4, 8], 'itemsize': 16})
        else:
            fields.append(PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1))
            dtype = np.dtype({'names': ['x', 'y', 'z', 'b', 'g', 'r'],
                              'formats': ['<f4'] * 3 + ['u1'] * 3,
                              'offsets': [0, 4, 8, 16, 17, 18], 'itemsize': 20})
        cloud = np.zeros(len(xyz), dtype=dtype)
        cloud['x'], cloud['y'], cloud['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        if rgb is not None:
            cloud['r'], cloud['g'], cloud['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        msg.height, msg.width = 1, len(cloud)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = dtype.itemsize
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = array.array('B', cloud.tobytes())
        return msg

    # ------------------------------------------------------------------ main loop
    def step(self):
        t = self.__robot.getTime()
        for node in (self.__lidar_node, self.__base_node, self.__camera_node):
            rclpy.spin_once(node, timeout_sec=0)

        if self.__due('subs', t, SUBSCRIBER_CHECK_PERIOD):
            self.__update_subscribers(t)
        if self.__due('imu', t, IMU_PERIOD):
            self.__publish_imu(t)
        if self.__due('odom', t, ODOM_PERIOD):
            self.__publish_odom(t)
        if self.__due('scan', t, SCAN_PERIOD):
            self.__publish_scan(t)
        if self.__due('tf', t, CAMERA_TF_PERIOD):
            self.__publish_camera_tf(t)
        if any(self.__enabled.values()) and self.__due('camera', t, CAMERA_PERIOD_MS / 1000.0):
            self.__publish_camera(t)
