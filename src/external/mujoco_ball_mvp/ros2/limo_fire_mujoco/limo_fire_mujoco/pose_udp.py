"""Send LIMO odometry and fire landmarks to the MuJoCo live viewer."""
import json
import math
import socket

import rclpy
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_geometry_msgs import do_transform_point, do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from .marker_cache import update_fire_marker_cache


class PoseUdp(Node):
    """Bridge team object-map fire markers (map) and LIMO odometry (odom)."""

    def __init__(self):
        super().__init__('limo_mujoco_pose_sync')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('markers_topic', '/object_mapper/markers')
        self.declare_parameter('target_topic', '/fire/detected_target_pose')
        self.declare_parameter('targets_topic', '/fire/detected_targets')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('fire_namespace', 'fire')
        self.declare_parameter('fire_marker_part', 3)
        self.declare_parameter('target_timeout', 0.75)
        self.declare_parameter('udp_host', '127.0.0.1')
        self.declare_parameter('udp_port', 8765)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.fire_namespace = str(self.get_parameter('fire_namespace').value)
        self.fire_marker_part = int(self.get_parameter('fire_marker_part').value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (
            str(self.get_parameter('udp_host').value),
            int(self.get_parameter('udp_port').value),
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.robot_pose = None
        self.pose_target = None
        self.pose_target_stamp = None
        self.pose_targets = None
        self.pose_targets_stamp = None
        self.map_targets = []
        self.map_targets_by_id = {}
        self.map_targets_known = False
        self.auto_fire_requested = False
        self._last_tf_warning = self.get_clock().now()

        self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self.on_odom,
            10,
        )
        marker_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter('markers_topic').value),
            self.on_markers,
            marker_qos,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter('target_topic').value),
            self.on_target,
            10,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter('targets_topic').value),
            self.on_targets,
            10,
        )
        self.create_subscription(
            Bool,
            '/fire/mujoco_auto_fire',
            self.on_auto_fire,
            10,
        )
        self.get_logger().info(
            f'Forwarding {self.fire_namespace!r} map markers and odometry '
            f'to MuJoCo viewer UDP {self.dest[0]}:{self.dest[1]}; '
            f'target frame is {self.target_frame!r}')

    def on_auto_fire(self, msg):
        self.auto_fire_requested = bool(msg.data)

    @staticmethod
    def stamp_seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def on_target(self, msg):
        pose = self.transform_pose(msg)
        if pose is None:
            return
        x, y = float(pose.position.x), float(pose.position.y)
        if math.isfinite(x) and math.isfinite(y):
            self.pose_target = (x, y)
            self.pose_target_stamp = self.stamp_seconds(msg.header.stamp)

    def on_targets(self, msg):
        points = []
        for pose in msg.poses:
            stamped = PoseStamped()
            stamped.header = msg.header
            stamped.pose = pose
            transformed = self.transform_pose(stamped)
            if transformed is None:
                continue
            x, y = float(transformed.position.x), float(transformed.position.y)
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        self.pose_targets = points
        self.pose_targets_stamp = self.stamp_seconds(msg.header.stamp)

    def transform_pose(self, pose):
        source = pose.header.frame_id
        if not source:
            self.get_logger().warning('Ignoring target with empty frame_id')
            return None
        if source == self.target_frame:
            return pose.pose
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, source, Time(), timeout=Duration(seconds=0.05))
            return do_transform_pose_stamped(pose, transform).pose
        except TransformException as exc:
            self.warn_tf(source, exc)
            return None

    def warn_tf(self, source_frame, exc):
        now = self.get_clock().now()
        if (now - self._last_tf_warning).nanoseconds > 2_000_000_000:
            self.get_logger().warning(
                f'Cannot transform {source_frame!r} -> {self.target_frame!r}: {exc}')
            self._last_tf_warning = now

    def on_markers(self, msg):
        # ObjectMapper publishes several visual pieces per landmark. Part 3
        # is the fire mesh icon; the center/risk cylinders and labels are ignored.
        def transform_marker(marker):
            if marker.type != Marker.MESH_RESOURCE:
                return None
            point = PointStamped()
            point.header = marker.header
            point.point = marker.pose.position
            source_frame = marker.header.frame_id
            try:
                if source_frame and source_frame != self.target_frame:
                    transform = self.tf_buffer.lookup_transform(
                        self.target_frame,
                        source_frame,
                        Time(),
                        timeout=Duration(seconds=0.05),
                    )
                    point = do_transform_point(point, transform)
            except TransformException as exc:
                self.warn_tf(source_frame, exc)
                return None

            x, y = float(point.point.x), float(point.point.y)
            if math.isfinite(x) and math.isfinite(y):
                return x, y
            return None

        changed = update_fire_marker_cache(
            self.map_targets_by_id,
            msg.markers,
            self.fire_namespace,
            self.fire_marker_part,
            transform_marker,
        )
        if changed:
            self.map_targets = list(self.map_targets_by_id.values())
            self.map_targets_known = True

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        robot_pose = PoseStamped()
        robot_pose.header = msg.header
        robot_pose.pose = msg.pose.pose
        transformed_robot = self.transform_pose(robot_pose)
        if transformed_robot is None:
            return
        p = transformed_robot.position
        q = transformed_robot.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        now = self.get_clock().now().nanoseconds * 1e-9
        timeout = float(self.get_parameter('target_timeout').value)
        array_fresh = (
            self.pose_targets is not None
            and self.pose_targets_stamp is not None
            and -0.5 <= now - self.pose_targets_stamp <= timeout
        )
        single_fresh = (
            self.pose_target is not None
            and self.pose_target_stamp is not None
            and -0.5 <= now - self.pose_target_stamp <= timeout
        )

        # Mapped landmarks remain valid in the static map until ObjectMapper
        # explicitly deletes them. Transient camera-pose topics still expire.
        if self.map_targets_known:
            targets = self.map_targets
        elif array_fresh:
            targets = self.pose_targets
        elif single_fresh:
            targets = [self.pose_target]
        else:
            targets = []

        targets = sorted(
            targets,
            key=lambda target: math.hypot(target[0] - p.x, target[1] - p.y),
        )[:8]
        target_x, target_y = targets[0] if targets else (0.0, 0.0)
        packet = json.dumps(
            {
                'x': p.x,
                'y': p.y,
                'yaw': yaw,
                'target_x': target_x,
                'target_y': target_y,
                'target_valid': bool(targets),
                'targets': targets,
                'fire_request': self.auto_fire_requested,
            },
            separators=(',', ':'),
        ).encode()
        try:
            self.sock.sendto(packet, self.dest)
        except OSError as exc:
            self.get_logger().warning(f'UDP pose send failed: {exc}')

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseUdp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
