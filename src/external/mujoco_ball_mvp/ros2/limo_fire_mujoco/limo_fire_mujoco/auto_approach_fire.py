"""Navigate to the nearest mapped fire, stop at standoff, then request one shot."""
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_geometry_msgs import do_transform_point, do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .approach_math import engagement_ready, standoff_goal
from .marker_cache import update_fire_marker_cache


class AutoApproachFire(Node):
    def __init__(self):
        super().__init__('mujoco_auto_approach_fire')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('markers_topic', '/object_mapper/markers')
        self.declare_parameter('navigation_action', '/navigate_to_pose')
        self.declare_parameter('auto_fire_topic', '/fire/mujoco_auto_fire')
        self.declare_parameter('standoff_m', 1.2)
        self.declare_parameter('min_fire_distance_m', 0.9)
        self.declare_parameter('max_fire_distance_m', 1.5)
        self.declare_parameter('stop_dwell_s', 0.6)
        self.declare_parameter('navigation_timeout_s', 120.0)
        self.declare_parameter('verification_timeout_s', 15.0)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.standoff = float(self.get_parameter('standoff_m').value)
        self.min_distance = float(self.get_parameter('min_fire_distance_m').value)
        self.max_distance = float(self.get_parameter('max_fire_distance_m').value)
        self.stop_dwell = float(self.get_parameter('stop_dwell_s').value)
        self.nav_timeout = float(self.get_parameter('navigation_timeout_s').value)
        self.verify_timeout = float(self.get_parameter('verification_timeout_s').value)
        if not (0.7 < self.min_distance < self.standoff < self.max_distance < 3.0):
            raise ValueError('require 0.7 < min_fire_distance < standoff < max_fire_distance < 3.0 m')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.targets = {}
        self.odom = None
        marker_qos = QoSProfile(depth=1,
                                reliability=QoSReliabilityPolicy.RELIABLE,
                                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(MarkerArray,
                                 str(self.get_parameter('markers_topic').value),
                                 self.on_markers, marker_qos)
        self.create_subscription(Odometry,
                                 str(self.get_parameter('odom_topic').value),
                                 self.on_odom, 10)
        self.fire_pub = self.create_publisher(
            Bool, str(self.get_parameter('auto_fire_topic').value), 10)
        self.nav_client = ActionClient(
            self, NavigateToPose,
            str(self.get_parameter('navigation_action').value))
        self.state = 'waiting'
        self.target_id = None
        self.goal_handle = None
        self.nav_started = 0.0
        self.verify_started = 0.0
        self.stopped_since = None
        self.fire_started = 0.0
        self.last_wait_log = 0.0
        self.done = False
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info(
            f'Auto approach enabled: standoff={self.standoff:.2f} m, '
            f'fire window={self.min_distance:.2f}–{self.max_distance:.2f} m')

    def on_odom(self, msg):
        self.odom = msg

    def on_markers(self, msg):
        def in_map(marker):
            source = marker.header.frame_id
            point = PointStamped()
            point.header = marker.header
            point.point = marker.pose.position
            try:
                if source and source != self.map_frame:
                    tf = self.tf_buffer.lookup_transform(
                        self.map_frame, source, Time(), timeout=Duration(seconds=0.05))
                    point = do_transform_point(point, tf)
                return point.point.x, point.point.y
            except TransformException:
                return None

        update_fire_marker_cache(self.targets, msg.markers, 'fire', 3, in_map)

    def robot_map_pose(self):
        if self.odom is None:
            return None
        pose = PoseStamped()
        pose.header = self.odom.header
        pose.pose = self.odom.pose.pose
        try:
            if pose.header.frame_id != self.map_frame:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, pose.header.frame_id, Time(),
                    timeout=Duration(seconds=0.05))
                pose = do_transform_pose_stamped(pose, tf)
        except TransformException:
            return None
        p, q = pose.pose.position, pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        twist = self.odom.twist.twist
        speed = math.hypot(twist.linear.x, twist.linear.y)
        return p.x, p.y, yaw, speed, twist.angular.z

    def send_navigation_goal(self):
        robot = self.robot_map_pose()
        reason = None
        if not self.targets:
            reason = 'no mapped fire mesh marker received'
        elif self.odom is None:
            reason = 'waiting for /odom'
        elif robot is None:
            reason = 'waiting for TF from odom into map'
        elif not self.nav_client.wait_for_server(timeout_sec=0.0):
            reason = 'waiting for Nav2 /navigate_to_pose action'
        if reason is not None:
            now = time.monotonic()
            if now - self.last_wait_log >= 5.0:
                self.get_logger().warning(f'Auto approach idle: {reason}; no robot motion or shot requested')
                self.last_wait_log = now
            return
        rx, ry, _, _, _ = robot
        self.target_id, target = min(
            self.targets.items(), key=lambda item: math.hypot(item[1][0] - rx,
                                                               item[1][1] - ry))
        tx, ty = target
        try:
            gx, gy, yaw = standoff_goal(rx, ry, tx, ty, self.standoff)
        except ValueError as exc:
            self.fail(str(exc))
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self.state = 'navigating'
        self.nav_started = time.monotonic()
        self.get_logger().info(
            f'Navigating to fire marker {self.target_id}: target=({tx:.2f},{ty:.2f}), '
            f'standoff goal=({gx:.2f},{gy:.2f}), distance={self.standoff:.2f} m')
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as exc:
            self.fail(f'Nav2 goal request failed: {exc}')
            return
        if not self.goal_handle.accepted:
            self.fail('Nav2 rejected the standoff goal')
            return
        self.goal_handle.get_result_async().add_done_callback(self.on_navigation_result)

    def on_navigation_result(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.fail(f'Nav2 result failed: {exc}')
            return
        if result.status != 4:  # GoalStatus.STATUS_SUCCEEDED
            self.fail(f'Nav2 ended with status {result.status}; no shot requested')
            return
        self.state = 'verifying_stop'
        self.verify_started = time.monotonic()
        self.stopped_since = None
        self.get_logger().info('Nav2 reached the standoff goal; verifying distance and stop')

    def tick(self):
        if self.done:
            return
        if self.state == 'waiting':
            self.send_navigation_goal()
        elif self.state == 'navigating':
            if time.monotonic() - self.nav_started > self.nav_timeout:
                if self.goal_handle is not None:
                    self.goal_handle.cancel_goal_async()
                self.fail('navigation timeout; no shot requested')
        elif self.state == 'verifying_stop':
            self.verify_engagement()
        elif self.state == 'publishing_fire':
            if time.monotonic() - self.fire_started < 1.0:
                self.fire_pub.publish(Bool(data=True))
            else:
                self.fire_pub.publish(Bool(data=False))
                self.get_logger().info('Auto-fire request sent to MuJoCo viewer')
                self.state = 'done'
                self.done = True
        elif self.state == 'failed':
            self.done = True

    def verify_engagement(self):
        robot = self.robot_map_pose()
        target = self.targets.get(self.target_id)
        if robot is None or target is None:
            return
        if time.monotonic() - self.verify_started > self.verify_timeout:
            self.fail('standoff/stop verification timeout; no shot requested')
            return
        rx, ry, yaw, speed, angular_speed = robot
        tx, ty = target
        distance = math.hypot(tx - rx, ty - ry)
        ready = engagement_ready(
            distance, yaw, tx - rx, ty - ry, speed, angular_speed,
            min_distance=self.min_distance, max_distance=self.max_distance)
        if ready:
            now = time.monotonic()
            if self.stopped_since is None:
                self.stopped_since = now
            if now - self.stopped_since >= self.stop_dwell:
                self.get_logger().info(
                    f'Stopped facing fire {self.target_id} at {distance:.2f} m; requesting MuJoCo shot')
                self.state = 'publishing_fire'
                self.fire_started = now
        else:
            self.stopped_since = None

    def fail(self, reason):
        if self.state == 'navigating' and self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.get_logger().error(reason)
        self.state = 'failed'


def main(args=None):
    rclpy.init(args=args)
    node = AutoApproachFire()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.state == 'navigating' and node.goal_handle is not None:
            node.goal_handle.cancel_goal_async()
        node.fire_pub.publish(Bool(data=False))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
