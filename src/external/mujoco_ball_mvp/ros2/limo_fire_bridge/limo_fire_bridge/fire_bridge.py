"""Gate a MuJoCo shot on a fresh target pose and stationary LIMO odometry."""
import math
import os
from pathlib import Path
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose_stamped


class FireBridge(Node):
    def __init__(self):
        super().__init__('limo_fire_bridge')
        self.declare_parameter('target_topic', '/fire/target_pose')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('result_topic', '/fire/shot_result')
        self.declare_parameter('target_height', .25)
        self.declare_parameter('speed', 5.5)
        self.declare_parameter('aim', 'surface')
        self.declare_parameter('stop_speed', .03)
        self.declare_parameter('settle_seconds', 1.0)
        self.declare_parameter('max_target_age', 3.0)
        self.declare_parameter('mujoco_root', str(Path(__file__).resolve().parents[3]))
        self.declare_parameter('python', '')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.result_pub = self.create_publisher(String, str(self.get_parameter('result_topic').value), 10)
        self.odom = None
        self.stopped_since = None
        self.last_target = None
        self.busy = False
        self.lock = threading.Lock()
        self.create_subscription(Odometry, str(self.get_parameter('odom_topic').value), self.on_odom, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter('target_topic').value), self.on_target, 10)
        self.get_logger().info('Waiting for /odom and PoseStamped target; target must be published in map/odom/base_link TF tree.')

    def on_odom(self, msg):
        self.odom = msg
        v = msg.twist.twist
        stopped = math.hypot(v.linear.x, v.linear.y) < float(self.get_parameter('stop_speed').value) and abs(v.angular.z) < float(self.get_parameter('stop_speed').value)
        now = self.get_clock().now()
        if stopped:
            if self.stopped_since is None:
                self.stopped_since = now
        else:
            self.stopped_since = None

    def on_target(self, msg):
        self.last_target = msg
        if self.busy or self.odom is None or self.stopped_since is None:
            self.publish('rejected', 'LIMO odometry missing or robot not confirmed stopped')
            return
        settled = (self.get_clock().now() - self.stopped_since).nanoseconds / 1e9
        if settled < float(self.get_parameter('settle_seconds').value):
            self.publish('waiting', 'waiting for stationary dwell')
            return
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp, clock_type=self.get_clock().clock_type)).nanoseconds / 1e9
        if msg.header.stamp.sec and (age < -0.5 or age > float(self.get_parameter('max_target_age').value)):
            self.publish('rejected', f'stale/future target pose age={age:.2f}s')
            return
        base = PoseStamped(); base.header.stamp = self.get_clock().now().to_msg(); base.header.frame_id = 'base_link'
        try:
            if msg.header.frame_id == 'base_link':
                relative = msg
            else:
                transform = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
                relative = do_transform_pose_stamped(msg, transform)
        except TransformException as exc:
            self.publish('rejected', f'TF unavailable: {exc}')
            return
        x, y = relative.pose.position.x, relative.pose.position.y
        if not all(math.isfinite(v) for v in (x, y)) or x <= 0:
            self.publish('rejected', 'target must be finite and in front of base_link (x > 0)')
            return
        with self.lock:
            if self.busy: return
            self.busy = True
        threading.Thread(target=self.run_shot, args=(x, y), daemon=True).start()

    def run_shot(self, x, y):
        root = Path(str(self.get_parameter('mujoco_root').value)).expanduser().resolve()
        python = str(self.get_parameter('python').value) or str(root / '.venv/bin/python')
        command = [python, str(root / 'launch.py'), '--headless', '--aim', str(self.get_parameter('aim').value), '--distance', f'{x + 0.05:.6f}', '--target-y', f'{y:.6f}', '--aim-height', str(float(self.get_parameter('target_height').value)), '--speed', str(float(self.get_parameter('speed').value))]
        self.publish('firing', f'target base_link=({x:.3f},{y:.3f}); launching MuJoCo child')
        try:
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30, check=False)
            output = completed.stdout + completed.stderr
            state = 'completed' if completed.returncode == 0 else 'failed'
            self.publish(state, output[-3500:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.publish('failed', str(exc))
        finally:
            with self.lock: self.busy = False

    def publish(self, state, detail):
        msg = String(); msg.data = f'{state}: {detail}'
        self.result_pub.publish(msg)
        self.get_logger().info(msg.data[:500])


def main(args=None):
    rclpy.init(args=args); node=FireBridge()
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
