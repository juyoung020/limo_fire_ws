"""Bounded ROS integration fixtures using fake Nav2, sensors and pitch transport.

Run on an isolated ROS_DOMAIN_ID. No robot driver or serial bridge is started.
"""
import json
import math
import threading
import time
import unittest

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from limo_fire_msgs.msg import FireObservation
from limo_msgs.msg import LimoStatus
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from limo_fire_hardware.approach_aim import ApproachAim


class FakeRobot(Node):
    def __init__(self, delay_navigation=False, nav_yaw_bias_deg=0.0):
        super().__init__('fake_approach_aim_robot')
        self.group = ReentrantCallbackGroup()
        self.pose = (0.0, 0.0, 0.0)
        self.target_x = 2.0
        self.delay_navigation = delay_navigation
        self.nav_yaw_bias_deg = nav_yaw_bias_deg
        self.angular_rate = 0.0
        self.motion_mode = 0
        self.yaw_commands = []
        self.last_sensor_at = time.monotonic()
        self.nav_goal_seen = threading.Event()
        self.nav_cancel_seen = threading.Event()
        self.navigation_goals = []
        self.pitch_commands = []
        self.shot_requests = 0
        self.statuses = []
        self.grid_stale = False
        self.send_malformed = False
        self.sequence = 0
        self.feedback_mode = 'none'
        self.feedback_stamp_ns = None
        self.bridge_status = dict(state='ready', sequence=0, command_deg=None,
                                  command_stamp=None, acked=False, measured=False,
                                  firing_enabled=True)
        self.tf_pub = TransformBroadcaster(self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/global_costmap/costmap', latched)
        self.fire_pub = self.create_publisher(FireObservation, '/fire/observations', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.base_status_pub = self.create_publisher(LimoStatus, '/limo_status', 10)
        self.bridge_pub = self.create_publisher(String, '/launcher/pitch_status', 10)
        self.shot_pub = self.create_publisher(String, '/launcher/shot_status', latched)
        self.joint_pub = self.create_publisher(JointState, '/launcher/joint_states', 10)
        self.create_subscription(Float64, '/launcher/pitch_command', self.pitch, 10)
        self.create_subscription(Twist, '/cmd_vel', self.yaw_command, 10)
        self.create_subscription(String, '/launcher/approach_aim_status', self.status, 10)
        self.planner = ActionServer(self, ComputePathToPose, '/compute_path_to_pose',
                                    self.plan, callback_group=self.group)
        self.navigator = ActionServer(self, NavigateToPose, '/navigate_to_pose',
                                      self.navigate, goal_callback=self.nav_accept,
                                      cancel_callback=self.nav_cancel, callback_group=self.group)
        self.start_client = self.create_client(Trigger, '/launcher/approach_and_aim')
        self.cancel_client = self.create_client(Trigger, '/launcher/cancel')
        self.shot_service = self.create_service(Trigger, '/launcher/shot', self.shot,
                                                callback_group=self.group)
        self.sensor_timer = self.create_timer(0.05, self.publish_sensors, callback_group=self.group)
        self.grid_timer = self.create_timer(0.2, self.publish_grid, callback_group=self.group)

    def status(self, msg):
        self.statuses.append(json.loads(msg.data))

    def pitch(self, msg):
        self.pitch_commands.append(msg.data)
        self.sequence += 1
        self.bridge_status = dict(state='acknowledged', sequence=self.sequence,
                                  command_deg=msg.data,
                                  command_stamp=self.get_clock().now().nanoseconds * 1e-9,
                                  acked=True, measured=False, firing_enabled=True)

    def shot(self, request, response):
        del request
        self.shot_requests += 1
        self.shot_pub.publish(String(data=json.dumps(dict(state='completed', command='shot',
                                                          sequence=self.shot_requests, measured=False))))
        response.success = True
        response.message = f'fixture accepted sequence={self.shot_requests}'
        return response

    def yaw_command(self, msg):
        self.angular_rate = msg.angular.z
        self.yaw_commands.append(msg.angular.z)

    def nav_accept(self, request):
        del request
        self.nav_goal_seen.set()
        if self.delay_navigation:
            time.sleep(0.7)
        return GoalResponse.ACCEPT

    def nav_cancel(self, handle):
        del handle
        self.nav_cancel_seen.set()
        return CancelResponse.ACCEPT

    def navigate(self, handle):
        self.navigation_goals.append(handle.request.pose)
        if self.delay_navigation:
            deadline = time.monotonic() + 5
            while rclpy.ok() and time.monotonic() < deadline:
                if handle.is_cancel_requested:
                    handle.canceled()
                    return NavigateToPose.Result()
                time.sleep(0.02)
            handle.abort()
            return NavigateToPose.Result()
        pose = handle.request.pose.pose
        q = pose.orientation
        yaw = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
        if len(self.navigation_goals) == 1:
            yaw += math.radians(self.nav_yaw_bias_deg)
        self.pose = (pose.position.x, pose.position.y, yaw)
        handle.succeed()
        return NavigateToPose.Result()

    def plan(self, handle):
        result = ComputePathToPose.Result()
        result.path.header.frame_id = 'map'
        result.path.header.stamp = self.get_clock().now().to_msg()
        result.path.poses = [handle.request.goal]
        handle.succeed()
        return result

    def publish_grid(self):
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        stamp = self.get_clock().now()
        if self.grid_stale:
            stamp -= Duration(seconds=20)
        grid.header.stamp = stamp.to_msg()
        grid.info.resolution = 0.05
        grid.info.width = 200
        grid.info.height = 200
        grid.info.origin.position.x = -5.0
        grid.info.origin.position.y = -5.0
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * 40000
        self.grid_pub.publish(grid)

    def publish_sensors(self):
        now = self.get_clock().now()
        sensor_at = time.monotonic()
        dt = min(0.1, sensor_at - self.last_sensor_at)
        self.last_sensor_at = sensor_at
        self.pose = (self.pose[0], self.pose[1], self.pose[2] + self.angular_rate * dt)
        x, y, yaw = self.pose
        transform = TransformStamped()
        transform.header.frame_id = 'map'
        transform.header.stamp = now.to_msg()
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = math.sin(yaw / 2)
        transform.transform.rotation.w = math.cos(yaw / 2)
        self.tf_pub.sendTransform(transform)
        fire = FireObservation()
        fire.header.frame_id = 'map'
        fire.header.stamp = now.to_msg()
        fire.id = 'fixture_fire'
        fire.position.x = self.target_x
        fire.position.z = 0.4
        fire.position_valid = True
        fire.confidence = 0.9
        self.fire_pub.publish(fire)
        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.header.stamp = now.to_msg()
        odom.child_frame_id = 'base_link'
        odom.twist.twist.angular.z = self.angular_rate
        self.odom_pub.publish(odom)
        base_status = LimoStatus()
        base_status.header.stamp = now.to_msg()
        base_status.motion_mode = self.motion_mode
        self.base_status_pub.publish(base_status)
        status = self.bridge_status.copy()
        status['stamp'] = now.nanoseconds * 1e-9
        if self.send_malformed:
            status.update(acked=True, sequence='bad', command_deg=None)
        self.bridge_pub.publish(String(data=json.dumps(status)))
        if self.feedback_mode != 'none':
            joint = JointState()
            joint.header.stamp = now.to_msg()
            if self.feedback_mode == 'stale':
                joint.header.stamp = (now - Duration(seconds=20)).to_msg()
            elif self.feedback_mode == 'precommand':
                joint.header.stamp = Time(nanoseconds=self.feedback_stamp_ns).to_msg()
            joint.name = ['launcher_pitch_joint']
            angle = self.pitch_commands[-1] if self.pitch_commands else 18.4
            joint.position = [math.radians(angle)]
            self.joint_pub.publish(joint)

    def close(self):
        self.planner.destroy()
        self.navigator.destroy()


class RosFixture(unittest.TestCase):
    motion_enabled = True
    mount_calibrated = True
    require_pitch_feedback = False
    auto_fire_enabled = False
    fire_on_pitch_ack = False
    delay_navigation = False
    yaw_alignment_mode = 'nav2'
    yaw_rate_drive_mode_verified = False
    nav_yaw_bias_deg = 0.0

    def setUp(self):
        args = ['--ros-args', '-p', f'motion_enabled:={str(self.motion_enabled).lower()}',
                '-p', f'mount_calibrated:={str(self.mount_calibrated).lower()}',
                '-p', f'require_pitch_feedback:={str(self.require_pitch_feedback).lower()}',
                '-p', f'auto_fire_enabled:={str(self.auto_fire_enabled).lower()}',
                '-p', f'fire_on_pitch_ack:={str(self.fire_on_pitch_ack).lower()}',
                '-p', f'yaw_alignment_mode:={self.yaw_alignment_mode}',
                '-p', f'yaw_rate_drive_mode_verified:={str(self.yaw_rate_drive_mode_verified).lower()}',
                '-p', 'stop_dwell_s:=0.1',
                '-p', 'plan_timeout_s:=2.0', '-p', 'navigation_timeout_s:=6.0',
                '-p', 'reacquire_timeout_s:=3.0', '-p', 'pitch_timeout_s:=2.0']
        rclpy.init(args=args)
        self.controller = ApproachAim()
        self.robot = FakeRobot(self.delay_navigation, self.nav_yaw_bias_deg)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.controller)
        self.executor.add_node(self.robot)
        self.spin_error = None

        def spin():
            try:
                self.executor.spin()
            except Exception as exc:
                self.spin_error = exc

        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()
        self.wait_for(lambda: self.robot.start_client.service_is_ready()
                      and self.controller.latest_fire is not None
                      and self.controller.grid is not None
                      and self.controller.nav.server_is_ready()
                      and self.controller.planner.server_is_ready(), 6)
        # Allow the independent TF listener callback to receive a fresh transform.
        time.sleep(0.15)

    def tearDown(self):
        self.controller.close()
        self.executor.shutdown(timeout_sec=6)
        self.thread.join(timeout=6)
        self.robot.close()
        self.robot.destroy_node()
        self.controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.spin_error:
            raise self.spin_error

    def wait_for(self, predicate, timeout=6):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.spin_error:
                raise self.spin_error
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f'Timed out after {timeout}s. Controller state {self.controller.phase}')

    def call(self, client):
        future = client.call_async(Trigger.Request())
        self.wait_for(future.done, 3)
        return future.result()


class ApproachAimRosTest(RosFixture):
    def test_nominal_ack_is_unmeasured(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'PITCH_ACKNOWLEDGED', 8)
        self.assertEqual(len(self.robot.navigation_goals), 2)
        self.assertEqual(len(self.robot.pitch_commands), 1)
        self.wait_for(lambda: any(s['state'] == 'PITCH_ACKNOWLEDGED' for s in self.robot.statuses))
        terminal = next(s for s in reversed(self.robot.statuses) if s['state'] == 'PITCH_ACKNOWLEDGED')
        self.assertFalse(terminal['measured'])
        self.assertFalse(any(s['state'] == 'AIM_VERIFIED' for s in self.robot.statuses))

    def test_malformed_bridge_status_does_not_crash(self):
        self.robot.send_malformed = True
        time.sleep(0.3)
        self.assertIsNone(self.spin_error)
        self.robot.send_malformed = False
        self.wait_for(lambda: self.controller.bridge.get('state') == 'ready')
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'PITCH_ACKNOWLEDGED', 8)

    def test_stale_costmap_rejected(self):
        self.robot.grid_stale = True
        # Wait longer than the configured freshness threshold so the preceding
        # valid sample expires even if the implementation drops stale updates.
        time.sleep(self.controller.cfg['costmap_freshness_s'] + 0.4)
        result = self.call(self.robot.start_client)
        self.assertFalse(result.success)
        self.assertIn('costmap', result.message.lower())
        self.assertEqual(self.robot.navigation_goals, [])
        self.assertEqual(self.robot.pitch_commands, [])


class PreviewRosTest(RosFixture):
    motion_enabled = False

    def test_preview_has_no_actuator_commands(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'PREVIEW_READY')
        self.assertEqual(self.robot.navigation_goals, [])
        self.assertEqual(self.robot.pitch_commands, [])


class AutoFireRosTest(RosFixture):
    auto_fire_enabled = True
    fire_on_pitch_ack = True

    def test_acknowledged_pitch_requests_one_firmware_shot(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'SHOT_COMMAND_ACKNOWLEDGED', 8)
        self.assertEqual(self.robot.shot_requests, 1)
        self.assertEqual(len(self.robot.pitch_commands), 1)
        self.wait_for(lambda: any(s['state'] == 'SHOT_COMMAND_ACKNOWLEDGED' for s in self.robot.statuses))
        terminal = next(s for s in self.robot.statuses if s['state'] == 'SHOT_COMMAND_ACKNOWLEDGED')
        self.assertFalse(terminal['measured'])


class AutoFireFeedbackRosTest(RosFixture):
    auto_fire_enabled = True
    require_pitch_feedback = True

    def test_measured_pitch_requests_one_firmware_shot(self):
        self.robot.feedback_mode = 'fresh'
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'SHOT_COMMAND_ACKNOWLEDGED', 8)
        self.assertEqual(self.robot.shot_requests, 1)


class CancellationRosTest(RosFixture):
    delay_navigation = True

    def test_cancel_before_late_goal_acceptance(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(self.robot.nav_goal_seen.is_set)
        self.assertTrue(self.call(self.robot.cancel_client).success)
        self.assertFalse(self.call(self.robot.start_client).success)
        self.wait_for(self.robot.nav_cancel_seen.is_set)
        self.wait_for(lambda: self.controller.phase == 'CANCELLED', 6)
        self.assertEqual(self.robot.pitch_commands, [])

    def test_cumulative_target_shift_cancels_without_resetting_dwell(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(self.robot.nav_goal_seen.is_set)
        self.robot.target_x = 2.15
        self.wait_for(lambda: abs(self.controller.target.position.x - 2.15) < 0.001)
        self.robot.target_x = 2.3
        self.wait_for(lambda: self.controller.phase == 'FAILED', 6)
        self.assertEqual(self.robot.pitch_commands, [])


class PitchFeedbackRosTest(RosFixture):
    require_pitch_feedback = True

    def test_stale_and_precommand_feedback_require_fresh_measured_dwell(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'WAIT_PITCH', 6)
        self.robot.feedback_mode = 'stale'
        time.sleep(0.25)
        self.assertEqual(self.controller.phase, 'WAIT_PITCH')
        self.assertFalse(any(s['state'] == 'AIM_VERIFIED' for s in self.robot.statuses))
        # This sample is received after the command but identifies a measurement
        # made 10 ms before it. Receipt freshness alone must not authorize it.
        self.robot.feedback_stamp_ns = self.controller.pitch_sent_ns - 10_000_000
        self.robot.feedback_mode = 'precommand'
        time.sleep(0.25)
        self.assertEqual(self.controller.phase, 'WAIT_PITCH')
        self.assertFalse(any(s['state'] == 'AIM_VERIFIED' for s in self.robot.statuses))
        self.robot.feedback_mode = 'fresh'
        self.wait_for(lambda: self.controller.phase == 'AIM_VERIFIED', 2)
        self.wait_for(lambda: any(s['state'] == 'AIM_VERIFIED' for s in self.robot.statuses))
        terminal = next(s for s in reversed(self.robot.statuses) if s['state'] == 'AIM_VERIFIED')
        self.assertTrue(terminal['measured'])

    def test_missing_feedback_times_out_without_measured_success(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'FAILED', 8)
        self.assertEqual(len(self.robot.pitch_commands), 1)
        self.assertFalse(any(s['state'] == 'AIM_VERIFIED' for s in self.robot.statuses))
        self.assertFalse(any(s['state'] == 'PITCH_ACKNOWLEDGED' for s in self.robot.statuses))


class UncalibratedMountRosTest(RosFixture):
    mount_calibrated = False

    def test_enabled_motion_requires_calibrated_mount(self):
        response = self.call(self.robot.start_client)
        self.assertFalse(response.success)
        self.assertIn('mount', response.message.lower())
        self.assertEqual(self.robot.navigation_goals, [])
        self.assertEqual(self.robot.pitch_commands, [])


class YawRateRosTest(RosFixture):
    yaw_alignment_mode = 'yaw_rate'
    yaw_rate_drive_mode_verified = True
    nav_yaw_bias_deg = 12.0

    def test_robot_rotates_to_target_then_commands_pitch(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: self.controller.phase == 'PITCH_ACKNOWLEDGED', 9)
        self.assertEqual(len(self.robot.navigation_goals), 1)
        self.assertTrue(any(rate < -0.01 for rate in self.robot.yaw_commands))
        self.assertAlmostEqual(self.robot.yaw_commands[-1], 0.0)
        self.assertLess(abs(math.degrees(self.robot.pose[2])), 3.0)
        self.assertEqual(len(self.robot.pitch_commands), 1)

    def test_cancel_during_rotation_sends_zero_and_skips_pitch(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: any(abs(rate) > 0.01 for rate in self.robot.yaw_commands), 6)
        self.assertTrue(self.call(self.robot.cancel_client).success)
        self.wait_for(lambda: self.controller.phase == 'CANCELLED', 5)
        self.assertEqual(self.robot.yaw_commands[-1], 0.0)
        self.assertEqual(self.robot.pitch_commands, [])

    def test_drive_mode_change_stops_rotation(self):
        self.assertTrue(self.call(self.robot.start_client).success)
        self.wait_for(lambda: any(abs(rate) > 0.01 for rate in self.robot.yaw_commands), 6)
        self.robot.motion_mode = 1
        self.wait_for(lambda: self.controller.phase == 'FAILED', 5)
        self.assertEqual(self.robot.yaw_commands[-1], 0.0)
        self.assertEqual(self.robot.pitch_commands, [])


class UnverifiedYawDriveRosTest(RosFixture):
    yaw_alignment_mode = 'yaw_rate'

    def test_pure_rotation_requires_verified_drive_mode(self):
        response = self.call(self.robot.start_client)
        self.assertFalse(response.success)
        self.assertIn('drive mode', response.message)
        self.assertEqual(self.robot.navigation_goals, [])
        self.assertEqual(self.robot.yaw_commands, [])


if __name__ == '__main__':
    unittest.main()
