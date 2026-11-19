"""One requested approach, body-yaw alignment and single-axis pitch command."""
import json
import math
import signal
import time
from dataclasses import replace

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from limo_fire_msgs.msg import FireObservation
from limo_msgs.msg import LimoStatus
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .aim_geometry import (approach_candidates, ballistic_pitch, body_yaw_for_target,
                           heading_error, pitch_for_target, yaw_rate_command)
from .navigation_checks import candidate_clear


class ApproachAim(Node):
    def __init__(self):
        super().__init__('launcher_approach_aim')
        defaults = {
            'motion_enabled': False, 'mount_calibrated': False,
            'map_frame': 'map', 'base_frame': 'base_link',
            'mount_xyz_m': [0.0, 0.0, 0.0], 'mount_yaw_deg': 0.0,
            'desired_distance_m': 1.2, 'distance_tolerance_m': 0.15,
            'pitch_min_deg': 5.0, 'pitch_max_deg': 30.0,
            'aiming_model': 'line_of_sight', 'measured_launch_speed_mps': 0.0,
            'gravity_mps2': 9.81,
            'candidate_offsets_deg': [0.0, 15.0, -15.0, 30.0, -30.0, 60.0, -60.0],
            'robot_radius_m': 0.22, 'footprint_margin_m': 0.03,
            'blocked_cost': 65, 'sight_target_margin_m': 0.15,
            'target_freshness_s': 1.0, 'target_shift_limit_m': 0.25,
            'min_confidence': 0.5, 'sensor_freshness_s': 0.5,
            'costmap_freshness_s': 3.0, 'tf_freshness_s': 0.5,
            'max_base_tilt_deg': 5.0,
            'plan_timeout_s': 5.0, 'navigation_timeout_s': 60.0,
            'reacquire_timeout_s': 8.0, 'pitch_timeout_s': 5.0,
            'stop_dwell_s': 0.5, 'stop_linear_mps': 0.01, 'stop_angular_rps': 0.02,
            'yaw_tolerance_deg': 3.0, 'pitch_tolerance_deg': 2.0,
            'yaw_alignment_mode': 'nav2', 'yaw_rate_drive_mode_verified': False,
            'yaw_velocity_topic': '/cmd_vel', 'yaw_rate_gain': 1.2,
            'limo_status_topic': '/limo_status',
            'yaw_rate_min_rps': 0.05, 'yaw_rate_max_rps': 0.25,
            'yaw_rate_timeout_s': 8.0, 'yaw_rate_max_error_deg': 45.0,
            'require_pitch_feedback': False, 'pitch_joint': 'launcher_pitch_joint',
            'pitch_feedback_dwell_s': 0.2,
            'fire_topic': '/fire/observations', 'odom_topic': '/odometry/filtered',
            'costmap_topic': '/global_costmap/costmap',
            'navigate_action': '/navigate_to_pose', 'planner_action': '/compute_path_to_pose',
            'planner_id': 'GridBased', 'pitch_command_topic': '/launcher/pitch_command',
            'pitch_status_topic': '/launcher/pitch_status',
            'pitch_feedback_topic': '/launcher/joint_states',
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.cfg = {key: self.get_parameter(key).value for key in defaults}
        self._validate()
        # Hardware configuration is fixed for a run. Restart to change calibration or enable motion.
        self.add_on_set_parameters_callback(self._parameter_update)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, self.cfg['navigate_action'])
        self.planner = ActionClient(self, ComputePathToPose, self.cfg['planner_action'])
        self.status_pub = self.create_publisher(
            String, '/launcher/approach_aim_status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pitch_pub = self.create_publisher(Float64, self.cfg['pitch_command_topic'], 10)
        self.yaw_pub = self.create_publisher(Twist, self.cfg['yaw_velocity_topic'], 10)
        self.create_subscription(FireObservation, self.cfg['fire_topic'], self._fire, 10)
        self.create_subscription(Odometry, self.cfg['odom_topic'], self._odom, qos_profile_sensor_data)
        self.create_subscription(LimoStatus, self.cfg['limo_status_topic'], self._limo_status, qos_profile_sensor_data)
        grid_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, self.cfg['costmap_topic'], self._grid, grid_qos)
        self.create_subscription(String, self.cfg['pitch_status_topic'], self._bridge, 10)
        self.create_subscription(JointState, self.cfg['pitch_feedback_topic'], self._joint, 10)
        self.create_service(Trigger, '/launcher/approach_and_aim', self._start)
        self.create_service(Trigger, '/launcher/cancel', self._cancel)
        self.phase = 'IDLE'
        self.active = False
        self.epoch = 0
        self.goal_handle = None
        self.handles = []
        self.pending_requests = 0
        self.cancel_terminal = ('CANCELLED', '')
        self.latest_fire = None
        self.target = None
        self.grid = None
        self.grid_at = 0.0
        self.odom = None
        self.odom_at = 0.0
        self.base_status = None
        self.base_status_at = 0.0
        self.bridge = {}
        self.bridge_at = 0.0
        self.joint = None
        self.joint_at = 0.0
        self.phase_at = time.monotonic()
        self.stop_since = None
        self.yaw_owned = False
        self.create_timer(0.1, self._tick)
        self.last_status = None
        self.create_timer(1.0, self._heartbeat)
        self._state('IDLE', 'Call /launcher/approach_and_aim for one run')

    def _validate(self):
        for key, value in self.cfg.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if len(self.cfg['mount_xyz_m']) != 3 or not all(map(math.isfinite, self.cfg['mount_xyz_m'])):
            raise ValueError('mount_xyz_m must contain three finite measured values')
        for key in ('desired_distance_m', 'distance_tolerance_m', 'robot_radius_m',
                    'target_freshness_s', 'sensor_freshness_s', 'costmap_freshness_s',
                    'tf_freshness_s', 'plan_timeout_s', 'navigation_timeout_s',
                    'reacquire_timeout_s', 'pitch_timeout_s', 'stop_dwell_s', 'pitch_feedback_dwell_s',
                    'target_shift_limit_m', 'yaw_tolerance_deg', 'pitch_tolerance_deg', 'max_base_tilt_deg'):
            if self.cfg[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if not 5 <= self.cfg['pitch_min_deg'] < self.cfg['pitch_max_deg'] <= 30:
            raise ValueError('pitch limits must stay inside firmware range 5..30 degrees')
        if self.cfg['aiming_model'] not in ('line_of_sight', 'ballistic'):
            raise ValueError('aiming_model must be line_of_sight or ballistic')
        if self.cfg['gravity_mps2'] <= 0 or self.cfg['measured_launch_speed_mps'] < 0:
            raise ValueError('invalid gravity or measured launch speed')
        if self.cfg['distance_tolerance_m'] >= self.cfg['desired_distance_m']:
            raise ValueError('distance tolerance must be smaller than desired distance')
        if self.cfg['aiming_model'] == 'ballistic' and self.cfg['measured_launch_speed_mps'] <= 0:
            raise ValueError('ballistic aiming requires a positive measured_launch_speed_mps')
        if not 0 < self.cfg['blocked_cost'] <= 100 or not 0 <= self.cfg['min_confidence'] <= 1:
            raise ValueError('invalid occupancy threshold or detection confidence')
        if self.cfg['footprint_margin_m'] < 0 or self.cfg['sight_target_margin_m'] < 0:
            raise ValueError('clearance margins must be nonnegative')
        if (self.cfg['stop_linear_mps'] < 0 or self.cfg['stop_angular_rps'] < 0
                or self.cfg['sight_target_margin_m'] >= self.cfg['desired_distance_m']):
            raise ValueError('invalid stop thresholds or target sight margin')
        if (not self.cfg['map_frame'] or not self.cfg['base_frame']
                or self.cfg['map_frame'] == self.cfg['base_frame']):
            raise ValueError('map_frame and base_frame must be distinct nonempty frames')
        if not self.cfg['candidate_offsets_deg'] or not all(map(math.isfinite, self.cfg['candidate_offsets_deg'])):
            raise ValueError('candidate_offsets_deg must contain finite values')
        if self.cfg['yaw_alignment_mode'] not in ('nav2', 'yaw_rate'):
            raise ValueError('yaw_alignment_mode must be nav2 or yaw_rate')
        if not self.cfg['yaw_velocity_topic'] or not self.cfg['limo_status_topic']:
            raise ValueError('yaw velocity and LIMO status topics must be nonempty')
        yaw_limits = ('yaw_rate_gain', 'yaw_rate_min_rps', 'yaw_rate_max_rps',
                      'yaw_rate_timeout_s', 'yaw_rate_max_error_deg')
        if any(not math.isfinite(self.cfg[key]) or self.cfg[key] <= 0 for key in yaw_limits):
            raise ValueError('yaw rate limits must be positive and finite')
        if (self.cfg['yaw_rate_min_rps'] > self.cfg['yaw_rate_max_rps']
                or self.cfg['yaw_rate_max_error_deg'] <= self.cfg['yaw_tolerance_deg']):
            raise ValueError('invalid yaw rate bounds')

    def _parameter_update(self, parameters):
        if any(p.name in self.cfg for p in parameters):
            return SetParametersResult(successful=False, reason='Restart the node to change hardware configuration')
        return SetParametersResult(successful=True)

    def _stamp_fresh(self, stamp, maximum):
        age = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds * 1e-9
        return -0.1 <= age <= maximum

    def _fire(self, msg):
        if self.phase in ('CANCELLING', 'STOP_UNCONFIRMED'):
            return
        valid = (msg.position_valid and msg.header.frame_id == self.cfg['map_frame']
                 and math.isfinite(msg.confidence) and msg.confidence >= self.cfg['min_confidence']
                 and all(map(math.isfinite, (msg.position.x, msg.position.y, msg.position.z)))
                 and self._stamp_fresh(msg.header.stamp, self.cfg['target_freshness_s']))
        if not valid:
            return
        if self.active and self.target:
            if msg.id != self.target.id:
                return
            displacement = math.dist(self._xyz(msg), self.initial_target_xyz)
            if displacement > self.cfg['target_shift_limit_m']:
                self._fail('Target moved beyond the configured limit')
                return
            self.target = msg
        self.latest_fire = msg

    @staticmethod
    def _xyz(msg):
        return (msg.position.x, msg.position.y, msg.position.z)

    def _odom(self, msg):
        self.odom, self.odom_at = msg, time.monotonic()

    def _limo_status(self, msg):
        self.base_status, self.base_status_at = msg, time.monotonic()

    def _yaw_drive_ready(self):
        return (self.base_status is not None
                and time.monotonic() - self.base_status_at <= self.cfg['sensor_freshness_s']
                and self._stamp_fresh(self.base_status.header.stamp, self.cfg['sensor_freshness_s'])
                and self.base_status.motion_mode in (0, 2))

    def _grid(self, msg):
        if (msg.header.frame_id == self.cfg['map_frame']
                and self._stamp_fresh(msg.header.stamp, self.cfg['costmap_freshness_s'])
                and msg.info.width > 0 and msg.info.height > 0
                and len(msg.data) == msg.info.width * msg.info.height
                and math.isfinite(msg.info.resolution) and msg.info.resolution > 0
                and all(map(math.isfinite, (msg.info.origin.position.x, msg.info.origin.position.y,
                                            msg.info.origin.orientation.x, msg.info.origin.orientation.y,
                                            msg.info.origin.orientation.z, msg.info.origin.orientation.w)))):
            self.grid, self.grid_at = msg, time.monotonic()

    def _grid_fresh(self):
        return (self.grid is not None and time.monotonic() - self.grid_at <= self.cfg['costmap_freshness_s']
                and self._stamp_fresh(self.grid.header.stamp, self.cfg['costmap_freshness_s']))

    def _bridge(self, msg):
        try:
            value = json.loads(msg.data)
            valid = (isinstance(value, dict) and isinstance(value.get('state'), str)
                     and isinstance(value.get('sequence'), int) and not isinstance(value['sequence'], bool)
                     and value['sequence'] >= 0 and isinstance(value.get('acked'), bool))
            if valid and value['acked']:
                valid = all(isinstance(value.get(key), (int, float))
                            and not isinstance(value[key], bool) and math.isfinite(value[key])
                            for key in ('command_deg', 'command_stamp'))
            if valid:
                self.bridge, self.bridge_at = value, time.monotonic()
        except (ValueError, TypeError):
            pass

    def _joint(self, msg):
        if self.cfg['pitch_joint'] in msg.name:
            index = msg.name.index(self.cfg['pitch_joint'])
            if index < len(msg.position) and math.isfinite(msg.position[index]):
                self.joint = (msg.header.stamp, msg.position[index])
                self.joint_at = time.monotonic()

    def _pose(self):
        transform = self.tf.lookup_transform(self.cfg['map_frame'], self.cfg['base_frame'], Time())
        if not self._stamp_fresh(transform.header.stamp, self.cfg['tf_freshness_s']):
            raise ValueError('Robot localization TF is stale')
        p, q = transform.transform.translation, transform.transform.rotation
        norm = sum(value * value for value in (q.x, q.y, q.z, q.w))
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
            raise ValueError('Localization quaternion is invalid')
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        if max(abs(roll), abs(pitch)) > math.radians(self.cfg['max_base_tilt_deg']):
            raise ValueError('Base roll/pitch exceeds the planar aiming limit')
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        result = (p.x, p.y, p.z, yaw)
        if not all(map(math.isfinite, result)):
            raise ValueError('Robot pose is nonfinite')
        return result

    def _stopped(self):
        if not self.odom or time.monotonic() - self.odom_at > self.cfg['sensor_freshness_s']:
            return False
        if not self._stamp_fresh(self.odom.header.stamp, self.cfg['sensor_freshness_s']):
            return False
        twist = self.odom.twist.twist
        values = (twist.linear.x, twist.linear.y, twist.angular.z)
        return (all(map(math.isfinite, values))
                and math.hypot(*values[:2]) <= self.cfg['stop_linear_mps']
                and abs(values[2]) <= self.cfg['stop_angular_rps'])

    def _state(self, phase, reason='', **details):
        self.phase = phase
        self.phase_at = time.monotonic()
        report = dict(state=phase, reason=reason, motion_enabled=self.cfg['motion_enabled'],
                      stamp=self.get_clock().now().nanoseconds * 1e-9, **details)
        self.last_status = String(data=json.dumps(report, allow_nan=False))
        self.status_pub.publish(self.last_status)
        self.get_logger().info(json.dumps(report, allow_nan=False))

    def _heartbeat(self):
        if self.last_status is not None:
            self.status_pub.publish(self.last_status)

    def _start(self, request, response):
        del request
        if self.active:
            response.success, response.message = False, 'A run or cancellation is already active'
            return response
        try:
            if not self.latest_fire or not self._stamp_fresh(self.latest_fire.header.stamp, self.cfg['target_freshness_s']):
                raise ValueError('A fresh localized fire observation in map frame is required')
            if not self._grid_fresh():
                raise ValueError('A fresh global costmap is required')
            pose = self._pose()
            if not self._stopped():
                raise ValueError('Fresh stationary odometry is required before starting')
            if not self.planner.server_is_ready():
                raise ValueError('Nav2 ComputePathToPose is unavailable')
            if self.cfg['motion_enabled']:
                if not self.cfg['mount_calibrated']:
                    raise ValueError('Measure launcher mount and set mount_calibrated before enabling motion')
                if self.cfg['yaw_alignment_mode'] == 'yaw_rate' and not self.cfg['yaw_rate_drive_mode_verified']:
                    raise ValueError('Verify differential or mecanum drive mode before yaw-rate aiming')
                if self.cfg['yaw_alignment_mode'] == 'yaw_rate' and not self._yaw_drive_ready():
                    raise ValueError('Fresh LIMO status must report differential or mecanum drive mode')
                if not self.nav.server_is_ready():
                    raise ValueError('Nav2 NavigateToPose is unavailable')
                if time.monotonic() - self.bridge_at > 2.5 or self.bridge.get('state') not in ('ready', 'acknowledged'):
                    raise ValueError('An enabled, identified and ready pitch bridge is required')
            self.target = self.latest_fire
            self.initial_target_xyz = self._xyz(self.target)
            candidates = approach_candidates(
                self._xyz(self.target), pose, self.cfg['desired_distance_m'],
                self.cfg['mount_xyz_m'], math.radians(self.cfg['mount_yaw_deg']),
                tuple(map(math.radians, self.cfg['candidate_offsets_deg'])),
                math.radians(self.cfg['pitch_min_deg']), math.radians(self.cfg['pitch_max_deg']),
                self.cfg['measured_launch_speed_mps'] if self.cfg['aiming_model'] == 'ballistic' else None,
                self.cfg['gravity_mps2'])
            self.candidates = iter(candidates)
            self.active = True
            self.epoch += 1
            self._next_candidate()
            response.success, response.message = True, 'Run accepted, follow /launcher/approach_aim_status'
        except (ValueError, TransformException) as exc:
            response.success, response.message = False, str(exc)
        return response

    def _cancel(self, request, response):
        del request
        if self.active:
            self._begin_cancel('CANCELLED', 'Cancellation requested')
        response.success, response.message = True, 'Cancellation requested'
        return response

    def _fail(self, reason):
        self._begin_cancel('FAILED', reason)

    def _begin_cancel(self, terminal, reason):
        if self.phase in ('CANCELLING', 'STOP_UNCONFIRMED'):
            return
        self.epoch += 1
        if self.yaw_owned:
            self._publish_yaw(0.0)
        for handle in self.handles:
            handle.cancel_goal_async()
        self.goal_handle = None
        self.cancel_terminal = (terminal, reason)
        self.stop_since = None
        self._state('CANCELLING', reason)

    def _goal_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.cfg['map_frame']
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return pose

    def _next_candidate(self):
        if not self._grid_fresh():
            self._fail('Global costmap became stale')
            return
        for candidate in self.candidates:
            if candidate_clear(self.grid, candidate, self._xyz(self.target),
                               self.cfg['robot_radius_m'] + self.cfg['footprint_margin_m'],
                               self.cfg['blocked_cost'], self.cfg['sight_target_margin_m']):
                self.candidate = candidate
                goal = ComputePathToPose.Goal()
                goal.goal = self._goal_pose(candidate.base_x, candidate.base_y, candidate.base_yaw)
                goal.planner_id = self.cfg['planner_id']
                goal.use_start = False
                self._state('PLANNING', goal_x=candidate.base_x, goal_y=candidate.base_y)
                self._send(self.planner, goal, self._planned)
                return
        self._fail('No reachable candidate passed the footprint, sight and pitch checks')

    def _send(self, client, goal, completed):
        token = self.epoch
        self.pending_requests += 1
        try:
            future = client.send_goal_async(goal)
        except Exception as exc:
            self.pending_requests -= 1
            self._fail(f'Action could not be sent: {exc}')
            return

        def accepted(result):
            self.pending_requests -= 1
            try:
                handle = result.result()
                if not handle.accepted:
                    if token == self.epoch:
                        completed(None)
                    return
                self.handles.append(handle)

                def finished(result):
                    self.handles.remove(handle)
                    if token != self.epoch:
                        return
                    self.goal_handle = None
                    try:
                        completed(result.result())
                    except Exception as exc:
                        self._fail(f'Action result failed: {exc}')

                handle.get_result_async().add_done_callback(finished)
                if token != self.epoch:
                    handle.cancel_goal_async()
                    return
                self.goal_handle = handle
            except Exception as exc:
                if token == self.epoch:
                    self._fail(f'Action request failed: {exc}')

        future.add_done_callback(accepted)

    def _planned(self, result):
        if not result or result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.path.poses:
            self._next_candidate()
            return
        path = result.result.path
        end = path.poses[-1].pose.position
        if path.header.frame_id != self.cfg['map_frame'] or not math.isfinite(end.x + end.y):
            self._next_candidate()
            return
        if math.hypot(end.x - self.candidate.base_x, end.y - self.candidate.base_y) > self.cfg['distance_tolerance_m']:
            self._next_candidate()
            return
        if not self.cfg['motion_enabled']:
            self.active = False
            self._state('PREVIEW_READY', 'Candidate and path computed',
                        base_x=self.candidate.base_x, base_y=self.candidate.base_y,
                        base_yaw_deg=math.degrees(self.candidate.base_yaw),
                        pitch_deg=math.degrees(self.candidate.pitch_rad),
                        mount_calibrated=self.cfg['mount_calibrated'])
            return
        if not self._stamp_fresh(self.target.header.stamp, self.cfg['target_freshness_s']):
            self._fail('Target expired while planning')
            return
        if (not self._grid_fresh()
                or not candidate_clear(self.grid, self.candidate, self._xyz(self.target),
                                       self.cfg['robot_radius_m'] + self.cfg['footprint_margin_m'],
                                       self.cfg['blocked_cost'], self.cfg['sight_target_margin_m'])):
            self._fail('Candidate costmap clearance changed while planning')
            return
        self._navigate(self._goal_pose(self.candidate.base_x, self.candidate.base_y, self.candidate.base_yaw), 'APPROACHING')

    def _navigate(self, pose, phase):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._state(phase)
        self._send(self.nav, goal, self._arrived)

    def _publish_yaw(self, rate):
        command = Twist()
        command.angular.z = rate
        self.yaw_pub.publish(command)

    def _arrived(self, result):
        if not result or result.status != GoalStatus.STATUS_SUCCEEDED:
            self._fail('Nav2 movement failed or was rejected')
            return
        self.after_alignment = self.phase == 'ALIGNING'
        self.stop_since = None
        self._state('WAIT_STOP')

    def _target_ready(self):
        return (self.target and self._stamp_fresh(self.target.header.stamp, self.cfg['target_freshness_s'])
                and Time.from_msg(self.target.header.stamp).nanoseconds >= self.reacquire_ros_ns)

    def _geometry(self):
        pose = self._pose()
        target = self._xyz(self.target)
        mount = self.cfg['mount_xyz_m']
        c, s = math.cos(pose[3]), math.sin(pose[3])
        px, py = pose[0] + c * mount[0] - s * mount[1], pose[1] + s * mount[0] + c * mount[1]
        distance = math.hypot(target[0] - px, target[1] - py)
        pitch = math.degrees(pitch_for_target(target, pose, mount, math.radians(self.cfg['mount_yaw_deg'])))
        if self.cfg['aiming_model'] == 'ballistic':
            pitch = math.degrees(ballistic_pitch(distance, target[2] - pose[2] - mount[2],
                                                  self.cfg['measured_launch_speed_mps'], self.cfg['gravity_mps2']))
        error = heading_error(target, pose, mount, math.radians(self.cfg['mount_yaw_deg']))
        if abs(distance - self.cfg['desired_distance_m']) > self.cfg['distance_tolerance_m']:
            raise ValueError(f'Measured pivot-to-target range {distance:.3f}m is outside tolerance')
        if not self.cfg['pitch_min_deg'] <= pitch <= self.cfg['pitch_max_deg']:
            raise ValueError('Required geometric elevation is outside the pitch limits')
        return pose, pitch, error

    def _clear_at_pose(self, pose):
        mount = self.cfg['mount_xyz_m']
        c, s = math.cos(pose[3]), math.sin(pose[3])
        actual = replace(self.candidate, base_x=pose[0], base_y=pose[1], base_yaw=pose[3],
                         pivot_x=pose[0] + c * mount[0] - s * mount[1],
                         pivot_y=pose[1] + s * mount[0] + c * mount[1], pivot_z=pose[2] + mount[2])
        return self._grid_fresh() and candidate_clear(
            self.grid, actual, self._xyz(self.target),
            self.cfg['robot_radius_m'] + self.cfg['footprint_margin_m'],
            self.cfg['blocked_cost'], self.cfg['sight_target_margin_m'])

    def _tick(self):
        if not self.active:
            return
        if self.yaw_owned and self.phase != 'YAW_ALIGNING':
            self._publish_yaw(0.0)
        elapsed = time.monotonic() - self.phase_at
        try:
            if self.phase == 'PLANNING' and elapsed > self.cfg['plan_timeout_s']:
                self._fail('Path planning timed out')
            elif self.phase in ('APPROACHING', 'ALIGNING'):
                if elapsed > self.cfg['navigation_timeout_s']:
                    self._fail('Nav2 movement timed out')
            elif self.phase in ('WAIT_STOP', 'CANCELLING', 'STOP_UNCONFIRMED'):
                settled = self._stopped()
                if self.phase != 'WAIT_STOP':
                    settled = settled and not self.handles and self.pending_requests == 0
                if settled:
                    if self.stop_since is None:
                        self.stop_since = time.monotonic()
                else:
                    self.stop_since = None
                if self.stop_since and time.monotonic() - self.stop_since >= self.cfg['stop_dwell_s']:
                    if self.phase != 'WAIT_STOP':
                        self.active = False
                        self.yaw_owned = False
                        self._state(*self.cancel_terminal)
                    else:
                        self.yaw_owned = False
                        self.reacquire_ros_ns = self.get_clock().now().nanoseconds
                        self._state('REACQUIRING')
                elif elapsed > self.cfg['reacquire_timeout_s']:
                    if self.phase == 'WAIT_STOP':
                        self._fail('Stationary odometry could not be confirmed')
                    elif self.phase == 'CANCELLING':
                        self._state('STOP_UNCONFIRMED', 'Check robot stop and outstanding Nav2 goals')
            elif self.phase == 'REACQUIRING':
                if self._target_ready():
                    pose, pitch, error = self._geometry()
                    if not self.after_alignment:
                        if self.cfg['yaw_alignment_mode'] == 'nav2':
                            yaw = body_yaw_for_target(self._xyz(self.target), pose, self.cfg['mount_xyz_m'],
                                                      math.radians(self.cfg['mount_yaw_deg']))
                            self._navigate(self._goal_pose(pose[0], pose[1], yaw), 'ALIGNING')
                        else:
                            if not self._stopped() or self.handles or self.pending_requests:
                                raise ValueError('Nav2 must be finished and robot stopped before yaw-rate aiming')
                            if not self._yaw_drive_ready():
                                raise ValueError('LIMO drive mode changed before yaw-rate aiming')
                            if abs(math.degrees(error)) > self.cfg['yaw_rate_max_error_deg']:
                                raise ValueError('Initial yaw error exceeds yaw-rate alignment limit')
                            if not self._clear_at_pose(pose):
                                raise ValueError('Costmap clearance changed before yaw-rate aiming')
                            self.yaw_owned = True
                            self._state('YAW_ALIGNING', yaw_error_deg=math.degrees(error))
                    elif abs(math.degrees(error)) > self.cfg['yaw_tolerance_deg']:
                        self._fail('Measured body heading is outside the aiming tolerance')
                    else:
                        self._pitch(pitch)
                elif elapsed > self.cfg['reacquire_timeout_s']:
                    self._fail('Target was not reacquired after movement')
            elif self.phase == 'YAW_ALIGNING':
                if elapsed > self.cfg['yaw_rate_timeout_s']:
                    raise ValueError('Yaw-rate alignment timed out')
                if not self._target_ready() or not self.odom or time.monotonic() - self.odom_at > self.cfg['sensor_freshness_s']:
                    raise ValueError('Fresh target or odometry lost during yaw-rate alignment')
                if not self._stamp_fresh(self.odom.header.stamp, self.cfg['sensor_freshness_s']):
                    raise ValueError('Odometry timestamp is stale during yaw-rate alignment')
                if not self._yaw_drive_ready():
                    raise ValueError('LIMO drive mode changed during yaw-rate alignment')
                pose, pitch, error = self._geometry()
                if not self._clear_at_pose(pose):
                    raise ValueError('Costmap clearance changed during yaw-rate alignment')
                if abs(math.degrees(error)) > self.cfg['yaw_rate_max_error_deg']:
                    raise ValueError('Yaw error exceeded alignment limit')
                rate = yaw_rate_command(error, math.radians(self.cfg['yaw_tolerance_deg']),
                                        self.cfg['yaw_rate_gain'], self.cfg['yaw_rate_min_rps'],
                                        self.cfg['yaw_rate_max_rps'])
                self._publish_yaw(rate)
                if rate == 0.0:
                    self.after_alignment = True
                    self.stop_since = None
                    self._state('WAIT_STOP', yaw_error_deg=math.degrees(error))
            elif self.phase == 'WAIT_PITCH':
                self._check_pitch(elapsed)
        except (ValueError, TransformException) as exc:
            self._fail(str(exc))

    def _pitch(self, pitch):
        if not self._stopped():
            raise ValueError('Robot moved before pitch command')
        if time.monotonic() - self.bridge_at > 2.5 or self.bridge.get('state') not in ('ready', 'acknowledged'):
            raise ValueError('Pitch bridge is not ready')
        if not self._clear_at_pose(self._pose()):
            raise ValueError('Candidate clearance cannot be confirmed before pitch command')
        self.pitch_deg = round(pitch, 1)
        if not self.cfg['pitch_min_deg'] <= self.pitch_deg <= self.cfg['pitch_max_deg']:
            raise ValueError('Rounded firmware angle exceeds the configured pitch limits')
        self.pitch_baseline = self.bridge.get('sequence', 0)
        self.pitch_sent_ns = self.get_clock().now().nanoseconds
        self.pitch_sent_at = time.monotonic()
        self.pitch_feedback_since = None
        self.pitch_pub.publish(Float64(data=self.pitch_deg))
        self._state('WAIT_PITCH', pitch_deg=self.pitch_deg, aiming_model=self.cfg['aiming_model'])

    def _check_pitch(self, elapsed):
        if not self._stopped() or not self._target_ready():
            raise ValueError('Stationary robot or fresh target confirmation lost while aiming')
        pose, required_pitch, error = self._geometry()
        if not self._clear_at_pose(pose):
            raise ValueError('Costmap clearance changed while aiming')
        if abs(required_pitch - self.pitch_deg) > self.cfg['pitch_tolerance_deg']:
            raise ValueError('Required pitch changed while awaiting the actuator')
        if abs(math.degrees(error)) > self.cfg['yaw_tolerance_deg']:
            raise ValueError('Body heading changed while aiming')
        status = self.bridge
        if status.get('state') in ('error', 'rejected'):
            raise ValueError('Pitch bridge reported a fault')
        acknowledged = (time.monotonic() - self.bridge_at <= 2.5 and status.get('acked') is True
                        and status.get('sequence', 0) > self.pitch_baseline
                        and isinstance(status.get('command_deg'), (int, float))
                        and isinstance(status.get('command_stamp'), (int, float))
                        and abs(status['command_deg'] - self.pitch_deg) <= 0.051
                        and status['command_stamp'] >= self.pitch_sent_ns * 1e-9)
        if acknowledged and not self.cfg['require_pitch_feedback']:
            self.active = False
            self._state('PITCH_ACKNOWLEDGED', 'Firmware acknowledged the command', measured=False,
                        pitch_deg=self.pitch_deg, aiming_model=self.cfg['aiming_model'])
            return
        if acknowledged and self.joint and self.joint_at >= self.pitch_sent_at:
            stamp, angle = self.joint
            if (self._stamp_fresh(stamp, self.cfg['sensor_freshness_s'])
                    and Time.from_msg(stamp).nanoseconds >= self.pitch_sent_ns
                    and time.monotonic() - self.joint_at <= self.cfg['sensor_freshness_s']
                    and abs(math.degrees(angle) - self.pitch_deg) <= self.cfg['pitch_tolerance_deg']):
                if self.pitch_feedback_since is None:
                    self.pitch_feedback_since = time.monotonic()
                if time.monotonic() - self.pitch_feedback_since >= self.cfg['pitch_feedback_dwell_s']:
                    self.active = False
                    self._state('AIM_VERIFIED', 'Fresh measured pitch feedback is within tolerance', measured=True)
                    return
            else:
                self.pitch_feedback_since = None
        if elapsed > self.cfg['pitch_timeout_s']:
            raise ValueError('Pitch acknowledgement or required measured feedback timed out')

    def close(self):
        self.epoch += 1
        if self.yaw_owned:
            self._publish_yaw(0.0)
        for handle in self.handles:
            handle.cancel_goal_async()


def main(args=None):
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = ApproachAim()
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            deadline = time.monotonic() + 5.0
            while rclpy.ok() and (node.handles or node.pending_requests) and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.handles or node.pending_requests:
                node.get_logger().error('Nav2 cancellation is unconfirmed at shutdown. Check the robot stop.')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
