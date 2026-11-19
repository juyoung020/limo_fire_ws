"""Single-owner serial adapter for pitch and explicitly enabled launcher shots.

The firmware reports the commanded pitch. Its reply is not an encoder reading.
ROS imports are local to main so protocol helpers can be tested without ROS.
"""

import json
import math
import re
import time


def pitch_command(value, minimum=5.0, maximum=30.0):
    """Validate the firmware range and encode only its pitch command."""
    if not all(math.isfinite(v) for v in (value, minimum, maximum)):
        raise ValueError('pitch and limits must be finite')
    if not 5.0 <= minimum <= maximum <= 30.0:
        raise ValueError('limits must be a subset of the firmware range 5..30 degrees')
    if not minimum <= value <= maximum:
        raise ValueError(f'pitch must be within {minimum}..{maximum} degrees')
    encoded = f'a{value:.1f}\n'
    commanded = float(encoded[1:].strip())
    if not minimum <= commanded <= maximum:
        raise ValueError('pitch rounded to firmware precision exceeds configured limits')
    return commanded, encoded.encode('ascii')


def parse_pitch_reply(line):
    """Return the acknowledged command, None for unrelated firmware output."""
    line = line.strip()
    if not line.startswith('pitch'):
        return None
    match = re.fullmatch(r'pitch\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))', line)
    if match is None:
        raise ValueError('malformed pitch reply')
    value = float(match.group(1))
    if not math.isfinite(value) or not 5.0 <= value <= 30.0:
        raise ValueError('firmware reply is outside the pitch range')
    return value


def matching_pitch_reply(line, commanded):
    value = parse_pitch_reply(line)
    if value is None:
        return False
    if abs(value - commanded) > 0.051:
        raise ValueError(f'pitch reply {value} does not match command {commanded}')
    return True


def launcher_startup_banner(line):
    """Recognize resets; the generic Uno banner alone cannot identify firmware."""
    line = line.strip()
    return line == 'READY' or line.startswith('launcher_test ready')


def launcher_identity(line):
    """Recognize this firmware before issuing a command shared by older sketches."""
    line = line.strip()
    if line.startswith('launcher_test ready'):
        return True
    number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))'
    prefix = rf'lever {number} \(rest {number} cock {number} fire {number}\) '
    esp32_pattern = prefix + rf'\| pitch {number} deg, tilt servo {number}'
    guarded_pattern = (rf'lever {number} \(rest {number} cock {number} fire {number}, '
                       rf'(?:ON|OFF)\) \| pitch {number} deg, tilt servo {number} \((?:ON|OFF)\)')
    uno_pattern = prefix + rf'\| pitch {number} \| tilt servo {number} \| targets OK'
    guarded = re.fullmatch(guarded_pattern, line)
    match = guarded or re.fullmatch(esp32_pattern, line) or re.fullmatch(uno_pattern, line)
    if match is None:
        return False
    values = tuple(float(value) for value in match.groups())
    return (all(math.isfinite(value) for value in values)
            and all(0 <= value <= 180 for value in values[:4])
            and 5 <= values[4] <= 30 and 0 <= values[5] <= (270 if guarded else 180))


def guarded_pitch_firmware(line):
    """The new ESP32 starts with PWM off and requires an e5 acknowledgement."""
    return ((line.startswith('launcher_test ready') and 'PWM OFF' in line)
            or (launcher_identity(line) and 'tilt servo ' in line
                and (line.endswith('(ON)') or line.endswith('(OFF)'))))


SHOT_REPLIES = {'cock': ('cocked',), 'fire': ('fired',),
                'shot': ('shot: cock', 'shot: fire', 'done')}
SHOT_BYTES = {'cock': b'c\n', 'fire': b'f\n', 'shot': b's\n'}
MOTION_REPLIES = frozenset(line for replies in SHOT_REPLIES.values() for line in replies)


def shot_reply_index(command, index, line):
    """Advance only on the next exact firmware reply; never infer physical motion."""
    expected = SHOT_REPLIES[command]
    if line.startswith('blocked:') or line.endswith('failed'):
        raise ValueError(line)
    if line == expected[index]:
        return index + 1
    if line in MOTION_REPLIES:
        raise ValueError(f'out-of-order launcher reply: {line}')
    return index


def main(args=None):
    import rclpy
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64, String
    from std_srvs.srv import Trigger

    class PitchSerialBridge(Node):
        def __init__(self):
            super().__init__('pitch_serial_bridge')
            readonly = ParameterDescriptor(read_only=True)

            def parameter(name, default):
                return self.declare_parameter(name, default, readonly).value

            self.enabled = parameter('enabled', False)
            self.port_name = parameter('serial_port', '')
            baud = parameter('baud_rate', 115200)
            self.minimum = parameter('min_pitch_deg', 5.0)
            self.maximum = parameter('max_pitch_deg', 30.0)
            self.ack_timeout = parameter('ack_timeout_s', 3.0)
            self.command_interval = parameter('min_command_interval_s', 0.2)
            self.firing_enabled = parameter('firing_enabled', False)
            self.shot_timeout = parameter('shot_timeout_s', 8.0)
            startup_delay = parameter('startup_delay_s', 2.0)
            startup_timeout = parameter('startup_timeout_s', 8.0)
            command_topic = parameter('command_topic', '/launcher/pitch_command')
            status_topic = parameter('status_topic', '/launcher/pitch_status')
            pitch_command(self.minimum, self.minimum, self.maximum)
            if (baud <= 0 or not math.isfinite(self.ack_timeout) or self.ack_timeout <= 0
                    or not math.isfinite(self.command_interval) or self.command_interval < 0
                    or not math.isfinite(startup_delay) or startup_delay < 0
                    or not math.isfinite(startup_timeout) or startup_timeout <= startup_delay
                    or not math.isfinite(self.shot_timeout) or self.shot_timeout <= 0):
                raise ValueError('invalid serial timing or baud configuration')

            status_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                    reliability=ReliabilityPolicy.RELIABLE)
            self.status_publisher = self.create_publisher(String, status_topic, status_qos)
            self.command_subscription = self.create_subscription(Float64, command_topic, self.command, 10)
            self.shot_status_publisher = self.create_publisher(String, '/launcher/shot_status', status_qos)
            for name in SHOT_BYTES:
                self.create_service(Trigger, '/launcher/' + name,
                                    lambda request, response, action=name: self.shot_service(action, response))
            self.port = None
            self.fault = False
            self.sequence = 0
            self.pending = None
            self.shot_pending = None
            self.shot_sequence = 0
            self.cocked = False
            self.lever_enabled = False
            self.guarded_firmware_seen = False
            self.receive_buffer = bytearray()
            self.last_sent = -math.inf
            self.ready_at = time.monotonic() + startup_delay
            self.startup_deadline = time.monotonic() + startup_timeout
            self.startup_reported = False
            self.identity_seen = False
            self.identity_query_sent = False
            self.pitch_enable_required = False
            self.pitch_enable_sent_at = None
            self.latest_status = None
            self.last_status_sent = -math.inf
            self.timer = self.create_timer(0.02, self.poll)
            if not self.enabled:
                self.publish_status('disabled', reason='enabled=false; serial port unopened')
                return
            if not self.port_name:
                self.fail('serial_port is required when enabled=true')
                return
            try:
                import serial
                self.port = serial.Serial(self.port_name, baudrate=baud, timeout=0,
                                          write_timeout=0.1, exclusive=True)
                self.publish_status('starting', reason='waiting for firmware startup')
            except Exception as exc:
                self.fail(f'cannot open serial port: {exc}')

        def now_seconds(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def publish_status(self, state, command=None, acked=False, reason=''):
            status = dict(state=state, stamp=self.now_seconds(), acked=acked,
                          measured=False, firing_enabled=self.firing_enabled, reason=reason)
            status.update(command or dict(sequence=self.sequence, command_deg=None, command_stamp=None))
            self.latest_status = status
            self.emit_status()

        def emit_status(self):
            self.latest_status['stamp'] = self.now_seconds()
            message = String()
            message.data = json.dumps(self.latest_status, allow_nan=False)
            self.status_publisher.publish(message)
            self.last_status_sent = time.monotonic()

        def fail(self, reason):
            self.fault = True
            self.publish_status('error', self.pending, reason=reason)
            self.pending = None
            if self.shot_pending is not None:
                self.publish_shot('error', reason)
                self.shot_pending = None
            self.cocked = False
            self.get_logger().error(reason)

        def publish_shot(self, state, reason=''):
            pending = self.shot_pending or {}
            message = String()
            message.data = json.dumps(dict(state=state, reason=reason,
                                           command=pending.get('command'),
                                           sequence=pending.get('sequence', self.shot_sequence),
                                           stamp=self.now_seconds(), measured=False), allow_nan=False)
            self.shot_status_publisher.publish(message)

        def shot_service(self, action, response):
            if not self.firing_enabled:
                response.success, response.message = False, 'firing_enabled=false'
            elif not self.enabled or self.port is None or self.fault or not self.startup_reported:
                response.success, response.message = False, 'identified launcher serial port is unavailable'
            elif not self.guarded_firmware_seen:
                response.success, response.message = False, 'guarded ESP32 launcher firmware is required'
            elif self.pending is not None or self.shot_pending is not None:
                response.success, response.message = False, 'another serial command is pending'
            elif action == 'fire' and not self.cocked:
                response.success, response.message = False, 'cock must be acknowledged before fire'
            elif action in ('cock', 'shot') and self.cocked:
                response.success, response.message = False, 'launcher is already cocked; fire or inspect before re-cocking'
            else:
                self.shot_sequence += 1
                self.shot_pending = dict(command=action, sequence=self.shot_sequence,
                                         index=0, started=time.monotonic(), enabling=not self.lever_enabled)
                try:
                    self.receive_buffer.clear()
                    self.port.reset_input_buffer()
                    self._write_shot(b'e4\n' if not self.lever_enabled else SHOT_BYTES[action])
                    self.publish_shot('accepted')
                    response.success = True
                    response.message = (f'accepted sequence={self.shot_sequence}; '
                                        'watch /launcher/shot_status for completion')
                except Exception as exc:
                    self.fail(f'launcher serial write failed: {exc}')
                    response.success, response.message = False, str(exc)
            return response

        def _write_shot(self, payload):
            if self.port.write(payload) != len(payload):
                raise OSError('incomplete launcher serial write')
            self.last_sent = time.monotonic()

        def _shot_line(self, line):
            pending = self.shot_pending
            if pending is None:
                return
            if pending['enabling']:
                if line == 'lever PWM ON':
                    self.lever_enabled = True
                    pending['enabling'] = False
                    pending['started'] = time.monotonic()
                    self._write_shot(SHOT_BYTES[pending['command']])
                    self.publish_shot('running')
                elif line.startswith('blocked:') or 'lever PWM attach failed' in line:
                    raise ValueError(line)
                elif line in MOTION_REPLIES:
                    raise ValueError(f'unexpected launcher motion reply before e4: {line}')
                return
            pending['index'] = shot_reply_index(pending['command'], pending['index'], line)
            if pending['index'] == len(SHOT_REPLIES[pending['command']]):
                self.cocked = pending['command'] == 'cock'
                self.publish_shot('completed', 'Firmware command acknowledged; physical motion unmeasured')
                self.shot_pending = None

        def command(self, message):
            self.sequence += 1
            request = dict(sequence=self.sequence,
                           command_deg=message.data if math.isfinite(message.data) else None,
                           command_stamp=self.now_seconds())
            if not self.enabled or self.port is None or self.fault:
                self.publish_status('rejected', request, reason='serial output unavailable')
                return
            if self.pending is not None or self.shot_pending is not None:
                self.publish_status('rejected', request, reason='previous command awaiting reply')
                return
            now = time.monotonic()
            if not self.startup_reported or now - self.last_sent < self.command_interval:
                self.publish_status('rejected', request, reason='firmware identity/startup or rate limit')
                return
            try:
                commanded, encoded = pitch_command(message.data, self.minimum, self.maximum)
                request['command_deg'] = commanded
            except ValueError as exc:
                self.publish_status('rejected', request, reason=str(exc))
                return
            self.pending = request
            try:
                # Clear complete or partial responses from earlier commands before transmission.
                self.receive_buffer.clear()
                self.port.reset_input_buffer()
                if self.port.write(encoded) != len(encoded):
                    raise OSError('incomplete serial write')
                self.last_sent = time.monotonic()
                self.publish_status('waiting_ack', self.pending)
            except Exception as exc:
                self.fail(f'serial write failed: {exc}')

        def poll(self):
            if self.latest_status and time.monotonic() - self.last_status_sent >= 1.0:
                self.emit_status()
            if not self.enabled or self.port is None or self.fault:
                return
            try:
                if self.pending and time.monotonic() - self.last_sent >= self.ack_timeout:
                    self.fail('pitch reply deadline exceeded; restart bridge before another command')
                    return
                if self.shot_pending and time.monotonic() - self.shot_pending['started'] >= self.shot_timeout:
                    self.fail('launcher reply deadline exceeded; inspect hardware before any retry')
                    return
                available = self.port.in_waiting
                if available:
                    data = self.port.read(min(available, 4096))
                    for byte in data:
                        if byte == 10:
                            line = self.receive_buffer.decode('ascii', errors='replace').strip()
                            self.receive_buffer.clear()
                            if self.identity_seen and launcher_startup_banner(line):
                                raise ValueError('firmware restarted; restart bridge to reestablish state')
                            if launcher_identity(line):
                                self.identity_seen = True
                                self.pitch_enable_required |= guarded_pitch_firmware(line)
                                self.guarded_firmware_seen |= guarded_pitch_firmware(line)
                            if self.pitch_enable_sent_at is not None:
                                if line == 'tilt PWM ON':
                                    self.pitch_enable_sent_at = None
                                    self.startup_reported = True
                                    self.publish_status('ready', reason='guarded ESP32 pitch PWM enabled')
                                elif 'tilt PWM attach failed' in line:
                                    raise ValueError('ESP32 pitch PWM attach failed')
                            if self.pending and matching_pitch_reply(line, self.pending['command_deg']):
                                self.publish_status('acknowledged', self.pending, acked=True)
                                self.pending = None
                            self._shot_line(line)
                        elif byte != 13:
                            self.receive_buffer.append(byte)
                            if len(self.receive_buffer) > 512:
                                raise ValueError('firmware response exceeds 512 bytes')
                if self.pitch_enable_sent_at is not None and time.monotonic() - self.pitch_enable_sent_at >= self.ack_timeout:
                    self.fail('pitch PWM enable reply deadline exceeded')
                    return
                if not self.startup_reported and self.pitch_enable_sent_at is None and time.monotonic() >= self.ready_at:
                    if self.identity_seen and self.pitch_enable_required:
                        if self.port.write(b'e5\n') != 3:
                            raise OSError('incomplete pitch PWM enable write')
                        self.pitch_enable_sent_at = time.monotonic()
                        self.publish_status('enabling', reason='waiting for guarded ESP32 e5 reply')
                    elif self.identity_seen:
                        self.startup_reported = True
                        self.publish_status('ready', reason='launcher firmware identity verified')
                    elif time.monotonic() >= self.startup_deadline:
                        self.fail('launcher firmware identity not received before startup deadline')
                    elif not self.identity_query_sent:
                        # The intended firmware returns status. Older one-letter test sketches
                        # ignore this query, so they cannot be mistaken for the pitch firmware.
                        if self.port.write(b'?\n') != 2:
                            raise OSError('incomplete firmware identity query')
                        self.identity_query_sent = True
            except Exception as exc:
                self.fail(f'serial read or pitch reply failed: {exc}')

        def close(self):
            if self.port is not None:
                try:
                    self.port.close()
                except Exception as exc:
                    self.get_logger().error(f'cannot close serial port: {exc}')

    rclpy.init(args=args)
    node = None
    try:
        node = PitchSerialBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
