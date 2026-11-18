"""Actual ROS bridge tests against a Linux pseudo-terminal firmware emulator.

No physical serial device is opened. Run in a dedicated ROS_DOMAIN_ID.
"""

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

try:
    import pty
    import tty
    import rclpy
    import serial  # noqa: F401: verifies the bridge's runtime dependency is installed
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64, String
    ROS_PTY_AVAILABLE = os.name == 'posix'
except ImportError:
    ROS_PTY_AVAILABLE = False


class FirmwareEmulator:
    def __init__(self, identity=True, mismatch=False, uno=False, guarded=False):
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        self.slave_name = os.ttyname(self.slave)
        self.identity = identity
        self.mismatch = mismatch
        self.uno = uno
        self.guarded = guarded
        self.received = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        buffer = bytearray()
        while not self.stop.is_set():
            readable, _, _ = select.select([self.master], [], [], 0.05)
            if not readable:
                continue
            try:
                data = os.read(self.master, 4096)
            except OSError:
                break
            if not data:
                break
            buffer.extend(data)
            while b'\n' in buffer:
                raw, _, remainder = buffer.partition(b'\n')
                buffer = bytearray(remainder)
                line = raw.decode('ascii').strip()
                with self.lock:
                    self.received.append(line)
                response = None
                if line == '?':
                    if not self.identity:
                        response = 'READY\r\n'
                    elif self.uno:
                        response = ('lever 100.0 (rest 100.0 cock 30.0 fire 125.0) '
                                    '| pitch 15.0 | tilt servo 90.0 | targets OK\r\n')
                    elif self.guarded:
                        response = ('lever 100.0 (rest 100.0 cock 30.0 fire 125.0, OFF) '
                                    '| pitch 15.0 deg, tilt servo 135.0 (OFF)\r\n')
                    else:
                        response = ('lever 90.0 (rest 90.0 cock 20.0 fire 115.0) '
                                    '| pitch 15.0 deg, tilt servo 90.0\r\n')
                elif line.startswith('a'):
                    angle = float(line[1:]) + (1.0 if self.mismatch else 0.0)
                    response = f'pitch {angle:.1f}\r\n'
                elif line == 'e5' and self.guarded:
                    response = 'tilt PWM ON\r\n'
                if response is not None:
                    os.write(self.master, response.encode('ascii'))

    def commands(self):
        with self.lock:
            return list(self.received)

    def restart(self):
        banner = 'READY\r\n' if self.uno else 'launcher_test ready\r\n'
        os.write(self.master, banner.encode('ascii'))

    def close(self):
        self.stop.set()
        self.worker.join(timeout=1.0)
        os.close(self.master)
        os.close(self.slave)


@unittest.skipUnless(ROS_PTY_AVAILABLE, 'requires ROS 2, pyserial and Linux pseudo-terminals')
class PitchBridgeRosTest(unittest.TestCase):
    def setUp(self):
        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = rclpy.create_node('pitch_bridge_pty_test', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.statuses = []
        qos = QoSProfile(depth=30, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.node.create_subscription(
            String, '/launcher/pitch_status', self.receive_status, qos)
        self.pub = self.node.create_publisher(Float64, '/launcher/pitch_command', 10)
        self.emulator = None
        self.process = None
        self.log = tempfile.TemporaryFile(mode='w+b')

    def receive_status(self, message):
        self.statuses.append(json.loads(message.data))

    def start_bridge(self, identity=True, mismatch=False, uno=False, guarded=False):
        self.emulator = FirmwareEmulator(identity=identity, mismatch=mismatch, uno=uno, guarded=guarded)
        command = [sys.executable, '-c',
                   'from limo_fire_hardware.pitch_serial_bridge import main; main()',
                   '--ros-args', '-p', 'enabled:=true',
                   '-p', f'serial_port:={self.emulator.slave_name}',
                   '-p', 'startup_delay_s:=0.1', '-p', 'startup_timeout_s:=1.0',
                   '-p', 'ack_timeout_s:=1.0', '-p', 'min_command_interval_s:=0.0']
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=os.environ.copy(), start_new_session=True)

    def wait_for(self, predicate, seconds=8.0):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
            if predicate():
                return
            if self.process is not None and self.process.poll() is not None:
                self.log.seek(0)
                self.fail(f'bridge exited with {self.process.returncode}: '
                          f'{self.log.read(16000).decode(errors="replace")}')
        self.fail(f'timed out; statuses={self.statuses[-8:]}, '
                  f'commands={self.emulator.commands() if self.emulator else []}')

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)

    def publish_pitch(self, degrees):
        self.wait_for(lambda: self.pub.get_subscription_count() > 0)
        message = Float64()
        message.data = degrees
        self.pub.publish(message)

    def test_nominal_pitch_and_command_only_acknowledgement(self):
        self.start_bridge()
        self.wait_for(lambda: any(s['state'] == 'ready' for s in self.statuses))
        self.assertEqual(self.emulator.commands(), ['?'])
        self.publish_pitch(15.0)
        self.wait_for(lambda: any(s['state'] == 'acknowledged' for s in self.statuses))
        acknowledgement = next(s for s in self.statuses if s['state'] == 'acknowledged')
        self.assertTrue(acknowledgement['acked'])
        self.assertFalse(acknowledgement['measured'])
        self.assertEqual(acknowledgement['command_deg'], 15.0)
        self.assertGreater(acknowledgement['sequence'], 0)
        self.assertEqual(self.emulator.commands(), ['?', 'a15.0'])
        before = len(self.statuses)
        self.wait_for(lambda: any(s['state'] == 'acknowledged' for s in self.statuses[before:]), seconds=3.0)

    def test_wrong_firmware_never_receives_pitch_command(self):
        self.start_bridge(identity=False)
        self.wait_for(lambda: any(s['state'] == 'error' for s in self.statuses))
        self.publish_pitch(15.0)
        self.wait_for(lambda: any(s['state'] == 'rejected' for s in self.statuses))
        self.spin_for(0.3)
        self.assertEqual(self.emulator.commands(), ['?'])
        self.assertFalse(any(s['acked'] for s in self.statuses))

    def test_uno_status_handshake_and_pitch_acknowledgement(self):
        self.start_bridge(uno=True)
        self.wait_for(lambda: any(s['state'] == 'ready' for s in self.statuses))
        self.assertEqual(self.emulator.commands(), ['?'])
        self.publish_pitch(22.5)
        self.wait_for(lambda: any(s['state'] == 'acknowledged' for s in self.statuses))
        acknowledgement = next(s for s in self.statuses if s['state'] == 'acknowledged')
        self.assertTrue(acknowledgement['acked'])
        self.assertFalse(acknowledgement['measured'])
        self.assertEqual(acknowledgement['command_deg'], 22.5)
        self.assertEqual(self.emulator.commands(), ['?', 'a22.5'])

    def test_guarded_esp32_enables_pitch_before_ready_and_command(self):
        self.start_bridge(guarded=True)
        self.wait_for(lambda: any(s['state'] == 'ready' for s in self.statuses))
        self.assertEqual(self.emulator.commands(), ['?', 'e5'])
        self.publish_pitch(15.0)
        self.wait_for(lambda: any(s['state'] == 'acknowledged' for s in self.statuses))
        self.assertEqual(self.emulator.commands(), ['?', 'e5', 'a15.0'])

    def test_uno_ready_after_handshake_latches_reset_fault(self):
        self.start_bridge(uno=True)
        self.wait_for(lambda: any(s['state'] == 'ready' for s in self.statuses))
        self.emulator.restart()
        self.wait_for(lambda: any(s['state'] == 'error' for s in self.statuses))
        error = next(s for s in self.statuses if s['state'] == 'error')
        self.assertIn('firmware restarted', error['reason'])
        self.publish_pitch(20.0)
        self.wait_for(lambda: any(s['state'] == 'rejected' and s['command_deg'] == 20.0
                                 for s in self.statuses))
        self.spin_for(0.3)
        self.assertEqual(self.emulator.commands(), ['?'])
        self.assertFalse(any(s['acked'] for s in self.statuses))

    def test_mismatched_reply_latches_fault(self):
        self.start_bridge(mismatch=True)
        self.wait_for(lambda: any(s['state'] == 'ready' for s in self.statuses))
        self.publish_pitch(15.0)
        self.wait_for(lambda: any(s['state'] == 'error' for s in self.statuses))
        self.publish_pitch(20.0)
        self.wait_for(lambda: any(s['state'] == 'rejected' and s['command_deg'] == 20.0
                                 for s in self.statuses))
        self.spin_for(0.3)
        self.assertEqual(self.emulator.commands(), ['?', 'a15.0'])
        self.assertFalse(any(s['acked'] for s in self.statuses))

    def tearDown(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2.0)
        if self.emulator is not None:
            self.emulator.close()
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
        self.log.close()


if __name__ == '__main__':
    unittest.main()
