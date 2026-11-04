"""Implement the Fire-Aware Extinguish action with a MuJoCo ball shot."""
import csv
import math
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from limo_fire_msgs.action import Extinguish
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class MujocoExtinguisher(Node):
    """Return success only when MuJoCo records a direct hit and a topple."""

    def __init__(self):
        super().__init__('mujoco_extinguisher')
        self.declare_parameter('action', '/fire/mujoco_extinguish')
        self.declare_parameter('mujoco_root', str(Path(__file__).resolve().parents[3]))
        self.declare_parameter('python', '')
        self.declare_parameter('aim', 'surface')
        self.declare_parameter('speed', 5.5)
        self.declare_parameter('aim_height', 0.40)
        self.declare_parameter('target_pivot_offset', 0.15)
        self.declare_parameter('show_gui', False)
        self.declare_parameter('gui_playback_speed', 0.35)
        self.declare_parameter('timeout_s', 30.0)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_distance', 0.7)
        self.declare_parameter('max_distance', 3.0)
        if not self.get_parameter('use_sim_time').value:
            raise RuntimeError('MuJoCo extinguisher is simulation-only; use_sim_time must be true')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._active = None
        self._viewers = []
        self._lock = threading.Lock()
        self._server = ActionServer(
            self, Extinguish, str(self.get_parameter('action').value),
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
        )
        self.get_logger().info('MuJoCo ball shot action server ready')

    @staticmethod
    def _goal(_goal):
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel(_goal_handle):
        return CancelResponse.ACCEPT

    def _to_base(self, target: PointStamped):
        if not target.header.frame_id:
            raise ValueError('target frame_id is empty')
        if target.header.frame_id == str(self.get_parameter('base_frame').value):
            point = target.point
        else:
            transform = self.tf_buffer.lookup_transform(
                str(self.get_parameter('base_frame').value), target.header.frame_id, Time())
            point = do_transform_point(target, transform).point
        x, y = float(point.x), float(point.y)
        if not math.isfinite(x) or not math.isfinite(y) or x <= 0.0:
            raise ValueError('target must be finite and in front of the robot')
        distance = math.hypot(x, y)
        lo = float(self.get_parameter('min_distance').value)
        hi = float(self.get_parameter('max_distance').value)
        # In scene.xml the target box center is 0.15 m behind its hinge.
        # Fire observations identify the box center, while --distance sets the hinge.
        pivot_x = x + float(self.get_parameter('target_pivot_offset').value)
        if not lo <= pivot_x <= hi:
            raise ValueError(f'target forward distance {pivot_x:.3f} m is outside MuJoCo range [{lo:.2f}, {hi:.2f}] m')
        return x, y, distance

    def _execute(self, goal_handle):
        # Keep at most one inspection window so repeated fires do not pile up.
        self._reap_viewers(close_all=True)
        result = Extinguish.Result()
        try:
            x, y, distance = self._to_base(goal_handle.request.target)
        except (ValueError, TransformException) as exc:
            result.success = False
            result.reason = f'MuJoCo target rejected: {exc}'
            goal_handle.abort()
            self.get_logger().warning(result.reason)
            return result

        root = Path(str(self.get_parameter('mujoco_root').value)).expanduser().resolve()
        python = str(self.get_parameter('python').value) or str(root / '.venv/bin/python')
        pivot_x = x + float(self.get_parameter('target_pivot_offset').value)
        show_gui = bool(self.get_parameter('show_gui').value)
        run_dir = Path(tempfile.mkdtemp(prefix='mujoco_extinguish_'))
        csv_path = run_dir / 'shot.csv'
        command = [
            python, str(root / 'launch.py'),
            '--aim', str(self.get_parameter('aim').value),
            '--speed', str(float(self.get_parameter('speed').value)),
            '--distance', f'{pivot_x:.6f}', '--target-y', f'{y:.6f}',
            '--aim-height', str(float(self.get_parameter('aim_height').value)),
            '--csv', str(csv_path), '--duration', '3.0',
        ]
        if show_gui:
            command.extend(['--autoplay', '--playback-speed',
                            str(float(self.get_parameter('gui_playback_speed').value))])
        else:
            command.append('--headless')

        process = None
        keep_viewer = False
        try:
            self.get_logger().info(
                f"MuJoCo shot for {goal_handle.request.fire_id}: relative=({x:.3f},{y:.3f}) m, range={distance:.3f} m, gui={show_gui}")
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            with self._lock:
                self._active = process
            summary_path = csv_path.with_name('shot_summary.csv')
            deadline = time.monotonic() + float(self.get_parameter('timeout_s').value)
            while True:
                if goal_handle.is_cancel_requested:
                    self._stop_process(process)
                    result.success = False
                    result.reason = 'MuJoCo shot canceled'
                    goal_handle.canceled()
                    return result
                if show_gui and summary_path.is_file():
                    # The shot is complete; return the result while the final
                    # frame remains visible beside the Webots mission.
                    break
                if process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('MuJoCo shot timed out')
                threading.Event().wait(0.05)

            output = ''
            if not show_gui or process.poll() is not None:
                output, _ = process.communicate()
            if process.returncode not in (None, 0) or not summary_path.is_file():
                raise RuntimeError(f'MuJoCo exited {process.returncode}: {(output or "")[-1200:]}')
            with summary_path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or rows[0].get('completed', '').lower() != 'true':
                raise RuntimeError(f'expected one completed shot result row, got {len(rows)}')
            row = rows[0]
            contact = row.get('contact', '').lower() == 'true'
            direct_hit = row.get('direct_hit', '').lower() == 'true'
            toppled = row.get('toppled', '').lower() == 'true'
            result.success = contact and direct_hit and toppled
            result.reason = (
                f"MuJoCo contact={contact}, direct_hit={direct_hit}, toppled={toppled}; "
                f"failure={row.get('failure', 'unknown')}"
            )
            if show_gui and process.poll() is None:
                with self._lock:
                    self._viewers.append((process, run_dir))
                process = None
                keep_viewer = True
                self.get_logger().info('MuJoCo result frame is open; it closes when the mission launch shuts down or another shot starts')
            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
        except (OSError, subprocess.TimeoutExpired, TimeoutError, RuntimeError) as exc:
            result.success = False
            result.reason = f'MuJoCo shot failed: {exc}'
            goal_handle.abort()
        finally:
            if process is not None:
                if process.poll() is None:
                    self._stop_process(process)
                process.communicate()
            if not keep_viewer:
                shutil.rmtree(run_dir, ignore_errors=True)
            with self._lock:
                self._active = None
        self.get_logger().info(result.reason)
        return result

    @staticmethod
    def _stop_process(process):
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _reap_viewers(self, close_all=False):
        with self._lock:
            viewers, self._viewers = self._viewers, []
        keep = []
        for process, run_dir in viewers:
            if close_all:
                self._stop_process(process)
            if process.poll() is None:
                keep.append((process, run_dir))
            else:
                process.communicate()
                shutil.rmtree(run_dir, ignore_errors=True)
        with self._lock:
            self._viewers.extend(keep)

    def destroy_node(self):
        self._reap_viewers(close_all=True)
        if self._active is not None:
            self._stop_process(self._active)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoExtinguisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
