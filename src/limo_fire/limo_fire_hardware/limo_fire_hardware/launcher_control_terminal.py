"""Interactive ROS launcher terminal sharing the aim bridge's serial connection."""

import argparse
import json
import math
import re
import time
from collections import deque
from datetime import datetime, timezone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', help='Append command and acknowledgement records as JSONL')
    parser.add_argument('--search-timeout-s', type=float, default=180.0,
                        help='Maximum frontier search time before an automatic mission gives up')
    parser.add_argument('--no-search', action='store_true',
                        help='Require a visible target when a is pressed; do not start frontier exploration')
    args = parser.parse_args(argv)
    if not math.isfinite(args.search_timeout_s) or args.search_timeout_s <= 0:
        parser.error('--search-timeout-s must be positive and finite')

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.time import Time
    from limo_fire_msgs.msg import FireObservation
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from std_srvs.srv import SetBool, Trigger

    rclpy.init()
    node = rclpy.create_node('launcher_control_terminal')
    statuses = []
    aim_statuses = []
    observations = deque(maxlen=40)
    odometry = deque(maxlen=10)
    explorer_states = deque(maxlen=10)
    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def status_received(message):
        try:
            statuses.append(json.loads(message.data))
        except ValueError:
            pass

    node.create_subscription(String, '/launcher/shot_status', status_received, qos)
    def aim_received(message):
        try:
            aim_statuses.append(json.loads(message.data))
        except ValueError:
            pass

    node.create_subscription(String, '/launcher/approach_aim_status', aim_received, qos)
    node.create_subscription(FireObservation, '/fire/observations',
                             lambda message: observations.append(message), 10)
    node.create_subscription(Odometry, '/odometry/filtered',
                             lambda message: odometry.append(message), 10)
    node.create_subscription(String, '/explore/frontier_explorer/state',
                             lambda message: explorer_states.append((message.data, time.monotonic())), qos)
    clients = {name: node.create_client(Trigger, '/launcher/' + name)
               for name in ('cock', 'fire', 'shot')}
    aim_client = node.create_client(Trigger, '/launcher/approach_and_aim')
    explore_client = node.create_client(SetBool, '/explore/frontier_explorer/enable')
    log = open(args.log, 'a', encoding='utf-8', buffering=1) if args.log else None

    def record(event, **fields):
        if log:
            log.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                      event=event, **fields), ensure_ascii=False) + '\n')

    def latest_aim_status():
        return max(aim_statuses, key=lambda item: item.get('stamp', 0)) if aim_statuses else None

    def fresh_stamp(stamp, maximum_s):
        age = (node.get_clock().now().nanoseconds - Time.from_msg(stamp).nanoseconds) * 1e-9
        return -0.1 <= age <= maximum_s

    def target_visible():
        if not observations:
            return False
        target = observations[-1]
        def valid(message):
            return (message.position_valid and message.header.frame_id == 'map'
                    and math.isfinite(message.confidence) and message.confidence >= 0.5
                    and all(math.isfinite(v) for v in (message.position.x, message.position.y,
                                                       message.position.z))
                    and fresh_stamp(message.header.stamp, 1.0))
        if not valid(target):
            return False
        seen_stamps = set()
        stable_count = 0
        for item in reversed(observations):
            if item.id != target.id or not valid(item):
                continue
            if math.dist((item.position.x, item.position.y, item.position.z),
                         (target.position.x, target.position.y, target.position.z)) > 0.25:
                continue
            stamp = (item.header.stamp.sec, item.header.stamp.nanosec)
            if stamp not in seen_stamps:
                seen_stamps.add(stamp)
                stable_count += 1
            if stable_count >= 3:
                return True
        return False

    def robot_stopped():
        if not odometry or not fresh_stamp(odometry[-1].header.stamp, 0.5):
            return False
        twist = odometry[-1].twist.twist
        values = (twist.linear.x, twist.linear.y, twist.angular.z)
        return (all(math.isfinite(v) for v in values)
                and math.hypot(*values[:2]) <= 0.01 and abs(values[2]) <= 0.02)

    def set_exploration(enabled):
        if not explore_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('프론티어 탐사 서비스가 없습니다.')
        future = explore_client.call_async(SetBool.Request(data=enabled))
        rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('프론티어 탐사 ' + ('시작' if enabled else '정지') + ' 응답을 확인하지 못했습니다.')
        record('exploration_enabled' if enabled else 'exploration_disabled',
               response=future.result().message)

    def search_until_target():
        # A fresh observation can be used directly, but never compete with an active explorer.
        warmup_deadline = time.monotonic() + 1.0
        while time.monotonic() < warmup_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if explorer_states and explorer_states[-1][0] == 'running':
            raise RuntimeError('이미 다른 프론티어 탐사가 진행 중입니다. 탐사를 정지한 뒤 다시 시도하세요.')
        if target_visible():
            return
        if args.no_search:
            raise RuntimeError('최근 1초 이내의 지도 좌표 표적이 없습니다.')
        if not explorer_states or explorer_states[-1][0] not in ('idle', 'done'):
            raise RuntimeError('프론티어 탐사 상태를 확인하지 못했습니다.')
        exploring = False
        try:
            set_exploration(True)
            exploring = True
            print('표적을 찾을 때까지 프론티어 탐사를 진행합니다.')
            deadline = time.monotonic() + args.search_timeout_s
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if target_visible():
                    break
                if explorer_states and explorer_states[-1][0] in ('done', 'idle'):
                    raise RuntimeError('탐사가 끝났지만 유효한 표적을 찾지 못했습니다.')
            else:
                raise RuntimeError('탐색 제한 시간이 지났거나 ROS가 종료됐습니다.')
            stop_requested_at = time.monotonic()
            set_exploration(False)
            exploring = False
            stopped_since = None
            handoff_deadline = time.monotonic() + 8.0
            while rclpy.ok() and time.monotonic() < handoff_deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                idle = (explorer_states and explorer_states[-1][0] == 'idle'
                        and explorer_states[-1][1] >= stop_requested_at)
                if idle and robot_stopped():
                    if stopped_since is None:
                        stopped_since = time.monotonic()
                    if time.monotonic() - stopped_since >= 0.5:
                        if not target_visible():
                            raise RuntimeError('정지 대기 중 표적 관측이 오래됐습니다.')
                        record('exploration_handoff', target_id=observations[-1].id)
                        return
                else:
                    stopped_since = None
            raise RuntimeError('탐사 정지와 차체 정지를 확인하지 못했습니다.')
        finally:
            if exploring:
                try:
                    set_exploration(False)
                except RuntimeError as error:
                    print('탐사 정지 확인 실패:', error)
                    record('exploration_stop_unconfirmed', reason=str(error))

    try:
        print('ROS launcher terminal: a=탐색·접근·조준·발사, c=장전, f=발사, s=장전+발사 1회, ?=상태, q=종료')
        print('조준 브리지가 같은 USB 포트를 소유합니다. 명령 완료는 실제 공의 비행을 측정하지 않습니다.')
        while rclpy.ok():
            command = input('launcher> ').strip().lower()
            if command == 'q':
                break
            if command == '?':
                rclpy.spin_once(node, timeout_sec=0.2)
                print('aim:', latest_aim_status() or '조준 상태 없음')
                print('launcher:', statuses[-1] if statuses else '장전·발사 상태 없음')
                continue
            if command == 'a':
                # Wait through one status heartbeat so a previous transient-local
                # publisher cannot mask the live controller after a restart.
                status_deadline = time.monotonic() + 1.2
                while time.monotonic() < status_deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                aim_status = latest_aim_status()
                if not aim_status or aim_status.get('auto_fire_enabled') is not True:
                    print('자동 발사가 꺼져 있습니다. launch에서 auto_fire_enabled:=true를 지정하세요.')
                    continue
                if not aim_client.wait_for_service(timeout_sec=2.0):
                    print('조준 임무 서비스가 없습니다.')
                    continue
                if aim_status.get('state') not in (
                        'IDLE', 'PREVIEW_READY', 'PITCH_ACKNOWLEDGED', 'AIM_VERIFIED',
                        'SHOT_COMMAND_ACKNOWLEDGED', 'SHOT_REJECTED', 'SHOT_UNCONFIRMED',
                        'FAILED', 'CANCELLED', 'STOP_UNCONFIRMED'):
                    print('다른 조준 임무가 진행 중입니다.')
                    continue
                try:
                    search_until_target()
                except RuntimeError as error:
                    print('자동 임무 시작 안 함:', error)
                    record('search_failed', reason=str(error))
                    continue
                baseline_stamp = aim_status.get('stamp', 0)
                future = aim_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
                if not future.done() or future.result() is None:
                    print('임무 요청 응답을 확인하지 못했습니다. 상태를 먼저 확인하세요.')
                    record('mission_unconfirmed')
                    continue
                reply = future.result()
                record('mission_request', accepted=reply.success, message=reply.message)
                if not reply.success:
                    print('임무 거부:', reply.message)
                    continue
                print('자동 임무 시작. 탐지·이동·조준·발사 상태를 기다립니다.')
                last_state = None
                terminal_states = {'SHOT_COMMAND_ACKNOWLEDGED', 'SHOT_REJECTED',
                                   'SHOT_UNCONFIRMED', 'FAILED', 'CANCELLED', 'STOP_UNCONFIRMED'}
                deadline = time.monotonic() + 130.0
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                    current_status = latest_aim_status()
                    if not current_status:
                        continue
                    if current_status.get('stamp', 0) <= baseline_stamp:
                        continue
                    state = current_status.get('state')
                    if state != last_state:
                        print('aim:', state, current_status.get('reason', ''))
                        record('mission_status', **current_status)
                        last_state = state
                    if state in terminal_states:
                        break
                else:
                    print('임무 완료를 확인하지 못했습니다. 재시작 전에 로봇과 발사기를 점검하세요.')
                    record('mission_unconfirmed')
                continue
            if command not in ('c', 'f', 's'):
                print('a, c, f, s, ?, q 중 하나를 입력하세요.')
                continue
            rclpy.spin_once(node, timeout_sec=0.2)
            current_status = latest_aim_status()
            if (current_status and current_status.get('state') not in
                    ('IDLE', 'PREVIEW_READY', 'PITCH_ACKNOWLEDGED', 'AIM_VERIFIED',
                     'SHOT_COMMAND_ACKNOWLEDGED', 'SHOT_REJECTED', 'SHOT_UNCONFIRMED',
                     'FAILED', 'CANCELLED', 'STOP_UNCONFIRMED')):
                print('자동 조준 임무가 진행 중입니다. 수동 장전·발사 명령을 보내지 않습니다.')
                continue
            action = {'c': 'cock', 'f': 'fire', 's': 'shot'}[command]
            client = clients[action]
            if not client.wait_for_service(timeout_sec=2.0):
                print('발사 브리지 서비스가 없습니다. launch 상태를 확인하세요.')
                continue
            baseline = max((item.get('sequence', 0) for item in statuses), default=0)
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
            if not future.done() or future.result() is None:
                print('서비스 응답을 확인하지 못했습니다. 재명령 전에 상태와 장치를 점검하세요.')
                record('unconfirmed', command=action)
                continue
            reply = future.result()
            record('request', command=action, accepted=reply.success, message=reply.message)
            if not reply.success:
                print('거부:', reply.message)
                continue
            match = re.search(r'\bsequence=(\d+)\b', reply.message)
            if not match or int(match.group(1)) <= baseline:
                print('새 명령 번호를 확인하지 못했습니다. 장치 상태를 점검하세요.')
                record('unconfirmed', command=action, message=reply.message)
                continue
            expected_sequence = int(match.group(1))
            deadline = time.monotonic() + 12.0
            final = None
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                for item in reversed(statuses):
                    if item.get('sequence') == expected_sequence and item.get('command') == action:
                        if item.get('state') in ('completed', 'error'):
                            final = item
                            break
                if final:
                    break
            if final:
                print(final['state'], final.get('reason', ''))
                record('result', **final)
            else:
                print('응답 시간 초과. 기구 상태를 확인하기 전에는 같은 명령을 반복하지 마세요.')
                record('unconfirmed', command=action)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if log:
            log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
