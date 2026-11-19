"""Interactive ROS launcher terminal sharing the aim bridge's serial connection."""

import argparse
import json
import re
import time
from datetime import datetime, timezone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', help='Append command and acknowledgement records as JSONL')
    args = parser.parse_args(argv)

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node('launcher_control_terminal')
    statuses = []
    aim_statuses = []
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
    clients = {name: node.create_client(Trigger, '/launcher/' + name)
               for name in ('cock', 'fire', 'shot')}
    aim_client = node.create_client(Trigger, '/launcher/approach_and_aim')
    log = open(args.log, 'a', encoding='utf-8', buffering=1) if args.log else None

    def record(event, **fields):
        if log:
            log.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                      event=event, **fields), ensure_ascii=False) + '\n')

    try:
        print('ROS launcher terminal: a=자동 임무 시작, c=장전, f=발사, s=장전+발사 1회, ?=상태, q=종료')
        print('조준 브리지가 같은 USB 포트를 소유합니다. 명령 완료는 실제 공의 비행을 측정하지 않습니다.')
        while rclpy.ok():
            command = input('launcher> ').strip().lower()
            if command == 'q':
                break
            if command == '?':
                rclpy.spin_once(node, timeout_sec=0.2)
                print('aim:', aim_statuses[-1] if aim_statuses else '조준 상태 없음')
                print('launcher:', statuses[-1] if statuses else '장전·발사 상태 없음')
                continue
            if command == 'a':
                status_deadline = time.monotonic() + 2.0
                while not aim_statuses and time.monotonic() < status_deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                if not aim_statuses or aim_statuses[-1].get('auto_fire_enabled') is not True:
                    print('자동 발사가 꺼져 있습니다. launch에서 auto_fire_enabled:=true를 지정하세요.')
                    continue
                if not aim_client.wait_for_service(timeout_sec=2.0):
                    print('조준 임무 서비스가 없습니다.')
                    continue
                baseline_stamp = aim_statuses[-1].get('stamp', 0)
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
                    if not aim_statuses:
                        continue
                    if aim_statuses[-1].get('stamp', 0) <= baseline_stamp:
                        continue
                    state = aim_statuses[-1].get('state')
                    if state != last_state:
                        print('aim:', state, aim_statuses[-1].get('reason', ''))
                        record('mission_status', **aim_statuses[-1])
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
            if (aim_statuses and aim_statuses[-1].get('state') not in
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
