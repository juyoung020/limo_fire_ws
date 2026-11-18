#!/usr/bin/env python3
"""Publish a surveyed, stationary test marker as FireObservation.

This is a coordinate-injection check for aiming, not visual detection. Run it
only while no perception node publishes /fire/observations for the same target.
"""

import argparse
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x', type=float, required=True, help='marker map x, metres')
    parser.add_argument('--y', type=float, required=True, help='marker map y, metres')
    parser.add_argument('--z', type=float, required=True, help='marker height above map ground, metres')
    parser.add_argument('--id', default='measured_test_marker')
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.x, args.y, args.z)) or args.z < 0:
        parser.error('marker coordinates must be finite; z must be nonnegative')

    import rclpy
    from limo_fire_msgs.msg import FireObservation

    rclpy.init()
    node = rclpy.create_node('launcher_measured_test_marker')
    publisher = node.create_publisher(FireObservation, '/fire/observations', 10)

    def publish():
        observation = FireObservation()
        observation.header.stamp = node.get_clock().now().to_msg()
        observation.header.frame_id = 'map'
        observation.id = args.id
        observation.position.x = args.x
        observation.position.y = args.y
        observation.position.z = args.z
        observation.position_valid = True
        observation.confidence = 1.0
        publisher.publish(observation)

    timer = node.create_timer(0.1, publish)
    node.get_logger().info('Publishing surveyed marker coordinates only; no camera detection is involved')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_timer(timer)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
