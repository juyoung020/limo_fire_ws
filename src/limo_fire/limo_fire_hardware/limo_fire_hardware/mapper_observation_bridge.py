"""Forward only freshly seen fire from Juyoung's RGB-D object map to aiming.

The mapper republishes its saved landmarks at 10 Hz. Its Detection3DArray
header is therefore *not* proof of a new camera measurement. The mapper puts
seconds since last sighting in results[0].pose.covariance[35]; this bridge
uses that age to stamp FireObservation at the actual sighting time.
"""

import math


def select_live_fire(detections, *, target_id='', min_confidence=0.5,
                     max_seen_age_s=0.5):
    """Return (detection, age, reason); reject stale or ambiguous map entries."""
    candidates = []
    for detection in detections:
        if not detection.id or (target_id and detection.id != target_id) or not detection.results:
            continue
        result = detection.results[0]
        if result.hypothesis.class_id != 'fire':
            continue
        score = result.hypothesis.score
        age = result.pose.covariance[35]
        position = detection.bbox.center.position
        if (not math.isfinite(score) or score < min_confidence or
                not math.isfinite(age) or age < 0 or age > max_seen_age_s or
                not all(map(math.isfinite, (position.x, position.y, position.z)))):
            continue
        candidates.append((detection, age))
    if not candidates:
        return None, None, 'no recently observed fire with valid map position'
    if len(candidates) != 1:
        return None, None, 'multiple fire targets: set target_id explicitly'
    return candidates[0][0], candidates[0][1], ''


def main():
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from vision_msgs.msg import Detection3DArray
    from limo_fire_msgs.msg import FireObservation

    class MapperObservationBridge(Node):
        def __init__(self):
            super().__init__('mapper_observation_bridge')
            self.declare_parameter('objects_topic', '/perception/objects')
            self.declare_parameter('fire_topic', '/fire/observations')
            self.declare_parameter('target_id', '')
            self.declare_parameter('min_confidence', 0.5)
            self.declare_parameter('max_seen_age_s', 0.5)
            self.declare_parameter('max_message_delay_s', 0.2)
            self.target_id = self.get_parameter('target_id').value
            self.min_confidence = self.get_parameter('min_confidence').value
            self.max_seen_age_s = self.get_parameter('max_seen_age_s').value
            self.max_message_delay_s = self.get_parameter('max_message_delay_s').value
            if not (0 <= self.min_confidence <= 1 and self.max_seen_age_s > 0 and
                    self.max_message_delay_s > 0):
                raise ValueError('invalid mapper bridge freshness or confidence limits')
            self.publisher = self.create_publisher(
                FireObservation, self.get_parameter('fire_topic').value, 10)
            self.subscription = self.create_subscription(
                Detection3DArray, self.get_parameter('objects_topic').value,
                self.on_objects, 10)
            self.last_issue = ''

        def on_objects(self, msg):
            if msg.header.frame_id != 'map':
                self.warn_change('mapper objects are not in map frame')
                return
            stamp = Time.from_msg(msg.header.stamp)
            delay = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if not math.isfinite(delay) or delay < -0.1 or delay > self.max_message_delay_s:
                self.warn_change('mapper objects message is delayed or has a future stamp')
                return
            detection, age, reason = select_live_fire(
                msg.detections, target_id=self.target_id,
                min_confidence=self.min_confidence,
                max_seen_age_s=self.max_seen_age_s)
            if reason:
                self.warn_change(reason)
                return
            observation = FireObservation()
            observation.header.frame_id = 'map'
            observation.header.stamp = (stamp - Duration(seconds=age)).to_msg()
            observation.id = detection.id
            observation.position = detection.bbox.center.position
            observation.position_valid = True
            observation.width_m = detection.bbox.size.x
            observation.size_valid = (math.isfinite(observation.width_m) and
                                      observation.width_m > 0)
            observation.confidence = detection.results[0].hypothesis.score
            self.publisher.publish(observation)
            self.last_issue = ''

        def warn_change(self, message):
            if message != self.last_issue:
                self.get_logger().warn(message)
                self.last_issue = message

    rclpy.init()
    node = MapperObservationBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
