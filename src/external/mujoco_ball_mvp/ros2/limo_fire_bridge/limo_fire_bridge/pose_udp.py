"""Forward Webots ROS odometry pose to the MuJoCo live viewer over localhost UDP."""
import json
import math
import socket

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, PoseStamped


class PoseUdp(Node):
    def __init__(self):
        super().__init__('limo_mujoco_pose_sync')
        self.declare_parameter('odom_topic','/odom')
        self.declare_parameter('udp_host','127.0.0.1')
        self.declare_parameter('udp_port',8765)
        self.declare_parameter('target_topic','/fire/detected_target_pose')
        self.declare_parameter('targets_topic','/fire/detected_targets')
        self.declare_parameter('target_frame','odom')
        self.declare_parameter('target_timeout',0.75)
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.dest=(str(self.get_parameter('udp_host').value),int(self.get_parameter('udp_port').value))
        self.target=None
        self.target_stamp=None
        self.targets=None
        self.targets_stamp=None
        self.create_subscription(Odometry,str(self.get_parameter('odom_topic').value),self.on_odom,10)
        self.create_subscription(PoseStamped,str(self.get_parameter('target_topic').value),self.on_target,10)
        self.create_subscription(PoseArray,str(self.get_parameter('targets_topic').value),self.on_targets,10)
        self.get_logger().info(f"Forwarding odom pose to MuJoCo viewer UDP {self.dest[0]}:{self.dest[1]}")
    def on_target(self,msg):
        expected=str(self.get_parameter('target_frame').value)
        if msg.header.frame_id != expected:
            self.get_logger().warning(f"Ignoring detected target in {msg.header.frame_id!r}; expected {expected!r}")
            return
        x=float(msg.pose.position.x);y=float(msg.pose.position.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        self.target=(x,y)
        self.target_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def on_targets(self,msg):
        expected=str(self.get_parameter('target_frame').value)
        if msg.header.frame_id != expected:
            return
        points=[]
        for pose in msg.poses:
            x=float(pose.position.x);y=float(pose.position.y)
            if math.isfinite(x) and math.isfinite(y):points.append((x,y))
        self.targets=points
        self.targets_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def on_odom(self,msg):
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        now=self.get_clock().now().nanoseconds*1e-9
        timeout=float(self.get_parameter('target_timeout').value)
        array_fresh=(self.targets is not None and self.targets_stamp is not None and
                     -0.5 <= now-self.targets_stamp <= timeout)
        single_fresh=(self.target is not None and self.target_stamp is not None and
                      -0.5 <= now-self.target_stamp <= timeout)
        targets=self.targets if array_fresh else ([self.target] if single_fresh else [])
        targets=sorted(targets,key=lambda t:math.hypot(t[0]-p.x,t[1]-p.y))[:8]
        valid=bool(targets)
        target_x,target_y=targets[0] if targets else (0.0,0.0)
        packet=json.dumps({'x':p.x,'y':p.y,'yaw':yaw,
                           'target_x':target_x,'target_y':target_y,
                           'target_valid':valid,'targets':targets},separators=(',',':')).encode()
        try:self.sock.sendto(packet,self.dest)
        except OSError as exc:self.get_logger().warning(f'UDP pose send failed: {exc}')
    def destroy_node(self):
        self.sock.close();super().destroy_node()

def main(args=None):
    rclpy.init(args=args);n=PoseUdp()
    try:rclpy.spin(n)
    finally:n.destroy_node();rclpy.shutdown()
