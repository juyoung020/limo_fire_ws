from geometry_msgs.msg import Pose
from enum import Enum
import numpy as np
import math
import tf_transformations
import copy

def quaternion_from_euler(a_1, a_2, a_3):
    # get roll, pitch, yaw and divide by 2
    a_1 /= 2.0 
    a_2 /= 2.0
    a_3 /= 2.0
    
    # calculate cos and sin
    c_1 = math.cos(a_1)
    s_1 = math.sin(a_1)
    c_2 = math.cos(a_2)
    s_2 = math.sin(a_2)
    c_3 = math.cos(a_3)
    s_3 = math.sin(a_3)

    # calculate quaternion with variable
    # q = q[0]i + q[1]j + q[2]k + q[3]
    q = np.empty((4, ))
    q[0] = s_1 * c_2 * c_3 - c_1 * s_2 * s_3
    q[1] = c_1 * s_2 * c_3 + s_1 * c_2 * s_3
    q[2] = c_1 * c_2 * s_3 - s_1 * s_2 * c_3
    q[3] = c_1 * c_2 * c_3 + s_1 * s_2 * s_3

    return q

def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class NavPose:
    def __init__(self):
        self.pos_ = Pose()
        
    def set_pose(self, x=0, y=0, theta=0): 
        # set the initial pose
        self.pos_.position.x = x
        self.pos_.position.y = y
        q = quaternion_from_euler(0, 0, theta)
        self.pos_.orientation.x = q[0]
        self.pos_.orientation.y = q[1]
        self.pos_.orientation.z = q[2]
        self.pos_.orientation.w = q[3]

    def get_pose(self):
        # get pose
        return copy.deepcopy(self.pos_)
    
    def set_relative(self, x=0, y=0, theta=0):
        # set pose relatively 
        # calculate coordinate
        self.pos_.position.x = self.pos_.position.x + x
        self.pos_.position.y = self.pos_.position.y + y

        # calculate orientation
        q_orig = [self.pos_.orientation.x,
                 self.pos_.orientation.y, 
                 self.pos_.orientation.z, 
                 self.pos_.orientation.w]

        q_rot = quaternion_from_euler(0, 0, theta)
        q_new = tf_transformations.quaternion_multiply(q_orig, q_rot)

        self.pos_.orientation.x = q_new[0]
        self.pos_.orientation.y = q_new[1]
        self.pos_.orientation.z = q_new[2]
        self.pos_.orientation.w = q_new[3]
        

# you can check the limo status with enum class
class NavigateStatus(Enum):
    DEFAULT = 0
    PENDING = 1
    ACTIVE = 2
    GOAL = 3
    FAILED = 4