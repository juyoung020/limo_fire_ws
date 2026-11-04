"""Pure geometry and stop checks for the ROS auto-approach controller."""
import math


def standoff_goal(robot_x, robot_y, target_x, target_y, distance):
    """Return map x/y/yaw on the robot's current side of the target."""
    dx, dy = target_x - robot_x, target_y - robot_y
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError('robot and fire positions are coincident')
    ux, uy = dx / norm, dy / norm
    goal_x, goal_y = target_x - distance * ux, target_y - distance * uy
    return goal_x, goal_y, math.atan2(target_y - goal_y, target_x - goal_x)


def engagement_ready(distance, robot_yaw, target_x, target_y, linear_speed,
                     angular_speed, min_distance=0.9, max_distance=1.5,
                     max_yaw_error=math.radians(45), max_linear_speed=0.02,
                     max_angular_speed=0.03):
    """Check standoff, aim direction, and measured stop before requesting fire."""
    target_yaw = math.atan2(target_y, target_x)
    yaw_error = math.atan2(math.sin(target_yaw - robot_yaw),
                           math.cos(target_yaw - robot_yaw))
    return (min_distance <= distance <= max_distance
            and abs(yaw_error) <= max_yaw_error
            and abs(linear_speed) <= max_linear_speed
            and abs(angular_speed) <= max_angular_speed)
