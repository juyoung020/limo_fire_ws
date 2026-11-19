"""Planar LIMO positioning for a launcher with one elevation axis.

Mount xyz is the pitch pivot in base_link. Mount yaw is the fixed horizontal
direction of the launcher's neutral forward axis. Pitch is positive upward.
Line-of-sight elevation is the default. The optional ideal projectile model
requires launch speed supplied by the caller. Base roll and pitch are zero.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


DEFAULT_ANGULAR_OFFSETS = tuple(
    math.radians(degrees) for degrees in (0, 15, -15, 30, -30, 60, -60)
)


@dataclass(frozen=True)
class ApproachCandidate:
    base_x: float
    base_y: float
    base_yaw: float
    pivot_x: float
    pivot_y: float
    pivot_z: float
    distance_m: float
    pitch_rad: float


def _finite_vector(values: Sequence[float], length: int, name: str) -> tuple:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _finite_angle(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _relative_target(
    target_xyz: Sequence[float],
    base_pose_xyz_yaw: Sequence[float],
    mount_xyz: Sequence[float],
    mount_yaw_rad: float,
) -> tuple:
    target = _finite_vector(target_xyz, 3, "target_xyz")
    base = _finite_vector(base_pose_xyz_yaw, 4, "base_pose_xyz_yaw")
    mount = _finite_vector(mount_xyz, 3, "mount_xyz")
    beta = _finite_angle(mount_yaw_rad, "mount_yaw_rad")
    cosine, sine = math.cos(base[3]), math.sin(base[3])
    pivot_x = base[0] + cosine * mount[0] - sine * mount[1]
    pivot_y = base[1] + sine * mount[0] + cosine * mount[1]
    pivot_z = base[2] + mount[2]
    dx, dy = target[0] - pivot_x, target[1] - pivot_y
    neutral_yaw = base[3] + beta
    cosine, sine = math.cos(neutral_yaw), math.sin(neutral_yaw)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        target[2] - pivot_z,
    )


def pitch_for_target(
    target_xyz: Sequence[float],
    base_pose_xyz_yaw: Sequence[float],
    mount_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    mount_yaw_rad: float = 0.0,
) -> float:
    """Return line-of-sight elevation in radians from the neutral pitch pivot.

    Check heading_error before commanding pitch. This elevation remains defined
    for a horizontally misaligned target so the approach planner can evaluate it.
    """
    forward, lateral, up = _relative_target(
        target_xyz, base_pose_xyz_yaw, mount_xyz, mount_yaw_rad
    )
    horizontal = math.hypot(forward, lateral)
    if horizontal <= 1e-9:
        raise ValueError("target must have nonzero horizontal distance from pitch pivot")
    return math.atan2(up, horizontal)


def heading_error(
    target_xyz: Sequence[float],
    base_pose_xyz_yaw: Sequence[float],
    mount_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    mount_yaw_rad: float = 0.0,
) -> float:
    """Return signed horizontal target error relative to the launcher axis."""
    forward, lateral, _ = _relative_target(
        target_xyz, base_pose_xyz_yaw, mount_xyz, mount_yaw_rad
    )
    if math.hypot(forward, lateral) <= 1e-9:
        raise ValueError("target must have nonzero horizontal distance from pitch pivot")
    return math.atan2(lateral, forward)


def yaw_rate_command(error_rad: float, tolerance_rad: float, gain: float,
                     minimum_rps: float, maximum_rps: float) -> float:
    """Bounded in-place angular velocity toward a freshly measured target."""
    values = (error_rad, tolerance_rad, gain, minimum_rps, maximum_rps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('yaw rate inputs must be finite')
    if tolerance_rad <= 0 or gain <= 0 or minimum_rps <= 0 or maximum_rps < minimum_rps:
        raise ValueError('invalid yaw rate limits')
    if abs(error_rad) <= tolerance_rad:
        return 0.0
    return math.copysign(min(maximum_rps, max(minimum_rps, gain * abs(error_rad))), error_rad)


def body_yaw_for_target(
    target_xyz: Sequence[float],
    base_pose_xyz_yaw: Sequence[float],
    mount_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    mount_yaw_rad: float = 0.0,
) -> float:
    """Return body yaw aligning the launcher while holding base x/y fixed.

    A lateral pivot offset changes the yaw required as the body rotates. If two
    forward-facing solutions exist, choose the one nearest the current yaw.
    """
    target = _finite_vector(target_xyz, 3, "target_xyz")
    base = _finite_vector(base_pose_xyz_yaw, 4, "base_pose_xyz_yaw")
    mount = _finite_vector(mount_xyz, 3, "mount_xyz")
    beta = _finite_angle(mount_yaw_rad, "mount_yaw_rad")
    dx, dy = target[0] - base[0], target[1] - base[1]
    distance = math.hypot(dx, dy)
    lateral_offset = -mount[0] * math.sin(beta) + mount[1] * math.cos(beta)
    forward_offset = mount[0] * math.cos(beta) + mount[1] * math.sin(beta)
    if distance <= 1e-9 or abs(lateral_offset) > distance:
        raise ValueError("target cannot be aligned from this base position")
    delta = math.asin(lateral_offset / distance)
    bearing = math.atan2(dy, dx)
    solutions = []
    for angle in (delta, math.pi - delta):
        forward_range = distance * math.cos(angle) - forward_offset
        if forward_range > 1e-9:
            solutions.append(_wrap(bearing - beta - angle))
    if not solutions:
        raise ValueError("target is behind the pitch pivot for all aligned body yaws")
    return min(solutions, key=lambda yaw: abs(_wrap(yaw - base[3])))


def ballistic_pitch(
    distance_m: float,
    height_delta_m: float,
    speed_mps: float,
    gravity_mps2: float = 9.81,
) -> float:
    """Return the low-angle elevation for an ideal projectile without drag.

    Distance and height are measured from the projectile release point. Speed
    must be supplied from calibration appropriate to the physical launcher.
    """
    distance = float(distance_m)
    height = float(height_delta_m)
    speed = float(speed_mps)
    gravity = float(gravity_mps2)
    if not all(math.isfinite(value) for value in (distance, height, speed, gravity)):
        raise ValueError("ballistic inputs must be finite")
    if distance <= 0.0 or speed <= 0.0 or gravity <= 0.0:
        raise ValueError("ballistic distance, speed and gravity must be positive")
    speed_squared = speed * speed
    discriminant = speed_squared * speed_squared - gravity * (
        gravity * distance * distance + 2.0 * height * speed_squared
    )
    if not math.isfinite(discriminant) or discriminant < 0.0:
        raise ValueError("target is outside the ideal ballistic envelope")
    # Rationalize the low-angle root to avoid cancellation at short ranges.
    denominator = distance * (speed_squared + math.sqrt(discriminant))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("ballistic solution exceeds floating-point range")
    tangent = (gravity * distance * distance + 2.0 * height * speed_squared) / denominator
    if not math.isfinite(tangent):
        raise ValueError("ballistic solution is not finite")
    return math.atan(tangent)


def approach_candidates(
    target_xyz: Sequence[float],
    base_pose_xyz_yaw: Sequence[float],
    distance_m: float,
    mount_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    mount_yaw_rad: float = 0.0,
    angular_offsets_rad: Iterable[float] = DEFAULT_ANGULAR_OFFSETS,
    pitch_min_rad: float = math.radians(5.0),
    pitch_max_rad: float = math.radians(30.0),
    launch_speed_mps: float | None = None,
    gravity_mps2: float = 9.81,
) -> list[ApproachCandidate]:
    """Return deterministic base poses on a launcher-pivot distance ring.

    The first angular offset is measured from the current pivot-to-target
    direction. Distance is horizontal distance from the pitch pivot. Candidates
    outside the elevation interval are omitted. Path, footprint and visibility
    checks belong to the caller. If launch_speed_mps is supplied, elevation uses
    the ideal ballistic model with the pitch pivot as release-point approximation.
    The caller must account for a distinct physical muzzle position separately.
    """
    target = _finite_vector(target_xyz, 3, "target_xyz")
    base = _finite_vector(base_pose_xyz_yaw, 4, "base_pose_xyz_yaw")
    mount = _finite_vector(mount_xyz, 3, "mount_xyz")
    beta = _finite_angle(mount_yaw_rad, "mount_yaw_rad")
    distance = float(distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be positive and finite")
    lower = _finite_angle(pitch_min_rad, "pitch_min_rad")
    upper = _finite_angle(pitch_max_rad, "pitch_max_rad")
    if not -math.pi / 2.0 <= lower <= upper <= math.pi / 2.0:
        raise ValueError("pitch limits must be ordered elevations within [-pi/2, pi/2]")
    offsets = tuple(_finite_angle(offset, "angular offset") for offset in angular_offsets_rad)
    if not offsets:
        raise ValueError("angular_offsets_rad must contain at least one angle")
    normalized_offsets = tuple(_wrap(offset) for offset in offsets)
    for index, offset in enumerate(normalized_offsets):
        if any(abs(_wrap(offset - previous)) <= 1e-9 for previous in normalized_offsets[:index]):
            raise ValueError("angular offsets must be distinct modulo 2*pi")

    cosine, sine = math.cos(base[3]), math.sin(base[3])
    current_pivot_x = base[0] + cosine * mount[0] - sine * mount[1]
    current_pivot_y = base[1] + sine * mount[0] + cosine * mount[1]
    dx, dy = target[0] - current_pivot_x, target[1] - current_pivot_y
    if math.hypot(dx, dy) <= 1e-9:
        direction = base[3] + beta
    else:
        direction = math.atan2(dy, dx)
    pivot_z = base[2] + mount[2]
    if launch_speed_mps is None:
        pitch = math.atan2(target[2] - pivot_z, distance)
    else:
        pitch = ballistic_pitch(distance, target[2] - pivot_z, launch_speed_mps, gravity_mps2)
    if not lower <= pitch <= upper:
        return []

    candidates = []
    for offset in normalized_offsets:
        launch_yaw = direction + offset
        base_yaw = _wrap(launch_yaw - beta)
        pivot_x = target[0] - distance * math.cos(launch_yaw)
        pivot_y = target[1] - distance * math.sin(launch_yaw)
        cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
        candidates.append(ApproachCandidate(
            base_x=pivot_x - cosine * mount[0] + sine * mount[1],
            base_y=pivot_y - sine * mount[0] - cosine * mount[1],
            base_yaw=base_yaw,
            pivot_x=pivot_x,
            pivot_y=pivot_y,
            pivot_z=pivot_z,
            distance_m=distance,
            pitch_rad=pitch,
        ))
    return candidates
