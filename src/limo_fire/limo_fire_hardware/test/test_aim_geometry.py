import math
import unittest

from limo_fire_hardware.aim_geometry import (
    approach_candidates,
    ballistic_pitch,
    body_yaw_for_target,
    heading_error,
    pitch_for_target,
    yaw_rate_command,
)


class AimGeometryTest(unittest.TestCase):
    def test_centered_target(self):
        target = (2.0, 0.0, 0.5)
        candidates = approach_candidates(target, (0.0, 0.0, 0.0, 0.0), 1.0)
        first = candidates[0]
        self.assertAlmostEqual(first.base_x, 1.0)
        self.assertAlmostEqual(first.base_y, 0.0)
        self.assertAlmostEqual(first.base_yaw, 0.0)
        self.assertAlmostEqual(first.pitch_rad, math.atan2(0.5, 1.0))

    def test_lateral_mount_and_axis_yaw(self):
        target = (2.7, -0.4, 0.85)
        current = (0.2, 0.5, 0.1, 0.4)
        mount = (0.3, -0.18, 0.25)
        beta = math.radians(23.0)
        distance = 1.4
        candidates = approach_candidates(target, current, distance, mount, beta)
        self.assertEqual(len(candidates), 7)
        for candidate in candidates:
            pose = (candidate.base_x, candidate.base_y, current[2], candidate.base_yaw)
            self.assertAlmostEqual(heading_error(target, pose, mount, beta), 0.0)
            self.assertAlmostEqual(pitch_for_target(target, pose, mount, beta), candidate.pitch_rad)
            self.assertAlmostEqual(
                math.hypot(target[0] - candidate.pivot_x, target[1] - candidate.pivot_y),
                distance,
            )
            self.assertAlmostEqual(candidate.pivot_z, 0.35)

    def test_existing_robot_pose_is_preserved_when_already_aimed(self):
        current = (0.7, -0.2, 0.1, 0.6)
        mount = (0.25, 0.12, 0.3)
        beta = -0.15
        cosine, sine = math.cos(current[3]), math.sin(current[3])
        pivot_x = current[0] + cosine * mount[0] - sine * mount[1]
        pivot_y = current[1] + sine * mount[0] + cosine * mount[1]
        distance = 1.1
        target = (
            pivot_x + distance * math.cos(current[3] + beta),
            pivot_y + distance * math.sin(current[3] + beta),
            current[2] + mount[2] + 0.2,
        )
        first = approach_candidates(target, current, distance, mount, beta)[0]
        self.assertAlmostEqual(first.base_x, current[0])
        self.assertAlmostEqual(first.base_y, current[1])
        self.assertAlmostEqual(first.base_yaw, current[3])

    def test_heading_wrap_and_sign(self):
        self.assertAlmostEqual(heading_error((0, 1, 0), (0, 0, 0, 0)), math.pi / 2)
        self.assertAlmostEqual(heading_error((1, 0, 0), (0, 0, 0, 2 * math.pi)), 0.0)
        self.assertAlmostEqual(pitch_for_target((1, 0, -0.5), (0, 0, 0, 0)), -math.atan(0.5))

    def test_bounded_yaw_rate_and_deadband(self):
        command = lambda error: yaw_rate_command(math.radians(error), math.radians(3), 1.2, 0.05, 0.25)
        self.assertEqual(command(2), 0.0)
        self.assertEqual(command(-2), 0.0)
        self.assertAlmostEqual(command(30), 0.25)
        self.assertAlmostEqual(command(-30), -0.25)
        self.assertGreater(command(4), 0.05)
        self.assertLess(command(4), 0.25)
        with self.assertRaises(ValueError):
            yaw_rate_command(math.nan, 0.05, 1.2, 0.05, 0.25)

    def test_fixed_position_body_yaw_accounts_for_lateral_offset(self):
        target = (2.0, 0.0, 0.5)
        base = (0.0, 0.0, 0.0, 0.4)
        mount = (0.3, 0.2, 0.25)
        yaw = body_yaw_for_target(target, base, mount)
        self.assertAlmostEqual(yaw, -math.asin(0.2 / 2.0))
        self.assertAlmostEqual(heading_error(target, (*base[:3], yaw), mount), 0.0)

    def test_fixed_position_body_yaw_accounts_for_mount_yaw(self):
        target = (1.5, -0.7, 0.6)
        base = (-0.2, 0.4, 0.1, -0.4)
        mount = (0.23, -0.12, 0.25)
        beta = 0.3
        yaw = body_yaw_for_target(target, base, mount, beta)
        self.assertAlmostEqual(heading_error(target, (*base[:3], yaw), mount, beta), 0.0)

    def test_fixed_position_body_yaw_rejects_unreachable(self):
        with self.assertRaises(ValueError):
            body_yaw_for_target((0.1, 0, 0.3), (0, 0, 0, 0), (0, 0.2, 0))
        with self.assertRaises(ValueError):
            body_yaw_for_target((0.1, 0, 0.3), (0, 0, 0, 0), (0.2, 0, 0))
        with self.assertRaises(ValueError):
            body_yaw_for_target((0, 0, 0.3), (0, 0, 0, 0))

    def test_pitch_limits_filter_candidates(self):
        self.assertEqual(approach_candidates((1, 0, 2), (0, 0, 0, 0), 1), [])
        self.assertEqual(approach_candidates((1, 0, -0.1), (0, 0, 0, 0), 1), [])

    def test_coincident_current_pivot_uses_current_axis(self):
        candidates = approach_candidates((0, 0, 0.2), (0, 0, 0, 0.3), 1)
        self.assertAlmostEqual(candidates[0].base_yaw, 0.3)

    def test_invalid_geometry(self):
        for distance in (0, -1, math.nan, math.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                approach_candidates((1, 0, 0.2), (0, 0, 0, 0), distance)
        for target in ((math.nan, 0, 1), (1, math.inf, 1), (1, 0)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pitch_for_target(target, (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            heading_error((0, 0, 1), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            pitch_for_target((0, 0, 1), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            pitch_for_target((1, 0, 0.2), (0, 0, 0, math.nan))
        with self.assertRaises(ValueError):
            pitch_for_target((1, 0, 0.2), (0, 0, 0, 0), mount_yaw_rad=math.inf)
        with self.assertRaises(ValueError):
            pitch_for_target((1, 0, 0.2), (0, 0, 0, 0), mount_xyz=(0, math.nan, 0))

    def test_invalid_offsets_and_limits(self):
        for offsets in ((), (0, 0), (0, 2 * math.pi), (0, math.nan)):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                approach_candidates((1, 0, 0.2), (0, 0, 0, 0), 1, angular_offsets_rad=offsets)
        for lower, upper in ((0.5, 0.1), (-2, 0.5), (0.0, math.inf)):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                approach_candidates(
                    (1, 0, 0.2), (0, 0, 0, 0), 1,
                    pitch_min_rad=lower, pitch_max_rad=upper,
                )

    def test_ballistic_pitch_reconstructs_trajectory(self):
        for distance, height, speed in ((2.0, 0.0, 6.0), (1.5, 0.3, 6.0), (2.0, -0.2, 5.0)):
            with self.subTest(distance=distance, height=height, speed=speed):
                pitch = ballistic_pitch(distance, height, speed)
                time = distance / (speed * math.cos(pitch))
                predicted_height = speed * math.sin(pitch) * time - 0.5 * 9.81 * time * time
                self.assertAlmostEqual(predicted_height, height)
                if height == 0.0:
                    self.assertAlmostEqual(pitch, 0.5 * math.asin(9.81 * distance / speed**2))

    def test_ballistic_candidates_can_aim_above_a_lower_target(self):
        target, base = (2.0, 0.0, 0.3), (0.0, 0.0, 0.0, 0.0)
        mount = (0.0, 0.0, 0.5)
        self.assertEqual(approach_candidates(target, base, 2.0, mount), [])
        candidates = approach_candidates(target, base, 2.0, mount, launch_speed_mps=5.0)
        self.assertEqual(len(candidates), 7)
        self.assertAlmostEqual(candidates[0].pitch_rad, ballistic_pitch(2.0, -0.2, 5.0))
        self.assertLess(pitch_for_target(target, base, mount), 0.0)

    def test_ballistic_invalid_and_unreachable(self):
        for arguments in (
            (0, 0, 5), (-1, 0, 5), (1, 0, 0), (1, 0, -1),
            (math.nan, 0, 5), (1, math.inf, 5), (1, 0, math.nan),
            (1, 0, 5, 0), (1, 0, 5, math.inf), (10, 0, 1), (1, 10, 1),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ballistic_pitch(*arguments)


if __name__ == "__main__":
    unittest.main()
