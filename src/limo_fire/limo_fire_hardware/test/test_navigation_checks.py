"""Grid clearance regression cases without ROS imports."""
import math
from types import SimpleNamespace
import unittest

from limo_fire_hardware.aim_geometry import ApproachCandidate
from limo_fire_hardware.navigation_checks import candidate_clear, grid_cell


def make_grid(origin_x=0.0, origin_y=0.0, yaw=0.0):
    origin = SimpleNamespace(
        position=SimpleNamespace(x=origin_x, y=origin_y),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)))
    info = SimpleNamespace(origin=origin, resolution=0.1, width=40, height=40)
    return SimpleNamespace(info=info, data=[0] * (info.width * info.height))


def set_cell(grid, col, row, value):
    grid.data[row * grid.info.width + col] = value


class NavigationChecksTest(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid()
        self.candidate = ApproachCandidate(
            base_x=0.85, base_y=0.85, base_yaw=0.0,
            pivot_x=0.85, pivot_y=0.85, pivot_z=0.2,
            distance_m=2.0, pitch_rad=0.1)
        self.target = (2.85, 0.85, 0.4)

    def clear(self, grid=None, candidate=None, margin=0.15):
        return candidate_clear(grid or self.grid, candidate or self.candidate,
                               self.target, 0.2, 65, margin)

    def test_free_footprint_and_sight_line(self):
        self.assertTrue(self.clear())

    def test_unknown_cell_in_footprint_rejected(self):
        set_cell(self.grid, 9, 9, -1)
        self.assertFalse(self.clear())

    def test_occupied_cell_in_footprint_rejected(self):
        set_cell(self.grid, 8, 9, 100)
        self.assertFalse(self.clear())

    def test_out_of_bounds_footprint_rejected(self):
        candidate = ApproachCandidate(
            base_x=0.1, base_y=0.85, base_yaw=0.0,
            pivot_x=0.1, pivot_y=0.85, pivot_z=0.2,
            distance_m=2.75, pitch_rad=0.1)
        self.assertFalse(self.clear(candidate=candidate))

    def test_unknown_cell_on_sight_line_rejected(self):
        set_cell(self.grid, 18, 8, -1)
        self.assertFalse(self.clear())

    def test_obstruction_before_target_margin_rejected(self):
        set_cell(self.grid, 24, 8, 100)
        self.assertFalse(self.clear(margin=0.25))

    def test_target_surface_inside_endpoint_margin_allowed(self):
        set_cell(self.grid, 28, 8, 100)
        self.assertTrue(self.clear(margin=0.25))
        self.assertFalse(self.clear(margin=0.0))

    def test_cost_threshold_is_inclusive(self):
        set_cell(self.grid, 18, 8, 64)
        self.assertTrue(self.clear())
        set_cell(self.grid, 18, 8, 65)
        self.assertFalse(self.clear())

    def test_rotated_origin_maps_world_position_into_grid(self):
        grid = make_grid(origin_x=5.0, origin_y=-1.0, yaw=math.pi / 2)
        set_cell(grid, 2, 3, 37)
        self.assertEqual(grid_cell(grid, 4.65, -0.75), 37)
        self.assertIsNone(grid_cell(grid, 5.2, -1.0))

    def test_rotated_origin_footprint_and_sight_are_checked(self):
        # Rotating the whole free-grid fixture by90deg preserves its clearance.
        grid = make_grid(origin_x=5.0, origin_y=-1.0, yaw=math.pi / 2)
        candidate = ApproachCandidate(
            base_x=4.15, base_y=-0.15, base_yaw=math.pi / 2,
            pivot_x=4.15, pivot_y=-0.15, pivot_z=0.2,
            distance_m=2.0, pitch_rad=0.1)
        target = (4.15, 1.85, 0.4)
        self.assertTrue(candidate_clear(grid, candidate, target, 0.2, 65, 0.15))
        set_cell(grid, 18, 8, 100)
        self.assertFalse(candidate_clear(grid, candidate, target, 0.2, 65, 0.15))

    def test_invalid_resolution_and_truncated_data_rejected(self):
        self.grid.info.resolution = 0.0
        self.assertFalse(self.clear())
        self.grid.info.resolution = 0.1
        self.grid.data = []
        self.assertFalse(self.clear())


if __name__ == '__main__':
    unittest.main()
