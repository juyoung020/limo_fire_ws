"""Physics regressions and independent verification of the exported tilt history."""
import csv
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from launch import arguments, load_model, run


class PhysicsTests(unittest.TestCase):
    def test_cad_projectile_topples_with_original_hold_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'cad-trace.csv'
            scene = Path(__file__).resolve().parent / 'scene_cad.xml'
            args = arguments(['--headless', '--scene', str(scene), '--speed', '4.2',
                              '--pitch', '30', '--duration', '6', '--csv', str(trace)])
            model = load_model(args)
            ball_id = model.geom('ball_geom').id
            self.assertAlmostEqual(float(model.geom_size[ball_id, 0]), 0.020, places=12)
            ball_body_id = model.body('ball').id
            self.assertAlmostEqual(float(model.body_mass[ball_body_id]), 0.0027, places=12)
            for name in ('cad_launcher_part_0', 'cad_launcher_part_1',
                         'cad_launcher_part_2', 'cad_launcher_part_3'):
                geom_id = model.geom(name).id
                self.assertEqual(int(model.geom_contype[geom_id]), 0)
                self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)

            result = run(args, quiet=True)
            with trace.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(result['contact'])
            self.assertTrue(result['toppled'])
            self.assertFalse(result['ground_before_target'])
            angles = [abs(float(row['target_tilt_deg'])) for row in rows]
            longest = current = 0
            for value in angles:
                if value >= 45:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            self.assertGreaterEqual(longest * result['timestep_s'], 0.20)

    def test_optional_aerodynamic_drag_reduces_ball_range(self):
        common = ['--headless', '--scene', str(Path(__file__).resolve().parent / 'scene_cad.xml'),
                  '--distance', '3', '--speed', '5', '--pitch', '30', '--duration', '0.3']
        no_drag = run(arguments(common), quiet=True)
        with_drag = run(arguments(common + ['--drag-coefficient', '0.45']), quiet=True)
        self.assertEqual(no_drag['drag_coefficient'], 0.0)
        self.assertAlmostEqual(with_drag['drag_coefficient'], 0.45)
        self.assertGreater(with_drag['initial_air_drag_N'], 0.0)
        self.assertLess(with_drag['ball_final_position_m'][0], no_drag['ball_final_position_m'][0])

    def test_legacy_clearance_reproduces_failure(self):
        result = run(arguments(['--headless', '--pivot-height', '0.02', '--duration', '3']), quiet=True)
        self.assertTrue(result['contact'])
        self.assertFalse(result['toppled'])
        self.assertAlmostEqual(result['final_tilt_deg'], 15.5, delta=0.1)
        self.assertLess(result['ground_contact_time_s'], 0.3)

    def test_exported_history_independently_satisfies_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.csv'
            result = run(arguments(['--headless', '--csv', str(path)]), quiet=True)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            # Find contiguous runs from exported angles, independently of the detector flag.
            spans, current = [], []
            for row in rows:
                if abs(float(row['target_tilt_deg'])) >= 45:
                    current.append(float(row['time_s']))
                elif current:
                    spans.append(current)
                    current = []
            if current:
                spans.append(current)
            qualifying = [span for span in spans if span[-1] - span[0] >= 0.2]
            self.assertTrue(qualifying)
            self.assertTrue(result['toppled'])
            start = qualifying[0][0]
            self.assertGreaterEqual(result['toppled_at_s'] - start, 0.2)
            self.assertLess(result['toppled_at_s'] - start, 0.2 + 2 * result['timestep_s'])
            self.assertLess(result['contact_time_s'], start)
            self.assertFalse(result['ground_before_target'])

    def test_no_launch_and_miss_do_not_topple(self):
        for overrides in ({'speed': 0}, {'yaw': 65}):
            args = arguments(['--headless'])
            for key, value in overrides.items():
                setattr(args, key, value)
            result = run(args, quiet=True)
            self.assertFalse(result['contact'])
            self.assertFalse(result['toppled'])

    def test_insufficient_hold_is_not_success(self):
        result = run(arguments(['--headless', '--duration', '0.45']), quiet=True)
        self.assertGreater(result['max_tilt_deg'], 45)
        self.assertFalse(result['toppled'])

    def test_mass_override_scales_inertia(self):
        nominal = load_model(arguments(['--headless']))
        heavy = load_model(arguments(['--headless', '--target-mass', '0.28']))
        bid = nominal.body('fire_target').id
        for expected, actual in zip(nominal.body_inertia[bid] * 2, heavy.body_inertia[bid]):
            self.assertAlmostEqual(float(expected), float(actual), places=12)

    def test_timestep_refinement_keeps_success(self):
        results = [run(arguments(['--headless', '--timestep', str(dt)]), quiet=True)
                   for dt in (0.002, 0.001, 0.0005)]
        self.assertTrue(all(r['toppled'] for r in results))
        times = [r['toppled_at_s'] for r in results]
        self.assertLess(max(times) - min(times), 0.003)

    def test_viewer_closed_early_saves_incomplete_result(self):
        # Exercise lifecycle with a fake viewer; this is not a rendered GUI test.
        from mujoco import viewer as mj_viewer

        def launch_fake(model, data):
            viewer = SimpleNamespace(
                cam=SimpleNamespace(lookat=[0, 0, 0]),
                lock=contextlib.nullcontext,
                is_running=lambda: data.time < 0.15,
                sync=lambda: None,
            )
            return contextlib.nullcontext(viewer)

        ticks = iter(i * 0.01 for i in range(1000))
        with tempfile.TemporaryDirectory() as folder:
            summary = Path(folder) / 'summary.json'
            trace = Path(folder) / 'trace.csv'
            args = arguments(['--summary', str(summary), '--csv', str(trace)])
            with patch.object(mj_viewer, 'launch_passive', side_effect=launch_fake), \
                    patch('launch.time.perf_counter', side_effect=lambda: next(ticks)), \
                    patch('launch.time.sleep'):
                result = run(args, quiet=True)
            self.assertEqual(result['termination'], 'viewer_closed_early')
            self.assertFalse(result['completed'])
            self.assertFalse(result['toppled'])
            self.assertGreaterEqual(result['simulated_s'], 0.15)
            self.assertLess(result['simulated_s'], args.duration)
            self.assertEqual(json.loads(summary.read_text()), result)
            with trace.open() as f:
                rows = list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[-1]['time_s']), result['simulated_s'])


if __name__ == '__main__':
    unittest.main()
