import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from limo_fire_hardware.aim_learning import FIELDS, fit_bandit, read_shots


def shot(**changes):
    row = dict(run_id='r1', shot_id='1', recorded_at_utc='2026-11-19T10:00:00Z',
               source='real_camera', ball_color='blue', launch_speed_mps='3.9',
               target_id='fire-1', range_m='1.2', height_delta_m='0.1',
               baseline_yaw_deg='10', baseline_pitch_deg='20',
               delta_yaw_deg='0', delta_pitch_deg='0',
               impact_error_right_m='0.1', impact_error_up_m='0',
               contact='false', toppled='false', notes='')
    row.update(changes)
    return row


class ShotLearningTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_shots_are_rejected_before_training(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'shots.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(shot())
                writer.writerow(shot())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_shots(path)
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(shot(impact_error_right_m='nan'))
            with self.assertRaisesRegex(ValueError, 'finite'):
                read_shots(path)

    def test_simulation_and_unmeasured_shots_cannot_train_real_cells(self):
        rows = [shot(shot_id=str(i)) for i in range(3)]
        rows += [shot(shot_id='sim', source='mujoco', impact_error_right_m='0')]
        rows += [shot(shot_id='unknown', impact_error_right_m='', impact_error_up_m='')]
        model = fit_bandit(rows, min_samples=3)
        self.assertEqual(model['measured_shots_used'], 3)
        self.assertEqual(len(model['cells']), 1)
        self.assertAlmostEqual(model['cells'][0]['q_value'], -0.1)
        self.assertFalse(model['runtime_enabled'])

    def test_each_action_needs_its_own_evidence(self):
        rows = [shot(shot_id=str(i), delta_pitch_deg='0.5',
                     impact_error_right_m='0.05') for i in range(3)]
        rows += [shot(shot_id='unproven', delta_pitch_deg='-0.5',
                      impact_error_right_m='0')]
        model = fit_bandit(rows, min_samples=3)
        self.assertEqual(len(model['cells']), 1)
        self.assertEqual(model['cells'][0]['delta_pitch_bin'], 1)
        self.assertEqual(model['cells'][0]['count'], 3)


if __name__ == '__main__':
    unittest.main()
