import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).parents[1] / 'tools' / 'write_mount_config.py'
SPEC = importlib.util.spec_from_file_location('write_mount_config', MODULE)
writer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(writer)


class MountConfigTest(unittest.TestCase):
    def test_cm_measurements_become_base_link_meters(self):
        template = writer.TEMPLATE.read_text(encoding='utf-8')
        result = writer.make_config(template, 7, -2, 31, 3)
        self.assertIn('mount_xyz_m: [0.0700, -0.0200, 0.1650]', result)
        self.assertIn('mount_yaw_deg: 3.000', result)
        self.assertIn('mount_calibrated: true', result)
        self.assertIn('motion_enabled: false', result)

    def test_bad_or_incomplete_template_is_rejected(self):
        with self.assertRaises(ValueError):
            writer.make_config('mount_xyz_m: [0, 0, 0]', 7, -2, 31, 3)
        template = writer.TEMPLATE.read_text(encoding='utf-8')
        for values in ((700, 0, 31, 0), (0, 0, 10, 0), (0, 0, 31, 181)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                writer.make_config(template, *values)


if __name__ == '__main__':
    unittest.main()
