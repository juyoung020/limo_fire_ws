import importlib.util
import math
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / 'limo_fire_hardware' / 'pitch_serial_bridge.py'
SPEC = importlib.util.spec_from_file_location('pitch_serial_bridge', MODULE_PATH)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


class PitchProtocolTest(unittest.TestCase):
    def test_only_pitch_ascii_is_emitted(self):
        self.assertEqual(protocol.pitch_command(15), (15.0, b'a15.0\n'))
        self.assertEqual(protocol.pitch_command(5), (5.0, b'a5.0\n'))
        self.assertEqual(protocol.pitch_command(30), (30.0, b'a30.0\n'))

    def test_invalid_values_are_rejected_before_firmware_clamps_them(self):
        for value in (math.nan, math.inf, -math.inf, 4.9, 30.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.pitch_command(value)

    def test_configured_limits_cannot_expand_firmware_limits(self):
        for limits in ((0, 30), (5, 45), (20, 10), (math.nan, 30)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                protocol.pitch_command(15, *limits)

    def test_rounding_must_stay_within_calibrated_limits(self):
        with self.assertRaises(ValueError):
            protocol.pitch_command(5.01, 5.01, 29.99)

    def test_unrelated_firmware_text_does_not_acknowledge_pitch(self):
        for line in ('launcher_test ready', 'ANGLE 15', 'lever servo 15.0', 'home', ''):
            self.assertFalse(protocol.matching_pitch_reply(line, 15.0))

    def test_matching_reply_accepts_crlf_and_precision(self):
        self.assertTrue(protocol.matching_pitch_reply('pitch 15.0\r\n', 15.0))

    def test_wrong_and_nonfinite_replies_are_rejected(self):
        for line in ('pitch 16.0', 'pitch nan', 'pitch inf', 'pitch 31', 'pitch 15 extra'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                protocol.matching_pitch_reply(line, 15.0)

    def test_only_launcher_firmware_identity_is_accepted(self):
        self.assertTrue(protocol.launcher_identity('launcher_test ready — ? 로 상태, s 로 한 발'))
        self.assertTrue(protocol.launcher_identity(
            'lever 90.0 (rest 90.0 cock 20.0 fire 115.0) | pitch 15.0 deg, tilt servo 90.0'))
        for line in ('READY', 'ANGLE 90', 'pitch 15.0', 'GPIO4 servo test: complete'):
            self.assertFalse(protocol.launcher_identity(line))

    def test_guarded_esp32_requires_pitch_enable(self):
        line = ('lever 100.0 (rest 100.0 cock 30.0 fire 125.0, OFF) '
                '| pitch 15.0 deg, tilt servo 135.0 (OFF)')
        self.assertTrue(protocol.launcher_identity(line))
        self.assertTrue(protocol.guarded_pitch_firmware(line))
        self.assertTrue(protocol.guarded_pitch_firmware('launcher_test ready — PWM OFF; ? state, e4/e5 enable'))
        self.assertFalse(protocol.guarded_pitch_firmware('launcher_test ready — ? 로 상태, s 로 한 발'))

    def test_bad_status_cannot_identify_firmware(self):
        for pitch in ('nan', 'inf', '31.0'):
            line = f'lever 90.0 (rest 90.0 cock 20.0 fire 115.0) | pitch {pitch} deg, tilt servo 90.0'
            self.assertFalse(protocol.launcher_identity(line))

    def test_uno_status_identifies_firmware_without_accepting_generic_ready(self):
        self.assertFalse(protocol.launcher_identity('READY\r\n'))
        self.assertTrue(protocol.launcher_identity(
            'lever 100.0 (rest 100.0 cock 30.0 fire 125.0) | pitch 15.0 | tilt servo 90.0 | targets OK\r\n'))

    def test_uno_invalid_configuration_or_angles_cannot_identify_firmware(self):
        valid = ('lever 100.0 (rest 100.0 cock 30.0 fire 125.0) '
                 '| pitch 15.0 | tilt servo 90.0 | targets OK')
        for line in (
                valid.replace('targets OK', 'targets INVALID'),
                valid.replace('targets OK', 'targets'),
                valid.replace('pitch 15.0', 'pitch 31.0'),
                valid.replace('pitch 15.0', 'pitch nan'),
                valid.replace('tilt servo 90.0', 'tilt servo 181.0'),
                valid.replace('cock 30.0', 'cock -1.0'),
                valid + ' extra'):
            with self.subTest(line=line):
                self.assertFalse(protocol.launcher_identity(line))

    def test_startup_banners_detect_uno_and_esp32_restarts(self):
        for line in ('READY\r\n', 'launcher_test ready — ? 로 상태, s 로 한 발'):
            self.assertTrue(protocol.launcher_startup_banner(line))
        for line in ('READY extra', 'pitch 15.0', 'targets OK', ''):
            self.assertFalse(protocol.launcher_startup_banner(line))

    def test_shot_requires_exact_ordered_firmware_replies(self):
        index = 0
        for line in ('lever PWM ON', 'shot: cock', 'shot: fire', 'done'):
            index = protocol.shot_reply_index('shot', index, line)
        self.assertEqual(index, 3)
        with self.assertRaises(ValueError):
            protocol.shot_reply_index('shot', 0, 'shot: fire')
        with self.assertRaises(ValueError):
            protocol.shot_reply_index('fire', 0, 'blocked: use e4 first')


if __name__ == '__main__':
    unittest.main()
