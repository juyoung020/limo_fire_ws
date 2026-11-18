import unittest
from types import SimpleNamespace as Obj

from limo_fire_hardware.mapper_observation_bridge import select_live_fire


def fire(id='fire-1', score=0.9, age=0.1, x=1.0):
    covariance = [0.0] * 36
    covariance[35] = age
    result = Obj(hypothesis=Obj(class_id='fire', score=score),
                 pose=Obj(covariance=covariance))
    box = Obj(center=Obj(position=Obj(x=x, y=2.0, z=0.3)))
    return Obj(id=id, results=[result], bbox=box)


class LiveMapperObservationTests(unittest.TestCase):
    def test_stale_republished_landmark_cannot_drive_aiming(self):
        detection, _, reason = select_live_fire([fire(age=2.0)])
        self.assertIsNone(detection)
        self.assertIn('no recently', reason)

    def test_two_fires_require_an_explicit_target(self):
        pair = [fire('fire-1'), fire('fire-2')]
        self.assertIn('multiple', select_live_fire(pair)[2])
        self.assertEqual(select_live_fire(pair, target_id='fire-2')[0].id, 'fire-2')

    def test_fresh_sensor_mapped_fire_is_selected(self):
        chosen, age, reason = select_live_fire([fire()])
        self.assertEqual((chosen.id, age, reason), ('fire-1', 0.1, ''))


if __name__ == '__main__':
    unittest.main()
