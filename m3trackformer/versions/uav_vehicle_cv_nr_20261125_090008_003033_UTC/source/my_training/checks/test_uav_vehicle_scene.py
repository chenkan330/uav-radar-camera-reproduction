"""Check the geometry and causal flight commands used by both trackers."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.uav_vehicle_scene import (
    ALTITUDE_M, DT, INITIAL_BEARING_FROM_NORTH_DEG, INITIAL_HORIZONTAL_RANGE_M,
    ROAD_ANGLE_FROM_EAST_WEST_DEG, VEHICLE_SPEED_MPS, build_scene,
)


class UavVehicleSceneTests(unittest.TestCase):
    def test_level_curved_flight_and_car_speed(self):
        saw_reversal = saw_sustained_turn = False
        for seed in range(24):
            scene = build_scene(seed)
            np.testing.assert_allclose(np.diff(scene.time), DT)
            np.testing.assert_array_equal(scene.platform[:, 2], ALTITUDE_M)
            np.testing.assert_array_equal(scene.vehicle[:, 2], 0.0)
            velocity = np.diff(scene.vehicle, axis=0) / DT
            np.testing.assert_allclose(np.max(np.abs(velocity - velocity[0])), 0, atol=3e-12)
            self.assertAlmostEqual(np.linalg.norm(velocity[0]), VEHICLE_SPEED_MPS)
            initial = scene.vehicle[0, :2] - scene.platform[0, :2]
            horizontal_range = np.linalg.norm(initial)
            bearing = np.rad2deg(np.arctan2(initial[0], initial[1]))
            road_angle = np.rad2deg(np.arctan2(velocity[0, 1], velocity[0, 0]))
            offset = (road_angle + 90) % 180 - 90
            for x, (low, high) in [(horizontal_range, INITIAL_HORIZONTAL_RANGE_M),
                                   (bearing, INITIAL_BEARING_FROM_NORTH_DEG),
                                   (offset, ROAD_ANGLE_FROM_EAST_WEST_DEG)]:
                self.assertTrue(low <= x <= high)
            heading = np.unwrap(np.arctan2(scene.command[:, 1], scene.command[:, 0]))
            self.assertGreater(np.ptp(heading), np.deg2rad(12))
            rates = np.diff(heading) / DT
            np.testing.assert_allclose(rates, scene.turn_rate[1:-1], atol=4e-4)
            saw_reversal |= rates.min() < 0 < rates.max()
            saw_sustained_turn |= rates.min() > 0 or rates.max() < 0
        self.assertTrue(saw_reversal and saw_sustained_turn)

    def test_preissued_command_and_clip_prefix(self):
        short = build_scene(25, frames=56)
        long = build_scene(25, frames=201)
        for name in ('platform', 'vehicle', 'command', 'noise', 'time', 'speed', 'turn_rate'):
            prefix = getattr(short, name)
            np.testing.assert_allclose(getattr(long, name)[:len(prefix)], prefix, rtol=0, atol=1e-12)
        np.testing.assert_allclose(long.platform[1:] - long.platform[:-1], long.command, atol=1e-12)
        self.assertAlmostEqual(long.time[-1], 10.0)
        other = build_scene(26)
        self.assertFalse(np.allclose(short.noise, other.noise[:56]))
        self.assertFalse(np.allclose(short.command, other.command[:55]))
        self.assertFalse(np.allclose(short.vehicle, other.vehicle[:56]))


if __name__ == '__main__':
    unittest.main()
