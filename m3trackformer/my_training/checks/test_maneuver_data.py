"""V4 mobility physics, reproducibility, split isolation and unknowable forks."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.maneuver_data import (ManeuverConfig, centered_dft_beams, fork_future,
                                generate_split, protocol)


class ManeuverDataTests(unittest.TestCase):
    def test_controls_preserve_speed_and_continuous_integration(self):
        data = generate_split(10, steps=1001, seed=4)
        speed = np.linalg.norm(data['velocity'][..., :2], axis=-1)
        self.assertGreaterEqual(float(speed.min()), 10.-1e-4)
        self.assertLessEqual(float(speed.max()), 35.+1e-4)
        self.assertLessEqual(float(np.abs(np.diff(speed, axis=1)/.05).max()), 8.+1e-3)
        old, new = data['velocity'][:, :-1, :2], data['velocity'][:, 1:, :2]
        cross = old[..., 0]*new[..., 1]-old[..., 1]*new[..., 0]
        dot = np.einsum('nti,nti->nt', old, new)
        angle_delta = np.abs(np.arctan2(cross, dot))
        lateral_acceleration = angle_delta/.05*np.maximum(speed[:, :-1], speed[:, 1:])
        self.assertLessEqual(float(lateral_acceleration.max()), 12.+1e-3)
        delta = np.diff(data['position'], axis=1)
        expected = (data['velocity'][:, 1:]+data['velocity'][:, :-1])*.025
        np.testing.assert_allclose(delta, expected, atol=1e-4)
        self.assertTrue(np.isfinite(data['position']).all())
        self.assertLessEqual(float(np.linalg.norm(delta, axis=-1).max()), 35.*.05+1e-3)
        self.assertTrue(np.all(data['velocity'][..., 2] == 0))

    def test_kinematic_regimes_and_event_labels_are_present(self):
        data = generate_split(20, steps=301, seed=12)
        self.assertEqual(set(data['trajectory_kind'].tolist()), set(range(5)))
        mixed = data['trajectory_kind'] == 4
        self.assertGreaterEqual(len(np.unique(data['motion_type'][mixed])), 3)
        self.assertTrue(data['maneuver_onset'][mixed].any())
        self.assertTrue(np.array_equal(data['beam'], centered_dft_beams(data['position'])))
        self.assertGreaterEqual(int(data['beam'].min()), 1)
        self.assertLessEqual(int(data['beam'].max()), 1024)

    def test_generation_is_reproducible_without_mutating_global_rng(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        a = generate_split(3, steps=51, seed=13, id_offset=20)
        np.testing.assert_equal(np.random.random(5), expected)
        b = generate_split(3, steps=51, seed=13, id_offset=20)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        alone = generate_split(1, steps=51, seed=13, id_offset=21)
        np.testing.assert_array_equal(a['position'][1], alone['position'][0])

    def test_split_seed_and_id_offsets_keep_synthetic_records_independent(self):
        train = generate_split(10, steps=51, seed=3, id_offset=0)
        test = generate_split(10, steps=51, seed=7, id_offset=1000)
        self.assertFalse(set(train['trajectory_id']) & set(test['trajectory_id']))
        self.assertFalse(np.array_equal(train['position'], test['position']))

    def test_initial_reference_geometry_is_used_without_copying_its_future(self):
        anchors = np.array([[200., 100., 60.], [210., 120., 70.]])
        data = generate_split(10, steps=41, seed=3, initial_references=anchors)
        self.assertTrue(all(any(np.array_equal(row, p) for p in anchors) for row in data['position'][:, 0]))
        self.assertTrue(np.any(data['position'][:, 1] != data['position'][:, 0]))
        directed = generate_split(1, steps=11, seed=4, kind='constant_speed',
                                  initial_references={'position': anchors[:1], 'velocity': np.array([[1., 0., 0.]])})
        self.assertTrue(np.allclose(directed['position'][0, :, 1], 100.))

    def test_shared_prefix_can_have_different_unobservable_future_branches(self):
        source = generate_split(1, steps=11, seed=8, kind='constant_speed')
        prefix_p, prefix_v = source['position'][0], source['velocity'][0]
        branches = [fork_future(prefix_p, prefix_v, steps=40, seed=i) for i in range(5)]
        for branch in branches:
            np.testing.assert_array_equal(branch['position'][:11], prefix_p)
            np.testing.assert_array_equal(branch['velocity'][:11], prefix_v)
            self.assertEqual(branch['fork_index'], 11)
        self.assertTrue(any(not np.array_equal(branches[0]['position'][11:], row['position'][11:])
                            for row in branches[1:]))

    def test_geofence_steers_without_position_reset(self):
        data = generate_split(1, steps=100, seed=7, kind='constant_speed',
                              initial_references={'position': np.array([[910., 0., 60.]]),
                                                  'velocity': np.array([[20., 0., 0.]])})
        self.assertTrue(np.any(data['motion_type'] == 2))
        self.assertLess(float(np.linalg.norm(np.diff(data['position'][0], axis=0), axis=1).max()), 2.)
        self.assertTrue(protocol()['tangential_acceleration_is_not_total_vector_acceleration'])

    def test_invalid_configuration_and_shapes_are_rejected(self):
        for args in ({'count': 0}, {'count': 1, 'dt': 0}, {'count': 1, 'steps': 1},
                     {'count': 1, 'kind': 'teleport'}, {'count': 1, 'id_offset': -1},
                     {'count': 1, 'initial_references': np.zeros((3, 2))}):
            with self.assertRaises(ValueError):
                generate_split(**args)
        with self.assertRaises(ValueError):
            generate_split(1, config=ManeuverConfig(minimum_speed_m_s=40.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
