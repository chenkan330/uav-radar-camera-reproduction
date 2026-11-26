"""Tracking regressions for the moving-platform, single-vehicle comparison."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import uav_vehicle_tracking as tracking
from core.uav_vehicle_scene import build_scene


def clean_batch(count, frames):
    scenes = [build_scene(seed, frames) for seed in range(count)]
    for scene in scenes:
        scene.noise.fill(0)
    return tracking.prepare(scenes)


def beam_scores(num_rows, beams):
    logits = torch.zeros((num_rows, 1, 1024))
    logits[:, 0, beams] = torch.arange(4, 0, -1, dtype=torch.float32)
    return logits


class PredictFirstFour:
    config = SimpleNamespace(failure_limit=5)

    def __call__(self, history, valid):
        # Identity normalization below gives an exact post-detection position head.
        idx = (valid * torch.arange(5)).max(dim=1).values
        last_measurement = history[torch.arange(len(history)), idx].numpy()
        position = tracking.cartesian(last_measurement) + tracking.REFERENCE_BS
        return {'logits': beam_scores(len(history), [0, 1, 2, 3]),
                'position': torch.from_numpy(position)}


class RecoverWithNextFour:
    def __init__(self):
        self.failed_histories = []

    def __call__(self, nnet, history, valid, failed, count):
        for f, c in zip(failed.numpy(), count.tolist()):
            self.failed_histories.append((c, f.copy()))
        return beam_scores(len(history), [4, 5, 6, 7])


class UavVehicleTrackingTests(unittest.TestCase):
    def test_array_conversion_and_measurement_geometry(self):
        enu = np.array([[2., -3., -60.], [-3., 4., -60.], [0., 0., -60.]])
        local = tracking.world_to_array(enu)
        np.testing.assert_array_equal(local, [[-2, -3, 60], [3, 4, 60], [0, 0, 60]])
        # At nadir azimuth is undefined; range and polar angle are observable.
        np.testing.assert_allclose(tracking.spherical(local)[2, [0, 2]], [60, 0])
        np.testing.assert_allclose(tracking.cartesian(tracking.spherical(local)), local, atol=1e-12)
        np.testing.assert_allclose(tracking.world_to_array(local), enu)
        batch = clean_batch(2, 14)
        np.testing.assert_allclose(batch.observed_world, batch.vehicle, atol=1e-12)

    def test_history_excludes_missed_samples_and_future_navigation(self):
        batch = clean_batch(2, 14)
        query_rows = np.array([1, 0])
        steps = np.array([9, 7])
        valid = np.array([[True, False, True, True, False], [False, True, True, False, True]])
        expected = []
        for row, step, mask in zip(query_rows, steps, valid):
            expected.append(tracking.world_to_array(batch.vehicle[row, step-5:step] - batch.platform[row, step]))
            batch.observed_world[row, np.arange(step-5, step)[~mask]] = np.nan
            batch.observed_world[row, step:] = np.nan
            batch.platform[row, step:] = np.nan
        hist = tracking.make_history(batch, query_rows, steps, valid)
        np.testing.assert_allclose(tracking.cartesian(hist)[valid], np.asarray(expected)[valid], atol=1e-4)
        np.testing.assert_array_equal(hist[~valid], 0)
        np.testing.assert_array_equal(np.isfinite(hist), True)

    def test_noiseless_cv_selects_exact_future_beam_across_missing_frames(self):
        batch = clean_batch(4, 201)
        rows = np.arange(4)
        valid = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1],
                          [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
        for step in (5, 25, 61, 114, 180, 200):
            hist = tracking.make_history(batch, rows, step, valid)
            picks = tracking.pick_cv(hist, valid)
            self.assertEqual(picks.shape, (4, tracking.BUDGET))
            np.testing.assert_array_equal(picks[:, 0], batch.beam[:, step])

    def test_routes_recoveries_five_misses_and_distinct_denominators(self):
        batch = clean_batch(3, 13)
        # Scene 0 loses and recovers twice, scene 1 terminates after five
        # misses, and scene 2 is always detected.
        batch.beam[:, 5:] = [[0, 7, 6, 9, 9, 4, 0, 0], [9]*8, [0]*8]
        ck = {'normalization': {'mean': [0]*3, 'std': [1]*3}}

        def fixed_cv(history, valid):
            return np.tile(np.arange(4), (len(history), 1))

        for method, normal_name in [('cv_rnet', 'CV'), ('nrnet', 'N')]:
            with self.subTest(method=method), patch.object(tracking, 'pick_cv', fixed_cv):
                recovery = RecoverWithNextFour()
                rows, events = tracking.simulate(batch, PredictFirstFour(), recovery, ck, 'cpu', method)
                modes = [[e['mode'] for e in events if e['scene'] == scene] for scene in range(3)]
                self.assertEqual(modes[0], [normal_name, normal_name, 'R', normal_name, 'R', 'R',
                                            normal_name, normal_name])
                self.assertEqual(modes[1], [normal_name] + ['R']*4)
                self.assertEqual(modes[2], [normal_name]*8)
                self.assertEqual([r['success'] for r in rows], [True, False, True])
                self.assertEqual([r['attempts'] for r in rows], [8, 5, 8])
                for count, failures in recovery.failed_histories:
                    np.testing.assert_array_equal(failures[0], [0, 1, 2, 3])
                    for item in failures[1:count]:
                        np.testing.assert_array_equal(item, [4, 5, 6, 7])
                    self.assertTrue((failures[count:] == -1).all())
                summary = tracking.summarize(rows)
                self.assertEqual(summary['hits'], 13)
                self.assertAlmostEqual(summary['detection_rate'], 13/24)
                self.assertAlmostEqual(summary['attempted_hit_rate'], 13/21)
                self.assertAlmostEqual(summary['recovery_rate'], 2/3)
                self.assertAlmostEqual(summary['normal_hit_rate'], 11/14)
                self.assertAlmostEqual(summary['r_hit_rate'], 2/7)
                self.assertLess(summary['position_rmse_m'], 1e-4)
                for event in events:
                    self.assertEqual(np.isfinite(event['estimate_x']), bool(event['hit']))


if __name__ == '__main__':
    unittest.main()
