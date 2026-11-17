"""V4 forecast windows preserve time direction and honest uncertainty baselines."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from core.motion_uncertainty import MotionConfig, MotionUncertaintyNet
from core.workflows.step10e_maneuver_uncertainty import ForecastWindows, _cv_prediction, _cv_distribution
from core.workflows.step10e_maneuver_uncertainty import add_observations, distribution_metrics


def trajectory(steps=60):
    times = np.arange(steps)*.05
    position = np.stack((100+20*times, np.full(steps, 100.), np.full(steps, 70.)), -1)[None].astype('float32')
    return {'position': position, 'beam': np.ones((1, steps), np.int64),
            'trajectory_id': np.array([100000]), 'motion_type': np.zeros((1, steps), np.int8),
            'trajectory_kind': np.array([0]), 'maneuver_onset': np.zeros((1, steps), bool)}


class V4WorkflowTests(unittest.TestCase):
    def test_targets_are_strictly_future_and_never_input_features(self):
        raw = trajectory()
        data, _ = add_observations(raw, 33)
        config = MotionConfig()
        first = ForecastWindows(data, config, stride=5)[0]
        end = config.history-1
        changed = {key: value.copy() for key, value in data.items()}
        changed['position'][:, end+1:] += 10000
        changed['motion_type'][:] = 3
        changed['maneuver_onset'][:] = True
        second = ForecastWindows(changed, config, stride=5)[0]
        for actual, expected in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(first[3], second[3]))
        np.testing.assert_array_equal(first[3], data['position'][0, end+np.asarray(config.horizons)])

    def test_offline_filter_prefix_never_uses_future_observations(self):
        raw = trajectory()
        a, _ = add_observations(raw, 41)
        changed = {key: value.copy() for key, value in raw.items()}
        changed['position'][:, 30:, 1] += 100
        b, _ = add_observations(changed, 41)
        np.testing.assert_array_equal(a['measurement'][:, :30], b['measurement'][:, :30])
        np.testing.assert_array_equal(a['filtered_measurement'][:, :30], b['filtered_measurement'][:, :30])

    def test_cv_baseline_preserves_last_observation_age(self):
        model = MotionUncertaintyNet(MotionConfig(d_model=16, heads=4, layers=1, dropout=0)).eval()
        data, _ = add_observations(trajectory(), 31)
        window = ForecastWindows(data, model.config)[0]
        history, valid, sigma = [torch.from_numpy(value)[None] for value in window[:3]]
        valid[:, -4:] = False
        with torch.inference_mode():
            pred = model(history, valid, sigma=sigma)
        base = _cv_prediction(pred, model.config.horizons)
        expected = pred['origin'][:, None]+pred['velocity'][:, None]*(
            torch.as_tensor(model.config.horizons)[None, :, None]*.05+.2)
        torch.testing.assert_close(base, expected)

    def test_fixed_spread_baseline_keeps_forecast_mean_and_valid_density(self):
        mean = torch.zeros(40, 4, 3)
        sigma = torch.full((4, 3), 2.)
        pred = _cv_distribution(mean, sigma)
        rows = distribution_metrics(pred, mean, seed=8)
        self.assertTrue(all(row['rmse_m'] == 0 for row in rows))
        self.assertTrue(all(row['marginal_coverage_90'] == 1 for row in rows))
        self.assertTrue(all(np.isfinite(row['nll']) and row['marginal_width_90_m'] > 0 for row in rows))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
