"""Optimizers cannot read missing measurements, change frozen networks, or use test ranking."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from core.measurement_quality import RobustMeasurementProcessor
from core.optimization_processors import RobustIMMProcessor, candidates, make_processor
from core.optimization_experiment import selection_key
from core.multi_target import simulate
from test_trajectory_figure import FixedPositionNet, NORMALIZATION, fixture


class OptimizationTests(unittest.TestCase):
    def test_defaults_preserve_existing_huber(self):
        d = fixture(); z, sigma = d['measurement'][..., :5, :], d['measurement_sigma'][..., :5, :]
        a = RobustMeasurementProcessor(z, sigma, .05)
        b = make_processor(z, sigma, .05, configuration={'kind': 'huber'})
        np.testing.assert_array_equal(a.filtered_initial_history, b.filtered_initial_history)
        for t in range(5, 12):
            args = (t, np.ones((1, 1), bool), d['measurement'][..., t, :], d['measurement_sigma'][..., t, :])
            np.testing.assert_array_equal(a.step(*args)[0], b.step(*args)[0])

    def test_all_processors_hide_misses_and_initialization_future(self):
        d = fixture(); z, sigma = d['measurement'][..., :5, :], d['measurement_sigma'][..., :5, :]
        configurations = [c['processor'] for c in candidates() if c['processor'] is not None]
        for config in configurations:
            a = make_processor(z, sigma, .05, configuration=config)
            b = make_processor(z, sigma, .05, configuration=config)
            for t in range(5, 9):
                empty = np.zeros((1, 1), bool)
                x = a.step(t, empty, np.full((1, 1, 3), np.nan), np.full((1, 1, 3), np.nan))[0]
                y = b.step(t, empty, np.full((1, 1, 3), 1e12), np.full((1, 1, 3), 1e6))[0]
                np.testing.assert_array_equal(x, y)
            changed = z.copy(); changed[..., -1, 0] += 80
            c = make_processor(changed, sigma, .05, configuration=config)
            initial = make_processor(z, sigma, .05, configuration=config)
            np.testing.assert_array_equal(c.filtered_initial_history[..., :-1, :], initial.filtered_initial_history[..., :-1, :])

    def test_imm_covariance_and_mode_probabilities_remain_valid(self):
        d = fixture(); z, sigma = d['measurement'][..., :5, :], d['measurement_sigma'][..., :5, :]
        p = RobustIMMProcessor(z, sigma, .05)
        for t in range(5, 40):
            measurement = z[..., 0, :].copy()
            measurement[..., 0] += 100 if t == 10 else (t-5)*.2
            filtered, _ = p.step(t, np.ones((1, 1), bool), measurement, sigma[..., 0, :])
            self.assertTrue(np.isfinite(filtered).all())
            np.testing.assert_allclose(p.probabilities.sum(-1), 1.)
            self.assertTrue((p.probabilities >= 0).all())
            for model in p.models:
                self.assertGreater(np.linalg.eigvalsh(model.covariance).min(), -1e-7)

    def test_factory_and_temperature_keep_nnet_trace_output(self):
        from functools import partial
        d, trace = fixture(), []
        factory = partial(make_processor, configuration={'kind': 'student', 'student_df': 3.})
        result = simulate(FixedPositionNet(), None, NORMALIZATION, d, torch.device('cpu'), .05, 3, 'joint',
            quality_mode='robust', measurement_processor_factory=factory, logit_temperature=1.3,
            trajectory_trace=trace, trajectory_estimator='nnet')
        self.assertTrue(all(r['estimated_x_m'] == 3. for r in trace if r['phase'] == 'tracking' and r['hit']))
        self.assertEqual(result[1][0]['recoveries'], 1)
        for value in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                simulate(FixedPositionNet(), None, NORMALIZATION, d, torch.device('cpu'), .05, 3, 'joint', logit_temperature=value)

    def test_selection_prioritizes_complete_success_over_low_conditional_rmse(self):
        a = {'aggregate': dict(success_rate=.8, mean_all_targets_duration_s=40., position_evaluation_coverage=.8)}
        b = {'aggregate': dict(success_rate=.7, mean_all_targets_duration_s=45., position_evaluation_coverage=.9)}
        self.assertGreater(selection_key(a), selection_key(b))


if __name__ == '__main__':
    unittest.main()
