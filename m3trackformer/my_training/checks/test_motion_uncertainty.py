"""V4 forecast distribution, masking, calibration and proper-score checks."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from core.motion_uncertainty import (
    MotionConfig, MotionUncertaintyNet, apply_scale_temperature,
    evaluate_distribution, fit_scale_temperature, gaussian_mixture_nll,
    mixture_marginal_intervals, predictive_mean, prepare_history, sample_mixture,
)


def spherical(xyz):
    rel = xyz-torch.tensor([0., 0., 10.], dtype=xyz.dtype)
    r = rel.norm(dim=-1)
    return torch.stack((r, torch.atan2(rel[..., 1], rel[..., 0]),
                        torch.acos(rel[..., 2]/r)), -1)


def normal_prediction(batch=1, horizons=1, components=1, scale=1.):
    return {'weights': np.full((batch, horizons, components), 1/components),
            'mean': np.zeros((batch, horizons, components, 3)),
            'scale': np.full((batch, horizons, components, 3), float(scale))}


class MotionUncertaintyTests(unittest.TestCase):
    def test_nll_matches_normal_density_and_mixture_permutation(self):
        p = {k: torch.from_numpy(v) for k, v in normal_prediction(2, 2, 3, 2.).items()}
        p['mean'][:, :, 0, 0] = -5
        p['mean'][:, :, 2, 0] = 5
        target = torch.randn(2, 2, 3, dtype=torch.float64)
        original = gaussian_mixture_nll(p, target, reduction='none')
        permuted = {k: v[:, :, [2, 0, 1]] for k, v in p.items()}
        torch.testing.assert_close(original, gaussian_mixture_nll(permuted, target, reduction='none'))
        single = {k: torch.from_numpy(v) for k, v in normal_prediction(1, 1, 1, 2.).items()}
        at_zero = gaussian_mixture_nll(single, torch.zeros(1, 1, 3, dtype=torch.float64))
        self.assertAlmostEqual(float(at_zero), 3*math.log(2.)+1.5*math.log(2*math.pi), places=10)

    def test_two_future_branches_remain_a_distribution_not_their_mean(self):
        p = normal_prediction(1, 1, 2, .25)
        p['mean'][0, 0, :, 0] = [-8., 8.]
        self.assertEqual(float(predictive_mean(p)[0, 0, 0]), 0.)
        target = torch.tensor([[[-8., 0., 0.]]], dtype=torch.float64)
        tp = {k: torch.from_numpy(v) for k, v in p.items()}
        mixture_nll = float(gaussian_mixture_nll(tp, target))
        collapsed = {k: torch.from_numpy(v) for k, v in normal_prediction(1, 1, 1, .25).items()}
        self.assertLess(mixture_nll, float(gaussian_mixture_nll(collapsed, target))-100)
        samples = sample_mixture(p, samples=4000, seed=10)
        self.assertAlmostEqual(float((samples[..., 0] < 0).mean()), .5, delta=.03)
        self.assertGreater(float(np.abs(samples[..., 0]).mean()), 7.)

    def test_model_outputs_finite_positive_scales_and_trainable_gradients(self):
        torch.manual_seed(12)
        c = MotionConfig(history=6, horizons=(1, 3), d_model=16, heads=4, layers=1, dropout=0.)
        net = MotionUncertaintyNet(c)
        xyz = torch.zeros(4, 6, 3)+torch.tensor([100., 80., 60.])
        xyz[:, :, 0] += torch.arange(6)[None]*1.2
        history = spherical(xyz)
        valid = torch.ones(4, 6, dtype=torch.bool)
        valid[1, 2] = False
        history[1, 2] = float('nan')
        result = net(history, valid)
        self.assertEqual(result['mean'].shape, (4, 2, 3, 3))
        self.assertTrue(bool((result['scale'] >= c.minimum_std_m).all()))
        torch.testing.assert_close(result['weights'].sum(-1), torch.ones(4, 2))
        target = xyz[:, -1, None]+torch.tensor([[[1.2, 0., 0.], [3.6, 0., 0.]]])
        loss = gaussian_mixture_nll(result, target)
        loss.backward()
        gradients = [p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_invalid_history_and_quality_are_invisible_even_as_nans(self):
        torch.manual_seed(4)
        c = MotionConfig(history=6, horizons=(1, 3), d_model=16, heads=4, layers=1, dropout=0.)
        net = MotionUncertaintyNet(c).eval()
        xyz = torch.tensor([100., 80., 60.])[None, None].expand(2, 6, 3).clone()
        xyz[:, :, 0] += torch.arange(6)[None]
        z = spherical(xyz)
        valid = torch.tensor([[1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        sigma = torch.full_like(z, .01)
        a = net(z, valid, sigma=sigma)
        changed, changed_sigma = z.clone(), sigma.clone()
        changed[~valid] = float('nan')
        changed_sigma[~valid] = float('nan')
        b = net(changed, valid, sigma=changed_sigma)
        for key in ('mean', 'scale', 'weights', 'base', 'observation_age_s'):
            torch.testing.assert_close(a[key], b[key])

    def test_causal_cv_extrapolation_includes_missing_sample_age(self):
        dt = .05
        xyz = torch.tensor([100., 80., 60.], dtype=torch.float64)[None, None].expand(1, 20, 3).clone()
        velocity = torch.tensor([12., -4., 2.], dtype=torch.float64)
        xyz += torch.arange(20, dtype=torch.float64)[None, :, None]*dt*velocity
        valid = torch.ones(1, 20, dtype=torch.bool)
        valid[:, -3:] = False
        _, origin, estimated, age, last = prepare_history(spherical(xyz), valid, dt=dt)
        torch.testing.assert_close(estimated[0], velocity, atol=1.e-10, rtol=1.e-10)
        self.assertAlmostEqual(float(age[0]), .15)
        predicted = origin+estimated*(age[:, None]+dt)
        torch.testing.assert_close(predicted[0], xyz[0, -1]+velocity*dt, atol=1.e-10, rtol=1.e-10)

    def test_independent_calibration_recovers_known_underdispersion(self):
        rng = np.random.default_rng(12)
        target = rng.normal(0, 4., (4000, 1, 3))
        p = normal_prediction(4000)
        temperature = fit_scale_temperature(p, target, split_name='calibration')
        self.assertAlmostEqual(temperature, 4., delta=.08)
        scaled = apply_scale_temperature(p, temperature)
        np.testing.assert_allclose(scaled['scale'], temperature)
        with self.assertRaises(ValueError):
            fit_scale_temperature(p, target, split_name='test')

    def test_marginal_intervals_have_nominal_coverage_for_known_distribution(self):
        n = 2500
        p = normal_prediction(n, components=2, scale=1.)
        p['mean'][:, :, :, 0] = [-6., 6.]
        target = sample_mixture(p, samples=1, seed=20)[0]
        low, high = mixture_marginal_intervals(p, levels=(.9,))[.9]
        coverage = ((target >= low) & (target <= high)).mean()
        self.assertAlmostEqual(float(coverage), .9, delta=.015)
        self.assertGreater(float((high-low)[..., 0].mean()), 12.)
        self.assertLess(float((high-low)[..., 1].mean()), 4.)

    def test_proper_scores_penalize_unnecessarily_broad_predictions(self):
        n = 700
        target = np.random.default_rng(10).normal(size=(n, 1, 3))
        calibrated = evaluate_distribution(normal_prediction(n), target, levels=(.9,), energy_samples=64, seed=11)
        broad = evaluate_distribution(normal_prediction(n, scale=8.), target, levels=(.9,), energy_samples=64, seed=11)
        self.assertLess(calibrated['nll_nats'], broad['nll_nats'])
        self.assertLess(calibrated['energy_score_m'], broad['energy_score_m'])
        self.assertAlmostEqual(calibrated['marginal_coverage_90'], .9, delta=.025)
        self.assertEqual(calibrated['per_horizon'][0]['sample_count'], n)

    def test_sampling_is_reproducible_without_modifying_global_numpy_rng(self):
        p = normal_prediction(2, 2, 2)
        np.random.seed(30)
        before = np.random.get_state()
        a = sample_mixture(p, samples=8, seed=8)
        b = sample_mixture(p, samples=8, seed=8)
        after = np.random.get_state()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_prediction_and_all_missing_history_fail(self):
        with self.assertRaises(ValueError):
            MotionConfig(horizons=(5, 1)).validate()
        net = MotionUncertaintyNet(MotionConfig(history=2, d_model=16, heads=4, layers=1))
        with self.assertRaises(ValueError):
            net(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))
        p = normal_prediction()
        p['scale'][0, 0, 0, 0] = 0
        with self.assertRaises(ValueError):
            sample_mixture(p)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
