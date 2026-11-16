"""Noise reproducibility and causal, uncertainty-aware robust measurement updates."""
import sys
import unittest
import csv
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.measurement_quality import (NoiseConfig, RobustMeasurementProcessor,
                                      cartesian_measurement_covariance, cartesian_to_spherical,
                                      perturb_scenes, render_noise_diagnostics, sigmas_from_snr, spherical_to_cartesian)


def initial(r=100., sigma_r=1., h=5):
    z = np.tile([r, 0., np.pi/2], (1, 1, h, 1))
    sigma = np.tile([sigma_r, .01, .01], (1, 1, h, 1))
    return z, sigma


class MeasurementQualityTests(unittest.TestCase):
    def test_noise_is_reproducible_regrouping_invariant_and_nonmutating(self):
        position = np.tile([300., 0., 10.], (2, 2, 100, 1)).astype(np.float32)
        source = {'position': position, 'measurement': position.copy(), 'beam': np.ones(position.shape[:-1], int),
                  'trajectory_id': np.arange(4).reshape(2, 2)}
        a, meta = perturb_scenes(source, seed=13)
        b, _ = perturb_scenes(source, seed=13)
        np.testing.assert_array_equal(a['measurement'], b['measurement'])
        np.testing.assert_array_equal(a['outlier_mask'], b['outlier_mask'])
        np.testing.assert_array_equal(source['measurement'], position)
        regrouped = {k: v[::-1].copy() for k, v in source.items()}
        c, _ = perturb_scenes(regrouped, seed=13)
        np.testing.assert_array_equal(a['measurement'][::-1], c['measurement'])
        self.assertFalse(a['outlier_mask'][..., :5].any())
        self.assertEqual(meta['measurement_count'], 400)

    def test_range_and_snr_control_nominal_precision(self):
        position = np.array([[[[150., 0., 10.]]], [[[300., 0., 10.]]], [[[600., 0., 10.]]]])
        source = {'position': position, 'trajectory_id': np.arange(3)[:, None]}
        config = replace(NoiseConfig(), fading_std_db=0.)
        a, _ = perturb_scenes(source, config)
        snr = a['snr_db'].ravel()
        self.assertAlmostEqual(snr[0]-snr[1], 40*np.log10(2), places=5)
        self.assertTrue(np.all(np.diff(a['measurement_sigma'][..., 0].ravel()) > 0))
        sigma = sigmas_from_snr(np.array([0., 10., 20.]), config)
        self.assertTrue(np.all(np.diff(sigma[:, 0]) < 0))

    def test_extreme_outlier_probability_and_bounds(self):
        source = {'position': np.tile([100., 0., 10.], (1, 1, 25, 1)), 'trajectory_id': np.array([[8]])}
        a, _ = perturb_scenes(source, replace(NoiseConfig(), outlier_probability=1., burst_start_probability=0.))
        self.assertTrue(a['outlier_mask'][..., 5:].all())
        self.assertFalse(a['outlier_mask'][..., :5].any())
        self.assertTrue((a['measurement'][..., 0] > 0).all())
        self.assertTrue((np.abs(a['measurement'][..., 1]) <= np.pi).all())
        self.assertTrue(((a['measurement'][..., 2] >= 0) & (a['measurement'][..., 2] <= np.pi)).all())

    def test_coordinate_wrap_and_covariance_positive(self):
        points = np.array([[-100., 1e-6, 10.], [-100., -1e-6, 10.], [50., 20., 80.]])
        z = cartesian_to_spherical(points)
        np.testing.assert_allclose(spherical_to_cartesian(z), points, atol=1e-10)
        covariance = cartesian_measurement_covariance(z, np.tile([1., .01, .01], (3, 1)))
        self.assertTrue((np.linalg.eigvalsh(covariance) > 0).all())
        init, sigma = initial()
        init[..., 0] = 100
        init[..., 1] = np.pi-1e-5
        p = RobustMeasurementProcessor(init, sigma, .05)
        filtered, diagnostics = p.step(5, np.ones((1, 1), bool), np.array([[[100., -np.pi+1e-5, np.pi/2]]]), sigma[..., 0, :])
        self.assertLess(diagnostics['nis'][0, 0], .01)
        self.assertLess(abs(abs(filtered[0, 0, 1])-np.pi), .001)

    def test_isolated_outlier_has_bounded_smaller_influence(self):
        z, sigma = initial()
        robust = RobustMeasurementProcessor(z, sigma, .05, initial_velocity_std=1.)
        ordinary = RobustMeasurementProcessor(z, sigma, .05, huber_threshold=1e12, initial_velocity_std=1.)
        outlier = np.array([[[200., 0., np.pi/2]]])
        observed = np.ones((1, 1), bool)
        a, diag = robust.step(5, observed, outlier, sigma[..., 0, :])
        b, _ = ordinary.step(5, observed, outlier, sigma[..., 0, :])
        self.assertLess(abs(a[0, 0, 0]-100), abs(b[0, 0, 0]-100)*.3)
        self.assertLess(diag['weight'][0, 0], .1)
        self.assertTrue((np.linalg.eigvalsh(robust.covariance) > -1e-9).all())

    def test_less_precise_observation_has_less_update_influence(self):
        z, sigma = initial()
        high = RobustMeasurementProcessor(z, sigma, .05, initial_velocity_std=1.)
        low = RobustMeasurementProcessor(z, sigma, .05, initial_velocity_std=1.)
        measurement = np.array([[[105., 0., np.pi/2]]])
        a, _ = high.step(5, np.ones((1, 1), bool), measurement, np.array([[[1., .01, .01]]]))
        b, _ = low.step(5, np.ones((1, 1), bool), measurement, np.array([[[20., .1, .1]]]))
        self.assertLess(abs(b[0, 0, 0]-100), abs(a[0, 0, 0]-100))

    def test_hidden_observations_are_never_used_and_time_advances(self):
        z, sigma = initial()
        a = RobustMeasurementProcessor(z, sigma, .05)
        b = RobustMeasurementProcessor(z, sigma, .05)
        for t in range(5, 9):
            hidden = np.zeros((1, 1), bool)
            a.step(t, hidden, np.full((1, 1, 3), np.nan), np.full((1, 1, 3), np.nan))
            b.step(t, hidden, np.full((1, 1, 3), 1e12), np.full((1, 1, 3), 1e6))
        np.testing.assert_array_equal(a.state, b.state)
        np.testing.assert_array_equal(a.covariance, b.covariance)
        self.assertEqual(a.time_index, 8)
        with self.assertRaises(ValueError):
            a.step(10, hidden, z[..., 0, :], sigma[..., 0, :])

    def test_initial_history_prefix_is_causal(self):
        z, sigma = initial()
        changed = z.copy()
        changed[..., -1, 0] += 50
        a = RobustMeasurementProcessor(z, sigma, .05)
        b = RobustMeasurementProcessor(changed, sigma, .05)
        np.testing.assert_array_equal(a.filtered_initial_history[..., :-1, :], b.filtered_initial_history[..., :-1, :])

    def test_real_turn_remains_updateable_and_filter_is_stable(self):
        times = np.arange(101)*.05
        points = np.column_stack((100+20*np.minimum(times, 2.),
                                  20*np.maximum(times-2., 0), np.full(len(times), 10.)))
        z = cartesian_to_spherical(points)[None, None]
        sigma = np.tile([1., .01, .01], z.shape[:-1]+(1,))
        processor = RobustMeasurementProcessor(z[..., :5, :], sigma[..., :5, :], .05)
        weights = []
        for t in range(5, len(times)):
            filtered, diagnostics = processor.step(t, np.ones((1, 1), bool), z[..., t, :], sigma[..., t, :])
            weights.append(diagnostics['weight'][0, 0])
        # A persistent new trajectory is incorporated; updates never hard-reject it.
        self.assertTrue(np.all(np.asarray(weights) > 0))
        error = np.linalg.norm(spherical_to_cartesian(filtered)[0, 0]-points[-1])
        self.assertLess(error, 5.)
        self.assertTrue((np.linalg.eigvalsh(processor.covariance) > -1e-8).all())

    def test_covariance_grows_through_missing_gap_then_update_remains_finite(self):
        z, sigma = initial()
        processor = RobustMeasurementProcessor(z, sigma, .05)
        before = np.trace(processor.covariance[0, 0, :3, :3])
        for t in range(5, 105):
            processor.step(t, np.zeros((1, 1), bool), np.full((1, 1, 3), np.nan), np.full((1, 1, 3), np.nan))
        self.assertGreater(np.trace(processor.covariance[0, 0, :3, :3]), before)
        result, diag = processor.step(105, np.ones((1, 1), bool), z[..., 0, :], sigma[..., 0, :])
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.isfinite(diag['nis']).all())

    def test_noise_diagnostic_bins_conserve_generated_counts(self):
        source = {'position': np.tile([300., 0., 10.], (1, 1, 20, 1)), 'trajectory_id': np.array([[0]])}
        scenes, metadata = perturb_scenes(source, replace(NoiseConfig(), outlier_probability=.5))
        cache = Path(__file__).resolve().parents[2]/'.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix='measurement_quality_check_') as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(cache.resolve()))
            result = render_noise_diagnostics(scenes, directory)
            self.assertGreater(Path(result['figure']).stat().st_size, 1000)
            with Path(result['bins_csv']).open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            for kind in ('range_m', 'snr_db'):
                current = [r for r in rows if r['bin_kind'] == kind]
                self.assertEqual(sum(int(r['count']) for r in current), 20)
                self.assertEqual(sum(int(r['outlier_count']) for r in current), metadata['outlier_count'])

    def test_invalid_configuration_and_sigmas_fail(self):
        for config in (replace(NoiseConfig(), outlier_probability=1.1),
                       replace(NoiseConfig(), reference_range_m=0),
                       replace(NoiseConfig(), fading_correlation=1.),
                       replace(NoiseConfig(), burst_min_length=2.5)):
            with self.assertRaises(ValueError):
                config.validate()
        z, sigma = initial()
        sigma[..., 0] = 0
        with self.assertRaises(ValueError):
            RobustMeasurementProcessor(z, sigma, .05)
        z, sigma = initial()
        with self.assertRaises(ValueError):
            RobustMeasurementProcessor(z, sigma, .05, acceleration_std=np.inf)


if __name__ == '__main__':
    unittest.main(verbosity=2)
