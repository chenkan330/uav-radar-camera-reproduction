"""Mathematical radar association checks; no real-flight accuracy claims."""

import unittest

import numpy as np

from kalman3d import Kalman3D
from radar import H_RADAR, RadarAssociator, RadarInitializer, validate_cloud


class RadarTests(unittest.TestCase):
    def make_filter(self):
        return Kalman3D(np.zeros(6), np.eye(6), acceleration_std=0)

    def test_cloud_shape_and_finite(self):
        self.assertEqual(validate_cloud(np.empty((0, 5))).shape, (0, 5))
        for value in ([], np.zeros((2, 4)), np.zeros((2, 6)),
                      [[0, 0, 0, 0, np.nan]]):
            with self.assertRaises(ValueError):
                validate_cloud(value)

    def test_initialization_two_consecutive_strongest_and_velocity(self):
        init = RadarInitializer(R=np.eye(4) * 0.25)
        cloud1 = np.array([[30, 0, 0, 1, 1], [2, 1, 3, .04, 10]])
        cloud2 = np.array([[30, 0, 0, 1, 1], [2.02, 1, 3, .05, 10]])
        self.assertIsNone(init.observe(0, cloud1))
        kf = init.observe(.1, cloud2)
        self.assertIsNotNone(kf)
        np.testing.assert_allclose(kf.x, [2.02, .05, 1, 0, 3, 0])
        np.testing.assert_allclose(H_RADAR @ kf.P @ H_RADAR.T, np.eye(4) * .25)
        self.assertAlmostEqual(init.last_speed, .2)
        self.assertIsNone(init.observe(.2, cloud2))

    def test_initialization_empty_breaks_pair_and_rejects_fast(self):
        init = RadarInitializer()
        first = [[1, 0, 0, 0, 10]]
        moved = [[1.05, 0, 0, 0, 10]]
        self.assertIsNone(init.observe(0, first))
        self.assertIsNone(init.observe(.1, moved))  # 0.5 m/s
        self.assertIsNone(init.observe(.2, np.empty((0, 5))))
        self.assertIsNone(init.observe(.3, moved))
        self.assertIsNotNone(init.observe(.4, moved))
        with self.assertRaises(ValueError):
            init.observe(.4, moved)

    def test_gate_precedes_strongest_and_uses_velocity(self):
        kf = self.make_filter()
        cloud = [[.1, 0, 0, 0, 1], [.1, 0, 0, 9, 1000], [9, 0, 0, 0, 2000]]
        result = RadarAssociator("strongest", R=np.eye(4), gate=2).update(kf, cloud)
        np.testing.assert_array_equal(result.gated_indices, [0])
        np.testing.assert_allclose(result.measurement, cloud[0])
        np.testing.assert_allclose(kf.x, [.05, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(result.distances[0], .1 / np.sqrt(2))

    def test_closest_means_xyz_distance_not_intensity_or_four_d(self):
        kf = self.make_filter()
        cloud = [[.1, 0, 0, 1, 1], [.2, 0, 0, 0, 50]]
        result = RadarAssociator("closest", R=np.eye(4), gate=3).update(kf, cloud)
        np.testing.assert_allclose(result.measurement, cloud[0])
        self.assertAlmostEqual(kf.x[1], .5)

    def test_single_kmeans_and_average_exact_centroid_all_five_fields(self):
        cloud = [[1, 0, 0, .2, 4], [0, 1, 0, .4, 10], [0, 0, 1, .6, 16]]
        filters = [self.make_filter(), self.make_filter()]
        results = [RadarAssociator(method, R=np.eye(4), gate=4).update(kf, cloud)
                   for method, kf in zip(("average", "kmeans"), filters)]
        expected = np.mean(cloud, axis=0)
        for result in results:
            np.testing.assert_allclose(result.measurement, expected)
        np.testing.assert_allclose(filters[0].x, filters[1].x)
        np.testing.assert_allclose(filters[0].P, filters[1].P)

    def test_pdaf_hand_calculated_mixture_covariance(self):
        # S=2I, K=0.5 on observed axes. beta=[0.2,0.6], beta0=.2.
        # x innovations +1 and -1 => mean -.4. Innovation spread .8-.16=.64.
        # Observed posterior variance .2*1+.8*.5+.25*.64=.76.
        kf = self.make_filter()
        result = RadarAssociator("weighted", R=np.eye(4), beta0=.2, gate=3).update(
            kf, [[1, 0, 0, 0, 1], [-1, 0, 0, 0, 3]])
        np.testing.assert_allclose(result.beta, [.2, .6])
        np.testing.assert_allclose(kf.x, [-.2, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(np.diag(kf.P), [.76, .6, .6, 1, .6, 1])
        # A centroid Kalman would produce .5 instead of .76, losing ambiguity.
        self.assertGreater(kf.P[0, 0], .5)

    def test_pdaf_matches_explicit_gaussian_mixture_for_correlated_covariance(self):
        rng = np.random.default_rng(14)
        A = rng.normal(size=(6, 6))
        P = A @ A.T + np.eye(6)
        x = rng.normal(size=6)
        kf = Kalman3D(x, P)
        R = np.diag([.5, .6, .7, .8])
        z = H_RADAR @ x + rng.normal(size=(3, 4)) * .3
        cloud = np.column_stack((z, [1., 2., 3.]))
        beta0 = .15
        beta = (1 - beta0) * np.array([1., 2., 3.]) / 6
        S = H_RADAR @ P @ H_RADAR.T + R
        K = np.linalg.solve(S, H_RADAR @ P).T
        Pc = P - K @ S @ K.T
        components = [x] + [x + K @ (zi - H_RADAR @ x) for zi in z]
        probabilities = np.r_[beta0, beta]
        mean = sum(w * component for w, component in zip(probabilities, components))
        covariances = [P, Pc, Pc, Pc]
        covariance = sum(w * (C + np.outer(component - mean, component - mean))
                         for w, C, component in zip(probabilities, covariances, components))
        RadarAssociator("weighted", R=R, gate=4, beta0=beta0).update(kf, cloud)
        np.testing.assert_allclose(kf.x, mean, atol=1e-12)
        np.testing.assert_allclose(kf.P, covariance, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(kf.P)[0], 0)

    def test_empty_and_empty_gate_preserve_state(self):
        for cloud in (np.empty((0, 5)), [[100, 0, 0, 0, 5]]):
            kf = self.make_filter()
            before = kf.x.copy(), kf.P.copy()
            result = RadarAssociator().update(kf, cloud)
            self.assertFalse(result.accepted)
            np.testing.assert_array_equal(kf.x, before[0])
            np.testing.assert_array_equal(kf.P, before[1])

    def test_bad_weighted_weights_leave_state_unchanged(self):
        for intensity in (-1, 0):
            kf = self.make_filter()
            with self.assertRaises(ValueError):
                RadarAssociator("weighted").update(kf, [[0, 0, 0, 0, intensity]])
            np.testing.assert_array_equal(kf.x, np.zeros(6))
            np.testing.assert_array_equal(kf.P, np.eye(6))

    def test_probability_one_missed_detection_preserves_prior(self):
        kf = self.make_filter()
        RadarAssociator("weighted", beta0=1).update(kf, [[.2, 0, 0, .1, 1]])
        np.testing.assert_allclose(kf.x, np.zeros(6))
        np.testing.assert_allclose(kf.P, np.eye(6))

    def test_invalid_configuration(self):
        for kwargs in ({"gate": 0}, {"gate": np.nan}, {"beta0": -1},
                       {"beta0": 1.01}, {"R": np.zeros((4, 4))}, {"method": "unknown"}):
            with self.assertRaises(ValueError):
                RadarAssociator(**kwargs)


if __name__ == "__main__":
    unittest.main()
