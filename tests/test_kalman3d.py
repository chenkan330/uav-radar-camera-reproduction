"""Mathematical checks for the handwritten, synchronous 3D Kalman filter.

Run from the project root with: python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from kalman3d import H_POSITION, POSITION, VELOCITY, Kalman3D


class Kalman3DTests(unittest.TestCase):
    def test_hand_calculated_prediction_and_update(self):
        # One-axis prior [[2, 1], [1, 3]], dt=1, sigma_a=2:
        # predicted covariance = [[8, 6], [6, 7]]. With R=4,
        # S=12, K=[2/3, 1/2], posterior=[[8/3, 2], [2, 4]].
        prior = np.kron(np.eye(3), [[2.0, 1.0], [1.0, 3.0]])
        tracker = Kalman3D([0, 1, 10, -2, -5, 0.5], prior, 2.0)
        state, covariance = tracker.predict(1.0)
        assert_allclose(state, [1, 1, 8, -2, -4.5, 0.5])
        assert_allclose(covariance, np.kron(np.eye(3), [[8, 6], [6, 7]]))

        result = tracker.update([4, 2, -3], 4.0 * np.eye(3))
        assert_allclose(result.innovation, [3, -6, 1.5])
        assert_allclose(result.innovation_covariance, 12.0 * np.eye(3))
        expected_gain = np.zeros((6, 3))
        expected_gain[POSITION, np.arange(3)] = 2.0 / 3.0
        expected_gain[VELOCITY, np.arange(3)] = 0.5
        assert_allclose(result.gain, expected_gain)
        assert_allclose(tracker.x, [3, 2.5, 4, -5, -3.5, 1.25])
        assert_allclose(tracker.P, np.kron(np.eye(3), [[8 / 3, 2], [2, 4]]))

    def test_two_independent_updates_equal_joint_observation_in_either_order(self):
        rng = np.random.default_rng(13)
        mixing = rng.normal(size=(6, 6))
        prior_covariance = mixing @ mixing.T + np.eye(6)
        prior_state = np.array([10, 0.4, -3, -0.2, 4, 0.5])
        radar = np.array([9.8, -3.3, 3.9])
        camera = np.array([10.2, -2.8, 4.4])
        radar_R = np.array([[0.5, 0.1, 0.05], [0.1, 0.3, 0], [0.05, 0, 0.8]])
        camera_R = np.array([[1.2, -0.1, 0.2], [-0.1, 0.6, 0.1], [0.2, 0.1, 2]])

        # Independent joint observation, evaluated in information form rather
        # than by repeating the implementation's gain/Joseph-form equations.
        joint_H = np.vstack([H_POSITION, H_POSITION])
        joint_z = np.concatenate([radar, camera])
        joint_R = np.zeros((6, 6))
        joint_R[:3, :3] = radar_R
        joint_R[3:, 3:] = camera_R
        prior_information = np.linalg.solve(prior_covariance, np.eye(6))
        posterior_information = prior_information + joint_H.T @ np.linalg.solve(joint_R, joint_H)
        information_mean = prior_information @ prior_state + joint_H.T @ np.linalg.solve(joint_R, joint_z)
        expected_state = np.linalg.solve(posterior_information, information_mean)
        expected_covariance = np.linalg.solve(posterior_information, np.eye(6))

        for observations in [((radar, radar_R), (camera, camera_R)),
                             ((camera, camera_R), (radar, radar_R))]:
            with self.subTest(first_observation=observations[0][0].tolist()):
                tracker = Kalman3D(prior_state, prior_covariance)
                # Both measurements refer to the same instant; no prediction
                # belongs between these two corrections.
                for measurement, measurement_covariance in observations:
                    tracker.update(measurement, measurement_covariance)
                assert_allclose(tracker.x, expected_state, rtol=1e-12, atol=1e-12)
                assert_allclose(tracker.P, expected_covariance, rtol=1e-12, atol=1e-12)

    def test_prediction_only_advances_motion_and_grows_uncertainty(self):
        initial = np.array([1, 2, -4, 0.5, 3, -1], dtype=float)
        tracker = Kalman3D(initial, np.eye(6), acceleration_std=0.4)
        for step in range(1, 11):
            previous_position_variance = np.diag(tracker.P)[POSITION].copy()
            previous_velocity_variance = np.diag(tracker.P)[VELOCITY].copy()
            tracker.predict(0.2)
            assert_allclose(tracker.x[POSITION], initial[POSITION] + step * 0.2 * initial[VELOCITY])
            assert_allclose(tracker.x[VELOCITY], initial[VELOCITY])
            self.assertTrue(np.all(np.diag(tracker.P)[POSITION] > previous_position_variance))
            self.assertTrue(np.all(np.diag(tracker.P)[VELOCITY] > previous_velocity_variance))

    def test_repeated_updates_preserve_symmetric_psd_covariance(self):
        rng = np.random.default_rng(24)
        tracker = Kalman3D(np.zeros(6), 100.0 * np.eye(6), acceleration_std=0.8)
        radar_R = np.diag([0.04, 0.09, 0.16])
        camera_R = np.diag([0.16, 0.25, 1.0])
        for _ in range(300):
            tracker.predict(rng.uniform(0.02, 0.2))
            for measurement_covariance in (radar_R, camera_R):
                before_update = tracker.P.copy()
                tracker.update(rng.normal(size=3), measurement_covariance)
                assert_allclose(tracker.P, tracker.P.T, rtol=0, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(tracker.P)[0], -1e-12)
                # Conditioning on a measurement cannot add uncertainty.
                self.assertGreaterEqual(np.linalg.eigvalsh(before_update - tracker.P)[0], -1e-12)
                self.assertTrue(np.all(np.isfinite(tracker.x)))

    def test_first_observation_initializes_once_with_its_full_covariance(self):
        position = np.array([1, -2, 3], dtype=float)
        R = np.array([[0.4, 0.1, 0], [0.1, 0.5, -0.1], [0, -0.1, 0.8]])
        tracker = Kalman3D.from_position(position, R, velocity_std=3.0)
        assert_allclose(tracker.x[POSITION], position)
        assert_allclose(tracker.x[VELOCITY], np.zeros(3))
        # A duplicate correction using the same sample would halve this R.
        assert_allclose(tracker.P[np.ix_(POSITION, POSITION)], R)
        assert_allclose(tracker.P[np.ix_(VELOCITY, VELOCITY)], 9.0 * np.eye(3))
        assert_allclose(tracker.P[np.ix_(POSITION, VELOCITY)], np.zeros((3, 3)))
        self.assertFalse(np.shares_memory(tracker.x, position))
        self.assertFalse(np.shares_memory(tracker.P, R))

    def test_invalid_state_and_covariance_are_rejected(self):
        for state in (np.zeros(5), np.zeros((6, 1)), [0, 0, 0, 0, 0, np.nan]):
            with self.subTest(state=state), self.assertRaises(ValueError):
                Kalman3D(state, np.eye(6))
        asymmetric = np.eye(6)
        asymmetric[0, 1] = 0.1
        nonfinite = np.eye(6)
        nonfinite[0, 0] = np.inf
        for covariance in (np.eye(3), -np.eye(6), asymmetric, nonfinite):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                Kalman3D(np.zeros(6), covariance)
        # A zero prior is mathematically valid positive semidefinite input.
        tracker = Kalman3D(np.zeros(6), np.zeros((6, 6)))
        tracker.update([1, 2, 3], np.eye(3))
        assert_allclose(tracker.x, np.zeros(6))

    def test_invalid_measurements_are_rejected_without_changing_filter(self):
        tracker = Kalman3D(np.arange(6), np.eye(6))
        for position in ([1, 2], [[1], [2], [3]], [1, np.nan, 3]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                tracker.update(position, np.eye(3))
        asymmetric = np.eye(3)
        asymmetric[0, 1] = 0.1
        for covariance in (np.eye(2), np.zeros((3, 3)), -np.eye(3), asymmetric,
                           np.diag([1.0, 1.0, np.inf])):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                tracker.update([1, 2, 3], covariance)
        assert_allclose(tracker.x, np.arange(6))
        assert_allclose(tracker.P, np.eye(6))

    def test_invalid_time_noise_and_initialization_parameters_are_rejected(self):
        tracker = Kalman3D(np.zeros(6), np.eye(6))
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(dt=value), self.assertRaises(ValueError):
                tracker.predict(value)
            with self.subTest(velocity_std=value), self.assertRaises(ValueError):
                Kalman3D.from_position([0, 0, 0], np.eye(3), velocity_std=value)
        for value in (-1, np.nan, np.inf):
            with self.subTest(acceleration_std=value), self.assertRaises(ValueError):
                Kalman3D(np.zeros(6), np.eye(6), acceleration_std=value)
        with self.assertRaises(ValueError):
            Kalman3D.from_position([0, 0], np.eye(3))
        with self.assertRaises(ValueError):
            Kalman3D.from_position([0, 0, 0], np.zeros((3, 3)))
        # Deterministic constant velocity is an allowed zero-noise model.
        deterministic = Kalman3D(np.zeros(6), np.eye(6), acceleration_std=0)
        _, Q = deterministic.motion_matrices(1.0)
        assert_allclose(Q, np.zeros((6, 6)))


if __name__ == "__main__":
    unittest.main()
