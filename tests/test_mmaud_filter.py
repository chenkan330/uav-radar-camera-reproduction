"""Independent checks for the explicitly adapted external-data filter."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kalman3d import POSITION, VELOCITY, Kalman3D
from mmaud_filter import Calib, CameraFrame, RadarFrame, normalized_bearing_model, run_comparison


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class MMAUDAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.R = np.diag([.04, .09, .16])
        self.calib = Calib(np.eye(3), [0, 0, 0], [.01, .02])

    def test_normalized_bearing_jacobian_matches_numerical_derivative(self):
        calib = Calib(rotation_y(.2), [.2, -.1, .3], .01)
        state = np.array([1.2, .1, -.5, .2, 7, -.3])
        prediction = normalized_bearing_model(state, calib)
        numerical = np.zeros((2, 6))
        epsilon = 1e-5
        for index in range(6):
            step = np.eye(6)[index] * epsilon
            numerical[:, index] = (
                normalized_bearing_model(state + step, calib).normalized_xy
                - normalized_bearing_model(state - step, calib).normalized_xy
            ) / (2 * epsilon)
        assert_allclose(prediction.H, numerical, atol=2e-11, rtol=2e-8)

    def test_camera_optical_center_translation_and_depth_null_direction(self):
        calib = Calib(np.eye(3), [1, 2, 3], .01)
        state = np.array([2, 0, 4, 0, 13, 0])
        prediction = normalized_bearing_model(state, calib)
        assert_allclose(prediction.normalized_xy, [.1, .2])
        radial = np.zeros(6)
        radial[POSITION] = state[POSITION] - calib.translation
        assert_allclose(prediction.H @ radial, [0, 0], atol=1e-14)

    def test_no_camera_both_groups_identical(self):
        radar = [RadarFrame(0, [[1, 2, 10]]), RadarFrame(.031, [[1.1, 2, 10]]),
                 RadarFrame(.06, np.empty((0, 3)))]
        comparison = run_comparison(radar, [], radar_R=self.R, end_time=.08)
        a, b = comparison["radar_only"], comparison["radar_camera"]
        assert_array_equal(a.times, b.times)
        assert_array_equal(a.state, b.state)
        assert_array_equal(a.covariance, b.covariance)
        assert_array_equal(a.prediction_grid, b.prediction_grid)

    def test_initial_radar_point_is_not_updated_twice(self):
        comparison = run_comparison([RadarFrame(0, [[1, 2, 10], [5, 6, 10]])], [],
                                    radar_R=self.R, initial_point_index=1, end_time=.03)
        result = comparison["radar_only"]
        assert_allclose(result.state[0], [5, 0, 6, 0, 10, 0])
        assert_allclose(result.covariance[0][np.ix_(POSITION, POSITION)], self.R)
        self.assertEqual(len(result.updates), 1)
        self.assertEqual(result.updates[0]["reason"], "initialization once")
        self.assertEqual(result.initial_source["point_index"], 1)
        self.assertFalse(result.initial_source["ground_truth_used"])
        self.assertFalse(result.initial_source["initialized_observation_reused"])

    def test_common_camera_event_partitions_are_applied_to_radar_baseline(self):
        initial = [1, 2, 10]
        radar = [RadarFrame(0, [initial])]
        camera = [CameraFrame(.003, [.1, .2]), CameraFrame(.017, [.1, .2])]
        comparison = run_comparison(radar, camera, self.calib, radar_R=self.R,
                                    acceleration_std=.8, end_time=.03)
        baseline, fused = comparison["radar_only"], comparison["radar_camera"]
        assert_array_equal(baseline.times, fused.times)
        assert_array_equal(baseline.prediction_grid, fused.prediction_grid)
        assert_allclose(baseline.prediction_grid, [0, .003, .01, .017, .02, .03])
        # Independent replay of the baseline's common intervals; no camera
        # information is applied to this reference but every split is retained.
        reference = Kalman3D.from_position(initial, self.R, 2, .8)
        previous = 0
        for timestamp in baseline.prediction_grid[1:]:
            reference.predict(timestamp - previous)
            previous = timestamp
        assert_allclose(baseline.state[-1], reference.x, atol=1e-14)
        assert_allclose(baseline.covariance[-1], reference.P, atol=1e-14)
        self.assertEqual(sum(u["sensor"] == "camera" for u in baseline.updates), 2)
        self.assertEqual(sum(u["sensor"] == "camera" and u["accepted"] for u in fused.updates), 2)

    def test_gate_then_nearest_does_not_use_truth_or_intensity(self):
        radar = [RadarFrame(0, [[0, 0, 10]]),
                 RadarFrame(.01, [[100, 100, 100], [.2, 0, 10], [.1, 0, 10]])]
        comparison = run_comparison(radar, [], radar_R=np.eye(3), acceleration_std=0,
                                    radar_gate_nis=5, end_time=.01)
        update = comparison["radar_only"].updates[-1]
        self.assertTrue(update["accepted"])
        self.assertEqual(update["gated_count"], 2)
        self.assertEqual(update["selected_point_index"], 2)
        assert_allclose(update["selected_xyz_m"], [.1, 0, 10])
        self.assertGreater(update["nis"], 0)

    def test_radar_and_camera_outliers_are_logged_without_correction(self):
        radar = [RadarFrame(0, [[0, 0, 10]]), RadarFrame(.01, [[100, 100, 100]])]
        camera = [CameraFrame(.01, [100, 100])]
        comparison = run_comparison(radar, camera, self.calib, radar_R=self.R, end_time=.02)
        for result in comparison.values():
            radar_update = next(u for u in result.updates if u["sensor"] == "radar" and u["timestamp"] == .01)
            self.assertFalse(radar_update["accepted"])
            self.assertEqual(radar_update["reason"], "NIS gate")
        fused_camera = next(u for u in comparison["radar_camera"].updates if u["sensor"] == "camera")
        self.assertFalse(fused_camera["accepted"])
        self.assertEqual(fused_camera["reason"], "NIS gate")
        assert_array_equal(comparison["radar_only"].state, comparison["radar_camera"].state)
        assert_array_equal(comparison["radar_only"].covariance, comparison["radar_camera"].covariance)

    def test_two_observation_velocity_initialization_has_propagated_cross_covariance(self):
        radar = [RadarFrame(0, [[1, 2, 10]]), RadarFrame(.1, [[1.2, 1.9, 10.3]])]
        result = run_comparison(radar, [], radar_R=self.R, initial_velocity_from_two=True)["radar_only"]
        assert_allclose(result.state[0][POSITION], [1.2, 1.9, 10.3])
        assert_allclose(result.state[0][VELOCITY], [2, -1, 3])
        assert_allclose(result.covariance[0][np.ix_(VELOCITY, VELOCITY)], 2 * self.R / .1**2)
        assert_allclose(result.covariance[0][np.ix_(POSITION, VELOCITY)], self.R / .1)
        self.assertGreater(np.linalg.eigvalsh(result.covariance[0]).min(), 0)
        self.assertEqual(len(result.updates), 1)
        self.assertEqual(result.initial_source["second_timestamp"], .1)

    def test_invalid_inputs_and_truth_argument_are_rejected(self):
        invalid = [
            lambda: RadarFrame(0, [[1, 2, np.nan]]),
            lambda: RadarFrame(-1, [[1, 2, 3]]),
            lambda: CameraFrame(0, [0, np.inf]),
            lambda: Calib(np.diag([1, 1, -1]), [0, 0, 0]),
            lambda: Calib(np.eye(3), [0, 0, 0], noise_norm=0),
            lambda: run_comparison([RadarFrame(0, [[0, 0, 10]])], [CameraFrame(0, [0, 0])]),
            lambda: run_comparison([RadarFrame(0, [[0, 0, 10]])], [], initial_point_index=3),
            lambda: run_comparison([RadarFrame(0, [[0, 0, 10]])], [], initial_velocity_from_two=True),
            lambda: run_comparison([RadarFrame(0, [[0, 0, 10]])], [], radar_R=-np.eye(3)),
            lambda: run_comparison([RadarFrame(0, [[0, 0, 10]]), RadarFrame(0, [[0, 0, 10]])], []),
        ]
        for call in invalid:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()
        with self.assertRaises(TypeError):
            run_comparison([RadarFrame(0, [[0, 0, 10]])], [], ground_truth=[[0, 0, 10]])


if __name__ == "__main__":
    unittest.main()
