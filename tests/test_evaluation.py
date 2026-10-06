"""Hand-calculated metric and temporal-coverage checks."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose

from evaluation import evaluate_positions


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_signed_errors_and_euclidean_metrics(self):
        result = evaluate_positions([0, 1], [[3, 4, 0], [-3, 0, 4]], [[0, 0, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["coverage_fraction"], 1)
        assert_allclose(result["signed_mean_xyz_m"], [0, 2, 2])
        assert_allclose(result["axis_rmse_m"], [3, np.sqrt(8), np.sqrt(8)])
        for key in ["mean_euclidean_m", "median_euclidean_m", "p95_euclidean_m", "rmse3d_m"]:
            self.assertEqual(result[key], 5)
        self.assertEqual(result["std_euclidean_m"], 0)
        self.assertEqual(result["matched_time_span_s"], [0, 1])
        json.dumps(result, allow_nan=False)

    def test_mean_distance_is_not_rmse(self):
        result = evaluate_positions([0, 1], [[0, 0, 0], [3, 4, 0]], [[0, 0, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(result["mean_euclidean_m"], 2.5)
        self.assertEqual(result["std_euclidean_m"], 2.5)
        self.assertAlmostEqual(result["rmse3d_m"], np.sqrt(12.5))
        self.assertAlmostEqual(result["p95_euclidean_m"], 4.75)

    def test_nonuniform_timestamp_interpolation_uses_actual_elapsed_time(self):
        # Ground truth is x=2t, y=-t, z=3t, sampled at unequal intervals.
        truth = [[0, 0, 0, 0], [0.2, 0.4, -0.2, 0.6], [0.8, 1.6, -0.8, 2.4]]
        result = evaluate_positions([0.1, 0.5], [[0.2, -0.1, 0.3], [1, -0.5, 1.5]], truth, 0.7)
        self.assertEqual(result["matched_count"], 2)
        self.assertAlmostEqual(result["rmse3d_m"], 0, places=14)

    def test_no_extrapolation_or_interpolation_across_large_truth_gap(self):
        times = [-0.1, 0, 0.1, 0.2, 1, 2, 2.1, 3]
        positions = np.zeros((len(times), 3))
        truth = [[0, 0, 0, 0], [0.2, 0, 0, 0], [2, 0, 0, 0], [2.2, 0, 0, 0]]
        result = evaluate_positions(times, positions, truth, 0.21)
        # Exact 0,0.2,2 and interpolated 0.1,2.1; no match for -0.1,1,3.
        self.assertEqual(result["matched_count"], 5)
        self.assertEqual(result["unmatched_count"], 3)
        self.assertEqual(result["coverage_fraction"], 5 / 8)
        self.assertEqual(result["matched_time_span_s"], [0, 2.1])

    def test_single_truth_sample_matches_only_that_timestamp(self):
        result = evaluate_positions([0, 1, 2], np.zeros((3, 3)), [[1, 0, 0, 0]])
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["matched_time_span_s"], [1, 1])

    def test_nominal_gap_boundary_allows_timestamp_rounding_only(self):
        for origin in [2.0, 1_000_000_000.0]:
            with self.subTest(origin=origin):
                result = evaluate_positions(
                    [origin + 0.1], [[0, 0, 0]],
                    [[origin, 0, 0, 0], [origin + 0.2, 0, 0, 0]], 0.2,
                )
                self.assertEqual(result["matched_count"], 1)
        result = evaluate_positions(
            [2.1], [[0, 0, 0]], [[2, 0, 0, 0], [2.20001, 0, 0, 0]], 0.2,
        )
        self.assertEqual(result["matched_count"], 0)

    def test_absent_truth_and_no_overlap_report_missing_metrics(self):
        for truth, status in [(None, "no_ground_truth"), ([[2, 0, 0, 0]], "no_overlap")]:
            with self.subTest(truth=truth):
                result = evaluate_positions([0], [[0, 0, 0]], truth)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["matched_count"], 0)
                self.assertIsNone(result["mean_euclidean_m"])
                self.assertIsNone(result["signed_mean_xyz_m"])
                json.dumps(result, allow_nan=False)

    def test_empty_estimates_and_empty_truth_are_valid(self):
        result = evaluate_positions([], np.empty((0, 3)), np.empty((0, 4)))
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["coverage_fraction"], 0)
        self.assertIsNone(result["estimate_time_span_s"])

    def test_bad_shapes_values_timestamps_and_gap_rejected(self):
        cases = [
            ([[0]], [[0, 0, 0]], None, 0.2),
            ([0], [[0, 0]], None, 0.2),
            ([0], [[np.nan, 0, 0]], None, 0.2),
            ([0, 0], np.zeros((2, 3)), None, 0.2),
            ([1, 0], np.zeros((2, 3)), None, 0.2),
            ([0], [[0, 0, 0]], [[0, 0, 0]], 0.2),
            ([0], [[0, 0, 0]], [[0, 0, 0, np.inf]], 0.2),
            ([0], [[0, 0, 0]], [[0, 0, 0, 0], [0, 1, 2, 3]], 0.2),
            ([0], [[0, 0, 0]], None, 0),
            ([0], [[0, 0, 0]], None, np.nan),
            ([0], [[0, 0, 0]], None, None),
        ]
        for times, positions, truth, gap in cases:
            with self.subTest(times=times, truth=truth, gap=gap):
                with self.assertRaises(ValueError):
                    evaluate_positions(times, positions, truth, gap)


if __name__ == "__main__":
    unittest.main()
