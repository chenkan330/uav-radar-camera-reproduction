"""Numerical checks for the B-group CSI representation and timing."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.isac_b_csi_predictor import (ARRAY_SIDE, beam_powers, causal_reference, csi_parameters,
                                      evaluate_predictions, make_channel, position_parameters)


class TestCsiPredictor(unittest.TestCase):
    def test_fft_beam_grid_matches_complex_codebook_for_general_channels(self):
        rng = np.random.default_rng(2)
        h = rng.normal(size=(2, ARRAY_SIDE**2)) + 1j*rng.normal(size=(2, ARRAY_SIDE**2))
        axis = np.arange(ARRAY_SIDE)
        grid = -1 + 2*(axis+.5)/ARRAY_SIDE
        weights = np.exp(1j*np.pi*axis[:, None]*grid)/np.sqrt(ARRAY_SIDE)
        expected = np.abs(h.conj() @ np.kron(weights, weights))**2
        np.testing.assert_allclose(beam_powers(h), expected, atol=1.e-11, rtol=1.e-11)

    def test_csi_representation_removes_only_common_phase(self):
        parameters = np.array([[-.342, .271, .2], [.31, -.41, -1.]])
        truth = make_channel(parameters)*np.exp(1j*np.array([.61, -2.91]))[:, None]
        np.testing.assert_allclose(csi_parameters(truth), parameters, atol=1.e-7)
        data = {"csi": truth, "trajectory_id": [0, 1], "session_group": np.array([2, 4]), "label_frame": [8, 9]}
        results, _ = evaluate_predictions({"method": parameters}, data)
        self.assertAlmostEqual(results["method"]["phase_aligned_nmse"], 0.)
        self.assertEqual(results["method"]["oracle_beam_in_top12_rate"], 1.)

    def test_next_frame_extrapolation_uses_frame_times_with_missing_observations(self):
        first = np.array([74., -29., 178.])
        motion_per_frame = np.array([.8, .25, -.5])
        positions = first + np.arange(5)[:, None]*motion_per_frame
        valid = np.array([[True, False, True, True, False]])
        positions[~valid[0]] = 0.
        last, predicted = causal_reference(positions[None], valid)
        np.testing.assert_allclose(last, position_parameters((first+3*motion_per_frame)[None]), atol=1.e-7)
        np.testing.assert_allclose(predicted, position_parameters((first+5*motion_per_frame)[None]), atol=1.e-7)


if __name__ == "__main__":
    unittest.main()
