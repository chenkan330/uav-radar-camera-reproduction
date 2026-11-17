"""The revised V4 must run real N/R branches, keep weights, and plot N-Net positions."""
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from core.common import state_digest
from core.maneuver_nnet_rnet import ORIGINAL_MODELS_SHA256, probability_summary, run_original_nr
from test_trajectory_figure import FixedPositionNet, NORMALIZATION, fixture


class NNet(torch.nn.Module):
    config = FixedPositionNet.config

    def forward(self, history, valid):
        return FixedPositionNet()(history, valid)


class RNet(torch.nn.Module):
    def forward(self, nnet, history, valid, failed, count):
        output = torch.zeros(len(history), 1, 8)
        output[..., 7] = 20
        return output


class OriginalNRTests(unittest.TestCase):
    def test_original_transformer_definitions_remain_byte_identical(self):
        path = Path(__file__).resolve().parents[1]/'core/models.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ORIGINAL_MODELS_SHA256)

    def test_runner_uses_rnet_after_misses_without_changing_weights(self):
        nnet, rnet = NNet(), RNet()
        before = state_digest(nnet), state_digest(rnet)
        result, trace, _, _, probability_rows, audit = run_original_nr(nnet, rnet,
            NORMALIZATION, fixture(), torch.device('cpu'), quality_mode='robust',
            dt=.05, budget=3, plot_id=400017)
        expected = sum(e['prior_misses'] > 0 for e in result[2])
        self.assertGreater(expected, 0)
        self.assertEqual(audit['rnet_target_frames'], expected)
        self.assertFalse(audit['alternative_prediction_network_used'])
        self.assertEqual(before, (state_digest(nnet), state_digest(rnet)))
        self.assertTrue(all(e['estimate_source'] == 'nnet_position_head' for e in trace if e['phase'] == 'tracking'))
        self.assertTrue(all(e['estimated_x_m'] == 3. for e in trace if e['phase'] == 'tracking' and e['hit']))
        self.assertEqual(len(probability_rows), len(result[2]))

    def test_probability_calibration_is_beam_only_and_not_fit_on_test(self):
        rows = [dict(confidence=.8, entropy_nats=.3, nll_nats=.5, brier_score=.1,
                     selected_probability_mass=.9, top1_correct=value) for value in (True, False)]
        result = probability_summary(rows)
        self.assertEqual(result['top1_accuracy'], .5)
        self.assertAlmostEqual(result['ece'], .3)
        self.assertFalse(result['is_3d_motion_uncertainty'])
        self.assertFalse(result['calibrated_on_test'])


if __name__ == '__main__':
    unittest.main()
