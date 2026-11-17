"""Probability projection and custom predictor causal access tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core.maneuver_tracking import MotionBeamPredictor
from core.multi_target import simulate
from core.models import JournalConfig


class FixedMotion(torch.nn.Module):
    config = SimpleNamespace(history=20)

    def forward(self, history, valid, dt, sigma):
        batch = len(history)
        mean = torch.tensor([200., 150., 80.])[None, None, None].expand(batch, 4, 3, 3).clone()
        return {'weights': torch.ones(batch, 4, 3)/3, 'mean': mean,
                'scale': torch.full_like(mean, 3.)}


class NoLegacyBeam(torch.nn.Module):
    config = JournalConfig(classes=1024)

    def forward(self, *args, **kwargs):
        raise AssertionError('Custom predictor should replace beam inference')


class ProjectionTests(unittest.TestCase):
    def test_probability_and_point_project_to_normalized_reproducible_beams(self):
        history = np.ones((2, 3, 20, 3), np.float32)
        valid = np.ones((2, 3, 20), bool)
        for mode in ('probability', 'point'):
            predictor = MotionBeamPredictor(FixedMotion(), 'cpu', mode=mode, fixed_scale=np.ones((4, 3)))
            a = predictor(history, valid, None, .05)
            b = predictor(history, valid, None, .05)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a.exp().sum(-1), torch.ones(2, 3))
            self.assertTrue(torch.isfinite(a).all())

    def test_tracker_hook_receives_only_past_and_hidden_measurements_cannot_change_decisions(self):
        class Recorder:
            history = 20
            def __init__(self):
                self.inputs = []
            def __call__(self, history, valid, sigma, dt):
                self.inputs.append((history.copy(), valid.copy()))
                logits = torch.zeros(*valid.shape[:2], 1024)
                logits[..., 0] = 30
                return logits
        beams = np.ones((1, 2, 13), dtype=np.int64)
        beams[:, 1, 5:] = 1000
        original = {'beam': beams, 'measurement': np.zeros((1, 2, 13, 3), np.float32),
                    'trajectory_id': np.array([[1, 2]])}
        changed = {k: v.copy() for k, v in original.items()}
        changed['measurement'][:, 1, 5:] = 123456.
        a, b = Recorder(), Recorder()
        norm = {'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]}}
        for scenes, hook in ((original, a), (changed, b)):
            rows = simulate(NoLegacyBeam(), None, norm, scenes, torch.device('cpu'), .05,
                            3, 'joint', beam_predictor=hook)[0]
            self.assertEqual(rows[0]['evaluated_blocks'], 5)
        for (xa, va), (xb, vb) in zip(a.inputs, b.inputs):
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(va, vb)
        self.assertFalse(a.inputs[0][1][..., :15].any())
        self.assertEqual(int(a.inputs[0][1].sum()), 10)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
