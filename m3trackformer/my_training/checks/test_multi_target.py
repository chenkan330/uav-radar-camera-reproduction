"""多目标关键语义：概率聚合、资源预算、逐目标恢复、因果性和终止口径。"""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from core.common import state_digest
from core.models import JournalConfig, JournalNNet, JournalRNet
from core.multi_target import _prediction, joint_beams, make_scenes, simulate
from step10d_journal_tracking import simulate as single_simulate


NORMALIZATION = {'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]}}


class FixedNNet:
    config = JournalConfig(classes=8)

    def __init__(self):
        self.inputs = []

    def __call__(self, history, valid):
        self.inputs.append((history.cpu().numpy().copy(), valid.cpu().numpy().copy()))
        logits = torch.arange(8., 0., -1., device=history.device)[None, None]
        return {'logits': logits.expand(len(history), 1, -1).clone()}


class RecoveryNet:
    def __init__(self):
        self.inputs = []

    def __call__(self, nnet, history, valid, failed, count):
        self.inputs.append((failed.cpu().numpy().copy(), count.cpu().numpy().copy()))
        logits = torch.zeros(len(history), 1, 8, device=history.device)
        logits[:, :, 7] = 20
        return logits


def scene(labels):
    beam = np.asarray(labels, dtype=np.int64)[None]
    return {'beam': beam, 'measurement': np.arange(beam.size*3, dtype=np.float32).reshape(*beam.shape, 3),
            'trajectory_id': np.arange(beam.shape[1])[None]+11}


class MultiTargetTests(unittest.TestCase):
    def test_probabilities_are_summed_not_logits(self):
        logits = torch.tensor([[[4., 1., 0.], [0., 1., 4.]]])
        shifted = logits + torch.tensor([[[1000.], [-3000.]]])
        torch.testing.assert_close(joint_beams(logits, 2), joint_beams(shifted, 2))
        self.assertEqual(set(joint_beams(logits, 2)[0].tolist()), {0, 2})

    def test_target_permutation_and_routing_with_actual_models(self):
        torch.manual_seed(13)
        c = JournalConfig(classes=32, d_model=32, heads=4, encoder_layers=1, decoder_layers=1, dropout=0)
        nnet = JournalNNet(c).eval().requires_grad_(False)
        rnet = JournalRNet(c).eval().requires_grad_(False)
        x = torch.randn(3, 5, 3)
        valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        counts = torch.tensor([0, 1, 4])
        failed = torch.full((3, 4, 6), -1, dtype=torch.long)
        failed[1, 0] = torch.arange(6)
        failed[2] = torch.arange(24).reshape(4, 6)
        before = state_digest(nnet), state_digest(rnet)
        with torch.inference_mode():
            original = _prediction(nnet, rnet, x, valid, failed, counts)
            normal = nnet(x[:1], valid[:1])['logits'][:, 0]
            permutation = torch.tensor([2, 0, 1])
            reordered = _prediction(nnet, rnet, x[permutation], valid[permutation],
                                    failed[permutation], counts[permutation])
        torch.testing.assert_close(original[:1], normal)
        torch.testing.assert_close(reordered, original[permutation], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(joint_beams(original[None], 6), joint_beams(reordered[None], 6))
        self.assertEqual(before, (state_digest(nnet), state_digest(rnet)))

    def test_one_shared_beam_can_hit_multiple_targets(self):
        data = scene([[1]*9, [1]*9])
        rows, _, events, scans, _ = simulate(FixedNNet(), None, NORMALIZATION, data,
                                             torch.device('cpu'), .05, 3, 'joint')
        self.assertTrue(rows[0]['success'])
        self.assertTrue(all(e['hit'] for e in events))
        self.assertTrue(all(s['beam_slots_used'] == 3 and s['unique_beams'] == 3 for s in scans))

    def test_recovery_is_per_target_and_records_all_global_beams(self):
        data = scene([[1]*9, [1]*5+[8, 8, 1, 1]])
        rnet = RecoveryNet()
        rows, targets, events, _, _ = simulate(FixedNNet(), rnet, NORMALIZATION, data,
                                              torch.device('cpu'), .05, 4, 'joint')
        self.assertTrue(rows[0]['success'])
        self.assertEqual(targets[1]['recoveries'], 1)
        self.assertEqual(len(rnet.inputs), 1)
        failed, counts = rnet.inputs[0]
        np.testing.assert_array_equal(counts, [1])
        np.testing.assert_array_equal(failed[0, 0], [0, 1, 2, 3])
        target_events = [e for e in events if e['target_index'] == 1]
        self.assertEqual([e['mode'] for e in target_events], ['N', 'R', 'N', 'N'])
        self.assertTrue(all(e['mode'] == 'N' for e in events if e['target_index'] == 0))

    def test_missing_measurements_are_hidden_and_peer_termination_is_censoring(self):
        data = scene([[1]*12, [1]*5+[8]*7])
        changed = {k: v.copy() for k, v in data.items()}
        changed['measurement'][0, 1, 5:] = 1e8
        for policy in ('joint', 'decoupled'):
            model1, model2 = FixedNNet(), FixedNNet()
            a = simulate(model1, None, NORMALIZATION, data, torch.device('cpu'), .05, 4, policy)
            b = simulate(model2, None, NORMALIZATION, changed, torch.device('cpu'), .05, 4, policy)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[0][0]['evaluated_blocks'], 5)
            self.assertFalse(a[0][0]['success'])
            self.assertTrue(a[1][0]['censored_by_peer_failure'])
            self.assertFalse(a[1][0]['terminal_failure'])
            self.assertTrue(a[1][1]['terminal_failure'])
            for (x1, v1), (x2, v2) in zip(model1.inputs, model2.inputs):
                np.testing.assert_array_equal(x1, x2)
                np.testing.assert_array_equal(v1, v2)

    def test_decoupled_floor_quota_and_duplicate_beams_consume_slots(self):
        data = scene([[1]*8, [1]*8])
        _, _, _, scans, _ = simulate(FixedNNet(), None, NORMALIZATION, data,
                                     torch.device('cpu'), .05, 5, 'decoupled')
        self.assertTrue(all(s['beam_slots_used'] == 4 and s['unique_beams'] == 2 for s in scans))

    def test_single_target_reduces_to_existing_single_loop(self):
        labels = [1]*5+[8, 1]+[8]*5+[1]
        data = scene([labels])
        memory = io.BytesIO()
        np.savez(memory, measurement=data['measurement'][:, 0], beam=data['beam'][:, 0],
                 trajectory_id=data['trajectory_id'][:, 0])
        memory.seek(0)
        expected, _ = single_simulate(FixedNNet(), None, NORMALIZATION, memory,
                                      torch.device('cpu'), .05, 4)
        result = simulate(FixedNNet(), None, NORMALIZATION, data,
                          torch.device('cpu'), .05, 4, 'joint')
        for key in ('success', 'duration_s', 'evaluated_blocks', 'recoveries'):
            self.assertEqual(result[0][0][key], expected[0][key])

    def test_scene_composition_is_reproducible_without_replacement(self):
        raw = {'trajectory_id': np.arange(7), 'measurement': np.zeros((7, 8, 3)),
               'beam': np.ones((7, 8), dtype=np.int64)}
        a, index, unused = make_scenes(raw, 3, seed=22)
        b, index2, _ = make_scenes(raw, 3, seed=22)
        np.testing.assert_array_equal(index, index2)
        self.assertEqual(a['beam'].shape, (2, 3, 8))
        self.assertEqual(len(np.unique(a['trajectory_id'])), 6)
        self.assertEqual(len(unused), 1)
        self.assertFalse(set(unused) & set(a['trajectory_id'].ravel()))
        with self.assertRaises(ValueError):
            make_scenes(raw, 3, limit=3)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
