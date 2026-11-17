"""Identity ambiguity, rejection, privacy boundary, and observer invariance."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from core.beam_identity import AnonymousDetections, associate, IdentityAudit
from core.multi_target import simulate
from core.optimization_experiment import load_tracking_choice, sha
from test_trajectory_figure import fixture, FixedPositionNet, NORMALIZATION


class BeamIdentityTests(unittest.TestCase):
    def packet(self, ranges):
        z = np.array([[distance, 0., np.pi/2] for distance in ranges])
        return AnonymousDetections(np.zeros(len(z), dtype=int), z,
            np.broadcast_to([.5, .001, .001], z.shape))

    def match(self, packet):
        return associate(np.array([[.9,.1],[.8,.2]]), packet,
            np.array([[100.,0.,10.],[200.,0.,10.]]), np.broadcast_to(np.eye(3)*.25,(2,3,3)))[0]

    def test_same_beam_argmax_cannot_separate_two_returns_but_range_can(self):
        result = self.match(self.packet([100.,200.]))
        np.testing.assert_array_equal(result['beam_argmax'], [0,0])
        np.testing.assert_array_equal(result['beam_geometry'], [0,1])

    def test_matching_is_invariant_to_detection_order_without_source_ids(self):
        original = self.match(self.packet([100.,200.]))
        shuffled = self.match(self.packet([200.,100.]))
        for name in original:
            np.testing.assert_array_equal(original[name],shuffled[name][::-1])
        self.assertEqual(set(AnonymousDetections.__dataclass_fields__), {'beams','measurements','sigmas'})
        with self.assertRaises(ValueError):
            self.packet([100.]).measurements[0,0] = 999.

    def test_outlier_can_be_unassigned_and_empty_frames_work(self):
        result = self.match(self.packet([1000.]))
        np.testing.assert_array_equal(result['beam_geometry'], [-1])
        packet = AnonymousDetections(np.empty(0,dtype=int),np.empty((0,3)),np.empty((0,3)))
        result = self.match(packet)
        self.assertTrue(all(len(values)==0 for values in result.values()))

    def test_global_assignment_does_not_duplicate_a_track(self):
        result = self.match(self.packet([100.,100.1,200.]))['beam_geometry']
        accepted = result[result>=0]
        self.assertEqual(len(accepted),len(np.unique(accepted)))
        self.assertEqual(int((result<0).sum()),1)

    def test_passive_audit_changes_neither_baseline_nor_network_trace(self):
        data = fixture()
        traces = [[],[]]
        a = simulate(FixedPositionNet(),None,NORMALIZATION,data,torch.device('cpu'),.05,3,'joint',
            quality_mode='robust',trajectory_trace=traces[0],trajectory_estimator='nnet')
        observer = IdentityAudit(data['trajectory_id'],20261116,5,.05)
        b = simulate(FixedPositionNet(),None,NORMALIZATION,data,torch.device('cpu'),.05,3,'joint',
            quality_mode='robust',trajectory_trace=traces[1],trajectory_estimator='nnet',detection_observer=observer)
        self.assertEqual(a[:4],b[:4]); self.assertEqual(traces[0],traces[1])
        self.assertEqual(observer.summary()['beam_argmax']['total'],1)
        changed = {k:v.copy() for k,v in data.items()}
        changed['measurement'][...,5:7,:] = np.nan
        changed['measurement'][...,8:,:] = np.nan
        other = IdentityAudit(data['trajectory_id'],20261116,5,.05)
        simulate(FixedPositionNet(),None,NORMALIZATION,changed,torch.device('cpu'),.05,3,'joint',
            quality_mode='robust',detection_observer=other)
        self.assertEqual(observer.rows,other.rows)
        self.assertEqual(observer.summary(),other.summary())

    def test_rejections_stay_in_all_return_denominator(self):
        audit = IdentityAudit(np.array([[11,12]]),1,5,.05)
        c = audit.counters['beam_geometry']
        c.update(total=10,assigned=6,correct=6)
        m = audit.summary()['beam_geometry']
        self.assertEqual(m['accuracy_all_returns'],.6)
        self.assertEqual(m['precision_assigned'],1.)
        self.assertEqual(m['assignment_coverage'],.6)

    def test_user_baseline_persists_over_new_automatic_sweep_results(self):
        cache = Path(__file__).resolve().parents[2]/'.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            base = Path(folder); output = base/'outputs'; output.mkdir()
            record = {'configuration':{'id':'huber_base'},'selection_basis':'user_requested_baseline'}
            lock = output/'user_choice.json'; lock.write_text(json.dumps(record),encoding='utf-8')
            choice = {**record,'selected_on_validation':False,'selection_record':str(lock),
                      'selection_lock_sha256':sha(lock)}
            (output/'current_tracking_configuration.json').write_text(json.dumps(choice),encoding='utf-8')
            (output/'current_optimized_tracking.json').write_text('{"configuration":{"id":"student_6"}}',encoding='utf-8')
            selected,_ = load_tracking_choice(base)
            self.assertEqual(selected['configuration']['id'],'huber_base')
            self.assertEqual(json.loads((output/'current_optimized_tracking.json').read_text())['configuration']['id'],'student_6')
            record['configuration']['id'] = 'student_6'
            lock.write_text(json.dumps(record),encoding='utf-8')
            with self.assertRaises(RuntimeError):
                load_tracking_choice(base)


if __name__ == '__main__':
    unittest.main()
