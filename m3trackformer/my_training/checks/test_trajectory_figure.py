"""Trajectory plots must reflect causal simulation estimates and event counts."""
import hashlib
import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from core.measurement_quality import spherical_to_cartesian
from core.models import JournalConfig
from core.multi_target import simulate, summarize
from core.trajectory_figure import save_trajectory_figure, trace_statistics


class FixedPositionNet:
    config = JournalConfig(classes=8)

    def __call__(self, history, valid):
        logits = torch.arange(8., 0., -1.)[None, None].expand(len(history), 1, -1)
        return {'logits': logits, 'position': torch.tensor([3., 4., 5.])[None].expand(len(history), 3)}


def fixture():
    measurement = np.broadcast_to([100., .1, 1.2], (1, 1, 17, 3)).copy()
    return {'measurement': measurement, 'measurement_sigma': np.ones_like(measurement)*.01,
        'position': spherical_to_cartesian(measurement), 'trajectory_id': np.array([[400017]]),
        'beam': np.array([[[1]*5+[8, 8, 1]+[8]*9]])}


NORMALIZATION = {'normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]}}


class TrajectoryFigureTests(unittest.TestCase):
    def run_simulation(self, data, trace=None, quality='raw'):
        return simulate(FixedPositionNet(), None, NORMALIZATION, data, torch.device('cpu'),
                        .05, 3, 'joint', quality_mode=quality,
                        trajectory_trace=trace, trace_trajectory_id=400017)

    def test_trace_does_not_change_decisions_and_counts_unresolved_loss(self):
        trace = []
        baseline = self.run_simulation(fixture())
        result = self.run_simulation(fixture(), trace)
        self.assertEqual(baseline[:4], result[:4])
        stats = trace_statistics(trace, .05)
        self.assertEqual(stats['lost_events'], 2)
        self.assertEqual(stats['recoveries'], 1)
        self.assertEqual(stats['recovery_rate'], .5)
        self.assertAlmostEqual(stats['tracking_duration_s'], .4)
        self.assertEqual([r['estimated_x_m'] for r in trace if r['time_index'] == 5], [None])
        hit = next(r for r in trace if r['time_index'] == 7)
        self.assertEqual(hit['estimated_x_m'], 3.)
        self.assertEqual(hit['estimate_source'], 'legacy_neural_position_head')

    def test_robust_trace_uses_filter_and_never_acquires_missed_measurements(self):
        data = fixture()
        changed = {k: v.copy() for k, v in data.items()}
        changed['measurement'][..., 5:7, :] = np.nan
        changed['measurement'][..., 8:, :] = np.nan
        traces = [[], []]
        a = self.run_simulation(data, traces[0], 'robust')
        b = self.run_simulation(changed, traces[1], 'robust')
        self.assertEqual(a[:4], b[:4])
        self.assertEqual(traces[0], traces[1])
        hit = next(r for r in traces[0] if r['time_index'] == 7)
        self.assertEqual(hit['estimate_source'], 'causal_filtered_measurement')
        self.assertAlmostEqual(hit['estimated_x_m'], float(data['position'][0, 0, 7, 0]), places=3)
        self.assertNotEqual(hit['estimated_x_m'], 3.)

    def test_saved_samples_leave_misses_and_unobserved_tail_blank_and_hashes_match(self):
        trace, data = [], fixture()
        result = self.run_simulation(data, trace)
        summary = summarize(result[0], result[1], result[4], .05, 3, 1, 'joint')
        cache = Path(__file__).resolve().parents[2]/'.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            png = save_trajectory_figure(directory, data, trace, result[1], summary,
                dt=.05, history=5, trajectory_id=400017, title='仿真记录验证')
            meta = json.loads(png.with_suffix('.json').read_text(encoding='utf-8'))
            csv_path = png.with_name('single_trajectory_tracking_samples.csv')
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 17)
            self.assertEqual(rows[5]['estimated_x_m'], '')
            self.assertEqual(rows[-1]['estimated_x_m'], '')
            self.assertEqual(rows[-1]['phase'], 'not_evaluated')
            self.assertEqual(rows[7]['estimated_x_m'], '3.0')
            self.assertEqual(meta['single_trajectory']['recovery_rate'], .5)
            self.assertEqual(meta['stop_reason'], 'target_failure')
            self.assertEqual(meta['figure_sha256'], hashlib.sha256(png.read_bytes()).hexdigest())
            self.assertEqual(meta['samples_sha256'], hashlib.sha256(csv_path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                save_trajectory_figure(directory, data, trace, result[1], summary,
                    dt=.05, history=5, trajectory_id=400017, title='拒绝覆盖')

    def test_no_loss_has_unavailable_rate_not_perfect_score(self):
        rows = [{'phase': 'initialization', 'time_index': 0, 'hit': False},
                {'phase': 'tracking', 'time_index': 1, 'hit': True}]
        stats = trace_statistics(rows, .05)
        self.assertIsNone(stats['recovery_rate'])
        self.assertEqual(stats['lost_events'], 0)

    def test_nnet_trace_with_robust_input_keeps_real_neural_output(self):
        data, trace, probabilities = fixture(), [], []
        result = simulate(FixedPositionNet(), None, NORMALIZATION, data, torch.device('cpu'),
            .05, 3, 'joint', quality_mode='robust', trajectory_trace=trace,
            trace_trajectory_id=400017, trajectory_estimator='nnet',
            beam_probability_metrics=probabilities)
        self.assertEqual(self.run_simulation(data, quality='robust')[:4], result[:4])
        hit = next(r for r in trace if r['time_index'] == 7)
        self.assertEqual(hit['estimate_source'], 'nnet_position_head')
        self.assertEqual(hit['estimated_x_m'], 3.)
        self.assertEqual(len(probabilities), result[0][0]['evaluated_blocks'])
        self.assertTrue(all(0 <= r['selected_probability_mass'] <= 1.000000001 for r in probabilities))


if __name__ == '__main__':
    unittest.main()
