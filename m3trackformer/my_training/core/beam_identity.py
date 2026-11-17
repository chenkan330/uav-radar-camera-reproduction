"""Anonymous, single-frame identity decisions; never update tracking histories.

The sensor/scoring adapter keeps origin labels separate from AnonymousDetections.
This is a conditional association audit on the original Huber tracker's history,
not a closed-loop tracker driven by its own association errors.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .measurement_quality import spherical_to_cartesian, cartesian_measurement_covariance

METHODS = {'beam_argmax': '波束贡献最高直接归属', 'beam_geometry': '波束评分＋几何门控＋全局匹配'}
DEFAULT_GATE = 16.26623619623813  # chi-square(3), 0.999; nominal Gaussian interpretation only


@dataclass(frozen=True)
class AnonymousDetections:
    beams: np.ndarray
    measurements: np.ndarray
    sigmas: np.ndarray

    def __post_init__(self):
        for name, dtype in (('beams', int), ('measurements', float), ('sigmas', float)):
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if (self.beams.ndim != 1 or self.measurements.shape != (len(self.beams), 3)
                or self.sigmas.shape != self.measurements.shape
                or not np.isfinite(self.measurements).all()
                or not np.isfinite(self.sigmas).all() or (self.sigmas <= 0).any()
                or (self.beams < 0).any()):
            raise ValueError('Invalid anonymous sensor returns')


def contribution_fractions(probabilities, beams):
    """Relative track contributions to each beam, not calibrated identity posteriors."""
    scores = np.asarray(probabilities, dtype=float)[:, beams]
    safe = np.maximum(scores, 1e-300)
    return safe/safe.sum(axis=0, keepdims=True)


def associate(probabilities, detections, predicted_position, predicted_covariance,
              *, gate=DEFAULT_GATE, beam_weight=1., unmatched_cost=20.):
    """Pure matching function: no source IDs, truth positions, or outlier labels."""
    probability = np.asarray(probabilities, dtype=float)
    if (probability.ndim != 2 or not np.isfinite(probability).all()
            or (probability < 0).any() or not np.allclose(probability.sum(-1), 1.)):
        raise ValueError('Expected per-track normalized beam probabilities')
    tracks, classes = probability.shape
    if (detections.beams >= classes).any():
        raise ValueError('Detection beam exceeds codebook')
    count = len(detections.beams)
    if not count:
        return {name: np.empty(0, dtype=int) for name in METHODS}, np.empty((tracks, 0))
    # Canonical order uses only observable attributes, never source order/identity.
    z = detections.measurements
    order = np.lexsort((z[:, 2], z[:, 1], z[:, 0], detections.beams))
    fractions = contribution_fractions(probability, detections.beams)
    direct = fractions.argmax(axis=0)
    residual = spherical_to_cartesian(z)[None] - np.asarray(predicted_position)[:, None]
    covariance = (np.asarray(predicted_covariance)[:, None]
                  + cartesian_measurement_covariance(z, detections.sigmas)[None])
    distance = np.maximum(np.einsum('tdi,tdi->td', residual,
        np.linalg.solve(covariance, residual[..., None])[..., 0]), 0.)
    costs = .5*distance-beam_weight*np.log(np.maximum(fractions, 1e-12))
    valid = distance <= gate
    # One dummy column per track allows no assignment instead of forcing an outlier.
    augmented = np.full((tracks, count+tracks), unmatched_cost)
    augmented[:, :count] = np.where(valid[:, order], costs[:, order], 1e9)
    track_indices, columns = linear_sum_assignment(augmented)
    matched = np.full(count, -1, dtype=int)
    for track, column in zip(track_indices, columns):
        if column < count and augmented[track, column] < unmatched_cost:
            matched[order[column]] = track
    return {'beam_argmax': direct, 'beam_geometry': matched}, distance


class IdentityAudit:
    """Sensor + offline evaluator around the label-free association function."""
    def __init__(self, membership, seed, history, dt, *, gate=DEFAULT_GATE,
                 beam_weight=1., unmatched_cost=20., record_rows=True):
        self.membership = np.asarray(membership)
        self.groups, self.targets = self.membership.shape
        self.seed, self.history, self.dt = seed, history, dt
        self.parameters = dict(gate=gate, beam_weight=beam_weight, unmatched_cost=unmatched_cost)
        self.record_rows = record_rows
        self.rows, self.contribution_rows = [], []
        self.snapshot, self.collision_snapshot = None, None
        self.scene_frames = np.zeros(self.groups, dtype=int)
        self.counters = {name: dict(total=0, assigned=0, correct=0, identity_switches=0,
            collision_total=0, collision_correct=0, duplicate_track_frames=0,
            confusion=np.zeros((self.targets, self.targets+1), dtype=int)) for name in METHODS}
        self.previous = {name: np.broadcast_to(np.arange(self.targets), self.membership.shape).copy()
                         for name in METHODS}
        self.beam_only_upper_correct = 0
        self.near_tie_returns = 0

    def observe(self, *, step, live, selected, probabilities, hit, beam_labels,
                measurements, sigmas, state, covariance, transition, process_covariance):
        """Called before the current Huber update. Only actual returns form packets."""
        predicted = state @ transition.T
        predicted_cov = transition @ covariance @ transition.T+process_covariance
        for i, group in enumerate(live):
            group = int(group)
            self.scene_frames[group] += 1
            source = np.flatnonzero(hit[i])
            # Labelled arrays belong only to this simulation sensor adapter.
            # Algorithms below receive none of `source` or `membership`.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, group, step]))
            source = source[rng.permutation(len(source))]
            packet = AnonymousDetections(beam_labels[i, source], measurements[i, source], sigmas[i, source])
            decisions, distances = associate(probabilities[i], packet, predicted[i, :, :3],
                predicted_cov[i, :, :3, :3], **self.parameters)
            fractions = contribution_fractions(probabilities[i], packet.beams)
            unique, counts = np.unique(packet.beams, return_counts=True)
            collisions = np.isin(packet.beams, unique[counts > 1])
            self.beam_only_upper_correct += len(unique)  # one return per real target in this sensor
            if len(source):
                ranked = np.sort(fractions, axis=0)
                margin = ranked[-1]-ranked[-2] if self.targets > 1 else ranked[-1]
                self.near_tie_returns += int((margin < .1).sum())
            else:
                margin = np.empty(0)
            # Source IDs enter evaluation only AFTER the two rules have decided.
            for method, assigned in decisions.items():
                c = self.counters[method]
                c['total'] += len(source)
                c['assigned'] += int((assigned >= 0).sum())
                c['correct'] += int((assigned == source).sum())
                c['collision_total'] += int(collisions.sum())
                c['collision_correct'] += int(((assigned == source) & collisions).sum())
                labels = assigned[assigned >= 0]
                c['duplicate_track_frames'] += int(len(labels) != len(np.unique(labels)))
                for j, origin in enumerate(source):
                    label = int(assigned[j])
                    c['confusion'][origin, label if label >= 0 else self.targets] += 1
                    if label >= 0:
                        c['identity_switches'] += int(self.previous[method][group, origin] != label)
                        self.previous[method][group, origin] = label
            if self.record_rows:
                xyz = spherical_to_cartesian(packet.measurements)
                for j, origin in enumerate(source):
                    self.rows.append(dict(noise_seed=self.seed, scene_id=group, time_index=step,
                        time_s=(step-self.history)*self.dt, detection_index=j,
                        true_target_index_evaluation_only=int(origin),
                        true_trajectory_id_evaluation_only=int(self.membership[group, origin]),
                        beam_1based=int(packet.beams[j]+1), same_beam_collision=bool(collisions[j]),
                        beam_argmax=int(decisions['beam_argmax'][j]),
                        beam_geometry=int(decisions['beam_geometry'][j]), contribution_margin=float(margin[j]),
                        measurement_x_m=float(xyz[j, 0]), measurement_y_m=float(xyz[j, 1]),
                        measurement_z_m=float(xyz[j, 2])))
            if self.snapshot is None or (collisions.any() and self.collision_snapshot is None):
                capture = dict(scene_id=group, time_index=step, noise_seed=self.seed,
                    selected_beams_1based=(selected[i]+1).tolist(), probabilities=probabilities[i][:, selected[i]].tolist(),
                    detected_beams_1based=(packet.beams+1).tolist(),
                    true_source_evaluation_only=source.tolist(),
                    decisions={k: v.tolist() for k, v in decisions.items()})
                if self.snapshot is None:
                    self.snapshot = capture
                if collisions.any() and self.collision_snapshot is None:
                    self.collision_snapshot = capture

    def summary(self):
        results = {}
        for method, values in self.counters.items():
            c = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()}
            c.update(accuracy_all_returns=c['correct']/c['total'] if c['total'] else None,
                precision_assigned=c['correct']/c['assigned'] if c['assigned'] else None,
                assignment_coverage=c['assigned']/c['total'] if c['total'] else None,
                accuracy_same_beam=c['collision_correct']/c['collision_total'] if c['collision_total'] else None,
                beam_only_upper_bound=self.beam_only_upper_correct/c['total'] if c['total'] else None,
                near_tie_returns=self.near_tie_returns)
            results[method] = c
        return results


def pool_summaries(summaries):
    pooled = {}
    for method in METHODS:
        rows = [s[method] for s in summaries]
        c = {key: sum(r[key] for r in rows) for key in ('total', 'assigned', 'correct',
            'identity_switches', 'collision_total', 'collision_correct', 'duplicate_track_frames', 'near_tie_returns')}
        c['confusion'] = np.sum([r['confusion'] for r in rows], axis=0).tolist()
        c.update(accuracy_all_returns=c['correct']/c['total'] if c['total'] else None,
            precision_assigned=c['correct']/c['assigned'] if c['assigned'] else None,
            assignment_coverage=c['assigned']/c['total'] if c['total'] else None,
            accuracy_same_beam=c['collision_correct']/c['collision_total'] if c['collision_total'] else None,
            beam_only_upper_bound=sum(r['beam_only_upper_bound']*r['total'] for r in rows if r['total'])/c['total']
                if c['total'] else None)
        pooled[method] = c
    return pooled
