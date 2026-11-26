"""Closed-loop single-vehicle beam tracking with a preplanned moving radar."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .common import BASE, load_nnet, normalized_history, read_json
from .dft_codebook import centered_dft_beam_coordinates, centered_dft_beam_ids
from .models import JournalRNet
from .uav_vehicle_scene import DT


HISTORY = 5
BUDGET = 4
GRID = centered_dft_beam_coordinates()[:32, 0]
REFERENCE_BS = np.array([0., 0., 10.])
MEASUREMENT_SIGMA = np.array([1., np.deg2rad(.5), np.deg2rad(.5)])


def world_to_array(delta):
    # Fixed west/north/down right-handed array; this rotation is its own inverse.
    return delta * np.array([-1., 1., -1.])


def spherical(vectors):
    x, y, z = np.moveaxis(vectors, -1, 0)
    xy = np.hypot(x, y)
    return np.stack([np.hypot(xy, z), np.arctan2(y, x), np.arctan2(xy, z)], -1)


def cartesian(measurements):
    distance, az, polar = np.moveaxis(measurements, -1, 0)
    xy = distance * np.sin(polar)
    return np.stack([xy*np.cos(az), xy*np.sin(az), distance*np.cos(polar)], -1)


@dataclass
class Batch:
    measurement: np.ndarray
    observed_world: np.ndarray
    platform: np.ndarray
    vehicle: np.ndarray
    command: np.ndarray
    beam: np.ndarray


def prepare(scenes):
    platform = np.stack([s.platform for s in scenes])
    vehicle = np.stack([s.vehicle for s in scenes])
    local = world_to_array(vehicle - platform)
    observed = spherical(local) + MEASUREMENT_SIGMA * np.stack([s.noise for s in scenes])
    observed[..., 0] = observed[..., 0].clip(min=1e-6)
    observed[..., 1] = (observed[..., 1] + np.pi) % (2*np.pi) - np.pi
    observed[..., 2] = observed[..., 2].clip(0, np.pi)
    measured_world = platform + world_to_array(cartesian(observed))
    truth_ids = centered_dft_beam_ids(local[..., :2] / np.linalg.norm(local, axis=-1, keepdims=True)) - 1
    return Batch(observed, measured_world, platform, vehicle,
                 np.stack([s.command for s in scenes]), truth_ids)


def array_origin(batch, rows, step):
    return batch.platform[rows, step - 1] + batch.command[rows, step - 1]


def make_history(batch, rows, step, valid, *, offset=0):
    """Mask unsuccessful probes; place received history in one common array frame.

    offset=0 forecasts step and uses its already-issued flight command. offset=1
    is a post-detection position estimate, using the actual current navigation pose.
    """
    frames = np.asarray(step)[..., None] + np.arange(-HISTORY + offset, offset)
    reference = array_origin(batch, rows, step) if offset == 0 else batch.platform[rows, step]
    points = np.zeros((*valid.shape, 3), dtype=np.float64)
    i, j = np.nonzero(valid)
    frame_ids = np.broadcast_to(frames, valid.shape)[i, j]
    points[i, j] = world_to_array(batch.observed_world[rows[i], frame_ids] - reference[i])
    return spherical(points).astype(np.float32)


def pick_cv(history, valid):
    """Batched least-squares constant velocity in a common frame, accounting for gaps."""
    times = np.arange(-HISTORY, 0, dtype=float)
    valid_count = valid.sum(axis=1)
    mean_time = (valid * times).sum(axis=1) / valid_count
    centered = (times - mean_time[:, None]) * valid
    xyz = cartesian(history)
    mean_xyz = (xyz * valid[..., None]).sum(axis=1) / valid_count[:, None]
    denominator = np.sum(centered**2, axis=1)
    # With one valid observation the zero slope gives a position hold.
    slope = (centered[..., None] * xyz).sum(axis=1) / np.where(denominator > 0, denominator, 1)[:, None]
    prediction = mean_xyz - mean_time[:, None] * slope
    uv = prediction[:, :2] / np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-9)
    delta = (GRID[None, :, None] - uv[:, None, :] + 1) % 2 - 1
    gain = (np.sinc(16 * delta) / np.sinc(delta / 2))**2
    powers = (gain[:, :, 1, None] * gain[:, None, :, 0]).reshape(len(history), -1)
    return np.argsort(-powers, axis=1, kind='stable')[:, :BUDGET]


def load_original(device):
    config = read_json(BASE / 'outputs/current_journal_experiment.json')

    def local_path(key):
        # Historical JSON may still name an earlier checkout.
        parts = Path(config[key].replace('\\', '/')).parts
        return BASE.joinpath(*parts[parts.index('outputs'):])

    npath, rpath = local_path('nnet_checkpoint'), local_path('rnet_checkpoint')
    nnet, ck = load_nnet(npath, device)
    rnet = JournalRNet(nnet.config).to(device)
    saved = torch.load(rpath, map_location=device, weights_only=False)
    rnet.load_state_dict(saved['model'], strict=True)
    return nnet, rnet.eval(), ck, npath, rpath


def tensor_history(raw, valid, ck, device):
    normalized = normalized_history(raw, valid, ck['normalization'])
    return torch.from_numpy(normalized).to(device), torch.as_tensor(valid, device=device)


@torch.inference_mode()
def simulate(batch, nnet, rnet, checkpoint, device, method, collect=None):
    """The oracle only gates the sensor after all active tracks have committed beams."""
    if method not in ('cv_rnet', 'nrnet'):
        raise ValueError(method)
    n, frames = batch.beam.shape
    limit = nnet.config.failure_limit
    valid_samples = np.zeros((n, frames), bool)
    valid_samples[:, :HISTORY] = True
    lost = np.zeros(n, int)
    missed_beams = np.full((n, limit - 1, BUDGET), -1, np.int64)
    stats = {key: np.zeros(n, int) for key in
             ('attempts', 'hits', 'lost_events', 'recoveries', 'normal_attempts', 'normal_hits', 'r_attempts', 'r_hits')}
    squared_error = np.zeros(n)
    events = []
    for step in range(HISTORY, frames):
        active = np.flatnonzero(lost < limit)
        if not len(active):
            break
        valid = valid_samples[active, step-HISTORY:step]
        raw = make_history(batch, active, step, valid)
        choices = np.empty((len(active), BUDGET), np.int64)
        normal = lost[active] == 0
        for is_normal in (True, False):
            group = normal if is_normal else ~normal
            if not group.any():
                continue
            ids = active[group]
            if is_normal and method == 'cv_rnet':
                choices[group] = pick_cv(raw[group], valid[group])
                continue
            x, mask = tensor_history(raw[group], valid[group], checkpoint, device)
            if is_normal:
                scores = nnet(x, mask)['logits']
            else:
                scores = rnet(nnet, x, mask, torch.as_tensor(missed_beams[ids], device=device),
                              torch.as_tensor(lost[ids], device=device))
            choices[group] = scores[:, 0].topk(BUDGET, dim=-1).indices.cpu().numpy()

        target_beam = batch.beam[active, step]
        hit = (choices == target_beam[:, None]).any(axis=1)
        if collect is not None:
            for j in np.flatnonzero(~normal):
                i = active[j]
                collect.append(dict(history=raw[j].copy(), valid=valid[j].copy(),
                                    failed=missed_beams[i].copy(), count=int(lost[i]), label=int(target_beam[j])))
        good, bad = active[hit], active[~hit]
        stats['attempts'][active] += 1
        stats['normal_attempts'][active[normal]] += 1
        stats['r_attempts'][active[~normal]] += 1
        stats['hits'][good] += 1
        stats['normal_hits'][active[normal & hit]] += 1
        stats['r_hits'][active[~normal & hit]] += 1
        stats['recoveries'][active[~normal & hit]] += 1
        stats['lost_events'][active[normal & ~hit]] += 1
        valid_samples[good, step] = True
        if collect is None:
            estimates = np.full((len(active), 3), np.nan)
            if len(good):
                if method == 'cv_rnet':
                    estimates[hit] = batch.observed_world[good, step]
                else:
                    update_mask = valid_samples[good, step-HISTORY+1:step+1]
                    hist = make_history(batch, good, step, update_mask, offset=1)
                    x, mask = tensor_history(hist, update_mask, checkpoint, device)
                    position = nnet(x, mask)['position'].cpu().numpy()
                    estimates[hit] = batch.platform[good, step] + world_to_array(position - REFERENCE_BS)
                squared_error[good] += ((estimates[hit] - batch.vehicle[good, step])**2).sum(axis=1)
            for j, i in enumerate(active):
                events.append(dict(method=method, scene=int(i), frame=step, time_s=step*DT,
                                   mode=('CV' if method == 'cv_rnet' else 'N') if normal[j] else 'R',
                                   prior_misses=int(lost[i]), hit=int(hit[j]), oracle_beam=int(target_beam[j])+1,
                                   selected_beams=' '.join(str(b) for b in choices[j]+1),
                                   estimate_x=estimates[j, 0], estimate_y=estimates[j, 1], estimate_z=estimates[j, 2]))
        storable = lost[bad] < limit - 1
        missed_beams[bad[storable], lost[bad[storable]]] = choices[~hit][storable]
        lost[bad] += 1
        lost[good] = 0
        missed_beams[good] = -1
    rows = [dict(method=method, scene=i, scheduled=frames-HISTORY, success=bool(lost[i] < limit),
                 misses=int(stats['attempts'][i]-stats['hits'][i]), squared_error_sum=float(squared_error[i]),
                 **{key: int(count[i]) for key, count in stats.items()}) for i in range(n)]
    return rows, events


def summarize(rows):
    total = {key: sum(r[key] for r in rows) for key in
             ('scheduled', 'hits', 'attempts', 'lost_events', 'recoveries', 'normal_attempts', 'normal_hits',
              'r_attempts', 'r_hits', 'squared_error_sum')}
    def fraction(num, den):
        return total[num] / total[den] if total[den] else None
    n = len(rows)
    completed = sum(r['success'] for r in rows)
    return dict(method=rows[0]['method'], num_scenes=n, completed=completed, completion_rate=completed/n,
                mean_valid_s=total['attempts']*DT/n, detection_rate=fraction('hits','scheduled'),
                attempted_hit_rate=fraction('hits','attempts'), recovery_rate=fraction('recoveries','lost_events'),
                position_rmse_m=np.sqrt(total['squared_error_sum']/total['hits']) if total['hits'] else None,
                normal_hit_rate=fraction('normal_hits','normal_attempts'), r_hit_rate=fraction('r_hits','r_attempts'),
                **total)
