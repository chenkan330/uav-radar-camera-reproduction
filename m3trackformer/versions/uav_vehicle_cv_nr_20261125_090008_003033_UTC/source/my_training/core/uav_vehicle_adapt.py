"""Warm-start the journal networks on independent moving-platform simulations."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .common import normalized_history, state_digest, write_json
from .journal_training_loss import training_classification_loss, training_distance_matrix
from .uav_vehicle_tracking import array_origin, make_history, prepare, simulate, world_to_array


SEED = 20261125
METHODS = ('cv_rnet', 'nrnet')


def normal_windows(batch, config, normalization, rng, per_scene, *, missed_history=False):
    # Equal numbers per trajectory; sample in disjoint temporal strata along each path.
    n, frames = batch.beam.shape
    h, future = config.history, config.future
    bins = np.linspace(h, frames - future + 1, per_scene + 1, dtype=int)
    if np.any(np.diff(bins) < 1):
        raise ValueError('Not enough frames for stratified windows.')
    steps = rng.integers(bins[:-1], bins[1:], size=(n, per_scene)).ravel()
    rows = np.repeat(np.arange(n), per_scene)
    valid = np.ones((len(rows), h), dtype=bool)
    if missed_history:
        # The normal branch is also used immediately after a successful reacquisition.
        masked = rng.random(len(rows)) < .15
        valid[masked, :-1] = rng.random((masked.sum(), h - 1)) > .25
    raw = make_history(batch, rows, steps, valid)
    history = normalized_history(raw, valid, normalization)
    xyz = world_to_array(batch.vehicle[rows, steps - 1] - array_origin(batch, rows, steps))
    xyz = (xyz + np.array([0., 0., 10.])).astype('float32')
    labels = batch.beam[rows[:, None], steps[:, None] + np.arange(future)].astype('int64')
    return tuple(map(torch.from_numpy, (history, valid, xyz, labels)))


def recovery_windows(cases, normalization):
    raw, valid, failed, count, labels = (np.stack([case[k] for case in cases])
        for k in ('history', 'valid', 'failed', 'count', 'label'))
    x = normalized_history(raw, valid, normalization)
    return tuple(map(torch.from_numpy, (x, valid.astype(bool), failed.astype('int64'),
        count.astype('int64'), labels.astype('int64').reshape(-1, 1))))


def epoch_pass(nnet, rnet, data, device, batch_size, distance, *, optimizer=None, rng=None):
    recovery = rnet is not None
    model = rnet if recovery else nnet
    model.train(optimizer is not None)
    n = len(data[0])
    indices = np.arange(n) if optimizer is None else rng.permutation(n)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(indices), batch_size):
            take = torch.as_tensor(indices[start:start + batch_size])
            x, v, *args, target = [array[take].to(device) for array in data]
            if recovery:
                logits = rnet(nnet, x, v, *args, future=1)
            else:
                result = nnet(x, v, nnet.config.future)
                logits = result['logits']
            loss = training_classification_loss(logits, target, distance, nnet.config.angular_weight)
            if not recovery:
                loss = loss + nnet.config.position_weight * (result['position'] - args[0]).square().sum(1).mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)
                optimizer.step()
            top = logits[:, 0].detach().topk(4).indices
            hits = top == target[:, :1]
            totals += torch.stack([hits.any(1).sum(), hits[:, 0].sum(), loss.detach() * len(take)])
    top4, top1, loss = (totals / len(indices)).tolist()
    return {'top4': top4, 'top1': top1, 'loss': loss}


def fit(nnet, rnet, val, training_data, device, epochs, batch_size, rng, distance, lr):
    model = nnet if rnet is None else rnet
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best = None
    log = []
    for epoch in range(epochs + 1):
        if epoch:
            train = training_data()
            epoch_pass(nnet, rnet, train, device, batch_size, distance, optimizer=optimizer, rng=rng)
        metrics = epoch_pass(nnet, rnet, val, device, batch_size, distance)
        score = (metrics['top4'], metrics['top1'], -metrics['loss'])
        log.append({'epoch': epoch, **metrics})
        if best is None or score > best:
            best = score
            best_epoch = epoch
            best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        elif epoch - best_epoch >= 4:
            break
    model.load_state_dict(best_weights)
    model.requires_grad_(False).eval()
    return {'selected_epoch': best_epoch, 'validation': log, 'learning_rate': lr}


def collect_recovery(batch, nnet, rnet, checkpoint, device):
    cases, counts = [], {}
    for method in METHODS:
        before = len(cases)
        simulate(batch, nnet, rnet, checkpoint, device, method, collect=cases)
        counts[method] = len(cases) - before
    return cases, counts


def adapt(nnet, rnet, checkpoint, device, out, train_scenes, r_scenes, validation_scenes):
    """Return (adapted N, adapted R, inference metadata); inputs never include the test split."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device(device)
    nnet, rnet = deepcopy(nnet).to(device), deepcopy(rnet).to(device)
    nnet.eval()
    rnet.eval()
    config = nnet.config
    source_hash = {'nnet': state_digest(nnet), 'rnet': state_digest(rnet)}
    norm = checkpoint['normalization']
    train, recover, validation = map(prepare, (train_scenes, r_scenes, validation_scenes))
    batch_size = 128 if device.type == 'cuda' else 32
    distance = training_distance_matrix(config, device)
    val = normal_windows(validation, config, norm, rng, 24)
    per_scene = min(12, max(1, 4096 // len(train.beam)))
    def training_data():
        return normal_windows(train, config, norm, rng, per_scene, missed_history=True)

    print('Adapting N-Net on independent training scenes...', flush=True)
    normal_fit = fit(nnet, None, val, training_data, device, 10, batch_size, rng, distance, lr=8e-5)
    print('Collecting actual R-Net attempts from both closed-loop policies...', flush=True)
    train_cases, train_counts = collect_recovery(recover, nnet, rnet, checkpoint, device)
    val_cases, val_counts = collect_recovery(validation, nnet, rnet, checkpoint, device)
    if train_cases and val_cases:
        train_data, val_data = [recovery_windows(cases, norm) for cases in (train_cases, val_cases)]
        recover_fit = fit(nnet, rnet, val_data, lambda: train_data, device, 12, batch_size, rng, distance, lr=8e-5)
    else:
        rnet.requires_grad_(False).eval()
        recover_fit = {'selected_epoch': 0, 'reason': 'No observed recovery attempt in training or validation.'}

    n_meta = {key: checkpoint[key] for key in ('format', 'model_config', 'normalization', 'matmul_precision') if key in checkpoint}
    n_meta['model'] = nnet.state_dict()
    torch.save(n_meta, out/'adapted_nnet.pth')
    torch.save({'format': checkpoint['format'], 'model_config': config.dictionary(),
        'model': rnet.state_dict(), 'nnet_checkpoint': str((out/'adapted_nnet.pth').resolve())},
        out/'adapted_rnet.pth')
    write_json(out/'adaptation.json', {'seed': SEED, 'device': str(device), 'batch_size': batch_size,
        'source_model_state_sha256': source_hash, 'adapted_model_state_sha256': {'nnet':state_digest(nnet), 'rnet':state_digest(rnet)},
        'window_selection': 'Equal temporal strata per trajectory; fixed validation windows; best (top4, top1, -loss) including epoch zero.',
        'nnet_training_scenes': len(train.beam), 'nnet_windows_per_scene_per_epoch': per_scene,
        'rnet_training_scenes': len(recover.beam), 'validation_scenes': len(validation.beam),
        'rnet_training_attempts': train_counts, 'rnet_validation_attempts': val_counts,
        'rnet_source': 'Only causal R attempts after actual misses in rollouts of each policy; N frozen during R training.',
        'normalization': norm, 'position_target': 'Unscaled metres in array frame at commanded origin plus [0,0,10].',
        'nnet_fit': normal_fit, 'rnet_fit': recover_fit})
    return nnet, rnet, n_meta
