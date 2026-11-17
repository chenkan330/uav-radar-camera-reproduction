"""V4: disjoint maneuver simulation, probabilistic Transformer training and calibration.

Motion labels are scoring metadata. They never enter the network. Calibration
uses a separate trajectory split after validation-only checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.common import BASE, new_run, read_json, seed_all, write_json, write_rows
from core.measurement_quality import NoiseConfig, RobustMeasurementProcessor, perturb_scenes


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_observations(data, seed, dt=.05):
    """Generate once, then causally preprocess all acquired observations for the forecast benchmark."""
    scene = {key: np.asarray(value)[:, None] for key, value in data.items()}
    disturbed, metadata = perturb_scenes(scene, NoiseConfig(), seed)
    data = {key: np.asarray(value)[:, 0] for key, value in disturbed.items()}
    raw, sigma = disturbed['measurement'], disturbed['measurement_sigma']
    processor = RobustMeasurementProcessor(raw[..., :5, :], sigma[..., :5, :], dt)
    filtered = np.empty_like(raw)
    filtered[..., :5, :] = processor.filtered_initial_history
    observed = np.ones(raw.shape[:2], bool)
    for step in range(5, raw.shape[-2]):
        filtered[..., step, :], _ = processor.step(step, observed, raw[..., step, :], sigma[..., step, :])
    data['filtered_measurement'] = filtered[:, 0]
    return data, metadata


def prepare_data(out, *, small=False, seed=20261121, original_data=None):
    """Create genuinely disjoint simulated trajectories; anchor geometry uses original train only."""
    from core.maneuver_data import generate_split
    out = Path(out)
    root = out/'data'
    root.mkdir(parents=True, exist_ok=False)
    if original_data is None:
        current = read_json(BASE/'outputs/current_journal_experiment.json')
        original_data = Path(current['data'])
    original_data = Path(original_data)
    with np.load(original_data/'nnet_train.npz') as original:
        anchors = {'position': original['position'].copy(),
                   'velocity': np.gradient(original['position'], .05, axis=1).astype('float32')}
    sizes = {'train': 40, 'validation': 15, 'calibration': 15, 'test': 15} if small else {
        'train': 800, 'validation': 160, 'calibration': 160, 'test': 51}
    splits, noise = {}, {}
    for index, (name, count) in enumerate(sizes.items()):
        steps = 121 if small else 1001 if name == 'test' else 401
        print(f'V4生成 {name}: {count}条 × {steps}帧', flush=True)
        generated = generate_split(count, steps=steps, dt=.05, seed=seed+1000*index,
                                   id_offset=(index+1)*100000, initial_references=anchors)
        splits[name], noise[name] = add_observations(generated, seed+50000, .05)
        np.savez_compressed(root/f'{name}.npz', **splits[name])
    identifiers = {name: set(data['trajectory_id'].tolist()) for name, data in splits.items()}
    names = list(splits)
    assert all(not identifiers[a] & identifiers[b] for i, a in enumerate(names) for b in names[i+1:])
    metadata = {'seed': seed, 'dt': .05, 'split_counts': sizes, 'trajectory_ids_disjoint': True,
                'split_steps': {name: int(data['position'].shape[1]) for name, data in splits.items()},
                'anchor_source_split': 'original_nnet_train_only',
                'anchor_source_sha256': _file_hash(original_data/'nnet_train.npz'),
                'split_sha256': {name: _file_hash(root/f'{name}.npz') for name in splits},
                'noise': noise, 'original_data': str(original_data.resolve()),
                'all_observation_forecast_benchmark': True,
                'limitations': ['Generated maneuvers are simulation assumptions, not new measured flight records.',
                                'Forecast benchmark has all acquired observations; closed-loop tracking is evaluated separately.',
                                'Full experiment train/validation/calibration length20s and test length50s; small checks use6s; windows share the same horizons.',
                                'Input tail masking regularizes 0-4 missing latest observations; older filtered states remain causal.',
                                'Initial spatial anchors come only from original training positions; no original test flight enters training.']}
    write_json(root/'config.json', metadata)
    return splits, metadata


class ForecastWindows(Dataset):
    def __init__(self, data, config, stride=5, *, tail_dropout=False):
        self.data, self.config, self.tail_dropout = data, config, tail_dropout
        last = data['position'].shape[1]-max(config.horizons)
        self.ends = np.arange(config.history-1, last, stride, dtype=np.int64)
        if not len(self.ends):
            raise ValueError('Trajectories too short for forecast history/horizons')

    def __len__(self):
        return len(self.ends)*len(self.data['position'])

    def __getitem__(self, index):
        track, end_index = divmod(index, len(self.ends))
        end = self.ends[end_index]
        h = self.config.history
        valid = np.ones(h, bool)
        if self.tail_dropout:
            missing = np.random.choice(5, p=[.6, .16, .10, .08, .06])
            if missing:
                valid[-missing:] = False
        return (self.data['filtered_measurement'][track, end-h+1:end+1], valid,
                self.data['measurement_sigma'][track, end-h+1:end+1],
                self.data['position'][track, end+np.asarray(self.config.horizons)])


def _loader(data, config, batch, stride=5, train=False):
    return DataLoader(ForecastWindows(data, config, stride, tail_dropout=train),
                      batch_size=batch, shuffle=train, num_workers=0, drop_last=False)


def _run_epoch(model, loader, device, optimizer=None):
    from core.motion_uncertainty import gaussian_mixture_nll
    model.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for history, valid, sigma, target in loader:
            history, valid, sigma, target = [value.to(device) for value in (history, valid, sigma, target)]
            pred = model(history, valid, dt=.05, sigma=sigma)
            loss = gaussian_mixture_nll(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite forecast NLL')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            total += float(loss.detach())*len(history)
            count += len(history)
    return total/count


def forecast_dataset(model, data, device, stride=10, batch=512):
    config = model.config
    dataset = ForecastWindows(data, config, stride)
    collected, truth = {}, []
    model.eval()
    with torch.inference_mode():
        for history, valid, sigma, target in DataLoader(dataset, batch_size=batch, num_workers=0):
            pred = model(history.to(device), valid.to(device), dt=.05, sigma=sigma.to(device))
            for key, value in pred.items():
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(history):
                    collected.setdefault(key, []).append(value.cpu())
            truth.append(target)
    predictions = {key: torch.cat(value) for key, value in collected.items()}
    target = torch.cat(truth)
    track_indices = np.repeat(np.arange(len(data['position'])), len(dataset.ends))
    end_indices = np.tile(dataset.ends, len(data['position']))
    metadata = {'trajectory_id': data['trajectory_id'][track_indices], 'history_end': end_indices,
                'trajectory_kind': data['trajectory_kind'][track_indices],
                'motion_type': data['motion_type'][track_indices, end_indices],
                'future_maneuver': np.array([data['maneuver_onset'][track, end+1:end+max(config.horizons)+1].any()
                                            for track, end in zip(track_indices, end_indices)])}
    return predictions, target, metadata


def _scale_prediction(prediction, temperature):
    return {key: value*temperature if key == 'scale' else value for key, value in prediction.items()}


def _cv_prediction(prediction, horizons, dt=.05):
    """Past-only CV forecast; its constant spread is fitted once on validation."""
    # Model module exposes causal CV extrapolation as cv_base (fallback explicit origin/velocity).
    mean = prediction.get('base')
    if mean is None:
        mean = prediction.get('base_mean')
    if mean is None:
        origin, velocity = prediction['origin'], prediction['velocity']
        offsets = torch.as_tensor(horizons, dtype=origin.dtype)*dt
        mean = origin[:, None]+velocity[:, None]*offsets[None, :, None]
    if mean.ndim == 4:
        mean = mean[:, :, 0]
    return mean


def _cv_distribution(mean, standard_deviation):
    return {'weights': torch.ones((*mean.shape[:2], 1)),
            'log_weights': torch.zeros((*mean.shape[:2], 1)), 'mean': mean[:, :, None],
            'scale': standard_deviation[None, :, None, :].expand(len(mean), -1, -1, -1)}


def distribution_metrics(prediction, target, seed=37):
    """Proper scores plus explicitly marginal coordinate intervals; no joint coverage claim."""
    from core.motion_uncertainty import evaluate_distribution, mixture_marginal_intervals
    evaluated = evaluate_distribution(prediction, target, energy_samples=32, seed=seed)
    lower90, upper90 = mixture_marginal_intervals(prediction, levels=(.9,))[.9]
    truth = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    results = []
    for horizon, values in enumerate(evaluated['per_horizon']):
        row = {'horizon_index': horizon, 'samples': len(target), 'nll': values['nll_nats'],
               'rmse_m': values['mean_position_rmse_m'], 'energy_score_m': values['energy_score_m']}
        for level in (50, 80, 90, 95):
            row[f'marginal_coverage_{level}'] = values[f'marginal_coverage_{level}']
            row[f'marginal_width_{level}_m'] = values[f'marginal_width_{level}_m']
        for coordinate, label in enumerate(('x', 'y', 'z')):
            row[f'coverage_90_{label}'] = float(((truth[:, horizon, coordinate] >= lower90[:, horizon, coordinate]) &
                                               (truth[:, horizon, coordinate] <= upper90[:, horizon, coordinate])).mean())
            row[f'width_90_{label}_m'] = float((upper90[:, horizon, coordinate]-lower90[:, horizon, coordinate]).mean())
        results.append(row)
    return results


def train_and_evaluate(out, device, *, small=False, seed=20261121, epochs=12):
    """Fixed training budget; checkpoint chosen by validation NLL, scale fitted on calibration only."""
    from core.motion_uncertainty import MotionConfig, MotionUncertaintyNet, fit_scale_temperature, predictive_mean, predictive_std
    out = Path(out)
    if int(epochs) != epochs or epochs < 1:
        raise ValueError('Training requires a positive integer epoch budget')
    seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    splits, data_config = prepare_data(out, small=small, seed=seed)
    config = MotionConfig()
    model = MotionUncertaintyNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train = _loader(splits['train'], config, 512, train=True)
    validation = _loader(splits['validation'], config, 512, stride=10)
    rows, best, best_epoch = [], float('inf'), None
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        loss = _run_epoch(model, train, device, optimizer)
        val = _run_epoch(model, validation, device)
        rows.append({'epoch': epoch, 'training_nll': loss, 'validation_nll': val,
                     'learning_rate': optimizer.param_groups[0]['lr']})
        print(f'V4 epoch {epoch}/{epochs}: trainNLL={loss:.4f}, valNLL={val:.4f}', flush=True)
        if val < best:
            best, best_epoch = val, epoch
            torch.save({'format': 'v4_motion_uncertainty_v1', 'model_config': asdict(config),
                        'model': model.state_dict(), 'epoch': epoch, 'validation_nll': val,
                        'training_seed': seed, 'data_config': data_config}, out/'best_motion_forecaster.pth')
        scheduler.step()
    training_seconds = time.perf_counter()-started
    write_rows(out/'training_curve.csv', rows)
    checkpoint = torch.load(out/'best_motion_forecaster.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model']); model.eval().requires_grad_(False)
    val_prediction, val_truth, _ = forecast_dataset(model, splits['validation'], device)
    cv_val = _cv_prediction(val_prediction, config.horizons)
    cv_sigma = ((cv_val-val_truth)**2).mean(0).sqrt().clamp_min(.2)
    point_sigma = ((predictive_mean(val_prediction)-val_truth)**2).mean(0).sqrt().clamp_min(.2)
    cal_prediction, cal_truth, _ = forecast_dataset(model, splits['calibration'], device)
    temperature = float(fit_scale_temperature(cal_prediction, cal_truth))
    test_prediction, target, metadata = forecast_dataset(model, splits['test'], device)
    calibrated = _scale_prediction(test_prediction, temperature)
    cv = _cv_distribution(_cv_prediction(test_prediction, config.horizons), cv_sigma)
    point = _cv_distribution(predictive_mean(test_prediction), point_sigma)
    metrics, metric_rows = {}, []
    for name, pred in (('cv_constant_spread', cv), ('transformer_point_fixed_spread', point),
                       ('transformer_uncalibrated', test_prediction),
                       ('transformer_calibrated', calibrated)):
        metrics[name] = distribution_metrics(pred, target, seed=seed)
        for row in metrics[name]:
            metric_rows.append({'method': name, 'cohort': 'all', 'horizon_s': config.horizons[row['horizon_index']]*.05, **row})
        cohorts = [(f'type_{cohort}', metadata['trajectory_kind'] == cohort) for cohort in range(5)]
        cohorts += [('future_onset', metadata['future_maneuver']), ('no_future_onset', ~metadata['future_maneuver'])]
        for cohort, mask in cohorts:
            selected = torch.from_numpy(mask)
            if not selected.any():
                continue
            own = {key: value[selected] for key, value in pred.items()
                   if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(target)}
            for row in distribution_metrics(own, target[selected], seed=seed):
                metric_rows.append({'method': name, 'cohort': cohort,
                                    'horizon_s': config.horizons[row['horizon_index']]*.05, **row})
    write_rows(out/'forecast_metrics.csv', metric_rows)
    np.savez_compressed(out/'forecast_predictions.npz', **{
        key: calibrated[key].numpy() for key in ('weights', 'mean', 'scale')},
        target=target.numpy(), cv_mean=_cv_prediction(test_prediction, config.horizons).numpy(),
        horizons_s=np.asarray(config.horizons)*.05, **metadata)
    calibration = {'standard_deviation_temperature': temperature, 'fit_split': 'calibration',
                   'selection_split': 'validation', 'test_used_for_model_selection_or_calibration': False,
                   'cv_constant_sigma_m': cv_sigma.tolist(), 'point_fixed_scale_m': point_sigma.tolist()}
    write_json(out/'calibration.json', calibration)
    summary = {'status': 'forecast_training_and_evaluation_complete', 'seed': seed,
               'model_config': asdict(config), 'epochs_budget': epochs, 'best_epoch': best_epoch,
               'trained_checkpoint_sha256': _file_hash(out/'best_motion_forecaster.pth'),
               'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
               'best_validation_nll': best, 'training_seconds': training_seconds,
               'train_windows_per_epoch': len(train.dataset), 'test_windows': len(target),
               'calibration': calibration, 'forecast_results': metrics,
               'data_config': data_config,
               'limitations': ['Conditional/aleatoric uncertainty only; epistemic uncertainty is not estimated.',
                               'Outputs are separate marginal distributions at each horizon, not a coherent joint path law.',
                               'Coverage averages one-dimensional coordinate intervals, not simultaneous3Dcoverage.',
                               'Unknown future random event onsets and turn directions cannot be known from identical pasts.',
                               'Energy score uses32independent samples for each of twoMonteCarlosets.',
                               'Forecast benchmark uses all observed filtered measurements; tracking has method-dependent availability.']}
    summary['forecast_cohort_results'] = {
        method: {cohort: [row for row in metric_rows if row['method'] == method and row['cohort'] == cohort]
                 for cohort in sorted(set(row['cohort'] for row in metric_rows))}
        for method in metrics}
    summary['uncertainty_scope'] = 'conditional_measurement_and_motion_uncertainty_not_pure_maneuver_or_epistemic'
    summary['future_onset_cohort_definition'] = 'An onset occurs during the next1.0s; label is for evaluation only and does not enter prediction.'
    std = predictive_std(calibrated).numpy()
    summary['by_horizon_uncertainty'] = [
        {'horizon_s': float(horizon*.05), 'mean_predictive_std_xyz_m': std[:, i].mean(0).tolist()}
        for i, horizon in enumerate(config.horizons)]
    summary['energy_score_sampling_seed'] = seed
    write_json(out/'forecast_summary.json', summary)
    return {'model': model, 'summary': summary, 'data': splits, 'data_config': data_config,
            'temperature': temperature, 'point_fixed_scale': point_sigma,
            'checkpoint': out/'best_motion_forecaster.pth', 'out': out}


def load_bundle(out, device):
    """Read the completed checkpoint without retraining or selecting by test performance."""
    from core.motion_uncertainty import MotionConfig, MotionUncertaintyNet
    out = Path(out)
    checkpoint = torch.load(out/'best_motion_forecaster.pth', map_location=device, weights_only=False)
    model = MotionUncertaintyNet(MotionConfig(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model']); model.eval().requires_grad_(False)
    splits = {}
    for name in ('train', 'validation', 'calibration', 'test'):
        with np.load(out/'data'/f'{name}.npz') as data:
            splits[name] = {key: data[key].copy() for key in data.files}
    summary = read_json(out/'forecast_summary.json')
    return {'model': model, 'summary': summary, 'data': splits,
            'data_config': read_json(out/'data/config.json'),
            'temperature': summary['calibration']['standard_deviation_temperature'],
            'point_fixed_scale': torch.tensor(summary['calibration']['point_fixed_scale_m']),
            'checkpoint': out/'best_motion_forecaster.pth', 'out': out}


def extend_saved_evaluation(out):
    """Additional predeclared strata/coordinate diagnostics; changes no learned parameters."""
    from core.motion_uncertainty import predictive_mean, predictive_std
    out = Path(out)
    summary = read_json(out/'forecast_summary.json')
    with np.load(out/'forecast_predictions.npz') as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    temperature = summary['calibration']['standard_deviation_temperature']
    calibrated = {key: torch.from_numpy(arrays[key]) for key in ('weights', 'mean', 'scale')}
    target = torch.from_numpy(arrays['target'])
    if 'cv_mean' not in arrays:
        # Existing completed runs saved only the calibrated distribution. Obtain
        # the exact causal baseline from the same frozen model, without training.
        bundle = load_bundle(out, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        prediction, reproduced_truth, _ = forecast_dataset(bundle['model'], bundle['data']['test'],
                                                          next(bundle['model'].parameters()).device)
        torch.testing.assert_close(reproduced_truth, target)
        arrays['cv_mean'] = _cv_prediction(prediction, bundle['model'].config.horizons).numpy()
    arrays['horizons_s'] = np.asarray(summary['model_config']['horizons'])*.05
    np.savez_compressed(out/'forecast_predictions.npz', **arrays)
    methods = {'cv_constant_spread': _cv_distribution(torch.from_numpy(arrays['cv_mean']),
                  torch.tensor(summary['calibration']['cv_constant_sigma_m'])),
               'transformer_point_fixed_spread': _cv_distribution(predictive_mean(calibrated),
                  torch.tensor(summary['calibration']['point_fixed_scale_m'])),
               'transformer_uncalibrated': _scale_prediction(calibrated, 1/temperature),
               'transformer_calibrated': calibrated}
    all_rows, results, cohorts_results = [], {}, {}
    cohorts = [('all', np.ones(len(target), bool))]
    cohorts += [(f'type_{cohort}', arrays['trajectory_kind'] == cohort) for cohort in range(5)]
    cohorts += [('future_onset', arrays['future_maneuver']), ('no_future_onset', ~arrays['future_maneuver'])]
    for method, pred in methods.items():
        cohorts_results[method] = {}
        for cohort, mask in cohorts:
            selected = torch.from_numpy(mask)
            if not selected.any():
                continue
            own = {key: value[selected] for key, value in pred.items()}
            values = distribution_metrics(own, target[selected], seed=summary['seed'])
            rows = [{'method': method, 'cohort': cohort,
                     'horizon_s': arrays['horizons_s'][row['horizon_index']], **row} for row in values]
            all_rows.extend(rows); cohorts_results[method][cohort] = rows
            if cohort == 'all':
                results[method] = values
    write_rows(out/'forecast_metrics.csv', all_rows)
    summary['forecast_results'] = results
    summary['forecast_cohort_results'] = cohorts_results
    summary['uncertainty_scope'] = 'conditional_measurement_and_motion_uncertainty_not_pure_maneuver_or_epistemic'
    summary['future_onset_cohort_definition'] = 'An onset occurs during next1.0s; evaluation label only, never a network input.'
    summary['energy_score_sampling_seed'] = summary['seed']
    std = predictive_std(calibrated).numpy()
    summary['by_horizon_uncertainty'] = [
        {'horizon_s': float(horizon), 'mean_predictive_std_xyz_m': std[:, i].mean(0).tolist()}
        for i, horizon in enumerate(arrays['horizons_s'])]
    write_json(out/'forecast_summary.json', summary)
    print(f'V4追加逐坐标与突发机动分组评价：{len(all_rows)}行；模型未修改。', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_v4_maneuver')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261121)
    args = parser.parse_args(argv)
    out = new_run(args.output_root)
    train_and_evaluate(out, torch.device(args.device), small=args.small, seed=args.seed, epochs=args.epochs)
    print(out.resolve(), flush=True)
    return out


if __name__ == '__main__':
    main()
