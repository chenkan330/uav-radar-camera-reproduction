"""V4 complete experiment: maneuver forecasts, calibration and shared-beam tracking."""
import argparse
import hashlib
import shutil
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .common import BASE, load_nnet, load_split, new_run, read_json, state_digest, write_json, write_rows
from .maneuver_tracking import MotionBeamPredictor
from .models import JournalRNet
from .motion_uncertainty import MotionConfig, MotionUncertaintyNet, evaluate_distribution
from .multi_target import make_scenes, simulate, summarize, summarize_measurement_updates


def closed_loop_scores(hook, events, scenes, history=5):
    """Score already-produced forecasts; future truth is never passed to the predictor."""
    if not hook.records:
        return None
    horizon = np.asarray(hook.model.config.horizons)
    predictions, targets = {k: [] for k in ('weights', 'mean', 'scale')}, []
    for offset, record in enumerate(hook.records):
        step = history+offset
        if step-1+max(horizon) >= scenes['position'].shape[-2]:
            continue
        live = sorted({row['scene_id'] for row in events if row['time_index'] == step})
        expected = len(live)*scenes['beam'].shape[1]
        if record['mean'].shape[0] != expected:
            raise ValueError('Closed-loop forecast batch does not match active scene membership')
        for key in predictions:
            predictions[key].append(record[key])
        targets.append(scenes['position'][np.asarray(live), :, :][:, :, step-1+horizon].reshape(expected, len(horizon), 3))
    pred = {k: torch.from_numpy(np.concatenate(v)) for k, v in predictions.items()}
    target = torch.from_numpy(np.concatenate(targets))
    scores = evaluate_distribution(pred, target, energy_samples=32, seed=99)
    scores['scope'] = 'all active target forecasts from acquired-history-only tracking; method-dependent exposure'
    scores['forecast_count'] = len(target)
    scores['horizons_s'] = (horizon*.05).tolist()
    return scores


def finish_run(out, device):
    out = Path(out)
    if (out/'tracking_metrics.json').exists():
        raise FileExistsError('已完成的V4结果不覆盖，请创建新运行目录')
    torch.set_num_threads(4)
    current = read_json(BASE/'outputs/current_journal_experiment.json')
    nnet, ck = load_nnet(current['nnet_checkpoint'], device)
    saved = torch.load(current['rnet_checkpoint'], map_location=device, weights_only=False)
    rnet = JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(saved['model'])
    checkpoint = torch.load(out/'best_motion_forecaster.pth', map_location=device, weights_only=False)
    model = MotionUncertaintyNet(MotionConfig(**checkpoint['model_config'])).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    calibration = read_json(out/'calibration.json')
    data = load_split(out/'data/test.npz')
    scenes, indices, unused = make_scenes(data, 3, seed=20261115)
    # Only raw external measurements and quality metadata enter this tracking loop.
    # The all-observation filtered stream is reserved for the separate forecast benchmark.
    for key in ('measurement_sigma', 'snr_db', 'outlier_mask'):
        scenes[key] = data[key][indices]
    before = state_digest(nnet), state_digest(rnet), state_digest(model)
    methods, loop_scores = {}, {}
    hooks = {'legacy_v3': None,
             'motion_point': MotionBeamPredictor(model, device, mode='point',
                 fixed_scale=calibration['point_fixed_scale_m'], record=True),
             'motion_probability': MotionBeamPredictor(model, device,
                 temperature=calibration['standard_deviation_temperature'], record=True)}
    for name, hook in hooks.items():
        print(f'V4闭环跟踪：{name}', flush=True)
        updates, timing = [], {}
        result = simulate(nnet, rnet, ck, scenes, device, .05, 12, 'joint', quality_mode='robust',
                          measurement_updates=updates, filter_timing=timing, beam_predictor=hook)
        summary = summarize(result[0], result[1], result[4], .05, 12, 3, 'joint')
        summary.update(summarize_measurement_updates(updates, timing))
        summary['current_filtered_position_rmse_m_on_detections'] = summary['measurement_processed_rmse_m_on_detections']
        summary['position_rmse_scope'] = 'legacy current-position head retained; filtered current-state RMSE reported separately'
        methods[name] = summary
        for suffix, rows in zip(('scenes', 'targets', 'events', 'scans'), result[:4]):
            write_rows(out/f'{name}_{suffix}.csv', rows)
        write_rows(out/f'{name}_measurement_updates.csv', updates)
        if hook is not None:
            loop_scores[name] = closed_loop_scores(hook, result[2], scenes)
        print({k: summary[k] for k in ('success_rate', 'mean_all_targets_duration_s',
                                     'current_filtered_position_rmse_m_on_detections')}, flush=True)
    if before != (state_digest(nnet), state_digest(rnet), state_digest(model)):
        raise RuntimeError('Inference modified model weights')
    record = {'methods': methods, 'targets': 3, 'total_beams': 12,
              'maximum_evaluated_duration_s': 49.8, 'maximum_duration_s': 49.8,
              'scene_membership': scenes['trajectory_id'].tolist(), 'unused_trajectory_ids': unused,
              'same_external_measurement_stream': True,
              'measurement_sha256': hashlib.sha256(scenes['measurement'].tobytes()).hexdigest(),
              'position_metric': 'filtered current-state estimate on detections only',
              'probability_projection': {'samples_per_component': 256, 'nodes': 'fixed Sobol Gaussian nodes',
                  'tiny_spatial_tie_breaker_mass': 1e-6, 'no_future_truth_or_motion_labels_in_predictor': True},
              'legacy_initial_acquisitions': 5, 'motion_initial_acquisitions': 5,
              'motion_history': 20, 'left_padding_is_invalid': True,
              'point_probability_comparison': 'same learned forecast means; fixed spread vs full calibrated mixture',
              'weights_unchanged': True}
    write_json(out/'tracking_metrics.json', record)
    write_json(out/'closed_loop_forecast_metrics.json', loop_scores)
    forecast = read_json(out/'forecast_summary.json')
    summary = {'version': 'v4_maneuver_uncertainty',
               'source': str((out/'data/test.npz').resolve()),
               'nnet_checkpoint': current['nnet_checkpoint'], 'rnet_checkpoint': current['rnet_checkpoint'],
               'motion_checkpoint': str((out/'best_motion_forecaster.pth').resolve()),
               'motion_checkpoint_sha256': hashlib.sha256((out/'best_motion_forecaster.pth').read_bytes()).hexdigest(),
               'tracking': record, 'forecast': forecast,
               'closed_loop_forecast': loop_scores,
               'fully_reproduced_paper_numerical_results': False,
               'limitations': ['V4 uses new simulated maneuver trajectories; old V3 percentages are not a matched baseline.',
                   'Motion labels and future event times are evaluation-only.',
                   'Conditional uncertainty includes measurement error and motion ambiguity, not epistemic uncertainty.',
                   'Each horizon has its own mixture; interval coverage is marginal, not joint path coverage.',
                   '17 scene groups and one simulated seed do not establish general performance guarantees.',
                   'New motion methods replace old N/R beam prediction; the legacy method remains a separate comparison.',
                   'Forecast benchmark sees all observations; closed-loop metrics have method-dependent availability.']}
    write_json(out/'summary.json', summary)
    from .maneuver_charts import render_v4
    render_v4(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-run', type=Path)
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--archive-version')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_v4_maneuver')
    args = parser.parse_args(argv)
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    else:
        import matplotlib.pyplot as plt
    device = torch.device(args.device)
    if args.resume_run:
        out = args.resume_run
    else:
        out = new_run(args.output_root)
        pointer = args.output_root/'latest_complete_run.txt'
        if not args.retrain and pointer.exists():
            prior = args.output_root/pointer.read_text(encoding='utf-8').strip()
            for name in ('data', 'best_motion_forecaster.pth', 'calibration.json', 'forecast_summary.json',
                         'forecast_metrics.csv', 'forecast_predictions.npz', 'training_curve.csv'):
                source = prior/name
                shutil.copytree(source, out/name) if source.is_dir() else shutil.copy2(source, out/name)
        else:
            from .workflows.step10e_maneuver_uncertainty import train_and_evaluate
            train_and_evaluate(out, device)
    finish_run(out, device)
    from .versions import save_version
    version = args.archive_version or ('v4_'+out.name)
    archived = save_version(version, out, '加速/急转弯/突发/混合机动；Transformer三分量未来位置分布与独立概率校准。')
    (args.output_root/'latest_complete_run.txt').write_text(out.name, encoding='utf-8')
    print(f'V4结果：{out.resolve()}\n版本快照：{archived.resolve()}', flush=True)
    if not args.no_show:
        import matplotlib.pyplot as plt
        for filename in ('forecast_scores.png', 'maneuver_tracking_comparison.png'):
            path = out/'figures'/filename
            if path.exists():
                fig, ax = plt.subplots(figsize=(12, 7)); ax.imshow(plt.imread(path)); ax.axis('off')
        plt.show()
    return out


if __name__ == '__main__':
    main()
