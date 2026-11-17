"""Real GNSS reference + distance/SNR-dependent errors/outliers, original N-Net/R-Net."""
import argparse
import hashlib
import shutil
from pathlib import Path

from .common import BASE, new_run, read_json, write_json
from .maneuver_nnet_rnet import ORIGINAL_MODELS_SHA256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--plot-trajectory-id', type=int)
    parser.add_argument('--noise-seed', type=int, default=20261116)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--archive-prefix')
    parser.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_real_snr_outliers')
    args = parser.parse_args(argv)
    current = read_json(BASE/'outputs/current_journal_experiment.json')
    data_root = args.data or Path(current['data'])
    config = read_json(data_root/'config.json')
    if 'real_GNSS' not in config.get('dataset_kind', ''):
        parser.error('必须提供真实GNSS参考轨迹分区，不能使用生成机动轨迹')
    model_source = Path(__file__).with_name('models.py')
    if sha(model_source) != ORIGINAL_MODELS_SHA256:
        raise RuntimeError('原N-Net/R-Net定义被修改，停止实验')
    original_hashes = {k: sha(current[k]) for k in ('nnet_checkpoint', 'rnet_checkpoint')}
    root = new_run(args.output_root)
    prefix = args.archive_prefix or 'real_'+root.name
    for suffix in ('raw', 'robust'):
        if (BASE/'experiments'/f'{prefix}_{suffix}').exists():
            parser.error('归档版本已存在，不能覆盖旧结果')
    backend = None
    if not args.no_show:
        import matplotlib.pyplot as plt
        backend = plt.get_backend()
    from .multi_target import main as run_tracking
    runs = {}
    for mode in ('raw', 'robust'):
        parameters = ['--measurement-regime', 'stress', '--quality-mode', mode,
            '--noise-seed', str(args.noise_seed), '--data', str(data_root),
            '--device', args.device, '--no-show', '--output-root', str(root/mode),
            '--archive-version', f'{prefix}_{mode}', '--targets', '3', '--total-beams', '12']
        if args.plot_trajectory_id is not None:
            parameters += ['--plot-trajectory-id', str(args.plot_trajectory_id)]
        print(f'真实轨迹重新仿真：{mode}', flush=True)
        runs[mode] = run_tracking(parameters)
    summaries = {mode: read_json(path/'summary.json') for mode, path in runs.items()}
    for key in ('scene_membership', 'measurement_realization_sha256', 'source_sha256'):
        if summaries['raw'][key] != summaries['robust'][key]:
            raise RuntimeError(f'两组对照条件不一致：{key}')
    if sha(model_source) != ORIGINAL_MODELS_SHA256 or original_hashes != {k: sha(current[k]) for k in original_hashes}:
        raise RuntimeError('N-Net/R-Net定义或权重文件发生变化')
    from .metric_charts import compare_versions
    comparison = root/'comparison'
    compare_versions({'扰动直接输入': runs['raw'], '相同扰动＋鲁棒预处理': runs['robust']}, comparison)
    pair = {'runs': {k: str(v.resolve()) for k, v in runs.items()},
        'versions': {k: f'{prefix}_{k}' for k in runs},
        'models_source_sha256': ORIGINAL_MODELS_SHA256, 'checkpoint_sha256': original_hashes,
        'same_truth_and_measurements': True, 'networks_and_weights_unchanged': True,
        'measurement_realization_sha256': summaries['raw']['measurement_realization_sha256'],
        'results': {k: v['results']['joint'] for k, v in summaries.items()},
        'real_reference_audit': summaries['raw']['real_reference_audit'],
        'test_is_new_blind_set': summaries['raw']['test_is_new_blind_set'],
        'primary_position_curve': 'original N-Net output on acquired frames',
        'GNSS_truth_is_reference_not_independent_high_precision_truth': True}
    write_json(root/'batch_summary.json', pair)
    write_json(comparison/'paired_experiment.json', pair)
    # Append separately locked comparison figures; immutable source/results stay untouched.
    from tools.verify_versions import lock_figures
    for mode in runs:
        archive = BASE/'experiments'/f'{prefix}_{mode}'
        shutil.copytree(comparison, archive/'figures')
        lock_figures(archive)
    (args.output_root/'latest_complete_run.txt').write_text(root.name, encoding='utf-8')
    print(f'真实轨迹结果：{root.resolve()}', flush=True)
    for mode, s in pair['results'].items():
        print(mode, {k: s[k] for k in ('success_rate', 'mean_all_targets_duration_s',
            'recoveries', 'lost_events', 'recovery_rate', 'position_rmse_m_on_detections')}, flush=True)
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.switch_backend(backend)
        for path in (runs['robust']/'figures/single_trajectory_tracking.png', comparison/'metrics_comparison.png'):
            fig, ax = plt.subplots(figsize=(12, 8)); ax.imshow(plt.imread(path)); ax.axis('off')
        plt.show()
    return root


if __name__ == '__main__':
    main()
