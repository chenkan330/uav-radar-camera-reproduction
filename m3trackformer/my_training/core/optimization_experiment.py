"""Validation-selected external algorithms; no N-Net/R-Net modification or retraining."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .common import BASE, load_nnet, load_split, new_run, read_json, state_digest, write_json, write_rows
from .maneuver_nnet_rnet import ORIGINAL_MODELS_SHA256, run_original_nr
from .measurement_quality import NoiseConfig, perturb_scenes
from .models import JournalRNet
from .multi_target import make_scenes, summarize
from .optimization_processors import candidates, make_processor
from .trajectory_figure import save_trajectory_figure
from .versions import save_version

VALIDATION_SEEDS = [20261116, 20261118]
TEST_SEEDS = [20261116, 20261118, 20261119]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tracking_choice(base=BASE):
    """A user-selected operating baseline takes precedence over sweep results."""
    active = base/'outputs/current_tracking_configuration.json'
    choice = read_json(active if active.exists() else base/'outputs/current_optimized_tracking.json')
    user_choice = choice.get('selection_basis') == 'user_requested_baseline'
    path = (Path(choice['selection_record']) if user_choice else
            Path(choice['experiment_root'])/'selection_locked_before_test.json')
    lock = read_json(path)
    if (sha(path) != choice['selection_lock_sha256']
            or lock['configuration'] != choice['configuration']
            or (user_choice and (lock.get('selection_basis') != 'user_requested_baseline'
                                or choice.get('selected_on_validation') is not False))):
        raise RuntimeError('当前运行配置与原始选择记录不一致')
    return choice, path


def aggregate(records, maximum_blocks):
    scenes = [r for record in records for r in record['scene_rows']]
    targets = [r for record in records for r in record['target_rows']]
    rec, lost = sum(t['recoveries'] for t in targets), sum(t['lost_events'] for t in targets)
    n = sum(t['position_count'] for t in targets)
    potential = len(scenes)*3*maximum_blocks
    return {'scenes_with_noise_repeats': len(scenes), 'unique_geometrical_scenes': len(records[0]['scene_rows']),
        'success_count': sum(s['success'] for s in scenes),
        'success_rate': float(np.mean([s['success'] for s in scenes])),
        'mean_all_targets_duration_s': float(np.mean([s['duration_s'] for s in scenes])),
        'recoveries': rec, 'lost_events': lost, 'recovery_rate': rec/lost if lost else None,
        'position_rmse_m_on_detections': (sum(t['position_squared_error_sum'] for t in targets)/n)**.5 if n else None,
        'position_evaluation_count': n, 'position_evaluation_coverage': n/potential,
        'filter_ms_per_detected_frame': sum(r['filter_timing'].get('total_ms', 0) for r in records)/n if n else None}


def selection_key(record):
    m = record['aggregate']
    return (m['success_rate'], m['mean_all_targets_duration_s'], m['position_evaluation_coverage'])


def prepare_scenes(data_root, split, seeds):
    from .real_data_audit import audit_real_reference
    data = load_split(data_root/f'{split}.npz')
    scenes, _, unused = make_scenes(data, 3, seed=20261115)
    variants = {}
    for seed in seeds:
        noisy, metadata = perturb_scenes(scenes, NoiseConfig(), seed)
        variants[seed] = (noisy, metadata)
    return variants, unused


def trial(root, prefix, stage, configuration, scene_variants, unused, current, nnet, rnet, checkpoint, device,
          selection_provenance=None):
    out = root/stage/configuration['id']
    out.mkdir(parents=True, exist_ok=False)
    data_root = Path(current['data'])
    from .real_data_audit import audit_real_reference
    # The split's raw-CVS audit was already done once; preserve it in this trial.
    import shutil
    for name in ('real_trajectory_sources.csv', 'real_trajectory_audit.json'):
        shutil.copy2(root/(stage+'_reference')/name, out/name)
    records, primary = [], None
    started = time.perf_counter()
    for i, (seed, (scenes, noise_meta)) in enumerate(scene_variants.items()):
        plot_id = int(scenes['trajectory_id'][0, 0])
        factory = None if configuration['processor'] is None else partial(make_processor,
            configuration=configuration['processor'])
        result, trace, _, timing, _, calls = run_original_nr(nnet, rnet, checkpoint, scenes, device,
            quality_mode='raw' if factory is None else 'robust', dt=.05, budget=12, plot_id=plot_id,
            measurement_processor_factory=factory, logit_temperature=configuration['temperature'],
            record_events=False, collect_probability=False, collect_updates=False)
        seed_record = {'noise_seed': seed, 'scene_rows': result[0], 'target_rows': result[1],
            'filter_timing': timing, 'network_calls': calls,
            'measurement_sha256': hashlib.sha256(scenes['measurement'].tobytes()).hexdigest()}
        records.append(seed_record)
        write_rows(out/f'scenes_{seed}.csv', result[0])
        write_rows(out/f'targets_{seed}.csv', result[1])
        if i == 0:
            primary = (scenes, trace, result, plot_id, seed)
        print(f'{stage} {configuration["id"]} seed={seed}: '
            f'{sum(s["success"] for s in result[0])}/{len(result[0])} scenes', flush=True)
    scenes, trace, result, plot_id, seed = primary
    maximum_blocks = scenes['beam'].shape[-1]-nnet.config.history
    total = aggregate(records, maximum_blocks)
    primary_summary = summarize(result[0], result[1], result[4], .05, 12, 3, 'joint')
    record = {'version': 'external_optimization_trial', 'phase': stage,
        'configuration': configuration, 'aggregate': total, 'noise_repeats': records,
        'source': str((data_root/f'{stage}.npz').resolve()),
        'nnet_checkpoint': current['nnet_checkpoint'], 'rnet_checkpoint': current['rnet_checkpoint'],
        'trajectory_figure': 'figures/single_trajectory_tracking.png',
        'models_source_sha256': ORIGINAL_MODELS_SHA256, 'networks_and_weights_unchanged': True,
        'same_geometries_across_methods': True, 'scene_membership': scenes['trajectory_id'].tolist(),
        'unused_trajectory_ids': unused, 'primary_plot_noise_seed': seed,
        'position_metric': 'original N-Net head on actual detections; methods have different coverage',
        'test_records_used_previously': True, 'wall_time_s': time.perf_counter()-started,
        'total_beams': 12, 'targets': 3, 'dt': .05,
        'maximum_evaluated_duration_s': maximum_blocks*.05,
        'alternative_prediction_network_used': False,
        'external_temperature_does_not_change_per_target_ranking': True,
        'temperature_objective': 'closed-loop validation selection, not test calibration'}
    save_trajectory_figure(out, scenes, trace, result[1], primary_summary, dt=.05,
        history=nnet.config.history, trajectory_id=plot_id,
        title='原 N-Net/R-Net：'+configuration['label'],
        selection_note=f'固定首场景首目标；本图仅展示噪声种子{seed}，汇总图包含全部固定种子。')
    from .optimization_charts import render_trial
    render_trial(out, f'{stage}：{configuration["label"]}（{configuration["id"]}）', total)
    if selection_provenance is not None:
        record['selection_provenance'] = selection_provenance
        write_json(out/'operating_configuration.json', selection_provenance)
        import shutil
        shutil.copy2(selection_provenance['selection_record'], out/'operating_choice_lock.json')
    write_json(out/'summary.json', record)
    version = f'{prefix}_{stage}_{configuration["id"]}'
    description = ('用户指定当前Huber基准；不改写历史验证集选择；原N/R及真实GPS来源保持。'
        if selection_provenance and selection_provenance.get('selection_basis') == 'user_requested_baseline'
        else '原N/R外部算法比较；验证集选参、测试前锁定；实际N-Net轨迹与真实GPS来源保留。')
    saved = save_version(version, out, description)
    return record, saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_optimization')
    parser.add_argument('--apply-selected', action='store_true')
    args = parser.parse_args(argv)
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    else:
        # Initialize the interactive backend before export helpers choose Agg.
        import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    current = read_json(BASE/'outputs/current_journal_experiment.json')
    data_root = Path(current['data'])
    if 'real_GNSS' not in read_json(data_root/'config.json').get('dataset_kind', ''):
        raise ValueError('优化比较只使用真实GNSS轨迹')
    model_source = Path(__file__).with_name('models.py')
    if sha(model_source) != ORIGINAL_MODELS_SHA256:
        raise RuntimeError('原N-Net/R-Net定义发生变化')
    weight_hashes = {k: sha(current[k]) for k in ('nnet_checkpoint', 'rnet_checkpoint')}
    device = torch.device(args.device)
    nnet, checkpoint = load_nnet(current['nnet_checkpoint'], device)
    saved = torch.load(current['rnet_checkpoint'], map_location=device, weights_only=False)
    rnet = JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(saved['model'])
    before = state_digest(nnet), state_digest(rnet)
    root = new_run(args.output_root)
    prefix = 'opt_'+root.name.removeprefix('run_').removesuffix('_UTC')
    from .real_data_audit import audit_real_reference
    if args.apply_selected:
        choice, selection_path = load_tracking_choice()
        if (choice['checkpoint_sha256'] != weight_hashes
                or choice['models_source_sha256'] != sha(model_source)
                or choice['selection_lock_sha256'] != sha(selection_path)):
            raise RuntimeError('已选配置、选参锁定文件或原N/R权重不匹配')
        audit_real_reference(data_root, 'test', root/'test_reference')
        scenes, unused = prepare_scenes(data_root, 'test', [TEST_SEEDS[0]])
        record, archive = trial(root, prefix, 'test', choice['configuration'], scenes, unused,
                                 current, nnet, rnet, checkpoint, device,
                                 selection_provenance={**choice, 'selection_record': str(selection_path.resolve())})
        print(f'按当前固定配置 {choice["configuration"]["id"]} 运行：{archive.resolve()}', flush=True)
        if (before != (state_digest(nnet), state_digest(rnet))
                or sha(model_source) != ORIGINAL_MODELS_SHA256
                or weight_hashes != {k: sha(current[k]) for k in weight_hashes}):
            raise RuntimeError('推理改变了原N/R结构或权重')
        (args.output_root/'latest_selected_run.txt').write_text(root.name, encoding='utf-8')
        if not args.no_show:
            from .optimization_charts import show_images
            show_images([archive/'results/figures/single_trajectory_tracking.png',
                         archive/'results/figures/trial_metrics.png'])
        return root
    protocol = {'candidates': candidates(), 'validation_seeds': VALIDATION_SEEDS, 'test_seeds': TEST_SEEDS,
        'ranking_order': ['validation_complete_scene_success_rate', 'validation_mean_tracking_duration',
                          'validation_detection_coverage'],
        'no_test_tuning': True, 'networks_frozen': True, 'models_source_sha256': ORIGINAL_MODELS_SHA256,
        'checkpoint_sha256': weight_hashes, 'same_scene_seed': 20261115, 'budget': 12,
        'test_geometry_groups': 17, 'test_records_previously_used': True,
        'method_notes': ['Student-t weights and robust IMM likelihood are declared approximations.',
            'External temperature changes confidence before the original probability-sum Top-B rule.',
            'No new predictor replaces N-Net or R-Net; position curve always uses the original N-Net head.']}
    write_json(root/'protocol.json', protocol)
    audit_real_reference(data_root, 'validation', root/'validation_reference')
    validation_scenes, unused_val = prepare_scenes(data_root, 'validation', VALIDATION_SEEDS)
    validation = []
    for configuration in protocol['candidates']:
        record, archive = trial(root, prefix, 'validation', configuration, validation_scenes,
                                unused_val, current, nnet, rnet, checkpoint, device)
        validation.append(record)
        write_json(root/'validation_progress.json', [{'configuration': r['configuration'],
            'aggregate': r['aggregate']} for r in validation])
    winner = max(validation, key=selection_key)
    families = {}
    for record in validation:
        family = record['configuration']['family']
        if family not in families or selection_key(record) > selection_key(families[family]):
            families[family] = record
    chosen = {r['configuration']['id']: r['configuration'] for r in families.values()}
    chosen['huber_base'] = next(c for c in candidates() if c['id'] == 'huber_base')
    selection = {'configuration': winner['configuration'], 'validation_aggregate': winner['aggregate'],
        'test_finalists': list(chosen.values()), 'selected_before_test': True,
        'protocol_sha256': sha(root/'protocol.json'), 'ranking_order': protocol['ranking_order']}
    write_json(root/'selection_locked_before_test.json', selection)
    selection_hash = sha(root/'selection_locked_before_test.json')
    print('验证集锁定最优方案：'+winner['configuration']['id'], flush=True)
    # Only after selection is locked do we load and simulate the held-out split.
    audit_real_reference(data_root, 'test', root/'test_reference')
    test_scenes, unused_test = prepare_scenes(data_root, 'test', TEST_SEEDS)
    tests, archives = [], {}
    for configuration in chosen.values():
        record, archive = trial(root, prefix, 'test', configuration, test_scenes, unused_test,
                                current, nnet, rnet, checkpoint, device)
        tests.append(record); archives[configuration['id']] = str(archive.resolve())
        write_json(root/'test_progress.json', [{'configuration': r['configuration'],
            'aggregate': r['aggregate']} for r in tests])
    if sha(root/'selection_locked_before_test.json') != selection_hash:
        raise RuntimeError('测试阶段改变了验证集选择')
    if (before != (state_digest(nnet), state_digest(rnet)) or sha(model_source) != ORIGINAL_MODELS_SHA256
            or weight_hashes != {k: sha(current[k]) for k in weight_hashes}):
        raise RuntimeError('原N/R结构或权重发生变化')
    from .optimization_charts import render_comparison, write_report
    render_comparison(root, tests, winner['configuration']['id'])
    write_report(root, validation, tests, winner['configuration']['id'])
    flat = []
    for record in tests:
        flat.append({'id': record['configuration']['id'], 'method': record['configuration']['label'],
            'validation_selected': record['configuration']['id'] == winner['configuration']['id'],
            **record['aggregate']})
    write_rows(root/'test_comparison.csv', flat)
    summary = {'protocol': protocol, 'selection': selection, 'test': flat,
        'test_version_archives': archives, 'networks_and_weights_unchanged': True,
        'test_source': str((data_root/'test.npz').resolve()),
        'limitation': 'Same 17 geometry groups with three noise repeats; records are not a new blind dataset.'}
    write_json(root/'summary.json', summary)
    selected = next(r for r in tests if r['configuration']['id'] == winner['configuration']['id'])
    pointer = {'configuration': winner['configuration'], 'selected_on_validation': True,
        'selection_lock_sha256': selection_hash, 'experiment_root': str(root.resolve()),
        'version_archive': archives[winner['configuration']['id']], 'test_aggregate': selected['aggregate'],
        'models_source_sha256': ORIGINAL_MODELS_SHA256, 'checkpoint_sha256': weight_hashes}
    write_json(BASE/'outputs/current_optimized_tracking.json', pointer)
    from tools.verify_versions import lock_figures
    import shutil
    for archive_path in archives.values():
        destination = Path(archive_path)/'figures'
        destination.mkdir()
        for name in ('optimization_comparison.png', 'test_comparison.csv', 'summary.json',
                     'selection_locked_before_test.json', 'protocol.json', 'results_report.md'):
            shutil.copy2(root/name, destination/name)
        lock_figures(archive_path)
    (args.output_root/'latest_complete_run.txt').write_text(root.name, encoding='utf-8')
    print(f'比较完成：{root.resolve()}\n固定选择：{winner["configuration"]["id"]}', flush=True)
    print(flat, flush=True)
    if not args.no_show:
        from .optimization_charts import show_images
        show_images([Path(pointer['version_archive'])/'results/figures/single_trajectory_tracking.png',
                     root/'optimization_comparison.png'])
    return root


if __name__ == '__main__':
    main()
