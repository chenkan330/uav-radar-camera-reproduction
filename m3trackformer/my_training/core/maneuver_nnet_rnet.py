"""Revised V4: maneuver tests with the original, unchanged Transformer N-Net/R-Net."""
import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch

from .common import BASE, load_nnet, load_split, new_run, read_json, state_digest, write_json, write_rows
from .models import JournalRNet
from .multi_target import make_scenes, simulate, summarize, summarize_measurement_updates
from .trajectory_figure import V4_REFERENCE_ID, save_trajectory_figure

ORIGINAL_MODELS_SHA256 = '43972fb03a96a01c94a914e763440943933ca28f024bbb5156e0e164d7ba925f'
METHODS = {'nr_raw': '原 N/R：直接测量', 'nr_robust': '原 N/R：鲁棒预处理'}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_maneuver_test():
    root = BASE/'outputs/step10_v4_maneuver'
    pointer = root/'latest_complete_run.txt'
    if not pointer.exists():
        raise FileNotFoundError('未找到已有V4机动测试数据，请用--data指定test.npz')
    return root/pointer.read_text(encoding='utf-8').strip()/'data/test.npz'


def run_original_nr(nnet, rnet, checkpoint, scenes, device, *, quality_mode, dt, budget, plot_id,
                    measurement_processor_factory=None, logit_temperature=1.,
                    record_events=True, collect_probability=True, collect_updates=True,
                    detection_observer=None):
    """Use the original routing; hooks count real calls without changing outputs."""
    audit = {'nnet_forward_calls': 0, 'nnet_target_frames': 0,
             'rnet_forward_calls': 0, 'rnet_target_frames': 0}

    def count_n(module, inputs):
        audit['nnet_forward_calls'] += 1
        audit['nnet_target_frames'] += len(inputs[0])

    def count_r(module, inputs):
        audit['rnet_forward_calls'] += 1
        audit['rnet_target_frames'] += len(inputs[1])

    hooks = [nnet.register_forward_pre_hook(count_n), rnet.register_forward_pre_hook(count_r)]
    trace, updates, timing, probability_rows = [], [], {}, []
    try:
        result = simulate(nnet, rnet, checkpoint, scenes, device, dt, budget, 'joint',
            quality_mode=quality_mode, measurement_updates=updates if collect_updates else None, filter_timing=timing,
            beam_predictor=None, trajectory_trace=trace, trace_trajectory_id=plot_id,
            trajectory_estimator='nnet', beam_probability_metrics=probability_rows if collect_probability else None,
            measurement_processor_factory=measurement_processor_factory, logit_temperature=logit_temperature,
            record_events=record_events, detection_observer=detection_observer)
    finally:
        for hook in hooks:
            hook.remove()
    expected_r = sum(r['missed_blocks']-int(r['lost_events'] > r['recoveries']) for r in result[1])
    if audit['rnet_target_frames'] != expected_r:
        raise RuntimeError('实际R-Net调用与失跟目标数不一致')
    audit.update(expected_rnet_target_frames=expected_r, original_nr_routing_verified=True,
                 alternative_prediction_network_used=False,
                 trajectory_estimator='nnet_position_head')
    return result, trace, updates, timing, probability_rows, audit


def probability_summary(rows):
    """Class probabilities refer to the next beam, not a 3-D position distribution."""
    if not rows:
        return {'count': 0, 'scope': 'no evaluated predictions'}
    result = {'count': len(rows), 'scope': 'next-block beam class on active target frames only',
        'calibrated_on_test': False, 'is_3d_motion_uncertainty': False}
    for key in ('confidence', 'entropy_nats', 'nll_nats', 'brier_score', 'selected_probability_mass'):
        result['mean_'+key] = float(np.mean([r[key] for r in rows]))
    result['top1_accuracy'] = float(np.mean([r['top1_correct'] for r in rows]))
    ece, bins = 0., []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        own = [r for r in rows if lo <= r['confidence'] and
               (r['confidence'] < hi or hi == 1.)]
        if not own:
            continue
        confidence = float(np.mean([r['confidence'] for r in own]))
        accuracy = float(np.mean([r['top1_correct'] for r in own]))
        ece += len(own)/len(rows)*abs(accuracy-confidence)
        bins.append({'lower': float(lo), 'upper': float(hi), 'count': len(own),
                     'confidence': confidence, 'accuracy': accuracy})
    result.update(ece=ece, bins=bins)
    return result


def render_figures(out, metrics, probabilities):
    from .metric_charts import _plotting
    plt = _plotting()
    output = out/'figures'
    output.mkdir(exist_ok=True)
    names = list(METHODS)
    colors = ['#d08239', '#2768a8']
    specs = [('mean_all_targets_duration_s', '全部目标同时跟踪时间', '秒', 1),
        ('recovery_rate', '重捕率', '%', 100), ('success_rate', '整组完整跟踪成功率', '%', 100),
        ('position_rmse_m_on_detections', '原 N-Net 检测帧位置 RMSE', 'm', 1),
        ('position_evaluation_coverage', '位置误差评价覆盖率', '%', 100),
        ('rnet_target_frames', '实际调用 R-Net 的目标帧数', '帧', 1)]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))
    for ax, (key, title, unit, scale) in zip(axes.flat, specs):
        values = [metrics[n].get(key) for n in names]
        bars = ax.bar(range(2), [np.nan if v is None else v*scale for v in values], color=colors, width=.55)
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value is None:
                ax.text(i, .1, 'N/A', transform=ax.get_xaxis_transform(), ha='center')
            else:
                text = f'{value*scale:.2f}' if key != 'rnet_target_frames' else str(int(value))
                if key == 'recovery_rate':
                    text += f'\n{metrics[names[i]]["recoveries"]}/{metrics[names[i]]["lost_events"]}'
                ax.annotate(text, (bar.get_x()+bar.get_width()/2, value*scale), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=10)
        finite = [v*scale for v in values if v is not None]
        ax.set_ylim(0, 115 if unit == '%' else max([1.]+finite)*1.28)
        ax.set(title=title, ylabel=unit)
        ax.set_xticks(range(2), ['直接测量\n原 N-Net/R-Net', '鲁棒预处理\n原 N-Net/R-Net'])
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('重做 V4：保留原 Transformer N-Net/R-Net', fontsize=17, fontweight='bold', y=.98)
    fig.text(.06, .025, '同一组旧权重、51条机动轨迹组成17场景、3目标共享12束；任一目标连续漏检5帧即整组终止。\n'
        '位置来自原N-Net，未替换为滤波结果。RMSE仅统计检测帧；不同方法的评价帧集合不同，需结合覆盖率解读。',
        fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(.02, .09, .99, .94), h_pad=2.3)
    fig.savefig(output/'nr_tracking_metrics.png', dpi=180, facecolor='white'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for n, color in zip(names, colors):
        p = probabilities[n]
        axes[0].plot([x['confidence'] for x in p['bins']], [x['accuracy'] for x in p['bins']],
                     'o-', color=color, label=f'{METHODS[n]}（ECE={p["ece"]:.3f}）')
        keys = ('mean_nll_nats', 'mean_brier_score')
        i = names.index(n)
        axes[1].bar(np.arange(2)+(i-.5)*.32, [p[k] for k in keys], width=.3, color=color, label=METHODS[n])
    axes[0].plot([0, 1], [0, 1], '--', color='#777777')
    axes[0].set(xlabel='预测最高波束概率', ylabel='该分箱实际 Top-1 正确率', xlim=(0, 1), ylim=(0, 1.05))
    axes[1].set_xticks(range(2), ['下一步波束 NLL', '波束 Brier 分数'])
    axes[1].set_ylabel('越小越好')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('原 N/R 网络的下一步波束概率是否可信', fontsize=16, y=.98)
    fig.text(.06, .025, '直接评价原网络Softmax，未用测试集校准。仅含场景存活期间的预测；不是三维位置或整条轨迹的不确定性。', fontsize=9)
    fig.tight_layout(rect=(.02, .09, .99, .92))
    fig.savefig(output/'beam_probability_reliability.png', dpi=180, facecolor='white'); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--archive-version')
    parser.add_argument('--plot-trajectory-id', type=int, default=V4_REFERENCE_ID)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_v4_original_nr')
    args = parser.parse_args(argv)
    if args.archive_version and (BASE/'experiments'/args.archive_version).exists():
        parser.error('版本已存在，不能覆盖旧版本')
    model_source = Path(__file__).with_name('models.py')
    if file_sha(model_source) != ORIGINAL_MODELS_SHA256:
        raise RuntimeError('N-Net/R-Net定义与V1不一致，停止实验')
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    else:
        import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    device = torch.device(args.device)
    current = read_json(BASE/'outputs/current_journal_experiment.json')
    nnet, checkpoint = load_nnet(current['nnet_checkpoint'], device)
    saved = torch.load(current['rnet_checkpoint'], map_location=device, weights_only=False)
    if saved['model_config'] != checkpoint['model_config']:
        raise ValueError('N-Net和R-Net配置不一致')
    rnet = JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(saved['model'])
    source = args.data or find_maneuver_test()
    source = source/'test.npz' if source.is_dir() else source
    data = load_split(source)
    scenes, indices, unused = make_scenes(data, 3, seed=20261115)
    if args.plot_trajectory_id not in scenes['trajectory_id']:
        raise ValueError('示例轨迹编号不在本次测试场景中')
    for key in ('measurement_sigma', 'snr_db', 'outlier_mask'):
        scenes[key] = data[key][indices]
    dt, budget, history = .05, 12, nnet.config.history
    maximum = (scenes['beam'].shape[-1]-history)*dt
    out = new_run(args.output_root)
    (out/'data').mkdir()
    shutil.copy2(source, out/'data/test.npz')
    before = state_digest(nnet), state_digest(rnet)
    weight_hashes = {k: file_sha(current[k]) for k in ('nnet_checkpoint', 'rnet_checkpoint')}
    metrics, probabilities, calls = {}, {}, {}
    for method, quality in (('nr_raw', 'raw'), ('nr_robust', 'robust')):
        print(f'重做V4：{METHODS[method]}', flush=True)
        result, trace, updates, timing, probability_rows, audit = run_original_nr(
            nnet, rnet, checkpoint, scenes, device, quality_mode=quality, dt=dt,
            budget=budget, plot_id=args.plot_trajectory_id)
        summary = summarize(result[0], result[1], result[4], dt, budget, 3, 'joint')
        summary.update(summarize_measurement_updates(updates, timing))
        summary['position_evaluation_count'] = sum(x['position_count'] for x in result[1])
        summary['position_evaluation_coverage'] = summary['position_evaluation_count']/(len(indices)*3*(scenes['beam'].shape[-1]-history))
        summary['rnet_target_frames'] = audit['rnet_target_frames']
        summary['position_metric'] = 'original N-Net position head on actually detected frames only'
        summary['quality_mode'] = quality
        metrics[method], calls[method] = summary, audit
        probabilities[method] = probability_summary(probability_rows)
        for suffix, rows in zip(('scenes', 'targets', 'events', 'scans'), result[:4]):
            write_rows(out/f'{method}_{suffix}.csv', rows)
        write_rows(out/f'{method}_measurement_updates.csv', updates)
        write_rows(out/f'{method}_beam_probability_events.csv', probability_rows)
        figure_root = out if method == 'nr_robust' else out/'raw_control'
        save_trajectory_figure(figure_root, scenes, trace, result[1], summary, dt=dt,
            history=history, trajectory_id=args.plot_trajectory_id,
            title='重做 V4：原 Transformer N-Net/R-Net'+('＋鲁棒预处理' if quality == 'robust' else '（直接测量）'),
            selection_note='固定旧V4示例 ID 400017；本次不根据跟踪结果重新挑选。全部指标来自原N/R实际闭环仿真。')
        print({k: summary[k] for k in ('mean_all_targets_duration_s', 'recovery_rate',
            'success_rate', 'position_rmse_m_on_detections')}, flush=True)
    if before != (state_digest(nnet), state_digest(rnet)) or file_sha(model_source) != ORIGINAL_MODELS_SHA256:
        raise RuntimeError('推理修改了原模型或参数')
    if weight_hashes != {k: file_sha(current[k]) for k in weight_hashes}:
        raise RuntimeError('原模型权重文件发生变化')
    record = {'version': 'v4_original_nr_maneuver', 'source': str((out/'data/test.npz').resolve()),
        'nnet_checkpoint': current['nnet_checkpoint'], 'rnet_checkpoint': current['rnet_checkpoint'],
        'trajectory_figure': 'figures/single_trajectory_tracking.png', 'targets': 3,
        'total_beams': budget, 'dt': dt, 'history': history, 'failure_limit': nnet.config.failure_limit,
        'maximum_evaluated_duration_s': maximum, 'scene_membership': scenes['trajectory_id'].tolist(),
        'unused_trajectory_ids': unused, 'results': metrics, 'beam_probabilities': probabilities,
        'models_source_sha256': ORIGINAL_MODELS_SHA256, 'checkpoint_sha256': weight_hashes,
        'actual_network_calls': calls, 'weights_unchanged': True, 'networks_unchanged': True,
        'alternative_prediction_network_used': False, 'same_external_measurement_stream': True,
        'measurement_sha256': hashlib.sha256(scenes['measurement'].tobytes()).hexdigest(),
        'reused_maneuver_test_sha256': file_sha(source), 'filtered_measurement_array_used': False,
        'fully_reproduced_paper_numerical_results': False,
        'limitations': ['Fixed original weights are tested on new simulated maneuvers; no retraining or tuning on test.',
            'The probability evaluation is next-beam classification, not 3-D position/path uncertainty.',
            'Same 17 groups as old V4; any target reaching five misses stops the entire group.',
            'Position RMSE uses N-Net output on detections only; exposures differ between methods.',
            'Ideal measurement identity and beam-hit feedback are still assumed; identity association is not implemented.',
            'Receiver sigmas are idealized; this is trajectory-driven proxy simulation, not physical radar data.']}
    write_json(out/'summary.json', record)
    write_json(out/'network_integrity.json', {k: record[k] for k in ('models_source_sha256',
        'checkpoint_sha256', 'actual_network_calls', 'networks_unchanged', 'weights_unchanged',
        'alternative_prediction_network_used')})
    render_figures(out, metrics, probabilities)
    report = ['重做V4：原Transformer N-Net/R-Net不修改、不替换；固定旧权重。',
        '同一批机动轨迹：匀速、加速制动、急转弯、突然机动、混合运动；51条轨迹组成17场景。',
        '主图蓝线为原N-Net位置输出；鲁棒滤波仅作为其输入预处理。']
    for method, m in metrics.items():
        report.append(f'{METHODS[method]}：平均同时跟踪{m["mean_all_targets_duration_s"]:.6f}秒，'
            f'重捕{m["recoveries"]}/{m["lost_events"]}，整组成功率{m["success_rate"]:.2%}，'
            f'N-Net检测帧位置RMSE={m["position_rmse_m_on_detections"]:.6f}米。')
    report += ['不承诺重做后指标提高；未使用测试结果调整网络或筛选展示轨迹。',
        '下一步：以原N/R为基础建立匿名检测集合，再逐步实现身份关联。']
    (out/'report.txt').write_text('\n'.join(report), encoding='utf-8')
    from .versions import save_version
    archive = save_version(args.archive_version or 'v4_nr_'+out.name, out,
        '重做V4：原Transformer N-Net/R-Net结构与权重不变；同机动数据的直接测量/鲁棒预处理对照；N-Net位置轨迹。')
    (args.output_root/'latest_complete_run.txt').write_text(out.name, encoding='utf-8')
    print(f'结果：{out.resolve()}\n版本：{archive.resolve()}', flush=True)
    if not args.no_show:
        import matplotlib.pyplot as plt
        for name in ('single_trajectory_tracking.png', 'nr_tracking_metrics.png'):
            fig, ax = plt.subplots(figsize=(12, 8)); ax.imshow(plt.imread(out/'figures'/name)); ax.axis('off')
        plt.show()
    return out


if __name__ == '__main__':
    main()
