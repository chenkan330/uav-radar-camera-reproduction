"""Chinese figures for V4 maneuver forecasts, calibrated uncertainty and tracking.

All uncertainty coverage here is averaged coordinate-marginal coverage. A 90%
coordinate interval is not a 90% three-dimensional joint prediction region.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

MODEL_LABELS = {
    'cv_fixed': '匀速预测＋固定误差', 'cv_gaussian': '匀速预测＋固定误差',
    'nn_raw': '概率网络（校准前）', 'nn_calibrated': '概率网络（校准后）',
    'legacy_v3': '原有网络＋鲁棒处理', 'motion_point': '新网络均值＋固定误差',
    'motion_probability': '新网络概率预测',
    'cv_constant_spread': '匀速预测＋固定误差',
    'transformer_point_fixed_spread': '新网络均值＋固定误差',
    'transformer_uncalibrated': '概率网络（校准前）',
    'transformer_calibrated': '概率网络（校准后）',
}
COHORT_LABELS = {
    'all': '全部运动', 'constant_speed': '匀速', 'acceleration_braking': '加速／制动',
    'sharp_turn': '急转弯', 'sudden_maneuver': '突然机动', 'mixed': '混合运动',
    'type_0': '匀速', 'type_1': '加速／制动', 'type_2': '急转弯',
    'type_3': '突然机动', 'type_4': '混合运动',
}
COLORS = ['#2768a8', '#e58a35', '#299779', '#9362b6', '#ce5966']
LEVELS = (.5, .8, .9, .95)
COVERAGE_NOTE = ('覆盖率为各坐标边际区间的平均覆盖率，不是三维联合覆盖率；'
                 '同时观察区间宽度，避免仅靠放大范围取得高覆盖。')


def _plotting():
    import matplotlib
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10,
                         'savefig.facecolor': 'white'})
    return plt


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _first(root, names):
    for name in names:
        path = root/name
        if path.exists():
            return path
    return None


def _number(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and value != '':
            result = float(value)
            return result if np.isfinite(result) else np.nan
    return np.nan


def _level(row, category, level):
    mapping = row.get(category, {})
    if isinstance(mapping, dict):
        for key in (str(level), f'{level:.2f}', str(int(level*100))):
            if key in mapping:
                return float(mapping[key])
    prefixes = ('coverage', 'marginal_coverage') if category == 'coverage' else ('mean_width_m', 'width_m', 'interval_width_m')
    for prefix in prefixes:
        for key in (f'{prefix}_{int(level*100)}', f'{prefix}_{level}'):
            if key in row:
                return float(row[key])
    if category == 'mean_width_m':
        key = f'marginal_width_{int(level*100)}_m'
        if key in row:
            return float(row[key])
    return np.nan


def _records(record):
    rows = record['records'] if isinstance(record, dict) else record
    result = []
    for original in rows:
        row = dict(original)
        if 'method' in row:
            row.setdefault('model', row['method'])
        row.setdefault('cohort', 'all')
        if 'horizon_s' not in row:
            row['horizon_s'] = float(row['horizon_steps'])*float(record.get('dt', .05))
        row['horizon_s'] = float(row['horizon_s'])
        if 'model' not in row:
            raise ValueError('Forecast metric rows require a model name')
        result.append(row)
    if not result:
        raise ValueError('No saved forecast metric rows to plot')
    return result


def _save(fig, output, name, paths):
    plt = _plotting()
    path = output/name
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    paths.append(path)


def _forecast_scores(rows, output, paths):
    plt = _plotting()
    selected = [r for r in rows if r['cohort'] == 'all']
    models = list(dict.fromkeys(r['model'] for r in selected))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    specs = ((('rmse_m', 'position_rmse_m', 'rmse'), '未来位置均值 RMSE', 'm'),
             (('nll', 'nll_nats'), '预测分布负对数似然', 'nats／预测位置'),
             (('energy_score', 'energy_score_m'), '预测分布能量分数', 'm'))
    for model_index, model in enumerate(models):
        own = sorted([r for r in selected if r['model'] == model], key=lambda r: r['horizon_s'])
        x = [r['horizon_s'] for r in own]
        for ax, (keys, title, unit) in zip(axes, specs):
            ax.plot(x, [_number(r, *keys) for r in own], marker='o', linewidth=2,
                    color=COLORS[model_index % len(COLORS)], label=MODEL_LABELS.get(model, model))
            ax.set(title=title, xlabel='未来预测时间／s', ylabel=unit)
            ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('V4：未来运动的误差与概率评分', x=.055, y=.98, ha='left', fontsize=16, fontweight='bold')
    fig.text(.055, .025, '分数越低越好；三种网络方案的预测均值相同，RMSE 曲线重合。模型由验证集选择，尺度只用独立校准集确定。', fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(.02, .1, .99, .88), w_pad=2.)
    _save(fig, output, 'forecast_scores.png', paths)


def _calibration(rows, output, paths):
    plt = _plotting()
    selected = [r for r in rows if r['cohort'] == 'all']
    horizons = sorted({r['horizon_s'] for r in selected})
    models = list(dict.fromkeys(r['model'] for r in selected))
    fig, axes = plt.subplots(2, len(horizons), figsize=(max(11, len(horizons)*3.4), 8), squeeze=False)
    for column, horizon in enumerate(horizons):
        for model_index, model in enumerate(models):
            matches = [r for r in selected if r['model'] == model and r['horizon_s'] == horizon]
            if len(matches) != 1:
                raise ValueError('Each model/horizon must have exactly one overall forecast row')
            row = matches[0]
            color = COLORS[model_index % len(COLORS)]
            axes[0, column].plot(100*np.array(LEVELS), [100*_level(row, 'coverage', q) for q in LEVELS],
                                 marker='o', color=color, label=MODEL_LABELS.get(model, model))
            axes[1, column].plot(100*np.array(LEVELS), [_level(row, 'mean_width_m', q) for q in LEVELS],
                                 marker='o', color=color)
        axes[0, column].plot([45, 100], [45, 100], color='#6b7280', linestyle='--', linewidth=1)
        axes[0, column].set(title=f'预测未来 {horizon:g} 秒', xlabel='标称覆盖率／%',
                            ylabel='实际边际覆盖率／%', xlim=(45, 100), ylim=(0, 102))
        axes[1, column].set(xlabel='标称覆盖率／%', ylabel='平均坐标区间宽度／m', xlim=(45, 100))
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('V4：预测的不确定性是否可信', x=.055, y=.98, ha='left', fontsize=16, fontweight='bold')
    fig.text(.055, .025, COVERAGE_NOTE, fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(.02, .08, .99, .92), h_pad=2.3, w_pad=1.7)
    _save(fig, output, 'uncertainty_calibration.png', paths)


def _motion_cohorts(rows, output, paths):
    plt = _plotting()
    available = [r for r in rows if r['cohort'].startswith('type_') or r['cohort'] in
                 ('constant_speed', 'acceleration_braking', 'sharp_turn', 'sudden_maneuver', 'mixed')]
    if not available:
        return
    horizon = max(r['horizon_s'] for r in available)
    selected = [r for r in available if r['horizon_s'] == horizon]
    cohorts = list(dict.fromkeys(r['cohort'] for r in selected))
    models = list(dict.fromkeys(r['model'] for r in selected))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5))
    width = .78/max(len(models), 1)
    positions = np.arange(len(cohorts))
    for model_index, model in enumerate(models):
        own = {r['cohort']: r for r in selected if r['model'] == model}
        x = positions-.39+width*(model_index+.5)
        values = ([_number(own.get(c, {}), 'rmse_m', 'position_rmse_m', 'rmse') for c in cohorts],
                  [100*_level(own.get(c, {}), 'coverage', .9) for c in cohorts],
                  [_level(own.get(c, {}), 'mean_width_m', .9) for c in cohorts])
        for ax, ys in zip(axes, values):
            ax.bar(x, ys, width=width*.9, color=COLORS[model_index % len(COLORS)],
                   label=MODEL_LABELS.get(model, model), zorder=3)
    axes[0].set(title=f'未来 {horizon:g} 秒的预测误差', ylabel='位置均值 RMSE／m')
    axes[1].axhline(90, color='#6b7280', linestyle='--', linewidth=1)
    axes[1].set(title='标称 90% 区间的实际覆盖率', ylabel='边际覆盖率／%', ylim=(0, 108))
    axes[2].set(title='标称 90% 区间的宽度', ylabel='平均坐标区间宽度／m')
    for ax in axes:
        ax.set_xticks(positions, [COHORT_LABELS.get(c, c) for c in cohorts])
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('V4：不同运动类型下的误差与不确定性', x=.065, y=.985, ha='left', fontsize=16, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.56, .952),
               ncol=min(4, len(models)), frameon=False, fontsize=9)
    fig.text(.065, .018, '运动类型标签只用于事后分组，不输入网络。'+COVERAGE_NOTE, fontsize=8.5, color='#4b5563')
    fig.tight_layout(rect=(.02, .065, .99, .9), h_pad=2.)
    _save(fig, output, 'motion_type_comparison.png', paths)


def _onset_diagnostics(rows, output, paths):
    plt = _plotting()
    cohorts = ('no_future_onset', 'future_onset')
    available = [r for r in rows if r['cohort'] in cohorts]
    if not available:
        return
    horizon = max(r['horizon_s'] for r in available)
    selected = [r for r in available if r['horizon_s'] == horizon]
    models = list(dict.fromkeys(r['model'] for r in selected))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2))
    for index, cohort in enumerate(cohorts):
        own = {r['model']: r for r in selected if r['cohort'] == cohort}
        title = '未来窗口内无新机动' if cohort == 'no_future_onset' else '未来窗口内发生新机动'
        for ax, values, name, unit in (
            (axes[index, 0], [_number(own.get(m, {}), 'rmse_m') for m in models], '预测均值误差', 'RMSE／m'),
            (axes[index, 1], [_level(own.get(m, {}), 'mean_width_m', .9) for m in models], '标称 90% 区间宽度', '平均坐标宽度／m'),
        ):
            bars = ax.bar(np.arange(len(models)), values, width=.65, color=COLORS[:len(models)], zorder=3)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.annotate(f'{value:.2f}', (bar.get_x()+bar.get_width()/2, value), xytext=(0, 4),
                                textcoords='offset points', ha='center', fontsize=8)
            ax.set_title(f'{title}\n{name}', loc='left', fontsize=10.5, pad=13)
            ax.set_ylabel(unit)
            short = {'cv_constant_spread': '匀速预测\n＋固定误差',
                     'transformer_point_fixed_spread': '新网络均值\n＋固定误差',
                     'transformer_uncalibrated': '概率网络\n校准前',
                     'transformer_calibrated': '概率网络\n校准后'}
            ax.set_xticks(np.arange(len(models)), [short.get(m, MODEL_LABELS.get(m, m)) for m in models], fontsize=7.8)
            finite = [v for v in values if np.isfinite(v)]
            ax.set_ylim(0, max([1.]+finite)*1.26)
            ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        ax = axes[index, 2]
        width = .78/len(models)
        for model_index, model in enumerate(models):
            values = [100*_number(own.get(model, {}), f'coverage_90_{coord}') for coord in ('x', 'y', 'z')]
            ax.bar(np.arange(3)-.39+width*(model_index+.5), values, width=width*.9,
                   color=COLORS[model_index % len(COLORS)], label=MODEL_LABELS.get(model, model), zorder=3)
        ax.axhline(90, color='#6b7280', linestyle='--', linewidth=1)
        ax.set(title=f'{title}\n各坐标实际覆盖率', ylabel='90% 边际区间实际覆盖率／%', ylim=(0, 108))
        ax.set_xticks(np.arange(3), ['东向 x', '北向 y', '高度 z'])
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.57, .947), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f'V4：未来 {horizon:g} 秒内突然出现机动时，预测是否过于确定',
                 x=.045, y=.99, ha='left', fontsize=16, fontweight='bold')
    fig.text(.045, .02, '是否发生未来机动仅用于事后分组，不能作为在线输入；突发随机机动不能由历史提前确定。\n各坐标分别评价，不能把三个坐标的平均覆盖率当作三维联合覆盖率。', fontsize=8.5, color='#4b5563')
    fig.tight_layout(rect=(.01, .09, .99, .89), h_pad=2.6, w_pad=2.)
    _save(fig, output, 'maneuver_onset_uncertainty.png', paths)


def _trajectory_examples(path, output, paths):
    plt = _plotting()
    from matplotlib.patches import Rectangle
    with np.load(path, allow_pickle=False) as saved:
        d = {k: saved[k] for k in saved.files}
    if 'history_xyz' not in d:
        d = _examples_from_predictions(path, d)
    history, truth = d['history_xyz'], d['truth_future']
    mean, weights = d['pred_mean'], d['pred_weight']
    lower, upper = d['lower95'], d['upper95']
    count = min(len(history), 5)
    fig, axes = plt.subplots(1, count, figsize=(max(10, count*3.6), 4.8), squeeze=False)
    cohorts = d.get('cohort', np.array(['示例']*count))
    ids = d.get('trajectory_id', np.arange(count))
    expected = (mean*weights[..., None]).sum(-2)
    for i, ax in enumerate(axes[0]):
        # Origin translation makes local geometry readable without exposing raw coordinates.
        origin = history[i, -1]
        h, t, m = history[i]-origin, truth[i]-origin, expected[i]-origin
        ax.plot(h[:, 0], h[:, 1], color='#6b7280', marker='.', linewidth=1.3, label='已有测量历史')
        ax.plot(t[:, 0], t[:, 1], color='#1f2937', marker='x', linewidth=1.5, label='实际未来位置')
        ax.plot(m[:, 0], m[:, 1], color='#2768a8', marker='o', linestyle='--', linewidth=1.5, label='预测分布均值')
        # Components are marginal at each horizon; never connect component IDs into a path.
        last = mean[i, -1]-origin
        ax.scatter(last[:, 0], last[:, 1], s=30+150*weights[i, -1], color='#299779',
                   edgecolors='white', linewidths=.4, label='最长时距的混合分量')
        lo, hi = lower[i, -1]-origin, upper[i, -1]-origin
        ax.add_patch(Rectangle(lo[:2], hi[0]-lo[0], hi[1]-lo[1],
                               facecolor='#299779', edgecolor='#299779', alpha=.13,
                               label='横纵坐标各 95% 区间'))
        ax.scatter([0], [0], color='#ce5966', marker='*', s=70, zorder=5)
        name = COHORT_LABELS.get(str(cohorts[i]), str(cohorts[i]))
        ax.set(title=f'{name}｜轨迹 {int(ids[i])}', xlabel='相对东向位移／m', ylabel='相对北向位移／m')
        ax.grid(alpha=.2); ax.set_aspect('equal', adjustable='datalim')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, .055))
    fig.suptitle('V4：历史测量、未来运动与概率范围', x=.055, y=.99, ha='left', fontsize=16, fontweight='bold')
    fig.text(.055, .015, '按预设轨迹顺序与固定时刻取例；矩形不是联合 95% 预测区域。未来机动的随机起点不能由历史提前确定。', fontsize=8.5, color='#4b5563')
    fig.tight_layout(rect=(.01, .21, .99, .89), w_pad=1.5)
    _save(fig, output, 'future_motion_examples.png', paths)


def _examples_from_predictions(path, d):
    """Choose first trajectory in each cohort, at the window closest to 5 s.

    This rule is independent of forecast errors and uncertainty calibration.
    """
    from scipy.special import ndtr
    from .measurement_quality import spherical_to_cartesian
    run = Path(path).parent
    data_path = _first(run, ('data/test.npz', 'test.npz'))
    if data_path is None:
        raise FileNotFoundError('V4 example plotting requires its saved test split')
    with np.load(data_path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    kinds = d['trajectory_kind']
    rows = []
    for kind in sorted(np.unique(kinds)):
        candidates = np.flatnonzero(kinds == kind)
        first_id = np.min(d['trajectory_id'][candidates])
        candidates = candidates[d['trajectory_id'][candidates] == first_id]
        rows.append(candidates[np.argmin(np.abs(d['history_end'][candidates]-99))])
    rows = np.asarray(rows, dtype=int)
    ids = d['trajectory_id'][rows]
    indices = {int(value): i for i, value in enumerate(data['trajectory_id'])}
    history_length = 20
    summary = run/'forecast_summary.json'
    if summary.exists():
        history_length = int(_load(summary)['model_config']['history'])
    history = np.array([spherical_to_cartesian(data['filtered_measurement'][indices[int(tid)],
                        int(d['history_end'][row])-history_length+1:int(d['history_end'][row])+1])
                        for row, tid in zip(rows, ids)])
    weights, mean, scale = d['weights'][rows], d['mean'][rows], d['scale'][rows]

    def quantile(q):
        lo = np.min(mean-12*scale, axis=-2)
        hi = np.max(mean+12*scale, axis=-2)
        for _ in range(65):
            middle = (lo+hi)*.5
            cdf = (weights[..., None]*ndtr((middle[..., None, :]-mean)/scale)).sum(-2)
            lo = np.where(cdf < q, middle, lo)
            hi = np.where(cdf >= q, middle, hi)
        return (lo+hi)*.5
    return {'history_xyz': history, 'truth_future': d['target'][rows],
            'pred_mean': mean, 'pred_scale': scale, 'pred_weight': weights,
            'lower95': quantile(.025), 'upper95': quantile(.975),
            'cohort': np.array([f'type_{int(kind)}' for kind in kinds[rows]]),
            'trajectory_id': ids}


def _training_curve(path, output, paths):
    plt = _plotting()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].plot(epochs, [float(r['training_nll']) for r in rows], marker='o', color=COLORS[0], label='训练集')
    axes[0].plot(epochs, [float(r['validation_nll']) for r in rows], marker='o', color=COLORS[1], label='验证集')
    axes[0].set(xlabel='训练轮次', ylabel='负对数似然／nats', title='训练与验证概率损失')
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [float(r['learning_rate']) for r in rows], marker='o', color=COLORS[2])
    axes[1].set(xlabel='训练轮次', ylabel='学习率', title='预先确定的学习率日程')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('V4：概率运动网络训练记录', x=.06, y=.98, ha='left', fontsize=15, fontweight='bold')
    fig.text(.06, .025, '模型按验证集 NLL 选择；测试集不用于模型选择，独立校准集只用于尺度校准。', fontsize=9, color='#4b5563')
    fig.tight_layout(rect=(.02, .1, .99, .88), w_pad=2.)
    _save(fig, output, 'motion_network_training.png', paths)


def _tracking(record, output, paths):
    plt = _plotting()
    methods = record['methods']
    keys = list(methods)
    labels = [MODEL_LABELS.get(k, k) for k in keys]
    specs = (('success_rate', '整组跟踪成功率', '%', 100.),
             ('mean_all_targets_duration_s', '全部目标同时持续跟踪时长', 's', 1.),
             ('current_filtered_position_rmse_m_on_detections', '检测帧滤波位置 RMSE', 'm', 1.),
             ('recovery_rate', '重捕获率', '%', 100.),
             ('mean_recovery_delay_s', '平均重捕获延迟', 'ms', 1000.),
             ('amortized_prediction_ms_per_scene', '场景均摊波束预测耗时', 'ms／scene', 1.))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (key, title, unit, factor) in zip(axes.flat, specs):
        values = [_number(methods[k], key)*factor for k in keys]
        bars = ax.bar(np.arange(len(keys)), values, width=.6, color=COLORS[:len(keys)], zorder=3)
        finite = [v for v in values if np.isfinite(v)]
        for bar, value, method in zip(bars, values, keys):
            if np.isfinite(value):
                text = f'{value:.2f}'
                if key == 'current_filtered_position_rmse_m_on_detections':
                    count = methods[method].get('measurement_update_count')
                    if count is not None:
                        text += f'\nn={int(count):,}'
                ax.annotate(text, (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
            else:
                ax.text(bar.get_x()+bar.get_width()/2, .04, 'N/A',
                        transform=ax.get_xaxis_transform(), ha='center')
        ax.set_title(title, loc='left', fontsize=11, pad=13)
        ax.set_ylabel(unit)
        ax.set_xticks(np.arange(len(keys)), [label.replace('＋', '\n＋') for label in labels], fontsize=8.5)
        ax.set_ylim(0, 115 if unit == '%' else max([1.]+finite)*1.24)
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    fig.suptitle('V4：相同机动场景、相同波束预算下的闭环跟踪', x=.055, y=.985, ha='left', fontsize=16, fontweight='bold')
    fig.text(.055, .022, '本图三种方法使用同一批新生成的机动轨迹；不能与 V1～V3 的旧飞行轨迹数字直接比较。\n均值／概率方案复用同一个新网络；旧网络对比还受训练数据与历史长度影响。\n任一目标连续漏检 5 次即终止整组。RMSE 是检测帧滤波位置误差，n 是评价帧数；均摊预测耗时不是独立场景端到端延迟。', fontsize=8.5, color='#4b5563')
    fig.tight_layout(rect=(.02, .09, .99, .93), h_pad=2.3, w_pad=2.)
    _save(fig, output, 'maneuver_tracking_comparison.png', paths)


def render_v4(run_dir, output_dir=None):
    """Render completed V4 metrics. Return the list of saved PNG paths."""
    run = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else run/'figures'
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    metrics = _first(run, ('forecast_metrics.json', 'forecasts/forecast_metrics.json', 'forecast_metrics.csv'))
    if metrics:
        if metrics.suffix == '.csv':
            with metrics.open(encoding='utf-8-sig', newline='') as handle:
                rows = _records(list(csv.DictReader(handle)))
        else:
            rows = _records(_load(metrics))
        _forecast_scores(rows, output, paths)
        _calibration(rows, output, paths)
        _motion_cohorts(rows, output, paths)
        _onset_diagnostics(rows, output, paths)
    examples = _first(run, ('forecast_examples.npz', 'forecasts/forecast_examples.npz', 'forecast_predictions.npz'))
    if examples:
        _trajectory_examples(examples, output, paths)
    tracking = _first(run, ('tracking_metrics.json', 'tracking/tracking_metrics.json'))
    if tracking:
        _tracking(_load(tracking), output, paths)
    curve = _first(run, ('training_curve.csv',))
    if curve:
        _training_curve(curve, output, paths)
    if not paths:
        raise FileNotFoundError('No saved V4 forecast/tracking metrics were found')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    for path in render_v4(args.run, args.output):
        print(path.resolve())


if __name__ == '__main__':
    main()
