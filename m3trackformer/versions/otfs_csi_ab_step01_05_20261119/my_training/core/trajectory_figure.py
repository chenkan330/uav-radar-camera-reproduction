"""Plot actual closed-loop estimates and events; never fabricate a trajectory.

V4's display ID is fixed once for comparisons. It is an illustration containing
a recovery in the existing V4 run, not a randomly selected or typical test case.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

V4_REFERENCE_ID = 400017


def trace_statistics(rows, dt):
    """Match the simulator's event definition; initialization is not scored."""
    evaluated = sorted((r for r in rows if r['phase'] == 'tracking'), key=lambda r: r['time_index'])
    lost, recovered, missing = 0, 0, False
    for row in evaluated:
        hit = row['hit'] is True or str(row['hit']).lower() == 'true'
        if hit:
            recovered += int(missing)
            missing = False
        elif not missing:
            lost += 1
            missing = True
    return {'tracking_duration_s': len(evaluated)*dt, 'lost_events': lost,
            'recoveries': recovered, 'recovery_rate': recovered/lost if lost else None,
            'evaluated_frames': len(evaluated),
            'missed_frames': sum(not (r['hit'] is True or str(r['hit']).lower() == 'true') for r in evaluated)}


def _rate(rate, recovered, lost):
    return f'{100*rate:.1f}%（{recovered}/{lost}）' if rate is not None else 'N/A（无丢失事件）'


def save_trajectory_figure(run_dir, scenes, trace_rows, target_rows, summary, *, dt,
                           history, trajectory_id, title, selection_note=None,
                           highlight_dropout=()):
    """Write PNG + exact plotted samples + definitions/hashes into a fresh run.

    Truth is read only here, after simulation, for the red line and loss markers.
    No interpolation is used to fill misses or the portion after termination.
    """
    root = Path(run_dir)
    output = root/'figures'
    output.mkdir(parents=True, exist_ok=True)
    png = output/'single_trajectory_tracking.png'
    csv_path = output/'single_trajectory_tracking_samples.csv'
    metadata_path = output/'single_trajectory_tracking.json'
    if any(p.exists() for p in (png, csv_path, metadata_path)):
        raise FileExistsError('Trajectory artifacts already exist; preserve them and create a new run')
    matching = np.argwhere(scenes['trajectory_id'] == trajectory_id)
    if matching.shape != (1, 2):
        raise ValueError('Display trajectory ID must occur exactly once')
    g, t = map(int, matching[0])
    rows = [r for r in trace_rows if int(r['trajectory_id']) == trajectory_id]
    if not rows or len({int(r['time_index']) for r in rows}) != len(rows):
        raise ValueError('Display trace is empty or has duplicate frame indices')
    stats = trace_statistics(rows, dt)
    target = next(r for r in target_rows if int(r['trajectory_id']) == trajectory_id)
    def enabled(value):
        return value is True or str(value).lower() == 'true'
    stop_reason = ('peer_failure_censoring' if enabled(target.get('censored_by_peer_failure'))
                   else 'target_failure' if enabled(target.get('terminal_failure')) else 'complete')
    stop_label = {'peer_failure_censoring': '同伴失跟，场景终止',
                  'target_failure': '本目标连续漏检终止', 'complete': '已完成全程评估'}[stop_reason]
    for key in ('lost_events', 'recoveries'):
        if stats[key] != int(target[key]):
            raise ValueError(f'Trace and simulator disagree on {key}')
    if not np.isclose(stats['tracking_duration_s'], float(target['observed_duration_s'])):
        raise ValueError('Trace and simulator disagree on duration')
    truth = np.asarray(scenes['position'][g, t], dtype=float)
    dropout = sorted({int(i) for i in highlight_dropout})
    if any(i < history or i >= len(truth) for i in dropout):
        raise ValueError('Highlighted dropout frames must be inside the tracking horizon')
    estimates = np.full_like(truth, np.nan)
    missed, recovered = [], []
    lookup = {int(r['time_index']): r for r in rows}
    association_mode = any('identity_correct' in r for r in rows)
    wrong_identity = []
    sources = sorted({r['estimate_source'] for r in rows if r['phase'] == 'tracking'})
    if not sources or len(sources) != 1:
        raise ValueError('A display trace must identify one current-position estimator')
    neural_trace = sources[0] in ('nnet_position_head', 'legacy_neural_position_head')
    sample_rows = []
    for i in range(len(truth)):
        row = lookup.get(i)
        hit = row is not None and (row['hit'] is True or str(row['hit']).lower() == 'true')
        if hit:
            estimates[i] = [row[f'estimated_{a}_m'] for a in 'xyz']
            if not np.isfinite(estimates[i]).all():
                raise ValueError('Acquired frames require actual finite estimates')
        if row is not None and row['phase'] == 'tracking':
            if hit and 'identity_correct' in row and not enabled(row['identity_correct']):
                wrong_identity.append(i)
            if not hit:
                missed.append(i)
            if row['recovered'] is True or str(row['recovered']).lower() == 'true':
                recovered.append(i)
        sample_rows.append({'trajectory_id': trajectory_id, 'scene_id': g, 'target_index': t,
            'time_index': i, 'time_s': (i-history)*dt,
            'phase': row['phase'] if row else 'not_evaluated',
            'hit': hit if row else None, 'recovered': i in recovered,
            'estimate_source': row['estimate_source'] if row else '',
            **({'identity_correct': row.get('identity_correct'),
                'assigned_source_evaluation_only': row.get('assigned_source_evaluation_only')} if association_mode and row else {}),
            **{f'true_{a}_m': float(truth[i, k]) for k, a in enumerate('xyz')},
            **{f'estimated_{a}_m': float(estimates[i, k]) if hit else None for k, a in enumerate('xyz')}})
    with csv_path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_rows[0])
        writer.writeheader()
        writer.writerows(sample_rows)

    import matplotlib
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Rectangle
    with plt.rc_context({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10}):
        fig = plt.figure(figsize=(11.5, 8.5), facecolor='white')
        ax = fig.add_axes((.09, .30, .62, .57))
        zoom = fig.add_axes((.76, .655, .215, .21))
        estimate_label = '原 N-Net 位置输出' if neural_trace else '因果滤波估计'
        blue = estimates.copy()
        if neural_trace:
            # Initial measurements are not neural network position predictions.
            blue[:history] = np.nan
        for plot in (ax, zoom):
            plot.plot(truth[:, 0], truth[:, 1], color='#ee2222', lw=1.7, label='真实轨迹', zorder=2)
            plot.plot(blue[:, 0], blue[:, 1], color='#1455d9', ls='--', lw=1.35,
                      label=estimate_label, zorder=3)
            plot.scatter([truth[0, 0]], [truth[0, 1]], color='#ee2222', s=62, zorder=5)
            if dropout:
                plot.scatter(truth[dropout, 0], truth[dropout, 1], marker='o',
                    facecolors='none', edgecolors='#e98a19', s=135, linewidths=1.5, zorder=5)
            if missed:
                plot.scatter(truth[missed, 0], truth[missed, 1], marker='x', color='black', s=48,
                             linewidths=1.4, zorder=6)
            if recovered:
                plot.scatter(estimates[recovered, 0], estimates[recovered, 1], marker='o',
                             facecolors='none', edgecolors='#16834b', s=75, linewidths=1.5, zorder=7)
            if wrong_identity:
                plot.scatter(estimates[wrong_identity,0],estimates[wrong_identity,1],marker='D',
                    facecolors='none',edgecolors='#a32da8',s=65,linewidths=1.2,zorder=8)
            plot.grid(alpha=.25)
            plot.set_aspect('equal', adjustable='box')
        ax.scatter([0], [0], marker='^', color='#0087bd', s=100, zorder=6)
        if stop_reason == 'peer_failure_censoring':
            last = max(int(r['time_index']) for r in rows)
            point = estimates[last, :2] if np.isfinite(estimates[last, :2]).all() else truth[last, :2]
            ax.scatter([point[0]], [point[1]], marker='s', color='#6b7280', s=40, zorder=7)
            ax.annotate(f'场景于 {stats["tracking_duration_s"]:.2f} s 终止', xy=point,
                xytext=(-8, -24), textcoords='offset points', fontsize=8.5, ha='right',
                color='#4b5563', arrowprops={'arrowstyle': '-', 'color': '#6b7280', 'lw': .8})
        ax.set(xlabel='X（m）', ylabel='Y（m）')
        ax.margins(.07)
        # First miss is selected for explanation, never the smallest-error segment.
        center_index = dropout[0] if dropout else missed[0] if missed else history
        left, right = max(0, center_index-8), min(len(truth), center_index+9)
        local = np.vstack((truth[left:right, :2], estimates[left:right, :2]))
        local = local[np.isfinite(local).all(1)]
        center = (local.min(0)+local.max(0))/2
        half = max(8., float(np.ptp(local, axis=0).max())*.68)
        zoom.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
                 title='注入断测附近放大' if dropout else '首次漏检附近放大' if missed else '闭环开始位置放大')
        zoom.tick_params(labelsize=8)
        ax.add_patch(Rectangle(center-half, 2*half, 2*half, fill=False,
                               linestyle='-.', edgecolor='#555555', linewidth=1))
        fig.add_artist(ConnectionPatch(xyA=center+[half, half], coordsA=ax.transData,
            xyB=(0, .05), coordsB=zoom.transAxes, color='#777777', lw=.8, linestyle='-.'))
        handles = [Line2D([], [], marker='^', color='#0087bd', ls='', ms=9, label='基站 BS'),
            Line2D([], [], color='#ee2222', lw=1.7, label='真实轨迹'),
            Line2D([], [], marker='o', color='#ee2222', ls='', ms=7, label='起点'),
            Line2D([], [], color='#1455d9', ls='--', lw=1.35, label=estimate_label),
            Line2D([], [], marker='x', color='black', ls='', ms=7,
                label='无可用关联测量' if association_mode else '漏检位置'),
            Line2D([], [], marker='o', markerfacecolor='none', color='#16834b', ls='', ms=8, label='重捕成功')]
        if association_mode:
            handles.append(Line2D([],[],marker='D',markerfacecolor='none',color='#a32da8',ls='',ms=7,label='错身份更新'))
        if dropout:
            handles.append(Line2D([], [], marker='o', markerfacecolor='none', color='#e98a19',
                                  ls='', ms=8, label='注入断测帧'))
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.745, .59),
                   frameon=True, fontsize=9 if dropout else 10, labelspacing=.6 if dropout else .7)
        fig.text(.76, .345 if association_mode else .365, f'轨迹 ID：{trajectory_id}\n场景：{g}，目标索引：{t}\n'
                 f'{"无可用更新" if association_mode else "实际漏检"}：{stats["missed_frames"]} 帧\n{stop_label}\n视图：水平面 XY 投影',
                 fontsize=9.5, linespacing=1.6, color='#374151', va='top')
        maximum = (len(truth)-history)*dt
        fig.text(.075, .95, title, fontsize=17, fontweight='bold')
        fig.text(.075, .914, '程序直接读取闭环仿真记录绘制；蓝线来自实际获取测量后的当前状态估计',
                 fontsize=10, color='#4b5563')
        fig.text(.09, .19, f'本条轨迹  {"可用测量跟踪时间" if association_mode else "跟踪时间"}：{stats["tracking_duration_s"]:.2f} / {maximum:.2f} 秒'
            f'    {"更新恢复率" if association_mode else "重捕率"}：{_rate(stats["recovery_rate"], stats["recoveries"], stats["lost_events"])}',
            fontsize=12, fontweight='bold')
        scene_scope = summary.get('scene_scope', f'全部 {summary["scenes"]} 个场景')
        fig.text(.09, .145, f'{scene_scope}  平均同时跟踪时间：'
            f'{summary["mean_all_targets_duration_s"]:.2f} 秒    {"更新恢复率" if association_mode else "重捕率"}：'
            f'{_rate(summary["recovery_rate"], summary["recoveries"], summary["lost_events"])}', fontsize=11)
        if association_mode and summary.get('physical_recovery'):
            physical=summary['physical_recovery'];single,overall=physical['single'],physical['all']
            scope = summary.get('physical_recovery_scope', '首组全部场景')
            fig.text(.09,.117,'物理漏检重捕率：本轨迹 '+_rate(single['recovery_rate'],single['recoveries'],single['lost_events'])+
                f'，{scope} '+_rate(overall['recovery_rate'],overall['recoveries'],overall['lost_events']),
                fontsize=9,color='#374151')
        footnote = ('图中时间/重捕率按可用更新统计；严格身份跟踪指标另列。紫菱形保留实际错配后的网络输出。\n'
            '未关联时蓝线留空；任一轨迹连续 5 帧没有可用测量即停止控制器。' if association_mode else
            '重捕率＝成功恢复次数／丢失事件数；连续漏检算一次事件。初始化不计入跟踪时间。\n'
            '蓝线在漏检及终止后留空；黑 × 用真实位置事后标记。任一目标连续漏检 5 帧即整组终止。')
        fig.text(.09, .064 if association_mode else .083, footnote,
            fontsize=9, color='#4b5563', linespacing=1.7)
        fig.text(.09, .014 if association_mode else .033, selection_note or '示例采用固定轨迹编号，仅用于展示；总体效果以全部场景指标为准。',
                 fontsize=8.5, color='#596273')
        fig.savefig(png, dpi=190, facecolor='white')
        plt.close(fig)
    metadata = {'trajectory_id': trajectory_id, 'scene_id': g, 'target_index': t,
        'dt_s': dt, 'history_frames': history, 'estimate_source': sources[0],
        'stop_reason': stop_reason,
        'association_mode': association_mode,'wrong_identity_updates':len(wrong_identity),
        'physical_recovery':summary.get('physical_recovery'),
        'single_trajectory': stats, 'all_scenes': {k: summary[k] for k in ('scenes',
            'mean_all_targets_duration_s', 'recoveries', 'lost_events', 'recovery_rate')},
        'maximum_evaluated_duration_s': (len(truth)-history)*dt,
        'selection_note': selection_note or 'fixed ID; otherwise minimum available trajectory ID',
        'no_interpolation_or_synthetic_estimates': True, 'truth_is_evaluation_only': True,
        'initial_measurements_drawn_as_neural_predictions': False,
        'xy_projection': True, 'zoom_time_index': center_index,
        'samples_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'figure_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
        'plot_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if dropout:
        metadata['injected_dropout_frames'] = dropout
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return png
