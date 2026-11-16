"""闭环测试结束后绘制状态时间图和持续跟踪比例曲线。"""
from pathlib import Path

import numpy as np


POLICIES = ('nnet_only', 'nnet_rnet')
LABELS = ('仅 N-Net', 'N-Net + R-Net')


def tracking_survival(rows, maximum_duration_s):
    """仅在彻底失跟时下降；完成测试的轨迹在右端点仍计为成功。"""
    failures = np.asarray([r['duration_s'] for r in rows if not r['success']])
    times, counts = np.unique(failures, return_counts=True)
    x = np.r_[0., times, maximum_duration_s]
    y = np.r_[100., 100. * (1. - counts.cumsum() / len(rows))]
    return x, np.r_[y, y[-1]]


def save_tracking_plots(output, rows_by_policy, events_by_policy, dt,
                        history, maximum_duration_s, *, show=True,
                        trajectory_id=None):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    output = Path(output)
    indexed = {name: {r['trajectory_id']: r for r in rows_by_policy[name]}
               for name in POLICIES}
    ids = set(indexed[POLICIES[0]])
    if not ids or ids != set(indexed[POLICIES[1]]):
        raise ValueError('两种方案必须使用同一批非空测试轨迹')
    automatic_selection = trajectory_id is None
    if automatic_selection:
        # 确定性选择有恢复过程的示例；整体曲线始终统计全部轨迹。
        trajectory_id = max(sorted(ids), key=lambda i: (
            indexed['nnet_rnet'][i]['recoveries'],
            indexed['nnet_rnet'][i]['recoveries'] - indexed['nnet_only'][i]['recoveries'],
            indexed['nnet_rnet'][i]['missed_blocks']))
    if trajectory_id not in ids:
        raise ValueError(f'当前测试集不包含轨迹 {trajectory_id}')

    steps = int(round(maximum_duration_s / dt))
    status = np.full((2, steps), -1, dtype=np.int8)
    selected_events = {}
    recovery_times = {}
    for row, name in enumerate(POLICIES):
        events = sorted((e for e in events_by_policy[name]
                         if e['trajectory_id'] == trajectory_id),
                        key=lambda e: e['time_index'])
        if not events:
            raise ValueError(f'{name} 缺少逐时刻记录，无法绘制状态图')
        selected_events[name] = events
        slots = np.asarray([e['time_index'] - history for e in events])
        status[row, slots] = [int(e['hit']) for e in events]
        recovery_times[name] = [(e['time_index'] - history + .5) * dt
                                for e in events
                                if e['hit'] and e['prior_consecutive_misses'] > 0]

    focus_events = selected_events['nnet_rnet']
    centers = recovery_times['nnet_rnet']
    if not centers:
        centers = [(e['time_index'] - history + .5) * dt
                   for e in focus_events if not e['hit']]
    if not centers:
        centers = [(e['time_index'] - history + .5) * dt
                   for e in selected_events['nnet_only'] if not e['hit']]
    center = centers[0] if centers else maximum_duration_s / 2
    width = min(2., maximum_duration_s)
    left = max(0., min(center - width / 2, maximum_duration_s - width))
    right = left + width
    cmap = ListedColormap(['#dce1e7', '#e35d6a', '#2caa78'])
    fig_status, axes = plt.subplots(2, 1, figsize=(13, 6.5))
    for ax in axes:
        ax.pcolormesh(np.arange(steps + 1) * dt, [-.5, .5, 1.5], status,
                      cmap=cmap, vmin=-1.5, vmax=1.5, shading='flat', rasterized=True)
        for row, name in enumerate(POLICIES):
            recovered = recovery_times[name]
            ax.scatter(recovered, [row] * len(recovered), marker='*', s=95,
                       color='#165abc', edgecolors='white', linewidths=.45, zorder=4)
            record = indexed[name][trajectory_id]
            if not record['success']:
                ax.scatter([record['duration_s']], [row], marker='X', s=85,
                           color='#202a35', zorder=5, clip_on=False)
        ax.set_yticks([0, 1], [f'{LABELS[i]}（恢复 {indexed[name][trajectory_id]["recoveries"]} 次）'
                               for i, name in enumerate(POLICIES)])
        ax.set_ylim(1.5, -.5)
        ax.set_xlabel('闭环评估时间（秒）')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_xlim(0, maximum_duration_s)
    axes[0].set_title('完整跟踪过程', loc='left', fontsize=11)
    axes[0].axvspan(left, right, color='#165abc', alpha=.10)
    axes[1].set_xlim(left, right)
    axes[1].set_title(f'局部放大：每格 {dt:g} 秒，星号表示漏检后再次命中', loc='left', fontsize=11)
    selection = '自动选取 R-Net 方案恢复次数最多的示例' if automatic_selection else '手动指定示例'
    fig_status.suptitle(f'跟踪状态时间图  |  轨迹 {trajectory_id}', fontsize=16, fontweight='bold')
    fig_status.legend(handles=[
        Patch(color='#2caa78', label='命中'), Patch(color='#e35d6a', label='漏检'),
        Patch(color='#dce1e7', label='终止后不再评估'),
        Line2D([], [], marker='*', color='#165abc', linestyle='None', markersize=11, label='恢复成功'),
        Line2D([], [], marker='X', color='#202a35', linestyle='None', markersize=8, label='跟踪终止'),
    ], loc='upper center', bbox_to_anchor=(.55, .925), ncol=5, frameon=False)
    fig_status.text(.5, .02, f'{selection}；横轴不含最初 {history} 个时刻的初始化观测。',
                    ha='center', fontsize=9, color='#566171')
    fig_status.tight_layout(rect=(0, .05, 1, .86), h_pad=1.6)

    fig_survival, ax = plt.subplots(figsize=(10, 6))
    for name, label, color, linestyle in zip(
            POLICIES, LABELS, ['#2563a6', '#e58a28'], ['-', '--']):
        rows = rows_by_policy[name]
        x, y = tracking_survival(rows, maximum_duration_s)
        mean = np.mean([r['duration_s'] for r in rows])
        ax.step(x, y, where='post', linewidth=2.5, color=color, linestyle=linestyle,
                label=f'{label}：结束时 {y[-1]:.2f}%，平均 {mean:.2f} 秒')
    ax.set(xlim=(0, maximum_duration_s), ylim=(0, 103),
           xlabel='闭环评估时间（秒）', ylabel='尚未彻底失跟的轨迹比例（%）',
           title=f'持续跟踪比例曲线  |  全部 {len(ids)} 段轨迹')
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', frameon=False)
    fig_survival.text(.5, .025,
                      '仅在连续漏检达到终止条件时下降；完成全程的轨迹在右端点仍计为成功。',
                      ha='center', fontsize=9, color='#566171')
    fig_survival.tight_layout(rect=(0, .06, 1, 1))

    figures = (fig_status, fig_survival)
    filenames = ('tracking_status.png', 'tracking_survival.png')
    for fig, filename in zip(figures, filenames):
        fig.savefig(output / filename, dpi=180, facecolor='white')
    return {'tracking_status': filenames[0], 'tracking_survival': filenames[1],
            'trajectory_id': int(trajectory_id),
            'trajectory_selection': 'most_rnet_recoveries' if automatic_selection else 'manual',
            'time_origin': 'first_evaluated_block_after_initialization'}, figures


def finish_tracking_plots(figures, *, show=True):
    import matplotlib.pyplot as plt
    try:
        if show:
            for fig, title in zip(figures, ('跟踪状态时间图', '持续跟踪比例曲线')):
                fig.canvas.manager.set_window_title(title)
            plt.show()
    finally:
        for fig in figures:
            plt.close(fig)
