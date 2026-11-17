"""Plots of real simulation decisions, including rejected and incorrect identities."""
import numpy as np

from .beam_identity import METHODS
from .metric_charts import _plotting

COLORS = ['#2470b5', '#e58824', '#25925a']


def render_budget(out, summary, audit, maximum, history, dt):
    plt = _plotting()
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    metrics = summary['identity_metrics']
    specs = [('accuracy_all_returns', '全部检出点的正确归属率'),
             ('precision_assigned', '已分配点的正确率'),
             ('assignment_coverage', '分配覆盖率'), ('accuracy_same_beam', '同束混叠点正确归属率')]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.8))
    for ax, (key, label) in zip(axes, specs):
        values = [metrics[k][key] for k in METHODS]
        ax.bar(range(2), [0 if v is None else v*100 for v in values], color=['#6c8cb2', '#20835a'])
        for j, value in enumerate(values):
            ax.text(j, 0 if value is None else value*100, 'N/A' if value is None else f'{value*100:.2f}%',
                    ha='center', va='bottom', fontsize=10)
        ax.set_xticks(range(2), ['最高贡献\n直接归属', '评分＋几何\n全局匹配'])
        ax.set(title=label, ylabel='%', ylim=(0, 115)); ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'身份判别审计：3目标、共享{summary["total_beams"]}束、3组固定噪声', fontsize=16)
    fig.text(.04, .025, '两种规则使用同一批匿名检出点；拒配计入全部点的错误项。过去历史由Huber基准维持正确关联，本版身份结果不反馈网络。', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .9)); fig.savefig(folder/'identity_metrics.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, (method, name) in zip(axes, METHODS.items()):
        counts = np.asarray(metrics[method]['confusion'])
        fractions = counts/np.maximum(counts.sum(-1, keepdims=True), 1)
        ax.imshow(fractions, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{counts[i,j]}\n{fractions[i,j]*100:.1f}%', ha='center', va='center',
                    color='white' if fractions[i,j] > .55 else '#222222', fontsize=10)
        ax.set_xticks(range(4), ['目标A', '目标B', '目标C', '拒配'])
        ax.set_yticks(range(3), ['真实A', '真实B', '真实C']); ax.set(title=name, xlabel='算法判断')
    fig.suptitle(f'{summary["total_beams"]}束：身份混淆矩阵（含全部检出点）', fontsize=16)
    fig.text(.05, .025, '目标A/B/C按每场景初始化轨迹的序号汇总，非机型分类；真实编号仅用于事后评分。', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .9)); fig.savefig(folder/'identity_confusion.png', dpi=170); plt.close(fig)

    rows = [r for r in audit.rows if r['scene_id'] == 0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    palette = ListedColormap(['#eeeeee', *COLORS, '#555555'])
    norm = BoundaryNorm(np.arange(-1.5, 4.5), palette.N)
    for ax, (method, label) in zip(axes, METHODS.items()):
        grid = np.full((3, maximum), -1)
        for row in rows:
            grid[row['true_target_index_evaluation_only'], row['time_index']-history] = row[method] if row[method] >= 0 else 3
        ax.imshow(grid, cmap=palette, norm=norm, aspect='auto', interpolation='nearest',
                  extent=(0, maximum*dt, 2.5, -.5))
        wrong = [r for r in rows if r[method] >= 0 and r[method] != r['true_target_index_evaluation_only']]
        if wrong:
            ax.scatter([r['time_s']+dt/2 for r in wrong], [r['true_target_index_evaluation_only'] for r in wrong],
                       s=9, c='#dd1538', marker='x', linewidths=.7)
        stop = audit.scene_frames[0]*dt
        ax.axvline(stop, color='#555555', ls='--', lw=.8)
        ax.set_yticks(range(3), ['真实A', '真实B', '真实C']); ax.set(title=label, xlim=(0, maximum*dt))
    axes[-1].set_xlabel('闭环开始后的时间 / 秒')
    handles = [Patch(color=COLORS[i], label=f'判为{letter}') for i, letter in enumerate('ABC')]
    handles += [Patch(color='#555555', label='拒配'), Patch(color='#eeeeee', label='无检出/未评估')]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(.5, .04), fontsize=9)
    fig.suptitle(f'固定首场景：逐帧身份判断（{summary["total_beams"]}束，噪声{audit.seed}）', fontsize=15)
    fig.text(.05, .01, '红×为已分配但身份错误；虚线为场景停止时间。此图不是关联结果反馈后的跟踪轨迹。', fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .93)); fig.savefig(folder/'identity_timeline.png', dpi=170); plt.close(fig)

    snap = audit.collision_snapshot or audit.snapshot
    if snap is not None:
        p = np.asarray(snap['probabilities'])
        relative = p/np.maximum(p.sum(0, keepdims=True), 1e-300)
        fig, ax = plt.subplots(figsize=(max(8, p.shape[1]*.7), 4.8))
        im = ax.imshow(relative, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        for i in range(3):
            for j in range(p.shape[1]):
                ax.text(j, i, f'{p[i,j]:.3f}\n({relative[i,j]*100:.0f}%)', ha='center', va='center',
                        fontsize=8, color='white' if relative[i,j] > .6 else '#222222')
        ax.set_xticks(range(p.shape[1]), snap['selected_beams_1based'])
        ax.set_yticks(range(3), ['目标A贡献', '目标B贡献', '目标C贡献'])
        ax.set(xlabel='按总评分排序的已选波束编号', title='原N/R概率贡献（括号为该波束内相对贡献）')
        fig.colorbar(im, ax=ax, label='相对贡献')
        detail = '；'.join(f'波束{b}来自真实{"ABC"[t]}→直接{"ABC"[a]} / 改进'
            f'{"ABC"[g] if g >= 0 else "拒配"}' for b,t,a,g in zip(snap['detected_beams_1based'],
            snap['true_source_evaluation_only'], snap['decisions']['beam_argmax'], snap['decisions']['beam_geometry']))
        fig.suptitle(f'评分展开示例：场景{snap["scene_id"]}，时间{(snap["time_index"]-history)*dt:.2f}s', fontsize=15)
        fig.text(.03, .05, detail, fontsize=9)
        fig.text(.03, .01, '若存在同束多回波，固定展示首个此类时刻；否则展示首个扫描。示例只用于解释，不代表平均效果。', fontsize=8.5)
        fig.tight_layout(rect=(0, .13, 1, .92)); fig.savefig(folder/'beam_contributions.png', dpi=170); plt.close(fig)


def render_comparison(root, records):
    plt = _plotting()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    labels = [f'{r["total_beams"]}束\n'+('最高贡献' if name == 'beam_argmax' else '评分＋几何')
              for r in records for name in METHODS]
    colors = ['#6c8cb2', '#20835a']*len(records)
    for ax, (key, title) in zip(axes, [('accuracy_all_returns','全部检出点正确归属率'),
                                      ('assignment_coverage','分配覆盖率'), ('accuracy_same_beam','同束混叠正确归属率')]):
        values = [r['identity_metrics'][name][key] for r in records for name in METHODS]
        ax.bar(range(len(labels)), [0 if v is None else v*100 for v in values], color=colors)
        for i,v in enumerate(values):
            ax.text(i, 0 if v is None else v*100, 'N/A' if v is None else f'{v*100:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set(title=title, ylabel='%', ylim=(0,115)); ax.grid(axis='y', alpha=.2)
    fig.suptitle('你的评分归属方案与改进版：同一Huber历史条件下比较', fontsize=15)
    fig.text(.04,.025,'每个预算内规则共享相同检出点；4束与12束的检出数量和场景存活时间不同。拒配计入全部点的错误项。',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.9)); fig.savefig(root/'identity_comparison.png',dpi=180); plt.close(fig)
