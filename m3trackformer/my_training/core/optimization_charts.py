"""Scientific charts from recorded closed-loop optimization trials."""
import numpy as np

from .metric_charts import _plotting


def render_trial(out, label, metrics):
    plt = _plotting()
    specs = [('success_rate', '整组完整跟踪成功率', '%', 100),
        ('mean_all_targets_duration_s', '平均同时跟踪时长', '秒', 1),
        ('recovery_rate', '重捕率', '%', 100),
        ('position_rmse_m_on_detections', 'N-Net检测帧位置RMSE', 'm', 1)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6))
    for ax, (key, title, unit, scale) in zip(axes, specs):
        value = metrics[key]
        ax.bar([0], [0 if value is None else value*scale], color='#2768a8', width=.45)
        ax.set_xticks([]); ax.set(title=title, ylabel=unit)
        ax.set_ylim(0, 110 if unit == '%' else max(1., (value or 0)*scale)*1.3)
        ax.text(0, (value or 0)*scale, 'N/A' if value is None else f'{value*scale:.2f}', ha='center', va='bottom')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(label, fontsize=15)
    fig.text(.05, .02, '汇总当前分区的全部固定噪声种子；位置误差仅统计检测帧。N-Net/R-Net和总波束预算保持不变。', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .9))
    (out/'figures').mkdir(exist_ok=True)
    fig.savefig(out/'figures/trial_metrics.png', dpi=160, facecolor='white'); plt.close(fig)


def render_comparison(out, rows, winner):
    plt = _plotting()
    labels = [r['configuration']['label']+'\n'+r['configuration']['id'] for r in rows]
    colors = ['#22865d' if r['configuration']['id'] == winner else '#668fb5' for r in rows]
    specs = [('success_rate', '完整跟踪成功率', '%', 100),
        ('mean_all_targets_duration_s', '平均同时跟踪时间', '秒', 1),
        ('recovery_rate', '重捕率', '%', 100),
        ('position_rmse_m_on_detections', 'N-Net检测帧位置RMSE', 'm', 1),
        ('position_evaluation_coverage', '检测位置评价覆盖率', '%', 100),
        ('filter_ms_per_detected_frame', '测量预处理耗时', 'ms/检测目标帧', 1)]
    fig, axes = plt.subplots(3, 2, figsize=(15, max(12, len(rows)*1.9)))
    for ax, (key, title, unit, scale) in zip(axes.flat, specs):
        values = [r['aggregate'].get(key) for r in rows]
        ax.barh(np.arange(len(rows)), [np.nan if v is None else v*scale for v in values], color=colors)
        ax.set_yticks(range(len(rows)), labels, fontsize=8.5)
        ax.invert_yaxis()
        maximum = max([1e-5]+[v*scale for v in values if v is not None])
        ax.set_xlim(0, max(110 if unit == '%' else 0, maximum*1.22))
        for i, v in enumerate(values):
            ax.text(0 if v is None else v*scale, i, ' N/A' if v is None else f' {v*scale:.2f}', va='center', fontsize=8.5)
        ax.set(title=title, xlabel=unit); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('外围算法优化对照：原 Transformer N-Net/R-Net 保持不变', fontsize=17, y=.99)
    fig.text(.045, .012, '绿色是测试前已由验证集锁定的方案。各方案使用相同的17组真实轨迹、3组噪声；重复噪声实验不等于新增独立场景。\n'
        '各方法检测帧集合不同，RMSE必须结合覆盖率解读。测量预处理耗时不包括神经网络推理和位置回归。', fontsize=9)
    fig.tight_layout(rect=(.01, .055, .99, .97), h_pad=2.5, w_pad=3)
    fig.savefig(out/'optimization_comparison.png', dpi=170, facecolor='white'); plt.close(fig)
