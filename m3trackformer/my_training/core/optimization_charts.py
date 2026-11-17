"""Scientific charts from recorded closed-loop optimization trials."""
import numpy as np

from .metric_charts import _plotting


def show_images(paths):
    """Display the two saved simulation figures in one blocking show call."""
    plt = _plotting()
    for path in paths:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(plt.imread(path))
        ax.axis('off')
        fig.tight_layout()
    plt.show()


def write_report(out, validation, tests, winner):
    """Readable report derived only from recorded trial metrics."""
    selected = next(r for r in tests if r['configuration']['id'] == winner)
    baseline = next(r for r in tests if r['configuration']['id'] == 'huber_base')
    best, old = selected['aggregate'], baseline['aggregate']
    lines = ['# 原 Transformer N-Net/R-Net 外围优化实测', '',
        f'验证集锁定方案：{selected["configuration"]["label"]}（{winner}）。', '',
        '15组预先列出的配置全部在验证集、两组固定噪声上运行。按完整跟踪成功率、平均同时跟踪时长、检测覆盖率依次选优；测试前锁定，随后比较8组入围配置。',
        '测试为17组轨迹组合 × 3组固定噪声，共51次运行；不等于51个独立几何场景。每组3个目标，共享12束；任一目标连续漏检5帧即整组终止。', '',
        '## 测试集结果', '',
        '|方案|完整成功率|平均同时跟踪时长/s|重捕率|N-Net检测帧RMSE/m|检测位置覆盖率|',
        '|---|---:|---:|---:|---:|---:|']
    for row in tests:
        c, m = row['configuration'], row['aggregate']
        mark = '（已选）' if c['id'] == winner else ''
        recovery = 'N/A' if m['recovery_rate'] is None else f'{m["recovery_rate"]*100:.2f}%'
        lines.append(f'|{c["label"]} / {c["id"]}{mark}|{m["success_rate"]*100:.2f}% '
            f'({m["success_count"]}/{m["scenes_with_noise_repeats"]})|{m["mean_all_targets_duration_s"]:.2f}|'
            f'{recovery}|{m["position_rmse_m_on_detections"]:.2f}|{m["position_evaluation_coverage"]*100:.2f}%|')
    lines += ['', f'相对当前Huber基准：成功率增加{(best["success_rate"]-old["success_rate"])*100:.2f}个百分点，'
        f'平均同时跟踪时间增加{best["mean_all_targets_duration_s"]-old["mean_all_targets_duration_s"]:.2f}秒；'
        f'重捕率变化{(best["recovery_rate"]-old["recovery_rate"])*100:+.2f}个百分点。',
        '提升有限：51次噪声重复中仅多1次完整成功，不能据此声称有统计显著或跨数据集收益；重捕率并未提升。', '',
        '## 验证集全部配置', '',
        '|配置|方案|完整成功率|平均同时跟踪时长/s|', '|---|---|---:|---:|']
    for row in validation:
        c, m = row['configuration'], row['aggregate']
        lines.append(f'|{c["id"]}|{c["label"]}|{m["success_rate"]*100:.2f}%|{m["mean_all_targets_duration_s"]:.4f}|')
    lines += ['', '## 算法和数据口径', '',
        '- 普通卡尔曼：按距离/SNR给出的名义协方差进行常速度测量预处理。',
        '- Huber调参：调整过程噪声和异常降权阈值。',
        '- Student-t式：用创新NIS计算 w=min(1,(df+3)/(df+NIS))，有效测量协方差为R/w。已选df=6；这是一次创新软降权近似，不是完整Student-t贝叶斯滤波。',
        '- 自适应过程噪声：只根据已获取的持续创新调大后续过程噪声。',
        '- 鲁棒IMM：两个不同过程噪声的常速度模型，混合状态和协方差；模式证据为鲁棒近似。',
        '- 外部温度：在原N/R输出后缩放logits，再执行原来的逐目标Softmax、求和、Top-12；网络和每目标波束排序不变。',
        '- 原N-Net/R-Net结构、权重、接口、失跟路由保持一致；位置曲线和RMSE均来自原N-Net位置头，滤波只处理已检测测量。',
        '- 真实数据为GNSS参考轨迹，原约5Hz插值为20Hz，不增加真实运动细节；雷达测量、SNR和异常值仍为仿真。',
        '- 多目标由不同实测片段组合，非真实同时采集多目标；检测仍使用理想身份，尚未加入未知身份关联。',
        '- RMSE只统计检测帧，各方案帧集合不同，需结合覆盖率理解；跟踪时长不含初始化，组终止后同伴轨迹截尾。',
        '- 这些测试记录此前已使用，不是全新盲测；当前不确定性主要来自有限记录及噪声重复。',
        '- 此前三组噪声中的首组Huber成功率为76.47%；本表68.63%是三组汇总，不能混用口径。',
        '- 每版单轨迹图固定首场景首目标、只绘首个噪声种子；它的时间和重捕率与三种子汇总不同，不代表平均效果。', '',
        '## 复现入口', '', '项目根目录执行：', '', '```powershell',
        r'.\.venv\Scripts\python.exe .\my_training\step10_single_target_tracking.py --best-real',
        '```', '',
        '--best-real：只运行锁定配置、首个噪声种子，保存新版本并显示轨迹和指标两张图；--no-show用于后台运行。',
        '--optimize-real：重新运行15组验证配置与8组测试配置；每次建立全新归档。', '',
        '对应配置和校验值见 current_optimized_tracking.json、protocol.json、selection_locked_before_test.json；完整指标见 test_comparison.csv。',
        '比较图为 optimization_comparison.png；所有历史快照保留，不覆盖旧版本。', '']
    (out/'results_report.md').write_text('\n'.join(lines), encoding='utf-8')


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
