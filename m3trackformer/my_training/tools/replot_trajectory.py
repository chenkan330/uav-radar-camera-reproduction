"""Reproduce a trajectory figure from actual saved simulation samples in a NEW version."""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.common import BASE, load_split, new_run, read_json, write_json
from core.multi_target import make_scenes
from core.trajectory_figure import save_trajectory_figure
from core.versions import save_version


def replot(prior, version):
    prior = Path(prior).resolve()
    if (BASE/'experiments'/version).exists():
        raise FileExistsError('Use a new archive name; old versions remain locked')
    summary = read_json(prior/'summary.json')
    saved = read_json(prior/'figures/single_trajectory_tracking.json')
    with (prior/'figures/single_trajectory_tracking_samples.csv').open(encoding='utf-8-sig', newline='') as stream:
        samples = list(csv.DictReader(stream))
    trace = []
    for row in samples:
        if row['phase'] == 'not_evaluated':
            continue
        row = dict(row)
        for key in ('trajectory_id', 'scene_id', 'target_index', 'time_index'):
            row[key] = int(row[key])
        for key in ('hit', 'recovered'):
            row[key] = row[key] == 'True'
        for key in [k for k in row if k.endswith('_m')] + ['time_s']:
            row[key] = float(row[key]) if row[key] else None
        trace.append(row)
    v4 = summary.get('version') == 'v4_maneuver_uncertainty'
    nr_v4 = summary.get('version') == 'v4_original_nr_maneuver'
    scene_info = summary['tracking'] if v4 else summary
    data = load_split(summary['source'])
    scenes, _, _ = make_scenes(data, scene_info['targets'], seed=summary.get('seed', 20261115))
    if scenes['trajectory_id'].tolist() != scene_info['scene_membership']:
        raise ValueError('Cannot reproduce original scene membership')
    name = 'motion_probability' if v4 else 'nr_robust' if nr_v4 else 'joint'
    with (prior/f'{name}_targets.csv').open(encoding='utf-8-sig', newline='') as stream:
        targets = list(csv.DictReader(stream))
    metrics = scene_info['methods'][name] if v4 else scene_info['results'][name]
    out = new_run(prior.parent)
    skip = {'single_trajectory_tracking.png', 'single_trajectory_tracking.json',
            'single_trajectory_tracking_samples.csv'}
    shutil.copytree(prior, out, dirs_exist_ok=True,
                    ignore=lambda directory, names: [n for n in names if n in skip])
    # Large local assets were copied exactly; point new version references to its own assets.
    for key, relative in (('source', 'data/test.npz'), ('motion_checkpoint', 'best_motion_forecaster.pth')):
        if (out/relative).exists():
            summary[key] = str((out/relative).resolve())
    summary['trajectory_figure_revision'] = {
        'source_run': str(prior), 'source_summary_sha256': hashlib.sha256((prior/'summary.json').read_bytes()).hexdigest(),
        'simulation_rerun': False, 'uses_exact_saved_simulation_samples': True,
        'changes': 'only figure layout; no tracking algorithm, data, model or metrics changes'}
    write_json(out/'summary.json', summary)
    png = save_trajectory_figure(out, scenes, trace, targets, metrics,
        dt=saved['dt_s'], history=saved['history_frames'], trajectory_id=saved['trajectory_id'],
        title='V4：运动概率预测与鲁棒测量处理' if v4 else
            '重做 V4：原 Transformer N-Net/R-Net＋鲁棒预处理' if nr_v4 else '多目标联合跟踪：单轨迹示例',
        selection_note=saved['selection_note'])
    (out/'report.txt').write_text(
        '单轨迹效果图：由实际闭环仿真保存的逐帧样本重绘，仅调整显示排版。\n'
        f'图：{png.name}\n跟踪时间：{saved["single_trajectory"]["tracking_duration_s"]:.2f}秒\n'
        f'单条重捕：{saved["single_trajectory"]["recoveries"]}/{saved["single_trajectory"]["lost_events"]}\n'
        f'全部场景平均同时跟踪时间：{metrics["mean_all_targets_duration_s"]:.6f}秒\n'
        f'全部场景重捕：{metrics["recoveries"]}/{metrics["lost_events"]}\n'
        f'蓝线：{"原N-Net实际位置输出" if nr_v4 else "因果滤波当前位置"}；真实轨迹及漏检位置只用于事后绘图。\n'
        '下一步：无身份标签检测输入；保持当前数据、权重、预算作为关联前基线。\n', encoding='utf-8')
    archive = save_version(version, out, 'V4单轨迹仿真图：固定示例、跟踪时间、重捕率；复用实际闭环记录，仅调整排版。')
    if v4 or nr_v4:
        (out.parent/'latest_complete_run.txt').write_text(out.name, encoding='utf-8')
    print(f'图：{png.resolve()}\n归档：{archive.resolve()}', flush=True)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--archive-version', required=True)
    args = parser.parse_args()
    replot(args.run, args.archive_version)
