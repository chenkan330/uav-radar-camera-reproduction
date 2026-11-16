"""原文第V节：共享单目标权重、概率求和选束及独立均分基线。"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .common import (BASE, audit_partitions, load_nnet, load_split, new_run,
                     normalized_history, read_json, state_digest, write_json, write_rows)
from .models import JournalRNet

POLICIES = ('joint', 'decoupled')
LABELS = {'joint': '论文联合选束', 'decoupled': '独立均分基线'}


def make_scenes(data, targets, seed=20261115, limit=None):
    """仅在同一个数据分区内，无放回地组成场景；不会虚增独立测试数量。"""
    ids = data['trajectory_id']
    if targets < 1 or len(ids) < targets or len(np.unique(ids)) != len(ids):
        raise ValueError('目标数无效、测试轨迹不足或轨迹ID重复')
    order = np.random.default_rng(seed).permutation(len(ids))
    count = len(ids) // targets
    if limit is not None:
        if not 1 <= limit <= count:
            raise ValueError(f'无放回组合最多只有 {count} 个场景')
        count = limit
    indices = order[:count * targets].reshape(count, targets)
    scenes = {key: data[key][indices] for key in ('measurement', 'beam', 'trajectory_id')}
    if 'position' in data:
        scenes['position'] = data['position'][indices]
    return scenes, indices, ids[order[count * targets:]].tolist()


def joint_beams(logits, budget):
    """先逐目标Softmax，再对概率求和；不对logits求和。"""
    if logits.ndim != 3 or not 1 <= budget <= logits.shape[-1]:
        raise ValueError('需要[场景,目标,码本]分数以及合法总波束数')
    scores = logits.double().softmax(-1).sum(1)
    return scores.argsort(dim=-1, descending=True, stable=True)[:, :budget]


def _prediction(nnet, rnet, x, valid, failed, counts):
    result = nnet(x, valid)['logits'][:, 0]
    lost = counts > 0
    if rnet is not None and lost.any():
        # 未失跟目标不进入LE/FF，避免零输入经过带bias的网络后改变N-Net输出。
        result[lost] = rnet(nnet, x[lost], valid[lost], failed[lost], counts[lost])[:, 0]
    return result


def simulate(nnet, rnet, checkpoint, scenes, device, dt, budget, policy,
             record_events=True):
    """同场景任一目标连续失跟达到阈值即终止；未失败的同伴记为截尾。"""
    if policy not in POLICIES:
        raise ValueError('未知多目标策略')
    measured, beams = scenes['measurement'], scenes['beam'] - 1
    groups, targets, steps = beams.shape
    c = nnet.config
    h, failure_limit = c.history, c.failure_limit
    if (dt <= 0 or steps <= h or failure_limit > h or failure_limit < 2
            or budget < 1 or budget > c.classes
            or (policy == 'decoupled' and budget < targets)):
        raise ValueError('采样、轨迹长度、失败阈值或波束预算不合法')
    if measured.shape != (*beams.shape, 3) or np.any(beams < 0) or np.any(beams >= c.classes):
        raise ValueError('测量或1-based码本标签不合法')
    valid = np.zeros_like(beams, dtype=bool)
    valid[:, :, :h] = True
    active = np.ones(groups, dtype=bool)
    counts = np.zeros((groups, targets), dtype=np.int64)
    failed = np.full((groups, targets, failure_limit - 1, budget), -1, dtype=np.int64)
    evaluated = np.zeros(groups, dtype=np.int64)
    misses = np.zeros_like(counts)
    recoveries = np.zeros_like(counts)
    delay_blocks = np.zeros_like(counts)
    squared_error = np.zeros_like(counts, dtype=float)
    position_count = np.zeros_like(counts)
    events, scans, timings = [], [], []
    with torch.inference_mode():
        for step in range(h, steps):
            live = np.flatnonzero(active)
            if not len(live):
                break
            mask = valid[live, :, step-h:step]
            history = normalized_history(measured[live, :, step-h:step], mask,
                                         checkpoint['normalization'])
            x = torch.from_numpy(history).to(device)
            v = torch.from_numpy(mask).to(device)
            f = torch.from_numpy(failed[live]).to(device)
            prior = counts[live].copy()
            ct = torch.from_numpy(prior).to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if policy == 'joint':
                logits = _prediction(nnet, rnet, x.flatten(0, 1), v.flatten(0, 1),
                                     f.flatten(0, 1), ct.flatten()).reshape(len(live), targets, -1)
                chosen = joint_beams(logits, budget).cpu().numpy()
                feedback = np.broadcast_to(chosen[:, None], (len(live), targets, budget)).copy()
                swept = chosen
            else:
                # 原文VI-D：各目标顺序推理，并各取floor(Bs/I)束；余数不另行分配。
                quota = budget // targets
                feedback = np.full((len(live), targets, budget), -1, dtype=np.int64)
                for target in range(targets):
                    logits = _prediction(nnet, rnet, x[:, target], v[:, target],
                                         f[:, target], ct[:, target])
                    feedback[:, target, :quota] = logits.argsort(
                        dim=-1, descending=True, stable=True)[:, :quota].cpu().numpy()
                swept = feedback[:, :, :quota].reshape(len(live), targets * quota)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000
            timings.append((elapsed, len(live)))

            # 真值仅在选束结束后用于模拟环境。独立基线只使用分配给本目标的扫描反馈。
            hit = (feedback == beams[live, :, step, None]).any(-1)
            recovered = hit & (prior > 0)
            recoveries[live] += recovered
            delay_blocks[live] += np.where(recovered, prior, 0)
            misses[live] += ~hit
            evaluated[live] += 1
            after = np.where(hit, 0, prior + 1)
            counts[live] = after
            scene_failed = (after >= failure_limit).any(1)
            active[live[scene_failed]] = False
            for row, scene in enumerate(live):
                for target in range(targets):
                    if hit[row, target]:
                        valid[scene, target, step] = True
                        failed[scene, target] = -1
                    elif prior[row, target] < failure_limit - 1:
                        failed[scene, target, prior[row, target]] = feedback[row, target]
                    if record_events:
                        events.append({'policy': policy, 'scene_id': int(scene),
                                       'target_index': target,
                                       'trajectory_id': int(scenes['trajectory_id'][scene, target]),
                                       'time_index': step, 'time_s': (step-h)*dt,
                                       'mode': 'R' if prior[row, target] else 'N',
                                       'group_mode': 'R' if prior[row].any() else 'N',
                                       'prior_misses': int(prior[row, target]), 'hit': bool(hit[row, target]),
                                       'recovered': bool(recovered[row, target]),
                                       'post_misses': int(after[row, target]),
                                       'true_beam_1based': int(beams[scene, target, step]+1),
                                       'snr_db': float(scenes['snr_db'][scene, target, step]) if 'snr_db' in scenes else None,
                                       'measurement_outlier': bool(scenes['outlier_mask'][scene, target, step]) if 'outlier_mask' in scenes else None,
                                       'feedback_beams_1based': ','.join(str(b+1) for b in feedback[row, target] if b >= 0),
                                       'scene_terminated': bool(scene_failed[row])})
                if record_events:
                    scans.append({'policy': policy, 'scene_id': int(scene), 'time_index': step,
                                  'beam_slots_used': int(swept.shape[1]),
                                  'unique_beams': len(np.unique(swept[row])),
                                  'beams_1based': ','.join(str(b+1) for b in swept[row])})

            # 只有环境返回命中测量以后才估计当前位置；坐标估计不反馈为虚构测量。
            good_scene, good_target = np.nonzero(hit)
            if 'position' in scenes and len(good_scene):
                gs = live[good_scene]
                vm = valid[gs, good_target, step-h+1:step+1]
                hm = normalized_history(measured[gs, good_target, step-h+1:step+1], vm,
                                        checkpoint['normalization'])
                positions = nnet(torch.from_numpy(hm).to(device), torch.from_numpy(vm).to(device))['position']
                errors = positions.cpu().numpy() - scenes['position'][gs, good_target, step]
                squared_error[gs, good_target] += (errors**2).sum(1)
                position_count[gs, good_target] += 1

    scene_rows, target_rows = [], []
    for scene in range(groups):
        scene_rows.append({'policy': policy, 'scene_id': scene, 'success': bool(active[scene]),
                           'duration_s': float(evaluated[scene]*dt),
                           'evaluated_blocks': int(evaluated[scene]),
                           'recoveries': int(recoveries[scene].sum()),
                           'missed_target_blocks': int(misses[scene].sum()),
                           'terminal_failed_targets': int((counts[scene] >= failure_limit).sum())})
        for target in range(targets):
            failed_here = bool(counts[scene, target] >= failure_limit)
            target_rows.append({'policy': policy, 'scene_id': scene, 'target_index': target,
                                'trajectory_id': int(scenes['trajectory_id'][scene, target]),
                                'terminal_failure': failed_here,
                                'censored_by_peer_failure': not active[scene] and not failed_here,
                                'observed_duration_s': float(evaluated[scene]*dt),
                                'missed_blocks': int(misses[scene, target]),
                                'lost_events': int(recoveries[scene, target]+(counts[scene, target] > 0)),
                                'recoveries': int(recoveries[scene, target]),
                                'recovery_delay_blocks_sum': int(delay_blocks[scene, target]),
                                'position_squared_error_sum': float(squared_error[scene, target]),
                                'position_count': int(position_count[scene, target])})
    return scene_rows, target_rows, events, scans, timings


def summarize(scene_rows, target_rows, timings, dt, budget, targets, policy):
    rec = sum(r['recoveries'] for r in target_rows)
    lost = sum(r['lost_events'] for r in target_rows)
    positions = sum(r['position_count'] for r in target_rows)
    return {'scenes': len(scene_rows), 'targets_per_scene': targets,
            'success_rate': float(np.mean([r['success'] for r in scene_rows])),
            'mean_all_targets_duration_s': float(np.mean([r['duration_s'] for r in scene_rows])),
            'recoveries': rec, 'lost_events': lost, 'recovery_rate': rec/lost if lost else None,
            'mean_recovery_delay_s': sum(r['recovery_delay_blocks_sum'] for r in target_rows)*dt/rec if rec else None,
            'position_rmse_m_on_detections': (sum(r['position_squared_error_sum'] for r in target_rows)/positions)**.5 if positions else None,
            'beam_slots_per_scene_block': budget if policy == 'joint' else (budget//targets)*targets,
            'mean_prediction_batch_ms': float(np.mean([t[0] for t in timings])),
            'amortized_prediction_ms_per_scene': float(np.mean([t[0]/t[1] for t in timings])),
            'latency_scope': 'beam_prediction_and_selection_only_excludes_position_regression_and_IO'}


def save_plots(out, results, events, targets, dt, maximum, show, failure_limit=5):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    scene = max(range(len(results['joint'][0])), key=lambda i: results['joint'][0][i]['recoveries'])
    steps = int(round(maximum/dt))
    state = np.full((len(POLICIES)*targets, steps), -1)
    recovery, endings = [], []
    for row, policy in enumerate(POLICIES):
        for e in events:
            if e['policy'] != policy or e['scene_id'] != scene:
                continue
            lane = row*targets + e['target_index']
            index = int(round(e['time_s']/dt))
            state[lane, index] = int(e['hit'])
            if e['recovered']:
                recovery.append(((index+.5)*dt, lane))
            if e['post_misses'] >= failure_limit:
                endings.append(((index+1)*dt, lane))
    centers = [p[0] for p in recovery]
    if not centers:
        centers = (np.flatnonzero((state == 0).any(0))[:1]*dt).tolist()
    center = centers[0] if centers else maximum/2
    width = min(2., maximum)
    left = max(0, min(center-width/2, maximum-width))
    fig, axes = plt.subplots(2, 1, figsize=(13, max(6.5, targets*1.3)))
    for ax in axes:
        ax.pcolormesh(np.arange(steps+1)*dt, np.arange(len(state)+1)-.5, state,
                      cmap=ListedColormap(['#dce1e7', '#e35d6a', '#2caa78']),
                      vmin=-1.5, vmax=1.5, shading='flat', rasterized=True)
        if recovery:
            ax.scatter(*zip(*recovery), marker='*', color='#165abc', s=55, edgecolors='white', linewidths=.3)
        if endings:
            ax.scatter(*zip(*endings), marker='X', color='#202a35', s=60, clip_on=False)
        ax.set_yticks(range(len(state)), [f'{LABELS[p]} / 目标 {i+1}' for p in POLICIES for i in range(targets)])
        ax.set_ylim(len(state)-.5, -.5)
        ax.set_xlabel('初始化后的闭环评估时间（秒）')
    axes[0].set_xlim(0, maximum)
    axes[0].set_title('完整过程；任一目标彻底失跟即终止整个场景', loc='left', fontsize=10)
    axes[1].set_xlim(left, left+width)
    axes[1].set_title('局部放大', loc='left', fontsize=10)
    fig.suptitle(f'多目标跟踪状态  |  场景 {scene}（按联合方案恢复次数选例）', fontsize=15)
    fig.legend(handles=[Patch(color='#2caa78', label='命中'), Patch(color='#e35d6a', label='漏检'),
                        Patch(color='#dce1e7', label='场景终止后不评估'),
                        Line2D([], [], marker='*', color='#165abc', linestyle='None', label='恢复'),
                        Line2D([], [], marker='X', color='#202a35', linestyle='None', label='失败')],
               loc='upper center', bbox_to_anchor=(.56, .94), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    curve, ax = plt.subplots(figsize=(10, 6))
    for policy, color, style in zip(POLICIES, ['#2563a6', '#e58a28'], ['-', '--']):
        rows = results[policy][0]
        failures, counts = np.unique([r['duration_s'] for r in rows if not r['success']], return_counts=True)
        y = np.r_[100., 100*(1-counts.cumsum()/len(rows))]
        ax.step(np.r_[0., failures, maximum], np.r_[y, y[-1]], where='post',
                color=color, linestyle=style, linewidth=2.3, label=LABELS[policy])
    ax.set(xlabel='初始化后的闭环评估时间（秒）', ylabel='全部目标尚未彻底失跟的场景比例（%）',
           xlim=(0, maximum), ylim=(0, 103), title='多目标同时持续跟踪比例')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    curve.tight_layout()
    for f, filename in [(fig, 'tracking_status.png'), (curve, 'tracking_survival.png')]:
        f.savefig(out/filename, dpi=180, facecolor='white')
    return (fig, curve)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', type=int, default=3)
    p.add_argument('--total-beams', type=int, default=12)
    p.add_argument('--scenes', type=int)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--rnet-checkpoint', '--checkpoint', dest='checkpoint', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--measurement-regime', choices=['original', 'stress'], default='original')
    p.add_argument('--noise-seed', type=int, default=20261116)
    p.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_multi_target')
    a = p.parse_args(argv)
    if a.targets < 1 or a.total_beams <= a.targets:
        p.error('按原文设置：目标数为正，且总波束数须大于目标数')
    if a.checkpoint is None:
        current = BASE/'outputs/current_journal_experiment.json'
        if not current.exists():
            p.error('没有现有正式版权重，请使用--rnet-checkpoint指定权重')
        a.checkpoint = Path(read_json(current)['rnet_checkpoint'])
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('当前环境不可使用CUDA')
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if a.device == 'auto' else a.device)
    torch.set_num_threads(4)
    saved = torch.load(a.checkpoint, map_location=device, weights_only=False)
    if saved.get('format') != 'journal_equations_v1':
        p.error('多目标模式须使用正式版单目标网络权重，不能使用旧教学版权重')
    nnet, ck = load_nnet(saved['nnet_checkpoint'], device)
    if saved['model_config'] != ck['model_config']:
        p.error('N-Net与R-Net模型配置不一致')
    if a.total_beams > nnet.config.classes:
        p.error('总波束数不能超过码本大小')
    rnet = JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(saved['model'])
    data_root = a.data or Path(ck['data'])
    cfg = read_json(data_root/'config.json')
    partition = audit_partitions(data_root)
    source = data_root/f'{a.split}.npz'
    data = load_split(source)
    try:
        scenes, indices, unused = make_scenes(data, a.targets, a.seed, a.scenes)
    except ValueError as error:
        p.error(str(error))
    noise_metadata = None
    if a.measurement_regime == 'stress':
        from .measurement_quality import NoiseConfig, perturb_scenes
        scenes, noise_metadata = perturb_scenes(scenes, NoiseConfig(), a.noise_seed)
    before = (state_digest(nnet), state_digest(rnet))
    out = new_run(a.output_root)
    np.savez_compressed(out/'scene_membership.npz', source_indices=indices,
                        trajectory_id=scenes['trajectory_id'])
    if noise_metadata is not None:
        np.savez_compressed(out/'measurement_realization.npz', **{
            key: scenes[key] for key in ('measurement', 'measurement_sigma', 'snr_db', 'outlier_mask')})
        write_json(out/'noise_config.json', noise_metadata)
    print(f'多目标：{len(indices)}个场景 × {a.targets}个目标，总预算{a.total_beams}束；设备{device}', flush=True)
    # 预热只访问初始化历史，不写入闭环状态，不计入时延统计。
    h = nnet.config.history
    with torch.inference_mode():
        v = np.ones((1, h), bool)
        x = normalized_history(scenes['measurement'][:1, 0, :h], v, ck['normalization'])
        warm_x, warm_v = torch.from_numpy(x).to(device), torch.from_numpy(v).to(device)
        warm_failed = torch.full((1, nnet.config.failure_limit-1, a.total_beams), -1,
                                 dtype=torch.long, device=device)
        warm_failed[:, 0] = torch.arange(a.total_beams, device=device)
        for _ in range(3):
            nnet(warm_x, warm_v)
            rnet(nnet, warm_x, warm_v, warm_failed, torch.ones(1, dtype=torch.long, device=device))
    results, summaries, all_events, all_scans = {}, {}, [], []
    for policy in POLICIES:
        print(f'正在运行：{LABELS[policy]}', flush=True)
        result = simulate(nnet, rnet, ck, scenes, device, cfg['dt'], a.total_beams, policy)
        results[policy] = result
        summaries[policy] = summarize(result[0], result[1], result[4], cfg['dt'], a.total_beams, a.targets, policy)
        write_rows(out/f'{policy}_scenes.csv', result[0])
        write_rows(out/f'{policy}_targets.csv', result[1])
        all_events.extend(result[2]); all_scans.extend(result[3])
        print(summaries[policy], flush=True)
    if before != (state_digest(nnet), state_digest(rnet)):
        raise RuntimeError('推理过程意外修改了网络权重')
    write_rows(out/'tracking_events.csv', all_events)
    write_rows(out/'beam_scans.csv', all_scans)
    maximum = (scenes['beam'].shape[-1]-h)*cfg['dt']
    figures = save_plots(out, results, all_events, a.targets, cfg['dt'], maximum,
                         not a.no_show, nnet.config.failure_limit)
    notes = [
        '按正式版第V节实现共享网络、逐目标N/R分支及Softmax后概率求和Top-Bs。',
        '独立基线按原文VI-D每目标floor(Bs/I)束、顺序推理，仅使用本目标分配的扫描反馈；重复方向仍消耗扫描名额，余数不用。',
        '本实现主指标为所有目标同时持续跟踪时长：任一目标连续5次失败即终止场景；其余目标记同伴失败截尾，不能当成自身失败或独立完整跟踪时长。',
        '轨迹在当前分区内无放回组合；拼成同步时间轴的仿真场景，不代表原始记录实际同时采集。',
        '只在选束完成后用正确波束判定命中；只有命中测量进入历史；身份关联和同束目标可分辨性是理想假设。',
        '测量仍是轨迹加高斯误差代理，未实现完整OFDM回波、检测器、杂波或新目标出现。',
        '复用现有单目标模型，保留JOURNAL_ALIGNMENT.txt中码本/角距损失、模型参数量及数据限制；不是全文数值严格复现。',
        'CUDA预热后计时；批次时延与摊销单场景时延分开，不能据此声称复现论文的独立单场景时延。',
    ]
    record = {'split': a.split, 'targets': a.targets, 'total_beams': a.total_beams, 'seed': a.seed,
              'dt': cfg['dt'], 'history': h, 'failure_limit': nnet.config.failure_limit,
              'maximum_evaluated_duration_s': maximum, 'source': str(source.resolve()),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'rnet_checkpoint': str(a.checkpoint.resolve()), 'nnet_checkpoint': saved['nnet_checkpoint'],
              'scene_membership': scenes['trajectory_id'].tolist(), 'unused_trajectory_ids': unused,
              'partition_audit': partition, 'weights_unchanged': True,
              'measurement_regime': a.measurement_regime,
              'quality_mode': 'raw', 'noise_seed': a.noise_seed,
              'noise_metadata': noise_metadata,
              'test_is_new_blind_set': not cfg.get('test_records_used_previously', False),
              'results': summaries, 'limitations': cfg.get('limitations', [])+notes,
              'fully_reproduced_paper_numerical_results': False}
    write_json(out/'summary.json', record)
    lines = ['原文多目标核心流程测试', f'{len(indices)}个场景，每场景{a.targets}目标，总预算{a.total_beams}束。']
    for policy, summary in summaries.items():
        lines.append(f'{LABELS[policy]}：整组成功率{summary["success_rate"]:.2%}，同时跟踪时长{summary["mean_all_targets_duration_s"]:.4f}秒，恢复{summary["recoveries"]}次。')
    (out/'report.txt').write_text('\n'.join(lines+['']+notes)+'\n', encoding='utf-8')
    (a.output_root/'latest_run.txt').write_text(out.name, encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'结果目录：{out.resolve()}', flush=True)
    print('图片：tracking_status.png、tracking_survival.png', flush=True)
    from .plots import finish_tracking_plots
    finish_tracking_plots(figures, show=not a.no_show)
    return out
