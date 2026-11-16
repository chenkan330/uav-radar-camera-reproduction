"""Phase 2采样：独立D_sim、滑窗短程仿真、K=3/4/5混合、去除完全重复状态。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from core.common import BASE, audit_partitions, finish_run, load_nnet, load_split, new_run, normalized_history, read_json, write_json


def collect_windows(model, checkpoint, path, device, stride=1, batch_size=512, seed=44):
    data = load_split(path); c = model.config; rng = np.random.default_rng(seed)
    measurement, beam = data['measurement'], data['beam']-1
    n, t = beam.shape; h, f = c.history, c.future
    starts = np.arange(0, t-h-f+1, stride)
    pairs = np.stack(np.meshgrid(np.arange(n), starts, indexing='ij'), -1).reshape(-1, 2)
    rows = {key: [] for key in ('history','valid','failed','count','label','trajectory_id','time_index','anchor_index','k')}
    seen = set(); candidates = lost_windows = 0
    for offset in range(0, len(pairs), batch_size):
        ids, anchor = pairs[offset:offset+batch_size].T; size = len(ids)
        # 概率来自公开脚本testtest；窗口内保持同一K，测试时不会使用该随机混合。
        k = np.full(size,rng.choice([3,4,5],p=[.7,.2,.1]),dtype='int64')
        available = np.zeros((size, h+f), bool); available[:, :h] = True
        active = np.ones(size, bool); counts = np.zeros(size, dtype='int64')
        failed = np.full((size, c.failure_limit-1, 5), -1, dtype='int64')
        for step in range(f):
            take = np.flatnonzero(active)
            if not len(take):
                break
            times = anchor[take]+h+step
            history_times = times[:, None]-h+np.arange(h)[None, :]
            valid = available[take, step:step+h]
            history = normalized_history(measurement[ids[take, None], history_times], valid, checkpoint['normalization'])
            lost = counts[take]>0
            if lost.any():
                selected = take[lost]
                for local, original in zip(np.flatnonzero(lost), selected):
                    current = int(anchor[original]+h+step)
                    key = (int(data['trajectory_id'][ids[original]]), current, int(counts[original]), failed[original].tobytes())
                    candidates += 1
                    if key in seen:
                        continue
                    seen.add(key)
                    label = np.full(f, -1, dtype='int64'); length = min(f, t-current)
                    label[:length] = beam[ids[original], current:current+length]
                    values = {'history': history[local], 'valid': valid[local].copy(), 'failed': failed[original].copy(),
                              'count': counts[original], 'label': label,
                              'trajectory_id': data['trajectory_id'][ids[original]], 'time_index': current,
                              'anchor_index': anchor[original], 'k': k[original]}
                    for name, value in values.items():
                        rows[name].append(value)
            with torch.no_grad():
                logits = model(torch.from_numpy(history).to(device), torch.from_numpy(valid).to(device))['logits'][:, 0]
                chosen = logits.topk(5, -1).indices.cpu().numpy()
            used = np.arange(5)[None, :]<k[take, None]
            hit = ((chosen==beam[ids[take], times, None]) & used).any(1)
            # 仅采集本窗口第一次漏检链；命中即结束该窗口，和作者短窗采样意图一致。
            active[take[hit]] = False
            bad = take[~hit]
            if len(bad):
                lost_windows += int((counts[bad]==0).sum())
                room = counts[bad]<c.failure_limit-1
                bad_room = bad[room]
                failed[bad_room, counts[bad_room]] = np.where(used[~hit][room], chosen[~hit][room], -1)
                counts[bad] += 1
                active[bad[counts[bad]>=c.failure_limit]] = False
        if (offset//batch_size+1)%100 == 0:
            print(f'windows {min(offset+batch_size,len(pairs))}/{len(pairs)} unique lost states {len(seen)}', flush=True)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    stats = {'candidate_windows': len(pairs), 'windows_starting_with_loss': lost_windows,
             'candidate_lost_states': candidates, 'unique_lost_states': len(seen),
             'unique_segments': len(np.unique(arrays['trajectory_id'])) if seen else 0,
             'sampling_k_values': [3,4,5], 'sampling_k_probabilities': [.7,.2,.1]}
    return arrays, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stride', type=int, default=1); p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=44)
    p.add_argument('--output-root', type=Path, default=BASE/'outputs/step08g_journal')
    a = p.parse_args()
    if min(a.stride,a.batch_size)<1:
        p.error('stride/batch-size必须为正数')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_nnet(a.checkpoint, device); data = Path(checkpoint['data'])
    audit_partitions(data); out = new_run(a.output_root); statistics = {}
    for split, source in [('train', 'rnet_sim'), ('validation', 'validation')]:
        rows, stats = collect_windows(model, checkpoint, data/f'{source}.npz', device, a.stride, a.batch_size,
                                     a.seed+(split=='validation'))
        if not stats['unique_lost_states']:
            raise RuntimeError(f'{split}无失跟样本，不能训练；未降低测试难度或伪造样本')
        np.savez_compressed(out/f'{split}.npz', **rows); statistics[split] = stats
        print(split, stats, flush=True)
    write_json(out/'config.json', {'format': 'journal_equations_v1', 'nnet_checkpoint': str(a.checkpoint.resolve()),
                                  'data': str(data.resolve()), 'statistics': statistics, 'stride': a.stride, 'seed': a.seed,
                                  'policy': 'frozen_NNet_only_independent_sliding_window_rollout_no_iterative_RNet',
                                  'not_paper_specified': ['窗口步长、采样窗口长度及全重复状态去重是显式工程约定。',
                                                         '公开脚本用一次多token预测；此处按论文当前历史定义逐时刻滚动带掩码N-Net。'],
                                  'future_targets_are_labels_only': True})
    finish_run(out); print(out, flush=True)


if __name__ == '__main__':
    main()
