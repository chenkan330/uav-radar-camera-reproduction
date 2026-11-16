"""正式版Phase 1：多token角距惩罚分类损失 + 当前坐标MSE。"""
import argparse
import csv
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.common import BASE, TrainingWindows, audit_partitions, collate_batched, finish_run, new_run, read_json, seed_all, write_json
from core.models import JournalConfig, JournalNNet, angular_distance_matrix, classification_loss


def run(model, loader, device, distance, config, optimizer=None):
    model.train(optimizer is not None); sums = np.zeros(5); total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for history, position, labels in loader:
            history, position, labels = [x.to(device, non_blocking=True) for x in (history, position, labels)]
            valid = torch.ones(history.shape[:2], device=device, dtype=torch.bool)
            result = model(history, valid, config.future)
            cls = classification_loss(result['logits'], labels, distance, config.angular_weight)
            loc = (result['position']-position).square().sum(1).mean()
            loss = cls+config.position_weight*loc
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            chosen = result['logits'][:, 0].topk(4, -1).indices
            metrics = [loss.item(), cls.item(), loc.item(), (chosen[:, 0]==labels[:, 0]).float().mean().item(),
                       (chosen==labels[:, :1]).any(1).float().mean().item()]
            sums += np.array(metrics)*len(history); total += len(history)
    return dict(zip(('loss', 'classification', 'position_mse', 'top1', 'top4'), sums/total))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--history', type=int, default=5); p.add_argument('--future', type=int, default=5)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=256); p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--position-weight', type=float, default=.001); p.add_argument('--angular-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=44)
    p.add_argument('--output-root', type=Path, default=BASE/'outputs/step07g_journal')
    a = p.parse_args()
    if min(a.epochs, a.history, a.future, a.stride, a.batch_size)<1 or min(a.position_weight, a.angular_weight)<0:
        p.error('参数无效')
    seed_all(a.seed); torch.set_float32_matmul_precision('high')
    partition_audit = audit_partitions(a.data)
    data_config = read_json(a.data/'config.json')
    if data_config['dt'] != .05:
        p.error('正式版协议需要0.05秒采样，不能直接使用早期0.1秒数据')
    config = JournalConfig(history=a.history, future=a.future, position_weight=a.position_weight, angular_weight=a.angular_weight)
    if config.history < config.failure_limit:
        p.error('历史长度至少为失败阈值，保证终止前仍有有效测量')
    with np.load(a.data/'nnet_train.npz') as d:
        normalization = {'mean': d['measurement'].mean((0, 1)).tolist(),
                         'std': np.maximum(d['measurement'].std((0, 1)), 1e-6).tolist()}
    datasets = [TrainingWindows(a.data/f'{s}.npz', normalization, a.history, a.future, a.stride)
                for s in ('nnet_train', 'validation')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = [DataLoader(d, batch_size=a.batch_size, shuffle=(i==0), num_workers=0,
                          pin_memory=device.type=='cuda',collate_fn=collate_batched) for i, d in enumerate(datasets)]
    model = JournalNNet(config).to(device); distance = angular_distance_matrix(mode=config.angular_grid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    out = new_run(a.output_root); start = time.monotonic(); best = (-1., -1.)
    meta = {'format': 'journal_equations_v1', 'data': str(a.data.resolve()), 'model_config': config.dictionary(),
            'normalization': normalization, 'training': {**vars(a), 'data': str(a.data.resolve()), 'output_root': str(a.output_root.resolve())},
            'partition_audit': partition_audit, 'device': str(device), 'matmul_precision': 'high',
            'window_counts': dict(zip(('train', 'validation'), map(len, datasets)))}
    meta['training'] = {k: str(v) if isinstance(v, Path) else v for k, v in meta['training'].items()}
    write_json(out/'config.json', meta)
    with (out/'training_log.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['epoch', *['train_'+k for k in ('loss','classification','position_mse','top1','top4')],
                                                *['val_'+k for k in ('loss','classification','position_mse','top1','top4')]])
        for epoch in range(1, a.epochs+1):
            train = run(model, loaders[0], device, distance, config, optimizer)
            val = run(model, loaders[1], device, distance, config)
            writer.writerow([epoch, *train.values(), *val.values()]); f.flush()
            if (val['top4'], val['top1']) > best:
                best = (val['top4'], val['top1'])
                torch.save({**meta, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'validation': val}, out/'best_nnet.pth')
            print(f"epoch {epoch}/{a.epochs} val Top1={val['top1']:.2%} Top4={val['top4']:.2%}", flush=True)
    write_json(out/'summary.json', {'epochs_completed': a.epochs, 'best_val_top4': best[0],
                                   'best_val_top1': best[1], 'seconds': time.monotonic()-start})
    finish_run(out); print(out, flush=True)


if __name__ == '__main__':
    main()
