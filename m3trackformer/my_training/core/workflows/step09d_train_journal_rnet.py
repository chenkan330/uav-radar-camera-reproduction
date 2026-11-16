"""冻结MAFE/TSP，仅训练LE/FF；使用正式版多时刻角距分类损失。"""
import argparse
import csv
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.common import BASE, finish_run, load_nnet, new_run, read_json, seed_all, state_digest, write_json
from core.models import JournalRNet, angular_distance_matrix, classification_loss


def dataset(path):
    with np.load(path) as d:
        return TensorDataset(*[torch.from_numpy(d[k].copy()) for k in ('history','valid','failed','count','label')])


def run(nnet, rnet, loader, device, distance, optimizer=None):
    nnet.eval(); rnet.train(optimizer is not None); total = correct = 0; loss_sum = 0.
    with torch.set_grad_enabled(optimizer is not None):
        for history, valid, failed, count, labels in loader:
            history, valid, failed, count, labels = [v.to(device) for v in (history, valid, failed, count, labels)]
            logits = rnet(nnet, history, valid, failed, count, nnet.config.future)
            loss = classification_loss(logits, labels, distance, nnet.config.angular_weight)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            correct += int((logits[:, 0].topk(4, -1).indices==labels[:, :1]).any(1).sum())
            total += len(history); loss_sum += loss.item()*len(history)
    return {'loss': loss_sum/total, 'top4': correct/total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=256); p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=44)
    p.add_argument('--output-root', type=Path, default=BASE/'outputs/step09g_journal')
    a = p.parse_args(); seed_all(a.seed)
    if min(a.epochs,a.batch_size)<1 or a.lr<=0:
        p.error('无效训练参数')
    meta = read_json(a.data/'config.json'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnet, checkpoint = load_nnet(meta['nnet_checkpoint'], device); before = state_digest(nnet)
    rnet = JournalRNet(nnet.config).to(device); distance = angular_distance_matrix(mode=nnet.config.angular_grid).to(device)
    optimizer = torch.optim.AdamW(rnet.parameters(), lr=a.lr)
    loaders = [DataLoader(dataset(a.data/f'{s}.npz'), batch_size=a.batch_size, shuffle=(s=='train')) for s in ('train','validation')]
    out = new_run(a.output_root); best = -1.; start = time.monotonic()
    write_json(out/'config.json', {'data': str(a.data.resolve()), 'nnet_checkpoint': meta['nnet_checkpoint'],
                                  'model_config': nnet.config.dictionary(), 'epochs': a.epochs, 'batch_size': a.batch_size,
                                  'matmul_precision':torch.get_float32_matmul_precision(),
                                  'lr': a.lr, 'seed': a.seed, 'frozen_nnet_sha256_before': before})
    with (out/'training_log.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['epoch','train_loss','train_top4','val_loss','val_top4'])
        for epoch in range(1,a.epochs+1):
            train = run(nnet,rnet,loaders[0],device,distance,optimizer)
            val = run(nnet,rnet,loaders[1],device,distance)
            writer.writerow([epoch,*train.values(),*val.values()]); f.flush()
            if val['top4']>best:
                best = val['top4']
                torch.save({'format':'journal_equations_v1','model':rnet.state_dict(),'model_config':nnet.config.dictionary(),
                            'nnet_checkpoint':meta['nnet_checkpoint'],'data':str(a.data.resolve()),'epoch':epoch,'validation':val},out/'best_rnet.pth')
            print(f"epoch {epoch}/{a.epochs} train Top4={train['top4']:.2%} val Top4={val['top4']:.2%}",flush=True)
    after = state_digest(nnet)
    if before!=after or any(p.grad is not None for p in nnet.parameters()):
        raise AssertionError('冻结N-Net的验证失败')
    write_json(out/'summary.json',{'epochs_completed':a.epochs,'best_val_top4':best,'seconds':time.monotonic()-start,
                                   'frozen_nnet_sha256_before':before,'frozen_nnet_sha256_after':after,'nnet_unchanged':True})
    finish_run(out); print(out,flush=True)


if __name__=='__main__':
    main()
