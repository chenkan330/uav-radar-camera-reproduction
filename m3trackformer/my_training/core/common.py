"""正式版复现的公共读写、训练窗口及仅训练集归一化工具。"""
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .models import JournalConfig, JournalNNet

BASE = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def new_run(root):
    out = Path(root) / ('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC'))
    out.mkdir(parents=True)
    return out


def finish_run(out):
    (out.parent / 'latest_run.txt').write_text(out.name, encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_split(path):
    with np.load(path) as d:
        return {key: d[key].copy() for key in d.files}


def load_nnet(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('format') != 'journal_equations_v1':
        raise ValueError('必须使用新正式版权重，旧教学版及作者发布版不是同一结构')
    torch.set_float32_matmul_precision(checkpoint.get('matmul_precision','highest'))
    model = JournalNNet(JournalConfig(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model']); model.eval(); model.requires_grad_(False)
    return model, checkpoint


def normalized_history(measurement, valid, normalization):
    x = (measurement - np.asarray(normalization['mean'])) / np.asarray(normalization['std'])
    return np.where(valid[..., None], x, 0).astype('float32')


class TrainingWindows(Dataset):
    def __init__(self, path, normalization, history, future, stride):
        self.data = load_split(path)
        self.h, self.f, self.normalization = history, future, normalization
        self.normalized = normalized_history(self.data['measurement'], np.ones(self.data['beam'].shape, bool), normalization)
        self.labels = self.data['beam']-1
        self.starts = np.arange(0, self.data['beam'].shape[1]-history-future+1, stride)
        if not len(self.starts):
            raise ValueError('轨迹过短')

    def __len__(self):
        return len(self.starts) * len(self.data['beam'])

    def __getitem__(self, index):
        i, j = divmod(index, len(self.starts)); start = self.starts[j]
        return (self.normalized[i,start:start+self.h], self.data['position'][i,start+self.h-1],
                self.labels[i,start+self.h:start+self.h+self.f])

    def __getitems__(self, indices):
        # DataLoader批量取窗口，与逐项__getitem__相同，避免每个样本分配多个小数组。
        trajectory, window = np.divmod(np.asarray(indices),len(self.starts))
        starts = self.starts[window]
        hi = starts[:,None]+np.arange(self.h)[None,:]
        fi = starts[:,None]+self.h+np.arange(self.f)[None,:]
        return (torch.from_numpy(self.normalized[trajectory[:,None],hi]),
                torch.from_numpy(self.data['position'][trajectory,starts+self.h-1]),
                torch.from_numpy(self.labels[trajectory[:,None],fi]))


def collate_batched(batch):
    return batch


def audit_partitions(data):
    ids, groups = {}, {}
    for split in ('nnet_train', 'rnet_sim', 'validation', 'test'):
        with np.load(Path(data)/f'{split}.npz') as d:
            ids[split] = set(d['trajectory_id'].tolist())
            if 'session_group' in d.files:
                groups[split] = set(d['session_group'].tolist())
    names = list(ids)
    if any(ids[x] & ids[y] for i, x in enumerate(names) for y in names[i+1:]):
        raise ValueError('轨迹ID跨分组重叠')
    if groups and any(groups[x] & groups[y] for i, x in enumerate(names) for y in names[i+1:]):
        raise ValueError('原始记录组跨分组重叠')
    return {'ids_disjoint': True, 'session_groups_disjoint': True if groups else None,
            'split_counts': {key: len(value) for key, value in ids.items()}}


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
