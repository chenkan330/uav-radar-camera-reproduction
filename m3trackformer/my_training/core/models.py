"""单目标正式版核心：MAFE/TSP、动态 LE、全序列 FF、特征残差融合。"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class JournalConfig:
    history: int = 5          # 正式版VI节历史窗口实验明确选5，也与公开代码一致
    future: int = 5           # 正式版未给数值；沿用公开代码 PREDICT_SIZE
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 2
    dropout: float = .1       # 公开代码
    classes: int = 1024       # 模型内部0..1023；磁盘标签仍为1..1024
    failure_limit: int = 5
    position_weight: float = .001  # 正式版未给数值，实验假设
    angular_weight: float = .1     # 正式版未给数值，公开损失函数默认值
    angular_grid: str = 'upstream_64x16'

    def dictionary(self):
        return asdict(self)


def positional_encoding(length, dim, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    scale = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
    return pe


class EncodingUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = nn.MultiheadAttention(c.d_model, c.heads, c.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(c.d_model, 4*c.d_model), nn.ReLU(), nn.Linear(4*c.d_model, c.d_model))
        self.norm1, self.norm2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.drop1, self.drop2 = nn.Dropout(c.dropout), nn.Dropout(c.dropout)

    def forward(self, x, padding):
        a = self.attention(x, x, x, key_padding_mask=padding, need_weights=False)[0]
        x = self.norm1(x + self.drop1(a))
        return self.norm2(x + self.drop2(self.ffn(x)))


class DecodingUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = nn.MultiheadAttention(c.d_model, c.heads, c.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(c.d_model, 4*c.d_model), nn.ReLU(), nn.Linear(4*c.d_model, c.d_model))
        self.norm1, self.norm2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.drop1, self.drop2 = nn.Dropout(c.dropout), nn.Dropout(c.dropout)

    def forward(self, lost, motion, motion_padding):
        a = self.attention(lost, motion, motion, key_padding_mask=motion_padding, need_weights=False)[0]
        x = self.norm1(lost + self.drop1(a))
        return self.norm2(x + self.drop2(self.ffn(x)))


class JournalNNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or JournalConfig()
        self.embedding = nn.Linear(3, c.d_model)
        # 论文的原始空间零token经共享线性层后可学习；所有预测位置共享token。
        self.prediction_token = nn.Parameter(torch.zeros(1, 1, 3))
        self.input_dropout = nn.Dropout(c.dropout)
        self.mafe = nn.ModuleList(EncodingUnit(c) for _ in range(c.encoder_layers))
        self.classifier = nn.Linear(c.d_model, c.classes)
        self.regressor = nn.Sequential(nn.Linear(c.d_model, c.d_model//2), nn.ReLU(), nn.Linear(c.d_model//2, 3))

    def encode(self, history, valid, future=1):
        if history.shape[1] != self.config.history or not valid.any(1).all():
            raise ValueError('历史长度不匹配或已无有效历史；应在失败阈值处终止')
        n, h, _ = history.shape
        raw = torch.cat((history.masked_fill(~valid[..., None], 0), self.prediction_token.expand(n, future, -1)), 1)
        x = self.embedding(raw)
        x = self.input_dropout(x + positional_encoding(h+future, x.shape[-1], x.device, x.dtype))
        padding = torch.cat((~valid, torch.ones(n, future, device=x.device, dtype=torch.bool)), 1)
        for unit in self.mafe:
            x = unit(x, padding)
        return x, padding

    def forward(self, history, valid, future=1):
        x, padding = self.encode(history, valid, future)
        h = self.config.history
        return {'logits': self.classifier(x[:, h:]), 'position': self.regressor(x[:, h-1]),
                'motion': x, 'padding': padding}


class JournalRNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or JournalConfig()
        self.embedding = nn.Linear(c.classes, c.d_model)
        self.prediction_token = nn.Parameter(torch.zeros(1, 1, c.classes))
        self.input_dropout = nn.Dropout(c.dropout)
        self.le = EncodingUnit(c)
        self.ff = nn.ModuleList(DecodingUnit(c) for _ in range(c.decoder_layers))

    def forward(self, nnet, history, valid, failed, count, future=1):
        """failed是已扫描失败的0-based波束，-1为批填充；未来标签不在接口内。"""
        if ((count < 1) | (count >= self.config.failure_limit)).any():
            raise ValueError('R-Net仅用于已漏检且未达到失败阈值的状态')
        if nnet.training:
            raise ValueError('Phase 2及推理时，共享MAFE/TSP必须保持eval状态')
        motion = nnet(history, valid, future)
        n, max_lost = len(history), int(count.max())
        positions = torch.arange(max_lost, device=history.device)[None, :]
        event_valid = positions < count[:, None]
        f = failed[:, :max_lost]
        selected = (f >= 0) & event_valid[..., None]
        hot = torch.zeros(n, max_lost, self.config.classes, device=history.device, dtype=history.dtype)
        hot.scatter_add_(2, f.clamp_min(0), selected.to(history.dtype))
        hot.clamp_max_(1)
        # token紧跟每条样本实际的失跟前缀，而不是固定放在第5个位置。
        raw = torch.zeros(n, max_lost+future, self.config.classes, device=history.device, dtype=history.dtype)
        raw[:, :max_lost] = hot
        rows = torch.arange(n, device=history.device)[:, None]
        query_index = count[:, None] + torch.arange(future, device=history.device)[None, :]
        raw[rows, query_index] = self.prediction_token.expand(n, future, -1)
        key_padding = torch.arange(max_lost+future, device=history.device)[None, :] > count[:, None]
        # LE中当前预测token有效；多token训练时额外未来token只作query，防止改变首步语义。
        x = self.embedding(raw)
        x = self.input_dropout(x + positional_encoding(x.shape[1], x.shape[2], x.device, x.dtype))
        x = self.le(x, key_padding)
        # 按论文FF对LE完整序列做交叉注意力，再提取各预测token。
        for unit in self.ff:
            x = unit(x, motion['motion'], motion['padding'])
        correction = x[rows, query_index]
        refined = motion['motion'][:, self.config.history:] + correction
        return nnet.classifier(refined)


def angular_distance_matrix(classes=1024, mode='upstream_64x16'):
    """Eq.(23)；角度索引沿用作者utils/tools.py，排序歧义见对齐说明。"""
    if mode != 'upstream_64x16' or classes != 1024:
        raise ValueError('未擅自添加未经原文/源码给出的码本角度映射')
    i = torch.arange(classes)
    az = (i % 64 + .5) * (2*math.pi/64)
    el = (i // 64 + .5) * (math.pi/2/16)
    da = (az[:, None] - az[None, :]).abs()
    da = torch.minimum(da, 2*math.pi-da)
    return da.square() + (el[:, None]-el[None, :]).square()


def classification_loss(logits, labels, distance, eta=.1):
    """正式版Eq.(24)：未来时刻求和、样本求平均；-1目标仅表示片段末尾无监督。"""
    valid = labels >= 0
    safe = labels.clamp_min(0)
    ce = F.cross_entropy(logits.flatten(0, 1), safe.flatten(), reduction='none').view_as(labels)
    expected = (logits.softmax(-1) * distance[safe]).sum(-1)
    angular = torch.sqrt(expected.clamp_min(1e-12))
    return ((ce + eta*angular) * valid).sum()/len(labels)
