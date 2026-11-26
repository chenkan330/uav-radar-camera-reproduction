"""Shared beam classification objective for both journal training stages."""

import torch
from torch.nn import functional as F

from .dft_codebook import DFT_CLASSES, centered_dft_beam_coordinates
from .models import angular_distance_matrix, classification_loss


DFT_UV_GRID = 'dft_uv_32x32'
UPSTREAM_GRID = 'upstream_64x16'


def training_distance_matrix(config, device):
    if config.angular_weight == 0:
        return None
    if config.angular_grid == DFT_UV_GRID:
        if config.classes != DFT_CLASSES:
            raise ValueError('DFT geometry requires 1024 beam classes')
        uv = torch.as_tensor(centered_dft_beam_coordinates(), dtype=torch.float32, device=device)
        # Every codeword has a center in (u, v), including centers outside the visible disk.
        return (uv[:, None] - uv[None, :]).square().sum(dim=-1)
    return angular_distance_matrix(classes=config.classes, mode=config.angular_grid).to(device)


def training_classification_loss(logits, labels, distance, weight):
    if distance is None:
        targets = labels.masked_fill(labels < 0, -1)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1, reduction='sum') / len(labels)
    return classification_loss(logits, labels, distance, weight)
