"""The journal dataset's half-bin centered DFT codebook."""

import numpy as np


DFT_AXIS_BINS = 32
DFT_CLASSES = DFT_AXIS_BINS ** 2
DFT_CODEBOOK_NAME = '32x32_half_bin_centered_DFT_labels_1_to_1024'


def centered_dft_beam_ids(uv):
    """Encode direction cosines as the dataset's one-based beam labels."""
    bins = np.floor((np.asarray(uv) + 1) * (DFT_AXIS_BINS / 2)).astype(np.int64)
    u_bin, v_bin = np.moveaxis(np.clip(bins, 0, DFT_AXIS_BINS - 1), -1, 0)
    return v_bin * DFT_AXIS_BINS + u_bin + 1


def centered_dft_beam_coordinates():
    """Return (u, v) centers indexed by zero-based model class."""
    v_bin, u_bin = np.divmod(np.arange(DFT_CLASSES), DFT_AXIS_BINS)
    return (np.stack((u_bin, v_bin), axis=-1) + .5) * (2 / DFT_AXIS_BINS) - 1
