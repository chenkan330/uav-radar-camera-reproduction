"""Fixed radar-only scene extraction for the selected central overhead clips.

No ground-truth inputs. Enhanced clouds are upstream products, not independent
raw detections: remove exact within-frame repetition and identical successive
target snapshots, without reducing covariance by their point count.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
from decimal import Decimal

import numpy as np


TIME_ORIGIN_NS = 1_692_846_887_000_000_000
CALIBRATION_INTERVALS_UNIX = [(1692846908.0, 1692846923.1), (1692847012.0, 1692847026.1)]
EVALUATION_INTERVAL_UNIX = (1692846959.5, 1692846969.0)


def timestamp_ns(path):
    return int(Decimal(Path(path).stem) * 1_000_000_000)


def relative_time(path):
    return (timestamp_ns(path) - TIME_ORIGIN_NS) / 1e9


def read_xyz(path):
    # Explicitly disable pickle; official files are plain numeric arrays.
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Expected actual finite Nx3 enhanced radar XYZ")
    return np.asarray(points, dtype=float)


def build_background(files, duration=8.0, voxel_m=.2, min_frames=30):
    files = sorted(map(Path, files), key=timestamp_ns)
    if not files or duration <= 0 or voxel_m <= 0 or min_frames < 1:
        raise ValueError("Invalid radar background settings")
    first_ns = timestamp_ns(files[0])
    counts = Counter()
    for path in files:
        if (timestamp_ns(path) - first_ns) / 1e9 > duration:
            break
        cells = np.unique(np.rint(read_xyz(path) / voxel_m).astype(np.int64), axis=0)
        counts.update(map(tuple, cells))
    return {cell for cell, count in counts.items() if count >= min_frames}


def extract_radar_target_points(points, background, voxel_m=.2,
                                half_width_m=2., min_z_m=2., link_distance_m=.6):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Expected finite Nx3 radar points")
    if min(voxel_m, half_width_m, link_distance_m) <= 0:
        raise ValueError("Radar extraction lengths must be positive")
    unique = np.unique(points, axis=0)
    retained = [point for point in unique
                if abs(point[0]) < half_width_m and abs(point[1]) < half_width_m
                and point[2] > min_z_m
                and tuple(np.rint(point / voxel_m).astype(np.int64)) not in background]
    retained = np.asarray(retained, dtype=float).reshape(-1, 3)
    info = {"original_points": len(points), "unique_points": len(unique),
            "roi_points": len(retained), "cluster_sizes": [], "selected_points": 0,
            "rule": "fixed background/central ROI/largest single-link cluster; radar only"}
    if not len(retained):
        return retained, info
    adjacency = np.sum((retained[:, None] - retained[None, :])**2, axis=2) <= link_distance_m**2
    remaining = set(range(len(retained)))
    clusters = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        pending = [seed]; cluster = [seed]
        while pending:
            current = pending.pop()
            additions = remaining.intersection(np.flatnonzero(adjacency[current]))
            remaining.difference_update(additions)
            pending.extend(additions); cluster.extend(additions)
        clusters.append(sorted(cluster))
    # Deterministic tie-break: centroid closest to the central sensor axis.
    clusters.sort(key=lambda c: (-len(c), float(np.linalg.norm(np.median(retained[c], axis=0)[:2]))))
    selected = retained[clusters[0]]
    info.update(cluster_sizes=[len(c) for c in clusters], selected_points=len(selected))
    return selected, info


def radar_frames_from_files(files, background, intervals_unix, suppress_identical=True):
    """Return observed snapshots and a complete per-file selection audit.

    Every accepted snapshot retains actual candidates. Identical consecutive
    selected point sets do not become repeated independent KF updates.
    """
    observations = []
    audit = []
    last_signature = None
    for path in sorted(map(Path, files), key=timestamp_ns):
        unix = timestamp_ns(path) / 1e9
        if not any(start <= unix <= end for start, end in intervals_unix):
            continue
        selected, info = extract_radar_target_points(read_xyz(path), background)
        record = {"source_file": path.name, "timestamp_s": relative_time(path), **info,
                  "accepted_snapshot": False, "identical_to_previous": False}
        if len(selected):
            signature = hashlib.sha256(selected.tobytes()).hexdigest()
            record["selected_set_sha256"] = signature
            if suppress_identical and signature == last_signature:
                record["identical_to_previous"] = True
            else:
                observations.append({"timestamp_s": relative_time(path), "points": selected,
                                     "path": path, "median_xyz": np.median(selected, axis=0)})
                record["accepted_snapshot"] = True
            last_signature = signature
        audit.append(record)
    return observations, audit
