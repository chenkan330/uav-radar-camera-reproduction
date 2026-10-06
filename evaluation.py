"""Evaluate positions without extrapolating or bridging missing truth segments."""

from __future__ import annotations

import numpy as np


def _span(times: np.ndarray) -> list[float] | None:
    return [float(times[0]), float(times[-1])] if len(times) else None


def evaluate_positions(
    times, positions, truth, max_interpolation_gap_s: float = 0.2,
) -> dict:
    """Evaluate Nx3 estimated positions against Nx4 timestamp/XYZ truth.

    Exact truth samples always match. Intermediate times match only when both
    surrounding truth samples exist and their gap is <= the configured maximum.
    Errors are estimate minus truth. Standard deviation is population std (ddof=0).
    Results are JSON-serializable; absent ground truth/coverage yields None metrics.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if times.ndim != 1 or positions.shape != (times.size, 3):
        raise ValueError("times must have shape (N,), positions must have shape (N, 3)")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("estimated timestamps and positions must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("estimated timestamps must be strictly increasing")
    try:
        max_gap = float(max_interpolation_gap_s)
    except (TypeError, ValueError) as error:
        raise ValueError("max_interpolation_gap_s must be finite and positive") from error
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_interpolation_gap_s must be finite and positive")

    result = {
        "status": "no_ground_truth" if truth is None else "no_overlap",
        "estimate_count": int(times.size), "matched_count": 0,
        "unmatched_count": int(times.size), "coverage_fraction": 0.0,
        "estimate_time_span_s": _span(times), "truth_time_span_s": None,
        "matched_time_span_s": None, "max_interpolation_gap_s": max_gap,
        "signed_mean_xyz_m": None, "mean_euclidean_m": None,
        "median_euclidean_m": None, "std_euclidean_m": None,
        "p95_euclidean_m": None, "rmse3d_m": None, "axis_rmse_m": None,
    }
    if truth is None:
        return result
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 4:
        raise ValueError("truth must have shape (M, 4): timestamp, x, y, z")
    if not np.all(np.isfinite(truth)):
        raise ValueError("truth timestamps and positions must be finite")
    if np.any(np.diff(truth[:, 0]) <= 0):
        raise ValueError("truth timestamps must be strictly increasing")
    result["truth_time_span_s"] = _span(truth[:, 0])
    if not len(truth) or not len(times):
        return result

    truth_times = truth[:, 0]
    errors = []
    matched_times = []
    for timestamp, estimate in zip(times, positions):
        if timestamp < truth_times[0] or timestamp > truth_times[-1]:
            continue
        upper = int(np.searchsorted(truth_times, timestamp, side="left"))
        if upper < len(truth) and timestamp == truth_times[upper]:
            truth_position = truth[upper, 1:]
        else:
            lower = upper - 1
            if lower < 0 or upper >= len(truth):
                continue
            gap = truth_times[upper] - truth_times[lower]
            # Subtraction of Unix/relative timestamps can put a nominal 0.2 s
            # gap a few floating-point ulps above 0.2. Accept only this rounding
            # margin; substantial missing intervals still cannot be bridged.
            rounding_margin = 4 * np.finfo(float).eps * max(
                abs(truth_times[lower]), abs(truth_times[upper]), max_gap, 1.0,
            )
            if gap > max_gap + rounding_margin:
                continue
            fraction = (timestamp - truth_times[lower]) / gap
            truth_position = (
                (1 - fraction) * truth[lower, 1:] + fraction * truth[upper, 1:]
            )
        errors.append(estimate - truth_position)
        matched_times.append(timestamp)

    if not errors:
        return result
    errors = np.asarray(errors)
    distances = np.linalg.norm(errors, axis=1)
    count = len(errors)
    result.update({
        "status": "evaluated", "matched_count": count,
        "unmatched_count": int(times.size) - count,
        "coverage_fraction": count / int(times.size),
        "matched_time_span_s": _span(np.asarray(matched_times)),
        "signed_mean_xyz_m": np.mean(errors, axis=0).tolist(),
        "mean_euclidean_m": float(np.mean(distances)),
        "median_euclidean_m": float(np.median(distances)),
        "std_euclidean_m": float(np.std(distances)),
        "p95_euclidean_m": float(np.percentile(distances, 95)),
        "rmse3d_m": float(np.sqrt(np.mean(np.sum(errors * errors, axis=1)))),
        "axis_rmse_m": np.sqrt(np.mean(errors * errors, axis=0)).tolist(),
    })
    return result
