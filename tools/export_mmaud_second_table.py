"""Export exact saved one-second states and a position-derived speed reference.

No tracking is rerun. MMAUD provides position, not directly measured velocity:
V_ref uses linearly interpolated original GT at t +/- 0.5 s, differentiated
over one second. It is a centred window-average approximation, not an exact
instantaneous velocity. Reference-velocity variance is unknown.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_METRICS_SHA256 = "20004bbd5c61974a2696724a258c80316cbf23720933c5f16ebd86e0dfee65c8"
FROZEN_TRACKS_SHA256 = "64f4d31b62742e97d1c0c1db3c6d315e968f03dee041bf08107e7736e4285137"
FROZEN_MODEL_SHA256 = "be9281c8dcd3fc7bae85c1b4b884ad039e2733027921f8e32326f3d713c88a8f"
METHODS = (("radar_only", "仅雷达"), ("radar_camera", "雷达＋相机"))
STATE_ORDER = ["x", "Vx", "y", "Vy", "z", "Vz"]
POSITION_INDICES = [0, 2, 4]
VELOCITY_INDICES = [1, 3, 5]
MAX_INTERPOLATION_GAP_S = .30
VELOCITY_HALF_WINDOW_S = .5
TRUTH_SUPPORT_PADDING_S = .6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _validate_truth(truth, max_gap):
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 4 or len(truth) < 2:
        raise ValueError("GT must be Nx4 [timestamp_s,x_m,y_m,z_m], N >= 2")
    if not np.isfinite(truth).all() or np.any(np.diff(truth[:, 0]) <= 0):
        raise ValueError("GT must be finite with strictly increasing timestamps")
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Interpolation maximum gap must be finite and positive")
    return truth


def interpolate_position(truth, timestamp, max_gap=MAX_INTERPOLATION_GAP_S):
    """Return interpolated XYZ plus actual sample indices/weights; no extrapolation."""
    truth = _validate_truth(truth, max_gap)
    timestamp = float(timestamp)
    if not np.isfinite(timestamp):
        raise ValueError("Query timestamp must be finite")
    times = truth[:, 0]
    upper = int(np.searchsorted(times, timestamp, side="left"))
    if upper < len(times) and timestamp == times[upper]:
        return truth[upper, 1:].copy(), dict(indices=[upper], weights=[1.0])
    lower = upper - 1
    if lower < 0 or upper >= len(times):
        raise ValueError(f"GT cannot extrapolate at {timestamp:.9f} s")
    gap = times[upper] - times[lower]
    rounding = 4 * np.finfo(float).eps * max(abs(times[lower]), abs(times[upper]), 1.0)
    if gap > max_gap + rounding:
        raise ValueError(f"GT interpolation gap {gap:.9f} s exceeds {max_gap} s")
    fraction = (timestamp - times[lower]) / gap
    weights = [float(1 - fraction), float(fraction)]
    return weights[0] * truth[lower, 1:] + weights[1] * truth[upper, 1:], dict(indices=[lower, upper], weights=weights)


def central_velocity_reference(truth, timestamp, half_window_s=VELOCITY_HALF_WINDOW_S,
                               max_gap=MAX_INTERPOLATION_GAP_S):
    """Position difference across the complete centred window, rejecting GT gaps."""
    truth = _validate_truth(truth, max_gap)
    if not np.isfinite(half_window_s) or half_window_s <= 0:
        raise ValueError("Velocity reference half-window must be finite and positive")
    start, end = float(timestamp) - half_window_s, float(timestamp) + half_window_s
    before, before_support = interpolate_position(truth, start, max_gap)
    after, after_support = interpolate_position(truth, end, max_gap)
    # Checking only the endpoints could silently bridge an outage inside the
    # one-second window. Include every actual sample gap used by the window.
    first = min(before_support["indices"])
    last = max(after_support["indices"])
    segment_times = truth[first:last + 1, 0]
    rounding = 4 * np.finfo(float).eps * max(float(np.max(np.abs(segment_times))), 1.0)
    if np.any(np.diff(segment_times) > max_gap + rounding):
        raise ValueError("GT velocity window contains a gap larger than the permitted maximum")
    velocity = (after - before) / (2 * half_window_s)
    return velocity, dict(window_start_s=start, window_end_s=end, window_duration_s=2 * half_window_s,
                         before=before_support, after=after_support,
                         maximum_actual_gap_in_window_s=float(np.max(np.diff(segment_times))) if len(segment_times) > 1 else 0.0)


def numerical_checks():
    """Independent analytic trajectories and an internal missing interval."""
    times = np.arange(16, dtype=float) / 5
    velocity = np.array([2., -3., .25])
    positions = np.array([4., 1., -2.]) + times[:, None] * velocity
    linear = np.column_stack((times, positions))
    actual, _ = central_velocity_reference(linear, 1.17)
    np.testing.assert_allclose(actual, velocity, rtol=0, atol=2e-14)
    # For a quadratic trajectory a centred difference is the analytic
    # derivative when its endpoints are actual samples (here 0.6 and 1.6 s).
    acceleration = np.array([1., -2., .4])
    quadratic = np.column_stack((times, positions + .5 * times[:, None] ** 2 * acceleration))
    actual, _ = central_velocity_reference(quadratic, 1.1)
    np.testing.assert_allclose(actual, velocity + 1.1 * acceleration, rtol=0, atol=2e-14)
    # Both endpoints interpolate within short gaps, but the centre has a
    # 0.6 s hole: the window must still be rejected.
    missing = linear[~np.isin(np.round(times, 1), [.8, 1.0])]
    try:
        central_velocity_reference(missing, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("An internal GT gap was silently bridged")
    try:
        central_velocity_reference(linear, .2)
    except ValueError:
        pass
    else:
        raise AssertionError("GT velocity was extrapolated")
    return dict(constant_velocity_analytic=True, quadratic_analytic=True,
                internal_gap_rejected=True, extrapolation_rejected=True)


def _timestamp_ns(path):
    return int(Decimal(Path(path).stem) * 1_000_000_000)


def _load_original_truth(data, manifest):
    origin = manifest["source"]["time_origin_ns"]
    selected = np.asarray(manifest["selected_interval_s"], dtype=float)
    support_interval = selected + [-TRUTH_SUPPORT_PADDING_S, TRUTH_SUPPORT_PADDING_S]
    files = sorted((data / "Mavic3/ground_truth").glob("*.npy"), key=_timestamp_ns)
    times = np.asarray([(_timestamp_ns(path) - origin) / 1e9 for path in files])
    if not len(files) or np.any(np.diff(times) <= 0):
        raise ValueError("Original GT timestamps must exist and be strictly increasing")
    inside = np.flatnonzero((times >= support_interval[0]) & (times <= support_interval[1]))
    if not len(inside):
        raise ValueError("No original GT within the fixed support interval")
    # Exactly one surrounding sample avoids truncating real interpolation
    # support at a padding boundary. It never permits extrapolation.
    first, last = max(0, int(inside[0]) - 1), min(len(files) - 1, int(inside[-1]) + 1)
    chosen = files[first:last + 1]
    hashes, rows = {}, []
    frozen_hashes = manifest["source"]["raw_sha256"]
    for index, path in enumerate(chosen, first):
        label = path.relative_to(data).as_posix()
        digest = sha256(path)
        if label in frozen_hashes and digest != frozen_hashes[label]:
            raise ValueError(f"Original GT file differs from frozen source: {label}")
        point = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Original GT must contain a finite XYZ position only")
        rows.append([times[index], *point])
        hashes[label] = digest
    return np.asarray(rows), [path.relative_to(data).as_posix() for path in chosen], hashes, support_interval


def _source_inputs(data, primary):
    primary_directory = Path(primary)
    metrics_path = primary_directory / "metrics.json"
    if sha256(metrics_path) != FROZEN_METRICS_SHA256:
        raise ValueError("Primary metrics differ from the frozen v1.6 result")
    if sha256(primary_directory / "tracks.npz") != FROZEN_TRACKS_SHA256:
        raise ValueError("Saved state/covariance NPZ differs from the frozen v1.6 result")
    metrics = read_json(metrics_path)
    manifest = read_json(data / "prepared_v16/clip.json")
    if manifest != metrics["manifest"] or metrics.get("ground_truth_filter_input") is not False:
        raise ValueError("Prepared manifest or no-GT filter declaration differs from the primary result")
    input_hashes = {}
    for name, expected in metrics["input_sha256"].items():
        digest = sha256(data / "prepared_v16" / name)
        if digest != expected:
            raise ValueError(f"Frozen prepared input has drifted: {name}")
        input_hashes[f"prepared_v16/{name}"] = digest
    model_path = data / "calibration/local_model.json"
    model_hash = sha256(model_path)
    if model_hash != FROZEN_MODEL_SHA256 or model_hash != manifest["source"]["raw_sha256"]["calibration/local_model.json"]:
        raise ValueError("Primary rigid-transform model has drifted")
    model = read_json(model_path)
    if model.get("evaluation_used_for_fit") is not False:
        raise ValueError("The frozen transform must not fit evaluation GT")
    alignment = manifest["config"]["evaluation"]["truth_frame_alignment"]
    rotation = np.asarray(model["radar_to_truth"]["R"], dtype=float)
    translation = np.asarray(model["radar_to_truth"]["t"], dtype=float)
    np.testing.assert_array_equal(rotation, alignment["R_radar_to_truth"])
    np.testing.assert_array_equal(translation, alignment["t_radar_in_truth"])
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Frozen GT frame transform must be finite")
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-10)
    archive_name = "ground_truth_archive_span.bin"
    archive_hash = sha256(data / archive_name)
    if archive_hash != manifest["source"]["raw_sha256"][archive_name]:
        raise ValueError("Original GT archive source differs from the frozen provenance")
    source_hashes = dict(primary_metrics_json=FROZEN_METRICS_SHA256,
                         primary_tracks_npz=FROZEN_TRACKS_SHA256,
                         local_model_json=model_hash, ground_truth_archive_span_bin=archive_hash,
                         prepared_inputs=input_hashes)
    return metrics, manifest, rotation, translation, source_hashes


def _baseline_stagnation_evidence(primary, states, times, manifest):
    updates_path = primary / "updates.json"
    updates = read_json(updates_path)["radar_only"]
    initial_xyz = np.asarray(manifest["radar_processing"]["initial_selected_xyz_m"])
    np.testing.assert_array_equal(states[0, POSITION_INDICES], initial_xyz)
    accepted = [item for item in updates if item["sensor"] == "radar" and item["accepted"] and "selected_xyz_m" in item]
    repeated = []
    first_changed = None
    for item in accepted:
        if not np.array_equal(item["selected_xyz_m"], initial_xyz):
            first_changed = item
            break
        if item["nis"] != 0.0 or item["reason"] != "nearest gated XYZ":
            raise ValueError("Baseline repeated-point diagnostics disagree with saved nearest-point updates")
        repeated.append(item)
    changed_ticks = np.flatnonzero(np.any(states != states[0], axis=1))
    return dict(updates_json_sha256=sha256(updates_path), initial_selected_xyz_m=initial_xyz.tolist(),
                accepted_initial_xyz_repeats_before_change=len(repeated),
                repeated_selection_interval_s=[repeated[0]["timestamp"], repeated[-1]["timestamp"]] if repeated else None,
                repeated_selected_point_indices=sorted({item["selected_point_index"] for item in repeated}),
                first_changed_selected_radar_update=first_changed,
                first_changed_saved_tick=int(changed_ticks[0]) if len(changed_ticks) else None,
                first_changed_saved_timestamp_s=float(times[changed_ticks[0]]) if len(changed_ticks) else None,
                explanation="Observed nearest-point reuse of the exact initial XYZ in enhanced historical point clouds; zero velocity is the saved baseline output, not missing-value padding or proof of a stationary target.")


def export_second_table(data, primary, output):
    data, primary, output = Path(data), Path(primary), Path(output)
    checks = numerical_checks()
    metrics, manifest, rotation, translation, hashes = _source_inputs(data, primary)
    with np.load(primary / "tracks.npz", allow_pickle=False) as saved:
        times = np.asarray(saved["timestamps"], dtype=float).copy()
        states = {name: np.asarray(saved[f"{name}_state"], dtype=float).copy() for name, _ in METHODS}
        covariances = {name: np.asarray(saved[f"{name}_covariance"], dtype=float).copy() for name, _ in METHODS}
    if times.shape != (949,) or not np.isfinite(times).all() or metrics["output_grid_hz"] != 100:
        raise ValueError("Expected the frozen 949 saved 100 Hz timestamps")
    expected_times = times[0] + np.arange(949) / 100
    np.testing.assert_array_equal(times, expected_times)
    initialization_time = (_timestamp_ns(manifest["radar_processing"]["initial_file"]) - manifest["source"]["time_origin_ns"]) / 1e9
    if times[0] != initialization_time or times[0] != 72.512097:
        raise ValueError("Saved first output must equal the actual first radar initialization")
    for name, _ in METHODS:
        if states[name].shape != (949, 6) or covariances[name].shape != (949, 6, 6):
            raise ValueError("Saved state/covariance dimensions differ from the frozen result")
        if not np.isfinite(states[name]).all() or not np.isfinite(covariances[name]).all():
            raise ValueError("Saved state/covariance must be finite")
        diagonal = np.diagonal(covariances[name], axis1=1, axis2=2)
        if np.any(diagonal < 0):
            raise ValueError("Saved posterior variances must be nonnegative")
    baseline_evidence = _baseline_stagnation_evidence(primary, states["radar_only"], times, manifest)
    raw_truth, truth_labels, raw_truth_hashes, support_interval = _load_original_truth(data, manifest)
    # All GT transformations use the fixed primary rig, shared by both methods.
    truth = raw_truth.copy()
    truth[:, 1:] = (raw_truth[:, 1:] - translation) @ rotation
    records, references = [], []
    for sample_index, tick in enumerate(range(0, 901, 100)):
        timestamp = float(times[tick])
        position, position_support = interpolate_position(truth, timestamp)
        # Compute in the original GT frame, then rotate velocity without adding
        # translation. This also independently checks the position transform.
        velocity_original, velocity_support = central_velocity_reference(raw_truth, timestamp)
        reference_velocity = rotation.T @ velocity_original
        velocity_from_transformed, _ = central_velocity_reference(truth, timestamp)
        np.testing.assert_allclose(reference_velocity, velocity_from_transformed, rtol=0, atol=1e-13)
        references.append(dict(sample_index=sample_index, timestamp_s=timestamp,
                               gt_position_support=position_support, V_ref_support=velocity_support))
        for name, _ in METHODS:
            state, covariance = states[name][tick], covariances[name][tick]
            filtered_position, filtered_velocity = state[POSITION_INDICES], state[VELOCITY_INDICES]
            diagonal = np.diag(covariance)
            record = dict(method=name, sample_index=sample_index, saved_tick_index=tick,
                          time_since_initialization_s=float(sample_index), timestamp_s=timestamp,
                          filtered_x_m=float(filtered_position[0]), filtered_y_m=float(filtered_position[1]), filtered_z_m=float(filtered_position[2]),
                          filtered_Vx_mps=float(filtered_velocity[0]), filtered_Vy_mps=float(filtered_velocity[1]), filtered_Vz_mps=float(filtered_velocity[2]),
                          filtered_V_mps=float(np.linalg.norm(filtered_velocity)),
                          P_xx_m2=float(diagonal[0]), P_yy_m2=float(diagonal[2]), P_zz_m2=float(diagonal[4]),
                          P_VxVx_m2ps2=float(diagonal[1]), P_VyVy_m2ps2=float(diagonal[3]), P_VzVz_m2ps2=float(diagonal[5]),
                          gt_x_m=float(position[0]), gt_y_m=float(position[1]), gt_z_m=float(position[2]),
                          V_ref_x_mps=float(reference_velocity[0]), V_ref_y_mps=float(reference_velocity[1]), V_ref_z_mps=float(reference_velocity[2]),
                          V_ref_mps=float(np.linalg.norm(reference_velocity)))
            # Export values, including the interleaved variance ordering, must
            # equal the actual stored posterior, rather than recomputed values.
            np.testing.assert_array_equal([record[f"filtered_{axis}_m"] for axis in "xyz"], state[POSITION_INDICES])
            np.testing.assert_array_equal([record[f"filtered_V{axis}_mps"] for axis in "xyz"], state[VELOCITY_INDICES])
            np.testing.assert_array_equal([record[f"P_{axis}{axis}_m2"] for axis in "xyz"], diagonal[POSITION_INDICES])
            np.testing.assert_array_equal([record[f"P_V{axis}V{axis}_m2ps2"] for axis in "xyz"], diagonal[VELOCITY_INDICES])
            records.append(record)
    checks.update(saved_state_exactly_equal=True, saved_variance_exactly_equal=True,
                  velocity_frame_rotation_without_translation=True, original_filter_not_rerun=True)
    metadata = dict(version="1.6", dataset="MMAUD", table_kind="exact saved one-second posterior samples",
                    method_labels=dict(METHODS), record_count=20, sample_count_per_method=10,
                    state_order=STATE_ORDER, frame="primary radar coordinate frame",
                    initialization_timestamp_s=initialization_time, time_origin_ns=manifest["source"]["time_origin_ns"],
                    selected_interval_s=manifest["selected_interval_s"], saved_tick_indices=list(range(0, 901, 100)),
                    sampling_interval_s=1.0, saved_filter_grid_hz=100, filter_rerun=False, filter_interpolation=False,
                    source_sha256=hashes, original_gt_file_sha256=raw_truth_hashes,
                    reference_source="original MMAUD Leica XYZ position files; no measured velocity channel",
                    V_ref_description="centred one-second window-average approximation from linearly interpolated GT positions at t +/- 0.5 s; not directly measured or guaranteed instantaneous velocity",
                    V_ref_formula="R_primary.T @ (p_truth(t+0.5)-p_truth(t-0.5))/1.0",
                    V_ref_scalar_description="norm of the one-second average velocity vector, i.e. net displacement / 1 s; not average travelled-distance speed",
                    V_ref_window_duration_s=1.0, V_ref_half_window_s=VELOCITY_HALF_WINDOW_S,
                    V_ref_variance_m2ps2=dict(Vx=None, Vy=None, Vz=None, V=None),
                    max_gt_interpolation_gap_s=MAX_INTERPOLATION_GAP_S,
                    requested_gt_support_interval_s=support_interval.tolist(),
                    original_gt_file_count=len(truth_labels),
                    gt_boundary_policy="all samples in selected interval +/- 0.6 s plus at most one actual preceding/following support sample; no extrapolation",
                    actual_gt_support_interval_s=raw_truth[[0, -1], 0].tolist(),
                    gt_source_files_in_sample_index_order=truth_labels, reference_interpolation_support=references,
                    primary_R_radar_to_truth=rotation.tolist(), primary_t_radar_in_truth=translation.tolist(),
                    variance_description="posterior Kalman/EKF model covariance diagonal conditional on fixed parameters, not measured error variance; complete calibration-parameter uncertainty is not explicitly propagated; R contains a mixed noise estimate from training residuals; reference-speed variance is unknown",
                    field_units=dict(position="m", velocity="m/s", position_variance="m^2", velocity_variance="m^2/s^2"),
                    numerical_checks=checks,
                    baseline_stagnation_evidence=baseline_evidence,
                    conditions="training-GT-supervised local experimental projection and manual-oracle camera; exploratory held-out clip, not official full calibration or paper Table I replication")
    output.mkdir(parents=True, exist_ok=True)
    document = dict(metadata=metadata, records=records)
    (output / "second_samples.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "second_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _write_report(output, document)
    return document


def _write_report(output, document):
    meta, records = document["metadata"], document["records"]
    lines = ["# 1.6 每隔1秒：位置、速度与后验方差", "",
             "从真实首雷达初始化 t=72.512097 s 起，直接取已保存100 Hz结果的 tick 0、100、…、900；共10个时刻、两种方法20行。没有插值或重新运行滤波。t 为相对初始化的秒数；原时间戳仍完整保留在CSV/JSON。", "",
             "**V_ref（GT位置差分估计，1 s窗口）是以t为中心的1 s平均速度近似：原GT在t±0.5 s做时间线性插值，再差分。MMAUD源GT只有位置，没有直接测速；V_ref不是瞬时精确速度。**", "",
             "V_ref标量是上述平均速度向量的模长，即1 s净位移的长度/1 s；它不等于按实际路径长度计算的平均速率。", "",
             "原GT位置与参考速度使用冻结主刚体变换到同一雷达坐标；速度只乘旋转Rᵀ，不加平移。插值不得外推，端点及整个1 s窗口中的任一原采样缺口不得大于0.30 s。参考窗口可使用固定测试段外的邻近实际GT，只用于本表差分，不参与滤波、拟合或调参。", "",
             "**P_xx/P_yy/P_zz、P_VxVx/P_VyVy/P_VzVz为固定参数条件下保存协方差的后验模型对角项，非实测误差方差，也不等于实际误差平方；未显式传播完整标定参数不确定度；R含训练残差的混合噪声估计。GT速度方差未知，JSON为null，不填0。**", "",
             "位置x/y/z单位为m；速度Vx/Vy/Vz与V=√(Vx²+Vy²+Vz²)单位为m/s；位置方差单位m²，速度方差单位m²/s²。下表显示6位小数，CSV/JSON保留完整浮点精度。"]
    for method, label in METHODS:
        lines += ["", f"## {label}：滤波后位置和速度", "",
                  "| t（s） | x（m） | y（m） | z（m） | Vx（m/s） | Vy（m/s） | Vz（m/s） | V（m/s） | V_ref（m/s） |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        fields = ["time_since_initialization_s", "filtered_x_m", "filtered_y_m", "filtered_z_m", "filtered_Vx_mps", "filtered_Vy_mps", "filtered_Vz_mps", "filtered_V_mps", "V_ref_mps"]
        for record in records:
            if record["method"] == method:
                lines.append("| " + " | ".join(f"{record[key]:.6f}" for key in fields) + " |")
    lines += ["", "## 两组共用GT位置和参考速度", "",
              "| t（s） | GT x（m） | GT y（m） | GT z（m） | V_ref,x（m/s） | V_ref,y（m/s） | V_ref,z（m/s） | V_ref（m/s） |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields = ["time_since_initialization_s", "gt_x_m", "gt_y_m", "gt_z_m", "V_ref_x_mps", "V_ref_y_mps", "V_ref_z_mps", "V_ref_mps"]
    for record in records[::2]:
        lines.append("| " + " | ".join(f"{record[key]:.6f}" for key in fields) + " |")
    lines += ["", "## 滤波后验方差P对角项", "",
              "| t（s） | 方法 | P_xx（m²） | P_yy（m²） | P_zz（m²） | P_VxVx（m²/s²） | P_VyVy（m²/s²） | P_VzVz（m²/s²） |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    fields = ["P_xx_m2", "P_yy_m2", "P_zz_m2", "P_VxVx_m2ps2", "P_VyVy_m2ps2", "P_VzVz_m2ps2"]
    for record in records:
        values = " | ".join(f"{record[key]:.6f}" for key in fields)
        lines.append(f"| {record['time_since_initialization_s']:.0f} | {dict(METHODS)[record['method']]} | {values} |")
    evidence = meta["baseline_stagnation_evidence"]
    lines += ["", "## 仅雷达前4秒为何速度为0", "",
              f"t=0–4 s的常数位置和零速度来自冻结NPZ，不是缺值补0。更新日志记录：初始化后连续{evidence['accepted_initial_xyz_repeats_before_change']}次门控后的nearest XYZ选择都返回初始实际观测{evidence['initial_selected_xyz_m']} m，时间为{evidence['repeated_selection_interval_s']} s、创新NIS为0；观测在点云中的索引变化，但XYZ保持逐值相同。",
              f"首次选择不同XYZ发生在{evidence['first_changed_selected_radar_update']['timestamp']:.6f} s，selected_point_index={evidence['first_changed_selected_radar_update']['selected_point_index']}；首次改变的保存状态为tick {evidence['first_changed_saved_tick']}，t_log={evidence['first_changed_saved_timestamp_s']:.6f} s。增强点云保留历史回波与最近点选择使基线长期停留初值，这是该输入和关联方法的局限，不能据此认定真实目标静止。未修改原算法或原结果。",
              "", "## 来源和核对", "",
              "NPZ状态顺序为[x,Vx,y,Vy,z,Vz]，位置/位置方差索引0/2/4，速度/速度方差索引1/3/5。每个导出值均已与保存的state/P逐值核对；两组使用同一参考位置和V_ref。匀速、二次轨迹解析参考、内部缺口拒绝与边界外推拒绝检查通过。",
              f"GT请求支撑区间为{meta['requested_gt_support_interval_s']} s，另保留最多1个前/后边界实际样本，实际为{meta['actual_gt_support_interval_s']} s。首V_ref左端72.012097 s需71.882726/72.083612 s两点支撑，不能在71.9 s人为截断。",
              f"冻结主metrics SHA256：`{meta['source_sha256']['primary_metrics_json']}`。",
              f"冻结模型 SHA256：`{meta['source_sha256']['local_model_json']}`。",
              "输入、原GT归档、所有实际使用GT文件及NPZ摘要，插值样本索引和权重均见second_samples.json；未复制完整原GT。",
              "当前仍是训练GT监督局部实验投影＋人工辅助视觉条件下的探索性短片段对照，不是完整官方标定或原论文表I逐数复现。",
              "", "生成方法：`.venv\\Scripts\\python.exe tools/export_mmaud_second_table.py`。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="导出冻结1.6每隔1秒状态/方差，以及GT位置差分V_ref；不重跑滤波")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/public/mmaud")
    parser.add_argument("--primary", type=Path, default=PROJECT_ROOT / "output/v16_comparison")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/v16_second_table")
    parser.add_argument("--self-check", action="store_true", help="仅检查解析轨迹/缺口，不读取实测数据")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(numerical_checks(), ensure_ascii=False))
        return
    result = export_second_table(args.data, args.primary, args.output)
    print(f"已保存每秒状态表：{len(result['records'])} 行，10时刻×2方法；V_ref为GT位置1 s中心差分近似。")


if __name__ == "__main__":
    main()
