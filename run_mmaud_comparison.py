"""Run a declared real MMAUD clip after provenance/calibration checks.

The selected test interval and parameters come exclusively from clip.json.
Both filters finish before this runner parses evaluation truth.csv or computes
metrics. Earlier source audit/preparation and training-only calibration may
use reference data as declared. Test truth is not a tracking input. This is an
XYZ/normalized-bearing adaptation, not Table I replication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

import numpy as np

from evaluation import evaluate_positions
from kalman3d import _covariance
from mmaud_filter import Calib, CameraFrame, RadarFrame, run_comparison


GROUPS = (("radar_only", "仅雷达"), ("radar_camera", "雷达＋相机视线"))


def _reject_private_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_private_paths(key)
            _reject_private_paths(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_paths(item)
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if re.search(r"\b[A-Za-z]:[\\/]", value) or value.startswith("\\\\") or PurePosixPath(normalized).is_absolute():
            raise ValueError("输入清单不得含本机绝对路径；请使用相对来源标签")


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须提供实际来源说明")
    if re.search(r"\b[A-Za-z]:[\\/]", value) or value.startswith("\\\\"):
        raise ValueError(f"{name} 请使用可公开的来源标签，不要写本机绝对路径")
    return value.strip()


def _relative_label(value, name):
    value = _text(value, name)
    normalized = value.replace("\\", "/")
    if PureWindowsPath(value).is_absolute() or PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"{name} 必须为安全的相对来源标签")
    return normalized


def _number(value, name, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必须是有限数值") from error
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} 必须是有限{'正' if positive else ''}数值")
    return number


def _interval(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} 必须明确为 [开始秒,结束秒]")
    start, end = [_number(item, name) for item in value]
    if start < 0 or end <= start:
        raise ValueError(f"{name} 必须为非负、递增的相对时间区间")
    return start, end


def _independent_source(value, name, selected):
    value = _object(value, name)
    if value.get("source_kind") not in ("official", "held_out"):
        raise ValueError(f"{name}.source_kind 必须为 official 或 held_out")
    _text(value.get("source_description"), f"{name}.source_description")
    if value.get("uses_test_truth") is not False:
        raise ValueError(f"{name} 必须声明 uses_test_truth=false，禁止用测试段真值拟合")
    fit = value.get("fit_interval_s")
    multiple = value.get("fit_intervals_s")
    if fit is not None and multiple is not None:
        raise ValueError(f"{name} 不得同时设置 fit_interval_s 与 fit_intervals_s")
    if multiple is not None:
        if not isinstance(multiple, list) or not multiple:
            raise ValueError(f"{name}.fit_intervals_s 必须含至少一个独立拟合区间")
        intervals = [_interval(item, f"{name}.fit_intervals_s") for item in multiple]
    elif fit is not None:
        intervals = [_interval(fit, f"{name}.fit_interval_s")]
    else:
        intervals = []
    if not intervals:
        if value["source_kind"] != "official":
            raise ValueError(f"{name} 的 held_out 标定必须注明独立拟合区间")
    else:
        # Check the actually used segments, rather than their bounding extent:
        # independent segments may lie before AND after the evaluation clip.
        for start, end in intervals:
            if max(start, selected[0]) <= min(end, selected[1]):
                raise ValueError(f"{name} 的标定/拟合区间与固定测试段重叠")
    return value


def _hash_declarations(value, name, required=True):
    value = _object(value, name)
    if required and not value:
        raise ValueError(f"{name} 必须保存实际原始素材 SHA256")
    for label, digest in value.items():
        _relative_label(label, f"{name} 的文件标签")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"{name}.{label} 必须是 SHA256 十六进制摘要")
    return value


def validate_manifest(manifest):
    """Validate the data contract before loading observations or any truth."""
    manifest = _object(manifest, "clip.json")
    _reject_private_paths(manifest)
    if manifest.get("data_kind") != "real" or manifest.get("dataset") != "MMAUD":
        raise ValueError("此入口只接受 data_kind=real、dataset=MMAUD 的实际素材")
    source = _object(manifest.get("source"), "source")
    for key in ("description", "url", "recording_id"):
        _text(source.get(key), f"source.{key}")
    origin = source.get("time_origin_ns")
    if not isinstance(origin, int) or isinstance(origin, bool) or origin < 0:
        raise ValueError("source.time_origin_ns 必须是记录相对时钟原点的非负整数纳秒")
    _hash_declarations(source.get("raw_sha256"), "source.raw_sha256")
    if "input_sha256" in manifest:
        _hash_declarations(manifest["input_sha256"], "input_sha256")
    selected = _interval(manifest.get("selected_interval_s"), "selected_interval_s")
    if "test_interval_s" in manifest and _interval(manifest["test_interval_s"], "test_interval_s") != selected:
        raise ValueError("test_interval_s 必须与固定 selected_interval_s 一致")
    config = _object(manifest.get("config"), "config")
    calibration = _independent_source(config.get("calibration"), "config.calibration", selected)
    projection_model = calibration.get("projection_model", "undistorted_pinhole")
    if projection_model not in ("undistorted_pinhole", "local_pinhole_experiment"):
        raise ValueError("projection_model 必须明确为 undistorted_pinhole 或 local_pinhole_experiment")
    if projection_model == "local_pinhole_experiment" and calibration["source_kind"] != "held_out":
        raise ValueError("局部实验投影须标明独立训练段 held_out 来源，不能冒称官方镜头标定")
    for key in ("rotation", "translation", "noise_norm"):
        if key not in calibration:
            raise ValueError(f"config.calibration 缺少实际 {key}")
    calib = Calib(calibration["rotation"], calibration["translation"], calibration["noise_norm"])
    evaluation = _object(config.get("evaluation"), "config.evaluation")
    _independent_source(evaluation.get("truth_frame_alignment"), "config.evaluation.truth_frame_alignment", selected)
    max_gap = _number(evaluation.get("max_interpolation_gap_s"), "max_interpolation_gap_s", True)
    settings = _object(config.get("filter"), "config.filter")
    required = ("radar_R", "acceleration_std", "velocity_std", "prediction_hz",
                "radar_gate_nis", "camera_gate_nis", "initial_point_index", "init_selection_description")
    for key in required:
        if key not in settings:
            raise ValueError(f"config.filter 缺少显式参数 {key}")
    _text(settings["init_selection_description"], "config.filter.init_selection_description")
    permitted = set(required) | {"initial_velocity_from_two", "initial_second_point_index"}
    if set(settings) - permitted:
        raise ValueError(f"未知 filter 参数：{sorted(set(settings) - permitted)}")
    parameters = {key: value for key, value in settings.items() if key != "init_selection_description"}
    parameters["radar_R"] = _covariance(settings["radar_R"], 3, "radar_R", True)
    for key in ("velocity_std", "prediction_hz", "radar_gate_nis", "camera_gate_nis"):
        parameters[key] = _number(settings[key], key, True)
    parameters["acceleration_std"] = _number(settings["acceleration_std"], "acceleration_std")
    if parameters["acceleration_std"] < 0:
        raise ValueError("acceleration_std 必须非负")
    if parameters["prediction_hz"] != 100:
        raise ValueError("1.6 此对照入口固定采用 100 Hz 共同输出网格")
    index = settings["initial_point_index"]
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("initial_point_index 必须是非负整数")
    if "initial_velocity_from_two" in settings and not isinstance(settings["initial_velocity_from_two"], bool):
        raise ValueError("initial_velocity_from_two 必须是布尔值")
    return calib, parameters, selected, max_gap


def _csv_rows(path, columns):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers) or not set(columns).issubset(headers):
            raise ValueError(f"{path.name} 缺少必要列或列名重复：{columns}")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(row.get(column) is None for column in columns):
                raise ValueError(f"{path.name}:{number} CSV 行格式无效")
            rows.append(row)
        return rows


def _timestamp(row):
    timestamp = _number(row["timestamp_s"], "timestamp_s")
    if not 0 <= timestamp < 1e7:
        raise ValueError("timestamp_s 必须是非负相对秒，不得直接传 Unix 十亿秒时间戳")
    return timestamp


def load_observations(directory, selected):
    """Only radar/camera CSV; no access to ground-truth files."""
    directory = Path(directory)
    frames = {}
    for row in _csv_rows(directory / "radar.csv", ("frame_id", "timestamp_s", "x_m", "y_m", "z_m")):
        frame_id = _text(row["frame_id"], "frame_id")
        timestamp = _timestamp(row)
        point = [_number(row[key], key) for key in ("x_m", "y_m", "z_m")]
        if frame_id in frames and frames[frame_id][0] != timestamp:
            raise ValueError(f"同一雷达 frame_id={frame_id!r} 时间不一致")
        frames.setdefault(frame_id, (timestamp, []))[1].append(point)
    radar = [RadarFrame(timestamp, points) for timestamp, points in frames.values()
             if selected[0] <= timestamp <= selected[1]]
    if not radar:
        raise ValueError("固定测试段没有实际雷达候选点")
    camera, provenance = [], []
    for row in _csv_rows(directory / "camera.csv", ("timestamp_s", "normalized_x", "normalized_y", "source_image", "annotation_method")):
        timestamp = _timestamp(row)
        xy = [_number(row[key], key) for key in ("normalized_x", "normalized_y")]
        source_image = _relative_label(row["source_image"], "source_image")
        method = _text(row["annotation_method"], "annotation_method")
        if method not in ("manual_oracle", "detector"):
            raise ValueError("annotation_method 必须为 manual_oracle 或 detector；不接受真值投影")
        if selected[0] <= timestamp <= selected[1]:
            camera.append(CameraFrame(timestamp, xy))
            provenance.append(dict(timestamp_s=timestamp, source_image=source_image, annotation_method=method))
    if not camera:
        raise ValueError("固定测试段没有实际相机视线，无法生成仅雷达/融合对照")
    return radar, camera, provenance


def load_truth(directory, selected):
    """Parse evaluation truth after both runs; retain nearest edge brackets.

    Brackets support interpolation at selected boundaries, not extrapolation
    or a wider reported test interval. Earlier dataset preparation is separate.
    """
    values = []
    for row in _csv_rows(Path(directory) / "truth.csv", ("timestamp_s", "x_m", "y_m", "z_m")):
        timestamp = _timestamp(row)
        xyz = [_number(row[key], key) for key in ("x_m", "y_m", "z_m")]
        values.append([timestamp, *xyz])
    truth = np.asarray(values, dtype=float).reshape(-1, 4)
    if not len(truth):
        raise ValueError("固定测试段没有真实参考位置，不能输出位置精度")
    truth = truth[np.argsort(truth[:, 0], kind="stable")]
    if np.any(np.diff(truth[:, 0]) <= 0):
        raise ValueError("truth.csv 的采集时间必须唯一")
    first = int(np.searchsorted(truth[:, 0], selected[0], side="left"))
    last = int(np.searchsorted(truth[:, 0], selected[1], side="right"))
    return truth[max(0, first - 1):min(len(truth), last + 1)]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_truth(times, truth, max_gap):
    """Plotting alignment using the evaluator's same time-coverage rules."""
    aligned = np.full((len(times), 3), np.nan)
    for index, timestamp in enumerate(times):
        if timestamp < truth[0, 0] or timestamp > truth[-1, 0]:
            continue
        upper = int(np.searchsorted(truth[:, 0], timestamp))
        if upper < len(truth) and truth[upper, 0] == timestamp:
            aligned[index] = truth[upper, 1:]
            continue
        lower = upper - 1
        if lower < 0 or upper >= len(truth):
            continue
        gap = truth[upper, 0] - truth[lower, 0]
        rounding = 4 * np.finfo(float).eps * max(abs(truth[lower, 0]), abs(truth[upper, 0]), max_gap, 1)
        if gap <= max_gap + rounding:
            fraction = (timestamp - truth[lower, 0]) / gap
            aligned[index] = (1 - fraction) * truth[lower, 1:] + fraction * truth[upper, 1:]
    return aligned


def save_results(output, manifest, hashes, runs, metrics, truth, camera_provenance, max_gap, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    oracle = any(item["annotation_method"] == "manual_oracle" for item in camera_provenance)
    projection_model = manifest["config"]["calibration"].get("projection_model", "undistorted_pinhole")
    local_projection = projection_model == "local_pinhole_experiment"
    future_sources = []
    for source_name, source in (("camera_calibration", manifest["config"]["calibration"]),
                                ("truth_frame_alignment", manifest["config"]["evaluation"]["truth_frame_alignment"])):
        intervals = source.get("fit_intervals_s")
        if intervals is None:
            interval = source.get("fit_interval_s")
            intervals = [] if interval is None else [interval]
        if any(interval[0] > manifest["selected_interval_s"][1] for interval in intervals):
            future_sources.append(source_name)
    summary = dict(version="1.6", experiment="MMAUD short-clip XYZ/bearing adaptation",
        data_kind="real", manifest=manifest, input_sha256=hashes,
        raw_hashes_verified_here=False, camera_oracle_condition=oracle,
        projection_model=projection_model, exploratory_heldout_projection=local_projection,
        initial_source=runs["radar_only"].initial_source,
        metrics=metrics, ground_truth_filter_input=False,
        output_grid_hz=manifest["config"]["filter"]["prediction_hz"], shared_prediction_partition=True,
        prepared_camera_observations=len(camera_provenance),
        replayed_camera_events=sum(u["sensor"] == "camera" for u in runs["radar_camera"].updates),
        output_time_range_s=[float(runs["radar_only"].times[0]), float(runs["radar_only"].times[-1])],
        output_samples_are_independent_sensor_measurements=False,
        noncausal_offline_calibration=bool(future_sources),
        calibration_sources_using_later_segments=future_sources,
        primary_scope="whole selected clip including initialization; no error-based trimming",
        secondary_scope="predeclared samples >= initialization timestamp + 1 second",
        caveats=["XYZ-only radar, no fabricated world vx or intensity; nearest association only.",
            "Normalized-bearing EKF adaptation, not paper_xyz or all five paper associations.",
            "Manual image annotations, when present, are an oracle visual condition, not a trained detector.",
            "This runner parsed evaluation truth only after both filters finished; prior source audits/preparation are separate.",
            "Test truth is not used for fitting, association, initialization, annotation, or tuning; training-only supervision is declared separately.",
            "File hashing reads bytes for provenance, not numeric labels supplied to the tracker.",
            "Discrete constant-acceleration Q depends on the common event partition.",
            "Independent calibration/alignment are declared in the manifest; verify their cited evidence.",
            "External-data generalization experiment, not a numerical reproduction of paper Table I."])
    if local_projection:
        summary["caveats"].append("Training-GT-supervised local pinhole mapping: calibration is unstable and exploratory, not complete fisheye rectification or official calibration.")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "camera_provenance.json").write_text(json.dumps(camera_provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "tracks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "timestamp_s", "x_m", "vx_mps", "y_m", "vy_mps", "z_m", "vz_mps",
                         "Pxx", "Pvxvx", "Pyy", "Pvyvy", "Pzz", "Pvzvz"])
        for name, _ in GROUPS:
            for timestamp, state, covariance in zip(runs[name].times, runs[name].state, runs[name].covariance):
                writer.writerow([name, timestamp, *state, *np.diag(covariance)])
    diagnostics = {name: run.updates for name, run in runs.items()}
    (output / "updates.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "tracks.npz", timestamps=runs["radar_only"].times,
        prediction_partition=runs["radar_only"].prediction_grid,
        radar_only_state=runs["radar_only"].state, radar_only_covariance=runs["radar_only"].covariance,
        radar_camera_state=runs["radar_camera"].state, radar_camera_covariance=runs["radar_camera"].covariance)
    interval = manifest["selected_interval_s"]
    lines = ["# 1.6：MMAUD 实测短片段对照", "",
        f"固定测试时间：{interval[0]}–{interval[1]} 秒（相对录制时钟）。",
        "本实验是三维位置雷达＋归一化相机视线的适配，验证外部数据泛化，不能逐数复现论文表 I。", "",
        "两组共同采用 100 Hz 输出及完全一致的预测分区。主要指标覆盖整个固定测试段的可评价输出，包含初始化样本；没有按误差择优截取。",
        "相机条件：" + ("含人工图像标注，属于 oracle 视觉观测；不是作者检测器。" if oracle else "使用实际检测器视线，模型/来源说明见输入清单。"), "",
        "| 评价范围 | 组别 | 平均欧氏误差（米） | 三维均方根误差（米） | 匹配数 | 真值覆盖率 |",
        "|---|---|---:|---:|---:|---:|"]
    if local_projection:
        condition = "训练GT监督的局部实验投影＋" + ("人工辅助视觉" if oracle else "检测器视线")
        lines[2:2] = [f"**实验条件：{condition}；标定不稳定，属于探索性 held-out 对照。**",
                      "该映射不代表官方标定，也不是完整鱼眼镜头去畸变；测试GT未用于拟合、关联、初始化、图像标注或调参。", ""]
    first, last = runs["radar_only"].times[[0, -1]]
    table_index = next(i for i, line in enumerate(lines) if line.startswith("| 评价范围 |"))
    camera_events = summary["replayed_camera_events"]
    lines[table_index:table_index] = [
        f"准备相机图像 {len(camera_provenance)} 帧，实际重放相机事件 {camera_events} 次；早于雷达初始化的图像不追溯更新。",
        f"有效输出时间 {first:.6f}–{last:.6f} 秒，共 {len(runs['radar_only'].times)} 个共同输出时刻；100 Hz插值评价样本不等于独立传感器样本数。",
        f"覆盖率的分母为有效输出时刻；区间开始至初始化的 {(first-interval[0])*1000:.3f} 毫秒没有跟踪输出。", ""]
    def number(value):
        return "无有效真值" if value is None else f"{value:.6f}"
    for scope, scope_label in (("whole_clip", "主指标：完整片段"), ("after_first_second", "补充：预设启动1秒后")):
        for name, label in GROUPS:
            m = metrics[name][scope]
            lines.append(f"| {scope_label} | {label} | {number(m['mean_euclidean_m'])} | {number(m['rmse3d_m'])} | {m['matched_count']} | {m['coverage_fraction']:.1%} |")
    lines += ["", "各轴有符号平均误差、各轴均方根误差、中位数、标准差、95%分位数及时间范围见 metrics.json。",
        "当前入口在两组核心跟踪完成后才解析评价真值；此前素材语义审计和准备可能读取参考数据，不能声称整个流程首次打开真值。测试GT未参与拟合、关联、初始化、图像标注或调参。",
        "相机外参及雷达至真值坐标转换是不同的标定链，分别记录其来源与独立拟合区间。哈希读取文件字节用于溯源，不等于将数值标签提供给跟踪器。",
        "初始化来自调用者指定的真实雷达候选点，默认零速度，选点依据见输入清单；不能认为首点天然是真目标。",
        "完整轨迹及协方差见 tracks.csv / tracks.npz；每次雷达/相机更新时间、NIS与门控结果见 updates.json。",
        "本入口计算准备文件摘要并记录原始素材声明摘要，未重新打开原始 bag 核验原始摘要。",
        "离散 Q 的统计含义依赖共同分区；不冒称四维 world vx 观测、强度加权、完整五关联或作者 MobileNet 检测器复现。"]
    if future_sources:
        lines += ["", "标定使用了测试段之后的独立片段：这是离线、非因果标定，不是仅使用过去信息的在线校准；实际拟合帧仍须逐段排除测试数据。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        _plots(output, runs, metrics, truth, max_gap, oracle, manifest["selected_interval_s"], local_projection)
    return summary


def _plots(output, runs, metrics, truth, max_gap, oracle, selected, local_projection):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import configure_chinese_font
    configure_chinese_font()
    colors = {"radar_only": "#2877b4", "radar_camera": "#d55e00"}
    figure = plt.figure(figsize=(11, 7), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    truth_in_scope = truth[(truth[:, 0] >= selected[0]) & (truth[:, 0] <= selected[1])]
    axis.plot(truth_in_scope[:, 1], truth_in_scope[:, 2], truth_in_scope[:, 3], color="#303030", label="独立参考位置")
    for name, label in GROUPS:
        positions = runs[name].positions
        axis.plot(*positions.T, color=colors[name], label=label)
        axis.scatter(*positions[0], color=colors[name], marker="o", s=35)
    axis.set(xlabel="雷达X位置（米）", ylabel="雷达Y位置（米）", zlabel="雷达Z位置（米）",
             title="实测固定片段三维轨迹" + ("（探索性局部投影＋人工视觉）" if local_projection and oracle else "（人工视觉观测）" if oracle else ""))
    axis.legend()
    figure.savefig(output / "trajectory.png", dpi=150)
    plt.close(figure)
    times = runs["radar_only"].times
    aligned = _aligned_truth(times, truth, max_gap)
    valid_count = int(np.isfinite(aligned[:, 0]).sum())
    for name, _ in GROUPS:
        if valid_count != metrics[name]["whole_clip"]["matched_count"]:
            raise AssertionError("绘图与指标的真值匹配范围不一致")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for axis, index in zip(axes.flat[:3], range(3)):
        for name, label in GROUPS:
            axis.plot(times, runs[name].positions[:, index] - aligned[:, index], color=colors[name], label=label)
        axis.set(xlabel="相对录制时间（秒）", ylabel=f"{'XYZ'[index]}轴位置误差（米）")
        axis.grid(alpha=.25)
    for name, label in GROUPS:
        distances = np.linalg.norm(runs[name].positions - aligned, axis=1)
        axes.flat[3].plot(times, distances, color=colors[name], label=label)
    axes.flat[3].set(xlabel="相对录制时间（秒）", ylabel="三维欧氏误差（米）")
    axes.flat[3].grid(alpha=.25)
    for axis in axes.flat:
        axis.axvline(times[0] + 1, color="#777777", linestyle="--", alpha=.5)
        axis.set_xlim(*selected)
    axes.flat[0].legend()
    figure.suptitle("完整片段误差：虚线为预设1秒启动期终点，缺真值处不连接" +
                   ("\n探索性局部投影＋人工辅助视觉" if local_projection else ""))
    figure.savefig(output / "time_error.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    x = np.arange(2)
    for offset, (name, label) in zip((-.18, .18), GROUPS):
        values = [metrics[name]["whole_clip"][key] for key in ("mean_euclidean_m", "rmse3d_m")]
        if all(value is not None for value in values):
            bars = axis.bar(x + offset, values, width=.36, color=colors[name], label=label)
            axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_xticks(x, ["平均欧氏误差", "三维均方根误差"])
    axis.set(ylabel="误差（米）", title="主指标：完整固定测试片段，包含初始化" +
             ("\n探索性局部投影＋人工辅助视觉" if local_projection else ""))
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    figure.savefig(output / "mean_error_comparison.png", dpi=150)
    plt.close(figure)


def run_prepared_clip(directory, output=Path("output/v16_comparison"), plots=True):
    directory = Path(directory)
    with (directory / "clip.json").open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    calib, parameters, selected, max_gap = validate_manifest(manifest)
    radar, camera, provenance = load_observations(directory, selected)
    # Deliberate separation: the core receives only observations/calibration.
    runs = run_comparison(radar, camera, calib, start_time=selected[0], end_time=selected[1], **parameters)
    a, b = runs["radar_only"], runs["radar_camera"]
    if not np.array_equal(a.times, b.times) or not np.array_equal(a.prediction_grid, b.prediction_grid):
        raise AssertionError("两组时间网格或预测分区不一致")
    # This runner parses evaluation numeric labels only after both complete
    # filter runs. Prior semantic audits/preparation and training-only fitting
    # are separate steps; file hashes are provenance, not filter measurements.
    truth = load_truth(directory, selected)
    hashes = {name: _sha256(directory / name) for name in ("clip.json", "radar.csv", "camera.csv", "truth.csv")}
    for name, expected in manifest.get("input_sha256", {}).items():
        if name not in hashes or hashes[name] != expected.lower():
            raise ValueError(f"准备输入 {name} 与锁定的 SHA256 不一致")
    metrics = {}
    for name, _ in GROUPS:
        result = runs[name]
        after = result.times >= result.times[0] + 1
        metrics[name] = dict(
            whole_clip=evaluate_positions(result.times, result.positions, truth, max_gap),
            after_first_second=evaluate_positions(result.times[after], result.positions[after], truth, max_gap),
            minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(result.covariance).min()),
            maximum_covariance_asymmetry=float(np.max(np.abs(result.covariance - result.covariance.transpose(0, 2, 1)))),
            accepted_radar_updates=sum(u["sensor"] == "radar" and u["accepted"] and u["reason"] != "initialization once" for u in result.updates),
            accepted_camera_updates=sum(u["sensor"] == "camera" and u["accepted"] for u in result.updates),
        )
    for scope in ("whole_clip", "after_first_second"):
        if metrics["radar_only"][scope]["matched_count"] != metrics["radar_camera"][scope]["matched_count"]:
            raise AssertionError("两组真值评价覆盖范围不同")
    return save_results(output, manifest, hashes, runs, metrics, truth, provenance, max_gap, plots)


def main():
    parser = argparse.ArgumentParser(description="MMAUD 真实短片段：固定范围、独立标定、仅雷达/融合公平对照")
    parser.add_argument("--clip", type=Path, required=True, help="已经核验的 prepared clip 目录")
    parser.add_argument("--output", type=Path, default=Path("output/v16_comparison"))
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    try:
        summary = run_prepared_clip(arguments.clip, arguments.output, not arguments.no_plots)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"未生成完整实测结果：{error}\n")
    print(f"真实短片段对照已保存：{arguments.output}")
    print("这是 XYZ/相机视线适配；人工视觉输入为 oracle 条件，不代表作者表 I 完整复现。")
    return summary


if __name__ == "__main__":
    main()
