"""Evaluate all 50 frozen training-bootstrap rig draws, without tuning.

Only the radar-to-truth rigid transform varies. Camera-to-truth pose, local K,
normalized observations, noise, process model, gates and initialization remain
fixed. This is conditional calibration sensitivity, not a confidence interval
or a joint bootstrap of camera/rig uncertainty.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import evaluate_positions
from mmaud_filter import Calib, run_comparison
from run_mmaud_comparison import GROUPS, _sha256, load_observations, load_truth, validate_manifest


def _read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _pose(rotation, translation):
    # Reuse the strict proper-rotation and finite translation validation.
    validated = Calib(rotation, translation, .01)
    return validated.rotation, validated.translation


def _verify_frozen_inputs(clip, model_path, primary_directory):
    manifest = _read_json(clip / "clip.json")
    primary = _read_json(primary_directory / "metrics.json")
    model = _read_json(model_path)
    model_hash = _sha256(model_path)
    expected = primary["manifest"]["source"]["raw_sha256"].get("calibration/local_model.json")
    if model_hash != expected or manifest != primary["manifest"]:
        raise ValueError("当前模型/准备清单与主结果不一致；禁止在输入漂移后混用敏感性结果")
    for name in ("clip.json", "radar.csv", "camera.csv", "truth.csv"):
        # Hash bytes establish provenance; numeric evaluation truth is parsed
        # later, after all tracker runs finish, and is never a filter input.
        if _sha256(clip / name) != primary["input_sha256"][name]:
            raise ValueError(f"准备输入 {name} 与冻结主结果摘要不同")
    calib, parameters, selected, max_gap = validate_manifest(manifest)
    if model.get("evaluation_used_for_fit") is not False or model.get("camera_fit_source") != "training_Leica_only_no_eval_truth":
        raise ValueError("敏感性模型必须明确只使用训练段参考数据，不使用评估段拟合")
    origin = manifest["source"]["time_origin_ns"] / 1e9
    intervals = [[start - origin, end - origin] for start, end in model["fit_intervals_unix_s"]]
    for start, end in intervals:
        if max(start, selected[0]) <= min(end, selected[1]):
            raise ValueError("冻结模型训练区间与评估片段相交")
    camera = model["camera_model"]
    rig = model["radar_to_truth"]
    primary_R, primary_t = _pose(rig["R"], rig["t"])
    camera_R, camera_t = _pose(camera["R_camera_to_truth"], camera["t_camera_in_truth"])
    np.testing.assert_allclose(primary_R.T @ camera_R, calib.rotation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(primary_R.T @ (camera_t - primary_t), calib.translation, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(camera["K"]), np.asarray(manifest["config"]["calibration"]["local_K"]))
    np.testing.assert_array_equal(np.asarray(camera["noise_norm"]), calib.noise_norm)
    alignment = manifest["config"]["evaluation"]["truth_frame_alignment"]
    np.testing.assert_array_equal(primary_R, np.asarray(alignment["R_radar_to_truth"]))
    np.testing.assert_array_equal(primary_t, np.asarray(alignment["t_radar_in_truth"]))
    draws = rig["bootstrap"]
    if not isinstance(draws, list) or len(draws) != 50:
        raise ValueError("必须使用冻结模型的全部50个训练bootstrap draw，不能择优截取")
    validated_draws = [_pose(draw["R"], draw["t"]) for draw in draws]
    return manifest, primary, model, model_hash, calib, parameters, selected, max_gap, validated_draws


def _score(run, truth, max_gap):
    after = run.times >= run.times[0] + 1
    return dict(
        whole_clip=evaluate_positions(run.times, run.positions, truth, max_gap),
        after_first_second=evaluate_positions(run.times[after], run.positions[after], truth, max_gap),
        accepted_camera_updates=sum(item["sensor"] == "camera" and item["accepted"] for item in run.updates),
        accepted_radar_updates=sum(item["sensor"] == "radar" and item["accepted"] and item["reason"] != "initialization once" for item in run.updates),
    )


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("全部draw必须有有效的共同真值覆盖，不能删掉失败draw")
    levels = [0, .025, .25, .5, .75, .975, 1]
    return {name: float(value) for name, value in zip(
        ("minimum", "p2_5", "p25", "median", "p75", "p97_5", "maximum"), np.quantile(values, levels),
    )}


def evaluate_frozen_draws(clip, model_path, primary_directory, output, plots=True):
    clip, model_path = Path(clip), Path(model_path)
    primary_directory, output = Path(primary_directory), Path(output)
    manifest, primary, model, model_hash, calib, parameters, selected, max_gap, poses = _verify_frozen_inputs(
        clip, model_path, primary_directory,
    )
    radar, camera, camera_provenance = load_observations(clip, selected)
    reference_runs = run_comparison(radar, camera, calib, start_time=selected[0], end_time=selected[1], **parameters)
    with np.load(primary_directory / "tracks.npz", allow_pickle=False) as saved:
        for group, _ in GROUPS:
            np.testing.assert_array_equal(reference_runs[group].times, saved["timestamps"])
            np.testing.assert_array_equal(reference_runs[group].state, saved[f"{group}_state"])
            np.testing.assert_array_equal(reference_runs[group].covariance, saved[f"{group}_covariance"])
    primary_rig = model["radar_to_truth"]
    primary_R, primary_t = _pose(primary_rig["R"], primary_rig["t"])
    camera_R, camera_t = _pose(model["camera_model"]["R_camera_to_truth"], model["camera_model"]["t_camera_in_truth"])
    pending = []
    for index, (rig_R, rig_t) in enumerate(poses):
        # Compose the same fixed virtual camera in each sampled radar frame.
        draw_calib = Calib(rig_R.T @ camera_R, rig_R.T @ (camera_t - rig_t), calib.noise_norm)
        runs = run_comparison(radar, camera, draw_calib, start_time=selected[0], end_time=selected[1], **parameters)
        baseline = runs["radar_only"]
        np.testing.assert_array_equal(baseline.state, reference_runs["radar_only"].state)
        np.testing.assert_array_equal(baseline.covariance, reference_runs["radar_only"].covariance)
        for run in runs.values():
            np.testing.assert_array_equal(run.times, baseline.times)
            np.testing.assert_array_equal(run.prediction_grid, reference_runs["radar_only"].prediction_grid)
            np.testing.assert_array_equal(run.initial_source["first_xyz_m"], reference_runs["radar_only"].initial_source["first_xyz_m"])
        pending.append((index, rig_R, rig_t, draw_calib, runs))
        if (index + 1) % 10 == 0:
            print(f"已完成 {index + 1}/50 个固定训练draw的两组跟踪；尚未解析评价真值。", flush=True)

    # All 100 draw-specific filter runs have completed. Parse truth only now.
    prepared_truth = load_truth(clip, selected)
    original_truth_xyz = prepared_truth[:, 1:] @ primary_R.T + primary_t
    scored_primary = {name: _score(run, prepared_truth, max_gap) for name, run in reference_runs.items()}
    for name, _ in GROUPS:
        for scope in ("whole_clip", "after_first_second"):
            for metric in ("mean_euclidean_m", "rmse3d_m", "coverage_fraction"):
                if not np.isclose(scored_primary[name][scope][metric], primary["metrics"][name][scope][metric], atol=1e-12, rtol=1e-12):
                    raise AssertionError("固定主标定未重现主报告指标；不保存混用的敏感性结果")
    results = []
    for index, rig_R, rig_t, draw_calib, runs in pending:
        # Undo the primary prepared-truth transform, then apply this draw's
        # inverse to BOTH groups. No separate per-group truth alignment.
        draw_truth = np.column_stack((prepared_truth[:, 0], (original_truth_xyz - rig_t) @ rig_R))
        metrics = {name: _score(run, draw_truth, max_gap) for name, run in runs.items()}
        for scope in ("whole_clip", "after_first_second"):
            count = metrics["radar_only"][scope]["matched_count"]
            if count != metrics["radar_camera"][scope]["matched_count"] or count != scored_primary["radar_only"][scope]["matched_count"]:
                raise AssertionError("draw真值时间覆盖改变，禁止通过删样本比较")
        results.append(dict(draw_index=index, R_radar_to_truth=rig_R.tolist(), t_radar_in_truth=rig_t.tolist(),
            R_camera_to_radar=draw_calib.rotation.tolist(), t_camera_in_radar=draw_calib.translation.tolist(),
            rotation_delta_degrees=float(primary_rig["bootstrap"][index]["rotation_delta_degrees"]),
            translation_delta_m=primary_rig["bootstrap"][index]["translation_delta_m"], metrics=metrics,
            radar_only_state_exactly_same=True, radar_only_covariance_exactly_same=True,
            shared_grid_exactly_same=True))
    distributions, fraction_better = {}, {}
    for scope in ("whole_clip", "after_first_second"):
        distributions[scope] = {}
        fraction_better[scope] = {}
        for metric in ("mean_euclidean_m", "rmse3d_m"):
            values = {name: [result["metrics"][name][scope][metric] for result in results] for name, _ in GROUPS}
            distributions[scope][metric] = {name: _quantiles(data) for name, data in values.items()}
            fraction_better[scope][metric] = float(np.mean(np.asarray(values["radar_camera"]) < np.asarray(values["radar_only"])))
    summary = dict(
        version="1.6", analysis="conditional training-bootstrap radar-rig sensitivity",
        draw_count=50, all_draws_used=True, selected_interval_s=list(selected),
        model_sha256=model_hash, primary_metrics_sha256=_sha256(primary_directory / "metrics.json"),
        prepared_input_sha256=primary["input_sha256"], fixed_filter_parameters=manifest["config"]["filter"],
        fixed_camera_model=model["camera_model"], training_fit_intervals_unix_s=model["fit_intervals_unix_s"],
        primary_metrics=scored_primary, draw_metrics=results, quantiles=distributions,
        fraction_fusion_better=fraction_better, common_output_count=len(reference_runs["radar_only"].times),
        prepared_camera_frame_count=len(camera_provenance), primary_camera_accepted_updates=scored_primary["radar_camera"]["accepted_camera_updates"],
        radar_only_state_exactly_same_all_draws=True, radar_only_covariance_exactly_same_all_draws=True,
        test_truth_used_for_fitting_or_tuning=False, numeric_truth_parsed_after_all_draw_runs=True,
        limitations=["Conditional on the fixed virtual camera-to-truth fit, K, noise and manual image annotations.",
            "Only radar-to-truth rig draws vary; camera-fit, timing, annotation and cross-parameter uncertainty are not jointly bootstrapped.",
            "Bootstrap draws are training resamples, not independent held-out flights.",
            "Sensitivity quantiles are not confidence intervals or demonstrated population accuracy.",
            "Baseline trajectories stay identical; their scores vary because each draw changes the common truth-frame transform.",
            "Exploratory training-GT-supervised local pinhole and manual-oracle conditions; not official fisheye calibration or Table I replication."])
    output.mkdir(parents=True, exist_ok=True)
    (output / "sensitivity.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(output, summary)
    if plots:
        _plot(output, summary)
    return summary


def _write_report(output, summary):
    lines = ["# 1.6：条件雷达刚体标定敏感性", "",
        "**这是训练重采样的条件 rig 敏感性，不是置信区间；未联合重采样相机模型误差。**", "",
        "使用冻结训练模型中的全部50个刚体变换draw；局部相机到参考坐标的模型、K、归一化视线、噪声、Q/R、门限、初始化与测试段全部保持不变，没有按测试结果择优。",
        "每个draw把相机模型合成到相应雷达坐标，并给两组使用同一个真值逆变换。100次跟踪全部完成后才解析评价真值。", "",
        f"共同输出 {summary['common_output_count']} 点；主结果实际使用相机更新 {summary['primary_camera_accepted_updates']} 次。每个draw两组与主结果的真值覆盖数一致。",
        "所有draw的仅雷达状态、协方差及时间网格逐元素完全相同。其评价误差会变化，是共同参考坐标变换变化所致，不是雷达基线重新调参。", "",
        "| 范围 | 指标 | 组别 | 最小值 | 2.5%分位 | 中位数 | 97.5%分位 | 最大值 |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for scope, scope_label in (("whole_clip", "完整片段主指标"), ("after_first_second", "预设启动1秒后")):
        for metric, metric_label in (("mean_euclidean_m", "平均欧氏误差（米）"), ("rmse3d_m", "三维均方根误差（米）")):
            for name, label in GROUPS:
                q = summary["quantiles"][scope][metric][name]
                values = " | ".join(f"{q[key]:.6f}" for key in ("minimum", "p2_5", "median", "p97_5", "maximum"))
                lines.append(f"| {scope_label} | {metric_label} | {label} | {values} |")
    lines += ["", "融合误差小于仅雷达的draw比例（不是总体概率保证）："]
    for scope, label in (("whole_clip", "完整片段"), ("after_first_second", "启动1秒后")):
        values = summary["fraction_fusion_better"][scope]
        lines.append(f"- {label}：平均欧氏误差 {values['mean_euclidean_m']:.1%}；三维均方根误差 {values['rmse3d_m']:.1%}。")
    lines += ["", f"冻结模型 SHA256：`{summary['model_sha256']}`。主结果、准备文件摘要和每个draw的全部评价见 sensitivity.json。",
        "这些50个draw来自训练重采样，不是50次独立测试飞行；分位范围不能称为置信区间。",
        "局部相机拟合、图像人工观测、时间偏差及它们与rig的相关误差未联合重采样，不能据此宣称完整标定不确定度或泛化到全部MMAUD。",
        "仍属于训练GT监督的局部实验投影＋人工辅助视觉、不稳定标定下的探索性 held-out 对照，不是官方镜头标定或原论文表 I 逐数复现。"]
    (output / "sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(output, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import configure_chinese_font
    configure_chinese_font()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    colors = {"radar_only": "#2877b4", "radar_camera": "#d55e00"}
    for axis, metric, title in zip(axes, ("mean_euclidean_m", "rmse3d_m"), ("平均欧氏误差", "三维均方根误差")):
        values = [[result["metrics"][name]["whole_clip"][metric] for result in summary["draw_metrics"]] for name, _ in GROUPS]
        boxes = []
        for name, label in GROUPS:
            q = summary["quantiles"]["whole_clip"][metric][name]
            boxes.append(dict(label=label, med=q["median"], q1=q["p25"], q3=q["p75"],
                              whislo=q["p2_5"], whishi=q["p97_5"], fliers=[]))
        # Plot the recorded, interpolated quantiles exactly; boxplot's ordinary
        # percentile whiskers otherwise snap to an interior observed draw.
        axis.bxp(boxes, showfliers=False)
        for position, (name, _) in enumerate(GROUPS, 1):
            axis.scatter(position + np.linspace(-.08, .08, 50), values[position - 1], color=colors[name], s=15, alpha=.5)
            axis.scatter(position, summary["primary_metrics"][name]["whole_clip"][metric], marker="D", s=55,
                         color="#222222", zorder=4, label="固定主标定" if position == 1 else None)
        axis.set(ylabel="完整片段误差（米）", title=title)
        axis.grid(axis="y", alpha=.2)
        axis.legend()
    figure.suptitle("条件刚体标定敏感性：全部50次训练重采样，非置信区间\n箱体为25%–75%分位，须线为2.5%–97.5%分位；相机模型固定")
    figure.savefig(output / "sensitivity.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="冻结50训练draw的条件刚体标定敏感性；不调参、不择优")
    parser.add_argument("--clip", type=Path, default=PROJECT_ROOT / "data/public/mmaud/prepared_v16")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "data/public/mmaud/calibration/local_model.json")
    parser.add_argument("--primary", type=Path, default=PROJECT_ROOT / "output/v16_comparison")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/v16_comparison")
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    summary = evaluate_frozen_draws(arguments.clip, arguments.model, arguments.primary, arguments.output, not arguments.no_plots)
    print(f"50draw敏感性完成：全部基线严格一致，主指标融合更好比例 {summary['fraction_fusion_better']['whole_clip']['mean_euclidean_m']:.1%}。")


if __name__ == "__main__":
    main()
