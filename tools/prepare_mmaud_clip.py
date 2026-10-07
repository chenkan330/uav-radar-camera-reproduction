"""Prepare the fixed held-out clip from audited real files, never synthetic data.

Observation construction has no evaluation-truth input. Separate final export
maps actual Leica samples using the already frozen training-only alignment.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmaud_adapter import (TIME_ORIGIN_NS, CALIBRATION_INTERVALS_UNIX,
                          EVALUATION_INTERVAL_UNIX, build_background,
                          radar_frames_from_files, relative_time, timestamp_ns)

DATA = ROOT / "data/public/mmaud"
CLIP = DATA / "prepared_v16"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(columns); writer.writerows(rows)


def prepare():
    model_path = DATA / "calibration/local_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("usable_for_evaluation") is not True or model.get("evaluation_used_for_fit") is not False:
        raise ValueError("Need a frozen, training-only model that passed the declared local checks")
    if model["fit_intervals_unix_s"] != [list(x) for x in CALIBRATION_INTERVALS_UNIX]:
        raise ValueError("Calibration intervals differ from the fixed selection")
    audit = json.loads((DATA / "evaluation_annotation_audit.json").read_text())
    annotation_path = DATA / "annotations_evaluation.csv"
    if audit.get("visual_review_completed") is not True or audit.get("annotations_sha256") != sha(annotation_path):
        raise ValueError("All actual image annotations need matching completed visual QA")
    CLIP.mkdir(parents=True, exist_ok=True)
    radar_files = list((DATA / "Mavic3/radar_enhance_pcl").glob("*.npy"))
    background = build_background(radar_files)
    radar, radar_audit = radar_frames_from_files(radar_files, background, [EVALUATION_INTERVAL_UNIX])
    if not radar:
        raise ValueError("The fixed clip has no actual target radar snapshot")
    first = radar[0]
    # Choose an observed point nearest the observed median, not the median as a
    # fabricated independent detection, and never a ground-truth-nearest point.
    initial_index = int(np.argmin(np.linalg.norm(first["points"]-first["median_xyz"], axis=1)))
    radar_rows = [[frame["path"].stem, frame["timestamp_s"], *point]
                  for frame in radar for point in frame["points"]]
    csv_write(CLIP / "radar.csv", ["frame_id", "timestamp_s", "x_m", "y_m", "z_m"], radar_rows)
    camera = model["camera_model"]
    K = np.asarray(camera["K"], dtype=float)
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid fitted local projection matrix")
    with annotation_path.open(newline="", encoding="utf-8") as file:
        annotations = list(csv.DictReader(file))
    if len(annotations) != 48:
        raise ValueError("The fixed clip requires all 48 actual image observations")
    camera_rows = []
    raw_hashes = {"independent_Leica.bag": sha(DATA / "2023-08-24-11-14-40_mavic3.bag"),
                  "ground_truth_archive_span.bin": sha(DATA / "ground_truth_archive_span.bin"),
                  "radar_enhance_pcl_archive_span.bin": sha(DATA / "radar_enhance_pcl_archive_span.bin"),
                  "calibration/local_model.json": sha(model_path)}
    for row in annotations:
        image_path = DATA / row["source_image"]
        if sha(image_path) != row["image_sha256"]:
            raise ValueError("Actual image differs from visually reviewed source")
        # The declared model is local pinhole on unrectified fisheye pixels.
        # This is NOT a full-fisheye hardware distortion calibration.
        normalized = [(float(row["u"])-K[0, 2])/K[0, 0],
                      (float(row["v"])-K[1, 2])/K[1, 1]]
        camera_rows.append([float(row["timestamp_s"]), *normalized,
                            row["source_image"], "manual_oracle"])
        raw_hashes[row["source_image"]] = row["image_sha256"]
    csv_write(CLIP / "camera.csv", ["timestamp_s", "normalized_x", "normalized_y", "source_image", "annotation_method"], camera_rows)

    # Only this separate evaluation export touches held-out position values;
    # it cannot modify the already created radar/camera files or parameters.
    R = np.asarray(model["radar_to_truth"]["R"])
    t = np.asarray(model["radar_to_truth"]["t"])
    truth_rows = []
    for path in sorted((DATA / "Mavic3/ground_truth").glob("*.npy"), key=timestamp_ns):
        unix = timestamp_ns(path) / 1e9
        if EVALUATION_INTERVAL_UNIX[0]-.3 <= unix <= EVALUATION_INTERVAL_UNIX[1]+.3:
            point = np.load(path, allow_pickle=False)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("Expected actual finite Leica XYZ")
            truth_rows.append([relative_time(path), *(R.T @ (point-t))])
            raw_hashes["Mavic3/ground_truth/"+path.name] = sha(path)
    csv_write(CLIP / "truth.csv", ["timestamp_s", "x_m", "y_m", "z_m"], truth_rows)
    origin_s = TIME_ORIGIN_NS / 1e9
    selected = [x-origin_s for x in EVALUATION_INTERVAL_UNIX]
    fitted = [[start-origin_s, end-origin_s] for start, end in CALIBRATION_INTERVALS_UNIX]
    source_description = ("仅独立训练段Leica＋人工图像机身中心拟合的局部虚拟针孔投影；"
                          "未知真实鱼眼K/D，不是物理外参或完整鱼眼去畸变。测试GT未参与拟合。")
    manifest = {
        "data_kind": "real", "dataset": "MMAUD", "selected_interval_s": selected,
        "source": {"description": "官方V1 Mavic3增强点云＋双鱼眼左图＋Leica棱镜真值；固定9.5秒探索性对照，人工辅助视觉、训练GT监督局部标定。",
                   "url": "https://ntu-aris.github.io/MMAUD/", "recording_id": "V1_Mavic3_2023-08-24",
                   "time_origin_ns": TIME_ORIGIN_NS, "raw_sha256": raw_hashes},
        "config": {"calibration": {"rotation": camera["R_camera_to_radar"],
                       "translation": camera["t_camera_in_radar"], "noise_norm": camera["noise_norm"],
                       "source_kind": "held_out", "source_description": source_description,
                       "uses_test_truth": False, "fit_intervals_s": fitted,
                       "projection_model": "local_pinhole_experiment", "local_K": camera["K"]},
                   "evaluation": {"max_interpolation_gap_s": .3,
                       "truth_frame_alignment": {"source_kind": "held_out", "uses_test_truth": False,
                           "source_description": "仅两个独立训练段雷达候选簇中位数与Leica位置做Huber刚体拟合，p_truth=R@p_radar+t；测试真值用共同逆变换，不为两方法分别对齐；外参稳定性较弱。",
                           "fit_intervals_s": fitted, "R_radar_to_truth": R.tolist(), "t_radar_in_truth": t.tolist()}},
                   "filter": {"radar_R": model["radar_R"], "acceleration_std": 4.0, "velocity_std": 2.0,
                              "prediction_hz": 100, "radar_gate_nis": 11.344866730144373,
                              "camera_gate_nis": 9.210340371976184, "initial_point_index": initial_index,
                              "init_selection_description": "首个实际雷达最大簇中，选择最靠近该簇中位数的一枚实际观测；无GT，速度为零，首点只用一次。"}},
        "radar_processing": {"within_frame_exact_deduplication": True,
               "background": "initial8s; .2m voxel present>=30 frames", "central_roi": "abs(x)<2,abs(y)<2,z>2 metres",
               "clustering": "largest .6m single-link component", "source_frames": len(radar_audit),
               "identical_snapshots_suppressed": sum(x["identical_to_previous"] for x in radar_audit),
               "accepted_snapshots": len(radar), "covariance_divided_by_point_count": False,
               "initial_file": first["path"].name, "initial_file_sha256": sha(first["path"]),
               "initial_selected_xyz_m": first["points"][initial_index].tolist(),
               "radar_R_status": "training centroid/reference residual covariance; not independently measured pointwise hardware noise"},
        "camera_annotation": {"source": "image-only manually seeded largest-dark-component tracking, all48 visually reviewed",
                              "training_detector_reproduced": False, "camera_half": "left", "sampling_hz": 5,
                              "annotations_sha256": sha(annotation_path), "qa_sha256": sha(DATA / "evaluation_annotation_audit.json")},
        "calibration_limits": model["limitations"] + ["雷达—真值刚体重采样旋转不稳定；仅探索性短片段，需敏感性分析。",
                "增强点云有相关历史回波与时滞，未把训练时滞诊断当作已验证的时钟校正。"],
        "input_sha256": {name: sha(CLIP/name) for name in ("radar.csv", "camera.csv", "truth.csv")}}
    (CLIP / "clip.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (CLIP / "radar_processing_audit.json").write_text(json.dumps(radar_audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Prepared fixed real clip: {len(radar)} radar snapshots, {len(camera_rows)} camera observations; no scoring performed.")


if __name__ == "__main__":
    prepare()
