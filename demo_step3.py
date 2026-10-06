"""Step 3: bounding-box center, camera frames, and two-pixel bearing updates.

All boxes, priors and calibration parameters in this demonstration are
synthetic. Independent prior states isolate one camera update per sample;
the asynchronous tracker is implemented separately in step 4.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from camera import (CameraExtrinsics, CameraIntrinsics, bbox_to_pixels,
                    camera_backproject, camera_bearing_observation,
                    camera_project, paper_pseudo_observation)
from kalman3d import Kalman3D, POSITION
from plot_style import configure_chinese_font


def run_demo(seed=42, samples=201):
    if samples < 2:
        raise ValueError("samples must be at least two")
    rng = np.random.default_rng(seed)
    intrinsics = CameraIntrinsics(420., 420., 320., 240.)
    pose = CameraExtrinsics()
    t = np.linspace(0., 20., samples)
    truth = np.column_stack([8. + .7 * np.sin(.25 * t),
                             .8 * np.sin(.4 * t), 1.2 + .35 * np.sin(.3 * t)])
    clean_pixels = np.array([camera_project(point, intrinsics, pose) for point in truth])
    pixels = clean_pixels + rng.normal(0., 2., (samples, 2))
    confidence = np.full(samples, .95)
    confidence[::9] = .65
    prior = np.zeros((samples, 6))
    prior[:, POSITION] = truth + rng.normal(0., [.7, .35, .35], (samples, 3))
    corrected = prior.copy()
    pseudo_xyz = np.full((samples, 3), np.nan)
    accepted = np.zeros(samples, dtype=bool)
    uncertainty_before = np.array([.7**2, 1., .35**2, 1., .35**2, 1.])
    uncertainty_after = np.tile(uncertainty_before[POSITION], (samples, 1))
    for k in range(samples):
        box = np.r_[pixels[k] - [12., 10.], pixels[k] + [12., 10.]]
        center = bbox_to_pixels(box, confidence[k])
        if center is None:
            continue
        accepted[k] = True
        pseudo = paper_pseudo_observation(center, prior[k], intrinsics, pose,
                                          pixel_std=2., depth_std=.7)
        pseudo_xyz[k] = pseudo.position
        observation = camera_bearing_observation(center, prior[k], intrinsics, pose, pixel_std=2.)
        filt = Kalman3D(prior[k], np.diag(uncertainty_before))
        filt.update_linear(observation.z, observation.R, observation.H)
        corrected[k] = filt.x
        uncertainty_after[k] = np.diag(filt.P)[POSITION]
    roundtrip = np.array([camera_backproject(p, xyz[0], intrinsics, pose)
                          for p, xyz in zip(clean_pixels, truth)])
    before_error = prior[:, POSITION] - truth
    after_error = corrected[:, POSITION] - truth
    metrics = {
        "step": 3, "data_kind": "synthetic", "seed": seed, "samples": samples,
        "accepted_detections": int(accepted.sum()),
        "confidence_threshold": .70, "pixel_std_px": [2., 2.],
        "intrinsics": {"fu": 420., "fv": 420., "u0": 320., "v0": 240.},
        "extrinsics_convention": "global = M.T @ camera - translation",
        "translation_m": [0., 0., 0.],
        "roundtrip_max_abs_error_m": float(np.abs(roundtrip - truth).max()),
        "prior_rmse_3d_m": float(np.sqrt(np.mean(np.sum(before_error**2, axis=1)))),
        "after_bearing_rmse_3d_m": float(np.sqrt(np.mean(np.sum(after_error**2, axis=1)))),
        "prior_lateral_rmse_m": float(np.sqrt(np.mean(np.sum(before_error[:, 1:]**2, axis=1)))),
        "after_bearing_lateral_rmse_m": float(np.sqrt(np.mean(np.sum(after_error[:, 1:]**2, axis=1)))),
        "paper_pseudo_observation_uses_correlated_state_depth": True,
        "caveat": "Independent synthetic priors; no detector training, real camera data, or paper accuracy claim. Bearing mode is an EKF extension to the paper's pseudo-XYZ observation.",
    }
    data = dict(time=t, truth=truth, pixels=pixels, clean_pixels=clean_pixels,
                confidence=confidence, accepted=accepted, prior=prior[:, POSITION],
                corrected=corrected[:, POSITION], pseudo_xyz=pseudo_xyz,
                variance=uncertainty_after)
    return data, metrics


def save_outputs(output, data, metrics, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "camera_observations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_s", "true_x_m", "true_y_m", "true_z_m", "u_px", "v_px", "confidence", "accepted",
                         "prior_x_m", "prior_y_m", "prior_z_m", "bearing_x_m", "bearing_y_m", "bearing_z_m",
                         "paper_x_m", "paper_y_m", "paper_z_m"])
        for k, t in enumerate(data["time"]):
            writer.writerow([t, *data["truth"][k], *data["pixels"][k], data["confidence"][k],
                             int(data["accepted"][k]), *data["prior"][k], *data["corrected"][k], *data["pseudo_xyz"][k]])
    if not plots:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    configure_chinese_font()
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    fig.suptitle("第3步｜相机像素、坐标转换与方位观测更新", fontsize=18, fontweight="bold")
    truth, prior, corrected, t = (data[k] for k in ["truth", "prior", "corrected", "time"])
    accepted = data["accepted"]
    ax = axes[0, 0]
    ax.plot(*data["clean_pixels"].T, color="#17243b", label="真实像素轨迹")
    ax.scatter(*data["pixels"][accepted].T, s=11, color="#087e85", alpha=.5, label="置信度≥0.70")
    ax.scatter(*data["pixels"][~accepted].T, s=28, color="#d98b2b", marker="x", label="低置信度检测（拒绝）")
    ax.set(xlim=(0, 640), ylim=(480, 0), xlabel="水平坐标 u（像素）", ylabel="垂直坐标 v（像素）", title="640×480图像中的合成检测中心")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.scatter(prior[:, 1], prior[:, 2], s=9, alpha=.25, color="#8596ba", label="更新前估计")
    ax.plot(truth[:, 1], truth[:, 2], color="#17243b", lw=2, label="真实轨迹")
    ax.scatter(corrected[accepted, 1], corrected[accepted, 2], s=9, alpha=.6, color="#087e85", label="方位观测更新后")
    ax.set(xlabel="全局 Y 位置（米）", ylabel="全局 Z 位置（米）", title="由像素观测修正横向位置")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.plot(t, prior[:, 0] - truth[:, 0], color="#8596ba", lw=.8, alpha=.7, label="更新前深度误差")
    ax.plot(t, corrected[:, 0] - truth[:, 0], color="#087e85", lw=.8, label="方位观测更新后")
    ax.axhline(0., color="#17243b", alpha=.4)
    ax.set(xlabel="时间（秒）", ylabel="全局 X 误差（米）", title="相机提供方位信息，深度沿用状态估计")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    prior_lateral = np.linalg.norm(prior[:, 1:] - truth[:, 1:], axis=1)
    after_lateral = np.linalg.norm(corrected[:, 1:] - truth[:, 1:], axis=1)
    ax.plot(t, prior_lateral, color="#8596ba", lw=.8, alpha=.7, label="更新前横向误差")
    ax.plot(t, after_lateral, color="#087e85", lw=.8, label="方位观测更新后")
    ax.set(xlabel="时间（秒）", ylabel="横向误差（米）", title="低置信度检测框保留更新前估计")
    ax.legend(fontsize=9)
    fig.get_layout_engine().set(rect=(0, .055, 1, .935))
    fig.text(.5, .014, "合成标定与检测框｜每个独立先验仅进行一次相机更新｜结果仅用于算法验证", ha="center", fontsize=10)
    fig.savefig(output / "camera_observations.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "step3")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    data, metrics = run_demo(args.seed, args.samples)
    save_outputs(args.output, data, metrics, plots=not args.no_plots)
    print(f"STEP 3: {metrics['accepted_detections']}/{metrics['samples']} synthetic detections accepted")
    print(f"Lateral RMSE: prior {metrics['prior_lateral_rmse_m']:.4f} m -> bearing {metrics['after_bearing_lateral_rmse_m']:.4f} m")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
