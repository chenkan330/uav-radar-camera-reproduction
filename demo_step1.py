"""Step 1: synthetic Radar/Camera -> handwritten 3D Kalman filter.

Run: python demo_step1.py
The Camera data here are fictitious independent XYZ observations, not pixels.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from kalman3d import H_POSITION, POSITION, VELOCITY, Kalman3D


RADAR_STD = np.array([0.20, 0.35, 0.35])  # XYZ, metres
CAMERA_STD = np.array([0.75, 0.12, 0.12])  # X is depth; synthetic only!
R_RADAR = np.diag(RADAR_STD**2)
R_CAMERA = np.diag(CAMERA_STD**2)


def make_synthetic_data(t, seed=42, trajectory="curve"):
    """Ground truth is used to generate/evaluate data, never fed to the KF."""
    if trajectory == "line":
        position = np.array([2.0, -1.0, 1.5]) + t[:, None] * np.array([0.15, 0.08, 0.03])
        velocity = np.tile([0.15, 0.08, 0.03], (len(t), 1))
    elif trajectory == "curve":
        position = np.column_stack([
            4.0 + 1.2 * np.sin(0.25 * t),
            1.3 * np.sin(0.4 * t),
            1.8 + 0.65 * np.sin(0.3 * t),
        ])
        velocity = np.column_stack([
            0.30 * np.cos(0.25 * t),
            0.52 * np.cos(0.4 * t),
            0.195 * np.cos(0.3 * t),
        ])
    else:
        raise ValueError("trajectory must be 'line' or 'curve'")
    rng = np.random.default_rng(seed)
    radar = position + rng.normal(size=position.shape) * RADAR_STD
    camera = position + rng.normal(size=position.shape) * CAMERA_STD
    state = np.empty((len(t), 6))
    state[:, POSITION], state[:, VELOCITY] = position, velocity
    return state, radar, camera


def run_simulation(seed=42, duration=20.0, dt=0.1, trajectory="curve", acceleration_std=0.8):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(duration) or duration < dt:
        raise ValueError("duration must be finite and at least dt")
    t = np.arange(int(np.floor(duration / dt + 1e-9)) + 1) * dt
    truth, radar, camera = make_synthetic_data(t, seed, trajectory)
    fused = Kalman3D.from_position(radar[0], R_RADAR, acceleration_std=acceleration_std)
    radar_only = Kalman3D.from_position(radar[0], R_RADAR, acceleration_std=acceleration_std)
    camera_only = Kalman3D.from_position(camera[0], R_CAMERA, acceleration_std=acceleration_std)
    state_names = ["predicted", "after_radar", "fused", "radar_only", "camera_only"]
    data = {name: np.empty_like(truth) for name in state_names}
    for name in ["P_predicted", "P_after_radar", "P_fused"]:
        data[name] = np.empty((len(t), 6, 6))
    data.update(time=t, truth=truth, radar=radar, camera=camera)
    traces = []
    initial_state, initial_P = fused.x.copy(), fused.P.copy()

    # t=0: the first Radar measurement has already been used for initialization.
    # Do not predict to t=dt yet, and do not use radar[0] twice.
    data["predicted"][0] = data["after_radar"][0] = fused.x
    data["P_predicted"][0] = data["P_after_radar"][0] = fused.P
    fused.update(camera[0], R_CAMERA)
    data["fused"][0], data["P_fused"][0] = fused.x, fused.P
    data["radar_only"][0], data["camera_only"][0] = radar_only.x, camera_only.x

    for k in range(1, len(t)):
        before_x, before_P = fused.x.copy(), fused.P.copy()
        interval = t[k] - t[k - 1]
        # One prediction, two corrections, all corrections at exactly t[k].
        data["predicted"][k], data["P_predicted"][k] = fused.predict(interval)
        radar_update = fused.update(radar[k], R_RADAR)
        data["after_radar"][k], data["P_after_radar"][k] = fused.x, fused.P
        camera_update = fused.update(camera[k], R_CAMERA)
        data["fused"][k], data["P_fused"][k] = fused.x, fused.P
        for filter_, name, observation, R in [
            (radar_only, "radar_only", radar[k], R_RADAR),
            (camera_only, "camera_only", camera[k], R_CAMERA),
        ]:
            filter_.predict(interval)
            filter_.update(observation, R)
            data[name][k] = filter_.x
        if k <= 3:
            F, Q = fused.motion_matrices(interval)
            traces.append({
                "k": k, "time_s": float(t[k]), "dt_s": float(interval),
                "previous_x": before_x.tolist(), "previous_P": before_P.tolist(),
                "F": F.tolist(), "Q": Q.tolist(), "H": H_POSITION.tolist(),
                "predicted_x": data["predicted"][k].tolist(),
                "predicted_P": data["P_predicted"][k].tolist(),
                **{name: {
                    "z": observation.tolist(), "R": R.tolist(),
                    "innovation": update.innovation.tolist(),
                    "S": update.innovation_covariance.tolist(), "K": update.gain.tolist(),
                    "updated_x": data[state_name][k].tolist(),
                    "updated_P": data[p_name][k].tolist(),
                } for name, observation, R, update, state_name, p_name in [
                    ("radar_update", radar[k], R_RADAR, radar_update, "after_radar", "P_after_radar"),
                    ("camera_update", camera[k], R_CAMERA, camera_update, "fused", "P_fused"),
                ]},
            })

    estimates = {"radar_raw": radar, "camera_raw": camera}
    estimates.update({name: data[name][:, POSITION] for name in ["radar_only", "camera_only", "fused"]})
    rmse = {}
    for name, estimate in estimates.items():
        error = estimate - truth[:, POSITION]
        rmse[name] = {
            "xyz_m": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "position_3d_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        }
    prediction_error = data["predicted"][1:, POSITION] - truth[1:, POSITION]
    metrics = {
        "scope": "STEP 1 SYNTHETIC ONLY: independent XYZ observations; no pixels, point clouds or timing alignment",
        "seed": seed, "trajectory": trajectory, "samples": len(t),
        "dt_s": dt, "duration_s": float(t[-1]), "acceleration_std_m_s2": acceleration_std,
        "radar_std_xyz_m": RADAR_STD.tolist(), "camera_std_xyz_m": CAMERA_STD.tolist(),
        "initial_velocity_std_m_s": 2.0,
        "rmse_definition": "sqrt(mean(||estimated_position - true_position||^2)), all samples including initialization",
        "rmse": rmse,
        "prediction_rmse_3d_m_excluding_initialization": float(np.sqrt(np.mean(np.sum(prediction_error**2, axis=1)))),
        "min_posterior_covariance_eigenvalue": float(np.linalg.eigvalsh(data["P_fused"]).min()),
        "max_posterior_covariance_asymmetry": float(np.abs(data["P_fused"] - data["P_fused"].transpose(0, 2, 1)).max()),
    }
    trace = {
        "state_order": ["x", "vx", "y", "vy", "z", "vz"],
        "initialization": {"source": "radar[0], consumed once", "state": initial_state.tolist(), "P": initial_P.tolist()},
        "steps": traces,
    }
    return data, metrics, trace


def save_data(output, data, metrics, trace):
    output.mkdir(parents=True, exist_ok=True)
    for name, content in [("metrics.json", metrics), ("step_trace.json", trace)]:
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "simulation.npz", **data)
    columns = {"time_s": data["time"]}
    for name in ["truth", "radar", "camera", "predicted", "after_radar", "fused", "radar_only", "camera_only"]:
        values = data[name]
        labels = ["x_m", "vx_m_s", "y_m", "vy_m_s", "z_m", "vz_m_s"] if values.shape[1] == 6 else ["x_m", "y_m", "z_m"]
        columns.update({f"{name}_{label}": values[:, i] for i, label in enumerate(labels)})
    with (output / "simulation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def make_plots(output, data, metrics):
    # Keep Matplotlib's cache inside the project for portable/restricted setups.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "#f5f7fb",
        "axes.facecolor": "white", "savefig.facecolor": "#f5f7fb",
    })
    colors = dict(truth="#17243b", radar="#d98b2b", camera="#8596ba", predicted="#9063b3", after_radar="#bd933a", fused="#087e85")
    t, truth = data["time"], data["truth"][:, POSITION]
    fused = data["fused"][:, POSITION]
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    fig.suptitle("STEP 1  |  Handwritten 3D Kalman filter", fontsize=21, fontweight="bold")
    axes = [fig.add_subplot(2, 2, 1, projection="3d")]
    axes += [fig.add_subplot(2, 2, i) for i in [2, 3, 4]]
    ax = axes[0]
    for name in ["radar", "camera"]:
        ax.scatter(*data[name].T, s=7, color=colors[name], alpha=0.23, label=f"{name.title()} XYZ (synthetic)")
    ax.plot(*truth.T, color=colors["truth"], lw=2, label="Ground truth")
    ax.plot(*fused.T, color=colors["fused"], lw=1.8, label="Fused KF")
    ax.set(xlabel="X / depth (m)", ylabel="Y (m)", zlabel="Z (m)", title="3D trajectory")
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=23, azim=-55)
    ax = axes[1]
    for name in ["radar", "camera"]:
        ax.scatter(data[name][:, 0], data[name][:, 1], s=10, alpha=0.22, color=colors[name])
    ax.plot(truth[:, 0], truth[:, 1], color=colors["truth"], lw=2, label="Ground truth")
    ax.plot(fused[:, 0], fused[:, 1], color=colors["fused"], lw=1.8, label="Fused KF")
    ax.scatter(*truth[0, :2], s=75, marker="*", color=colors["truth"], label="Start", zorder=5)
    ax.set(title="XY view: noise reduction while tracking motion", xlabel="X / depth (m)", ylabel="Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9)
    ax = axes[2]
    for name, label in [("radar", "Raw Radar"), ("camera", "Synthetic Camera")]:
        ax.plot(t, np.linalg.norm(data[name] - truth, axis=1), color=colors[name], alpha=0.48, lw=0.8, label=label)
    ax.plot(t[1:], np.linalg.norm(data["predicted"][1:, POSITION] - truth[1:], axis=1), color=colors["predicted"], lw=1, alpha=0.8, label="Before updates")
    ax.plot(t, np.linalg.norm(fused - truth, axis=1), color=colors["fused"], lw=1.7, label="After both updates")
    ax.set(title="3D position error over time", xlabel="Time (s)", ylabel="Euclidean error (m)")
    ax.legend(ncol=2, fontsize=8)
    ax = axes[3]
    names = ["radar_raw", "camera_raw", "radar_only", "camera_only", "fused"]
    values = [metrics["rmse"][name]["position_3d_m"] for name in names]
    bars = ax.barh(["Raw Radar", "Synthetic Camera", "Radar-only KF", "Camera-only KF", "Fused KF"], values,
                   color=[colors["radar"], colors["camera"], "#b5a282", "#aeb9ce", colors["fused"]], height=0.58)
    ax.bar_label(bars, fmt="%.3f m", padding=5, fontsize=11)
    ax.invert_yaxis()
    ax.set(xlabel="3D position RMSE (m), all samples", title="Same simulated data, same motion model", xlim=(0, max(values) * 1.27))
    fig.text(0.5, -0.015, f"Seed {metrics['seed']}  |  dt = {metrics['dt_s']:g} s  |  Both sensors provide independent synthetic XYZ; this is not a real-data result.", ha="center", fontsize=10, color="#526079")
    fig.savefig(output / "overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, layout="constrained")
    fig.suptitle("Prediction -> Radar update -> Camera update", fontsize=20, fontweight="bold")
    # A short window exposes individual updates. All stages of a sample share t.
    left = min(5.0, t[-1] * 0.25)
    right = min(left + 2.5, t[-1])
    for i, ax in enumerate(axes):
        index = POSITION[i]
        ax.scatter(t, data["radar"][:, i], color=colors["radar"], s=22, marker="x", alpha=0.65, label="Radar XYZ")
        ax.scatter(t, data["camera"][:, i], color=colors["camera"], s=18, alpha=0.5, label="Synthetic Camera XYZ")
        ax.plot(t, truth[:, i], color=colors["truth"], lw=2, label="Ground truth")
        for name, marker, label in [("predicted", "^", "Prediction"), ("after_radar", "s", "After Radar"), ("fused", "o", "After Camera (fused)")]:
            start = 1 if name == "predicted" else 0
            ax.plot(t[start:], data[name][start:, index], color=colors[name], marker=marker, markersize=3.5, lw=1.2, label=label)
        sigma = np.sqrt(data["P_fused"][:, index, index])
        ax.fill_between(t, fused[:, i] - 2 * sigma, fused[:, i] + 2 * sigma, color=colors["fused"], alpha=0.1, label="Fused +/- 2 sigma (model)")
        visible = (t >= left) & (t <= right)
        shown = np.concatenate([data["radar"][visible, i], data["camera"][visible, i], truth[visible, i], fused[visible, i] - 2*sigma[visible], fused[visible, i] + 2*sigma[visible]])
        margin = max(0.05, np.ptp(shown) * 0.12)
        ax.set(ylabel=f"{'XYZ'[i]} (m)", xlim=(left, right), ylim=(shown.min()-margin, shown.max()+margin))
    axes[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.31))
    axes[-1].set_xlabel("Time (s) | At each time: predict once, update twice")
    fig.savefig(output / "prediction_update.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=20.0, help="seconds")
    parser.add_argument("--dt", type=float, default=0.1, help="seconds, shared fixed interval")
    parser.add_argument("--trajectory", choices=["curve", "line"], default="curve")
    parser.add_argument("--acceleration-std", type=float, default=0.8, help="m/s^2, per-step discrete acceleration std")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "step1")
    parser.add_argument("--no-plots", action="store_true", help="run numerical simulation with NumPy only")
    args = parser.parse_args()
    data, metrics, trace = run_simulation(args.seed, args.duration, args.dt, args.trajectory, args.acceleration_std)
    save_data(args.output, data, metrics, trace)
    if not args.no_plots:
        make_plots(args.output, data, metrics)
    print("STEP 1: synthetic XYZ only; handwritten 3D Kalman filter")
    print(f"{metrics['samples']} samples, dt={args.dt:g} s, seed={args.seed}, trajectory={args.trajectory}")
    print("3D position RMSE, all samples (including initialization):")
    for name, value in metrics["rmse"].items():
        print(f"  {name:12s} {value['position_3d_m']:.4f} m")
    print("\nFirst 3 cycles, XYZ in metres:")
    for k in range(1, min(4, len(data["time"]))):
        parts = [f"{name}={np.array2string(data[name][k, POSITION], precision=3)}" for name in ["predicted", "after_radar", "fused"]]
        print(f"  t={data['time'][k]:.1f}s: " + " -> ".join(parts))
    print(f"\nOutputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
