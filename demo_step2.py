"""Controlled synthetic radar clouds, initialization, clutter and missing frames."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from radar import DEFAULT_RADAR_R, METHODS, RadarAssociator, RadarInitializer
from plot_style import METHOD_LABELS, configure_chinese_font


def make_clouds(seed=42, duration=12.0, dt=0.1):
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be finite and positive")
    rng = np.random.default_rng(seed)
    time = np.arange(int(np.floor(duration / dt)) + 1) * dt
    moving_time = np.maximum(time - .6, 0)
    truth = np.zeros((len(time), 6))
    truth[:, 0] = 4 + .8 * np.sin(.4 * moving_time)
    truth[:, 2] = .5 * np.sin(.5 * moving_time)
    truth[:, 4] = 1.5 + .25 * np.sin(.3 * moving_time)
    moving = time > .6
    truth[moving, 1] = .32 * np.cos(.4 * moving_time[moving])
    truth[moving, 3] = .25 * np.cos(.5 * moving_time[moving])
    truth[moving, 5] = .075 * np.cos(.3 * moving_time[moving])
    clouds = []
    for k, target in enumerate(truth):
        if k in (30, 31, 32, 33, 34, 75):
            clouds.append(np.empty((0, 5)))
            continue
        z = target[[0, 2, 4, 1]]
        # Extremely quiet first pair explicitly guarantees the paper's
        # initialization condition in this controlled demonstration.
        noise_std = np.array([.002, .002, .002, .002]) if k < 2 else np.array([.16, .2, .2, .08])
        target_returns = z + rng.normal(size=(3, 4)) * noise_std
        cloud = np.column_stack((target_returns, [12., 9., 7.]))
        clutter = np.column_stack((rng.uniform([-3, -5, -1, -2], [10, 5, 5, 2], (8, 4)),
                                   rng.uniform(1, 11, 8)))
        # Some clutter falls near the target to expose ambiguous association.
        if k > 2 and k % 9 == 0:
            clutter[0, :4] = z + [.35, -.35, .25, .05]
            clutter[0, 4] = 15
        clouds.append(np.vstack((cloud, clutter)))
    return time, truth, clouds


def run(seed=42, duration=12.0, dt=.1, gate=3.0, beta0=.1):
    time, truth, clouds = make_clouds(seed, duration, dt)
    estimates = {}
    metrics = {}
    for method in METHODS:
        initializer = RadarInitializer()
        association = RadarAssociator(method, gate=gate, beta0=beta0)
        state = np.full_like(truth, np.nan)
        kf = None
        counts = np.zeros(len(time), dtype=int)
        accepted = 0
        predicted_only = 0
        eigenvalues = []
        asymmetry = []
        initial_time = None
        for k, (timestamp, cloud) in enumerate(zip(time, clouds)):
            if kf is None:
                kf = initializer.observe(timestamp, cloud)
                if kf is None:
                    continue
                initial_time = float(timestamp)
                # Initialization consumed the second strongest return once.
            else:
                kf.predict(float(timestamp - time[k - 1]))
                result = association.update(kf, cloud)
                counts[k] = result.used_count
                accepted += int(result.accepted)
                predicted_only += int(not result.accepted)
            state[k] = kf.x
            eigenvalues.append(float(np.linalg.eigvalsh(kf.P)[0]))
            asymmetry.append(float(np.max(np.abs(kf.P - kf.P.T))))
        valid = np.all(np.isfinite(state), axis=1)
        if not np.any(valid):
            raise RuntimeError("synthetic initialization did not succeed")
        error = state[valid][:, [0, 2, 4]] - truth[valid][:, [0, 2, 4]]
        estimates[method] = state
        metrics[method] = {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "axis_rmse_m": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "initialized_at_s": initial_time,
            "accepted_updates": accepted,
            "prediction_only_frames": predicted_only,
            "minimum_covariance_eigenvalue": min(eigenvalues),
            "maximum_covariance_asymmetry": max(asymmetry),
            "gated_return_count": int(np.sum(counts)),
        }
    return time, truth, clouds, estimates, metrics


def save(output, seed, time, truth, clouds, estimates, metrics, gate, beta0, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "data_kind": "controlled_synthetic_radar_point_clouds",
        "paper_hardware_reproduction": False,
        "seed": seed,
        "frames": len(time),
        "empty_frames": sum(len(cloud) == 0 for cloud in clouds),
        "state_order": ["x", "vx", "y", "vy", "z", "vz"],
        "cloud_order": ["x", "y", "z", "vx", "linear_intensity"],
        "assumptions": {"gate_mahalanobis_distance": gate, "beta0": beta0,
                        "R": DEFAULT_RADAR_R.tolist(), "acceleration_std_m_s2": .8,
                        "initial_velocity_y_z": 0, "initial_velocity_std_m_s": 2,
                        "centroid_R_policy": "same as one return, no independence claim"},
        "methods": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "simulation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        labels = ["x", "vx", "y", "vy", "z", "vz"]
        writer.writerow(["time_s", "point_count"] + ["truth_" + label for label in labels] +
                        [method + "_" + label for method in METHODS for label in labels])
        for k, timestamp in enumerate(time):
            writer.writerow([timestamp, len(clouds[k]), *truth[k],
                             *(value for method in METHODS for value in estimates[method][k])])
    with (output / "clouds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "x_m", "y_m", "z_m", "vx_m_s", "linear_intensity"])
        for frame, (timestamp, cloud) in enumerate(zip(time, clouds)):
            for point in cloud:
                writer.writerow([frame, timestamp, *point])
    if plots:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        configure_chinese_font()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        colors = dict(zip(METHODS, ("#e67e22", "#2874a6", "#27ae60", "#884ea0", "#c0392b")))
        for axis, index, label in zip(axes.flat, (0, 2, 4), ("X", "Y", "Z")):
            axis.plot(time, truth[:, index], color="black", linewidth=2, label="真实轨迹")
            for method in METHODS:
                axis.plot(time, estimates[method][:, index], color=colors[method], label=METHOD_LABELS[method], alpha=.85)
            for start, end in ((3, 3.5), (7.5, 7.6)):
                axis.axvspan(start, end, color="gray", alpha=.12)
            axis.set(xlabel="时间（秒）", ylabel=f"{label}（米）")
            axis.grid(alpha=.2)
        axes[0, 0].legend(ncol=2, fontsize=8)
        ax = axes[1, 1]
        values = [metrics[method]["position_rmse_m"] for method in METHODS]
        ax.bar([METHOD_LABELS[method] for method in METHODS], values, color=[colors[method] for method in METHODS])
        ax.set(ylabel="三维位置均方根误差（米）", title="合成点云结果，仅用于算法验证")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
        fig.suptitle("第 2 步：雷达初始化、门控与五种关联方法")
        fig.savefig(output / "overview.png", dpi=150)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--gate", type=float, default=3)
    parser.add_argument("--beta0", type=float, default=.1)
    parser.add_argument("--output", default="output/step2")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    time, truth, clouds, estimates, metrics = run(args.seed, args.duration, args.dt, args.gate, args.beta0)
    summary = save(args.output, args.seed, time, truth, clouds, estimates, metrics,
                   args.gate, args.beta0, not args.no_plots)
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
