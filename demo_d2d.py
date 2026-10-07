"""Controlled v2.0 synthetic experiment; never a real-flight benchmark.

All methods see the same noisy measurements and first-radar initialization.
Ego pose is an exact synthetic oracle to isolate geometry, not a measured VIO.
The v1.5 baseline only adapts Cartesian position and has no fictitious vx.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from d2d import (CameraIntrinsics, D2DEKF, EgoPose, POSITION, VELOCITY,
                 SensorExtrinsics, camera_model, radar_model, sensor_kinematics)
from kalman3d import Kalman3D
from plot_style import configure_chinese_font

METHODS = {
    "v15_position_adapter": "1.5位置接口适配基线",
    "v2_radar": "2.0仅雷达",
    "v2_range_rate_camera": "2.0距离·径向速度·相机",
    "v2_full": "2.0完整观测融合",
    "frozen_ego_fault": "固定自机状态（故障消融）",
}
SCENARIOS = {
    "static_line": "固定自机·匀速目标",
    "static_maneuver": "固定自机·机动目标",
    "moving_line": "运动自机·匀速目标",
    "moving_maneuver": "运动自机·机动目标",
}
PLOT_METHODS = {key: value for key, value in METHODS.items() if key != "frozen_ego_fault"}
RADAR_STD = np.array([0.25, 0.15, np.deg2rad(0.8), np.deg2rad(1.2)])
RADAR_R = np.diag(RADAR_STD**2)
CAMERA_R = np.diag([4.0, 4.0])
RADAR_EXTRINSIC = SensorExtrinsics(np.array([0.18, 0.05, 0.02]))
CAMERA_EXTRINSIC = SensorExtrinsics(
    np.array([0.15, -0.04, 0.04]),
    np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]),
)
INTRINSICS = CameraIntrinsics(600., 600., 640., 360.)
DT = 0.1
ACCELERATION_PSD = 0.2


def fixture(timestamp, scenario):
    """Generate world target truth and exact ego kinematics, separately."""
    t = timestamp
    if scenario.endswith("line"):
        position = np.array([35 + .25*t, 2 + .07*t, 4 + .02*t])
        velocity = np.array([.25, .07, .02])
    else:
        position = np.array([35 + 2*np.sin(.4*t), 3*np.sin(.3*t), 5 + np.sin(.35*t)])
        velocity = np.array([.8*np.cos(.4*t), .9*np.cos(.3*t), .35*np.cos(.35*t)])
    truth = np.empty(6)
    truth[POSITION], truth[VELOCITY] = position, velocity
    if scenario.startswith("static"):
        ego = EgoPose(np.zeros(3), np.zeros(3))
    else:
        angle = .15*np.sin(.18*t)
        c, s = np.cos(angle), np.sin(angle)
        ego = EgoPose(
            np.array([3*np.sin(.2*t), 4*np.sin(.25*t), np.sin(.17*t)]),
            np.array([.6*np.cos(.2*t), np.cos(.25*t), .17*np.cos(.17*t)]),
            np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]),
            np.array([0., 0., .027*np.cos(.18*t)]),
        )
    return truth, ego


def position_from_radar(observation, ego):
    """Convert measured range/angles, not truth, into world XYZ and its R."""
    distance, _, azimuth, elevation = observation
    ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(elevation), np.sin(elevation)
    direction = np.array([ce*ca, ce*sa, se])
    J = np.column_stack((direction, distance*np.array([-ce*sa, ce*ca, 0.]),
                         distance*np.array([-se*ca, -se*sa, ce])))
    sensor = sensor_kinematics(ego, RADAR_EXTRINSIC)
    world = sensor.position + sensor.rotation @ (distance*direction)
    J = sensor.rotation @ J
    covariance = J @ RADAR_R[np.ix_([0, 2, 3], [0, 2, 3])] @ J.T
    return world, covariance


def generate_observations(seed, scenario, duration):
    rng = np.random.default_rng(seed)
    times = np.arange(int(round(duration/DT)) + 1)*DT
    truth, egos, radar, pixels = [], [], [], []
    for t in times:
        state, ego = fixture(t, scenario)
        truth.append(state)
        egos.append(ego)
        radar.append(radar_model(state, ego, RADAR_EXTRINSIC).z + rng.normal(size=4)*RADAR_STD)
        pixels.append(camera_model(state, ego, INTRINSICS, CAMERA_EXTRINSIC).z + rng.normal(size=2)*2.)
    return times, np.asarray(truth), egos, np.asarray(radar), np.asarray(pixels)


def track(method, times, egos, radar, pixels):
    """Tracker receives no target truth; first radar is used only once."""
    position, R_position = position_from_radar(radar[0], egos[0])
    state = np.zeros(6)
    state[POSITION] = position
    P = np.zeros((6, 6))
    P[np.ix_(POSITION, POSITION)] = R_position
    P[np.ix_(VELOCITY, VELOCITY)] = np.eye(3)*9.
    if method == "v15_position_adapter":
        # Match velocity diffusion q*dt at the experiment's fixed cadence;
        # v1.5's discrete Q still differs in position/cross terms.
        tracker = Kalman3D(state, P, acceleration_std=np.sqrt(ACCELERATION_PSD/DT))
    else:
        tracker = D2DEKF(state, P, acceleration_psd=ACCELERATION_PSD)
    frozen_ego = EgoPose(egos[0].position, np.zeros(3), egos[0].rotation)
    states, covariances, innovations = [], [], []
    for i, t in enumerate(times):
        if i:
            tracker.predict(float(t-times[i-1]))
        if method == "v15_position_adapter":
            if i:
                position, covariance = position_from_radar(radar[i], egos[i])
                result = tracker.update(position, covariance)
                nis = float(result.innovation @ np.linalg.solve(result.innovation_covariance, result.innovation))
                innovations.append((t, "radar_xyz", 3, nis, True))
        else:
            ego = frozen_ego if method == "frozen_ego_fault" else egos[i]
            if i:
                use_angles = method != "v2_range_rate_camera"
                result = tracker.update_radar(
                    radar[i] if use_angles else radar[i, :2],
                    RADAR_R if use_angles else RADAR_R[:2, :2], ego,
                    RADAR_EXTRINSIC, include_angles=use_angles,
                    gate_nis=13.2767 if use_angles else 9.2103,
                )
                innovations.append((t, "radar", len(result.innovation), result.nis, result.accepted))
            if method != "v2_radar":
                result = tracker.update_camera(pixels[i], CAMERA_R, ego, INTRINSICS,
                                               CAMERA_EXTRINSIC, gate_nis=9.2103)
                innovations.append((t, "camera", 2, result.nis, result.accepted))
        states.append(tracker.x.copy())
        covariances.append(tracker.P.copy())
    return np.asarray(states), np.asarray(covariances), innovations


def metrics(states, covariances, truth, innovations):
    error = states - truth
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(error, covariances)])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, POSITION]**2, axis=1)))),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(error[:, VELOCITY]**2, axis=1)))),
        "mean_nees_6d": float(np.mean(nees)),
        "mean_nis_per_dimension": float(np.mean([x[3]/x[2] for x in innovations])) if innovations else None,
        "accepted_updates": sum(bool(x[4]) for x in innovations),
        "rejected_updates": sum(not x[4] for x in innovations),
        "minimum_covariance_eigenvalue": float(min(np.linalg.eigvalsh(P)[0] for P in covariances)),
    }


def plot_results(output, runs, traces, truth_by_scenario, times):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, field, title, unit in zip(axes, ("position_rmse_m", "velocity_rmse_m_s"),
                                     ("三维位置均方根误差", "三维速度均方根误差"), ("米", "米/秒")):
        for index, (method, label) in enumerate(PLOT_METHODS.items()):
            means = [np.mean([r[field] for r in runs if r["scenario"] == scenario and r["method"] == method])
                     for scenario in SCENARIOS]
            ax.bar(np.arange(4)+(index-1.5)*.18, means, .18, label=label)
        ax.set_xticks(np.arange(4), list(SCENARIOS.values()), rotation=15)
        ax.set_ylabel(f"{title}（{unit}）")
        ax.set_title(title+"·三个种子均值")
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("2.0受控合成实验：准确自机输入；故障消融另见报告完整表格")
    fig.tight_layout(rect=[0, .14, 1, .94])
    fig.savefig(output/"comparison.png", dpi=150)
    plt.close(fig)
    scenario = "moving_maneuver"
    truth = truth_by_scenario[scenario]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for method, label in PLOT_METHODS.items():
        states = traces[(scenario, method)]
        axes[0].plot(times, np.linalg.norm(states[:, POSITION]-truth[:, POSITION], axis=1), label=label)
        axes[1].plot(times, np.linalg.norm(states[:, VELOCITY]-truth[:, VELOCITY], axis=1), label=label)
    for ax, title, unit in zip(axes, ("位置误差", "速度误差"), ("米", "米/秒")):
        ax.set_xlabel("时间（秒）")
        ax.set_ylabel(f"{title}（{unit}）")
        ax.set_title(title+"·包含启动阶段")
        ax.grid(alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("运动自机与机动目标·首个种子·合成数据")
    fig.tight_layout(rect=[0, .14, 1, .94])
    fig.savefig(output/"moving_maneuver.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="2.0无人机对无人机感知：受控合成实验")
    parser.add_argument("--output", type=Path, default=Path("output/local_d2d"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--duration", type=float, default=12.)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.seeds != [42, 43, 44] and not args.no_plots:
        parser.error("自定义种子请加--no-plots；参考图按三个默认种子标注")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("种子必须是互不重复的非负整数")
    if not np.isfinite(args.duration) or args.duration < DT or args.duration > 60:
        parser.error("实验时长必须在0.1至60秒之间")
    if not np.isclose(args.duration/DT, round(args.duration/DT)):
        parser.error("实验时长必须是0.1秒的整数倍")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    runs, traces, truth_by_scenario, archive = [], {}, {}, {}
    with (output/"states.csv").open("w", newline="", encoding="utf-8") as file, \
         (output/"innovations.csv").open("w", newline="", encoding="utf-8") as innovation_file:
        writer, iw = csv.writer(file), csv.writer(innovation_file)
        writer.writerow(["scenario", "seed", "method", "t_s", "x_m", "vx_m_s", "y_m", "vy_m_s",
                         "z_m", "vz_m_s", "truth_x_m", "truth_vx_m_s", "truth_y_m", "truth_vy_m_s",
                         "truth_z_m", "truth_vz_m_s", "P_x", "P_vx", "P_y", "P_vy", "P_z", "P_vz",
                         "relative_body_x_m", "relative_body_y_m", "relative_body_z_m",
                         "relative_body_vx_m_s", "relative_body_vy_m_s", "relative_body_vz_m_s"])
        iw.writerow(["scenario", "seed", "method", "t_s", "sensor", "dimension", "nis", "accepted"])
        for scenario in SCENARIOS:
            for seed in args.seeds:
                times, truth, egos, radar, pixels = generate_observations(seed, scenario, args.duration)
                for method in METHODS:
                    states, covariance, innovations = track(method, times, egos, radar, pixels)
                    values = metrics(states, covariance, truth, innovations)
                    runs.append({"scenario": scenario, "seed": seed, "method": method, **values})
                    for t, state, P, gt, ego in zip(times, states, covariance, truth, egos):
                        relative_position = ego.rotation.T @ (state[POSITION]-ego.position)
                        relative_velocity = ego.rotation.T @ (state[VELOCITY]-ego.velocity)
                        writer.writerow([scenario, seed, method, t, *state, *gt, *np.diag(P),
                                         *relative_position, *relative_velocity])
                    prefix = f"{scenario}_seed{seed}_{method}"
                    archive[prefix+"_states"] = states
                    archive[prefix+"_covariances"] = covariance
                    for row in innovations:
                        iw.writerow([scenario, seed, method, *row])
                    if seed == args.seeds[0]:
                        traces[(scenario, method)] = states
                        truth_by_scenario[scenario] = truth
    archive["times_s"] = times
    np.savez_compressed(output/"states_covariances.npz", **archive)
    summary = {
        "version": "2.0-stage1-prototype", "data_kind": "synthetic", "seeds": args.seeds,
        "duration_s": args.duration, "dt_s": DT, "run_count": len(runs),
        "ego_input": "exact synthetic oracle; not VIO/GNSS evaluation",
        "target_truth_input_to_tracker": False, "acceleration_psd_m2_s3": ACCELERATION_PSD,
        "radar_std_range_rate_azimuth_elevation": RADAR_STD.tolist(), "camera_std_pixels": [2., 2.],
        "radar_lever_arm_body_m": RADAR_EXTRINSIC.lever_arm.tolist(),
        "radar_rotation_body_from_sensor": RADAR_EXTRINSIC.rotation.tolist(),
        "camera_lever_arm_body_m": CAMERA_EXTRINSIC.lever_arm.tolist(),
        "camera_rotation_body_from_sensor": CAMERA_EXTRINSIC.rotation.tolist(),
        "camera_intrinsics_fu_fv_u0_v0": [INTRINSICS.fu, INTRINSICS.fv, INTRINSICS.u0, INTRINSICS.v0],
        "initialization": "first noisy radar XYZ; zero velocity; sigma_v=3 m/s; no static-speed gate",
        "gate_nis_radar4": 13.2767, "gate_nis_radar2_camera2": 9.2103,
        "runs": runs,
    }
    (output/"metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for scenario, label in SCENARIOS.items():
        for method, name in METHODS.items():
            selected = [r for r in runs if r["scenario"] == scenario and r["method"] == method]
            rows.append(f"| {label} | {name} | {np.mean([r['position_rmse_m'] for r in selected]):.4f} | "
                        f"{np.mean([r['velocity_rmse_m_s'] for r in selected]):.4f} | "
                        f"{sum(r['rejected_updates'] for r in selected)} |")
    report = "\n".join([
        "# 2.0第1阶段：运动平台融合原型", "", "**数据性质：合成；自机输入为准确的合成先验。**",
        "本报告不代表真实视频检测、真实飞行定位精度或设备实时性能。", "",
        f"四种场景、{len(args.seeds)}个种子、五种配置，共{len(runs)}组实验；启动阶段保留。",
        "每组使用同一噪声观测、时序、首个雷达初始化；目标真值只生成观测和事后评价，不传给track函数。", "",
        "| 场景 | 配置 | 位置均方根误差（米） | 速度均方根误差（米/秒） | 拒绝观测数（所有种子） |",
        "|---|---|---:|---:|---:|", *rows, "",
        "## 怎样理解对照", "",
        "- 1.5位置接口适配基线使用原Kalman3D及测得的雷达XYZ，经当前自机坐标变换；没有给它伪造全局vx。它没有相机和Doppler，不能把与完整观测方法的差异全部归因于滤波器改进。",
        "- 1.5适配基线不使用NIS门控；2.0使用按维数指定的99%阈值。基线仍记录每次更新的创新、NIS和实际接受数。",
        "- 2.0仅雷达与完整观测方法使用相同EKF、过程噪声和初值，比较增加二维相机信息的影响。",
        "- 距离·径向速度·相机配置仅在初始化使用首个雷达方向；之后不用雷达角度，检查相机方位可提供的约束。",
        "- 固定自机状态是显式故障消融，用于说明遗漏自机运动的后果，不能作为算法优越性的主要基线。",
        "- 1.5离散Q与2.0连续Q不同；示例只匹配固定0.1秒时速度扩散量。完整1.5流程仍由冻结版本单独保存。", "",
        "## 当前限制", "",
        "观测同步10 Hz，只有一个目标；无点云关联、视频检测、延迟、掉帧、长时间导航漂移。",
        "运动自机实验仅改变平移和偏航；三轴旋转与杆臂的几何正确性另由数学测试核验。",
        "目标机动与常速度模型存在偏差，因此NIS/NEES是诊断数值，不据此宣称统计一致性。",
        "原型使用首个雷达观测起轨，尚无多帧起轨确认、误起轨删除或搜索策略。",
        "准确自机是受控上界；未来必须加入真实VIO/GNSS状态、测量时刻插值、共享相关误差、标定与实测评价。", "",
        "指标和轨迹：metrics.json、states.csv、innovations.csv；states_covariances.npz保存所有实验的完整状态与协方差。",
        "CSV相对位置以自机机体原点为参考；相对速度为Rᵀ(v目标-v自机)，与旋转机体系位置导数不同，后者还需减ω×相对位置。",
        "主图比较四个正常配置；故障消融在上述完整表格和原始结果中保留。机器字段保持稳定，图表标注使用中文。", "",
        "![误差比较](comparison.png)" if not args.no_plots else "本次未生成图表。", "",
        "![运动自机与机动目标](moving_maneuver.png)" if not args.no_plots else "",
    ])
    (output/"report.md").write_text(report+"\n", encoding="utf-8")
    if not args.no_plots:
        plot_results(output, runs, traces, truth_by_scenario, times)
    print(f"完成{len(runs)}组合成实验；报告：{output/'report.md'}")


if __name__ == "__main__":
    main()
