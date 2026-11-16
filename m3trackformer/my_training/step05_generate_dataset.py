"""第5步：批量生成不同轨迹，按轨迹分组，并准备训练、验证和测试数据。"""

import csv
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 导入已有程序中的参数和波束计算函数，不会执行它们的 main()。
from step01_generate_truth import BASE_STATION, DT, DURATION
from step02_generate_measurements import AZIMUTH_STD_DEG, POLAR_STD_DEG, RANGE_STD_M
from step03_generate_beam_labels import NX, NY, calculate_beam_gains
from step04_build_windows import FUTURE_SIZE, HISTORY_SIZE


# 先用100条轨迹学习完整流程。这些是教学参数，不代表论文的完整实验配置。
TRAIN_COUNT = 70
VALIDATION_COUNT = 15
TEST_COUNT = 15
DATA_SEED = 20261109       # 第 i 条轨迹使用 DATA_SEED + i 作为种子。
SPLIT_SEED = 42            # 单独控制哪些轨迹进入哪个数据集。
INITIAL_RADIUS_M = (60.0, 160.0)
HEIGHT_ABOVE_BASE_M = (30.0, 80.0)
SPEED_M_PER_S = (5.0, 15.0)
TURN_RATE_DEG_PER_S = (-15.0, 15.0)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "step05"
INPUT_FIELDS = ["measured_range_m", "measured_azimuth_rad", "measured_polar_rad"]
DATASET_NOTE = "teaching_synthetic_constant_height_straight_or_constant_turn"


def generate_one_trajectory(time_s, trajectory_id, weights_x, weights_y):
    """将前3步的方法应用到一条新轨迹，返回真值、测量和波束标签。"""
    rng = np.random.default_rng(DATA_SEED + trajectory_id)
    radius = rng.uniform(*INITIAL_RADIUS_M)
    initial_azimuth = rng.uniform(-np.pi, np.pi)
    height_above_base = rng.uniform(*HEIGHT_ABOVE_BASE_M)
    speed = rng.uniform(*SPEED_M_PER_S)
    initial_heading_deg = rng.uniform(-180.0, 180.0)
    # 每5条有1条走直线，其余采用不同方向和快慢的匀速转弯。
    turn_rate = 0.0 if trajectory_id % 5 == 0 else rng.uniform(*TURN_RATE_DEG_PER_S)
    position = np.zeros((len(time_s), 3))
    position[0] = BASE_STATION + [radius * np.cos(initial_azimuth),
                                 radius * np.sin(initial_azimuth), height_above_base]
    for index in range(len(time_s) - 1):
        heading = np.deg2rad(initial_heading_deg + turn_rate * time_s[index])
        velocity = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0])
        position[index + 1] = position[index] + velocity * DT

    relative = position - BASE_STATION
    distance = np.linalg.norm(relative, axis=1)
    azimuth = np.arctan2(relative[:, 1], relative[:, 0])
    polar = np.arccos(np.clip(relative[:, 2] / distance, -1.0, 1.0))
    truth = np.column_stack((distance, azimuth, polar))

    # 测量层仿真：给距离、方位角和极角加独立高斯误差。
    measured = truth.copy()
    measured[:, 0] = np.maximum(distance + rng.normal(0, RANGE_STD_M, len(time_s)), 1e-6)
    measured[:, 1] += rng.normal(0, np.deg2rad(AZIMUTH_STD_DEG), len(time_s))
    measured[:, 1] = (measured[:, 1] + np.pi) % (2 * np.pi) - np.pi
    measured[:, 2] = np.clip(polar + rng.normal(0, np.deg2rad(POLAR_STD_DEG), len(time_s)), 0, np.pi)
    measured_range, measured_azimuth, measured_polar = measured.T
    observed_position = BASE_STATION + np.column_stack((
        measured_range * np.sin(measured_polar) * np.cos(measured_azimuth),
        measured_range * np.sin(measured_polar) * np.sin(measured_azimuth),
        measured_range * np.cos(measured_polar),
    ))

    # 监督标签始终由真实方向计算，带噪测量用于模型输入。
    u = np.sin(polar) * np.cos(azimuth)
    v = np.sin(polar) * np.sin(azimuth)
    gains = calculate_beam_gains(u, v, weights_x, weights_y)
    best_beam = gains.argmax(axis=1).astype(np.int64) + 1
    parameters = {
        "seed": DATA_SEED + trajectory_id, "speed_m_per_s": speed,
        "initial_heading_deg": initial_heading_deg, "turn_rate_deg_per_s": turn_rate,
    }
    return position, truth, measured, observed_position, best_beam, parameters


def build_windows(measured, position, observed_position, best_beam, trajectory_id):
    """沿用第4步的时间对齐方式，一条轨迹内部切窗口。"""
    count = len(measured) - HISTORY_SIZE - FUTURE_SIZE + 1
    starts = np.arange(count)[:, None]
    history_indices = starts + np.arange(HISTORY_SIZE)
    future_indices = starts + HISTORY_SIZE + np.arange(FUTURE_SIZE)
    current_indices = history_indices[:, -1]
    return {
        "x_history": measured[history_indices].astype(np.float32),
        "y_future_beams": best_beam[future_indices],
        "y_current_position_m": position[current_indices].astype(np.float32),
        "current_observed_position_m": observed_position[current_indices].astype(np.float32),
        "trajectory_id": np.full(count, trajectory_id, dtype=np.int64),
        "history_indices": history_indices, "future_indices": future_indices,
    }


def save_overview(positions, split_ids, windows_per_trajectory):
    """每组画至多10条真实轨迹，观察数据集包含哪些不同运动。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    names = {"train": "训练集", "validation": "验证集", "test": "测试集"}
    colors = {"train": "#2254B1", "validation": "#C2762C", "test": "#197453"}
    lower = positions[:, :, :2].min(axis=(0, 1)) - 5
    upper = positions[:, :, :2].max(axis=(0, 1)) + 5
    for axis, (split, ids) in zip(axes, split_ids.items()):
        for trajectory_id in ids[:10]:
            xy = positions[trajectory_id, :, :2]
            axis.plot(xy[:, 0], xy[:, 1], color=colors[split], alpha=0.65, linewidth=1)
            axis.scatter(*xy[0], color=colors[split], s=14)
        axis.scatter(*BASE_STATION[:2], color="black", marker="*", s=100)
        axis.set(title=f"{names[split]}：{len(ids)}条轨迹，{len(ids) * windows_per_trajectory}个窗口",
                 xlabel="X（米）", ylabel="Y（米）", xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    fig.suptitle("第5步教学仿真数据：按整条轨迹划分训练、验证和测试集", fontsize=13)
    fig.text(0.5, 0.025, "每组最多展示10条真实轨迹；圆点为起点，星号为基站；各条轨迹高度不同", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "dataset_overview.png", dpi=160)
    plt.close(fig)


def main():
    # 一、读取上方参数，建立时间轴和与第3步完全相同的DFT码本。
    split_counts = {"train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "test": TEST_COUNT}
    if any(not isinstance(count, int) or count < 1 for count in split_counts.values()):
        raise ValueError("三个数据集都应至少有1条轨迹，数量必须是整数。")
    if DT <= 0 or DURATION <= 0 or min(HISTORY_SIZE, FUTURE_SIZE, NX, NY) < 1:
        raise ValueError("采样间隔、持续时间、窗口长度和阵列大小必须为正。")
    if min(RANGE_STD_M, AZIMUTH_STD_DEG, POLAR_STD_DEG) < 0:
        raise ValueError("测量噪声的三个标准差不能为负。")
    time_s = np.arange(int(round(DURATION / DT)) + 1) * DT
    windows_per_trajectory = len(time_s) - HISTORY_SIZE - FUTURE_SIZE + 1
    if windows_per_trajectory < 1:
        raise ValueError("轨迹太短，不足以切出一个完整的历史和未来窗口。")
    grid_u = -1.0 + 2.0 * np.arange(NX) / NX
    grid_v = -1.0 + 2.0 * np.arange(NY) / NY
    weights_x = np.exp(1j * np.pi * np.outer(np.arange(NX), grid_u)) / np.sqrt(NX)
    weights_y = np.exp(1j * np.pi * np.outer(np.arange(NY), grid_v)) / np.sqrt(NY)
    code_u, code_v = np.meshgrid(grid_u, grid_v)

    # 二、先分配整条轨迹的归属，再切窗口。同一轨迹不会出现在两个数据集中。
    total = sum(split_counts.values())
    permutation = np.random.default_rng(SPLIT_SEED).permutation(total)
    split_ids, assignment = {}, np.empty(total, dtype="U10")
    offset = 0
    for split, count in split_counts.items():
        ids = np.sort(permutation[offset:offset + count])
        split_ids[split] = ids
        assignment[ids] = split
        offset += count

    # 三、为每条轨迹重新生成运动和噪声，再应用前面学过的标签和窗口方法。
    positions, true_spherical, measurements, observations, beams, parameters = [], [], [], [], [], []
    grouped_windows = {split: [] for split in split_counts}
    for trajectory_id in range(total):
        position, truth, measured, observed, best_beam, params = generate_one_trajectory(
            time_s, trajectory_id, weights_x, weights_y,
        )
        positions.append(position)
        true_spherical.append(truth)
        measurements.append(measured)
        observations.append(observed)
        beams.append(best_beam)
        parameters.append(params)
        windows = build_windows(measured, position, observed, best_beam, trajectory_id)
        grouped_windows[assignment[trajectory_id]].append(windows)
        if (trajectory_id + 1) % 10 == 0 or trajectory_id + 1 == total:
            print(f"已生成 {trajectory_id + 1}/{total} 条轨迹。", flush=True)

    positions, true_spherical = np.asarray(positions), np.asarray(true_spherical)
    measurements, observations, beams = np.asarray(measurements), np.asarray(observations), np.asarray(beams)
    datasets = {}
    for split, chunks in grouped_windows.items():
        datasets[split] = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}

    # 四、只从训练集的历史输入计算均值和标准差，然后对三个数据集使用同一组参数。
    # 每个特征各自计算：(原始数值 - 训练集均值) / 训练集标准差。
    # 验证集、测试集不参与参数估计。真实标签也不进入输入标准化计算。
    training_inputs = datasets["train"]["x_history"].reshape(-1, 3).astype(np.float64)
    input_mean = training_inputs.mean(axis=0)
    input_std = np.maximum(training_inputs.std(axis=0), 1e-8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common = {
        "input_fields": np.array(INPUT_FIELDS), "history_size": HISTORY_SIZE,
        "future_size": FUTURE_SIZE, "num_classes": NX * NY + 1,
        "codebook_u": code_u.ravel(), "codebook_v": code_v.ravel(),
        "base_station_m": BASE_STATION, "dt_s": DT, "time_s": time_s,
        "input_mean": input_mean, "input_std": input_std, "dataset_note": DATASET_NOTE,
    }
    for split, samples in datasets.items():
        samples["x_history_normalized"] = ((samples["x_history"] - input_mean) / input_std).astype(np.float32)
        np.savez_compressed(OUTPUT_DIR / f"{split}.npz", **samples, **common)
    np.savez_compressed(
        OUTPUT_DIR / "normalization.npz", input_mean=input_mean, input_std=input_std,
        input_fields=np.array(INPUT_FIELDS), fitted_trajectory_ids=split_ids["train"],
        fit_source="training_history_inputs_only",
    )

    # 五、保留原始轨迹和生成参数，后续可以从窗口反查源数据。
    np.savez_compressed(
        OUTPUT_DIR / "trajectories.npz", time_s=time_s, dt_s=DT, base_station_m=BASE_STATION,
        trajectory_id=np.arange(total), split=assignment,
        true_position_m=positions, observed_position_m=observations, best_beam=beams,
        true_range_m=true_spherical[:, :, 0], true_azimuth_rad=true_spherical[:, :, 1],
        true_polar_rad=true_spherical[:, :, 2], measured_range_m=measurements[:, :, 0],
        measured_azimuth_rad=measurements[:, :, 1], measured_polar_rad=measurements[:, :, 2],
        **{key: np.array([params[key] for params in parameters]) for key in parameters[0]},
        array_nx=NX, array_ny=NY, codebook_u=code_u.ravel(), codebook_v=code_v.ravel(),
        dataset_note=DATASET_NOTE,
    )
    with (OUTPUT_DIR / "trajectory_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trajectory_id", "split", "seed", "motion", "speed_m_per_s",
                         "initial_heading_deg", "turn_rate_deg_per_s", "initial_x_m", "initial_y_m",
                         "height_m", "end_x_m", "end_y_m", "window_count", "unique_beam_count"])
        for trajectory_id, params in enumerate(parameters):
            writer.writerow([
                trajectory_id, assignment[trajectory_id], params["seed"],
                "straight" if params["turn_rate_deg_per_s"] == 0 else "constant_turn",
                params["speed_m_per_s"], params["initial_heading_deg"], params["turn_rate_deg_per_s"],
                *positions[trajectory_id, 0], *positions[trajectory_id, -1, :2],
                windows_per_trajectory, len(np.unique(beams[trajectory_id])),
            ])
    config = {
        "dataset_note": DATASET_NOTE, "data_seed": DATA_SEED, "split_seed": SPLIT_SEED,
        "per_trajectory_seed_rule": "data_seed + trajectory_id", "dt_s": DT,
        "requested_duration_s": DURATION, "actual_duration_s": float(time_s[-1]),
        "points_per_trajectory": len(time_s), "windows_per_trajectory": windows_per_trajectory,
        "history_size": HISTORY_SIZE, "future_size": FUTURE_SIZE, "split_counts": split_counts,
        "split_trajectory_ids": {split: ids.tolist() for split, ids in split_ids.items()},
        "base_station_m": BASE_STATION.tolist(), "array_nx": NX, "array_ny": NY,
        "initial_radius_m": INITIAL_RADIUS_M, "height_above_base_m": HEIGHT_ABOVE_BASE_M,
        "speed_m_per_s": SPEED_M_PER_S, "turn_rate_deg_per_s": TURN_RATE_DEG_PER_S,
        "straight_trajectory_rule": "trajectory_id % 5 == 0",
        "range_std_m": RANGE_STD_M, "azimuth_std_deg": AZIMUTH_STD_DEG, "polar_std_deg": POLAR_STD_DEG,
        "measurement_model": "independent_gaussian_range_and_angles_with_range_and_angle_limits",
        "input_fields": INPUT_FIELDS, "normalization_fit_source": "training_history_inputs_only",
        "azimuth_convention": "atan2(y,x), radians in [-pi,pi)",
        "polar_convention": "angle from +z axis, radians in [0,pi]",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "notes": ["教学合成数据，运动模型和噪声参数尚未对齐论文完整实验。",
                  "当前每个时刻均有角距测量，尚未模拟波束选择导致的缺测和重捕获。",
                  "重叠窗口只存在于各自所属的数据集内；测试集留到最终评估使用。"],
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overview(positions, split_ids, windows_per_trajectory)

    print(f"\n共生成 {total} 条轨迹，每条 {len(time_s)} 个采样点、{windows_per_trajectory} 个窗口。")
    for split, samples in datasets.items():
        print(f"{split:10s}: {split_counts[split]} 条轨迹，{len(samples['x_history'])} 个样本，输入形状 {samples['x_history'].shape}。")
    print("已按整条轨迹划分数据集，标准化参数仅使用训练集计算。")
    print("这是用于学习训练流程的合成数据；本程序还没有训练网络。")
    print(f"输出文件夹 {OUTPUT_DIR}")
    print("先打开 dataset_overview.png 看轨迹，打开 trajectory_summary.csv 看每条轨迹的参数和归属。")


if __name__ == "__main__":
    main()
