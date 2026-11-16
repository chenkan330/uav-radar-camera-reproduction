"""第4步，将连续测量切成历史输入和未来标签，并展开第一个样本。"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HISTORY_SIZE = 10     # 每个样本使用最近5个时刻的测量作为输入。
FUTURE_SIZE = 5      # 每个样本监督之后5个时刻的最优波束。
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "step03" / "beam_labels.npz"
OUTPUT_DIR = BASE_DIR / "outputs" / "step04"


def save_csv(filename, header, rows):
    with (OUTPUT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    # 一、读取上一程序保存的带噪测量与真实标签。
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到 {INPUT_FILE}，请先运行 step03_generate_beam_labels.py。")
    if HISTORY_SIZE < 1 or FUTURE_SIZE < 1:
        raise ValueError("历史长度和预测长度都至少为1。")
    with np.load(INPUT_FILE) as stored:
        data = dict(stored)
    time_s = data["time_s"]
    measurements = np.column_stack((
        data["measured_range_m"], data["measured_azimuth_rad"], data["measured_polar_rad"],
    ))
    best_beam = data["best_beam"]
    true_position = data["true_position_m"]
    observed_position = data["observed_position_m"]

    # 二、一个窗口要用到历史5点和未来5点，末尾允许刚好取到最后一个点。
    sample_count = len(time_s) - HISTORY_SIZE - FUTURE_SIZE + 1
    if sample_count < 1:
        raise ValueError("轨迹太短，需要至少包含历史长度加预测长度这么多个采样点。")
    history_values, future_beams, current_positions = [], [], []
    current_observations, history_indices, future_indices = [], [], []
    for start in range(sample_count):
        history_end = start + HISTORY_SIZE  # Python切片不包含右端点。
        future_end = history_end + FUTURE_SIZE
        current_index = history_end - 1     # 当前时刻是历史输入的最后一点。
        history_values.append(measurements[start:history_end])
        future_beams.append(best_beam[history_end:future_end])
        current_positions.append(true_position[current_index])
        current_observations.append(observed_position[current_index])
        history_indices.append(np.arange(start, history_end))
        future_indices.append(np.arange(history_end, future_end))

    # 三、整理成数组。每一行是一个训练样本，整个数组可供后续分批读取。
    x_history = np.asarray(history_values, dtype=np.float32)
    y_beams = np.asarray(future_beams, dtype=np.int64)
    y_current = np.asarray(current_positions, dtype=np.float32)
    current_observed = np.asarray(current_observations, dtype=np.float32)
    history_indices = np.asarray(history_indices, dtype=np.int64)
    future_indices = np.asarray(future_indices, dtype=np.int64)
    current_indices = history_indices[:, -1]
    num_classes = int(data["array_nx"] * data["array_ny"]) + 1

    # 当前只有一条轨迹，因此所有窗口都属于trajectory_id为0的同一条轨迹。
    # 此阶段用于检查样本对应关系。多轨迹数据准备好后，再按轨迹划分训练和验证。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "samples.npz", x_history=x_history, y_future_beams=y_beams,
        y_current_position_m=y_current, current_observed_position_m=current_observed,
        history_indices=history_indices, future_indices=future_indices,
        history_time_s=time_s[history_indices], future_time_s=time_s[future_indices],
        trajectory_id=np.zeros(sample_count, dtype=np.int64),
        input_fields=np.array(["measured_range_m", "measured_azimuth_rad", "measured_polar_rad"]),
        history_size=HISTORY_SIZE, future_size=FUTURE_SIZE, num_classes=num_classes,
        codebook_u=data["codebook_u"], codebook_v=data["codebook_v"],
        dt_s=data["dt_s"], dataset_note="single_trajectory_window_demo",
    )

    # 四、将全部窗口的时间范围和标签写入CSV，便于用Excel核对。
    window_rows = []
    for sample in range(sample_count):
        window_rows.append([
            sample, 0, int(history_indices[sample, 0]), int(current_indices[sample]),
            int(future_indices[sample, 0]), int(future_indices[sample, -1]),
            time_s[history_indices[sample, 0]], time_s[current_indices[sample]],
            time_s[future_indices[sample, 0]], time_s[future_indices[sample, -1]],
            *y_beams[sample].tolist(), *y_current[sample].tolist(),
        ])
    save_csv(
        "windows.csv",
        ["sample_id", "trajectory_id", "history_start_index", "current_index", "future_start_index", "future_end_index",
         "history_start_time_s", "current_time_s", "future_start_time_s", "future_end_time_s"]
        + [f"target_beam_step_{i + 1}" for i in range(FUTURE_SIZE)]
        + ["target_current_x_m", "target_current_y_m", "target_current_z_m"],
        window_rows,
    )

    # 五、单独展开第一个样本。未来时刻的测量输入留空，只展示其波束标签。
    first_rows = []
    for index in range(HISTORY_SIZE + FUTURE_SIZE):
        inputs = ["", "", ""]
        target_beam = ""
        target_position = ["", "", ""]
        if index < HISTORY_SIZE:
            role = "history_input"
            inputs = measurements[index].tolist()
        else:
            role = "future_label"
            target_beam = int(best_beam[index])
        if index == HISTORY_SIZE - 1:
            target_position = true_position[index].tolist()
        first_rows.append([index, index + 1, time_s[index], role,
                           *inputs, target_beam, *target_position])
    save_csv(
        "first_sample.csv",
        ["source_index_zero_based", "source_sample_one_based", "time_s", "role",
         "input_range_m", "input_azimuth_rad", "input_polar_rad", "target_future_beam",
         "target_current_x_m", "target_current_y_m", "target_current_z_m"],
        first_rows,
    )

    # 六、画出第一个窗口。图中采样点从1开始编号，CSV同时保存Python下标。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), gridspec_kw={"height_ratios": [1, 1.65]})
    timeline = axes[0]
    total = HISTORY_SIZE + FUTURE_SIZE
    timeline.axvspan(0.5, HISTORY_SIZE + 0.5, color="#DCECF7", alpha=0.8)
    timeline.axvspan(HISTORY_SIZE + 0.5, total + 0.5, color="#FCEBD9", alpha=0.8)
    timeline.plot(np.arange(1, total + 1), np.zeros(total), color="gray", linewidth=1)
    timeline.scatter(np.arange(1, HISTORY_SIZE + 1), np.zeros(HISTORY_SIZE), color="#2254B1", s=55)
    timeline.scatter(np.arange(HISTORY_SIZE + 1, total + 1), np.zeros(FUTURE_SIZE), color="#C2762C", s=55)
    timeline.text((HISTORY_SIZE + 1) / 2, 0.44, f"历史输入，{HISTORY_SIZE}个测量", ha="center")
    timeline.text(HISTORY_SIZE + (FUTURE_SIZE + 1) / 2, 0.44, f"未来标签，{FUTURE_SIZE}个波束", ha="center")
    timeline.annotate("当前定位标签对应这里", xy=(HISTORY_SIZE, 0), xytext=(HISTORY_SIZE, -0.55),
                      ha="center", arrowprops={"arrowstyle": "->", "color": "gray"})
    timeline.set(xlim=(0.5, total + 0.5), ylim=(-0.9, 0.9), xticks=np.arange(1, total + 1),
                 yticks=[], xlabel="原始轨迹中的采样点编号（从1开始）")
    axes[1].axis("off")
    table_rows = []
    for index in range(total):
        if index < HISTORY_SIZE:
            table_rows.append([index + 1, f"{time_s[index]:.2f}", "输入",
                               f"{measurements[index, 0]:.3f}", f"{measurements[index, 1]:.4f}",
                               f"{measurements[index, 2]:.4f}", ""])
        else:
            table_rows.append([index + 1, f"{time_s[index]:.2f}", "标签", "", "", "", int(best_beam[index])])
    table = axes[1].table(cellText=table_rows,
                         colLabels=["采样点", "时间（s）", "用途", "输入距离（m）", "输入方位角（rad）", "输入极角（rad）", "未来波束标签"],
                         cellLoc="center", loc="center", colWidths=[0.08, 0.10, 0.08, 0.17, 0.20, 0.20, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    current_text = ", ".join(f"{value:.3f}" for value in true_position[HISTORY_SIZE - 1])
    fig.suptitle("第4步示例数据，展开第一个训练窗口", fontsize=13)
    fig.text(0.5, 0.025, f"当前定位标签是采样点{HISTORY_SIZE}的真实坐标（米） [{current_text}]", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "sample_window.png", dpi=160)
    plt.close(fig)

    print(f"原始轨迹 {len(time_s)} 个采样点，历史长度 {HISTORY_SIZE}，预测长度 {FUTURE_SIZE}。")
    print(f"共切出 {sample_count} 个互相重叠的窗口。")
    print(f"历史测量输入 x_history 的形状 {x_history.shape}，每点依次是距离、方位角、极角。")
    print(f"未来波束标签 y_future_beams 的形状 {y_beams.shape}。")
    print(f"当前定位标签 y_current_position_m 的形状 {y_current.shape}。")
    print(f"第一个窗口的历史下标 {history_indices[0]}，未来下标 {future_indices[0]}。")
    print(f"第一个窗口的未来波束标签 {y_beams[0]}。")
    print("当前数据是一条轨迹的窗口示例，多条独立轨迹准备好后再进行训练和验证划分。")
    print(f"输出文件夹 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
