"""第2步，给距离和角度加入测量误差，并转换成观测坐标。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 这是用于学习流程的测量层仿真。本例分别给距离和两个角度加独立高斯误差。
# 这些标准差是本例自行设定的参数，修改后重新运行可以观察测量误差的变化。
RANGE_STD_M = 1.0          # 距离误差的标准差，单位为米。
AZIMUTH_STD_DEG = 0.5      # 方位角误差的标准差，单位为度。
POLAR_STD_DEG = 0.5        # 相对竖直 +z 轴的极角误差标准差，单位为度。
SEED = 42                 # 固定随机种子，让相同参数得到相同的一组噪声。

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "step01" / "truth.npz"
OUTPUT_DIR = BASE_DIR / "outputs" / "step02"


def main():
    # 一、读取第1步保存的真值。后续计算直接使用NPZ里的弧度制角度。
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到 {INPUT_FILE}，请先运行 step01_generate_truth.py。")
    if min(RANGE_STD_M, AZIMUTH_STD_DEG, POLAR_STD_DEG) < 0:
        raise ValueError("三个误差标准差都应大于或等于0。")
    with np.load(INPUT_FILE) as truth:
        time_s = truth["time_s"]
        true_position = truth["true_position_m"]
        base_station = truth["base_station_m"]
        true_range = truth["range_m"]
        true_azimuth = truth["azimuth_rad"]
        true_polar = truth["polar_rad"]
        dt = float(truth["dt_s"])

    # 二、分别生成三种测量误差。角度标准差先从度转换为弧度。
    rng = np.random.default_rng(SEED)
    count = len(time_s)
    range_noise = rng.normal(0.0, RANGE_STD_M, count)
    azimuth_noise = rng.normal(0.0, np.deg2rad(AZIMUTH_STD_DEG), count)
    polar_noise = rng.normal(0.0, np.deg2rad(POLAR_STD_DEG), count)

    # 三、真实值加上误差，得到测量值。限制范围以保持几何意义。
    measured_range = np.maximum(true_range + range_noise, 1e-6)
    measured_azimuth = true_azimuth + azimuth_noise
    measured_azimuth = (measured_azimuth + np.pi) % (2 * np.pi) - np.pi
    measured_polar = np.clip(true_polar + polar_noise, 0.0, np.pi)

    # 四、用同一份带噪角距测量计算观测坐标。
    # phi从竖直+z轴量起，因此水平分量使用sin(phi)，竖直分量使用cos(phi)。
    observed_relative = np.column_stack((
        measured_range * np.sin(measured_polar) * np.cos(measured_azimuth),
        measured_range * np.sin(measured_polar) * np.sin(measured_azimuth),
        measured_range * np.cos(measured_polar),
    ))
    observed_position = observed_relative + base_station

    # 五、计算这次生成的实际误差，方便检查仿真结果。
    range_error = measured_range - true_range
    azimuth_error = (measured_azimuth - true_azimuth + np.pi) % (2 * np.pi) - np.pi
    polar_error = measured_polar - true_polar
    position_error = np.linalg.norm(observed_position - true_position, axis=1)
    range_rmse = np.sqrt(np.mean(range_error**2))
    azimuth_rmse_deg = np.rad2deg(np.sqrt(np.mean(azimuth_error**2)))
    polar_rmse_deg = np.rad2deg(np.sqrt(np.mean(polar_error**2)))
    position_rmse = np.sqrt(np.mean(position_error**2))

    # 六、保存结果。CSV用度便于查看，NPZ用弧度保留后续计算需要的数据。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((
        time_s, true_position, observed_position, true_range, measured_range,
        np.rad2deg(true_azimuth), np.rad2deg(measured_azimuth),
        np.rad2deg(true_polar), np.rad2deg(measured_polar),
        range_error, np.rad2deg(azimuth_error), np.rad2deg(polar_error), position_error,
    ))
    header = (
        "time_s,true_x_m,true_y_m,true_z_m,observed_x_m,observed_y_m,observed_z_m,"
        "true_range_m,measured_range_m,true_azimuth_deg,measured_azimuth_deg,"
        "true_polar_deg,measured_polar_deg,range_error_m,azimuth_error_deg,"
        "polar_error_deg,position_error_m"
    )
    np.savetxt(OUTPUT_DIR / "measurements.csv", table, delimiter=",", fmt="%.6f",
               header=header, comments="", encoding="utf-8-sig")
    np.savez(
        OUTPUT_DIR / "measurements.npz", time_s=time_s, dt_s=dt,
        true_position_m=true_position, observed_position_m=observed_position,
        base_station_m=base_station, true_range_m=true_range, measured_range_m=measured_range,
        true_azimuth_rad=true_azimuth, measured_azimuth_rad=measured_azimuth,
        true_polar_rad=true_polar, measured_polar_rad=measured_polar,
        range_error_m=range_error, azimuth_error_rad=azimuth_error, polar_error_rad=polar_error,
        position_error_m=position_error, range_std_m=RANGE_STD_M,
        azimuth_std_deg=AZIMUTH_STD_DEG, polar_std_deg=POLAR_STD_DEG, seed=SEED,
        measurement_model="independent_gaussian_range_and_angles_with_range_and_angle_limits",
    )

    # 七、画图，观察角距测量误差最终造成了多大的位置误差。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(true_position[:, 0], true_position[:, 1], label="真实轨迹", color="#197453")
    axes[0, 0].scatter(observed_position[:, 0], observed_position[:, 1], s=9,
                       alpha=0.65, color="#E1842C", label="带误差的观测位置")
    axes[0, 0].set(title="水平面轨迹与观测", xlabel="X（米）", ylabel="Y（米）")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].plot(time_s, range_error, linewidth=1, label="测量距离减去真实距离")
    axes[0, 1].axhline(RANGE_STD_M, color="gray", linestyle="--", linewidth=0.8, label="设定的正负1个标准差")
    axes[0, 1].axhline(-RANGE_STD_M, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 1].set(title="距离测量误差", xlabel="时间（秒）", ylabel="误差（米）")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(time_s, np.rad2deg(azimuth_error), linewidth=1, label="方位角误差")
    axes[1, 0].plot(time_s, np.rad2deg(polar_error), linewidth=1, label="极角误差")
    axes[1, 0].set(title="角度测量误差", xlabel="时间（秒）", ylabel="误差（度）")
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].plot(time_s, position_error, color="#7351A8", linewidth=1, label="每时刻的三维位置误差")
    axes[1, 1].axhline(position_rmse, color="gray", linestyle="--", linewidth=0.8,
                       label=f"整条轨迹RMSE为{position_rmse:.3f}米")
    axes[1, 1].set(title="由角距误差造成的位置误差", xlabel="时间（秒）", ylabel="位置误差（米）")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("第2步示例数据，直接在距离和角度上加入高斯测量误差", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "measurements.png", dpi=160)
    plt.close(fig)

    print(f"已读取第1步的 {count} 个时刻，并生成对应测量值。")
    print(f"设定的标准差 距离{RANGE_STD_M}米，方位角{AZIMUTH_STD_DEG}度，极角{POLAR_STD_DEG}度。")
    print(f"本次距离RMSE {range_rmse:.3f}米。")
    print(f"本次方位角RMSE {azimuth_rmse_deg:.3f}度，极角RMSE {polar_rmse_deg:.3f}度。")
    print(f"本次三维位置RMSE {position_rmse:.3f}米。")
    print(f"输出文件夹 {OUTPUT_DIR}")
    print("打开 measurements.png 看图，打开 measurements.csv 对照真值和测量值。")


if __name__ == "__main__":
    main()
