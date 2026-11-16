"""第 1 步，生成目标真实轨迹，以及相对基站的距离和角度真值。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 将图保存为图片，运行时可以通过文件资源管理器打开。
import matplotlib.pyplot as plt
import numpy as np


# 先阅读这里的参数。本例采用固定高度、匀速转弯，方便逐步检查结果。
DT = 0.05                              # 相邻时刻间隔，单位为秒。
DURATION = 10.0                        # 轨迹持续时间，单位为秒。
BASE_STATION = np.array([0., 0., 10.])  # 基站的 x、y、z 坐标，单位为米。
INITIAL_POSITION = np.array([80., -40., 60.])
SPEED = 10.0                           # 目标的水平速度，单位为米每秒。
INITIAL_HEADING_DEG = 30.0              # 初始运动方向，从 +x 轴向 +y 轴量起。
TURN_RATE_DEG_PER_S = 12.0              # 运动方向每秒增加的角度。

# 输出位置由本文件的位置决定，方便在不同终端工作目录下运行。
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "step01"


def main():
    # 一、建立时间轴，包含初始时刻和最终时刻。
    interval_count = int(round(DURATION / DT))
    time_s = np.arange(interval_count + 1) * DT

    # 二、逐时刻计算真实位置。每次的位置变化等于速度乘以时间间隔。
    position = np.zeros((len(time_s), 3))
    position[0] = INITIAL_POSITION
    for index in range(interval_count):
        heading_deg = INITIAL_HEADING_DEG + TURN_RATE_DEG_PER_S * time_s[index]
        heading_rad = np.deg2rad(heading_deg)
        velocity = np.array([SPEED * np.cos(heading_rad),
                             SPEED * np.sin(heading_rad), 0.0])
        position[index + 1] = position[index] + velocity * DT

    # 三、先减去基站坐标，再计算目标相对基站的距离和角度。
    relative = position - BASE_STATION
    dx, dy, dz = relative[:, 0], relative[:, 1], relative[:, 2]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    azimuth = np.arctan2(dy, dx)  # 方位角，内部采用弧度。
    polar = np.arccos(np.clip(dz / distance, -1.0, 1.0))
    # polar 从竖直 +z 方向量起，与论文中的 phi 定义一致。
    # 上面的运动方向 heading 和此处的目标方位角 azimuth 含义不同。

    # 四、保存数据。CSV 使用角度制便于查看，NPZ 使用弧度便于后续计算。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((time_s, position, distance,
                             np.rad2deg(azimuth), np.rad2deg(polar)))
    np.savetxt(
        OUTPUT_DIR / "truth.csv", table, delimiter=",", fmt="%.6f",
        header="time_s,x_m,y_m,z_m,range_m,azimuth_deg,polar_deg",
        comments="", encoding="utf-8-sig",
    )
    np.savez(
        OUTPUT_DIR / "truth.npz", time_s=time_s,
        true_position_m=position, base_station_m=BASE_STATION,
        range_m=distance, azimuth_rad=azimuth, polar_rad=polar,
        dt_s=DT, speed_m_per_s=SPEED,
        initial_heading_deg=INITIAL_HEADING_DEG,
        turn_rate_deg_per_s=TURN_RATE_DEG_PER_S,
    )

    # 五、画出运动轨迹，以及该轨迹对应的距离和角度。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].plot(position[:, 0], position[:, 1], label="目标轨迹")
    axes[0].scatter(*BASE_STATION[:2], marker="*", s=120, label="基站")
    axes[0].scatter(*position[0, :2], marker="o", s=35, label="目标起点")
    axes[0].set(title=f"水平面轨迹，目标高度固定为{INITIAL_POSITION[2]:g}米", xlabel="X（米）", ylabel="Y（米）")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=9)
    axes[1].plot(time_s, distance)
    axes[1].set(title="目标到基站的距离", xlabel="时间（秒）", ylabel="距离（米）")
    axes[2].plot(time_s, np.rad2deg(azimuth), label="方位角 theta")
    axes[2].plot(time_s, np.rad2deg(polar), label="相对+z轴的极角 phi")
    axes[2].set(title="目标相对基站的角度", xlabel="时间（秒）", ylabel="角度（度）")
    axes[2].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("第1步示例数据，固定高度、匀速转弯的真实轨迹", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "truth.png", dpi=160)
    plt.close(fig)

    print(f"已生成 {len(time_s)} 个时刻，覆盖 {time_s[-1]:.2f} 秒。")
    print(f"初始真实位置（米） {position[0]}")
    print(f"初始距离 {distance[0]:.3f} 米，方位角 {np.rad2deg(azimuth[0]):.3f} 度。")
    print(f"初始极角 {np.rad2deg(polar[0]):.3f} 度，该角度从竖直 +z 轴量起。")
    print(f"输出文件夹 {OUTPUT_DIR}")
    print("打开 truth.png 看图，打开 truth.csv 看数字，truth.npz 留给下一步程序读取。")


if __name__ == "__main__":
    main()
