"""第3步，构造平面阵列DFT码本，并用真实方向生成最优波束标签。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 阵列位于xy平面，沿x、y方向各有32个天线，天线间距为半波长。
# 码本包含32乘32共1024组波束权重。编号从1开始，0留给后续缺测填充。
NX = 32
NY = 32
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "step02" / "measurements.npz"
OUTPUT_DIR = BASE_DIR / "outputs" / "step03"


def calculate_beam_gains(u, v, weights_x, weights_y):
    """计算每个方向对应的全部波束增益，输出形状为[时刻数, NY*NX]。"""
    # u、v是两个方向余弦。半波长间距下，相邻天线的相位差分别为pi*u和pi*v。
    nx = weights_x.shape[0]
    ny = weights_y.shape[0]
    steering_x = np.exp(1j * np.pi * u[:, None] * np.arange(nx)) / np.sqrt(nx)
    steering_y = np.exp(1j * np.pi * v[:, None] * np.arange(ny)) / np.sqrt(ny)
    gain_x = np.abs(steering_x @ weights_x.conj())**2
    gain_y = np.abs(steering_y @ weights_y.conj())**2
    # 平面阵列可分解成x、y两个方向，整体功率增益是两项的乘积。
    gains = gain_y[:, :, None] * gain_x[:, None, :]
    return gains.reshape(len(u), ny * nx)


def main():
    # 一、读取第2步数据，同时保留真值和带误差的测量值。
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到 {INPUT_FILE}，请先运行 step02_generate_measurements.py。")
    with np.load(INPUT_FILE) as stored:
        data = dict(stored)
    time_s = data["time_s"]
    count = len(time_s)

    # 二、建立DFT网格。这里均匀划分的是方向余弦，而非直接均匀划分角度。
    grid_u = -1.0 + 2.0 * np.arange(NX) / NX
    grid_v = -1.0 + 2.0 * np.arange(NY) / NY
    weights_x = np.exp(1j * np.pi * np.outer(np.arange(NX), grid_u)) / np.sqrt(NX)
    weights_y = np.exp(1j * np.pi * np.outer(np.arange(NY), grid_v)) / np.sqrt(NY)
    code_u, code_v = np.meshgrid(grid_u, grid_v)
    code_u, code_v = code_u.ravel(), code_v.ravel()
    # 使用一维下标展开，x方向编号先变化，波束编号等于 y下标*NX+x下标+1。
    beam_ids = np.arange(1, NX * NY + 1)
    visible = code_u**2 + code_v**2 <= 1.0 + 1e-12

    # 三、根据真实角度计算方向余弦，再比较所有波束的阵列增益。
    true_u = np.sin(data["true_polar_rad"]) * np.cos(data["true_azimuth_rad"])
    true_v = np.sin(data["true_polar_rad"]) * np.sin(data["true_azimuth_rad"])
    true_gains = calculate_beam_gains(true_u, true_v, weights_x, weights_y)
    true_index = np.argmax(true_gains, axis=1)
    best_beam = true_index + 1
    best_gain = true_gains[np.arange(count), true_index]

    # 四、另外用带噪测量方向直接选一次波束，作为对照。
    # best_beam才是监督标签。measurement_based_beam用于观察测量误差的影响。
    measured_u = np.sin(data["measured_polar_rad"]) * np.cos(data["measured_azimuth_rad"])
    measured_v = np.sin(data["measured_polar_rad"]) * np.sin(data["measured_azimuth_rad"])
    measured_gains = calculate_beam_gains(measured_u, measured_v, weights_x, weights_y)
    measured_index = np.argmax(measured_gains, axis=1)
    measurement_based_beam = measured_index + 1
    measured_choice_true_gain = true_gains[np.arange(count), measured_index]
    agrees = best_beam == measurement_based_beam
    best_gain_db = 10 * np.log10(np.maximum(best_gain, 1e-12))
    measured_choice_true_gain_db = 10 * np.log10(np.maximum(measured_choice_true_gain, 1e-12))

    # 五、保存标签和码本。归一化单程功率增益的最大可能值为1。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_table = np.column_stack((
        time_s, np.rad2deg(data["true_azimuth_rad"]), np.rad2deg(data["true_polar_rad"]),
        true_u, true_v, best_beam, true_index % NX, true_index // NX,
        best_gain, measurement_based_beam, measured_choice_true_gain, agrees.astype(int),
    ))
    np.savetxt(
        OUTPUT_DIR / "beam_labels.csv", label_table, delimiter=",",
        fmt=["%.6f"] * 5 + ["%d", "%d", "%d", "%.8f", "%d", "%.8f", "%d"],
        header="time_s,true_azimuth_deg,true_polar_deg,true_u,true_v,best_beam,x_grid_index,y_grid_index,best_gain,measurement_based_beam,measurement_choice_true_gain,same_beam",
        comments="", encoding="utf-8-sig",
    )
    np.savetxt(
        OUTPUT_DIR / "codebook.csv", np.column_stack((beam_ids, code_u, code_v, visible)),
        delimiter=",", fmt=["%d", "%.6f", "%.6f", "%d"],
        header="beam_id,u_grid,v_grid,visible_direction", comments="", encoding="utf-8-sig",
    )
    # 继续带上前两步的数据，后续制作训练窗口时可以从这个文件统一读取。
    np.savez_compressed(
        OUTPUT_DIR / "beam_labels.npz", **data,
        best_beam=best_beam, measurement_based_beam=measurement_based_beam,
        best_gain=best_gain, measurement_choice_true_gain=measured_choice_true_gain,
        true_u=true_u, true_v=true_v, codebook_u=code_u, codebook_v=code_v,
        weights_x=weights_x, weights_y=weights_y, array_nx=NX, array_ny=NY,
        antenna_spacing_wavelengths=0.5, reserved_beam_label=0,
        beam_label_definition="argmax of normalized one-way array gain at the true direction",
    )

    # 六、画图，展示方向、波束增益，以及每个时刻的分类标签。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes[0, 0].scatter(code_u, code_v, s=5, color="lightgray", label="DFT网格")
    circle = np.linspace(0, 2 * np.pi, 300)
    axes[0, 0].plot(np.cos(circle), np.sin(circle), "--", color="gray", linewidth=0.8, label="可见方向边界")
    axes[0, 0].plot(true_u, true_v, color="#197453", label="目标真实方向")
    axes[0, 0].scatter(code_u[true_index], code_v[true_index], s=15, color="#E1842C", label="所选波束网格点")
    axes[0, 0].set(title="方向余弦与DFT网格", xlabel="u", ylabel="v", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend(fontsize=8, loc="lower left")

    first_gain_db = 10 * np.log10(np.maximum(true_gains[0].reshape(NY, NX), 1e-12))
    heatmap = axes[0, 1].pcolormesh(grid_u, grid_v, first_gain_db, shading="nearest", vmin=-40, vmax=0, cmap="viridis")
    axes[0, 1].scatter(true_u[0], true_v[0], s=45, facecolors="none", edgecolors="white", label="真实方向")
    axes[0, 1].scatter(code_u[true_index[0]], code_v[true_index[0]], s=55, color="#F26A3C", marker="*", label="最大增益波束")
    axes[0, 1].set(title=f"第0时刻的波束增益，最优编号为{best_beam[0]}", xlabel="u", ylabel="v")
    axes[0, 1].legend(fontsize=8, loc="lower left")
    fig.colorbar(heatmap, ax=axes[0, 1], label="归一化单程功率增益（dB）")

    axes[1, 0].step(time_s, best_beam, where="post", label="真实方向生成的标签")
    axes[1, 0].scatter(time_s, measurement_based_beam, s=8, alpha=0.5, label="带噪方向直接选出的波束")
    axes[1, 0].set(title="每时刻对应的波束编号", xlabel="时间（秒）", ylabel="波束编号")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(time_s, best_gain_db, color="#197453", label="选择最优标签时的增益")
    axes[1, 1].plot(time_s, measured_choice_true_gain_db, color="#E1842C", alpha=0.8, label="按带噪方向选波束后的实际增益")
    axes[1, 1].set(title="不同选择在真实方向上的增益", xlabel="时间（秒）", ylabel="归一化单程功率增益（dB）")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    fig.suptitle("第3步示例数据，为每个时刻生成最优波束监督标签", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beam_labels.png", dpi=160)
    plt.close(fig)

    print(f"已建立 {NX}乘{NY} 平面阵列的 {NX * NY} 个DFT波束权重。")
    print(f"已生成 {count} 个最优波束标签，共用到 {len(np.unique(best_beam))} 个不同编号。")
    print(f"第0时刻的最优波束编号 {best_beam[0]}，归一化单程增益 {best_gain[0]:.4f}。")
    print(f"观测方向直接选波束与真实标签的一致率 {100 * agrees.mean():.2f}%。")
    print("这项一致率用于对照测量误差的影响，标签由真实方向生成。")
    print(f"输出文件夹 {OUTPUT_DIR}")
    print("打开 beam_labels.csv 查看标签，打开 beam_labels.png 查看方向和波束增益。")


if __name__ == "__main__":
    main()
