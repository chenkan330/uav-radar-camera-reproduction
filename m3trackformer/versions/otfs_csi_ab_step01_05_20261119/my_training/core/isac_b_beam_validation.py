"""B 组第 3 步，离线核对真实 CSI 的波束增益和共享选束。"""
import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np

from .common import BASE, finish_run, new_run, read_json, write_json, write_rows
from .isac_b_csi_samples import VERSION as SAMPLE_VERSION
from .isac_beam_selection import downlink_channel_column
from .isac_multi_target import BUDGET, RADAR_CONFIG, STATION
from .versions import save_version, sha256


VERSION = "isac_b_step03_beam_validation"
FIGURE = "csi_beam_gain_validation.png"
CONTROLLED_DIRECTIONS_AND_RANGES = {
    "separated": [(-.538, .253, 94.), (.405, .176, 200.), (.073, -.496, 318.)],
    "overlap": [(-.293, .194, 80.), (-.293, .194, 325.), (.332, -.407, 181.)],
}


def select_downlink_beams(channel_columns, codebook, budget):
    """逐终端归一化全码本功率并按总分降序选束，平分时编号较小的束优先。"""
    gains = np.abs(channel_columns.conj() @ codebook)**2
    totals = gains.sum(axis=-1, keepdims=True)
    if not 1 <= budget <= gains.shape[-1] or np.any(~np.isfinite(totals) | (totals <= 0)):
        raise ValueError("波束预算或信道总功率无效")
    scores = gains/totals
    selected = np.argsort(-scores.sum(axis=0), kind="stable")[:budget]
    return gains, scores, selected


def nearest_geometric_beams(direction, grid):
    """空间频率周期为 2，保留每个轴上等距离的候选中心。"""
    distances = np.abs((grid[:, None]-direction+1.) % 2.-1.)
    candidates = np.isclose(distances, distances.min(axis=0), atol=1e-12, rtol=0)
    bx = np.flatnonzero(candidates[:, 0])
    by = np.flatnonzero(candidates[:, 1])
    return (by[:, None]*len(grid)+bx).ravel()


def geometric_gain(relative_position, grid, reference_range_m):
    """按等比级数的模平方计算 DFT 阵列增益，独立于复 CSI 和码本矩阵。"""
    distance = np.linalg.norm(relative_position)
    direction = relative_position[:2]/distance
    delta = (grid[:, None]-direction+1.) % 2.-1.
    axis_power = (np.sinc(len(grid)*delta/2.)/np.sinc(delta/2.))**2
    return (reference_range_m/distance)**2 * np.outer(axis_power[:, 1], axis_power[:, 0]).ravel()


def verify_single_channels(channels, frame_ids, times, positions, codebook, grid):
    records = []
    power = np.abs(channels.conj() @ codebook)**2
    for i, (gain, position) in enumerate(zip(power, positions)):
        u, v = position[:2]/np.linalg.norm(position)
        candidates = nearest_geometric_beams([u, v], grid)
        best = int(np.argmax(gain))
        reference = geometric_gain(position, grid, RADAR_CONFIG.reference_range_m)
        np.testing.assert_allclose(gain, reference, rtol=1e-9, atol=1e-12)
        if best not in candidates:
            raise ValueError(f"样本 {i} 的 CSI 最佳束与几何方向不一致")
        records.append({
            "sample_index": i, "label_frame": int(frame_ids[i]), "label_time_s": float(times[i]),
            "direction_u": float(u), "direction_v": float(v), "csi_best_beam_0based": best,
            "geometric_best_beams_0based": ",".join(map(str, candidates)),
            "max_gain": float(gain[best]), "max_gain_error": float(np.max(abs(gain-reference))),
        })
    return records, power


def verify_joint_case(direction_and_range, indices, grid, codebook):
    uv = np.array([entry[:2] for entry in direction_and_range])
    distance = np.array([entry[2] for entry in direction_and_range])
    positions = distance[:, None]*np.column_stack((uv, np.sqrt(1.-(uv**2).sum(axis=1))))
    channels = np.stack([downlink_channel_column(p, RADAR_CONFIG, indices) for p in positions])
    gain, normalized, selected = select_downlink_beams(channels, codebook, BUDGET)

    reference = np.stack([geometric_gain(p, grid, RADAR_CONFIG.reference_range_m) for p in positions])
    reference /= reference.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(normalized, reference, rtol=1e-9, atol=1e-12)
    total = normalized.sum(axis=0)
    reference_total = reference.sum(axis=0)
    expected = sorted(range(len(reference_total)), key=lambda b: (-reference_total[b], b))[:BUDGET]
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_allclose(normalized.sum(axis=1), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(total.sum(), len(positions), atol=1e-12, rtol=0)
    per_target_best = gain.argmax(axis=1)
    for direction, best in zip(uv, per_target_best):
        if best not in nearest_geometric_beams(direction, grid):
            raise ValueError("三终端样例中的单目标最佳束与几何方向不一致")

    return {
        "direction_cosines_uv": uv.tolist(), "ranges_m": distance.tolist(),
        "target_best_beams_0based": per_target_best.tolist(), "selected_beams_0based": selected.tolist(),
        "selected_scores": total[selected].tolist(), "selected_beam_count": len(np.unique(selected)),
        "per_target_normalized_gain_sums": normalized.sum(axis=1).tolist(),
        "max_analytic_score_error": float(np.max(abs(total-reference_total))),
    }, gain, normalized


def draw_validation(output, grid, first_power, first_record, joint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def relative_db(power):
        return 10*np.log10(np.maximum(power/power.max(), 1e-12))

    extent = [-1., 1., -1., 1.]
    joint_result, _, normalized = joint
    panels = (
        (first_power, [first_record["csi_best_beam_0based"]],
         [[first_record["direction_u"], first_record["direction_v"]]], "单终端 · 下一帧真实 CSI 增益"),
        (normalized.sum(axis=0), joint_result["selected_beams_0based"],
         joint_result["direction_cosines_uv"], "三终端 · 归一化后求和"),
    )
    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.7), layout="constrained")
        for ax, (power, selected, targets, title) in zip(axes, panels):
            mesh = ax.imshow(relative_db(power).reshape(len(grid), len(grid)), origin="lower",
                             extent=extent, vmin=-35, vmax=0, cmap="viridis", interpolation="nearest")
            by, bx = np.divmod(selected, len(grid))
            ax.scatter(grid[bx], grid[by], facecolors="none", edgecolors="white", s=45,
                       linewidths=1.4, label="入选波束")
            u, v = np.array(targets).T
            ax.scatter(u, v, c="#ff8b48", marker="x", s=55, linewidths=1.6, label="终端方向")
            ax.set(title=title, xlabel="方向余弦 u", ylabel="方向余弦 v", xlim=extent[:2], ylim=extent[2:])
            ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(mesh, ax=axes, shrink=.84, label="相对本图峰值（dB）")
        fig.savefig(output/FIGURE, dpi=175, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-run", type=Path, help="第 2 步运行目录，默认使用最近一次完整运行")
    args = parser.parse_args()
    sample_root = BASE/"outputs"/SAMPLE_VERSION
    sample_run = args.samples_run or sample_root/(sample_root/"latest_run.txt").read_text(encoding="utf-8")
    sample_meta = read_json(sample_run/"summary.json")
    protocol_file = sample_run/"step01_protocol.json"
    protocol = read_json(protocol_file)
    if sha256(protocol_file) != sample_meta["sample_protocol_sha256"]:
        raise ValueError("第 2 步归档的实验协议校验失败")
    if protocol["timing"]["beams_per_frame"] != BUDGET:
        raise ValueError("第 2 步协议与当前波束预算不一致")

    labels_file = sample_run/"csi_labels.npz"
    with np.load(labels_file, allow_pickle=False) as labels:
        channels, frame_ids, times = (labels[name] for name in ("h", "label_frame_indices", "label_time_s"))
    with np.load(sample_meta["source"], allow_pickle=False) as data:
        matches = np.flatnonzero(data["trajectory_id"] == sample_meta["trajectory_id"])
        if len(matches) != 1:
            raise ValueError("第 2 步源轨迹的编号缺失或不唯一")
        positions = data["position"][matches[0], frame_ids] - STATION
    np.testing.assert_allclose(times, frame_ids*protocol["timing"]["frame_period_s"], atol=1e-12, rtol=0)

    height, width = protocol["channel_label"]["antenna_array_shape_yx"]
    if height != width or channels.shape != (len(frame_ids), height*width) or not len(frame_ids):
        raise ValueError("CSI 标签与方形 DFT 码本尺寸不一致")
    indices = np.arange(width)
    grid = -1. + 2.*(indices+.5)/width
    weights = np.exp(1j*np.pi*indices[:, None]*grid)/np.sqrt(width)
    codebook = np.kron(weights, weights)
    single, power = verify_single_channels(channels, frame_ids, times, positions, codebook, grid)
    joint = {name: verify_joint_case(geometry, indices, grid, codebook)
             for name, geometry in CONTROLLED_DIRECTIONS_AND_RANGES.items()}

    output = new_run(BASE/"outputs"/VERSION)
    copyfile(protocol_file, output/"step01_protocol.json")
    write_rows(output/"single_target_beams.csv", single)
    for name, (record, gain, normalized) in joint.items():
        ranks = {beam: rank for rank, beam in enumerate(record["selected_beams_0based"], start=1)}
        write_rows(output/f"joint_scores_{name}.csv", [
            {"beam_id_0based": b, "beam_x": b % width, "beam_y": b//width,
             **{f"target_{i+1}_gain": float(gain[i,b]) for i in range(len(gain))},
             **{f"target_{i+1}_normalized": float(normalized[i,b]) for i in range(len(normalized))},
             "total_score": float(normalized[:, b].sum()), "selected_rank": ranks.get(b)}
            for b in range(codebook.shape[1])
        ])
    draw_validation(output, grid, power[0], single[0], joint["separated"])
    write_json(output/"summary.json", {
        "version": VERSION, "scope": "固定阵列、单径、使用仿真下行真值的离线公式验收",
        "samples_run": str(sample_run.resolve()), "source": str(labels_file.resolve()),
        "reference_trajectory": sample_meta["source"], "trajectory_id": sample_meta["trajectory_id"],
        "codebook_size": codebook.shape[1], "scan_budget": BUDGET,
        "single_target_count": len(single), "single_target_geometry_matches": len(single),
        "first_sample": single[0],
        "joint_cases": {name: item[0] for name, item in joint.items()},
        "beam_numbering": "0-based, x changes fastest",
    })
    archive = save_version(f"{VERSION}_{output.name[4:]}", output, "B 组第 3 步离线 CSI 增益选束验收。",
                           asset_keys=("source", "reference_trajectory"))
    finish_run(output)
    print(f"单终端最优束 {len(single)}/{len(single)} 一致，"
          f"三终端验收 {len(joint)} 组通过。\n结果目录：{output.resolve()}\n归档目录：{archive.resolve()}", flush=True)
