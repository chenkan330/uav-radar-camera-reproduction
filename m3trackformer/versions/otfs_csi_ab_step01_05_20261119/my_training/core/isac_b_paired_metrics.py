"""第 5 步离线评价。按各组实际扫描时隙计算单程视距下行增益。"""
import csv
from pathlib import Path

import numpy as np

from .common import write_json, write_rows
from .isac_multi_target import BUDGET


GAIN_COLUMNS = (
    "scene_id", "frame", "frame_start_s", "target_index", "trajectory_id",
    "oracle_gain", "oracle_beam_1based", "oracle_slot_0based", "oracle_tx_time_s",
    "a_best_scanned_gain", "a_best_beam_1based", "a_best_slot_0based", "a_best_tx_time_s", "a_loss_db",
    "b_best_scanned_gain", "b_best_beam_1based", "b_best_slot_0based", "b_best_tx_time_s", "b_loss_db",
    "b_minus_a_gain_db",
)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _read_scans(directory, group_count, frame_count, codebook_size):
    records = {}
    for row in _read_csv(directory/"scans.csv"):
        scene, frame = int(row["scene"]), int(row["frame"])
        beams = tuple(int(value)-1 for value in row["selected_beams_1based"].split(","))
        if not 0 <= scene < group_count or not 0 <= frame < frame_count:
            raise ValueError(f"扫描记录超出场景范围 {directory} {(scene, frame)}")
        if len(beams) != BUDGET or len(set(beams)) != BUDGET or not all(0 <= b < codebook_size for b in beams):
            raise ValueError(f"扫描记录不符合波束预算 {directory} {(scene, frame)}")
        if (scene, frame) in records:
            raise ValueError(f"扫描记录重复 {directory} {(scene, frame)}")
        # CSV 的顺序是发射次序，直接转为槽位编号。
        records[scene, frame] = beams
    return records


def _axis_power(grid_coordinate, direction, side):
    delta = (grid_coordinate-direction+1.) % 2.-1.
    return (np.sinc(side*delta/2.)/np.sinc(delta/2.))**2


def _nearest_grid_index(direction, side):
    # DFT 网格中心位于每格中点。等距时两候选具有相同增益。
    return np.floor((direction+1.)*(side/2.)).astype(np.intp) % side


def _gain_arrays(radar, keys, a_scans, b_scans, grid):
    """返回 [配对帧, 目标] 上三种最佳增益，以及达到该增益的束与槽位。"""
    group, frame = np.asarray(keys, dtype=np.intp).T
    delay = np.arange(BUDGET, dtype=np.float64)*radar.burst_s
    positions = radar.position[group, :, frame, None, :] + radar.velocity[group, :, frame, None, :]*delay[None, None, :, None]
    distance = np.linalg.norm(positions, axis=-1)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("评估时隙出现无效的阵列相对距离")
    u, v = positions[..., 0]/distance, positions[..., 1]/distance
    scale = (radar.config.reference_range_m/distance)**2
    side = len(grid)
    results = {}

    def best(gain, beams):
        slots = gain.argmax(axis=-1)
        gain = np.take_along_axis(gain, slots[..., None], axis=-1)[..., 0]
        beam = np.take_along_axis(np.broadcast_to(beams, (len(keys), gain.shape[1], BUDGET)),
                                  slots[..., None], axis=-1)[..., 0]
        return gain, beam, slots

    for label, scans in (("a", a_scans), ("b", b_scans)):
        beams = np.asarray([scans[key] for key in keys], dtype=np.intp)[:, None, :]
        by, bx = np.divmod(beams, side)
        power = scale*_axis_power(grid[bx], u, side)*_axis_power(grid[by], v, side)
        results[label] = best(power, beams)

    # DFT 全码本的轴向增益分子在格点间仅变号。最大模平方位于最近格点，
    # 因此逐槽找最近 x/y 格点后再对槽取最大，等价于遍历所有槽位和所有波束。
    bx, by = _nearest_grid_index(u, side), _nearest_grid_index(v, side)
    power = scale*_axis_power(grid[bx], u, side)*_axis_power(grid[by], v, side)
    results["oracle"] = best(power, by*side+bx)
    return results


def _paired_gain_rows(radar, keys, a_scans, b_scans, grid, trajectory_ids):
    dt = radar.config.decision_interval_s
    rows = []
    losses = {"a": [], "b": []}
    for start in range(0, len(keys), 512):
        batch = keys[start:start+512]
        evaluated = _gain_arrays(radar, batch, a_scans, b_scans, grid)
        oracle_gain, oracle_beam, oracle_slot = evaluated["oracle"]
        arm_losses = {}
        for label in ("a", "b"):
            gain = evaluated[label][0]
            if np.any(gain > oracle_gain*(1.+1e-12)):
                raise ValueError("已扫波束增益超过全槽位全码本参考")
            arm_losses[label] = np.maximum(10*np.log10(oracle_gain/np.maximum(gain, np.finfo(float).tiny)), 0.)
            losses[label].append(arm_losses[label].ravel())
        for i, (scene, frame) in enumerate(batch):
            frame_start = frame*dt
            for target, trajectory_id in enumerate(trajectory_ids[scene]):
                slot = int(oracle_slot[i, target])
                row = {
                    "scene_id": scene, "frame": frame, "frame_start_s": frame_start,
                    "target_index": target, "trajectory_id": int(trajectory_id),
                    "oracle_gain": float(oracle_gain[i, target]),
                    "oracle_beam_1based": int(oracle_beam[i, target])+1,
                    "oracle_slot_0based": slot, "oracle_tx_time_s": frame_start+slot*radar.burst_s,
                }
                for label in ("a", "b"):
                    gain, beam, best_slot = evaluated[label]
                    slot = int(best_slot[i, target])
                    row.update({
                        f"{label}_best_scanned_gain": float(gain[i, target]),
                        f"{label}_best_beam_1based": int(beam[i, target])+1,
                        f"{label}_best_slot_0based": slot,
                        f"{label}_best_tx_time_s": frame_start+slot*radar.burst_s,
                        f"{label}_loss_db": float(arm_losses[label][i, target]),
                    })
                row["b_minus_a_gain_db"] = row["a_loss_db"]-row["b_loss_db"]
                rows.append(row)
    return rows, {label: np.concatenate(chunks) if chunks else np.empty(0) for label, chunks in losses.items()}


def _scene_rows(a_dir, b_dir, scans, common_keys, trajectory_ids, first_frame, frame_count):
    scene_rows = {}
    for label, directory in (("a", a_dir), ("b", b_dir)):
        source_rows = _read_csv(directory/"identity_scenes.csv")
        scene_rows[label] = {int(row["scene_id"]): row for row in source_rows}
        if len(source_rows) != len(trajectory_ids) or set(scene_rows[label]) != set(range(len(trajectory_ids))):
            raise ValueError(f"身份评价未覆盖全部场景 {directory}")
        target_rows = _read_csv(directory/"identity_targets.csv")
        recorded = {(int(row["scene_id"]), int(row["target_index"])): int(row["trajectory_id"]) for row in target_rows}
        expected = {(scene, i): int(t) for scene, targets in enumerate(trajectory_ids) for i, t in enumerate(targets)}
        if len(target_rows) != len(expected) or recorded != expected:
            raise ValueError(f"身份评价的场景与轨迹编号不一致 {directory}")

    def counts(keys):
        return np.bincount([scene for scene, frame in keys if frame >= first_frame], minlength=len(trajectory_ids))

    a_count, b_count, paired = counts(scans["a"]), counts(scans["b"]), counts(common_keys)
    rows = []
    for scene, targets in enumerate(trajectory_ids):
        row = {"scene_id": scene, "trajectory_ids": ",".join(str(t) for t in targets),
               "configured_frames_from_b_start": frame_count-first_frame,
               "a_scanned_frames_from_b_start": int(a_count[scene]),
               "b_scanned_frames_from_b_start": int(b_count[scene]), "paired_frames": int(paired[scene])}
        for label in ("a", "b"):
            source = scene_rows[label][scene]
            row.update({
                f"{label}_identity_success": source["success"].lower() == "true",
                f"{label}_identity_duration_s": float(source["duration_s"]),
                f"{label}_identity_evaluated_frames": int(source["evaluated_blocks"]),
                f"{label}_identity_recoveries": int(source["recoveries"]),
                f"{label}_wrong_updates": int(source["wrong_updates"]),
            })
        row["b_minus_a_identity_duration_s"] = row["b_identity_duration_s"]-row["a_identity_duration_s"]
        rows.append(row)
    return rows


def _plot_comparison(output, tracking, losses, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("a", "b")
    labels = ("A 原 V6", "B 预测 CSI")
    colors = ("#345479", "#168477")
    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplot_mosaic([["success", "duration"], ["gain", "gain"]],
                                       figsize=(9.1, 6.1), layout="constrained")
        setup = (
            ("success", "identity_success_count", "三目标均完成的场景", "场景数",
             coverage["all_scenes"], "%.0f"),
            ("duration", "identity_mean_duration_s", "平均同时跟踪时长", "秒",
             coverage["maximum_identity_duration_s"], "%.2f"),
        )
        for key, measure, title, units, ceiling, decimals in setup:
            axis = axes[key]
            bars = axis.bar(labels, [tracking[name][measure] for name in methods], color=colors, width=.51)
            axis.set(title=title, ylabel=units, ylim=(0., max(ceiling, 1.)*1.22))
            axis.bar_label(bars, fmt=decimals, padding=3)
        axis = axes["gain"]
        if coverage["paired_target_frames"]:
            for name, title, color in zip(methods, labels, colors):
                values = np.sort(losses[name])
                axis.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=title, color=color)
            axis.legend(loc="lower right")
        else:
            axis.text(.5, .5, "无共同扫描帧", transform=axis.transAxes, ha="center")
        axis.set(title=f"共同扫描的 {coverage['paired_target_frames']:,} 个目标帧",
                 xlabel="相对全槽位全码本最优的增益损失（dB）", ylabel="累积比例", ylim=(0., 1.04))
        for axis in axes.values():
            axis.grid(axis="y", alpha=.24)
            axis.set_axisbelow(True)
        fig.suptitle("固定阵列单径 OTFS 闭环配对结果")
        fig.savefig(output/"stage05_ab_comparison.png", dpi=170, facecolor="white")
        plt.close(fig)


def paired_experiment_metrics(output, scenes, radar, a_dir, b_dir, a_metrics, b_metrics, *, b_first_frame):
    """返回完整跟踪指标及从 B 启用帧起共同扫描帧的实际槽位增益。"""
    output, a_dir, b_dir = map(Path, (output, a_dir, b_dir))
    trajectory_ids = np.asarray(scenes["trajectory_id"])
    group_count, target_count, frame_count, _ = radar.position.shape
    if trajectory_ids.shape != (group_count, target_count) or not 0 <= b_first_frame < frame_count:
        raise ValueError("场景、轨迹编号或 B 启用帧不合法")
    side = len(radar.indices)
    grid = -1.+2.*(np.arange(side)+.5)/side
    scans = {label: _read_scans(directory, group_count, frame_count, side**2)
             for label, directory in (("a", a_dir), ("b", b_dir))}
    keys = sorted(key for key in scans["a"].keys() & scans["b"].keys() if key[1] >= b_first_frame)
    all_scenes = _scene_rows(a_dir, b_dir, scans, keys, trajectory_ids, b_first_frame, frame_count)
    gains, losses = _paired_gain_rows(radar, keys, scans["a"], scans["b"], grid, trajectory_ids)
    with (output/"paired_downlink_gain.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=GAIN_COLUMNS)
        writer.writeheader()
        writer.writerows(gains)
    write_rows(output/"paired_scenes.csv", all_scenes)

    count_a = sum(row["a_scanned_frames_from_b_start"] for row in all_scenes)
    count_b = sum(row["b_scanned_frames_from_b_start"] for row in all_scenes)
    paired = len(keys)
    counts = (count_a, count_b)
    # 原 V6 的初始化窗口来自冻结协议，长度为 5 帧。图的时长范围用真实评价时长即可。
    maximum_identity_duration_s = max(max(row[f"{label}_identity_duration_s"] for row in all_scenes)
                                      for label in ("a", "b"))
    coverage = {
        "all_scenes": group_count, "first_paired_eligible_frame_0based": int(b_first_frame),
        "maximum_identity_duration_s": maximum_identity_duration_s,
        "configured_scene_frames_from_b_start": group_count*(frame_count-b_first_frame),
        "a_scene_frames_from_b_start": count_a, "b_scene_frames_from_b_start": count_b,
        "paired_scene_frames": paired, "paired_target_frames": paired*target_count,
        "a_only_scene_frames": count_a-paired, "b_only_scene_frames": count_b-paired,
        "paired_fraction_of_each_arm": {
            label: paired/count if count else None for label, count in zip(("a", "b"), counts)
        },
        "paired_scene_count": sum(row["paired_frames"] > 0 for row in all_scenes),
    }
    write_json(output/"paired_coverage.json", coverage)

    tracking = {}
    for label, metrics in (("a", a_metrics), ("b", b_metrics)):
        identity, physical = metrics["identity"], metrics["physical"]
        success_count = identity["success_count"]
        if success_count != sum(row[f"{label}_identity_success"] for row in all_scenes):
            raise ValueError(f"{label} 组场景成功数与汇总不一致")
        tracking[label] = {
            "identity_success_count": success_count, "identity_success_rate": identity["success_rate"],
            "identity_mean_duration_s": identity["mean_all_targets_duration_s"],
            "identity_recoveries": identity["recoveries"], "identity_lost_events": identity["lost_events"],
            "identity_recovery_rate": identity["recovery_rate"], "physical_recoveries": physical["recoveries"],
            "physical_lost_events": physical["lost_events"], "physical_recovery_rate": physical["recovery_rate"],
            "position_rmse_m_on_detections": identity["position_rmse_m_on_detections"],
            "position_evaluation_count": identity["position_evaluation_count"],
            "position_evaluation_coverage": identity["position_evaluation_coverage"],
        }
    _plot_comparison(output, tracking, losses, coverage)
    mean_loss = {label: float(values.mean()) if paired else None for label, values in losses.items()}
    return {
        "scope": "固定阵列单径视距，真实信道仅作离线评分",
        "oracle_definition": "同一帧同一目标，12 个实际槽位起点及全 1024 束的最大理论增益",
        "scanned_gain_definition": "每组选束按本组发射次序在各槽位起点评估，取本组 12 个增益的最大值",
        "paired_scene_frames": paired, "paired_target_frames": paired*target_count,
        "coverage": coverage, "tracking": tracking, "mean_loss_db": mean_loss,
        "p95_loss_db": {label: float(np.percentile(values, 95)) if paired else None for label, values in losses.items()},
        "mean_b_minus_a_gain_db": mean_loss["a"]-mean_loss["b"] if paired else None,
    }
