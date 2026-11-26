"""Draw the paired closed-loop results from the recorded frame decisions."""
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from .common import write_json, write_rows


METHODS = {"cv_rnet": "匀速外推 + R-Net", "nrnet": "N-Net + R-Net"}
COLORS = {"cv_rnet": "#27698B", "nrnet": "#8853A1"}
VEHICLE, UAV, MISS, RECOVERY = "#A65B28", "#228177", "#28313A", "#138253"


def _rate_text(value):
    return "无丢失事件" if value is None else f"{100 * value:.1f}%"


def _frame_samples(scene, frame_rows):
    lookup = {(r["method"], int(r["frame"])): r for r in frame_rows if int(r["scene"]) == 0}
    samples = []
    for method in METHODS:
        for frame, time in enumerate(scene.time):
            row = lookup.get((method, frame))
            hit = bool(int(row["hit"])) if row else None
            recovered = bool(hit and row["mode"] == "R")
            sample = {"scene": 0, "method": method, "frame": frame, "time_s": float(time),
                      "mode": row["mode"] if row else "", "hit": hit, "recovered": recovered}
            for k, axis in enumerate("xyz"):
                sample[f"vehicle_{axis}"] = float(scene.vehicle[frame, k])
                sample[f"platform_{axis}"] = float(scene.platform[frame, k])
                sample[f"estimate_{axis}"] = float(row[f"estimate_{axis}"]) if hit else None
            samples.append(sample)
    return samples


def _style_axes(axis):
    axis.set_axisbelow(True)
    axis.grid(color="#DFE5E9", lw=.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#9DAAB1")
    axis.tick_params(color="#9DAAB1", labelsize=9)


def _first_scene_plot(folder, scene, samples):
    fig = plt.figure(figsize=(11.5, 7.8), layout="constrained")
    layout = fig.add_gridspec(4, 2, height_ratios=(5.2, .32, 1, .7))
    first = fig.add_subplot(layout[0, 0])
    second = fig.add_subplot(layout[0, 1], sharex=first, sharey=first)
    timeline = fig.add_subplot(layout[2, :])
    legend = fig.add_subplot(layout[1, :])
    note = fig.add_subplot(layout[3, :])
    legend.axis("off")
    note.axis("off")
    fig.suptitle("机载雷达跟踪 80 km/h 地面车辆  ·  第一个测试场景", fontsize=16, fontweight="bold")
    stats = {}
    dt = float(np.median(np.diff(scene.time)))
    for y, (method, axis) in enumerate(zip(METHODS, (first, second))):
        rows = [r for r in samples if r["method"] == method and r["hit"] is not None]
        hits = [r for r in rows if r["hit"]]
        misses = [r for r in rows if not r["hit"]]
        recoveries = [r for r in hits if r["recovered"]]
        lost_events = sum(1 for j, r in enumerate(rows)
                          if not r["hit"] and (j == 0 or rows[j - 1]["hit"]))
        rate = len(recoveries) / lost_events if lost_events else None
        stats[method] = {"attempted_frames": len(rows), "attempted_duration_s": len(rows) * dt,
                         "missed_frames": len(misses), "loss_episodes": lost_events,
                         "recoveries": len(recoveries), "recovery_rate": rate}
        axis.plot(*scene.platform[:, :2].T, color=UAV, lw=2)
        axis.plot(*scene.vehicle[:, :2].T, color=VEHICLE, ls="--", lw=1.5)
        for path, color in ((scene.platform, UAV), (scene.vehicle, VEHICLE)):
            axis.scatter(*path[0, :2], color=color, marker="s", s=22, zorder=6)
        axis.scatter([r["estimate_x"] for r in hits], [r["estimate_y"] for r in hits],
                     s=8, color=COLORS[method], alpha=.75, linewidths=0, zorder=3)
        axis.scatter([r["vehicle_x"] for r in misses], [r["vehicle_y"] for r in misses],
                     s=23, color=MISS, marker="x", linewidths=1, zorder=4)
        axis.scatter([r["estimate_x"] for r in recoveries], [r["estimate_y"] for r in recoveries],
                     s=45, facecolors="none", edgecolors=RECOVERY, lw=1.25, zorder=5)
        axis.set(aspect="equal", xlabel="东向坐标（m）")
        axis.set_title(METHODS[method], loc="left", color=COLORS[method], fontweight="bold")
        _style_axes(axis)
        for group, marker, color in ((hits, "|", COLORS[method]), (misses, "x", MISS),
                                     (recoveries, "o", RECOVERY)):
            keyword = ({"edgecolors": color, "facecolors": "none"} if marker == "o"
                       else {"color": color})
            timeline.scatter([r["time_s"] for r in group], [y] * len(group),
                             s=42 if marker == "o" else 21, marker=marker, linewidths=1, **keyword)
        if rows and rows[-1]["frame"] < len(scene.time) - 1:
            start = rows[-1]["time_s"] + dt / 2
            timeline.fill_between([start, scene.time[-1] + dt / 2], y - .16, y + .16,
                                  color="#E8ECEF", lw=0, zorder=0)
        note.text(y * .5, .98,
                  f'{METHODS[method]}　实际跟踪 {len(rows)*dt:.2f} s　'
                  f'重捕 {len(recoveries)}/{lost_events}（{_rate_text(rate)}）',
                  transform=note.transAxes, color=COLORS[method], fontsize=9.5, va="top")
    first.set_ylabel("北向坐标（m）")
    handles = [Line2D([], [], color=UAV, lw=2, label="无人机地面投影"),
               Line2D([], [], color=VEHICLE, lw=1.5, ls="--", label="车辆真值"),
               Line2D([], [], marker="s", color=MISS, ms=4, ls="", label="起点"),
               Line2D([], [], marker=".", color="#4D6380", ms=9, ls="", label="检出帧估计"),
               Line2D([], [], marker="x", color=MISS, ls="", label="漏检帧真值"),
               Line2D([], [], marker="o", color=RECOVERY, mfc="none", ls="", label="重捕成功")]
    legend.legend(handles=handles, loc="center", ncol=6, frameon=False, fontsize=9)
    timeline.set(xlabel="场景时间（s）", yticks=[0, 1], yticklabels=list(METHODS.values()),
                 ylim=(1.43, -.43), xlim=(scene.time[0] - dt / 2, scene.time[-1] + dt / 2))
    _style_axes(timeline)
    note.text(0, .28, "时间轴上，竖线表示检出，叉号表示漏检，圆圈表示重捕，灰带表示终止后的时段。",
              va="center", fontsize=9, color="#4A5961")
    path = folder / "single_trajectory_tracking.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return stats


def _summary_plot(folder, summary, scene_count):
    data = {r["method"]: r for r in summary}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), width_ratios=(1.6, 1, 1))
    fig.subplots_adjust(top=.83, bottom=.28, left=.066, right=.98, wspace=.36)
    fig.suptitle(f"两种跟踪方法的配对结果  ·  {scene_count} 个测试场景", fontsize=16, fontweight="bold")
    fields = ("completion_rate", "detection_rate", "recovery_rate")
    axes[0].set(xticks=range(3), xticklabels=["完整跟踪", "检出", "重捕"], ylabel="比例（%）",
                ylim=(0, 116))
    for i, method in enumerate(METHODS):
        row = data[method]
        for j, field in enumerate(fields):
            x = j + (i - .5) * .36
            value = row[field]
            if value is None or not np.isfinite(value):
                axes[0].text(x, 3, "—", ha="center", color=COLORS[method])
                continue
            pct = 100 * value
            axes[0].bar(x, pct, width=.33, color=COLORS[method])
            axes[0].text(x, pct + 1, f"{pct:.1f}", ha="center", fontsize=8, color=COLORS[method])
    for axis, field, title, unit in zip(axes[1:], ("mean_valid_s", "position_rmse_m"),
                                       ("平均有效跟踪时间", "检出帧位置 RMSE"), ("s", "m")):
        for i, method in enumerate(METHODS):
            value = data[method][field]
            if value is None or not np.isfinite(value):
                axis.text(i, .06, "—", ha="center", transform=axis.get_xaxis_transform())
                continue
            axis.bar(i, value, color=COLORS[method], width=.58)
            axis.annotate(f"{value:.2f}", (i, value), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=9)
        axis.set(xticks=[], ylabel=f"{title}（{unit}）", xlim=(-.65, 1.65))
        axis.set_ylim(0, max(1e-6, axis.get_ylim()[1] * 1.16))
    for axis in axes:
        _style_axes(axis)
        axis.xaxis.grid(False)
    handles = [Line2D([], [], lw=7, color=COLORS[m], label=METHODS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .105), ncol=2, frameon=False)
    fig.text(.07, .07, "检出率按全部计划跟踪帧计，终止后的帧计为未检出。重捕率按丢失事件计。", fontsize=9)
    fig.text(.07, .025, "位置 RMSE 仅统计检出帧。匀速外推版使用回波量测位置，N-Net 版使用更新量测后的网络位置头。", fontsize=9)
    fig.savefig(folder / "paired_tracking_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)


def plot_results(out, scenes, summary, frame_rows):
    """Plot scene 0 and aggregate metrics. All coordinates use east/north/up."""
    folder = Path(out) / "figures"
    folder.mkdir(exist_ok=True, parents=True)
    scene = scenes[0]
    samples = _frame_samples(scene, frame_rows)
    fonts = ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "PingFang SC")
    available = set(font_manager.get_font_names())
    with plt.rc_context({"font.family": "sans-serif", "axes.unicode_minus": False,
                         "font.sans-serif": [f for f in fonts if f in available] + ["DejaVu Sans"]}):
        stats = _first_scene_plot(folder, scene, samples)
        _summary_plot(folder, summary, len(scenes))
    csv_path = folder / "single_trajectory_tracking_samples.csv"
    write_rows(csv_path, samples)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (csv_path, folder / "single_trajectory_tracking.png",
                        folder / "paired_tracking_comparison.png", Path(__file__))}
    write_json(folder / "single_trajectory_tracking.json", {
        "example_scene": 0, "selection": "first input test scene, independent of its outcomes",
        "aggregate_scene_count": len(scenes), "coordinates": "metres ENU, plotted XY",
        "marker_sources": {"UAV and vehicle": "simulated trajectory for display only",
                           "cv_rnet estimate": "received radar measurement mapped to ENU",
                           "nrnet estimate": "N-Net position head with newly received measurement mapped to ENU",
                           "miss": "simulated vehicle position at attempted missed frame",
                           "recovery": "an R-Net frame with a detection"},
        "unscored_frames": "no frame row for the method, plotted only on truth paths",
        "displayed_scene": stats, "sha256": hashes
    })
