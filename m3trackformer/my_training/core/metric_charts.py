"""Plot saved multi-target results without modifying their simulation or weights.

Run from my_training: python -m core.metric_charts --run OUTPUT_RUN --output FIGURES
Compare saved versions with repeated --version "V1=OUTPUT_RUN" arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

POLICIES = ("joint", "decoupled")
POLICY_LABELS = {"joint": "联合波束选择", "decoupled": "逐目标均分"}
COLORS = ("#2768a8", "#e58a35", "#299779", "#9362b6", "#ce5966", "#6c7d8b")
METRICS = (
    ("success_rate", "整组跟踪成功率", "%", 100.),
    ("mean_all_targets_duration_s", "全部目标同时连续跟踪时长", "s", 1.),
    ("position_rmse_m_on_detections", "检测帧位置 RMSE", "m", 1.),
    ("recovery_rate", "重捕获成功率", "%", 100.),
    ("mean_recovery_delay_s", "平均重捕获延迟", "ms", 1000.),
    ("missed_target_block_rate", "观测期目标帧漏检率", "%", 100.),
    ("recoveries", "重捕获次数", "次", 1.),
    ("lost_events", "丢失事件次数", "次", 1.),
    ("mean_unique_beams", "每场景每帧不同扫描方向数", "束", 1.),
)
RESOURCE_METRICS = (
    ("mean_beam_slots_used", "每场景每帧实际扫描开销", "束次", 1.),
    ("mean_prediction_batch_ms", "平均整批波束预测耗时", "ms / batch", 1.),
    ("amortized_prediction_ms_per_scene", "场景均摊波束预测耗时", "ms / scene", 1.),
)
FOOTNOTE = (
    "任一目标连续漏检达到阈值即终止整组；时长是整组同时跟踪时长。\n"
    "位置 RMSE 仅统计已检测帧；漏检率仅统计整组终止前已观测目标帧，各方法样本覆盖范围可能不同。"
)
LATENCY_NOTE = (
    "计时仅含波束预测与选择，不含位置回归及读写。整批大小随存活场景数变化；\n"
    "均摊耗时是批耗时除以活动场景数，不能视为独立场景的端到端推理延迟。"
)


def _rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_run_metrics(run_dir):
    """Return a run record and policy metrics; unavailable values remain None."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    targets = int(record["targets"])
    scans = _rows(run_dir / "beam_scans.csv")
    result = {}
    for policy, original in record["results"].items():
        values = dict(original)
        scenes = _rows(run_dir / f"{policy}_scenes.csv")
        targets_rows = _rows(run_dir / f"{policy}_targets.csv")
        if scenes:
            denominator = sum(int(row["evaluated_blocks"]) for row in scenes) * targets
            numerator = sum(int(row["missed_target_blocks"]) for row in scenes)
            values["missed_target_block_rate"] = numerator / denominator if denominator else None
            values["evaluated_target_blocks"] = denominator
            values["detected_target_blocks"] = denominator - numerator
        else:
            values["missed_target_block_rate"] = None
        own_scans = [row for row in scans if row["policy"] == policy]
        for column, key in (("unique_beams", "mean_unique_beams"),
                            ("beam_slots_used", "mean_beam_slots_used")):
            values[key] = float(np.mean([float(row[column]) for row in own_scans])) if own_scans else None
        values["position_evaluation_count"] = sum(int(row["position_count"]) for row in targets_rows) if targets_rows else None
        result[policy] = values
    return record, result


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "font.size": 10,
        "savefig.facecolor": "white",
    })
    return plt


def write_metrics_csv(path, versions):
    """versions maps visible labels to saved run directories; values use displayed units."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("version", "policy", "metric", "label", "value", "unit", "source_run"))
        writer.writeheader()
        for version, run_dir in versions.items():
            _, policies = load_run_metrics(run_dir)
            specs = METRICS + RESOURCE_METRICS + (
                ("evaluated_target_blocks", "已观测目标帧数", "帧", 1.),
                ("detected_target_blocks", "已检测目标帧数", "帧", 1.),
                ("position_evaluation_count", "位置 RMSE 评价帧数", "帧", 1.),
            )
            for policy, values in policies.items():
                for key, label, unit, scale in specs:
                    value = values.get(key)
                    writer.writerow({"version": version, "policy": policy, "metric": key,
                                     "label": label, "value": "" if value is None else value * scale,
                                     "unit": unit, "source_run": str(Path(run_dir).resolve())})
    return path


def _bar_grid(versions, specs, output, title, note, rows, columns, figsize):
    plt = _plotting()
    prepared = [(version, *load_run_metrics(run)) for version, run in versions.items()]
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    labels = [POLICY_LABELS.get(policy, policy) for policy in POLICIES]
    width = .76 / max(1, len(prepared))
    positions = np.arange(len(POLICIES))
    for ax, (key, name, unit, scale) in zip(axes.flat, specs):
        maximum = 0.
        for version_index, (version, record, metrics) in enumerate(prepared):
            x = positions - .38 + width * (version_index + .5)
            values = [metrics.get(policy, {}).get(key) for policy in POLICIES]
            displayed = [float(value) * scale if value is not None else np.nan for value in values]
            finite = [value for value in displayed if np.isfinite(value)]
            maximum = max([maximum] + finite)
            bars = ax.bar(x, displayed, width=width * .9, color=COLORS[version_index % len(COLORS)],
                          label=version, zorder=3)
            for bar, value in zip(bars, displayed):
                if np.isfinite(value):
                    text_value = f"{value:.0f}" if key in ("recoveries", "lost_events") else f"{value:.2f}"
                    ax.annotate(text_value, (bar.get_x()+bar.get_width()/2, value),
                                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
                else:
                    ax.text(bar.get_x()+bar.get_width()/2, .03, "N/A", transform=ax.get_xaxis_transform(),
                            ha="center", color="#6b7280", fontsize=9)
        ax.set_title(name, loc="left", pad=13)
        ax.set_ylabel(unit)
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, max(maximum * 1.24, 1.))
        if unit == "%" and key != "missed_target_block_rate":
            ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=.18, zorder=0)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    fig.suptitle(title, x=.055, y=.985, ha="left", fontsize=16, fontweight="bold")
    handles, legends = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper right", bbox_to_anchor=(.98, .99), ncol=min(3, len(prepared)), frameon=False)
    fig.text(.055, .017, note, fontsize=8.5, color="#4b5563", va="bottom")
    fig.tight_layout(rect=(.02, .085 if rows > 1 else .19, .99, .94 if rows > 1 else .87), h_pad=2.2, w_pad=2.)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def _survival(ax, rows, maximum, label, color, style="-"):
    failures, counts = np.unique([float(row["duration_s"]) for row in rows if str(row["success"]).lower() == "false"], return_counts=True)
    y = np.r_[100., 100 * (1-counts.cumsum()/len(rows))]
    ax.step(np.r_[0., failures, maximum], np.r_[y, y[-1]], where="post", label=label,
            color=color, linestyle=style, linewidth=2.)


def plot_distributions(run_dir, output, title="当前多目标结果：场景分布"):
    plt = _plotting()
    record, _ = load_run_metrics(run_dir)
    run_dir = Path(run_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    durations, rmses = [], []
    for policy in POLICIES:
        scenes = _rows(run_dir / f"{policy}_scenes.csv")
        target_rows = _rows(run_dir / f"{policy}_targets.csv")
        durations.append([float(row["duration_s"]) for row in scenes])
        rmses.append([(float(row["position_squared_error_sum"])/int(row["position_count"]))**.5
                      for row in target_rows if int(row["position_count"])])
        _survival(axes[2], scenes, record["maximum_evaluated_duration_s"], POLICY_LABELS[policy],
                  COLORS[POLICIES.index(policy)], "-" if policy == "joint" else "--")
    for ax, data, subtitle, ylabel in (
        (axes[0], durations, "各场景的同时跟踪时长", "s"),
        (axes[1], rmses, "各目标的检测帧位置 RMSE", "m"),
    ):
        for i, values in enumerate(data):
            if values:
                jitter = np.linspace(-.08, .08, len(values))
                ax.scatter(i+jitter, values, s=16, alpha=.6, color=COLORS[i], zorder=3)
        box = ax.boxplot(data, positions=np.arange(len(POLICIES)), widths=.4, patch_artist=True,
                         showfliers=False, medianprops={"color": "#1f2937", "linewidth": 1.8})
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color); patch.set_alpha(.2)
        ax.set_xticks(np.arange(len(POLICIES)), [POLICY_LABELS[p] for p in POLICIES])
        ax.set(title=subtitle, ylabel=ylabel)
        ax.grid(axis="y", alpha=.18)
    axes[2].set(title="全部目标尚未终止的场景比例", xlabel="闭环跟踪时间 / s", ylabel="%",
                xlim=(0, record["maximum_evaluated_duration_s"]), ylim=(0, 105))
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(alpha=.18)
    fig.suptitle(title, x=.055, y=.98, ha="left", fontsize=16, fontweight="bold")
    fig.text(.055, .02, FOOTNOTE, fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(.02, .12, .99, .9), w_pad=2.)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def plot_comparison_survival(versions, output):
    plt = _plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for version_index, (version, run_dir) in enumerate(versions.items()):
        record, _ = load_run_metrics(run_dir)
        for ax, policy in zip(axes, POLICIES):
            _survival(ax, _rows(Path(run_dir)/f"{policy}_scenes.csv"), record["maximum_evaluated_duration_s"],
                      version, COLORS[version_index % len(COLORS)])
            ax.set(title=POLICY_LABELS[policy], xlabel="闭环跟踪时间 / s", ylim=(0, 105))
            ax.grid(alpha=.18)
    axes[0].set_ylabel("全部目标尚未终止的场景比例 / %")
    axes[1].legend(frameon=False)
    fig.suptitle("版本对照：多目标同时连续跟踪比例", x=.055, y=.98, ha="left", fontsize=16, fontweight="bold")
    fig.text(.055, .02, FOOTNOTE, fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(.02, .12, .99, .9), w_pad=2.)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def render_run_dashboard(run_dir, output_dir, label="V1 原始多目标"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record, _ = load_run_metrics(run_dir)
    subtitle = f"{label}  |  {len(record['scene_membership'])} 场景 × {record['targets']} 目标，预算 {record['total_beams']} 束"
    versions = {label: Path(run_dir)}
    return [
        _bar_grid(versions, METRICS, output_dir/"metric_dashboard.png", subtitle, FOOTNOTE, 3, 3, (13.8, 11.5)),
        _bar_grid(versions, RESOURCE_METRICS, output_dir/"resource_latency.png", f"{label}：扫描开销与计算耗时", LATENCY_NOTE, 1, 3, (13.8, 5.)),
        plot_distributions(run_dir, output_dir/"scene_distributions.png", f"{label}：场景与目标分布"),
        write_metrics_csv(output_dir/"metrics.csv", versions),
    ]


def compare_versions(versions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _bar_grid(versions, METRICS, output_dir/"metrics_comparison.png", "多目标跟踪：三个版本的指标对照", FOOTNOTE, 3, 3, (15, 12)),
        _bar_grid(versions, RESOURCE_METRICS, output_dir/"resource_latency_comparison.png", "版本对照：扫描开销与计算耗时", LATENCY_NOTE, 1, 3, (15, 5.)),
        plot_comparison_survival(versions, output_dir/"tracking_survival_comparison.png"),
        write_metrics_csv(output_dir/"metrics_comparison.csv", versions),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="V1 原始多目标")
    parser.add_argument("--version", action="append", default=[], metavar="LABEL=RUN_DIR")
    args = parser.parse_args(argv)
    if args.version:
        versions = dict(value.split("=", 1) for value in args.version)
        paths = compare_versions(versions, args.output)
    elif args.run:
        paths = render_run_dashboard(args.run, args.output, args.label)
    else:
        parser.error("需要 --run 或至少一个 --version")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
