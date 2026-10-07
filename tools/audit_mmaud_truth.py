"""Inspect actual Leica bag; GT is never imported by the tracking filter.

Writes the extracted evaluation-only CSV under ignored data/, while committing
only source hashes, coverage summaries and a Chinese timing diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from plot_style import configure_chinese_font


def audit(bag: Path, out: Path, max_gap: float = 0.30) -> dict:
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be positive and finite")
    rows: dict[str, list] = {}
    frames: dict[str, set] = {}
    transforms: dict[str, list] = {}
    with AnyReader([bag]) as reader:
        topics = [{"topic": c.topic, "type": c.msgtype, "count": c.msgcount}
                  for c in reader.connections]
        for connection, receipt, raw in reader.messages():
            if connection.topic not in ("/leica/point/absolute", "/leica/point/relative", "/tf"):
                continue
            msg = reader.deserialize(raw, connection.msgtype)
            if connection.topic == "/tf":
                for transform in msg.transforms:
                    key = f"{transform.header.frame_id} -> {transform.child_frame_id}"
                    t = transform.transform.translation
                    q = transform.transform.rotation
                    transforms.setdefault(key, []).append([t.x, t.y, t.z, q.x, q.y, q.z, q.w])
                continue
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            rows.setdefault(connection.topic, []).append(
                [stamp_ns, msg.point.x, msg.point.y, msg.point.z, receipt])
            frames.setdefault(connection.topic, set()).add(msg.header.frame_id)
    result = {"dataset": "MMAUD V1 Mavic3 independent Leica ground truth",
              "source_sha256": hashlib.sha256(bag.read_bytes()).hexdigest(),
              "source_bytes": bag.stat().st_size, "topics": topics,
              "max_continuous_gap_s": max_gap, "position_topics": {},
              "transforms": {}, "truth_used_for_tracking": False}
    for topic, values in rows.items():
        stamp_ns = np.array([x[0] for x in values], dtype=np.int64)
        if np.any(np.diff(stamp_ns) <= 0):
            raise ValueError("Truth header timestamps must strictly increase")
        xyz = np.array([x[1:4] for x in values], dtype=float)
        if not np.all(np.isfinite(xyz)):
            raise ValueError("Nonfinite Leica sample")
        gaps = np.diff(stamp_ns) / 1e9
        segments = np.split(np.arange(len(values)), np.where(gaps > max_gap)[0] + 1)
        record = {"count": len(values), "frame_ids": sorted(frames[topic]),
                  "header_start_ns": int(stamp_ns[0]), "header_end_ns": int(stamp_ns[-1]),
                  "duration_s": float((stamp_ns[-1] - stamp_ns[0]) / 1e9),
                  "gap_percentiles_s": dict(zip(["min", "p25", "median", "p75", "p95", "max"],
                                                 np.percentile(gaps, [0, 25, 50, 75, 95, 100]).tolist())),
                  "coordinate_min_m": xyz.min(axis=0).tolist(),
                  "coordinate_max_m": xyz.max(axis=0).tolist(),
                  "segments": [{"start_ns": int(stamp_ns[s[0]]), "end_ns": int(stamp_ns[s[-1]]),
                                "samples": len(s)} for s in segments]}
        result["position_topics"][topic] = record
        dest = bag.parent / ("truth_" + topic.rsplit("/", 1)[-1] + ".csv")
        with dest.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp_ns", "x_m", "y_m", "z_m", "bag_receipt_ns"])
            writer.writerows(values)
    for key, values in transforms.items():
        array = np.asarray(values)
        result["transforms"][key] = {"count": len(array),
                    "translation_range_m": np.ptp(array[:, :3], axis=0).tolist(),
                    "quaternion_range": np.ptp(array[:, 3:], axis=0).tolist(),
                    "meaning": "dynamic Leica target transform; not a radar/camera rig extrinsic"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    configure_chinese_font()
    relative = rows["/leica/point/relative"]
    stamp_ns = np.array([x[0] for x in relative], dtype=np.int64)
    elapsed = (stamp_ns - stamp_ns[0]) / 1e9
    gaps = np.diff(stamp_ns) / 1e9
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.plot(elapsed[1:], gaps, lw=1.1, label="相邻真值时间间隔")
    ax.axhline(max_gap, color="#bd3434", ls="--", label=f"连续区间阈值：{max_gap:.2f} 秒")
    ax.set(title="MMAUD 真值连续性审计（未运行精度评价）", xlabel="距首个真值样本的时间（秒）", ylabel="相邻样本间隔（秒）")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(out / "truth_timing.png", dpi=170); plt.close(fig)
    r = result["position_topics"]["/leica/point/relative"]
    report = "# 实际真值文件审计\n\n"
    report += "来源：[MMAUD官方页面](https://ntu-aris.github.io/MMAUD/)，Mavic3独立真值bag。真值只供评价，未输入跟踪器。原始bag和提取CSV位于本地忽略目录。\n\n"
    report += f"SHA-256：`{result['source_sha256']}`；{result['source_bytes']:,}字节。\n\n"
    report += f"实际相对位置样本{r['count']}个，跨度{r['duration_s']:.3f}秒，名义约5 Hz。最大相邻间隔{r['gap_percentiles_s']['max']:.6f}秒。官网321.1秒是传感器序列标称时长，独立真值覆盖实际较短，不能假设全程有真值。\n\n"
    report += "| 连续区间开始时间（Unix秒） | 结束时间（Unix秒） | 样本数 |\n|---:|---:|---:|\n"
    for segment in r["segments"]:
        report += f"| {segment['start_ns']/1e9:.6f} | {segment['end_ns']/1e9:.6f} | {segment['samples']} |\n"
    report += f"\n连续区间以相邻间隔≤{max_gap:.2f}秒定义，仅用于覆盖审计。后续评价必须记录实际插值阈值，禁止跨过缺口或外推。\n\n"
    report += "`/leica/point/absolute`与`relative`都声明`world`，但这不足以证明与雷达或相机同坐标。`/tf`仅有随目标运动的`world→leica_abs`、`world→leica_rel`，旋转恒为单位，不能把它们当作传感器外参。棱镜位置与机身回波/图像框中心参考点也需另行说明。\n\n![真值连续性](truth_timing.png)\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, default=ROOT / "data/public/mmaud/2023-08-24-11-14-40_mavic3.bag")
    parser.add_argument("--out", type=Path, default=ROOT / "output/v16_truth_audit")
    parser.add_argument("--max-gap", type=float, default=.30)
    args = parser.parse_args()
    result = audit(args.bag, args.out, args.max_gap)
    print(json.dumps({"sha256": result["source_sha256"], "position_topics": result["position_topics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
