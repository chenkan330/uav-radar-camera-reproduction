"""第10步：默认运行原文多目标流程；--single-target运行旧教学单目标对照。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import os
import subprocess
import sys

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    project_python = project_root/'.venv/Scripts/python.exe'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('MPLCONFIGDIR', str(project_root/'.cache/matplotlib'))
    if project_python.exists() and Path(sys.executable).resolve() != project_python.resolve():
        print(f'自动使用项目环境：{project_python}', flush=True)
        raise SystemExit(subprocess.run([str(project_python), '-X', 'utf8',
                                        str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)
    if '--maneuver-uncertainty' in sys.argv[1:]:
        from core.maneuver_nnet_rnet import main as maneuver_main
        maneuver_main([value for value in sys.argv[1:] if value != '--maneuver-uncertainty'])
        raise SystemExit(0)
    if '--single-target' not in sys.argv[1:]:
        from core.multi_target import main as multi_target_main
        multi_target_main()
        raise SystemExit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from step03_generate_beam_labels import NX, NY, calculate_beam_gains
from step06_build_nnet import NNet
from step09_train_rnet import RNet, frozen_nnet_context

BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / "outputs/step10"
POLICIES = ("nnet_only", "nnet_rnet", "geometry_hold")


def observed_xyz(measurement, base_station):
    distance, azimuth, polar = measurement
    return np.array([distance * np.sin(polar) * np.cos(azimuth),
                     distance * np.sin(polar) * np.sin(azimuth),
                     distance * np.cos(polar)]) + base_station


def track_trajectory(nnet, rnet, model_kwargs, measured, truth_positions, true_beams,
                     dt_s, beam_count, max_lost_steps, policy, device,
                     context_function=frozen_nnet_context):
    """真值仅用于模拟检测和最后计分；模型只接收已经命中的测量。"""
    history_size = model_kwargs["history_size"]
    mean, std = np.asarray(model_kwargs["input_mean"]), np.asarray(model_kwargs["input_std"])
    base_station = np.asarray(model_kwargs["base_station_m"])
    count = len(measured)
    if count <= history_size or max_lost_steps > model_kwargs["future_size"]:
        raise ValueError("轨迹长度或连续漏检上限不满足模型预测长度。")
    if policy not in POLICIES:
        raise ValueError("未知跟踪策略。")
    available = np.zeros((count, 3), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    available[:history_size] = measured[:history_size]
    valid[:history_size] = True
    rows, squared_errors, latencies = [], [], []
    for index in range(history_size):
        position = observed_xyz(measured[index], base_station)
        rows.append({"step": index, "time_s": index * dt_s, "mode": "initialization",
                     "hit": True, "consecutive_misses": 0, "true_beam": int(true_beams[index]),
                     "scanned_beams": "[]", "inference_ms": 0.,
                     **{f"true_{axis}_m": float(truth_positions[index, j]) for j, axis in enumerate("xyz")},
                     **{f"observed_{axis}_m": float(position[j]) for j, axis in enumerate("xyz")},
                     **{f"estimated_{axis}_m": float(position[j]) for j, axis in enumerate("xyz")}})
    failed, context, misses = [], None, 0
    miss_events, recoveries, correct_top1, evaluated = 0, 0, 0, 0
    succeeded = True
    max_failed = max_lost_steps - 1
    if policy == "geometry_hold":
        wx = np.exp(1j * np.pi * np.outer(np.arange(NX), -1 + 2 * np.arange(NX) / NX)) / np.sqrt(NX)
        wy = np.exp(1j * np.pi * np.outer(np.arange(NY), -1 + 2 * np.arange(NY) / NY)) / np.sqrt(NY)

    def make_context(end):
        start = end - history_size + 1
        mask = valid[start:end + 1].copy()
        x = ((available[start:end + 1] - mean) / std).astype(np.float32)
        x[~mask] = 0.
        return context_function(nnet, torch.from_numpy(x[None]).to(device),
                                torch.from_numpy(mask[None]).to(device))

    def prepare(end, offset, reuse_context=False, prior_inference_ms=0.):
        nonlocal context
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if policy == "geometry_hold":
            last_index = np.flatnonzero(valid[:end + 1])[-1]
            _, azimuth, polar = available[last_index]
            u = np.array([np.sin(polar) * np.cos(azimuth)])
            v = np.array([np.sin(polar) * np.sin(azimuth)])
            score = calculate_beam_gains(u, v, wx, wy)[0]
            candidates = (np.argsort(-score)[:beam_count] + 1).tolist()
        else:
            if offset == 0 and not reuse_context:
                context = make_context(end)
            base_logits = context["beam_logits"][:, offset]
            if policy == "nnet_rnet" and offset > 0:
                scan_history = np.zeros((1, max_failed, beam_count), dtype=np.int64)
                scan_history[0, :len(failed)] = np.asarray(failed)
                result = rnet(base_logits, context["memory"],
                              torch.from_numpy(scan_history).to(device),
                              torch.tensor([len(failed)], device=device), context["memory_valid"])
                base_logits = result["beam_logits"]
            candidates = base_logits.topk(beam_count, dim=-1).indices[0].cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return candidates, (time.perf_counter() - started) * 1000. + prior_inference_ms

    with torch.no_grad():
        candidates, inference_ms = prepare(history_size - 1, 0)
        for index in range(history_size, count):
            # 最优波束包含于扫描集合即模拟检测成功；没有将真值传给网络。
            hit = int(true_beams[index]) in candidates
            mode = "normal" if misses == 0 else "reacquisition"
            evaluated += 1
            correct_top1 += int(candidates[0] == int(true_beams[index]))
            latencies.append(inference_ms)
            observed = estimate = np.full(3, np.nan)
            next_context_ms = 0.
            if hit:
                available[index] = measured[index]
                valid[index] = True
                observed = observed_xyz(measured[index], base_station)
                if misses:
                    recoveries += 1
                misses = 0
                failed = []
                if policy == "geometry_hold":
                    estimate = observed
                else:
                    # 此次命中后才允许将当前测量加入历史，再计算当前位置估计。
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    context_started = time.perf_counter()
                    context = make_context(index)
                    estimate = context["current_position_m"][0].cpu().numpy()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    next_context_ms = (time.perf_counter() - context_started) * 1000.
                squared_errors.append(float(np.sum((estimate - truth_positions[index])**2)))
            else:
                if misses == 0:
                    miss_events += 1
                misses += 1
                failed.append(candidates.copy())
            rows.append({"step": index, "time_s": index * dt_s, "mode": mode,
                         "hit": bool(hit), "consecutive_misses": misses,
                         "true_beam": int(true_beams[index]), "scanned_beams": json.dumps(candidates),
                         "inference_ms": inference_ms,
                         **{f"true_{axis}_m": float(truth_positions[index, j]) for j, axis in enumerate("xyz")},
                         **{f"observed_{axis}_m": float(observed[j]) for j, axis in enumerate("xyz")},
                         **{f"estimated_{axis}_m": float(estimate[j]) for j, axis in enumerate("xyz")}})
            if misses >= max_lost_steps:
                succeeded = False
                break
            if index + 1 < count:
                candidates, inference_ms = prepare(index, misses, reuse_context=hit,
                                                   prior_inference_ms=next_context_ms)
    metrics = {"success": succeeded, "evaluated_steps": evaluated,
               "tracking_duration_s": evaluated * dt_s,
               "maximum_duration_s": (count - history_size) * dt_s,
               "missing_events": miss_events, "recoveries": recoveries,
               "recovery_rate": recoveries / miss_events if miss_events else None,
               "top1_during_tracking": correct_top1 / evaluated,
               "hit_rate_during_tracking": sum(row["hit"] for row in rows[history_size:]) / evaluated,
               "position_rmse_m_on_detections": float(np.sqrt(np.mean(squared_errors))) if squared_errors else None,
               "mean_inference_ms": float(np.mean(latencies)),
               "final_consecutive_misses": misses}
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-target', action='store_true', help='使用旧教学版单目标流程')
    parser.add_argument("--rnet-checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--trajectory-limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--targets", type=int, help="原文多目标模式，例如 --targets 3 --total-beams 12")
    parser.add_argument("--total-beams", type=int, help="多目标共同使用的总波束预算")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用CUDA。")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.trajectory_limit is not None and args.trajectory_limit < 1:
        parser.error("trajectory-limit必须为正。")
    checkpoint_path = args.rnet_checkpoint
    if checkpoint_path is None:
        run = (BASE / "outputs/step09/latest_run.txt").read_text(encoding="utf-8").strip()
        checkpoint_path = BASE / "outputs/step09" / run / "best_rnet.pth"
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    nnet_path = Path(cp["nnet_checkpoint"])
    if hashlib.sha256(nnet_path.read_bytes()).hexdigest() != cp["nnet_sha256"]:
        raise ValueError("N-Net权重与R-Net训练时不匹配。")
    ncp = torch.load(nnet_path, map_location="cpu", weights_only=True)
    nnet = NNet(**ncp["model_kwargs"])
    nnet.load_state_dict(ncp["model_state_dict"], strict=True)
    nnet.to(device).eval()
    rnet = RNet(**cp["model_kwargs"])
    rnet.load_state_dict(cp["model_state_dict"], strict=True)
    rnet.to(device).eval()
    torch.set_num_threads(4)
    source = BASE / "outputs/step05/trajectories.npz"
    with np.load(source, allow_pickle=False) as data:
        indices = np.flatnonzero(data["split"] == args.split)
        if args.trajectory_limit:
            indices = indices[:args.trajectory_limit]
        if len(indices) == 0:
            raise ValueError("没有选择到轨迹。")
        ids = data["trajectory_id"][indices].copy()
        positions = data["true_position_m"][indices].copy()
        beams = data["best_beam"][indices].copy()
        measured = np.stack([data[key][indices] for key in
                             ("measured_range_m", "measured_azimuth_rad", "measured_polar_rad")], axis=-1)
        dt_s = float(data["dt_s"])
    if args.split == "test":
        used = set(ncp["training_config"]["train_trajectory_ids"] + ncp["training_config"]["validation_trajectory_ids"])
        used.update(cp.get("train_trajectory_ids", []))
        used.update(cp.get("validation_trajectory_ids", []))
        if used.intersection(ids.tolist()):
            raise ValueError("测试轨迹与训练或选模轨迹重叠。")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output = OUTPUT_ROOT / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    all_metrics, all_events, plots = [], [], {}
    for policy in POLICIES:
        for offset, trajectory_id in enumerate(ids):
            metrics, events = track_trajectory(nnet, rnet, ncp["model_kwargs"], measured[offset],
                                               positions[offset], beams[offset], dt_s,
                                               cp["beam_count"], cp["max_lost_steps"], policy, device)
            all_metrics.append({"policy": policy, "trajectory_id": int(trajectory_id), **metrics})
            all_events.extend({"policy": policy, "trajectory_id": int(trajectory_id), **event} for event in events)
            if offset == 0:
                plots[policy] = events
        print(f"{policy}：完成 {len(ids)} 条 {args.split} 轨迹。", flush=True)
    summaries = []
    for policy in POLICIES:
        rows = [row for row in all_metrics if row["policy"] == policy]
        missed = sum(row["missing_events"] for row in rows)
        recovered = sum(row["recoveries"] for row in rows)
        summaries.append({"policy": policy, "trajectories": len(rows),
                          "success_rate": float(np.mean([row["success"] for row in rows])),
                          "mean_tracking_duration_s": float(np.mean([row["tracking_duration_s"] for row in rows])),
                          "maximum_duration_s": rows[0]["maximum_duration_s"],
                          "missing_events": missed, "recoveries": recovered,
                          "recovery_rate": recovered / missed if missed else None})
    for filename, rows in (("trajectory_metrics.csv", all_metrics), ("tracking_events.csv", all_events),
                           ("summary.csv", summaries)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(positions[0, :, 0], positions[0, :, 1], color="black", label="真实轨迹")
    for policy, events in plots.items():
        xs = [event["estimated_x_m"] for event in events]
        ys = [event["estimated_y_m"] for event in events]
        axes[0].plot(xs, ys, ".-", markersize=3, label=policy)
    axes[0].set(title=f"轨迹 {int(ids[0])}：仅显示实际检测到的位置", xlabel="x / m", ylabel="y / m")
    axes[0].legend(fontsize=7)
    axes[1].bar([row["policy"] for row in summaries], [row["mean_tracking_duration_s"] for row in summaries])
    axes[1].axhline(summaries[0]["maximum_duration_s"], color="gray", linestyle="--", label="完整评估时长")
    axes[1].set(title="平均连续跟踪时长", ylabel="秒")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "tracking_comparison.png", dpi=150)
    plt.close(fig)
    config = {"rnet_checkpoint": str(checkpoint_path.resolve()), "nnet_checkpoint": str(nnet_path),
              "source_trajectories": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "split": args.split, "trajectory_ids": ids.tolist(), "dt_s": dt_s,
              "beam_count": cp["beam_count"], "max_lost_steps": cp["max_lost_steps"],
              "history_size": ncp["model_kwargs"]["history_size"],
              "notes": ["最优波束在候选中即模拟命中；没有模拟真实回波检测。",
                        "只有命中后才能读入该时刻的测量；连续4次漏检立即结束，不执行第5次扫描。",
                        "重捕获成功后重新滑动历史窗口，失联位置使用掩码；N-Net没有专门接受缺测训练。",
                        "初始化历史视为已获取；跟踪时长从初始化结束后计算。",
                        "本脚本是单目标教学闭环，不是论文完整多目标复现。"]}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({"split": args.split, "policies": summaries}, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "latest_run.txt").write_text(output.name, encoding="utf-8")
    print(json.dumps({"output": str(output), "policies": summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
