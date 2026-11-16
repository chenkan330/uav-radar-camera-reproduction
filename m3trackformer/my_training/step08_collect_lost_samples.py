"""第8步：冻结教学N-Net，模拟波束扫描并收集R-Net所需的漏检样本。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from step01_generate_truth import DT, DURATION
from step03_generate_beam_labels import NX, NY
from step05_generate_dataset import generate_one_trajectory
from step06_build_nnet import NNet

BASE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE / "outputs/step07/run_20261110_083100_336717_UTC/best_nnet.pth"
SOURCE_TRAJECTORIES = BASE / "outputs/step05/trajectories.npz"
OUTPUT_ROOT = BASE / "outputs/step08"


def collect_events(logits, labels, beam_count, max_failed, lost_limit=None):
    """连续漏检达到lost_limit即终止；仅为仍会执行的下一次扫描收集样本。"""
    if lost_limit is None:
        lost_limit = max_failed + 1
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("波束分数与标签维度不匹配。")
    if (not 1 <= beam_count < logits.shape[2] or max_failed < 1
            or lost_limit != max_failed + 1 or lost_limit > logits.shape[1]):
        raise ValueError("最大失败历史应等于终止阈值减1，且终止阈值不得超过预测长度。")
    if (not labels.size or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 1) or np.any(labels >= logits.shape[2])):
        raise ValueError("真实标签应为有效的非零波束编号。")
    if not np.isfinite(logits).all():
        raise ValueError("波束分数包含无效值。")
    # 禁止类别0后排序；排序结果只依赖N-Net分数，不依赖真实标签。
    ordered = np.argsort(-logits, axis=-1)[:, :, :beam_count]
    if np.any(ordered == 0):
        raise ValueError("N-Net候选中出现预留编号0。")
    window_ids, target_offsets, target_beams, histories, lengths = [], [], [], [], []
    # 第lost_limit次连续漏检后不再扫描，因此不生成它之后的训练标签。
    hit_counts = np.zeros(lost_limit, dtype=np.int64)
    first_miss_count = 0
    terminal_failure_count = 0
    for window in range(len(logits)):
        history = np.zeros((max_failed, beam_count), dtype=np.int16)
        for step in range(lost_limit):
            scanned = ordered[window, step]
            if labels[window, step] in scanned:
                hit_counts[step] += 1
                break
            if step == 0:
                first_miss_count += 1
            if step + 1 == lost_limit:
                terminal_failure_count += 1
                break
            history[step] = scanned
            window_ids.append(window)
            # 下标step+1是下一时刻；保存为从1开始的未来时刻编号。
            target_offsets.append(step + 2)
            target_beams.append(labels[window, step + 1])
            histories.append(history.copy())
            lengths.append(step + 1)
    return {
        "sample_window_index": np.asarray(window_ids, dtype=np.int32),
        "target_future_step": np.asarray(target_offsets, dtype=np.int16),
        "target_beam": np.asarray(target_beams, dtype=np.int16),
        "failed_beam_history": np.asarray(histories, dtype=np.int16).reshape(-1, max_failed, beam_count),
        "failure_count": np.asarray(lengths, dtype=np.int16),
    }, {"windows": len(logits), "first_miss_windows": first_miss_count,
        "samples": len(window_ids), "hit_counts_before_stop": hit_counts.tolist(),
        "terminal_failure_windows": terminal_failure_count}


def generate_split(ids, windows_per_trajectory, model, checkpoint, device, beam_count, max_failed,
                   lost_limit):
    model_kwargs = checkpoint["model_kwargs"]
    history_size, future_size = model_kwargs["history_size"], model_kwargs["future_size"]
    interval_count = int(round(DURATION / DT))
    time_s = np.arange(interval_count + 1) * DT
    possible_anchors = np.arange(history_size - 1, len(time_s) - future_size)
    if windows_per_trajectory > len(possible_anchors):
        raise ValueError(f"每条轨迹最多可选 {len(possible_anchors)} 个窗口。")
    anchors = possible_anchors[np.linspace(0, len(possible_anchors) - 1,
                                           windows_per_trajectory, dtype=int)]
    grid_u = -1.0 + 2.0 * np.arange(NX) / NX
    grid_v = -1.0 + 2.0 * np.arange(NY) / NY
    weights_x = np.exp(1j * np.pi * np.outer(np.arange(NX), grid_u)) / np.sqrt(NX)
    weights_y = np.exp(1j * np.pi * np.outer(np.arange(NY), grid_v)) / np.sqrt(NY)
    histories, future_labels, trajectory_ids, anchor_indices = [], [], [], []
    mean = np.asarray(model_kwargs["input_mean"])
    std = np.asarray(model_kwargs["input_std"])
    for index, trajectory_id in enumerate(ids):
        position, truth, measured, observed, best_beam, params = generate_one_trajectory(
            time_s, int(trajectory_id), weights_x, weights_y)
        for anchor in anchors:
            histories.append((measured[anchor - history_size + 1:anchor + 1] - mean) / std)
            future_labels.append(best_beam[anchor + 1:anchor + 1 + future_size])
            trajectory_ids.append(trajectory_id)
            anchor_indices.append(anchor)
        if (index + 1) % 10 == 0 or index + 1 == len(ids):
            print(f"已生成 {index + 1}/{len(ids)} 条新轨迹。", flush=True)
    x = np.asarray(histories, dtype=np.float32)
    labels = np.asarray(future_labels, dtype=np.int16)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int32)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int16)
    score_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 128):
            result = model(torch.from_numpy(x[start:start + 128]).to(device))
            score_parts.append(result["beam_logits"].cpu().numpy())
    logits = np.concatenate(score_parts).astype(np.float32, copy=False)
    samples, statistics = collect_events(logits, labels, beam_count, max_failed, lost_limit)
    sample_index = samples["sample_window_index"]
    offsets = samples["target_future_step"]
    assert np.array_equal(samples["target_beam"], labels[sample_index, offsets - 1])
    for i, window in enumerate(sample_index):
        for step in range(int(samples["failure_count"][i])):
            assert labels[window, step] not in samples["failed_beam_history"][i, step]
    if len(samples["target_beam"]) == 0:
        raise RuntimeError("没有收集到漏检样本；可以调整扫描波束数或轨迹数量。")
    result = {"x_history_normalized": x, "base_logits": logits,
              "future_beam_labels": labels, "window_trajectory_id": trajectory_ids,
              "window_anchor_index": anchor_indices, **samples,
              "sample_trajectory_id": trajectory_ids[sample_index],
              "sample_anchor_index": anchor_indices[sample_index],
              "input_mean": mean, "input_std": std,
              "history_size": np.int32(history_size), "future_size": np.int32(future_size),
              "num_classes": np.int32(model_kwargs["num_classes"]),
              "beam_count": np.int32(beam_count), "max_failed": np.int32(max_failed),
              "max_lost_steps": np.int32(lost_limit), "data_format_version": np.int32(2)}
    return result, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-trajectories", type=int, default=30)
    parser.add_argument("--validation-trajectories", type=int, default=10)
    parser.add_argument("--start-trajectory-id", type=int, default=100)
    parser.add_argument("--windows-per-trajectory", type=int, default=10)
    parser.add_argument("--beam-count", type=int, default=3)
    parser.add_argument("--max-failed", type=int, default=3,
                        help="R-Net可用的最大失败历史，必须等于--lost-limit减1。")
    parser.add_argument("--lost-limit", type=int, default=4,
                        help="连续漏检达到此数即终止，不再执行下一次扫描。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if min(args.train_trajectories, args.validation_trajectories, args.windows_per_trajectory,
           args.beam_count, args.max_failed, args.lost_limit) < 1 or args.start_trajectory_id < 0:
        parser.error("轨迹数、窗口数和扫描参数必须为正；轨迹编号不得为负。")
    if args.max_failed != args.lost_limit - 1:
        parser.error("--max-failed必须等于--lost-limit减1。")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用CUDA。")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if not SOURCE_TRAJECTORIES.is_file():
        raise FileNotFoundError(f"缺少原轨迹记录 {SOURCE_TRAJECTORIES}。")
    with np.load(SOURCE_TRAJECTORIES, allow_pickle=False) as data:
        existing_ids = data["trajectory_id"].copy()
    all_ids = np.arange(args.start_trajectory_id,
                        args.start_trajectory_id + args.train_trajectories + args.validation_trajectories)
    if np.intersect1d(existing_ids, all_ids).size:
        raise ValueError("新轨迹编号与第5步轨迹重叠，请调整 --start-trajectory-id。")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    kwargs = checkpoint["model_kwargs"]
    if kwargs["num_classes"] != NX * NY + 1 or args.lost_limit > kwargs["future_size"]:
        raise ValueError("模型类别数或预测长度与本程序参数不匹配。")
    expected_hash = checkpoint.get("training_config", {}).get("source_sha256", {}).get("step06_build_nnet.py")
    actual_hash = hashlib.sha256((BASE / "step06_build_nnet.py").read_bytes()).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("第6步模型定义与权重训练时不一致。")
    model = NNet(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))
    train_ids = all_ids[:args.train_trajectories]
    val_ids = all_ids[args.train_trajectories:]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output = args.output_root / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"冻结N-Net：{args.checkpoint}；设备：{device}。", flush=True)
    stats = {}
    for split, ids in (("train", train_ids), ("validation", val_ids)):
        print(f"生成 {split} 轨迹和漏检样本。", flush=True)
        result, info = generate_split(ids, args.windows_per_trajectory, model, checkpoint,
                                      device, args.beam_count, args.max_failed, args.lost_limit)
        np.savez_compressed(output / f"lost_{split}.npz", **result)
        with (output / f"lost_{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "trajectory_id", "anchor_index", "target_future_step",
                             "consecutive_failures", "failed_beams_oldest_first", "target_beam"])
            for i in range(len(result["target_beam"])):
                count = int(result["failure_count"][i])
                scanned = result["failed_beam_history"][i, :count].tolist()
                writer.writerow([i, int(result["sample_trajectory_id"][i]),
                                 int(result["sample_anchor_index"][i]), int(result["target_future_step"][i]),
                                 count, json.dumps(scanned), int(result["target_beam"][i])])
        stats[split] = {**info, "trajectory_ids": ids.tolist(),
                        "failure_length_counts": {str(k): int((result["failure_count"] == k).sum())
                                                  for k in range(1, args.max_failed + 1)}}
        print(f"{split}：{len(ids)}条轨迹，{info['windows']}个窗口，{info['samples']}个漏检训练样本。", flush=True)
    config = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "checkpoint_epoch": int(checkpoint["epoch"]), "device": str(device),
              "history_size": int(kwargs["history_size"]), "future_size": int(kwargs["future_size"]),
              "beam_count": args.beam_count, "max_failed": args.max_failed,
              "max_lost_steps": args.lost_limit, "data_format_version": 2,
              "windows_per_trajectory": args.windows_per_trajectory, "statistics": stats,
              "notes": ["N-Net权重冻结；每个独立窗口使用初始历史测量一次预测未来5步。",
                        "依次扫描N-Net给出的Top-K；漏检时记录已扫描波束，并以下一时刻真实波束为标签。",
                        f"连续{args.lost_limit}次漏检即终止；仅收集失败1到{args.max_failed}次后、仍会实际执行的下一时刻标签。",
                        "没有调用R-Net；本步骤是离线失联样本收集，尚非完整闭环。",
                        "假设目标最优波束在候选中即命中；训练和验证使用与第5步不重叠的新轨迹。",
                        "教学N-Net缺少论文原始模型的历史波束输入和缺测掩码，数据格式不直接适配作者R-Net。"]}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # 自定义输出路径时不更改项目默认输出指针。
    if args.output_root.resolve() == OUTPUT_ROOT.resolve():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "latest_run.txt").write_text(output.name, encoding="utf-8")
    print(f"第8步完成。输出文件夹：{output}", flush=True)
    print("CSV用于查看漏检过程，NPZ留给下一步搭建和训练教学版R-Net。", flush=True)


if __name__ == "__main__":
    main()
