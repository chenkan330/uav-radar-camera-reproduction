"""第9步：固定N-Net，训练教学R-Net，用失败波束历史修正下一时刻分数。"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from step06_build_nnet import NNet

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / "outputs" / "step08"
OUTPUT_ROOT = BASE / "outputs" / "step09"
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
SEED = 44
METRIC_KEYS = ("cross_entropy", "top1_accuracy", "topk_accuracy")


@torch.no_grad()
def frozen_nnet_context(nnet, x_history_normalized, history_valid=None):
    """复用N-Net子层，返回运动特征；可在闭环中屏蔽缺失的历史测量。

    全部历史有效时，计算与第6步forward相同。缺测掩码不增加参数，也不代表
    原N-Net已接受缺测训练；这一局限会记录在闭环结果中。
    """
    if x_history_normalized.ndim != 3 or x_history_normalized.shape[1:] != (nnet.history_size, 3):
        raise ValueError("历史测量形状不符合N-Net配置。")
    if not torch.isfinite(x_history_normalized).all():
        raise ValueError("历史测量存在非有限数值。")
    nnet.eval()
    batch = x_history_normalized.shape[0]
    if history_valid is None:
        history_valid = torch.ones((batch, nnet.history_size), dtype=torch.bool,
                                   device=x_history_normalized.device)
    else:
        history_valid = torch.as_tensor(history_valid, dtype=torch.bool,
                                        device=x_history_normalized.device)
    if history_valid.shape != (batch, nnet.history_size) or not history_valid.any(dim=1).all():
        raise ValueError("历史有效掩码形状错误，或某条样本没有任何有效历史。")
    clean_input = x_history_normalized.masked_fill(~history_valid.unsqueeze(-1), 0.)
    features = nnet.measurement_embedding(clean_input)
    hidden = torch.cat([features, nnet.future_tokens.expand(batch, -1, -1)], dim=1)
    hidden = nnet.embedding_dropout(hidden + nnet.time_encoding)
    padding_mask = None
    if not bool(history_valid.all()):
        padding_mask = torch.cat([~history_valid,
                                  torch.zeros((batch, nnet.future_size), dtype=torch.bool,
                                              device=history_valid.device)], dim=1)
    for layer in nnet.encoder_layers:
        hidden = layer(hidden, src_key_padding_mask=padding_mask)
    logits = nnet.beam_head(hidden[:, nnet.history_size:])
    logits = logits.masked_fill(nnet.padding_class_mask, torch.finfo(logits.dtype).min)
    measurement = clean_input[:, -1] * nnet.input_std + nnet.input_mean
    distance, azimuth, polar = measurement.unbind(dim=-1)
    observed = torch.stack([distance * torch.sin(polar) * torch.cos(azimuth),
                            distance * torch.sin(polar) * torch.sin(azimuth),
                            distance * torch.cos(polar)], dim=-1) + nnet.base_station_m
    correction = nnet.position_head(hidden[:, nnet.history_size - 1])
    return {"beam_logits": logits, "memory": hidden[:, :nnet.history_size],
            "memory_valid": history_valid, "observed_position_m": observed,
            "current_position_m": observed + correction}


class RNet(nn.Module):
    """失败扫描编码→因果解码器→与冻结运动特征交互→波束分数修正。"""

    def __init__(self, num_classes=1025, memory_dim=64, max_failed=3,
                 d_model=64, n_head=4, num_layers=1, dropout=.1):
        super().__init__()
        if min(num_classes, memory_dim, max_failed, d_model, n_head, num_layers) < 1:
            raise ValueError("网络维度和层数必须为正。")
        if num_classes < 2 or d_model % 2 or d_model % n_head or not 0 <= dropout < 1:
            raise ValueError("类别数、特征维度或Dropout配置错误。")
        self.num_classes, self.max_failed = num_classes, max_failed
        self.failed_embedding = nn.Linear(num_classes, d_model)
        self.memory_projection = nn.Identity() if memory_dim == d_model else nn.Linear(memory_dim, d_model)
        position = torch.arange(max_failed, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model))
        encoding = torch.zeros(1, max_failed, d_model)
        encoding[0, :, 0::2], encoding[0, :, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("failure_encoding", encoding)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head,
                                       dim_feedforward=4 * d_model, dropout=dropout,
                                       batch_first=True)
            for _ in range(num_layers)
        ])
        self.projection = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, base_logits, motion_features, failed_beams, failure_count, memory_valid=None):
        if base_logits.ndim != 2 or base_logits.shape[-1] != self.num_classes:
            raise ValueError("基础波束分数形状错误。")
        batch = base_logits.shape[0]
        failed_beams = failed_beams.long()
        failure_count = failure_count.long()
        if failed_beams.ndim != 3 or failed_beams.shape[:2] != (batch, self.max_failed):
            raise ValueError("失败扫描历史形状应为[batch,max_failed,K]。")
        if failure_count.shape != (batch,) or not ((failure_count >= 1) & (failure_count <= self.max_failed)).all():
            raise ValueError("连续漏检次数超出范围。")
        if not ((failed_beams >= 0) & (failed_beams < self.num_classes)).all():
            raise ValueError("失败波束编号超出类别范围。")
        failure_valid = torch.arange(self.max_failed, device=base_logits.device)[None] < failure_count[:, None]
        if ((failed_beams == 0) & failure_valid.unsqueeze(-1)).any():
            raise ValueError("真实扫描记录不能包含预留编号0。")
        multi_hot = base_logits.new_zeros((batch, self.max_failed, self.num_classes))
        multi_hot.scatter_(2, failed_beams, 1.)
        multi_hot[:, :, 0] = 0.
        multi_hot = multi_hot * failure_valid.unsqueeze(-1)
        hidden = self.failed_embedding(multi_hot) + self.failure_encoding
        memory = self.memory_projection(motion_features)
        if memory_valid is None:
            memory_valid = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        else:
            memory_valid = memory_valid.to(device=memory.device, dtype=torch.bool)
        if memory_valid.shape != memory.shape[:2] or not memory_valid.any(dim=1).all():
            raise ValueError("运动特征的有效掩码错误。")
        causal_mask = torch.triu(torch.ones(self.max_failed, self.max_failed,
                                           dtype=torch.bool, device=hidden.device), diagonal=1)
        for layer in self.decoder_layers:
            hidden = layer(hidden, memory, tgt_mask=causal_mask,
                           tgt_key_padding_mask=~failure_valid,
                           memory_key_padding_mask=~memory_valid)
        last = hidden[torch.arange(batch, device=hidden.device), failure_count - 1]
        delta = self.projection(last)
        # 类别0的基础分数是浮点最小值，先清零再相加，避免加法下溢。
        clean_base = base_logits.clone()
        clean_base[:, 0] = 0.
        adjusted = clean_base + delta
        adjusted = adjusted.masked_fill(torch.arange(self.num_classes, device=adjusted.device) == 0,
                                        torch.finfo(adjusted.dtype).min)
        return {"beam_logits": adjusted, "delta_logits": delta}


def _latest_data_dir():
    pointer = DATA_ROOT / "latest_run.txt"
    if not pointer.is_file():
        raise FileNotFoundError("请先运行修正后的第8步，生成R-Net漏检样本。")
    return DATA_ROOT / pointer.read_text(encoding="utf-8").strip()


def prepare_dataset(path, nnet, device, expected_max_failed):
    with np.load(path, allow_pickle=False) as data:
        x = data["x_history_normalized"].copy()
        stored_logits = data["base_logits"].copy()
        window = data["sample_window_index"].astype(np.int64)
        steps = data["target_future_step"].astype(np.int64)
        labels = data["target_beam"].astype(np.int64)
        failed = data["failed_beam_history"].astype(np.int64)
        counts = data["failure_count"].astype(np.int64)
        trajectory_ids = data["sample_trajectory_id"].astype(np.int64)
        if int(data["max_failed"]) != expected_max_failed:
            raise ValueError("第8步数据的连续漏检边界不一致，请重新生成。")
        if not len(labels) or failed.shape[1] != expected_max_failed:
            raise ValueError("没有有效漏检样本，或失败历史长度不正确。")
        if not np.array_equal(steps, counts + 1):
            raise ValueError("漏检次数和下一时刻标签没有对齐。")
        if not np.array_equal(labels, data["future_beam_labels"][window, steps - 1]):
            raise ValueError("波束监督标签与窗口不对应。")
    memory_parts = []
    nnet.eval()
    for start in range(0, len(x), 128):
        context = frozen_nnet_context(nnet, torch.from_numpy(x[start:start + 128]).float().to(device))
        actual_logits = context["beam_logits"].cpu().numpy()
        if not np.allclose(actual_logits, stored_logits[start:start + 128], atol=1e-4, rtol=1e-5):
            raise ValueError("第8步分数与当前冻结N-Net不一致，请检查权重或重新生成数据。")
        memory_parts.append(context["memory"].cpu())
    memory = torch.cat(memory_parts)[torch.from_numpy(window)]
    base = torch.from_numpy(stored_logits[window, steps - 1])
    dataset = TensorDataset(base.float(), memory.float(), torch.from_numpy(failed),
                            torch.from_numpy(counts), torch.from_numpy(labels))
    return dataset, np.unique(trajectory_ids)


def run_epoch(model, loader, device, top_k, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct1, correctk, count = 0., 0, 0, 0
    with torch.set_grad_enabled(training):
        for base, memory, failed, lengths, label in loader:
            base, memory, failed, lengths, label = [x.to(device, non_blocking=True)
                                                   for x in (base, memory, failed, lengths, label)]
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(base, memory, failed, lengths)["beam_logits"]
            loss = nn.functional.cross_entropy(logits, label)
            if not torch.isfinite(loss):
                raise FloatingPointError("R-Net损失出现非有限值。")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                candidates = logits.topk(top_k, dim=-1).indices
                correct1 += int((candidates[:, 0] == label).sum())
                correctk += int((candidates == label[:, None]).any(dim=-1).sum())
                loss_sum += float(loss) * len(label)
                count += len(label)
    return {"cross_entropy": loss_sum / count, "top1_accuracy": correct1 / count,
            "topk_accuracy": correctk / count}


def _selection(metrics):
    return metrics["topk_accuracy"], -metrics["cross_entropy"]


def save_curves(rows, best_epoch, beam_count, path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, key, label in zip(axes, METRIC_KEYS, ("波束交叉熵", "Top-1准确率", f"Top-{beam_count}命中率")):
        factor = 1 if key == "cross_entropy" else 100
        for split, name in (("train", "训练集"), ("val", "验证集")):
            axis.plot([r["epoch"] for r in rows], [r[split + "_" + key] * factor for r in rows], label=name)
        axis.axvline(best_epoch, color="gray", linestyle=":")
        axis.set(xlabel="训练轮数（0为未修正基线）", ylabel=label)
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle(f"第9步：教学R-Net；按验证Top-{beam_count}选择第{best_epoch}轮")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.d_model, args.num_layers) < 1:
        parser.error("训练轮数、批量和网络大小必须为正。")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("学习率必须为正有限数。")
    if args.d_model % 4 or not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("特征维度应能被4整除，Dropout应在[0,1)。")
    if not 0 <= args.seed < 2**32:
        parser.error("随机种子应在0到2**32-1之间。")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用CUDA。")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    data_dir = (args.data_dir or _latest_data_dir()).resolve()
    data_config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    lost_limit = int(data_config.get("lost_limit", data_config.get("max_lost_steps", 4)))
    max_failed = int(data_config["max_failed"])
    if lost_limit != 4 or max_failed != lost_limit - 1:
        raise ValueError("本教学闭环在第4次连续漏检后终止；请先重跑边界为max_failed=3、lost_limit=4的第8步。")
    checkpoint_path = Path(data_config["checkpoint"])
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != data_config["checkpoint_sha256"]:
        raise ValueError("第8步记录的N-Net权重文件已改变。")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    nnet = NNet(**checkpoint["model_kwargs"])
    nnet.load_state_dict(checkpoint["model_state_dict"], strict=True)
    nnet.to(device).eval()
    for parameter in nnet.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    train_set, train_ids = prepare_dataset(data_dir / "lost_train.npz", nnet, device, max_failed)
    val_set, val_ids = prepare_dataset(data_dir / "lost_validation.npz", nnet, device, max_failed)
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError("R-Net训练/验证集有相同轨迹。")
    model_kwargs = {"num_classes": nnet.num_classes, "memory_dim": nnet.measurement_embedding.out_features,
                    "max_failed": max_failed, "d_model": args.d_model, "n_head": 4,
                    "num_layers": args.num_layers, "dropout": args.dropout}
    model = RNet(**model_kwargs).to(device)
    beam_count = int(data_config["beam_count"])
    loader_options = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_options)
    train_check_loader = DataLoader(train_set, shuffle=False, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output = OUTPUT_ROOT / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "seed": args.seed, "device": str(device), "model_kwargs": model_kwargs,
              "beam_count": beam_count, "max_lost_steps": lost_limit, "step08_data_dir": str(data_dir),
              "nnet_checkpoint": str(checkpoint_path.resolve()), "nnet_sha256": checkpoint_hash,
              "train_trajectory_ids": train_ids.tolist(), "validation_trajectory_ids": val_ids.tolist(),
              "train_samples": len(train_set), "validation_samples": len(val_set),
              "selection_rule": "maximum_validation_topk_then_minimum_cross_entropy_including_epoch0",
              "data_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                              for name in ("lost_train.npz", "lost_validation.npz")},
              "notes": ["冻结教学N-Net，R-Net使用失败波束编码和冻结运动特征的交叉注意力。",
                        "R-Net只学习波束分数修正；标签仅用于损失，不作为网络输入。",
                        "本步骤不读取最终测试集。教学N-Net尚未接受缺测训练，闭环泛化能力需要后续评价。"]}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_checkpoint(epoch, metrics):
        torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model_kwargs": model_kwargs, "nnet_checkpoint": str(checkpoint_path.resolve()),
                    "nnet_sha256": checkpoint_hash, "nnet_model_kwargs": checkpoint["model_kwargs"],
                    "beam_count": beam_count, "max_lost_steps": lost_limit,
                    "step08_data_dir": str(data_dir), "train_trajectory_ids": train_ids.tolist(),
                    "validation_trajectory_ids": val_ids.tolist(), "epoch": epoch,
                    "validation_metrics": metrics, "config": config}, output / "best_rnet.pth")

    started = time.perf_counter()
    baseline_train = run_epoch(model, train_check_loader, device, beam_count)
    baseline_val = run_epoch(model, val_loader, device, beam_count)
    best_metrics, best_epoch = baseline_val.copy(), 0
    save_checkpoint(0, best_metrics)
    rows = []

    def log_row(epoch, train, val, seconds):
        return {"epoch": epoch, **{"train_" + k: train[k] for k in METRIC_KEYS},
                **{"val_" + k: val[k] for k in METRIC_KEYS}, "epoch_seconds": seconds}

    rows.append(log_row(0, baseline_train, baseline_val, 0.))
    print(f"第9步：冻结N-Net；R-Net训练样本{len(train_set)}，验证样本{len(val_set)}；输出{output}", flush=True)
    print(f"未修正基线：Top-1 {baseline_val['top1_accuracy']:.2%}；Top-{beam_count} {baseline_val['topk_accuracy']:.2%}", flush=True)
    with (output / "training_log.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerow(rows[0])
        stream.flush()
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            train = run_epoch(model, train_loader, device, beam_count, optimizer)
            val = run_epoch(model, val_loader, device, beam_count)
            row = log_row(epoch, train, val, time.perf_counter() - epoch_started)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            improved = _selection(val) > _selection(best_metrics)
            if improved:
                best_metrics, best_epoch = val.copy(), epoch
                save_checkpoint(epoch, val)
            print(f"Epoch {epoch:02d}/{args.epochs} | 训练CE {train['cross_entropy']:.4f} | 验证CE {val['cross_entropy']:.4f} | "
                  f"验证Top-1 {val['top1_accuracy']:.2%} | Top-{beam_count} {val['topk_accuracy']:.2%}" +
                  (" | 保存最佳" if improved else ""), flush=True)
    saved = torch.load(output / "best_rnet.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    measured = run_epoch(model, val_loader, device, beam_count)
    if any(not np.isclose(measured[key], best_metrics[key], rtol=1e-5, atol=1e-6) for key in METRIC_KEYS):
        raise RuntimeError("重载R-Net后验证指标与保存时不一致。")
    if _selection(measured) < _selection(baseline_val):
        raise RuntimeError("保存模型劣于未修正基线。")
    save_curves(rows, best_epoch, beam_count, output / "training_curves.png")
    summary = {"epochs_completed": args.epochs, "best_epoch": best_epoch,
               "elapsed_seconds": time.perf_counter() - started, "baseline_train": baseline_train,
               "baseline_validation": baseline_val, "best_validation": measured,
               "beam_count": beam_count, "test_set_used": False, "output_directory": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    text = (f"第9步：教学R-Net训练{args.epochs}轮，最佳第{best_epoch}轮。\n"
            f"冻结N-Net未修正基线：验证Top-1 {baseline_val['top1_accuracy']:.2%}，Top-{beam_count} {baseline_val['topk_accuracy']:.2%}。\n"
            f"加入R-Net后：验证Top-1 {measured['top1_accuracy']:.2%}，Top-{beam_count} {measured['topk_accuracy']:.2%}。\n"
            "指标只针对漏检数据中的下一时刻分类，不能等同于整条轨迹重捕获率。\n"
            "按验证Top-K选择模型并保留epoch0基线；如果最佳轮数为0，表示训练未改善该选择指标。\n"
            "下一步用固定模型进行单目标闭环；N-Net未接受缺测训练，该限制需与结果一起说明。\n")
    (output / "training_summary.txt").write_text(text, encoding="utf-8-sig")
    (OUTPUT_ROOT / "latest_run.txt").write_text(output.name, encoding="utf-8")
    print(text, flush=True)
    return output


if __name__ == "__main__":
    main()
