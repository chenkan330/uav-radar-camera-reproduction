"""第7步：训练教学版N-Net，记录训练曲线，默认按验证Top-1保存最佳模型。"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 复用第6步的网络定义，导入时不会执行第6步的前向检查程序。
from step06_build_nnet import D_MODEL, DROPOUT, MODEL_NOTE, N_HEAD, NUM_LAYERS, NNet


# 先用这些默认参数学习训练过程，也可以通过运行命令调整。
EPOCHS = 30                # 一轮epoch表示把训练集完整学习一遍。
BATCH_SIZE = 128           # 每次参数更新使用128个样本。
LEARNING_RATE = 0.0003     # 控制每次参数更新的步长。
POSITION_WEIGHT = 0.01     # 总损失 = 波束交叉熵 + 此权重 × 坐标均方误差。
SEED = 42
TOP_K = 3                 # 检查真实波束是否落在得分最高的3个候选内。
SELECTION_METRIC = "top1"  # top1选单波束准确率最高；topk选候选命中率最高；loss保留旧规则。
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "step05"
OUTPUT_ROOT = BASE_DIR / "outputs" / "step07"
STATUS = "trained_on_teaching_synthetic_data"
METRIC_KEYS = ("total_loss", "beam_loss", "position_mse_m2", "top1_accuracy", "topk_accuracy", "position_rmse_m")
SELECTION_LABELS = {
    "top1": "验证Top-1最高，同分时波束交叉熵最低",
    "topk": "验证Top-K最高，同分时波束交叉熵最低",
    "loss": "验证总损失最低，同分时Top-1最高",
}


def selection_key(metrics, selection_metric):
    """统一使用越大的排序键越好；模型选择只使用验证指标。"""
    if selection_metric == "top1":
        return metrics["top1_accuracy"], -metrics["beam_loss"]
    if selection_metric == "topk":
        return metrics["topk_accuracy"], -metrics["beam_loss"]
    if selection_metric == "loss":
        return -metrics["total_loss"], metrics["top1_accuracy"]
    raise ValueError(f"未知模型选择指标：{selection_metric}")


def read_dataset(path):
    """读取角距输入和两种监督标签，保留轨迹编号以检查训练/验证是否重叠。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，请先运行 step05_generate_dataset.py。")
    with np.load(path) as data:
        x = data["x_history_normalized"].copy()
        beam = data["y_future_beams"].copy()
        position = data["y_current_position_m"].copy()
        observed = data["current_observed_position_m"].copy()
        trajectory_ids = np.unique(data["trajectory_id"])
        metadata = {key: data[key].copy() for key in (
            "history_size", "future_size", "num_classes", "input_mean", "input_std",
            "base_station_m", "input_fields", "dataset_note",
        )}
    history, future, classes = (int(metadata[key]) for key in ("history_size", "future_size", "num_classes"))
    if not len(x) or x.shape != (len(x), history, 3) or beam.shape != (len(x), future):
        raise ValueError(f"{path.name}的历史输入或未来标签形状不正确。")
    if position.shape != (len(x), 3) or observed.shape != position.shape:
        raise ValueError(f"{path.name}的当前坐标形状不正确。")
    if not all(np.isfinite(array).all() for array in (x, position, observed)):
        raise ValueError(f"{path.name}中存在非有限数值。")
    if not np.issubdtype(beam.dtype, np.integer) or beam.min() < 1 or beam.max() >= classes:
        raise ValueError(f"{path.name}的波束标签必须是1到{classes - 1}之间的整数。")
    if metadata["input_fields"].tolist() != ["measured_range_m", "measured_azimuth_rad", "measured_polar_rad"]:
        raise ValueError("输入应依次为距离、方位角、极角，角度使用弧度。")
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(beam).long(),
                            torch.from_numpy(position).float())
    baseline_rmse = float(np.sqrt(np.mean(np.sum((observed.astype(np.float64) - position)**2, axis=1))))
    return dataset, metadata, trajectory_ids, baseline_rmse


def run_epoch(model, loader, device, position_weight, top_k, optimizer=None):
    """有优化器时更新参数；验证时只计算指标。每项损失按样本数加权汇总。"""
    training = optimizer is not None
    model.train(training)
    beam_sum, position_mse_sum, distance_squared_sum = 0.0, 0.0, 0.0
    correct_top1, correct_topk, sample_count, future_count = 0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for x, target_beam, target_position in loader:
            x, target_beam, target_position = (value.to(device, non_blocking=True) for value in (x, target_beam, target_position))
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(x)  # 真实标签不传入网络，只在下面计算误差。
            logits = outputs["beam_logits"]

            # 一、波束损失：把每个未来时刻看作一次多类别分类。
            # CrossEntropyLoss内部包含log_softmax，不能先对logits做softmax再传入。
            beam_loss = nn.functional.cross_entropy(logits.reshape(-1, model.num_classes), target_beam.reshape(-1))
            # 二、定位损失：当前预测坐标与真实坐标的逐分量均方误差，单位为平方米。
            position_loss = nn.functional.mse_loss(outputs["current_position_m"], target_position)
            loss = beam_loss + position_weight * position_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("损失出现非有限值，请检查数据或降低学习率。")
            if training:
                loss.backward()   # 反向传播：计算各参数应如何改变。
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()  # 根据梯度真正更新网络参数。

            # 三、指标不参与梯度计算。Top-K统计每个未来时刻，而非整条轨迹存活率。
            with torch.no_grad():
                candidates = logits.topk(top_k, dim=-1).indices
                correct_top1 += int((candidates[..., 0] == target_beam).sum())
                correct_topk += int((candidates == target_beam.unsqueeze(-1)).any(dim=-1).sum())
                distance_squared_sum += float(((outputs["current_position_m"] - target_position)**2).sum())
                count = len(x)
                beam_sum += float(beam_loss) * count
                position_mse_sum += float(position_loss) * count
                sample_count += count
                future_count += target_beam.numel()
    beam_average = beam_sum / sample_count
    position_average = position_mse_sum / sample_count
    return {
        "total_loss": beam_average + position_weight * position_average,
        "beam_loss": beam_average, "position_mse_m2": position_average,
        "top1_accuracy": correct_top1 / future_count, "topk_accuracy": correct_topk / future_count,
        # 这里按三维距离计算RMSE，等于sqrt(3 × 逐分量MSE)。
        "position_rmse_m": math.sqrt(distance_squared_sum / sample_count),
    }


def make_log_row(epoch, train, validation, seconds):
    return {"epoch": epoch, **{f"train_{key}": train[key] for key in METRIC_KEYS},
            **{f"val_{key}": validation[key] for key in METRIC_KEYS}, "epoch_seconds": seconds}


def save_curves(rows, best_epoch, train_baseline, validation_baseline, top_k, output_dir,
                selection_metric=SELECTION_METRIC):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    panels = [("total_loss", "总损失，越低越好", "损失", 1),
              ("top1_accuracy", "未来波束Top-1准确率", "准确率（%）", 100),
              ("topk_accuracy", f"未来波束Top-{top_k}命中率", "命中率（%）", 100),
              ("position_rmse_m", "当前三维定位RMSE，越低越好", "RMSE（米）", 1)]
    for axis, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        axis.plot(epochs, [row[f"train_{metric}"] * scale for row in rows], label="训练集", color="#2254B1")
        axis.plot(epochs, [row[f"val_{metric}"] * scale for row in rows], label="验证集", color="#C2762C")
        axis.axvline(best_epoch, color="gray", linestyle=":", linewidth=0.9)
        axis.set(title=title, xlabel="训练轮数（0表示训练前）", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    axes[1, 1].axhline(train_baseline, color="#2254B1", linestyle="--", linewidth=0.8, label="训练集原始观测")
    axes[1, 1].axhline(validation_baseline, color="#C2762C", linestyle="--", linewidth=0.8, label="验证集原始观测")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"第7步教学训练：共{epochs[-1]}轮，按{selection_metric}选择第{best_epoch}轮", fontsize=13)
    fig.text(0.5, 0.025, "简化N-Net与教学合成数据；波束指标按未来时刻统计；虚线为原始观测定位误差", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--position-weight", type=float, default=POSITION_WEIGHT)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="可选的单次训练时间上限；完成当前轮后保存并结束。")
    parser.add_argument("--selection-metric", choices=("top1", "topk", "loss"), default=SELECTION_METRIC)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("训练轮数、批量大小和学习率必须大于0。")
    if not np.isfinite(args.position_weight) or args.position_weight < 0:
        parser.error("定位损失权重必须是大于或等于0的有限数值。")
    if args.d_model < 2 or args.d_model % 2 or args.d_model % N_HEAD or args.num_layers < 1:
        parser.error("特征维度应为正偶数且能被注意力头数整除，层数至少为1。")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("Dropout必须在[0,1)范围内。")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("权重衰减必须为非负有限数。")
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("训练时间上限必须是正有限数。")
    if not 0 <= args.seed < 2**32:
        parser.error("随机种子应在0到2**32-1之间。")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用CUDA设备，可使用 --device cpu。")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # 四、读取训练/验证集，检查同一轨迹没有同时进入两个集合。
    train_set, metadata, train_ids, train_baseline = read_dataset(DATA_DIR / "train.npz")
    validation_set, validation_metadata, validation_ids, validation_baseline = read_dataset(DATA_DIR / "validation.npz")
    if np.intersect1d(train_ids, validation_ids).size:
        raise ValueError("训练集与验证集包含相同轨迹，请重新检查第5步的数据划分。")
    if any(not np.array_equal(value, validation_metadata[key]) for key, value in metadata.items()):
        raise ValueError("训练集与验证集的特征、标准化参数或模型相关配置不一致。")
    model_kwargs = {
        "history_size": int(metadata["history_size"]), "future_size": int(metadata["future_size"]),
        "num_classes": int(metadata["num_classes"]), "input_mean": metadata["input_mean"].tolist(),
        "input_std": metadata["input_std"].tolist(), "base_station_m": metadata["base_station_m"].tolist(),
        "d_model": args.d_model, "n_head": N_HEAD, "num_layers": args.num_layers, "dropout": args.dropout,
    }
    model = NNet(**model_kwargs).to(device)
    top_k = min(TOP_K, model.num_classes - 1)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_options)
    train_check_loader = DataLoader(train_set, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 每次运行创建一个新文件夹，保留以前训练的模型、曲线和日志。
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output_dir = OUTPUT_ROOT / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    best_path = output_dir / "best_nnet.pth"
    source_paths = [DATA_DIR / "train.npz", DATA_DIR / "validation.npz",
                    BASE_DIR / "step06_build_nnet.py", Path(__file__).resolve()]
    config = {
        "model_status": STATUS, "model_note": MODEL_NOTE, "dataset_note": str(metadata["dataset_note"]),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "position_weight": args.position_weight, "seed": args.seed, "top_k": top_k,
        "weight_decay": args.weight_decay, "max_seconds": args.max_seconds,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cpu_threads": torch.get_num_threads(), "model_kwargs": model_kwargs,
        "train_samples": len(train_set), "validation_samples": len(validation_set),
        "train_trajectory_ids": train_ids.tolist(), "validation_trajectory_ids": validation_ids.tolist(),
        "validation_observation_rmse_m": validation_baseline,
        "loss_definition": "cross_entropy_over_future_steps + position_weight * mean_squared_error_over_xyz",
        "selection_metric": args.selection_metric,
        "selection_rule": SELECTION_LABELS[args.selection_metric],
        "train_metrics_source": "minibatch_outputs_during_parameter_updates; epoch_0_is_eval_mode",
        "source_sha256": {str(path.relative_to(BASE_DIR)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "versions": {"python": platform.python_version(), "numpy": str(np.__version__), "torch": str(torch.__version__),
                     "matplotlib": str(matplotlib.__version__)},
        "notes": ["使用第6步的教学版网络和第5步的合成数据，训练损失尚未对齐完整论文。",
                  "只读取训练集和验证集；测试集留待后续独立评估。",
                  "Top-1和Top-K按所有样本的未来时刻统计，不是轨迹持续跟踪成功率。"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"设备：{device}；训练样本：{len(train_set)}；验证样本：{len(validation_set)}。", flush=True)
    print(f"共训练{args.epochs}轮，每批{args.batch_size}个样本；定位损失权重{args.position_weight}。", flush=True)
    print(f"最佳模型选择标准：{SELECTION_LABELS[args.selection_metric]}。", flush=True)
    print(f"本次输出文件夹：{output_dir}", flush=True)
    start_time = time.perf_counter()

    # 五、先记录训练前的表现，作为曲线中epoch=0的参考点。
    initial_train = run_epoch(model, train_check_loader, device, args.position_weight, top_k)
    initial_validation = run_epoch(model, validation_loader, device, args.position_weight, top_k)
    rows = [make_log_row(0, initial_train, initial_validation, 0.0)]
    print(f"训练前：验证损失{initial_validation['total_loss']:.4f}，验证Top-1 {initial_validation['top1_accuracy']:.2%}，"
          f"验证定位RMSE {initial_validation['position_rmse_m']:.3f}米。", flush=True)

    # 六、逐轮学习：训练时更新参数，验证时关闭梯度；按指定验证指标选择模型。
    best_key, best_epoch, best_metrics = None, 0, None
    with (output_dir / "training_log.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerow(rows[0])
        stream.flush()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train = run_epoch(model, train_loader, device, args.position_weight, top_k, optimizer)
            validation = run_epoch(model, validation_loader, device, args.position_weight, top_k)
            row = make_log_row(epoch, train, validation, time.perf_counter() - epoch_start)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            candidate_key = selection_key(validation, args.selection_metric)
            improved = best_key is None or candidate_key > best_key
            if improved:
                best_key, best_epoch, best_metrics = candidate_key, epoch, validation.copy()
                torch.save({
                    "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                    "model_kwargs": model_kwargs, "epoch": epoch, "validation_metrics": validation,
                    "training_config": config,
                }, best_path)
            print(f"Epoch {epoch:02d}/{args.epochs} | 训练损失 {train['total_loss']:.4f} | 验证损失 {validation['total_loss']:.4f} | "
                  f"验证Top-1 {validation['top1_accuracy']:.2%} | Top-{top_k} {validation['topk_accuracy']:.2%} | "
                  f"定位RMSE {validation['position_rmse_m']:.3f}米" + (" | 已保存最佳模型" if improved else ""), flush=True)
            if args.max_seconds is not None and time.perf_counter() - start_time >= args.max_seconds:
                print("已达到本次时间上限，保留最佳模型并结束训练。", flush=True)
                break

    # 七、重新加载保存的最佳权重，检查文件能读取并计算验证集指标。
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    best_model = NNet(**checkpoint["model_kwargs"]).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_validation = run_epoch(best_model, validation_loader, device, args.position_weight, top_k)
    if any(not np.isclose(best_validation[key], best_metrics[key], rtol=1e-5, atol=1e-6)
           for key in METRIC_KEYS):
        raise RuntimeError("重新读取模型后的验证指标与保存时不一致。")
    save_curves(rows, best_epoch, train_baseline, validation_baseline, top_k, output_dir,
                args.selection_metric)
    elapsed = time.perf_counter() - start_time
    summary = {
        "model_status": STATUS, "epochs_completed": rows[-1]["epoch"], "best_epoch": best_epoch,
        "elapsed_seconds": elapsed, "initial_validation": initial_validation,
        "best_validation": best_validation, "validation_observation_rmse_m": validation_baseline,
        "best_checkpoint": best_path.name, "top_k": top_k, "output_directory": str(output_dir),
        "selection_metric": args.selection_metric, "selection_rule": SELECTION_LABELS[args.selection_metric],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    description = (
        f"第7步：教学版N-Net训练完成，共{rows[-1]['epoch']}轮。\n"
        f"最佳模型来自第{best_epoch}轮，选择依据是{SELECTION_LABELS[args.selection_metric]}。\n"
        f"训练前验证损失：{initial_validation['total_loss']:.4f}\n最佳模型验证损失：{best_validation['total_loss']:.4f}\n"
        f"最佳模型验证Top-1准确率：{best_validation['top1_accuracy']:.2%}\n"
        f"最佳模型验证Top-{top_k}命中率：{best_validation['topk_accuracy']:.2%}\n"
        f"验证集原始观测定位RMSE：{validation_baseline:.3f}米\n"
        f"最佳模型验证定位RMSE：{best_validation['position_rmse_m']:.3f}米\n"
        f"本次总耗时：{elapsed:.1f}秒\n\n"
        "Top-1表示得分最高的波束是否正确，Top-K表示正确波束是否在前K个候选中。\n"
        "以上波束指标按未来时刻分类统计，不能等同于整条轨迹的持续跟踪成功率。\n"
        "定位RMSE按预测与真值的三维欧氏距离计算，单位为米。\n"
        "当前训练使用教学合成数据和简化网络；完整论文的损失与闭环跟踪仍需后续对齐。\n"
        "本步骤使用验证集选择模型，最终效果应在独立测试集上评价。\n"
    )
    (output_dir / "training_summary.txt").write_text(description, encoding="utf-8-sig")
    (OUTPUT_ROOT / "latest_run.txt").write_text(output_dir.name, encoding="utf-8")
    print(f"\n训练完成。最佳模型来自第{best_epoch}轮，验证Top-1 {best_validation['top1_accuracy']:.2%}，"
          f"定位RMSE {best_validation['position_rmse_m']:.3f}米。", flush=True)
    print(f"最佳模型：{best_path}", flush=True)
    print(f"训练曲线：{output_dir / 'training_curves.png'}", flush=True)
    print("先看 training_summary.txt，再看曲线；training_log.csv可以用Excel打开。", flush=True)


if __name__ == "__main__":
    main()
