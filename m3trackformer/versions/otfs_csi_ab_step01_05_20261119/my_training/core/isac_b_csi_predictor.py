"""B 组第 4 步，在原始飞行记录分区上训练和评估下一帧 CSI 预测器。"""
import argparse
import hashlib
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import BASE, finish_run, new_run, read_json, seed_all, write_json, write_rows
from .isac_b_csi_samples import PROTOCOL
from .isac_multi_target import BUDGET, RADAR_CONFIG
from .versions import save_version


VERSION = "isac_b_step04_csi_predictor"
DATA_VERSION = "isac_b_step04_csi_training_data"
SPLITS = ("train", "validation", "test")
SEED = 20261119
ARRAY_SIDE = 32
PARAMETER_SCALE = (.035, .035, .15)
LABELS = {
    "last_observation": "末次量测",
    "constant_velocity": "匀速外推",
    "learned_measurement": "小模型 · 仅角距",
    "learned_iq": "小模型 · 角距与 IQ",
    "pretrained_iq": "预训练骨干 · 角距与 IQ",
}


def csi_parameters(channel):
    """从阵元域标签读取两轴相位斜率和总振幅，消去公共相位。"""
    rows = channel.reshape(-1, ARRAY_SIDE, ARRAY_SIDE)
    ux = np.angle(np.sum(rows[:, :, 1:]*rows[:, :, :-1].conj(), axis=(1, 2)))/np.pi
    vy = np.angle(np.sum(rows[:, 1:, :]*rows[:, :-1, :].conj(), axis=(1, 2)))/np.pi
    return np.column_stack((ux, vy, np.log(np.linalg.norm(channel, axis=1)))).astype(np.float32)


def position_parameters(position):
    distance = np.maximum(np.linalg.norm(position, axis=-1), 1.)
    return np.column_stack((position[:, :2]/distance[:, None],
                            np.log(RADAR_CONFIG.reference_range_m/distance))).astype(np.float32)


def causal_reference(positions, valid):
    """两项基线使用与学习模型相同的已接收角距历史。"""
    count = valid.sum(axis=1, keepdims=True)
    if not count.min():
        raise ValueError("预测样本缺少全部历史角距观测")
    frame = np.arange(positions.shape[1])
    last_index = np.where(valid, frame, -1).max(axis=1)
    last = positions[np.arange(len(positions)), last_index]
    centre = (valid*frame).sum(axis=1, keepdims=True)/count
    shifted = (frame-centre)*valid
    denom = (shifted**2).sum(axis=1, keepdims=True)
    velocity = (positions*shifted[..., None]).sum(axis=1)/np.maximum(denom, 1.)
    average = (positions*valid[..., None]).sum(axis=1)/count
    predicted = average + velocity*(len(frame)-centre)
    return position_parameters(last), position_parameters(predicted)


class ResidualPredictor(nn.Module):
    """共享逐帧特征投影，在匀速参考上预测下行信道参数的残差。"""
    def __init__(self, frame_features, history, pretrained_dir=None):
        super().__init__()
        if pretrained_dir is None:
            self.backbone = None
            self.frame_encoder = nn.Sequential(nn.Linear(frame_features, 48), nn.GELU())
            head_features = history*48
        else:
            try:
                from transformers import AutoModel
            except ImportError as error:
                raise RuntimeError("本地预训练分支需要已安装的 transformers") from error
            if not (pretrained_dir/"config.json").is_file():
                raise FileNotFoundError(f"本地预训练模型目录缺少 config.json：{pretrained_dir}")
            self.backbone = AutoModel.from_pretrained(str(pretrained_dir), local_files_only=True)
            self.backbone.requires_grad_(False).eval()
            head_features = self.backbone.config.hidden_size
            self.frame_encoder = nn.Linear(frame_features, head_features)
        self.head = nn.Sequential(nn.Linear(head_features+3, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("residual_scale", torch.tensor(PARAMETER_SCALE))

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def forward(self, history, reference):
        tokens = self.frame_encoder(history)
        if self.backbone is None:
            encoded = tokens.flatten(start_dim=1)
        else:
            # 冻结权重，仍保留对数值输入投影的梯度。
            encoded = self.backbone(inputs_embeds=tokens.to(next(self.backbone.parameters()).dtype),
                                    use_cache=False).last_hidden_state[:, -1].float()
        residual = self.head(torch.cat((encoded, reference), dim=-1))
        return reference + self.residual_scale*residual


def load_dataset(root):
    groups = {}
    data = {}
    for split in SPLITS:
        with np.load(root/f"{split}.npz", allow_pickle=False) as saved:
            data[split] = {name: saved[name] for name in saved.files}
        item = data[split]
        n = len(item["csi"])
        if not n or item["csi"].shape != (n, ARRAY_SIDE**2) or item["positions"].shape != (n, 5, 3):
            raise ValueError(f"{split} 样本的 CSI 或历史位置尺寸无效")
        groups[split] = set(map(int, item["session_group"]))
        if "history_frames" in item:
            if (np.diff(item["history_frames"], axis=1) != 1).any() or (
                    item["history_frames"][:, -1]+1 != item["label_frame"]).any():
                raise ValueError(f"{split} 的观测与未来标签时间不一致")
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("原始飞行组跨分区重复")
    return data, {name: sorted(ids) for name, ids in groups.items()}


def model_inputs(data, use_iq):
    measured = np.concatenate((data["positions"]/RADAR_CONFIG.reference_range_m,
                               data["valid"][..., None]), axis=-1).astype(np.float32)
    return np.concatenate((measured, data["features"]), axis=-1) if use_iq else measured


def fit_predictor(data, references, output, method, epochs, device, pretrained_dir=None):
    use_iq = method != "learned_measurement"
    train_input = model_inputs(data["train"], use_iq)
    mean = train_input.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = train_input.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale[scale < 1.e-5] = 1.
    seed_all(SEED)
    model = ResidualPredictor(train_input.shape[-1], train_input.shape[1], pretrained_dir).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8e-4, weight_decay=.02)

    def prepare(split):
        x = (model_inputs(data[split], use_iq)-mean)/scale
        return tuple(torch.from_numpy(a).to(device) for a in
                     (x, references[split]["constant_velocity"], csi_parameters(data[split]["csi"])))

    train, validation = prepare("train"), prepare("validation")
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    best_loss = np.inf
    best_state = None
    trace = []
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.
        for indices in torch.randperm(len(train[0]), generator=rng).split(96):
            x, reference, label = (array[indices.to(device)] for array in train)
            residual = (model(x, reference)-label)/model.residual_scale
            loss = F.smooth_l1_loss(residual, torch.zeros_like(residual))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            train_loss += float(loss.detach())*len(indices)
        model.eval()
        with torch.inference_mode():
            x, reference, label = validation
            residual = (model(x, reference)-label)/model.residual_scale
            val_loss = float(F.smooth_l1_loss(residual, torch.zeros_like(residual)))
        trace.append({"method": method, "epoch": epoch, "train_loss": train_loss/len(train[0]),
                      "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                          if name in trainable_names}
    if best_state is None:
        raise RuntimeError(f"{method} 的验证误差无有效数值")

    model.load_state_dict(best_state, strict=False)
    torch.save({"model_state_dict": best_state, "input_mean": torch.from_numpy(mean),
                "input_std": torch.from_numpy(scale), "use_iq": use_iq, "history": train_input.shape[1],
                "pretrained_dir": str(pretrained_dir) if pretrained_dir is not None else None,
                "best_epoch": best_epoch}, output/f"{method}.pt")
    print(f"{LABELS[method]}：验证选择第 {best_epoch} 轮，损失 {best_loss:.5f}。", flush=True)
    result = {}
    with torch.inference_mode():
        for split in ("validation", "test"):
            x, reference, _ = prepare(split)
            result[split] = torch.cat([model(a, b).cpu() for a, b in zip(x.split(96), reference.split(96))]).numpy()
    return result, trace, {"best_epoch": best_epoch, "validation_loss": best_loss,
                           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                           "total_parameters": sum(p.numel() for p in model.parameters())}


def make_channel(parameters):
    """以正实数路径振幅为公共相位规范，生成 32×32 阵元域复 CSI。"""
    uv = parameters[:, :2].astype(np.float64)
    uv /= np.maximum(1., np.linalg.norm(uv, axis=1, keepdims=True))
    y, x = np.indices((ARRAY_SIDE, ARRAY_SIDE))
    phase = np.pi*(uv[:, 0, None]*x.ravel() + uv[:, 1, None]*y.ravel())
    amplitude = np.exp(parameters[:, 2].astype(np.float64))
    return amplitude[:, None]*np.exp(1j*phase)/ARRAY_SIDE


def beam_powers(channel):
    """用二维 FFT 批量计算与第 3 步 DFT 码本相同的功率。"""
    axis = np.arange(ARRAY_SIDE)
    offset = np.exp(1j*np.pi*axis*(-1.+1./ARRAY_SIDE))
    modulated = channel.conj().reshape(-1, ARRAY_SIDE, ARRAY_SIDE)*offset[:, None]*offset[None, :]
    responses = np.fft.ifft2(modulated, axes=(1, 2))*ARRAY_SIDE
    return (np.abs(responses)**2).reshape(-1, ARRAY_SIDE**2)


def evaluate_predictions(predictions, truth):
    raw = truth["csi"].astype(np.complex128)
    power = beam_powers(raw)
    oracle = power.argmax(axis=1)
    best_gain = power.max(axis=1)
    ref_energy = np.sum(np.abs(raw)**2, axis=1)
    result = {}
    rows = []
    for method, parameters in predictions.items():
        channel = make_channel(parameters)
        dot = np.abs(np.sum(channel.conj()*raw, axis=1))
        pred_energy = np.sum(np.abs(channel)**2, axis=1)
        nmse = np.maximum((pred_energy+ref_energy-2*dot)/ref_energy, 0.)
        correlation = np.minimum(dot**2/(pred_energy*ref_energy), 1.)
        selected = np.argsort(-beam_powers(channel), axis=1, kind="stable")[:, :BUDGET]
        actual_gain = np.take_along_axis(power, selected, axis=1).max(axis=1)
        loss_db = 10*np.log10(best_gain/np.maximum(actual_gain, 1e-30))
        hit = (selected == oracle[:, None]).any(axis=1)

        def aggregate(mask):
            return {"windows": int(mask.sum()), "phase_aligned_nmse": float(nmse[mask].mean()),
                    "channel_coherence": float(correlation[mask].mean()),
                    "mean_beam_gain_loss_db": float(loss_db[mask].mean()),
                    "p95_beam_gain_loss_db": float(np.percentile(loss_db[mask], 95)),
                    "oracle_beam_in_top12_rate": float(hit[mask].mean())}

        groups = truth["session_group"]
        result[method] = aggregate(np.ones(len(raw), dtype=bool))
        result[method]["by_flight_group"] = {str(group): aggregate(groups == group) for group in np.unique(groups)}
        rows.extend({"method": method, "trajectory_id": int(truth["trajectory_id"][i]),
                     "session_group": int(groups[i]), "label_frame": int(truth["label_frame"][i]),
                     "phase_aligned_nmse": float(nmse[i]), "actual_best_selected_gain": float(actual_gain[i]),
                     "relative_gain_loss_db": float(loss_db[i]), "oracle_beam_in_top12": bool(hit[i])}
                    for i in range(len(raw)))
    return result, rows


def draw_metrics(output, trace, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
        for method in dict.fromkeys(row["method"] for row in trace):
            rows = [row for row in trace if row["method"] == method]
            axes[0].plot([row["epoch"] for row in rows], [row["validation_loss"] for row in rows],
                         label=LABELS[method])
        axes[0].set(xlabel="训练轮数", ylabel="验证损失", title="按飞行组划分的验证集")
        axes[0].legend()
        methods = list(metrics)
        bars = axes[1].barh([LABELS[name] for name in methods],
                            [metrics[name]["phase_aligned_nmse"]*1000 for name in methods], color="#277c8f")
        axes[1].bar_label(bars, fmt="%.3f", padding=3)
        axes[1].margins(x=.17)
        axes[1].set(xlabel=r"公共相位对齐后 NMSE（$\times10^{-3}$）", title="测试集 · 下一帧 CSI 误差")
        axes[1].invert_yaxis()
        for ax in axes:
            ax.grid(axis="x", alpha=.24)
            ax.set_axisbelow(True)
        fig.savefig(output/"csi_prediction_metrics.png", dpi=165)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", type=Path, help="重用已保存的第 4 步数据目录")
    parser.add_argument("--collect-only", action="store_true", help="只生成第 4 步三个数据分区")
    parser.add_argument("--epochs", type=int, default=50, help="每个学习模型的固定训练轮数")
    parser.add_argument("--pretrained-model-dir", type=Path, help="可选的本地 Hugging Face 因果语言模型目录")
    args = parser.parse_args()
    if args.epochs < 1 or (args.dataset_run is not None and args.collect_only):
        parser.error("训练轮数须为正整数，重用数据时无需指定 --collect-only")
    torch.set_num_threads(4)
    protocol = read_json(PROTOCOL)
    if args.dataset_run is None:
        from .isac_b_csi_training_data import collect_dataset

        dataset = new_run(BASE/"outputs"/DATA_VERSION)
        current = read_json(BASE/"outputs/current_journal_experiment.json")
        frozen = read_json(BASE/"outputs/current_identity_closed_loop.json")
        dataset_meta = collect_dataset(dataset, current, frozen, protocol)
        copyfile(PROTOCOL, dataset/"step01_protocol.json")
        write_json(dataset/"summary.json", dataset_meta)
        finish_run(dataset)
        print(f"数据目录：{dataset.resolve()}", flush=True)
        if args.collect_only:
            return
    else:
        dataset = args.dataset_run.resolve()
    dataset_protocol = dataset/"step01_protocol.json"
    if dataset_protocol.is_file() and dataset_protocol.read_bytes() != PROTOCOL.read_bytes():
        raise ValueError("保存的数据与当前第 1 步协议不一致")
    data, flight_groups = load_dataset(dataset)
    references = {}
    for split, item in data.items():
        last, cv = causal_reference(item["positions"], item["valid"])
        references[split] = {"last_observation": last, "constant_velocity": cv}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = new_run(BASE/"outputs"/VERSION)
    fit_log, model_meta = [], {}
    methods = {"learned_measurement": None, "learned_iq": None}
    if args.pretrained_model_dir is not None:
        methods["pretrained_iq"] = args.pretrained_model_dir.resolve()
    predictions = {split: references[split].copy() for split in ("validation", "test")}
    for method, pretrained_dir in methods.items():
        estimates, trace, model_meta[method] = fit_predictor(data, references, output, method, args.epochs,
                                                             device, pretrained_dir)
        for split, estimate in estimates.items():
            predictions[split][method] = estimate
        fit_log.extend(trace)

    metrics = {}
    for split, estimates in predictions.items():
        metrics[split], rows = evaluate_predictions(estimates, data[split])
        write_rows(output/f"{split}_metrics.csv", rows)
    test = data["test"]
    np.savez_compressed(output/"test_predicted_iq_csi.npz",
        h=make_channel(predictions["test"]["learned_iq"]).astype(np.complex64),
        trajectory_id=test["trajectory_id"], session_group=test["session_group"], label_frame=test["label_frame"])
    write_rows(output/"training_log.csv", fit_log)
    draw_metrics(output, fit_log, metrics["test"])
    if (dataset/"summary.json").is_file():
        copyfile(dataset/"summary.json", output/"dataset_summary.json")
    copyfile(PROTOCOL, output/"step01_protocol.json")
    source_keys = {f"{split}_dataset": str((dataset/f"{split}.npz").resolve()) for split in SPLITS}
    write_json(output/"summary.json", {
        "version": VERSION, "dataset_run": str(dataset.resolve()), **source_keys,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "original_flight_groups": flight_groups, "window_counts": {split: len(data[split]["csi"]) for split in SPLITS},
        "selection": "只以验证集参数回归损失选择各学习模型的检查点",
        "scope": "固定阵列单径单终端仿真中的下一帧 CSI 离线预测",
        "phase": "去公共相位的阵元域复 CSI；增益评估使用保存的未来真实标签",
        "saved_test_prediction": "test_predicted_iq_csi.npz 保存角距与 IQ 小模型的下一帧阵元域复 CSI",
        "test_role": "其原始飞行组与本次训练、验证不重叠；现有测试分区曾用于历史实验",
        "inference_device": str(device), "seed": SEED, "epochs": args.epochs,
        "models": model_meta, "metrics": metrics,
    })
    archive = save_version(f"{VERSION}_{output.name[4:]}", output, "B 组第 4 步未来 CSI 预测。",
                           asset_keys=tuple(source_keys))
    finish_run(output)
    print(f"结果目录：{output.resolve()}\n归档目录：{archive.resolve()}", flush=True)
