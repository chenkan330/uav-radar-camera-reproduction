"""第6步：搭建教学版N-Net，读取一批历史测量，检查网络的两个输出。"""

import csv
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
from torch import nn


# 先用较小的网络理解数据流，这些参数可以在后续训练时调整。
D_MODEL = 64              # 把每个时刻的3个测量数值映射成64维特征。
N_HEAD = 4                # 每个Transformer层使用4个注意力头。
NUM_LAYERS = 2            # 堆叠2个Transformer编码层。
DROPOUT = 0.1
BATCH_SIZE = 8            # 本步骤只取8个样本检查前向计算。
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "step05" / "train.npz"
OUTPUT_DIR = BASE_DIR / "outputs" / "step06"
MODEL_NOTE = "teaching_measurement_only_transformer_with_future_tokens_and_current_position_residual"
MODEL_STATUS = "untrained_forward_check_only"


class NNet(nn.Module):
    """教学版：历史角距测量 -> 未来波束分数 + 当前坐标估计。

    forward只接收历史测量，不接收真实坐标或未来波束标签。
    与作者完整模型相比，本版先省去历史波束嵌入及缺测掩码。
    """

    def __init__(self, history_size, future_size, num_classes, input_mean, input_std,
                 base_station_m, d_model=D_MODEL, n_head=N_HEAD,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        if min(history_size, future_size, num_layers, n_head) < 1 or num_classes < 2:
            raise ValueError("窗口长度、层数、注意力头数应为正，类别数应包含0及至少一个有效波束。")
        if d_model < 2 or d_model % 2 or d_model % n_head:
            raise ValueError("特征维度应为正偶数，且能被注意力头数整除。")
        self.history_size = history_size
        self.future_size = future_size
        self.num_classes = num_classes
        for name, value in [("input_mean", input_mean), ("input_std", input_std),
                            ("base_station_m", base_station_m)]:
            tensor = torch.as_tensor(value, dtype=torch.float32).clone()
            if tensor.shape != (3,) or not torch.isfinite(tensor).all():
                raise ValueError(f"{name}必须包含3个有限数值。")
            self.register_buffer(name, tensor)
        if not torch.all(self.input_std > 0):
            raise ValueError("输入标准差必须大于0。")

        # 一、测量嵌入：每个历史时刻的[距离、方位角、极角]变成一条特征向量。
        self.measurement_embedding = nn.Linear(3, d_model)

        # 二、未来占位符：5条可学习的向量，表示等待预测的5个未来时刻。
        # 它们是模型参数，初始化时为小随机数，其中没有未来测量或真实标签。
        self.future_tokens = nn.Parameter(torch.empty(1, future_size, d_model))
        nn.init.normal_(self.future_tokens, std=0.02)

        # 三、时间位置编码：告诉网络哪条向量属于哪个时刻。
        length = history_size + future_size
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(1, length, d_model)
        encoding[0, :, 0::2] = torch.sin(position * scale)
        encoding[0, :, 1::2] = torch.cos(position * scale)
        self.register_buffer("time_encoding", encoding)
        self.embedding_dropout = nn.Dropout(dropout)

        # 四、Transformer编码层：让历史特征和未来占位符通过注意力交换信息。
        # 每一层分别创建，参数相互独立。当前所有历史点都有测量，无需缺测掩码。
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
                dropout=dropout, activation="relu", batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # 五、波束分支：每个未来时刻输出各个波束类别的分数，尚未做softmax。
        self.beam_head = nn.Linear(d_model, num_classes)
        self.register_buffer("padding_class_mask", torch.arange(num_classes) == 0)

        # 六、定位分支：预测当前测量坐标需要修正多少米。
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 3),
        )
        # 初始修正设为0，因此未训练时定位输出等于由角距测量计算的观测位置。
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

    def forward(self, x_history_normalized):
        # 输入形状为[一批样本数, 历史长度, 3]，本步骤默认是[8, 5, 3]。
        if x_history_normalized.ndim != 3 or x_history_normalized.shape[1:] != (self.history_size, 3):
            raise ValueError(f"输入形状应为[batch, {self.history_size}, 3]。")
        batch = x_history_normalized.shape[0]
        history_features = self.measurement_embedding(x_history_normalized)
        future_features = self.future_tokens.expand(batch, -1, -1)
        hidden = torch.cat([history_features, future_features], dim=1)
        hidden = self.embedding_dropout(hidden + self.time_encoding)
        for layer in self.encoder_layers:
            hidden = layer(hidden)

        # 取最后5条未来特征，得到[batch, 5, 类别数]。
        beam_logits = self.beam_head(hidden[:, self.history_size:])
        # 数据集中0是预留编号，不是真实波束，禁止选择它。
        beam_logits = beam_logits.masked_fill(self.padding_class_mask, torch.finfo(beam_logits.dtype).min)

        # 当前时刻对应历史的最后一点。将同一个输入还原为角距测量，再计算观测位置。
        current_measurement = x_history_normalized[:, -1] * self.input_std + self.input_mean
        distance, azimuth, polar = current_measurement.unbind(dim=-1)
        observed_position = torch.stack([
            distance * torch.sin(polar) * torch.cos(azimuth),
            distance * torch.sin(polar) * torch.sin(azimuth),
            distance * torch.cos(polar),
        ], dim=-1) + self.base_station_m
        correction = self.position_head(hidden[:, self.history_size - 1])
        return {
            "beam_logits": beam_logits,
            "position_correction_m": correction,
            "observed_position_m": observed_position,
            "current_position_m": observed_position + correction,
        }


def save_network_diagram(history_size, future_size, num_classes):
    """画出网络结构，方框中的B表示一批样本的数量。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axis = plt.subplots(figsize=(10.5, 8.0))
    axis.set(xlim=(0, 10), ylim=(-0.2, 10))
    axis.axis("off")

    def box(x, y, text, color="#E5EFF9", width=3.1, height=0.72):
        axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                     boxstyle="round,pad=0.08", facecolor=color, edgecolor="#4E6476"))
        axis.text(x, y, text, ha="center", va="center", fontsize=10)

    def arrow(start, end, color="#4E6476"):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                      color=color, linewidth=1.2))

    box(3.0, 9.1, f"历史测量输入\n[B, {history_size}, 3]")
    box(3.0, 7.8, f"测量嵌入 Linear\n[B, {history_size}, {D_MODEL}]")
    box(7.1, 7.8, f"可学习的未来占位符\n[B, {future_size}, {D_MODEL}]", "#FCEBD9")
    box(5.0, 6.4, f"拼接，并加入时间位置编码\n[B, {history_size + future_size}, {D_MODEL}]", width=4.2)
    box(5.0, 5.0, f"Transformer编码层 × {NUM_LAYERS}\n{N_HEAD}个注意力头", width=4.2)
    box(2.7, 3.5, f"未来{future_size}条特征 → 波束分支\n[B, {future_size}, {num_classes}]个分数", "#FCEBD9", width=3.8)
    box(7.1, 3.5, "当前历史特征 → 定位分支\n预测坐标修正量 [B, 3]", "#E0F0E8", width=3.8)
    box(2.7, 1.9, f"对每个未来时刻选择波束\n有效编号1～{num_classes - 1}", "#FCEBD9", width=3.8)
    box(7.1, 1.9, "观测坐标 + 预测修正量\n得到当前坐标 [B, 3]", "#E0F0E8", width=3.8)
    arrow((3, 8.65), (3, 8.24))
    arrow((3, 7.36), (4, 6.84))
    arrow((7.1, 7.36), (6, 6.84))
    arrow((5, 5.96), (5, 5.44))
    arrow((4, 4.56), (2.7, 3.94))
    arrow((6, 4.56), (7.1, 3.94))
    arrow((2.7, 3.06), (2.7, 2.34))
    arrow((7.1, 3.06), (7.1, 2.34))
    axis.text(5, 0.65, "观测坐标由历史输入的最后一组角距测量计算。\n未来占位符没有未来信息；真实标签只用于后续训练监督。",
              ha="center", va="center", fontsize=10)
    fig.suptitle("第6步：教学版N-Net结构与前向计算", fontsize=14, y=0.97)
    fig.text(0.5, 0.025, "本步骤的网络尚未训练，输出仅用于检查数据流。", ha="center", fontsize=10, color="#A35421")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "network_overview.png", dpi=160)
    plt.close(fig)


def main():
    # 一、只从训练集取一批数据。标签单独保留，不传入模型。
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到 {INPUT_FILE}，请先运行 step05_generate_dataset.py。")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE至少为1。")
    with np.load(INPUT_FILE) as data:
        x = data["x_history_normalized"][:BATCH_SIZE].copy()
        target_beams = data["y_future_beams"][:BATCH_SIZE].copy()
        target_position = data["y_current_position_m"][:BATCH_SIZE].copy()
        saved_observed = data["current_observed_position_m"][:BATCH_SIZE].copy()
        history_size, future_size, num_classes = (int(data[key]) for key in ["history_size", "future_size", "num_classes"])
        input_mean, input_std, base_station = (data[key].copy() for key in ["input_mean", "input_std", "base_station_m"])
        input_fields = data["input_fields"].copy()
    if len(x) == 0:
        raise ValueError("训练集没有样本，请检查第5步的输出。")
    expected_fields = ["measured_range_m", "measured_azimuth_rad", "measured_polar_rad"]
    if input_fields.tolist() != expected_fields:
        raise ValueError("输入字段顺序应为距离、方位角和极角，角度采用弧度。")

    # 二、随机初始化模型；自动选择当前环境中的GPU或CPU。
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNet(history_size, future_size, num_classes, input_mean, input_std, base_station).to(device)
    batch_input = torch.from_numpy(x).to(device)

    # 三、前向计算：检查网络能否正常接收输入并产生两个输出。
    # 本程序没有优化器和参数更新。eval()关闭dropout，便于观察同一个模型的输出。
    model.eval()
    with torch.no_grad():
        outputs = model(batch_input)
        probability = torch.softmax(outputs["beam_logits"], dim=-1)
        top_probability, top_beam = probability.topk(min(3, num_classes - 1), dim=-1)
    arrays = {key: value.cpu().numpy() for key, value in outputs.items()}
    top_beam, top_probability = top_beam.cpu().numpy(), top_probability.cpu().numpy()

    # 检查标准化还原和坐标转换是否与第5步的观测位置一致。
    np.testing.assert_allclose(arrays["observed_position_m"], saved_observed, atol=1e-4, rtol=1e-6)
    assert arrays["beam_logits"].shape == (len(x), future_size, num_classes)
    assert arrays["current_position_m"].shape == (len(x), 3)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert not np.any(top_beam == 0)

    # 四、保存结构和前向检查结果。随机波束选择不代表训练后的预测效果。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    structure = (f"第6步：教学版N-Net；状态：尚未训练。\n模型参数数量：{parameter_count:,}\n"
                 "输入：标准化后的历史距离、方位角和极角。\n"
                 "输出：未来波束类别分数，以及当前观测位置加上网络预测的坐标修正。\n"
                 "定位修正分支的末层初始化为0，因此初始定位输出等于观测位置。\n"
                 "与作者完整模型的区别：当前没有历史波束嵌入和缺测掩码，网络宽度及层数也更小。\n\n"
                 + str(model) + "\n")
    (OUTPUT_DIR / "model_structure.txt").write_text(structure, encoding="utf-8-sig")
    np.savez_compressed(
        OUTPUT_DIR / "forward_check.npz", **arrays, x_history_normalized=x,
        future_beam_labels=target_beams, true_current_position_m=target_position,
        top_beam=top_beam, top_probability=top_probability,
        model_status=MODEL_STATUS, model_note=MODEL_NOTE,
    )
    with (OUTPUT_DIR / "untrained_beams.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_status", "sample_id", "future_step", "true_beam", "random_model_top1_beam", "top1_probability"])
        for sample in range(len(x)):
            for future in range(future_size):
                writer.writerow([MODEL_STATUS, sample, future + 1, int(target_beams[sample, future]),
                                 int(top_beam[sample, future, 0]), float(top_probability[sample, future, 0])])
    with (OUTPUT_DIR / "current_position.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_status", "sample_id", "observed_x_m", "observed_y_m", "observed_z_m",
                         "predicted_dx_m", "predicted_dy_m", "predicted_dz_m", "output_x_m", "output_y_m", "output_z_m",
                         "true_x_m", "true_y_m", "true_z_m"])
        for sample in range(len(x)):
            writer.writerow([MODEL_STATUS, sample, *arrays["observed_position_m"][sample],
                             *arrays["position_correction_m"][sample], *arrays["current_position_m"][sample],
                             *target_position[sample]])
    config = {
        "model_status": MODEL_STATUS, "model_note": MODEL_NOTE, "seed": SEED,
        "history_size": history_size, "future_size": future_size, "num_classes": num_classes,
        "d_model": D_MODEL, "n_head": N_HEAD, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "parameter_count": parameter_count, "checked_batch_size": len(x), "device": str(device),
        "input_fields": input_fields.tolist(), "input_mean": input_mean.tolist(), "input_std": input_std.tolist(),
        "base_station_m": base_station.tolist(), "input_file": str(INPUT_FILE),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "matplotlib": matplotlib.__version__},
        "notes": ["尚未更新模型参数，随机波束输出不能用来判断算法效果。",
                  "每个未来时刻使用一个可学习占位符，不使用未来测量或真实标签。",
                  "当前版本对应前5步的简化数据，尚未包含完整论文模型的历史波束嵌入与缺测处理。"],
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_network_diagram(history_size, future_size, num_classes)

    print(f"计算设备：{device}，PyTorch版本：{torch.__version__}。")
    print(f"历史测量输入：{tuple(batch_input.shape)}。")
    print(f"未来波束分数：{arrays['beam_logits'].shape}；编号0已禁止选择。")
    print(f"当前坐标修正：{arrays['position_correction_m'].shape}。")
    print(f"当前坐标输出：{arrays['current_position_m'].shape}。")
    print(f"模型参数数量：{parameter_count:,}。")
    print("前向检查通过。网络尚未训练，随机波束输出不代表预测效果。")
    print("定位分支的初始修正为0，因此当前坐标输出等于由测量计算的观测位置。")
    print(f"输出文件夹 {OUTPUT_DIR}")
    print("先看 network_overview.png 理解结构，再看 model_structure.txt 对应代码中的各个层。")


if __name__ == "__main__":
    main()
