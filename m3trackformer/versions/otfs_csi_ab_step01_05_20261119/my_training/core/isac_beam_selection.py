"""第 2 步：在同一单目标轨迹上比较旧信道选束、因果外推和离线最优。"""
import argparse
import csv
from dataclasses import asdict, dataclass

import numpy as np

from .isac_echo import BASE, C, ScenarioConfig, cases, make_probe, propagate, receive


VERSION = "isac_step02_single_target_beam"
FIGURE = "beam_selection_comparison.png"
STRATEGIES = ("stale", "predicted", "oracle")
HISTORY_SIZE = 5


@dataclass(frozen=True)
class Observation:
    frame: int
    sample_time_s: float
    available_time_s: float
    position_m: np.ndarray


def _downlink_los_terms(position_m, config, antenna_indices):
    distance = np.linalg.norm(position_m)
    direction = position_m[:2]/distance
    steering = np.exp(1j * np.pi * antenna_indices[:, None] * direction)
    phase = -2*np.pi * distance * config.carrier_hz/C
    path = (config.reference_range_m/distance) * np.exp(1j*phase)
    return steering, path


def downlink_csi(position_m, config, antenna_indices, weights):
    """返回单程视距信道在 1024 个波束上的等效复数系数，x 方向编号先变化。"""
    steering, path = _downlink_los_terms(position_m, config, antenna_indices)
    response_x, response_y = steering.conj().T @ weights / np.sqrt(len(antenna_indices))
    return path * np.outer(response_y, response_x).ravel()


def downlink_channel_column(position_m, config, antenna_indices):
    """返回满足 h^H w 的阵元域信道列向量，按 [y,x] 展平。"""
    steering, path = _downlink_los_terms(position_m, config, antenna_indices)
    return path.conjugate() * np.outer(steering[:, 1], steering[:, 0]).ravel()/len(antenna_indices)


def estimate_positions(observations, apply_time_s):
    """从已被接收机处理完的雷达观测构造旧位置和匀速外推位置。"""
    history = [sample for sample in observations if sample.available_time_s <= apply_time_s]
    history = history[-HISTORY_SIZE:]
    if not history:
        return None
    last = history[-1]
    if len(history) < 2:
        return last.position_m, last.position_m, last
    fit_times = np.array([sample.sample_time_s for sample in history])
    fit_positions = np.stack([sample.position_m for sample in history])
    # 中心化最小二乘，只拟合已经收到的样本。仅一条观测时保持上一位置。
    mean_time = fit_times.mean()
    rel_time = fit_times - mean_time
    avg_position = fit_positions.mean(axis=0)
    velocity = rel_time @ (fit_positions - avg_position) / (rel_time @ rel_time)
    predicted = avg_position + (apply_time_s - mean_time) * velocity
    return last.position_m, predicted, last


def collect_observations(scans, config, probe, antenna_indices, weights, rng):
    """两束探测时隙对各策略相同。对每帧只保留峰值最高且检出的观测。"""
    pilots, spectrum, frequency, sample_times, burst_s = probe
    samples = []
    for first, second in zip(scans[::2], scans[1::2]):
        estimates = []
        for _, _, scan_s, beam, scan_position, scan_velocity in (first, second):
            echo, direction, _, _ = propagate(
                config, spectrum, frequency, sample_times, scan_position, scan_velocity,
                beam, antenna_indices, weights
            )
            estimate, _, _, _ = receive(config, pilots, sample_times, echo, direction, antenna_indices, rng)
            if estimate["detected"]:
                estimates.append((scan_s, estimate))
        if estimates:
            scan_s, estimate = max(estimates, key=lambda item: item[1]["echo_peak_db"])
            u, v = estimate["measured_u"], estimate["measured_v"]
            direction = np.array([u, v, np.sqrt(max(0., 1.-u*u-v*v))])
            position_hat = estimate["measured_range_m"] * direction/np.linalg.norm(direction)
            samples.append(Observation(first[1], scan_s+burst_s/2, second[2]+burst_s, position_hat))
    return samples


def compare_scene(scene, scans, observations, delay_frames, config, antenna_indices, weights):
    rows = []
    for _, frame, time_s, _, true_position, _ in scans[::2][HISTORY_SIZE+delay_frames-1:]:
        # 一帧延迟时只使用 k-1 帧及更早的探测。可用时间再作一次检查。
        available = [sample for sample in observations if sample.frame <= frame-delay_frames]
        estimates = estimate_positions(available, time_s)
        if estimates is None:
            continue
        stale_position, predicted_position, latest = estimates
        stale_csi = downlink_csi(stale_position, config, antenna_indices, weights)
        predicted_csi = downlink_csi(predicted_position, config, antenna_indices, weights)
        stale_beam, predicted_beam = (int(np.argmax(np.abs(csi)**2))
                                      for csi in (stale_csi, predicted_csi))

        # 真实未来信道仅供计算实际增益和离线最优束。对照组在上面已作完决策。
        actual_power = np.abs(downlink_csi(true_position, config, antenna_indices, weights))**2
        oracle_beam = int(np.argmax(actual_power))
        row = {
            "scenario": scene, "frame": frame, "apply_time_s": time_s,
            "last_observation_frame": latest.frame,
            "last_sample_time_s": latest.sample_time_s,
            "last_available_time_s": latest.available_time_s,
            "true_u": true_position[0]/np.linalg.norm(true_position),
            "stale_u": stale_position[0]/np.linalg.norm(stale_position),
            "predicted_u": predicted_position[0]/np.linalg.norm(predicted_position),
        }
        for name, beam in zip(STRATEGIES, (stale_beam, predicted_beam, oracle_beam)):
            row[f"{name}_beam"] = beam+1
            row[f"{name}_actual_power_db"] = float(10*np.log10(max(actual_power[beam], 1e-30)))
            loss = actual_power[oracle_beam]/max(actual_power[beam], 1e-30)
            row[f"{name}_loss_db"] = float(10*np.log10(loss))
        rows.append(row)
    return rows


def summarize(rows):
    metrics = {}
    for strategy in STRATEGIES:
        losses = np.array([row[f"{strategy}_loss_db"] for row in rows])
        metrics[strategy] = {
            "matches_oracle_beam": int(sum(row[f"{strategy}_beam"] == row["oracle_beam"] for row in rows)),
            "mean_loss_db": float(losses.mean()),
            "max_loss_db": float(losses.max()),
        }
    return {
        "frames_evaluated": len(rows), "strategies": metrics,
        "prediction_gain_over_stale_db": float(np.mean([
            row["stale_loss_db"]-row["predicted_loss_db"] for row in rows
        ])),
        "frames_with_different_decisions": [
            {key: row[key] for key in ("frame", "apply_time_s", "stale_beam", "predicted_beam", "oracle_beam",
                                       "stale_loss_db", "predicted_loss_db")}
            for row in rows if row["stale_beam"] != row["predicted_beam"]
        ],
    }


def render_comparison(rows, output, delay_frames, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.2), sharex="col", layout="constrained")
    delay_ms = delay_frames*config.decision_interval_s*1000
    fig.suptitle(f"单目标视距下行选束对照 · 标称观测滞后 {delay_ms:g} ms", fontsize=14)
    labels = ("沿用旧信道", "因果外推信道", "未来真值（离线）")
    colors = ("#A64718", "#216E65", "#263D67")
    for column, (scene, title) in enumerate((("crossing", "横向跨束"), ("radial", "径向远离"))):
        results = [row for row in rows if row["scenario"] == scene]
        time = [row["apply_time_s"] for row in results]
        for name, label, color, style in zip(reversed(STRATEGIES), reversed(labels), reversed(colors),
                                             ("-", "--", ":")):
            beam = [row[f"{name}_beam"] for row in results]
            loss = [row[f"{name}_loss_db"] for row in results]
            axes[0, column].step(time, beam, where="post", label=label, color=color,
                                  linestyle=style, linewidth=2.)
            axes[1, column].plot(time, loss, label=label, color=color, linestyle=style,
                                  marker="o" if name == "stale" else None, markersize=3)
        axes[0, column].set_title(title)
        axes[0, column].set_yticks(sorted({row[f"{name}_beam"] for row in results for name in STRATEGIES}))
        all_beams = [row[f"{name}_beam"] for row in results for name in STRATEGIES]
        axes[0, column].set_ylim(min(all_beams)-.18, max(all_beams)+.18)
        axes[1, column].set_xlabel("发射时刻（秒）")
        axes[1, column].yaxis.set_major_locator(MaxNLocator(nbins=5))
        for axis in axes[:, column]:
            axis.grid(alpha=.22)
    max_loss = max(row[f"{name}_loss_db"] for row in rows for name in STRATEGIES)
    for axis in axes[1]:
        axis.set_ylim(-0.035*max_loss, max(1.13*max_loss, .04))
    axes[0, 0].set_ylabel("发射波束编号")
    axes[1, 0].set_ylabel("相对同帧最优束的下行损失（dB）")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles[::-1], legend_labels[::-1], loc="lower right")
    fig.savefig(output/FIGURE, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-frames", type=int, choices=(1, 2), default=1,
                        help="用几帧前最近可用的测量，默认 1 帧（50 ms）")
    parser.add_argument("--show", action="store_true", help="保存后打开结果图")
    args = parser.parse_args()

    config = ScenarioConfig()
    rng = np.random.default_rng(config.seed)
    pilots, tx, frequency, sample_times = make_probe(config, rng)
    spectrum = np.fft.fft(tx[:, config.cp_samples:], axis=1, norm="ortho")
    burst_s = tx.size/(config.subcarriers*config.subcarrier_spacing_hz)
    probe = pilots, spectrum, frequency, sample_times, burst_s
    antenna_indices = np.arange(32)
    grid = -1+2*(antenna_indices+.5)/len(antenna_indices)
    weights = np.exp(1j*np.pi*antenna_indices[:, None]*grid)/np.sqrt(len(antenna_indices))
    _, radial, *crossing = cases(config, grid, burst_s)
    radial_start, radial_velocity = radial[4:6]
    radial_scans = [("radial", frame, time_s, beam, radial_start+time_s*radial_velocity, radial_velocity)
                    for _, frame, time_s, beam, _, _ in crossing]

    all_rows, results = [], {}
    for scene, scans in (("crossing", crossing), ("radial", radial_scans)):
        observations = collect_observations(scans, config, probe, antenna_indices, weights, rng)
        rows = compare_scene(scene, scans, observations, args.delay_frames, config, antenna_indices, weights)
        expected = len(scans)//2 - HISTORY_SIZE - args.delay_frames + 1
        if len(rows) != expected or any(row["last_available_time_s"] > row["apply_time_s"] for row in rows):
            raise RuntimeError(f"{scene} 缺少有效历史，或观测完成时间晚于发射时刻")
        results[scene] = {"frames_with_radar_observation": len(observations), **summarize(rows)}
        all_rows.extend(rows)

    from .common import finish_run, new_run, write_json
    from .versions import save_version

    output = new_run(BASE/"outputs"/VERSION)
    with (output/"beam_selection_frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output/"summary.json", {
        "version": VERSION, "config": asdict(config), "delay_frames": args.delay_frames,
        "nominal_decision_delay_s": args.delay_frames*config.decision_interval_s,
        "history_window": HISTORY_SIZE, "fixed_probe_beams": [564, 565],
        "selected_downlink_beams_per_strategy": 1,
        "model_scope": [
            "单目标视距，复数下行模型使用载频处的窄带阵列响应及单程自由空间衰减",
            "旧信道和外推信道均由历史 OFDM 雷达角距观测重建，匀速模型拟合位置",
            "固定两束共用探测时隙，评估同一时刻的单束下行增益，真值用作离线评价",
            "相对同帧离线最优束的增益损失为正值，0 dB 表示达到该帧的最优值",
        ],
        "scenarios": results,
    })
    render_comparison(all_rows, output, args.delay_frames, config)
    archive = save_version(f"{VERSION}_{output.name[4:]}", output, "单目标因果选束对照。", asset_keys=())
    finish_run(output)
    print(f"结果目录：{output.resolve()}\n归档目录：{archive.resolve()}", flush=True)
    for name, label in (("crossing", "横向跨束"), ("radial", "径向远离")):
        item = results[name]
        old, new = (item["strategies"][kind] for kind in ("stale", "predicted"))
        print(f'{label}：旧信道 {old["matches_oracle_beam"]}/{item["frames_evaluated"]}，'
              f'外推 {new["matches_oracle_beam"]}/{item["frames_evaluated"]}；'
              f'平均下行增益变化 {item["prediction_gain_over_stale_db"]:+.4f} dB。', flush=True)
    if args.show:
        import os
        os.startfile(output/FIGURE)
