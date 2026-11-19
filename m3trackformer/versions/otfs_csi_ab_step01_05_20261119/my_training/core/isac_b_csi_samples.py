"""记录 B 组第 2 步单终端 OTFS 观测与下一帧下行 CSI 样例。"""
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch

from .common import BASE, finish_run, new_run, read_json, state_digest, write_json, write_rows
from .identity_closed_loop import evaluate, simulate_anonymous
from .isac_beam_selection import downlink_channel_column
from .isac_multi_target import BUDGET, RADAR_CONFIG, load_v6_models, physical_metrics
from .isac_otfs import OTFSRadar
from .measurement_quality import spherical_to_cartesian
from .trajectory_figure import save_trajectory_figure
from .versions import save_version, sha256


VERSION = "isac_b_step02_csi_samples"
PROTOCOL = BASE/"isac/b_group_step01_protocol.json"
DEMO_FRAMES = 18


def save_aligned_samples(output, radar, captures, history, protocol):
    """保存每帧一次 IQ，各监督窗口用行索引复用同一份回波。"""
    dt = radar.config.decision_interval_s
    schedule = protocol["timing"]
    if (history != protocol["observation"]["history_frames"] or BUDGET != schedule["beams_per_frame"]
            or not np.isclose(dt, schedule["frame_period_s"], rtol=0, atol=1e-12)
            or not np.isclose(radar.burst_s, schedule["burst_duration_s"], rtol=0, atol=1e-12)
            or schedule["predicted_frame_offset"] != 1):
        raise ValueError("运行配置与第 1 步时序不一致")
    frames = np.array(sorted(step for group, step in radar.coverage if group == 0))
    if len(frames) <= history:
        raise RuntimeError(f"仅采到 {len(frames)} 帧，无法配成 {history} 帧输入与下一帧标签")
    np.testing.assert_array_equal(frames, np.arange(history, history+len(frames)))
    beams = np.stack([radar.coverage[0, int(step)] for step in frames])
    if len(captures) != beams.size or beams.shape[1] != BUDGET:
        raise ValueError("扫描回执与 IQ 记录的波束数量不一致")

    iq_frames = []
    for step, scanned in zip(frames, beams):
        stored = [captures[int(step), slot] for slot in range(BUDGET)]
        np.testing.assert_array_equal([beam for beam, _ in stored], scanned)
        iq_frames.append([iq for _, iq in stored])
    iq = np.array(iq_frames, dtype=np.complex64)
    if iq.shape != (len(frames), BUDGET, radar.config.symbols, radar.config.subcarriers):
        raise ValueError("参考阵元 IQ 尺寸不一致")
    if not np.isfinite(iq).all():
        raise ValueError("接收 IQ 中存在非有限值")

    slot_start = frames[:, None]*dt + np.arange(BUDGET)*radar.burst_s
    slot_end = slot_start + radar.burst_s
    valid = np.ones_like(beams, dtype=bool)
    occupied = np.stack([np.isin(row, radar.receipts[0, int(step)][0].occupied_beams)
                         for step, row in zip(frames, beams)])
    window_rows = np.lib.stride_tricks.sliding_window_view(np.arange(len(frames)-1), history)
    decision_rows = window_rows[:, -1]
    label_frames = frames[decision_rows+1]
    label_time = label_frames*dt
    cutoff = slot_end[decision_rows, -1]
    if (slot_end[window_rows] > cutoff[:, None, None] + 1e-12).any():
        raise ValueError("输入包含观测截止后的采样")
    lead_time = label_time-cutoff
    np.testing.assert_allclose(lead_time, schedule["lead_time_from_scan_end_s"], atol=1e-12, rtol=0)

    # 下行标签在控制器结束后生成。雷达接收 IQ 和下行信道来自各自的传播模型。
    label_positions = radar.position[0, 0, label_frames]
    channel = np.stack([downlink_channel_column(p, radar.config, radar.indices) for p in label_positions])
    expected_size = protocol["channel_label"]["per_terminal_complex_elements"]
    if channel.shape != (len(label_frames), expected_size) or not np.isfinite(channel).all():
        raise ValueError("下行信道标签尺寸或数值无效")
    projected = channel.conj() @ np.kron(radar.weights, radar.weights)
    existing = np.stack([radar.csi(p) for p in label_positions])
    np.testing.assert_allclose(projected, existing, atol=1e-12, rtol=1e-10)

    np.savez_compressed(output/"observations.npz", reference_iq=iq, frame_indices=frames, beam_ids=beams,
                        slot_start_s=slot_start, slot_end_s=slot_end, valid_mask=valid,
                        peak_detected=occupied, history_rows=window_rows, decision_cutoff_s=cutoff,
                        useful_symbol_start_offsets_s=radar.times, sample_rate_hz=radar.fs)
    np.savez_compressed(output/"csi_labels.npz", h=channel, label_frame_indices=label_frames, label_time_s=label_time)
    rms = np.sqrt(np.mean(abs(iq)**2, axis=(2, 3)))
    write_rows(output/"scans.csv", [
        {"observation_row": i, "frame": int(frame), "slot": slot, "beam_id_0based": int(beam),
         "start_s": round(float(slot_start[i, slot]), 7), "end_s": round(float(slot_end[i, slot]), 7),
         "valid_iq": bool(valid[i, slot]), "peak_detected": bool(occupied[i, slot]),
         "iq_rms": float(rms[i, slot])}
        for i, (frame, row) in enumerate(zip(frames, beams)) for slot, beam in enumerate(row)
    ])
    windows = [
        {"sample_index": i, "history_frames": ",".join(map(str, frames[rows])),
         "history_observation_rows": ",".join(map(str, rows)), "cutoff_s": round(float(cutoff[i]), 7),
         "label_frame": int(label_frames[i]), "label_time_s": round(float(label_time[i]), 7),
         "lead_time_ms": round(float(lead_time[i]*1000), 6)}
        for i, rows in enumerate(window_rows)
    ]
    write_rows(output/"windows.csv", windows)
    return {
        "window_count": len(windows), "first_window": windows[0], "last_window": windows[-1],
        "scanned_frames": frames.tolist(), "captured_beam_slots": int(beams.size),
        "captured_beam_slots_with_peak": int(occupied.sum()),
        "observation_shape": list(iq.shape), "observation_dtype": str(iq.dtype),
        "channel_label_shape": list(channel.shape), "channel_label_dtype": str(channel.dtype),
        "max_complex_response_error": float(np.max(abs(projected-existing))),
    }


def plot_tracking(output, radar, scenes, initial, events, target, trajectory_id):
    history = initial.shape[2]
    physical = physical_metrics(events, radar)
    trace = [
        {"trajectory_id": trajectory_id, "time_index": i, "phase": "initialization", "hit": True,
         "recovered": False, "estimate_source": "initial_measurement",
         **{f"estimated_{axis}_m": float(value) for axis, value in zip("xyz", position)}}
        for i, position in enumerate(spherical_to_cartesian(initial[0, 0]))
    ]
    trace.extend(events)
    summary = {
        "scenes": 1, "mean_all_targets_duration_s": target["observed_duration_s"],
        "recoveries": target["recoveries"], "lost_events": target["lost_events"],
        "recovery_rate": target["recoveries"]/target["lost_events"] if target["lost_events"] else None,
        "physical_recovery": {"single": physical, "all": physical},
    }
    save_trajectory_figure(output, scenes, trace, [target], summary, dt=radar.config.decision_interval_s,
        history=history, trajectory_id=trajectory_id, title="B 组第 2 步 · 单终端 OTFS 样例采集",
        selection_note="固定训练分区首条轨迹。跟踪帧的扫描由 V6 决定，观测和标签用于验证样本对齐。")


def main():
    torch.set_num_threads(4)
    protocol = read_json(PROTOCOL)
    current = read_json(BASE/"outputs/current_journal_experiment.json")
    frozen = read_json(BASE/"outputs/current_identity_closed_loop.json")
    source = Path(current["data"])/"nnet_train.npz"
    with np.load(source) as data:
        scenes = {key: data[key][:1, None, :DEMO_FRAMES].copy() for key in ("position", "beam")}
        scenes["trajectory_id"] = data["trajectory_id"][:1, None].copy()
    trajectory_id = int(scenes["trajectory_id"][0, 0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet, rnet, checkpoint = load_v6_models(current, frozen, device)
    before = state_digest(nnet), state_digest(rnet)
    radar = OTFSRadar(scenes["position"], RADAR_CONFIG, dropout=None)
    history = nnet.config.history
    initial, sigma = radar.initialization(history)
    captures = {}

    def capture(group, step, slot, beam, iq):
        if group != 0 or (step, slot) in captures:
            raise ValueError("预期每个单终端扫描时隙只记录一次")
        captures[step, slot] = beam, iq.astype(np.complex64, copy=True)

    radar.received_iq_observer = capture
    dt = radar.config.decision_interval_s
    steps = scenes["position"].shape[2]
    events, _, calls = simulate_anonymous(nnet, rnet, checkpoint, initial, sigma, radar,
        steps, device, dt, BUDGET, frozen["configuration"]["parameters"])
    if before != (state_digest(nnet), state_digest(rnet)):
        raise RuntimeError("V6 权重状态发生变化")
    scored, events = evaluate(events, radar, scenes, history, dt, nnet.config.failure_limit)

    output = new_run(BASE/"outputs"/VERSION)
    samples = save_aligned_samples(output, radar, captures, history, protocol)
    copyfile(PROTOCOL, output/"step01_protocol.json")
    write_rows(output/"tracking_events.csv", events)
    target = scored["controller_target_rows"][0]
    plot_tracking(output, radar, scenes, initial, events, target, trajectory_id)
    write_json(output/"summary.json", {
        "version": VERSION, "purpose": "单终端、单径、归一化仿真数据的观测与标签对齐",
        "trajectory_id": trajectory_id, "source": str(source), "source_split": "nnet_train",
        "trajectory_selection": "固定分区存储顺序首条轨迹的前 18 帧",
        "nnet_checkpoint": current["nnet_checkpoint"], "rnet_checkpoint": current["rnet_checkpoint"],
        "initialization_frames": history, "sample_protocol_sha256": sha256(PROTOCOL),
        "beam_selection": "原 V6 闭环中实际发射的波束，初始化结束后开始记录",
        "iq_representation": "参考阵元的有噪时域复基带，去 CP 后、FFT 和检测前",
        "iq_axes": ["scan_frame_row", "beam_slot", "otfs_symbol", "useful_fast_time_sample"],
        "label_axes": ["sample_index", "flattened_antenna_yx"],
        "valid_mask_meaning": "实际采样的时隙有效，接收机峰值检测结果独立记录",
        "timing_reference": "采样与标签以轨迹首帧为 0 s",
        "demo_only": True, "network_calls": calls, "samples": samples,
    })
    archive = save_version(f"{VERSION}_{output.name[4:]}", output, "B 组第 2 步单终端 OTFS 观测及未来 CSI。")
    finish_run(output)
    print(f"轨迹 {trajectory_id}，记录 {samples['captured_beam_slots']} 个扫描时隙，"
          f"生成 {samples['window_count']} 条对齐样本。\n结果目录：{output.resolve()}\n归档目录：{archive.resolve()}",
          flush=True)
