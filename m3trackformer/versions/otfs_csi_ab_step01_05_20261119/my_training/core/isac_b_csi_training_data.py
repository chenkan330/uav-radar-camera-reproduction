"""通过冻结 V6 的单终端 OTFS 扫描采集未来下行 CSI 监督样本。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import fft

from .common import audit_partitions, state_digest
from .identity_closed_loop import simulate_anonymous
from .isac_beam_selection import downlink_channel_column
from .isac_echo import C
from .isac_multi_target import BUDGET, RADAR_CONFIG, SIGMA, STATION, load_v6_models
from .isac_otfs import OTFSRadar
from .measurement_quality import spherical_to_cartesian


FEATURE_NAMES = (
    "beam_u", "beam_v", "range_over_reference_range", "doppler_khz",
    "dd_peak_real", "dd_peak_imag", "peak_to_median_db", "slot_start_s",
)
SPLITS = (("train", "nnet_train", 3), ("validation", "validation", 8), ("test", "test", 8))


class CohortRadar(OTFSRadar):
    """允许剔除初始化失败的场景，同时保持仿真组编号和噪声种子。"""

    def scan(self, step, live, selected):
        return super().scan(step, self.controller_groups[live], selected)

    def initialize_cohort(self, history):
        available, failed, initial = [], [], []
        for group, trajectory in enumerate(self.position[:, 0]):
            observations = []
            for step in range(history):
                beam = int(np.argmax(abs(self.csi(trajectory[step]))**2))
                returns = self.returns(group, step, 0, beam, [0])
                if not returns:
                    failed.append((group, step))
                    break
                observations.append(max(returns, key=lambda row: row[0])[2])
            if len(observations) == history:
                available.append(group)
                initial.append(observations)
        self.controller_groups = np.array(available, dtype=int)
        if not available:
            raise RuntimeError("当前分区所有样例均初始化失败")
        z = np.asarray(initial)[:, None]
        return z, np.broadcast_to(SIGMA, z.shape), failed


def _choose_clips(source, clips_per_group, clip_frames, rng):
    positions, clips = [], []
    with np.load(source, allow_pickle=False) as data:
        groups, ids, all_positions = (data[key] for key in ("session_group", "trajectory_id", "position"))
        if clip_frames > all_positions.shape[1]:
            raise ValueError(f"{source.name}的轨迹长度不足")
        for group in np.unique(groups):
            pool = np.flatnonzero(groups == group)
            if len(pool) < clips_per_group:
                raise ValueError(f"{source.name}原始飞行组 {group} 的轨迹数不足")
            for row in rng.choice(pool, size=clips_per_group, replace=False):
                start = int(rng.integers(all_positions.shape[1]-clip_frames+1))
                positions.append(all_positions[row, start:start+clip_frames])
                clips.append({
                    "session_group": int(group), "trajectory_id": int(ids[row]), "clip_start": start,
                })
    return np.asarray(positions)[:, None], clips


def _feature_observer(radar):
    """从接收机的同一份有噪 IQ 提取零位导频的时延-多普勒峰。"""
    grid = -1. + 2.*(radar.indices+.5)/len(radar.indices)
    config = radar.config
    slow = radar.slow.astype(np.float32)
    doppler = fft.fftfreq(config.symbols, d=radar.times[1]-radar.times[0])
    storage_shape = (len(radar.position), radar.position.shape[2], BUDGET)
    features = np.empty((*storage_shape, len(FEATURE_NAMES)), dtype=np.float32)
    beam_ids = np.full(storage_shape, -1, dtype=np.int16)
    range_scale = C/(2*radar.fs*config.reference_range_m)
    iq_scale = np.sqrt(config.subcarriers)

    def record(group, step, slot, beam, iq):
        useful = iq[:, :config.cp_samples].astype(np.complex64)
        dd = fft.fft(useful*slow[:, None], axis=0)
        power = abs(dd)**2
        d, r = np.unravel_index(np.argmax(power), power.shape)
        peak = dd[d, r]/iq_scale
        by, bx = divmod(beam, len(grid))
        snr = 10*np.log10(max(float(power[d, r]), 1e-30)/max(float(np.median(power)), 1e-30))
        features[group, step, slot] = (
            grid[bx], grid[by], r*range_scale, doppler[d]/1000., peak.real, peak.imag, snr, slot*radar.burst_s,
        )
        beam_ids[group, step, slot] = beam

    return record, features, beam_ids


def _save_windows(path, radar, clips, features, beam_ids, events, history):
    observed = {}
    for event in events:
        group = int(radar.controller_groups[event["scene_id"]])
        step = event["time_index"]
        packet, _ = radar.receipts[group, step]
        index = event["assigned_detection_index"]
        if index >= 0:
            observed[group, step] = spherical_to_cartesian(packet.measurements[index], base_station=(0., 0., 0.))

    rows = {key: [] for key in (
        "features", "positions", "valid", "csi", "trajectory_id", "session_group", "label_frame",
        "history_frames", "decision_frame", "beam_ids",
    )}
    for group, clip in enumerate(clips):
        frames = np.array(sorted(step for sensor_group, step in radar.coverage if sensor_group == group), dtype=int)
        clip["scanned_frames"] = len(frames)
        clip["accepted_measurement_frames"] = sum((group, frame) in observed for frame in frames)
        clip["sample_count"] = max(0, len(frames)-history)
        if len(frames) <= history:
            continue
        np.testing.assert_array_equal(frames, np.arange(history, history+len(frames)))
        scanned = np.stack([radar.coverage[group, int(step)] for step in frames])
        np.testing.assert_array_equal(scanned, beam_ids[group, frames])
        for i, label in enumerate(frames[history:]):
            past = frames[i:i+history]
            accepted = [(group, step) in observed for step in past]
            rows["features"].append(features[group, past].reshape(history, -1))
            rows["positions"].append([observed.get((group, step), (0., 0., 0.)) for step in past])
            rows["valid"].append(accepted)
            rows["csi"].append(downlink_channel_column(radar.position[group, 0, label], radar.config, radar.indices))
            rows["trajectory_id"].append(clip["trajectory_id"])
            rows["session_group"].append(clip["session_group"])
            rows["label_frame"].append(clip["clip_start"]+label)
            rows["decision_frame"].append(clip["clip_start"]+past[-1])
            rows["history_frames"].append(clip["clip_start"]+past)
            rows["beam_ids"].append(scanned[i:i+history])
    if not rows["csi"]:
        raise RuntimeError(f"{path.stem}没有可用历史窗口")
    dtypes = {"features": np.float32, "positions": np.float32, "valid": bool, "csi": np.complex64}
    arrays = {key: np.asarray(value, dtype=dtypes.get(key, np.int32)) for key, value in rows.items()}
    for key in ("features", "positions", "csi"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path.stem}的 {key} 含非有限值")
    expected_history = arrays["label_frame"][:, None]-np.arange(history, 0, -1)
    np.testing.assert_array_equal(arrays["history_frames"], expected_history)
    if np.any(arrays["beam_ids"] < 0):
        raise ValueError("窗口内有未记录的实际扫描波束")
    np.savez_compressed(path, **arrays)
    return {
        "sample_count": len(arrays["csi"]), "sample_coverage_of_requested_windows":
            len(arrays["csi"])/(len(clips)*(radar.position.shape[2]-2*history)),
        "windows_with_measurement": int(arrays["valid"].any(axis=1).sum()),
        "windows_with_latest_measurement": int(arrays["valid"][:, -1].sum()),
        "session_groups_with_samples": np.unique(arrays["session_group"]).tolist(),
        "clips": clips,
    }


def collect_dataset(output, current, frozen, protocol, seed=20261119, scan_workers=6, clip_frames=28):
    """采集按原始飞行隔离的 train、validation、test 三个压缩数据文件。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(current["data"])
    partitions = audit_partitions(source_root)
    if partitions["session_groups_disjoint"] is not True:
        raise ValueError("必须提供互斥的原始飞行组")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet, rnet, checkpoint = load_v6_models(current, frozen, device)
    state_before = state_digest(nnet), state_digest(rnet)
    history = nnet.config.history
    timing = protocol["timing"]
    dt = RADAR_CONFIG.decision_interval_s
    if (history != protocol["observation"]["history_frames"] or timing["beams_per_frame"] != BUDGET
            or timing["frame_period_s"] != dt or timing["predicted_frame_offset"] != 1
            or clip_frames <= 2*history or scan_workers < 1):
        raise ValueError("采集配置与第 1 步时序不一致")
    rng = np.random.default_rng(seed)
    association = frozen["configuration"]["parameters"]
    statistics = {}
    with ThreadPoolExecutor(max_workers=scan_workers) as executor:
        for split_index, (split, filename, per_group) in enumerate(SPLITS):
            source = source_root/f"{filename}.npz"
            positions, clips = _choose_clips(source, per_group, clip_frames, rng)
            config = replace(RADAR_CONFIG, seed=seed+split_index)
            radar = CohortRadar(positions, config, dropout=None, scan_executor=executor)
            if not np.isclose(radar.burst_s, timing["burst_duration_s"], atol=1e-12, rtol=0):
                raise ValueError("仿真波束周期与采集约定不一致")
            print(f"CSI 采集 {split}，{len(clips)} 条轨迹 × {clip_frames} 帧，设备 {device}。", flush=True)
            initial, sigma, init_failed = radar.initialize_cohort(history)
            for group, step in init_failed:
                clips[group]["initialization_failure_frame"] = step
            record, features, beam_ids = _feature_observer(radar)
            radar.received_iq_observer = record
            events, _, calls = simulate_anonymous(nnet, rnet, checkpoint, initial, sigma, radar,
                clip_frames, device, dt, BUDGET, association)
            saved = _save_windows(output/f"{split}.npz", radar, clips, features, beam_ids, events, history)
            statistics[split] = {
                "source": str(source), "selected_clips": len(clips), "initialization_failures": len(init_failed),
                "early_stops": sum(0 < clip["scanned_frames"] < clip_frames-history for clip in clips),
                "network_calls": calls, **saved,
            }
            print(f"CSI 采集 {split}，已保存 {saved['sample_count']} 个窗口，初始化失败 {len(init_failed)} 条。", flush=True)
    if state_before != (state_digest(nnet), state_digest(rnet)):
        raise RuntimeError("V6 权重状态发生变化")
    return {
        "splits": statistics, "seed": int(seed), "clip_frames": int(clip_frames), "scan_workers": int(scan_workers),
        "flight_partitions": partitions, "features_per_beam": list(FEATURE_NAMES),
        "feature_layout": "每帧按实际扫描时隙展平，单时隙按 features_per_beam 排列",
        "positions": "当前接收机已接受的匿名量测，转为相对阵列的 XYZ，valid 为假时填 0",
        "frame_clock": "采样和监督使用各原始轨迹内的帧编号",
        "channel_labels": "仿真结束后计算的下一帧首束时刻单程 LOS CSI",
    }
