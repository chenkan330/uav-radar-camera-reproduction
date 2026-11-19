"""三目标 OTFS 回波驱动的 V6 跟踪。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import BASE, finish_run, new_run, read_json, state_digest, write_json
from .isac_echo import C
from .isac_multi_target import BUDGET, DROPOUT, RADAR_CONFIG, SIGMA, WaveformRadar
from .isac_multi_target import load_tracking_scenes, load_v6_models, trial
from .versions import save_version


VERSION = "isac_step03_v6_otfs_tracking"


def make_otfs_probe(config, _rng=None):
    """轴 0 为多普勒，轴 1 为时延。发送单个 DD 导频，每个时隙加 CP。"""
    dd = np.zeros((config.symbols, config.subcarriers), dtype=complex)
    dd[0, 0] = np.sqrt(dd.size)
    tf = np.fft.fft(np.fft.ifft(dd, axis=0, norm="ortho"), axis=1, norm="ortho")
    useful = np.fft.ifft(tf, axis=1, norm="ortho")
    tx = np.concatenate((useful[:, -config.cp_samples:], useful), axis=1)
    fs = config.subcarriers * config.subcarrier_spacing_hz
    times = (np.arange(config.symbols) * tx.shape[1] + config.cp_samples)/fs
    return dd, tx, np.fft.fftfreq(config.subcarriers, 1/fs), times


class OTFSRadar(WaveformRadar):
    waveform = "OTFS"
    make_probe = staticmethod(make_otfs_probe)

    def delay_samples(self, iq):
        """去 CP 样本经 Wigner FFT 和 SFFT 的时延 IFFT，定位以零位 DD 导频为参考。"""
        tf = np.fft.fft(iq, axis=-1, norm="ortho")
        # 频域窗压低分数时延的旁瓣。非酉 IFFT 与零位导频的时域幅度归一化相配。
        return np.fft.ifft(tf*self.fast, axis=-1)

    def scan(self, step, live, selected):
        if step % 100 == 0:
            print(f"OTFS 帧 {step}/{self.position.shape[2]-1}，仍在跟踪 {len(live)} 组场景。", flush=True)
        return super().scan(step, live, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-test", action="store_true", help="使用测试集全部三目标场景和完整轨迹")
    parser.add_argument("--show", action="store_true", help="保存后打开单轨迹图")
    args = parser.parse_args()
    torch.set_num_threads(4)

    current = read_json(BASE/"outputs/current_journal_experiment.json")
    frozen = read_json(BASE/"outputs/current_identity_closed_loop.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet, rnet, checkpoint = load_v6_models(current, frozen, device)
    model_hashes = state_digest(nnet), state_digest(rnet)
    data_path = Path(current["data"])/"test.npz"
    if args.full_test:
        scenes = load_tracking_scenes(data_path, scene_limit=None, frame_limit=None)
        version, dropout = f"{VERSION}_full", None
    else:
        scenes = load_tracking_scenes(data_path)
        version, dropout = VERSION, DROPOUT
    started = perf_counter()
    groups, targets, frames = scenes["beam"].shape
    print(f"OTFS 回波闭环，{groups} 组 × {targets} 目标 × {frames} 帧，推理设备 {device}。", flush=True)
    association = frozen["configuration"]["parameters"]
    scan_workers = 6 if args.full_test else 1
    with (ThreadPoolExecutor(scan_workers) if scan_workers > 1 else nullcontext()) as executor:
        radar = OTFSRadar(scenes["position"], RADAR_CONFIG, dropout=dropout, scan_executor=executor)
        initial, sigmas = radar.initialization(nnet.config.history)
        output = new_run(BASE/"outputs"/version)
        metrics = trial(output, "v6", radar, nnet, rnet, checkpoint, initial, sigmas, scenes, device, association)
    if model_hashes != (state_digest(nnet), state_digest(rnet)):
        raise RuntimeError("运行过程中 N/R 状态发生变化")

    config = radar.config
    symbol_period = radar.times[1] - radar.times[0]
    write_json(output/"summary.json", {
        "version": version, "waveform": "CP-OTFS", "config": asdict(config),
        "dd_pilot_indices_doppler_delay": [0, 0], "burst_s": radar.burst_s,
        "range_resolution_m": C/(2*radar.fs),
        "doppler_resolution_hz": 1/(config.symbols*symbol_period),
        "scan_budget": BUDGET, "source": str(data_path), "nnet_checkpoint": current["nnet_checkpoint"],
        "rnet_checkpoint": current["rnet_checkpoint"], "scenario_trajectory_ids": scenes["trajectory_id"].tolist(),
        "nominal_sigma_m_rad_rad": SIGMA.tolist(), "association": association,
        "dropout_scene_target_frame": list(dropout) if dropout is not None else None,
        "initialization_frames": nnet.config.history, "frames_total": frames,
        "full_test_split": args.full_test, "weights_unchanged": True,
        "runtime_s": perf_counter()-started, "inference_device": str(device), "scan_workers": scan_workers,
        "beam_selection": "原 V6。逐目标 N/R 波束概率求和，取前 12 束。",
        "scope": "既有 GNSS 参考轨迹、自由空间点散射、单个 DD 探测导频；前五帧假设独立指向且已知身份。",
        "metrics": metrics,
    })
    archive = save_version(f"{version}_{output.name[4:]}", output, "V6 三目标 OTFS 回波闭环。")
    finish_run(output)
    print(f"结果目录：{output.resolve()}\n归档目录：{archive.resolve()}", flush=True)
    if args.show:
        import os
        os.startfile(output/"v6"/"figures"/"single_trajectory_tracking.png")
