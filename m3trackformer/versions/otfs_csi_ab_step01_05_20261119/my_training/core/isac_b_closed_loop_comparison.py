"""B 组第 5 步，冻结原 V6 与未来 CSI 选束做配对闭环比较。"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from shutil import copyfile, copytree
from time import perf_counter

import numpy as np
import torch

from .common import BASE, finish_run, new_run, read_json, state_digest, write_json, write_rows
from .isac_b_csi_predictor import (VERSION as PREDICTOR_VERSION, ResidualPredictor, beam_powers,
                                  causal_reference, make_channel, model_inputs)
from .isac_b_csi_samples import PROTOCOL
from .isac_b_csi_training_data import _feature_observer
from .isac_multi_target import BUDGET, RADAR_CONFIG, SIGMA, load_tracking_scenes, load_v6_models, trial
from .isac_otfs import OTFSRadar, VERSION as OTFS_VERSION
from .measurement_quality import spherical_to_cartesian
from .versions import save_version, sha256


VERSION = "isac_b_step05_ab_closed_loop"
BASELINE_VERSION = f"{OTFS_VERSION}_full"


def latest_run(version):
    parent = BASE/"outputs"/version
    return parent/(parent/"latest_run.txt").read_text(encoding="utf-8").strip()


def verified_manifest(run, files):
    summary = read_json(run/"summary.json")
    archive = BASE/"experiments"/f"{summary['version']}_{run.name.removeprefix('run_')}"
    manifest_path = archive/"manifest.json"
    manifest = read_json(manifest_path)
    for name in files:
        if sha256(run/name) != manifest["result_sha256"][name]:
            raise RuntimeError(f"历史实验文件校验失败：{run/name}")
    return summary, manifest, manifest_path


def load_baseline(run, scenes, source, current, frozen, history):
    files = ("summary.json", "v6/events.csv", "v6/identity_scenes.csv", "v6/identity_targets.csv",
             "v6/scans.csv", "v6/figures/single_trajectory_tracking.png",
             "v6/figures/single_trajectory_tracking_samples.csv", "v6/figures/single_trajectory_tracking.json")
    recorded, manifest, manifest_path = verified_manifest(run, files)
    if (recorded["version"] != BASELINE_VERSION or recorded["config"] != asdict(RADAR_CONFIG)
            or recorded["scenario_trajectory_ids"] != scenes["trajectory_id"].tolist()
            or recorded["nominal_sigma_m_rad_rad"] != SIGMA.tolist()
            or recorded["initialization_frames"] != history or recorded["frames_total"] != scenes["beam"].shape[-1]
            or recorded["scan_budget"] != BUDGET or recorded["dropout_scene_target_frame"] is not None
            or recorded["association"] != frozen["configuration"]["parameters"]):
        raise RuntimeError("历史 A 组与本次场景、波形、预算或跟踪配置不一致")
    for name, path in {"source": source, "nnet_checkpoint": current["nnet_checkpoint"],
                       "rnet_checkpoint": current["rnet_checkpoint"]}.items():
        if manifest["external_assets"][name]["sha256"] != sha256(path):
            raise RuntimeError(f"A 组资产与本次实验不一致：{name}")
    return recorded, manifest_path


class CsiBeamPolicy:
    """接受实时匿名量测和混合 IQ，由冻结的小模型给正常目标打分。"""
    def __init__(self, radar, checkpoint, device, history):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.history = history
        self.first_frame = 2*history
        dims = radar.position.shape[:3]
        self.positions = np.zeros((*dims, 3), dtype=np.float32)
        self.valid = np.zeros(dims, dtype=bool)
        record, self.features, self.beam_ids = _feature_observer(radar)
        frame_width = 4 + np.prod(self.features.shape[-2:])
        if not saved["use_iq"] or saved["history"] != history or saved["input_mean"].numel() != frame_width:
            raise ValueError("第 4 步检查点与当前 OTFS 特征不兼容")
        if saved["pretrained_dir"] is not None:
            raise ValueError("第 5 步使用第 4 步冻结的 IQ 小模型")
        self.device = device
        self.model = ResidualPredictor(int(frame_width), history).to(device).eval().requires_grad_(False)
        missing, unexpected = self.model.load_state_dict(saved["model_state_dict"], strict=False)
        if missing != ["residual_scale"] or unexpected:
            raise ValueError("CSI 小模型检查点字段不完整")
        self.mean = saved["input_mean"].numpy()
        self.scale = saved["input_std"].numpy()
        self.timing = []
        self.nr_target_frames = 0
        self.csi_target_frames = 0
        self.missing_history_target_frames = 0
        radar.received_iq_observer = record

    def observe(self, step, live, accepted, measurements):
        group_i, target = np.nonzero(accepted)
        group = live[group_i]
        self.valid[group, target, step] = True
        self.positions[group, target, step] = spherical_to_cartesian(
            measurements[group_i, target], base_station=(0., 0., 0.))

    def select(self, step, live, probabilities, misses, budget):
        start = perf_counter()
        past = np.arange(step-self.history, step)
        valid = self.valid[live[:, None, None], np.arange(misses.shape[1])[None, :, None], past]
        ready = (misses == 0) & valid.any(axis=-1)
        group_i, target = np.nonzero(ready)
        self.nr_target_frames += int((misses > 0).sum())
        self.missing_history_target_frames += int(((misses == 0) & ~ready).sum())
        self.csi_target_frames += len(group_i)
        scores = probabilities.copy()
        if len(group_i):
            group = live[group_i]
            if (self.beam_ids[group[:, None], past] < 0).any():
                raise RuntimeError("CSI 决策窗口包含未扫描的时隙")
            data = {
                "positions": self.positions[group[:, None], target[:, None], past],
                "valid": valid[group_i, target],
                "features": self.features[group[:, None], past].reshape(len(group), self.history, -1),
            }
            _, reference = causal_reference(data["positions"], data["valid"])
            x = (model_inputs(data, True)-self.mean)/self.scale
            prediction = self.model(torch.from_numpy(x).to(self.device), torch.from_numpy(reference).to(self.device))
            power = beam_powers(make_channel(prediction.cpu().numpy()))
            total = power.sum(axis=1, keepdims=True)
            if not np.isfinite(power).all() or (total <= 0).any():
                raise RuntimeError("CSI 模型输出的归一化波束功率无效")
            scores[group_i, target] = power/total
        selection = np.argsort(-scores.sum(axis=1), axis=1, kind="stable")[:, :budget]
        self.timing.append({"frame": step, "active_scenes": len(live), "csi_targets": len(group_i),
                            "b_policy_time_ms": (perf_counter()-start)*1000})
        return selection


def verify_warmup(a_dir, b_dir, first_frame):
    def read(directory):
        with (directory/"scans.csv").open(encoding="utf-8-sig", newline="") as stream:
            return [row for row in csv.DictReader(stream) if int(row["frame"]) < first_frame]

    a_rows, b_rows = read(a_dir), read(b_dir)
    if not a_rows or a_rows != b_rows:
        raise RuntimeError("预热时段 A/B 的实际扫描、波束顺序或回波检出结果不一致")
    return len(a_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, help="完整 A 组归档的运行目录")
    parser.add_argument("--predictor-run", type=Path, help="冻结第 4 步预测器的运行目录")
    args = parser.parse_args()
    torch.set_num_threads(4)
    current = read_json(BASE/"outputs/current_journal_experiment.json")
    frozen = read_json(BASE/"outputs/current_identity_closed_loop.json")
    source = Path(current["data"])/"test.npz"
    scenes = load_tracking_scenes(source, scene_limit=None, frame_limit=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet, rnet, checkpoint = load_v6_models(current, frozen, device)
    before = state_digest(nnet), state_digest(rnet)
    history = nnet.config.history
    baseline = (args.baseline_run or latest_run(BASELINE_VERSION)).resolve()
    a, baseline_manifest = load_baseline(baseline, scenes, source, current, frozen, history)
    predictor_run = (args.predictor_run or latest_run(PREDICTOR_VERSION)).resolve()
    predictor_summary, _, _ = verified_manifest(predictor_run, ("summary.json", "learned_iq.pt"))
    predictor_path = predictor_run/"learned_iq.pt"
    if predictor_summary["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("预测器训练协议与当前项目协议不一致")

    output = new_run(BASE/"outputs"/VERSION)
    a_dir = output/"a_original_v6"
    copytree(baseline/"v6", a_dir)
    copyfile(baseline/"summary.json", output/"a_frozen_summary.json")
    copyfile(PROTOCOL, output/"step01_protocol.json")
    print(f"已校验冻结 A 组，开始 B 组闭环，{len(scenes['position'])} 组三目标。", flush=True)
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=6) as executor:
        radar = OTFSRadar(scenes["position"], RADAR_CONFIG, dropout=None, scan_executor=executor)
        initial, sigmas = radar.initialization(history)
        policy = CsiBeamPolicy(radar, predictor_path, device, history)
        b = trial(output, "b_csi_prediction", radar, nnet, rnet, checkpoint, initial, sigmas, scenes,
                  device, frozen["configuration"]["parameters"], beam_policy=policy, method_label="B 组 · 未来 CSI 选束")
    b_runtime = perf_counter()-start
    if before != (state_digest(nnet), state_digest(rnet)):
        raise RuntimeError("V6 权重发生变化")
    b_dir = output/"b_csi_prediction"
    warmup_frames = verify_warmup(a_dir, b_dir, policy.first_frame)
    write_rows(output/"b_policy_timing.csv", policy.timing)
    # Preserve the completed roll-out before producing offline aggregate tables.
    write_json(output/"b_rollout_summary.json", {
        "metrics": b, "warmup_frames_verified": warmup_frames, "first_csi_frame": policy.first_frame,
        "csi_target_frames": policy.csi_target_frames, "rnet_target_frames_after_warmup": policy.nr_target_frames,
        "normal_target_frames_without_usable_history": policy.missing_history_target_frames,
        "wall_clock_runtime_s": b_runtime,
        "b_policy_batch_p95_ms": float(np.percentile([row["b_policy_time_ms"] for row in policy.timing], 95)),
    })

    from .isac_b_paired_metrics import paired_experiment_metrics

    paired = paired_experiment_metrics(output, scenes, radar, a_dir, b_dir, a["metrics"], b,
                                        b_first_frame=policy.first_frame)
    write_json(output/"summary.json", {
        "version": VERSION, "source": str(source), "baseline_run": str(baseline),
        "baseline_manifest": str(baseline_manifest), "predictor_checkpoint": str(predictor_path),
        "nnet_checkpoint": current["nnet_checkpoint"], "rnet_checkpoint": current["rnet_checkpoint"],
        "scope": "固定阵列、单径、17 组三目标 OTFS 仿真，同一测试分区",
        "comparison": "A 为已核对的冻结原 V6。B 正常目标用未来 CSI 归一化增益，失跟目标用原 R-Net",
        "b_input": "目标自身接受的角距量测与本组共同接收的混合 IQ 特征，分区训练采用单终端回波",
        "configuration": asdict(RADAR_CONFIG), "scene_trajectory_ids": scenes["trajectory_id"].tolist(),
        "a": a["metrics"], "b": b, "b_rollout": read_json(output/"b_rollout_summary.json"), "paired": paired,
    })
    archive = save_version(f"{VERSION}_{output.name[4:]}", output, "B 组第 5 步 OTFS 三目标闭环 A/B 对比。",
        asset_keys=("source", "baseline_manifest", "predictor_checkpoint", "nnet_checkpoint", "rnet_checkpoint"))
    finish_run(output)
    print(f"结果目录：{output.resolve()}\n归档目录：{archive.resolve()}", flush=True)
