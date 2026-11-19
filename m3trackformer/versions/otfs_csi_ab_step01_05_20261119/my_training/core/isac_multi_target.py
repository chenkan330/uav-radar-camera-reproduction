"""第 3 步：三目标 OFDM 回波接入 V6，比较原选束和视距信道外推选束。"""
import argparse
from dataclasses import asdict, replace
from itertools import islice, starmap
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from .beam_identity import WaveformDetections
from .common import BASE, finish_run, load_nnet, load_split, new_run, read_json, state_digest, write_json, write_rows
from .identity_closed_loop import aggregate_association, evaluate, simulate_anonymous
from .isac_beam_selection import downlink_csi
from .isac_echo import C, ScenarioConfig, make_probe, peak_shift, propagate, thermal_noise
from .measurement_quality import spherical_to_cartesian
from .models import JournalRNet
from .multi_target import make_scenes
from .optimization_experiment import aggregate
from .trajectory_figure import save_trajectory_figure
from .versions import save_version, sha256


VERSION = "isac_step03_v6_waveform"
FIGURE = "v6_waveform_comparison.png"
STATION = np.array([0., 0., 10.])
BUDGET = 12
SCENES = 2
FRAMES = 30
DROPOUT = (0, 0, 12)  # 组、目标、帧。两组方法共同使用同一个受控单帧断测。
SIGMA = np.array([1.5, np.deg2rad(.35), np.deg2rad(.35)])
METHODS = {"v6": "原 V6 选束", "csi_forecast": "视距外推选束"}
RADAR_CONFIG = replace(ScenarioConfig(), cp_samples=192, symbols=96, sample_noise_power=.015,
                       peak_to_floor_threshold_db=23.)


class WaveformRadar:
    """基带环境相加后检测。真值来源只写到离线评价回执。"""
    waveform = "OFDM"
    make_probe = staticmethod(make_probe)

    def __init__(self, position, config, dropout=DROPOUT, *, scan_executor=None):
        self.position = position - STATION
        self.velocity = np.diff(self.position, axis=2, append=self.position[:, :, -1:])/config.decision_interval_s
        self.velocity[:, :, -1] = self.velocity[:, :, -2]
        self.config = config
        self.dropout = dropout
        self.scan_executor = scan_executor
        self.received_iq_observer = None
        self.receipts, self.sources, self.coverage = {}, {}, {}
        self.indices = np.arange(32)
        grid = -1 + 2*(self.indices+.5)/len(self.indices)
        self.weights = np.exp(1j*np.pi*self.indices[:, None]*grid)/np.sqrt(len(self.indices))
        self.probe_grid, tx, self.frequency, self.times = self.make_probe(config, np.random.default_rng(config.seed))
        self.spectrum = np.fft.fft(tx[:, config.cp_samples:], axis=1, norm="ortho")
        self.burst_s = tx.size/(config.subcarriers*config.subcarrier_spacing_hz)
        if BUDGET*self.burst_s > config.decision_interval_s:
            raise ValueError("扫描时隙超出决策帧")
        self.slow = np.hanning(config.symbols)
        self.slow /= self.slow.sum()
        self.fast = np.fft.ifftshift(np.hanning(config.subcarriers))
        self.fast /= self.fast.mean()
        self.doppler = np.fft.fftshift(np.fft.fftfreq(2*config.symbols, self.times[1]-self.times[0]))
        self.fs = config.subcarriers*config.subcarrier_spacing_hz

    def delay_samples(self, iq):
        frequency_samples = np.fft.fft(iq, axis=-1, norm="ortho")
        return np.fft.ifft(frequency_samples*(self.fast/self.probe_grid), axis=-1)

    def returns(self, group, step, slot, beam, targets):
        config = self.config
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, group, step, slot]))
        echoes, uv = [], []
        for target in targets:
            velocity = self.velocity[group, target, step]
            position = self.position[group, target, step] + slot*self.burst_s*velocity
            echo, direction, _, _ = propagate(config, self.spectrum, self.frequency, self.times, position,
                                             velocity, (beam%32, beam//32), self.indices, self.weights)
            if (group, target, step) == self.dropout:
                echo.fill(0.)
            echoes.append(echo)
            uv.append(direction[:2])
        echoes = np.stack(echoes)
        total = echoes.sum(axis=0) + thermal_noise(rng, echoes.shape[1:], config.sample_noise_power)
        if self.received_iq_observer is not None:
            samples = total.view()
            samples.flags.writeable = False
            self.received_iq_observer(group, step, slot, beam, samples)
        received = self.delay_samples(total)
        # 按时隙做多普勒 FFT。对 OTFS，这一步完成带窗的 SFFT。
        spectrum = np.fft.fftshift(np.fft.fft(received*self.slow[:, None], n=len(self.doppler), axis=0), axes=0)
        power = abs(spectrum[:, :config.cp_samples])**2
        floor = np.median(power)
        # 两维 Hann 压旁瓣。弱于峰值 26 dB 或未过噪声门限的峰不进入候选。
        threshold = max(floor*10**(config.peak_to_floor_threshold_db/10), power.max()*10**(-2.6))
        peaks = np.argwhere((power >= threshold) & (power == maximum_filter(power, (9, 5))))
        if not len(peaks):
            return []
        clean_range = self.delay_samples(echoes)
        steering = np.exp(1j*np.pi*self.indices[None, :, None]*np.asarray(uv)[:, None, :])
        detections = []
        for d, r in peaks:
            p = power[d, r]
            delta_r = peak_shift(*power[d, r-1:r+2]) if 0 < r < power.shape[1]-1 else 0.
            delta_d = peak_shift(power[(d-1)%len(self.doppler), r], p, power[(d+1)%len(self.doppler), r])
            frequency = self.doppler[d] + delta_d*(self.doppler[1]-self.doppler[0])
            coherent = self.slow * np.exp(-2j*np.pi*frequency*self.times)
            contributions = clean_range[:, :, r] @ coherent
            # 只合成检测单元的接收阵元快拍。先叠加所有路径在各阵元的复数贡献。
            cuts = np.einsum("t,tij->ij", contributions, steering)
            noise_power = (config.sample_noise_power*np.sum(abs(coherent)**2)
                           *np.sum(self.fast**2)/config.subcarriers**2)
            cuts += thermal_noise(rng, cuts.shape, noise_power)
            cuts[0] = coherent @ received[:, r]
            direction = np.angle(np.sum(cuts[:-1].conj()*cuts[1:], axis=0))/np.pi
            u, v = direction
            distance = C*(r+delta_r)/(2*self.fs)
            radial_velocity = -frequency*C/(2*config.carrier_hz)
            measurement = np.array([distance, np.arctan2(v, u), np.arcsin(min(1., np.linalg.norm(direction)))])
            # 以下来源推断不参与峰值检测、测角或匹配。
            strongest = int(np.argmax(abs(contributions)**2))
            source = targets[strongest] if abs(contributions[strongest])**2 >= threshold/2 else -1
            detections.append((p, beam, measurement, direction, radial_velocity, source))
        return detections

    def initialization(self, history):
        """独立指向校准，为初始化帧提供有身份标记的回波测量。"""
        groups, targets = self.position.shape[:2]
        z = np.empty((groups, targets, history, 3))
        for group in range(groups):
            for step in range(history):
                for target in range(targets):
                    beam = int(np.argmax(abs(self.csi(self.position[group, target, step]))**2))
                    returns = self.returns(group, step, target, beam, [target])
                    if not returns:
                        raise RuntimeError(f"初始化缺少回波，场景 {group} 目标 {target} 帧 {step}")
                    z[group, target, step] = max(returns, key=lambda row: row[0])[2]
        return z, np.broadcast_to(SIGMA, z.shape)

    def csi(self, relative_position):
        return downlink_csi(relative_position, self.config, self.indices, self.weights)

    def forecast_beams(self, probabilities, predicted_position, misses, budget):
        scores = probabilities.copy()
        for group, target in zip(*np.nonzero(misses == 0)):
            relative = predicted_position[group, target] - STATION
            if np.linalg.norm(relative) > 0:
                power = abs(self.csi(relative))**2
                scores[group, target] = power/power.sum()
        return np.argsort(-scores.sum(axis=1), axis=1, kind="stable")[:, :budget]

    def scan(self, step, live, selected):
        targets = list(range(self.position.shape[1]))
        jobs = ((int(group), step, slot, int(beam), targets) for group, beams in zip(live, selected)
                for slot, beam in enumerate(beams))
        # 各束回波独立计算。按原扫描顺序收集结果后再写入回执。
        returns_by_beam = (self.scan_executor.map(lambda job: self.returns(*job), jobs)
                           if self.scan_executor is not None else starmap(self.returns, jobs))
        packets = []
        for group, beams in zip(live, selected):
            group = int(group)
            raw = [row for beam_returns in islice(returns_by_beam, len(beams)) for row in beam_returns]
            kept = []
            for row in sorted(raw, key=lambda item: -item[0]):
                _, _, z, uv, velocity, _ = row
                if not any(abs(z[0]-other[2][0]) < 5 and abs(velocity-other[4]) < 3
                           and np.linalg.norm(uv-other[3]) < .022 for other in kept):
                    kept.append(row)
            occupied = np.array(sorted({row[1] for row in raw}), dtype=int)
            packet = WaveformDetections(np.array([row[1] for row in kept], dtype=int),
                np.array([row[2] for row in kept]).reshape(-1, 3),
                np.broadcast_to(SIGMA, (len(kept), 3)), occupied)
            self.receipts[group, step] = packet, np.array([row[5] for row in kept], dtype=int)
            self.sources[group, step] = {row[5] for row in raw if row[5] >= 0}
            self.coverage[group, step] = beams.copy()
            packets.append(packet)
        return packets


def physical_metrics(events, radar):
    """从接收峰回执统计物理漏检，包含归并前检出，区别于身份关联。"""
    streak = np.zeros(radar.position.shape[:2], dtype=int)
    result = dict(recoveries=0, lost_events=0, missed_target_frames=0, evaluated_target_frames=0)
    for event in events:
        group, step, target = (event[key] for key in ("scene_id", "time_index", "target_index"))
        hit = target in radar.sources[group, step]
        count = streak[group, target]
        result["recoveries"] += int(hit and count > 0)
        result["lost_events"] += int(not hit and count == 0)
        result["missed_target_frames"] += int(not hit)
        result["evaluated_target_frames"] += 1
        streak[group, target] = 0 if hit else count+1
    lost = result["lost_events"]
    result["recovery_rate"] = result["recoveries"]/lost if lost else None
    return result


def compare_gain(radars, output):
    """两个闭环都扫描过的帧，评价同一个帧首信道快照。"""
    v6, forecast = radars
    rows = []
    for group, step in sorted(v6.coverage.keys() & forecast.coverage.keys()):
        for target, position in enumerate(v6.position[group, :, step]):
            power = abs(v6.csi(position))**2
            oracle = power.max()
            row = {"scene": group, "frame": step, "target": target, "oracle_beam_1based": int(power.argmax())+1}
            for key, radar in zip(METHODS, radars):
                beams = radar.coverage[group, step]
                selected_power = power[beams].max()
                row[f"{key}_loss_db"] = float(10*np.log10(oracle/max(selected_power, 1e-30)))
            rows.append(row)
    write_rows(output/"paired_downlink_gain.csv", rows)
    return {"paired_target_frames": len(rows), "mean_loss_db": {
        key: float(np.mean([row[f"{key}_loss_db"] for row in rows])) for key in METHODS
    }}


def trial(output, name, radar, nnet, rnet, checkpoint, initialization, sigma, scenes, device, association,
          *, beam_policy=None, method_label=None):
    history, steps = nnet.config.history, scenes["beam"].shape[-1]
    dt = radar.config.decision_interval_s
    label = method_label or METHODS[name]
    events, timing, calls = simulate_anonymous(nnet, rnet, checkpoint, initialization, sigma, radar,
        steps, device, dt, BUDGET, association, beam_selector=radar.forecast_beams if name == "csi_forecast" else None,
        beam_policy=beam_policy)
    scored, events = evaluate(events, radar, scenes, history, dt, nnet.config.failure_limit)
    scored["filter_timing"] = timing
    identity = aggregate([scored], steps-history)
    controller = aggregate([{**scored, "scene_rows": scored["controller_scene_rows"],
                              "target_rows": scored["controller_target_rows"]}], steps-history)
    physical = physical_metrics(events, radar)
    local = output/name
    local.mkdir()
    for filename, rows in (("events", events), ("identity_scenes", scored["scene_rows"]),
                           ("identity_targets", scored["target_rows"])):
        write_rows(local/f"{filename}.csv", rows)
    write_rows(local/"scans.csv", [
        {"scene": group, "frame": step, "selected_beams_1based": ",".join(str(b+1) for b in beams),
         "occupied_beams_1based": ",".join(str(b+1) for b in radar.receipts[group, step][0].occupied_beams),
         "detected_source_ids_offline": ",".join(str(t) for t in sorted(radar.sources[group, step]))}
        for (group, step), beams in sorted(radar.coverage.items())])
    plot_id = int(scenes["trajectory_id"][0, 0])
    trace = [
        dict(trajectory_id=plot_id, time_index=i, phase="initialization", hit=True, recovered=False,
             estimate_source="initial_measurement",
             **{f"estimated_{a}_m": float(xyz[j]) for j, a in enumerate("xyz")})
        for i, xyz in enumerate(spherical_to_cartesian(initialization[0, 0]))
    ]
    trace += [event for event in events if event["scene_id"] == 0 and event["target_index"] == 0]
    single = physical_metrics([event for event in trace if event["phase"] == "tracking"], radar)
    controller["physical_recovery"] = {"single": single, "all": physical}
    controller["scenes"] = len(scored["scene_rows"])
    save_trajectory_figure(local, scenes, trace, scored["controller_target_rows"], controller,
        dt=dt, history=history, trajectory_id=plot_id, title=f"{label} · {radar.waveform} 回波",
        selection_note="固定首组首目标，蓝点为实际 N-Net 位置输出。黑叉表示控制器漏检。",
        highlight_dropout=(radar.dropout[2],) if radar.dropout is not None and radar.dropout[:2] == (0, 0) else ())
    print(f'{label}：完整场景 {identity["success_count"]}/{len(scored["scene_rows"])}，'
          f'物理重捕 {physical["recoveries"]}/{physical["lost_events"]}，R-Net {calls["rnet_target_frames"]} 目标帧。', flush=True)
    return {"identity": identity, "association": aggregate_association([scored], steps-history, 3),
            "physical": physical, "network_calls": calls,
            "scenario_identity_durations_s": [row["duration_s"] for row in scored["scene_rows"]]}


def draw_comparison(output, trials, gains, duration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout="constrained")
    series = [
        ([trial["identity"]["mean_all_targets_duration_s"] for trial in trials.values()], "身份有效时长（秒）"),
        ([trial["physical"]["recoveries"] for trial in trials.values()], "物理重捕次数"),
        (list(gains["mean_loss_db"].values()), "共同帧下行增益损失（dB）"),
    ]
    for axis, (values, title) in zip(axes, series):
        bars = axis.bar(list(METHODS.values()), values, color=["#355781", "#24836F"], width=.56)
        axis.set(title=title, ylim=(0, max(max(values)*1.2, .1)))
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.grid(axis="y", alpha=.22)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0, duration*1.18)
    fig.suptitle("三目标 · 每帧 12 束 · 共同 OFDM 回波前端")
    fig.savefig(output/FIGURE, dpi=160)
    plt.close(fig)


def load_v6_models(current, frozen, device):
    for name, digest in frozen["checkpoint_sha256"].items():
        if sha256(current[name]) != digest:
            raise RuntimeError(f"V6 权重校验失败：{name}")
    if sha256(Path(__file__).with_name("models.py")) != frozen["models_source_sha256"]:
        raise RuntimeError("原始 N/R 模型结构校验失败")
    nnet, checkpoint = load_nnet(current["nnet_checkpoint"], device)
    rnet = JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(torch.load(current["rnet_checkpoint"], map_location=device, weights_only=False)["model"])
    return nnet, rnet, checkpoint


def load_tracking_scenes(data_path, *, scene_limit=SCENES, frame_limit=FRAMES):
    scenes, _, _ = make_scenes(load_split(data_path), 3, limit=scene_limit)
    return {key: value if key == "trajectory_id" else value[:, :, :frame_limit] for key, value in scenes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="保存后打开对照图和两张轨迹图")
    args = parser.parse_args()
    torch.set_num_threads(4)
    current = read_json(BASE/"outputs/current_journal_experiment.json")
    frozen = read_json(BASE/"outputs/current_identity_closed_loop.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nnet, rnet, checkpoint = load_v6_models(current, frozen, device)
    model_hashes = state_digest(nnet), state_digest(rnet)
    data_path = Path(current["data"])/"test.npz"
    scenes = load_tracking_scenes(data_path)
    config = RADAR_CONFIG
    radars = [WaveformRadar(scenes["position"], config) for _ in METHODS]
    print("生成两组共用的有身份 OFDM 初始化测量。", flush=True)
    initial, sigmas = radars[0].initialization(nnet.config.history)
    output = new_run(BASE/"outputs"/VERSION)
    trials = {name: trial(output, name, radar, nnet, rnet, checkpoint, initial, sigmas, scenes,
                          device, frozen["configuration"]["parameters"])
              for name, radar in zip(METHODS, radars)}
    if model_hashes != (state_digest(nnet), state_digest(rnet)):
        raise RuntimeError("运行过程中 N/R 状态发生变化")
    gain = compare_gain(radars, output)
    draw_comparison(output, trials, gain, (FRAMES-nnet.config.history)*config.decision_interval_s)
    write_json(output/"summary.json", {
        "version": VERSION, "config": asdict(config), "burst_s": radars[0].burst_s,
        "scan_budget": BUDGET, "source": str(data_path), "nnet_checkpoint": current["nnet_checkpoint"],
        "rnet_checkpoint": current["rnet_checkpoint"], "scenario_trajectory_ids": scenes["trajectory_id"].tolist(),
        "nominal_sigma_m_rad_rad": SIGMA.tolist(), "dropout_scene_target_frame": list(DROPOUT),
        "initialization_frames": nnet.config.history, "frames_total": FRAMES,
        "csi_forecast": "Huber 一步预测位置构造视距 CSI，正常目标使用归一化下行波束增益，失跟目标使用原 R-Net 分数。",
        "scope": "既有测试 GNSS 的前两组短片段，单次配对噪声和固定视距散射，分配与跟踪均由回波驱动。",
        "trials": trials, "downlink": gain})
    archive = save_version(f"{VERSION}_{output.name[4:]}", output, "V6 三目标 OFDM 回波闭环对照。")
    finish_run(output)
    print(f"结果目录：{output.resolve()}\n归档目录：{archive.resolve()}", flush=True)
    if args.show:
        import os
        os.startfile(output/FIGURE)
        for name in METHODS:
            os.startfile(output/name/"figures"/"single_trajectory_tracking.png")
