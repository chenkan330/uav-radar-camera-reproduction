"""对比实验第1步：已知 OFDM 探测、单站单目标复基带回波和角距速度估计。"""
import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

C = 299_792_458.0
BASE = Path(__file__).resolve().parents[1]
VERSION = "isac_step01_single_target_echo"
FIGURES = ("beam_crossing.png", "range_doppler.png")


@dataclass(frozen=True)
class ScenarioConfig:
    carrier_hz: float = 28e9
    subcarrier_spacing_hz: float = 120e3
    subcarriers: int = 256
    cp_samples: int = 64
    symbols: int = 192
    decision_interval_s: float = 0.05
    seed: int = 20261118
    reference_range_m: float = 150.
    sample_noise_power: float = 0.5
    peak_to_floor_threshold_db: float = 18.


def make_probe(config, rng):
    """生成单位平均功率 QPSK 探测符号，IFFT 后给每个符号加循环前缀。"""
    n = config.subcarriers
    shape = (config.symbols, n)
    pilots = ((2*rng.integers(0, 2, shape)-1)
              + 1j*(2*rng.integers(0, 2, shape)-1))/np.sqrt(2.)
    useful = np.fft.ifft(pilots, axis=1, norm="ortho")
    tx = np.concatenate((useful[:, -config.cp_samples:], useful), axis=1)
    fs = n * config.subcarrier_spacing_hz
    useful_times = (np.arange(config.symbols) * tx.shape[1] + config.cp_samples)/fs
    return pilots, tx, np.fft.fftfreq(n, 1/fs), useful_times


def beam_response(direction_xy, beam_xy, indices, weights):
    """以单位范数 32×32 DFT 码字发射，返回目标方向的归一化复响应。"""
    steering = np.exp(1j*np.pi*indices[:, None]*direction_xy)
    response_x = np.vdot(steering[:, 0], weights[:, beam_xy[0]])
    response_y = np.vdot(steering[:, 1], weights[:, beam_xy[1]])
    return response_x * response_y / len(indices)


def propagate(config, tx_spectrum, frequency, times, position, velocity, beam_xy, indices, weights):
    """在有效符号窗内实现分数延迟。CP 覆盖时，其结果等价于时域循环移位。"""
    distance = np.linalg.norm(position)
    direction = position/distance
    radial = np.dot(direction, velocity)
    fs = config.subcarriers * config.subcarrier_spacing_hz
    distance_by_symbol = distance + radial*times
    delay = 2*distance_by_symbol/C
    if (delay >= config.cp_samples/fs).any():
        raise ValueError("回波时延超出循环前缀覆盖范围")
    delay_phase = np.exp(-2j*np.pi * delay[:, None]*frequency)
    delayed = np.fft.ifft(tx_spectrum * delay_phase, axis=1, norm="ortho")
    sample_range = distance_by_symbol[:, None] + radial*np.arange(len(frequency))/fs
    wavelength = C/config.carrier_hz
    # 往返载波相位随距离连续变化，已包含多普勒。符号内包络时延按常量近似。
    carrier_phase = np.exp(-4j*np.pi*sample_range/wavelength)
    amplitude = (config.reference_range_m / distance_by_symbol[:, None])**2
    tx_gain = beam_response(direction[:2], beam_xy, indices, weights)
    echo_at_reference = tx_gain * amplitude * carrier_phase * delayed
    return echo_at_reference, direction, distance, radial


def thermal_noise(rng, shape, power):
    return np.sqrt(power/2) * (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))


def peak_shift(left, middle, right):
    """在对数功率谱上做三点抛物线插值。"""
    left, middle, right = np.log(np.maximum((left, middle, right), 1e-30))
    curvature = left - 2*middle + right
    return float(np.clip(0.5*(left-right)/curvature, -.5, .5)) if curvature < 0 else 0.


def receive(config, pilots, times, echo, direction, indices, rng):
    """距离/多普勒由参考阵元波形估计。测角阵元采用等价的线性处理后噪声。"""
    noisy_iq = echo + thermal_noise(rng, echo.shape, config.sample_noise_power)
    symbol_period = times[1] - times[0]
    fs = config.subcarriers * config.subcarrier_spacing_hz
    slow_window = np.hanning(len(times))
    noisy_channel = np.fft.fft(noisy_iq, axis=1, norm="ortho") / pilots
    range_samples = np.fft.ifft(noisy_channel, axis=1)[:, :config.cp_samples]
    padded_symbols = 4 * len(times)
    fft_bins_hz = np.fft.fftshift(np.fft.fftfreq(padded_symbols, symbol_period))
    spectrum = np.fft.fftshift(
        np.fft.fft(range_samples*slow_window[:, None], n=padded_symbols, axis=0), axes=0
    ) / slow_window.sum()
    power = np.abs(spectrum)**2
    doppler_index, range_index = np.unravel_index(np.argmax(power), power.shape)
    peak = power[doppler_index, range_index]
    peak_to_floor_db = 10*np.log10(peak/np.median(power))
    measured = {
        "detected": bool(peak_to_floor_db >= config.peak_to_floor_threshold_db),
        "echo_peak_db": float(10*np.log10(peak)),
        "peak_to_floor_db": float(peak_to_floor_db),
        "measured_range_m": None, "measured_doppler_hz": None, "measured_radial_mps": None,
        "measured_u": None, "measured_v": None,
        "measured_azimuth_deg": None, "measured_polar_deg": None,
    }
    if measured["detected"]:
        range_delta = 0.
        if 0 < range_index < power.shape[1]-1:
            range_delta = peak_shift(*power[doppler_index, range_index-1:range_index+2])
        doppler_delta = peak_shift(power[(doppler_index-1)%padded_symbols, range_index],
                                   peak, power[(doppler_index+1)%padded_symbols, range_index])
        doppler_hz = fft_bins_hz[doppler_index] + doppler_delta/(padded_symbols*symbol_period)
        measured["measured_range_m"] = float(C*(range_index+range_delta)/(2*fs))
        measured["measured_doppler_hz"] = float(doppler_hz)
        measured["measured_radial_mps"] = float(-doppler_hz*C/(2*config.carrier_hz))

        # 每个测角阵元共享同一无噪距离/多普勒单元，乘自身阵列相位后叠加独立噪声。
        # 只构造该单元的阵元快拍，避免把所有阵元的 IQ 立方体物化。
        clean_channel = np.fft.fft(echo, axis=1, norm="ortho") / pilots
        clean_range_bin = np.fft.ifft(clean_channel, axis=1)[:, range_index]
        coherent = slow_window * np.exp(-2j*np.pi*doppler_hz*times) / slow_window.sum()
        clean_snapshot = coherent @ clean_range_bin
        snapshot_noise_power = config.sample_noise_power * np.sum(abs(coherent)**2)/config.subcarriers
        cuts = clean_snapshot * np.exp(1j*np.pi*indices[:, None]*direction[:2])
        cuts += thermal_noise(rng, cuts.shape, snapshot_noise_power)
        cuts[0] = coherent @ range_samples[:, range_index]
        uv = np.angle(np.sum(cuts[:-1].conj()*cuts[1:], axis=0))/np.pi
        measured["measured_u"], measured["measured_v"] = map(float, uv)
        measured["measured_azimuth_deg"] = float(np.degrees(np.arctan2(uv[1], uv[0])))
        # 当前场景位于 xy 阵列上方，使用正 z 半空间消除平面阵的镜像歧义。
        measured["measured_polar_deg"] = float(np.degrees(np.arcsin(min(1., np.linalg.norm(uv)))))
    return measured, noisy_iq, power, fft_bins_hz


def cases(config, grid, tx_time_s):
    """两个相邻 x 波束在同一 y 行，目标沿 x 穿过两束中心。"""
    beam_a, beam_b = (19, 17), (20, 17)
    u, v = grid[list(beam_a)]
    direction = np.array([u, v, np.sqrt(1-u*u-v*v)])
    position = direction*config.reference_range_m
    cross_time = config.decision_interval_s * 18
    transverse_distance = np.linalg.norm(position[1:])
    end_u = grid[beam_b[0]]
    end_x = end_u * transverse_distance / np.sqrt(1-end_u**2)
    transverse_velocity = np.array([(end_x-position[0])/cross_time, 0., 0.])
    yield "static", 0, 0., beam_a, position, np.zeros(3)
    yield "radial", 0, 0., beam_a, position, 18.*direction
    for frame in range(19):
        for slot, beam in enumerate((beam_a, beam_b)):
            time = frame*config.decision_interval_s + slot*tx_time_s
            yield "crossing", frame, time, beam, position + time*transverse_velocity, transverse_velocity


def evaluate(rows, config):
    static, radial, *crossing = rows
    first_a, first_b = crossing[:2]
    last_a, last_b = crossing[-2:]
    best = [max(crossing[i:i+2], key=lambda row: row["echo_peak_db"])
            for i in range(0, len(crossing), 2)]
    detected = [row for row in best if row["detected"]]
    direction_error = [np.hypot(row["measured_u"]-row["true_u"], row["measured_v"]-row["true_v"])
                       for row in detected]
    range_half_bin = C/(4*config.subcarriers*config.subcarrier_spacing_hz)
    checks = {
        "static_detection_range_velocity": static["detected"]
        and abs(static["measured_range_m"]-static["true_range_m"]) <= range_half_bin
        and abs(static["measured_radial_mps"]) < .25,
        "radial_detection_range_velocity": radial["detected"]
        and abs(radial["measured_range_m"]-radial["true_range_m"]) <= range_half_bin
        and abs(radial["measured_radial_mps"]-radial["true_radial_mps"]) < .5,
        "adjacent_beam_crossing": first_a["echo_peak_db"] > first_b["echo_peak_db"] + 10
        and last_b["echo_peak_db"] > last_a["echo_peak_db"] + 10,
        "crossing_angle_from_echo": len(detected) == len(best) and max(direction_error) < .01,
    }
    return {
        "passed": all(checks.values()), "checks": {key: bool(value) for key, value in checks.items()},
        "static": static, "radial": radial,
        "crossing": {
            "beam_ids": [first_a["beam_id"], first_b["beam_id"]],
            "frames_with_detection_on_stronger_beam": len(detected), "total_frames": len(best),
            "start_beam_a_minus_b_db": first_a["echo_peak_db"] - first_b["echo_peak_db"],
            "end_beam_b_minus_a_db": last_b["echo_peak_db"] - last_a["echo_peak_db"],
            "max_stronger_beam_direction_cosine_error": max(direction_error, default=None),
        },
    }


def plot_results(rows, examples, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"]})
    crossing = rows[2:]
    colors = ("#235789", "#BC542D")
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.6), sharex=True, layout="constrained")
    for color, beam_rows in zip(colors, (crossing[::2], crossing[1::2])):
        time = [row["time_s"] for row in beam_rows]
        label = f'波束 {beam_rows[0]["beam_id"]}（编号从 1 算）'
        axes[0].plot(time, [row["echo_peak_db"] for row in beam_rows], "o-", ms=3, color=color, label=label)
        visible = [row for row in beam_rows if row["detected"]]
        axes[1].scatter([row["time_s"] for row in visible],
                        [row["measured_u"] for row in visible], s=19, facecolors="none",
                        edgecolors=color, label=label+"检测")
    axes[1].plot([row["time_s"] for row in crossing], [row["true_u"] for row in crossing],
                 color="#242930", linewidth=1.4, label="参考方向")
    axes[0].set(title="单目标横向跨越相邻波束", ylabel="回波谱峰（dB，相对仿真参考）")
    axes[1].set(xlabel="实际扫描时刻（秒）", ylabel="x 方向余弦 u")
    for axis in axes:
        axis.grid(alpha=.22)
        axis.legend(loc="best")
    fig.savefig(output/FIGURES[0], dpi=160)

    rd, panels = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True, layout="constrained")
    distance_bins = C*np.arange(config.cp_samples)/(2*config.subcarriers*config.subcarrier_spacing_hz)
    for axis, (row, power, doppler) in zip(panels, examples):
        velocity_bins = (-doppler*C/(2*config.carrier_hz))[::-1]
        velocity_take = abs(velocity_bins) <= 33
        dist_take = (distance_bins >= 100) & (distance_bins <= 200)
        region = power[::-1][np.ix_(velocity_take, dist_take)]
        mesh = axis.pcolormesh(distance_bins[dist_take], velocity_bins[velocity_take],
                               10*np.log10(region/np.max(power)), shading="nearest",
                               vmin=-40, vmax=0, cmap="viridis")
        axis.scatter(row["measured_range_m"], row["measured_radial_mps"], c="white",
                     marker="+", s=70, label="回波估计值")
        axis.set(title="静止目标" if row["case"] == "static" else "径向远离目标", xlabel="距离（米）")
        axis.legend(loc="upper right")
    panels[0].set_ylabel("径向速度（米/秒，远离为正）")
    rd.colorbar(mesh, ax=panels, label="归一化谱峰功率（dB）")
    rd.savefig(output/FIGURES[1], dpi=160)
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="保存后打开两张结果图")
    args = parser.parse_args()
    config = ScenarioConfig()
    rng = np.random.default_rng(config.seed)
    pilots, tx, frequency, times = make_probe(config, rng)
    tx_spectrum = np.fft.fft(tx[:, config.cp_samples:], axis=1, norm="ortho")
    indices = np.arange(32)
    grid = -1 + 2*(indices+.5)/len(indices)
    weights = np.exp(1j*np.pi*indices[:, None]*grid)/np.sqrt(len(indices))
    burst_time = tx.size/(config.subcarriers*config.subcarrier_spacing_hz)

    from .common import finish_run, new_run, write_json
    from .versions import save_version

    output = new_run(BASE/"outputs"/VERSION)
    rows, examples, example_iq = [], [], []
    for name, frame, time, beam, position, velocity in cases(config, grid, burst_time):
        echo, direction, true_range, true_radial = propagate(
            config, tx_spectrum, frequency, times, position, velocity, beam, indices, weights
        )
        observed, iq, power, doppler = receive(config, pilots, times, echo, direction, indices, rng)
        row = {"case": name, "frame": frame, "time_s": time, "beam_id": beam[1]*32+beam[0]+1,
               "true_range_m": float(true_range), "true_radial_mps": float(true_radial),
               "true_delay_us": float(2e6*true_range/C),
               "true_doppler_hz": float(-2*true_radial*config.carrier_hz/C),
               "true_u": float(direction[0]), "true_v": float(direction[1]), **observed}
        rows.append(row)
        if name in {"static", "radial"}:
            examples.append((row, power, doppler))
            example_iq.append(iq)
    summary = evaluate(rows, config)
    with (output/"echo_measurements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output/"ofdm_iq_examples.npz", tx_with_cp=tx, known_pilots=pilots,
                        static_rx_after_cp=example_iq[0], radial_rx_after_cp=example_iq[1])
    write_json(output/"summary.json", {
        "version": VERSION, "config": asdict(config), "burst_duration_s": burst_time,
        "units": {"echo_peak_db": "relative to aligned reference array element signal power"},
        "model_scope": [
            "单站单目标、自由空间点散射，强度按参考距离归一化，噪声为复白噪声",
            "32×32 半波长平面发射阵列，接收方向已知位于阵列正半空间",
            "距离和多普勒从参考阵元的复数采样估计，其余阵元在已检测单元生成等价线性处理后的快拍",
            "预设两束依次发射，用于波形前端开环验证",
        ], **summary,
    })
    plot_results(rows, examples, config, output)
    print(f'结果目录：{output.resolve()}', flush=True)
    if not summary["passed"]:
        raise SystemExit(f'物理校验未全部通过：{summary["checks"]}')
    archive = save_version(f"{VERSION}_{output.name[4:]}", output,
                           "对比实验第1步，单目标 OFDM 回波和横向跨束校验。", asset_keys=())
    finish_run(output)
    print(f'物理校验 4/4 通过。归档：{archive.resolve()}', flush=True)
    if args.show:
        import os
        for figure in FIGURES:
            os.startfile(output/figure)
