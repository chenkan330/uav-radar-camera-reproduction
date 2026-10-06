"""Synthetic asynchronous radar/cloud and camera/box replay validation."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from camera import CameraIntrinsics, CameraExtrinsics, camera_project
from demo_step2 import make_clouds
from fusion import FusionTracker, FusionConfig
from plot_style import configure_chinese_font, MODE_LABELS


def make_events(seed=42, duration=12.):
    radar_times, truth, clouds = make_clouds(seed=seed, duration=duration)
    intrinsics, pose = CameraIntrinsics(420., 420., 320., 240.), CameraExtrinsics()
    rng = np.random.default_rng(seed + 1000)
    events = []
    for k, (timestamp, cloud) in enumerate(zip(radar_times, clouds)):
        delay = 0. if k < 2 else (.08 if k % 11 == 0 else .006)
        events.append(dict(sensor="radar", timestamp=float(timestamp),
                           arrival_time=float(timestamp + delay), event_id=f"r{k:05d}", cloud=cloud))
    for k, timestamp in enumerate(np.arange(0., duration + 1e-9, .08)):
        if 4. <= timestamp <= 4.4:
            continue  # Independent camera outage.
        xyz = np.array([np.interp(timestamp, radar_times, truth[:, axis]) for axis in (0, 2, 4)])
        pixels = camera_project(xyz, intrinsics, pose) + rng.normal(0., 2., 2)
        box = np.r_[pixels - [12., 10.], pixels + [12., 10.]]
        delay = .16 if k % 13 == 0 else .04 + .012 * (1 + np.sin(k))
        events.append(dict(sensor="camera", timestamp=float(timestamp),
                           arrival_time=float(timestamp + delay), event_id=f"c{k:05d}",
                           bbox=box, confidence=.65 if k % 9 == 0 else .95))
    return events, np.column_stack((radar_times, truth[:, [0, 2, 4]])), intrinsics, pose


def run_events(events, intrinsics, pose, config=None, acquisition_order=False, end_time=None):
    tracker = FusionTracker(intrinsics, pose, config)
    ordered = sorted(events, key=lambda e: (e["timestamp"] if acquisition_order else e["arrival_time"],
                                          e["timestamp"], 0 if e["sensor"] == "radar" else 1, e["event_id"]))
    for event in ordered:
        payload = dict(event)
        if acquisition_order:
            payload["arrival_time"] = payload["timestamp"]
        tracker.ingest(**payload)
    final_time = end_time if end_time is not None else max(e["arrival_time"] for e in events) + .05
    tracker.advance_to(final_time)
    return tracker


def run_demo(seed=42, duration=12., mode="paper_xyz", output=Path("output/step4"), plots=True):
    events, truth, intrinsics, pose = make_events(seed, duration)
    config = FusionConfig(camera_mode=mode)
    end_time = max(e["arrival_time"] for e in events) + .05
    delayed = run_events(events, intrinsics, pose, config, end_time=end_time)
    ordered = run_events(events, intrinsics, pose, config, True, end_time)
    t, x, P = delayed.tick_history()
    ref_t, ref_x, ref_P = ordered.tick_history()
    if not np.array_equal(t, ref_t):
        raise AssertionError("replay prediction timeline differs from ordered reference")
    dx, dP = float(np.max(np.abs(x-ref_x))), float(np.max(np.abs(P-ref_P)))
    np.testing.assert_allclose(x, ref_x, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(P, ref_P, atol=1e-12, rtol=1e-12)
    summary = dict(step=4, seed=seed, data_kind="controlled_synthetic", camera_mode=mode,
                   prediction_hz=config.prediction_hz, ticks=len(t),
                   replay_vs_ordered_max_state_difference=dx,
                   replay_vs_ordered_max_covariance_difference=dP,
                   minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(P).min()),
                   maximum_covariance_asymmetry=float(np.max(np.abs(P-P.transpose(0,2,1)))),
                   diagnostics=delayed.diagnostics(),
                   assumptions=dict(radar_hz=10.,camera_hz=12.5,pixel_std_px=2.,
                       radar_R=config.radar_R.tolist(),gate=config.gate,beta0=config.beta0,
                       acceleration_std_m_s2=config.acceleration_std,pseudo_depth_std_m=config.depth_std),
                   caveat="Synthetic boxes, no trained detector or field accuracy. History is retrospectively corrected. paper_xyz uses correlated state depth; bearing is an EKF extension.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (output/"timeline.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream)
        writer.writerow(["timestamp_s","x_m","vx_mps","y_m","vy_mps","z_m","vz_mps",
                         "Pxx","Pvxvx","Pyy","Pvyvy","Pzz","Pvzvz"])
        for timestamp,state,covariance in zip(t,x,P):
            writer.writerow([timestamp,*state,*np.diag(covariance)])
    trace=[dict(timestamp_s=e.timestamp,sensor=e.sensor,event_id=e.key,diagnostics=e.diagnostics)
           for e in delayed.items if e.sensor!="tick"]
    (output/"update_trace.json").write_text(json.dumps(trace,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if plots:
        os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).parent/".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        configure_chinese_font()
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(2,2,figsize=(12,7),layout="constrained")
        for axis,index in zip(axes.flat[:3],range(3)):
            axis.plot(truth[:,0],truth[:,index+1],"k",label="合成真实轨迹")
            axis.plot(t,x[:,index*2],label="迟到数据回放",alpha=.85)
            axis.plot(ref_t,ref_x[:,index*2],"--",label="按采集时间处理",alpha=.65)
            axis.set(xlabel="时间（秒）",ylabel=f"{'XYZ'[index]} 位置（米）",
                     title=f"{'XYZ'[index]} 轴轨迹对比")
            axis.grid(alpha=.2)
        axes.flat[0].legend(fontsize=9)
        for sensor in ("radar","camera"):
            selected=[e for e in events if e["sensor"]==sensor]
            axes.flat[3].scatter([e["timestamp"] for e in selected],
                [1000*(e["arrival_time"]-e["timestamp"]) for e in selected],s=8,
                label={"radar":"雷达","camera":"相机"}[sensor])
        axes.flat[3].set(xlabel="采集时间（秒）",ylabel="延迟（毫秒）",title="传感器数据到达延迟")
        axes.flat[3].legend()
        fig.suptitle(f"第4步：100赫兹回放与按采集时间处理一致（{MODE_LABELS[mode]}，合成数据）")
        fig.savefig(output/"replay.png",dpi=140)
        plt.close(fig)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--duration",type=float,default=12.)
    parser.add_argument("--camera-mode",choices=("paper_xyz","bearing"),default="paper_xyz")
    parser.add_argument("--output",type=Path,default=Path("output/step4"))
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run_demo(args.seed,args.duration,args.camera_mode,args.output,not args.no_plots),indent=2))
