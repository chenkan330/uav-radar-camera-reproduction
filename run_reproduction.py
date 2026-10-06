"""Step 5: timestamped CSV replay, association comparison and paper metrics.

Choose --synthetic for controlled validation or --data DIRECTORY for supplied
logs. Ground truth is passed only to the evaluator, never to FusionTracker.
"""
import argparse
import csv
from dataclasses import fields
import json
import os
from pathlib import Path
import numpy as np
from camera import CameraIntrinsics, CameraExtrinsics
from dataset_io import (read_dataset, RADAR_COLUMNS, RADAR_FRAME_COLUMNS,
                        CAMERA_COLUMNS, TRUTH_COLUMNS)
from demo_step4 import make_events, run_events
from evaluation import evaluate_positions
from fusion import FusionConfig, FusionTracker
from radar import METHODS
from plot_style import (configure_chinese_font, METHOD_LABELS, MODE_LABELS,
                        DATA_KIND_LABELS, dataset_label)


def example_config():
    return dict(data_kind="synthetic", source="controlled fixture, not author flight logs",
        coordinate_frame="global", radar_velocity="cartesian_vx",
        intensity_scale="linear_nonnegative", timestamps="seconds_shared_clock",
        camera=dict(intrinsics=dict(fu=420.,fv=420.,u0=320.,v0=240.),
                    rotation=[[0,0,1],[-1,0,0],[0,-1,0]],translation=[0,0,0]),
        fusion=dict(prediction_hz=100.,gate=3.,beta0=.1,acceleration_std=.8,
                    velocity_std=2.,initialization_speed=.3,pixel_std=2.,
                    depth_std=.5,confidence_threshold=.7,camera_mode="paper_xyz",
                    radar_R=np.diag(np.array([.2,.35,.35,.12])**2).tolist()),
        evaluation=dict(max_interpolation_gap_s=.2))


def parse_config(config,require_camera=True):
    if not isinstance(config,dict):raise ValueError("config must be a JSON object")
    requirements=dict(coordinate_frame="global",radar_velocity="cartesian_vx",
                      intensity_scale="linear_nonnegative",timestamps="seconds_shared_clock")
    for key,value in requirements.items():
        if config.get(key)!=value:
            raise ValueError(f"config requires {key}={value!r}; convert/calibrate inputs explicitly")
    if config.get("data_kind") not in ("synthetic","real"):
        raise ValueError("data_kind must explicitly be synthetic or real")
    camera=config.get("camera",{})
    if not isinstance(camera,dict):raise ValueError("camera must be a JSON object")
    if not require_camera and not camera:
        # Unused construction placeholders, not guessed camera calibration.
        intrinsics,pose=CameraIntrinsics(1.,1.,0.,0.),CameraExtrinsics()
    elif not all(key in camera for key in ("intrinsics","rotation","translation")):
        raise ValueError("explicit camera intrinsics, rotation and translation are required")
    else:
        try:
            intrinsics=CameraIntrinsics(**camera["intrinsics"])
            pose=CameraExtrinsics(camera["rotation"],camera["translation"])
        except (TypeError,ValueError) as error:
            raise ValueError("camera parameters must contain valid calibrated numbers") from error
    if not isinstance(config.get("fusion",{}),dict):raise ValueError("fusion must be a JSON object")
    if not isinstance(config.get("evaluation",{}),dict):raise ValueError("evaluation must be a JSON object")
    settings=dict(config.get("fusion",{}))
    unknown=set(settings)-{f.name for f in fields(FusionConfig)}
    if unknown:raise ValueError(f"unknown fusion fields: {sorted(unknown)}")
    for name,value in settings.items():
        if name not in ("radar_R","radar_method","camera_mode") and (isinstance(value,bool) or not isinstance(value,(int,float))):
            raise ValueError(f"fusion.{name} must be a number")
    if "radar_R" in settings:settings["radar_R"]=np.asarray(settings["radar_R"],dtype=float)
    FusionTracker(intrinsics,pose,FusionConfig(**settings))  # Validate before calculating clock periods.
    return intrinsics,pose,settings


def write_fixture(directory,seed,duration=12.):
    """Generate CSV then load it through the same parser as supplied data."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    events,truth,_,_=make_events(seed,duration)
    (directory/"config.json").write_text(json.dumps(example_config(),indent=2)+"\n",encoding="utf-8")
    def write(name,header,rows):
        with (directory/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    radar=[e for e in events if e["sensor"]=="radar"]
    camera=[e for e in events if e["sensor"]=="camera"]
    write("radar.csv",RADAR_COLUMNS,([e["timestamp"],e["arrival_time"],e["event_id"],*point]
                                       for e in radar for point in e["cloud"]))
    write("radar_frames.csv",RADAR_FRAME_COLUMNS,
          ([e["timestamp"],e["arrival_time"],e["event_id"]] for e in radar))
    write("camera.csv",CAMERA_COLUMNS,
          ([e["timestamp"],e["arrival_time"],e["event_id"],*e["bbox"],e["confidence"]] for e in camera))
    write("ground_truth.csv",TRUTH_COLUMNS,truth)
    return read_dataset(directory)


def run_dataset(dataset,label,modes=None):
    intrinsics,pose,settings=parse_config(dataset["config"],bool(dataset["camera"]))
    modes=modes or [settings.get("camera_mode","paper_xyz")]
    all_events=[dict(e,sensor="radar") for e in dataset["radar"]]+[dict(e,sensor="camera") for e in dataset["camera"]]
    if not all_events:raise ValueError("dataset has no radar or camera frames")
    end_time=max(e["arrival_time"] for e in all_events)+1/settings.get("prediction_hz",100.)
    variants=[("none","radar_only","average",False)]
    variants += [(mode,method,method,True) for mode in modes for method in METHODS]
    reports=[];tracks={}
    for mode,name,method,use_camera in variants:
        parameters=dict(settings,radar_method=method,camera_mode="paper_xyz" if mode=="none" else mode)
        tracker=run_events(all_events if use_camera else [e for e in all_events if e["sensor"]=="radar"],
                           intrinsics,pose,FusionConfig(**parameters),end_time=end_time)
        t,x,P=tracker.tick_history()
        metrics=evaluate_positions(t,x[:,[0,2,4]],dataset["ground_truth"],
            dataset["config"].get("evaluation",{}).get("max_interpolation_gap_s",.2))
        report=dict(dataset=label,camera_mode=mode,method=name,metrics=metrics,
                    camera_input_count=len(dataset["camera"]) if use_camera else 0,
                    diagnostics=tracker.diagnostics(),
                    minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(P).min()) if len(P) else None,
                    maximum_covariance_asymmetry=float(np.max(np.abs(P-P.transpose(0,2,1)))) if len(P) else None)
        reports.append(report);tracks[(mode,name)]=(t,x)
    # The paper uses one cluster, so identical centroids must give identical tracks.
    for mode in modes:
        np.testing.assert_allclose(tracks[(mode,"average")][1],tracks[(mode,"kmeans")][1],atol=1e-12,rtol=1e-12)
    return reports,tracks


def save_report(output,reports,representative_tracks,truth,config,plots=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    aggregates=[]
    for mode,name in sorted({(r["camera_mode"],r["method"]) for r in reports}):
        selected=[r for r in reports if r["camera_mode"]==mode and r["method"]==name]
        values=[r["metrics"]["mean_euclidean_m"] for r in selected if r["metrics"]["mean_euclidean_m"] is not None]
        aggregates.append(dict(camera_mode=mode,method=name,datasets=len(selected),
            evaluated_datasets=len(values),mean_of_mean_euclidean_m=float(np.mean(values)) if values else None,
            std_of_mean_euclidean_m=float(np.std(values)) if values else None))
    summary=dict(step=5,data_kind=config["data_kind"],config=config,runs=reports,aggregates=aggregates,
        executed_camera_modes=sorted({r["camera_mode"] for r in reports if r["camera_mode"]!="none"}),
        camera_stream_present=any(r["camera_input_count"]>0 for r in reports),
        paper_field_reproduction=False,
        completed="algorithm, geometry, asynchronous replay, CSV interface and synthetic evaluation",
        pending="author flight logs/calibration, detector training images and weights, ROS/Coral/hardware validation",
        caveats=["Mean Euclidean error and signed XYZ means are Table I metrics; RMSE is separately labelled.",
                 "No extrapolation of ground truth or interpolation across long gaps.",
                 "Tracks are retrospectively corrected histories; not causal published trajectories.",
                 "paper_xyz copies correlated state depth; bearing is an EKF extension.",
                 "Synthetic seeds are algorithm fixtures, not the author's two flight datasets."])
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    with (output/"representative_tracks.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["camera_mode","method","timestamp_s","x_m","y_m","z_m"])
        for (mode,name),(t,x) in representative_tracks.items():
            for timestamp,state in zip(t,x):writer.writerow([mode,name,timestamp,*state[[0,2,4]]])
    lines=["# 第5步：统一复现报告","",f"数据类型：{DATA_KIND_LABELS[config['data_kind']]}。本报告不代表作者真实飞行数据复现。",
           "","本步完成CSV导入、五方法比较、雷达单传感器基线、时间对齐评价和结果保存。",
           "","| 数据 | 相机模式 | 方法 | 平均欧氏误差（米） | 三维均方根误差（米） | 真值匹配率 |",
           "|---|---|---|---:|---:|---:|"]
    def number(value):return "无有效真值" if value is None else f"{value:.6f}"
    for report in reports:
        m=report["metrics"]
        lines.append(f"| {dataset_label(report['dataset'])} | {MODE_LABELS[report['camera_mode']]} | {METHOD_LABELS[report['method']]} | {number(m['mean_euclidean_m'])} | {number(m['rmse3d_m'])} | {m['coverage_fraction']:.1%} |")
    lines += ["","各轴有符号平均误差、匹配时间范围、协方差和迟到事件统计见summary.json。",
              "真实飞行精度、MobileNet V2训练/mAP和Coral/ROS硬件运行仍需原始数据、模型及设备。",
              "论文三维观测模式使用相关状态深度；像素方位观测为另行标明的改进。代表性轨迹是回放修正的历史。"]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if plots:
        os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).parent/".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        configure_chinese_font()
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
        plotted_mode="paper_xyz" if any(mode=="paper_xyz" for mode,_ in representative_tracks) else "bearing"
        for axis,index in zip(axes.flat[:3],range(3)):
            if truth is not None and len(truth):axis.plot(truth[:,0],truth[:,index+1],"k",lw=2,label="参考真值")
            for (mode,name),(t,x) in representative_tracks.items():
                if mode==plotted_mode or mode=="none":axis.plot(t,x[:,index*2],label=METHOD_LABELS[name],alpha=.75)
            axis.set(xlabel="时间（秒）",ylabel=f"{'XYZ'[index]}位置（米）");axis.grid(alpha=.2)
        axes.flat[0].legend(ncol=2,fontsize=8)
        valid=[a for a in aggregates if a["mean_of_mean_euclidean_m"] is not None]
        if valid:
            labels=[METHOD_LABELS[a["method"]] if a["camera_mode"]=="none" else
                    METHOD_LABELS[a["method"]]+"｜"+MODE_LABELS[a["camera_mode"]] for a in valid]
            axes.flat[3].barh(range(len(valid)),[a["mean_of_mean_euclidean_m"] for a in valid],
                xerr=[a["std_of_mean_euclidean_m"] for a in valid],capsize=3)
            axes.flat[3].set_yticks(range(len(valid)),labels,fontsize=8)
            axes.flat[3].invert_yaxis()
            axes.flat[3].set(xlabel="平均欧氏误差（米）",title="跨数据组均值与总体标准差")
        else:axes.flat[3].text(.5,.5,"无匹配真值，无法评价定位精度",ha="center")
        modes_title="与".join(MODE_LABELS[mode] for mode in summary["executed_camera_modes"])
        fig.suptitle(f"第5步：五种关联方法与仅雷达基线（{DATA_KIND_LABELS[config['data_kind']]}；{modes_title}）")
        fig.savefig(output/"comparison.png",dpi=140);plt.close(fig)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic",action="store_true")
    source.add_argument("--data",type=Path)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--duration",type=float,default=12.)
    parser.add_argument("--camera-mode",choices=("paper_xyz","bearing","both"))
    parser.add_argument("--output",type=Path,default=Path("output/local_reproduction"))
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    reports=[];tracks={};truth=None;config=None
    modes=["paper_xyz","bearing"] if args.camera_mode=="both" or (args.synthetic and args.camera_mode is None) else ([args.camera_mode] if args.camera_mode else None)
    if args.synthetic:
        if len(args.seeds)!=len(set(args.seeds)):parser.error("seeds must be unique")
        for i,seed in enumerate(args.seeds):
            dataset=write_fixture(Path("tmp/step5")/f"seed{seed}",seed,args.duration)
            new_reports,new_tracks=run_dataset(dataset,f"synthetic_seed{seed}",modes)
            reports.extend(new_reports)
            if i==0:tracks,truth,config=new_tracks,dataset["ground_truth"],dataset["config"]
    else:
        dataset=read_dataset(args.data)
        reports,tracks=run_dataset(dataset,args.data.name,modes)
        truth,config=dataset["ground_truth"],dataset["config"]
    summary=save_report(args.output,reports,tracks,truth,config,not args.no_plots)
    print(f"Saved {len(reports)} runs to {args.output}; data_kind={summary['data_kind']}")
    print("Algorithm validation only. Field/detector/hardware reproduction requires the original materials.")


if __name__=="__main__":
    try:main()
    except (ValueError,OSError) as error:
        raise SystemExit(f"Cannot run reproduction: {error}") from error
