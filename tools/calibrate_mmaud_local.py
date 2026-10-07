"""Experimental, local MMAUD calibration from explicitly disjoint training intervals.

The official fisheye K/D and rig extrinsics have not been obtained.  This tool
does NOT create a full-fisheye calibration.  Pixel annotations must be obtained
from images alone. Leica ground truth is used only inside the locked training
intervals for rigid-frame, local camera-projection and noise fitting. It does
not supply pixel labels, radar associations, evaluation initialization or tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmaud_adapter import (TIME_ORIGIN_NS, CALIBRATION_INTERVALS_UNIX,
                          build_background, radar_frames_from_files)
DEFAULT_DATA = ROOT / "data/public/mmaud/Mavic3"
DEFAULT_OUT = ROOT / "data/public/mmaud/calibration"
INTERVALS = CALIBRATION_INTERVALS_UNIX
# Manually selected central fuselage positions from training image pixels only.
# Neither radar nor Leica positions were projected to choose these regions.
MANUAL_BODY_SEEDS = {
    6908: (602, 537), 6909: (619, 604), 6910: (632, 507),
    6911: (638, 410), 6912: (655, 416), 6913: (700, 487),
    6914: (699, 539), 6915: (651, 560), 6916: (586, 532),
    6917: (572, 514), 6918: (602, 497), 6919: (620, 495),
    6920: (623, 458), 6921: (614, 359), 6922: (607, 255),
    6923: (607, 177), 7012: (612, 373), 7013: (612, 402),
    7014: (612, 431), 7015: (614, 460), 7016: (614, 491),
    7017: (619, 502), 7018: (617, 510), 7019: (608, 519),
    7020: (597, 525), 7021: (601, 535), 7022: (612, 539),
    7023: (618, 523), 7024: (620, 503), 7025: (622, 490),
    7026: (626, 516),
}


def annotate_training(data: Path, out: Path):
    folder = out / "training_annotations_review"
    folder.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for t, path in timed_files(data / "image", ".png"):
        if not in_intervals(t):
            continue
        sec = int(t) - 1692840000
        if sec not in MANUAL_BODY_SEEDS:
            continue
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        u0, v0 = MANUAL_BODY_SEEDS[sec]
        radius = 35 if sec < 7000 else 9
        x0, y0 = u0-radius, v0-radius
        patch = gray[y0:v0+radius+1, x0:u0+radius+1]
        response = cv2.morphologyEx(patch, cv2.MORPH_BLACKHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        # Local photometric refinement inside an image-only manual body region.
        peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        mask = ((xx-peak_x)**2 + (yy-peak_y)**2 <= 3**2)
        weights = np.maximum(response.astype(float) - float(response.max())*.35, 0) * mask
        if sec < 7000:
            # Clear near drone: 3x3 opening removes slender arms; the largest
            # dark component is the fuselage. This threshold is image-only.
            dark = cv2.morphologyEx(np.uint8(patch<95)*255, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(dark)
            if count < 2:
                raise ValueError("No central fuselage component")
            label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            weights=np.maximum(95-patch.astype(float),0)*(labels==label)
        if weights.sum() < 1:
            raise ValueError(f"No visible dark body in manually reviewed region {path.name}")
        u, v = x0 + float((xx*weights).sum()/weights.sum()), y0 + float((yy*weights).sum()/weights.sum())
        rows.append({"timestamp": t, "source_image": path.name, "u": u, "v": v,
                     "method": "image_only_manual_fuselage_region_local_blackhat_centroid",
                     "sha256": digest(path), "manual_seed_u": u0, "manual_seed_v": v0,
                     "bbox_x": x0, "bbox_y": y0, "bbox_width": 2*radius+1,
                     "bbox_height": 2*radius+1, "confidence": "visually_identifiable",
                     "reference_point": "visible_dark_fuselage_not_guaranteed_Leica_prism"})
        x1, y1 = max(0, int(u)-40), max(0, int(v)-40)
        crop = im[y1:y1+80, x1:x1+80].copy()
        cv2.drawMarker(crop, (int(round(u))-x1, int(round(v))-y1), (0,0,255),
                       cv2.MARKER_CROSS, 7, 1)
        tile = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, str(sec), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 1)
        cv2.imwrite(str(folder / f"{path.stem}.png"), tile)
        tiles.append(tile)
    with (out / "annotations_training.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for first in range(0, len(tiles), 12):
        page = tiles[first:first+12]
        while len(page) % 4:
            page.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(page[i:i+4]) for i in range(0, len(page), 4)])
        cv2.imwrite(str(folder / f"sheet_{first:02d}.png"), sheet)
    print(json.dumps({"training_annotations": len(rows), "csv": str(out/"annotations_training.csv")}))


def rigid_fit(a, b, weights=None):
    if weights is None:
        weights=np.ones(len(a))
    weights=np.asarray(weights,float); weights=weights/weights.sum()
    ca, cb = weights@a, weights@b
    u, s, vt = np.linalg.svd((a-ca).T @ ((b-cb)*weights[:,None]))
    r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    t = cb-r@ca
    residual = a@r.T+t-b
    return r, t, residual, s


def robust_rigid_fit(a,b,huber_threshold_m=.5):
    weights=np.ones(len(a))
    for _ in range(50):
        r,t,residual,s=rigid_fit(a,b,weights)
        lengths=np.linalg.norm(residual,axis=1)
        next_weights=np.minimum(1.,huber_threshold_m/np.maximum(lengths,1e-12))
        if np.max(abs(next_weights-weights))<1e-8:
            break
        weights=next_weights
    return r,t,residual,s,weights


def camera_fit(points, pixels, initial_f):
    k = np.array([[initial_f,0,700.], [0,initial_f,480.], [0,0,1.]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
        [points.astype(np.float32)], [pixels.astype(np.float32)], (1280,960),
        k, np.zeros(5), flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-10))
    r = cv2.Rodrigues(rvecs[0])[0]
    t = tvecs[0].ravel()
    projected = cv2.projectPoints(points, rvecs[0], tvecs[0], k, np.zeros(5))[0][:,0]
    residual = projected - pixels
    return {"rms_pixels": float(rms), "K": k, "R_reference_to_camera": r,
            "t_reference_to_camera": t, "residual_pixels": residual,
            "depths_camera": (points@r.T+t)[:,2]}


def fit_training(data: Path, out: Path, bootstrap_count: int, camera_reference: str):
    if bootstrap_count < 10:
        raise ValueError("At least 10 bootstrap fits are required for model diagnostics")
    rows = list(csv.DictReader((out/"annotations_training.csv").open(encoding="utf-8")))
    radar_files = [p for _,p in timed_files(data/"radar_enhance_pcl", ".npy")]
    background = build_background(radar_files)
    observations, radar_audit = radar_frames_from_files(radar_files, background, INTERVALS)
    rt = np.array([o["timestamp_s"]+TIME_ORIGIN_NS/1e9 for o in observations])
    gtfiles = [(t,p) for t,p in timed_files(data/"ground_truth", ".npy") if in_intervals(t)]
    gt_times = np.array([t for t,_ in gtfiles])
    gt_xyz = np.array([np.load(p, allow_pickle=False) for _,p in gtfiles])
    # No reads of the ground-truth arrays in the held-out interval.
    rp, tp, pairing = [], [], []
    for o, t in zip(observations, rt):
        j = int(np.searchsorted(gt_times, t))
        if j==0 or j==len(gt_times) or gt_times[j]-gt_times[j-1] > .30:
            continue
        if not any(a <= gt_times[j-1] <= t <= gt_times[j] <= b for a,b in INTERVALS):
            continue
        alpha=(t-gt_times[j-1])/(gt_times[j]-gt_times[j-1])
        rp.append(o["median_xyz"]); tp.append(gt_xyz[j-1]*(1-alpha)+gt_xyz[j]*alpha)
        pairing.append({"radar_file":o["path"].name,"timestamp_unix":float(t),
                        "gt_left_file":gtfiles[j-1][1].name,"gt_right_file":gtfiles[j][1].name})
    rp, tp = np.array(rp), np.array(tp)
    ordinary_r,ordinary_t,ordinary_residual,_=rigid_fit(rp,tp)
    r_truth,t_truth,residual_truth,sv_truth,rigid_weights=robust_rigid_fit(rp,tp)
    xyz, uv, correspondences = [], [], []
    used_snapshots = set()
    for row in rows:
        t = float(row["timestamp"])
        if not in_intervals(t):
            raise ValueError("Annotation outside locked calibration intervals")
        image_path=data/"image"/row["source_image"]
        if digest(image_path) != row["sha256"]:
            raise ValueError("Annotation image hash mismatch")
        if camera_reference=="radar":
            j=int(np.argmin(abs(rt-t)))
            if abs(rt[j]-t) > .15 or j in used_snapshots:
                continue
            used_snapshots.add(j)
            point=observations[j]["median_xyz"]
            metadata={"radar_file":observations[j]["path"].name,
                      "radar_timestamp_unix":float(rt[j]),"time_difference_s":float(rt[j]-t),
                      "radar_xyz":point.tolist()}
        else:
            j=int(np.searchsorted(gt_times,t))
            if (0<j<len(gt_times) and gt_times[j]-gt_times[j-1]<=.30 and
                any(a<=gt_times[j-1]<=t<=gt_times[j]<=b for a,b in INTERVALS)):
                alpha=(t-gt_times[j-1])/(gt_times[j]-gt_times[j-1])
                point=gt_xyz[j-1]*(1-alpha)+gt_xyz[j]*alpha
                metadata={"gt_left_file":gtfiles[j-1][1].name,"gt_right_file":gtfiles[j][1].name,
                          "gt_pair_method":"training_only_linear_interpolation","interpolation_fraction":float(alpha)}
            else:
                j=int(np.argmin(abs(gt_times-t)))
                if abs(gt_times[j]-t)>.15:
                    continue
                point=gt_xyz[j]
                metadata={"gt_left_file":gtfiles[j][1].name,"gt_right_file":gtfiles[j][1].name,
                          "gt_pair_method":"training_boundary_nearest_no_extrapolation",
                          "time_difference_s":float(gt_times[j]-t)}
            metadata["truth_xyz"]=point.tolist()
        xyz.append(point);uv.append([float(row["u"]),float(row["v"])])
        correspondences.append({**row,**metadata})
    xyz,uv=np.array(xyz),np.array(uv)
    fits=[]
    for focal in (250.,400.,600.,900.):
        try:
            fit=camera_fit(xyz,uv,focal); fit["initial_focal_pixels"]=focal; fits.append(fit)
        except cv2.error as exc:
            print(str(exc))
    if not fits:
        raise ValueError("All camera fitting starts failed")
    best=min(fits,key=lambda f:f["rms_pixels"])
    generator=np.random.default_rng(160016)
    boot=[]
    for _ in range(bootstrap_count):
        # Stratify by the two predetermined depth intervals.
        indices=[]
        for a,b in INTERVALS:
            choices=np.array([i for i,c in enumerate(correspondences) if a<=float(c["timestamp"])<=b])
            indices.extend(generator.choice(choices,len(choices),replace=True))
        try:
            alternatives=[]
            for initial in (250.,float(best["K"][0,0]),900.):
                try:
                    alternatives.append(camera_fit(xyz[indices],uv[indices],initial))
                except cv2.error:
                    continue
            f=min(alternatives,key=lambda item:item["rms_pixels"])
            q=xyz@f["R_reference_to_camera"].T+f["t_reference_to_camera"]
            predicted=(q[:,:2]/q[:,2:3])@np.diag([f["K"][0,0],f["K"][1,1]])+f["K"][:2,2]
            boot.append({"rms_pixels":f["rms_pixels"],"K":f["K"].tolist(),
                         "virtual_optical_center_in_fit_reference":(-f["R_reference_to_camera"].T@f["t_reference_to_camera"]).tolist(),
                         "R_reference_to_camera":f["R_reference_to_camera"].tolist(),
                         "t_reference_to_camera":f["t_reference_to_camera"].tolist(),
                         "calibration_prediction_rms_pixels":float(np.sqrt(np.mean(np.sum((predicted-uv)**2,axis=1))))})
        except (cv2.error,ValueError):
            continue
    reference_camera_center=-best["R_reference_to_camera"].T@best["t_reference_to_camera"]
    if camera_reference=="radar":
        camera_center=reference_camera_center
        camera_to_radar_rotation=best["R_reference_to_camera"].T
    else:
        camera_center=r_truth.T@(reference_camera_center-t_truth)
        camera_to_radar_rotation=r_truth.T@best["R_reference_to_camera"].T
    singular=np.linalg.svd(xyz-xyz.mean(axis=0),compute_uv=False)
    noise_pixels=np.maximum(np.std(best["residual_pixels"],axis=0,ddof=1),3.)
    noise_norm=noise_pixels/np.array([best["K"][0,0],best["K"][1,1]])
    def serial_fit(f):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()}
    prediction_rms=[f["calibration_prediction_rms_pixels"] for f in boot]
    # Fixed probes from the TRAINING coordinate domain only; not test points.
    # They check interpolation between the two calibration depth planes.
    probes=np.array([[x,y,z] for z in (8.,10.,12.) for x in (-.5,0.,.5) for y in (-.5,0.,.5)])
    def pixel_projection(x,f):
        rr=np.asarray(f["R_reference_to_camera"]);tt=np.asarray(f["t_reference_to_camera"]);kk=np.asarray(f["K"])
        q=x@rr.T+tt
        return q[:,:2]/q[:,2:3]*kk[[0,1],[0,1]]+kk[:2,2]
    ref_projection=pixel_projection(probes,{"R_reference_to_camera":best["R_reference_to_camera"],
                                           "t_reference_to_camera":best["t_reference_to_camera"],"K":best["K"]})
    probe_rms=np.array([np.sqrt(np.mean(np.sum((pixel_projection(probes,f)-ref_projection)**2,axis=1))) for f in boot])
    probe_axis_rms=np.array([np.sqrt(np.mean((pixel_projection(probes,f)-ref_projection)**2,axis=0)) for f in boot])
    model_uncertainty_pixels=np.percentile(probe_axis_rms,95,axis=0)
    noise_pixels=np.sqrt(noise_pixels**2+model_uncertainty_pixels**2)
    noise_norm=noise_pixels/np.array([best["K"][0,0],best["K"][1,1]])
    focal_boot=np.array([np.asarray(f["K"])[[0,1],[0,1]] for f in boot])
    centers_boot=np.array([f["virtual_optical_center_in_fit_reference"] for f in boot])
    bootstrap_summary={"reprojection_prediction_rms_p50_p90_p95_max_pixels":np.percentile(prediction_rms,[50,90,95,100]).tolist(),
                       "focal_p5_p50_p95_pixels":np.percentile(focal_boot,[5,50,95],axis=0).tolist(),
                       "virtual_center_p5_p50_p95_m":np.percentile(centers_boot,[5,50,95],axis=0).tolist(),
                       "fixed_probe_xyz_in_fit_reference":probes.tolist(),
                       "fixed_probe_prediction_rms_p50_p90_p95_max_pixels":np.percentile(probe_rms,[50,90,95,100]).tolist(),
                       "fixed_probe_prediction_rms_p95_each_axis_pixels":model_uncertainty_pixels.tolist(),
                       "physical_parameter_stability_verified":False}
    # Enhanced clouds are temporally dependent: bootstrap contiguous 1-second
    # blocks within each TRAINING segment, rather than independent radar points.
    rigid_boot=[]
    for _ in range(bootstrap_count):
        selected=[]
        for start,end in INTERVALS:
            groups={}
            for i,row in enumerate(pairing):
                stamp=row["timestamp_unix"]
                if start<=stamp<=end:
                    groups.setdefault(int(np.floor(stamp-start)),[]).append(i)
            keys=sorted(groups)
            for key in generator.choice(keys,len(keys),replace=True):
                selected.extend(groups[int(key)])
        br,bt,bres,bs,bw=robust_rigid_fit(rp[selected],tp[selected])
        relative_rotation=br@r_truth.T
        angle=np.degrees(np.arccos(np.clip((np.trace(relative_rotation)-1)/2,-1,1)))
        rigid_boot.append({"R":br.tolist(),"t":bt.tolist(),
                           "rotation_delta_degrees":float(angle),"translation_delta_m":(bt-t_truth).tolist()})
    rigid_boot_summary={"method":"stratified contiguous 1-second block bootstrap; seed 160016 after camera bootstrap",
                        "count":len(rigid_boot),
                        "rotation_delta_p50_p90_p95_degrees":np.percentile([r["rotation_delta_degrees"] for r in rigid_boot],[50,90,95]).tolist(),
                        "translation_delta_p5_p50_p95_m":np.percentile([r["translation_delta_m"] for r in rigid_boot],[5,50,95],axis=0).tolist()}
    usable=(camera_reference=="training_truth" and 100<best["K"][0,0]<2000 and
            100<best["K"][1,1]<2000 and np.min(best["depths_camera"])>0 and
            best["rms_pixels"]<20 and len(boot)>=max(1,bootstrap_count*.8) and
            np.percentile(prediction_rms,90)<20 and np.percentile(probe_rms,95)<20)
    report={"schema":"mmaud_local_experiment_v1", "official_calibration":False,
            "input_provenance":{"annotation_csv_sha256":digest(out/"annotations_training.csv"),
                                "calibration_tool_sha256":digest(Path(__file__)),
                                "image_count":len(rows),"image_hashes_retained_in_annotation_csv":True,
                                "training_truth_array_count":len(gtfiles)},
            "usable_for_evaluation": bool(usable),
            "evaluation_scope":"exploratory calibration-limited local held-out comparison only",
            "verified_physical_calibration":False,
            "supports_global_hardware_accuracy_claim":False,
            "validation_status": "training only checks passed for explicitly limited virtual local camera model" if usable else "failed training model checks; do not use as verified calibration",
            "camera_fit_source":"training_Leica_only_no_eval_truth" if camera_reference=="training_truth" else "training_image_radar_only_failed_attempt",
            "fit_intervals_unix_s":INTERVALS, "evaluation_used_for_fit":False,
            "experiment_scope":"offline held-out short central overhead clip; later calibration interval included",
            "camera_model":{"model":"experimental_local_pinhole", "K":best["K"].tolist(),
                            "distortion":[0.,0.,0.,0.,0.],
                            "distortion_status":"assumed_zero_in_local_pinhole_not_official_fisheye_undistortion",
                            "R_camera_to_radar":camera_to_radar_rotation.tolist(),
                            "t_camera_in_radar":camera_center.tolist(),"noise_norm":noise_norm.tolist(),
                            "noise_pixels":noise_pixels.tolist(),"pixel_scope_left_camera":[1280,960],
                            "noise_estimation":"training reprojection std (minimum 3 px) combined in quadrature with 95th percentile bootstrap fixed-probe uncertainty",
                            "virtual_optical_center_in_truth":reference_camera_center.tolist() if camera_reference=="training_truth" else None,
                            "R_camera_to_truth":best["R_reference_to_camera"].T.tolist() if camera_reference=="training_truth" else None,
                            "t_camera_in_truth":reference_camera_center.tolist() if camera_reference=="training_truth" else None,
                            "extrinsics_status":"virtual_local_projection_composed_with_training_rigid_fit_not_measured_physical_rig"},
            "radar_to_truth":{"R":r_truth.tolist(),"t":t_truth.tolist(),
                              "convention":"p_truth = R @ p_radar + t", "fit_count":len(rp),
                              "rmse_m":float(np.sqrt(np.mean(np.sum(residual_truth**2,axis=1)))),
                              "residual_std_m":np.std(residual_truth,axis=0,ddof=1).tolist(),
                              "fit_method":"Huber IRLS rigid SE3; threshold 0.5m; 50 iterations max; training only",
                              "huber_threshold_m":.5,"unit_weight_count":int(np.sum(rigid_weights>=.999)),
                              "weights":rigid_weights.tolist(),
                              "bootstrap_summary":rigid_boot_summary,
                              "bootstrap":rigid_boot,
                              "ordinary_all_pairs":{"R":ordinary_r.tolist(),"t":ordinary_t.tolist(),
                                                    "rmse_m":float(np.sqrt(np.mean(np.sum(ordinary_residual**2,axis=1))))}},
            "radar_R":np.diag(np.maximum(np.std(residual_truth@r_truth,axis=0,ddof=1),[.20,.20,.30])**2).tolist(),
            "time_alignment":{"clock_offsets_s":0.,"status":"publication timestamps; clock offset not independently verified",
                              "camera_radar_max_abs_gap_s":.15,"GT_interpolation_max_gap_s":.30},
            "camera_fit":{"count":len(xyz),"xyz_centered_singular_values_m":singular.tolist(),
                          "xyz_rank":int(np.linalg.matrix_rank(xyz-xyz.mean(axis=0))),
                          "pixel_extent":np.ptp(uv,axis=0).tolist(),"best":serial_fit(best),
                          "multi_start":[serial_fit(f) for f in fits],"bootstrap_seed":160016,
                          "bootstrap_success":len(boot),"bootstrap_requested":bootstrap_count,
                          "bootstrap":boot,"bootstrap_summary":bootstrap_summary},
            "radar_training_audit":{"source_frames":len(radar_audit),"accepted_snapshots":len(observations),
                                    "background_cells":len(background),"unique_clouds_not_independent_returns":True},
            "limitations":["Unknown true K/D and camera-radar/Leica rig matrices",
                           "Local pinhole approximation for unrectified fisheye pixels",
                           "Image body, enhanced radar centroid, and Leica prism need not be identical physical points",
                           "Training residuals combine calibration, observation, timing and reference-point errors",
                           "No covariance reduction by duplicate/accumulated radar point count"]}
    (out/"local_model.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"correspondences_training.json").write_text(json.dumps(correspondences,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"radar_truth_pairing_training.json").write_text(json.dumps(pairing,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"camera_count":len(xyz),"camera_rms_px":best["rms_pixels"],"K":best["K"].tolist(),
                      "camera_center":camera_center.tolist(),"xyz_singular":singular.tolist(),
                      "radar_truth_count":len(rp),"radar_truth_rmse":report["radar_to_truth"]["rmse_m"],
                      "bootstrap_success":len(boot)},ensure_ascii=False))


def timed_files(folder: Path, suffix: str) -> list[tuple[float, Path]]:
    return sorted((float(p.stem), p) for p in folder.glob("*" + suffix))


def in_intervals(t: float, intervals=INTERVALS) -> bool:
    return any(a <= t <= b for a, b in intervals)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspection(data: Path, out: Path):
    """Training images only. Crops/blackhat candidates are annotation aids."""
    folder = out / "training_inspection"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    tiles = []
    for t, path in timed_files(data / "image", ".png"):
        if not in_intervals(t):
            continue
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(path)
        # Fixed broad near-axis patch, identified visually before calibration.
        x0, y0, x1, y1 = 430, 260, 980, 740
        patch = original[y0:y1, x0:x1]
        cv2.imwrite(str(folder / f"{path.stem}_patch.png"), patch)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        _, mask = cv2.threshold(black, 25, 255, cv2.THRESH_BINARY)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = []
        marked = patch.copy()
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            if area < 2 or area > 250 or max(w, h) > 35:
                continue
            u, v = centers[i] + [x0, y0]
            candidates.append({"u": float(u), "v": float(v), "area": area,
                               "bbox": [x + x0, y + y0, w, h]})
            cv2.rectangle(marked, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 1)
        records.append({"timestamp": t, "source_image": path.name, "sha256": digest(path),
                        "candidates": candidates})
        cv2.imwrite(str(folder / f"{path.stem}_marked.png"), marked)
        tile = cv2.resize(marked, (550, 480))
        cv2.putText(tile, path.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,255), 1)
        tiles.append(tile)
    for first in range(0, len(tiles), 8):
        page = tiles[first:first+8]
        while len(page) % 2:
            page.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(page[i:i+2]) for i in range(0, len(page), 2)])
        cv2.imwrite(str(folder / f"sheet_{first:02d}.png"), sheet)
    (out / "training_candidates_image_only.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"training_images": len(records), "inspection_folder": str(folder)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inspect-training", action="store_true")
    parser.add_argument("--annotate-training", action="store_true")
    parser.add_argument("--fit-training", action="store_true")
    parser.add_argument("--bootstrap-count", type=int, default=50)
    parser.add_argument("--camera-reference", choices=("radar","training_truth"), default="training_truth")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inspect_training:
        inspection(args.data, args.output)
    elif args.annotate_training:
        annotate_training(args.data,args.output)
    elif args.fit_training:
        fit_training(args.data,args.output,args.bootstrap_count,args.camera_reference)
    else:
        parser.error("Use --inspect-training; fitting requires image-only annotations.")


if __name__ == "__main__":
    main()
