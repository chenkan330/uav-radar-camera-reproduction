"""Compare constant-velocity extrapolation + R-Net with N-Net + R-Net on moving-UAV scenes."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from .common import BASE, finish_run, new_run, seed_all, write_json, write_rows
from .uav_vehicle_scene import (ALTITUDE_M, DT, INITIAL_BEARING_FROM_NORTH_DEG,
                                INITIAL_HORIZONTAL_RANGE_M, ROAD_ANGLE_FROM_EAST_WEST_DEG,
                                SUBSTEPS, VEHICLE_SPEED_MPS, build_scene)
from .versions import save_version


VERSION = "uav_vehicle_cv_nr"
METHODS = ("cv_rnet", "nrnet")
FRAMES = 205
SEED_START = {"train": 110000, "r_train": 210000, "validation": 310000, "test": 410000}
DEFAULT_COUNTS = {"train": 384, "r_train": 128, "validation": 50, "test": 50}
MODEL_SEED = 20261125


def split_scenes(out, split, count):
    """Keep the exact noise draws and preissued displacement commands for independent replay."""
    seeds = np.arange(SEED_START[split], SEED_START[split] + count, dtype=np.int64)
    scenes = [build_scene(int(seed), FRAMES) for seed in seeds]
    arrays = {field: np.stack([getattr(scene, field) for scene in scenes])
              for field in ("platform", "vehicle", "command", "noise", "speed", "turn_rate")}
    np.savez_compressed(out / f"{split}_scenes.npz", seed=seeds, time=scenes[0].time, **arrays)
    return scenes


def evaluate(batch, models, device, split, weights):
    from .uav_vehicle_tracking import simulate, summarize

    print(f"Evaluating {weights} checkpoints on {split} ...", flush=True)
    summary, scenes, frames = [], [], []
    for method in METHODS:
        scene_rows, frame_rows = simulate(batch, *models, device, method)
        result = {"weights": weights, **summarize(scene_rows)}
        summary.append(result)
        if split == "test":
            for rows in (scene_rows, frame_rows):
                for row in rows:
                    row["seed"] = SEED_START[split] + int(row["scene"])
            for row in frame_rows:
                if not row["hit"]:
                    for axis in "xyz":
                        row[f"estimate_{axis}"] = None
            scenes.extend(scene_rows)
            frames.extend(frame_rows)
        print(f"  {method}: completion {result['completed']}/{result['num_scenes']}, "
              f"detection coverage {result['detection_rate']:.2%}, "
              f"recoveries {result['recoveries']}/{result['lost_events']}", flush=True)
    return summary, scenes, frames


def nr_score(summary):
    nr = next(row for row in summary if row["method"] == "nrnet")
    return nr["completion_rate"], nr["detection_rate"]


def adaptation_reasons(summary):
    nr = next(row for row in summary if row["method"] == "nrnet")
    reasons = []
    if nr["completion_rate"] < 0.50:
        reasons.append("N+R completed fewer than half of validation scenes")
    for row in summary:
        if row["lost_events"] >= 10 and row["recoveries"] == 0:
            reasons.append(f"{row['method']} recovered none of its {row['lost_events']} validation losses")
    return reasons


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="PyTorch execution device")
    parser.add_argument("--output-root", type=Path, default=BASE / "outputs" / VERSION,
                        help="create a new, timestamped result folder here")
    for split, flag in (("train", "train"), ("r_train", "r"), ("validation", "validation"), ("test", "test")):
        parser.add_argument(f"--{flag}-scenes", type=int, default=DEFAULT_COUNTS[split], dest=split,
                            help=f"number of independent {split} scenes")
    args = parser.parse_args(argv)
    if any(not 1 <= getattr(args, split) <= 100000 for split in DEFAULT_COUNTS):
        parser.error("scene counts must be in [1, 100000] to keep split seed ranges independent")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but is unavailable in this Python environment")
    return args


def main(argv=None):
    args = arguments(argv)
    from .uav_vehicle_plots import plot_results
    from .uav_vehicle_tracking import BUDGET, MEASUREMENT_SIGMA, load_original, prepare

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_all(MODEL_SEED)
    out = new_run(args.output_root.resolve())
    print(f"Run folder: {out}\nDevice: {device}", flush=True)
    counts = {name: getattr(args, name) for name in DEFAULT_COUNTS}
    nnet, rnet, ck, npath, rpath = load_original(device)
    original = nnet, rnet, ck
    config = {
        "scene_counts": counts, "default_counts": counts == DEFAULT_COUNTS, "first_seeds": SEED_START,
        "frames": FRAMES, "dt_s": DT, "flight_integration_dt_s": DT / SUBSTEPS,
        "initial_observations": nnet.config.history, "evaluation_duration_s": (FRAMES - nnet.config.history) * DT,
        "uav_altitude_m": ALTITUDE_M, "vehicle_speed_kmh": VEHICLE_SPEED_MPS * 3.6,
        "initial_ground_range_m": list(INITIAL_HORIZONTAL_RANGE_M),
        "initial_bearing_from_north_deg": list(INITIAL_BEARING_FROM_NORTH_DEG),
        "road_heading_offset_from_east_or_west_deg": list(ROAD_ANGLE_FROM_EAST_WEST_DEG),
        "platform": "Level curved flight with variable speed; fixed array attitude; exact past navigation and preissued next step.",
        "target": "One ground vehicle moving in a straight line at constant speed; noisily observed on hits.",
        "array_axes": "west/north/down; past detected world observations are expressed in the upcoming array frame",
        "measurement_noise_std_range_m_az_polar_rad": MEASUREMENT_SIGMA.tolist(),
        "detection_model": "selected set includes the nearest one of 32x32 centered DFT beam labels; Gaussian range/angle measurement proxy",
        "method_names": list(METHODS), "beam_budget_per_frame": BUDGET, "failure_limit": nnet.config.failure_limit,
        "model_config": nnet.config.dictionary(), "model_seed": MODEL_SEED, "device": str(device),
        "original_nnet_checkpoint": str(Path(npath).resolve()), "original_rnet_checkpoint": str(Path(rpath).resolve()),
        "adaptation_trigger": "N+R validation completion < 0.5 or either method has 10+ loss episodes and zero recoveries",
        "selection_rule": "Maximize N+R validation completion rate, then all-scheduled-frame detection coverage. Exact tie keeps original.",
        "paired_comparison": "Identical test scenes, noise, beam budget, detection/stop rules, and N/R checkpoint pair for both methods.",
    }
    write_json(out / "config.json", config)
    validation_scenes = split_scenes(out, "validation", counts["validation"])
    validation_batch = prepare(validation_scenes)
    val_original, _, _ = evaluate(validation_batch, original, device, "validation", "original")
    val = list(val_original)
    reasons = adaptation_reasons(val_original)
    selected, chosen = "original", original
    weight_paths = {key: config[key] for key in ("original_nnet_checkpoint", "original_rnet_checkpoint")}
    adapted_paths = {}
    if reasons:
        from .uav_vehicle_adapt import adapt

        print("Adapting from the original checkpoints: " + ", ".join(reasons), flush=True)
        train = split_scenes(out, "train", counts["train"])
        r_train = split_scenes(out, "r_train", counts["r_train"])
        adapt_out = out.parent / "adapted_checkpoints" / out.name
        adapt_out.mkdir(parents=True)
        candidate = adapt(*original, device, adapt_out, train, r_train, validation_scenes)
        shutil.copy2(adapt_out / "adaptation.json", out / "adaptation.json")
        weight_paths.update(adapted_nnet_checkpoint=str(adapt_out / "adapted_nnet.pth"),
                            adapted_rnet_checkpoint=str(adapt_out / "adapted_rnet.pth"))
        val_adapted, _, _ = evaluate(validation_batch, candidate, device, "validation", "adapted")
        val.extend(val_adapted)
        if nr_score(val_adapted) > nr_score(val_original):
            selected, chosen = "adapted", candidate
        # Restore an independent original pair even if adaptation changed its input modules in place.
        original = load_original(device)[:3]
        if selected == "original":
            chosen = original
        adapted_paths = {"adapted_checkpoint_folder": str(adapt_out)}
    write_rows(out / "validation_summary.csv", val)
    selection = {"selected_weights": selected, "adaptation_reasons": reasons, "validation": val, **adapted_paths}
    write_json(out / "selection.json", selection)
    print(f"Validation selection locked: {selected}. Generating test scenes now.", flush=True)
    test_scenes = split_scenes(out, "test", counts["test"])
    test_batch = prepare(test_scenes)
    original_test, final_scenes, final_frames = evaluate(test_batch, original, device, "test", "original")
    test = list(original_test)
    final_summary = original_test
    if selected == "adapted":
        final_summary, final_scenes, final_frames = evaluate(test_batch, chosen, device, "test", "adapted")
        test.extend(final_summary)
    write_rows(out / "test_summary.csv", test)
    write_rows(out / "final_scene_results.csv", final_scenes)
    write_rows(out / "final_frame_results.csv", final_frames)
    plot_results(out, test_scenes, final_summary, final_frames)
    write_json(out / "summary.json", {
        "experiment": VERSION, **config, **selection, **weight_paths,
        "nnet_checkpoint": weight_paths[f"{selected}_nnet_checkpoint"],
        "rnet_checkpoint": weight_paths[f"{selected}_rnet_checkpoint"],
        "test": test, "final_test": final_summary,
        "coverage_denominator": "all scheduled evaluation frames, including those after early termination",
        "position_rmse": "world-frame position errors on detected evaluation frames only; see detection coverage",
    })
    version = VERSION + "_" + out.name.removeprefix("run_")
    archive = save_version(version, out, "Paired vehicle tracking from a moving UAV.", asset_keys=tuple(weight_paths))
    finish_run(out)
    print(f"Results: {out}\nSnapshot: {archive}", flush=True)
    return out


if __name__ == "__main__":
    main()
