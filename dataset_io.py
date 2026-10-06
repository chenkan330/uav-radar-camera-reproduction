"""Load explicitly named, timestamped radar/camera CSV data for offline replay.

The loader does not infer units, calibrate reference frames, detect objects, or
feed ground truth into a filter. See docs/data_format.md for the input contract.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


RADAR_COLUMNS = (
    "timestamp_s", "arrival_time_s", "frame_id", "x_m", "y_m", "z_m",
    "vx_mps", "intensity",
)
RADAR_FRAME_COLUMNS = ("timestamp_s", "arrival_time_s", "frame_id")
CAMERA_COLUMNS = (
    "timestamp_s", "arrival_time_s", "detection_id", "xmin_px", "ymin_px",
    "xmax_px", "ymax_px", "confidence",
)
TRUTH_COLUMNS = ("timestamp_s", "x_m", "y_m", "z_m")


def _rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f"{path.name}: missing or duplicate CSV column names")
        missing = set(required).difference(headers)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"{path.name}:{line_number}: malformed CSV row")
            row["__location__"] = f"{path.name}:{line_number}"
            rows.append(row)
        return rows


def _number(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (ValueError, TypeError) as error:
        raise ValueError(f"{row['__location__']}: invalid {name}") from error
    if not np.isfinite(value):
        raise ValueError(f"{row['__location__']}: {name} must be finite")
    return value


def _identifier(row: dict[str, str], name: str) -> str:
    value = row[name].strip()
    if not value:
        raise ValueError(f"{row['__location__']}: empty {name}")
    return value


def _times(row: dict[str, str]) -> tuple[float, float]:
    timestamp = _number(row, "timestamp_s")
    arrival = _number(row, "arrival_time_s")
    if timestamp < 0 or arrival < timestamp:
        raise ValueError(
            f"{row['__location__']}: require 0 <= timestamp_s <= arrival_time_s; "
            "both times must use the same clock"
        )
    return timestamp, arrival


def read_dataset(directory: str | Path) -> dict:
    """Return config, raw radar/camera events, and optional Nx4 ground truth.

    Each radar cloud has shape (N, 5), ordered x/y/z/vx/intensity. Camera bbox
    has shape (4,), ordered xmin/ymin/xmax/ymax. Event identifiers are namespaced
    as radar:<frame_id> and camera:<detection_id>. Events are sorted by sensor
    timestamp, not by arrival; the replay layer decides arrival processing.
    """
    directory = Path(directory)
    with (directory / "config.json").open("r", encoding="utf-8-sig") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")

    frames: dict[str, dict] = {}

    def ensure_frame(row: dict[str, str]) -> dict:
        identifier = _identifier(row, "frame_id")
        timestamp, arrival = _times(row)
        frame = frames.get(identifier)
        if frame is None:
            frame = {
                "timestamp": timestamp, "arrival_time": arrival,
                "event_id": f"radar:{identifier}", "points": [],
            }
            frames[identifier] = frame
        elif (frame["timestamp"], frame["arrival_time"]) != (timestamp, arrival):
            raise ValueError(
                f"{row['__location__']}: frame_id {identifier!r} has inconsistent "
                "timestamp_s or arrival_time_s"
            )
        return frame

    for row in _rows(directory / "radar.csv", RADAR_COLUMNS):
        frame = ensure_frame(row)
        point = [_number(row, key) for key in RADAR_COLUMNS[3:]]
        if point[4] < 0:
            raise ValueError(f"{row['__location__']}: linear intensity must be >= 0")
        frame["points"].append(point)

    frame_table = directory / "radar_frames.csv"
    if frame_table.exists():
        seen = set()
        for row in _rows(frame_table, RADAR_FRAME_COLUMNS):
            identifier = _identifier(row, "frame_id")
            if identifier in seen:
                raise ValueError(f"{row['__location__']}: duplicate frame_id {identifier!r}")
            seen.add(identifier)
            ensure_frame(row)

    radar = []
    for frame in frames.values():
        points = frame.pop("points")
        frame["cloud"] = np.asarray(points, dtype=float).reshape(-1, 5)
        radar.append(frame)
    radar.sort(key=lambda event: (event["timestamp"], event["event_id"]))
    if any(first["timestamp"] == second["timestamp"] for first, second in zip(radar, radar[1:])):
        raise ValueError("radar.csv/radar_frames.csv: one radar frame per timestamp is required")

    camera = []
    camera_file = directory / "camera.csv"
    if camera_file.exists():
        seen = set()
        for row in _rows(camera_file, CAMERA_COLUMNS):
            identifier = _identifier(row, "detection_id")
            if identifier in seen:
                raise ValueError(f"{row['__location__']}: duplicate detection_id {identifier!r}")
            seen.add(identifier)
            timestamp, arrival = _times(row)
            bbox = np.asarray([_number(row, key) for key in CAMERA_COLUMNS[3:7]])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"{row['__location__']}: bounding box must have positive area")
            confidence = _number(row, "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError(f"{row['__location__']}: confidence must be in [0, 1]")
            camera.append({
                "timestamp": timestamp, "arrival_time": arrival,
                "event_id": f"camera:{identifier}", "bbox": bbox,
                "confidence": confidence,
            })
    camera.sort(key=lambda event: (event["timestamp"], event["event_id"]))
    if any(first["timestamp"] == second["timestamp"] for first, second in zip(camera, camera[1:])):
        raise ValueError("camera.csv: one selected target detection per timestamp is required")

    truth = None
    truth_file = directory / "ground_truth.csv"
    if truth_file.exists():
        truth_rows = []
        for row in _rows(truth_file, TRUTH_COLUMNS):
            values = [_number(row, key) for key in TRUTH_COLUMNS]
            if values[0] < 0:
                raise ValueError(f"{row['__location__']}: timestamp_s must be >= 0")
            truth_rows.append(values)
        truth = np.asarray(truth_rows, dtype=float).reshape(-1, 4)
        if len(truth):
            truth = truth[np.argsort(truth[:, 0], kind="stable")]
            if np.any(np.diff(truth[:, 0]) <= 0):
                raise ValueError("ground_truth.csv: timestamps must be unique")

    return {"config": config, "radar": radar, "camera": camera, "ground_truth": truth}
