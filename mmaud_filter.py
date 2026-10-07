"""MMAUD 1.6 adaptation: XYZ-only radar and genuine camera bearing updates.

This module does not accept ground truth, Cartesian velocity observations,
intensity, detections generated from truth, or a trained image detector. It
preserves the handwritten 1.5 Kalman core but explicitly changes radar H/R to
three-dimensional position and uses a normalized-image bearing EKF. Both
comparison groups share every prediction interval and the same 100 Hz outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman3d import H_POSITION, POSITION, VELOCITY, Kalman3D, _covariance


def _time(value, name="timestamp"):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _vector(value, size, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _readonly(value):
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class RadarFrame:
    timestamp: float
    points: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "timestamp", _time(self.timestamp))
        points = np.array(self.points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("radar points must be a finite Nx3 XYZ array, in metres")
        object.__setattr__(self, "points", _readonly(points))


@dataclass(frozen=True)
class CameraFrame:
    timestamp: float
    normalized_xy: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "timestamp", _time(self.timestamp))
        object.__setattr__(self, "normalized_xy", _readonly(_vector(self.normalized_xy, 2, "normalized_xy")))


@dataclass(frozen=True)
class Calib:
    """p_radar = rotation @ p_camera + translation, no implied sign change.

    normalized_xy must already be undistorted and equal camera [X/Z,Y/Z].
    noise_norm is one or two standard deviations in normalized-image units.
    Static calibration is treated as exact; calibration uncertainty is not
    included in this initial external-data adaptation.
    """

    rotation: np.ndarray
    translation: np.ndarray
    noise_norm: float | np.ndarray = 0.005

    def __post_init__(self):
        rotation = np.array(self.rotation, dtype=float, copy=True)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0,
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0):
            raise ValueError("rotation must be a proper orthonormal camera-to-radar rotation")
        translation = _vector(self.translation, 3, "translation")
        noise = np.array(self.noise_norm, dtype=float, copy=True)
        if noise.shape == ():
            noise = np.repeat(noise, 2)
        if noise.shape != (2,) or not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError("noise_norm must be one or two finite positive standard deviations")
        object.__setattr__(self, "rotation", _readonly(rotation))
        object.__setattr__(self, "translation", _readonly(translation))
        object.__setattr__(self, "noise_norm", _readonly(noise))

    @property
    def covariance(self):
        return np.diag(self.noise_norm**2)


@dataclass
class BearingPrediction:
    normalized_xy: np.ndarray
    H: np.ndarray
    optical_depth: float


def normalized_bearing_model(state, calib):
    """Analytic two-dimensional pinhole model at the current estimated state."""
    state = _vector(state, 6, "state")
    if not isinstance(calib, Calib):
        raise ValueError("camera calibration must be Calib")
    camera = calib.rotation.T @ (state[POSITION] - calib.translation)
    x, y, depth = camera
    if depth <= 1e-9:
        raise ValueError("predicted camera optical depth must be positive and nonsingular")
    J = np.array([[1 / depth, 0, -x / depth**2], [0, 1 / depth, -y / depth**2]])
    H = J @ calib.rotation.T @ H_POSITION
    normalized = camera[:2] / depth
    if not np.isfinite(H).all() or not np.isfinite(normalized).all():
        raise ValueError("camera geometry exceeds finite numerical range")
    return BearingPrediction(normalized, H, float(depth))


@dataclass
class FilterRun:
    times: np.ndarray
    state: np.ndarray
    covariance: np.ndarray
    updates: list[dict]
    initial_source: dict
    prediction_grid: np.ndarray

    @property
    def positions(self):
        return self.state[:, POSITION]


def _frames(values, frame_type, name):
    values = list(values)
    if any(not isinstance(value, frame_type) for value in values):
        raise ValueError(f"{name} must contain {frame_type.__name__} objects")
    values.sort(key=lambda frame: frame.timestamp)
    if any(a.timestamp == b.timestamp for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} requires one frame per sensor timestamp")
    return values


def _point_index(value, frame, name):
    if not isinstance(value, (int, np.integer)) or not 0 <= value < len(frame.points):
        raise ValueError(f"{name} must select an existing point in the initialization radar frame")
    return int(value)


def _radar_update(tracker, frame, R, gate):
    base = dict(sensor="radar", timestamp=float(frame.timestamp), accepted=False,
                nis=None, gate_nis=float(gate), candidate_count=len(frame.points),
                gated_count=0, selected_point_index=None)
    if not len(frame.points):
        return dict(base, reason="empty radar frame")
    innovation = frame.points - tracker.x[POSITION]
    S = H_POSITION @ tracker.P @ H_POSITION.T + R
    nis = np.einsum("ij,ij->i", innovation, np.linalg.solve(S, innovation.T).T)
    if not np.isfinite(nis).all():
        raise ValueError("radar NIS exceeds finite numerical range")
    indices = np.flatnonzero(nis <= gate)
    base["gated_count"] = int(len(indices))
    base["nis"] = float(np.min(nis))
    if not len(indices):
        return dict(base, reason="NIS gate")
    selected = int(indices[np.argmin(np.linalg.norm(innovation[indices], axis=1))])
    tracker.update(frame.points[selected], R)
    return dict(base, accepted=True, nis=float(nis[selected]), selected_point_index=selected,
                selected_xyz_m=frame.points[selected].tolist(), reason="nearest gated XYZ")


def _camera_update(tracker, frame, calib, gate):
    base = dict(sensor="camera", timestamp=float(frame.timestamp), accepted=False,
                nis=None, gate_nis=float(gate), observation_normalized_xy=frame.normalized_xy.tolist())
    camera = calib.rotation.T @ (tracker.x[POSITION] - calib.translation)
    if camera[2] <= 1e-9:
        return dict(base, reason="nonpositive/singular predicted optical depth")
    prediction = normalized_bearing_model(tracker.x, calib)
    R = calib.covariance
    innovation = frame.normalized_xy - prediction.normalized_xy
    S = prediction.H @ tracker.P @ prediction.H.T + R
    nis = float(innovation @ np.linalg.solve(S, innovation))
    if not np.isfinite(nis):
        raise ValueError("camera NIS exceeds finite numerical range")
    base.update(nis=nis, prediction_normalized_xy=prediction.normalized_xy.tolist(),
                optical_depth_m=prediction.optical_depth)
    if nis > gate:
        return dict(base, reason="NIS gate")
    equivalent_observation = frame.normalized_xy - prediction.normalized_xy + prediction.H @ tracker.x
    tracker.update_linear(equivalent_observation, R, prediction.H)
    return dict(base, accepted=True, reason="normalized bearing EKF")


def run_comparison(
    radar_frames, camera_frames, calib=None, *, radar_R=None, acceleration_std=0.8,
    velocity_std=2.0, prediction_hz=100.0, radar_gate_nis=11.344866730144373,
    camera_gate_nis=9.210340371976184, initial_point_index=0,
    initial_velocity_from_two=False, initial_second_point_index=None,
    start_time=None, end_time=None,
):
    """Run nearest XYZ radar-only and radar+camera on a common event partition.

    Initialization uses a caller-selected point, default points[0], in the
    first nonempty radar frame. This is a reference strategy, not a claim that
    the first point belongs to the UAV. Optional two-point velocity estimation
    requires the caller to select the same target in the next nonempty frame.
    All units and static calibration must be established before this call.
    """
    radar = _frames(radar_frames, RadarFrame, "radar_frames")
    camera = _frames(camera_frames, CameraFrame, "camera_frames")
    if camera and not isinstance(calib, Calib):
        raise ValueError("camera frames require explicit Calib")
    R = _covariance(np.diag(np.array([.2, .35, .35])**2) if radar_R is None else radar_R, 3, "radar_R", True)
    prediction_hz = _positive(prediction_hz, "prediction_hz")
    velocity_std = _positive(velocity_std, "velocity_std")
    radar_gate_nis = _positive(radar_gate_nis, "radar_gate_nis")
    camera_gate_nis = _positive(camera_gate_nis, "camera_gate_nis")
    if not isinstance(initial_velocity_from_two, (bool, np.bool_)):
        raise ValueError("initial_velocity_from_two must be boolean")
    cut_start = 0.0 if start_time is None else _time(start_time, "start_time")
    if not radar:
        raise ValueError("at least one nonempty radar frame is required for initialization")
    cut_end = max(frame.timestamp for frame in radar + camera) if end_time is None else _time(end_time, "end_time")
    if cut_end < cut_start:
        raise ValueError("end_time must be >= start_time")
    radar = [frame for frame in radar if cut_start <= frame.timestamp <= cut_end]
    camera = [frame for frame in camera if cut_start <= frame.timestamp <= cut_end]
    nonempty = [frame for frame in radar if len(frame.points)]
    if not nonempty:
        raise ValueError("selected interval has no nonempty radar initialization frame")
    first = nonempty[0]
    first_index = _point_index(initial_point_index, first, "initial_point_index")
    anchor = first.timestamp
    position = first.points[first_index]
    initial_tracker = Kalman3D.from_position(position, R, velocity_std, acceleration_std)
    source = dict(rule="caller-selected first nonempty radar point; default index is a reference strategy",
                  first_timestamp=float(first.timestamp), point_index=first_index,
                  first_xyz_m=position.tolist(), velocity_rule="zero",
                  ground_truth_used=False, initialized_observation_reused=False)
    if initial_velocity_from_two:
        if len(nonempty) < 2:
            raise ValueError("two-observation initialization requires two nonempty radar frames")
        second = nonempty[1]
        second_index = _point_index(first_index if initial_second_point_index is None else initial_second_point_index,
                                    second, "initial_second_point_index")
        interval = second.timestamp - first.timestamp
        state = np.zeros(6)
        state[POSITION] = second.points[second_index]
        state[VELOCITY] = (second.points[second_index] - position) / interval
        P = np.zeros((6, 6))
        P[np.ix_(POSITION, POSITION)] = R
        P[np.ix_(VELOCITY, VELOCITY)] = 2 * R / interval**2
        P[np.ix_(POSITION, VELOCITY)] = R / interval
        P[np.ix_(VELOCITY, POSITION)] = R / interval
        initial_tracker = Kalman3D(state, P, acceleration_std)
        anchor = second.timestamp
        source.update(second_timestamp=float(anchor), second_point_index=second_index,
                      second_xyz_m=second.points[second_index].tolist(),
                      velocity_rule="caller-associated two observed XYZ points; independent equal-R noise",
                      estimated_velocity_mps=state[VELOCITY].tolist())
    source["initialization_timestamp"] = float(anchor)

    tick_count = int(np.floor((cut_end - anchor) * prediction_hz + 1e-9))
    output_times = anchor + np.arange(tick_count + 1) / prediction_hz
    # Identical partition for both filters. Camera event timestamps intentionally
    # divide BOTH groups' prediction intervals under the retained discrete Q.
    radar_events = {frame.timestamp: frame for frame in radar if frame.timestamp > anchor}
    camera_events = {frame.timestamp: frame for frame in camera if frame.timestamp >= anchor}
    partition = np.array(sorted(set(output_times).union(radar_events, camera_events)))
    outputs = set(output_times)
    runs = {}
    for name, use_camera in (("radar_only", False), ("radar_camera", True)):
        tracker = Kalman3D(initial_tracker.x, initial_tracker.P, acceleration_std)
        updates = [dict(sensor="radar", timestamp=float(anchor), accepted=True, nis=None,
                        reason="initialization once", selected_point_index=source.get("second_point_index", first_index))]
        states, covariances = [], []
        previous = anchor
        for timestamp in partition:
            interval = float(timestamp - previous)
            if interval > 0:
                tracker.predict(interval)
            if timestamp in radar_events:
                updates.append(_radar_update(tracker, radar_events[timestamp], R, radar_gate_nis))
            if timestamp in camera_events:
                frame = camera_events[timestamp]
                if use_camera:
                    updates.append(_camera_update(tracker, frame, calib, camera_gate_nis))
                else:
                    updates.append(dict(sensor="camera", timestamp=float(timestamp), accepted=False,
                                        nis=None, reason="disabled in radar-only baseline"))
            if timestamp in outputs:
                states.append(tracker.x.copy())
                covariances.append(tracker.P.copy())
            previous = timestamp
        runs[name] = FilterRun(output_times.copy(), np.asarray(states).reshape(-1, 6),
                               np.asarray(covariances).reshape(-1, 6, 6), updates,
                               dict(source), partition.copy())
    return runs
