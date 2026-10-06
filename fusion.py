"""Step 4: event-time sensor fusion with 100 Hz ticks and delayed replay.

Raw clouds/boxes and every prediction tick are retained. Late insertion
restores the preceding posterior and recomputes all subsequent associations.
The returned history is retrospectively corrected, not an online publication log.
"""
from bisect import bisect_left
from dataclasses import dataclass, field
import numpy as np

from kalman3d import Kalman3D, POSITION
from radar import RadarInitializer, RadarAssociator
from camera import (bbox_to_pixels, camera_bearing_observation,
                    paper_pseudo_observation)


@dataclass
class FusionConfig:
    radar_method: str = "closest"
    radar_R: np.ndarray = field(default_factory=lambda: np.diag(np.array([.2, .35, .35, .12])**2))
    gate: float = 3.0
    beta0: float = .1
    acceleration_std: float = .8
    velocity_std: float = 2.0
    initialization_speed: float = .3
    prediction_hz: float = 100.0
    camera_mode: str = "paper_xyz"
    pixel_std: float = 2.0
    depth_std: float = .5
    confidence_threshold: float = .7


@dataclass
class TimelineItem:
    timestamp: float
    priority: int
    key: str
    sensor: str
    payload: dict
    x: np.ndarray | None = None
    P: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def order(self):
        return self.timestamp, self.priority, self.key


class FusionTracker:
    def __init__(self, intrinsics, extrinsics, config=None):
        self.config = config or FusionConfig()
        c = self.config
        for name in ("prediction_hz", "pixel_std", "depth_std", "velocity_std", "initialization_speed"):
            if not np.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= c.confidence_threshold <= 1 or c.camera_mode not in ("paper_xyz", "bearing"):
            raise ValueError("invalid camera mode or confidence threshold")
        if not np.isfinite(c.acceleration_std) or c.acceleration_std < 0:
            raise ValueError("acceleration_std must be finite and nonnegative")
        self.intrinsics, self.extrinsics = intrinsics, extrinsics
        self.associator = RadarAssociator(c.radar_method, R=c.radar_R, gate=c.gate, beta0=c.beta0)
        self.filter = None
        self.items = []
        self.events = []
        self.seen = set()
        self.wall_time = -np.inf
        self.anchor_time = None
        self.anchor_x = self.anchor_P = None
        self.next_tick = None
        self.late_events = 0
        self.replayed_items = 0
        self.ignored_before_initialization = 0

    def _initializer(self):
        c = self.config
        return RadarInitializer(R=c.radar_R, velocity_std=c.velocity_std,
                                acceleration_std=c.acceleration_std,
                                speed_threshold=c.initialization_speed)

    def ingest(self, sensor, timestamp, arrival_time, event_id, **payload):
        """Arrival-time ordered input; event_id must be unique per sensor.

        Radar payload: cloud Nx5. Camera payload: bbox(4), confidence.
        Acquisition and arrival timestamps use the same clock, in seconds.
        """
        timestamp, arrival_time = float(timestamp), float(arrival_time)
        key = f"{sensor}:{event_id}"
        if sensor not in ("radar", "camera") or key in self.seen:
            raise ValueError("unknown sensor or duplicate event_id")
        if any(e.sensor == sensor and e.timestamp == timestamp for e in self.events):
            raise ValueError("one frame/detection per sensor timestamp is required")
        if not np.isfinite([timestamp, arrival_time]).all() or timestamp < 0 or arrival_time < timestamp:
            raise ValueError("finite nonnegative timestamps require arrival >= acquisition")
        if arrival_time < self.wall_time - 1e-12:
            raise ValueError("ingest inputs must be ordered by arrival_time")
        if sensor == "radar":
            cloud = np.array(payload.get("cloud"), dtype=float, copy=True)
            if cloud.ndim != 2 or cloud.shape[1] != 5 or not np.isfinite(cloud).all() or np.any(cloud[:, 4] < 0):
                raise ValueError("cloud must be finite Nx5 with nonnegative linear intensity")
            payload = {"cloud": cloud}
        else:
            bbox = np.array(payload.get("bbox"), dtype=float, copy=True)
            confidence = float(payload.get("confidence"))
            bbox_to_pixels(bbox, confidence, self.config.confidence_threshold)
            payload = {"bbox": bbox, "confidence": confidence}
        event = TimelineItem(timestamp, 0 if sensor == "radar" else 1, key, sensor, payload)
        self.seen.add(key)
        self.events.append(event)
        if self.filter is None:
            self.wall_time = arrival_time
            initializer = self._initializer()
            for candidate in sorted((e for e in self.events if e.sensor == "radar"), key=lambda e:e.order):
                result = initializer.observe(candidate.timestamp, candidate.payload["cloud"])
                if result is not None:
                    self.filter = result
                    self.anchor_time = candidate.timestamp
                    self.anchor_x, self.anchor_P = result.x.copy(), result.P.copy()
                    self.next_tick = int(np.floor(self.anchor_time * self.config.prediction_hz + 1e-9)) + 1
                    self.items = [e for e in self.events if e.timestamp > self.anchor_time or
                                  (e.timestamp == self.anchor_time and e.sensor == "camera")]
                    self.items.sort(key=lambda e:e.order)
                    self.ignored_before_initialization = len(self.events) - len(self.items)
                    self._replay(0)
                    self.advance_to(arrival_time)
                    break
            return self.filter is not None
        self.advance_to(arrival_time)
        if timestamp < self.anchor_time or (timestamp == self.anchor_time and sensor == "radar"):
            self.ignored_before_initialization += 1
            return False
        if self.items and timestamp < self.items[-1].timestamp - 1e-12:
            self.late_events += 1
        index = bisect_left([e.order for e in self.items], event.order)
        self.items.insert(index, event)
        self._replay(index)
        return True

    def advance_to(self, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < self.wall_time - 1e-12:
            raise ValueError("advance_to must be finite and monotonic")
        self.wall_time = timestamp
        if self.filter is None:
            return
        end_tick = int(np.floor(timestamp * self.config.prediction_hz + 1e-9))
        first_changed = len(self.items)
        while self.next_tick <= end_tick:
            tick = TimelineItem(self.next_tick / self.config.prediction_hz, 2,
                                f"tick:{self.next_tick:012d}", "tick", {})
            index = bisect_left([e.order for e in self.items], tick.order)
            self.items.insert(index, tick)
            first_changed = min(first_changed, index)
            self.next_tick += 1
        if first_changed < len(self.items):
            self._replay(first_changed)

    def _replay(self, start):
        if start == 0:
            tracker = Kalman3D(self.anchor_x, self.anchor_P, self.config.acceleration_std)
            previous_time = self.anchor_time
        else:
            previous = self.items[start - 1]
            tracker = Kalman3D(previous.x, previous.P, self.config.acceleration_std)
            previous_time = previous.timestamp
        for item in self.items[start:]:
            dt = item.timestamp - previous_time
            if dt > 1e-12:
                tracker.predict(dt)
            item.diagnostics = {}
            if item.sensor == "radar":
                try:
                    result = self.associator.update(tracker, item.payload["cloud"])
                    item.diagnostics = {"accepted": bool(result.accepted),
                                        "gated_count": int(len(result.gated_indices))}
                except ValueError as error:
                    if str(error) != "Weighted needs positive total gated linear intensity":
                        raise
                    item.diagnostics = {"accepted": False, "reason": "zero gated intensity"}
            elif item.sensor == "camera":
                pixels = bbox_to_pixels(item.payload["bbox"], item.payload["confidence"],
                                        self.config.confidence_threshold)
                if pixels is None:
                    item.diagnostics = {"accepted": False, "reason": "confidence"}
                else:
                    try:
                        if self.config.camera_mode == "paper_xyz":
                            observation = paper_pseudo_observation(pixels, tracker.x, self.intrinsics,
                                self.extrinsics, pixel_std=self.config.pixel_std, depth_std=self.config.depth_std)
                        else:
                            observation = camera_bearing_observation(pixels, tracker.x, self.intrinsics,
                                self.extrinsics, pixel_std=self.config.pixel_std)
                        tracker.update_linear(observation.z, observation.R, observation.H)
                        item.diagnostics = {"accepted": True, "depth_m": float(observation.depth)}
                    except ValueError as error:
                        # Out-of-view/nonpositive-depth geometry is a rejected observation.
                        item.diagnostics = {"accepted": False, "reason": str(error)}
            item.x, item.P = tracker.x.copy(), tracker.P.copy()
            previous_time = item.timestamp
        self.replayed_items += len(self.items) - start
        self.filter = tracker

    def tick_history(self):
        ticks = [e for e in self.items if e.sensor == "tick"]
        return (np.array([e.timestamp for e in ticks]),
                np.array([e.x for e in ticks]).reshape(-1, 6),
                np.array([e.P for e in ticks]).reshape(-1, 6, 6))

    def diagnostics(self):
        updates = [e for e in self.items if e.sensor != "tick"]
        return {"initialized": self.filter is not None, "initialization_time_s": self.anchor_time,
                "late_events": self.late_events, "replayed_items": self.replayed_items,
                "ignored_before_initialization": self.ignored_before_initialization,
                "accepted_radar": sum(e.sensor == "radar" and e.diagnostics.get("accepted", False) for e in updates),
                "accepted_camera": sum(e.sensor == "camera" and e.diagnostics.get("accepted", False) for e in updates),
                "history_kind": "retrospectively corrected 100 Hz ticks; not causal published history"}
