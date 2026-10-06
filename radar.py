"""Paper radar initialization and gated association, without hardware dependencies.

Cloud columns are [x, y, z, vx, intensity]; SI units and *linear*
nonnegative intensity are required by the Weighted method. Do not pass dB SNR
as weights. R, the gate and beta0 are explicit experimental assumptions.
"""

from dataclasses import dataclass

import numpy as np

from kalman3d import Kalman3D, _covariance


RADAR_STATE = np.array([0, 2, 4, 1])
H_RADAR = np.eye(6)[RADAR_STATE]
DEFAULT_RADAR_STD = np.array([0.20, 0.35, 0.35, 0.12])
DEFAULT_RADAR_R = np.diag(DEFAULT_RADAR_STD**2)
METHODS = ("strongest", "closest", "average", "kmeans", "weighted")


def validate_cloud(cloud):
    """Return an owned, finite Nx5 array, including well-shaped empty frames."""
    result = np.array(cloud, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[1] != 5 or not np.all(np.isfinite(result)):
        raise ValueError("cloud must be a finite Nx5 [x,y,z,vx,intensity] array")
    return result


class RadarInitializer:
    """Start after consecutive strongest returns move slower than 0.3 m/s.

    The paper does not specify the initial covariance/velocities. Here the
    second strongest return initializes xyz and measured vx once; vy/vz=0,
    with independent velocity_std uncertainty. Empty frames break continuity.
    """

    def __init__(self, R=None, velocity_std=2.0, acceleration_std=0.8,
                 speed_threshold=0.3):
        self.R = _covariance(DEFAULT_RADAR_R if R is None else R, 4, "R", True)
        if not np.isfinite(velocity_std) or velocity_std <= 0:
            raise ValueError("velocity_std must be finite and positive")
        if not np.isfinite(acceleration_std) or acceleration_std < 0:
            raise ValueError("acceleration_std must be finite and nonnegative")
        if not np.isfinite(speed_threshold) or speed_threshold <= 0:
            raise ValueError("speed_threshold must be finite and positive")
        self.velocity_std = float(velocity_std)
        self.acceleration_std = float(acceleration_std)
        self.speed_threshold = float(speed_threshold)
        self.previous = None
        self.last_time = None
        self.filter = None
        self.last_speed = None

    def observe(self, timestamp, cloud):
        cloud = validate_cloud(cloud)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        timestamp = float(timestamp)
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("initialization frames must have increasing timestamps")
        self.last_time = timestamp
        if self.filter is not None:
            return None
        if len(cloud) == 0:
            self.previous = None
            self.last_speed = None
            return None
        strongest = cloud[np.argmax(cloud[:, 4])].copy()
        prior = self.previous
        self.previous = (timestamp, strongest)
        if prior is None:
            return None
        self.last_speed = float(np.linalg.norm(strongest[:3] - prior[1][:3]) /
                                (timestamp - prior[0]))
        if self.last_speed >= self.speed_threshold:
            return None
        state = np.zeros(6)
        state[RADAR_STATE] = strongest[:4]
        covariance = np.zeros((6, 6))
        covariance[np.ix_(RADAR_STATE, RADAR_STATE)] = self.R
        covariance[3, 3] = covariance[5, 5] = self.velocity_std**2
        self.filter = Kalman3D(state, covariance, self.acceleration_std)
        return self.filter


@dataclass
class RadarUpdate:
    method: str
    accepted: bool
    gated_indices: np.ndarray
    distances: np.ndarray
    measurement: np.ndarray | None = None
    innovation: np.ndarray | None = None
    gain: np.ndarray | None = None
    beta: np.ndarray | None = None
    beta0: float | None = None

    @property
    def used_count(self):
        return len(self.gated_indices)


class RadarAssociator:
    """Mahalanobis gate then one of the paper's five association rules.

    R is the covariance of one radar return and is deliberately not divided by
    cloud size for centroids: independence of multiple target returns is not
    supplied by the paper. Weighted uses its full covariance mixture (21-26).
    """

    def __init__(self, method="closest", R=None, gate=3.0, beta0=0.1):
        method = str(method).lower()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not np.isfinite(gate) or gate <= 0:
            raise ValueError("gate must be finite and positive")
        if not np.isfinite(beta0) or not 0 <= beta0 <= 1:
            raise ValueError("beta0 must lie in [0,1]")
        self.method = method
        self.R = _covariance(DEFAULT_RADAR_R if R is None else R, 4, "R", True)
        self.gate = float(gate)
        self.beta0 = float(beta0)

    def update(self, kf, cloud):
        cloud = validate_cloud(cloud)
        # Validate before mutation; weighted dB-like negative values are invalid.
        if self.method == "weighted" and np.any(cloud[:, 4] < 0):
            raise ValueError("Weighted requires nonnegative linear intensity, not dB SNR")
        predicted = H_RADAR @ kf.x
        S = H_RADAR @ kf.P @ H_RADAR.T + self.R
        innovations = cloud[:, :4] - predicted
        squared = np.einsum("ij,ij->i", innovations,
                            np.linalg.solve(S, innovations.T).T)
        distances = np.sqrt(np.maximum(squared, 0))
        indices = np.flatnonzero(distances <= self.gate)
        result = RadarUpdate(self.method, False, indices, distances)
        if len(indices) == 0:
            return result
        candidates = cloud[indices]
        if self.method == "strongest":
            point = candidates[np.argmax(candidates[:, 4])]
        elif self.method == "closest":
            position = kf.x[[0, 2, 4]]
            point = candidates[np.argmin(np.linalg.norm(candidates[:, :3] - position,
                                                        axis=1))]
        elif self.method in ("average", "kmeans"):
            # Single-cluster k-means minimizes sum ||point-centroid||^2:
            # its exact minimizer is the arithmetic mean of all five fields.
            point = np.mean(candidates, axis=0)
        else:
            weights = candidates[:, 4]
            if np.max(weights) <= 0:
                raise ValueError("Weighted needs positive total gated linear intensity")
            # Scaling first avoids overflow for valid but very large weights.
            relative = weights / np.max(weights)
            beta = (1 - self.beta0) * relative / np.sum(relative)
            v = innovations[indices]
            combined = beta @ v
            K = np.linalg.solve(S, H_RADAR @ kf.P).T
            residual = np.eye(6) - K @ H_RADAR
            correct_P = residual @ kf.P @ residual.T + K @ self.R @ K.T
            spread = np.einsum("i,ij,ik->jk", beta, v, v) - np.outer(combined, combined)
            posterior_P = (self.beta0 * kf.P + (1 - self.beta0) * correct_P +
                           K @ spread @ K.T)
            posterior_x = kf.x + K @ combined
            kf.x = posterior_x
            kf.P = 0.5 * (posterior_P + posterior_P.T)
            result.accepted = True
            result.innovation = combined.copy()
            result.gain = K.copy()
            result.beta = beta.copy()
            result.beta0 = self.beta0
            return result
        update = kf.update_linear(point[:4], self.R, H_RADAR)
        result.accepted = True
        result.measurement = point.copy()
        result.innovation = update.innovation
        result.gain = update.gain
        return result
