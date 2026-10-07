"""Version 2 research prototype: moving-platform radar and pixel EKF.

Target state is world-frame [x,vx,y,vy,z,vz]. Ego orientation maps body to
world; angular velocity and sensor lever arms are expressed in body axes.
Radar +x is forward; camera +z is optical forward. Doppler is positive when
range increases. No detector, multi-target association, ROS driver, or truth
input is provided. Ego covariance propagation assumes independence from the
target state and ignores temporal and cross-sensor correlations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


POSITION = np.array([0, 2, 4])
VELOCITY = np.array([1, 3, 5])
_GEOMETRY_EPS = 1e-9


def _positive_scalar(value, name):
    try:
        scalar = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if scalar.shape != () or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(scalar)


def _vector(value, size, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _rotation(value, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3x3 rotation")
    if not np.allclose(result.T @ result, np.eye(3), rtol=0, atol=1e-10) or not np.isclose(
        np.linalg.det(result), 1.0, rtol=0, atol=1e-10,
    ):
        raise ValueError(f"{name} must be a proper orthonormal rotation")
    return result


def _covariance(value, size, name, positive_definite=False):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size, size) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}x{size} covariance")
    if not np.allclose(result, result.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    minimum = np.linalg.eigvalsh(result)[0]
    if minimum < -1e-12 or (positive_definite and minimum <= 0):
        label = "positive definite" if positive_definite else "positive semidefinite"
        raise ValueError(f"{name} must be {label}")
    return 0.5 * (result + result.T)


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _readonly(value):
    value.setflags(write=False)
    return value


def wrap_angle(value):
    """Map a finite angle/array to [-pi, pi)."""
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("angle must be finite")
    return (value + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class EgoPose:
    """Ego kinematics at the measurement's acquisition time.

    Covariance order is [delta_p_world(3), delta_v_world(3),
    delta_theta_body(3), delta_omega_body(3)]. Orientation perturbations are
    right-multiplicative: R_actual = R @ exp(skew(delta_theta_body)).
    The supplied covariance does not establish independence across updates.
    """

    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray | None = None

    def __post_init__(self):
        for name in ("position", "velocity", "angular_velocity"):
            object.__setattr__(self, name, _readonly(_vector(getattr(self, name), 3, name)))
        object.__setattr__(self, "rotation", _readonly(_rotation(self.rotation, "ego rotation")))
        if self.covariance is not None:
            object.__setattr__(self, "covariance", _readonly(_covariance(self.covariance, 12, "ego covariance")))


@dataclass(frozen=True)
class SensorExtrinsics:
    """Body-frame lever arm and rotation R_body_from_sensor, in SI units."""

    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        object.__setattr__(self, "lever_arm", _readonly(_vector(self.lever_arm, 3, "lever_arm")))
        object.__setattr__(self, "rotation", _readonly(_rotation(self.rotation, "sensor rotation")))


@dataclass(frozen=True)
class CameraIntrinsics:
    fu: float
    fv: float
    u0: float
    v0: float

    def __post_init__(self):
        values = _vector([self.fu, self.fv, self.u0, self.v0], 4, "camera intrinsics")
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        for name, value in zip(("fu", "fv", "u0", "v0"), values):
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class SensorKinematics:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray


def sensor_kinematics(ego, extrinsic=None):
    """Sensor-center world p/v and R_world_from_sensor, including rotation."""
    if not isinstance(ego, EgoPose):
        raise ValueError("ego must be an EgoPose at acquisition time")
    extrinsic = SensorExtrinsics() if extrinsic is None else extrinsic
    if not isinstance(extrinsic, SensorExtrinsics):
        raise ValueError("extrinsic must be SensorExtrinsics")
    return SensorKinematics(
        ego.position + ego.rotation @ extrinsic.lever_arm,
        ego.velocity + ego.rotation @ np.cross(ego.angular_velocity, extrinsic.lever_arm),
        ego.rotation @ extrinsic.rotation,
    )


@dataclass
class SensorPrediction:
    z: np.ndarray
    H: np.ndarray
    ego_covariance: np.ndarray
    ego_jacobian: np.ndarray
    angle_indices: tuple[int, ...] = ()

    @property
    def h(self):
        return self.z


def _geometry(state, ego, extrinsic):
    state = _vector(state, 6, "state")
    extrinsic = SensorExtrinsics() if extrinsic is None else extrinsic
    sensor = sensor_kinematics(ego, extrinsic)
    relative = state[POSITION] - sensor.position
    q = sensor.rotation.T @ relative
    # q = R_BS.T @ (R_WB.T @ (p_target-p_ego) - lever).
    # A right-multiplicative attitude perturbation acts on the entire relative
    # position before the lever arm is subtracted, not just on q.
    body_relative = ego.rotation.T @ (state[POSITION] - ego.position)
    Jq_ego = np.zeros((3, 12))
    Jq_ego[:, :3] = -sensor.rotation.T
    Jq_ego[:, 6:9] = extrinsic.rotation.T @ _skew(body_relative)
    return state, extrinsic, sensor, relative, q, Jq_ego


def _extra_covariance(J, ego, enabled):
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("propagate_ego_uncertainty must be boolean")
    extra = np.zeros((J.shape[0], J.shape[0]))
    if enabled and ego.covariance is not None:
        extra = J @ ego.covariance @ J.T
    if not np.isfinite(extra).all():
        raise ValueError("propagated ego covariance exceeds finite numerical range")
    return 0.5 * (extra + extra.T)


def radar_model(state, ego, extrinsic=None, include_angles=True, propagate_ego_uncertainty=True):
    """Predict [range, range_rate, azimuth, elevation], or first two values.

    Azimuth=atan2(y_sensor,x_sensor), elevation=atan2(z_sensor,hypot(x,y)).
    H is the analytic Jacobian with respect to the world-frame target state.
    The ego Jacobian includes the angular velocity of an offset sensor origin.
    """
    if not isinstance(include_angles, (bool, np.bool_)):
        raise ValueError("include_angles must be boolean")
    state, extrinsic, sensor, relative, q, Jq = _geometry(state, ego, extrinsic)
    distance = float(np.linalg.norm(relative))
    if distance <= _GEOMETRY_EPS:
        raise ValueError("radar target range is singular at the sensor origin")
    direction = relative / distance
    relative_velocity = state[VELOCITY] - sensor.velocity
    range_rate = float(direction @ relative_velocity)
    range_rate_position = (relative_velocity - direction * range_rate) / distance
    size = 4 if include_angles else 2
    H = np.zeros((size, 6))
    H[0, POSITION] = direction
    H[1, POSITION] = range_rate_position
    H[1, VELOCITY] = direction
    J = np.zeros((size, 12))
    J[0, :3] = -direction
    J[0, 6:9] = direction @ ego.rotation @ _skew(extrinsic.lever_arm)
    J[1, :3] = -range_rate_position
    J[1, 3:6] = -direction
    center_body_velocity = np.cross(ego.angular_velocity, extrinsic.lever_arm)
    J[1, 6:9] = (
        range_rate_position @ ego.rotation @ _skew(extrinsic.lever_arm)
        + direction @ ego.rotation @ _skew(center_body_velocity)
    )
    J[1, 9:12] = direction @ ego.rotation @ _skew(extrinsic.lever_arm)
    z = np.array([distance, range_rate])
    if include_angles:
        horizontal = float(np.hypot(q[0], q[1]))
        if horizontal <= _GEOMETRY_EPS:
            raise ValueError("radar azimuth is singular on the sensor vertical axis")
        angle_q = np.array([
            [-q[1] / horizontal**2, q[0] / horizontal**2, 0.0],
            [-q[0] * q[2] / (distance**2 * horizontal),
             -q[1] * q[2] / (distance**2 * horizontal), horizontal / distance**2],
        ])
        H[2:, POSITION] = angle_q @ sensor.rotation.T
        J[2:] = angle_q @ Jq
        z = np.r_[z, np.arctan2(q[1], q[0]), np.arctan2(q[2], horizontal)]
    if not np.isfinite(z).all() or not np.isfinite(H).all() or not np.isfinite(J).all():
        raise ValueError("radar geometry exceeds finite numerical range")
    return SensorPrediction(z, H, _extra_covariance(J, ego, propagate_ego_uncertainty), J,
                            (2, 3) if include_angles else ())


def camera_model(state, ego, intrinsics, extrinsic=None, propagate_ego_uncertainty=True):
    """Predict true pinhole pixels, without a synthetic depth observation."""
    if not isinstance(intrinsics, CameraIntrinsics):
        raise ValueError("intrinsics must be CameraIntrinsics")
    _, _, sensor, _, q, Jq = _geometry(state, ego, extrinsic)
    if q[2] <= _GEOMETRY_EPS:
        raise ValueError("camera target must have nonsingular positive optical depth")
    z = np.array([intrinsics.u0 + intrinsics.fu * q[0] / q[2],
                  intrinsics.v0 + intrinsics.fv * q[1] / q[2]])
    Jpixel = np.array([[intrinsics.fu / q[2], 0.0, -intrinsics.fu * q[0] / q[2]**2],
                       [0.0, intrinsics.fv / q[2], -intrinsics.fv * q[1] / q[2]**2]])
    H = np.zeros((2, 6))
    H[:, POSITION] = Jpixel @ sensor.rotation.T
    J = Jpixel @ Jq
    if not np.isfinite(z).all() or not np.isfinite(H).all() or not np.isfinite(J).all():
        raise ValueError("camera geometry exceeds finite numerical range")
    return SensorPrediction(z, H, _extra_covariance(J, ego, propagate_ego_uncertainty), J)


@dataclass
class InnovationStatistics:
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    gain: np.ndarray
    effective_R: np.ndarray
    nis: float


@dataclass
class UpdateResult(InnovationStatistics):
    accepted: bool


class D2DEKF:
    """Single-target EKF with continuous white-acceleration process noise.

    acceleration_psd is a scalar or three world-axis spectral densities, in
    m^2/s^3. Per-axis Q=q[[dt^3/3,dt^2/2],[dt^2/2,dt]] composes exactly
    across split prediction intervals. An explicit initial estimate/P is
    required; the class never obtains an initial state from ground truth.
    """

    def __init__(self, state, covariance, acceleration_psd=0.2):
        self.x = _vector(state, 6, "state")
        self.P = _covariance(covariance, 6, "P")
        psd = np.array(acceleration_psd, dtype=float, copy=True)
        if psd.shape == ():
            psd = np.repeat(psd, 3)
        if psd.shape != (3,) or not np.isfinite(psd).all() or np.any(psd < 0):
            raise ValueError("acceleration_psd must be one or three finite nonnegative values")
        self.acceleration_psd = psd

    def motion_matrices(self, dt):
        dt = _positive_scalar(dt, "dt")
        F = np.eye(6)
        F[POSITION, VELOCITY] = dt
        Q = np.zeros((6, 6))
        interval = np.float64(dt)
        for axis, q in enumerate(self.acceleration_psd):
            pair = [2 * axis, 2 * axis + 1]
            with np.errstate(over="ignore", invalid="ignore"):
                Q[np.ix_(pair, pair)] = q * np.array(
                    [[interval**3 / 3, interval**2 / 2], [interval**2 / 2, interval]],
                )
        if not np.isfinite(F).all() or not np.isfinite(Q).all():
            raise ValueError("prediction interval exceeds finite numerical range")
        return F, Q

    def predict(self, dt):
        F, Q = self.motion_matrices(dt)
        state = F @ self.x
        covariance = F @ self.P @ F.T + Q
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise ValueError("predicted state/covariance exceeds finite numerical range")
        self.x = state
        self.P = 0.5 * (covariance + covariance.T)
        return self.x.copy(), self.P.copy()

    def radar_prediction(self, ego, extrinsic=None, include_angles=True, propagate_ego_uncertainty=True):
        return radar_model(self.x, ego, extrinsic, include_angles, propagate_ego_uncertainty)

    def camera_prediction(self, ego, intrinsics, extrinsic=None, propagate_ego_uncertainty=True):
        return camera_model(self.x, ego, intrinsics, extrinsic, propagate_ego_uncertainty)

    def innovation_statistics(self, observation, R, prediction):
        if not isinstance(prediction, SensorPrediction):
            raise ValueError("prediction must be SensorPrediction at the current state")
        predicted = np.asarray(prediction.z, dtype=float)
        if predicted.ndim != 1 or not len(predicted) or not np.isfinite(predicted).all():
            raise ValueError("predicted measurement must be a nonempty finite vector")
        size = len(predicted)
        observation = _vector(observation, size, "observation")
        H = np.asarray(prediction.H, dtype=float)
        if H.shape != (size, 6) or not np.isfinite(H).all():
            raise ValueError("measurement Jacobian must be finite measurement-size x 6")
        R = _covariance(R, size, "R", positive_definite=True)
        extra = _covariance(prediction.ego_covariance, size, "propagated ego covariance")
        effective_R = R + extra
        innovation = observation - predicted
        indices = prediction.angle_indices
        if len(set(indices)) != len(indices) or any(not isinstance(index, int) or not 0 <= index < size for index in indices):
            raise ValueError("angle_indices must be distinct valid integer indices")
        if indices:
            innovation[list(indices)] = wrap_angle(innovation[list(indices)])
        S = _covariance(H @ self.P @ H.T + effective_R, size, "innovation covariance", True)
        gain = np.linalg.solve(S, H @ self.P).T
        nis = float(innovation @ np.linalg.solve(S, innovation))
        if not np.isfinite(nis) or not np.isfinite(gain).all():
            raise ValueError("innovation statistics exceed finite numerical range")
        return InnovationStatistics(innovation, S, gain, effective_R, nis)

    def nis(self, observation, R, prediction):
        """Squared normalized innovation; no state mutation."""
        return self.innovation_statistics(observation, R, prediction).nis

    def update(self, observation, R, prediction, gate_nis=None):
        """EKF correction; a finite squared-NIS gate rejects outliers unchanged.

        Invalid measurements/geometry raise ValueError rather than being counted
        as an ordinary gated outlier. Prediction must be linearized at self.x.
        """
        if gate_nis is not None:
            gate_nis = _positive_scalar(gate_nis, "gate_nis")
        stats = self.innovation_statistics(observation, R, prediction)
        accepted = gate_nis is None or stats.nis <= gate_nis
        if accepted:
            H = np.asarray(prediction.H, dtype=float)
            state = self.x + stats.gain @ stats.innovation
            residual = np.eye(6) - stats.gain @ H
            covariance = residual @ self.P @ residual.T + stats.gain @ stats.effective_R @ stats.gain.T
            if not np.isfinite(state).all() or not np.isfinite(covariance).all():
                raise ValueError("updated state/covariance exceeds finite numerical range")
            self.x = state
            self.P = 0.5 * (covariance + covariance.T)
        return UpdateResult(**vars(stats), accepted=accepted)

    def update_radar(self, observation, R, ego, extrinsic=None, include_angles=True,
                     gate_nis=None, propagate_ego_uncertainty=True):
        size = 4 if include_angles else 2
        observation = _vector(observation, size, "radar observation")
        if observation[0] <= 0:
            raise ValueError("observed radar range must be positive")
        if include_angles and abs(observation[3]) > np.pi / 2:
            raise ValueError("observed radar elevation must be in [-pi/2, pi/2]")
        prediction = self.radar_prediction(ego, extrinsic, include_angles, propagate_ego_uncertainty)
        return self.update(observation, R, prediction, gate_nis)

    def update_camera(self, observation, R, ego, intrinsics, extrinsic=None,
                      gate_nis=None, propagate_ego_uncertainty=True):
        prediction = self.camera_prediction(ego, intrinsics, extrinsic, propagate_ego_uncertainty)
        return self.update(observation, R, prediction, gate_nis)
