"""Handwritten linear Kalman filter; state order follows the paper.

State: [x, vx, y, vy, z, vz], SI units. Step 1 uses independent,
already-aligned 3D position observations for BOTH synthetic sensors.
Only NumPy linear algebra is used, not a Kalman-filter library.
"""

from dataclasses import dataclass

import numpy as np


POSITION = np.array([0, 2, 4])
VELOCITY = np.array([1, 3, 5])
H_POSITION = np.eye(6)[POSITION]


def _covariance(value, size, name, positive_definite=False):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size}x{size} matrix")
    if not np.allclose(result, result.T, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(result)
    if (positive_definite and eigenvalues[0] <= 0) or eigenvalues[0] < -1e-12:
        raise ValueError(f"{name} must be {'positive definite' if positive_definite else 'positive semidefinite'}")
    return result


@dataclass
class UpdateResult:
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    gain: np.ndarray


class Kalman3D:
    """Constant-velocity model with a random, constant acceleration per step.

    acceleration_std is in m/s^2. For each axis, G=[dt^2/2, dt]^T
    and Q_axis = acceleration_std^2 * G @ G.T. This is a discrete
    acceleration model, not continuous white-noise spectral density.
    """

    def __init__(self, state, covariance, acceleration_std=0.8):
        self.x = np.array(state, dtype=float, copy=True)
        if self.x.shape != (6,) or not np.all(np.isfinite(self.x)):
            raise ValueError("state must contain 6 finite values")
        self.P = _covariance(covariance, 6, "P")
        if not np.isfinite(acceleration_std) or acceleration_std < 0:
            raise ValueError("acceleration_std must be finite and nonnegative")
        self.acceleration_std = float(acceleration_std)

    @classmethod
    def from_position(cls, position, R, velocity_std=2.0, acceleration_std=0.8):
        """Initialize using ONE measurement; do not update with it again."""
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must contain 3 finite values")
        R = _covariance(R, 3, "R", positive_definite=True)
        if not np.isfinite(velocity_std) or velocity_std <= 0:
            raise ValueError("velocity_std must be finite and positive")
        state = np.zeros(6)
        state[POSITION] = position
        covariance = np.zeros((6, 6))
        covariance[np.ix_(POSITION, POSITION)] = R
        covariance[np.ix_(VELOCITY, VELOCITY)] = velocity_std**2 * np.eye(3)
        return cls(state, covariance, acceleration_std)

    def motion_matrices(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        F = np.eye(6)
        F[POSITION, VELOCITY] = dt
        G = np.zeros((6, 3))
        G[POSITION, np.arange(3)] = 0.5 * dt**2
        G[VELOCITY, np.arange(3)] = dt
        Q = self.acceleration_std**2 * (G @ G.T)
        return F, Q

    def predict(self, dt):
        """Advance the state and uncertainty ONCE per time interval."""
        F, Q = self.motion_matrices(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def update(self, position, R):
        """Correct at the CURRENT time with one independent 3D observation."""
        return self.update_linear(position, R, H_POSITION)

    def update_linear(self, observation, R, H):
        """Correct with a finite m-dimensional linear model z = H x + noise.

        An EKF caller can pass its linearized equivalent z; linearization and
        source-dependent covariance belong to the sensor model, not this class.
        Validate all inputs before changing the state.
        """
        z = np.asarray(observation, dtype=float)
        H = np.asarray(H, dtype=float)
        if z.ndim != 1 or len(z) == 0 or not np.all(np.isfinite(z)):
            raise ValueError("observation must be a nonempty finite vector")
        if H.shape != (len(z), 6) or not np.all(np.isfinite(H)):
            raise ValueError("H must be a finite observation-size x 6 matrix")
        R = _covariance(R, len(z), "R", positive_definite=True)
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        # K = P H^T S^-1, calculated without an explicit matrix inverse.
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ innovation
        # Joseph form is algebraically equivalent to the paper's covariance
        # update, with better preservation of symmetry/positive semidefiniteness.
        residual_map = np.eye(6) - K @ H
        self.P = residual_map @ self.P @ residual_map.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return UpdateResult(innovation.copy(), S.copy(), K.copy())
