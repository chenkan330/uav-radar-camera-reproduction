"""Calibrated bounding-box observations, following paper equations 11--15.

The paper's XYZ pseudo-observation reuses the current state's x as depth;
it is consequently correlated with that state. The recommended bearing
observation instead updates two pixel coordinates using a pinhole Jacobian.
No image detector or camera calibration is trained by this module.
"""

from dataclasses import dataclass, field

import numpy as np

from kalman3d import H_POSITION, POSITION


PAPER_CAMERA_TO_GLOBAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _pixel_std(value):
    result = np.asarray(value, dtype=float)
    if result.shape == ():
        result = np.repeat(result, 2)
    if result.shape != (2,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("pixel_std must contain one or two finite positive values")
    return result


@dataclass(frozen=True)
class CameraIntrinsics:
    fu: float
    fv: float
    u0: float
    v0: float

    def __post_init__(self):
        values = np.array([self.fu, self.fv, self.u0, self.v0], dtype=float)
        if not np.all(np.isfinite(values)) or self.fu <= 0 or self.fv <= 0:
            raise ValueError("camera focal lengths must be positive; all intrinsics must be finite")


@dataclass(frozen=True)
class CameraExtrinsics:
    """Column-vector convention: global_xyz = rotation @ camera_xyz - translation.

    The paper writes row_camera @ M - t. Its equivalent column rotation
    is M.T, not M. A custom rotation can include a measured attitude.
    Translation retains the sign convention in equation 15.
    """

    rotation: np.ndarray = field(default_factory=lambda: PAPER_CAMERA_TO_GLOBAL.copy())
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        rotation = np.array(self.rotation, dtype=float, copy=True)
        translation = _vector(self.translation, 3, "translation").copy()
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10):
            raise ValueError("rotation must be a proper orthonormal rotation")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)


@dataclass
class CameraObservation:
    z: np.ndarray
    H: np.ndarray
    R: np.ndarray
    mode: str
    depth: float
    position: np.ndarray | None = None
    uses_state_as_depth: bool = False


def bbox_to_pixels(bbox_xyxy, confidence, threshold=0.70):
    """Return the center of a valid [xmin,ymin,xmax,ymax] detection or None.

    Confidence exactly at the threshold is accepted. Coordinates are pixels,
    not normalized box coordinates. Bad geometry is rejected explicitly.
    """
    box = _vector(bbox_xyxy, 4, "bbox_xyxy")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bounding-box width and height must be positive")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between zero and one")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if confidence < threshold:
        return None
    return 0.5 * (box[:2] + box[2:])


def camera_backproject(pixels, depth, intrinsics, extrinsics=None):
    """Apply equations 12--15 with an explicitly supplied optical depth."""
    pixels = _vector(pixels, 2, "pixels")
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("camera depth must be finite and positive")
    pose = extrinsics if extrinsics is not None else CameraExtrinsics()
    point = np.array([(pixels[0] - intrinsics.u0) * depth / intrinsics.fu,
                      (pixels[1] - intrinsics.v0) * depth / intrinsics.fv, depth])
    return pose.rotation @ point - pose.translation


def camera_project(global_xyz, intrinsics, extrinsics=None):
    """Inverse extrinsics followed by a pinhole projection; no distortion."""
    point = _vector(global_xyz, 3, "global_xyz")
    pose = extrinsics if extrinsics is not None else CameraExtrinsics()
    camera_xyz = pose.rotation.T @ (point + pose.translation)
    if camera_xyz[2] <= 0:
        raise ValueError("point must have positive camera-forward depth")
    return np.array([intrinsics.u0 + intrinsics.fu * camera_xyz[0] / camera_xyz[2],
                     intrinsics.v0 + intrinsics.fv * camera_xyz[1] / camera_xyz[2]])


def paper_pseudo_observation(pixels, state, intrinsics, extrinsics=None,
                             pixel_std=2.0, depth_std=2.0,
                             depth_source="paper_state_x"):
    """Paper-compatible XYZ observation; its copied depth is NOT independent.

    depth_source='paper_state_x' reproduces the literal X(0,0) in equations
    12--14. 'camera_forward' uses the calibrated optical depth of the state
    for custom poses. R propagates a configurable hypothetical depth_std;
    it does not repair correlation with the state. Prefer bearing updates.
    """
    pixels = _vector(pixels, 2, "pixels")
    state = _vector(state, 6, "state")
    sigma = _pixel_std(pixel_std)
    if not np.isfinite(depth_std) or depth_std <= 0:
        raise ValueError("depth_std must be finite and positive")
    pose = extrinsics if extrinsics is not None else CameraExtrinsics()
    if depth_source == "paper_state_x":
        depth = float(state[0])
    elif depth_source == "camera_forward":
        depth = float((pose.rotation.T @ (state[POSITION] + pose.translation))[2])
    else:
        raise ValueError("depth_source must be paper_state_x or camera_forward")
    xyz = camera_backproject(pixels, depth, intrinsics, pose)
    du = (pixels[0] - intrinsics.u0) / intrinsics.fu
    dv = (pixels[1] - intrinsics.v0) / intrinsics.fv
    J = np.array([[depth / intrinsics.fu, 0, du],
                  [0, depth / intrinsics.fv, dv], [0, 0, 1.]])
    covariance_camera = J @ np.diag([sigma[0]**2, sigma[1]**2, depth_std**2]) @ J.T
    R = pose.rotation @ covariance_camera @ pose.rotation.T
    return CameraObservation(xyz.copy(), H_POSITION.copy(), R, "paper_xyz",
                             depth, xyz.copy(), True)


def camera_bearing_observation(pixels, state, intrinsics, extrinsics=None,
                               pixel_std=2.0):
    """Return a two-row EKF observation for a generic linear Kalman update.

    h(state) is the pixel projection. At this state's linearization point,
    H is dh/dstate and z = pixels - h(state) + H@state, so the ordinary
    innovation z-H@state equals the observed minus predicted pixels.
    R is pixel noise covariance. There is no artificial depth measurement.
    Recompute this observation immediately before each update.
    """
    pixels = _vector(pixels, 2, "pixels")
    state = _vector(state, 6, "state")
    sigma = _pixel_std(pixel_std)
    pose = extrinsics if extrinsics is not None else CameraExtrinsics()
    camera_xyz = pose.rotation.T @ (state[POSITION] + pose.translation)
    cx, cy, depth = camera_xyz
    if depth <= 0:
        raise ValueError("state must have positive camera-forward depth")
    predicted_pixels = camera_project(state[POSITION], intrinsics, pose)
    J = np.array([[intrinsics.fu / depth, 0., -intrinsics.fu * cx / depth**2],
                  [0., intrinsics.fv / depth, -intrinsics.fv * cy / depth**2]])
    H = J @ pose.rotation.T @ H_POSITION
    z = pixels - predicted_pixels + H @ state
    return CameraObservation(z, H, np.diag(sigma**2), "bearing", float(depth))
