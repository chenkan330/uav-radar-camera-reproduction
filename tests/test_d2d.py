"""Independent geometry, observability, covariance, and innovation checks."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from d2d import (
    CameraIntrinsics, D2DEKF, EgoPose, POSITION, SensorExtrinsics,
    camera_model, radar_model, sensor_kinematics, wrap_angle,
)


def state_xyz_velocity(position, velocity):
    return np.column_stack((position, velocity)).reshape(6)


def rotation_vector(vector):
    """Independent Rodrigues rotation, also used for tiny body perturbations."""
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    x, y, z = np.asarray(vector) / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def perturb_ego(ego, index, amount):
    position = ego.position.copy()
    velocity = ego.velocity.copy()
    rotation = ego.rotation.copy()
    omega = ego.angular_velocity.copy()
    if index < 3:
        position[index] += amount
    elif index < 6:
        velocity[index - 3] += amount
    elif index < 9:
        delta = np.zeros(3)
        delta[index - 6] = amount
        rotation = rotation @ rotation_vector(delta)
    else:
        omega[index - 9] += amount
    return EgoPose(position, velocity, rotation, omega)


class MovingPlatformEKFTests(unittest.TestCase):
    def setUp(self):
        self.state = state_xyz_velocity([8, 1, 7], [.7, -.5, .2])
        self.ego = EgoPose([1, -2, .5], [.3, .8, -.1],
                           rotation_vector([.12, -.17, .35]), [.2, -.1, .7])
        self.sensor = SensorExtrinsics([.3, -.2, .1], rotation_vector([-.1, .15, -.25]))
        self.intrinsics = CameraIntrinsics(430, 410, 320, 240)

    def test_radar_state_jacobian_matches_independent_finite_differences(self):
        prediction = radar_model(self.state, self.ego, self.sensor)
        numerical = np.zeros((4, 6))
        epsilon = 1e-5
        for index in range(6):
            delta = np.eye(6)[index] * epsilon
            upper = radar_model(self.state + delta, self.ego, self.sensor).z
            lower = radar_model(self.state - delta, self.ego, self.sensor).z
            difference = upper - lower
            difference[2:] = wrap_angle(difference[2:])
            numerical[:, index] = difference / (2 * epsilon)
        assert_allclose(prediction.H, numerical, atol=2e-9, rtol=2e-8)

    def test_camera_state_jacobian_matches_independent_finite_differences(self):
        prediction = camera_model(self.state, self.ego, self.intrinsics, self.sensor)
        numerical = np.zeros((2, 6))
        epsilon = 1e-5
        for index in range(6):
            delta = np.eye(6)[index] * epsilon
            numerical[:, index] = (
                camera_model(self.state + delta, self.ego, self.intrinsics, self.sensor).z
                - camera_model(self.state - delta, self.ego, self.intrinsics, self.sensor).z
            ) / (2 * epsilon)
        assert_allclose(prediction.H, numerical, atol=2e-8, rtol=2e-8)

    def test_ego_pose_jacobians_include_rotation_and_angular_rate(self):
        # Derivatives are checked against right-multiplicative body rotations,
        # not against the same skew-matrix expressions used in the module.
        for sensor_kind in ("radar", "camera"):
            with self.subTest(sensor=sensor_kind):
                def model(ego):
                    if sensor_kind == "radar":
                        return radar_model(self.state, ego, self.sensor)
                    return camera_model(self.state, ego, self.intrinsics, self.sensor)
                prediction = model(self.ego)
                numerical = np.zeros_like(prediction.ego_jacobian)
                epsilon = 1e-5
                for index in range(12):
                    difference = model(perturb_ego(self.ego, index, epsilon)).z - model(
                        perturb_ego(self.ego, index, -epsilon),
                    ).z
                    if prediction.angle_indices:
                        difference[list(prediction.angle_indices)] = wrap_angle(difference[list(prediction.angle_indices)])
                    numerical[:, index] = difference / (2 * epsilon)
                assert_allclose(prediction.ego_jacobian, numerical, atol=3e-8, rtol=3e-8)

    def test_stationary_world_target_on_forward_moving_platform_has_negative_doppler(self):
        target = state_xyz_velocity([10, 0, 0], [0, 0, 0])
        ego = EgoPose([0, 0, 0], [2, 0, 0])
        prediction = radar_model(target, ego)
        assert_allclose(prediction.z, [10, -2, 0, 0])
        # Tangential world velocity is not radar's radial velocity.
        target = state_xyz_velocity([10, 0, 0], [0, 3, 0])
        self.assertEqual(radar_model(target, EgoPose([0, 0, 0], [0, 0, 0])).z[1], 0)

    def test_rotating_lever_arm_changes_sensor_origin_velocity_and_doppler(self):
        ego = EgoPose([0, 0, 0], [0, 0, 0], angular_velocity=[0, 0, 2])
        extrinsic = SensorExtrinsics([0, 1, 0])
        sensor = sensor_kinematics(ego, extrinsic)
        assert_allclose(sensor.position, [0, 1, 0])
        assert_allclose(sensor.velocity, [-2, 0, 0])
        target = state_xyz_velocity([10, 1, 0], [0, 0, 0])
        assert_allclose(radar_model(target, ego, extrinsic).z[:2], [10, 2])
        rotated_ego = EgoPose([0, 0, 0], [0, 0, 0], rotation_vector([0, 0, np.pi / 2]), [0, 0, 2])
        sensor = sensor_kinematics(rotated_ego, extrinsic)
        assert_allclose(sensor.position, [-1, 0, 0], atol=1e-15)
        assert_allclose(sensor.velocity, [0, -2, 0], atol=1e-15)

    def test_rotation_at_zero_lever_arm_does_not_invent_radial_velocity(self):
        target = state_xyz_velocity([10, 2, 3], [.4, -.2, .7])
        static = EgoPose([0, 0, 0], [0, 0, 0])
        rotating = EgoPose([0, 0, 0], [0, 0, 0], rotation_vector([0, 0, 1.2]), [0, 0, 20])
        assert_allclose(radar_model(target, static).z[:2], radar_model(target, rotating).z[:2])

    def test_single_camera_view_does_not_create_depth_information(self):
        state = state_xyz_velocity([1, 2, 10], [0, 0, 0])
        ego = EgoPose([0, 0, 0], [0, 0, 0])
        tracker = D2DEKF(state, np.eye(6), acceleration_psd=0)
        prediction = tracker.camera_prediction(ego, self.intrinsics)
        radial = np.zeros(6)
        radial[POSITION] = state[POSITION] / np.linalg.norm(state[POSITION])
        assert_allclose(prediction.H @ radial, [0, 0], atol=1e-14)
        prior_radial_variance = float(radial @ tracker.P @ radial)
        tracker.update_camera(prediction.z, np.eye(2), ego, self.intrinsics)
        self.assertAlmostEqual(float(radial @ tracker.P @ radial), prior_radial_variance, places=13)
        # Pixels are invariant to positive scaling along the same line of sight.
        doubled = state.copy()
        doubled[POSITION] *= 2
        assert_allclose(camera_model(doubled, ego, self.intrinsics).z, prediction.z)

    def test_continuous_noise_prediction_composes_across_nonuniform_splits(self):
        rng = np.random.default_rng(71)
        matrix = rng.normal(size=(6, 6))
        prior = matrix @ matrix.T
        entire = D2DEKF(self.state, prior, [.2, .4, .7])
        split = D2DEKF(self.state, prior, [.2, .4, .7])
        entire.predict(.73)
        for interval in [.11, .07, .23, .32]:
            split.predict(interval)
        assert_allclose(split.x, entire.x, atol=1e-14, rtol=1e-14)
        assert_allclose(split.P, entire.P, atol=2e-14, rtol=2e-14)
        F, Q = entire.motion_matrices(2)
        assert_allclose(F[:2, :2], [[1, 2], [0, 1]])
        assert_allclose(Q[:2, :2], .2 * np.array([[8 / 3, 2], [2, 2]]))

    def test_nis_gate_rejects_outlier_without_mutating_state(self):
        target = state_xyz_velocity([10, 0, 0], [0, 0, 0])
        tracker = D2DEKF(target, np.eye(6))
        ego = EgoPose([0, 0, 0], [0, 0, 0])
        predicted = tracker.radar_prediction(ego, include_angles=False)
        before_x, before_P = tracker.x.copy(), tracker.P.copy()
        # S_range=2, S_Doppler=2; a +10 m range outlier gives NIS=50.
        self.assertEqual(tracker.nis([20, 0], np.eye(2), predicted), 50)
        result = tracker.update_radar([20, 0], np.eye(2), ego, include_angles=False, gate_nis=9)
        self.assertFalse(result.accepted)
        self.assertEqual(result.nis, 50)
        assert_array_equal(tracker.x, before_x)
        assert_array_equal(tracker.P, before_P)
        accepted = tracker.update_radar([11, 0], np.eye(2), ego, include_angles=False, gate_nis=9)
        self.assertTrue(accepted.accepted)
        self.assertAlmostEqual(tracker.x[0], 10.5)
        self.assertAlmostEqual(tracker.P[0, 0], .5)

    def test_azimuth_boundary_innovation_is_wrapped(self):
        angle = np.pi - .01
        state = state_xyz_velocity([10 * np.cos(angle), 10 * np.sin(angle), 0], [0, 0, 0])
        ego = EgoPose([0, 0, 0], [0, 0, 0])
        tracker = D2DEKF(state, np.eye(6))
        result = tracker.update_radar([10, 0, -np.pi + .01, 0], np.diag([1, 1, .01, .01]), ego)
        self.assertAlmostEqual(result.innovation[2], .02)
        self.assertLess(result.nis, .05)

    def test_optional_ego_uncertainty_weakens_correction(self):
        state = state_xyz_velocity([10, 0, 0], [0, 0, 0])
        pose_covariance = np.zeros((12, 12))
        pose_covariance[0, 0] = 4
        uncertain = EgoPose([0, 0, 0], [0, 0, 0], covariance=pose_covariance)
        included = D2DEKF(state, np.eye(6))
        ignored = D2DEKF(state, np.eye(6))
        a = included.update_radar([11, 0], np.eye(2), uncertain, include_angles=False)
        b = ignored.update_radar([11, 0], np.eye(2), uncertain, include_angles=False, propagate_ego_uncertainty=False)
        self.assertEqual(a.effective_R[0, 0], 5)
        self.assertEqual(b.effective_R[0, 0], 1)
        self.assertAlmostEqual(included.x[0], 10 + 1 / 6)
        self.assertAlmostEqual(ignored.x[0], 10.5)
        self.assertGreater(included.P[0, 0], ignored.P[0, 0])

    def test_long_mixed_sensor_updates_keep_covariance_symmetric_and_psd(self):
        ego = EgoPose([0, 0, 0], [.2, -.1, .1], angular_velocity=[0, .2, 0])
        tracker = D2DEKF(self.state, np.eye(6), [.2, .3, .4])
        for _ in range(200):
            tracker.predict(.013)
            radar = tracker.radar_prediction(ego)
            tracker.update_radar(radar.z + [0.01, -.02, .001, -.001], np.diag([.04, .01, .0004, .0004]), ego)
            camera = tracker.camera_prediction(ego, self.intrinsics)
            tracker.update_camera(camera.z + [.1, -.1], np.eye(2), ego, self.intrinsics)
            self.assertGreaterEqual(np.linalg.eigvalsh(tracker.P).min(), -1e-12)
            assert_allclose(tracker.P, tracker.P.T, atol=1e-14, rtol=0)

    def test_bad_ego_extrinsics_covariance_and_intrinsics_are_rejected(self):
        invalid_calls = [
            lambda: EgoPose([np.nan, 0, 0], [0, 0, 0]),
            lambda: EgoPose([0, 0, 0], [0, 0, 0], np.diag([1, 1, -1])),
            lambda: EgoPose([0, 0, 0], [0, 0, 0], covariance=np.eye(6)),
            lambda: EgoPose([0, 0, 0], [0, 0, 0], covariance=-np.eye(12)),
            lambda: SensorExtrinsics([0, np.inf, 0]),
            lambda: SensorExtrinsics(rotation=np.ones((3, 3))),
            lambda: CameraIntrinsics(0, 420, 320, 240),
            lambda: D2DEKF(self.state, -np.eye(6)),
            lambda: D2DEKF(self.state, np.eye(6), acceleration_psd=[1, -1, 2]),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_singular_geometry_and_bad_measurements_do_not_change_filter(self):
        ego = EgoPose([0, 0, 0], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "origin"):
            radar_model(np.zeros(6), ego)
        vertical = state_xyz_velocity([0, 0, 10], [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "vertical"):
            radar_model(vertical, ego)
        assert_allclose(radar_model(vertical, ego, include_angles=False).z, [10, 0])
        with self.assertRaisesRegex(ValueError, "depth"):
            camera_model(state_xyz_velocity([1, 2, -1], [0, 0, 0]), ego, self.intrinsics)
        tracker = D2DEKF(self.state, np.eye(6))
        before_x, before_P = tracker.x.copy(), tracker.P.copy()
        for observation, covariance in [
            ([np.nan, 0, 0, 0], np.eye(4)),
            ([-1, 0, 0, 0], np.eye(4)),
            ([10, 0, 0, np.pi], np.eye(4)),
            ([10, 0, 0, 0], np.zeros((4, 4))),
            ([10, 0, 0, 0], np.diag([1, -1, 1, 1])),
        ]:
            with self.subTest(observation=observation):
                with self.assertRaises(ValueError):
                    tracker.update_radar(observation, covariance, ego)
                assert_array_equal(tracker.x, before_x)
                assert_array_equal(tracker.P, before_P)
        with self.assertRaises(ValueError):
            tracker.update_camera([np.inf, 0], np.eye(2), ego, self.intrinsics)
        for interval in [0, -1, np.nan, None, "invalid", [1, 2], 1e200]:
            with self.assertRaises(ValueError):
                tracker.predict(interval)
        with self.assertRaises(ValueError):
            tracker.update_radar([10, 0], np.eye(2), ego, include_angles=False, gate_nis=0)
        assert_array_equal(tracker.x, before_x)
        assert_array_equal(tracker.P, before_P)

    def test_truth_is_not_a_legal_observation_input(self):
        # The complete update requires only an initial estimate, ego kinematics,
        # one raw sensor observation and its covariance; no truth argument.
        tracker = D2DEKF(self.state, np.eye(6))
        prediction = tracker.radar_prediction(self.ego, self.sensor)
        result = tracker.update_radar(prediction.z, np.eye(4), self.ego, self.sensor)
        self.assertTrue(result.accepted)
        assert_allclose(tracker.x, self.state)
        with self.assertRaises(TypeError):
            tracker.update_radar(prediction.z, np.eye(4), self.ego, self.sensor, ground_truth=self.state)


if __name__ == "__main__":
    unittest.main()
