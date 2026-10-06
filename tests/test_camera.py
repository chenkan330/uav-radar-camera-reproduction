import unittest

import numpy as np

from camera import (CameraExtrinsics, CameraIntrinsics, bbox_to_pixels,
                    camera_backproject, camera_bearing_observation,
                    camera_project, paper_pseudo_observation)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(400., 500., 320., 240.)

    def test_paper_row_to_column_transform_and_translation(self):
        pose = CameraExtrinsics(translation=np.array([.1, .2, .3]))
        xyz = camera_backproject([360., 340.], 10., self.intrinsics, pose)
        # Camera [1,2,10] becomes global [10,-1,-2] minus translation.
        np.testing.assert_allclose(xyz, [9.9, -1.2, -2.3])

    def test_projection_roundtrip_with_custom_rotation(self):
        theta = .25
        attitude = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        paper_pose = CameraExtrinsics()
        pose = CameraExtrinsics(attitude @ paper_pose.rotation, [.3, -.1, .2])
        for pixels, depth in [([351., 213.], 8.), ([200., 450.], 13.)]:
            xyz = camera_backproject(pixels, depth, self.intrinsics, pose)
            np.testing.assert_allclose(camera_project(xyz, self.intrinsics, pose), pixels, atol=1e-12)

    def test_detection_confidence_and_bbox(self):
        np.testing.assert_equal(bbox_to_pixels([10, 20, 30, 40], .70), [20, 30])
        self.assertIsNone(bbox_to_pixels([10, 20, 30, 40], .699))
        for box, confidence in [([1, 2, 1, 3], .9), ([0, 0, 3, 3], 1.1), ([0, 0, np.nan, 3], .9)]:
            with self.assertRaises(ValueError):
                bbox_to_pixels(box, confidence)

    def test_bearing_jacobian_matches_numerical_projection(self):
        state = np.array([9., .3, -.7, .1, 1.2, -.2])
        pose = CameraExtrinsics(translation=[.12, -.07, .02])
        pixels = camera_project(state[[0, 2, 4]], self.intrinsics, pose) + [2., -3.]
        observation = camera_bearing_observation(pixels, state, self.intrinsics, pose, [1.5, 2.5])
        numerical = np.zeros((2, 6))
        eps = 1e-5
        for i in range(6):
            plus, minus = state.copy(), state.copy()
            plus[i] += eps
            minus[i] -= eps
            numerical[:, i] = (camera_project(plus[[0, 2, 4]], self.intrinsics, pose) -
                               camera_project(minus[[0, 2, 4]], self.intrinsics, pose)) / (2 * eps)
        np.testing.assert_allclose(observation.H, numerical, atol=1e-8)
        np.testing.assert_allclose(observation.z - observation.H @ state, [2., -3.], atol=1e-12)
        np.testing.assert_allclose(observation.R, np.diag([2.25, 6.25]))
        self.assertEqual(np.linalg.matrix_rank(observation.H), 2)

    def test_center_ray_does_not_invent_independent_depth_information(self):
        state = np.array([10., 0., 0., 0., 0., 0.])
        observation = camera_bearing_observation([320., 240.], state, self.intrinsics)
        np.testing.assert_equal(observation.H[:, [0, 1, 3, 5]], 0.)
        covariance = np.diag([9., 1., 4., 1., 4., 1.])
        H, R = observation.H, observation.R
        gain = np.linalg.solve(H @ covariance @ H.T + R, H @ covariance).T
        posterior = covariance - gain @ (H @ covariance @ H.T + R) @ gain.T
        self.assertEqual(posterior[0, 0], covariance[0, 0])
        self.assertLess(posterior[2, 2], covariance[2, 2])

    def test_pseudo_observation_covariance_and_correlation_flag(self):
        state = np.array([10., 0., 0., 0., 0., 0.])
        obs = paper_pseudo_observation([320., 240.], state, self.intrinsics,
                                       pixel_std=[2., 3.], depth_std=1.2)
        np.testing.assert_allclose(obs.position, [10., 0., 0.])
        np.testing.assert_allclose(obs.R, np.diag([1.44, .0025, .0036]))
        self.assertTrue(obs.uses_state_as_depth)
        self.assertTrue(np.all(np.linalg.eigvalsh(obs.R) > 0))
        pose = CameraExtrinsics(translation=[.5, 0., 0.])
        literal = paper_pseudo_observation([320., 240.], state, self.intrinsics, pose)
        corrected = paper_pseudo_observation([320., 240.], state, self.intrinsics, pose, depth_source="camera_forward")
        self.assertEqual(literal.position[0], 9.5)
        self.assertEqual(corrected.position[0], 10.)

    def test_invalid_depth_intrinsics_pose_and_noise(self):
        for depth in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                camera_backproject([320., 240.], depth, self.intrinsics)
        with self.assertRaises(ValueError):
            camera_project([-1., 0., 0.], self.intrinsics)
        for value in [0., -1., np.inf]:
            with self.assertRaises(ValueError):
                CameraIntrinsics(value, 500, 320, 240)
        with self.assertRaises(ValueError):
            CameraExtrinsics(rotation=np.diag([1., 1., -1.]))
        state = np.array([-1., 0., 0., 0., 0., 0.])
        with self.assertRaises(ValueError):
            camera_bearing_observation([320., 240.], state, self.intrinsics)
        state[0] = 10.
        for noise in [0., [-1., 1.], [1., 2., 3.]]:
            with self.assertRaises(ValueError):
                camera_bearing_observation([320., 240.], state, self.intrinsics, pixel_std=noise)


if __name__ == "__main__":
    unittest.main()
