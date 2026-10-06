"""Delayed camera replay and independent linear observation checks."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from camera import CameraIntrinsics, CameraExtrinsics
from kalman3d import Kalman3D
from fusion import FusionTracker, FusionConfig
from demo_step4 import make_events, run_events


class FusionTests(unittest.TestCase):
    def test_delayed_raw_boxes_and_clouds_match_ordered_full_covariance(self):
        events, _, intrinsics, pose=make_events(21,2.)
        for mode in ("paper_xyz","bearing"):
            with self.subTest(mode=mode):
                cfg=FusionConfig(camera_mode=mode)
                delayed=run_events(events,intrinsics,pose,cfg)
                ordered=run_events(events,intrinsics,pose,cfg,True)
                t,x,P=delayed.tick_history()
                rt,rx,rP=ordered.tick_history()
                assert_allclose(t,rt,atol=1e-12,rtol=0)
                assert_allclose(x,rx,atol=1e-12,rtol=1e-12)
                assert_allclose(P,rP,atol=1e-12,rtol=1e-12)
                self.assertGreater(delayed.late_events,0)
                self.assertGreaterEqual(np.linalg.eigvalsh(P).min(),-1e-12)

    def test_camera_alone_cannot_initialize_or_create_depth(self):
        tracker=FusionTracker(CameraIntrinsics(400,400,320,240),CameraExtrinsics())
        tracker.ingest("camera",0.,.05,"c0",bbox=[310,230,330,250],confidence=.95)
        tracker.advance_to(.1)
        self.assertIsNone(tracker.filter)
        self.assertEqual(tracker.tick_history()[0].size,0)

    def test_weighted_zero_gated_intensity_is_rejected_without_losing_prediction(self):
        cfg=FusionConfig(radar_method="weighted", acceleration_std=0.)
        tracker=FusionTracker(CameraIntrinsics(400,400,320,240),CameraExtrinsics(),cfg)
        for k in range(2):
            tracker.ingest("radar",k*.1,k*.1,str(k),cloud=[[4,0,0,0,1]])
        tracker.ingest("radar",.2,.2,"2",cloud=[[4,0,0,0,0],[100,0,0,0,1]])
        item=next(e for e in tracker.items if e.key=="radar:2")
        self.assertFalse(item.diagnostics["accepted"])
        assert_allclose(tracker.filter.x,[4,0,0,0,0,0])
        self.assertTrue(np.isfinite(tracker.filter.P).all())

    def test_invalid_process_noise_rejected_before_any_event(self):
        for value in (-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):
                FusionTracker(CameraIntrinsics(400,400,320,240),CameraExtrinsics(),FusionConfig(acceleration_std=value))

    def test_generic_four_dimensional_update_matches_information_solution(self):
        H=np.eye(6)[[0,2,4,1]]
        R=np.diag([1.,2.,3.,4.])
        prior=np.eye(6)*2
        z=np.array([4.,2.,1.,.5])
        tracker=Kalman3D(np.zeros(6),prior)
        tracker.update_linear(z,R,H)
        information=np.linalg.inv(prior)+H.T@np.linalg.solve(R,H)
        assert_allclose(tracker.x,np.linalg.solve(information,H.T@np.linalg.solve(R,z)))
        assert_allclose(tracker.P,np.linalg.inv(information))
        before=tracker.x.copy()
        with self.assertRaises(ValueError):tracker.update_linear(z,R,H[:3])
        assert_allclose(tracker.x,before)


if __name__=="__main__":unittest.main()
