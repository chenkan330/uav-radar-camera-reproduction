"""Independent radar-only checks of delayed raw-measurement replay."""

import unittest

import numpy as np

from camera import CameraExtrinsics, CameraIntrinsics
from fusion import FusionConfig, FusionTracker
from kalman3d import Kalman3D
from radar import RadarAssociator, RadarInitializer


class FusionRadarReplayTests(unittest.TestCase):
    def tracker(self, **kwargs):
        return FusionTracker(CameraIntrinsics(400, 400, 320, 240),
                             CameraExtrinsics(), FusionConfig(**kwargs))

    def cloud(self, x=4.0, vx=0.):
        return np.array([[x, 0., 1.5, vx, 10.]])

    def initialize(self, tracker):
        self.assertFalse(tracker.ingest("radar", 0., 0., "first", cloud=self.cloud()))
        self.assertTrue(tracker.ingest("radar", .1, .1, "second", cloud=self.cloud()))

    def test_delayed_raw_cloud_is_gated_again_after_prior_correction(self):
        delayed = self.tracker(gate=2.)
        self.initialize(delayed)
        delayed.ingest("radar", .3, .3, "later", cloud=self.cloud(4.70))
        later = next(item for item in delayed.items if item.key == "radar:later")
        self.assertFalse(later.diagnostics["accepted"])
        self.assertEqual(later.diagnostics["gated_count"], 0)

        delayed.ingest("radar", .2, .45, "late", cloud=self.cloud(4.45))
        later = next(item for item in delayed.items if item.key == "radar:later")
        self.assertTrue(later.diagnostics["accepted"])
        self.assertEqual(later.diagnostics["gated_count"], 1)
        self.assertEqual(delayed.late_events, 1)

        chronological = self.tracker(gate=2.)
        self.initialize(chronological)
        chronological.ingest("radar", .2, .2, "late", cloud=self.cloud(4.45))
        chronological.ingest("radar", .3, .3, "later", cloud=self.cloud(4.70))
        chronological.advance_to(.45)
        np.testing.assert_allclose(delayed.filter.x, chronological.filter.x, atol=1e-12)
        np.testing.assert_allclose(delayed.filter.P, chronological.filter.P, atol=1e-12)
        td, xd, pd = delayed.tick_history()
        tc, xc, pc = chronological.tick_history()
        np.testing.assert_allclose(td, tc, atol=1e-15)
        np.testing.assert_allclose(xd, xc, atol=1e-12)
        np.testing.assert_allclose(pd, pc, atol=1e-12)

    def test_100_hz_predict_only_matches_independent_recurrence(self):
        tracker = self.tracker(acceleration_std=.8)
        self.initialize(tracker)
        reference = Kalman3D(tracker.anchor_x, tracker.anchor_P, acceleration_std=.8)
        expected_x, expected_P = [], []
        for _ in range(25):
            reference.predict(.01)
            expected_x.append(reference.x.copy())
            expected_P.append(reference.P.copy())
        tracker.advance_to(.35)
        time, x, P = tracker.tick_history()
        self.assertEqual(len(time), 25)
        np.testing.assert_allclose(time, np.arange(11, 36) / 100, atol=1e-15)
        np.testing.assert_allclose(x, expected_x, atol=1e-12)
        np.testing.assert_allclose(P, expected_P, atol=1e-12)
        self.assertGreater(P[-1, 0, 0], tracker.anchor_P[0, 0])
        # One long prediction does not produce the same Q as 25 independent
        # per-tick acceleration draws; replay must retain every original tick.
        one_jump = Kalman3D(tracker.anchor_x, tracker.anchor_P, acceleration_std=.8)
        one_jump.predict(.25)
        self.assertGreater(np.max(np.abs(one_jump.P - reference.P)), 1e-3)

    def test_off_grid_sensor_splits_match_independent_tick_and_event_loop(self):
        tracker = self.tracker(gate=4., acceleration_std=.8)
        self.initialize(tracker)
        events = [(.205, "a", self.cloud(4.10, .1)),
                  (.307, "b", self.cloud(4.15, .1))]
        tracker.ingest("radar", events[1][0], .4, "b", cloud=events[1][2])
        tracker.ingest("radar", events[0][0], .5, "a", cloud=events[0][2])
        tracker.advance_to(.6)

        # This reference does not use FusionTracker's replay/history logic.
        init = RadarInitializer()
        init.observe(0., self.cloud())
        reference = init.observe(.1, self.cloud())
        associator = RadarAssociator(gate=4.)
        timeline = [(tick / 100, 1, None) for tick in range(11, 61)]
        timeline += [(timestamp, 0, cloud) for timestamp, _, cloud in events]
        previous = .1
        expected_ticks = []
        for timestamp, priority, cloud in sorted(timeline, key=lambda item: (item[0], item[1])):
            dt = timestamp - previous
            if dt > 1e-12:
                reference.predict(dt)
            if cloud is not None:
                associator.update(reference, cloud)
            else:
                expected_ticks.append((reference.x.copy(), reference.P.copy()))
            previous = timestamp
        _, actual_x, actual_P = tracker.tick_history()
        np.testing.assert_allclose(actual_x, [pair[0] for pair in expected_ticks], atol=1e-12)
        np.testing.assert_allclose(actual_P, [pair[1] for pair in expected_ticks], atol=1e-12)

    def test_duplicate_sensor_timestamp_rejected_before_state_mutation(self):
        tracker = self.tracker()
        self.initialize(tracker)
        before_x, before_P = tracker.filter.x.copy(), tracker.filter.P.copy()
        before_events = list(tracker.events)
        before_items = list(tracker.items)
        before_seen = tracker.seen.copy()
        before_wall = tracker.wall_time
        with self.assertRaisesRegex(ValueError, "per sensor timestamp"):
            tracker.ingest("radar", .1, .2, "duplicate_timestamp", cloud=self.cloud())
        np.testing.assert_array_equal(tracker.filter.x, before_x)
        np.testing.assert_array_equal(tracker.filter.P, before_P)
        self.assertEqual(tracker.events, before_events)
        self.assertEqual(tracker.items, before_items)
        self.assertEqual(tracker.seen, before_seen)
        self.assertEqual(tracker.wall_time, before_wall)


if __name__ == "__main__":
    unittest.main()
