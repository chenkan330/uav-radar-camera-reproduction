"""Input-contract checks, independent of the replay/filter implementation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from dataset_io import (
    CAMERA_COLUMNS, RADAR_COLUMNS, RADAR_FRAME_COLUMNS, TRUTH_COLUMNS, read_dataset,
)


class DatasetIOTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.directory = Path(self.temporary.name)
        self.config = {"coordinate_frame": "global", "source": "unit test", "clock": "relative_s"}
        (self.directory / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.write_csv("radar.csv", RADAR_COLUMNS, [])

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, columns, rows):
        with (self.directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_group_frames_preserve_order_and_raw_values(self):
        self.write_csv("radar.csv", RADAR_COLUMNS, [
            [1, 1.08, "f1", 2, -1, 3, -0.2, 4],
            [0, 0.03, "f0", 5, 6, 7, 0.1, 2],
            [1, 1.08, "f1", 8, 9, 10, 0.3, 0],
        ])
        self.write_csv("camera.csv", CAMERA_COLUMNS, [
            [1, 1.3, "d0", 10, 20, 30, 40, 0.6],
        ])
        dataset = read_dataset(self.directory)
        self.assertEqual(dataset["config"], self.config)
        self.assertEqual([event["event_id"] for event in dataset["radar"]], ["radar:f0", "radar:f1"])
        assert_allclose(dataset["radar"][1]["cloud"], [[2, -1, 3, -0.2, 4], [8, 9, 10, 0.3, 0]])
        self.assertEqual(dataset["radar"][1]["arrival_time"], 1.08)
        assert_allclose(dataset["camera"][0]["bbox"], [10, 20, 30, 40])
        # Keep low confidence raw input; selection belongs to the camera model.
        self.assertEqual(dataset["camera"][0]["confidence"], 0.6)
        self.assertIsNone(dataset["ground_truth"])

    def test_empty_frame_and_empty_optional_streams(self):
        self.write_csv("radar_frames.csv", RADAR_FRAME_COLUMNS, [[0, 0.1, "empty"]])
        dataset = read_dataset(self.directory)
        self.assertEqual(dataset["radar"][0]["cloud"].shape, (0, 5))
        self.assertEqual(dataset["camera"], [])
        self.write_csv("ground_truth.csv", TRUTH_COLUMNS, [])
        self.assertEqual(read_dataset(self.directory)["ground_truth"].shape, (0, 4))

    def test_frame_table_can_register_nonempty_frames(self):
        self.write_csv("radar.csv", RADAR_COLUMNS, [[0, 0.1, "f", 1, 2, 3, 4, 5]])
        self.write_csv("radar_frames.csv", RADAR_FRAME_COLUMNS, [[0, 0.1, "f"]])
        self.assertEqual(len(read_dataset(self.directory)["radar"]), 1)

    def test_frame_timestamp_and_arrival_must_be_consistent(self):
        for timestamp, arrival in [(0.1, 0.2), (0, 0.2)]:
            with self.subTest(timestamp=timestamp, arrival=arrival):
                self.write_csv("radar.csv", RADAR_COLUMNS, [
                    [0, 0.1, "f", 1, 2, 3, 4, 5],
                    [timestamp, arrival, "f", 1, 2, 3, 4, 5],
                ])
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    read_dataset(self.directory)

    def test_duplicate_frame_table_and_camera_ids_rejected(self):
        self.write_csv("radar_frames.csv", RADAR_FRAME_COLUMNS, [[0, 0, "f"], [0, 0, "f"]])
        with self.assertRaisesRegex(ValueError, "duplicate frame_id"):
            read_dataset(self.directory)
        (self.directory / "radar_frames.csv").unlink()
        self.write_csv("camera.csv", CAMERA_COLUMNS, [
            [0, 0, "d", 0, 0, 10, 10, 1], [1, 1, "d", 0, 0, 10, 10, 1],
        ])
        with self.assertRaisesRegex(ValueError, "duplicate detection_id"):
            read_dataset(self.directory)

    def test_finite_values_linear_intensity_and_clock_validation(self):
        bad_rows = [
            [0, 0, "f", "nan", 2, 3, 4, 5],
            [0, 0, "f", 1, 2, 3, "inf", 5],
            [0, 0, "f", 1, 2, 3, 4, -1],
            [1, 0, "f", 1, 2, 3, 4, 5],
            [-1, 0, "f", 1, 2, 3, 4, 5],
            [0, 0, " ", 1, 2, 3, 4, 5],
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                self.write_csv("radar.csv", RADAR_COLUMNS, [row])
                with self.assertRaises(ValueError):
                    read_dataset(self.directory)

    def test_radial_velocity_column_is_not_silently_accepted(self):
        columns = tuple("radial_velocity_mps" if item == "vx_mps" else item for item in RADAR_COLUMNS)
        self.write_csv("radar.csv", columns, [])
        with self.assertRaisesRegex(ValueError, "vx_mps"):
            read_dataset(self.directory)

    def test_distinct_ids_cannot_double_count_same_sensor_timestamp(self):
        self.write_csv("radar.csv", RADAR_COLUMNS, [[0, 0.1, "f0", 1, 2, 3, 4, 5]])
        # A separate empty-frame ID at the same acquisition time would also
        # represent a duplicate sensor event, rather than missing information.
        self.write_csv("radar_frames.csv", RADAR_FRAME_COLUMNS, [[0, 0.2, "f1"]])
        with self.assertRaisesRegex(ValueError, "one radar frame per timestamp"):
            read_dataset(self.directory)
        (self.directory / "radar_frames.csv").unlink()
        self.write_csv("camera.csv", CAMERA_COLUMNS, [
            [0, 0.1, "d0", 0, 0, 5, 10, 0.9],
            [0, 0.2, "d1", 10, 10, 15, 20, 0.95],
        ])
        with self.assertRaisesRegex(ValueError, "one selected target detection"):
            read_dataset(self.directory)

    def test_radar_and_camera_can_share_a_timestamp(self):
        self.write_csv("radar.csv", RADAR_COLUMNS, [[0, 0.1, "f0", 1, 2, 3, 4, 5]])
        self.write_csv("camera.csv", CAMERA_COLUMNS, [[0, 0.2, "d0", 0, 0, 5, 10, 0.9]])
        dataset = read_dataset(self.directory)
        self.assertEqual(len(dataset["radar"]), 1)
        self.assertEqual(len(dataset["camera"]), 1)

    def test_invalid_bbox_confidence_and_camera_clock(self):
        for row in [
            [0, 0, "d", 10, 0, 5, 10, 1],
            [0, 0, "d", 0, 0, 5, 10, 1.01],
            [0, 0, "d", 0, 0, "nan", 10, 1],
            [1, 0.9, "d", 0, 0, 5, 10, 1],
        ]:
            with self.subTest(row=row):
                self.write_csv("camera.csv", CAMERA_COLUMNS, [row])
                with self.assertRaises(ValueError):
                    read_dataset(self.directory)

    def test_truth_sorted_and_duplicate_truth_time_rejected(self):
        self.write_csv("ground_truth.csv", TRUTH_COLUMNS, [[1, 1, 2, 3], [0, 4, 5, 6]])
        assert_allclose(read_dataset(self.directory)["ground_truth"], [[0, 4, 5, 6], [1, 1, 2, 3]])
        self.write_csv("ground_truth.csv", TRUTH_COLUMNS, [[0, 1, 2, 3], [0, 4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "unique"):
            read_dataset(self.directory)

    def test_malformed_rows_headers_and_config_rejected(self):
        for columns, rows in [
            (RADAR_COLUMNS, [[0, 0, "f", 1]]),
            (RADAR_COLUMNS + ("intensity",), []),
        ]:
            with self.subTest(columns=columns):
                self.write_csv("radar.csv", columns, rows)
                with self.assertRaises(ValueError):
                    read_dataset(self.directory)
        self.write_csv("radar.csv", RADAR_COLUMNS, [])
        (self.directory / "config.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "object"):
            read_dataset(self.directory)


if __name__ == "__main__":
    unittest.main()
