"""Verify reported rates, missing-data handling and comparable scene coverage."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metric_charts import compare_versions, load_run_metrics, write_metrics_csv

TEMP_ROOT = Path(__file__).resolve().parents[2] / ".cache"
TEMP_ROOT.mkdir(exist_ok=True)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


class MetricChartsTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir(exist_ok=True)
        record = {"targets": 2, "total_beams": 6, "dt": .05,
                  "maximum_evaluated_duration_s": 1., "scene_membership": [[1, 2], [3, 4]],
                  "results": {"joint": {"success_rate": .5, "recovery_rate": None,
                                        "mean_recovery_delay_s": None}}}
        (root / "summary.json").write_text(json.dumps(record), encoding="utf-8")
        write_csv(root / "joint_scenes.csv", [
            {"evaluated_blocks": 10, "missed_target_blocks": 2},
            {"evaluated_blocks": 5, "missed_target_blocks": 3},
        ])
        write_csv(root / "joint_targets.csv", [{"position_count": 25}])
        return record

    def test_miss_denominator_uses_observed_target_frames(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory); self.fixture(root)
            _, values = load_run_metrics(root)
            self.assertAlmostEqual(values["joint"]["missed_target_block_rate"], 5/30)
            self.assertEqual(values["joint"]["evaluated_target_blocks"], 30)
            self.assertEqual(values["joint"]["detected_target_blocks"], 25)
            self.assertEqual(values["joint"]["potential_target_blocks"], 80)
            self.assertAlmostEqual(values["joint"]["position_evaluation_coverage"], 25/80)
            self.assertIsNone(values["joint"]["mean_unique_beams"])

    def test_resource_use_distinguishes_slots_from_directions(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory); self.fixture(root)
            write_csv(root / "beam_scans.csv", [
                {"policy": "joint", "unique_beams": 6, "beam_slots_used": 6},
                {"policy": "joint", "unique_beams": 4, "beam_slots_used": 6},
                {"policy": "decoupled", "unique_beams": 1, "beam_slots_used": 2},
            ])
            _, values = load_run_metrics(root)
            self.assertEqual(values["joint"]["mean_unique_beams"], 5)
            self.assertEqual(values["joint"]["mean_beam_slots_used"], 6)

    def test_export_scales_percent_and_preserves_unavailable_values(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory); self.fixture(root)
            output = write_metrics_csv(root / "metrics.csv", {"V1": root})
            with output.open(encoding="utf-8-sig") as stream:
                rows = {row["metric"]: row for row in csv.DictReader(stream)}
            self.assertEqual(float(rows["success_rate"]["value"]), 50)
            self.assertEqual(rows["recovery_rate"]["value"], "")
            self.assertEqual(rows["mean_unique_beams"]["value"], "")

    def test_different_scene_membership_rejects_comparison(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            root = Path(directory)
            first = root / "a"; second = root / "b"
            self.fixture(first); record = self.fixture(second)
            record["scene_membership"] = [[4, 3], [2, 1]]
            (second / "summary.json").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scene_membership"):
                compare_versions({"V1": first, "V2": second}, root / "figures")


if __name__ == "__main__":
    unittest.main()
