"""Integration checks of explicit data contracts and no-truth replay."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose
from run_reproduction import example_config,parse_config,write_fixture,run_dataset


class ReproductionTests(unittest.TestCase):
    def test_velocity_units_cannot_be_silently_reinterpreted(self):
        config=example_config();config["radar_velocity"]="radial_doppler"
        with self.assertRaises(ValueError):parse_config(config)

    def test_invalid_configuration_fails_before_dividing_prediction_period(self):
        for section in ("camera","fusion","evaluation"):
            config=example_config();config[section]=[]
            with self.assertRaises(ValueError):parse_config(config)
        config=example_config();config["fusion"]["prediction_hz"]=0
        with self.assertRaises(ValueError):parse_config(config)
        config=example_config();config["camera"].pop("rotation")
        with self.assertRaises(ValueError):parse_config(config)

    def test_csv_replay_and_missing_truth_do_not_change_estimates(self):
        root=Path(__file__).resolve().parents[1]/"tmp";root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            dataset=write_fixture(directory,42,.8)
            reports,tracks=run_dataset(dataset,"fixture",["bearing"])
            dataset["ground_truth"]=None
            no_truth_reports,no_truth_tracks=run_dataset(dataset,"fixture",["bearing"])
            for key in tracks:assert_allclose(tracks[key][1],no_truth_tracks[key][1],atol=0,rtol=0)
            self.assertTrue(all(r["metrics"]["status"]=="no_ground_truth" for r in no_truth_reports))
            self.assertTrue(all(r["diagnostics"]["initialized"] for r in reports))
            assert_allclose(tracks[("bearing","average")][1],tracks[("bearing","kmeans")][1])

    def test_failed_initialization_reports_no_estimates(self):
        dataset=dict(config=example_config(),radar=[dict(timestamp=0.,arrival_time=0.,event_id="empty",cloud=np.empty((0,5)))],camera=[],ground_truth=None)
        reports,tracks=run_dataset(dataset,"empty",["paper_xyz"])
        self.assertTrue(all(not r["diagnostics"]["initialized"] for r in reports))
        self.assertTrue(all(r["metrics"]["estimate_count"]==0 for r in reports))

    def test_radar_only_data_does_not_require_unused_camera_calibration(self):
        config=example_config();config.pop("camera")
        dataset=dict(config=config,radar=[dict(timestamp=k*.1,arrival_time=k*.1,event_id=str(k),cloud=np.array([[4,0,0,0,1.]])) for k in range(3)],camera=[],ground_truth=None)
        reports,_=run_dataset(dataset,"radar",["paper_xyz"])
        self.assertTrue(all(r["diagnostics"]["initialized"] for r in reports))


if __name__=="__main__":unittest.main()
