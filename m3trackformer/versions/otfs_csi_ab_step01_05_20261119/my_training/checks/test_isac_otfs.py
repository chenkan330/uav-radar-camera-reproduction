"""Verify the DD pilot and the OTFS receiver using synthetic point targets."""
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.isac_multi_target import RADAR_CONFIG, STATION
from core.isac_otfs import OTFSRadar, make_otfs_probe


class TestOTFS(unittest.TestCase):
    def test_delay_doppler_pilot_round_trip(self):
        config = RADAR_CONFIG
        dd, tx, _, _ = make_otfs_probe(config)
        useful = tx[:, config.cp_samples:]
        tf = np.fft.fft(useful, axis=1, norm="ortho")
        recovered = np.fft.fft(np.fft.ifft(tf, axis=1, norm="ortho"), axis=0, norm="ortho")
        self.assertEqual(np.count_nonzero(dd), 1)
        self.assertAlmostEqual(np.mean(abs(useful)**2), 1.)
        np.testing.assert_allclose(recovered, dd, atol=1e-12)

    def test_two_targets_and_single_return_dropout(self):
        beam_u, beam_v = 19, 17
        u, v = -1 + 2*(np.array([beam_u, beam_v])+.5)/32
        direction = np.array([u, v, np.sqrt(1-u*u-v*v)])
        distance = np.array([152.6, 244.3])
        radial_velocity = np.array([18., -11.])
        t = np.arange(3)*RADAR_CONFIG.decision_interval_s
        position = (distance[:, None] + radial_velocity[:, None]*t)[:, :, None]*direction
        radar = OTFSRadar((position + STATION)[None, ...], RADAR_CONFIG, dropout=(0, 0, 1))
        beam = beam_u + 32*beam_v
        hits = radar.returns(0, 0, 0, beam, [0, 1])
        for expected_range, expected_speed in zip(distance, radial_velocity):
            hit = min(hits, key=lambda row: abs(row[2][0]-expected_range))
            self.assertAlmostEqual(hit[2][0], expected_range, delta=.5)
            self.assertAlmostEqual(hit[4], expected_speed, delta=.5)
            np.testing.assert_allclose(hit[3], direction[:2], atol=.01)

        hits = radar.returns(0, 1, 0, beam, [0, 1])
        self.assertTrue(any(abs(hit[2][0]-(distance[1] + radial_velocity[1]*t[1])) < .5 for hit in hits))
        self.assertFalse(any(abs(hit[2][0]-(distance[0] + radial_velocity[0]*t[1])) < 5. for hit in hits))

    def test_parallel_beam_scans_preserve_packet_order_and_noise(self):
        relative_positions = np.array([[12., 18., 129.], [-15., 18., 138.]])
        positions = np.broadcast_to(relative_positions[:, None, None, :] + STATION, (2, 1, 2, 3))
        serial = OTFSRadar(positions, RADAR_CONFIG, dropout=None)
        groups = [1, 0]
        selected = [np.argsort(-abs(serial.csi(relative_positions[group]))**2)[:3] for group in groups]
        reference = serial.scan(1, groups, selected)
        with ThreadPoolExecutor(2) as executor:
            parallel = OTFSRadar(positions, RADAR_CONFIG, dropout=None, scan_executor=executor)
            actual = parallel.scan(1, groups, selected)

        for group, expected, packet in zip(groups, reference, actual):
            self.assertGreater(len(expected.measurements), 0)
            np.testing.assert_array_equal(packet.beams, expected.beams)
            np.testing.assert_array_equal(packet.measurements, expected.measurements)
            np.testing.assert_array_equal(packet.occupied_beams, expected.occupied_beams)
            np.testing.assert_array_equal(parallel.receipts[group, 1][1], serial.receipts[group, 1][1])

    def test_iq_observer_records_noisy_receiver_input_even_on_dropout(self):
        positions = np.broadcast_to(STATION + [12., 18., 129.], (1, 1, 2, 3))
        radar = OTFSRadar(positions, RADAR_CONFIG, dropout=(0, 0, 1))
        with patch.object(radar, "delay_samples", wraps=radar.delay_samples) as receiver:
            captures = []
            radar.received_iq_observer = lambda *observation: captures.append(observation)
            radar.returns(0, 1, 0, 600, [0])
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0][:4], (0, 1, 0, 600))
        captured = captures[0][4]
        self.assertFalse(captured.flags.writeable)
        self.assertGreater(float(np.linalg.norm(captured)), 0.)
        np.testing.assert_array_equal(captured, receiver.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
