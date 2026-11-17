"""Experiment versions preserve exact source/results and reject destructive reuse."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import versions
from tools.verify_versions import lock_figures, verify_version


class VersionTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[2]/'.cache'
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache, prefix='versions_check_')
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(cache.resolve()))
        self.base = self.root/'my_training'
        self.base.mkdir()
        self.mock_base = patch.object(versions, 'BASE', self.base)
        self.mock_base.start()
        self.run = self.base/'outputs'/'original_run'
        self.run.mkdir(parents=True)
        self.assets = {}
        for key, relative in [('source', 'datasets/fixture.npz'),
                              ('nnet_checkpoint', 'outputs/nnet.pth'),
                              ('rnet_checkpoint', 'outputs/rnet.pth')]:
            path = self.base/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((key+' fixture').encode())
            self.assets[key] = str(path)
        (self.run/'summary.json').write_text(json.dumps(self.assets), encoding='utf-8')
        (self.run/'metrics.csv').write_text('metric,value\nsuccess,0.5\n', encoding='utf-8')
        for name in ('tracking_status.png', 'tracking_survival.png'):
            (self.run/name).write_bytes(b'\x89PNG\r\n\x1a\n'+b'x'*64)
        (self.base/'entry.py').write_text('print("first version")\n', encoding='utf-8')
        (self.base/'skip_me.py').write_text('temporary workflow', encoding='utf-8')
        (self.root/'requirements.txt').write_text('numpy\n', encoding='utf-8')
        for part in ('outputs', 'datasets', '__pycache__', 'experiments'):
            folder = self.base/part
            folder.mkdir(exist_ok=True)
            (folder/'not_source.py').write_text('excluded', encoding='utf-8')

    def tearDown(self):
        self.mock_base.stop()
        self.temporary.cleanup()

    def snapshot(self, name='v1'):
        return versions.save_version(name, self.run, 'fixture', exclude_modules=('skip_me.py',))

    def test_v4_requires_its_forecast_and_tracking_figures(self):
        summary = dict(self.assets, version='v4_maneuver_uncertainty')
        (self.run/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        figures = self.run/'figures'
        figures.mkdir()
        names = ('forecast_scores.png', 'uncertainty_calibration.png',
                 'motion_type_comparison.png', 'maneuver_onset_uncertainty.png',
                 'future_motion_examples.png', 'maneuver_tracking_comparison.png',
                 'motion_network_training.png')
        for name in names[:-1]:
            (figures/name).write_bytes(b'\x89PNG\r\n\x1a\n'+b'x'*64)
        incomplete = self.snapshot('v4_incomplete')
        audit = verify_version(incomplete)
        self.assertFalse(audit['ok'])
        self.assertTrue(any(names[-1] in error for error in audit['errors']))
        (figures/names[-1]).write_bytes(b'\x89PNG\r\n\x1a\n'+b'x'*64)
        self.assertTrue(verify_version(self.snapshot('v4_complete'))['ok'])

    @staticmethod
    def digest_tree(path):
        return {p.relative_to(path).as_posix(): versions.sha256(p) for p in path.rglob('*') if p.is_file()}

    def test_archive_source_result_and_external_asset_hashes_match(self):
        saved = self.snapshot()
        manifest = json.loads((saved/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(versions.sha256(saved/'source.zip'), manifest['source_archive_sha256'])
        with zipfile.ZipFile(saved/'source.zip') as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(manifest['source_sha256']))
            for name, digest in manifest['source_sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
        for name, digest in manifest['result_sha256'].items():
            self.assertEqual(versions.sha256(saved/'results'/name), digest)
            self.assertEqual((saved/'results'/name).read_bytes(), (self.run/name).read_bytes())
        for key, asset in manifest['external_assets'].items():
            self.assertEqual(versions.sha256(self.assets[key]), asset['sha256'])
            self.assertEqual(Path(self.assets[key]).stat().st_size, asset['bytes'])
        self.assertTrue(verify_version(saved)['ok'])

    def test_large_assets_and_previous_results_never_enter_source_archive(self):
        saved = self.snapshot()
        with zipfile.ZipFile(saved/'source.zip') as archive:
            names = archive.namelist()
        self.assertIn('my_training/entry.py', names)
        self.assertIn('requirements.txt', names)
        self.assertFalse(any(name.endswith('skip_me.py') for name in names))
        self.assertFalse(any(part in name.split('/') for name in names
                             for part in ('outputs', 'datasets', '__pycache__', 'experiments')))
        self.assertFalse(any(name.endswith(('.pth', '.npz')) for name in names))

    def test_completed_version_cannot_be_overwritten_even_after_source_changes(self):
        saved = self.snapshot()
        before = self.digest_tree(saved)
        (self.base/'entry.py').write_text('print("changed")', encoding='utf-8')
        (self.run/'metrics.csv').write_text('changed result', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.snapshot()
        self.assertEqual(before, self.digest_tree(saved))

    def test_existing_archive_refuses_reuse_before_copying_any_result(self):
        destination = self.base/'experiments'/'v1'
        destination.mkdir()
        (destination/'source.zip').write_bytes(b'old partial source archive')
        before = self.digest_tree(destination)
        with self.assertRaises(FileExistsError):
            self.snapshot()
        self.assertEqual(before, self.digest_tree(destination))
        self.assertFalse((destination/'results').exists())

    def test_existing_results_refuse_reuse_and_are_preserved(self):
        result = self.base/'experiments'/'v1'/'results'
        result.mkdir(parents=True)
        (result/'metrics.csv').write_text('old', encoding='utf-8')
        before = self.digest_tree(result.parent)
        with self.assertRaises(FileExistsError):
            self.snapshot()
        self.assertEqual(before, self.digest_tree(result.parent))

    def test_new_version_is_append_only_and_keeps_previous_version_bytes(self):
        first = self.snapshot('v1')
        before = self.digest_tree(first)
        (self.base/'entry.py').write_text('print("new version")\n', encoding='utf-8')
        second = self.snapshot('v2')
        self.assertEqual(before, self.digest_tree(first))
        self.assertNotEqual(versions.sha256(first/'source.zip'), versions.sha256(second/'source.zip'))
        self.assertTrue(verify_version(first)['ok'])
        self.assertTrue(verify_version(second)['ok'])

    def test_missing_external_asset_fails_before_version_is_created(self):
        Path(self.assets['nnet_checkpoint']).unlink()
        with self.assertRaises(FileNotFoundError):
            self.snapshot()
        self.assertFalse((self.base/'experiments'/'v1').exists())

    def test_invalid_names_cannot_escape_or_alias_version_directory(self):
        for name in ('../escape', 'a/b', 'a\\b', '', '.', '..', str(self.root/'absolute')):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.snapshot(name)
        self.assertFalse((self.root/'escape').exists())

    def test_verifier_detects_result_source_archive_and_asset_tampering(self):
        saved = self.snapshot()
        before = self.digest_tree(saved)
        self.assertTrue(verify_version(saved)['ok'])
        self.assertEqual(before, self.digest_tree(saved), 'Verifier must never mutate the snapshot')
        (saved/'results'/'metrics.csv').write_text('tampered', encoding='utf-8')
        audit = verify_version(saved)
        self.assertFalse(audit['ok'])
        self.assertTrue(any('Saved result SHA256 differs' in e for e in audit['errors']))
        (saved/'results'/'metrics.csv').write_bytes((self.run/'metrics.csv').read_bytes())
        original_archive = (saved/'source.zip').read_bytes()
        (saved/'source.zip').write_bytes(b'broken zip')
        self.assertFalse(verify_version(saved)['ok'])
        (saved/'source.zip').write_bytes(original_archive)
        Path(self.assets['rnet_checkpoint']).write_bytes(b'tampered external weights')
        audit = verify_version(saved)
        self.assertFalse(audit['ok'])
        self.assertTrue(any('External asset size/SHA256 differs' in e for e in audit['errors']))

    def test_separate_figure_lock_is_write_once_detects_tampering_and_keeps_old_manifest(self):
        saved = self.snapshot()
        original_manifest = (saved/'manifest.json').read_bytes()
        figures = saved/'figures'
        figures.mkdir()
        (figures/'dashboard.png').write_bytes((self.run/'tracking_status.png').read_bytes())
        (figures/'metrics.csv').write_text('metric,value\nsuccess,0.5\n', encoding='utf-8')
        locked = lock_figures(saved)
        self.assertEqual(locked, [saved/'figures_manifest.json'])
        self.assertEqual((saved/'manifest.json').read_bytes(), original_manifest)
        self.assertTrue(verify_version(saved)['ok'])
        lock_bytes = locked[0].read_bytes()
        with self.assertRaises(FileExistsError):
            lock_figures(saved)
        self.assertEqual(locked[0].read_bytes(), lock_bytes)
        (figures/'metrics.csv').write_text('tampered', encoding='utf-8')
        audit = verify_version(saved)
        self.assertFalse(audit['ok'])
        self.assertTrue(any('Locked figure/CSV artifact size/SHA256 differs' in e for e in audit['errors']))
        self.assertEqual((saved/'manifest.json').read_bytes(), original_manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
