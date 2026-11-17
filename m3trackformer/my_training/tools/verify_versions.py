r"""Read-only validation of preserved experiment versions and external assets.

Run from the project root:
  .venv\Scripts\python.exe -X utf8 my_training\tools\verify_versions.py
No files are extracted, written, replaced or deleted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and '..' not in path.parts
                and '\\' not in name and ':' not in name)


def verify_version(directory, verify_assets=True, require_figures=True):
    """Return a JSON-compatible audit. Old code is compared to its ZIP, not live code."""
    directory = Path(directory).resolve()
    errors, warnings = [], []
    audit = {'version_directory': str(directory), 'ok': False, 'errors': errors,
             'warnings': warnings, 'source_files_checked': 0, 'result_files_checked': 0,
             'external_assets_checked': 0, 'figures_checked': []}
    manifest_path = directory/'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        errors.append(f'Cannot read manifest: {exc}')
        return audit
    audit['version'] = manifest.get('version')
    source_hashes = manifest.get('source_sha256', {})
    results_hashes = manifest.get('result_sha256', {})
    if not source_hashes or not results_hashes:
        errors.append('Manifest must contain nonempty source and result hash maps')
    archive_path = directory/'source.zip'
    try:
        if file_sha256(archive_path) != manifest.get('source_archive_sha256'):
            errors.append('Source archive SHA256 differs from manifest')
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                errors.append('Source archive contains duplicate entries')
            if set(names) != set(source_hashes):
                errors.append('Source archive entry list differs from manifest')
            bad_crc = archive.testzip()
            if bad_crc:
                errors.append(f'Source archive CRC check failed: {bad_crc}')
            for name in names:
                if not _safe_relative(name):
                    errors.append(f'Unsafe source archive path: {name}')
                    continue
                parts = PurePosixPath(name).parts
                if (any(part in {'datasets', 'outputs', '__pycache__'} for part in parts)
                        or PurePosixPath(name).suffix.lower() in {'.pth', '.pt', '.npz', '.mat'}):
                    errors.append(f'Unexpected data/weight artifact in source archive: {name}')
                if hashlib.sha256(archive.read(name)).hexdigest() != source_hashes.get(name):
                    errors.append(f'Archived source SHA256 differs: {name}')
                audit['source_files_checked'] += 1
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        errors.append(f'Cannot validate source archive: {exc}')
    results = directory/'results'
    actual_results = {p.relative_to(results).as_posix() for p in results.rglob('*') if p.is_file()}
    if actual_results != set(results_hashes):
        errors.append('Saved result file list differs from manifest')
    original = Path(manifest.get('original_run', ''))
    original_available = bool(manifest.get('original_run')) and original.is_dir()
    if not original_available:
        warnings.append('Original run directory unavailable; preserved results still checked against manifest')
    for name, expected in results_hashes.items():
        if not _safe_relative(name):
            errors.append(f'Unsafe result path in manifest: {name}')
            continue
        path = results/name
        if not path.is_file():
            errors.append(f'Saved result is missing: {name}')
            continue
        if file_sha256(path) != expected:
            errors.append(f'Saved result SHA256 differs: {name}')
        if original_available:
            original_file = original/name
            if not original_file.is_file() or file_sha256(original_file) != expected:
                errors.append(f'Preserved result does not match original run: {name}')
        audit['result_files_checked'] += 1
    if manifest.get('large_data_and_weights_in_archive') is not False:
        errors.append('Manifest must explicitly state that large assets are external')
    if verify_assets:
        for key, record in manifest.get('external_assets', {}).items():
            path = Path(record['path'])
            if not path.is_file():
                errors.append(f'External asset missing: {key}: {path}')
                continue
            if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
                errors.append(f'External asset size/SHA256 differs: {key}')
            audit['external_assets_checked'] += 1
    else:
        warnings.append('External asset verification explicitly skipped')
    expected_figures = ['tracking_status.png', 'tracking_survival.png'] if require_figures else []
    for name in expected_figures:
        if not (results/name).is_file():
            errors.append(f'Required tracking figure missing: {name}')
    figure_paths = list(results.rglob('*.png')) + list((directory/'figures').rglob('*.png'))
    for path in sorted(figure_paths):
        with path.open('rb') as handle:
            if handle.read(8) != b'\x89PNG\r\n\x1a\n' or path.stat().st_size < 32:
                errors.append(f'Figure is empty or not a PNG: {path.relative_to(directory)}')
        audit['figures_checked'].append(path.relative_to(directory).as_posix())
    figure_manifest = directory/'figures_manifest.json'
    if figure_manifest.exists():
        try:
            record = json.loads(figure_manifest.read_text(encoding='utf-8'))
            if record.get('original_manifest_sha256') != file_sha256(manifest_path):
                errors.append('Original manifest SHA256 differs from figure-lock reference')
            artifact_hashes = record['artifacts']
            current_artifacts = {p.relative_to(directory).as_posix()
                                 for p in (directory/'figures').rglob('*') if p.is_file()}
            if not artifact_hashes or current_artifacts != set(artifact_hashes):
                errors.append('Figure/CSV artifact list differs from figure manifest')
            for name, asset in artifact_hashes.items():
                if not _safe_relative(name) or not name.startswith('figures/'):
                    errors.append(f'Unsafe figure artifact path: {name}')
                    continue
                path = directory/name
                if not path.is_file():
                    errors.append(f'Locked figure/CSV artifact missing: {name}')
                elif path.stat().st_size != asset['bytes'] or file_sha256(path) != asset['sha256']:
                    errors.append(f'Locked figure/CSV artifact size/SHA256 differs: {name}')
            audit['figure_artifacts_checked'] = len(artifact_hashes)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'Cannot validate figure manifest: {exc}')
    elif (directory/'figures').exists():
        warnings.append('Additional figures outside results are checked as PNGs; their hashes are not in the original manifest')
    audit['ok'] = not errors
    return audit


def lock_figures(version_directories):
    """Append figure locks once; never edit completed original manifests or locks.

    Lock all final files in figures/, including CSV tables. This is a separate
    explicit write operation; verify_version and the default CLI remain read-only.
    A supplied list is preflighted before any locks are created.
    """
    if isinstance(version_directories, (str, Path)):
        version_directories = [version_directories]
    plans = []
    for supplied in version_directories:
        directory = Path(supplied).resolve()
        target = directory/'figures_manifest.json'
        if target.exists():
            raise FileExistsError(f'Existing figure lock will not be overwritten: {target}')
        audit = verify_version(directory)
        if not audit['ok']:
            raise ValueError(f'Cannot lock invalid snapshot {directory}: {audit["errors"]}')
        files = sorted(p for p in (directory/'figures').rglob('*') if p.is_file())
        if not files:
            raise ValueError(f'No figure artifacts to lock: {directory}')
        if any(not p.stat().st_size for p in files):
            raise ValueError(f'Empty figure/CSV artifact: {directory}')
        record = {'version': audit['version'], 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'scope': 'all files under figures/; original source/results manifest remains unchanged',
                  'original_manifest_sha256': file_sha256(directory/'manifest.json'),
                  'artifacts': {p.relative_to(directory).as_posix():
                                {'bytes': p.stat().st_size, 'sha256': file_sha256(p)} for p in files}}
        plans.append((target, record))
    for target, record in plans:
        with target.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    return [target for target, _ in plans]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('versions', nargs='*', type=Path)
    parser.add_argument('--skip-external-assets', action='store_true',
                        help='For exported source/results without local data and model files')
    args = parser.parse_args(argv)
    directories = args.versions or sorted((Path(__file__).resolve().parents[1]/'experiments').glob('*/manifest.json'))
    directories = [p.parent if p.name == 'manifest.json' else p for p in directories]
    if not directories:
        parser.error('No completed versions found')
    audits = [verify_version(p, not args.skip_external_assets) for p in directories]
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(json.dumps({'ok': all(a['ok'] for a in audits), 'versions': audits}, ensure_ascii=False, indent=2))
    return 0 if all(a['ok'] for a in audits) else 1


if __name__ == '__main__':
    raise SystemExit(main())
