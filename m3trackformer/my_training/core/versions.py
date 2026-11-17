"""Append-only experiment snapshots: source, configuration, metrics and artifact hashes."""
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .common import BASE, read_json, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_version(name, run, description, exclude_modules=()):
    """Never overwrite a completed version. Large data/weights are referenced by hash."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('版本名必须是单个安全目录名')
    destination = BASE/'experiments'/name
    if any((destination/item).exists() for item in ('manifest.json', 'source.zip', 'results')):
        raise FileExistsError(f'旧版本已锁定，请使用新版本名：{destination}')
    run = Path(run)
    summary = read_json(run/'summary.json')
    references = {}
    for key in ('source', 'nnet_checkpoint', 'rnet_checkpoint'):
        path = Path(summary[key])
        references[key] = {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}
    destination.mkdir(parents=True, exist_ok=True)
    result_dir = destination/'results'
    if result_dir.exists():
        raise FileExistsError(result_dir)
    shutil.copytree(run, result_dir)
    sources = sorted(p for p in BASE.rglob('*.py')
                     if not any(part in {'outputs', 'experiments', '__pycache__', 'datasets'}
                                for part in p.relative_to(BASE).parts)
                     and p.name not in exclude_modules)
    sources += [p for p in (BASE/'README.txt', BASE/'AGENTS.md', BASE/'JOURNAL_ALIGNMENT.txt',
                            BASE.parent/'requirements.txt') if p.exists()]
    hashes = {}
    with zipfile.ZipFile(destination/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            relative = path.relative_to(BASE.parent).as_posix()
            content = path.read_bytes()
            archive.writestr(relative, content)
            hashes[relative] = hashlib.sha256(content).hexdigest()
    record = {'version': name, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'description': description, 'original_run': str(run.resolve()),
              'source_sha256': hashes, 'external_assets': references,
              'source_archive_sha256': sha256(destination/'source.zip'),
              'result_sha256': {p.relative_to(result_dir).as_posix(): sha256(p)
                                for p in result_dir.rglob('*') if p.is_file()},
              'large_data_and_weights_in_archive': False}
    write_json(destination/'manifest.json', record)
    return destination
