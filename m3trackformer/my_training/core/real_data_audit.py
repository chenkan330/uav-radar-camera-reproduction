"""Verify real GNSS references and retain each evaluated slice's provenance."""
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np

from .common import load_split, read_json, write_json, write_rows


def audit_real_reference(data_root, split, output):
    root, output = Path(data_root), Path(output)
    config = read_json(root/'config.json')
    if 'real_GNSS' not in config.get('dataset_kind', ''):
        raise ValueError('本入口只接受真实GNSS参考轨迹')
    parent = Path(config['source_data'])
    original_config = read_json(parent/'config.json')
    data, original = load_split(root/f'{split}.npz'), load_split(parent/f'{split}.npz')
    old_times = np.arange(original['position'].shape[1])*original_config['dt']
    new_times = np.arange(data['position'].shape[1])*config['dt']
    expected = np.asarray([[np.interp(new_times, old_times, row[:, a]) for a in range(3)]
                           for row in original['position']]).transpose(0, 2, 1).astype('float32')
    if not np.array_equal(expected, data['position']):
        raise ValueError('参考位置与原始真实切片的插值结果不一致')
    for key in ('trajectory_id', 'source_lap', 'session_group'):
        if not np.array_equal(data[key], original[key]):
            raise ValueError(f'真实切片来源字段不一致：{key}')
    with (parent/'segments.csv').open(encoding='utf-8-sig', newline='') as stream:
        lookup = {int(r['id']): r for r in csv.DictReader(stream)}
    rows = []
    for trajectory_id, lap, group in zip(data['trajectory_id'], data['source_lap'], data['session_group']):
        row = dict(lookup[int(trajectory_id)])
        if int(row['lap']) != int(lap) or int(row['group']) != int(group) or row['split'] != split:
            raise ValueError('真实切片清单与数据分区不一致')
        row['source_csv_in_archive'] = f'Speedybee DataSet/processed/GPS/Lap{int(lap):03d}_gps.csv'
        rows.append(row)
    # Read the original CSVs again, rather than trusting only the cached NPZ label.
    archive = Path(original_config['archive'])
    with archive.open('rb') as stream:
        archive_md5 = hashlib.file_digest(stream, 'md5').hexdigest()
    if archive_md5 != original_config['archive_md5']:
        raise ValueError('真实飞行压缩包校验值与切片来源不一致')
    from tools.prepare_real_flights import to_enu
    references = {int(r['group']): r['origin_lla'] for r in original_config['session_references']}
    cached = {}
    with zipfile.ZipFile(archive) as zipped:
        for i, row in enumerate(rows):
            name, group = row['source_csv_in_archive'], int(row['group'])
            if (name, group) not in cached:
                with zipped.open(name) as stream:
                    gps = list(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')))
                values = np.asarray([[float(r[k]) for k in ('time (us)', 'GPS_coord[0]',
                    'GPS_coord[1]', 'GPS_altitude')] for r in gps])
                cached[name, group] = values[:, 0]/1e6, to_enu(values[:, 1:4], references[group])
            times, position = cached[name, group]
            target_times = float(row['start_s'])+old_times
            if target_times[0] < times[0] or target_times[-1] > times[-1]+1e-8:
                raise ValueError('真实切片时间超出原始GPS记录')
            restored = np.column_stack([np.interp(target_times, times, position[:, a])
                                        for a in range(3)]).astype('float32')
            if not np.allclose(restored, original['position'][i], rtol=0, atol=1e-4):
                raise ValueError('原始GPS CSV无法重建参考轨迹')
    result = {'real_GNSS_reference_verified': True, 'positions_equal_parent_reference_interpolation': True,
        'positions_reconstructed_from_original_GPS_CSV': True,
        'raw_archive': original_config['archive'], 'raw_archive_md5': archive_md5,
        'source_segments_manifest': str((parent/'segments.csv').resolve()),
        'split': split, 'segments': len(rows), 'source_flights': len(set(map(int, data['source_lap']))),
        'duration_s': config['duration'], 'input_dt_s': config['dt'],
        'original_GNSS_approximately_hz': 5,
        'interpolation_adds_no_real_motion_detail': True,
        'GNSS_is_reference_not_independent_high_precision_truth': True,
        'test_is_new_blind_set': not config.get('test_records_used_previously', False)}
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/'real_trajectory_sources.csv', rows)
    write_json(output/'real_trajectory_audit.json', result)
    return result
