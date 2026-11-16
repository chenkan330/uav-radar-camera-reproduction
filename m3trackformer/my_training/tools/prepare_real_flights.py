"""将 Speedybee GPS 实测位置切成 50 秒片段；按起飞参考位置分组防止泄漏。"""
import argparse, csv, hashlib, io, json, re, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.common import BASE

BS = np.array([0., 0., 10.])
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def direction_beams(u, v):
    """保持原始50秒参考数据的中间标签；正式标签由主数据准备程序重算。"""
    ix = np.floor((u + 1) * 16 + .5).astype(np.int64) % 32
    iy = np.floor((v + 1) * 16 + .5).astype(np.int64) % 32
    return iy * 32 + ix + 1


def csv_rows(z, name):
    with z.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline=''))


def ecef(lat, lon, altitude):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    a, e2 = 6378137., 6.69437999014e-3
    n = a / np.sqrt(1-e2*np.sin(lat)**2)
    return np.stack(((n+altitude)*np.cos(lat)*np.cos(lon),
                     (n+altitude)*np.cos(lat)*np.sin(lon),
                     (n*(1-e2)+altitude)*np.sin(lat)), axis=-1)


def to_enu(lla, origin):
    lat, lon = np.deg2rad(origin[:2])
    rot = np.array([[-np.sin(lon), np.cos(lon), 0.],
                    [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                    [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return (ecef(lla[:, 0], lla[:, 1], lla[:, 2])-ecef(*origin))@rot.T


def build(archive, seed=20261112):
    # 发布方 MD5；只校验，不解压或修改原始文件。
    with archive.open('rb') as f:
        md5 = hashlib.file_digest(f, 'md5').hexdigest()
    if md5 != 'fa6210078d8f615118027a5310dfc99c':
        raise ValueError('压缩包 MD5 不匹配，请检查下载完整性')
    out = BASE/'outputs/step05d_real'/('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC'))
    out.mkdir(parents=True)
    laps, excluded, references = [], [], {}
    nonempty_count = 0
    with zipfile.ZipFile(archive) as z:
        names = sorted([n for n in z.namelist() if re.search(r'/GPS/Lap\d+_gps\.csv$', n, re.I)])
        for name in names:
            lap = int(re.search(r'Lap(\d+)_gps', name).group(1))
            rows = list(csv_rows(z, name))
            if not rows:
                excluded.append({'lap': lap, 'reason': 'empty_gps'}); continue
            nonempty_count += 1
            values = np.array([[float(r[k]) for k in ('time (us)', 'GPS_coord[0]', 'GPS_coord[1]', 'GPS_altitude', 'GPS_speed (m/s)', 'GPS_home_lat', 'GPS_home_lon')] for r in rows])
            if not np.isfinite(values).all() or (np.diff(values[:, 0]) <= 0).any():
                raise ValueError(f'Lap {lap} 数值或时间戳异常，请人工检查')
            if (values[:, 1:3] == 0).all(axis=1).any(): raise ValueError(f'Lap {lap} 无效坐标')
            homes = np.unique(values[:, 5:7], axis=0)
            if len(homes) != 1: raise ValueError(f'Lap {lap} 存在多个 home，需先核实来源')
            home = tuple(homes[0]); times = values[:, 0]/1e6
            if times[-1]-times[0] < 50:
                excluded.append({'lap': lap, 'reason': 'shorter_than_50s'}); continue
            # 同一 home 的首次 GPS 时刻，用此前飞控 NEU 高度对齐高度基准。
            # INAV 文档：navPos 为 NEU 厘米；只使用初始时刻以前的数据，不用未来位置做对齐。
            if home not in references:
                tele = name.replace('/GPS/', '/telemetry/').replace('_gps.csv', '.csv')
                last = None
                if tele in z.namelist():
                    for r in csv_rows(z, tele):
                        t = float(r['time (us)'])/1e6
                        if t > times[0]: break
                        last = r
                if last is None: raise ValueError(f'Lap {lap} 缺少初始高度参考')
                height_up = float(last['navPos[2]'])/100
                references[home] = {'origin_lla': [*home, values[0, 3]-height_up],
                                    'initial_lap': lap, 'initial_gps_time_s': float(times[0]),
                                    'reference_nav_time_s': float(last['time (us)'])/1e6}
            pos = to_enu(values[:, 1:4], references[home]['origin_lla'])
            laps.append({'lap': lap, 'home': home, 'time': times, 'position': pos,
                         'speed': values[:, 4], 'source': name})

    homes = sorted(references)
    groups = {home: index for index, home in enumerate(homes)}
    segments = []
    for data in laps:
        times = data['time']; cuts = np.r_[0, np.flatnonzero(np.diff(times)>1.)+1, len(times)]
        made = 0
        for start, stop in zip(cuts[:-1], cuts[1:]):
            t = times[start:stop]; pos = data['position'][start:stop]
            if not len(t): continue
            count = int(np.floor((t[-1]-t[0])/50.))
            for part in range(count):
                tt = t[0]+50*part+np.arange(501)*.1
                xyz = np.column_stack([np.interp(tt, t, pos[:, axis]) for axis in range(3)])
                segments.append({'position': xyz, 'group': groups[data['home']], 'lap': data['lap'],
                                 'start_s': float(tt[0]), 'end_s': float(tt[-1]),
                                 'speed': float(np.interp(tt, times, data['speed']).mean())})
                made += 1
        if not made: excluded.append({'lap': data['lap'], 'reason': 'no_contiguous_50s'})

    # 起飞参考坐标相同的所有 Lap 永远属于同一分组；先分组、后输出片段。
    eligible = np.unique([s['group'] for s in segments])
    perm = np.random.default_rng(seed).permutation(eligible)
    nv = max(1, round(len(perm)*.1)); nt = nv
    ntrain = len(perm)-nv-nt; nsim = max(1, round(ntrain*.2))
    split_groups = {'nnet_train': perm[:ntrain-nsim], 'rnet_sim': perm[ntrain-nsim:ntrain],
                    'validation': perm[ntrain:ntrain+nv], 'test': perm[ntrain+nv:]}
    assignment = {int(g): name for name, gs in split_groups.items() for g in gs}
    manifest = []
    for i, s in enumerate(segments):
        s['id'] = i; s['split'] = assignment[s['group']]
        manifest.append({k: s[k] for k in ('id','group','lap','start_s','end_s','speed','split')})
    for name in split_groups:
        selected = [s for s in segments if s['split']==name]
        if not selected: raise ValueError(f'{name} 没有足够轨迹')
        position = np.stack([s['position'] for s in selected])
        rel = position-BS; distance = np.linalg.norm(rel, axis=-1)
        az = np.arctan2(rel[:, :, 1], rel[:, :, 0]); polar = np.arccos(np.clip(rel[:, :, 2]/distance, -1, 1))
        measurement = np.stack((distance, az, polar), axis=-1)
        # GNSS位置作为参考轨迹；输入为另加独立角距误差的模拟雷达测量。
        for i, s in enumerate(selected):
            rng = np.random.default_rng(seed+100000+s['id'])
            measurement[i, :, 0] = np.maximum(distance[i]+rng.normal(0, 1., 501), 1e-6)
            measurement[i, :, 1] = (az[i]+rng.normal(0, np.deg2rad(.5), 501)+np.pi)%(2*np.pi)-np.pi
            measurement[i, :, 2] = np.clip(polar[i]+rng.normal(0, np.deg2rad(.5), 501), 0, np.pi)
        beam = direction_beams(rel[:, :, 0]/distance, rel[:, :, 1]/distance)
        np.savez_compressed(out/f'{name}.npz', position=position.astype('float32'),
                            measurement=measurement.astype('float32'), beam=beam.astype('int64'),
                            speed=np.array([s['speed'] for s in selected], dtype='float32'),
                            trajectory_id=np.array([s['id'] for s in selected], dtype='int64'),
                            session_group=np.array([s['group'] for s in selected], dtype='int64'),
                            source_lap=np.array([s['lap'] for s in selected], dtype='int64'))
        print(f'{name}: {len(selected)} 段, {len(split_groups[name])} 组', flush=True)
    with (out/'segments.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys()); w.writeheader(); w.writerows(manifest)
    config = {'dataset_kind': 'real_GNSS_trajectory_with_simulated_radar_measurement',
              'archive': str(archive.resolve()), 'archive_md5': md5, 'seed': seed,
              'duration': 50., 'dt': .1, 'steps': 501, 'base_station_m': BS.tolist(),
              'max_source_gap_s': 1., 'segment_stride_s': 50.,
              'source_gps_files': len(names), 'nonempty_gps_files': nonempty_count,
              'segment_count': len(segments), 'excluded_laps': excluded,
              'split_counts': {name: sum(s['split']==name for s in segments) for name in split_groups},
              'split_groups': {k: v.tolist() for k, v in split_groups.items()},
              'session_references': [{'group': groups[k], 'home': list(k), **v} for k, v in references.items()],
              'limitations': ['约5Hz GNSS位置线性插值至10Hz，不增加实测时间分辨率；不跨超过1秒的缺口。',
                 'GNSS及初始飞控高度是参考估计，并非独立高精度真值；高度基准采用初始navPos对齐。',
                 '假设基站位于起飞参考位置上方10米，雷达距离误差1米、角度误差0.5度是仿真设定。',
                 '分组依据相同home坐标，属于保守记录组，不保证不同日期完全独立；同航线重复仍存在。',
                 '测试集仅用于冻结模型后的评价。']}
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (out.parent/'latest_run.txt').write_text(out.name, encoding='utf-8')
    print('数据目录:', out, flush=True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=BASE/'datasets/idf_ds/Speedybee DataSet.zip')
    p.add_argument('--seed', type=int, default=20261112)
    args = p.parse_args(); build(args.archive, args.seed)
