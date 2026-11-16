"""正式版0.05秒采样/半格DFT标签；明确区分真实轨迹扩展与合成代理数据。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.common import BASE, audit_partitions, finish_run, load_split, new_run, read_json, write_json

BS = np.array([0., 0., 10.])


def centered_dft_beams(position):
    rel = position-BS
    distance = np.maximum(np.linalg.norm(rel, axis=-1), 1e-8)
    uv = rel[..., :2]/distance[..., None]
    ij = np.clip(np.floor((uv+1)*16).astype('int64'), 0, 31)
    return ij[..., 1]*32+ij[..., 0]+1


def measure(position, rng, range_sigma, angle_sigma):
    rel = position-BS; distance = np.linalg.norm(rel, axis=-1)
    az = np.arctan2(rel[..., 1], rel[..., 0])
    polar = np.arccos(np.clip(rel[..., 2]/np.maximum(distance, 1e-8), -1, 1))
    return np.stack((np.maximum(distance+rng.normal(0, range_sigma, distance.shape), 1e-6),
                     (az+rng.normal(0, np.deg2rad(angle_sigma), az.shape)+np.pi)%(2*np.pi)-np.pi,
                     np.clip(polar+rng.normal(0, np.deg2rad(angle_sigma), polar.shape), 0, np.pi)), -1).astype('float32')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-data', type=Path)
    p.add_argument('--count', type=int, default=10000)
    p.add_argument('--validation-count', type=int, default=1000)
    p.add_argument('--test-count', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261113)
    p.add_argument('--range-sigma', type=float, default=1.)
    p.add_argument('--angle-sigma-deg', type=float, default=.5)
    p.add_argument('--z-sigma', type=float, default=.1)
    p.add_argument('--output-root', type=Path, default=BASE/'outputs/step05g_journal')
    a = p.parse_args()
    if min(a.count, a.validation_count, a.test_count) < 5 or min(a.range_sigma, a.angle_sigma_deg, a.z_sigma) < 0:
        p.error('数量或噪声参数无效')
    out = new_run(a.output_root)
    cfg = {'target': 'local_TWC_2026_3720890_single_target', 'dt': .05, 'base_station_m': BS.tolist(),
           'seed': a.seed, 'source_data': str(a.source_data.resolve()) if a.source_data else None,
           'measurement_model': 'Gaussian_range_angle_proxy_NOT_full_OFDM_Swerling_I_receiver',
           'measurement_parameters': {'range_sigma_m': a.range_sigma, 'angle_sigma_deg': a.angle_sigma_deg},
           'codebook': '32x32_half_bin_centered_DFT_labels_1_to_1024',
           'limitations': ['原训练集及回波/参数估计器未公开；当前高斯距离角度测量是代理，不是完整物理数值复现。',
                           '初始水平半径、z扰动方差、损失权重等未给出的数值均为显式实验设定。']}
    counts = {}
    if a.source_data:
        old = read_json(a.source_data/'config.json'); dt = old['dt']
        if 'real_GNSS' not in old.get('dataset_kind', ''):
            p.error('--source-data仅接受已分组真实GNSS参考轨迹')
        new_times = np.arange(round(old['duration']/.05)+1)*.05
        cfg.update({'dataset_kind': 'real_GNSS_extension_journal_network_protocol', 'duration': old['duration'],
                    'steps': len(new_times), 'split_groups': old['split_groups'],
                    'test_records_used_previously': True, 'source_reference_interpolation': 'offline_position_only'})
        cfg['limitations'].extend(['约5Hz GPS参考位置插值至20Hz不增加真实动态细节；短间隔预测可能受插值平滑影响。',
                                   '原始记录组互斥，但同一飞行路线重复，且测试记录组在此前实验已使用。'])
        for split in ('nnet_train', 'rnet_sim', 'validation', 'test'):
            data = load_split(a.source_data/f'{split}.npz'); old_times = np.arange(data['position'].shape[1])*dt
            position = np.array([[np.interp(new_times, old_times, row[:, j]) for j in range(3)]
                                 for row in data['position']], dtype='float32').transpose(0, 2, 1)
            data['position'] = position
            data['measurement'] = np.stack([measure(row, np.random.default_rng(a.seed+100000+int(i)), a.range_sigma, a.angle_sigma_deg)
                                            for row, i in zip(position, data['trajectory_id'])])
            data['beam'] = centered_dft_beams(position)
            np.savez_compressed(out/f'{split}.npz', **data)
            counts[split] = len(position); print(split, counts[split], position.shape, flush=True)
    else:
        # 1000个50ms评估时刻；额外5个初始化时刻的约定沿用公开数据1005点。
        steps = 1005
        cfg.update({'dataset_kind': 'synthetic_journal_mobility_Gaussian_measurement_proxy', 'duration': 50.,
                    'steps': steps, 'warmup_samples': 5, 'evaluated_blocks': 1000,
                    'requested_training_trajectories': a.count, 'independent_validation_extra': a.validation_count,
                    'initial_horizontal_radius_m': [60., 160.], 'z_increment_sigma_m': a.z_sigma})
        nbase = int(a.count*.8)
        sizes = {'nnet_train': nbase, 'rnet_sim': a.count-nbase,
                 'validation': a.validation_count, 'test': a.test_count}
        offset = 0
        for split, count in sizes.items():
            position = np.empty((count, steps, 3), 'float32'); measurements = np.empty_like(position)
            ids = np.arange(offset, offset+count); speeds = np.empty(count, 'float32'); offset += count
            for row, i in enumerate(ids):
                rng = np.random.default_rng(a.seed+int(i))
                radius, az = rng.uniform(60, 160), rng.uniform(-np.pi, np.pi)
                speed, heading = rng.uniform(10, 35), rng.uniform(0, 2*np.pi)
                angles = heading+np.r_[0., np.cumsum(rng.uniform(-np.pi/9, np.pi/9, steps-2))]
                xyz = np.empty((steps, 3)); xyz[0] = [radius*np.cos(az), radius*np.sin(az), rng.uniform(50, 80)]
                xyz[1:, 0] = xyz[0, 0]+np.cumsum(speed*.05*np.cos(angles))
                xyz[1:, 1] = xyz[0, 1]+np.cumsum(speed*.05*np.sin(angles))
                xyz[1:, 2] = xyz[0, 2]+np.cumsum(rng.normal(0, a.z_sigma, steps-1))
                position[row], speeds[row] = xyz, speed
                measurements[row] = measure(xyz, rng, a.range_sigma, a.angle_sigma_deg)
            np.savez_compressed(out/f'{split}.npz', position=position, measurement=measurements,
                                beam=centered_dft_beams(position), speed=speeds, trajectory_id=ids)
            counts[split] = count; print(split, count, flush=True)
    cfg['split_counts'] = counts
    write_json(out/'config.json', cfg); write_json(out/'audit.json', audit_partitions(out))
    finish_run(out); print(out, flush=True)


if __name__ == '__main__':
    main()
