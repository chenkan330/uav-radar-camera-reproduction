"""Reproducible range/SNR-dependent radar proxy and causal robust preprocessing.

The proxy is an experiment setting, not an OFDM receiver. Ground truth is used
only by the simulator to generate observations and to evaluate their errors.
The processor receives observed spherical measurements and reported sigmas only.
"""
from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    reference_range_m: float = 300.0
    reference_snr_db: float = 20.0
    range_power_exponent: float = 4.0
    fading_std_db: float = 2.0
    fading_correlation: float = 0.95
    minimum_snr_db: float = -10.0
    maximum_snr_db: float = 40.0
    reference_range_sigma_m: float = 1.0
    reference_angle_sigma_deg: float = 0.5
    range_sigma_floor_m: float = 0.3
    angle_sigma_floor_deg: float = 0.1
    outlier_probability: float = 0.02
    burst_start_probability: float = 0.001
    burst_min_length: int = 2
    burst_max_length: int = 4
    outlier_range_min_m: float = 20.0
    outlier_range_max_m: float = 80.0
    outlier_angle_min_deg: float = 3.0
    outlier_angle_max_deg: float = 10.0
    initialization_samples_without_outliers: int = 5

    def validate(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Noise parameters must be finite')
        positive = ('reference_range_m', 'reference_range_sigma_m',
                    'reference_angle_sigma_deg', 'range_sigma_floor_m',
                    'angle_sigma_floor_deg')
        if any(values[k] <= 0 for k in positive):
            raise ValueError('Reference range and measurement sigmas must be positive')
        if self.range_power_exponent < 0 or self.fading_std_db < 0:
            raise ValueError('Range exponent and fading standard deviation must be nonnegative')
        if not 0 <= self.fading_correlation < 1:
            raise ValueError('Fading correlation must be in [0, 1)')
        if not self.minimum_snr_db < self.maximum_snr_db:
            raise ValueError('SNR bounds are invalid')
        if any(not 0 <= p <= 1 for p in (self.outlier_probability, self.burst_start_probability)):
            raise ValueError('Outlier probabilities must be in [0, 1]')
        if not 1 <= self.burst_min_length <= self.burst_max_length:
            raise ValueError('Burst lengths are invalid')
        if any(int(v) != v for v in (self.burst_min_length, self.burst_max_length,
                                     self.initialization_samples_without_outliers)):
            raise ValueError('Burst and initialization lengths must be integers')
        if self.initialization_samples_without_outliers < 0:
            raise ValueError('Initialization length must be nonnegative')
        if not (0 <= self.outlier_range_min_m <= self.outlier_range_max_m
                and 0 <= self.outlier_angle_min_deg <= self.outlier_angle_max_deg):
            raise ValueError('Outlier magnitude bounds are invalid')
        return self


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def spherical_to_cartesian(z, base_station=(0., 0., 10.)):
    z = np.asarray(z, dtype=float)
    r, az, polar = np.moveaxis(z, -1, 0)
    sp = np.sin(polar)
    return np.stack((r * sp * np.cos(az), r * sp * np.sin(az), r * np.cos(polar)), -1) + base_station


def cartesian_to_spherical(position, base_station=(0., 0., 10.)):
    rel = np.asarray(position, dtype=float) - base_station
    r = np.maximum(np.linalg.norm(rel, axis=-1), 1e-8)
    return np.stack((r, np.arctan2(rel[..., 1], rel[..., 0]),
                     np.arccos(np.clip(rel[..., 2] / r, -1, 1))), -1)


def sigmas_from_snr(snr_db, config):
    """Independent spherical sigmas; inverse-square-root *linear* SNR scaling."""
    config.validate()
    scale2 = 10. ** ((config.reference_snr_db - np.asarray(snr_db)) / 10.)
    sr = np.sqrt(config.range_sigma_floor_m ** 2 + config.reference_range_sigma_m ** 2 * scale2)
    sa = np.deg2rad(np.sqrt(config.angle_sigma_floor_deg ** 2 + config.reference_angle_sigma_deg ** 2 * scale2))
    return np.stack((sr, sa, sa), -1)


def perturb_scenes(scenes, config=None, seed=20261115, base_station=(0., 0., 10.)):
    """Draw the entire external observation stream once, independently of policy.

    Every trajectory uses its own seed, so regrouping scenes does not alter its
    observations. First five acquisition samples have Gaussian noise but no
    injected outliers. Masks are evaluation labels and must never reach filters.
    """
    config = (config or NoiseConfig()).validate()
    if 'position' not in scenes:
        raise ValueError('Reference Cartesian trajectories are needed by the observation simulator')
    position = np.asarray(scenes['position'])
    ids = np.asarray(scenes['trajectory_id'])
    if position.ndim != 4 or position.shape[-1] != 3 or ids.shape != position.shape[:2]:
        raise ValueError('Expected positions [scene, target, time, 3] and IDs [scene, target]')
    if not np.isfinite(position).all() or np.any(ids < 0):
        raise ValueError('Trajectories and IDs must be finite and IDs nonnegative')
    truth = cartesian_to_spherical(position, base_station)
    shape = position.shape[:-1]
    measurement = np.empty_like(position, dtype=np.float32)
    sigma = np.empty_like(position, dtype=np.float32)
    snr = np.empty(shape, dtype=np.float32)
    outlier = np.empty(shape, dtype=bool)
    for index in np.ndindex(ids.shape):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(ids[index])]))
        z = truth[index]
        n = len(z)
        fade = np.empty(n)
        fade[0] = rng.normal(0, config.fading_std_db)
        innovations = rng.normal(0, config.fading_std_db * np.sqrt(1-config.fading_correlation**2), n-1)
        for t in range(1, n):
            fade[t] = config.fading_correlation * fade[t-1] + innovations[t-1]
        local_snr = np.clip(config.reference_snr_db - 10*config.range_power_exponent *
                            np.log10(np.maximum(z[:, 0], 1e-6)/config.reference_range_m) + fade,
                            config.minimum_snr_db, config.maximum_snr_db)
        local_sigma = sigmas_from_snr(local_snr, config)
        noisy = z + rng.normal(size=z.shape) * local_sigma
        mask = rng.random(n) < config.outlier_probability
        starts = np.flatnonzero(rng.random(n) < config.burst_start_probability)
        for start in starts:
            length = rng.integers(config.burst_min_length, config.burst_max_length+1)
            mask[start:min(n, start+length)] = True
        mask[:config.initialization_samples_without_outliers] = False
        # Each corrupted sample affects range and both angles; signs are independent.
        bias = np.column_stack((rng.uniform(config.outlier_range_min_m, config.outlier_range_max_m, n),
                                np.deg2rad(rng.uniform(config.outlier_angle_min_deg, config.outlier_angle_max_deg, n)),
                                np.deg2rad(rng.uniform(config.outlier_angle_min_deg, config.outlier_angle_max_deg, n))))
        bias *= rng.choice([-1., 1.], size=z.shape)
        noisy += mask[:, None] * bias
        noisy[:, 0] = np.maximum(noisy[:, 0], 1e-6)
        noisy[:, 1] = wrap_angle(noisy[:, 1])
        noisy[:, 2] = np.clip(noisy[:, 2], 0, np.pi)
        measurement[index], sigma[index], snr[index], outlier[index] = noisy, local_sigma, local_snr, mask
    result = {key: value.copy() if hasattr(value, 'copy') else value for key, value in scenes.items()}
    result.update(measurement=measurement, measurement_sigma=sigma, snr_db=snr, outlier_mask=outlier)
    metadata = {'noise_config': asdict(config), 'seed': int(seed),
                'base_station_m': list(base_station), 'observation_model': 'heteroscedastic_spherical_radar_proxy',
                'snr_model': 'reference_snr_db - 10*range_power_exponent*log10(range/reference_range) + AR1_fading',
                'outlier_count': int(outlier.sum()), 'measurement_count': int(outlier.size),
                'actual_outlier_fraction': float(outlier.mean()),
                'snr_db_min': float(snr.min()), 'snr_db_median': float(np.median(snr)), 'snr_db_max': float(snr.max()),
                'range_m_min': float(truth[..., 0].min()), 'range_m_max': float(truth[..., 0].max()),
                'labels_evaluation_only': ['outlier_mask', 'position', 'beam'],
                'limitations': ['Monostatic r^-4 received-power proxy; not a physical waveform/receiver simulation.',
                                'Reported SNR and nominal covariance are idealized metadata; no online covariance estimator is simulated.',
                                'Outliers contaminate acquired measurements; beam misses retain the original ideal feedback model.',
                                'Initialization observations have Gaussian noise but injected outliers are excluded explicitly.']}
    return result, metadata


def cartesian_measurement_covariance(z, sigma):
    """First-order covariance propagation J diag(sigma**2) J^T."""
    z, sigma = np.asarray(z, dtype=float), np.asarray(sigma, dtype=float)
    r, az, polar = np.moveaxis(z, -1, 0)
    ca, sa, cp, sp = np.cos(az), np.sin(az), np.cos(polar), np.sin(polar)
    jac = np.stack((np.stack((sp*ca, -r*sp*sa, r*cp*ca), -1),
                    np.stack((sp*sa, r*sp*ca, r*cp*sa), -1),
                    np.stack((cp, np.zeros_like(r), -r*sp), -1)), -2)
    return (jac * sigma[..., None, :]**2) @ np.swapaxes(jac, -1, -2) + np.eye(3)*1e-6


def render_noise_diagnostics(scenes, out, base_station=(0., 0., 10.)):
    """Plot the complete generated stream; evaluation labels never enter tracking.

    Error plots use Cartesian reference positions only for offline evaluation.
    Bins include every generated observation, irrespective of later beam hits,
    so tracking exposure/survivorship does not affect these diagnostic plots.
    """
    import matplotlib
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(scenes['position'])
    measurement = np.asarray(scenes['measurement'])
    sigmas = np.asarray(scenes['measurement_sigma']).reshape(-1, 3)
    snr = np.asarray(scenes['snr_db']).ravel()
    bad = np.asarray(scenes['outlier_mask'], bool).ravel()
    distance = np.linalg.norm(truth-np.asarray(base_station), axis=-1).ravel()
    error = np.linalg.norm(spherical_to_cartesian(measurement, base_station)-truth, axis=-1).ravel()
    if not len(distance) or not all(np.isfinite(x).all() for x in (distance, error, snr, sigmas)):
        raise ValueError('Noise diagnostics require nonempty finite arrays')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rows = []
    for kind, coordinate in [('range_m', distance), ('snr_db', snr)]:
        lo, hi = float(coordinate.min()), float(coordinate.max())
        if np.isclose(lo, hi):
            lo, hi = lo-.5, hi+.5
        boundaries = np.linspace(lo, hi, 9)
        index = np.clip(np.searchsorted(boundaries, coordinate, side='right')-1, 0, 7)
        for b in range(8):
            mask = index == b
            clean, dirty = mask & ~bad, mask & bad
            if not mask.any():
                continue
            def rmse(selection):
                return float(np.sqrt(np.mean(error[selection]**2))) if selection.any() else None
            rows.append({'bin_kind': kind, 'lower_bound': float(boundaries[b]),
                         'upper_bound': float(boundaries[b+1]), 'count': int(mask.sum()),
                         'clean_count': int(clean.sum()), 'outlier_count': int(dirty.sum()),
                         'mean_coordinate': float(coordinate[mask].mean()),
                         'median_snr_db': float(np.median(snr[mask])),
                         'median_range_sigma_m': float(np.median(sigmas[mask, 0])),
                         'median_angle_sigma_deg': float(np.rad2deg(np.median(sigmas[mask, 1]))),
                         'raw_cartesian_rmse_m': rmse(mask), 'clean_cartesian_rmse_m': rmse(clean),
                         'outlier_cartesian_rmse_m': rmse(dirty),
                         'clean_cartesian_median_error_m': float(np.median(error[clean])) if clean.any() else None,
                         'outlier_cartesian_median_error_m': float(np.median(error[dirty])) if dirty.any() else None})
    csv_path = out/'noise_bins.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(distance), min(len(distance), 5000), replace=False)
    axes[0, 0].scatter(distance[sample], snr[sample], s=3, alpha=.2, color='#3769a2', rasterized=True)
    axes[0, 0].set(xlabel='真实距离（米）', ylabel='信噪比（dB）', title='距离相关信噪比＋相关衰落')
    by_range = [r for r in rows if r['bin_kind'] == 'range_m']
    by_snr = [r for r in rows if r['bin_kind'] == 'snr_db']
    axes[0, 1].plot([r['mean_coordinate'] for r in by_range], [r['median_range_sigma_m'] for r in by_range],
                    'o-', color='#3769a2')
    axes[0, 1].set(xlabel='真实距离分箱均值（米）', ylabel='距离误差标准差中位数（米）', title='距离误差随距离变化')
    axes[0, 2].plot([r['mean_coordinate'] for r in by_snr], [r['median_angle_sigma_deg'] for r in by_snr],
                    'o-', color='#3769a2')
    axes[0, 2].set(xlabel='信噪比分箱均值（dB）', ylabel='角度误差标准差中位数（度）', title='角度误差随信噪比变化')
    bins = np.linspace(0, max(float(error.max()), 1.), 55)
    for select, color, label in [(~bad, '#3769a2', '未注入异常值'), (bad, '#ca5b45', '注入异常值')]:
        if select.any():
            axes[1, 0].hist(error[select], bins=bins, histtype='step', linewidth=1.8, color=color,
                            label=f'{label}（{select.sum():,}）')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set(xlabel='原始测量的三维位置误差（米）', ylabel='样本数（对数）', title='正常测量与异常测量的误差分布')
    axes[1, 0].legend(frameon=False, fontsize=9)
    for key, color, label in [('clean_cartesian_rmse_m', '#3769a2', '未注入异常值'),
                              ('outlier_cartesian_rmse_m', '#ca5b45', '注入异常值')]:
        points = [r for r in by_range if r[key] is not None]
        if points:
            axes[1, 1].plot([r['mean_coordinate'] for r in points], [r[key] for r in points], 'o-', color=color, label=label)
    axes[1, 1].set(xlabel='真实距离分箱均值（米）', ylabel='原始测量位置 RMSE（米）', title='各距离段的三维测量误差')
    axes[1, 1].legend(frameon=False, fontsize=9)
    centers = [r['mean_coordinate'] for r in by_range]
    widths = [.7*(r['upper_bound']-r['lower_bound']) for r in by_range]
    axes[1, 2].bar(centers, [r['outlier_count'] for r in by_range], width=widths, color='#ca5b45', alpha=.85)
    axes[1, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1, 2].set(xlabel='真实距离分箱均值（米）', ylabel='注入异常值数量', title=f'异常值共 {bad.sum():,}/{len(bad):,}（{bad.mean():.2%}）')
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('扰动数据诊断｜完整生成测量流，包含未被波束命中的时刻', fontsize=15)
    fig.supxlabel('SNR 与名义误差范围为理想化接收机质量信息；真实位置和异常标签仅供离线评价。', fontsize=10)
    figure_path = out/'noise_diagnostics.png'
    fig.savefig(figure_path, dpi=180, facecolor='white')
    plt.close(fig)
    return {'figure': str(figure_path), 'bins_csv': str(csv_path), 'measurement_count': len(bad),
            'outlier_count': int(bad.sum()), 'raw_cartesian_rmse_m': float(np.sqrt(np.mean(error**2)))}


class RobustMeasurementProcessor:
    """CV Kalman filter with converted measurements and Huber innovation weights.

    State is Cartesian position/velocity. Prediction runs for every target at
    every time step, while updates access observed entries only. There is no hard
    rejection, no use of outlier labels, and no insertion of missed observations.
    This first-order converted-measurement filter is approximate, not exact EKF.
    """
    def __init__(self, measurement_init, sigma_init, dt, base_station=(0., 0., 10.),
                 huber_threshold=3.0, acceleration_std=12.0, initial_velocity_std=35.0):
        z, sig = np.asarray(measurement_init, dtype=float), np.asarray(sigma_init, dtype=float)
        if z.ndim != 4 or z.shape != sig.shape or z.shape[-1] != 3 or z.shape[-2] < 1:
            raise ValueError('Initialization requires [scene, target, history, 3] measurements and sigmas')
        if not np.isfinite(z).all() or not np.isfinite(sig).all() or np.any(sig <= 0):
            raise ValueError('Initialization measurements must be finite and sigmas positive')
        parameters = (dt, huber_threshold, acceleration_std, initial_velocity_std)
        if not (all(np.isfinite(v) for v in parameters) and all(v > 0 for v in parameters)):
            raise ValueError('Filter parameters must be finite and positive')
        self.shape, self.dt, self.base_station = z.shape[:2], float(dt), np.asarray(base_station, dtype=float)
        if self.base_station.shape != (3,) or not np.isfinite(self.base_station).all():
            raise ValueError('Base station must contain three finite coordinates')
        self.huber_threshold = float(huber_threshold)
        self.transition = np.eye(6)
        self.transition[:3, 3:] = np.eye(3)*dt
        gain = np.vstack((np.eye(3)*dt**2/2, np.eye(3)*dt))
        self.process_covariance = acceleration_std**2 * gain @ gain.T
        self.state = np.zeros((*self.shape, 6))
        self.covariance = np.zeros((*self.shape, 6, 6))
        self.state[..., :3] = spherical_to_cartesian(z[..., 0, :], self.base_station)
        self.covariance[..., :3, :3] = cartesian_measurement_covariance(z[..., 0, :], sig[..., 0, :])
        self.covariance[..., 3:, 3:] = np.eye(3)*initial_velocity_std**2
        self.time_index = 0
        self.filtered_initial_history = np.empty_like(z, dtype=np.float32)
        self.filtered_initial_history[..., 0, :] = z[..., 0, :]
        self.initial_weights = np.ones(z.shape[:-1])
        for t in range(1, z.shape[-2]):
            filtered, diagnostics = self.step(t, np.ones(self.shape, bool), z[..., t, :], sig[..., t, :])
            self.filtered_initial_history[..., t, :] = filtered
            self.initial_weights[..., t] = diagnostics['weight']

    def step(self, step_index, observed, measurement, sigma):
        observed = np.asarray(observed, dtype=bool)
        if step_index != self.time_index + 1:
            raise ValueError('Call once per consecutive time index, including time steps with no detections')
        if observed.shape != self.shape:
            raise ValueError('Observed mask shape differs from scene/target shape')
        z, sig = np.asarray(measurement), np.asarray(sigma)
        if z.shape != (*self.shape, 3) or sig.shape != z.shape:
            raise ValueError('Measurements and sigmas must have shape [scene, target, 3]')
        # Deliberately validate and access only entries for which a measurement exists.
        zo, so = z[observed].astype(float), sig[observed].astype(float)
        if not np.isfinite(zo).all() or not np.isfinite(so).all() or np.any(so <= 0):
            raise ValueError('Observed measurements must be finite and sigmas positive')
        f = self.transition
        self.state = self.state @ f.T
        self.covariance = f @ self.covariance @ f.T + self.process_covariance
        self.time_index = int(step_index)
        weights = np.full(self.shape, np.nan)
        nis = np.full(self.shape, np.nan)
        if observed.any():
            x, p = self.state[observed], self.covariance[observed]
            r = cartesian_measurement_covariance(zo, so)
            residual = spherical_to_cartesian(zo, self.base_station) - x[:, :3]
            s = p[:, :3, :3] + r
            solved = np.linalg.solve(s, residual[..., None])[..., 0]
            local_nis = np.maximum(np.einsum('bi,bi->b', residual, solved), 0.)
            weight = np.minimum(1., self.huber_threshold/np.maximum(np.sqrt(local_nis), 1e-12))
            # IRLS: Huber weight reduces observation information, R_eff=R/w.
            effective_r = r/np.maximum(weight[:, None, None], 1e-3)
            se = p[:, :3, :3]+effective_r
            k = np.swapaxes(np.linalg.solve(se, np.swapaxes(p[:, :, :3], -1, -2)), -1, -2)
            x += np.einsum('bij,bj->bi', k, residual)
            ikh = np.broadcast_to(np.eye(6), p.shape).copy()
            ikh[:, :, :3] -= k
            # Joseph form preserves positive-semidefinite covariance numerically.
            updated = ikh @ p @ np.swapaxes(ikh, -1, -2) + k @ effective_r @ np.swapaxes(k, -1, -2)
            self.state[observed] = x
            self.covariance[observed] = (updated + np.swapaxes(updated, -1, -2))/2
            weights[observed], nis[observed] = weight, local_nis
        filtered = cartesian_to_spherical(self.state[..., :3], self.base_station).astype(np.float32)
        return filtered, {'weight': weights, 'nis': nis}
