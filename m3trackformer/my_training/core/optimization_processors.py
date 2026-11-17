"""Causal preprocessing outside the frozen Transformer N-Net/R-Net."""
import numpy as np

from .measurement_quality import (RobustMeasurementProcessor, cartesian_measurement_covariance,
    cartesian_to_spherical, spherical_to_cartesian)


class RobustIMMProcessor:
    """Two interacting CV models, different process noise, robust measurement updates.

    Mode evidence uses a Student-like innovation likelihood to limit outlier
    influence. This is an experimental robust IMM approximation, not an exact
    Bayesian Student-t filter. No mode or motion ground-truth labels are read.
    """
    def __init__(self, measurement_init, sigma_init, dt, low_acceleration=3., high_acceleration=24.):
        z, sigma = np.asarray(measurement_init), np.asarray(sigma_init)
        self.models = [RobustMeasurementProcessor(z[..., :1, :], sigma[..., :1, :], dt,
                        acceleration_std=a) for a in (low_acceleration, high_acceleration)]
        self.shape = z.shape[:2]
        self.probabilities = np.broadcast_to([.8, .2], (*self.shape, 2)).copy()
        self.transition = np.array([[.98, .02], [.05, .95]])
        self.time_index = 0
        self.filtered_initial_history = np.empty_like(z, dtype=np.float32)
        self.filtered_initial_history[..., 0, :] = z[..., 0, :]
        for step in range(1, z.shape[-2]):
            filtered, _ = self.step(step, np.ones(self.shape, bool), z[..., step, :], sigma[..., step, :])
            self.filtered_initial_history[..., step, :] = filtered

    def step(self, step_index, observed, measurement, sigma):
        if step_index != self.time_index+1:
            raise ValueError('IMM requires consecutive causal steps')
        observed = np.asarray(observed, dtype=bool)
        old_x = np.stack([m.state.copy() for m in self.models], -2)
        old_p = np.stack([m.covariance.copy() for m in self.models], -3)
        prior = self.probabilities @ self.transition
        likelihoods, diagnostics = [], []
        for j, model in enumerate(self.models):
            mixing = self.probabilities*self.transition[:, j]/prior[..., j, None]
            mean = (mixing[..., None]*old_x).sum(-2)
            difference = old_x-mean[..., None, :]
            covariance = (mixing[..., None, None]*(old_p+
                difference[..., :, :, None]*difference[..., :, None, :])).sum(-3)
            model.state, model.covariance = mean, covariance
            log_likelihood = np.zeros(self.shape)
            if observed.any():
                z, sig = np.asarray(measurement)[observed], np.asarray(sigma)[observed]
                xp = model.state[observed] @ model.transition.T
                pp = model.transition @ model.covariance[observed] @ model.transition.T + model.process_covariance
                innovation = spherical_to_cartesian(z)-xp[:, :3]
                innovation_cov = pp[:, :3, :3]+cartesian_measurement_covariance(z, sig)
                nis = np.einsum('bi,bi->b', innovation,
                    np.linalg.solve(innovation_cov, innovation[..., None])[..., 0])
                log_likelihood[observed] = -.5*np.linalg.slogdet(innovation_cov)[1]-3.5*np.log1p(np.maximum(nis, 0)/4.)
            likelihoods.append(log_likelihood)
            _, diag = model.step(step_index, observed, measurement, sigma)
            diagnostics.append(diag)
        log_p = np.log(np.maximum(prior, 1e-300))+np.stack(likelihoods, -1)
        exp_p = np.exp(log_p-log_p.max(-1, keepdims=True))
        self.probabilities = exp_p/exp_p.sum(-1, keepdims=True)
        self.time_index = step_index
        fused = sum(self.probabilities[..., j, None]*m.state[..., :3] for j, m in enumerate(self.models))
        weights = sum(self.probabilities[..., j]*d['weight'] for j, d in enumerate(diagnostics))
        nis = sum(prior[..., j]*d['nis'] for j, d in enumerate(diagnostics))
        return cartesian_to_spherical(fused).astype('float32'), {'weight': weights, 'nis': nis}


def make_processor(measurement_init, sigma_init, dt, *, configuration):
    config = dict(configuration)
    kind = config.pop('kind', 'huber')
    if kind == 'imm':
        return RobustIMMProcessor(measurement_init, sigma_init, dt, **config)
    loss = {'kalman': 'none', 'student': 'student', 'adaptive': 'huber', 'huber': 'huber'}[kind]
    return RobustMeasurementProcessor(measurement_init, sigma_init, dt,
        robust_loss=loss, adaptive_process=kind == 'adaptive', **config)


def candidates():
    result = [dict(id='raw', family='raw', label='扰动直接输入', processor=None, temperature=1.),
        dict(id='huber_base', family='huber', label='当前Huber基准', processor={'kind': 'huber'}, temperature=1.)]
    for a in (6., 24.):
        result.append(dict(id=f'kalman_q{int(a)}', family='kalman', label='普通卡尔曼',
            processor={'kind': 'kalman', 'acceleration_std': a}, temperature=1.))
    for df in (3., 6.):
        result.append(dict(id=f'student_{int(df)}', family='student', label='Student-t式降权',
            processor={'kind': 'student', 'student_df': df}, temperature=1.))
    for scale in (9., 25.):
        result.append(dict(id=f'adaptive_{int(scale)}', family='adaptive', label='自适应过程噪声',
            processor={'kind': 'adaptive', 'adaptive_max_scale': scale}, temperature=1.))
    for low, high in ((3., 24.), (6., 48.)):
        result.append(dict(id=f'imm_{int(low)}_{int(high)}', family='imm', label='鲁棒双模型IMM',
            processor={'kind': 'imm', 'low_acceleration': low, 'high_acceleration': high}, temperature=1.))
    for a, threshold in ((6., 2.), (24., 4.)):
        result.append(dict(id=f'huber_q{int(a)}_c{int(threshold)}', family='huber', label='Huber参数优化',
            processor={'kind': 'huber', 'acceleration_std': a, 'huber_threshold': threshold}, temperature=1.))
    for temp in (.7, 1.3, 2.):
        result.append(dict(id=f'temperature_{str(temp).replace(".", "p")}', family='temperature',
            label='Huber＋外部概率温度', processor={'kind': 'huber'}, temperature=temp))
    return result
