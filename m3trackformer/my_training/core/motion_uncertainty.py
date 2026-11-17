"""V4: causal Transformer forecasting with a heteroscedastic Gaussian mixture.

The output describes conditional future-motion uncertainty. It is not an
epistemic uncertainty estimate or evidence that random future commands are known.
All forecast coordinates and standard deviations are expressed in metres.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import positional_encoding


@dataclass
class MotionConfig:
    history: int = 20
    horizons: tuple = (1, 5, 10, 20)
    components: int = 3
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = .1
    position_scale: float = 10.
    minimum_std_m: float = .25
    maximum_std_m: float = 200.
    velocity_limit_m_s: float = 80.
    dt: float = .05

    def validate(self):
        if self.history < 2 or self.components < 1 or self.layers < 1:
            raise ValueError('History, mixture components and layer counts are invalid')
        if not self.horizons or any(int(h) != h or h < 1 for h in self.horizons):
            raise ValueError('Forecast horizons must be positive integer sample offsets')
        if sorted(set(self.horizons)) != list(self.horizons):
            raise ValueError('Forecast horizons must be unique and ascending')
        if self.d_model < 4 or self.heads < 1 or self.d_model % self.heads:
            raise ValueError('Transformer dimensions and heads are incompatible')
        if not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be in [0,1)')
        if not (0 < self.minimum_std_m < self.maximum_std_m
                and self.position_scale > 0 and self.velocity_limit_m_s > 0 and self.dt > 0):
            raise ValueError('Forecast scales and sampling interval must be positive')
        if not all(np.isfinite(v) for v in (self.position_scale, self.minimum_std_m,
                                            self.maximum_std_m, self.velocity_limit_m_s, self.dt)):
            raise ValueError('Forecast scales must be finite')
        return self

    def dictionary(self):
        return asdict(self)


def _cartesian(z):
    r, az, polar = z.unbind(-1)
    sp = polar.sin()
    return torch.stack((r*sp*az.cos(), r*sp*az.sin(), r*polar.cos()+10.), -1)


def _cartesian_variance(z, sigma):
    r, az, polar = z.unbind(-1)
    sr, sa, sp = sigma.unbind(-1)
    cp, pp, ca, aa = polar.cos(), polar.sin(), az.cos(), az.sin()
    return torch.stack(((pp*ca*sr).square() + (r*pp*aa*sa).square() + (r*cp*ca*sp).square(),
                        (pp*aa*sr).square() + (r*pp*ca*sa).square() + (r*cp*aa*sp).square(),
                        (cp*sr).square() + (r*pp*sp).square()), -1).clamp_min(1.e-6)


def prepare_history(history, valid, sigma=None, dt=.05, position_scale=10., velocity_limit=80.):
    """Use past observations only; invalid entries, including NaNs, are invisible.

    Returns ten input features, last observed Cartesian origin, past-only weighted
    least-squares velocity, age of the last observation and last valid indices.
    """
    if history.ndim != 3 or history.shape[-1] != 3 or valid.shape != history.shape[:2]:
        raise ValueError('Expected spherical history [batch,time,3] and valid [batch,time]')
    if valid.dtype != torch.bool or not bool(valid.any(1).all()):
        raise ValueError('Every history requires at least one valid past observation')
    if not np.isfinite(dt) or dt <= 0 or position_scale <= 0 or velocity_limit <= 0:
        raise ValueError('Sampling interval and normalization scales must be positive')
    if not bool(torch.isfinite(history[valid]).all()):
        raise ValueError('Valid measurements must be finite')
    # Sanitize BEFORE trigonometric operations so masked NaNs cannot contaminate
    # either predictions or backward gradients.
    z = torch.where(valid[..., None], history, torch.zeros_like(history))
    xyz = _cartesian(z)
    if sigma is None:
        variance = torch.ones_like(xyz)
    else:
        if sigma.shape != history.shape or not bool((torch.isfinite(sigma[valid]) & (sigma[valid] > 0)).all()):
            raise ValueError('Valid nominal spherical standard deviations must be positive and finite')
        safe_sigma = torch.where(valid[..., None], sigma, torch.ones_like(sigma))
        variance = _cartesian_variance(z, safe_sigma)
    b, h, _ = history.shape
    rows = torch.arange(b, device=history.device)
    indices = torch.arange(h, device=history.device)
    last = torch.where(valid, indices[None], -1).max(1).values
    origin = xyz[rows, last]
    times = (indices.to(history.dtype)-(h-1))*dt
    age = -times[last]
    weight = valid.to(history.dtype)/(variance.sum(-1)+1.)
    weight = weight/weight.sum(1, keepdim=True).clamp_min(1.e-12)
    time_center = (weight*times[None]).sum(1)
    position_center = (weight[..., None]*xyz).sum(1)
    centered_time = times[None]-time_center[:, None]
    denominator = (weight*centered_time.square()).sum(1)
    velocity = (weight[..., None]*centered_time[..., None]*(xyz-position_center[:, None])).sum(1)
    velocity = velocity/denominator[:, None].clamp_min(1.e-8)
    velocity = torch.where((denominator > 1.e-8)[:, None], velocity, torch.zeros_like(velocity))
    speed = velocity.norm(dim=-1)
    velocity = velocity*(velocity_limit/speed.clamp_min(velocity_limit))[:, None]
    relative = (xyz-origin[:, None])/position_scale
    quality = torch.log1p(variance.sqrt()/position_scale)
    time_feature = (times/max((h-1)*dt, dt))[None, :, None].expand(b, -1, -1)
    features = torch.cat((relative, time_feature, valid[..., None].to(history.dtype), quality,
                          (velocity.norm(dim=-1)/35.)[:, None, None].expand(-1, h, -1),
                          (age/max(h*dt, dt))[:, None, None].expand(-1, h, -1)), -1)
    features = features.masked_fill(~valid[..., None], 0.)
    return features, origin, velocity, age, last


class MotionUncertaintyNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = (config or MotionConfig()).validate()
        self.embedding = nn.Linear(10, c.d_model)
        layer = nn.TransformerEncoderLayer(c.d_model, c.heads, c.d_model*2, c.dropout,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(c.d_model*2+4, c.d_model*2), nn.GELU(),
                                  nn.Linear(c.d_model*2, len(c.horizons)*c.components*7))
        final = self.head[-1]
        nn.init.normal_(final.weight, std=.01)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            bias = final.bias.view(len(c.horizons), c.components, 7)
            for i, horizon in enumerate(c.horizons):
                std = max(3., horizon*c.dt*10.)
                bias[i, :, 4:] = math.log(math.expm1((std-c.minimum_std_m)/c.position_scale))
                if c.components > 1:
                    bias[i, :, 1] = torch.linspace(-1, 1, c.components)*horizon*c.dt*.2

    def forward(self, history, valid, dt=None, sigma=None):
        c = self.config
        if history.shape[1] != c.history:
            raise ValueError(f'This forecaster requires {c.history} history samples')
        dt = c.dt if dt is None else dt
        features, origin, velocity, age, last = prepare_history(
            history, valid, sigma, dt, c.position_scale, c.velocity_limit_m_s)
        encoded = self.embedding(features)
        encoded = encoded+positional_encoding(c.history, c.d_model, encoded.device, encoded.dtype)
        encoded = self.encoder(encoded, src_key_padding_mask=~valid)
        pool = (encoded*valid[..., None]).sum(1)/valid.sum(1, keepdim=True)
        latest = encoded[torch.arange(len(history), device=history.device), last]
        context = torch.cat((pool, latest, velocity/35., age[:, None]), -1)
        raw = self.head(context).view(len(history), len(c.horizons), c.components, 7)
        log_weights = raw[..., 0].log_softmax(-1)
        horizon = torch.as_tensor(c.horizons, device=history.device, dtype=history.dtype)*dt
        base = origin[:, None]+velocity[:, None]*(horizon[None, :, None]+age[:, None, None])
        mean = base[:, :, None]+c.position_scale*raw[..., 1:4]
        scale = (c.minimum_std_m+c.position_scale*F.softplus(raw[..., 4:7])).clamp_max(c.maximum_std_m)
        return {'weights': log_weights.exp(), 'log_weights': log_weights, 'mean': mean, 'scale': scale,
                'origin': origin, 'velocity': velocity, 'base': base, 'observation_age_s': age}


def gaussian_mixture_nll(prediction, target, reduction='mean', valid=None):
    """Exact marginal 3D mixture NLL, including constants, in physical metre units."""
    mean, scale = prediction['mean'], prediction['scale']
    if target.shape != (*mean.shape[:2], 3) or scale.shape != mean.shape:
        raise ValueError('Target or Gaussian mixture shape is invalid')
    if valid is not None:
        if valid.shape != target.shape[:2] or not bool(valid.any()):
            raise ValueError('Future validity mask is invalid or empty')
        target = torch.where(valid[..., None], target, torch.zeros_like(target))
    log_weights = prediction.get('log_weights')
    if log_weights is None:
        log_weights = prediction['weights'].clamp_min(1.e-30).log()
    difference = (target[:, :, None]-mean)/scale
    log_component = -.5*(difference.square()+2*scale.log()+math.log(2*math.pi)).sum(-1)
    nll = -torch.logsumexp(log_weights+log_component, -1)
    if reduction == 'none':
        return nll if valid is None else nll.masked_fill(~valid, 0)
    values = nll if valid is None else nll[valid]
    if reduction == 'mean':
        return values.mean()
    if reduction == 'sum':
        return values.sum()
    raise ValueError('Unknown NLL reduction')


def predictive_mean(prediction):
    return (prediction['weights'][..., None]*prediction['mean']).sum(-2)


def predictive_std(prediction):
    mean = predictive_mean(prediction)
    variance = (prediction['weights'][..., None]*(prediction['scale']**2+
                (prediction['mean']-mean[..., None, :])**2)).sum(-2)
    return variance.clamp_min(0).sqrt() if torch.is_tensor(variance) else np.sqrt(np.maximum(variance, 0))


def _numpy_prediction(prediction, temperature=1.):
    def convert(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    result = {key: convert(prediction[key]).astype(np.float64) for key in ('weights', 'mean', 'scale')}
    weights, mean, scale = result['weights'], result['mean'], result['scale']
    if mean.ndim != 4 or mean.shape[-1] != 3 or scale.shape != mean.shape or weights.shape != mean.shape[:-1]:
        raise ValueError('Mixture must have [batch,horizon,component,3] means and scales')
    if not all(np.isfinite(a).all() for a in result.values()) or np.any(scale <= 0) or np.any(weights < 0):
        raise ValueError('Mixture parameters must be finite with positive scales and nonnegative weights')
    if not np.allclose(weights.sum(-1), 1., atol=1.e-5) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Mixture weights or scale temperature are invalid')
    result['scale'] *= temperature
    return result


def apply_scale_temperature(prediction, temperature):
    """A standard-deviation multiplier; covariance is multiplied by its square."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Scale temperature must be finite and positive')
    return {**prediction, 'scale': prediction['scale']*temperature}


def _numpy_nll(prediction, target):
    from scipy.special import logsumexp
    z = (np.asarray(target)[:, :, None]-prediction['mean'])/prediction['scale']
    component = -.5*(z*z+2*np.log(prediction['scale'])+math.log(2*math.pi)).sum(-1)
    return -logsumexp(np.log(np.maximum(prediction['weights'], 1.e-300))+component, axis=-1)


def fit_scale_temperature(calibration_prediction, calibration_target, bounds=(.2, 8.), split_name='calibration'):
    """Fit on an independent calibration split, never on evaluation/test targets."""
    from scipy.optimize import minimize_scalar
    if split_name not in ('calibration', 'validation'):
        raise ValueError('Scale calibration cannot be fitted on test/evaluation observations')
    prediction = _numpy_prediction(calibration_prediction)
    target = np.asarray(calibration_target.detach().cpu() if torch.is_tensor(calibration_target) else calibration_target)
    if target.shape != (*prediction['mean'].shape[:2], 3) or not np.isfinite(target).all():
        raise ValueError('Calibration targets must be finite and aligned with the predictions')
    if not (0 < bounds[0] < bounds[1] and np.isfinite(bounds).all()):
        raise ValueError('Temperature bounds must be positive and finite')
    objective = lambda log_t: float(_numpy_nll({**prediction, 'scale': prediction['scale']*np.exp(log_t)}, target).mean())
    result = minimize_scalar(objective, bounds=np.log(bounds), method='bounded', options={'xatol': 1.e-5})
    if not result.success:
        raise RuntimeError('Scale calibration did not converge')
    return float(np.exp(result.x))


def sample_mixture(prediction, samples=64, seed=0, temperature=1.):
    """Independent predictive samples with shape [samples,batch,horizon,3]."""
    if int(samples) != samples or samples < 1:
        raise ValueError('Sample count must be a positive integer')
    p = _numpy_prediction(prediction, temperature)
    rng = np.random.default_rng(seed)
    b, f, k = p['weights'].shape
    cumulative = np.cumsum(p['weights'], -1)
    cumulative[..., -1] = 1.
    uniforms = rng.random((int(samples), b, f))
    component = (uniforms[..., None] >= cumulative[None]).sum(-1).clip(max=k-1)
    bi, fi = np.arange(b)[None, :, None], np.arange(f)[None, None, :]
    mean, scale = p['mean'][bi, fi, component], p['scale'][bi, fi, component]
    return mean+scale*rng.normal(size=mean.shape)


def mixture_marginal_intervals(prediction, levels=(.5, .8, .9, .95), temperature=1.):
    """Central coordinate-wise mixture intervals; these are NOT joint 3D regions."""
    from scipy.special import ndtr
    p = _numpy_prediction(prediction, temperature)
    mean, scale, weights = p['mean'], p['scale'], p['weights']
    bounds_low, bounds_high = (mean-10*scale).min(-2), (mean+10*scale).max(-2)
    def quantile(probability):
        low, high = bounds_low.copy(), bounds_high.copy()
        for _ in range(48):
            mid = (low+high)/2
            cdf = (weights[..., None]*ndtr((mid[..., None, :]-mean)/scale)).sum(-2)
            low = np.where(cdf < probability, mid, low)
            high = np.where(cdf >= probability, mid, high)
        return (low+high)/2
    result = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError('Interval levels must be in (0,1)')
        result[float(level)] = (quantile((1-level)/2), quantile((1+level)/2))
    return result


def evaluate_distribution(prediction, target, levels=(.5, .8, .9, .95), energy_samples=32, seed=0, temperature=1.):
    """Proper NLL/energy score plus marginal calibration and sharpness diagnostics."""
    p = _numpy_prediction(prediction, temperature)
    target = np.asarray(target.detach().cpu() if torch.is_tensor(target) else target)
    if target.shape != (*p['mean'].shape[:2], 3) or not np.isfinite(target).all():
        raise ValueError('Evaluation targets must be finite and aligned')
    mean = (p['weights'][..., None]*p['mean']).sum(-2)
    squared_error = ((mean-target)**2).sum(-1)
    nll = _numpy_nll(p, target)
    draw1 = sample_mixture(p, energy_samples, seed)
    draw2 = sample_mixture(p, energy_samples, seed+1)
    energy = np.linalg.norm(draw1-target[None], axis=-1).mean(0) - .5*np.linalg.norm(draw1-draw2, axis=-1).mean(0)
    intervals = mixture_marginal_intervals(p, levels)
    per_horizon = []
    for h in range(mean.shape[1]):
        row = {'sample_count': int(len(target)), 'nll_nats': float(nll[:, h].mean()),
               'mean_position_rmse_m': float(np.sqrt(squared_error[:, h].mean())),
               'energy_score_m': float(energy[:, h].mean())}
        for level, (lower, upper) in intervals.items():
            suffix = str(int(round(level*100)))
            row['marginal_coverage_'+suffix] = float(((target[:, h] >= lower[:, h]) &
                                                     (target[:, h] <= upper[:, h])).mean())
            row['marginal_width_'+suffix+'_m'] = float((upper[:, h]-lower[:, h]).mean())
        per_horizon.append(row)
    summary = {'sample_count': len(target), 'nll_nats': float(nll.mean()),
               'mean_position_rmse_m': float(np.sqrt(squared_error.mean())),
               'energy_score_m': float(energy.mean()), 'per_horizon': per_horizon,
               'coverage_scope': 'central_marginal_coordinate_intervals_not_joint_3D_coverage',
               'uncertainty_scope': 'conditional_predictive_mixture_not_epistemic_model_uncertainty'}
    for level, (lower, upper) in intervals.items():
        suffix = str(int(round(level*100)))
        summary['marginal_coverage_'+suffix] = float(((target >= lower) & (target <= upper)).mean())
        summary['marginal_width_'+suffix+'_m'] = float((upper-lower).mean())
    return summary
