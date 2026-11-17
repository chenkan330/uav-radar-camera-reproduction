"""Causal mixture-to-codebook projection for the V4 controlled tracking comparison."""
import numpy as np
import torch

from .motion_uncertainty import predictive_mean


class MotionBeamPredictor:
    def __init__(self, model, device, temperature=1., mode='probability', fixed_scale=None,
                 samples=256, seed=20261120, record=False):
        if mode not in ('probability', 'point') or temperature <= 0:
            raise ValueError('Invalid motion projection mode/temperature')
        self.model, self.device = model.eval(), torch.device(device)
        self.history = model.config.history
        self.temperature, self.mode = float(temperature), mode
        self.records = [] if record else None
        self.fixed_scale = None if fixed_scale is None else torch.as_tensor(fixed_scale, device=device)
        if mode == 'point' and self.fixed_scale is None:
            raise ValueError('Point comparison requires validation-only fixed spread')
        uniform = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(samples)
        self.nodes = (2**.5*torch.erfinv(2*uniform.clamp(1e-6, 1-1e-6)-1)).to(device)
        i = torch.arange(1024, device=device)
        self.uv = torch.stack(((i % 32+.5)/16-1, (i//32+.5)/16-1), -1)

    @torch.inference_mode()
    def __call__(self, history, valid, sigma, dt):
        shape = history.shape[:2]
        x = torch.as_tensor(history.reshape(-1, self.history, 3), device=self.device, dtype=torch.float32)
        v = torch.as_tensor(valid.reshape(-1, self.history), device=self.device)
        s = None if sigma is None else torch.as_tensor(sigma.reshape(-1, self.history, 3), device=self.device)
        prediction = self.model(x, v, dt=dt, sigma=s)
        if self.records is not None:
            if self.mode == 'point':
                pm = predictive_mean(prediction)[:, :, None]
                fs = self.fixed_scale
                ps = fs[None, :, None].expand_as(pm) if fs.ndim == 2 else fs[None, None, None].expand_as(pm)
                record = {'weights': torch.ones(*pm.shape[:2], 1, device=self.device),
                          'mean': pm, 'scale': ps}
            else:
                record = {key: prediction[key]*(self.temperature if key == 'scale' else 1.)
                          for key in ('weights', 'mean', 'scale')}
            self.records.append({key: value.detach().cpu().numpy() for key, value in record.items()})
        mean, scale, weight = prediction['mean'][:, 0], prediction['scale'][:, 0], prediction['weights'][:, 0]
        if self.mode == 'point':
            mean = predictive_mean(prediction)[:, :1]
            spread = self.fixed_scale[0] if self.fixed_scale.ndim == 2 else self.fixed_scale
            scale = spread[None, None, :].expand_as(mean)
            weight = torch.ones(len(x), 1, device=self.device)
        else:
            scale = scale*self.temperature
        samples = mean[:, :, None, :]+scale[:, :, None, :]*self.nodes[None, None]
        relative = samples-torch.tensor([0., 0., 10.], device=self.device)
        uv = relative[..., :2]/relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ij = ((uv+1)*16).floor().long().clamp(0, 31)
        index = ij[..., 1]*32+ij[..., 0]
        masses = weight[:, :, None].expand_as(index)/len(self.nodes)
        probabilities = torch.zeros(len(x), 1024, device=self.device)
        probabilities.scatter_add_(1, index.flatten(1), masses.flatten(1))
        # Small, fixed spatial tie-breaker avoids selecting arbitrary distant code indices
        # when quadrature puts all mass in fewer directions than the scan budget.
        centre = (weight[..., None]*mean).sum(1)-torch.tensor([0., 0., 10.], device=self.device)
        centre_uv = centre[:, :2]/centre.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        distance = ((self.uv[None]-centre_uv[:, None])**2).sum(-1)
        tail = (-distance/.02**2).softmax(-1)
        probabilities = (probabilities+1e-6*tail+1e-15)
        probabilities = probabilities/probabilities.sum(-1, keepdim=True)
        return probabilities.log().reshape(*shape, 1024)
