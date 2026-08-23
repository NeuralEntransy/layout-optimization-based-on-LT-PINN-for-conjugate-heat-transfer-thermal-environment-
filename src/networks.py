# -*- coding: utf-8 -*-
"""
Networks for milestone 1 (pure multi-material conduction).

Per doc 04 section 8.2/8.4: temperature is represented by *separate per-domain
branches* (aerogel / aluminum wall / cavity air / three devices) instead of one
global network, so that the steep gradient jump across material interfaces
(k_Al/k_f ~ 6423) does not have to be represented by a single smooth function.

Each branch gets a fixed affine input normalization to O([-1,1]) so that the
thin aerogel strip (5 mm) and the device patches are well conditioned.
Derivatives obtained by autograd w.r.t. *physical* coordinates automatically
include the 1/scale chain factor because the transform is applied inside
forward().

Milestone 1 uses TemperatureField only. Stage B will add an independent
FlowNet (u, v, p) without touching the temperature branch structure.
"""
import torch
import torch.nn as nn

import config as C
import geometry as G

# fixed input affine transforms: z = (x - center) / scale
_INPUT_NORM = dict(
    tbl=((C.X_TBL / 2, 0.5), (C.L_TBL / 2, 0.5)),
    wall_l=((C.W_IN / 2, 0.5), (C.W_IN / 2, 0.5)),
    wall_r=((1 - C.W_IN / 2, 0.5), (C.W_IN / 2, 0.5)),
    wall_b=((0.5, C.W_IN / 2), ((1 - 2 * C.W_IN) / 2, C.W_IN / 2)),
    wall_t=((0.5, 1 - C.W_IN / 2),
            ((1 - 2 * C.W_IN) / 2, C.W_IN / 2)),
    air=((0.5, 0.5), ((1 - 2 * C.W_IN) / 2,
                      (1 - 2 * C.W_IN) / 2)),
    dev1=((0.5, C.W_IN + C.D1 / 2), (0.5, C.D1 / 2)),
    dev2=((0.5, 1 - C.W_IN - C.D2 / 2), (0.5, C.D2 / 2)),
    dev3=(C.C3, (0.5, C.R3)),
)

# Per-domain output scales: theta = out_scale * net(z).  The stagnant-air
# solution reaches theta ~ 3 around the un-bridged disk device (20 W now);
# an O(1) tanh output would need large output weights to get there (slow).
# Scales are rough FEM-informed magnitudes and only act linearly.
_OUTPUT_SCALE = dict(tbl=2.0, wall_l=2.0, wall_r=2.0, wall_b=2.0,
                     wall_t=2.0, air=3.0, dev1=2.0, dev2=2.5, dev3=4.0)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, depth, fourier_sigma=0.0,
                 fourier_dim=64):
        super().__init__()
        self.fourier_dim = 0
        if fourier_sigma > 0:
            # random Fourier features (Tancik et al. 2020): accelerate the
            # learning of steep gradients (air gap around the disk device
            # reaches dtheta/dr ~ 40 in theta units)
            B = torch.randn(n_in, fourier_dim) * fourier_sigma
            self.register_buffer("fourier_B", B)
            self.fourier_dim = fourier_dim
            n_in = n_in + 2 * fourier_dim
        layers = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.fourier_dim:
            proj = 2 * torch.pi * x @ self.fourier_B
            x = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)


# default per-domain Fourier sigma (0 = disabled).  With the v4 parameters
# (disk device 20 W) the field is smooth enough for plain tanh MLPs; the
# --fourier-sigma CLI flag can re-enable them if steep gradients reappear.
_FOURIER_SIGMA = dict(tbl=0.0, wall_l=0.0, wall_r=0.0, wall_b=0.0,
                      wall_t=0.0, air=0.0, dev1=0.0, dev2=0.0, dev3=0.0)


class TemperatureField(nn.Module):
    """theta(x) per domain; physical coords in, normalized theta out."""

    DOMAINS = G.DOMAINS

    def __init__(self, width=96, depth=5, theta_init=None,
                 fourier_sigma=None, fourier_dim=64):
        super().__init__()
        sig = dict(_FOURIER_SIGMA)
        if fourier_sigma is not None:           # global override (CLI)
            sig = {d: fourier_sigma for d in self.DOMAINS}
        self.nets = nn.ModuleDict(
            {d: MLP(2, 1, width, depth, sig[d], fourier_dim)
             for d in self.DOMAINS})
        for d in self.DOMAINS:
            c, s = _INPUT_NORM[d]
            self.register_buffer(f"center_{d}", torch.tensor(c).float())
            self.register_buffer(f"scale_{d}", torch.tensor(s).float())
            self.register_buffer(f"outscale_{d}",
                                 torch.tensor(float(_OUTPUT_SCALE[d])))
        if theta_init is not None and theta_init > 0:
            # warm start: uniform initial temperature level theta_init
            for d in self.DOMAINS:
                out = self.nets[d].net[-1]
                nn.init.zeros_(out.weight)
                nn.init.constant_(out.bias,
                                  theta_init / _OUTPUT_SCALE[d])

    def forward(self, domain, pts):
        z = (pts - getattr(self, f"center_{domain}")) \
            / getattr(self, f"scale_{domain}")
        out = getattr(self, f"outscale_{domain}") * self.nets[domain](z)
        return out

    def temperature_K(self, domain, pts):
        import config as C
        return C.T_C + C.DT * self.forward(domain, pts)
