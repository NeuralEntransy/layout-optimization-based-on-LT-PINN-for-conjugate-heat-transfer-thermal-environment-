# -*- coding: utf-8 -*-
"""
Per-domain steady conduction residuals (doc 04, sections 3.1 and 4).

Each domain equation is divided by its own conductivity:

    devices : nabla^2 theta + S_k * power_scale = 0
    others  : nabla^2 theta = 0

with S_k = q'''_k L_ref^2 / (k_Al * DT_ref)  (config.S_OF_DOMAIN).
"""
import torch

import config as C


def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


class ConductionLoss(torch.nn.Module):
    def __init__(self, field, power_scale=1.0, w_dev=1000.0):
        super().__init__()
        self.field = field
        self.power_scale = power_scale
        self.w_dev = w_dev          # extra weight on device Poisson residual

    def forward(self, samples):
        """samples: dict domain -> (n,2) tensor. Returns (total, details)."""
        details = {}
        total = 0.0
        for dom, pts in samples.items():
            pts.requires_grad_(True)
            th = self.field(dom, pts)
            g = grad(th, pts)
            th_xx = grad(g[:, 0:1], pts)[:, 0:1]
            th_yy = grad(g[:, 1:2], pts)[:, 1:2]
            res = th_xx + th_yy
            w = C.PDE_W_OF_DOMAIN[dom]
            if dom in C.S_OF_DOMAIN:
                res = res + C.S_OF_DOMAIN[dom] * self.power_scale
                w = w * self.w_dev
            res = res * C.PDE_RES_SCALE[dom]
            l = w * torch.mean(res ** 2)
            details[f"pde_{dom}"] = l
            total = total + l
        return total, details
