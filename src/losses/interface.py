# -*- coding: utf-8 -*-
"""
Material-interface continuity losses (doc 04, section 2.3 / 8.4).

For every internal interface:
  * temperature continuity   r_T = theta_a - theta_b
  * heat-flux continuity     r_q = (k_a dtheta_a/dxi - k_b dtheta_b/dxi)
                                   * DT_ref / (L_ref * Q_REF)
where xi is a fixed global direction (x / y / radial) supplied by the
interface sampler, so both sides are differentiated along the *same*
direction and no normal-orientation ambiguity arises.

Conductivity ratios (up to k_Al/k_f = 6423) appear only here, per the
normalization strategy of doc section 4. Interface losses are scaled
independently via the training weights (w_if_T, w_if_q).
"""
import torch

import config as C
from .conduction import grad


class InterfaceLoss(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field

    def forward(self, iface_samples):
        """iface_samples: name -> dict(pts, dirs, a, b)."""
        total_T, total_q = 0.0, 0.0
        details = {}
        for name, s in iface_samples.items():
            pts, dirs = s["pts"], s["dirs"]
            pts.requires_grad_(True)
            th_a = self.field(s["a"], pts)
            th_b = self.field(s["b"], pts)
            g_a = torch.sum(grad(th_a, pts) * dirs, dim=1, keepdim=True)
            g_b = torch.sum(grad(th_b, pts) * dirs, dim=1, keepdim=True)
            r_T = th_a - th_b
            r_q = (C.K_OF_DOMAIN[s["a"]] * g_a
                   - C.K_OF_DOMAIN[s["b"]] * g_b) * (C.DT / C.L_REF) / C.Q_IF
            l_T, l_q = torch.mean(r_T ** 2), torch.mean(r_q ** 2)
            details[f"ifT_{name}"] = l_T
            details[f"ifq_{name}"] = l_q
            total_T = total_T + l_T
            total_q = total_q + l_q
        return total_T, total_q, details
