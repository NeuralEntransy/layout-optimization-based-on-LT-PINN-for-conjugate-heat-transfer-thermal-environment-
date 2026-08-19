# -*- coding: utf-8 -*-
"""
Outer-boundary losses (doc 04, sections 2.3 and 4).

  left  : with USE_TBL_1D (default), the 5 cm aerogel is degenerated to its
          1-D resistance -> Robin on the wall_l branch at x = 0:
              k_Al dtheta/dx = h_TBL * theta     (h_TBL = k_TBL / l_TBL)
          imposed in physical-flux form normalized by Q_REF:
              r = (k_Al/L_ref * dtheta/dx - h_TBL * theta) * DT / Q_REF
          (the resolved-layer variant instead uses Dirichlet theta = 0
          at x = -0.05 on the tbl branch)
  right : Robin  -k_Al dtheta/dx = h_ext (theta - 1)  on the wall_r branch:
              r = (-k_Al/L_ref * dtheta/dx - h_ext * (theta - 1)) * DT / Q_REF
  top/bottom : adiabatic dtheta/dn = 0 on every sampled top_*/bottom_* strip:
              r = k_m * DT/L_ref * dtheta/dn / Q_REF
"""
import torch

import config as C
from .conduction import grad


class BoundaryLoss(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field

    def forward(self, bnd_samples):
        details = {}
        total = 0.0

        # left boundary
        s = bnd_samples["left"]
        if C.USE_TBL_1D:
            pts = s["pts"]
            pts.requires_grad_(True)
            th = self.field(s["dom"], pts)
            g = grad(th, pts)[:, 0:1]
            r = (C.K_AL / C.L_REF * g - C.H_TBL * th) * C.DT / C.Q_REF
            l = torch.mean(r ** 2)
            details["bc_left_robin_tbl1d"] = l
        else:
            th = self.field(s["dom"], s["pts"])
            l = torch.mean((th - C.THETA_COLD) ** 2)
            details["bc_left_dirichlet"] = l
        total = total + l

        # right Robin
        s = bnd_samples["right"]
        pts = s["pts"]
        pts.requires_grad_(True)
        th = self.field(s["dom"], pts)
        g = grad(th, pts)[:, 0:1]
        r = (-C.K_AL / C.L_REF * g
             - C.H_EXT * (th - C.THETA_INF)) * C.DT / C.Q_REF
        l = torch.mean(r ** 2)
        details["bc_right_robin"] = l
        total = total + l

        # top / bottom adiabatic on every sampled top_* / bottom_* strip
        for key, s in bnd_samples.items():
            if not (key.startswith("top_") or key.startswith("bottom_")):
                continue
            dom = s["dom"]
            pts = s["pts"]
            pts.requires_grad_(True)
            th = self.field(dom, pts)
            g_n = torch.sum(grad(th, pts) * s["dirs"], dim=1, keepdim=True)
            r = C.K_OF_DOMAIN[dom] * C.DT / C.L_REF * g_n / C.Q_REF
            l = torch.mean(r ** 2)
            details[f"bc_{key}_adiab"] = l
            total = total + l
        return total, details
