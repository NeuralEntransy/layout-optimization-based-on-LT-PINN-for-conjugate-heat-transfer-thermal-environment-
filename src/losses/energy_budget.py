# -*- coding: utf-8 -*-
"""
Per-device and global steady heat-budget losses (integral constraints).

At steady state, each device must export exactly its own power through its
surface, and the whole domain must export P_tot through the outer boundary.
These integral constraints are implied by the pointwise PDE + interface
losses at convergence, but during training they are NOT redundant: they
provide a direct O(1) driving signal for the field amplitude (the pointwise
flux-continuity residual carries the low-conductivity side with a k_F/k_Al
~ 1/6400 coefficient, which stalls the air-side learning).

Residuals are normalized by the corresponding power, so they are O(1)
when violated and ~0 at convergence:
    r = (Q_out - P_k * power_scale) / (P_k * power_scale)

Both the device-side and the receiving-side fluxes are constrained, so the
weak (air / receiving) branch is driven directly.
"""
import torch

import config as C
import geometry as G
from .conduction import grad

P_OF_DEV = dict(dev1=C.P1, dev2=C.P2, dev3=C.P3)


def _n_out_tensor(meta, dirs, device):
    if meta["n_out"] == "radial":
        return dirs                      # sampler dirs already point outward
    v = torch.tensor(meta["n_out"], dtype=torch.float32, device=device)
    return v.expand(dirs.shape[0], 2)


class EnergyBudgetLoss(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field
        self.power_scale = 1.0

    def _face_flux(self, dom, pts, n_out):
        """Mean outward heat flux [W/m^2] across a face, using dom's net."""
        pts.requires_grad_(True)
        th = self.field(dom, pts)
        g = grad(th, pts)
        q_out = -C.K_OF_DOMAIN[dom] * C.DT / C.L_REF \
            * torch.sum(g * n_out, dim=1, keepdim=True)
        return q_out.mean()

    def forward(self, iface_samples, bnd_samples):
        details = {}
        total = 0.0
        ps = self.power_scale
        for dev, ifaces in G.DEV_IFACES.items():
            P = P_OF_DEV[dev] * ps
            for side in ("recv", "dev"):
                Q = 0.0
                for name in ifaces:
                    s = iface_samples[name]
                    a, b = s["a"], s["b"]
                    recv = b if a == dev else a
                    dom = recv if side == "recv" else dev
                    n_out = _n_out_tensor(G.IFACE_META[name], s["dirs"],
                                          s["pts"].device)
                    qm = self._face_flux(dom, s["pts"], n_out)
                    Q = Q + qm * G.IFACE_META[name]["length"] * C.B
                r = (Q - P) / P
                details[f"eng_{dev}_{side}"] = r.detach()
                total = total + r ** 2

        # global balance: outer boundary outflow = total power
        ptsL = bnd_samples["left"]["pts"]; ptsL.requires_grad_(True)
        thL = self.field("wall_l", ptsL)
        gL = grad(thL, ptsL)[:, 0:1]
        Q_left = C.K_AL * C.DT / C.L_REF * gL.mean() * 1.0 * C.B
        ptsR = bnd_samples["right"]["pts"]; ptsR.requires_grad_(True)
        thR = self.field("wall_r", ptsR)
        gR = grad(thR, ptsR)[:, 0:1]
        Q_right = -C.K_AL * C.DT / C.L_REF * gR.mean() * 1.0 * C.B
        P_tot = C.P_TOT * ps
        r_g = (Q_left + Q_right - P_tot) / P_tot
        details["eng_global"] = r_g.detach()
        total = total + r_g ** 2
        return total, details
