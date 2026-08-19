# -*- coding: utf-8 -*-
"""
Monitoring / acceptance metrics for milestone 1 (doc 04, section 7.2).

All quantities are computed from the PINN field with input-gradients enabled
but detached from the training graph.

  * boundary heat rates  Q_left / Q_right (+ Robin cross-check) / Q_top / Q_bottom
  * total energy balance vs P_TOT * power_scale
  * per-device T_max (dense evaluation)
  * mounting-interface continuity errors (dev1|wall, dev2|wall)
  * aerogel 1D-resistance equivalence (dT across layer vs q_left * R'')
"""
import numpy as np
import torch

import config as C
from losses.conduction import grad


def _line(x, y0, y1, n, device):
    p = torch.zeros(n, 2, device=device)
    p[:, 0] = x
    p[:, 1] = torch.linspace(y0, y1, n, device=device)
    return p


def _flux_line(field, dom, pts, comp, k, sign, device):
    """Integral of signed normal heat flux q_n = -sign * k dT/d(comp) over a
    unit-length line, times extrusion thickness b. Returns Q [W]."""
    pts = pts.clone().requires_grad_(True)
    th = field(dom, pts)
    g = grad(th, pts)[:, comp:comp + 1]
    q_n = -sign * k * C.DT / C.L_REF * g          # W/m^2
    return float(q_n.mean() * 1.0 * C.B), th.detach()


@torch.no_grad()
def device_Tmax(field, x1, x2, device, n=81):
    """Dense per-device max temperature [K]."""
    out = {}
    grids = {
        "dev1": _grid(x1 - C.D1 / 2, x1 + C.D1 / 2, C.DEV1_Y[0], C.DEV1_Y[1],
                      n, device),
        "dev2": _grid(x2 - C.D2 / 2, x2 + C.D2 / 2, C.DEV2_Y[0], C.DEV2_Y[1],
                      n, device),
        "dev3": _grid(C.C3[0] - C.R3, C.C3[0] + C.R3, C.C3[1] - C.R3,
                      C.C3[1] + C.R3, n, device),
    }
    import geometry as G
    for name, p in grids.items():
        if name == "dev3":
            p = p[G.mask_dev3(p)]
        th = field(name, p)
        out[name] = float((C.T_C + C.DT * th).max())
    return out


def _grid(x0, x1_, y0, y1, n, device):
    gx = torch.linspace(x0, x1_, n, device=device)
    gy = torch.linspace(y0, y1, n, device=device)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    return torch.stack([GX.ravel(), GY.ravel()], dim=1)


def boundary_fluxes(field, device, n=1025):
    """Heat rates [W] leaving the domain through each outer boundary."""
    if C.USE_TBL_1D:
        # heat leaving through the 1-D aerogel resistance = wall flux at x=0
        # (outward normal -x):  Q = integral k_Al dT/dx dy  (x increasing
        # into the wall); cross-checked by the Robin form h_TBL*(T - T_cold)
        Q_left, th0 = _flux_line(field, "wall_l", _line(0.0, 0, 1, n, device),
                                 0, C.K_AL, sign=-1.0, device=device)
        Q_left_robin = float(C.H_TBL * C.DT *
                             (th0 - C.THETA_COLD).mean() * 1.0 * C.B)
    else:
        Q_left, th0 = _flux_line(field, "tbl",
                                 _line(C.X_TBL, 0, 1, n, device),
                                 0, C.K_TBL, sign=-1.0, device=device)
        Q_left_robin = Q_left
    Q_right, th_r = _flux_line(field, "wall_r", _line(1.0, 0, 1, n, device),
                               0, C.K_AL, sign=+1.0, device=device)
    # top/bottom outer edges
    Q_top, _ = _flux_line(field, "wall_t", _line2(0, 1, 1.0, n, device),
                          1, C.K_AL, sign=+1.0, device=device)
    Q_bottom, _ = _flux_line(field, "wall_b", _line2(0, 1, 0.0, n, device),
                             1, C.K_AL, sign=-1.0, device=device)
    if not C.USE_TBL_1D:
        q_tbl_t, _ = _flux_line(field, "tbl",
                                _line2(C.X_TBL, 0, 1.0, 65, device),
                                1, C.K_TBL, sign=+1.0, device=device)
        Q_top += q_tbl_t * abs(C.X_TBL)   # per-length integral x strip width
        q_tbl_b, _ = _flux_line(field, "tbl",
                                _line2(C.X_TBL, 0, 0.0, 65, device),
                                1, C.K_TBL, sign=-1.0, device=device)
        Q_bottom += q_tbl_b * abs(C.X_TBL)
    Q_right_robin = float(C.H_EXT * C.DT *
                          (th_r - C.THETA_INF).mean() * 1.0 * C.B)
    return dict(Q_left=Q_left, Q_left_robin=Q_left_robin,
                Q_right=Q_right, Q_right_robin=Q_right_robin,
                Q_top=Q_top, Q_bottom=Q_bottom)


def _line2(x0, x1_, y, n, device):
    p = torch.zeros(n, 2, device=device)
    p[:, 0] = torch.linspace(x0, x1_, n, device=device)
    p[:, 1] = y
    return p


def energy_report(field, device, power_scale=1.0, n=1025):
    fl = boundary_fluxes(field, device, n)
    P = C.P_TOT * power_scale
    fl["P_in"] = P
    fl["balance_err"] = abs(P - fl["Q_left"] - fl["Q_right"]) / P
    fl["adiabatic_leak"] = abs(fl["Q_top"]) + abs(fl["Q_bottom"])
    return fl


def aerogel_check(field, device, n=401):
    """dT across the aerogel vs the 1D resistance prediction q'' * R''.
    With USE_TBL_1D the layer is a Robin BC: dT = theta(0)*DT and the 1D
    relation holds by construction; still cross-checked against the FEM
    explicit-layer reference in validate.py."""
    if C.USE_TBL_1D:
        pts = _line(0.0, 1e-4, 1 - 1e-4, n, device)
        with torch.no_grad():
            T_x0 = (C.T_C + C.DT * field("wall_l", pts)).mean()
        Q_left = boundary_fluxes(field, device, n=1025)["Q_left"]
        q_pp = Q_left / C.B / 1.0
        dT = float(T_x0 - C.T_COLD)
        return dict(dT=dT, dT_1D=q_pp * C.RPP_TBL, q_left_Wm2=q_pp,
                    Rpp=C.RPP_TBL,
                    rel_err=abs(dT - q_pp * C.RPP_TBL) / max(dT, 1e-12))
    pts = _line(0.0, 1e-4, 1 - 1e-4, n, device)
    with torch.no_grad():
        T_x0 = (C.T_C + C.DT * field("tbl", pts)).mean()
    Q_left = boundary_fluxes(field, device, n=1025)["Q_left"]
    q_pp = Q_left / C.B / 1.0
    Rpp = abs(C.X_TBL) / C.K_TBL
    dT = float(T_x0 - C.T_COLD)
    return dict(dT=dT, dT_1D=q_pp * Rpp, q_left_Wm2=q_pp, Rpp=Rpp,
                rel_err=abs(dT - q_pp * Rpp) / max(dT, 1e-12))


def mount_continuity(field, x1, x2, device, n=512):
    """RMS / max temperature and flux jumps on the two mounting interfaces."""
    out = {}
    for name, dom, strip, y, xa, xb in [
            ("dev1_wall", "dev1", "wall_b", C.DEV1_Y[0],
             x1 - C.D1 / 2, x1 + C.D1 / 2),
            ("dev2_wall", "dev2", "wall_t", C.DEV2_Y[1],
             x2 - C.D2 / 2, x2 + C.D2 / 2)]:
        pts = _line2(xa, xb, y, n, device).requires_grad_(True)
        th_s = field(dom, pts)
        th_w = field(strip, pts)
        g_s = grad(th_s, pts)[:, 1:2]
        g_w = grad(th_w, pts)[:, 1:2]
        r_T = (th_s - th_w).detach().cpu().numpy().ravel()
        r_q = ((C.K_OF_DOMAIN[dom] * g_s - C.K_OF_DOMAIN[strip] * g_w)
               * C.DT / C.L_REF).detach().cpu().numpy().ravel()  # W/m^2
        out[name] = dict(
            dT_rms=float(np.sqrt((r_T ** 2).mean()) * C.DT),
            dT_max=float(np.abs(r_T).max() * C.DT),
            dq_rms_Wm2=float(np.sqrt((r_q ** 2).mean())),
            dq_max_Wm2=float(np.abs(r_q).max()))
    return out


def full_report(field, x1, x2, device, power_scale=1.0):
    rep = dict(energy=energy_report(field, device, power_scale),
               T_max_K=device_Tmax(field, x1, x2, device),
               aerogel=aerogel_check(field, device),
               mount=mount_continuity(field, x1, x2, device))
    rep["T_max_C"] = {k: v - 273.15 for k, v in rep["T_max_K"].items()}
    rep["feasible_70C"] = all(v <= C.T_LIM for v in rep["T_max_K"].values())
    return rep
