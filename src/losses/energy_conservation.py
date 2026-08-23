# -*- coding: utf-8 -*-
"""Deterministic integral energy constraints for Milestone 1.

The pointwise PDE, interface and boundary losses imply these balances only
at the exact solution. During optimization the integral constraints provide
direct low-frequency signals for:

* each powered device (device and receiving side);
* the source-free air domain;
* every source-free wall strip (and the explicit aerogel, when enabled);
* the complete four-sided outer-boundary balance;
* zero integrated leakage on every top/bottom boundary segment.

All residuals are normalized by the current total/device power. The samples
used here are deterministic midpoint quadrature points generated separately
from the randomly resampled pointwise collocation sets.
"""
import math

import torch

import config as C
import geometry as G
from .conduction import grad


P_OF_DEV = dict(dev1=C.P1, dev2=C.P2, dev3=C.P3)


# Outward normals of the AIR domain. On device contacts these point into the
# excluded solid and are therefore the negatives of the device outward normal.
AIR_IFACE_META = {
    "wall_air_left": ((-1.0, 0.0), 1.0 - 2.0 * C.W_IN),
    "wall_air_right": ((1.0, 0.0), 1.0 - 2.0 * C.W_IN),
    "wall_air_bottom": ((0.0, -1.0),
                        1.0 - 2.0 * C.W_IN - C.D1),
    "wall_air_top": ((0.0, 1.0),
                     1.0 - 2.0 * C.W_IN - C.D2),
    "dev1_air_left": ((1.0, 0.0), C.D1),
    "dev1_air_right": ((-1.0, 0.0), C.D1),
    "dev1_air_top": ((0.0, -1.0), C.D1),
    "dev2_air_left": ((1.0, 0.0), C.D2),
    "dev2_air_right": ((-1.0, 0.0), C.D2),
    "dev2_air_bottom": ((0.0, 1.0), C.D2),
    "dev3_air": ("inward_radial", 2.0 * math.pi * C.R3),
}


def _passive_balance_meta():
    """Boundary partition for each source-free solid domain.

    Each term contains sample group, sample name, outward normal and length.
    The four wall strips are independent networks, so their corner interfaces
    must be included explicitly to close each strip's energy account.
    """
    wall_l_left = (
        ("bnd", "left", (-1.0, 0.0), 1.0)
        if C.USE_TBL_1D
        else ("iface", "tbl_wall", (-1.0, 0.0), 1.0)
    )
    meta = {
        "wall_l": [
            wall_l_left,
            ("iface", "wall_air_left", (1.0, 0.0),
             1.0 - 2.0 * C.W_IN),
            ("iface", "corner_l_b", (1.0, 0.0), C.W_IN),
            ("iface", "corner_l_t", (1.0, 0.0), C.W_IN),
            ("bnd", "top_wall_l", (0.0, 1.0), C.W_IN),
            ("bnd", "bottom_wall_l", (0.0, -1.0), C.W_IN),
        ],
        "wall_r": [
            ("bnd", "right", (1.0, 0.0), 1.0),
            ("iface", "wall_air_right", (-1.0, 0.0),
             1.0 - 2.0 * C.W_IN),
            ("iface", "corner_r_b", (-1.0, 0.0), C.W_IN),
            ("iface", "corner_r_t", (-1.0, 0.0), C.W_IN),
            ("bnd", "top_wall_r", (0.0, 1.0), C.W_IN),
            ("bnd", "bottom_wall_r", (0.0, -1.0), C.W_IN),
        ],
        "wall_b": [
            ("bnd", "bottom_wall_b", (0.0, -1.0),
             1.0 - 2.0 * C.W_IN),
            ("iface", "wall_air_bottom", (0.0, 1.0),
             1.0 - 2.0 * C.W_IN - C.D1),
            ("iface", "dev1_wall", (0.0, 1.0), C.D1),
            ("iface", "corner_l_b", (-1.0, 0.0), C.W_IN),
            ("iface", "corner_r_b", (1.0, 0.0), C.W_IN),
        ],
        "wall_t": [
            ("bnd", "top_wall_t", (0.0, 1.0),
             1.0 - 2.0 * C.W_IN),
            ("iface", "wall_air_top", (0.0, -1.0),
             1.0 - 2.0 * C.W_IN - C.D2),
            ("iface", "dev2_wall", (0.0, -1.0), C.D2),
            ("iface", "corner_l_t", (-1.0, 0.0), C.W_IN),
            ("iface", "corner_r_t", (1.0, 0.0), C.W_IN),
        ],
    }
    if not C.USE_TBL_1D:
        meta["tbl"] = [
            ("bnd", "left", (-1.0, 0.0), 1.0),
            ("iface", "tbl_wall", (1.0, 0.0), 1.0),
            ("bnd", "top_tbl", (0.0, 1.0), C.L_TBL),
            ("bnd", "bottom_tbl", (0.0, -1.0), C.L_TBL),
        ]
    return meta


def _outer_horizontal_segments(side):
    """Return sample name, normal and length for one horizontal edge."""
    normal = (0.0, 1.0) if side == "top" else (0.0, -1.0)
    middle = "wall_t" if side == "top" else "wall_b"
    segments = [
        (f"{side}_wall_l", normal, C.W_IN),
        (f"{side}_{middle}", normal, 1.0 - 2.0 * C.W_IN),
        (f"{side}_wall_r", normal, C.W_IN),
    ]
    if not C.USE_TBL_1D:
        segments.append((f"{side}_tbl", normal, C.L_TBL))
    return segments


def _normal_tensor(normal, sample):
    """Expand a fixed normal or obtain a radial normal from the sampler."""
    if normal == "radial":
        return sample["dirs"]
    if normal == "inward_radial":
        return -sample["dirs"]
    value = torch.tensor(normal, dtype=sample["pts"].dtype,
                         device=sample["pts"].device)
    return value.expand(sample["pts"].shape[0], 2)


class EnergyConservationLoss(torch.nn.Module):
    """Integral energy budgets with independently tunable component weights."""

    def __init__(self, field, w_device=1.0, w_air=1.0, w_wall=1.0,
                 w_global=1.0, w_adiabatic=1.0):
        super().__init__()
        self.field = field
        self.power_scale = 1.0
        self.w_device = w_device
        self.w_air = w_air
        self.w_wall = w_wall
        self.w_global = w_global
        self.w_adiabatic = w_adiabatic

    def _face_flux(self, dom, pts, n_out):
        """Mean outward heat flux density [W/m^2] using dom's branch."""
        pts.requires_grad_(True)
        theta = self.field(dom, pts)
        gradient = grad(theta, pts)
        return (-C.K_OF_DOMAIN[dom] * C.DT / C.L_REF
                * torch.sum(gradient * n_out, dim=1, keepdim=True)).mean()

    def _integral(self, dom, sample, normal, length):
        n_out = _normal_tensor(normal, sample)
        return self._face_flux(dom, sample["pts"], n_out) * length * C.B

    def _passive_balance(self, dom, terms, iface_samples, bnd_samples):
        heat_rate = 0.0
        for group, name, normal, length in terms:
            samples = iface_samples if group == "iface" else bnd_samples
            heat_rate = heat_rate + self._integral(
                dom, samples[name], normal, length)
        return heat_rate

    def forward(self, iface_samples, bnd_samples):
        details = {}
        ps = self.power_scale
        if ps <= 0.0:
            raise ValueError("power_scale must be positive for normalized "
                             "energy constraints")
        p_total = C.P_TOT * ps
        loss_device = 0.0

        # Each device must export its power. The receiving-side heat rate is
        # projected along the device outward normal, so it represents heat
        # accepted from the device rather than outward flux of the receiver.
        for dev, iface_names in G.DEV_IFACES.items():
            target = P_OF_DEV[dev] * ps
            for side in ("dev", "recv"):
                heat_rate = 0.0
                for name in iface_names:
                    sample = iface_samples[name]
                    recv = sample["b"] if sample["a"] == dev else sample["a"]
                    dom = dev if side == "dev" else recv
                    meta = G.IFACE_META[name]
                    heat_rate = heat_rate + self._integral(
                        dom, sample, meta["n_out"], meta["length"])
                residual = (heat_rate - target) / target
                details[f"eng_{dev}_{side}"] = residual.detach()
                details[f"eng_Q_{dev}_{side}_W"] = heat_rate.detach()
                loss_device = loss_device + residual ** 2

        # Air is source-free: all of its boundary outflows must sum to zero.
        q_air = 0.0
        for name, (normal, length) in AIR_IFACE_META.items():
            q_air = q_air + self._integral(
                "air", iface_samples[name], normal, length)
        r_air = q_air / p_total
        details["eng_air"] = r_air.detach()
        details["eng_Q_air_W"] = q_air.detach()

        # Close the transport chain inside every independently represented,
        # source-free wall strip (and the explicit aerogel when present).
        loss_wall = 0.0
        for dom, terms in _passive_balance_meta().items():
            q_dom = self._passive_balance(
                dom, terms, iface_samples, bnd_samples)
            residual = q_dom / p_total
            details[f"eng_{dom}"] = residual.detach()
            details[f"eng_Q_{dom}_W"] = q_dom.detach()
            loss_wall = loss_wall + residual ** 2

        # Signed heat rates leaving all four sides of the complete domain.
        left_dom = "wall_l" if C.USE_TBL_1D else "tbl"
        q_left = self._integral(
            left_dom, bnd_samples["left"], (-1.0, 0.0), 1.0)
        q_right = self._integral(
            "wall_r", bnd_samples["right"], (1.0, 0.0), 1.0)

        q_horizontal = {}
        loss_adiabatic = 0.0
        for side in ("top", "bottom"):
            q_side = 0.0
            for name, normal, length in _outer_horizontal_segments(side):
                dom = bnd_samples[name]["dom"]
                q_segment = self._integral(
                    dom, bnd_samples[name], normal, length)
                q_side = q_side + q_segment
                segment_residual = q_segment / p_total
                details[f"eng_adiabatic_{name}"] = \
                    segment_residual.detach()
                details[f"eng_Q_{name}_W"] = q_segment.detach()
                loss_adiabatic = loss_adiabatic + segment_residual ** 2
            q_horizontal[side] = q_side
            details[f"eng_adiabatic_{side}"] = (q_side / p_total).detach()
            details[f"eng_Q_{side}_W"] = q_side.detach()

        q_outer = q_left + q_right + q_horizontal["top"] \
            + q_horizontal["bottom"]
        r_global = (q_outer - p_total) / p_total
        details["eng_global"] = r_global.detach()
        details["eng_Q_left_W"] = q_left.detach()
        details["eng_Q_right_W"] = q_right.detach()
        details["eng_Q_outer_W"] = q_outer.detach()

        total = (self.w_device * loss_device
                 + self.w_air * r_air ** 2
                 + self.w_wall * loss_wall
                 + self.w_global * r_global ** 2
                 + self.w_adiabatic * loss_adiabatic)
        return total, details
