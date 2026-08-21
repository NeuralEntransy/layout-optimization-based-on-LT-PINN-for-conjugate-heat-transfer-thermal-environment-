# -*- coding: utf-8 -*-
"""
Geometry for the closed-cavity thermal layout benchmark (doc 04, section 1,
v4 parameters: 5 mm aerogel, 1 cm wall frame, cavity [0.01, 0.99]^2,
disk device 20 W).

Domains (physical coordinates [m]):
    tbl    : aerogel layer  [-0.005, 0] x [0, 1]  (only when not USE_TBL_1D)
    wall_* : aluminum frame split into four strips:
             wall_l [0, 0.01]x[0, 1]        wall_r [0.99, 1]x[0, 1]
             wall_b [0.01, 0.99]x[0, 0.01]  wall_t [0.01, 0.99]x[0.99, 1]
    air    : cavity         [0.01, 0.99]^2 minus the three devices
    dev1   : square  [x1-0.1, x1+0.1] x [0.01, 0.21]     (bottom wall mount)
    dev2   : square  [x2-0.175, x2+0.175] x [0.64, 0.99] (top wall mount)
    dev3   : disk    center (0.5, 0.4), R = 0.1          (fixed)

The wall frame is split because a single frame network puts the thin strips
at the edge of its normalized input range and systematically under-predicts
the wall temperature near the device mounts (~25 K mismatch).  Four strips
are tensor-product domains and are individually well conditioned; the strips
are re-joined thermally by the four corner interface losses
(corner_l_b / corner_l_t / corner_r_b / corner_r_t).

Provides:
  * DesignVars     - x1/x2 as sigmoid-parameterized learnable variables
                     (frozen in milestone 1, released in stage A)
  * domain samplers (uniform, per epoch, device-resident)
  * interface samplers returning points + fixed global direction e_xi
    (flux continuity: k_a dtheta_a/dxi = k_b dtheta_b/dxi)
  * outer-boundary samplers (left Robin via 1-D aerogel resistance /
    right Dirichlet baseline (or Robin variant) / top+bottom adiabatic)
"""
import math
import torch
import torch.nn as nn

import config as C

WALL_STRIPS = {
    "wall_l": (0.0, C.W_IN, 0.0, 1.0),
    "wall_r": (1 - C.W_IN, 1.0, 0.0, 1.0),
    "wall_b": (C.W_IN, 1 - C.W_IN, 0.0, C.W_IN),
    "wall_t": (C.W_IN, 1 - C.W_IN, 1 - C.W_IN, 1.0),
}

DOMAINS = ["wall_l", "wall_r", "wall_b", "wall_t", "air",
           "dev1", "dev2", "dev3"]
if not C.USE_TBL_1D:
    # resolved aerogel layer variant (kept for comparison)
    DOMAINS = ["tbl"] + DOMAINS

# canonical ordering of the region labels returned by label_points
LABEL_ORDER = ["tbl", "wall_l", "wall_r", "wall_b", "wall_t",
               "air", "dev1", "dev2", "dev3"]

# (name, domain_a, domain_b) sharing one geometric line
_INTERFACES_ALL = [
    ("tbl_wall",        "tbl",    "wall_l"),
    ("wall_air_left",   "wall_l", "air"),
    ("wall_air_right",  "wall_r", "air"),
    ("wall_air_bottom", "wall_b", "air"),
    ("wall_air_top",    "wall_t", "air"),
    ("dev1_wall",       "dev1",   "wall_b"),
    ("dev1_air_left",   "dev1",   "air"),
    ("dev1_air_right",  "dev1",   "air"),
    ("dev1_air_top",    "dev1",   "air"),
    ("dev2_wall",       "dev2",   "wall_t"),
    ("dev2_air_left",   "dev2",   "air"),
    ("dev2_air_right",  "dev2",   "air"),
    ("dev2_air_bottom", "dev2",   "air"),
    ("dev3_air",        "dev3",   "air"),
    # frame corner joints: the four wall strips share edges at the corners
    # (x=0.01/0.99, y in the strip thickness) and must be thermally
    # continuous so heat can flow around the frame
    ("corner_l_b",      "wall_l", "wall_b"),
    ("corner_l_t",      "wall_l", "wall_t"),
    ("corner_r_b",      "wall_r", "wall_b"),
    ("corner_r_t",      "wall_r", "wall_t"),
]
INTERFACES = [t for t in _INTERFACES_ALL
              if not (C.USE_TBL_1D and t[0] == "tbl_wall")]

# device-adjacent interface metadata for the per-device energy-budget loss:
#   n_out  = outward unit normal of the DEVICE on that face ('radial' for the
#            disk, equals the sampler direction),
#   length = face length [m] (extrusion b = 1 m).
IFACE_META = {
    "dev1_wall":       dict(n_out=(0.0, -1.0), length=C.D1),
    "dev1_air_left":   dict(n_out=(-1.0, 0.0), length=C.D1),
    "dev1_air_right":  dict(n_out=(1.0, 0.0), length=C.D1),
    "dev1_air_top":    dict(n_out=(0.0, 1.0), length=C.D1),
    "dev2_wall":       dict(n_out=(0.0, 1.0), length=C.D2),
    "dev2_air_left":   dict(n_out=(-1.0, 0.0), length=C.D2),
    "dev2_air_right":  dict(n_out=(1.0, 0.0), length=C.D2),
    "dev2_air_bottom": dict(n_out=(0.0, -1.0), length=C.D2),
    "dev3_air":        dict(n_out="radial", length=2 * math.pi * C.R3),
}
DEV_IFACES = {
    "dev1": ["dev1_wall", "dev1_air_left", "dev1_air_right",
             "dev1_air_top"],
    "dev2": ["dev2_wall", "dev2_air_left", "dev2_air_right",
             "dev2_air_bottom"],
    "dev3": ["dev3_air"],
}


# --------------------------------------------------------------------------- #
# design variables: x1 in X1_RANGE, x2 in X2_RANGE via sigmoid
# --------------------------------------------------------------------------- #
class DesignVars(nn.Module):
    def __init__(self, x1_0, x2_0, trainable=False, device="cpu"):
        super().__init__()
        self.lo1, self.hi1 = C.X1_RANGE
        self.lo2, self.hi2 = C.X2_RANGE

        def logit(u):
            return math.log(u / (1.0 - u))

        u1 = (x1_0 - self.lo1) / (self.hi1 - self.lo1)
        u2 = (x2_0 - self.lo2) / (self.hi2 - self.lo2)
        self.z1 = nn.Parameter(torch.tensor(logit(u1), device=device),
                               requires_grad=trainable)
        self.z2 = nn.Parameter(torch.tensor(logit(u2), device=device),
                               requires_grad=trainable)

    def x1(self):
        return self.lo1 + (self.hi1 - self.lo1) * torch.sigmoid(self.z1)

    def x2(self):
        return self.lo2 + (self.hi2 - self.lo2) * torch.sigmoid(self.z2)

    def values(self):
        return float(self.x1()), float(self.x2())


# --------------------------------------------------------------------------- #
# domain membership masks (also used by validate.py for the FEM grid)
# --------------------------------------------------------------------------- #
def mask_tbl(p):
    return p[:, 0] < 0.0


def mask_strip(p, strip):
    x0, x1_, y0, y1 = WALL_STRIPS[strip]
    return (p[:, 0] >= x0) & (p[:, 0] <= x1_) & (p[:, 1] >= y0) \
        & (p[:, 1] <= y1)


def mask_dev1(p, x1):
    return ((p[:, 0] - x1).abs() <= C.D1 / 2) & \
           (p[:, 1] >= C.DEV1_Y[0]) & (p[:, 1] <= C.DEV1_Y[1])


def mask_dev2(p, x2):
    return ((p[:, 0] - x2).abs() <= C.D2 / 2) & \
           (p[:, 1] >= C.DEV2_Y[0]) & (p[:, 1] <= C.DEV2_Y[1])


def mask_dev3(p):
    return (p[:, 0] - C.C3[0]) ** 2 + (p[:, 1] - C.C3[1]) ** 2 <= C.R3 ** 2


def label_points(p, x1, x2):
    """Assign each (n,2) point to exactly one region of LABEL_ORDER
    (priority: devices > tbl > wall strips > air).  With USE_TBL_1D the
    aerogel points (x<0) are labelled -1 (handled analytically)."""
    idx_of = {d: i for i, d in enumerate(LABEL_ORDER)}
    lab = torch.full((p.shape[0],), -1, dtype=torch.long, device=p.device)
    lab[...] = idx_of["air"]
    for strip in WALL_STRIPS:
        lab[mask_strip(p, strip)] = idx_of[strip]
    lab[mask_tbl(p)] = idx_of["tbl"]
    lab[mask_dev3(p)] = idx_of["dev3"]
    lab[mask_dev2(p, x2)] = idx_of["dev2"]
    lab[mask_dev1(p, x1)] = idx_of["dev1"]
    if C.USE_TBL_1D:
        lab[mask_tbl(p)] = -1
    return lab


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _box(n, xmin, xmax, ymin, ymax, device):
    p = torch.rand(n, 2, device=device)
    p[:, 0] = p[:, 0] * (xmax - xmin) + xmin
    p[:, 1] = p[:, 1] * (ymax - ymin) + ymin
    return p


def _union_segment(n, segs, y, device):
    """Sample n points on y=const from a union of x-intervals `segs`."""
    lengths = torch.tensor([b - a for a, b in segs], device=device)
    cum = torch.cumsum(lengths, 0)
    u = torch.rand(n, device=device) * cum[-1]
    x = torch.empty(n, device=device)
    for i, (a, b) in enumerate(segs):
        lo = cum[i] - lengths[i]
        m = (u >= lo) & (u < cum[i]) if i < len(segs) - 1 else (u >= lo)
        x[m] = a + (u[m] - lo)
    return torch.stack([x, torch.full_like(x, y)], dim=1)


# --------------------------------------------------------------------------- #
# domain samplers
# --------------------------------------------------------------------------- #
def sample_domain(name, n, x1, x2, device):
    x1f, x2f = float(x1), float(x2)
    if name == "tbl":
        return _box(n, C.X_TBL, 0.0, 0.0, 1.0, device)
    if name in WALL_STRIPS:
        x0, x1_, y0, y1 = WALL_STRIPS[name]
        return _box(n, x0, x1_, y0, y1, device)
    if name == "air":
        pts, need, tries = [], n, 0
        while need > 0 and tries < 60:
            cand = _box(max(2 * need, 128), C.W_IN, 1 - C.W_IN,
                        C.W_IN, 1 - C.W_IN, device)
            keep = ~(mask_dev1(cand, x1f) | mask_dev2(cand, x2f)
                     | mask_dev3(cand))
            sel = cand[keep][:need]
            pts.append(sel)
            need -= sel.shape[0]
            tries += 1
        out = torch.cat(pts, 0)
        if out.shape[0] < n:
            rep = (n // out.shape[0]) + 1
            out = out.repeat(rep, 1)[:n]
        return out
    if name == "dev1":
        return _box(n, x1f - C.D1 / 2, x1f + C.D1 / 2, C.DEV1_Y[0],
                    C.DEV1_Y[1], device)
    if name == "dev2":
        return _box(n, x2f - C.D2 / 2, x2f + C.D2 / 2, C.DEV2_Y[0],
                    C.DEV2_Y[1], device)
    if name == "dev3":
        u = torch.rand(n, device=device)
        v = torch.rand(n, device=device)
        r = C.R3 * torch.sqrt(u)
        t = 2 * math.pi * v
        return torch.stack([C.C3[0] + r * torch.cos(t),
                            C.C3[1] + r * torch.sin(t)], dim=1)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# interface samplers -> (pts, dirs) with unit direction e_xi
# --------------------------------------------------------------------------- #
def _v(n, x, y, device):
    p = torch.rand(n, 2, device=device)
    p[:, 0] = x
    p[:, 1] = p[:, 1] * (y[1] - y[0]) + y[0]
    return p


def sample_interface(name, n, x1, x2, device):
    x1f, x2f = float(x1), float(x2)
    wi = C.W_IN
    if name == "tbl_wall":
        p = _v(n, 0.0, (0.0, 1.0), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "wall_air_left":
        p = _v(n, wi, (wi, 1 - wi), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "wall_air_right":
        p = _v(n, 1 - wi, (wi, 1 - wi), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "wall_air_bottom":
        p = _union_segment(n, [(wi, x1f - C.D1 / 2), (x1f + C.D1 / 2, 1 - wi)],
                           wi, device)
        d = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "wall_air_top":
        p = _union_segment(n, [(wi, x2f - C.D2 / 2), (x2f + C.D2 / 2, 1 - wi)],
                           1 - wi, device)
        d = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev1_wall":
        p = _box(n, x1f - C.D1 / 2, x1f + C.D1 / 2, wi, wi, device)
        d = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev1_air_left":
        p = _v(n, x1f - C.D1 / 2, C.DEV1_Y, device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev1_air_right":
        p = _v(n, x1f + C.D1 / 2, C.DEV1_Y, device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev1_air_top":
        p = _box(n, x1f - C.D1 / 2, x1f + C.D1 / 2, C.DEV1_Y[1],
                 C.DEV1_Y[1], device)
        d = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev2_wall":
        p = _box(n, x2f - C.D2 / 2, x2f + C.D2 / 2, 1 - wi, 1 - wi, device)
        d = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev2_air_left":
        p = _v(n, x2f - C.D2 / 2, C.DEV2_Y, device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev2_air_right":
        p = _v(n, x2f + C.D2 / 2, C.DEV2_Y, device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev2_air_bottom":
        p = _box(n, x2f - C.D2 / 2, x2f + C.D2 / 2, C.DEV2_Y[0],
                 C.DEV2_Y[0], device)
        d = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev3_air":
        t = 2 * math.pi * torch.rand(n, device=device)
        p = torch.stack([C.C3[0] + C.R3 * torch.cos(t),
                         C.C3[1] + C.R3 * torch.sin(t)], dim=1)
        d = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    elif name == "corner_l_b":
        p = _v(n, wi, (0.0, wi), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "corner_l_t":
        p = _v(n, wi, (1 - wi, 1.0), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "corner_r_b":
        p = _v(n, 1 - wi, (0.0, wi), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "corner_r_t":
        p = _v(n, 1 - wi, (1 - wi, 1.0), device)
        d = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    else:
        raise KeyError(name)
    return p, d


def sample_all_interfaces(n, x1, x2, device):
    out = {}
    for name, a, b in INTERFACES:
        p, d = sample_interface(name, n, x1, x2, device)
        out[name] = dict(pts=p, dirs=d, a=a, b=b)
    return out


# --------------------------------------------------------------------------- #
# outer boundary samplers
# --------------------------------------------------------------------------- #
def sample_boundaries(n, device):
    bnd = {}
    if C.USE_TBL_1D:
        # wall_l left face: Robin to the cold reservoir through the 1-D
        # aerogel resistance:  k_Al dtheta/dx = h_TBL * theta  at x = 0
        bnd["left"] = dict(pts=_v(n, 0.0, (0.0, 1.0), device), dom="wall_l")
    else:
        bnd["left"] = dict(pts=_v(n, C.X_TBL, (0.0, 1.0), device), dom="tbl")
    bnd["right"] = dict(pts=_v(n, 1.0, (0.0, 1.0), device), dom="wall_r")
    if not C.USE_TBL_1D:
        for side, y in [("top", 1.0), ("bottom", 0.0)]:
            bnd[f"{side}_tbl"] = dict(
                pts=_box(n, C.X_TBL, 0.0, y, y, device), dom="tbl",
                dirs=torch.tensor([0.0, 1.0], device=device).expand(n, 2))
    # the y=0 / y=1 outer edges span wall_l / wall_t|b / wall_r strips
    for side, y in [("top", 1.0), ("bottom", 0.0)]:
        mid = "wall_t" if side == "top" else "wall_b"
        for strip, (x0, x1_) in [("wall_l", (0.0, C.W_IN)),
                                 (mid, (C.W_IN, 1 - C.W_IN)),
                                 ("wall_r", (1 - C.W_IN, 1.0))]:
            bnd[f"{side}_{strip}"] = dict(
                pts=_box(n, x0, x1_, y, y, device), dom=strip,
                dirs=torch.tensor([0.0, 1.0], device=device).expand(n, 2))
    return bnd
