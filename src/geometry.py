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
  * signed-distance functions (SDFs): F(x, y) < 0 inside, = 0 on boundary
  * domain definitions and point-membership masks derived from the SDFs
  * interface topology, device outward normals and interface lengths

Sampling implementations live in sampling.py so this module contains only
geometry shapes, topology and metadata.
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
# signed-distance geometry
# --------------------------------------------------------------------------- #
def sdf_box(p, bounds):
    """Exact SDF of an axis-aligned rectangle.

    ``bounds`` is ``(xmin, xmax, ymin, ymax)``.  The sign convention used by
    every SDF in this module is negative inside, zero on the boundary and
    positive outside.
    """
    xmin, xmax, ymin, ymax = bounds
    as_scalar = lambda value: torch.as_tensor(value, dtype=p.dtype,
                                               device=p.device)
    xmin, xmax, ymin, ymax = map(as_scalar, (xmin, xmax, ymin, ymax))
    center = torch.stack([(xmin + xmax) / 2, (ymin + ymax) / 2])
    half_size = torch.stack([(xmax - xmin) / 2, (ymax - ymin) / 2])
    q = torch.abs(p - center) - half_size
    outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=1)
    inside = torch.clamp(torch.amax(q, dim=1), max=0.0)
    return outside + inside


def sdf_circle(p, center, radius):
    """Exact SDF of a circle."""
    center_tensor = torch.as_tensor(center, dtype=p.dtype, device=p.device)
    return torch.linalg.vector_norm(p - center_tensor, dim=1) - radius


def sdf_tbl(p):
    return sdf_box(p, (C.X_TBL, 0.0, 0.0, 1.0))


def sdf_strip(p, strip):
    return sdf_box(p, WALL_STRIPS[strip])


def sdf_dev1(p, x1):
    bounds = (x1 - C.D1 / 2, x1 + C.D1 / 2,
              C.DEV1_Y[0], C.DEV1_Y[1])
    return sdf_box(p, bounds)


def sdf_dev2(p, x2):
    bounds = (x2 - C.D2 / 2, x2 + C.D2 / 2,
              C.DEV2_Y[0], C.DEV2_Y[1])
    return sdf_box(p, bounds)


def sdf_dev3(p):
    return sdf_circle(p, C.C3, C.R3)


def sdf_air(p, x1, x2):
    """SDF-like CSG field for cavity minus all three devices.

    The zero level set describes the cavity walls and device surfaces.  The
    max composition preserves the sign needed for membership and is exact at
    each constituent boundary, although it is non-smooth where two distance
    fields tie.
    """
    cavity = sdf_box(p, (C.W_IN, 1 - C.W_IN,
                         C.W_IN, 1 - C.W_IN))
    return torch.maximum(
        cavity,
        torch.maximum(-sdf_dev1(p, x1),
                      torch.maximum(-sdf_dev2(p, x2), -sdf_dev3(p))))


def domain_sdf(name, p, x1=None, x2=None):
    """Return the implicit field F(x,y) for a named physical domain."""
    if name == "tbl":
        return sdf_tbl(p)
    if name in WALL_STRIPS:
        return sdf_strip(p, name)
    if name == "dev1":
        if x1 is None:
            raise ValueError("x1 is required for the dev1 SDF")
        return sdf_dev1(p, x1)
    if name == "dev2":
        if x2 is None:
            raise ValueError("x2 is required for the dev2 SDF")
        return sdf_dev2(p, x2)
    if name == "dev3":
        return sdf_dev3(p)
    if name == "air":
        if x1 is None or x2 is None:
            raise ValueError("x1 and x2 are required for the air SDF")
        return sdf_air(p, x1, x2)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# domain membership derived from F(x,y) <= 0
# --------------------------------------------------------------------------- #
def mask_tbl(p):
    return sdf_tbl(p) <= 0.0


def mask_strip(p, strip):
    return sdf_strip(p, strip) <= 0.0


def mask_dev1(p, x1):
    return sdf_dev1(p, x1) <= 0.0


def mask_dev2(p, x2):
    return sdf_dev2(p, x2) <= 0.0


def mask_dev3(p):
    return sdf_dev3(p) <= 0.0


def mask_air(p, x1, x2):
    return sdf_air(p, x1, x2) <= 0.0


def mask_domain(name, p, x1=None, x2=None):
    """Membership test derived uniformly from the named domain SDF."""
    return domain_sdf(name, p, x1, x2) <= 0.0


def label_points(p, x1, x2):
    """Assign each (n,2) point to exactly one region of LABEL_ORDER
    from SDF signs (priority: devices > tbl > wall strips > air).  Points
    outside the represented geometry receive -1.  With USE_TBL_1D the
    aerogel is handled analytically and therefore also receives -1."""
    idx_of = {d: i for i, d in enumerate(LABEL_ORDER)}
    lab = torch.full((p.shape[0],), -1, dtype=torch.long, device=p.device)
    lab[mask_air(p, x1, x2)] = idx_of["air"]
    for strip in WALL_STRIPS:
        lab[mask_strip(p, strip)] = idx_of[strip]
    if not C.USE_TBL_1D:
        lab[mask_tbl(p)] = idx_of["tbl"]
    lab[mask_dev3(p)] = idx_of["dev3"]
    lab[mask_dev2(p, x2)] = idx_of["dev2"]
    lab[mask_dev1(p, x1)] = idx_of["dev1"]
    return lab
