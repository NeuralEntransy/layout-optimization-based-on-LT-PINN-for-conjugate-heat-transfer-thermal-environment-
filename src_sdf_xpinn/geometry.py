"""Differentiable SDF/CSG geometry for five physical regions."""
import math
import torch
import torch.nn as nn

import config as C

DOMAINS = list(C.DOMAINS)
LABEL_ORDER = list(C.DOMAINS)

INTERFACES = [
    ("wall_air_left", "wall", "air"),
    ("wall_air_right", "wall", "air"),
    ("wall_air_bottom", "wall", "air"),
    ("wall_air_top", "wall", "air"),
    ("dev1_wall", "dev1", "wall"),
    ("dev1_air_left", "dev1", "air"),
    ("dev1_air_right", "dev1", "air"),
    ("dev1_air_top", "dev1", "air"),
    ("dev2_wall", "dev2", "wall"),
    ("dev2_air_left", "dev2", "air"),
    ("dev2_air_right", "dev2", "air"),
    ("dev2_air_bottom", "dev2", "air"),
    ("dev3_air", "dev3", "air"),
]

IFACE_META = {
    "dev1_wall": ((0., -1.), C.D1),
    "dev1_air_left": ((-1., 0.), C.D1),
    "dev1_air_right": ((1., 0.), C.D1),
    "dev1_air_top": ((0., 1.), C.D1),
    "dev2_wall": ((0., 1.), C.D2),
    "dev2_air_left": ((-1., 0.), C.D2),
    "dev2_air_right": ((1., 0.), C.D2),
    "dev2_air_bottom": ((0., -1.), C.D2),
    "dev3_air": ("radial", 2 * math.pi * C.R3),
}
DEV_IFACES = {
    "dev1": ["dev1_wall", "dev1_air_left", "dev1_air_right", "dev1_air_top"],
    "dev2": ["dev2_wall", "dev2_air_left", "dev2_air_right", "dev2_air_bottom"],
    "dev3": ["dev3_air"],
}


class DesignVars(nn.Module):
    def __init__(self, x1, x2, trainable=False, device="cpu"):
        super().__init__()
        def make(value, limits):
            u = (value - limits[0]) / (limits[1] - limits[0])
            return math.log(u / (1 - u))
        self.z1 = nn.Parameter(torch.tensor(make(x1, C.X1_RANGE), device=device),
                               requires_grad=trainable)
        self.z2 = nn.Parameter(torch.tensor(make(x2, C.X2_RANGE), device=device),
                               requires_grad=trainable)

    def x1(self):
        lo, hi = C.X1_RANGE
        return lo + (hi - lo) * torch.sigmoid(self.z1)

    def x2(self):
        lo, hi = C.X2_RANGE
        return lo + (hi - lo) * torch.sigmoid(self.z2)


def sdf_box(p, bounds):
    vals = [torch.as_tensor(v, dtype=p.dtype, device=p.device) for v in bounds]
    xmin, xmax, ymin, ymax = vals
    center = torch.stack(((xmin + xmax) / 2, (ymin + ymax) / 2))
    half = torch.stack(((xmax - xmin) / 2, (ymax - ymin) / 2))
    q = torch.abs(p - center) - half
    return (torch.linalg.vector_norm(torch.clamp(q, min=0), dim=1)
            + torch.clamp(torch.amax(q, dim=1), max=0))


def sdf_circle(p, center, radius):
    c = torch.as_tensor(center, dtype=p.dtype, device=p.device)
    return torch.linalg.vector_norm(p - c, dim=1) - radius


def sdf_outer(p):
    return sdf_box(p, (0., 1., 0., 1.))


def sdf_cavity(p):
    return sdf_box(p, (C.W_IN, 1-C.W_IN, C.W_IN, 1-C.W_IN))


def sdf_wall(p):
    """Outer enclosure minus the open cavity: max(F_outer, -F_cavity)."""
    return torch.maximum(sdf_outer(p), -sdf_cavity(p))


def sdf_dev1(p, x1):
    return sdf_box(p, (x1-C.D1/2, x1+C.D1/2, *C.DEV1_Y))


def sdf_dev2(p, x2):
    return sdf_box(p, (x2-C.D2/2, x2+C.D2/2, *C.DEV2_Y))


def sdf_dev3(p):
    return sdf_circle(p, C.C3, C.R3)


def sdf_air(p, x1, x2):
    """Cavity minus the union of all three devices."""
    holes = torch.minimum(sdf_dev1(p, x1),
                          torch.minimum(sdf_dev2(p, x2), sdf_dev3(p)))
    return torch.maximum(sdf_cavity(p), -holes)


def domain_sdf(name, p, x1=None, x2=None):
    if name == "wall": return sdf_wall(p)
    if name == "air": return sdf_air(p, x1, x2)
    if name == "dev1": return sdf_dev1(p, x1)
    if name == "dev2": return sdf_dev2(p, x2)
    if name == "dev3": return sdf_dev3(p)
    raise KeyError(name)


def mask_domain(name, p, x1=None, x2=None):
    return domain_sdf(name, p, x1, x2) <= 0


def label_points(p, x1, x2):
    labels = torch.full((p.shape[0],), -1, dtype=torch.long, device=p.device)
    for name in ("wall", "air", "dev3", "dev2", "dev1"):
        labels[mask_domain(name, p, x1, x2)] = LABEL_ORDER.index(name)
    return labels


def smooth_indicator(name, p, x1, x2, beta=100.0):
    """LT-PINN-style differentiable region visibility, optional in later stages."""
    return torch.sigmoid(-beta * domain_sdf(name, p, x1, x2))
