# -*- coding: utf-8 -*-
"""Collocation samplers for the v4 multi-domain thermal benchmark.

Geometry shapes, masks, topology and interface metadata live in geometry.py.
This module turns that static geometry information into random tensors for:
  * interior PDE collocation points,
  * material/contact interface points and derivative directions,
  * outer-boundary points and their owning domains.

Milestone 1 uses fixed layouts.  The current float(x1/x2) conversion therefore
intentionally detaches layout coordinates; stage-A differentiable layout
optimization must replace it with reparameterized moving samples.
"""
import math

import torch

import config as C
import geometry as G


def _box(n, xmin, xmax, ymin, ymax, device):
    """Uniform samples in an axis-aligned box (or line if one span is zero)."""
    p = torch.rand(n, 2, device=device)
    p[:, 0] = p[:, 0] * (xmax - xmin) + xmin
    p[:, 1] = p[:, 1] * (ymax - ymin) + ymin
    return p


def _vertical_line(n, x, y_range, device):
    """Uniform samples on x=constant, y in y_range."""
    p = torch.rand(n, 2, device=device)
    p[:, 0] = x
    p[:, 1] = p[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    return p


def _line_midpoints(n, start, end, device):
    """Deterministic equal-weight midpoint rule on a straight segment."""
    u = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    start = torch.tensor(start, device=device, dtype=torch.float32)
    end = torch.tensor(end, device=device, dtype=torch.float32)
    return start + u[:, None] * (end - start)


def _union_midpoints(n, segments, y, device):
    """Deterministic midpoint rule on a length-parameterized segment union."""
    lengths = torch.tensor([b - a for a, b in segments], device=device)
    cumulative = torch.cumsum(lengths, 0)
    u = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) \
        / n * cumulative[-1]
    x = torch.empty_like(u)
    for i, (a, _b) in enumerate(segments):
        lower = cumulative[i] - lengths[i]
        mask = (u >= lower) & (u < cumulative[i]) \
            if i < len(segments) - 1 else (u >= lower)
        x[mask] = a + (u[mask] - lower)
    return torch.stack([x, torch.full_like(x, y)], dim=1)


def _union_segment(n, segments, y, device):
    """Length-weighted uniform samples on a union of horizontal segments."""
    lengths = torch.tensor([b - a for a, b in segments], device=device)
    cumulative = torch.cumsum(lengths, 0)
    u = torch.rand(n, device=device) * cumulative[-1]
    x = torch.empty(n, device=device)
    for i, (a, b) in enumerate(segments):
        lower = cumulative[i] - lengths[i]
        mask = (u >= lower) & (u < cumulative[i]) \
            if i < len(segments) - 1 else (u >= lower)
        x[mask] = a + (u[mask] - lower)
    return torch.stack([x, torch.full_like(x, y)], dim=1)


def sample_domain(name, n, x1, x2, device):
    """Sample n interior collocation points from one active domain."""
    x1f, x2f = float(x1), float(x2)
    if name == "tbl":
        return _box(n, C.X_TBL, 0.0, 0.0, 1.0, device)
    if name in G.WALL_STRIPS:
        x0, x1_, y0, y1 = G.WALL_STRIPS[name]
        return _box(n, x0, x1_, y0, y1, device)
    if name == "air":
        points, needed, tries = [], n, 0
        while needed > 0 and tries < 60:
            candidates = _box(max(2 * needed, 128), C.W_IN, 1 - C.W_IN,
                              C.W_IN, 1 - C.W_IN, device)
            keep = G.mask_air(candidates, x1f, x2f)
            selected = candidates[keep][:needed]
            points.append(selected)
            needed -= selected.shape[0]
            tries += 1
        out = torch.cat(points, 0)
        if out.shape[0] < n:
            repeats = (n // out.shape[0]) + 1
            out = out.repeat(repeats, 1)[:n]
        return out
    if name == "dev1":
        return _box(n, x1f - C.D1 / 2, x1f + C.D1 / 2,
                    C.DEV1_Y[0], C.DEV1_Y[1], device)
    if name == "dev2":
        return _box(n, x2f - C.D2 / 2, x2f + C.D2 / 2,
                    C.DEV2_Y[0], C.DEV2_Y[1], device)
    if name == "dev3":
        radial_u = torch.rand(n, device=device)
        angular_u = torch.rand(n, device=device)
        radius = C.R3 * torch.sqrt(radial_u)
        angle = 2 * math.pi * angular_u
        return torch.stack([C.C3[0] + radius * torch.cos(angle),
                            C.C3[1] + radius * torch.sin(angle)], dim=1)
    raise KeyError(name)


def sample_interface(name, n, x1, x2, device, deterministic=False):
    """Return interface points and a shared unit derivative direction."""
    x1f, x2f = float(x1), float(x2)
    wall = C.W_IN
    vline = (lambda x, yr: _line_midpoints(
        n, (x, yr[0]), (x, yr[1]), device)) if deterministic else \
        (lambda x, yr: _vertical_line(n, x, yr, device))
    hline = (lambda xr, y: _line_midpoints(
        n, (xr[0], y), (xr[1], y), device)) if deterministic else \
        (lambda xr, y: _box(n, xr[0], xr[1], y, y, device))
    union = _union_midpoints if deterministic else _union_segment

    if name == "tbl_wall":
        points = vline(0.0, (0.0, 1.0))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "wall_air_left":
        points = vline(wall, (wall, 1 - wall))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "wall_air_right":
        points = vline(1 - wall, (wall, 1 - wall))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "wall_air_bottom":
        segments = [(wall, x1f - C.D1 / 2),
                    (x1f + C.D1 / 2, 1 - wall)]
        points = union(n, segments, wall, device)
        directions = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "wall_air_top":
        segments = [(wall, x2f - C.D2 / 2),
                    (x2f + C.D2 / 2, 1 - wall)]
        points = union(n, segments, 1 - wall, device)
        directions = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev1_wall":
        points = hline((x1f - C.D1 / 2, x1f + C.D1 / 2), wall)
        directions = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev1_air_left":
        points = vline(x1f - C.D1 / 2, C.DEV1_Y)
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev1_air_right":
        points = vline(x1f + C.D1 / 2, C.DEV1_Y)
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev1_air_top":
        points = hline((x1f - C.D1 / 2, x1f + C.D1 / 2), C.DEV1_Y[1])
        directions = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev2_wall":
        points = hline((x2f - C.D2 / 2, x2f + C.D2 / 2), 1 - wall)
        directions = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev2_air_left":
        points = vline(x2f - C.D2 / 2, C.DEV2_Y)
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev2_air_right":
        points = vline(x2f + C.D2 / 2, C.DEV2_Y)
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "dev2_air_bottom":
        points = hline((x2f - C.D2 / 2, x2f + C.D2 / 2), C.DEV2_Y[0])
        directions = torch.tensor([0.0, 1.0], device=device).expand(n, 2)
    elif name == "dev3_air":
        angle = 2 * math.pi * ((torch.arange(n, device=device) + 0.5) / n \
            if deterministic else torch.rand(n, device=device))
        points = torch.stack([C.C3[0] + C.R3 * torch.cos(angle),
                              C.C3[1] + C.R3 * torch.sin(angle)], dim=1)
        directions = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    elif name == "corner_l_b":
        points = vline(wall, (0.0, wall))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "corner_l_t":
        points = vline(wall, (1 - wall, 1.0))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "corner_r_b":
        points = vline(1 - wall, (0.0, wall))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    elif name == "corner_r_t":
        points = vline(1 - wall, (1 - wall, 1.0))
        directions = torch.tensor([1.0, 0.0], device=device).expand(n, 2)
    else:
        raise KeyError(name)
    return points, directions


def sample_all_interfaces(n, x1, x2, device, deterministic=False):
    """Sample every active interface and attach its topology metadata."""
    samples = {}
    for name, domain_a, domain_b in G.INTERFACES:
        points, directions = sample_interface(
            name, n, x1, x2, device, deterministic=deterministic)
        samples[name] = dict(pts=points, dirs=directions,
                             a=domain_a, b=domain_b)
    return samples


def sample_boundaries(n, device, deterministic=False):
    """Sample outer boundaries and identify the owning temperature branch."""
    boundaries = {}
    vline = (lambda x, yr: _line_midpoints(
        n, (x, yr[0]), (x, yr[1]), device)) if deterministic else \
        (lambda x, yr: _vertical_line(n, x, yr, device))
    hline = (lambda xr, y: _line_midpoints(
        n, (xr[0], y), (xr[1], y), device)) if deterministic else \
        (lambda xr, y: _box(n, xr[0], xr[1], y, y, device))
    if C.USE_TBL_1D:
        # Robin resistance from wall_l at x=0 to the cold reservoir.
        boundaries["left"] = dict(
            pts=vline(0.0, (0.0, 1.0)), dom="wall_l")
    else:
        boundaries["left"] = dict(
            pts=vline(C.X_TBL, (0.0, 1.0)), dom="tbl")

    boundaries["right"] = dict(
        pts=vline(1.0, (0.0, 1.0)), dom="wall_r")

    if not C.USE_TBL_1D:
        for side, y in [("top", 1.0), ("bottom", 0.0)]:
            boundaries[f"{side}_tbl"] = dict(
                pts=hline((C.X_TBL, 0.0), y), dom="tbl",
                dirs=torch.tensor([0.0, 1.0], device=device).expand(n, 2))

    # y=0/1 spans wall_l, wall_b/wall_t and wall_r.
    for side, y in [("top", 1.0), ("bottom", 0.0)]:
        middle = "wall_t" if side == "top" else "wall_b"
        strips = [("wall_l", (0.0, C.W_IN)),
                  (middle, (C.W_IN, 1 - C.W_IN)),
                  ("wall_r", (1 - C.W_IN, 1.0))]
        for strip, (x0, x1_) in strips:
            boundaries[f"{side}_{strip}"] = dict(
                pts=hline((x0, x1_), y), dom=strip,
                dirs=torch.tensor([0.0, 1.0], device=device).expand(n, 2))
    return boundaries
