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


def sample_near_outer_boundary(name, n, width, device):
    """Sample wall PDE points concentrated just inside top/bottom boundaries."""
    if n <= 0:
        return torch.empty((0, 2), device=device)
    if not 0.0 < width <= C.W_IN:
        raise ValueError("near-boundary width must lie in (0, wall thickness]")
    if name not in {"wall_b", "wall_t"}:
        raise KeyError(name)

    x = torch.rand(n, device=device) * (1.0 - 2.0 * C.W_IN) + C.W_IN
    # Squared distances bias points toward the physical boundary while still
    # covering the complete boundary layer.
    distance = (width * torch.rand(n, device=device).square()).clamp_min(1e-6)
    y = distance if name == "wall_b" else 1.0 - distance
    return torch.stack([x, y], dim=1)


def sample_near_device_boundary(name, n, width, x1, x2, device):
    """Sample points inside a device and close to its material boundary."""
    if n <= 0:
        return torch.empty((0, 2), device=device)
    x1f, x2f = float(x1), float(x2)
    if name in {"dev1", "dev2"}:
        side = C.D1 if name == "dev1" else C.D2
        xc = x1f if name == "dev1" else x2f
        yr = C.DEV1_Y if name == "dev1" else C.DEV2_Y
        width = min(width, side / 2)
        points = _box(max(2 * n, 128), xc - side / 2, xc + side / 2,
                      yr[0], yr[1], device)
        distance = torch.minimum(
            torch.minimum(points[:, 0] - (xc - side / 2),
                          (xc + side / 2) - points[:, 0]),
            torch.minimum(points[:, 1] - yr[0], yr[1] - points[:, 1]))
        selected = points[distance <= width]
        while selected.shape[0] < n:
            more = _box(max(2 * (n - selected.shape[0]), 128),
                        xc - side / 2, xc + side / 2, yr[0], yr[1], device)
            d = torch.minimum(
                torch.minimum(more[:, 0] - (xc - side / 2),
                              (xc + side / 2) - more[:, 0]),
                torch.minimum(more[:, 1] - yr[0], yr[1] - more[:, 1]))
            selected = torch.cat((selected, more[d <= width]), dim=0)
        return selected[:n]
    if name == "dev3":
        inner = max(0.0, C.R3 - width)
        u = torch.rand(n, device=device)
        radius = torch.sqrt(inner ** 2 + u * (C.R3 ** 2 - inner ** 2))
        angle = 2 * math.pi * torch.rand(n, device=device)
        return torch.stack([C.C3[0] + radius * torch.cos(angle),
                            C.C3[1] + radius * torch.sin(angle)], dim=1)
    raise KeyError(name)


def sample_near_wall_surface(name, n, width, device):
    """Sample points inside a wall strip near one named inner/outer face."""
    if n <= 0:
        return torch.empty((0, 2), device=device)
    dom, face = name.rsplit("_", 1)
    if dom not in G.WALL_STRIPS or face not in {"inner", "outer"}:
        raise KeyError(name)
    x0, x1, y0, y1 = G.WALL_STRIPS[dom]
    if dom == "wall_l":
        lo, hi = ((x0, min(x0 + width, x1)) if face == "outer"
                  else (max(x1 - width, x0), x1))
        return _box(n, lo, hi, y0, y1, device)
    if dom == "wall_r":
        lo, hi = ((max(x1 - width, x0), x1) if face == "outer"
                  else (x0, min(x0 + width, x1)))
        return _box(n, lo, hi, y0, y1, device)
    if dom == "wall_b":
        lo, hi = ((y0, min(y0 + width, y1)) if face == "outer"
                  else (max(y1 - width, y0), y1))
        return _box(n, x0, x1, lo, hi, device)
    lo, hi = ((max(y1 - width, y0), y1) if face == "outer"
              else (y0, min(y0 + width, y1)))
    return _box(n, x0, x1, lo, hi, device)


def sample_air_near(name, n, width, x1, x2, device):
    """Sample air points in a narrow layer next to a device or long wall."""
    if n <= 0:
        return torch.empty((0, 2), device=device)
    x1f, x2f = float(x1), float(x2)
    points = []
    needed = n
    while needed > 0:
        if name in {"wall_b", "wall_t"}:
            if name == "wall_b":
                candidates = _box(max(3 * needed, 256), C.W_IN,
                                  1 - C.W_IN, C.W_IN,
                                  C.W_IN + width, device)
            else:
                candidates = _box(max(3 * needed, 256), C.W_IN,
                                  1 - C.W_IN, 1 - C.W_IN - width,
                                  1 - C.W_IN, device)
        elif name in {"dev1", "dev2"}:
            side = C.D1 if name == "dev1" else C.D2
            xc = x1f if name == "dev1" else x2f
            yr = C.DEV1_Y if name == "dev1" else C.DEV2_Y
            candidates = _box(max(5 * needed, 256),
                              xc - side / 2 - width,
                              xc + side / 2 + width,
                              yr[0] - width, yr[1] + width, device)
            dx = torch.clamp(torch.abs(candidates[:, 0] - xc)
                             - side / 2, min=0.0)
            yc = (yr[0] + yr[1]) / 2
            dy = torch.clamp(torch.abs(candidates[:, 1] - yc)
                             - side / 2, min=0.0)
            close = torch.sqrt(dx ** 2 + dy ** 2) <= width
            candidates = candidates[close]
        elif name.startswith(("dev1_air_", "dev2_air_")):
            dev, face = name.split("_air_", 1)
            side = C.D1 if dev == "dev1" else C.D2
            xc = x1f if dev == "dev1" else x2f
            yr = C.DEV1_Y if dev == "dev1" else C.DEV2_Y
            xl, xr = xc - side / 2, xc + side / 2
            if face == "left":
                candidates = _box(max(2 * needed, 256),
                                  xl - width, xl, yr[0], yr[1], device)
            elif face == "right":
                candidates = _box(max(2 * needed, 256),
                                  xr, xr + width, yr[0], yr[1], device)
            elif face == "bottom" and dev == "dev2":
                candidates = _box(max(2 * needed, 256),
                                  xl, xr, yr[0] - width, yr[0], device)
            elif face == "top" and dev == "dev1":
                candidates = _box(max(2 * needed, 256),
                                  xl, xr, yr[1], yr[1] + width, device)
            else:
                raise KeyError(name)
        elif name == "dev3":
            angle = 2 * math.pi * torch.rand(max(2 * needed, 256),
                                               device=device)
            radius = torch.sqrt(C.R3 ** 2 + torch.rand_like(angle)
                                * ((C.R3 + width) ** 2 - C.R3 ** 2))
            candidates = torch.stack(
                [C.C3[0] + radius * torch.cos(angle),
                 C.C3[1] + radius * torch.sin(angle)], dim=1)
        else:
            raise KeyError(name)
        keep = G.mask_air(candidates, x1f, x2f)
        selected = candidates[keep][:needed]
        points.append(selected)
        needed -= selected.shape[0]
    return torch.cat(points, dim=0)


def _sample_count(spec, name):
    """Resolve a scalar or per-name sample-count specification."""
    if isinstance(spec, dict):
        if name not in spec:
            raise KeyError(f"missing sample count for {name}")
        return int(spec[name])
    return int(spec)


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
        count = _sample_count(n, name)
        points, directions = sample_interface(
            name, count, x1, x2, device, deterministic=deterministic)
        samples[name] = dict(pts=points, dirs=directions,
                             a=domain_a, b=domain_b)
    return samples


def sample_boundaries(n, device, deterministic=False):
    """Sample outer boundaries and identify the owning temperature branch."""
    boundaries = {}
    def vline(name, x, yr):
        count = _sample_count(n, name)
        return (_line_midpoints(count, (x, yr[0]), (x, yr[1]), device)
                if deterministic else _vertical_line(count, x, yr, device))

    def hline(name, xr, y):
        count = _sample_count(n, name)
        return (_line_midpoints(count, (xr[0], y), (xr[1], y), device)
                if deterministic else _box(count, xr[0], xr[1], y, y,
                                            device))
    if C.USE_TBL_1D:
        # Robin resistance from wall_l at x=0 to the cold reservoir.
        boundaries["left"] = dict(
            pts=vline("left", 0.0, (0.0, 1.0)), dom="wall_l")
    else:
        boundaries["left"] = dict(
            pts=vline("left", C.X_TBL, (0.0, 1.0)), dom="tbl")

    boundaries["right"] = dict(
        pts=vline("right", 1.0, (0.0, 1.0)), dom="wall_r")

    if not C.USE_TBL_1D:
        for side, y in [("top", 1.0), ("bottom", 0.0)]:
            boundaries[f"{side}_tbl"] = dict(
                pts=hline(f"{side}_tbl", (C.X_TBL, 0.0), y), dom="tbl")
            count = boundaries[f"{side}_tbl"]["pts"].shape[0]
            boundaries[f"{side}_tbl"]["dirs"] = torch.tensor(
                [0.0, 1.0], device=device).expand(count, 2)

    # y=0/1 spans wall_l, wall_b/wall_t and wall_r.
    for side, y in [("top", 1.0), ("bottom", 0.0)]:
        middle = "wall_t" if side == "top" else "wall_b"
        strips = [("wall_l", (0.0, C.W_IN)),
                  (middle, (C.W_IN, 1 - C.W_IN)),
                  ("wall_r", (1 - C.W_IN, 1.0))]
        for strip, (x0, x1_) in strips:
            key = f"{side}_{strip}"
            pts = hline(key, (x0, x1_), y)
            boundaries[key] = dict(
                pts=pts, dom=strip,
                dirs=torch.tensor([0.0, 1.0], device=device).expand(
                    pts.shape[0], 2))
    return boundaries
