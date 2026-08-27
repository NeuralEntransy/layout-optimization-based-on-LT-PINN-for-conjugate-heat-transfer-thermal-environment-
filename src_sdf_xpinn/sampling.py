"""SDF-aware interior, interface and exterior-boundary samplers."""
import math
import torch
import config as C
import geometry as G


def box(n, bounds, device):
    xmin, xmax, ymin, ymax = bounds
    p = torch.rand(n, 2, device=device)
    p[:, 0] = xmin + (xmax-xmin) * p[:, 0]
    p[:, 1] = ymin + (ymax-ymin) * p[:, 1]
    return p


def line(n, start, end, device, deterministic=False):
    u = ((torch.arange(n, device=device) + .5) / n if deterministic
         else torch.rand(n, device=device))
    a = torch.tensor(start, dtype=torch.float32, device=device)
    b = torch.tensor(end, dtype=torch.float32, device=device)
    return a + u[:, None] * (b-a)


def union_hline(n, segments, y, device, deterministic=False):
    lengths = torch.tensor([b-a for a, b in segments], device=device)
    total = lengths.sum()
    u = ((torch.arange(n, device=device)+.5)/n*total if deterministic
         else torch.rand(n, device=device)*total)
    cumulative = torch.cumsum(lengths, 0)
    x = torch.empty_like(u)
    for i, (a, _b) in enumerate(segments):
        low = cumulative[i] - lengths[i]
        mask = (u >= low) & ((u < cumulative[i]) if i+1 < len(segments) else True)
        x[mask] = a + u[mask] - low
    return torch.stack((x, torch.full_like(x, y)), dim=1)


def sample_domain(name, n, x1, x2, device):
    x1, x2 = float(x1), float(x2)
    if name == "wall":
        # Exact length-weighted union of four non-overlapping strips.
        pieces = [(0., C.W_IN, 0., 1.), (1-C.W_IN, 1., 0., 1.),
                  (C.W_IN, 1-C.W_IN, 0., C.W_IN),
                  (C.W_IN, 1-C.W_IN, 1-C.W_IN, 1.)]
        areas = torch.tensor([(b-a)*(d-c) for a,b,c,d in pieces], device=device)
        ids = torch.multinomial(areas/areas.sum(), n, replacement=True)
        out = torch.empty(n, 2, device=device)
        for i, bounds in enumerate(pieces):
            mask = ids == i
            out[mask] = box(int(mask.sum()), bounds, device)
        return out
    if name == "dev1":
        return box(n, (x1-C.D1/2, x1+C.D1/2, *C.DEV1_Y), device)
    if name == "dev2":
        return box(n, (x2-C.D2/2, x2+C.D2/2, *C.DEV2_Y), device)
    if name == "dev3":
        r = C.R3 * torch.sqrt(torch.rand(n, device=device))
        a = 2*math.pi*torch.rand(n, device=device)
        return torch.stack((C.C3[0]+r*torch.cos(a), C.C3[1]+r*torch.sin(a)), 1)
    if name == "air":
        chunks, need = [], n
        while need:
            candidate = box(max(2*need, 256),
                            (C.W_IN, 1-C.W_IN, C.W_IN, 1-C.W_IN), device)
            keep = candidate[G.mask_domain("air", candidate, x1, x2)][:need]
            chunks.append(keep); need -= keep.shape[0]
        return torch.cat(chunks)
    raise KeyError(name)


def sample_near_boundary(name, n, width, x1, x2, device):
    """Sample inside a domain's SDF boundary layer: -width <= F <= 0."""
    if n <= 0:
        return torch.empty((0, 2), device=device)
    chunks, need = [], n
    while need:
        candidate = sample_domain(name, max(3*need, 256), x1, x2, device)
        distance = G.domain_sdf(name, candidate, x1, x2)
        selected = candidate[(distance <= 0) & (distance >= -width)][:need]
        chunks.append(selected); need -= selected.shape[0]
    return torch.cat(chunks)


def sample_interface(name, n, x1, x2, device, deterministic=False):
    x1, x2, w = float(x1), float(x2), C.W_IN
    direction = None
    if name == "wall_air_left":
        pts = line(n, (w,w), (w,1-w), device, deterministic); direction=(1.,0.)
    elif name == "wall_air_right":
        pts = line(n, (1-w,w), (1-w,1-w), device, deterministic); direction=(1.,0.)
    elif name == "wall_air_bottom":
        pts = union_hline(n, [(w,x1-C.D1/2),(x1+C.D1/2,1-w)], w, device, deterministic); direction=(0.,1.)
    elif name == "wall_air_top":
        pts = union_hline(n, [(w,x2-C.D2/2),(x2+C.D2/2,1-w)], 1-w, device, deterministic); direction=(0.,1.)
    elif name == "dev1_wall":
        pts=line(n,(x1-C.D1/2,w),(x1+C.D1/2,w),device,deterministic); direction=(0.,1.)
    elif name == "dev1_air_left":
        pts=line(n,(x1-C.D1/2,*C.DEV1_Y[:1]),(x1-C.D1/2,C.DEV1_Y[1]),device,deterministic); direction=(1.,0.)
    elif name == "dev1_air_right":
        pts=line(n,(x1+C.D1/2,C.DEV1_Y[0]),(x1+C.D1/2,C.DEV1_Y[1]),device,deterministic); direction=(1.,0.)
    elif name == "dev1_air_top":
        pts=line(n,(x1-C.D1/2,C.DEV1_Y[1]),(x1+C.D1/2,C.DEV1_Y[1]),device,deterministic); direction=(0.,1.)
    elif name == "dev2_wall":
        pts=line(n,(x2-C.D2/2,1-w),(x2+C.D2/2,1-w),device,deterministic); direction=(0.,1.)
    elif name == "dev2_air_left":
        pts=line(n,(x2-C.D2/2,C.DEV2_Y[0]),(x2-C.D2/2,C.DEV2_Y[1]),device,deterministic); direction=(1.,0.)
    elif name == "dev2_air_right":
        pts=line(n,(x2+C.D2/2,C.DEV2_Y[0]),(x2+C.D2/2,C.DEV2_Y[1]),device,deterministic); direction=(1.,0.)
    elif name == "dev2_air_bottom":
        pts=line(n,(x2-C.D2/2,C.DEV2_Y[0]),(x2+C.D2/2,C.DEV2_Y[0]),device,deterministic); direction=(0.,1.)
    elif name == "dev3_air":
        a = 2*math.pi*((torch.arange(n,device=device)+.5)/n if deterministic else torch.rand(n,device=device))
        dirs=torch.stack((torch.cos(a),torch.sin(a)),1)
        pts=torch.tensor(C.C3,device=device)+C.R3*dirs
        return pts, dirs
    else: raise KeyError(name)
    dirs = torch.tensor(direction, device=device).expand(n,2)
    return pts, dirs


def sample_all_interfaces(spec, x1, x2, device, deterministic=False):
    result = {}
    for name, a, b in G.INTERFACES:
        n = int(spec[name] if isinstance(spec, dict) else spec)
        pts, dirs = sample_interface(name,n,x1,x2,device,deterministic)
        result[name] = dict(pts=pts, dirs=dirs, a=a, b=b)
    return result


def sample_boundaries(spec, device, deterministic=False):
    def n(name): return int(spec[name] if isinstance(spec,dict) else spec)
    data = {
        "left": ((0.,0.),(0.,1.),(-1.,0.)),
        "right": ((1.,0.),(1.,1.),(1.,0.)),
        "top": ((0.,1.),(1.,1.),(0.,1.)),
        "bottom": ((0.,0.),(1.,0.),(0.,-1.)),
    }
    return {name: dict(pts=line(n(name),a,b,device,deterministic),
                       dirs=torch.tensor(normal,device=device).expand(n(name),2),
                       dom="wall") for name,(a,b,normal) in data.items()}
