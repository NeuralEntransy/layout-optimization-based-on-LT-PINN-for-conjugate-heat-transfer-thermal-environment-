# -*- coding: utf-8 -*-
"""
Surface-to-surface radiation helpers for the 2-D enclosure (doc 04, sec 3.4).

The air is non-participating; all exposed surfaces are gray-diffuse with
emissivity eps.  Radiosity model:
    J = eps*sigma*T^4 + (1-eps) G,   G = F J,   q_rad = J - G  (net leaving)

View factors are computed for the infinitely-extruded 2-D enclosure via the
crossed-string method with ray-cast occlusion by the devices, then corrected
to enforce reciprocity (A_i F_ij = A_j F_ji) and closure (sum_j F_ij = 1),
which are hard acceptance metrics.
"""
import numpy as np

from dolfinx.geometry import bb_tree, compute_collisions_points

SIGMA = 5.670374419e-8      # Stefan-Boltzmann [W/m^2/K^4]


# --------------------------------------------------------------------------- #
# cavity-surface extraction from the conforming mesh
# --------------------------------------------------------------------------- #
def extract_cavity_facets(domain, cell_tags, air_tag):
    """Interior facets shared by an air cell and a solid cell = cavity surface.

    Returns dict of numpy arrays: p1, p2 (endpoints, (N,2)), mid (N,2),
    normal (N,2, unit, pointing INTO the air), length (N,), owner (N,),
    facet_indices.
    """
    from dolfinx import mesh as dmesh
    from dolfinx.geometry import bb_tree, compute_collisions_points
    tdim = domain.topology.dim
    domain.topology.create_connectivity(1, tdim)
    domain.topology.create_connectivity(1, 0)
    f2c = domain.topology.connectivity(1, tdim)
    ctags = cell_tags.values

    nfac = domain.topology.index_map(1).size_local
    cavity = []
    for f in range(nfac):
        cells = f2c.links(f)
        if len(cells) != 2:
            continue
        t0, t1 = ctags[cells[0]], ctags[cells[1]]
        if (t0 == air_tag) != (t1 == air_tag):
            cavity.append((f, cells[0], cells[1], t0, t1))
    cavity = np.array(cavity, dtype=np.int64)
    facets = cavity[:, 0]

    # geometry
    # geometry: map topology vertices -> geometry nodes (needed because the
    # 2nd-order gmsh mesh has mid-edge nodes, so vertex index != node index)
    mids = dmesh.compute_midpoints(domain, 1, facets)[:, :2]
    f2nodes = dmesh.entities_to_geometry(domain, 1, facets)
    x = domain.geometry.x
    p1 = np.zeros((len(facets), 2))
    p2 = np.zeros((len(facets), 2))
    for i in range(len(facets)):
        nodes = x[f2nodes[i], :2]
        # endpoints = the two most distant geometry nodes of the facet
        dd = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
        a, b = np.unravel_index(np.argmax(dd), dd.shape)
        p1[i], p2[i] = nodes[a], nodes[b]
    length = np.linalg.norm(p2 - p1, axis=1)

    # normal pointing into the air: robustly oriented by testing which side
    # of each facet contains an air cell (independent of mesh node ordering)
    tang = (p2 - p1) / length[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    tree = bb_tree(domain, tdim)
    eps_probe = 0.2 * length
    owner = np.where(cavity[:, 3] == air_tag, cavity[:, 4], cavity[:, 3])
    for i in range(len(facets)):
        probe = mids[i] + eps_probe[i] * nrm[i]
        links = compute_collisions_points(tree, np.array(
            [[probe[0], probe[1], 0.0]]))
        cell = links.array[links.offsets[0]:links.offsets[1]]
        is_air = len(cell) > 0 and ctags[cell[0]] == air_tag
        if not is_air:
            nrm[i] *= -1.0

    return dict(p1=p1, p2=p2, mid=mids, normal=nrm, length=length,
                owner=owner, facet_indices=facets)


# --------------------------------------------------------------------------- #
# visibility (occlusion by the devices)
# --------------------------------------------------------------------------- #
def _seg_disk_blocked(P, Q, Cc, R):
    """Vectorized: does open segment P->Q pass within R of Cc? (N,N) bool."""
    d = Q - P                       # (...,2)
    w = Cc - P
    tt = np.sum(w * d, axis=-1) / np.sum(d * d, axis=-1)
    tt = np.clip(tt, 0.0, 1.0)
    dist = np.linalg.norm(P + tt[..., None] * d - Cc, axis=-1)
    return dist < R


def _seg_rect_blocked(P, Q, xmin, xmax, ymin, ymax):
    """Vectorized slab-method segment/rectangle intersection. (N,N) bool."""
    d = Q - P
    tiny = 1e-300
    dx = np.where(np.abs(d[..., 0]) < tiny, tiny, d[..., 0])
    dy = np.where(np.abs(d[..., 1]) < tiny, tiny, d[..., 1])
    tx1 = (xmin - P[..., 0]) / dx
    tx2 = (xmax - P[..., 0]) / dx
    ty1 = (ymin - P[..., 1]) / dy
    ty2 = (ymax - P[..., 1]) / dy
    tlo = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    thi = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    return thi > np.maximum(tlo, 0.005)  # exclude the immediate start region


def compute_view_factors(pan, devices, shrink=0.02):
    """Crossed-string view factors with occlusion.

    pan: dict from extract_cavity_facets.  devices: dict with 'rects' (list of
    (xmin,xmax,ymin,ymax,owner_tag)) and 'disk' (cx,cy,R,owner_tag).
    Returns F (N,N) with reciprocity + closure enforced.
    """
    p1, p2, mid, nrm, length, owner = (pan["p1"], pan["p2"], pan["mid"],
                                       pan["normal"], pan["length"],
                                       pan["owner"])
    N = len(mid)
    A = length.copy()                    # area per unit extrusion = length*b

    # mutual front-facing test: each midpoint must lie in the other's
    # front half-space (normals point into the air)
    D = mid[None, :, :] - mid[:, None, :]          # D[i,j] = mid_j - mid_i
    # mutual front-facing: j in front of i (D.n_i > 0) and i in front of j
    dot_i = np.sum(D * nrm[:, None, :], axis=-1)   # (mid_j - mid_i).n_i
    dot_j = np.sum(-D * nrm[None, :, :], axis=-1)  # (mid_i - mid_j).n_j
    facing = (dot_i > 1e-12) & (dot_j > 1e-12)

    # occlusion: shrink test segment to middle (1-2*shrink) to avoid the
    # facets' own endpoints grazing their owner device
    P = mid[:, None, :] + (mid[None, :, :] - mid[:, None, :]) * shrink
    Q = mid[:, None, :] - (mid[None, :, :] - mid[:, None, :]) * shrink
    blocked = np.zeros((N, N), dtype=bool)
    for (xmin, xmax, ymin, ymax, tag) in devices.get("rects", []):
        b = _seg_rect_blocked(P, Q, xmin, xmax, ymin, ymax)
        own = (owner[:, None] == tag) | (owner[None, :] == tag)
        blocked |= b & ~own
    if "disk" in devices:
        cx, cy, R, tag = devices["disk"]
        b = _seg_disk_blocked(P, Q, np.array([cx, cy]), R)
        own = (owner[:, None] == tag) | (owner[None, :] == tag)
        blocked |= b & ~own

    visible = facing & ~blocked
    np.fill_diagonal(visible, False)

    # crossed-string view factors (exact for fully visible 2-D segment pairs)
    a1, a2 = p1[:, None, :], p2[:, None, :]
    b1, b2 = p1[None, :, :], p2[None, :, :]

    def dist(U, V):
        return np.linalg.norm(U - V, axis=-1)

    S1 = dist(a1, b2) + dist(a2, b1)   # candidate "crossed"
    S2 = dist(a1, b1) + dist(a2, b2)   # candidate "uncrossed"
    crossed = np.maximum(S1, S2)
    uncrossed = np.minimum(S1, S2)
    F = np.clip(crossed - uncrossed, 0.0, None) / (2.0 * A[:, None])
    F *= visible

    # ---- enforce reciprocity: H_ij = A_i F_ij symmetric
    H = A[:, None] * F
    H = 0.5 * (H + H.T)
    # ---- enforce closure by iterative row-scaling + symmetrization
    for _ in range(50):
        F = H / A[:, None]
        rowsum = F.sum(axis=1, keepdims=True)
        rowsum[rowsum < 1e-300] = 1.0
        F = F / rowsum
        H = A[:, None] * F
        H = 0.5 * (H + H.T)
    F = H / A[:, None]
    return F, dict(visible_pairs=int(visible.sum()))


# --------------------------------------------------------------------------- #
# radiosity solve
# --------------------------------------------------------------------------- #
def radiosity(T_surf, F, eps):
    """Net radiative flux q_rad [W/m^2] leaving each surface element.

    Solves (I - (1-eps) F) J = eps sigma T^4, then q = (I - F) J.
    """
    N = len(T_surf)
    E = np.full(N, eps)
    M = np.eye(N) - np.diag(1 - E) @ F
    rhs = E * SIGMA * np.asarray(T_surf).ravel() ** 4
    J = np.linalg.solve(M, rhs)
    G = F @ J
    return J - G, J, G


def enclosure_checks(F, area):
    """Closure and reciprocity residuals (acceptance metrics)."""
    closure = float(np.abs(F.sum(axis=1) - 1.0).max())
    H = area[:, None] * F
    rec = float(np.abs(H - H.T).max() / max(H.max(), 1e-300))
    return dict(closure_max_err=closure, reciprocity_rel_err=rec)
