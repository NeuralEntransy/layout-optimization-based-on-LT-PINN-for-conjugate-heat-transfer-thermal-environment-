# -*- coding: utf-8 -*-
"""
Milestone 0: deterministic FEM reference solutions (FEniCSx / DOLFINx 0.10)
==========================================================================
Closed low-pressure cavity with left aerogel layer, aluminum wall frame,
stagnant air (stage A reduced model: conduction only, no radiation, no flow).

Solves steady multi-material conduction
    div(k_m grad T) + q'''_m = 0
for >= 3 fixed layouts (squares left / center / right) and reports:
  * T_max of the three devices (feasibility vs 70 C limit)
  * Q_left / Q_right and total energy balance vs P_tot = 1050 W
  * right-wall area-averaged temperature and Robin residual
  * aerogel 1D thermal-resistance equivalence metrics

Boundary conditions (doc 04, section 2.3):
  x = -0.005 : Dirichlet T = 218.15 K (-55 C)
  x =  1     : Robin  -k_Al dT/dn = h_ext (T - T_inf), h_ext = 15, T_inf = 303.15 K
  y = 0 / 1  : adiabatic (natural)

All internal material interfaces are conforming in the mesh, hence
temperature and flux continuity are satisfied exactly by construction.

Run with the fenicsx conda env:
    ~/miniforge/envs/fenicsx-env/bin/python fem/milestone0_reference.py
"""

import json
import os
import sys
import numpy as np

import gmsh
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.geometry import bb_tree, compute_collisions_points
from dolfinx.fem.petsc import LinearProblem

# ---------------------------------------------------------------- parameters
P = dict(
    # geometry [m]
    x_tbl=-0.05,            # aerogel outer face
    w_in=0.05,              # wall inner offset (frame thickness)
    d1=0.2,  P1=300.0,      # device 1: small square on bottom wall
    d2=0.35, P2=600.0,      # device 2: large square on top wall
    c3=(0.5, 0.4), R3=0.1, P3=20.0,    # device 3: fixed disk
    b=1.0,                  # extrusion thickness
    # materials
    k_Al=167.0, k_f=0.026, k_TBL=0.018,
    # thermal environment
    T_cold=218.15,          # left Dirichlet [K]
    h_ext=20.0, T_inf=303.15,  # right Robin
    # mesh
    h_global=0.006, h_device=0.003,
)

P["q1"] = P["P1"] / (P["d1"] ** 2 * P["b"])          # 7500 W/m3
P["q2"] = P["P2"] / (P["d2"] ** 2 * P["b"])          # 4898 W/m3
P["q3"] = P["P3"] / (np.pi * P["R3"] ** 2 * P["b"])  #  637 W/m3
P["P_tot"] = P["P1"] + P["P2"] + P["P3"]             #  920 W
P["T_lim"] = 343.15                                   # 70 C

LAYOUTS = {
    "left":   dict(x1=0.25, x2=0.35),
    "center": dict(x1=0.50, x2=0.50),
    "right":  dict(x1=0.75, x2=0.65),
}

# subdomain / boundary tags
TAG_TBL, TAG_WALL, TAG_AIR, TAG_D1, TAG_D2, TAG_D3 = 1, 2, 3, 4, 5, 6
B_LEFT, B_RIGHT, B_BOTTOM, B_TOP = 11, 12, 13, 14


# ------------------------------------------------------------------ geometry
def build_mesh(x1, x2, order=2):
    """Fragment all regions -> conforming multi-material mesh with tags."""
    gmsh.initialize()
    gmsh.model.add("milestone0")
    occ = gmsh.model.occ
    wi = P["w_in"]
    r_tbl = occ.addRectangle(P["x_tbl"], 0, 0, -P["x_tbl"], 1.0)
    r_wall = occ.addRectangle(0.0, 0.0, 0, 1.0, 1.0)
    r_air = occ.addRectangle(wi, wi, 0, 1 - 2 * wi, 1 - 2 * wi)
    r_d1 = occ.addRectangle(x1 - P["d1"] / 2, wi, 0, P["d1"], P["d1"])
    r_d2 = occ.addRectangle(x2 - P["d2"] / 2, 1 - wi - P["d2"], 0, P["d2"], P["d2"])
    r_d3 = occ.addDisk(P["c3"][0], P["c3"][1], 0, P["R3"], P["R3"])

    # Boolean to non-overlapping regions first:
    #   wall frame = wall rect \ cavity;  air = cavity \ devices
    wall_frame, _ = occ.cut([(2, r_wall)], [(2, r_air)],
                            removeObject=True, removeTool=False)
    air_cut, _ = occ.cut([(2, r_air)], [(2, r_d1), (2, r_d2), (2, r_d3)],
                         removeObject=True, removeTool=False)
    inputs = [(2, r_tbl), wall_frame[0], air_cut[0],
              (2, r_d1), (2, r_d2), (2, r_d3)]
    _, out = occ.fragment(inputs, [])   # out = per-input entity map
    occ.synchronize()

    # out[i] = entities produced from inputs[i] -> material tags
    mat_of_input = [TAG_TBL, TAG_WALL, TAG_AIR, TAG_D1, TAG_D2, TAG_D3]
    mat_entities = {t: [] for t in mat_of_input}

    def _as_dimtags(entry):
        # gmsh may return (dim, tag) tuples, flat [dim, tag, ...] lists or tags
        out_ = []
        if len(entry) == 0:
            return out_
        if isinstance(entry[0], (tuple, list)):
            for e in entry:
                out_.append((int(e[0]), int(e[1])))
        elif isinstance(entry[0], (int, np.integer)):
            if len(entry) == 2 and entry[0] in (1, 2, 3):
                out_.append((int(entry[0]), int(entry[1])))
            else:  # flat list of same-dim tags (all our inputs are 2D)
                out_.extend((2, int(t)) for t in entry)
        return out_

    for inp, mat in zip(out, mat_of_input):
        for (dim, tag) in _as_dimtags(inp):
            if dim == 2:
                mat_entities[mat].append(tag)
    assert all(len(v) > 0 for v in mat_entities.values()), mat_entities

    # classify boundary curves by bounding box (gmsh pads bboxes by ~1e-7)
    tol = 1e-6
    bnd = {B_LEFT: [], B_RIGHT: [], B_BOTTOM: [], B_TOP: []}
    for (dim, tag) in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmin - P["x_tbl"]) < tol and abs(xmax - P["x_tbl"]) < tol:
            bnd[B_LEFT].append(tag)
        elif abs(xmin - 1.0) < tol and abs(xmax - 1.0) < tol:
            bnd[B_RIGHT].append(tag)
        elif abs(ymin) < tol and abs(ymax) < tol:
            bnd[B_BOTTOM].append(tag)
        elif abs(ymin - 1.0) < tol and abs(ymax - 1.0) < tol:
            bnd[B_TOP].append(tag)

    # mesh size: fine inside devices, coarse elsewhere
    for (dim, tag) in gmsh.model.getEntities(0):
        x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        in_dev = (abs(x - x1) <= P["d1"] / 2 + 1e-9 and wi - 1e-9 <= y <= wi + P["d1"] + 1e-9) or \
                 (abs(x - x2) <= P["d2"] / 2 + 1e-9 and 1 - wi - P["d2"] - 1e-9 <= y <= 1 - wi + 1e-9) or \
                 ((x - P["c3"][0]) ** 2 + (y - P["c3"][1]) ** 2 <= (P["R3"] + 1e-9) ** 2)
        gmsh.model.mesh.setSize([(0, tag)],
                                P["h_device"] if in_dev else P["h_global"])

    for name, tags in [("tbl", [TAG_TBL]), ("wall", [TAG_WALL]), ("air", [TAG_AIR]),
                       ("dev1", [TAG_D1]), ("dev2", [TAG_D2]), ("dev3", [TAG_D3])]:
        gmsh.model.addPhysicalGroup(2, mat_entities[tags[0]], tags[0], name)
    for name, tag in [("left", B_LEFT), ("right", B_RIGHT),
                      ("bottom", B_BOTTOM), ("top", B_TOP)]:
        gmsh.model.addPhysicalGroup(1, bnd[tag], tag, name)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    try:
        from dolfinx.io.gmsh import model_to_mesh          # dolfinx >= 0.9
    except ImportError:                                    # pragma: no cover
        from dolfinx.io.gmshio import model_to_mesh
    data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return data.mesh, data.cell_tags, data.facet_tags


# -------------------------------------------------------------------- solve
def _eval_at(func, pts3):
    """Evaluate a dolfinx Function at (n,3) points; one cell per point."""
    tree = bb_tree(func.function_space.mesh, 2)
    links = compute_collisions_points(tree, pts3)
    arr, off = links.array, links.offsets
    cells = np.full(pts3.shape[0], -1, dtype=np.int32)
    for i in range(off.size - 1):
        if off[i + 1] > off[i]:
            cells[i] = arr[off[i]]
    if np.any(cells < 0):
        raise RuntimeError(f"{int(np.sum(cells < 0))} points outside mesh")
    return func.eval(pts3, cells)


def solve_layout(name, x1, x2, outdir, power_scale=1.0):
    domain, cell_tags, facet_tags = build_mesh(x1, x2)
    V = fem.functionspace(domain, ("Lagrange", 2))
    T, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # piecewise k and volumetric source on DG0
    Q = fem.functionspace(domain, ("DG", 0))
    k_fun, q_fun = fem.Function(Q), fem.Function(Q)
    k_map = {TAG_TBL: P["k_TBL"], TAG_WALL: P["k_Al"], TAG_AIR: P["k_f"],
             TAG_D1: P["k_Al"], TAG_D2: P["k_Al"], TAG_D3: P["k_Al"]}
    q_map = {TAG_D1: P["q1"] * power_scale, TAG_D2: P["q2"] * power_scale,
             TAG_D3: P["q3"] * power_scale,
             TAG_TBL: 0.0, TAG_WALL: 0.0, TAG_AIR: 0.0}
    for tag in k_map:
        cells = cell_tags.find(tag)
        k_fun.x.array[cells] = k_map[tag]
        q_fun.x.array[cells] = q_map[tag]

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    a = k_fun * ufl.dot(ufl.grad(T), ufl.grad(v)) * dx \
        + P["h_ext"] * T * v * ds(B_RIGHT)
    L = q_fun * v * dx + P["h_ext"] * P["T_inf"] * v * ds(B_RIGHT)

    # left Dirichlet
    left_facets = facet_tags.find(B_LEFT)
    left_dofs = fem.locate_dofs_topological(V, 1, left_facets)
    bc = fem.dirichletbc(fem.Constant(domain, P["T_cold"]), left_dofs, V)

    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="m0_",
                            petsc_options={
        "ksp_type": "cg", "pc_type": "gamg",
        "ksp_rtol": 1e-12, "ksp_atol": 1e-14})
    Th = problem.solve()
    Th.name = "T"

    # ------------------------------------------------------------- post-proc
    n = ufl.FacetNormal(domain)

    def flux(btag):
        return fem.assemble_scalar(fem.form(
            -k_fun * ufl.dot(ufl.grad(Th), n) * ds(btag))) * P["b"]

    Q_left, Q_right = flux(B_LEFT), flux(B_RIGHT)
    Q_top, Q_bottom = flux(B_TOP), flux(B_BOTTOM)
    Q_right_robin = fem.assemble_scalar(fem.form(
        P["h_ext"] * (Th - P["T_inf"]) * ds(B_RIGHT))) * P["b"]
    P_in = P["P_tot"] * power_scale
    bal_err = abs(P_in - Q_left - Q_right) / P_in

    # device T_max from DOF coordinates inside device bounds
    X = V.tabulate_dof_coordinates()
    Tv = Th.x.array
    dev_bounds = {
        "dev1": lambda p: (np.abs(p[:, 0] - x1) <= P["d1"] / 2 + 1e-9)
                          & (p[:, 1] >= P["w_in"] - 1e-9)
                          & (p[:, 1] <= P["w_in"] + P["d1"] + 1e-9),
        "dev2": lambda p: (np.abs(p[:, 0] - x2) <= P["d2"] / 2 + 1e-9)
                          & (p[:, 1] >= 1 - P["w_in"] - P["d2"] - 1e-9)
                          & (p[:, 1] <= 1 - P["w_in"] + 1e-9),
        "dev3": lambda p: ((p[:, 0] - P["c3"][0]) ** 2
                           + (p[:, 1] - P["c3"][1]) ** 2) <= (P["R3"] + 1e-9) ** 2,
    }
    T_max, T_max_loc = {}, {}
    for dev, sel in dev_bounds.items():
        idx = np.where(sel(X))[0]
        i = idx[np.argmax(Tv[idx])]
        T_max[dev] = float(Tv[i])
        T_max_loc[dev] = [float(X[i, 0]), float(X[i, 1])]

    # right wall mean temperature
    A_right = fem.assemble_scalar(fem.form(1.0 * ds(B_RIGHT)))
    T_right_mean = fem.assemble_scalar(fem.form(Th * ds(B_RIGHT))) / A_right
    # Robin residual L2 norm (natural BC -> should be ~0)
    robin_res = np.sqrt(fem.assemble_scalar(fem.form(
        (k_fun * ufl.dot(ufl.grad(Th), n)
         + P["h_ext"] * (Th - P["T_inf"])) ** 2 * ds(B_RIGHT))))
    # aerogel 1D-equivalence metrics: mean T on x=0 (aerogel|wall interface)
    ys = np.linspace(1e-4, 1 - 1e-4, 201)
    pts = np.zeros((ys.size, 3)); pts[:, 0] = 0.0; pts[:, 1] = ys
    T_at_x0 = _eval_at(Th, pts).mean()
    dT_tbl = float(T_at_x0 - P["T_cold"])              # drop across aerogel
    q_left_pp = Q_left / P["b"] / 1.0                   # per m^2
    Rpp_TBL = -P["x_tbl"] / P["k_TBL"]
    dT_tbl_1d = q_left_pp * Rpp_TBL                     # 1D resistance prediction

    result = dict(
        layout=name, x1=x1, x2=x2, power_scale=power_scale,
        n_cells=domain.topology.index_map(2).size_local,
        n_dofs=V.dofmap.index_map.size_local,
        T_max_K=T_max, T_max_C={k: v - 273.15 for k, v in T_max.items()},
        T_max_loc=T_max_loc,
        feasible_all_below_70C=all(v <= P["T_lim"] for v in T_max.values()),
        Q_left_W=Q_left, Q_right_W=Q_right,
        Q_right_robin_W=Q_right_robin,
        Q_top_W=Q_top, Q_bottom_W=Q_bottom,
        energy_balance_err=bal_err,
        T_right_wall_mean_K=float(T_right_mean),
        robin_residual_L2=float(robin_res),
        aerogel=dict(dT_K=dT_tbl, q_left_Wm2=float(q_left_pp),
                     Rpp=Rpp_TBL, dT_1D_pred=float(dT_tbl_1d),
                     rel_err_1D=float(abs(dT_tbl - dT_tbl_1d) / max(dT_tbl, 1e-12))),
    )

    os.makedirs(outdir, exist_ok=True)
    suffix = name if power_scale == 1.0 else f"{name}_ps{power_scale:g}"
    with open(os.path.join(outdir, f"summary_{suffix}.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # export field for PINN validation: uniform grid CSV (x, y, T)
    nx, ny = 251, 251
    gx = np.linspace(P["x_tbl"], 1.0, nx)
    gy = np.linspace(0.0, 1.0, ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.vstack([GX.ravel(), GY.ravel(), np.zeros(GX.size)]).T
    Tg = _eval_at(Th, pts)
    csv_path = os.path.join(outdir, f"Tfield_{suffix}.csv")
    np.savetxt(csv_path, np.column_stack([GX.ravel(), GY.ravel(), Tg]),
               delimiter=",", header="x,y,T_K", comments="")

    # xdmf for paraview
    try:
        from dolfinx.io import XDMFFile
        with XDMFFile(MPI.COMM_WORLD,
                      os.path.join(outdir, f"T_{suffix}.xdmf"), "w") as xf:
            xf.write_mesh(domain)
            xf.write_function(Th)
    except Exception as e:                                # pragma: no cover
        print("xdmf export skipped:", e)

    return result


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "results/milestone0"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(LAYOUTS)
    power_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    all_results = {}
    for name in only:
        x1, x2 = LAYOUTS[name]["x1"], LAYOUTS[name]["x2"]
        print(f"\n=== layout {name}: x1={x1}, x2={x2}, "
              f"power_scale={power_scale} ===", flush=True)
        res = solve_layout(name, x1, x2, outdir, power_scale)
        suffix = name if power_scale == 1.0 else f"{name}_ps{power_scale:g}"
        all_results[suffix] = res
        print(json.dumps({k: res[k] for k in
                          ["T_max_C", "feasible_all_below_70C", "Q_left_W",
                           "Q_right_W", "energy_balance_err",
                           "T_right_wall_mean_K", "aerogel"]},
                         indent=2, ensure_ascii=False), flush=True)
    with open(os.path.join(outdir, "all_layouts.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
