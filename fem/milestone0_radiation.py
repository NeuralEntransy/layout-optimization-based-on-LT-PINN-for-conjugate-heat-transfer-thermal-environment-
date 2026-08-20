# -*- coding: utf-8 -*-
"""
Milestone 0 + radiation: deterministic FEM reference with surface-to-surface
radiation added to the stagnant-air conduction (doc 04 sections 3.1 + 3.4).

Same conduction model as milestone0_reference.py; in addition the cavity
enclosure (device exposed faces + inner wall faces) exchanges gray-diffuse
radiation (eps = 0.8, air non-participating).  Radiation enters the
conduction solve as an interior-facet surface source

    solid-conducted = air-conducted + q_rad      on the cavity boundary,

implemented as  - integral_cavity q_rad * avg(v) dS  in the load, with q_rad
from the radiosity system (Picard / fixed-point iteration over T^4).

Run with the fenicsx env:
    ~/miniforge/envs/fenicsx-env/bin/python fem/milestone0_radiation.py
"""
import json
import os
import sys

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem

from milestone0_reference import (build_mesh, P, LAYOUTS, _eval_at,
                                  TAG_TBL, TAG_WALL, TAG_AIR,
                                  TAG_D1, TAG_D2, TAG_D3,
                                  B_LEFT, B_RIGHT, B_BOTTOM, B_TOP)
import radiation as rad

CAVITY = 99
EPS = 0.8


# --------------------------------------------------------------------------- #
def solve_conduction(domain, cell_tags, ftags, hfun, sfun, V, right_bc):
    """One linear conduction solve with the radiation surface flux applied as
        q_applied,i = q_rad,i(T_k) + h_loc,i (T_i - T_k,i),
        h_loc,i = 4 eps sigma T_k,i^3   (local isolated-surface sensitivity)
    where q_rad(T_k) is the EXACT radiosity net flux at the current iterate
    (net ~0 over the closed enclosure).  The implicit h_loc*T term goes to
    the lhs (a += hfun T v dS), the constant (h_loc*T_k - q_rad) to the rhs
    (L += sfun v dS).  At T = T_k the applied flux equals the exact radiosity
    flux -> energy is conserved at convergence (needs tight surface tol).
    right_bc: 'robin' (h_ext convection to 30 C) or 'dirichlet' (clamp 30 C).
    """
    T, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    Q = fem.functionspace(domain, ("DG", 0))
    k_fun, q_fun = fem.Function(Q), fem.Function(Q)
    k_map = {TAG_TBL: P["k_TBL"], TAG_WALL: P["k_Al"], TAG_AIR: P["k_f"],
             TAG_D1: P["k_Al"], TAG_D2: P["k_Al"], TAG_D3: P["k_Al"]}
    q_map = {TAG_D1: P["q1"], TAG_D2: P["q2"], TAG_D3: P["q3"],
             TAG_TBL: 0.0, TAG_WALL: 0.0, TAG_AIR: 0.0}
    for tag in k_map:
        cells = cell_tags.find(tag)
        k_fun.x.array[cells] = k_map[tag]
        q_fun.x.array[cells] = q_map[tag]

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ftags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ftags)

    a = k_fun * ufl.dot(ufl.grad(T), ufl.grad(v)) * dx \
        + hfun * ufl.avg(T) * ufl.avg(v) * dS(CAVITY)
    L = q_fun * v * dx + sfun * ufl.avg(v) * dS(CAVITY)

    left_facets = ftags.find(B_LEFT)
    left_dofs = fem.locate_dofs_topological(V, 1, left_facets)
    bcs = [fem.dirichletbc(fem.Constant(domain, P["T_cold"]), left_dofs, V)]

    if right_bc == "robin":
        a += P["h_ext"] * T * v * ds(B_RIGHT)
        L += P["h_ext"] * P["T_inf"] * v * ds(B_RIGHT)
    else:  # dirichlet: clamp right wall outer face to T_inf
        right_facets = ftags.find(B_RIGHT)
        right_dofs = fem.locate_dofs_topological(V, 1, right_facets)
        bcs.append(fem.dirichletbc(fem.Constant(domain, P["T_inf"]),
                                   right_dofs, V))

    problem = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="m0r_",
                            petsc_options={"ksp_type": "cg", "pc_type": "gamg",
                                           "ksp_rtol": 1e-12,
                                           "ksp_atol": 1e-14})
    return problem.solve()


def solve_layout_radiation(name, x1, x2, outdir, picard_tol=1e-3,
                           max_iter=400, right_bc="robin"):
    domain, cell_tags, facet_tags = build_mesh(x1, x2)
    V = fem.functionspace(domain, ("Lagrange", 2))
    V1 = fem.functionspace(domain, ("Lagrange", 1))

    # cavity facets + combined facet tags (outer boundaries + cavity)
    cavity = rad.extract_cavity_facets(domain, cell_tags, TAG_AIR)
    ft_indices = list(facet_tags.indices) + list(cavity["facet_indices"])
    ft_values = list(facet_tags.values) + [CAVITY] * len(cavity["facet_indices"])
    order = np.argsort(np.array(ft_indices))
    ft = dmesh.meshtags(domain, 1, np.array(ft_indices)[order],
                        np.array(ft_values)[order])

    # devices for occlusion (rects use (xmin,xmax,ymin,ymax,tag))
    wi = P["w_in"]
    devices = dict(
        rects=[(x1 - P["d1"] / 2, x1 + P["d1"] / 2, wi, wi + P["d1"], TAG_D1),
               (x2 - P["d2"] / 2, x2 + P["d2"] / 2, 1 - wi - P["d2"],
                1 - wi, TAG_D2)],
        disk=(P["c3"][0], P["c3"][1], P["R3"], TAG_D3))

    # view factors + enclosure checks (computed ONCE per layout)
    F, info = rad.compute_view_factors(cavity, devices)
    checks = rad.enclosure_checks(F, cavity["length"] * P["b"])
    print(f"[{name}] cavity facets: {len(cavity['mid'])}, "
          f"visible pairs: {info['visible_pairs']}, "
          f"closure max err: {checks['closure_max_err']:.2e}, "
          f"reciprocity rel err: {checks['reciprocity_rel_err']:.2e}",
          flush=True)

    # nodal mapping: vertex dof <- adjacent facet values
    coords1 = V1.tabulate_dof_coordinates()[:, :2]
    key2dof = {(round(float(c[0]), 9), round(float(c[1]), 9)): i
               for i, c in enumerate(coords1)}
    facet_dofs = []
    for i in range(len(cavity["mid"])):
        d1 = key2dof[(round(float(cavity["p1"][i, 0]), 9),
                      round(float(cavity["p1"][i, 1]), 9))]
        d2 = key2dof[(round(float(cavity["p2"][i, 0]), 9),
                      round(float(cavity["p2"][i, 1]), 9))]
        facet_dofs.append((d1, d2))

    qfun_h = fem.Function(V1)   # implicit radiation coefficient h_loc
    qfun_s = fem.Function(V1)   # source h_loc*T_k - q_rad(T_k)
    qfun_h.x.array[:] = 0.0
    qfun_s.x.array[:] = 0.0

    def set_nodal(fun, values_per_facet):
        acc = np.zeros(V1.dofmap.index_map.size_local)
        cnt = np.zeros_like(acc)
        for (d1, d2), val in zip(facet_dofs, values_per_facet):
            acc[d1] += val; acc[d2] += val
            cnt[d1] += 1;  cnt[d2] += 1
        mask = cnt > 0
        fun.x.array[:] = 0.0
        fun.x.array[mask] = acc[mask] / cnt[mask]

    # ---------------- fixed-point iteration over the T^4 nonlinearity
    # radiation reference = EXACT radiosity (conservative net ~0); implicit
    # local term h_loc = 4 eps sigma T^3 stabilizes; converge tightly on the
    # surface temperature so the applied flux matches the exact one.
    Th = solve_conduction(domain, cell_tags, ft, qfun_h, qfun_s, V, right_bc)
    history = []
    converged = False
    q_applied = np.zeros(len(cavity["mid"]))
    Tm_prev = None
    for it in range(1, max_iter + 1):
        Tm = _eval_at(Th, np.column_stack([cavity["mid"],
                                           np.zeros(len(cavity["mid"]))]))
        Tm = np.asarray(Tm, dtype=float).ravel()
        Tm = np.maximum(Tm, 1.0)             # guard against nonphysical cold
        q_rad, J, G = rad.radiosity(Tm, F, EPS)
        q_applied = q_rad

        h_loc = 4.0 * EPS * rad.SIGMA * Tm ** 3
        s_loc = h_loc * Tm - q_rad
        set_nodal(qfun_h, h_loc)
        set_nodal(qfun_s, s_loc)

        Th = solve_conduction(domain, cell_tags, ft, qfun_h, qfun_s, V,
                              right_bc)

        X = V.tabulate_dof_coordinates()
        tmax = device_tmax(x1, x2, X, Th.x.array)
        history.append(dict(iter=it, **{k: v for k, v in tmax.items()}))
        dmax = (np.abs(Tm - Tm_prev).max() if Tm_prev is not None
                else np.inf)
        Tm_prev = Tm
        if it % 10 == 0 or it < 4:
            qrad_net = float((q_rad * cavity["length"] * P["b"]).sum())
            print(f"[{name}] it {it}: Tmax dev1 {tmax['dev1']-273.15:.1f}C "
                  f"dev2 {tmax['dev2']-273.15:.1f}C dev3 {tmax['dev3']-273.15:.1f}C "
                  f"| dTsurf {dmax:.3f}K | q_rad net {qrad_net:.2e} W",
                  flush=True)
        if dmax < picard_tol:
            converged = True
            print(f"[{name}] converged at iter {it} (surface dT={dmax:.2e})",
                  flush=True)
            break

    # ---------------- final diagnostics
    n = ufl.FacetNormal(domain)
    Q = fem.functionspace(domain, ("DG", 0))
    k_fun = fem.Function(Q)
    for tag, kv in {TAG_TBL: P["k_TBL"], TAG_WALL: P["k_Al"],
                    TAG_AIR: P["k_f"], TAG_D1: P["k_Al"], TAG_D2: P["k_Al"],
                    TAG_D3: P["k_Al"]}.items():
        k_fun.x.array[cell_tags.find(tag)] = kv
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    def flux(btag):
        return float(fem.assemble_scalar(fem.form(
            -k_fun * ufl.dot(ufl.grad(Th), n) * ds(btag))) * P["b"])

    Q_left, Q_right = flux(B_LEFT), flux(B_RIGHT)
    # net radiation over the closed enclosure must be ~0 (redistribution only)
    Q_rad_total = float((q_applied * cavity["length"] * P["b"]).sum())
    P_in = P["P_tot"]
    bal = abs(P_in - Q_left - Q_right) / P_in

    X = V.tabulate_dof_coordinates()
    tmax = device_tmax(x1, x2, X, Th.x.array)
    result = dict(
        layout=name, x1=x1, x2=x2, radiation=True, eps=EPS,
        right_bc=right_bc,
        n_cavity_facets=len(cavity["mid"]),
        view_factor_checks=checks,
        T_max_C={k: v - 273.15 for k, v in tmax.items()},
        T_max_K=tmax,
        feasible_all_below_70C=all(v <= P["T_lim"] for v in tmax.values()),
        Q_left_W=Q_left, Q_right_W=Q_right,
        Q_rad_total_W=Q_rad_total,
        energy_balance_err=bal,
        picard_iters=len(history), picard_converged=converged,
        T_history_C=[{k: v - 273.15 for k, v in h.items() if k != "iter"}
                     for h in history],
    )
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"summary_rad_{name}.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # field CSV for comparison
    nx = ny = 251
    gx = np.linspace(P["x_tbl"], 1.0, nx)
    gy = np.linspace(0.0, 1.0, ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.vstack([GX.ravel(), GY.ravel(), np.zeros(GX.size)]).T
    Tg = _eval_at(Th, pts)
    np.savetxt(os.path.join(outdir, f"Tfield_rad_{name}.csv"),
               np.column_stack([GX.ravel(), GY.ravel(), Tg]),
               delimiter=",", header="x,y,T_K", comments="")
    return result


def device_tmax(x1, x2, X, Tv):
    dev_bounds = {
        "dev1": lambda p: (np.abs(p[:, 0] - x1) <= P["d1"] / 2 + 1e-9)
                          & (p[:, 1] >= P["w_in"] - 1e-9)
                          & (p[:, 1] <= P["w_in"] + P["d1"] + 1e-9),
        "dev2": lambda p: (np.abs(p[:, 0] - x2) <= P["d2"] / 2 + 1e-9)
                          & (p[:, 1] >= 1 - P["w_in"] - P["d2"] - 1e-9)
                          & (p[:, 1] <= 1 - P["w_in"] + 1e-9),
        "dev3": lambda p: ((p[:, 0] - P["c3"][0]) ** 2
                           + (p[:, 1] - P["c3"][1]) ** 2)
                          <= (P["R3"] + 1e-9) ** 2,
    }
    out = {}
    for dev, sel in dev_bounds.items():
        idx = np.where(sel(X))[0]
        out[dev] = float(Tv[idx].max())
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "results/milestone0_radiation"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(LAYOUTS)
    right_bc = sys.argv[3] if len(sys.argv) > 3 else "robin"
    all_results = {}
    for name in only:
        x1, x2 = LAYOUTS[name]["x1"], LAYOUTS[name]["x2"]
        print(f"\n=== layout {name} + radiation: x1={x1}, x2={x2}, "
              f"right_bc={right_bc} ===", flush=True)
        res = solve_layout_radiation(name, x1, x2, outdir, right_bc=right_bc)
        all_results[name] = res
        print(json.dumps({k: res[k] for k in
                          ["T_max_C", "feasible_all_below_70C", "Q_left_W",
                           "Q_right_W", "Q_rad_total_W", "energy_balance_err",
                           "picard_iters", "view_factor_checks"]},
                         indent=2, ensure_ascii=False), flush=True)
    with open(os.path.join(outdir, "all_layouts.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
