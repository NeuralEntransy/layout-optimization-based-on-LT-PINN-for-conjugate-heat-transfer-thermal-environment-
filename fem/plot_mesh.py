# -*- coding: utf-8 -*-
"""Visualize the milestone-0 FEM mesh (material subdomains + boundaries).

Rebuilds the gmsh mesh via milestone0_reference.build_mesh and draws
triangle edges colored by material tag, plus tagged boundary curves.

Run with the fenicsx conda env:
    ~/miniforge/envs/fenicsx-env/bin/python fem/plot_mesh.py [left,center,right]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from milestone0_reference import (build_mesh, LAYOUTS,
                                  TAG_TBL, TAG_WALL, TAG_AIR,
                                  TAG_D1, TAG_D2, TAG_D3,
                                  B_LEFT, B_RIGHT, B_BOTTOM, B_TOP)

MAT_STYLE = {
    TAG_TBL:  ("aerogel (TBL)",  "#9ecae1"),
    TAG_WALL: ("Al wall frame",  "#c7c7c7"),
    TAG_AIR:  ("stagnant air",   "#f7f7f7"),
    TAG_D1:   ("device 1",       "#fb6a4a"),
    TAG_D2:   ("device 2",       "#cb181d"),
    TAG_D3:   ("device 3",       "#fd8d3c"),
}
BND_STYLE = {
    B_LEFT:   ("left: Dirichlet -55 °C", "#08519c"),
    B_RIGHT:  ("right: Robin h=20",      "#006d2c"),
    B_BOTTOM: ("bottom: adiabatic",      "#636363"),
    B_TOP:    ("top: adiabatic",         "#636363"),
}

layouts = sys.argv[1].split(",") if len(sys.argv) > 1 else list(LAYOUTS)
outdir = "results/milestone0_mesh"
os.makedirs(outdir, exist_ok=True)

for name in layouts:
    x1, x2 = LAYOUTS[name]["x1"], LAYOUTS[name]["x2"]
    domain, cell_tags, facet_tags = build_mesh(x1, x2, order=1)
    domain.topology.create_connectivity(2, 0)
    domain.topology.create_connectivity(1, 0)
    domain.topology.create_connectivity(2, 1)

    geom = domain.geometry.x[:, :2]
    cells = domain.topology.connectivity(2, 0).array.reshape(-1, 3)
    cvals = cell_tags.values                       # material tag per cell
    edges_c2e = domain.topology.connectivity(2, 1)
    e2v = domain.topology.connectivity(1, 0).array.reshape(-1, 2)
    fmap = domain.topology.index_map(1)
    nfacets = fmap.size_local
    edge_of_cell = edges_c2e.array.reshape(-1, 3)

    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)

    # one LineCollection per material -> colored wireframe
    for tag, (label, color) in MAT_STYLE.items():
        idx = np.where(cvals == tag)[0]
        if idx.size == 0:
            continue
        segs = []
        for c in idx:
            for e in edge_of_cell[c]:
                if e < nfacets:
                    segs.append(geom[e2v[e]])
        ax.add_collection(LineCollection(segs, colors=color,
                                         linewidths=0.4, zorder=1))
    # boundary curves on top, thick
    fvals = facet_tags.values
    fidx = facet_tags.indices
    for tag, (label, color) in BND_STYLE.items():
        segs = [geom[e2v[f]] for f, v in zip(fidx, fvals) if v == tag]
        if segs:
            ax.add_collection(LineCollection(segs, colors=color,
                                             linewidths=2.0, zorder=3))

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Milestone 0 FEM mesh — layout '{name}'  "
                 f"({cells.shape[0]} triangles)")
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for l, c in MAT_STYLE.values()]
    handles += [Line2D([0], [0], color=c, lw=2, label=l)
                for l, c in BND_STYLE.values()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, frameon=False)

    path = os.path.join(outdir, f"mesh_{name}.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print("saved", path, f"| cells={cells.shape[0]}, nodes={geom.shape[0]}")
