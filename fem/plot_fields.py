# -*- coding: utf-8 -*-
"""Plot FEM temperature fields of the three milestone-0 layouts."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1] if len(sys.argv) > 1 else "results/milestone0_v2"
prefix = sys.argv[3] if len(sys.argv) > 3 else "Tfield_"
layouts = sys.argv[2].split(",") if len(sys.argv) > 2 else ["left", "center",
                                                            "right"]

fig, axes = plt.subplots(1, len(layouts), figsize=(6.4 * len(layouts), 5.2),
                         constrained_layout=True)
if len(layouts) == 1:
    axes = [axes]
for ax, name in zip(axes, layouts):
    d = np.loadtxt(os.path.join(outdir, f"{prefix}{name}.csv"),
                   delimiter=",", skiprows=1)
    nx = ny = int(round(np.sqrt(len(d))))
    X = d[:, 0].reshape(nx, ny)
    Y = d[:, 1].reshape(nx, ny)
    T = d[:, 2].reshape(nx, ny) - 273.15
    # fixed color card: -55 C (cold wall) .. 70 C (device limit)
    cf = ax.pcolormesh(X, Y, T, cmap="inferno", shading="auto",
                       vmin=-55, vmax=70)
    ax.contour(X, Y, T, levels=[70], colors="cyan", linewidths=1.2)
    ax.set_aspect("equal")
    ax.set_title(f"layout {name}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(cf, ax=ax, label="T [°C]", shrink=0.85)
fig.suptitle("Milestone 0 FEM reference\n"
             "(color card clipped to [-55, 70] °C; cyan contour = 70 °C limit)")
fig.savefig(os.path.join(outdir, f"{prefix}all_layouts.png"), dpi=300)
print("saved", os.path.join(outdir, f"{prefix}all_layouts.png"))
