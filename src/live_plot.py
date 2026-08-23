# -*- coding: utf-8 -*-
"""Non-blocking live temperature-field plot used during PINN training."""
import os

import numpy as np
import torch

import config as C
import geometry as G


class LiveTemperaturePlot:
    """Assemble domain branches on a grid and refresh one Matplotlib window."""

    def __init__(self, outdir, resolution=151):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.outpath = os.path.join(outdir, "temperature_live.png")
        self.resolution = resolution
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7.2, 6.0))
        self.cbar = None
        self.fig.show()

    @torch.no_grad()
    def update(self, field, x1, x2, device, step_label, loss=None,
               power_scale=1.0):
        import matplotlib.patches as patches

        axis = torch.linspace(0.0, 1.0, self.resolution, device=device)
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")
        points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        labels = G.label_points(points, x1, x2)
        temperature = torch.full((points.shape[0],), torch.nan,
                                 device=device)
        for domain in G.DOMAINS:
            mask = labels == G.LABEL_ORDER.index(domain)
            if mask.any():
                temperature[mask] = field.temperature_K(
                    domain, points[mask]).squeeze(1)

        image_data = temperature.reshape(self.resolution,
                                         self.resolution).cpu().numpy()
        image_data = image_data - 273.15
        finite = np.isfinite(image_data)
        if not finite.any():
            return

        self.ax.clear()
        image = self.ax.imshow(
            image_data, origin="lower", extent=(0, 1, 0, 1),
            cmap="inferno", aspect="equal",
            vmin=float(np.nanmin(image_data)),
            vmax=float(np.nanmax(image_data)))
        if self.cbar is None:
            self.cbar = self.fig.colorbar(image, ax=self.ax,
                                          label="Temperature [°C]")
        else:
            self.cbar.update_normal(image)

        x1f, x2f = float(x1), float(x2)
        self.ax.add_patch(patches.Rectangle(
            (x1f - C.D1 / 2, C.DEV1_Y[0]), C.D1, C.D1,
            fill=False, edgecolor="cyan", linewidth=1.4))
        self.ax.add_patch(patches.Rectangle(
            (x2f - C.D2 / 2, C.DEV2_Y[0]), C.D2, C.D2,
            fill=False, edgecolor="cyan", linewidth=1.4))
        self.ax.add_patch(patches.Circle(
            C.C3, C.R3, fill=False, edgecolor="cyan", linewidth=1.4))
        self.ax.add_patch(patches.Rectangle(
            (C.W_IN, C.W_IN), 1 - 2 * C.W_IN, 1 - 2 * C.W_IN,
            fill=False, edgecolor="white", linewidth=0.7, alpha=0.8))

        loss_text = "" if loss is None else f"  loss={float(loss):.3e}"
        self.ax.set_title(
            f"PINN temperature — {step_label}{loss_text}  "
            f"power={power_scale:g}")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.savefig(self.outpath, dpi=150)
        self.plt.pause(0.001)

