"""Render a headless temperature field from a Milestone-1 checkpoint.

Supports optional filled colormap + overlaid temperature contours, or a
dedicated contour-line-only panel, to help judge whether the predicted
thermal field follows the expected physics (isotherm spacing, hot-spot
location, 70 °C device-limit boundary, etc.)."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
import geometry as G  # noqa: E402
from networks import TemperatureField  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=401)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--levels", type=str, default=None,
                        help="comma-separated contour levels in °C "
                             "(default: auto, ~12 levels between T_min and T_max)")
    parser.add_argument("--no-fill", action="store_true",
                        help="draw contour lines only, skip the filled colormap")
    parser.add_argument("--side-by-side", action="store_true",
                        help="add a second panel showing contour lines only")
    parser.add_argument("--clip-vmin", type=float, default=None,
                        help="lower clip for the color scale [°C]")
    parser.add_argument("--clip-vmax", type=float, default=None,
                        help="upper clip for the color scale [°C]")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    saved = checkpoint["args"]
    field = TemperatureField(
        saved["width"], saved["depth"], theta_init=saved["theta_init"],
        fourier_sigma=saved["fourier_sigma"],
        fourier_dim=saved["fourier_dim"]).to(device)
    field.load_state_dict(checkpoint["field"])
    field.eval()

    layout = saved.get("layout", "center")
    x10, x20 = C.LAYOUTS[layout]
    design = G.DesignVars(x10, x20, trainable=False, device=device)
    with torch.no_grad():
        design.z1.fill_(checkpoint["design"]["z1"])
        design.z2.fill_(checkpoint["design"]["z2"])
        x1, x2 = design.x1(), design.x2()

        axis = torch.linspace(0, 1, args.resolution, device=device)
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")
        points = torch.stack((gx.ravel(), gy.ravel()), dim=1)
        labels = G.label_points(points, x1, x2)
        temp_c = torch.full((points.shape[0],), torch.nan, device=device)
        for domain in G.DOMAINS:
            mask = labels == G.LABEL_ORDER.index(domain)
            if mask.any():
                temp_c[mask] = field.temperature_K(
                    domain, points[mask]).squeeze(1) - 273.15
        image = temp_c.reshape(args.resolution, args.resolution).cpu().numpy()
        X = gx.cpu().numpy()
        Y = gy.cpu().numpy()

    finite = image[np.isfinite(image)]
    t_min = float(finite.min())
    t_max = float(finite.max())
    t_lim_c = C.T_LIM - 273.15  # 70 °C device limit

    # resolve contour levels
    if args.levels:
        levels = [float(v.strip()) for v in args.levels.split(",")]
    else:
        levels = list(np.linspace(t_min, t_max, 12))
        if t_min < t_lim_c < t_max and not any(abs(v - t_lim_c) < 1e-6
                                                for v in levels):
            levels.append(t_lim_c)
        levels = sorted({round(v, 4) for v in levels})

    # build the figure: one panel by default, two if side-by-side requested
    if args.side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                                 constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
        axes = [ax]

    ax = axes[0]
    vmin = args.clip_vmin if args.clip_vmin is not None else t_min
    vmax = args.clip_vmax if args.clip_vmax is not None else t_max

    if not args.no_fill:
        shown = ax.imshow(image, origin="lower", extent=(0, 1, 0, 1),
                          cmap="inferno", aspect="equal",
                          vmin=vmin, vmax=vmax)
        fig.colorbar(shown, ax=ax, label="Temperature [°C]")

    # masked array so contour leaves gaps in undefined (outside-domain) cells
    masked = np.ma.masked_invalid(image)
    cs = ax.contour(X, Y, masked, levels=levels, colors="white",
                    linewidths=0.9, alpha=0.9)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f°C")

    # highlight the 70 °C device-limit isotherm
    if t_min < t_lim_c < t_max:
        ax.contour(X, Y, masked, levels=[t_lim_c], colors="cyan",
                   linewidths=2.2, linestyles="-")

    title_fill = "PINN temperature" if not args.no_fill else "PINN temperature contours"

    x1f, x2f = float(x1), float(x2)
    ax.add_patch(patches.Rectangle(
        (x1f - C.D1 / 2, C.DEV1_Y[0]), C.D1, C.D1,
        fill=False, edgecolor="cyan", linewidth=1.3))
    ax.add_patch(patches.Rectangle(
        (x2f - C.D2 / 2, C.DEV2_Y[0]), C.D2, C.D2,
        fill=False, edgecolor="cyan", linewidth=1.3))
    ax.add_patch(patches.Circle(
        C.C3, C.R3, fill=False, edgecolor="cyan", linewidth=1.3))
    ax.add_patch(patches.Rectangle(
        (C.W_IN, C.W_IN), 1 - 2 * C.W_IN, 1 - 2 * C.W_IN,
        fill=False, edgecolor="white", linewidth=0.7, alpha=0.8))
    ax.set(title=(f"{title_fill} | epoch {checkpoint['epoch']} "
                  f"({checkpoint.get('phase', 'unknown')})"),
           xlabel="x [m]", ylabel="y [m]")

    # optional second panel: line-only contours (best for tracing isotherms)
    if args.side_by_side:
        ax2 = axes[1]
        norm = Normalize(vmin=vmin, vmax=vmax)
        cs2 = ax2.contour(X, Y, masked, levels=levels, cmap="inferno",
                          norm=norm, linewidths=1.1)
        ax2.clabel(cs2, inline=True, fontsize=8, fmt="%.0f°C")
        if t_min < t_lim_c < t_max:
            ax2.contour(X, Y, masked, levels=[t_lim_c], colors="cyan",
                        linewidths=2.2, linestyles="-")
        ax2.set_aspect("equal")
        ax2.set(title=f"Contour lines | epoch {checkpoint['epoch']}",
                xlabel="x [m]", ylabel="y [m]")
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax2, label="Temperature [°C]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())
    print(f"range_C={t_min:.6f},{t_max:.6f} x1={x1f:.6f} x2={x2f:.6f}")
    print(f"contour levels [{len(levels)}]: "
          f"{', '.join(f'{v:.1f}' for v in levels)}")


if __name__ == "__main__":
    main()
