"""Render a headless temperature field from a Milestone-1 checkpoint."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
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

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    finite = image[np.isfinite(image)]
    shown = ax.imshow(image, origin="lower", extent=(0, 1, 0, 1),
                      cmap="inferno", aspect="equal",
                      vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(shown, ax=ax, label="Temperature [°C]")

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
    ax.set(title=(f"PINN temperature | epoch {checkpoint['epoch']} "
                  f"({checkpoint.get('phase', 'unknown')})"),
           xlabel="x [m]", ylabel="y [m]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())
    print(f"range_C={finite.min():.6f},{finite.max():.6f} x1={x1f:.6f} x2={x2f:.6f}")


if __name__ == "__main__":
    main()
