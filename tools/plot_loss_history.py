"""Plot the Milestone-1 training history from a loss_log.csv.

Supports arbitrary checkpoint/result directories and optional FEM reference
lines.  The default weights are read from src/config.py but can be overridden."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=ROOT / "checkpoint_m1" / "center_v4_dirichlet" /
                   "loss_log.csv",
                   help="path to loss_log.csv")
    p.add_argument("--output", type=Path, default=None,
                   help="output image path (default: next to the CSV)")
    p.add_argument("--title", type=str, default=None,
                   help="figure suptitle prefix")
    p.add_argument("--fem-q-left", type=float, default=None,
                   help="FEM reference Q_left [W]")
    p.add_argument("--fem-q-right", type=float, default=None,
                   help="FEM reference Q_right [W]")
    p.add_argument("--p-in", type=float, default=C.P_TOT,
                   help="total input power [W]")
    p.add_argument("--w-pde", type=float, default=C.TRAIN["w_pde"])
    p.add_argument("--w-if-T", type=float, default=C.TRAIN["w_if_T"])
    p.add_argument("--w-if-q", type=float, default=C.TRAIN["w_if_q"])
    p.add_argument("--w-bc", type=float, default=C.TRAIN["w_bc"])
    p.add_argument("--w-eng", type=float, default=C.TRAIN["w_eng"])
    p.add_argument("--zoom-epoch", type=int, default=None,
                   help="epoch threshold for the refinement-zoom panel")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    CSV = args.csv
    data = pd.read_csv(CSV)
    adam = data[data["lr"] >= 0].sort_values("epoch")
    lbfgs = data[data["lr"] < 0].sort_values("epoch")

    OUT = args.output or CSV.with_name("loss_curves.png")
    tag = CSV.parent.name  # e.g. center_v4_dirichlet

    weights = {
        "pde": args.w_pde,
        "if_T": args.w_if_T,
        "if_q": args.w_if_q,
        "bc": args.w_bc,
        "eng": args.w_eng,
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(adam["epoch"], adam["loss"], label="Adam", lw=1.5)
    if not lbfgs.empty:
        ax.semilogy(lbfgs["epoch"], lbfgs["loss"], ".-", label="old L-BFGS", alpha=0.75)
    ax.set(title="Weighted total loss", xlabel="epoch", ylabel="loss")
    ax.legend()

    ax = axes[0, 1]
    for col in ["pde", "if_T", "if_q", "bc", "eng"]:
        ax.semilogy(adam["epoch"], adam[col].clip(lower=1e-12), label=col, lw=1.2)
    ax.set(title="Raw loss components (Adam)", xlabel="epoch", ylabel="raw loss")
    ax.legend(ncol=2)

    ax = axes[1, 0]
    weights = {"pde": 1, "if_T": 10, "if_q": 10, "bc": 20, "eng": 30}
    for col, weight in weights.items():
        ax.semilogy(adam["epoch"], (weight * adam[col]).clip(lower=1e-12),
                    label=f"{weight} x {col}", lw=1.2)
    ax.set(title="Weighted component contributions", xlabel="epoch", ylabel="contribution")
    ax.legend(ncol=2)

    ax = axes[1, 1]
    zoom_epoch = args.zoom_epoch if args.zoom_epoch is not None \
        else int(adam["epoch"].max() * 0.5)
    late = adam[adam["epoch"] >= zoom_epoch]
    ax.semilogy(late["epoch"], late["loss"].clip(lower=1e-12),
                label="total", lw=1.3)
    ax.semilogy(late["epoch"], (weights["if_q"] * late["if_q"]).clip(lower=1e-12),
                label=f"{weights['if_q']} x if_q", lw=1.1)
    ax.semilogy(late["epoch"], (weights["eng"] * late["eng"]).clip(lower=1e-12),
                label=f"{weights['eng']} x eng", lw=1.1)
    ax.set(title=f"Refinement stage zoom (epoch >= {zoom_epoch})",
           xlabel="epoch", ylabel="weighted loss")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(adam["epoch"], adam["Q_left"], label="Q_left", lw=1.1)
    ax.plot(adam["epoch"], adam["Q_right"], label="Q_right", lw=1.1)
    ax.plot(adam["epoch"], adam["Q_left"] + adam["Q_right"],
            label="Q_left + Q_right", lw=1.2)
    if args.fem_q_left is not None:
        ax.axhline(args.fem_q_left, color="C0", ls="--", alpha=0.7,
                   label="FEM Q_left")
    if args.fem_q_right is not None:
        ax.axhline(args.fem_q_right, color="C1", ls="--", alpha=0.7,
                   label="FEM Q_right")
    ax.axhline(args.p_in, color="k", ls=":", alpha=0.8,
               label=f"P_in = {args.p_in:g} W")
    ax.set(title="Boundary heat flow", xlabel="epoch", ylabel="W")
    ax.legend(ncol=2)

    ax = axes[2, 1]
    ax.semilogy(adam["epoch"], adam["balance_err"].clip(lower=1e-8), label="balance error")
    ax.axhline(0.01, color="C3", ls="--", label="1% target")
    ax.set(title="Global energy balance", xlabel="epoch", ylabel="relative error")
    ax.legend()

    for ax in axes.flat:
        ax.grid(True, which="both", alpha=0.25)

    last = adam.iloc[-1]
    title = args.title or tag
    fig.suptitle(
        f"{title} | {len(data)} records | latest Adam epoch {int(last['epoch'])} | "
        f"loss {last['loss']:.3g}", fontsize=14
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=args.dpi)
    print(OUT)
    print(
        f"latest epoch={int(last['epoch'])} loss={last['loss']:.6g} "
        f"Q_left={last['Q_left']:.3f} Q_right={last['Q_right']:.3f} "
        f"balance={last['balance_err']:.6g}"
    )


if __name__ == "__main__":
    main()
