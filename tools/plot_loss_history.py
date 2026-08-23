"""Plot the current Milestone-1 training history from loss_log.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "checkpoint_m1" / "center_v4_dirichlet" / "loss_log.csv"
OUT = ROOT / "results" / "milestone1" / "center_v4_dirichlet" / "loss_curves.png"


def main() -> None:
    data = pd.read_csv(CSV)
    adam = data[data["lr"] >= 0].sort_values("epoch")
    lbfgs = data[data["lr"] < 0].sort_values("epoch")

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
    late = adam[adam["epoch"] >= 30000]
    ax.plot(late["epoch"], late["loss"], label="total", lw=1.3)
    ax.plot(late["epoch"], 10 * late["if_q"], label="10 x if_q", lw=1.1)
    ax.plot(late["epoch"], 30 * late["eng"], label="30 x eng", lw=1.1)
    ax.set(title="Refinement stage zoom", xlabel="epoch", ylabel="weighted loss")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(adam["epoch"], adam["Q_left"], label="Q_left", lw=1.1)
    ax.plot(adam["epoch"], adam["Q_right"], label="Q_right", lw=1.1)
    ax.plot(adam["epoch"], adam["Q_left"] + adam["Q_right"], label="Q_left + Q_right", lw=1.2)
    ax.axhline(324.8728662343759, color="C0", ls="--", alpha=0.7, label="FEM Q_left")
    ax.axhline(506.55670618121286, color="C1", ls="--", alpha=0.7, label="FEM Q_right")
    ax.axhline(830.0, color="k", ls=":", alpha=0.8, label="P_in = 830 W")
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
    fig.suptitle(
        f"center_v4_dirichlet | {len(data)} records | latest Adam epoch {int(last['epoch'])} | "
        f"loss {last['loss']:.3g}", fontsize=14
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)
    print(
        f"latest epoch={int(last['epoch'])} loss={last['loss']:.6g} "
        f"Q_left={last['Q_left']:.3f} Q_right={last['Q_right']:.3f} "
        f"balance={last['balance_err']:.6g}"
    )


if __name__ == "__main__":
    main()
