# -*- coding: utf-8 -*-
"""
Milestone 1 validation against the milestone-0 FEM reference.

Checks (doc 04, sections 6 and 7.2):
  1. mounting-interface temperature / flux continuity (PINN soft losses)
  2. aerogel layer degenerates to the 1D thermal-resistance result
  3. total energy balance  |P_in - Q_left - Q_right| / P_in
  4. pointwise agreement with the FEM temperature field (per-domain rel-L2)
  5. per-device T_max: PINN vs FEM

Usage:
    python src/validate.py --layout center
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import geometry as G
import monitors as M
from networks import TemperatureField

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="center", choices=list(C.LAYOUTS))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--fem", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--power-scale", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda:0" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu")

    x1, x2 = C.LAYOUTS[args.layout]
    tag = args.layout + (f"_ps{args.power_scale:g}"
                         if args.power_scale != 1.0 else "")
    ckpt = args.ckpt or os.path.join(ROOT, "checkpoint_m1", tag, "latest.pt")
    suffix = args.layout + (f"_ps{args.power_scale:g}"
                            if args.power_scale != 1.0 else "")
    fem_csv = args.fem or os.path.join(ROOT, "results", "milestone0_v2",
                                       f"Tfield_{suffix}.csv")
    fem_dir = os.path.dirname(os.path.abspath(fem_csv))
    fem_json = os.path.join(fem_dir, f"summary_{suffix}.json")
    outdir = args.outdir or os.path.join(ROOT, "results", "milestone1", tag)
    os.makedirs(outdir, exist_ok=True)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    field = TemperatureField(ck["args"]["width"], ck["args"]["depth"],
                             fourier_sigma=ck["args"].get("fourier_sigma"),
                             fourier_dim=ck["args"].get("fourier_dim", 64)
                             ).to(device)
    field.load_state_dict(ck["field"])
    field.eval()
    print(f"loaded {ckpt} (epoch {ck['epoch']})")

    # ---------------------------------------------------------- FEM comparison
    data = np.loadtxt(fem_csv, delimiter=",", skiprows=1)
    pts = torch.tensor(data[:, :2], dtype=torch.float32, device=device)
    T_fem = data[:, 2]
    labels = G.label_points(pts, x1, x2)
    T_pinn = torch.empty(pts.shape[0], device=device)
    with torch.no_grad():
        # aerogel points (label -1 under the 1-D model): linear profile from
        # T_cold to the wall left-face temperature theta(0, y)
        m_tbl = labels == -1
        if m_tbl.any():
            pts0 = pts[m_tbl].clone()
            pts0[:, 0] = 0.0
            th0 = field("wall_l", pts0)
            frac = (pts[m_tbl][:, 0:1] - C.X_TBL) / C.L_TBL
            T_pinn[m_tbl] = (C.T_COLD + C.DT * th0 * frac).squeeze(1)
        for dom in G.DOMAINS:
            i = G.LABEL_ORDER.index(dom)
            m = labels == i
            if not m.any():
                continue
            T_pinn[m] = (C.T_C + C.DT * field(dom, pts[m])).squeeze(1)
    T_pinn = T_pinn.cpu().numpy()

    per_domain, err_all = {}, None
    denom = np.sqrt(np.mean((T_fem - C.T_COLD) ** 2))
    all_doms = [d for d in G.DOMAINS]
    # include the aerogel (1-D model) points in the per-domain breakdown
    m_tbl_np = (labels.cpu().numpy() == -1)
    if m_tbl_np.any():
        e = T_pinn[m_tbl_np] - T_fem[m_tbl_np]
        per_domain["tbl_1D"] = dict(
            rel_L2=float(np.sqrt(np.mean(e ** 2)) / denom),
            max_abs_K=float(np.abs(e).max()), n=int(m_tbl_np.sum()))
    for dom in all_doms:
        i = G.LABEL_ORDER.index(dom)
        m = (labels.cpu().numpy() == i)
        if not m.any():
            continue
        e = T_pinn[m] - T_fem[m]
        per_domain[dom] = dict(
            rel_L2=float(np.sqrt(np.mean(e ** 2)) / denom),
            max_abs_K=float(np.abs(e).max()), n=int(m.sum()))
    e_all = T_pinn - T_fem
    global_cmp = dict(rel_L2=float(np.sqrt(np.mean(e_all ** 2)) / denom),
                      max_abs_K=float(np.abs(e_all).max()))

    # ---------------------------------------------------------- PINN metrics
    rep_pinn = M.full_report(field, x1, x2, device, args.power_scale)
    with open(fem_json) as f:
        rep_fem = json.load(f)

    dev_cmp = {}
    for dom in ("dev1", "dev2", "dev3"):
        dev_cmp[dom] = dict(
            pinn_K=rep_pinn["T_max_K"][dom],
            fem_K=rep_fem["T_max_K"][dom],
            diff_K=rep_pinn["T_max_K"][dom] - rep_fem["T_max_K"][dom])

    result = dict(
        layout=args.layout, x1=x1, x2=x2, power_scale=args.power_scale,
        ckpt=ckpt, ckpt_epoch=ck["epoch"],
        pinn=rep_pinn,
        fem_energy=dict(Q_left=rep_fem["Q_left_W"],
                        Q_right=rep_fem["Q_right_W"],
                        balance_err=rep_fem["energy_balance_err"]),
        device_Tmax=dev_cmp,
        field_cmp_global=global_cmp,
        field_cmp_per_domain=per_domain,
        acceptance=dict(
            energy_balance_lt_1pc=
            rep_pinn["energy"]["balance_err"] < 0.01,
            aerogel_1D_rel_err=rep_pinn["aerogel"]["rel_err"],
            all_below_70C=rep_pinn["feasible_70C"],
        ))
    with open(os.path.join(outdir, "validation.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    # (a) horizontal centerline y=0.5
    m = np.abs(data[:, 1] - 0.5) < 0.0021
    order = np.argsort(data[m, 0])
    axes[0].plot(data[m, 0][order], T_fem[m][order] - 273.15, "k-",
                 lw=1.6, label="FEM")
    axes[0].plot(data[m, 0][order], T_pinn[m][order] - 273.15, "r--",
                 lw=1.2, label="PINN")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("T [°C]")
    axes[0].set_title("centerline y = 0.5"); axes[0].legend(); axes[0].grid(alpha=.3)
    # (b) vertical line x=0.5
    m = np.abs(data[:, 0] - 0.5) < 0.0021
    order = np.argsort(data[m, 1])
    axes[1].plot(data[m, 1][order], T_fem[m][order] - 273.15, "k-", lw=1.6)
    axes[1].plot(data[m, 1][order], T_pinn[m][order] - 273.15, "r--", lw=1.2)
    axes[1].set_xlabel("y [m]"); axes[1].set_ylabel("T [°C]")
    axes[1].set_title("vertical x = 0.5"); axes[1].grid(alpha=.3)
    # (c) device T_max bars
    doms = ["dev1", "dev2", "dev3"]
    xp = np.arange(3)
    axes[2].bar(xp - 0.18, [rep_fem["T_max_C"][d] for d in doms], width=0.36,
                label="FEM")
    axes[2].bar(xp + 0.18, [rep_pinn["T_max_C"][d] for d in doms], width=0.36,
                label="PINN")
    axes[2].axhline(70, color="r", ls=":", label="70 °C limit")
    axes[2].set_xticks(xp, doms); axes[2].set_ylabel("T_max [°C]")
    axes[2].set_title("device T_max"); axes[2].legend(); axes[2].grid(alpha=.3)
    fig.suptitle(f"milestone 1 validation — layout {args.layout} "
                 f"(x1={x1}, x2={x2})")
    fig.tight_layout()
    fig_path = os.path.join(outdir, f"validation_{tag}.png")
    fig.savefig(fig_path, dpi=150)
    print("figure:", fig_path)

    print(json.dumps(result["acceptance"], indent=2))
    print("\ndevice T_max [°C]  PINN / FEM")
    for d in doms:
        print(f"  {d}: {rep_pinn['T_max_C'][d]:8.2f} / "
              f"{rep_fem['T_max_C'][d]:8.2f}")
    print(f"\nglobal rel L2: {global_cmp['rel_L2']:.3e}, "
          f"max abs: {global_cmp['max_abs_K']:.2f} K")
    print(f"PINN energy: Q_left {rep_pinn['energy']['Q_left']:.1f} W, "
          f"Q_right {rep_pinn['energy']['Q_right']:.1f} W, "
          f"balance err {rep_pinn['energy']['balance_err']:.2e}")
    print(f"FEM  energy: Q_left {rep_fem['Q_left_W']:.1f} W, "
          f"Q_right {rep_fem['Q_right_W']:.1f} W, "
          f"balance err {rep_fem['energy_balance_err']:.2e}")
    print(f"aerogel 1D: dT {rep_pinn['aerogel']['dT']:.2f} K vs 1D "
          f"{rep_pinn['aerogel']['dT_1D']:.2f} K "
          f"(rel err {rep_pinn['aerogel']['rel_err']:.2e})")
    print("mount continuity:", json.dumps(rep_pinn["mount"], indent=2))


if __name__ == "__main__":
    main()
