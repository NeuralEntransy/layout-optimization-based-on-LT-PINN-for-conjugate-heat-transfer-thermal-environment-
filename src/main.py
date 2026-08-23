# -*- coding: utf-8 -*-
"""
Milestone 1: solid thermal bridges + pure multi-material conduction
====================================================================
Reproduction plan: docs/04_复现方案.md, section 6 (Milestone 1).

Physics: steady multi-domain conduction in the aluminum wall frame, stagnant
cavity air and three devices with volumetric heat sources.  The 5 mm aerogel
is represented by its one-dimensional resistance by default.  Buoyancy and
radiation are switched OFF.  Interfaces impose temperature + flux continuity;
outer BCs are left aerogel resistance, top/bottom adiabatic and the v4 right
30 C Dirichlet heat sink (an optional Robin comparison is retained).

The two design variables x1/x2 are created through the sigmoid
parameterization of geometry.DesignVars but are FROZEN in milestone 1
(fixed layout validation against the milestone-0 FEM reference).

Usage (torch env):
    python src/main.py --lr 1e-3 --epochs 30000 --layout center --device cuda:0
    python src/main.py --layout center --power-scale 0.1 --epochs 2000  # debug
    python src/main.py --layout center --resume                          # continue

Outputs:
    <ckptdir>/latest.pt, loss_log.csv, final_report.json
    <outdir>/  (shared with validate.py results)
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import config as C
import geometry as G
import sampling as S
import monitors as M
from networks import TemperatureField
from live_plot import LiveTemperaturePlot
from losses import ConductionLoss, InterfaceLoss, BoundaryLoss, \
    EnergyConservationLoss

# old 3.1.2 checkpoints live in ../checkpoint and must NOT be touched here
DEFAULT_CKPT_ROOT = os.path.join(os.path.dirname(__file__), "..",
                                 "checkpoint_m1")


def parse_args():
    p = argparse.ArgumentParser(description="Milestone 1 conduction PINN")
    p.add_argument("--layout", default="center", choices=list(C.LAYOUTS),
                   help="fixed layout name from config.LAYOUTS")
    p.add_argument("--x1", type=float, default=None, help="override layout x1")
    p.add_argument("--x2", type=float, default=None, help="override layout x2")
    p.add_argument("--epochs", type=int, default=C.TRAIN["epochs"],
                   help="Adam-phase epochs")
    p.add_argument("--lbfgs-steps", type=int,
                   default=C.TRAIN["lbfgs_steps"],
                   help="L-BFGS polish steps after the Adam phase "
                        "(quasi-Newton handles the stiff coupled valley)")
    p.add_argument("--lbfgs-max-iter", type=int,
                   default=C.TRAIN["lbfgs_max_iter"])
    p.add_argument("--lbfgs-history", type=int,
                   default=C.TRAIN["lbfgs_history"])
    p.add_argument("--lbfgs-resample", type=int,
                   default=C.TRAIN["lbfgs_resample"],
                   help="resample collocation points every N L-BFGS steps "
                        "(0 = keep fixed)")
    p.add_argument("--lr", type=float, default=C.TRAIN["lr"])
    p.add_argument("--width", type=int, default=C.TRAIN["width"])
    p.add_argument("--depth", type=int, default=C.TRAIN["depth"])
    p.add_argument("--power-scale", type=float,
                   default=C.TRAIN["power_scale"],
                   help="final heat-source scale")
    p.add_argument("--right-bc", choices=["dirichlet", "robin"],
                   default=C.RIGHT_BC,
                   help="right boundary: v4 ideal heat sink or comparison")
    p.add_argument("--power-start", type=float,
                   default=C.TRAIN["power_start"],
                   help="ramp start scale (default: = power-scale, no ramp)")
    p.add_argument("--ramp", choices=["none", "linear", "exp"],
                   default=C.TRAIN["ramp"],
                   help="power continuation schedule")
    p.add_argument("--ramp-frac", type=float,
                   default=C.TRAIN["ramp_frac"],
                   help="fraction of epochs over which power ramps up")
    p.add_argument("--w-pde", type=float, default=C.TRAIN["w_pde"])
    p.add_argument("--w-pde-dev", type=float, default=C.TRAIN["w_pde_dev"])
    p.add_argument("--w-eng", type=float, default=C.TRAIN["w_eng"])
    p.add_argument("--w-if-T", type=float, default=C.TRAIN["w_if_T"])
    p.add_argument("--w-if-q", type=float, default=C.TRAIN["w_if_q"])
    p.add_argument("--w-bc", type=float, default=C.TRAIN["w_bc"])
    p.add_argument("--fourier-sigma", type=float,
                   default=C.TRAIN["fourier_sigma"],
                   help="override per-domain Fourier sigma globally "
                        "(default: per-domain table in networks.py; "
                        "0 disables Fourier features)")
    p.add_argument("--fourier-dim", type=int,
                   default=C.TRAIN["fourier_dim"])
    p.add_argument("--theta-init", type=float,
                   default=C.TRAIN["theta_init"],
                   help="uniform warm-start temperature level (theta units)")
    p.add_argument("--seed", type=int, default=C.TRAIN["seed"])
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true",
                   help="load <ckptdir>/latest.pt (default: fresh start)")
    p.add_argument("--resume-lr", type=float, default=None,
                   help="override the current Adam learning rate after "
                        "loading a resume checkpoint; keeps optimizer "
                        "moments and scheduler progress")
    p.add_argument("--init-from", default=None,
                   help="warm-start field weights from another checkpoint "
                        "(fresh optimizer/scheduler; for power ramp stages)")
    p.add_argument("--eval-every", type=int, default=C.TRAIN["eval_every"])
    p.add_argument("--save-every", type=int, default=C.TRAIN["save_every"])
    p.add_argument("--no-plot", action="store_false", dest="live_plot",
                   default=C.TRAIN["live_plot"],
                   help="disable the live temperature-field window")
    p.add_argument("--ckptdir", default=None)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda:0" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float32)

    x1_0, x2_0 = C.LAYOUTS[args.layout]
    x1_0 = args.x1 if args.x1 is not None else x1_0
    x2_0 = args.x2 if args.x2 is not None else x2_0

    layout_tag = args.layout if (args.x1 is None and args.x2 is None) \
        else "custom"
    tag = f"{layout_tag}_{C.CASE_VERSION}_{args.right_bc}"
    if args.power_scale != 1.0:
        tag += f"_ps{args.power_scale:g}"
    ckptdir = args.ckptdir or os.path.join(DEFAULT_CKPT_ROOT, tag)
    outdir = args.outdir or os.path.join(
        os.path.dirname(__file__), "..", "results", "milestone1", tag)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------ build model
    field = TemperatureField(args.width, args.depth,
                             theta_init=args.theta_init,
                             fourier_sigma=args.fourier_sigma,
                             fourier_dim=args.fourier_dim).to(device)
    design = G.DesignVars(x1_0, x2_0, trainable=False, device=device)
    x1, x2 = design.x1(), design.x2()          # frozen tensors (milestone 1)
    plotter = LiveTemperaturePlot(
        outdir, C.TRAIN["plot_resolution"]) if args.live_plot else None

    condLoss = ConductionLoss(field, power_scale=args.power_scale,
                              w_dev=args.w_pde_dev)
    ifaceLoss = InterfaceLoss(field)
    bcLoss = BoundaryLoss(field, right_bc=args.right_bc)
    engLoss = EnergyConservationLoss(field)

    optimizer = torch.optim.Adam(field.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    start_epoch = 1
    ckpt_path = os.path.join(ckptdir, "latest.pt")

    def save_checkpoint(epoch, lbfgs_optimizer=None):
        """Save a numbered training snapshot and refresh latest.pt."""
        state = dict(field=field.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),
                     case=C.CASE_VERSION,
                     right_bc=args.right_bc,
                     design=dict(z1=float(design.z1),
                                 z2=float(design.z2)),
                     epoch=epoch,
                     phase="lbfgs" if lbfgs_optimizer is not None else "adam",
                     args=vars(args))
        if lbfgs_optimizer is not None:
            state["lbfgs"] = lbfgs_optimizer.state_dict()
        history_path = os.path.join(ckptdir, f"epoch_{epoch:06d}.pt")
        torch.save(state, history_path)
        torch.save(state, ckpt_path)

    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        field.load_state_dict(ck["field"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        print(f"[resume] from {ckpt_path} @ epoch {ck['epoch']}")
        if args.resume_lr is not None:
            if args.resume_lr <= 0:
                raise ValueError("--resume-lr must be positive")
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
                group["initial_lr"] = args.resume_lr
            scheduler.base_lrs = [args.resume_lr] * len(optimizer.param_groups)
            scheduler.eta_min = args.resume_lr * 1e-2
            scheduler._last_lr = [args.resume_lr] * len(optimizer.param_groups)
            print(f"[resume] learning rate overridden to "
                  f"{args.resume_lr:g}; scheduler epoch retained")
    elif args.init_from and os.path.exists(args.init_from):
        ck = torch.load(args.init_from, map_location=device,
                        weights_only=False)
        field.load_state_dict(ck["field"])
        print(f"[warm-start] field weights from {args.init_from} "
              f"(epoch {ck['epoch']}); fresh optimizer")
    else:
        print("[fresh] training from scratch "
              "(old 3.1.2 checkpoints are never loaded)")

    print(f"layout={layout_tag}  case={C.CASE_VERSION}  "
          f"right_bc={args.right_bc}  x1={float(x1):.4f}  "
          f"x2={float(x2):.4f}  device={device}  "
          f"power_scale={args.power_scale}")
    print(f"ckptdir={ckptdir}\noutdir={outdir}")

    log_path = os.path.join(ckptdir, "loss_log.csv")
    log_header = ["epoch", "loss", "pde", "if_T", "if_q", "bc", "eng",
                  "Q_left", "Q_right", "Q_right_robin",
                  "right_T_rms_err_K", "balance_err",
                  "Tmax1_C", "Tmax2_C", "Tmax3_C", "lr", "sec"]
    if start_epoch == 1 or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_header)

    # ------------------------------------------------------------ train loop
    nd = C.TRAIN["n_dom"]
    n_if, n_bnd = C.TRAIN["n_iface"], C.TRAIN["n_bnd"]

    def compute_loss(dom_samples, if_samples, bnd_samples):
        l_pde, _ = condLoss(dom_samples)
        l_ifT, l_ifq, _ = ifaceLoss(if_samples)
        l_bc, _ = bcLoss(bnd_samples)
        engLoss.power_scale = condLoss.power_scale
        l_eng, _ = engLoss(if_samples, bnd_samples)
        loss = (args.w_pde * l_pde + args.w_if_T * l_ifT
                + args.w_if_q * l_ifq + args.w_bc * l_bc
                + args.w_eng * l_eng)
        return loss, (l_pde, l_ifT, l_ifq, l_bc, l_eng)

    # Continuous power continuation: the physical solution is O(power) away
    # from the trivial flat state, so a slow ramp lets the field track the
    # moving target instead of crossing a loss barrier at fixed power.
    p_start = args.power_start if args.power_start is not None \
        else args.power_scale

    def power_at(epoch):
        f = min(epoch / max(1.0, args.ramp_frac * args.epochs), 1.0)
        if args.ramp == "exp" and p_start > 0:
            return p_start * (args.power_scale / p_start) ** f
        if args.ramp == "linear":
            return p_start + (args.power_scale - p_start) * f
        return args.power_scale
#### Training loop: Adam phase, then optional L-BFGS polish
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        ## sample collocation points
        t0 = time.time()
        condLoss.power_scale = power_at(epoch)

        dom_samples = {d: S.sample_domain(d, nd[d], x1, x2, device)
                       for d in G.DOMAINS}
        if_samples = S.sample_all_interfaces(n_if, x1, x2, device)
        bnd_samples = S.sample_boundaries(n_bnd, device)
        ## compute loss and backprop
        loss, (l_pde, l_ifT, l_ifq, l_bc, l_eng) = compute_loss(
            dom_samples, if_samples, bnd_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ps_now = condLoss.power_scale
            flux = M.energy_report(field, device, ps_now, n=257,
                                   right_bc=args.right_bc)
            tmax = M.device_Tmax(field, float(x1), float(x2), device, n=41)
            row = [epoch, float(loss), float(l_pde), float(l_ifT),
                   float(l_ifq), float(l_bc), float(l_eng),
                   flux["Q_left"], flux["Q_right"], flux["Q_right_robin"],
                   flux["right_T_rms_err_K"], flux["balance_err"],
                   tmax["dev1"] - 273.15, tmax["dev2"] - 273.15,
                   tmax["dev3"] - 273.15,
                   scheduler.get_last_lr()[0], time.time() - t0]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"ep {epoch:6d}  loss {float(loss):.3e}  "
                  f"pde {float(l_pde):.2e}  ifT {float(l_ifT):.2e}  "
                  f"ifq {float(l_ifq):.2e}  bc {float(l_bc):.2e}  "
                  f"eng {float(l_eng):.2e} | "
                  f"Q_L {flux['Q_left']:7.1f}  Q_R {flux['Q_right']:7.1f}  "
                  f"bal {flux['balance_err']:.2e} | "
                  f"T1 {tmax['dev1']-273.15:6.1f}C  T2 {tmax['dev2']-273.15:6.1f}C  "
                  f"T3 {tmax['dev3']-273.15:6.1f}C", flush=True)

        if plotter is not None and (epoch % C.TRAIN["plot_every"] == 0
                                    or epoch == args.epochs):
            plotter.update(field, x1, x2, device, f"epoch {epoch}",
                           loss, condLoss.power_scale)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(epoch)

    # ------------------------------------------------------------ L-BFGS phase
    # The coupled device-paraboloid -> interface-flux -> air-hotspot chain
    # forms a stiff valley for first-order optimizers; L-BFGS (quasi-Newton)
    # takes big steps along it.  Collocation points are fixed per step block.
    if args.lbfgs_steps > 0:
        condLoss.power_scale = args.power_scale
        print(f"[lbfgs] {args.lbfgs_steps} steps at fixed power "
              f"{args.power_scale:g}")
        lb = torch.optim.LBFGS(
            field.parameters(), max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history, tolerance_grad=1e-10,
            tolerance_change=1e-14, line_search_fn="strong_wolfe")

        def _samples():
            return ({d: S.sample_domain(d, nd[d], x1, x2, device)
                     for d in G.DOMAINS},
                    S.sample_all_interfaces(n_if, x1, x2, device),
                    S.sample_boundaries(n_bnd, device))

        fixed = _samples()
        for step in range(1, args.lbfgs_steps + 1):
            if args.lbfgs_resample and step % args.lbfgs_resample == 0:
                fixed = _samples()

            def closure():
                lb.zero_grad()
                l, _ = compute_loss(*fixed)
                l.backward()
                return l

            t0 = time.time()
            l_val = lb.step(closure)
            if step % 10 == 0 or step == args.lbfgs_steps:
                _, (l_pde, l_ifT, l_ifq, l_bc, l_eng) = compute_loss(*fixed)
                flux = M.energy_report(field, device, args.power_scale,
                                       n=257, right_bc=args.right_bc)
                tmax = M.device_Tmax(field, float(x1), float(x2), device,
                                     n=41)
                epoch_tag = args.epochs + step
                row = [epoch_tag, float(l_val), float(l_pde), float(l_ifT),
                       float(l_ifq), float(l_bc), float(l_eng),
                       flux["Q_left"], flux["Q_right"],
                       flux["Q_right_robin"],
                       flux["right_T_rms_err_K"], flux["balance_err"],
                       tmax["dev1"] - 273.15, tmax["dev2"] - 273.15,
                       tmax["dev3"] - 273.15, -1.0, time.time() - t0]
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(f"lb {step:5d}  loss {float(l_val):.3e}  "
                      f"pde {float(l_pde):.2e}  ifT {float(l_ifT):.2e}  "
                      f"ifq {float(l_ifq):.2e}  bc {float(l_bc):.2e}  "
                      f"eng {float(l_eng):.2e} | "
                      f"Q_L {flux['Q_left']:7.1f}  "
                      f"Q_R {flux['Q_right']:7.1f}  "
                      f"bal {flux['balance_err']:.2e} | "
                      f"T1 {tmax['dev1']-273.15:6.1f}C  "
                      f"T2 {tmax['dev2']-273.15:6.1f}C  "
                      f"T3 {tmax['dev3']-273.15:6.1f}C", flush=True)
            epoch_tag = args.epochs + step
            if plotter is not None and (
                    epoch_tag % C.TRAIN["plot_every"] == 0
                    or step == args.lbfgs_steps):
                plotter.update(field, x1, x2, device,
                               f"L-BFGS {step}", l_val,
                               args.power_scale)
            if step % 50 == 0 or step == args.lbfgs_steps:
                save_checkpoint(args.epochs + step, lb)

    # ------------------------------------------------------------ final report
    condLoss.power_scale = args.power_scale       # ensure full-power metrics
    report = M.full_report(field, float(x1), float(x2), device,
                           args.power_scale, right_bc=args.right_bc)
    report.update(dict(layout=layout_tag, case=C.CASE_VERSION,
                       right_bc=args.right_bc,
                       x1=float(x1), x2=float(x2),
                       power_scale=args.power_scale, epochs=args.epochs,
                       wall_time_min=(time.time() - t_start) / 60))
    with open(os.path.join(outdir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== final report ===")
    print(json.dumps(report, indent=2))
    print(f"\ncheckpoint: {ckpt_path}\nreport: {outdir}/final_report.json")
    print("next: python src/validate.py --layout", args.layout,
          "--right-bc", args.right_bc)


if __name__ == "__main__":
    main()
