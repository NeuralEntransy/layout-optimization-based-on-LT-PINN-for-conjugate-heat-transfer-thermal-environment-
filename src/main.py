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
    python src/main.py --layout center --device cuda:0
    python src/main.py --layout center --resume
    python src/main.py --layout center --init-from OLD.pt \
        --ckptdir NEW_CKPT_DIR --outdir NEW_RESULT_DIR

Outputs:
    <ckptdir>/latest.pt, loss_log.csv, energy_log.csv, physics_log.csv
    <outdir>/final_report.json
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
    p.add_argument("--lr-scheduler", choices=["none", "cosine"],
                   default=C.TRAIN["lr_scheduler"],
                   help="Adam learning-rate schedule; 'none' keeps --lr "
                        "constant (default)")
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
    p.add_argument("--eng-w-device", type=float,
                   default=C.TRAIN["eng_w_device"],
                   help="inner weight of per-device double-sided budgets")
    p.add_argument("--eng-w-face", type=float,
                   default=C.TRAIN["eng_w_face"],
                   help="inner weight of per-interface integrated "
                        "two-sided flux continuity")
    p.add_argument("--eng-w-air", type=float,
                   default=C.TRAIN["eng_w_air"],
                   help="inner weight of the source-free air balance")
    p.add_argument("--eng-w-wall", type=float,
                   default=C.TRAIN["eng_w_wall"],
                   help="inner weight of source-free wall balances")
    p.add_argument("--eng-w-global", type=float,
                   default=C.TRAIN["eng_w_global"],
                   help="inner weight of the four-side global balance")
    p.add_argument("--eng-w-lr", type=float,
                   default=C.TRAIN["eng_w_lr"],
                   help="inner weight of Q_left + Q_right = total power")
    p.add_argument("--eng-w-adiabatic", type=float,
                   default=C.TRAIN["eng_w_adiabatic"],
                   help="inner weight of integrated top/bottom leakage")
    p.add_argument("--w-if-T", type=float, default=C.TRAIN["w_if_T"])
    p.add_argument("--w-if-q", type=float, default=C.TRAIN["w_if_q"])
    p.add_argument("--w-bc", type=float, default=C.TRAIN["w_bc"])
    p.add_argument("--bc-w-adiabatic", type=float,
                   default=C.TRAIN["bc_w_adiabatic"],
                   help="inner multiplier for pointwise top/bottom BC loss")
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
    p.add_argument("--resume-from", default=None,
                   help="checkpoint to resume instead of <ckptdir>/latest.pt; "
                        "requires the same loss version/settings and an Adam "
                        "snapshot")
    p.add_argument("--resume-lr", type=float, default=None,
                   help="override the current Adam learning rate after "
                        "loading a resume checkpoint; keeps optimizer "
                        "moments and follows --lr-scheduler")
    p.add_argument("--init-from", default=None,
                   help="warm-start field weights from another checkpoint "
                        "with a fresh optimizer/scheduler; use a new ckptdir")
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
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    if args.resume_from and not args.resume:
        raise ValueError("--resume-from requires --resume")
    if args.resume_lr is not None and not args.resume:
        raise ValueError("--resume-lr requires --resume")
    if args.epochs < 0 or args.lbfgs_steps < 0:
        raise ValueError("epoch/step counts must be non-negative")
    if args.eval_every <= 0 or args.save_every <= 0:
        raise ValueError("evaluation/save intervals must be positive")
    if args.lbfgs_max_iter <= 0 or args.lbfgs_history <= 0:
        raise ValueError("L-BFGS max_iter/history must be positive")
    if args.lbfgs_resample < 0:
        raise ValueError("--lbfgs-resample must be non-negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.power_scale <= 0.0 or args.power_start <= 0.0:
        raise ValueError("power scales must be positive")
    if not 0.0 <= args.ramp_frac <= 1.0:
        raise ValueError("--ramp-frac must lie in [0, 1]")
    nonnegative_weights = {
        "w_pde": args.w_pde,
        "w_pde_dev": args.w_pde_dev,
        "w_if_T": args.w_if_T,
        "w_if_q": args.w_if_q,
        "w_bc": args.w_bc,
        "bc_w_adiabatic": args.bc_w_adiabatic,
        "w_eng": args.w_eng,
        "eng_w_device": args.eng_w_device,
        "eng_w_face": args.eng_w_face,
        "eng_w_air": args.eng_w_air,
        "eng_w_wall": args.eng_w_wall,
        "eng_w_global": args.eng_w_global,
        "eng_w_lr": args.eng_w_lr,
        "eng_w_adiabatic": args.eng_w_adiabatic,
    }
    invalid_weights = [
        name for name, value in nonnegative_weights.items() if value < 0.0]
    if invalid_weights:
        raise ValueError(
            "loss weights must be non-negative: "
            + ", ".join(invalid_weights))

    device = torch.device(
        "cuda:0" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float32)

    def capture_rng_state():
        """Capture the random streams that determine future collocation sets."""
        return dict(
            torch_cpu=torch.get_rng_state(),
            torch_cuda=(torch.cuda.get_rng_state_all()
                        if device.type == "cuda" else None),
            numpy=np.random.get_state(),
        )

    def restore_rng_state(state, require_cuda=False):
        """Restore a checkpointed training stream without touching the model."""
        torch.set_rng_state(state["torch_cpu"])
        np.random.set_state(state["numpy"])
        if device.type != "cuda":
            return
        cuda_rng = state.get("torch_cuda")
        if cuda_rng is None:
            if require_cuda:
                raise ValueError(
                    "checkpoint lacks CUDA RNG state required for resume")
            return
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count differs from the resume checkpoint")
        torch.cuda.set_rng_state_all(cuda_rng)

    x1_0, x2_0 = C.LAYOUTS[args.layout]
    x1_0 = args.x1 if args.x1 is not None else x1_0
    x2_0 = args.x2 if args.x2 is not None else x2_0

    layout_tag = args.layout if (args.x1 is None and args.x2 is None) \
        else "custom"
    tag = f"{layout_tag}_{C.CASE_VERSION}_{args.right_bc}"
    if args.power_scale != 1.0:
        tag += f"_ps{args.power_scale:g}"
    default_ckptdir = os.path.join(DEFAULT_CKPT_ROOT, tag)
    default_outdir = os.path.join(
        os.path.dirname(__file__), "..", "results", "milestone1", tag)
    ckptdir = args.ckptdir or default_ckptdir
    outdir = args.outdir or default_outdir
    if args.init_from:
        if args.ckptdir is None or args.outdir is None:
            raise ValueError(
                "--init-from requires explicit new --ckptdir and --outdir "
                "so legacy checkpoints, logs, reports and plots are preserved")
        if (os.path.isdir(ckptdir) and os.listdir(ckptdir)) \
                or (os.path.isdir(outdir) and os.listdir(outdir)):
            raise FileExistsError(
                "warm-start destinations must be empty; choose new "
                "--ckptdir and --outdir")
    elif not args.resume and os.path.isdir(outdir) and os.listdir(outdir):
        raise FileExistsError(
            f"fresh-run outdir is not empty: {outdir}; choose a new --outdir")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------ build model
    field = TemperatureField(args.width, args.depth,
                             theta_init=args.theta_init,
                             fourier_sigma=args.fourier_sigma,
                             fourier_dim=args.fourier_dim).to(device)
    design = G.DesignVars(x1_0, x2_0, trainable=False, device=device)
    x1, x2 = design.x1(), design.x2()          # frozen tensors (milestone 1)
    plotter = None

    def update_live_plot(step_label, loss_value, power_value):
        nonlocal plotter
        if plotter is None:
            return
        try:
            plotter.update(field, x1, x2, device, step_label,
                           loss_value, power_value)
        except Exception as exc:
            # A GUI backend failure must never terminate a long training run.
            print(f"[plot warning] live window disabled: {exc}", flush=True)
            plotter = None

    condLoss = ConductionLoss(field, power_scale=args.power_scale,
                              w_dev=args.w_pde_dev)
    ifaceLoss = InterfaceLoss(field)
    bcLoss = BoundaryLoss(field, right_bc=args.right_bc,
                          w_adiabatic=args.bc_w_adiabatic)
    engLoss = EnergyConservationLoss(
        field,
        w_device=args.eng_w_device,
        w_face=args.eng_w_face,
        w_air=args.eng_w_air,
        w_wall=args.eng_w_wall,
        w_global=args.eng_w_global,
        w_lr=args.eng_w_lr,
        w_adiabatic=args.eng_w_adiabatic)

    optimizer = torch.optim.Adam(field.parameters(), lr=args.lr)

    def build_scheduler(base_lr, total_epochs):
        if args.lr_scheduler == "none":
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda _epoch: 1.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs),
            eta_min=base_lr * 1e-2)

    scheduler = build_scheduler(args.lr, args.epochs)

    start_epoch = 1
    ckpt_path = os.path.join(ckptdir, "latest.pt")

    network_snapshot = dict(
        width=args.width,
        depth=args.depth,
        fourier_sigma=args.fourier_sigma,
        fourier_dim=args.fourier_dim,
    )
    objective_snapshot = dict(
        loss_version=C.LOSS_VERSION,
        case=C.CASE_VERSION,
        network=network_snapshot,
        right_bc=args.right_bc,
        power_scale=args.power_scale,
        power_start=args.power_start,
        ramp=args.ramp,
        ramp_frac=args.ramp_frac,
        ramp_total_epochs=(
            args.epochs if args.ramp != "none" else None),
        lr_scheduler=args.lr_scheduler,
        w_pde=args.w_pde,
        w_pde_dev=args.w_pde_dev,
        w_if_T=args.w_if_T,
        w_if_q=args.w_if_q,
        w_bc=args.w_bc,
        bc_w_adiabatic=args.bc_w_adiabatic,
        w_eng=args.w_eng,
        eng_w_device=args.eng_w_device,
        eng_w_face=args.eng_w_face,
        eng_w_air=args.eng_w_air,
        eng_w_wall=args.eng_w_wall,
        eng_w_global=args.eng_w_global,
        eng_w_lr=args.eng_w_lr,
        eng_w_adiabatic=args.eng_w_adiabatic,
        pde_res_scale=dict(C.PDE_RES_SCALE),
        pde_domain_weights=dict(C.PDE_W_OF_DOMAIN),
        q_ref=C.Q_REF,
        q_if=C.Q_IF,
        n_energy=C.TRAIN["n_energy"],
        sampling=dict(
            n_dom=dict(C.TRAIN["n_dom"]),
            n_near_device=dict(C.TRAIN["n_near_device"]),
            device_layer_width=C.TRAIN["device_layer_width"],
            n_near_wall=dict(C.TRAIN["n_near_wall"]),
            wall_layer_width=C.TRAIN["wall_layer_width"],
            n_near_air=dict(C.TRAIN["n_near_air"]),
            air_layer_width=C.TRAIN["air_layer_width"],
            n_iface=dict(C.TRAIN["n_iface"]),
            n_bnd=dict(C.TRAIN["n_bnd"]),
            n_energy=C.TRAIN["n_energy"],
            pde_microbatch=C.TRAIN["pde_microbatch"],
            interface_microbatch=C.TRAIN["interface_microbatch"],
            boundary_microbatch=C.TRAIN["boundary_microbatch"],
        ),
        runtime=dict(device=str(device), seed=args.seed),
        geometry=dict(
            x1=float(x1), x2=float(x2), x_tbl=C.X_TBL,
            wall_thickness=C.W_IN, d1=C.D1, d2=C.D2,
            dev1_y=tuple(C.DEV1_Y), dev2_y=tuple(C.DEV2_Y),
            c3=tuple(C.C3), r3=C.R3, thickness=C.B,
            use_tbl_1d=C.USE_TBL_1D,
        ),
        materials=dict(k_al=C.K_AL, k_air=C.K_F, k_tbl=C.K_TBL),
        powers=dict(p1=C.P1, p2=C.P2, p3=C.P3),
        temperatures=dict(
            t_cold=C.T_COLD, t_inf=C.T_INF, dt=C.DT),
    )

    def save_checkpoint(epoch, lbfgs_optimizer=None):
        """Save a numbered training snapshot and refresh latest.pt."""
        state = dict(field=field.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),
                     case=C.CASE_VERSION,
                     loss_version=C.LOSS_VERSION,
                     objective=objective_snapshot,
                     network=network_snapshot,
                     right_bc=args.right_bc,
                     design=dict(z1=float(design.z1),
                                 z2=float(design.z2)),
                     epoch=epoch,
                     phase="lbfgs" if lbfgs_optimizer is not None else "adam",
                     rng_state=capture_rng_state(),
                     args=vars(args))
        if lbfgs_optimizer is not None:
            # L-BFGS snapshots are intentionally field-only for future stage
            # initialization. Exact L-BFGS resume would also require the
            # fixed collocation tensors and block position; storing only its
            # large history is both insufficient and wasteful.
            state["lbfgs_meta"] = dict(
                max_iter=args.lbfgs_max_iter,
                history=args.lbfgs_history,
                resample=args.lbfgs_resample,
            )
        history_path = os.path.join(ckptdir, f"epoch_{epoch:06d}.pt")
        torch.save(state, history_path)
        torch.save(state, ckpt_path)

    resume_path = args.resume_from or ckpt_path
    resume_checkpoint_epoch = None

    def checkpoint_xy(checkpoint):
        geometry_snapshot = checkpoint.get(
            "objective", {}).get("geometry")
        if geometry_snapshot is not None:
            return (float(geometry_snapshot["x1"]),
                    float(geometry_snapshot["x2"]))
        saved_design = checkpoint.get("design")
        if saved_design is not None:
            sigmoid_z1 = 1.0 / (1.0 + np.exp(-float(saved_design["z1"])))
            sigmoid_z2 = 1.0 / (1.0 + np.exp(-float(saved_design["z2"])))
            return (
                C.X1_RANGE[0]
                + (C.X1_RANGE[1] - C.X1_RANGE[0]) * sigmoid_z1,
                C.X2_RANGE[0]
                + (C.X2_RANGE[1] - C.X2_RANGE[0]) * sigmoid_z2,
            )
        saved_args = checkpoint.get("args", {})
        saved_layout = saved_args.get("layout")
        if saved_layout in C.LAYOUTS:
            return C.LAYOUTS[saved_layout]
        raise ValueError("checkpoint does not identify its fixed layout")

    if args.init_from:
        source_dir = os.path.normcase(os.path.abspath(
            os.path.dirname(args.init_from)))
        destination_dir = os.path.normcase(os.path.abspath(ckptdir))
        if source_dir == destination_dir:
            raise ValueError(
                "--init-from must write to a different --ckptdir; otherwise "
                "the old history and latest.pt would be overwritten")
        if os.path.exists(ckpt_path):
            raise FileExistsError(
                f"warm-start destination already contains {ckpt_path}; "
                "use a new --ckptdir or --resume that run")
    elif not args.resume and os.path.exists(ckpt_path):
        raise FileExistsError(
            f"fresh-run destination already contains {ckpt_path}; use "
            "--resume or choose a new --ckptdir")

    if args.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_loss_version = ck.get("loss_version")
        if checkpoint_loss_version != C.LOSS_VERSION:
            raise ValueError(
                "checkpoint objective is incompatible with the current "
                f"{C.LOSS_VERSION!r} loss (checkpoint: "
                f"{checkpoint_loss_version!r}). Use --init-from to load only "
                "TemperatureField weights with a fresh optimizer.")
        if ck.get("network") != network_snapshot:
            raise ValueError(
                "checkpoint network/Fourier settings differ from the current "
                "model. Exact --resume is unsafe.")
        if ck.get("objective") != objective_snapshot:
            raise ValueError(
                "checkpoint objective settings differ from the current "
                "settings. Exact --resume is unsafe; use --init-from for a "
                "field-only warm start.")
        if ck.get("phase") != "adam":
            raise ValueError(
                "this checkpoint was saved during L-BFGS. Use --init-from "
                "to start a new Adam stage, or resume from an Adam snapshot.")
        field.load_state_dict(ck["field"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        rng_state = ck.get("rng_state")
        if rng_state is None:
            raise ValueError(
                "checkpoint lacks RNG state required for exact --resume; "
                "use --init-from for a non-exact warm start")
        restore_rng_state(rng_state, require_cuda=True)
        start_epoch = ck["epoch"] + 1
        resume_checkpoint_epoch = int(ck["epoch"])
        print(f"[resume] from {resume_path} @ epoch {ck['epoch']} "
              f"(phase={ck.get('phase', 'unknown')})")
        if args.resume_lr is not None:
            if args.resume_lr <= 0:
                raise ValueError("--resume-lr must be positive")
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
                group["initial_lr"] = args.resume_lr
            if args.lr_scheduler == "none":
                scheduler = build_scheduler(
                    args.resume_lr, max(1, args.epochs - ck["epoch"]))
                print("[resume] constant learning rate retained")
            else:
                scheduler.base_lrs = [args.resume_lr] * len(
                    optimizer.param_groups)
                scheduler.eta_min = args.resume_lr * 1e-2
                scheduler._last_lr = [args.resume_lr] * len(
                    optimizer.param_groups)
                old_t_max = scheduler.T_max
                if (scheduler.last_epoch >= old_t_max
                        and args.epochs >= start_epoch):
                    remaining = args.epochs - ck["epoch"]
                    scheduler = build_scheduler(args.resume_lr, remaining)
                    print(f"[resume] completed cosine schedule replaced by "
                          f"a new {remaining}-epoch schedule")
                else:
                    print("[resume] cosine scheduler progress retained")
            print(f"[resume] learning rate overridden to {args.resume_lr:g}")
    elif args.resume:
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    elif args.init_from and os.path.exists(args.init_from):
        # Field-only warm starts may point to a large L-BFGS checkpoint.
        # Load it on CPU so discarded optimizer history never consumes GPU
        # memory, then copy only the field state into the current model.
        ck = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)
        if ck.get("case") != C.CASE_VERSION:
            raise ValueError(
                f"warm-start checkpoint case {ck.get('case')!r} is not "
                f"compatible with {C.CASE_VERSION!r}")
        source_right_bc = ck.get(
            "right_bc", ck.get("args", {}).get("right_bc"))
        if source_right_bc != args.right_bc:
            raise ValueError(
                f"warm-start right_bc {source_right_bc!r} differs from "
                f"requested {args.right_bc!r}")
        source_x1, source_x2 = checkpoint_xy(ck)
        if (abs(source_x1 - float(x1)) > 1e-7
                or abs(source_x2 - float(x2)) > 1e-7):
            raise ValueError(
                "warm-start checkpoint layout differs from the requested "
                f"fixed geometry: source=({source_x1:.6f}, {source_x2:.6f}), "
                f"current=({float(x1):.6f}, {float(x2):.6f})")
        source_network = ck.get("network")
        if source_network is None:
            saved_args = ck.get("args", {})
            network_keys = (
                "width", "depth", "fourier_sigma", "fourier_dim")
            if all(key in saved_args for key in network_keys):
                source_network = {
                    key: saved_args[key] for key in network_keys}
        if source_network is not None and source_network != network_snapshot:
            raise ValueError(
                "warm-start checkpoint network/Fourier settings differ from "
                "the requested model; keep the original architecture")
        source_objective = ck.get("objective", {})
        for snapshot_name in (
                "geometry", "materials", "powers", "temperatures"):
            source_snapshot = source_objective.get(snapshot_name)
            if (source_snapshot is not None
                    and source_snapshot != objective_snapshot[snapshot_name]):
                raise ValueError(
                    f"warm-start {snapshot_name} settings differ from the "
                    "current physical case")
        field.load_state_dict(ck["field"])
        source_epoch = ck.get("epoch", "unknown")
        source_loss_version = ck.get("loss_version", "legacy")
        del ck
        print(f"[warm-start] field weights from {args.init_from} "
              f"(epoch {source_epoch}, loss={source_loss_version}); "
              "fresh optimizer")
    elif args.init_from:
        raise FileNotFoundError(
            f"warm-start checkpoint not found: {args.init_from}")
    else:
        print("[fresh] training from scratch "
              "(old 3.1.2 checkpoints are never loaded)")

    print(f"layout={layout_tag}  case={C.CASE_VERSION}  "
          f"loss={C.LOSS_VERSION}  "
          f"right_bc={args.right_bc}  x1={float(x1):.4f}  "
          f"x2={float(x2):.4f}  device={device}  "
          f"power_scale={args.power_scale}  lr={args.lr:g}  "
          f"lr_scheduler={args.lr_scheduler}")
    print(f"ckptdir={ckptdir}\noutdir={outdir}")
    # Plot/log setup is outside the mathematical training objective. Preserve
    # the post-initialization (or restored checkpoint) random stream so those
    # side effects cannot alter future randomly resampled collocation sets.
    training_rng_state = capture_rng_state()
    if args.live_plot:
        try:
            plotter = LiveTemperaturePlot(
                outdir, C.TRAIN["plot_resolution"])
        except Exception as exc:
            print(f"[plot warning] live window disabled during setup: {exc}")

    # ------------------------------------------------------------ train loop
    nd = C.TRAIN["n_dom"]
    n_if, n_bnd = C.TRAIN["n_iface"], C.TRAIN["n_bnd"]
    n_energy = C.TRAIN["n_energy"]
    # Integral constraints use a persistent deterministic midpoint rule.
    # PDE/interface/boundary collocation remains randomly resampled.
    eng_if_samples = S.sample_all_interfaces(
        n_energy, x1, x2, device, deterministic=True)
    eng_bnd_samples = S.sample_boundaries(
        n_energy, device, deterministic=True)

    log_path = os.path.join(ckptdir, "loss_log.csv")
    energy_log_path = os.path.join(ckptdir, "energy_log.csv")
    physics_log_path = os.path.join(ckptdir, "physics_log.csv")
    log_header = [
        "epoch", "phase", "loss", "pde", "if_T", "if_q", "bc", "eng",
        "Q_left", "Q_right", "Q_top", "Q_bottom", "Q_outer",
        "Q_right_robin", "right_T_rms_err_K", "balance_lr_err",
        "balance_err", "adiabatic_net_leak", "adiabatic_leak",
        "Tmax1_C", "Tmax2_C", "Tmax3_C", "lr",
        "w_pde_dev", "w_bc", "w_eng", "eng_w_face", "eng_w_wall",
        "sec",
    ]

    passive_domains = ["wall_l", "wall_r", "wall_b", "wall_t"]
    if not C.USE_TBL_1D:
        passive_domains.append("tbl")
    horizontal_boundary_keys = [
        key for key in eng_bnd_samples
        if key.startswith("top_") or key.startswith("bottom_")
    ]
    energy_residual_keys = []
    energy_heat_keys = []
    for dev_name in ("dev1", "dev2", "dev3"):
        for side_name in ("dev", "recv"):
            energy_residual_keys.append(f"eng_{dev_name}_{side_name}")
            energy_heat_keys.append(
                f"eng_Q_{dev_name}_{side_name}_W")
        for iface_name in G.DEV_IFACES[dev_name]:
            energy_residual_keys.append(f"eng_face_{iface_name}")
            energy_heat_keys.extend([
                f"eng_Q_{iface_name}_dev_W",
                f"eng_Q_{iface_name}_recv_W",
                f"eng_dQ_{iface_name}_W",
            ])
    energy_residual_keys.append("eng_air")
    energy_heat_keys.append("eng_Q_air_W")
    for dom_name in passive_domains:
        energy_residual_keys.append(f"eng_{dom_name}")
        energy_heat_keys.append(f"eng_Q_{dom_name}_W")
    energy_residual_keys.extend(["eng_global", "eng_lr"])
    for key in horizontal_boundary_keys:
        energy_residual_keys.append(f"eng_adiabatic_{key}")
        energy_heat_keys.append(f"eng_Q_{key}_W")
    for side_name in ("top", "bottom"):
        energy_residual_keys.append(f"eng_adiabatic_{side_name}")
        energy_heat_keys.append(f"eng_Q_{side_name}_W")
    energy_heat_keys.extend(
        ["eng_Q_left_W", "eng_Q_right_W", "eng_Q_outer_W"])
    energy_component_keys = [
        "eng_loss_device", "eng_loss_face", "eng_loss_air",
        "eng_loss_wall",
        "eng_loss_global", "eng_loss_lr", "eng_loss_adiabatic",
        "eng_loss_total",
    ]
    energy_header = (
        ["epoch", "phase"] + energy_residual_keys
        + energy_component_keys + energy_heat_keys)

    left_bc_key = (
        "bc_left_robin_tbl1d" if C.USE_TBL_1D
        else "bc_left_dirichlet")
    right_bc_key = (
        "bc_right_dirichlet" if args.right_bc == "dirichlet"
        else "bc_right_robin")
    physics_detail_keys = [f"pde_{dom}" for dom in G.DOMAINS]
    for iface_name, _domain_a, _domain_b in G.INTERFACES:
        physics_detail_keys.extend(
            [f"ifT_{iface_name}", f"ifq_{iface_name}"])
    physics_detail_keys.extend([left_bc_key, right_bc_key])
    physics_detail_keys.extend(
        f"bc_{key}_adiab" for key in horizontal_boundary_keys)
    physics_header = ["epoch", "phase"] + physics_detail_keys

    log_paths = [log_path, energy_log_path, physics_log_path]
    log_preexisting = [os.path.exists(path) for path in log_paths]
    if args.resume and any(log_preexisting) and not all(log_preexisting):
        raise ValueError(
            "resume logs are incomplete: loss_log.csv, energy_log.csv and "
            "physics_log.csv must either all exist or all be absent")

    def initialize_log(path, header):
        if start_epoch == 1 and os.path.exists(path):
            raise FileExistsError(
                f"refusing to overwrite existing training log: {path}")
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
            return None
        with open(path, newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            last_row = None
            for row in reader:
                if row:
                    last_row = row
        if existing_header != header:
            raise ValueError(
                f"log schema mismatch for {path}; use a new --ckptdir")
        return last_row

    last_log_rows = [
        initialize_log(log_path, log_header),
        initialize_log(energy_log_path, energy_header),
        initialize_log(physics_log_path, physics_header),
    ]
    if args.resume and all(log_preexisting):
        if any(row is None for row in last_log_rows):
            raise ValueError(
                "resume logs exist but one or more contain no records")
        log_positions = {
            (int(float(row[0])), row[1]) for row in last_log_rows}
        if len(log_positions) != 1:
            raise ValueError(
                "resume logs end at different epoch/phase values; repair or "
                "branch with --init-from into a new directory")
        last_log_epoch, last_log_phase = next(iter(log_positions))
        if last_log_phase != "adam":
            raise ValueError(
                "resume logs do not end in the Adam phase")
        if last_log_epoch != resume_checkpoint_epoch:
            raise ValueError(
                f"logs end at epoch {last_log_epoch}, but the checkpoint is "
                f"epoch {resume_checkpoint_epoch}; exact resume requires "
                "synchronized history files")

    restore_rng_state(training_rng_state)

    active_weights = dict(
        w_bc=args.w_bc,
        w_eng=args.w_eng,
        w_pde_dev=args.w_pde_dev,
        eng_w_face=args.eng_w_face,
        eng_w_wall=args.eng_w_wall,
    )

    def sample_domains():
        domains = {d: S.sample_domain(d, nd[d], x1, x2, device)
                   for d in G.DOMAINS}
        for d, count in C.TRAIN["n_near_device"].items():
            local = S.sample_near_device_boundary(
                d, count, C.TRAIN["device_layer_width"], x1, x2, device)
            domains[d] = torch.cat((domains[d], local), dim=0)
        for name, count in C.TRAIN["n_near_wall"].items():
            dom = name.rsplit("_", 1)[0]
            local = S.sample_near_wall_surface(
                name, count, C.TRAIN["wall_layer_width"], device)
            domains[dom] = torch.cat((domains[dom], local), dim=0)
        air_layers = [S.sample_air_near(
            name, count, C.TRAIN["air_layer_width"], x1, x2, device)
            for name, count in C.TRAIN["n_near_air"].items()]
        domains["air"] = torch.cat([domains["air"]] + air_layers, dim=0)
        return domains

    def sample_chunks(sample, chunk_size):
        count = sample["pts"].shape[0]
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            chunk = {}
            for key, value in sample.items():
                if (torch.is_tensor(value) and value.ndim > 0
                        and value.shape[0] == count):
                    chunk[key] = value[start:stop]
                else:
                    chunk[key] = value
            yield chunk, (stop - start) / count

    def tensor_chunks(points, chunk_size):
        count = points.shape[0]
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            yield points[start:stop], (stop - start) / count

    def accumulate_pointwise(dom_samples, if_samples, bnd_samples,
                             do_backward):
        """Evaluate exact mean losses in microbatches and optionally backprop."""
        zero = torch.zeros((), device=device)
        totals = dict(pde=zero, ifT=zero, ifq=zero, bc=zero)
        details = {}

        def consume(value, outer_weight):
            if do_backward:
                (outer_weight * value).backward()

        for dom, points in dom_samples.items():
            key = f"pde_{dom}"
            detail = zero
            for chunk, fraction in tensor_chunks(
                    points, C.TRAIN["pde_microbatch"]):
                loss_part, part_details = condLoss({dom: chunk})
                scaled = fraction * loss_part
                consume(scaled, args.w_pde)
                detail = detail + fraction * part_details[key].detach()
            totals["pde"] = totals["pde"] + detail
            details[key] = detail

        for name, sample in if_samples.items():
            value_T, value_q = zero, zero
            for chunk, fraction in sample_chunks(
                    sample, C.TRAIN["interface_microbatch"]):
                l_T, l_q, _ = ifaceLoss({name: chunk})
                if do_backward:
                    (fraction * (args.w_if_T * l_T
                                 + args.w_if_q * l_q)).backward()
                value_T = value_T + fraction * l_T.detach()
                value_q = value_q + fraction * l_q.detach()
            totals["ifT"] = totals["ifT"] + value_T
            totals["ifq"] = totals["ifq"] + value_q
            details[f"ifT_{name}"] = value_T
            details[f"ifq_{name}"] = value_q

        for name, sample in bnd_samples.items():
            detail_key = (left_bc_key if name == "left" else
                          right_bc_key if name == "right" else
                          f"bc_{name}_adiab")
            value = zero
            for chunk, fraction in sample_chunks(
                    sample, C.TRAIN["boundary_microbatch"]):
                loss_part, _ = bcLoss({name: chunk})
                consume(fraction * loss_part, active_weights["w_bc"])
                # BoundaryLoss includes the inner adiabatic multiplier in its
                # total, while physics_log intentionally stores the raw term.
                raw = (loss_part / bcLoss.w_adiabatic
                       if name.startswith(("top_", "bottom_"))
                       else loss_part)
                value = value + fraction * raw.detach()
                totals["bc"] = totals["bc"] + fraction * loss_part.detach()
            details[detail_key] = value
        return totals, details

    def accumulated_loss(dom_samples, if_samples, bnd_samples,
                         do_backward=False):
        point, physics_details = accumulate_pointwise(
            dom_samples, if_samples, bnd_samples, do_backward)
        engLoss.power_scale = condLoss.power_scale
        l_eng, eng_details = engLoss(eng_if_samples, eng_bnd_samples)
        if do_backward:
            (active_weights["w_eng"] * l_eng).backward()
        l_eng_detached = l_eng.detach()
        total = (args.w_pde * point["pde"]
                 + args.w_if_T * point["ifT"]
                 + args.w_if_q * point["ifq"]
                 + active_weights["w_bc"] * point["bc"]
                 + active_weights["w_eng"] * l_eng_detached)
        return (total.detach(),
                (point["pde"], point["ifT"], point["ifq"],
                 point["bc"], l_eng_detached),
                eng_details, physics_details)

    def write_detail_log(path, header, epoch, phase, details):
        row = [epoch, phase] + [
            float(details[key].detach()) for key in header[2:]
        ]
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

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
        dom_samples = sample_domains()
        if_samples = S.sample_all_interfaces(n_if, x1, x2, device)
        bnd_samples = S.sample_boundaries(n_bnd, device)
        ## Accumulate exact mean-loss gradients over memory-bounded chunks.
        optimizer.zero_grad(set_to_none=True)
        loss, (l_pde, l_ifT, l_ifq, l_bc, l_eng), eng_details, \
            physics_details = accumulated_loss(
                dom_samples, if_samples, bnd_samples, do_backward=True)
        optimizer.step()
        scheduler.step()
        reported_loss = loss
        ## log and report
        if (epoch % args.eval_every == 0
                or epoch % args.save_every == 0
                or epoch == args.epochs):
            # Re-evaluate after the optimizer update so every quantity in a
            # CSV row describes the same field state.
            reported_loss, (
                l_pde, l_ifT, l_ifq, l_bc, l_eng
            ), eng_details, physics_details = accumulated_loss(
                dom_samples, if_samples, bnd_samples, do_backward=False)
            ps_now = condLoss.power_scale
            flux = M.energy_report(field, device, ps_now, n=257,
                                   right_bc=args.right_bc)
            tmax = M.device_Tmax(field, float(x1), float(x2), device, n=41)
            row = [epoch, "adam", float(reported_loss),
                   float(l_pde), float(l_ifT),
                   float(l_ifq), float(l_bc), float(l_eng),
                   flux["Q_left"], flux["Q_right"],
                   flux["Q_top"], flux["Q_bottom"], flux["Q_outer"],
                   flux["Q_right_robin"], flux["right_T_rms_err_K"],
                   flux["balance_lr_err"], flux["balance_err"],
                   flux["adiabatic_net_leak"], flux["adiabatic_leak"],
                   tmax["dev1"] - 273.15, tmax["dev2"] - 273.15,
                   tmax["dev3"] - 273.15,
                   scheduler.get_last_lr()[0],
                   active_weights["w_pde_dev"],
                   active_weights["w_bc"],
                   active_weights["w_eng"],
                   active_weights["eng_w_face"],
                   active_weights["eng_w_wall"],
                   time.time() - t0]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            write_detail_log(
                energy_log_path, energy_header, epoch, "adam", eng_details)
            write_detail_log(
                physics_log_path, physics_header, epoch, "adam",
                physics_details)
            print(f"ep {epoch:6d}  loss {float(reported_loss):.3e}  "
                  f"pde {float(l_pde):.2e}  ifT {float(l_ifT):.2e}  "
                  f"ifq {float(l_ifq):.2e}  bc {float(l_bc):.2e}  "
                  f"eng {float(l_eng):.2e} | "
                  f"Q_L {flux['Q_left']:7.1f}  Q_R {flux['Q_right']:7.1f}  "
                  f"Q_T {flux['Q_top']:7.1f}  Q_B {flux['Q_bottom']:7.1f}  "
                  f"bal4 {flux['balance_err']:.2e}  "
                  f"balLR {flux['balance_lr_err']:.2e} | "
                  f"T1 {tmax['dev1']-273.15:6.1f}C  T2 {tmax['dev2']-273.15:6.1f}C  "
                  f"T3 {tmax['dev3']-273.15:6.1f}C | "
                  f"wdev {active_weights['w_pde_dev']:.0f}  "
                  f"wbc {active_weights['w_bc']:.0f}  "
                  f"weng {active_weights['w_eng']:.0f}", flush=True)

        if plotter is not None and (epoch % C.TRAIN["plot_every"] == 0
                                    or epoch == args.epochs):
            update_live_plot(
                f"epoch {epoch}", reported_loss, condLoss.power_scale)

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
        def _new_lbfgs():
            return torch.optim.LBFGS(
                field.parameters(), max_iter=args.lbfgs_max_iter,
                history_size=args.lbfgs_history, tolerance_grad=1e-10,
                tolerance_change=1e-14, line_search_fn="strong_wolfe")

        lb = _new_lbfgs()

        def _samples():
            return (sample_domains(),
                    S.sample_all_interfaces(n_if, x1, x2, device),
                    S.sample_boundaries(n_bnd, device))

        fixed = _samples()
        for step in range(1, args.lbfgs_steps + 1):
            if (args.lbfgs_resample and step > 1
                    and (step - 1) % args.lbfgs_resample == 0):
                fixed = _samples()
                # Curvature pairs belong to the old discrete objective. They
                # are invalid after changing collocation points.
                lb = _new_lbfgs()
                print(f"[lbfgs] resampled points and reset history "
                      f"before step {step}")

            def closure():
                lb.zero_grad(set_to_none=True)
                l, _, _, _ = accumulated_loss(
                    *fixed, do_backward=True)
                return l

            t0 = time.time()
            l_val = lb.step(closure)
            display_loss = l_val
            if step % 10 == 0 or step == args.lbfgs_steps:
                current_loss, (l_pde, l_ifT, l_ifq, l_bc, l_eng), \
                    eng_details, physics_details = accumulated_loss(
                        *fixed, do_backward=False)
                display_loss = current_loss
                flux = M.energy_report(field, device, args.power_scale,
                                       n=257, right_bc=args.right_bc)
                tmax = M.device_Tmax(field, float(x1), float(x2), device,
                                     n=41)
                epoch_tag = args.epochs + step
                row = [epoch_tag, "lbfgs", float(current_loss),
                       float(l_pde), float(l_ifT), float(l_ifq),
                       float(l_bc), float(l_eng),
                       flux["Q_left"], flux["Q_right"],
                       flux["Q_top"], flux["Q_bottom"], flux["Q_outer"],
                       flux["Q_right_robin"], flux["right_T_rms_err_K"],
                       flux["balance_lr_err"], flux["balance_err"],
                       flux["adiabatic_net_leak"],
                       flux["adiabatic_leak"],
                       tmax["dev1"] - 273.15, tmax["dev2"] - 273.15,
                       tmax["dev3"] - 273.15, -1.0,
                       active_weights["w_pde_dev"],
                       active_weights["w_bc"],
                       active_weights["w_eng"],
                       active_weights["eng_w_face"],
                       active_weights["eng_w_wall"],
                       time.time() - t0]
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                write_detail_log(
                    energy_log_path, energy_header, epoch_tag, "lbfgs",
                    eng_details)
                write_detail_log(
                    physics_log_path, physics_header, epoch_tag, "lbfgs",
                    physics_details)
                print(f"lb {step:5d}  loss {float(current_loss):.3e}  "
                      f"pde {float(l_pde):.2e}  ifT {float(l_ifT):.2e}  "
                      f"ifq {float(l_ifq):.2e}  bc {float(l_bc):.2e}  "
                      f"eng {float(l_eng):.2e} | "
                      f"Q_L {flux['Q_left']:7.1f}  "
                      f"Q_R {flux['Q_right']:7.1f}  "
                      f"Q_T {flux['Q_top']:7.1f}  "
                      f"Q_B {flux['Q_bottom']:7.1f}  "
                      f"bal4 {flux['balance_err']:.2e}  "
                      f"balLR {flux['balance_lr_err']:.2e} | "
                      f"T1 {tmax['dev1']-273.15:6.1f}C  "
                      f"T2 {tmax['dev2']-273.15:6.1f}C  "
                      f"T3 {tmax['dev3']-273.15:6.1f}C", flush=True)
            epoch_tag = args.epochs + step
            if plotter is not None and (
                    epoch_tag % C.TRAIN["plot_every"] == 0
                    or step == args.lbfgs_steps):
                update_live_plot(
                    f"L-BFGS {step}", display_loss, args.power_scale)
            if step % 50 == 0 or step == args.lbfgs_steps:
                save_checkpoint(args.epochs + step, lb)

    # ------------------------------------------------------------ final report
    condLoss.power_scale = args.power_scale       # ensure full-power metrics
    report = M.full_report(field, float(x1), float(x2), device,
                           args.power_scale, right_bc=args.right_bc)
    report.update(dict(layout=layout_tag, case=C.CASE_VERSION,
                       loss_version=C.LOSS_VERSION,
                       objective=objective_snapshot,
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
