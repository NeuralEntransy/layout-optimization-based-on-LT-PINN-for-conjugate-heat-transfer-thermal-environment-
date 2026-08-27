# -*- coding: utf-8 -*-
"""
Milestone 1 configuration for the v4 benchmark.

Implements the benchmark of docs/04_复现方案.md:
closed low-pressure cavity, left aerogel layer, aluminum wall frame,
two wall-mounted square devices (movable in x, frozen in milestone 1)
and one fixed disk device. Stage-A reduced physics in milestone 1:
multi-material conduction only (no buoyancy, no radiation).

Normalization (doc section 4):
    L_ref = 1 m,  theta = (T - 218.15 K) / 85 K
    theta_cold = 0,  theta_inf = 1,  theta_lim = 1.4706
    right boundary = theta_inf = 1 (Dirichlet baseline)
    Bi = h_ext L_ref / k_Al = 0.1198 (optional Robin variant)
Per-domain equations are divided by the domain's own conductivity, so
conductivity ratios only enter the interface flux-continuity losses.
"""
import math

# ---------------------------------------------------------------- case
CASE_VERSION = "v4"
# Checkpoints with a different loss version may still provide compatible
# TemperatureField weights, but their optimizer/L-BFGS state must not be
# resumed because the objective and its gradient scales have changed.
LOSS_VERSION = "m1_energy_v5_face_flux"

# ---------------------------------------------------------------- geometry [m]
X_TBL = -0.005          # aerogel outer face (cold side)
W_IN = 0.01             # wall frame thickness / cavity inner offset
D1, D2 = 0.2, 0.35      # device 1 / 2 side lengths
C3 = (0.5, 0.4)         # device 3 (fixed disk) center
R3 = 0.1
B = 1.0                 # extrusion thickness
X1_RANGE = (0.12, 0.88)     # feasible ranges (doc 1.3)
X2_RANGE = (0.195, 0.805)

# device bounds helpers (y-ranges are fixed)
DEV1_Y = (W_IN, W_IN + D1)              # [0.01, 0.21]
DEV2_Y = (1.0 - W_IN - D2, 1.0 - W_IN)  # [0.64, 0.99]

# ---------------------------------------------------------------- heat power
P1, P2, P3 = 270.0, 540.0, 20.0         # W
P_TOT = P1 + P2 + P3                    # 830 W
Q1 = P1 / (D1 ** 2 * B)                 # 6750 W/m^3
Q2 = P2 / (D2 ** 2 * B)                 # 4408 W/m^3
Q3 = P3 / (math.pi * R3 ** 2 * B)       # 637 W/m^3

# ---------------------------------------------------------------- materials
K_AL = 167.0        # devices + wall
K_F = 0.026         # stagnant air
K_TBL = 0.018       # aerogel

# Milestone-1 choice: the 5 mm aerogel layer is degenerated to its
# 1-D thermal resistance (doc 04 section 6: "verify the aerogel layer can
# degenerate to a 1-D resistance"; the FEM reference with an explicit layer
# confirms the 1-D model to 7e-4).  It enters the PINN as a Robin condition
# on the wall's left face:  k_Al dtheta/dx = h_TBL * theta  at x = 0,
# h_TBL = k_TBL / l_TBL.  Set USE_TBL_1D = False to resolve the layer as a
# separate domain instead (kept for comparison).
USE_TBL_1D = True
L_TBL = abs(X_TBL)                  # 0.005 m
H_TBL = K_TBL / L_TBL               # 3.6 W/m^2/K
RPP_TBL = L_TBL / K_TBL             # 0.278 m^2 K/W

# ---------------------------------------------------------------- environment
T_COLD = 218.15     # left Dirichlet [K]  (-55 C)
T_INF = 303.15      # right heat-sink / Robin ambient [K] (30 C)
RIGHT_BC = "dirichlet"  # v4 baseline; "robin" is the comparison variant
H_EXT = 20.0        # W/(m^2 K), only used by the Robin variant
T_LIM = 343.15      # device limit [K] (70 C)

# ---------------------------------------------------------------- normalization
L_REF = 1.0
T_C = T_COLD
DT = 85.0
THETA_COLD = 0.0
THETA_INF = 1.0
THETA_LIM = (T_LIM - T_C) / DT          # 1.4706
BI = H_EXT * L_REF / K_AL               # 0.1198 (optional right Robin)
BI_TBL = H_TBL * L_REF / K_AL           # 0.02156 (left aerogel Robin)
S1 = Q1 * L_REF ** 2 / (K_AL * DT)      # 0.476
S2 = Q2 * L_REF ** 2 / (K_AL * DT)      # 0.311
S3 = Q3 * L_REF ** 2 / (K_AL * DT)      # 0.0449
Q_REF = 500.0       # characteristic boundary flux [W/m^2]
Q_IF = 150.0        # characteristic interface flux [W/m^2].  Smaller Q_IF
                    # stiffens flux continuity but creates a loss barrier:
                    # the device paraboloid (gain ~ w_dev*(S*p)^2*N_pts)
                    # only pays off once the air accepts the flux, while the
                    # interface penalty (cost ~ (k_Al*g_dev*DT/Q_IF)^2*w_ifq*
                    # N_if) acts immediately.  150 keeps gain > cost with
                    # w_pde_dev ~ 400 at any power level.

# Per-domain PDE residual weights. The device Poisson residual supplies the
# local heat-source curvature (the integral energy budgets supply additional
# global power signals). A uniform temperature shift leaves the PDE residual
# invariant, so the source needs amplification via --w-pde-dev to keep its
# learning rate comparable to the boundary losses.
PDE_W_OF_DOMAIN = dict(tbl=1.0, wall_l=1.0, wall_r=1.5, wall_b=1.0,
                       wall_t=2.0, air=1.0, dev1=1.0, dev2=1.5, dev3=1.0)

# Per-domain PDE residual scale (only relevant for the resolved-layer
# variant; with USE_TBL_1D the thin aerogel domain is absent).
PDE_RES_SCALE = dict(tbl=(0.005 ** 2), wall_l=1.0, wall_r=1.0,
                     wall_b=1.0, wall_t=1.0, air=1.0,
                     dev1=1.0, dev2=1.0, dev3=1.0)

K_OF_DOMAIN = dict(tbl=K_TBL, air=K_F,
                   wall_l=K_AL, wall_r=K_AL, wall_b=K_AL, wall_t=K_AL,
                   dev1=K_AL, dev2=K_AL, dev3=K_AL)
S_OF_DOMAIN = dict(dev1=S1, dev2=S2, dev3=S3)   # only devices have sources
if USE_TBL_1D:
    K_OF_DOMAIN.pop("tbl")

# ---------------------------------------------------------------- layouts
LAYOUTS = dict(left=(0.25, 0.35), center=(0.50, 0.50), right=(0.75, 0.65))

# ---------------------------------------------------------------- training defaults
TRAIN = dict(
    # Face-flux recovery stage: field-only warm start from the best v4
    # dense checkpoint. The network shape itself is unchanged.
    epochs=5000,
    # A field-only warm start creates a fresh Adam state. Thin-wall physical
    # derivatives make the first normalized Adam update very sensitive; CUDA
    # smoke tests showed 1e-5 and 1e-6 produce immediate heat-flux overshoot.
    lr=2e-7,
    # Learning-rate decay must be explicit. This recovery stage uses a
    # constant small Adam learning rate unless cosine is requested on CLI.
    lr_scheduler="none",
    width=96, depth=5,
    # Current warm-start stage remains at the full physical power.
    power_scale=1.0,
    power_start=1.0,
    ramp="none",
    ramp_frac=1.0,
    # optional quasi-Newton polish after Adam
    lbfgs_steps=0,
    lbfgs_max_iter=20,
    lbfgs_history=50,
    lbfgs_resample=0,
    # feature mapping / field initialization
    fourier_sigma=None,       # None = use per-domain values in networks.py
    fourier_dim=64,
    theta_init=0.0,
    # non-blocking live temperature window
    live_plot=True,
    plot_every=100,
    plot_resolution=151,
    # collocation points per epoch
    n_dom=dict(tbl=1500, wall_l=3000, wall_r=3000, wall_b=4000, wall_t=4000,
               air=15000, dev1=3000, dev2=14000, dev3=2500),
    # Additional PDE points in material-side boundary layers.
    n_near_device=dict(dev1=1000, dev2=3000, dev3=500),
    device_layer_width=0.01,
    n_near_wall=dict(wall_l_outer=1000, wall_l_inner=1000,
                     wall_r_outer=1000, wall_r_inner=1000,
                     wall_b_outer=2000, wall_b_inner=3000,
                     wall_t_outer=2000, wall_t_inner=3000),
    wall_layer_width=0.002,
    # Air-side layers around devices and the two long wall/air interfaces.
    n_near_air=dict(dev1=1500,
                    dev2_air_left=1500, dev2_air_right=1500,
                    dev2_air_bottom=1500,
                    dev3=1500, wall_b=2000, wall_t=2000),
    air_layer_width=0.01,
    # Random point counts are specified per named interface/boundary.
    n_iface=dict(tbl_wall=834,
                 wall_air_left=817, wall_air_right=817,
                 wall_air_bottom=650, wall_air_top=768,
                 dev1_wall=300, dev1_air_left=300,
                 dev1_air_right=300, dev1_air_top=300,
                 dev2_wall=512, dev2_air_left=512,
                 dev2_air_right=512, dev2_air_bottom=512,
                 dev3_air=524,
                 corner_l_b=256, corner_l_t=256,
                 corner_r_b=256, corner_r_t=256),
    n_bnd=dict(left=1024, right=1024,
               top_wall_l=64, top_wall_t=1024, top_wall_r=64,
               bottom_wall_l=64, bottom_wall_b=1024,
               bottom_wall_r=64,
               top_tbl=64, bottom_tbl=64),
    # Microbatch sizes for one-update-per-epoch gradient accumulation.
    pde_microbatch=2000,
    interface_microbatch=512,
    boundary_microbatch=512,
    # Fixed midpoint quadrature used only by integral energy constraints.
    # Keeping it separate from random collocation removes Monte-Carlo target
    # noise without changing the neural-network architecture.
    n_energy=1024,
    # loss weights
    w_pde=1.5, w_if_T=20.0, w_if_q=20.0, w_bc=75.0,
    w_pde_dev=600.0,   # explicit device-Poisson emphasis; residual scale=1
    w_eng=100.0,
    bc_w_adiabatic=8.0,
    # Inner energy-budget weights. Per-face flux continuity prevents an
    # incorrect air-side heat rate from being hidden by another device face.
    eng_w_device=2.0,
    eng_w_face=2.0,
    eng_w_air=1.0,
    eng_w_wall=0.75,
    eng_w_global=3.0,
    eng_w_lr=5.0,
    eng_w_adiabatic=20.0,
    eval_every=100, save_every=1000,
    seed=20231028,
)
