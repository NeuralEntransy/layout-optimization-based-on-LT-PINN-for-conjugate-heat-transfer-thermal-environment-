"""Physical and training defaults for the independent five-domain solver."""

CASE_VERSION = "sdf_xpinn_v1"

# Geometry [m]
W_IN = 0.01
X_TBL = -0.005
L_TBL = 0.005
D1 = 0.20
D2 = 0.35
DEV1_Y = (W_IN, W_IN + D1)
DEV2_Y = (1.0 - W_IN - D2, 1.0 - W_IN)
C3 = (0.5, 0.4)
R3 = 0.1
X1_RANGE = (W_IN + D1 / 2, 1.0 - W_IN - D1 / 2)
X2_RANGE = (W_IN + D2 / 2, 1.0 - W_IN - D2 / 2)
LAYOUTS = {"left": (0.25, 0.35), "center": (0.50, 0.50),
           "right": (0.75, 0.65)}

# Thermal data
K_AL = 167.0
K_F = 0.026
K_TBL = 0.018
B = 1.0
T_COLD = 218.15
T_C = T_COLD
T_INF = 303.15
DT = T_INF - T_C
L_REF = 1.0
Q1, Q2, Q3 = 270.0, 540.0, 20.0
P_TOT = Q1 + Q2 + Q3
H_TBL = K_TBL / L_TBL
THETA_COLD = 0.0
THETA_INF = 1.0
Q_IF = 150.0
Q_REF = 500.0

S1 = Q1 / (D1 * D1 * B) * L_REF ** 2 / (K_AL * DT)
S2 = Q2 / (D2 * D2 * B) * L_REF ** 2 / (K_AL * DT)
S3 = Q3 / (3.141592653589793 * R3 ** 2 * B) * L_REF ** 2 / (K_AL * DT)

DOMAINS = ("wall", "air", "dev1", "dev2", "dev3")
K_OF_DOMAIN = {"wall": K_AL, "air": K_F,
               "dev1": K_AL, "dev2": K_AL, "dev3": K_AL}
S_OF_DOMAIN = {"dev1": S1, "dev2": S2, "dev3": S3}
P_OF_DEVICE = {"dev1": Q1, "dev2": Q2, "dev3": Q3}

TRAIN = dict(
    width=96, depth=5, fourier_sigma=0.0, fourier_dim=64,
    lr=2e-7, epochs=20000, seed=1234,
    power_scale=1.0,
    n_dom={"wall": 12000, "air": 15000,
           "dev1": 3000, "dev2": 14000, "dev3": 2500},
    # Extra interior PDE points selected by -width <= SDF <= 0.  This gives
    # every complex boundary a geometry-driven refinement layer.
    n_near={"wall": 4000, "air": 6000,
            "dev1": 1000, "dev2": 3000, "dev3": 500},
    near_width={"wall": 0.002, "air": 0.01,
                "dev1": 0.01, "dev2": 0.01, "dev3": 0.01},
    # Residual-adaptive random refinement (RAR).  These points are appended
    # to the uniform and SDF-boundary-layer points every epoch.  A candidate
    # pool is ranked probabilistically by squared strong-form PDE residual.
    n_rar={"wall": 2000, "air": 3000,
           "dev1": 500, "dev2": 1000, "dev3": 500},
    rar_candidate_factor=4,
    rar_power=1.0,
    rar_uniform_mix=0.05,
    rar_score_microbatch=2000,
    n_iface={
        "wall_air_left": 817, "wall_air_right": 817,
        "wall_air_bottom": 650, "wall_air_top": 768,
        "dev1_wall": 670, "dev1_air_left": 670,
        "dev1_air_right": 670, "dev1_air_top": 670,
        "dev2_wall": 1200, "dev2_air_left": 1200,
        "dev2_air_right": 1200, "dev2_air_bottom": 1200,
        "dev3_air": 524,
    },
    n_bnd={"left": 1024, "right": 1024,
           "top": 1024, "bottom": 1024},
    n_energy=1024,
    pde_microbatch=2000, interface_microbatch=512,
    w_pde=1.5, w_pde_dev=600.0,
    w_if_T=20.0, w_if_q=20.0,
    w_bc=75.0, w_adiabatic=8.0, w_eng=100.0,
    eval_every=100, save_every=1000,
)
