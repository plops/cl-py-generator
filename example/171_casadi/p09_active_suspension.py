from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ========================================================================
# AKTIVE FAHRWERKSREGELUNG MITTELS KOMPAKTER MAP-MPC (GEN09)
# ========================================================================
# Thema: Aktive Fahrwerksregelung fuer ein Viertelfahrzeug-Modell (Quarter-Car)
# Dieses Skript implementiert die MPC-Formulierung effizient unter Verwendung
# von CasADi's map() und mapaccum() Funktions-Objekten anstelle von klassischen
# For-Schleifen. Dies reduziert die Groesse des Ausdrucksgraphen von O(N) auf O(1)
# und beschleunigt die Initialisierungszeit erheblich.
#
# Modellbeschreibung:
# Viertelfahrzeug-Modell (Quarter-Car Model) zur Entkopplung der Ecken.
#
# Zustandsdefinitionen des Vektors x = [x1, x2, x3, x4]^T:
#   x1: zs - zu  -> Einfederweg der Karosserie relativ zum Rad (m).
#                   Grenze: +/- 0.08 m (Aufhaengungsanschlag).
#   x2: dot_zs   -> Vertikalgeschwindigkeit der gefederten Masse (Chassis) (m/s).
#   x3: zu - zr  -> Einfederweg des Reifens relativ zur Strasse (m).
#                   Reifeneinfederung ist proportional zur Radlast.
#   x4: dot_zu   -> Vertikalgeschwindigkeit der ungefederten Masse (Rad) (m/s).
#
# Eingangsdefinitionen:
#   u:  Aktive Kraft des Dämpfer-Aktuators (N). Grenze: +/- 1500 N.
#   vr: Strassengeschwindigkeitsstoerung dot_zr (m/s) (Differentieller Strasseneinfluss).
# ========================================================================
ms = 3.0e2
mu = 4.0e1
ks = 1.5e4
cs = 1.0e3
kt = 1.5e5
dt = 1.0e-2
N = 30
# --- Physikalische Systemmatrizen (Kontinuierlich) und Diskretisierung ---
A_np = np.array(
    [
        [0.0, 1.0, 0.0, -1.0],
        [-ks / ms, -cs / ms, 0.0, cs / ms],
        [0.0, 0.0, 0.0, 1.0],
        [ks / mu, cs / mu, -kt / mu, -cs / mu],
    ]
)
B_np = np.array([0.0, 1.0 / ms, 0.0, -1.0 / mu])
G_np = np.array([0.0, 0.0, -1.0, 0.0])
M = np.zeros(
    (
        6,
        6,
    )
)
M[0:4, 0:4] = A_np
M[0:4, 4] = B_np
M[0:4, 5] = G_np
M_disc = expm(M * dt)
A_d_np = M_disc[0:4, 0:4]
B_d_np = M_disc[0:4, 4]
G_d_np = M_disc[0:4, 5]
A = DM(A_d_np)
B = DM(B_d_np)
G = DM(G_d_np)
# --- MPC Gewichtungsfaktoren (mit SI-Einheiten) ---
# q1: Strafe fuer Auslenkung der Aufhaengung (1/m^2).
# q2: Strafe fuer Aufbaugeschwindigkeit (1/(m/s)^2).
# q3: Strafe fuer Reifeneinfederung (1/m^2).
# q4: Strafe fuer Radgeschwindigkeit (1/(m/s)^2).
# r:  Strafe fuer die Stellkraft des aktiven Daempfers (1/N^2).
q1 = 1.0e4
q2 = 5.0e5
q3 = 1.0e3
q4 = 1.0
r = 1.0e-6
q1_N = 1.0e1 * q1
q2_N = 1.0e1 * q2
q3_N = 1.0e1 * q3
q4_N = 1.0e1 * q4
# --- Symbolische QP-Formulierung mit map() in CasADi ---
X = SX.sym("X", 4, N + 1)
U = SX.sym("U", 1, N)
p = SX.sym("p", 4 + N)
x_init = p[0:4]
V_r = p[4:].T
x_curr_sym = SX.sym("x_curr", 4)
x_next_sym = SX.sym("x_next", 4)
u_curr_sym = SX.sym("u_curr", 1)
v_curr_sym = SX.sym("v_curr", 1)
f_dyn = Function(
    "f_dyn",
    [x_next_sym, x_curr_sym, u_curr_sym, v_curr_sym],
    [x_next_sym - (A @ x_curr_sym + B @ u_curr_sym + G * v_curr_sym)],
)
F_dyn = f_dyn.map(N)
g_dyn = F_dyn(X[:, 1:], X[:, 0:N], U, V_r)
g_init = (X[:, 0]) - x_init
g = vertcat(g_init, reshape(g_dyn, -1, 1))
f_stage = Function(
    "f_stage",
    [x_curr_sym, u_curr_sym],
    [
        q1 * ((x_curr_sym[0]) ** 2)
        + q2 * ((x_curr_sym[1]) ** 2)
        + q3 * ((x_curr_sym[2]) ** 2)
        + q4 * ((x_curr_sym[3]) ** 2)
        + r * ((u_curr_sym[0]) ** 2)
    ],
)
F_stage = f_stage.map(N)
stage_costs = F_stage(X[:, 0:N], U)
f = (
    sum2(stage_costs)
    + q1_N * ((X[0, N]) ** 2)
    + q2_N * ((X[1, N]) ** 2)
    + q3_N * ((X[2, N]) ** 2)
    + q4_N * ((X[3, N]) ** 2)
)
qp = {
    ("x"): (vertcat(reshape(X, -1, 1), reshape(U, -1, 1))),
    ("p"): (p),
    ("f"): (f),
    ("g"): (g),
}
S = qpsol("S", "qpoases", qp, {("printLevel"): ("none")})
# --- Simulations-Setup ---
sim_time = 3.0
N_steps = int(sim_time / dt)
t_vec = np.linspace(0.0, sim_time, N_steps)
# Bodenschwelle (5cm Hoehe) von t=0.5s bis t=0.7s
z_r_vec = np.zeros(N_steps)
v_r_vec = np.zeros(N_steps)
for j in range(N_steps):
    t_curr = t_vec[j]
    if 0.5 <= t_curr and t_curr <= 0.7:
        z_r_vec[j] = 2.5e-2 * (1.0 - np.cos((2.0 * np.pi * (t_curr - 0.5)) / 0.2))
        v_r_vec[j] = (
            2.5e-2
            * ((2.0 * np.pi) / 0.2)
            * np.sin((2.0 * np.pi * (t_curr - 0.5)) / 0.2)
        )
    else:
        z_r_vec[j] = 0.0
        v_r_vec[j] = 0.0
# --- Simulation des passiven Systems mittels mapaccum() ---
# Da das passive System ohne aktiven Regler u=0 laeuft, koennen wir
# die gesamte Zustandshistorie ohne Schleife mit mapaccum() berechnen.
x_curr_sym_p = SX.sym("x_curr_p", 4)
v_curr_sym_p = SX.sym("v_curr_p", 1)
f_passive_step = Function(
    "f_passive_step",
    [x_curr_sym_p, v_curr_sym_p],
    [A @ x_curr_sym_p + G * v_curr_sym_p],
)
f_passive_sim = f_passive_step.mapaccum("f_passive_sim", N_steps - 1)
x_curr_passive_init = np.array([0.0, 0.0, 0.0, 0.0])
x_hist_passive_matrix = f_passive_sim(x_curr_passive_init, v_r_vec[:-1])
x_hist_passive = hcat([x_curr_passive_init, x_hist_passive_matrix]).full()
# --- Simulation des aktiven (MPC) Systems ---
x_hist_mpc = np.zeros(
    (
        4,
        N_steps,
    )
)
u_hist_mpc = np.zeros(N_steps)
x_curr = np.array([0.0, 0.0, 0.0, 0.0])
lb_state = np.array([-8.0e-2, -1.0e1, -5.0e-2, -2.0e1])
ub_state = np.array([8.0e-2, 1.0e1, 5.0e-2, 2.0e1])
lb_input = np.array([-1.5e3])
ub_input = np.array([1.5e3])
lbx = np.concatenate(
    (
        np.tile(lb_state, N + 1),
        np.tile(lb_input, N),
    )
)
ubx = np.concatenate(
    (
        np.tile(ub_state, N + 1),
        np.tile(ub_input, N),
    )
)
lbg = np.zeros(4 * (N + 1))
ubg = np.zeros(4 * (N + 1))
x0_guess = np.zeros(4 * (N + 1) + N)
for j in range(N_steps - 1):
    x_hist_mpc[:, j] = x_curr
    V_r_horiz = np.zeros(N)
    for k in range(N):
        idx = j + k
        if idx < N_steps:
            V_r_horiz[k] = v_r_vec[idx]
        else:
            V_r_horiz[k] = 0.0
    p_val = np.concatenate(
        (
            x_curr,
            V_r_horiz,
        )
    )
    sol = S(x0=x0_guess, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    x_opt = sol["x"]
    u_opt = float(x_opt[4 * (N + 1)])
    u_hist_mpc[j] = u_opt
    x_curr = A_d_np @ x_curr + B_d_np * u_opt + G_d_np * v_r_vec[j]
x_hist_mpc[:, -1] = x_curr
# --- Berechnungen der Komfortmetriken ---
acc_mpc = (
    (-ks / ms) * x_hist_mpc[0]
    + (-cs / ms) * ((x_hist_mpc[1]) - (x_hist_mpc[3]))
    + (1.0 / ms) * u_hist_mpc
)
acc_passive = (-ks / ms) * x_hist_passive[0] + (-cs / ms) * (
    (x_hist_passive[1]) - (x_hist_passive[3])
)
zs_mpc = x_hist_mpc[0] + x_hist_mpc[2] + z_r_vec
zs_passive = x_hist_passive[0] + x_hist_passive[2] + z_r_vec
# --- Plotting und Visualisierung ---
plt.rcParams.update(
    {
        ("font.family"): ("sans-serif"),
        ("font.sans-serif"): (["DejaVu Sans", "Arial"]),
        ("axes.edgecolor"): ("#cccccc"),
        ("axes.linewidth"): (0.8),
        ("grid.color"): ("#eeeeee"),
        ("grid.linestyle"): ("-"),
    }
)
fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        14,
        10,
    ),
)
fig.suptitle(
    "Kompaktes Active MPC vs. Passives Fahrwerk (CasADi Map-Formulierung)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
ax1 = axes[0, 0]
ax1.fill_between(
    t_vec, 0, z_r_vec, color="#e0e0e0", alpha=0.5, label="Strassenprofil (Schwelle)"
)
ax1.plot(
    t_vec,
    zs_passive,
    label="Passives Fahrwerk",
    color="#ff4d4d",
    linestyle="--",
    lw=1.5,
)
ax1.plot(t_vec, zs_mpc, label="Aktives Fahrwerk (Map-MPC)", color="#1a73e8", lw=2.5)
ax1.set_title("Aufbauposition (zs)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Zeit (s)", fontsize=10)
ax1.set_ylabel("Auslenkung (m)", fontsize=10)
ax1.grid(True, alpha=0.6)
ax1.legend(frameon=True, facecolor="white", edgecolor="none")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2 = axes[0, 1]
ax2.axhspan(-0.5, 0.5, color="#e2f0d9", alpha=0.6, label="Komfortzone (ISO 2631)")
ax2.plot(t_vec, acc_passive, label="Passiv", color="#ff4d4d", linestyle="--", lw=1.5)
ax2.plot(t_vec, acc_mpc, label="Aktiv (Map-MPC)", color="#1a73e8", lw=2.5)
ax2.set_title("Aufbaubeschleunigung (Chassis)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Zeit (s)", fontsize=10)
ax2.set_ylabel("Beschleunigung (m/s^2)", fontsize=10)
ax2.grid(True, alpha=0.6)
ax2.legend(frameon=True, facecolor="white", edgecolor="none")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax3 = axes[1, 0]
ax3.axhspan(-8.0e-2, 8.0e-2, color="#f1f3f4", alpha=0.6, zorder=0)
ax3.plot(
    t_vec, x_hist_passive[0], label="Passiv", color="#ff4d4d", linestyle="--", lw=1.5
)
ax3.plot(t_vec, x_hist_mpc[0], label="Aktiv (Map-MPC)", color="#1a73e8", lw=2.5)
ax3.axhline(
    y=8.0e-2,
    color="#cc0000",
    linestyle=":",
    lw=1.2,
    label="Aufhaengungsweg-Grenze (+/- 8cm)",
)
ax3.axhline(y=-8.0e-2, color="#cc0000", linestyle=":", lw=1.2)
ax3.set_title("Federweg (Aufbau relativ zu Rad)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Zeit (s)", fontsize=10)
ax3.set_ylabel("Federweg (m)", fontsize=10)
ax3.grid(True, alpha=0.6)
ax3.legend(frameon=True, facecolor="white", edgecolor="none")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax4 = axes[1, 1]
ax4.axhspan(-1.5e3, 1.5e3, color="#f1f3f4", alpha=0.6, zorder=0)
ax4.step(
    t_vec,
    u_hist_mpc,
    label="Aktive Stellkraft (Map-MPC)",
    color="#34a853",
    lw=2,
    where="post",
)
ax4.axhline(
    y=1.5e3,
    color="#cc0000",
    linestyle=":",
    lw=1.2,
    label="Aktuator-Kraftgrenze (+/- 1500N)",
)
ax4.axhline(y=-1.5e3, color="#cc0000", linestyle=":", lw=1.2)
ax4.set_title("Aktuator-Stellkraft", fontsize=12, fontweight="bold")
ax4.set_xlabel("Zeit (s)", fontsize=10)
ax4.set_ylabel("Kraft (N)", fontsize=10)
ax4.grid(True, alpha=0.6)
ax4.legend(frameon=True, facecolor="white", edgecolor="none")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("active_suspension_mpc_map.png", dpi=150)
print("Plot gespeichert als active_suspension_mpc_map.png")
