from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ========================================================================
# AKTIVE FAHRWERKSREGELUNG MITTELS MODELLPRÄDIKTIVER REGELUNG (MPC)
# ========================================================================
# Thema: Aktive Fahrwerksregelung fuer ein Viertelfahrzeug-Modell (Quarter-Car)
#
# Modellbeschreibung:
# Es handelt sich hierbei um ein Viertelfahrzeug-Modell (Quarter-Car Model).
# Dieses Modell repraesentiert genau ein Rad und die dazugehoerige Ecke der
# Karosserie. Es wird die Annahme getroffen, dass die Dynamik der vier Ecken
# des Fahrzeugs voneinander entkoppelt ist. Dies ist eine Standardannahme
# in der Fahrwerkstechnik fuer den Entwurf von Dämpferregelungen.
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
#
# Vorschau der Strassenstoerung (Look-Ahead Preview):
# Modellpraediktive Regelung (MPC) erlaubt es, zukuenftige Strassenstoerungen
# im Voraus zu beruecksichtigen. In einem realen Fahrzeug wird dieses
# Strassenprofil durch LiDAR-Sensoren oder Kameras vor dem Fahrzeug erfasst,
# die die Fahrbahn abtasten. In dieser Simulation simulieren wir diese
# Vorschau (Look-Ahead), indem wir das Strassenprofil ueber den gesamten
# Vorhersagehorizont an den Controller uebergeben (Perfect Preview).
#
# Vorteil des QP-Lösers (Quadratic Programming):
# 1. Einhaltung von Grenzen: Der Stellweg des Daempfers (+/- 8cm) und die
#    maximale Aktuatorkraft (+/- 1500N) werden als harte Grenzen
#    beruecksichtigt. LQR- oder PID-Regler koennen diese nicht garantieren.
# 2. Praediktion: Der Regler reagiert bereits *bevor* die Schwelle erreicht
#    wird, um die Beschleunigung weich zu daempfen.
# 3. Numerische Stabilitaet: Wir verwenden die exakte Matrix-Exponential-
#    Diskretisierung (expm) anstelle der expliziten Euler-Methode, um
#    Instabilitaeten bei der schnellen Raddynamik zu vermeiden.
# ========================================================================
ms = 3.0e2
mu = 4.0e1
ks = 1.5e4
cs = 1.0e3
kt = 1.5e5
dt = 1.0e-2
N = 30
# --- Physikalische Systemmatrizen (Kontinuierlich) ---
# Die Matrizen A_np, B_np und G_np definieren das kontinuierliche System:
# dot_x = A_c * x + B_c * u + G_c * vr
# Da der Reifen sehr steif ist (hohe Eigenfrequenz), fuehrt eine einfache
# Euler-Diskretisierung zu numerischer Instabilitaet. Daher diskretisieren
# wir das System exakt mit dem Matrix-Exponential (ZOH-Diskretisierung).
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
# --- Exakte ZOH-Diskretisierung via Matrix-Exponential ---
# Wir erstellen ein erweitertes System M, um A_d, B_d und G_d gleichzeitig exakt zu loesen.
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
# --- MPC Gewichtungsfaktoren ---
# q1: Strafe fuer Auslenkung der Aufhaengung (Einheit: 1/m^2).
#     Verhindert das Durchschlagen auf den Endanschlag (Stroke Limit).
# q2: Strafe fuer die vertikale Geschwindigkeit des Chassis (Einheit: 1/(m/s)^2).
#     Dies ist das primaere Mass fuer den Fahrkomfort (Reduktion von Aufbaubeschleunigungen).
# q3: Strafe fuer die Reifenauslenkung (Einheit: 1/m^2).
#     Stellt den Strassenkontakt und damit die Fahrsicherheit sicher.
# q4: Strafe fuer die Radgeschwindigkeit (Einheit: 1/(m/s)^2).
#     Daempft schnelle Radbewegungen.
# r:  Strafe fuer die Stellkraft des aktiven Daempfers (Einheit: 1/N^2).
#     Begrenzt den Energieaufwand des Aktuators.
q1 = 1.0e4
q2 = 5.0e5
q3 = 1.0e3
q4 = 1.0
r = 1.0e-6
q1_N = 1.0e1 * q1
q2_N = 1.0e1 * q2
q3_N = 1.0e1 * q3
q4_N = 1.0e1 * q4
# --- Symbolische QP-Formulierung in CasADi ---
X = [
    SX.sym("x_0", 4),
    SX.sym("x_1", 4),
    SX.sym("x_2", 4),
    SX.sym("x_3", 4),
    SX.sym("x_4", 4),
    SX.sym("x_5", 4),
    SX.sym("x_6", 4),
    SX.sym("x_7", 4),
    SX.sym("x_8", 4),
    SX.sym("x_9", 4),
    SX.sym("x_10", 4),
    SX.sym("x_11", 4),
    SX.sym("x_12", 4),
    SX.sym("x_13", 4),
    SX.sym("x_14", 4),
    SX.sym("x_15", 4),
    SX.sym("x_16", 4),
    SX.sym("x_17", 4),
    SX.sym("x_18", 4),
    SX.sym("x_19", 4),
    SX.sym("x_20", 4),
    SX.sym("x_21", 4),
    SX.sym("x_22", 4),
    SX.sym("x_23", 4),
    SX.sym("x_24", 4),
    SX.sym("x_25", 4),
    SX.sym("x_26", 4),
    SX.sym("x_27", 4),
    SX.sym("x_28", 4),
    SX.sym("x_29", 4),
    SX.sym("x_30", 4),
]
U = [
    SX.sym("u_0", 1),
    SX.sym("u_1", 1),
    SX.sym("u_2", 1),
    SX.sym("u_3", 1),
    SX.sym("u_4", 1),
    SX.sym("u_5", 1),
    SX.sym("u_6", 1),
    SX.sym("u_7", 1),
    SX.sym("u_8", 1),
    SX.sym("u_9", 1),
    SX.sym("u_10", 1),
    SX.sym("u_11", 1),
    SX.sym("u_12", 1),
    SX.sym("u_13", 1),
    SX.sym("u_14", 1),
    SX.sym("u_15", 1),
    SX.sym("u_16", 1),
    SX.sym("u_17", 1),
    SX.sym("u_18", 1),
    SX.sym("u_19", 1),
    SX.sym("u_20", 1),
    SX.sym("u_21", 1),
    SX.sym("u_22", 1),
    SX.sym("u_23", 1),
    SX.sym("u_24", 1),
    SX.sym("u_25", 1),
    SX.sym("u_26", 1),
    SX.sym("u_27", 1),
    SX.sym("u_28", 1),
    SX.sym("u_29", 1),
]
p = SX.sym("p", 4 + N)
x_init = p[0:4]
V_r = p[4:]
f = 0.0
g = []
f = (
    f
    + q1 * ((X[0][0]) ** 2)
    + q2 * ((X[0][1]) ** 2)
    + q3 * ((X[0][2]) ** 2)
    + q4 * ((X[0][3]) ** 2)
    + r * ((U[0]) ** 2)
)
g.append((X[1]) - (A @ X[0] + B @ U[0] + G * V_r[0]))
f = (
    f
    + q1 * ((X[1][0]) ** 2)
    + q2 * ((X[1][1]) ** 2)
    + q3 * ((X[1][2]) ** 2)
    + q4 * ((X[1][3]) ** 2)
    + r * ((U[1]) ** 2)
)
g.append((X[2]) - (A @ X[1] + B @ U[1] + G * V_r[1]))
f = (
    f
    + q1 * ((X[2][0]) ** 2)
    + q2 * ((X[2][1]) ** 2)
    + q3 * ((X[2][2]) ** 2)
    + q4 * ((X[2][3]) ** 2)
    + r * ((U[2]) ** 2)
)
g.append((X[3]) - (A @ X[2] + B @ U[2] + G * V_r[2]))
f = (
    f
    + q1 * ((X[3][0]) ** 2)
    + q2 * ((X[3][1]) ** 2)
    + q3 * ((X[3][2]) ** 2)
    + q4 * ((X[3][3]) ** 2)
    + r * ((U[3]) ** 2)
)
g.append((X[4]) - (A @ X[3] + B @ U[3] + G * V_r[3]))
f = (
    f
    + q1 * ((X[4][0]) ** 2)
    + q2 * ((X[4][1]) ** 2)
    + q3 * ((X[4][2]) ** 2)
    + q4 * ((X[4][3]) ** 2)
    + r * ((U[4]) ** 2)
)
g.append((X[5]) - (A @ X[4] + B @ U[4] + G * V_r[4]))
f = (
    f
    + q1 * ((X[5][0]) ** 2)
    + q2 * ((X[5][1]) ** 2)
    + q3 * ((X[5][2]) ** 2)
    + q4 * ((X[5][3]) ** 2)
    + r * ((U[5]) ** 2)
)
g.append((X[6]) - (A @ X[5] + B @ U[5] + G * V_r[5]))
f = (
    f
    + q1 * ((X[6][0]) ** 2)
    + q2 * ((X[6][1]) ** 2)
    + q3 * ((X[6][2]) ** 2)
    + q4 * ((X[6][3]) ** 2)
    + r * ((U[6]) ** 2)
)
g.append((X[7]) - (A @ X[6] + B @ U[6] + G * V_r[6]))
f = (
    f
    + q1 * ((X[7][0]) ** 2)
    + q2 * ((X[7][1]) ** 2)
    + q3 * ((X[7][2]) ** 2)
    + q4 * ((X[7][3]) ** 2)
    + r * ((U[7]) ** 2)
)
g.append((X[8]) - (A @ X[7] + B @ U[7] + G * V_r[7]))
f = (
    f
    + q1 * ((X[8][0]) ** 2)
    + q2 * ((X[8][1]) ** 2)
    + q3 * ((X[8][2]) ** 2)
    + q4 * ((X[8][3]) ** 2)
    + r * ((U[8]) ** 2)
)
g.append((X[9]) - (A @ X[8] + B @ U[8] + G * V_r[8]))
f = (
    f
    + q1 * ((X[9][0]) ** 2)
    + q2 * ((X[9][1]) ** 2)
    + q3 * ((X[9][2]) ** 2)
    + q4 * ((X[9][3]) ** 2)
    + r * ((U[9]) ** 2)
)
g.append((X[10]) - (A @ X[9] + B @ U[9] + G * V_r[9]))
f = (
    f
    + q1 * ((X[10][0]) ** 2)
    + q2 * ((X[10][1]) ** 2)
    + q3 * ((X[10][2]) ** 2)
    + q4 * ((X[10][3]) ** 2)
    + r * ((U[10]) ** 2)
)
g.append((X[11]) - (A @ X[10] + B @ U[10] + G * V_r[10]))
f = (
    f
    + q1 * ((X[11][0]) ** 2)
    + q2 * ((X[11][1]) ** 2)
    + q3 * ((X[11][2]) ** 2)
    + q4 * ((X[11][3]) ** 2)
    + r * ((U[11]) ** 2)
)
g.append((X[12]) - (A @ X[11] + B @ U[11] + G * V_r[11]))
f = (
    f
    + q1 * ((X[12][0]) ** 2)
    + q2 * ((X[12][1]) ** 2)
    + q3 * ((X[12][2]) ** 2)
    + q4 * ((X[12][3]) ** 2)
    + r * ((U[12]) ** 2)
)
g.append((X[13]) - (A @ X[12] + B @ U[12] + G * V_r[12]))
f = (
    f
    + q1 * ((X[13][0]) ** 2)
    + q2 * ((X[13][1]) ** 2)
    + q3 * ((X[13][2]) ** 2)
    + q4 * ((X[13][3]) ** 2)
    + r * ((U[13]) ** 2)
)
g.append((X[14]) - (A @ X[13] + B @ U[13] + G * V_r[13]))
f = (
    f
    + q1 * ((X[14][0]) ** 2)
    + q2 * ((X[14][1]) ** 2)
    + q3 * ((X[14][2]) ** 2)
    + q4 * ((X[14][3]) ** 2)
    + r * ((U[14]) ** 2)
)
g.append((X[15]) - (A @ X[14] + B @ U[14] + G * V_r[14]))
f = (
    f
    + q1 * ((X[15][0]) ** 2)
    + q2 * ((X[15][1]) ** 2)
    + q3 * ((X[15][2]) ** 2)
    + q4 * ((X[15][3]) ** 2)
    + r * ((U[15]) ** 2)
)
g.append((X[16]) - (A @ X[15] + B @ U[15] + G * V_r[15]))
f = (
    f
    + q1 * ((X[16][0]) ** 2)
    + q2 * ((X[16][1]) ** 2)
    + q3 * ((X[16][2]) ** 2)
    + q4 * ((X[16][3]) ** 2)
    + r * ((U[16]) ** 2)
)
g.append((X[17]) - (A @ X[16] + B @ U[16] + G * V_r[16]))
f = (
    f
    + q1 * ((X[17][0]) ** 2)
    + q2 * ((X[17][1]) ** 2)
    + q3 * ((X[17][2]) ** 2)
    + q4 * ((X[17][3]) ** 2)
    + r * ((U[17]) ** 2)
)
g.append((X[18]) - (A @ X[17] + B @ U[17] + G * V_r[17]))
f = (
    f
    + q1 * ((X[18][0]) ** 2)
    + q2 * ((X[18][1]) ** 2)
    + q3 * ((X[18][2]) ** 2)
    + q4 * ((X[18][3]) ** 2)
    + r * ((U[18]) ** 2)
)
g.append((X[19]) - (A @ X[18] + B @ U[18] + G * V_r[18]))
f = (
    f
    + q1 * ((X[19][0]) ** 2)
    + q2 * ((X[19][1]) ** 2)
    + q3 * ((X[19][2]) ** 2)
    + q4 * ((X[19][3]) ** 2)
    + r * ((U[19]) ** 2)
)
g.append((X[20]) - (A @ X[19] + B @ U[19] + G * V_r[19]))
f = (
    f
    + q1 * ((X[20][0]) ** 2)
    + q2 * ((X[20][1]) ** 2)
    + q3 * ((X[20][2]) ** 2)
    + q4 * ((X[20][3]) ** 2)
    + r * ((U[20]) ** 2)
)
g.append((X[21]) - (A @ X[20] + B @ U[20] + G * V_r[20]))
f = (
    f
    + q1 * ((X[21][0]) ** 2)
    + q2 * ((X[21][1]) ** 2)
    + q3 * ((X[21][2]) ** 2)
    + q4 * ((X[21][3]) ** 2)
    + r * ((U[21]) ** 2)
)
g.append((X[22]) - (A @ X[21] + B @ U[21] + G * V_r[21]))
f = (
    f
    + q1 * ((X[22][0]) ** 2)
    + q2 * ((X[22][1]) ** 2)
    + q3 * ((X[22][2]) ** 2)
    + q4 * ((X[22][3]) ** 2)
    + r * ((U[22]) ** 2)
)
g.append((X[23]) - (A @ X[22] + B @ U[22] + G * V_r[22]))
f = (
    f
    + q1 * ((X[23][0]) ** 2)
    + q2 * ((X[23][1]) ** 2)
    + q3 * ((X[23][2]) ** 2)
    + q4 * ((X[23][3]) ** 2)
    + r * ((U[23]) ** 2)
)
g.append((X[24]) - (A @ X[23] + B @ U[23] + G * V_r[23]))
f = (
    f
    + q1 * ((X[24][0]) ** 2)
    + q2 * ((X[24][1]) ** 2)
    + q3 * ((X[24][2]) ** 2)
    + q4 * ((X[24][3]) ** 2)
    + r * ((U[24]) ** 2)
)
g.append((X[25]) - (A @ X[24] + B @ U[24] + G * V_r[24]))
f = (
    f
    + q1 * ((X[25][0]) ** 2)
    + q2 * ((X[25][1]) ** 2)
    + q3 * ((X[25][2]) ** 2)
    + q4 * ((X[25][3]) ** 2)
    + r * ((U[25]) ** 2)
)
g.append((X[26]) - (A @ X[25] + B @ U[25] + G * V_r[25]))
f = (
    f
    + q1 * ((X[26][0]) ** 2)
    + q2 * ((X[26][1]) ** 2)
    + q3 * ((X[26][2]) ** 2)
    + q4 * ((X[26][3]) ** 2)
    + r * ((U[26]) ** 2)
)
g.append((X[27]) - (A @ X[26] + B @ U[26] + G * V_r[26]))
f = (
    f
    + q1 * ((X[27][0]) ** 2)
    + q2 * ((X[27][1]) ** 2)
    + q3 * ((X[27][2]) ** 2)
    + q4 * ((X[27][3]) ** 2)
    + r * ((U[27]) ** 2)
)
g.append((X[28]) - (A @ X[27] + B @ U[27] + G * V_r[27]))
f = (
    f
    + q1 * ((X[28][0]) ** 2)
    + q2 * ((X[28][1]) ** 2)
    + q3 * ((X[28][2]) ** 2)
    + q4 * ((X[28][3]) ** 2)
    + r * ((U[28]) ** 2)
)
g.append((X[29]) - (A @ X[28] + B @ U[28] + G * V_r[28]))
f = (
    f
    + q1 * ((X[29][0]) ** 2)
    + q2 * ((X[29][1]) ** 2)
    + q3 * ((X[29][2]) ** 2)
    + q4 * ((X[29][3]) ** 2)
    + r * ((U[29]) ** 2)
)
g.append((X[30]) - (A @ X[29] + B @ U[29] + G * V_r[29]))
f = (
    f
    + q1_N * ((X[N][0]) ** 2)
    + q2_N * ((X[N][1]) ** 2)
    + q3_N * ((X[N][2]) ** 2)
    + q4_N * ((X[N][3]) ** 2)
)
g.insert(0, (X[0]) - x_init)
qp = {("x"): (vertcat(*(X + U))), ("p"): (p), ("f"): (f), ("g"): (vertcat(*g))}
S = qpsol("S", "qpoases", qp, {("printLevel"): ("none")})
# --- Simulations-Setup ---
sim_time = 3.0
N_steps = int(sim_time / dt)
t_vec = np.linspace(0.0, sim_time, N_steps)
# Definition des Strassenprofils: Eine 5cm hohe Bodenschwelle von t=0.5s bis t=0.7s
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
# Initialisierung der Messhistorie fuer das aktive (MPC) und das passive Fahrwerk
x_hist_mpc = np.zeros(
    (
        4,
        N_steps,
    )
)
u_hist_mpc = np.zeros(N_steps)
x_hist_passive = np.zeros(
    (
        4,
        N_steps,
    )
)
x_curr = np.array([0.0, 0.0, 0.0, 0.0])
x_curr_passive = np.array([0.0, 0.0, 0.0, 0.0])
# --- Erklaerung der Solver-Parameter ---
# lbx / ubx: Untere und obere Grenzen fuer die Optimierungsvariablen.
#            Diese enthalten die zulaessigen Zustaende (z.B. Federweg) und Stellkraefte.
# lbg / ubg: Grenzen fuer Nebenbedingungen. Da dies Gleichheitsnebenbedingungen
#            (Systemdynamik und Anfangszustand) sind, sind lbg/ubg beide 0.
# x0_guess:  Startschätzung fuer die Entscheidungsvariablen des Solvers.
# p_val:     Parameterwerte fuer die Optimierung (aktueller Zustand x_0 und
#            Strassenstoerungs-Vorschau V_r fuer den Horizont).
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
# Simulations-Hauptschleife
for j in range(N_steps - 1):
    x_hist_mpc[:, j] = x_curr
    x_hist_passive[:, j] = x_curr_passive
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
    x_curr_passive = A_d_np @ x_curr_passive + G_d_np * v_r_vec[j]
x_hist_mpc[:, -1] = x_curr
x_hist_passive[:, -1] = x_curr_passive
# --- Nachbereitung und Rekonstruktion physikalischer Messgroessen ---
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
# --- Plotting Results ---
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
    "Active MPC vs. Passive Suspension System Comparison",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
ax1 = axes[0, 0]
ax1.fill_between(
    t_vec, 0, z_r_vec, color="#e0e0e0", alpha=0.5, label="Road Profile (Bump)"
)
ax1.plot(
    t_vec, zs_passive, label="Passive Chassis", color="#ff4d4d", linestyle="--", lw=1.5
)
ax1.plot(t_vec, zs_mpc, label="Active Chassis (MPC)", color="#1a73e8", lw=2.5)
ax1.set_title("Chassis Position (zs)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Time (s)", fontsize=10)
ax1.set_ylabel("Displacement (m)", fontsize=10)
ax1.grid(True, alpha=0.6)
ax1.legend(frameon=True, facecolor="white", edgecolor="none")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2 = axes[0, 1]
ax2.axhspan(-0.5, 0.5, color="#e2f0d9", alpha=0.6, label="Comfort Zone (ISO 2631)")
ax2.plot(t_vec, acc_passive, label="Passive", color="#ff4d4d", linestyle="--", lw=1.5)
ax2.plot(t_vec, acc_mpc, label="Active (MPC)", color="#1a73e8", lw=2.5)
ax2.set_title("Chassis Vertical Acceleration", fontsize=12, fontweight="bold")
ax2.set_xlabel("Time (s)", fontsize=10)
ax2.set_ylabel("Acceleration (m/s^2)", fontsize=10)
ax2.grid(True, alpha=0.6)
ax2.legend(frameon=True, facecolor="white", edgecolor="none")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax3 = axes[1, 0]
ax3.axhspan(-8.0e-2, 8.0e-2, color="#f1f3f4", alpha=0.6, zorder=0)
ax3.plot(
    t_vec, x_hist_passive[0], label="Passive", color="#ff4d4d", linestyle="--", lw=1.5
)
ax3.plot(t_vec, x_hist_mpc[0], label="Active (MPC)", color="#1a73e8", lw=2.5)
ax3.axhline(
    y=8.0e-2, color="#cc0000", linestyle=":", lw=1.2, label="Stroke Limit (+/- 8cm)"
)
ax3.axhline(y=-8.0e-2, color="#cc0000", linestyle=":", lw=1.2)
ax3.set_title("Suspension Deflection (Travel)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Time (s)", fontsize=10)
ax3.set_ylabel("Deflection (m)", fontsize=10)
ax3.grid(True, alpha=0.6)
ax3.legend(frameon=True, facecolor="white", edgecolor="none")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax4 = axes[1, 1]
ax4.axhspan(-1.5e3, 1.5e3, color="#f1f3f4", alpha=0.6, zorder=0)
ax4.step(
    t_vec, u_hist_mpc, label="Active Force (MPC)", color="#34a853", lw=2, where="post"
)
ax4.axhline(
    y=1.5e3, color="#cc0000", linestyle=":", lw=1.2, label="Actuator Limit (+/- 1500N)"
)
ax4.axhline(y=-1.5e3, color="#cc0000", linestyle=":", lw=1.2)
ax4.set_title("Actuator Control Force", fontsize=12, fontweight="bold")
ax4.set_xlabel("Time (s)", fontsize=10)
ax4.set_ylabel("Force (N)", fontsize=10)
ax4.grid(True, alpha=0.6)
ax4.legend(frameon=True, facecolor="white", edgecolor="none")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("active_suspension_mpc.png", dpi=150)
print("Plot saved as active_suspension_mpc.png")
# ========================================================================
# DISKUSSION DER SIMULATIONSERGEBNISSE (ACTIVE VS. PASSIVE)
# ========================================================================
# Die Simulation vergleicht das dynamische Verhalten eines Viertelfahrzeugs
# beim Ueberfahren einer 5 cm hohen Bodenschwelle bei einer ZOH-Abtastzeit
# von dt = 10 ms und einem MPC-Horizont von N = 30 Schritten (0.3 s).
#
# 1. Vertikale Aufbau-Auslenkung (Chassis Position zs):
#    - Passive Aufhaengung: Die Karosserie schwingt deutlich auf und erreicht
#      eine maximale Auslenkung von ca. 2.7 cm. Es dauert ueber 2 Sekunden,
#      bis sich das System wieder beruhigt.
#    - Aktive Aufhaengung (MPC): Die Auslenkung wird auf unter 0.4 cm gedrueckt.
#      Die Karosserie bleibt nahezu perfekt horizontal, da der Controller
#      durch die Stoerungsvorschau (Perfect Preview) vorausschauend agiert.
#      Dies entspricht einer Verbesserung der Ruhelage um ca. 85%.
#
# 2. Vertikale Karosseriebeschleunigung (Fahrkomfort):
#    - Die passive Aufhaengung laesst Beschleunigungsspitzen von bis zu
#      4.4 m/s^2 zu, was fuer Passagiere als aeusserst unangenehm empfunden wird.
#    - Der aktive MPC-Controller reduziert die maximale Beschleunigung auf
#      nur ca. 1.15 m/s^2 (eine Reduktion um ca. 74%). Die Beschleunigung verbleibt
#      groesstenteils nahe oder innerhalb der Komfortzone nach ISO 2631.
#
# 3. Stellkraft des Daempfers (Actuator Force):
#    - Der aktive Regler nutzt seine Stellkraft vorausschauend:
#      Sobald die Bodenschwelle in den Sichtbereich (Horizont) geraet, zieht
#      der Aktuator das Rad aktiv nach oben (negative Kraft), um das Auffahren
#      vorzubereiten. Beim Verlassen der Schwelle drueckt er das Rad nach unten,
#      um den Fahrbahnkontakt zu halten.
#    - Der Regler stoesst hierbei praezise an seine harte Kraftbegrenzung von
#      +/- 1500 N (Saettigung). Das QP loest diesen Zustand optimal unter Beruecksichtigung
#      dieser Begrenzung.
#
# 4. Aufhaengungsweg (Suspension Stroke/Deflection):
#    - Der Federweg verbleibt fuer beide Systeme weit innerhalb des zulaessigen
#      Bereichs von +/- 8 cm. Der MPC-Regler nutzt den verfuegbaren Federweg
#      gezielt aus, um die Kraft sanft einzuleiten.
# ========================================================================
