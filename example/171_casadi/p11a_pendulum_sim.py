from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# --- Physikalische Modellierung (Inverted Pendulum) ---
# M: Masse des Wagens (kg)
# m: Masse des Pendels (kg)
# l: Laenge des Pendels zum Schwerpunkt (m)
# g: Erdbeschleunigung (m/s^2)
M = 1.0
m = 0.1
l = 0.5
g = 9.81
# Zustandsvariablen:
# s     : Position des Wagens (m)
# v     : Geschwindigkeit des Wagens (m/s)
# theta : Winkel des Pendels (rad), 0 = aufrecht, pi = haengend
# omega : Winkelgeschwindigkeit des Pendels (rad/s)
nx = 4
nu = 1
x = SX.sym("x", nx)
u = SX.sym("u", nu)
s_ = x[0]
v_ = x[1]
theta_ = x[2]
omega_ = x[3]
F_ = u
# Nichtlineare Systemdynamik (dx/dt = f(x, u))
sin_theta = np.sin(theta_)
cos_theta = np.cos(theta_)
den = M + m * (1.0 - (cos_theta * cos_theta))
ds = v_
dv = (F_ + m * l * omega_ * omega_ * sin_theta + m * g * cos_theta * sin_theta) / den
dtheta = omega_
domega = (
    (-1.0 * F_ * cos_theta)
    - (m * l * omega_ * omega_ * sin_theta * cos_theta)
    - ((M + m) * g * sin_theta)
) / (l * den)
f_ode = Function("f_ode", [x, u], [vertcat(ds, dv, dtheta, domega)])
# --- Opti-Stack & Kollokation Setup ---
opti = Opti()
N = 50
T = 3.0
h = T / N
# Kollokations-Parameter (Radau-Stützstellen, Polynomgrad d=3)
d = 3
tau_root = np.append(0.0, collocation_points(d, "radau"))
# Konstruktion der Kollokationsmatrizen C und D fuer die Lagrange-Polynome
C = np.zeros(
    (
        d + 1,
        d + 1,
    )
)
D = np.zeros(d + 1)
for j in range(d + 1):
    p = np.poly1d([1.0])
    for r in range(d + 1):
        if r != j:
            p = p * (np.poly1d([1.0, -tau_root[r]]) / ((tau_root[j]) - (tau_root[r])))
    D[j] = p(1.0)
    pder = np.polyder(p)
    for r in range(d + 1):
        C[j, r] = pder(tau_root[r])
# Entscheidungsvariablen im Opti-Stack definieren
X = opti.variable(nx, N + 1)
Xc = []
for k in range(N):
    Xc_k = []
    for r in range(d):
        Xc_k.append(opti.variable(nx))
    Xc.append(Xc_k)
U = opti.variable(nu, N)
# Start- und Endbedingungen (Swing-Up)
x_start = np.array([0.0, 0.0, np.pi, 0.0])
x_target = np.array([1.0, 0.0, 0.0, 0.0])
opti.subject_to(X[:, 0] == x_start)
opti.subject_to(X[:, N] == x_target)
# Nebenbedingungen (Bounds)
opti.subject_to(opti.bounded(-2.0, X[0, :], 2.0))
opti.subject_to(opti.bounded(-1.5e1, U, 1.5e1))
# Kollokations-Gleichungen & Intervall-Kontinuitaet erzwingen
for k in range(N):
    Xk = X[:, k]
    x_end = D[0] * Xk
    for j in range(1, d + 1):
        xp = C[0, j] * Xk
        for r in range(d):
            xp = xp + C[r + 1, j] * Xc[k][r]
        f_eval = f_ode(Xc[k][j - 1], U[:, k])
        opti.subject_to(xp == h * f_eval)
    for r in range(d):
        x_end = x_end + D[r + 1] * Xc[k][r]
    opti.subject_to(X[:, k + 1] == x_end)
# Kostenfunktion (Regelungsaufwand minimieren)
cost = 0.0
for k in range(N):
    cost = cost + 1.0e-2 * ((U[0, k]) ** 2)
opti.minimize(cost)
# Startschätzung fuer den Solver (Warm-Start hilft enorm)
opti.set_initial(X, np.linspace(x_start, x_target, N + 1).T)
# Solver Konfiguration: IPOPT
opti.solver("ipopt", {}, {("print_level"): (5)})
sol = opti.solve()
# --- Ergebnisse auslesen und Plotten ---
t_grid = np.linspace(0.0, T, N + 1)
X_res = sol.value(X)
U_res = sol.value(U)
fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        10,
        12,
    ),
)
axes[0].plot(t_grid, X_res[0, :], label="s (Wagenposition)")
axes[0].set_ylabel("Position (m)")
axes[0].legend()
axes[0].grid()
axes[1].plot(t_grid, X_res[2, :], label="theta (Pendelwinkel)")
axes[1].set_ylabel("Winkel (rad)")
axes[1].legend()
axes[1].grid()
axes[2].step(t_grid[0:-1], U_res, label="F (Aktuatorkraft)", where="post")
axes[2].set_ylabel("Kraft (N)")
axes[2].set_xlabel("Zeit (s)")
axes[2].legend()
axes[2].grid()
plt.tight_layout()
plt.savefig("p11a_pendulum_sim.png")
print("Simulation erfolgreich. Plot gespeichert.")
