from __future__ import annotations
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 2-STUFIGER DIODEN-GEKLEMMTER KONDENSATORFILTER (DAE-SYSTEM)
# ========================================================================
#
# 1. BESCHREIBUNG DER SCHALTUNG & VERWENDUNGSZWECK:
#    Diese Schaltung repraesentiert einen zweistufigen passiven Tiefpassfilter,
#    bei dem jeder Kondensatorstufe eine Diode parallel geschaltet ist.
#    Die Dioden dienen als Spannungsbegrenzer (Clamping-Dioden).
#    Die Schaltung wird von einer konstanten Ladestromquelle I_in gespeist.
#
#    Anwendungsbereiche:
#    - Signalbegrenzung (Cipping/Limiting) in HF- und Mischsignalschaltungen,
#      um empfindliche nachfolgende Stufen vor Ueberspannungen zu schuetzen.
#    - Spitzenwertdetektoren (Peak Detector) und Huellkurvendetektoren.
#    - Ueberspannungsschutz- und Gleichrichterschaltungen.
#
# 2. VOR- UND NACHTEILE DER SCHALTUNG:
#    Vorteile:
#    - Einfaches, rein passives Design mit sehr wenigen Bauelementen.
#    - Sehr schnelle Reaktionszeit (Klemmung) durch das passive Diodenverhalten.
#    - Effektive Begrenzung von Spannungsspitzen.
#    Nachteile:
#    - Starke Abhaengigkeit von Diodeneigenschaften (Schnittspannung, Sättigungsstrom),
#      die stark temperaturabhaengig sind.
#    - Verlustleistung ueber die Dioden.
#    - Schwer manuell abzustimmen wegen der exponentiellen Nichtlinearitaeten.
#
# 3. RELEVANZ FUER ELEKTRONIKINGENIEURE & ZIELSETZUNG:
#    Fuer das Design solcher Schaltungen ist die korrekte Dimensionierung
#    der Kapazitaeten C1 und C2 entscheidend, um Spannungsziele bei bestimmten
#    Zeitkonstanten einzuhalten. Hier nutzen wir die Sensitivitaetsanalyse und
#    die DAE-beschraenkte Optimierung (IPOPT mit CasADi IDAS), um die Kapazitaeten
#    automatisch so zu optimieren, dass die Ausgangsspannung V_C2 am Ende eines
#    Zeitraums maximiert wird, waehrend das gesamte Kapazitaetsbudget beschraenkt ist.
#
# 4. SCHALTPLAN (ASCII-ART):
#
#                    Knoten 1           Knoten 2
#    I_in              V_C1               V_C2
#     o-------+-----------o------[ R ]-------o-----------o (Ausgang V_out)
#             |           |                  |
#             |           |                  |
#           [I_in]      [ C1 ]             [ C2 ]
#             |           |                  |
#             |         + | -              + | -
#             |         ( | )              ( | )
#             |           |                  |
#             |       D1  |              D2  |
#             |          / \                / \
#             |         / v \              / v \
#             |         -----              -----
#             |           |                  |
#             |           |                  |
#    ---------+-----------+------------------+----------- GND
#
# 5. ERGEBNISSE DIESES PROJEKTS (ZUSAMMENFASSUNG):
#    - DAE-Modellierung: Formulierung eines Widerstands-Kondensator-Dioden-Netzwerks
#      in symbolischen CasADi-Ausdruecken mit 2 differenziellen Zustaenden (Spannungen),
#      2 algebraischen Zustaenden (Diodenstroemen) und 3 Parametern.
#    - Integrator-Erstellung: Initialisierung von Einschritt- und Zeitgitter-Lösern
#      ueber die CasADi 'integrator'-Schnittstelle mit dem IDAS-Plugin.
#    - Sensitivitaetsanalyse: Symbolisches Einwickeln des DAE-Lösers zur Berechnung
#      von Gradienten (1. Ordnung) und praezise Kreuzvalidierung der Hessian-Matrizen
#      (2. Ordnung) mittels Adjoint-over-Adjoint (AOA) und Forward-over-Adjoint (FOA).
#    - DAE-Optimierung: Erfolgreiches Auffinden der optimalen Kapazitaetsaufteilung
#      (C1 = 1.9F, C2 = 0.1F), um die Ausgangsspannung V_C2 zu maximieren.
# ========================================================================
x = ca.SX.sym("x", 2)
z = ca.SX.sym("z", 2)
p = ca.SX.sym("p", 3)
V_C1 = x[0]
V_C2 = x[1]
I_D1 = z[0]
I_D2 = z[1]
I_in = p[0]
C1 = p[1]
C2 = p[2]
R = 2.0
Is1 = 0.10
Is2 = 0.10
Vt1 = 0.50
Vt2 = 0.50
ode = ca.vertcat(
    (((I_in) - (I_D1) - (((V_C1) - (V_C2)) / (R))) / (C1)),
    (((((V_C1) - (V_C2)) / (R)) - (I_D2)) / (C2)),
)
alg = ca.vertcat(
    ((I_D1) - ((Is1) * ((ca.exp(((V_C1) / (Vt1)))) - (1.0)))),
    ((I_D2) - ((Is2) * ((ca.exp(((V_C2) / (Vt2)))) - (1.0)))),
)
dae = {("x"): (x), ("z"): (z), ("p"): (p), ("ode"): (ode), ("alg"): (alg)}
t0 = 0.0
t_grid = np.linspace((0.0), (1.0), 100)
# Integratoren unter Verwendung des 'idas'-Plugins definieren
# F_tf   - Integriert bis zur Endzeit tf=1.0 (optimiert fuer NLP/Hessian)
# F_grid - Integriert auf dem gesamten t_grid (fuer Trajektorien und Sensitivitaetsverlaeufe)
F_tf = ca.integrator("F_tf", "idas", dae, (0.0), (1.0))
F_grid = ca.integrator("F_grid", "idas", dae, t0, t_grid)
# ========================================================================
# SYMBOLIC SENSITIVITY WRAPPERS
# ========================================================================
# Wir wickeln die DAE-Integration in eine Standard-CasADi-Funktion ein,
# um analytische Ableitungen durch den Solver-Block zu ermoeglichen.
res_tf = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p)
V_C2_tf = res_tf["xf"][1]
# Gradient (Sensitivitaet 1. Ordnung) von V_C2(t_f) bzgl. der Parameter [I_in, C1, C2]
grad_V_C2 = ca.jacobian(V_C2_tf, p).T
# Hessian (Sensitivitaet 2. Ordnung) von V_C2(t_f) bzgl. der Parameter
# Modus 1: Adjoint-over-Adjoint (AOA) - Rueckwaerts-ueber-Adjungierte
H_aoa, g_aoa = ca.hessian(V_C2_tf, p)
f_aoa = ca.Function("f_aoa", [p], [H_aoa, g_aoa])
# Modus 2: Forward-over-Adjoint (FOA) - Vorwaerts-Ableitung des adjungierten Gradienten
H_foa = ca.jacobian(grad_V_C2, p)
f_foa = ca.Function("f_foa", [p], [H_foa])
# ========================================================================
# NOMINALE SENSITIVITAETSANALYSE
# ========================================================================
p_nom = [(2.0), (1.0), (1.0)]
H_aoa_val, g_aoa_val = f_aoa(p_nom)
H_foa_val = f_foa(p_nom)
print("--- Nominal Parameter Sensitivity (p = [I_in=2.0, C1=1.0, C2=1.0]) ---")
print(f"Gradient (AOA):\n{g_aoa_val}")
print(f"Hessian (AOA):\n{H_aoa_val}")
print(f"Hessian (FOA):\n{H_foa_val}")
print(
    f"Hessian agreement check (max absolute difference): {np.max(np.abs(np.array(H_aoa_val) - np.array(H_foa_val)))}"
)
# ========================================================================
# DAE-BESCHRAENKTE PARAMETEROPTIMIERUNG
# ========================================================================
# Maximierung der Ausgangsspannung V_C2(t_f) unter der Randbedingung
# der Gesamtkapazitaet C1 + C2 = 2.0 F, bei festem Ladestrom I_in = 2.0 A.
opti = ca.Opti()
p_var = opti.variable(3)
opti.subject_to(((p_var[0]) == (2.0)))
opti.subject_to((((p_var[1]) + (p_var[2])) == (2.0)))
opti.subject_to(((p_var[1]) >= (0.10)))
opti.subject_to(((p_var[2]) >= (0.10)))
opti.set_initial(p_var, p_nom)
res_opt = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_var)
V_C2_tf_opt = res_opt["xf"][1]
opti.minimize(((-1.0) * (V_C2_tf_opt)))
opti.solver("ipopt", {}, {("print_level"): (0)})
sol = opti.solve()
p_opt = sol.value(p_var)
V_C2_tf_max = sol.value(V_C2_tf_opt)
print("--- DAE-Constrained Optimization Results ---")
print(f"Optimal Capacitances: C1 = {p_opt[1]:.4f} F, C2 = {p_opt[2]:.4f} F")
print(
    f"Maximum V_C2(t_f): {V_C2_tf_max:.4f} V (Nominal: {float(F_tf(x0=[0, 0], z0=[0, 0], p=p_nom)['xf'][1]):.4f} V)"
)
# ========================================================================
# TRAJEKTORIEN-SIMULATION UND SENSITIVITAETS-VERLAEUFE
# ========================================================================
sim_nom = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_nom)
sim_opt = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt)
xf_nom = np.array(sim_nom["xf"])
zf_nom = np.array(sim_nom["zf"])
xf_opt = np.array(sim_opt["xf"])
zf_opt = np.array(sim_opt["zf"])
# Berechnung der Zeitverlaeufe der Sensitivitaeten erster Ordnung fuer V_C2(t)
res_grid_sym = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p)
V_C2_traj_sym = res_grid_sym["xf"][1, :]
J_traj_sym = ca.jacobian(V_C2_traj_sym, p)
J_func = ca.Function("J_func", [p], [J_traj_sym])
J_val = np.array(J_func(p_nom))
# 2D Parametersweep ueber C1 und C2 zur Konturdarstellung
C1_vals = np.linspace((0.10), (3.0), 40)
C2_vals = np.linspace((0.10), (3.0), 40)
C1_grid, C2_grid = np.meshgrid(C1_vals, C2_vals)
V_C2_tf_grid = np.zeros_like(C1_grid)
for i in range(len(C2_vals)):
    for j in range(len(C1_vals)):
        p_val_ij = [(2.0), C1_grid[i, j], C2_grid[i, j]]
        out = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_val_ij)
        V_C2_tf_grid[i, j] = float(out["xf"][1])
# ========================================================================
# VISUALISIERUNG UND PLOT-ERSTELLUNG
# ========================================================================
fig, axs = plt.subplots(
    2,
    2,
    figsize=(
        14,
        10,
    ),
)
if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
else:
    plt.style.use("default")
axs[0, 0].plot(t_grid, xf_nom[0, :], "b-", label="V_C1 (Nominal)")
axs[0, 0].plot(t_grid, xf_nom[1, :], "r-", label="V_C2 (Nominal)")
axs[0, 0].plot(t_grid, xf_opt[0, :], "b--", alpha=(0.70), label="V_C1 (Optimal)")
axs[0, 0].plot(t_grid, xf_opt[1, :], "r--", alpha=(0.70), label="V_C2 (Optimal)")
axs[0, 0].set_title(
    "Kondensatorspannungen ueber die Zeit", fontsize=12, fontweight="bold"
)
axs[0, 0].set_xlabel("Zeit [s]")
axs[0, 0].set_ylabel("Spannung [V]")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=(0.50))
axs[0, 1].plot(t_grid, zf_nom[0, :], "g-", label="I_D1 (Nominal)")
axs[0, 1].plot(t_grid, zf_nom[1, :], "m-", label="I_D2 (Nominal)")
axs[0, 1].plot(t_grid, zf_opt[0, :], "g--", alpha=(0.70), label="I_D1 (Optimal)")
axs[0, 1].plot(t_grid, zf_opt[1, :], "m--", alpha=(0.70), label="I_D2 (Optimal)")
axs[0, 1].set_title("Diodenstroeme ueber die Zeit", fontsize=12, fontweight="bold")
axs[0, 1].set_xlabel("Zeit [s]")
axs[0, 1].set_ylabel("Strom [A]")
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=(0.50))
axs[1, 0].plot(t_grid, J_val[:, 0], "k-", label="dV_C2/dI_in")
axs[1, 0].plot(t_grid, J_val[:, 1], "c-", label="dV_C2/dC1")
axs[1, 0].plot(t_grid, J_val[:, 2], "y-", label="dV_C2/dC2")
axs[1, 0].set_title(
    "Sensitivitaet von V_C2(t) bzgl. der Parameter", fontsize=12, fontweight="bold"
)
axs[1, 0].set_xlabel("Zeit [s]")
axs[1, 0].set_ylabel("Sensitivitaet")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=(0.50))
contour = axs[1, 1].contourf(C1_grid, C2_grid, V_C2_tf_grid, levels=20, cmap="viridis")
cbar = fig.colorbar(contour, ax=axs[1, 1])
cbar.set_label("V_C2(t_f) [V]", fontsize=11)
c1_line = np.linspace((0.10), (1.90), 100)
c2_line = (2.0) - (c1_line)
axs[1, 1].plot(c1_line, c2_line, "r--", lw=(2.0), label="Constraint: C1 + C2 = 2.0")
axs[1, 1].plot(
    [p_nom[1]], [p_nom[2]], "wo", ms=8, markeredgecolor="black", label="Nominal Point"
)
axs[1, 1].plot(
    [p_opt[1]], [p_opt[2]], "r*", ms=12, markeredgecolor="black", label="Optimal Point"
)
axs[1, 1].set_title("V_C2(t_f) vs Kapazitaeten", fontsize=12, fontweight="bold")
axs[1, 1].set_xlabel("C1 [F]")
axs[1, 1].set_ylabel("C2 [F]")
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=(0.30))
plt.suptitle(
    "2-Stufiger Dioden-Geklemmter Kondensatorfilter: Simulation, Sensitivitaeten & Optimierung",
    fontsize=14,
    fontweight="bold",
    y=(0.980),
)
plt.tight_layout()
plt.savefig("diode_sensitivities.png", dpi=150)
print("Plot saved to diode_sensitivities.png")
