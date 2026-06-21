from __future__ import annotations
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time as time

# ========================================================================
# 2-STUFIGER DIODEN-GEKLEMMTER KONDENSATORFILTER (DAE-SYSTEM)
# ========================================================================
#
# 1. BESCHREIBUNG DER SCHALTUNG & TOPOLOGIE:
#    Diese Schaltung ist ein zweistufiger passiver Tiefpassfilter.
#    An jedem Kondensatorknoten liegt eine Halbleiterdiode nach Masse.
#    Die Schaltung wird von einer konstanten Ladestromquelle I_in gespeist.
#
#    Dieselbe Grundtopologie (RC + Diode) dient je nach Beschaltung
#    unterschiedlichen Zwecken:
#
#    (a) Signalbegrenzung (Clipping/Clamping):
#        Die Dioden sind PARALLEL zu einer Signalleitung geschaltet.
#        Sobald die Spannung am Knoten die Dioden-Schwellspannung erreicht,
#        leitet die Diode und klemmt die Spannung. Die Ausgangsspannung
#        am SELBEN Knoten wird so begrenzt. Anwendung: Schutz empfindlicher
#        HF-Eingaenge oder ADC-Vorstufen vor Ueberspannungsspitzen.
#
#    (b) Spitzenwertdetektor (Peak Detector):
#        Die Diode ist IN SERIE zum Signalpfad geschaltet (Kathode zum
#        Kondensator). Der Kondensator laedt sich ueber die Diode bis zum
#        Spitzenwert auf und haelt ihn, weil die Diode den Rueckfluss sperrt.
#        Die Ausgangsspannung wird AM KONDENSATOR abgegriffen.
#        Anwendung: Huellkurvendemodulatoren, AGC-Regelschleifen.
#
#    In unserem Modell verwenden wir die parallele Beschaltung (Fall a).
#    Die Dioden sind PARALLEL zu den Kondensatoren nach Masse geschaltet.
#
# 2. VOR- UND NACHTEILE:
#    Vorteile:
#    - Einfaches, rein passives Design mit sehr wenigen Bauelementen.
#    - Sehr schnelle Reaktion (Klemmung beginnt sofort bei Ueberspannung).
#    - Keine Versorgungsspannung noetig.
#    Nachteile:
#    - Starke Abhaengigkeit von Diodeneigenschaften (Sperrstrom Is,
#      Temperaturspannung Vt), die stark temperaturabhaengig sind.
#    - Verlustleistung ueber die Dioden (Waermeentwicklung).
#    - Die exponentiellen Nichtlinearitaeten machen eine analytische
#      Dimensionierung der Kapazitaeten praktisch unmoeglich.
#
# 3. RELEVANZ FUER ELEKTRONIKINGENIEURE & ZIELSETZUNG:
#    Bei der Dimensionierung dieser Schaltung stellt sich die Frage:
#    'Wie muss ich die Kapazitaeten C1 und C2 aufteilen, wenn mir ein
#     festes Gesamtkapazitaetsbudget (z.B. C1+C2 = 2F) zur Verfuegung
#     steht, um ein bestimmtes Ziel am Ausgang zu erreichen?'
#
#    Diese Frage beantworten wir mithilfe von CasADi (symbolisches DAE).
#    Wir vergleichen hier ZWEI moegliche Entwicklungsziele (Optima):
#    1. Spannungs-Maximierung: Wenn nur ein Spannungspegel triggern soll
#       (z.B. CMOS-Logikgatter, hochohmiger Eingang).
#    2. Energie-Maximierung: Wenn die Schaltung danach eine Last treiben
#       soll oder die Energie geerntet wird (Energy Harvesting). Hier
#       nuetzt eine hohe Spannung nichts, wenn die Kapazitaet C2 zu winzig
#       ist, um relevante Ladungsmengen bereitzustellen.
#
# 4. SCHALTPLAN (ASCII-ART):
#
#                    Knoten 1           Knoten 2
#    I_in              V_C1               V_C2
#     o-------+-----------o------[ R ]-------o-----------o (Ausgang)
#             |           |                  |
#             |        ---+---            ---+---
#           [I_in]     | C1  |            | C2  |
#             |        ---+---            ---+---
#             |           |                  |
#             |         D1|                D2|
#             |          /|                 /|
#             |         / |                / |
#             |        v  |               v  |
#             |           |                  |
#    ---------+-----------+------------------+--------- GND
#
#    Stromfluss an Knoten 1 (KCL): I_in = I_C1 + I_R + I_D1
#      I_C1 = C1 * dV_C1/dt
#      I_R  = (V_C1 - V_C2)/R
#      I_D1 = Is1*(exp(V_C1/Vt1) - 1)
#
#    Stromfluss an Knoten 2 (KCL): I_R = I_C2 + I_D2
#      I_C2 = C2 * dV_C2/dt
#      I_D2 = Is2*(exp(V_C2/Vt2) - 1)
#
# 5. BEOBACHTUNGEN UND INTERPRETATION DER ERGEBNISSE:
#    - Verlustleistung (Panel 3):
#      Die Dioden sollen Ueberspannungen ableiten. In einem schlecht
#      dimensionierten (nominalen) Filter oeffnen sie zu frueh, und
#      massiv viel Leistung (P = V * I) wird als Waerme verbrannt.
#      Das belastet die Quelle und kann die Dioden zerstoeren.
#      Die optimierten Designs lenken den Strom in die Kondensatoren
#      und minimieren so den thermischen Verlust (Flaeche unter der Kurve).
#    - Optima-Lage (Panel 5 & 6):
#      Spannungs-Optimum (Panel 5): Liegt am aeussersten physikalischen Rand
#      (C2=0.1F). Ein winziges C2 ist immer schneller voll. Es gibt hier keinen
#      Sweet Spot im Inneren.
#      Energie-Optimum (Panel 6): Energie (E = 0.5*C2*V_C2^2) erfordert einen
#      Kompromiss. Ein zu kleines C2 speichert keine Energie, ein zu grosses
#      C2 erreicht keine Spannung. Hier liegt das Optimum tief im Inneren
#      des Parameterraums! Dies ist ein klassischer Sweet Spot.
# ========================================================================
# ========================================================================
# SYMBOLISCHE DAE-DEFINITION
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
F_tf = ca.integrator("F_tf", "idas", dae, (0.0), (1.0))
F_grid = ca.integrator("F_grid", "idas", dae, t0, t_grid)
# ========================================================================
# SENSITIVITAETSANALYSE: GRADIENT UND HESSIAN
# ========================================================================
res_tf = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p)
V_C2_tf = res_tf["xf"][1]
grad_V_C2 = ca.jacobian(V_C2_tf, p).T
H_aoa, g_aoa = ca.hessian(V_C2_tf, p)
f_aoa = ca.Function("f_aoa", [p], [H_aoa, g_aoa])
H_foa = ca.jacobian(grad_V_C2, p)
f_foa = ca.Function("f_foa", [p], [H_foa])
p_nom = [(2.0), (1.0), (1.0)]
t_start = time.time()
H_aoa_val, g_aoa_val = f_aoa(p_nom)
t_aoa = (time.time()) - (t_start)
t_start = time.time()
H_foa_val = f_foa(p_nom)
t_foa = (time.time()) - (t_start)
print("--- Nominale Sensitivitaeten (p = [I_in=2.0 A, C1=1.0 F, C2=1.0 F]) ---")
print(f"Gradient: {g_aoa_val}")
print(f"Hessian (AOA): {H_aoa_val}  (Rechenzeit: {t_aoa * 1000:.2f} ms)")
print(f"Hessian (FOA): {H_foa_val}  (Rechenzeit: {t_foa * 1000:.2f} ms)")
print(
    f"Max. Abweichung AOA vs FOA: {np.max(np.abs(np.array(H_aoa_val) - np.array(H_foa_val))):.2e}"
)
# ========================================================================
# OPTIMIERUNG 1: SPANNUNGS-MAXIMIERUNG
# ========================================================================
opti_V = ca.Opti()
p_var_V = opti_V.variable(3)
opti_V.subject_to(((p_var_V[0]) == (2.0)))
opti_V.subject_to((((p_var_V[1]) + (p_var_V[2])) == (2.0)))
opti_V.subject_to(((p_var_V[1]) >= (0.10)))
opti_V.subject_to(((p_var_V[2]) >= (0.10)))
opti_V.set_initial(p_var_V, p_nom)
res_opt_V = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_var_V)
V_C2_tf_opt_V = res_opt_V["xf"][1]
opti_V.minimize(((-1.0) * (V_C2_tf_opt_V)))
opti_V.solver("ipopt", {}, {("print_level"): (0)})
t_start = time.time()
sol_V = opti_V.solve()
p_opt_V = sol_V.value(p_var_V)
V_C2_tf_max = sol_V.value(V_C2_tf_opt_V)
t_opt_V = (time.time()) - (t_start)
print("--- DAE-beschraenkte Optimierung: Spannung ---")
print(f"Optimale Kapazitaeten: C1 = {p_opt_V[1]:.4f} F, C2 = {p_opt_V[2]:.4f} F")
print(f"Max V_C2(t_f): {V_C2_tf_max:.4f} V")
print(f"Rechenzeit IPOPT: {t_opt_V * 1000:.1f} ms")
# ========================================================================
# OPTIMIERUNG 2: ENERGIE-MAXIMIERUNG
# ========================================================================
opti_E = ca.Opti()
p_var_E = opti_E.variable(3)
opti_E.subject_to(((p_var_E[0]) == (2.0)))
opti_E.subject_to((((p_var_E[1]) + (p_var_E[2])) == (2.0)))
opti_E.subject_to(((p_var_E[1]) >= (0.10)))
opti_E.subject_to(((p_var_E[2]) >= (0.10)))
opti_E.set_initial(p_var_E, p_nom)
res_opt_E = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_var_E)
V_C2_tf_opt_E = res_opt_E["xf"][1]
E_opt_E = (0.50) * (p_var_E[2]) * ((V_C2_tf_opt_E) ** (2))
opti_E.minimize(((-1.0) * (E_opt_E)))
opti_E.solver("ipopt", {}, {("print_level"): (0)})
t_start = time.time()
sol_E = opti_E.solve()
p_opt_E = sol_E.value(p_var_E)
E_tf_max = sol_E.value(E_opt_E)
t_opt_E = (time.time()) - (t_start)
print("--- DAE-beschraenkte Optimierung: Energie ---")
print(f"Optimale Kapazitaeten: C1 = {p_opt_E[1]:.4f} F, C2 = {p_opt_E[2]:.4f} F")
print(f"Max Energie E(t_f): {E_tf_max:.4f} J")
print(f"Rechenzeit IPOPT: {t_opt_E * 1000:.1f} ms")
# ========================================================================
# TRAJEKTORIEN-SIMULATION, VERLUSTLEISTUNG UND ZEITAUFGELOESTE SENSITIVITAETEN
# ========================================================================
t_start = time.time()
sim_nom = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_nom)
sim_optV = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt_V)
sim_optE = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt_E)
t_sim = (time.time()) - (t_start)
xf_nom = np.array(sim_nom["xf"])
zf_nom = np.array(sim_nom["zf"])
xf_optV = np.array(sim_optV["xf"])
zf_optV = np.array(sim_optV["zf"])
xf_optE = np.array(sim_optE["xf"])
zf_optE = np.array(sim_optE["zf"])
P_loss_nom = ((xf_nom[0, :]) * (zf_nom[0, :])) + ((xf_nom[1, :]) * (zf_nom[1, :]))
P_loss_optV = ((xf_optV[0, :]) * (zf_optV[0, :])) + ((xf_optV[1, :]) * (zf_optV[1, :]))
P_loss_optE = ((xf_optE[0, :]) * (zf_optE[0, :])) + ((xf_optE[1, :]) * (zf_optE[1, :]))
res_grid_sym = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p)
V_C2_traj_sym = res_grid_sym["xf"][1, :]
t_start = time.time()
J_traj_sym = ca.jacobian(V_C2_traj_sym, p)
J_func = ca.Function("J_func", [p], [J_traj_sym])
J_val = np.array(J_func(p_nom))
t_jac = (time.time()) - (t_start)
# ========================================================================
# 2D PARAMETERSWEEP FUER HEATMAPS (SPANNUNG UND ENERGIE)
# ========================================================================
C1_vals = np.linspace((0.10), (3.0), 40)
C2_vals = np.linspace((0.10), (3.0), 40)
C1_grid, C2_grid = np.meshgrid(C1_vals, C2_vals)
V_C2_tf_grid = np.zeros_like(C1_grid)
E_tf_grid = np.zeros_like(C1_grid)
t_start = time.time()
for i in range(len(C2_vals)):
    for j in range(len(C1_vals)):
        p_val_ij = [(2.0), C1_grid[i, j], C2_grid[i, j]]
        out = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_val_ij)
        v_tf = float(out["xf"][1])
        V_C2_tf_grid[i, j] = v_tf
        E_tf_grid[i, j] = (0.50) * (C2_grid[i, j]) * ((v_tf) ** (2))
t_sweep = (time.time()) - (t_start)
# ========================================================================
# VISUALISIERUNG (2x3 GRID)
# ========================================================================
fig, axs = plt.subplots(
    2,
    3,
    figsize=(
        18,
        10,
    ),
)
ax = axs[0, 0]
ax.plot(t_grid, xf_nom[1, :], "r:", label="Nominal")
ax.plot(t_grid, xf_optV[1, :], "r--", alpha=(0.80), label="Optimum (Spannung)")
ax.plot(t_grid, xf_optE[1, :], "r-", alpha=(0.80), label="Optimum (Energie)")
ax.set_title("Ausgangsspannung V_C2(t)", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Spannung [V]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[0, 1]
ax.plot(t_grid, zf_nom[0, :], "g:", label="Nominal")
ax.plot(t_grid, zf_optV[0, :], "g--", alpha=(0.80), label="Optimum (Spannung)")
ax.plot(t_grid, zf_optE[0, :], "g-", alpha=(0.80), label="Optimum (Energie)")
ax.set_title("Verluststrom durch Diode 1", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Strom [A]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[0, 2]
ax.plot(t_grid, P_loss_nom, "k:", label="Nominal")
ax.plot(t_grid, P_loss_optV, "k--", label="Optimum (Spannung)", alpha=(0.80))
ax.plot(t_grid, P_loss_optE, "k-", label="Optimum (Energie)", alpha=(0.80))
ax.fill_between(t_grid, 0, P_loss_nom, color="gray", alpha=(0.20))
ax.set_title("Verlustleistung (Waerme in Dioden)", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Leistung [W]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[1, 0]
ax.plot(t_grid, J_val[:, 0], "k-", label="dV_C2/dI_in [V/A]")
ax.plot(t_grid, J_val[:, 1], "c-", label="dV_C2/dC1 [V/F]")
ax.plot(t_grid, J_val[:, 2], "y-", label="dV_C2/dC2 [V/F]")
ax.set_title("Nominale Sensitivitaeten", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Sensitivitaet [V/Einheit]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[1, 1]
contour = ax.contourf(C1_grid, C2_grid, V_C2_tf_grid, levels=20, cmap="Blues")
fig.colorbar(contour, ax=ax).set_label("V_C2(t_f) [V]", fontsize=11)
c1_line = np.linspace((0.10), (1.90), 100)
c2_line = (2.0) - (c1_line)
ax.plot(c1_line, c2_line, "r--", lw=(2.0), label="Budget C1+C2=2.0F")
ax.plot([p_nom[1]], [p_nom[2]], "ko", ms=8, markeredgecolor="white", label="Nominal")
ax.plot(
    [p_opt_V[1]],
    [p_opt_V[2]],
    "r*",
    ms=12,
    markeredgecolor="black",
    label="Optimum (Spannung)",
)
ax.set_title("Zielfunktion: Spannung", fontsize=12, fontweight="bold")
ax.set_xlabel("C1 [F]")
ax.set_ylabel("C2 [F]")
ax.legend(fontsize=9)
ax.grid(True, alpha=(0.30))
ax = axs[1, 2]
contour = ax.contourf(C1_grid, C2_grid, E_tf_grid, levels=20, cmap="Oranges")
fig.colorbar(contour, ax=ax).set_label("E(t_f) [J]", fontsize=11)
ax.plot(c1_line, c2_line, "r--", lw=(2.0), label="Budget C1+C2=2.0F")
ax.plot([p_nom[1]], [p_nom[2]], "ko", ms=8, markeredgecolor="white", label="Nominal")
ax.plot(
    [p_opt_E[1]],
    [p_opt_E[2]],
    "r*",
    ms=12,
    markeredgecolor="black",
    label="Optimum (Energie)",
)
ax.set_title("Zielfunktion: Gespeicherte Energie", fontsize=12, fontweight="bold")
ax.set_xlabel("C1 [F]")
ax.set_ylabel("C2 [F]")
ax.legend(fontsize=9)
ax.grid(True, alpha=(0.30))
plt.suptitle(
    "Design-Tradeoffs im Dioden-Geklemmten Filter: Spannung vs. Energie",
    fontsize=16,
    fontweight="bold",
    y=(0.980),
)
plt.tight_layout()
plt.savefig("diode_optima.png", dpi=150)
print("Plot gespeichert: diode_optima.png")
