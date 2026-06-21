from __future__ import annotations
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time as time

# ========================================================================
# 2-STUFIGER DIODEN-GEKLEMMTER KONDENSATORFILTER - MULTI-ZIEL OPTIMIERUNG
# ========================================================================
#
# PHYSIKALISCHES MODELL:
#    Zweistufiger passiver Tiefpassfilter mit Klemm-Dioden parallel
#    zu den Kondensatoren. Gespeist von einer Konstantstromquelle I_in.
#
#    Schaltplan:
#                    Knoten 1           Knoten 2
#    I_in              V_C1               V_C2
#     o-------+-----------o------[ R ]-------o-----------o (Ausgang)
#             |           |                  |
#             |        ---+---            ---+---
#           [I_in]     | C1  |            | C2  |
#             |        ---+---            ---+---
#             |         D1|                D2|
#             |          / |               /  |
#             |         v  |              v   |
#    ---------+-----------+------------------+-- GND
#
# DESIGNFRAGE:
#    Gegeben ein Kapazitaetsbudget C1+C2=2F und fester I_in=2A:
#    Wie ist C1,C2 aufzuteilen, um ein Entwurfsziel zu erreichen?
#
# DREI ENTWURFSZIELE IM VERGLEICH:
#    1. Spannungs-Optimum:  Maximiere V_C2(t_f)
#       -> Optimum liegt am Rand (C2 minimal). Kein Sweet Spot.
#       -> Anwendung: Digitale Logikgatter, hochohmige Eingaenge.
#
#    2. Energie-Optimum:   Maximiere E = 0.5*C2*V_C2(t_f)^2
#       -> Optimum liegt am gegenueberligenden Rand (C2 maximal).
#       -> Anwendung: Ideale Energiespeicherung ohne Nebenbedingungen.
#       -> Problem: Diode 1 leitet massiv (2W Verlust), zerstoert
#          in der Praxis das Bauteil.
#
#    3. Netto-Energie-Optimum (NEU): Maximiere E_netto = E - lambda * E_verlust
#       -> Hier: lambda = 0.060 (Gewichtungsfaktor fuer Thermoverluste)
#       -> E_verlust = Integral der Verlustleistung P_D = V_D * I_D
#       -> Physikalische Bedeutung von lambda:
#          lambda = Wirkungsgrad-Koeffizient. Er beschreibt, wie teuer
#          ein Joule Verlust im Vergleich zu einem Joule gespeicherter
#          Energie ist. Bei lambda=0.060 werden 60 mJ Verlust mit 1 mJ
#          nutzbarer Energie aufgewogen.
#       -> DAS ERGEBNIS LIEGT IM INNEREN DES PARAMETERRAUMS!
#          Das ist ein klassischer 'Sweet Spot': Ein zu kleines C2
#          speichert kaum Energie, ein zu grosses C2 verbrennt zu viel
#          Leistung. Der Optimierer findet automatisch den Kompromiss.
#       -> Anwendung: IoT-Energieernte, batterielos betriebene Sensoren,
#          bei denen Bauteiltemperatur und Effizienz gleichzeitig
#          beschraenkt werden muessen.
#
# TECHNISCHER SCHLUESSEL: CasADi QUADRATURZUSTAND
#    Um E_verlust = Integral(V_C1*I_D1 + V_C2*I_D2, dt) zu berechnen,
#    verwenden wir das 'quad'-Feld im DAE-Dictionary. Dadurch integriert
#    IDAS automatisch einen Quadraturzustand q(t) mit:
#       dq/dt = V_C1*I_D1 + V_C2*I_D2  (Verlustleistung)
#    Das Ergebnis F_tf(...)[qf] ist E_verlust am Integrations-Ende.
#    Gradienten durch diesen Quadraturzustand sind vollautomatisch
#    via adjoint-Sensitivitaeten verfuegbar - keine Extraarbeit!
# ========================================================================
# ========================================================================
# SYMBOLISCHE DAE-DEFINITION MIT QUADRATURZUSTAND
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
P_loss_sym = ((V_C1) * (I_D1)) + ((V_C2) * (I_D2))
dae = {
    ("x"): (x),
    ("z"): (z),
    ("p"): (p),
    ("ode"): (ode),
    ("alg"): (alg),
    ("quad"): (P_loss_sym),
}
t0 = 0.0
t_grid = np.linspace((0.0), (1.0), 100)
F_tf = ca.integrator("F_tf", "idas", dae, (0.0), (1.0))
F_grid = ca.integrator("F_grid", "idas", dae, t0, t_grid)
# ========================================================================
# 2D PARAMETERSWEEP (60x60) FUER ALLE DREI ZIELFUNKTIONEN
# ========================================================================
C1_vals = np.linspace((0.10), (2.80), 60)
C2_vals = np.linspace((0.10), (2.80), 60)
C1_grid, C2_grid = np.meshgrid(C1_vals, C2_vals)
V_C2_tf_grid = np.zeros_like(C1_grid)
E_stored_grid = np.zeros_like(C1_grid)
E_loss_grid = np.zeros_like(C1_grid)
E_netto_grid = np.zeros_like(C1_grid)
lam_penalty = 6.00e-2
t_start = time.time()
for i in range(len(C2_vals)):
    for j in range(len(C1_vals)):
        p_ij = [(2.0), C1_grid[i, j], C2_grid[i, j]]
        out = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_ij)
        v_tf = float(out["xf"][1])
        e_loss = float(out["qf"])
        e_stored = (0.50) * (C2_grid[i, j]) * ((v_tf) ** (2))
        V_C2_tf_grid[i, j] = v_tf
        E_stored_grid[i, j] = e_stored
        E_loss_grid[i, j] = e_loss
        E_netto_grid[i, j] = (e_stored) - ((lam_penalty) * (e_loss))
t_sweep = (time.time()) - (t_start)
print(f"Rechenzeit 2D-Sweep (60x60): {t_sweep:.1f} s")
# ========================================================================
# OPTIMIERUNG 1: SPANNUNGS-MAXIMIERUNG
# ========================================================================
opti_V = ca.Opti()
p_var_V = opti_V.variable(3)
opti_V.subject_to(((p_var_V[0]) == (2.0)))
opti_V.subject_to((((p_var_V[1]) + (p_var_V[2])) == (2.0)))
opti_V.subject_to(((p_var_V[1]) >= (0.10)))
opti_V.subject_to(((p_var_V[2]) >= (0.10)))
opti_V.set_initial(p_var_V, [(2.0), (1.0), (1.0)])
res_V = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_var_V)
V_C2_V = res_V["xf"][1]
opti_V.minimize(((-1.0) * (V_C2_V)))
opti_V.solver("ipopt", {}, {("print_level"): (0)})
sol_V = opti_V.solve()
p_opt_V = sol_V.value(p_var_V)
print(
    f"Optimum Spannung: C1={p_opt_V[1]:.3f} F, C2={p_opt_V[2]:.3f} F, V_C2={-sol_V.value(opti_V.f):.4f} V"
)
# ========================================================================
# OPTIMIERUNG 2: ENERGIE-MAXIMIERUNG
# ========================================================================
opti_E = ca.Opti()
p_var_E = opti_E.variable(3)
opti_E.subject_to(((p_var_E[0]) == (2.0)))
opti_E.subject_to((((p_var_E[1]) + (p_var_E[2])) == (2.0)))
opti_E.subject_to(((p_var_E[1]) >= (0.10)))
opti_E.subject_to(((p_var_E[2]) >= (0.10)))
opti_E.set_initial(p_var_E, [(2.0), (1.0), (1.0)])
res_E = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_var_E)
V_C2_E = res_E["xf"][1]
E_obj = (0.50) * (p_var_E[2]) * ((V_C2_E) ** (2))
opti_E.minimize(((-1.0) * (E_obj)))
opti_E.solver("ipopt", {}, {("print_level"): (0)})
sol_E = opti_E.solve()
p_opt_E = sol_E.value(p_var_E)
print(
    f"Optimum Energie: C1={p_opt_E[1]:.3f} F, C2={p_opt_E[2]:.3f} F, E={-sol_E.value(opti_E.f):.5f} J"
)
# ========================================================================
# OPTIMIERUNG 3: NETTO-ENERGIE (SWEET SPOT IM INNEREN!)
# ========================================================================
# Zielfunktion: E_netto = E_gespeichert - lambda * E_verlust
#   E_gespeichert = 0.5 * C2 * V_C2(tf)^2
#   E_verlust = Integral(V_C1*I_D1 + V_C2*I_D2, dt)  [qf-Ausgang von F_tf]
#   lambda = 0.060: Abwaegung zwischen Energie-Ernte und Thermoverlust
# Da grosse C2 mehr Energie speichern, aber D1 massiv leitend machen,
# und kleine C2 kaum Energie speichern, muss das Optimum irgendwo
# dazwischen liegen -- nicht am Rand!
opti_N = ca.Opti()
p_var_N = opti_N.variable(3)
opti_N.subject_to(((p_var_N[0]) == (2.0)))
opti_N.subject_to((((p_var_N[1]) + (p_var_N[2])) == (2.0)))
opti_N.subject_to(((p_var_N[1]) >= (0.10)))
opti_N.subject_to(((p_var_N[2]) >= (0.10)))
opti_N.set_initial(p_var_N, [(2.0), (1.0), (1.0)])
res_N = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_var_N)
V_C2_N = res_N["xf"][1]
E_loss_N = res_N["qf"]
E_stored_N = (0.50) * (p_var_N[2]) * ((V_C2_N) ** (2))
E_netto_N = (E_stored_N) - ((lam_penalty) * (E_loss_N))
opti_N.minimize(((-1.0) * (E_netto_N)))
opti_N.solver("ipopt", {}, {("print_level"): (0)})
sol_N = opti_N.solve()
p_opt_N = sol_N.value(p_var_N)
print(
    f"Optimum Netto-Energie: C1={p_opt_N[1]:.3f} F, C2={p_opt_N[2]:.3f} F, E_netto={-sol_N.value(opti_N.f):.5f} J"
)
is_interior = (
    ((p_opt_N[1]) > (0.110))
    and ((p_opt_N[2]) > (0.110))
    and ((p_opt_N[1]) < (1.890))
    and ((p_opt_N[2]) < (1.890))
)
if is_interior:
    print("VERIFIKATION: Optimum liegt IM INNEREN des Parameterraums [OK]")
else:
    print("VERIFIKATION: Optimum liegt am Rand [NICHT OK - lambda anpassen!]")
# ========================================================================
# TRAJEKTORIEN FUER ALLE DREI OPTIMA
# ========================================================================
sim_nom = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=[(2.0), (1.0), (1.0)])
sim_optV = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt_V)
sim_optE = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt_E)
sim_optN = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt_N)
xf_nom = np.array(sim_nom["xf"])
zf_nom = np.array(sim_nom["zf"])
xf_optV = np.array(sim_optV["xf"])
zf_optV = np.array(sim_optV["zf"])
xf_optE = np.array(sim_optE["xf"])
zf_optE = np.array(sim_optE["zf"])
xf_optN = np.array(sim_optN["xf"])
zf_optN = np.array(sim_optN["zf"])
P_nom = ((xf_nom[0, :]) * (zf_nom[0, :])) + ((xf_nom[1, :]) * (zf_nom[1, :]))
P_optV = ((xf_optV[0, :]) * (zf_optV[0, :])) + ((xf_optV[1, :]) * (zf_optV[1, :]))
P_optE = ((xf_optE[0, :]) * (zf_optE[0, :])) + ((xf_optE[1, :]) * (zf_optE[1, :]))
P_optN = ((xf_optN[0, :]) * (zf_optN[0, :])) + ((xf_optN[1, :]) * (zf_optN[1, :]))
# ========================================================================
# VISUALISIERUNG: 2x3 PLOT-GRID
# ========================================================================
fig, axs = plt.subplots(
    2,
    3,
    figsize=(
        18,
        11,
    ),
)
ax = axs[0, 0]
ax.plot(t_grid, xf_nom[1, :], "r:", alpha=(0.60), label="Nominal (C1=1, C2=1)")
ax.plot(t_grid, xf_optV[1, :], "b-", alpha=(0.90), label="Opt. Spannung")
ax.plot(t_grid, xf_optE[1, :], "g-", alpha=(0.90), label="Opt. Energie")
ax.plot(
    t_grid, xf_optN[1, :], "m-", alpha=(1.0), label="Opt. Netto-Energie (Sweet Spot)"
)
ax.set_title("Ausgangsspannung V_C2(t)", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Spannung [V]")
ax.legend(fontsize=9)
ax.grid(True, alpha=(0.40))
ax = axs[0, 1]
ax.plot(t_grid, zf_nom[0, :], "r:", alpha=(0.60), label="Nominal")
ax.plot(t_grid, zf_optV[0, :], "b-", alpha=(0.90), label="Opt. Spannung")
ax.plot(t_grid, zf_optE[0, :], "g-", alpha=(0.90), label="Opt. Energie")
ax.plot(t_grid, zf_optN[0, :], "m-", alpha=(1.0), label="Opt. Netto-Energie")
ax.set_title("Verluststrom I_D1 (Diode 1)", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Strom [A]")
ax.legend(fontsize=9)
ax.grid(True, alpha=(0.40))
ax = axs[0, 2]
ax.plot(t_grid, P_nom, "r:", alpha=(0.60), label="Nominal")
ax.plot(t_grid, P_optV, "b-", alpha=(0.90), label="Opt. Spannung")
ax.plot(t_grid, P_optE, "g-", alpha=(0.90), label="Opt. Energie")
ax.plot(t_grid, P_optN, "m-", alpha=(1.0), label="Opt. Netto-Energie")
ax.fill_between(t_grid, 0, P_optN, color="mediumpurple", alpha=(0.20))
ax.set_title("Verlustleistung P_D = V_D * I_D", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Leistung [W]")
ax.legend(fontsize=9)
ax.grid(True, alpha=(0.40))
ax = axs[1, 0]
contV = ax.contourf(C1_grid, C2_grid, V_C2_tf_grid, levels=25, cmap="Blues")
fig.colorbar(contV, ax=ax).set_label("V_C2(t_f) [V]")
budget_c1 = np.linspace((0.10), (1.90), 100)
ax.plot(budget_c1, ((2.0) - (budget_c1)), "w--", lw=(2.0), label="Budget C1+C2=2F")
ax.plot(
    [p_opt_V[1]],
    [p_opt_V[2]],
    "r*",
    ms=14,
    markeredgecolor="white",
    label="Opt. Spannung",
)
ax.set_title("Zielfkt. 1: Spannung", fontsize=12, fontweight="bold")
ax.set_xlabel("C1 [F]")
ax.set_ylabel("C2 [F]")
ax.legend(fontsize=9)
ax = axs[1, 1]
contN = ax.contourf(C1_grid, C2_grid, E_netto_grid, levels=25, cmap="RdYlGn")
fig.colorbar(contN, ax=ax).set_label("E_netto [J]")
ax.plot(budget_c1, ((2.0) - (budget_c1)), "w--", lw=(2.0), label="Budget C1+C2=2F")
ax.plot(
    [p_opt_N[1]],
    [p_opt_N[2]],
    "r*",
    ms=16,
    markeredgecolor="black",
    label="Opt. Netto-Energie [SWEET SPOT]",
)
ax.plot(
    [p_opt_V[1]],
    [p_opt_V[2]],
    "b^",
    ms=10,
    markeredgecolor="white",
    label="Opt. Spannung (Ref.)",
)
ax.plot(
    [p_opt_E[1]],
    [p_opt_E[2]],
    "gv",
    ms=10,
    markeredgecolor="white",
    label="Opt. Energie (Ref.)",
)
ax.set_title("Zielfkt. 3: E_netto = E - 0.060*E_loss", fontsize=12, fontweight="bold")
ax.set_xlabel("C1 [F]")
ax.set_ylabel("C2 [F]")
ax.legend(fontsize=8)
ax = axs[1, 2]
c1_scan = np.linspace((0.10), (1.90), 80)
c2_scan = (2.0) - (c1_scan)
e_netto_scan = np.zeros_like(c1_scan)
e_stored_scan = np.zeros_like(c1_scan)
e_loss_scan = np.zeros_like(c1_scan)
for k in range(len(c1_scan)):
    out_k = F_tf(
        x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=[(2.0), c1_scan[k], c2_scan[k]]
    )
    v_k = float(out_k["xf"][1])
    el_k = float(out_k["qf"])
    es_k = (0.50) * (c2_scan[k]) * ((v_k) ** (2))
    e_netto_scan[k] = (es_k) - ((lam_penalty) * (el_k))
    e_stored_scan[k] = es_k
    e_loss_scan[k] = (lam_penalty) * (el_k)
ax.plot(c1_scan, e_stored_scan, "g-", label="E_gespeichert")
ax.plot(c1_scan, ((-1.0) * (e_loss_scan)), "r-", label="-lambda*E_verlust")
ax.plot(c1_scan, e_netto_scan, "m-", lw=(2.50), label="E_netto (Summe)")
ax.axvline(p_opt_N[1], color="black", ls="--", lw=(1.50), label="Optimum")
ax.set_title("E_netto auf Budget-Linie C1+C2=2F", fontsize=12, fontweight="bold")
ax.set_xlabel("C1 [F]   (C2 = 2 - C1)")
ax.set_ylabel("Energie [J]")
ax.legend(fontsize=9)
ax.grid(True, alpha=(0.40))
plt.suptitle(
    "Sweet Spot im Inneren: Netto-Energie-Optimierung unter Thermoverlust-Penalty",
    fontsize=15,
    fontweight="bold",
    y=(0.990),
)
plt.tight_layout()
plt.savefig("diode_sweetspot.png", dpi=150)
print("Plot gespeichert: diode_sweetspot.png")
