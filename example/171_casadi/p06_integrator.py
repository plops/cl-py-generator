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
#     steht, um ein bestimmtes Spannungsziel am Ausgang zu erreichen?'
#
#    Diese Frage kann man NICHT analytisch beantworten, weil die
#    Shockley-Diodengleichung (exponentiell) zu einem nichtlinearen DAE
#    fuehrt. Stattdessen nutzen wir hier CasADi, um:
#    (1) Das DAE symbolisch zu formulieren und mit dem IDAS-Solver zu loesen.
#    (2) Automatische Ableitungen (Sensitivitaeten) durch den Solver zu
#        propagieren, um zu verstehen, wie empfindlich die Ausgangsspannung
#        auf Aenderungen von I_in, C1 und C2 reagiert.
#    (3) Mit IPOPT die optimale Kapazitaetsaufteilung zu finden.
#
#    HINWEIS: Wir maximieren hier V_C2(t_f) als Demonstrationsbeispiel
#    fuer DAE-gestuetzte Optimierung. In der Praxis koennte das Ziel
#    auch anders lauten (z.B. Minimierung der Einschwingzeit oder
#    Maximierung der Bandbreite). Die Methodik bleibt identisch.
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
#      I_C1 = C1 * dV_C1/dt   (Kondensatorladestrom)
#      I_R  = (V_C1 - V_C2)/R (Strom durch den Kopplungswiderstand)
#      I_D1 = Is1*(exp(V_C1/Vt1) - 1)  (Shockley-Diodenstrom)
#
#    Stromfluss an Knoten 2 (KCL): I_R = I_C2 + I_D2
#      I_C2 = C2 * dV_C2/dt
#      I_D2 = Is2*(exp(V_C2/Vt2) - 1)
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
# ========================================================================
# INTEGRATOR-ERSTELLUNG (IDAS-PLUGIN)
# ========================================================================
# CasADi bietet zwei wesentliche DAE-Integratoren aus dem SUNDIALS-Paket:
#   'cvodes' - Fuer reine ODEs (ohne algebraische Zustaende z).
#   'idas'   - Fuer DAE-Systeme (mit algebraischen Zustaenden z).
# Da wir hier Diodenstroeme als algebraische Variable haben, MUESSEN
# wir 'idas' verwenden.
#
# Wir erstellen ZWEI Integratoren mit unterschiedlichem Zeithorizont:
#
# F_tf: Integriert nur bis zur Endzeit tf = 1.0 s.
#   Vorteil: Liefert nur den Endwert xf. Ideal als Baustein fuer
#   Optimierung (Opti/IPOPT) und Hessian-Berechnung, weil CasADi
#   keine unnoetig grossen Zwischenergebnisse speichern muss.
#
# F_grid: Integriert auf einem Zeitgitter mit 100 aequidistanten Punkten
#   von t0=0.0 bis tf=1.0 Sekunden.
#   Vorteil: Liefert den Zustandsverlauf x(t) an JEDEM Gitterpunkt.
#   Das brauchen wir fuer:
#   - Plotten der Spannungs-/Stromtrajektorien ueber die Zeit.
#   - Berechnung der Sensitivitaet dV_C2(t)/dp an jedem Zeitpunkt,
#     um zu sehen, WANN im Zeitverlauf welcher Parameter den groessten
#     Einfluss hat (z.B. frueh vs. spaet in der Ladeperiode).
t0 = 0.0
t_grid = np.linspace((0.0), (1.0), 100)
F_tf = ca.integrator("F_tf", "idas", dae, (0.0), (1.0))
F_grid = ca.integrator("F_grid", "idas", dae, t0, t_grid)
# ========================================================================
# SENSITIVITAETSANALYSE: GRADIENT UND HESSIAN
# ========================================================================
# Sensitivitaetsanalyse beantwortet die Frage:
#   'Wie aendert sich die Ausgangsspannung V_C2(t_f), wenn ich einen
#    Parameter (z.B. C1) um einen kleinen Betrag dp veraendere?'
#
# 1. Ordnung (Gradient): dV_C2/dp = [dV_C2/dI_in, dV_C2/dC1, dV_C2/dC2]
#    Gibt die Richtung und Staerke des linearen Einflusses jedes Parameters.
#
# 2. Ordnung (Hessian): d^2 V_C2 / (dp_i * dp_j)
#    Gibt die Kruemmung der Zielfunktion an. Die Hessian-Matrix zeigt:
#    - Diagonale: Wie stark ein Parameter nichtlinear wirkt.
#    - Nebendiagonale: Wie stark zwei Parameter GEKOPPELT sind
#      (d.h. ob die Wirkung von C1 davon abhaengt, welchen Wert C2 hat).
#
# Wir berechnen den Hessian auf zwei Arten, um die Korrektheit zu pruefen:
res_tf = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p)
V_C2_tf = res_tf["xf"][1]
grad_V_C2 = ca.jacobian(V_C2_tf, p).T
# Modus 1: Adjoint-over-Adjoint (AOA) -- ca.hessian()
#   CasADi berechnet zuerst den Gradienten durch Rueckwaerts-AD (adjoint),
#   dann leitet es den Gradienten NOCHMALS rueckwaerts ab.
#   Vorteil: Effizient fuer skalare Zielfunktionen mit vielen Parametern,
#     weil die rueckwaerts-Propagation unabhaengig von der Parameterzahl ist.
#   Nachteil: Erfordert die Loesung eines adjungierten DAE-Systems (IDAS
#     integriert das Originalproblem + ein adjungiertes System rueckwaerts
#     in der Zeit), was numerisch anspruchsvoller sein kann.
H_aoa, g_aoa = ca.hessian(V_C2_tf, p)
f_aoa = ca.Function("f_aoa", [p], [H_aoa, g_aoa])
# Modus 2: Forward-over-Adjoint (FOA) -- jacobian(gradient, p)
#   Zuerst berechnen wir den Gradienten durch Rueckwaerts-AD (adjoint),
#   dann leiten wir den Gradienten-Vektor VORWAERTS ab (jacobian).
#   Vorteil: Numerisch robuster, weil der Vorwaerts-Pass weniger
#     empfindlich auf die Anfangswert-Berechnung des adjungierten Systems ist.
#   Nachteil: Kosten skalieren mit der Parameterzahl (hier 3, also kein Problem).
#
#   Wir vergleichen AOA und FOA, um die numerische Konsistenz zu validieren.
#   Bei einem gut konditionierten Problem sollten beide Matrizen bis auf
#   Rundungsfehler (ca. 1e-7) identisch sein.
H_foa = ca.jacobian(grad_V_C2, p)
f_foa = ca.Function("f_foa", [p], [H_foa])
# ========================================================================
# AUSWERTUNG DER SENSITIVITAETEN AM NOMINALEN ARBEITSPUNKT
# ========================================================================
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
# DAE-BESCHRAENKTE PARAMETEROPTIMIERUNG MIT OPTI/IPOPT
# ========================================================================
# CasADi bietet die 'Opti'-Klasse als komfortable High-Level-Schnittstelle
# zur Formulierung nichtlinearer Optimierungsprobleme (NLP). Sie verwaltet
# automatisch Variablen, Nebenbedingungen und die Solver-Kopplung.
#
# Als NLP-Solver verwenden wir IPOPT (Interior Point OPTimizer):
#   - IPOPT nutzt ein Innere-Punkte-Verfahren (Barrier-Methode) und ist
#     der Standardsolver fuer allgemeine, nichtlineare, beschraenkte
#     Optimierungsprobleme mit glatten Zielfunktionen.
#   - Alternativen waeren z.B. SNOPT (SQP-Methode, besser fuer duenn
#     besetzte, grosse Probleme) oder KNITRO (kommerziell, sehr robust).
#   - IPOPT benoetigt Gradienten und Hessians, die CasADi automatisch
#     via AD bereitstellt -- genau das, was wir oben validiert haben.
#
# Fragestellung: Gegeben ein festes Kapazitaetsbudget (C1+C2 = 2.0 F),
# finde die Aufteilung, die V_C2(t_f=1s) maximiert.
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
t_start = time.time()
sol = opti.solve()
p_opt = sol.value(p_var)
V_C2_tf_max = sol.value(V_C2_tf_opt)
t_opt = (time.time()) - (t_start)
print("--- DAE-beschraenkte Optimierung ---")
print(f"Optimale Kapazitaeten: C1 = {p_opt[1]:.4f} F, C2 = {p_opt[2]:.4f} F")
print(
    f"Max V_C2(t_f): {V_C2_tf_max:.4f} V (Nominal: {float(F_tf(x0=[0, 0], z0=[0, 0], p=p_nom)['xf'][1]):.4f} V)"
)
print(f"Rechenzeit IPOPT: {t_opt * 1000:.1f} ms")
# ========================================================================
# TRAJEKTORIEN-SIMULATION UND ZEITAUFGELOESTE SENSITIVITAETEN
# ========================================================================
# Wir simulieren die Zustandsverlaeufe fuer den nominalen und den
# optimalen Parametersatz ueber das gesamte Zeitgitter (100 Punkte).
t_start = time.time()
sim_nom = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_nom)
sim_opt = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_opt)
t_sim = (time.time()) - (t_start)
print(f"Rechenzeit Trajektorien (2x 100 Gitterpunkte): {t_sim * 1000:.1f} ms")
xf_nom = np.array(sim_nom["xf"])
zf_nom = np.array(sim_nom["zf"])
xf_opt = np.array(sim_opt["xf"])
zf_opt = np.array(sim_opt["zf"])
# Zeitaufgeloeste Sensitivitaeten: dV_C2(t)/dp fuer jeden Zeitpunkt t
#
# Die Sensitivitaet dV_C2(t)/dC1 sagt uns z.B.:
#   'Wenn ich C1 um 1 F erhoehe, um wieviel Volt aendert sich V_C2
#    zum Zeitpunkt t?'
# Wir berechnen das an allen 100 Gitterpunkten, um zu sehen, ob der
# Parametereinfluss frueh (Einschwingphase) oder spaet (stationaerer
# Zustand) dominant ist. Das hilft dem Ingenieur zu verstehen, in
# welcher Betriebsphase eine Bauteiltoleranz am kritischsten ist.
res_grid_sym = F_grid(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p)
V_C2_traj_sym = res_grid_sym["xf"][1, :]
t_start = time.time()
J_traj_sym = ca.jacobian(V_C2_traj_sym, p)
J_func = ca.Function("J_func", [p], [J_traj_sym])
J_val = np.array(J_func(p_nom))
t_jac = (time.time()) - (t_start)
print(
    f"Rechenzeit Sensitivitaets-Trajektorie (100 Punkte, 3 Parameter): {t_jac * 1000:.1f} ms"
)
# 2D Parametersweep: V_C2(tf) fuer ein 40x40-Gitter ueber C1 und C2
# Erzeugt die Daten fuer eine Konturdarstellung (Heatmap) der Zielfunktion.
C1_vals = np.linspace((0.10), (3.0), 40)
C2_vals = np.linspace((0.10), (3.0), 40)
C1_grid, C2_grid = np.meshgrid(C1_vals, C2_vals)
V_C2_tf_grid = np.zeros_like(C1_grid)
t_start = time.time()
for i in range(len(C2_vals)):
    for j in range(len(C1_vals)):
        p_val_ij = [(2.0), C1_grid[i, j], C2_grid[i, j]]
        out = F_tf(x0=[(0.0), (0.0)], z0=[(0.0), (0.0)], p=p_val_ij)
        V_C2_tf_grid[i, j] = float(out["xf"][1])
t_sweep = (time.time()) - (t_start)
print(f"Rechenzeit 2D-Sweep (40x40 = 1600 DAE-Loesungen): {t_sweep * 1000:.0f} ms")
# ========================================================================
# VISUALISIERUNG
# ========================================================================
fig, axs = plt.subplots(
    2,
    2,
    figsize=(
        14,
        10,
    ),
)
ax = axs[0, 0]
ax.plot(t_grid, xf_nom[0, :], "b-", label="V_C1 (Nominal)")
ax.plot(t_grid, xf_nom[1, :], "r-", label="V_C2 (Nominal)")
ax.plot(t_grid, xf_opt[0, :], "b--", alpha=(0.70), label="V_C1 (Optimal)")
ax.plot(t_grid, xf_opt[1, :], "r--", alpha=(0.70), label="V_C2 (Optimal)")
ax.set_title("Kondensatorspannungen [V] vs Zeit", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Spannung [V]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[0, 1]
ax.plot(t_grid, zf_nom[0, :], "g-", label="I_D1 (Nominal)")
ax.plot(t_grid, zf_nom[1, :], "m-", label="I_D2 (Nominal)")
ax.plot(t_grid, zf_opt[0, :], "g--", alpha=(0.70), label="I_D1 (Optimal)")
ax.plot(t_grid, zf_opt[1, :], "m--", alpha=(0.70), label="I_D2 (Optimal)")
ax.set_title("Diodenstroeme [A] vs Zeit", fontsize=12, fontweight="bold")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Strom [A]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[1, 0]
ax.plot(t_grid, J_val[:, 0], "k-", label="dV_C2/dI_in [V/A]")
ax.plot(t_grid, J_val[:, 1], "c-", label="dV_C2/dC1 [V/F]")
ax.plot(t_grid, J_val[:, 2], "y-", label="dV_C2/dC2 [V/F]")
ax.set_title(
    "Sensitivitaet von V_C2(t) bzgl. Parameter", fontsize=12, fontweight="bold"
)
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Sensitivitaet [V/Einheit]")
ax.legend()
ax.grid(True, alpha=(0.50))
ax = axs[1, 1]
contour = ax.contourf(C1_grid, C2_grid, V_C2_tf_grid, levels=20, cmap="viridis")
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("V_C2(t_f) [V]", fontsize=11)
c1_line = np.linspace((0.10), (1.90), 100)
c2_line = (2.0) - (c1_line)
ax.plot(c1_line, c2_line, "r--", lw=(2.0), label="C1 + C2 = 2.0 F")
ax.plot(
    [p_nom[1]], [p_nom[2]], "wo", ms=8, markeredgecolor="black", label="Nominalpunkt"
)
ax.plot([p_opt[1]], [p_opt[2]], "r*", ms=12, markeredgecolor="black", label="Optimum")
ax.set_title("V_C2(t_f) vs Kapazitaeten", fontsize=12, fontweight="bold")
ax.set_xlabel("C1 [F]")
ax.set_ylabel("C2 [F]")
ax.legend(fontsize=8)
ax.grid(True, alpha=(0.30))
plt.suptitle(
    "2-Stufiger Dioden-Geklemmter Kondensatorfilter: Simulation, Sensitivitaeten & Optimierung",
    fontsize=14,
    fontweight="bold",
    y=(0.980),
)
plt.tight_layout()
plt.savefig("diode_sensitivities.png", dpi=150)
print("Plot gespeichert: diode_sensitivities.png")
