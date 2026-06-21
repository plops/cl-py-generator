from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time as time

# ========================================================================
# PHYSIKALISCHE KONSTANTEN UND ORBITALELEMENTE
# ========================================================================
#
# Einheitensystem: Astronomische Einheiten (AU), Jahre (yr), Sonnenmassen.
# In diesen Einheiten gilt mu_Sonne = 4*pi^2 AU^3/yr^2 (aus dem dritten
# Keplerschen Gesetz: T^2 = (4*pi^2 / GM) * a^3, mit T_Erde = 1 yr,
# a_Erde = 1 AU).
#
# Kreisbahnnaehrung: Fuer dieses Demonstrationsbeispiel werden die Bahnen
# von Erde und Mars als koplanare Kreisbahnen approximiert. Die wahren
# Bahnen haben kleine Exzentrizitaeten (e_Erde = 0.017, e_Mars = 0.093)
# und eine relative Inklination von ~1.85 Grad, was die Ergebnisse nur
# leicht veraendert.
MU_SUN = 4.0 * (np.pi**2)
AU_KM = 1.5e8
YR_S = 365.25 * 24 * 3600
V_CONV = AU_KM / YR_S
A_EARTH = 1.0
T_EARTH = 1.0
A_MARS = 1.524
T_MARS = 1.881


# ========================================================================
# HILFSFUNKTION: PLANETENPOSITION AUF KREISBAHN
# ========================================================================
def planet_state(t_yr: float, a: float, period: float) -> tuple:
    """Berechnet Position und Geschwindigkeit eines Planeten auf einer Kreisbahn.

    Die Kreisbahngeschwindigkeit ist v = 2*pi*a/T (Umfang / Periode).
    Die Winkelgeschwindigkeit ist omega = 2*pi/T.
    Zum Zeitpunkt t steht der Planet beim Winkel theta = omega * t.

    Parameter:
        t_yr: Zeit in Jahren (Epoche relativ zum Referenzzeitpunkt)
        a: Halbachse der Kreisbahn in AU
        period: Umlaufperiode in Jahren

    Rueckgabe:
        (x, y, vx, vy) in AU und AU/yr"""
    omega = (2 * np.pi) / period
    theta = omega * t_yr
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    v = (2 * np.pi * a) / period
    vx = -1 * v * np.sin(theta)
    vy = v * np.cos(theta)
    return (
        x,
        y,
        vx,
        vy,
    )


# ========================================================================
# CASADI-INTEGRATOR: KEPLERSCHE ZWEIKÖRPERDYNAMIK
# ========================================================================
#
# Die Bewegungsgleichungen im heliozentrisch-kartesischen System lauten:
#   d^2 r / dt^2 = -mu * r / |r|^3
#
# Als System erster Ordnung mit Zustand s = [x, y, vx, vy]:
#   dx/dt  = vx
#   dy/dt  = vy
#   dvx/dt = -mu * x / r^3
#   dvy/dt = -mu * y / r^3
#
# Zeitskalierungstrick: Um die Transferzeit als Parameter (nicht als feste
# Integrationsgrenze) behandeln zu koennen, substituieren wir tau = t/TOF.
# Dann integrieren wir immer von tau=0 bis tau=1, und die ODE wird mit dem
# Faktor TOF skaliert: ds/d(tau) = TOF * f(s).
# Dies ermoeglicht es, einen einzigen Integrator fuer alle Transferzeiten
# wiederzuverwenden.
def create_integrator() -> object:
    """Erzeugt einen CasADi-CVODES-Integrator fuer die Keplersche Bahnmechanik.

    Der Integrator propagiert den Zustandsvektor [x, y, vx, vy] von tau=0 bis tau=1,
    wobei die physikalische Transferzeit TOF als Parameter uebergeben wird.

    CVODES (aus dem SUNDIALS-Paket) verwendet ein implizites BDF-Verfahren
    (Backward Differentiation Formula) mit adaptiver Schrittweitensteuerung.
    Es liefert automatisch exakte Ableitungen der Loesung nach den Anfangswerten
    und Parametern (Sensitivitaetsanalyse), die CasADi fuer die NLP-Gradienten nutzt."""
    x = SX.sym("x")
    y = SX.sym("y")
    vx = SX.sym("vx")
    vy = SX.sym("vy")
    state = vertcat(x, y, vx, vy)
    tof = SX.sym("tof")
    r = sqrt((x**2) + (y**2))
    ode = tof * vertcat(vx, vy, (-1 * MU_SUN * x) / (r**3), (-1 * MU_SUN * y) / (r**3))
    dae = {("x"): (state), ("p"): (tof), ("ode"): (ode)}
    F = integrator(
        "F",
        "cvodes",
        dae,
        {("t0"): (0), ("tf"): (1), ("reltol"): (1.0e-10), ("abstol"): (1.0e-12)},
    )
    return F


# ========================================================================
# LAMBERT-LOESER: NLP-FORMULIERUNG MIT CASADI OPTI STACK
# ========================================================================
#
# Fuer jedes Paar (Abflugdatum, Transferdauer) wird ein kleines NLP geloest:
#   Entscheidungsvariablen: v0 = [vx0, vy0] (Anfangsgeschwindigkeit)
#   Nebenbedingungen: Endposition = Marsposition (2 Gleichungen)
#   Zielfunktion: min Delta-v_total = |v0 - v_Erde| + |v_f - v_Mars|
#
# Da die 2 Nebenbedingungen die 2 Variablen eindeutig festlegen (Lambert-
# Theorem), ist dies effektiv ein Gleichungssystem. Die Zielfunktion dient
# als Regularisierung und berechnet gleichzeitig den Treibstoffbedarf.
def solve_lambert_single(args) -> float:
    """Loest das Lambert-Problem fuer ein einzelnes (t_dep, tof)-Paar.

    Diese Funktion wird von multiprocessing.Pool.map aufgerufen und muss daher
    den Integrator selbst erzeugen (CasADi-Objekte koennen nicht zwischen
    Prozessen serialisiert/gepickelt werden).

    args: Tuple (t_dep, tof) mit:
        t_dep: Abflugzeitpunkt in Jahren
        tof:   Transferzeit in Jahren

    Rueckgabe:
        Delta-v total in km/s, oder np.nan falls die Optimierung fehlschlaegt."""
    t_dep, tof = args
    F = create_integrator()
    x_E, y_E, vx_E, vy_E = planet_state(t_dep, A_EARTH, T_EARTH)
    x_M, y_M, vx_M, vy_M = planet_state(t_dep + tof, A_MARS, T_MARS)
    dx = x_M - x_E
    dy = y_M - y_E
    dist = np.sqrt((dx**2) + (dy**2))
    if dist < 1.0e-2:
        return np.nan
    opti = casadi.Opti()
    v0 = opti.variable(2)
    x0 = vertcat(x_E, y_E, v0[0], v0[1])
    result = F(x0=x0, p=tof)
    xf = result["xf"]
    opti.subject_to(xf[0] == x_M)
    opti.subject_to(xf[1] == y_M)
    dv_dep = sqrt((((v0[0]) - vx_E) ** 2) + (((v0[1]) - vy_E) ** 2))
    dv_arr = sqrt((((xf[2]) - vx_M) ** 2) + (((xf[3]) - vy_M) ** 2))
    opti.minimize(dv_dep + dv_arr)
    opti.set_initial(v0, vertcat(vx_E, vy_E))
    angle_to_mars = np.arctan2(dy, dx)
    v_earth_mag = (2 * np.pi * A_EARTH) / T_EARTH
    guess_vx = 1.1 * v_earth_mag * np.cos(angle_to_mars + (np.pi / 6))
    guess_vy = 1.1 * v_earth_mag * np.sin(angle_to_mars + (np.pi / 6))
    opti.set_initial(v0, vertcat(guess_vx, guess_vy))
    opti.solver(
        "ipopt",
        {("print_time"): (False)},
        {("print_level"): (0), ("max_iter"): (1000), ("tol"): (1.0e-6)},
    )
    try:
        sol = opti.solve()
        total_dv = float(sol.value(dv_dep + dv_arr))
        if total_dv < 5.0e1 * (YR_S / AU_KM):
            return total_dv * V_CONV
        else:
            return np.nan
    except Exception:
        return np.nan


# ========================================================================
# HAUPTPROGRAMM: PARALLELISIERTE GITTERBERECHNUNG UND VISUALISIERUNG
# ========================================================================
if __name__ == "__main__":
    # Gitterparameter:
    # - Abflugdaten: 0 bis 2.5 Jahre (deckt mehr als eine synodische Periode
    #   Erde-Mars von 2.135 Jahren ab, um mindestens ein Startfenster zu zeigen)
    # - Transferdauer: 100 bis 450 Tage (Hohmann-Transfer: ~259 Tage)
    n_dep = 50
    n_tof = 50
    t_dep_arr = np.linspace(0.0, 2.5, n_dep)
    tof_days_arr = np.linspace(1.0e2, 4.5e2, n_tof)
    tof_yr_arr = tof_days_arr / 365.25
    args_list = []
    for t_dep in t_dep_arr:
        for tof in tof_yr_arr:
            args_list.append(
                (
                    t_dep,
                    tof,
                )
            )
    n_cpus = mp.cpu_count()
    n_total = n_dep * n_tof
    print(
        f"Starte Gitterberechnung: {n_dep}x{n_tof} = {n_total} Punkte auf {n_cpus} Kernen"
    )
    t_start = time.time()
    with mp.Pool(processes=n_cpus) as pool:
        results = pool.map(solve_lambert_single, args_list)
    elapsed = time.time() - t_start
    print(f"Berechnung abgeschlossen in {elapsed:.1f} Sekunden")
    dv_grid = np.array(results).reshape(n_dep, n_tof)
    # ========================================================================
    # VISUALISIERUNG: PORK-CHOP-DIAGRAMM
    # ========================================================================
    #
    # Das Diagramm zeigt Konturen des gesamten Delta-v in km/s als Funktion
    # von Abflugdatum (x-Achse) und Transferdauer (y-Achse).
    #
    # Interpretation:
    # - Die geschlossenen Konturen niedriger Delta-v-Werte zeigen die
    #   optimalen Startfenster (typischerweise nahe dem Hohmann-Transfer
    #   von ~259 Tagen bei ~5.6 km/s Gesamt-Delta-v).
    # - Die bananenartige Form ('Pork Chop') entsteht durch die relative
    #   Geometrie der Erde- und Marspositionen: Guenstige Konstellationen
    #   wiederholen sich mit der synodischen Periode von ~780 Tagen.
    # - Rote Regionen zeigen energetisch teure Transfers (kurze Transferzeit
    #   bei unguentstiger Geometrie).
    # - Der weisse Stern markiert das globale Minimum.
    T_DEP, TOF_DAYS = np.meshgrid(t_dep_arr, tof_days_arr, indexing="ij")
    valid_mask = np.isfinite(dv_grid)
    if np.any(valid_mask):
        min_idx = np.nanargmin(dv_grid)
        i_min, j_min = np.unravel_index(min_idx, dv_grid.shape)
        dv_min = dv_grid[i_min, j_min]
        t_dep_min = t_dep_arr[i_min]
        tof_min = tof_days_arr[j_min]
        print(f"Globales Minimum: Delta-v = {dv_min:.2f} km/s")
        print(f"  Abflug: t = {t_dep_min:.3f} yr, Transferdauer: {tof_min:.0f} Tage")
    fig_and_ax = plt.subplots(
        1,
        1,
        figsize=(
            12,
            9,
        ),
    )
    fig, ax = fig_and_ax
    vmin = np.nanmin(dv_grid)
    vmax = np.clip(np.nanpercentile(dv_grid, 85), 0, 30)
    levels = np.linspace(vmin, vmax, 25)
    cf = ax.contourf(
        T_DEP, TOF_DAYS, dv_grid, levels=levels, cmap="RdYlGn_r", extend="max"
    )
    cs = ax.contour(
        T_DEP,
        TOF_DAYS,
        dv_grid,
        levels=levels,
        colors="black",
        linewidths=0.3,
        alpha=0.5,
    )
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Total $\Delta v$ [km/s]", fontsize=13)
    if np.any(valid_mask):
        ax.plot(
            t_dep_min,
            tof_min,
            marker="*",
            color="white",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=10,
        )
        ax.annotate(
            f"$\Delta v_{{min}}$ = {dv_min:.2f} km/s",
            xy=(
                t_dep_min,
                tof_min,
            ),
            xytext=(
                t_dep_min + 0.15,
                tof_min + 25,
            ),
            fontsize=11,
            fontweight="bold",
            color="white",
            arrowprops={("arrowstyle"): ("->"), ("color"): ("white"), ("lw"): (1.5)},
        )
    ax.set_xlabel("Departure date [years from epoch]", fontsize=13)
    ax.set_ylabel("Transfer duration [days]", fontsize=13)
    ax.set_title(
        "Earth $\rightarrow$ Mars: Pork Chop Diagram (Impulsive $\Delta v$)",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, color="white", linestyle="--")
    a_hohmann = (A_EARTH + A_MARS) / 2.0
    tof_hohmann_yr = np.pi * np.sqrt((a_hohmann**3) / (MU_SUN / (4 * (np.pi**2))))
    tof_hohmann_days = tof_hohmann_yr * 365.25
    ax.axhline(
        y=tof_hohmann_days,
        color="cyan",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Hohmann ({tof_hohmann_days:.0f} d)",
    )
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig("porkchop_earth_mars.png", dpi=150)
    print("Plot gespeichert: porkchop_earth_mars.png")
    plt.show()
