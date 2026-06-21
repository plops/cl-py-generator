import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import p10_porkchop_cpp as p10

# Global plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee'
})

print("Lese CSV-Daten...")
t_dep_list = []
tof_days_list = []
dv_tot_list = []

with open("porkchop_cpp.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Header überspringen
    for row in reader:
        if row:
            t_dep_list.append(float(row[0]))
            tof_days_list.append(float(row[1]))
            dv_tot_list.append(float(row[2]))

t_dep = np.array(t_dep_list)
tof_days = np.array(tof_days_list)
dv_tot = np.array(dv_tot_list)

X = np.unique(t_dep)
Y = np.unique(tof_days)

Z = np.full((len(Y), len(X)), np.nan)
x_map = {val: idx for idx, val in enumerate(X)}
y_map = {val: idx for idx, val in enumerate(Y)}

for t, tof, dv in zip(t_dep, tof_days, dv_tot):
    ix = x_map[t]
    iy = y_map[tof]
    Z[iy, ix] = dv

start_epoch = datetime.datetime(2026, 1, 1)

# ========================================================================
# 1. ERMITTLUNG ALLER 5 LOKALEN MINIMA
# ========================================================================
intervals = [
    (0.0, 1.0),
    (1.5, 2.8),
    (3.5, 4.8),
    (5.5, 6.8),
    (7.5, 9.5)
]

minima = []
print("Suche lokale Minima in den 5 synodischen Fenstern...")
for idx, (t_min, t_max) in enumerate(intervals):
    mask = (t_dep >= t_min) & (t_dep <= t_max)
    if np.any(mask):
        sub_idx = np.argmin(dv_tot[mask])
        sub_t = t_dep[mask][sub_idx]
        sub_tof = tof_days[mask][sub_idx]
        sub_dv = dv_tot[mask][sub_idx]
        sub_date = start_epoch + datetime.timedelta(days=float(sub_t) * 365.25)
        
        minima.append({
            'idx': idx + 1,
            't_dep': sub_t,
            'tof': sub_tof,
            'dv': sub_dv,
            'date': sub_date
        })
        print(f"  Fenster {idx+1}: {sub_date.strftime('%d.%m.%Y')} | Delta-v = {sub_dv:.2f} km/s | Flugzeit = {sub_tof:.1f} Tage")

# Finde das globale Minimum (für Trajektorienplot)
glob_min = min(minima, key=lambda x: x['dv'])

# ========================================================================
# 2. FLUGBAHN-INTEGRATION FÜR DAS GLOBALE MINIMUM (2034)
# ========================================================================
def get_spacecraft_trajectory(t_dep, tof):
    # Newton-Löser in Python zur Ermittlung von vx0, vy0
    p_val = [t_dep, tof / 365.25]
    g_res, J_res, s_end, v_E, v_M = p10.lambert_eval([0.0, 0.0], p_val)
    v_E = np.array(v_E).flatten()
    
    # Newton Loop
    z_sol = 1.1 * v_E
    for _ in range(15):
        g, J, s_end, v_E, v_M = p10.lambert_eval(z_sol, p_val)
        g = np.array(g).flatten()
        J = np.array(J).reshape((2,2))
        det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        if abs(det) < 1e-12:
            break
        dz0 = (J[1,1]*g[0] - J[0,1]*g[1]) / det
        dz1 = (J[0,0]*g[1] - J[1,0]*g[0]) / det
        z_sol[0] -= dz0; z_sol[1] -= dz1
        if np.sqrt(dz0**2 + dz1**2) < 1e-6:
            break

    # Trajektorienintegration
    x_E, y_E, vx_E, vy_E = p10.planet_state(t_dep, p10.A_EARTH, p10.T_EARTH)
    s_curr = np.array([float(x_E), float(y_E), z_sol[0], z_sol[1]])
    
    N_steps = 150
    h = (tof / 365.25) / N_steps
    x_traj, y_traj = [s_curr[0]], [s_curr[1]]
    
    for _ in range(N_steps):
        r = np.sqrt(s_curr[0]**2 + s_curr[1]**2)
        ax = -p10.MU_SUN * s_curr[0] / r**3
        ay = -p10.MU_SUN * s_curr[1] / r**3
        k1 = np.array([s_curr[2], s_curr[3], ax, ay])
        
        s2 = s_curr + 0.5 * h * k1
        r2 = np.sqrt(s2[0]**2 + s2[1]**2)
        ax2 = -p10.MU_SUN * s2[0] / r2**3
        ay2 = -p10.MU_SUN * s2[1] / r2**3
        k2 = np.array([s2[2], s2[3], ax2, ay2])
        
        s3 = s_curr + 0.5 * h * k2
        r3 = np.sqrt(s3[0]**2 + s3[1]**2)
        ax3 = -p10.MU_SUN * s3[0] / r3**3
        ay3 = -p10.MU_SUN * s3[1] / r3**3
        k3 = np.array([s3[2], s3[3], ax3, ay3])
        
        s4 = s_curr + h * k3
        r4 = np.sqrt(s4[0]**2 + s4[1]**2)
        ax4 = -p10.MU_SUN * s4[0] / r4**3
        ay4 = -p10.MU_SUN * s4[1] / r4**3
        k4 = np.array([s4[2], s4[3], ax4, ay4])
        
        s_curr = s_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_traj.append(s_curr[0])
        y_traj.append(s_curr[1])
        
    return np.array(x_traj), np.array(y_traj), float(x_E), float(y_E), float(s_end[0]), float(s_end[1])

# Berechne Bahndaten
x_t, y_t, xe, ye, xm, ym = get_spacecraft_trajectory(glob_min['t_dep'], glob_min['tof'])

# Berechne Strahlungsbelastung (Sonnensturm-Expositions-Index)
# Integral(1/r^2 dt) in Erden-Jahren.
r_t = np.sqrt(x_t**2 + y_t**2)
dt_yr = (glob_min['tof'] / 365.25) / 150
radiation_index = np.sum(1.0 / (r_t**2)) * dt_yr

# ========================================================================
# 3. 2-PANEL-PLOT ERSTELLEN
# ========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# ----------------- Subplot 1: Porkchop Plot mit allen Minima -----------------
X_dates = [start_epoch + datetime.timedelta(days=float(t) * 365.25) for t in X]
im = ax1.pcolormesh(X_dates, Y, Z, cmap="RdYlGn_r", shading="auto", vmin=5.5, vmax=25.0)
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Gesamt-$\\Delta v$ [km/s]", fontsize=12)

levels = [6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
contour = ax1.contour(X_dates, Y, Z, levels=levels, colors="black", linewidths=0.5, alpha=0.7)
ax1.clabel(contour, inline=True, fmt="%.1f", fontsize=8, colors="black")

# Alle 5 Minima einzeichnen und beschriften
for m in minima:
    is_glob = (m['idx'] == glob_min['idx'])
    ax1.plot(m['date'], m['tof'],
             marker="*" if is_glob else "o", 
             color="gold" if is_glob else "darkblue",
             markersize=14 if is_glob else 8,
             markeredgecolor="black", markeredgewidth=1.0, zorder=10)
    
    # Textplatzierung versetzen, um Überschneidungen zu vermeiden
    offset_y = 15 if is_glob else -22
    ax1.annotate(f"{m['date'].strftime('%d.%m.%y')}\n{m['dv']:.2f} km/s",
                xy=(m['date'], m['tof']),
                xytext=(0, offset_y), textcoords='offset points',
                fontsize=9, fontweight="bold" if is_glob else "normal",
                ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray", lw=0.5))

ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_xlabel("Abflugsdatum", fontsize=12)
ax1.set_ylabel("Flugzeit [Tage]", fontsize=12)
ax1.set_title("10-Jahres Porkchop-Gitter mit allen 5 synodischen Minima", fontsize=13, fontweight="bold")
ax1.set_xlim(start_epoch, start_epoch + datetime.timedelta(days=10.0 * 365.25))
ax1.set_ylim(80, 450)

# ------------ Subplot 2: Heliocentrische Flugbahnen (Transfer Orbits) ------------
# Orbits von Erde (r=1.0) und Mars (r=1.524)
theta = np.linspace(0, 2*np.pi, 200)
ax2.plot(np.cos(theta), np.sin(theta), color="#1f77b4", linestyle="--", label="Erde Orbit (1.0 AU)", lw=1.0, alpha=0.6)
ax2.plot(1.524 * np.cos(theta), 1.524 * np.sin(theta), color="#d62728", linestyle="--", label="Mars Orbit (1.524 AU)", lw=1.0, alpha=0.6)

# Sonne im Zentrum
ax2.plot(0, 0, marker="o", color="orange", markersize=15, label="Sonne", zorder=5)

# Farben für die 5 synodischen Minima
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
box_lines = []

for idx, m in enumerate(minima):
    color = colors[idx]
    x_t, y_t, xe, ye, xm, ym = get_spacecraft_trajectory(m['t_dep'], m['tof'])
    r_t = np.sqrt(x_t**2 + y_t**2)
    dt_yr = (m['tof'] / 365.25) / 150
    rad_index = np.sum(1.0 / (r_t**2)) * dt_yr
    
    # Sonnenzyklus-Multiplikator: M(t) = 1.0 + 0.9 * cos(2*pi*(t + 1)/11)
    # t ist Jahre ab 2026.0 (t_dep). Der Peak von Zyklus 25 lag um 2025.0 (t = -1.0).
    times = m['t_dep'] + np.arange(len(r_t)) * dt_yr
    multipliers = 1.0 + 0.9 * np.cos(2.0 * np.pi * (times + 1.0) / 11.0)
    rad_index_dyn = np.sum(multipliers / (r_t**2)) * dt_yr
    
    # Pfad zeichnen
    label_str = f"Fenster {m['idx']} ({m['date'].strftime('%Y')}): {m['dv']:.1f} km/s"
    ax2.plot(x_t, y_t, color=color, lw=2.0, label=label_str, zorder=8)
    
    # Start (Erde) und Ziel (Mars) einzeichnen
    ax2.plot(xe, ye, marker="o", color=color, markersize=6, zorder=9)
    ax2.plot(xm, ym, marker="^", color=color, markersize=7, zorder=9)
    
    # Verbindungsstriche zur Sonne
    ax2.plot([0, xe], [0, ye], color=color, alpha=0.15, lw=0.8)
    ax2.plot([0, xm], [0, ym], color=color, alpha=0.15, lw=0.8)
    
    box_lines.append(f"F{m['idx']} ({m['date'].strftime('%Y')}): ToF={m['tof']:.1f}d | Const={rad_index:.3f} | Dyn={rad_index_dyn:.3f} EE-J")

ax2.set_aspect("equal")
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.8, 1.8)
ax2.set_xlabel("x [AU]", fontsize=12)
ax2.set_ylabel("y [AU]", fontsize=12)
ax2.set_title("Heliocentrische Flugbahnen der 5 Transfer-Minima", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

# Box für Sonnensturm-Expositions-Indizes
text_box = "Sonnensturm-Dosis-Index:\n" + "\n".join(box_lines)
ax2.text(-1.75, -1.75, text_box, fontsize=8.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.95, ec="orange", lw=1.0))

fig.suptitle("Erde -> Mars Transfer-Missionsanalyse (C++ / CasADi)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()

output_png = "porkchop_cpp.png"
plt.savefig(output_png, dpi=150)
print(f"2-Panel Plot erfolgreich gespeichert als: {output_png}")
