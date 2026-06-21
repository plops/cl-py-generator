import numpy as np
import matplotlib.pyplot as plt
import csv

# Global plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee'
})

print("Lese CSV-Daten ohne Pandas...")
t_dep_list = []
tof_days_list = []
dv_tot_list = []

# CSV einlesen
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

# Finden der eindeutigen sortierten Achsenbeschriftungen für das Gitter
X = np.unique(t_dep)
Y = np.unique(tof_days)

# 2D-Gitter mit NaNs vorbelegen
Z = np.full((len(Y), len(X)), np.nan)

# Zuordnungstabellen für schnellen Index-Lookup
x_map = {val: idx for idx, val in enumerate(X)}
y_map = {val: idx for idx, val in enumerate(Y)}

# Werte in das 2D-Gitter einsortieren
for t, tof, dv in zip(t_dep, tof_days, dv_tot):
    ix = x_map[t]
    iy = y_map[tof]
    Z[iy, ix] = dv

# Globales Minimum finden
min_idx = np.argmin(dv_tot)
min_t = t_dep[min_idx]
min_tof = tof_days[min_idx]
min_dv = dv_tot[min_idx]
print(f"Globales Minimum gefunden bei Delta-v = {min_dv:.2f} km/s (t_dep={min_t:.3f}, tof={min_tof:.1f} Tage)")

# Plot erstellen
fig, ax = plt.subplots(1, 1, figsize=(11, 8))

# Konturstufen festlegen
levels = np.arange(5.5, 15.0, 0.5)

# Konturen zeichnen
contour = ax.contour(X, Y, Z, levels=levels, cmap="RdYlGn_r", linewidths=1.5)
ax.clabel(contour, inline=True, fmt="%.1f", fontsize=9, colors="black")

# Konturen ausfüllen für bessere Ästhetik
contourf = ax.contourf(X, Y, Z, levels=levels, cmap="RdYlGn_r", alpha=0.3)

# Colorbar hinzufügen
cbar = fig.colorbar(contourf, ax=ax)
cbar.set_label("Gesamt-$\\Delta v$ [km/s]", fontsize=12)

# Minimum eintragen
ax.plot(min_t, min_tof,
        marker="*", color="gold", markersize=16,
        markeredgecolor="black", markeredgewidth=1.2, zorder=10)

ax.annotate(f"$\\Delta v_{{min}}$ = {min_dv:.2f} km/s\n(Hohmann)",
            xy=(min_t, min_tof),
            xytext=(min_t + 0.1, min_tof + 20),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

# Achsenbeschriftung und Titel
ax.set_xlabel("Abflugsdatum [Jahre ab Epoche]", fontsize=12)
ax.set_ylabel("Flugzeit [Tage]", fontsize=12)
ax.set_title("Erde -> Mars Porkchop-Diagramm\n(Berechnet in C++ auf mehreren CPU-Kernen)", 
             fontsize=14, fontweight="bold")

plt.tight_layout()
output_png = "porkchop_cpp.png"
plt.savefig(output_png, dpi=150)
print(f"Plot erfolgreich gespeichert als: {output_png}")
