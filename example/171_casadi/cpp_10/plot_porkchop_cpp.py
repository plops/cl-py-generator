import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import csv

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

# Umrechnung in echte Kalenderdaten (Epoche: 1. Januar 2026)
start_epoch = datetime.datetime(2026, 1, 1)
X_dates = [start_epoch + datetime.timedelta(days=float(t) * 365.25) for t in X]
min_date = start_epoch + datetime.timedelta(days=float(min_t) * 365.25)

print(f"Globales Minimum gefunden bei Delta-v = {min_dv:.2f} km/s")
print(f"  Abflugsdatum: {min_date.strftime('%Y-%m-%d')} (t_dep={min_t:.3f} Jahre)")
print(f"  Flugzeit:     {min_tof:.1f} Tage")

# Plot erstellen
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# 1. Vollständige Heatmap aller berechneten Punkte zeichnen
im = ax.pcolormesh(X_dates, Y, Z, cmap="RdYlGn_r", shading="auto", vmin=5.5, vmax=25.0)

# Colorbar hinzufügen
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Gesamt-$\\Delta v$ [km/s]", fontsize=12)

# 2. Ausgewählte schwarze Höhenlinien für markante Werte drüberlegen
levels = [6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
# Um die Konturlinien über Dates zu zeichnen, übergeben wir X_dates an contour
contour = ax.contour(X_dates, Y, Z, levels=levels, colors="black", linewidths=0.6, alpha=0.8)
ax.clabel(contour, inline=True, fmt="%.1f", fontsize=8, colors="black")

# Minimum eintragen
ax.plot(min_date, min_tof,
        marker="*", color="gold", markersize=16,
        markeredgecolor="black", markeredgewidth=1.2, zorder=10)

ax.annotate(f"$\\Delta v_{{min}}$ = {min_dv:.2f} km/s\n({min_date.strftime('%d.%m.%Y')})",
            xy=(min_date, min_tof),
            xytext=(min_date + datetime.timedelta(days=45), min_tof + 30),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

# Datumsachsenformatierung
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Beschriftung alle 3 Monate
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format: Jan 2026
fig.autofmt_xdate()  # Schräge Datumsbeschriftung für bessere Lesbarkeit

# Achsenbeschriftung und Titel
ax.set_xlabel("Abflugsdatum", fontsize=12)
ax.set_ylabel("Flugzeit [Tage]", fontsize=12)
ax.set_title("Erde -> Mars Porkchop-Diagramm (Missionsfenster ab 2026)\n(Berechnet in C++ auf mehreren CPU-Kernen)", 
             fontsize=14, fontweight="bold")

# Grenzen der Datumsachse festlegen
date_min = start_epoch
date_max = start_epoch + datetime.timedelta(days=2.5 * 365.25)
ax.set_xlim(date_min, date_max)
ax.set_ylim(100, 450)

plt.tight_layout()
output_png = "porkchop_cpp.png"
plt.savefig(output_png, dpi=150)
print(f"Plot erfolgreich gespeichert als: {output_png}")
