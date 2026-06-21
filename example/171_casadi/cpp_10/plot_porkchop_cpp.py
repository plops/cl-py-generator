import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee'
})

print("Lese CSV-Daten...")
df = pd.read_csv("porkchop_cpp.csv")

print("Erstelle Grid via Pivot...")
pivot_df = df.pivot(index="tof_days", columns="t_dep", values="dv_tot")

X = pivot_df.columns.values
Y = pivot_df.index.values
Z = pivot_df.values

# Finde das Minimum
min_idx = df["dv_tot"].idxmin()
min_row = df.loc[min_idx]
print(f"Globales Minimum gefunden bei Delta-v = {min_row['dv_tot']:.2f} km/s (t_dep={min_row['t_dep']:.3f}, tof={min_row['tof_days']:.1f} Tage)")

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
ax.plot(min_row["t_dep"], min_row["tof_days"],
        marker="*", color="gold", markersize=16,
        markeredgecolor="black", markeredgewidth=1.2, zorder=10)

ax.annotate(f"$\\Delta v_{{min}}$ = {min_row['dv_tot']:.2f} km/s\n(Hohmann)",
            xy=(min_row["t_dep"], min_row["tof_days"]),
            xytext=(min_row["t_dep"] + 0.1, min_row["tof_days"] + 20),
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
