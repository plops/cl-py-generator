from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
# ========================================================================
# CASADI SYMBOLIK & ROOTFINDER KONFIGURATION
# ========================================================================
# Wir loesen die Kinematik eines ebenen Viergelenkgetriebes (Four-bar linkage).
# Gegeben ist der Kurbelwinkel theta2 (Antrieb). Gesucht sind die Winkel
# theta3 (Koppel) und theta4 (Schwinge).
# Die Schleifenschlussgleichungen lauten:
#   g0: l2 * cos(theta2) + l3 * cos(theta3) - l4 * cos(theta4) - d = 0
#   g1: l2 * sin(theta2) + l3 * sin(theta3) - l4 * sin(theta4) = 0
# Dabei ist:
#   z = [theta3, theta4] (die Unbekannten)
#   x = [theta2, d, l2, l3, l4] (die Parameter)
z=SX.sym("z", 2)
theta3=z[0]
theta4=z[1]
x=SX.sym("x", 5)
theta2=x[0]
d=x[1]
l2=x[2]
l3=x[3]
l4=x[4]
g0=((l2*cos(theta2))-(l4*cos(theta4)))+l3*cos(theta3)+ -1*d
g1=((l2*sin(theta2))-(l4*sin(theta4)))+l3*sin(theta3)
g=Function("g", [z, x], [vertcat(g0, g1)])
G=rootfinder("G", "newton", g, {("error_on_fail"):(False)})
# ========================================================================
# HILFSFUNKTION FUER DIE KINEMATIK
# ========================================================================
# Loest das Gleichungssystem fuer einen gegebenen Zustand.
# Bei Konvergenzfehlern wird der letzte Schätzwert zurückgegeben.
def solve_kinematics(theta2_val, d_val, l2_val, l3_val, l4_val, z_guess_val):
    x_val=[theta2_val, d_val, l2_val, l3_val, l4_val]
    try:
        sol=G(z_guess_val, x_val)
        return np.array(sol).flatten()
    except Exception as e:
        return z_guess_val
# ========================================================================
# INITIALISIERUNG DES MATPLOTLIB AXIS & WIDGETS
# ========================================================================
fig, ax=plt.subplots(figsize=(8,8,))
plt.subplots_adjust(bottom=0.3    )
ax.set_xlim(-4, 7)
ax.set_ylim(-5, 5)
ax.set_aspect("equal")
ax.grid(True)
line_crank=ax.plot([], [], "o-", lw=4, color="royalblue", label="Kurbel (l2)")[0]
line_coupler=ax.plot([], [], "o-", lw=4, color="forestgreen", label="Koppel (l3)")[0]
line_rocker=ax.plot([], [], "o-", lw=4, color="crimson", label="Schwinge (l4)")[0]
line_ground=ax.plot([], [], "o--", lw=2, color="gray", label="Gestell (d)")[0]
point_P=ax.plot([], [], "ro", ms=8, label="Koppelpunkt P")[0]
line_locus=ax.plot([], [], "r-", lw=2, alpha=0.6    , label="Koppelkurve (Locus)")[0]
ax.plot([0.    ], [0.    ], "ks", ms=10, zorder=5)
point_O4=ax.plot([], [], "ks", ms=10, zorder=5)[0]
ax.legend(loc="upper right")
# ========================================================================
# SLIDER-GUI FOR PARAMETER CONTROL
# ========================================================================
AX_SLIDER_D=plt.axes([0.15    , 0.2    , 0.65    , 3.0e-2])
slider_d=Slider(AX_SLIDER_D, "d (Gestell)", 1.    , 6.    , valinit=3.    )
AX_SLIDER_L2=plt.axes([0.15    , 0.15    , 0.65    , 3.0e-2])
slider_l2=Slider(AX_SLIDER_L2, "l2 (Kurbel)", 0.2    , 3.    , valinit=1.    )
AX_SLIDER_L3=plt.axes([0.15    , 0.1    , 0.65    , 3.0e-2])
slider_l3=Slider(AX_SLIDER_L3, "l3 (Koppel)", 1.    , 6.    , valinit=3.    )
AX_SLIDER_L4=plt.axes([0.15    , 5.0e-2, 0.65    , 3.0e-2])
slider_l4=Slider(AX_SLIDER_L4, "l4 (Schwinge)", 0.5    , 6.    , valinit=2.5    )
# ========================================================================
# GLOBALE ZUSTAENDE FUER SCHAETZWERTE UND LOKUS-SPUR
# ========================================================================
last_z=np.array([np.pi/4, np.pi/2])
locus_x=[]
locus_y=[]
theta2_vals=np.linspace(0.    , 2*np.pi, 200)
# ========================================================================
# ANIMATIONS-UPDATE FUNKTION
# ========================================================================
# Wird fuer jeden Frame aufgerufen. Liest die Slider-Werte und loest die
# Kinematik fuer den aktuellen Kurbelwinkel theta2 auf.
def update(frame):
    global last_z
    d_val=slider_d.val
    l2_val=slider_l2.val
    l3_val=slider_l3.val
    l4_val=slider_l4.val
    theta2_val=theta2_vals[frame]
    sol=solve_kinematics(theta2_val, d_val, l2_val, l3_val, l4_val, last_z)
    last_z=sol
    theta3_val=sol[0]
    theta4_val=sol[1]
    x_O2=0.    
    y_O2=0.    
    x_A=l2_val*np.cos(theta2_val)
    y_A=l2_val*np.sin(theta2_val)
    x_B=d_val+l4_val*np.cos(theta4_val)
    y_B=l4_val*np.sin(theta4_val)
    x_O4=d_val
    y_O4=0.    
    l_AP=1.5    
    alpha=0.5    
    x_P=x_A+l_AP*np.cos(theta3_val+alpha)
    y_P=y_A+l_AP*np.sin(theta3_val+alpha)
    line_crank.set_data([x_O2, x_A], [y_O2, y_A])
    line_coupler.set_data([x_A, x_B], [y_A, y_B])
    line_rocker.set_data([x_B, x_O4], [y_B, y_O4])
    line_ground.set_data([x_O2, x_O4], [y_O2, y_O4])
    point_O4.set_data([x_O4], [y_O4])
    point_P.set_data([x_P], [y_P])
    locus_x.append(x_P)
    locus_y.append(y_P)
    if len(locus_x)>len(theta2_vals):
        locus_x.pop(0)
        locus_y.pop(0)
    line_locus.set_data(locus_x, locus_y)
    return (line_crank,line_coupler,line_rocker,line_ground,point_O4,point_P,line_locus,)
# ========================================================================
# SLIDER EVENT-HANDLER
# ========================================================================
# Löscht die Koppelkurve und setzt die Startschätzung zurück, wenn
# der Benutzer die Geometrie ändert.
def on_slider_change(val):
    global locus_x
    global locus_y
    global last_z
    locus_x.clear()
    locus_y.clear()
    last_z=np.array([np.pi/4, np.pi/2])
slider_d.on_changed(on_slider_change)
slider_l2.on_changed(on_slider_change)
slider_l3.on_changed(on_slider_change)
slider_l4.on_changed(on_slider_change)
ani=FuncAnimation(fig, update, frames=len(theta2_vals), interval=20, blit=True)
ani.save("i05_linkage.gif", writer="pillow", fps=30)