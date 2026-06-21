from __future__ import annotations
from casadi import *
import numpy as np
import os
import shutil
import time as time

# ========================================================================
# PHYSIKALISCHE KONSTANTEN
# ========================================================================
MU_SUN = 4.0 * (np.pi**2)
AU_KM = 1.5e8
YR_S = 365.25 * 24 * 3600
V_CONV = AU_KM / YR_S
A_EARTH = 1.0
T_EARTH = 1.0
A_MARS = 1.524
T_MARS = 1.881


# ========================================================================
# ANALYTISCHE PLANETENBAHNEN
# ========================================================================
def planet_state(t_yr, a, period):
    omega = (2 * np.pi) / period
    theta = omega * t_yr
    x = a * cos(theta)
    y = a * sin(theta)
    v = (2 * np.pi * a) / period
    vx = -1 * v * sin(theta)
    vy = v * cos(theta)
    return (
        x,
        y,
        vx,
        vy,
    )


# ========================================================================
# SYMBOLISCHER RK4-INTEGRATOR
# ========================================================================
def integrate_rk4(x0, y0, vx0, vy0, tof):
    s_curr = vertcat(x0, y0, vx0, vy0)
    N_steps = 150
    h = tof / N_steps
    for i in range(N_steps):
        r1 = sqrt(((s_curr[0]) ** 2) + ((s_curr[1]) ** 2))
        ax1 = (-1 * MU_SUN * s_curr[0]) / (r1**3)
        ay1 = (-1 * MU_SUN * s_curr[1]) / (r1**3)
        k1 = vertcat(s_curr[2], s_curr[3], ax1, ay1)
        s2 = s_curr + 0.5 * h * k1
        r2 = sqrt(((s2[0]) ** 2) + ((s2[1]) ** 2))
        ax2 = (-1 * MU_SUN * s2[0]) / (r2**3)
        ay2 = (-1 * MU_SUN * s2[1]) / (r2**3)
        k2 = vertcat(s2[2], s2[3], ax2, ay2)
        s3 = s_curr + 0.5 * h * k2
        r3 = sqrt(((s3[0]) ** 2) + ((s3[1]) ** 2))
        ax3 = (-1 * MU_SUN * s3[0]) / (r3**3)
        ay3 = (-1 * MU_SUN * s3[1]) / (r3**3)
        k3 = vertcat(s3[2], s3[3], ax3, ay3)
        s4 = s_curr + h * k3
        r4 = sqrt(((s4[0]) ** 2) + ((s4[1]) ** 2))
        ax4 = (-1 * MU_SUN * s4[0]) / (r4**3)
        ay4 = (-1 * MU_SUN * s4[1]) / (r4**3)
        k4 = vertcat(s4[2], s4[3], ax4, ay4)
        s_curr = s_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return s_curr


# ========================================================================
# EVALUIERUNGS-FUNKTION (DYNAMIK + JACOBIAN)
# ========================================================================
z = SX.sym("z", 2)
p = SX.sym("p", 2)
vx0 = z[0]
vy0 = z[1]
t_dep = p[0]
tof = p[1]
x_E, y_E, vx_E, vy_E = planet_state(t_dep, A_EARTH, T_EARTH)
x_M, y_M, vx_M, vy_M = planet_state(t_dep + tof, A_MARS, T_MARS)
s_end = integrate_rk4(x_E, y_E, vx0, vy0, tof)
g_eq = vertcat((s_end[0]) - x_M, (s_end[1]) - y_M)
J = jacobian(g_eq, z)
lambert_eval = Function(
    "lambert_eval", [z, p], [g_eq, J, s_end, vertcat(vx_E, vy_E), vertcat(vx_M, vy_M)]
)
# ========================================================================
# C++ CODEGEN EXPORT & VERSCHIEBUNG NACH 'cpp_10'
# ========================================================================
file_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(file_dir, "cpp_10")
os.makedirs(target_dir, exist_ok=True)
opts = {("cpp"): (True), ("with_header"): (True)}
lambert_eval.generate("lambert_solver.cpp", opts)
shutil.move("lambert_solver.cpp", os.path.join(target_dir, "lambert_solver.cpp"))
shutil.move("lambert_solver.h", os.path.join(target_dir, "lambert_solver.h"))
print("C++-Code erfolgreich exportiert")
