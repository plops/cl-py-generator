from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
def simulate_hh(E: float, y0_list: list, t_max: float = (1.00e+3), n_steps: int = 100000)->list:
    x=SX.sym("x")
    y=SX.sym("y")
    px=SX.sym("px")
    py=SX.sym("py")
    state=vertcat(x, y, px, py)
    lambda_val=(1.0    )
    V=(((((0.50    ))*(((((x)**(2)))+(((y)**(2)))))))+(((lambda_val)*(((((((x)**(2)))*(y)))-(((((y)**(3)))/((3.0    )))))))))
    T_kin=(((0.50    ))*(((((px)**(2)))+(((py)**(2))))))
    H=((T_kin)+(V))
    ode=vertcat(px, py, ((-1)*(jacobian(H, x))), ((-1)*(jacobian(H, y))))
    dae={("x"):(state), ("ode"):(ode)}
    t_grid=np.linspace((0.    ), t_max, n_steps)
    F=integrator("F", "cvodes", dae, {("grid"):(t_grid), ("output_t0"):(True)})
    results=[]
    for y0 in y0_list:
        V_0=(((((0.50    ))*(((y0)**(2)))))-(((((y0)**(3)))/((3.0    )))))
        if ( ((V_0)<(E)) ):
            px0=np.sqrt(((2)*(((E)-(V_0)))))
            w0=np.array([(0.    ), y0, px0, (0.    )])
            sol=F(x0=w0)
            xf=sol["xf"].full()
            x_arr=xf[0,:]
            y_arr=xf[1,:]
            px_arr=xf[2,:]
            py_arr=xf[3,:]
            idx=np.where(((((((x_arr[:-1])<=(0))) & (((x_arr[1:])>(0))))) & (((px_arr[:-1])>(0)))))[0]
            y_crossings=[]
            py_crossings=[]
            for i in idx:
                t_frac=((((-1)*(x_arr[i])))/(((x_arr[((i)+(1))])-(x_arr[i]))))
                y_cross=((y_arr[i])+(((t_frac)*(((y_arr[((i)+(1))])-(y_arr[i]))))))
                py_cross=((py_arr[i])+(((t_frac)*(((py_arr[((i)+(1))])-(py_arr[i]))))))
                y_crossings.append(y_cross)
                py_crossings.append(py_cross)
            results.append({("y0"):(y0), ("x"):(x_arr), ("y"):(y_arr), ("y_cross"):(np.array(y_crossings)), ("py_cross"):(np.array(py_crossings))})
        else:
            print(f"Initial condition y0={y0} is energetically inaccessible for energy E={E}")
    return results
fig_and_axs=plt.subplots(2, 2, figsize=(12,10,))
fig, axs=fig_and_axs
ax_ol=axs[0,0]
ax_pl=axs[0,1]
ax_oh=axs[1,0]
ax_ph=axs[1,1]
E_low=(8.3330e-2)
y0_low=[(-0.250    ), (-0.150    ), (-5.00e-2), (5.00e-2), (0.150    ), (0.250    ), (0.350    ), (0.420    )]
results_low=simulate_hh(E_low, y0_low, t_max=(2.00e+3), n_steps=200000)
E_high=(0.150    )
y0_high=[(-0.350    ), (-0.20    ), (-5.00e-2), (0.10    ), (0.250    ), (0.40    ), (0.550    )]
results_high=simulate_hh(E_high, y0_high, t_max=(2.00e+3), n_steps=200000)
x_g=np.linspace((-1.20    ), (1.20    ), 200)
y_g=np.linspace((-1.20    ), (1.20    ), 200)
X, Y=np.meshgrid(x_g, y_g)
Z=(((((0.50    ))*(((((X)**(2)))+(((Y)**(2)))))))+(((((((X)**(2)))*(Y)))-(((((Y)**(3)))/((3.0    )))))))
for res in results_low:
    ax_ol.plot(res["x"], res["y"], alpha=(0.50    ))
    ax_pl.scatter(res["y_cross"], res["py_cross"], s=(1.50    ), alpha=(0.80    ))
ax_ol.contour(X, Y, Z, levels=[E_low], colors="red", linestyles="dashed")
for res in results_high:
    ax_oh.plot(res["x"], res["y"], alpha=(0.50    ))
    ax_ph.scatter(res["y_cross"], res["py_cross"], s=(1.50    ), alpha=(0.80    ))
ax_oh.contour(X, Y, Z, levels=[E_high], colors="red", linestyles="dashed")
ax_ol.set_aspect("equal")
ax_ol.set_title(f"Orbits (E = {E_low})")
ax_ol.set_xlabel("x")
ax_ol.set_ylabel("y")
ax_ol.grid(True)
ax_pl.set_title(f"Poincare Section x=0, px>0 (E = {E_low})")
ax_pl.set_xlabel("y")
ax_pl.set_ylabel("py")
ax_pl.grid(True)
ax_oh.set_aspect("equal")
ax_oh.set_title(f"Orbits (E = {E_high})")
ax_oh.set_xlabel("x")
ax_oh.set_ylabel("y")
ax_oh.grid(True)
ax_ph.set_title(f"Poincare Section x=0, px>0 (E = {E_high})")
ax_ph.set_xlabel("y")
ax_ph.set_ylabel("py")
ax_ph.grid(True)
plt.tight_layout()
plt.savefig("henon_heiles_chaos.png", dpi=150)
print("Plot saved to henon_heiles_chaos.png")