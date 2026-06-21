from __future__ import annotations
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time as time
# ========================================================================
# PHYSIKALISCHE KONSTANTEN
# ========================================================================
MU_SUN=(((4.0    ))*(((np.pi)**(2))))
AU_KM=(1.5e+8)
YR_S=(((365.250    ))*(24)*(3600))
V_CONV=((AU_KM)/(YR_S))
A_EARTH=(1.0    )
T_EARTH=(1.0    )
A_MARS=(1.5240    )
T_MARS=(1.8810    )
def planet_state(t_yr, a, period):
    omega=((((2)*(np.pi)))/(period))
    theta=((omega)*(t_yr))
    x=((a)*(np.cos(theta)))
    y=((a)*(np.sin(theta)))
    v=((((2)*(np.pi)*(a)))/(period))
    vx=((-1)*(v)*(np.sin(theta)))
    vy=((v)*(np.cos(theta)))
    return (x,y,vx,vy,)
def create_integrator():
    x=SX.sym("x")
    y=SX.sym("y")
    vx=SX.sym("vx")
    vy=SX.sym("vy")
    state=vertcat(x, y, vx, vy)
    tof=SX.sym("tof")
    r=sqrt(((((x)**(2)))+(((y)**(2)))))
    ode=((tof)*(vertcat(vx, vy, ((((-1)*(MU_SUN)*(x)))/(((r)**(3)))), ((((-1)*(MU_SUN)*(y)))/(((r)**(3)))))))
    dae={("x"):(state), ("p"):(tof), ("ode"):(ode)}
    return integrator("F", "cvodes", dae, {("t0"):(0), ("tf"):(1), ("reltol"):((1.00e-10)), ("abstol"):((1.00e-12))})
# ========================================================================
# WORKER-INITIALISIERUNG: SOLVER DEFINITIONEN
# ========================================================================
# Die CasADi-Rootfinder werden einmal pro Worker-Prozess initialisiert.
# Das vermeidet Serialisierungsprobleme und Overhead.
worker_state={}
def worker_init():
    F_int=create_integrator()
    # --- Iso-Delta-v Continuation Solver ---
    z=SX.sym("z", 4)
    vx0=z[0]
    vy0=z[1]
    t_dep=z[2]
    tof=z[3]
    p=SX.sym("p", 9)
    C_val=p[0]
    z_pred=p[1:5]
    T_pred=p[5:9]
    x_E, y_E, vx_E, vy_E=planet_state(t_dep, A_EARTH, T_EARTH)
    x_M, y_M, vx_M, vy_M=planet_state(((t_dep)+(tof)), A_MARS, T_MARS)
    x0_vec=vertcat(x_E, y_E, vx0, vy0)
    res_int=F_int(x0=x0_vec, p=tof)
    xf=res_int["xf"]
    dv_dep=sqrt(((((((vx0)-(vx_E)))**(2)))+(((((vy0)-(vy_E)))**(2)))))
    dv_arr=sqrt(((((((xf[2])-(vx_M)))**(2)))+(((((xf[3])-(vy_M)))**(2)))))
    dv_tot=((((dv_dep)+(dv_arr)))*(V_CONV))
    g1=((xf[0])-(x_M))
    g2=((xf[1])-(y_M))
    g3=((dv_tot)-(C_val))
    g_phys=vertcat(g1, g2, g3)
    J=jacobian(g_phys, z)
    J_iso_dv_func=Function("J", [z], [J])
    g4=casadi.dot(((z)-(z_pred)), T_pred)
    g_all=vertcat(g_phys, g4)
    rf_dict={("x"):(z), ("p"):(p), ("g"):(g_all)}
    solver_iso_dv=rootfinder("rf", "newton", rf_dict, {("error_on_fail"):(False)})
    worker_state["solver_iso_dv"]=solver_iso_dv
    worker_state["J_iso_dv_func"]=J_iso_dv_func
# ========================================================================
# WORKER-FUNKTION: KONTUR-VERFOLGUNG
# ========================================================================
def trace_contour_worker(args):
    solver_iso_dv=worker_state["solver_iso_dv"]
    J_iso_dv_func=worker_state["J_iso_dv_func"]
    C_target, z_start_arr=args
    path=[]
    z_curr=np.array(z_start_arr)
    T_prev=None
    ds=(1.50e-2)
    should_break=False
    for step in range(3000):
        if ( not(should_break) ):
            J_mat=np.array(J_iso_dv_func(z_curr))
            U, S, Vh=np.linalg.svd(J_mat)
            T=Vh[-1,:]
            if ( (T_prev is not None) ):
                if ( ((np.dot(T, T_prev))<(0)) ):
                    T=((-1)*(T))
            T_prev=T
            z_pred=((z_curr)+(((ds)*(T))))
            p_val=np.concatenate([[C_target], z_pred, T])
            try:
                sol=solver_iso_dv(z_pred, p_val)
                z_next=np.array(sol).flatten()
            except Exception:
                should_break=True
                z_next=np.array([np.nan, np.nan, np.nan, np.nan])
            if ( np.any(np.isnan(z_next)) ):
                should_break=True
            if ( not(should_break) ):
                path.append(z_next)
                z_curr=z_next
                if ( ((step)>(50)) ):
                    dist=np.linalg.norm(((z_curr)-(z_start_arr)))
                    if ( ((dist)<(((ds)*((1.50    ))))) ):
                        path.append(z_start_arr)
                        should_break=True
    return (C_target,np.array(path),)
# ========================================================================
# HAUPTPROGRAMM: MINIMUM-SUCHE & PARALLELES TRACING
# ========================================================================
if ( ((__name__)==("__main__")) ):
    print("Suche globales Minimum (Hohmann-Region)...")
    F_int_main=create_integrator()
    min_dv=(1.0e+9)
    z_min=None
    for t_guess in np.linspace((0.10    ), (2.0    ), 10):
        opti=casadi.Opti()
        z_var=opti.variable(4)
        vx0=z_var[0]
        vy0=z_var[1]
        t_dep=z_var[2]
        tof=z_var[3]
        opti.subject_to(opti.bounded((0.    ), t_dep, (2.50    )))
        opti.subject_to(opti.bounded((0.20    ), tof, (1.50    )))
        x_E, y_E, vx_E, vy_E=planet_state(t_dep, A_EARTH, T_EARTH)
        x_M, y_M, vx_M, vy_M=planet_state(((t_dep)+(tof)), A_MARS, T_MARS)
        x0_vec=vertcat(x_E, y_E, vx0, vy0)
        res_int=F_int_main(x0=x0_vec, p=tof)
        xf=res_int["xf"]
        opti.subject_to(((xf[0])==(x_M)))
        opti.subject_to(((xf[1])==(y_M)))
        dv_dep=sqrt(((((((vx0)-(vx_E)))**(2)))+(((((vy0)-(vy_E)))**(2)))))
        dv_arr=sqrt(((((((xf[2])-(vx_M)))**(2)))+(((((xf[3])-(vy_M)))**(2)))))
        dv_tot=((((dv_dep)+(dv_arr)))*(V_CONV))
        opti.minimize(dv_tot)
        hohmann_guess_vx=(((1.10    ))*(((((((2)*(np.pi)*(A_EARTH)))/(T_EARTH)))*(np.cos(((np.pi)/(6)))))))
        hohmann_guess_vy=(((1.10    ))*(((((((2)*(np.pi)*(A_EARTH)))/(T_EARTH)))*(np.sin(((np.pi)/(6)))))))
        opti.set_initial(vx0, hohmann_guess_vx)
        opti.set_initial(vy0, hohmann_guess_vy)
        opti.set_initial(t_dep, t_guess)
        opti.set_initial(tof, (0.70    ))
        opti.solver("ipopt", {("print_time"):(False)}, {("print_level"):(0)})
        try:
            sol_min=opti.solve()
            val=sol_min.value(dv_tot)
            if ( ((val)<(min_dv)) ):
                min_dv=val
                z_min=np.array(sol_min.value(z_var))
        except Exception:
            pass
    C_min=min_dv
    print(f"Globales Minimum gefunden: Delta-v = {C_min:.2f} km/s at t_dep={z_min[2]:.3f}, tof={z_min[3]:.3f}")
    # Startpunkte für Höhenlinien suchen
    contour_levels=np.arange(((np.ceil(C_min))+((0.50    ))), (15.    ), (0.50    ))
    start_tasks=[]
    for C in contour_levels:
        opti_C=casadi.Opti()
        z_C=opti_C.variable(4)
        vx0_C=z_C[0]
        vy0_C=z_C[1]
        t_dep_C=z_C[2]
        tof_C=z_C[3]
        x_E_C, y_E_C, vx_E_C, vy_E_C=planet_state(t_dep_C, A_EARTH, T_EARTH)
        x_M_C, y_M_C, vx_M_C, vy_M_C=planet_state(((t_dep_C)+(tof_C)), A_MARS, T_MARS)
        res_int_C=F_int_main(x0=vertcat(x_E_C, y_E_C, vx0_C, vy0_C), p=tof_C)
        xf_C=res_int_C["xf"]
        opti_C.subject_to(((xf_C[0])==(x_M_C)))
        opti_C.subject_to(((xf_C[1])==(y_M_C)))
        dv_dep_C=sqrt(((((((vx0_C)-(vx_E_C)))**(2)))+(((((vy0_C)-(vy_E_C)))**(2)))))
        dv_arr_C=sqrt(((((((xf_C[2])-(vx_M_C)))**(2)))+(((((xf_C[3])-(vy_M_C)))**(2)))))
        dv_tot_C=((((dv_dep_C)+(dv_arr_C)))*(V_CONV))
        opti_C.subject_to(((dv_tot_C)==(C)))
        opti_C.subject_to(((tof_C)==(z_min[3])))
        opti_C.subject_to(opti_C.bounded(((z_min[2])-((1.50    ))), t_dep_C, z_min[2]))
        opti_C.minimize(0)
        opti_C.set_initial(z_C, z_min)
        opti_C.solver("ipopt", {}, {("print_level"):(0)})
        try:
            sol_C=opti_C.solve()
            start_tasks.append((C,np.array(sol_C.value(z_C)),))
        except Exception:
            print(f"Kein Startpunkt für C={C} gefunden.")
    print(f"Gefundene Startpunkte: {len(start_tasks)}")
    if ( ((len(start_tasks))==(0)) ):
        print("FEHLER: Keine Startpunkte für die Höhenlinien gefunden!")
    else:
        n_cpus=mp.cpu_count()
        print(f"Starte paralleles Tracing auf {n_cpus} Kernen...")
        t_start=time.time()
        with mp.Pool(processes=n_cpus, initializer=worker_init) as pool:
            results=pool.map(trace_contour_worker, start_tasks)
        elapsed=((time.time())-(t_start))
        print(f"Tracing abgeschlossen in {elapsed:.1f} Sekunden")
        # ========================================================================
# VISUALISIERUNG: PORK-CHOP-DIAGRAMM
# ========================================================================
        fig_and_ax=plt.subplots(1, 1, figsize=(12,9,))
        fig, ax=fig_and_ax
        cmap=plt.get_cmap("RdYlGn_r")
        norm=plt.Normalize(vmin=np.min(contour_levels), vmax=np.max(contour_levels))
        for C, path_arr in results:
            if ( ((len(path_arr))>(0)) ):
                t_dep_path=path_arr[:,2]
                tof_days_path=((path_arr[:,3])*((365.250    )))
                color=cmap(norm(C))
                ax.plot(t_dep_path, tof_days_path, color=color, linewidth=(2.0    ))
                ax.text(t_dep_path[0], tof_days_path[0], f"{C}", color=color, fontsize=10, fontweight="bold", ha="right", va="center")
        ax.plot(z_min[2], ((z_min[3])*((365.250    ))), marker="*", color="gold", markersize=20, markeredgecolor="black", markeredgewidth=(1.50    ), zorder=10)
        ax.annotate(f"$\\Delta v_{{min}}$ = {C_min:.2f} km/s", xy=(z_min[2],((z_min[3])*((365.250    ))),), xytext=(((z_min[2])+((0.150    ))),((((z_min[3])*((365.250    ))))+(25)),), fontsize=11, fontweight="bold", color="black", arrowprops={("arrowstyle"):("->"), ("color"):("black"), ("lw"):((1.50    ))})
        sm=plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar=fig.colorbar(sm, ax=ax)
        cbar.set_label("Total $\\Delta v$ [km/s]", fontsize=13)
        ax.set_xlim((0.    ), (2.50    ))
        ax.set_ylim(100, 450)
        ax.set_xlabel("Departure date [years from epoch]", fontsize=13)
        ax.set_ylabel("Transfer duration [days]", fontsize=13)
        ax.set_title("Earth $\\rightarrow$ Mars: Pork Chop Diagram (Continuation Method)", fontsize=15, fontweight="bold")
        ax.grid(True, alpha=(0.30    ), linestyle="--")
        plt.tight_layout()
        plt.savefig("porkchop_fast.png", dpi=150)
        print("Plot gespeichert: porkchop_fast.png")
        plt.show()