from __future__ import annotations
from casadi import *
import numpy as np
import time as time


def setup_mpc(use_jit):
    opti = Opti()
    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    nx = 4
    nu = 1
    x = SX.sym("x", nx)
    u = SX.sym("u", nu)
    s_ = x[0]
    v_ = x[1]
    theta_ = x[2]
    omega_ = x[3]
    F_ = u
    sin_theta = np.sin(theta_)
    cos_theta = np.cos(theta_)
    den = M + m * (1.0 - (cos_theta * cos_theta))
    ds = v_
    dv = (
        F_ + m * l * omega_ * omega_ * sin_theta + m * g * cos_theta * sin_theta
    ) / den
    dtheta = omega_
    domega = (
        (-1.0 * F_ * cos_theta)
        - (m * l * omega_ * omega_ * sin_theta * cos_theta)
        - ((M + m) * g * sin_theta)
    ) / (l * den)
    f_ode = Function("f_ode", [x, u], [vertcat(ds, dv, dtheta, domega)])
    N = 20
    T = 1.0
    h = T / N
    d = 3
    tau_root = np.append(0.0, collocation_points(d, "radau"))
    C = np.zeros(
        (
            d + 1,
            d + 1,
        )
    )
    D = np.zeros(d + 1)
    for j in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p = p * (
                    np.poly1d([1.0, -tau_root[r]]) / ((tau_root[j]) - (tau_root[r]))
                )
        D[j] = p(1.0)
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])
    X = opti.variable(nx, N + 1)
    Xc = []
    for k in range(N):
        Xc_k = []
        for r in range(d):
            Xc_k.append(opti.variable(nx))
        Xc.append(Xc_k)
    U = opti.variable(nu, N)
    current_x = opti.parameter(nx)
    target_x = opti.parameter(nx)
    opti.set_value(current_x, np.array([0.0, 0.0, np.pi, 0.0]))
    opti.set_value(target_x, np.array([1.0, 0.0, 0.0, 0.0]))
    opti.subject_to(X[:, 0] == current_x)
    opti.subject_to(opti.bounded(-2.0, X[0, :], 2.0))
    opti.subject_to(opti.bounded(-1.5e1, U, 1.5e1))
    for k in range(N):
        Xk = X[:, k]
        x_end = D[0] * Xk
        for j in range(1, d + 1):
            xp = C[0, j] * Xk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[k][r]
            f_eval = f_ode(Xc[k][j - 1], U[:, k])
            opti.subject_to(xp == h * f_eval)
        for r in range(d):
            x_end = x_end + D[r + 1] * Xc[k][r]
        opti.subject_to(X[:, k + 1] == x_end)
    cost = 0.0
    Q = np.diag([1.0e1, 1.0, 1.0e1, 1.0])
    R = 1.0e-2
    for k in range(N):
        err = (X[:, k]) - target_x
        cost = cost + mtimes(mtimes(err.T, Q), err) + R * ((U[0, k]) ** 2)
    err_term = (X[:, N]) - target_x
    cost = cost + 1.0e1 * mtimes(mtimes(err_term.T, Q), err_term)
    opti.minimize(cost)
    solver_opts = {("print_time"): (False), ("ipopt"): ({("print_level"): (0)})}
    if use_jit:
        print("JIT aktiviert: Kompiliere NLP zu C-Code... (das dauert einen Moment)")
        solver_opts["jit"] = True
        solver_opts["compiler"] = "shell"
        solver_opts["jit_options"] = {("flags"): (["-O3"])}
    opti.solver("ipopt", solver_opts)
    return (
        opti,
        X,
        U,
        current_x,
        target_x,
        f_ode,
    )


def run_benchmark(opti, X, U, current_x, target_x, f_ode, name):
    print("")
    print("--- Benchmark: " + name + " ---")
    x0 = np.array([0.0, 0.0, np.pi, 0.0])
    opti.set_value(current_x, x0)
    opti.set_initial(X, np.linspace(x0, np.array([1.0, 0.0, 0.0, 0.0]), 20 + 1).T)
    t0 = time.time()
    sol = opti.solve()
    t_compile_and_cold = time.time() - t0
    # Zuruecksetzen auf Kaltstart-Guess, um den reinen Solve ohne GCC Kompilierungszeit zu messen
    opti.set_value(current_x, x0)
    opti.set_initial(X, np.linspace(x0, np.array([1.0, 0.0, 0.0, 0.0]), 20 + 1).T)
    opti.set_initial(
        U,
        np.zeros(
            (
                1,
                20,
            )
        ),
    )
    t0 = time.time()
    sol = opti.solve()
    t_cold = time.time() - t0
    print(f"JIT + Cold Start Time: {t_compile_and_cold * 1000:.1f} ms")
    print(f"Pure Cold Start Time : {t_cold * 1000:.1f} ms")
    n_steps = 50
    times = []
    state = x0
    for i in range(n_steps):
        X_res = sol.value(X)
        U_res = sol.value(U)
        X_guess = np.hstack(
            (
                X_res[:, 1:],
                X_res[:, -1:][:, np.newaxis :],
            )
        )
        X_guess = np.hstack(
            (
                X_res[:, 1:],
                X_res[:, -1:],
            )
        )
        U_guess = np.append(U_res[1:], U_res[-1])
        opti.set_initial(X, X_guess)
        opti.set_initial(U, U_guess)
        u_applied = U_res[0]
        k1 = f_ode(state, u_applied)
        k2 = f_ode(state + 2.5e-2 * k1, u_applied)
        k3 = f_ode(state + 2.5e-2 * k2, u_applied)
        k4 = f_ode(state + 5.0e-2 * k3, u_applied)
        state = state + np.array(
            np.squeeze((5.0e-2 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        )
        opti.set_value(current_x, state)
        t0 = time.time()
        sol = opti.solve()
        times.append(time.time() - t0)
    avg_time = 1.0e3 * np.mean(times)
    max_time = 1.0e3 * np.max(times)
    min_time = 1.0e3 * np.min(times)
    print(f"MPC Loop (Warm Start) - {n_steps} steps:")
    print(f"  Avg: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    return (
        t_cold,
        avg_time,
    )


opti_no_jit, X_no, U_no, c_no, t_no, f_ode_no = setup_mpc(False)
run_benchmark(opti_no_jit, X_no, U_no, c_no, t_no, f_ode_no, "Python (No JIT/C-Code)")
opti_jit, X_jit, U_jit, c_jit, t_jit, f_ode_jit = setup_mpc(True)
run_benchmark(opti_jit, X_jit, U_jit, c_jit, t_jit, f_ode_jit, "C-Code Export (JIT)")
print("nBenchmark beendet.")
