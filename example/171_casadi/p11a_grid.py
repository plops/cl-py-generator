from __future__ import annotations
from casadi import *
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


# =========================================================================================
#  INVERTED PENDULUM MPC GRID SEARCH (HEADLESS - JIT-free)
# =========================================================================================
class PendulumMPC:
    def __init__(self, h_mpc=5.0e-2, N=20, use_dual_warmstart=True, use_map=True):
        self.opti = Opti()
        self.nx = 4
        self.nu = 1
        self.N = N
        self.h_mpc = h_mpc
        self.T_horizon = self.N * self.h_mpc
        self.d = 3
        self.use_dual_warmstart = use_dual_warmstart
        self.use_map = use_map
        self.M_p = self.opti.parameter()
        self.m_p = self.opti.parameter()
        self.l_p = self.opti.parameter()
        self.wind_p = self.opti.parameter()
        self.Q_s = self.opti.parameter()
        self.Q_v = self.opti.parameter()
        self.Q_theta = self.opti.parameter()
        self.Q_omega = self.opti.parameter()
        self.R_F = self.opti.parameter()
        self.max_pos = self.opti.parameter()
        self.max_force = self.opti.parameter()
        x = SX.sym("x", self.nx)
        u = SX.sym("u", self.nu)
        p_ode = SX.sym("p_ode", 4)
        s_ = x[0]
        v_ = x[1]
        theta_ = x[2]
        omega_ = x[3]
        F_ = u
        M_s = p_ode[0]
        m_s = p_ode[1]
        l_s = p_ode[2]
        wind_s = p_ode[3]
        sin_theta = np.sin(theta_)
        cos_theta = np.cos(theta_)
        den = M_s + m_s * (1.0 - (cos_theta * cos_theta))
        F_total = F_ + wind_s * cos_theta
        ds = v_
        dv = (
            (F_total + m_s * l_s * omega_ * omega_ * sin_theta)
            - (m_s * 9.81 * cos_theta * sin_theta)
        ) / den
        dtheta = omega_
        domega = (
            (
                (-1.0 * F_total * cos_theta)
                - (m_s * l_s * omega_ * omega_ * sin_theta * cos_theta)
            )
            + (M_s + m_s) * 9.81 * sin_theta
        ) / (l_s * den)
        self.f_ode = Function("f_ode", [x, u, p_ode], [vertcat(ds, dv, dtheta, domega)])
        tau_root = np.append(0.0, collocation_points(self.d, "radau"))
        self.C = np.zeros(
            (
                self.d + 1,
                self.d + 1,
            )
        )
        self.D = np.zeros(self.d + 1)
        for j in range(self.d + 1):
            p = np.poly1d([1.0])
            for r in range(self.d + 1):
                if r != j:
                    p = p * (
                        np.poly1d([1.0, -tau_root[r]]) / ((tau_root[j]) - (tau_root[r]))
                    )
            self.D[j] = p(1.0)
            pder = np.polyder(p)
            for r in range(self.d + 1):
                self.C[j, r] = pder(tau_root[r])
        if self.use_map:
            self.X = self.opti.variable(self.nx, self.N + 1)
            self.Xc_var = self.opti.variable(self.nx * self.d, self.N)
            self.U = self.opti.variable(self.nu, self.N)
            self.current_x = self.opti.parameter(self.nx)
            self.target_x = self.opti.parameter(self.nx)
            Xk_sym = SX.sym("Xk", self.nx)
            Xck_vec_sym = SX.sym("Xck_vec", self.nx * self.d)
            Uk_sym = SX.sym("Uk", self.nu)
            p_sym = SX.sym("p", 4)
            Xck_mat = reshape(Xck_vec_sym, self.nx, self.d)
            x_end = self.D[0] * Xk_sym
            res_list = []
            for j in range(1, self.d + 1):
                xp = self.C[0, j] * Xk_sym
                for r in range(self.d):
                    xp = xp + self.C[r + 1, j] * Xck_mat[:, r]
                f_eval = self.f_ode(Xck_mat[:, j - 1], Uk_sym, p_sym)
                res_list.append(xp - (self.h_mpc * f_eval))
            for r in range(self.d):
                x_end = x_end + self.D[r + 1] * Xck_mat[:, r]
            res_vec = vertcat(*res_list)
            self.colloc_interval = Function(
                "colloc_interval",
                [Xk_sym, Xck_vec_sym, Uk_sym, p_sym],
                [res_vec, x_end],
            )
            colloc_map = self.colloc_interval.map(self.N)
            p_stacked = repmat(
                vertcat(self.M_p, self.m_p, self.l_p, self.wind_p), 1, self.N
            )
            res_all, x_end_all = colloc_map(
                self.X[:, : self.N], self.Xc_var, self.U, p_stacked
            )
            self.opti.subject_to(self.X[:, 0] == self.current_x)
            self.opti.subject_to(
                self.opti.bounded(-1.0 * self.max_pos, self.X[0, :], self.max_pos)
            )
            self.opti.subject_to(
                self.opti.bounded(-1.0 * self.max_force, self.U, self.max_force)
            )
            self.opti.subject_to(res_all == 0)
            self.opti.subject_to(self.X[:, 1:] == x_end_all)
        else:
            self.X = self.opti.variable(self.nx, self.N + 1)
            self.Xc = []
            self.U = self.opti.variable(self.nu, self.N)
            self.current_x = self.opti.parameter(self.nx)
            self.target_x = self.opti.parameter(self.nx)
            for k in range(self.N):
                Xc_k = []
                for r in range(self.d):
                    Xc_k.append(self.opti.variable(self.nx))
                self.Xc.append(Xc_k)
            self.opti.subject_to(self.X[:, 0] == self.current_x)
            self.opti.subject_to(
                self.opti.bounded(-1.0 * self.max_pos, self.X[0, :], self.max_pos)
            )
            self.opti.subject_to(
                self.opti.bounded(-1.0 * self.max_force, self.U, self.max_force)
            )
            for k in range(self.N):
                Xk = self.X[:, k]
                x_end = self.D[0] * Xk
                for j in range(1, self.d + 1):
                    xp = self.C[0, j] * Xk
                    for r in range(self.d):
                        xp = xp + self.C[r + 1, j] * self.Xc[k][r]
                    f_eval = self.f_ode(
                        self.Xc[k][j - 1],
                        self.U[:, k],
                        vertcat(self.M_p, self.m_p, self.l_p, self.wind_p),
                    )
                    self.opti.subject_to(xp == self.h_mpc * f_eval)
                for r in range(self.d):
                    x_end = x_end + self.D[r + 1] * self.Xc[k][r]
                self.opti.subject_to(self.X[:, k + 1] == x_end)
        cost = 0.0
        Q = diag(vertcat(self.Q_s, self.Q_v, self.Q_theta, self.Q_omega))
        for k in range(self.N):
            err = (self.X[:, k]) - self.target_x
            cost = (
                cost + mtimes(mtimes(err.T, Q), err) + self.R_F * ((self.U[0, k]) ** 2)
            )
        err_term = (self.X[:, self.N]) - self.target_x
        cost = cost + 1.0e1 * mtimes(mtimes(err_term.T, Q), err_term)
        self.opti.minimize(cost)
        self.n_constraints = self.opti.g.shape[0]
        solver_opts = {("print_time"): (False), ("error_on_fail"): (True)}
        ipopt_opts = {("print_level"): (0), ("sb"): ("yes"), ("max_iter"): (150)}
        self.opti.solver("ipopt", solver_opts, ipopt_opts)
        self.opti.set_initial(
            self.X,
            np.linspace(
                np.array([0.0, 0.0, np.pi, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                self.N + 1,
            ).T,
        )
        self.opti.set_initial(
            self.U,
            np.zeros(
                (
                    self.nu,
                    self.N,
                )
            ),
        )
        if self.use_map:
            self.opti.set_initial(
                self.Xc_var,
                np.zeros(
                    (
                        self.nx * self.d,
                        self.N,
                    )
                ),
            )
        self.sol = None
        self.last_X = np.zeros(
            (
                self.nx,
                self.N + 1,
            )
        )
        self.last_U = np.zeros(
            (
                self.nu,
                self.N,
            )
        )
        self.last_lam_g = None

    def step(self, state, target_state, params):
        self.opti.set_value(self.current_x, state)
        self.opti.set_value(self.target_x, target_state)
        self.opti.set_value(self.M_p, params["M"])
        self.opti.set_value(self.m_p, params["m"])
        self.opti.set_value(self.l_p, params["l"])
        self.opti.set_value(self.wind_p, params["wind"])
        self.opti.set_value(self.Q_s, params["Q_s"])
        self.opti.set_value(self.Q_v, params["Q_v"])
        self.opti.set_value(self.Q_theta, params["Q_theta"])
        self.opti.set_value(self.Q_omega, params["Q_omega"])
        self.opti.set_value(self.R_F, params["R_F"])
        self.opti.set_value(self.max_pos, params["max_pos"])
        self.opti.set_value(self.max_force, params["max_force"])
        if self.sol is not None:
            X_res = self.sol.value(self.X)
            U_res = self.sol.value(self.U)
            X_guess = np.hstack(
                (
                    X_res[:, 1:],
                    X_res[:, -1:],
                )
            )
            U_guess = np.append(U_res[1:], U_res[-1])
            self.opti.set_initial(self.X, X_guess)
            self.opti.set_initial(self.U, U_guess[np.newaxis, :])
            if self.use_map:
                self.opti.set_initial(self.Xc_var, self.sol.value(self.Xc_var))
            if self.use_dual_warmstart:
                lam_g_res = self.sol.value(self.opti.lam_g)
                self.opti.set_initial(self.opti.lam_g, lam_g_res)
        else:
            X_guess = np.linspace(state, target_state, self.N + 1).T
            U_guess = np.zeros(self.N)
            self.opti.set_initial(self.X, X_guess)
            self.opti.set_initial(self.U, U_guess[np.newaxis, :])
            if self.use_map:
                self.opti.set_initial(
                    self.Xc_var,
                    np.zeros(
                        (
                            self.nx * self.d,
                            self.N,
                        )
                    ),
                )
            if self.use_dual_warmstart:
                self.opti.set_initial(self.opti.lam_g, np.zeros(self.n_constraints))
        t0 = time.time()
        success = True
        try:
            self.sol = self.opti.solve()
        except Exception as e:
            success = False
        t_solve = time.time() - t0
        if success:
            self.last_X = self.sol.value(self.X)
            self.last_U = np.squeeze(self.sol.value(self.U))
            u_opt = self.last_U[0]
            if self.use_map:
                self.last_Xc_var = self.sol.value(self.Xc_var)
        else:
            self.last_X = np.hstack(
                (
                    self.last_X[:, 1:],
                    self.last_X[:, -1:],
                )
            )
            self.last_U = np.append(self.last_U[1:], self.last_U[-1])
            u_opt = self.last_U[0]
        return (
            float(u_opt),
            self.last_X,
            self.last_U,
            t_solve,
            success,
        )


def run_simulation(h_mpc, l):
    params = {
        ("M"): (1.0),
        ("m"): (0.1),
        ("l"): (l),
        ("wind"): (0.0),
        ("Q_s"): (1.0e1),
        ("Q_v"): (1.0),
        ("Q_theta"): (1.0e2),
        ("Q_omega"): (1.0),
        ("R_F"): (0.1),
        ("max_pos"): (5.0),
        ("max_force"): (1.5e1),
        ("h_mpc"): (h_mpc),
        ("N"): (20),
        ("dt_sim"): (h_mpc),
    }
    mpc = PendulumMPC(h_mpc=h_mpc, N=20, use_map=True)
    state = np.array([0.0, 0.0, np.pi, 0.0])
    target_state = np.array([1.0, 0.0, 0.0, 0.0])
    T_max = 1.2e1
    dt = h_mpc
    n_steps = int(T_max / dt)
    s_hist = []
    v_hist = []
    theta_hist = []
    omega_hist = []
    t_hist = []
    for step_idx in range(n_steps):
        u_opt, X_pred, U_pred, t_solve, success = mpc.step(state, target_state, params)

        def f_real(st):
            s_st = st[0]
            v_st = st[1]
            theta_st = st[2]
            omega_st = st[3]
            sin_t = np.sin(theta_st)
            cos_t = np.cos(theta_st)
            denom = params["M"] + params["m"] * (1.0 - (cos_t * cos_t))
            F_tot = u_opt
            l_val = params["l"]
            ds = v_st
            dv = (
                (F_tot + params["m"] * l_val * omega_st * omega_st * sin_t)
                - (params["m"] * 9.81 * cos_t * sin_t)
            ) / denom
            dtheta = omega_st
            domega = (
                (
                    (-1.0 * F_tot * cos_t)
                    - (params["m"] * l_val * omega_st * omega_st * sin_t * cos_t)
                )
                + (params["M"] + params["m"]) * 9.81 * sin_t
            ) / (l_val * denom)
            return np.array([ds, dv, dtheta, domega])

        k1 = f_real(state)
        k2 = f_real(state + (dt / 2.0) * k1)
        k3 = f_real(state + (dt / 2.0) * k2)
        k4 = f_real(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s_hist.append(state[0])
        v_hist.append(state[1])
        theta_hist.append(state[2])
        omega_hist.append(state[3])
        t_hist.append((step_idx + 1) * dt)
    if len(s_hist) < n_steps:
        return np.nan
    s_tol = 5.0e-2
    v_tol = 5.0e-2
    theta_tol = 5.0e-2
    omega_tol = 5.0e-2
    stabilization_step = -1
    for i in range(len(s_hist) - 1, -1, -1):
        s_val = s_hist[i]
        v_val = v_hist[i]
        theta_val = theta_hist[i]
        omega_val = omega_hist[i]
        theta_wrapped = ((theta_val + np.pi) % (2.0 * np.pi)) - np.pi
        outside = (
            np.abs(s_val - 1.0) > s_tol
            or np.abs(v_val) > v_tol
            or np.abs(theta_wrapped) > theta_tol
            or np.abs(omega_val) > omega_tol
        )
        if outside:
            stabilization_step = i + 1
            break
    if stabilization_step == -1:
        return 0.0
    else:
        if stabilization_step >= len(s_hist):
            return np.nan
        else:
            return t_hist[stabilization_step]


if __name__ == "__main__":
    print("Starting Inverted Pendulum MPC Grid Search...")
    h_mpc_vals = [2.0e-2, 3.0e-2, 4.0e-2, 5.0e-2, 6.0e-2, 8.0e-2, 0.1]
    l_vals = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    grid_results = np.zeros(
        (
            len(h_mpc_vals),
            len(l_vals),
        )
    )
    for i, h_val in enumerate(h_mpc_vals):
        for j, l_val in enumerate(l_vals):
            print(f"Simulating h_mpc={h_val:.2f}s, l={l_val:.2f}m...")
            t_stable = run_simulation(h_val, l_val)
            grid_results[i, j] = t_stable
            print(f"  Stabilization time: {t_stable:.3f}s")
    print("\nGrid Search Results (Stabilization Time in seconds):")
    header = "h_mpc \ l   "
    for l_val in l_vals:
        header = header + f"{l_val:8.2f}"
    print(header)
    print("-" * len(header))
    for i, h_val in enumerate(h_mpc_vals):
        line = f"{h_val:10.3f} |"
        for j, l_val in enumerate(l_vals):
            val = grid_results[i, j]
            if np.isnan(val):
                line = line + "     nan"
            else:
                line = line + f"{val:8.2f}"
        print(line)
    # Plot the heatmap
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            8,
            6,
        ),
    )
    cax = ax.imshow(
        grid_results, interpolation="nearest", cmap="viridis", origin="lower"
    )
    fig.colorbar(cax, label="Stabilisierungszeit [s]")
    ax.set_xticks(np.arange(len(l_vals)))
    ax.set_xticklabels([f"{l:.2f}" for l in l_vals])
    ax.set_yticks(np.arange(len(h_mpc_vals)))
    ax.set_yticklabels([f"{h:.2f}" for h in h_mpc_vals])
    ax.set_xlabel("Pendellaenge l [m]")
    ax.set_ylabel("Schrittweite h_mpc [s]")
    ax.set_title("MPC Grid Search: Stabilisierungszeit des Pendels")
    # Annotate text on the heatmap cells
    for i in range(len(h_mpc_vals)):
        for j in range(len(l_vals)):
            val = grid_results[i, j]
            text_val = "NaN" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text_val, ha="center", va="center", color="w")
    plt.tight_layout()
    plt.savefig("p11a_grid_heatmap.png")
    print("Heatmap saved as p11a_grid_heatmap.png")
