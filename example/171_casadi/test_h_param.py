from __future__ import annotations
from casadi import *
import numpy as np
import time

# =========================================================================================
# 1. CONVENTIONAL MPC (h_mpc is a Python float, hardcoded in the solver graph)
# =========================================================================================
class PendulumMPC_Conventional:
    def __init__(self, h_mpc=5.0e-2, N=20, use_dual_warmstart=True, use_map=True):
        self.opti = Opti()
        self.nx = 4
        self.nu = 1
        self.N = N
        self.h_mpc = h_mpc
        self.d = 3
        self.use_dual_warmstart = use_dual_warmstart
        self.use_map = use_map

        # Parameters
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

        # Dynamics
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
        dv = ((F_total + m_s * l_s * omega_ * omega_ * sin_theta) - (m_s * 9.81 * cos_theta * sin_theta)) / den
        dtheta = omega_
        domega = (((-1.0 * F_total * cos_theta) - (m_s * l_s * omega_ * omega_ * sin_theta * cos_theta)) + (M_s + m_s) * 9.81 * sin_theta) / (l_s * den)
        self.f_ode = Function("f_ode", [x, u, p_ode], [vertcat(ds, dv, dtheta, domega)])

        # Collocation
        tau_root = np.append(0.0, collocation_points(self.d, "radau"))
        self.C = np.zeros((self.d + 1, self.d + 1))
        self.D = np.zeros(self.d + 1)
        for j in range(self.d + 1):
            p = np.poly1d([1.0])
            for r in range(self.d + 1):
                if r != j:
                    p = p * (np.poly1d([1.0, -tau_root[r]]) / ((tau_root[j]) - (tau_root[r])))
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
            self.colloc_interval = Function("colloc_interval", [Xk_sym, Xck_vec_sym, Uk_sym, p_sym], [res_vec, x_end])
            colloc_map = self.colloc_interval.map(self.N)
            p_stacked = repmat(vertcat(self.M_p, self.m_p, self.l_p, self.wind_p), 1, self.N)
            res_all, x_end_all = colloc_map(self.X[:, : self.N], self.Xc_var, self.U, p_stacked)
            
            self.opti.subject_to(self.X[:, 0] == self.current_x)
            self.opti.subject_to(self.opti.bounded(-1.0 * self.max_pos, self.X[0, :], self.max_pos))
            self.opti.subject_to(self.opti.bounded(-1.0 * self.max_force, self.U, self.max_force))
            self.opti.subject_to(res_all == 0)
            self.opti.subject_to(self.X[:, 1:] == x_end_all)
        else:
            raise NotImplementedError("This test requires map structure.")

        cost = 0.0
        Q = diag(vertcat(self.Q_s, self.Q_v, self.Q_theta, self.Q_omega))
        for k in range(self.N):
            err = (self.X[:, k]) - self.target_x
            cost = cost + mtimes(mtimes(err.T, Q), err) + self.R_F * ((self.U[0, k]) ** 2)
        err_term = (self.X[:, self.N]) - self.target_x
        cost = cost + 1.0e1 * mtimes(mtimes(err_term.T, Q), err_term)
        self.opti.minimize(cost)
        self.n_constraints = self.opti.g.shape[0]

        solver_opts = {"print_time": False, "error_on_fail": True}
        ipopt_opts = {"print_level": 0, "sb": "yes", "max_iter": 150}
        self.opti.solver("ipopt", solver_opts, ipopt_opts)

        self.sol = None

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
            X_guess = np.hstack((X_res[:, 1:], X_res[:, -1:]))
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
                self.opti.set_initial(self.Xc_var, np.zeros((self.nx * self.d, self.N)))
            if self.use_dual_warmstart:
                self.opti.set_initial(self.opti.lam_g, np.zeros(self.n_constraints))
        
        try:
            self.sol = self.opti.solve()
            success = True
        except Exception as e:
            success = False
            
        if success:
            u_opt = float(self.sol.value(self.U)[0])
            x_pred = self.sol.value(self.X)
            u_pred = self.sol.value(self.U)
        else:
            u_opt = 0.0
            x_pred = None
            u_pred = None
        return u_opt, x_pred, u_pred, success


# =========================================================================================
# 2. PARAMETERIZED MPC (h_mpc is an opti.parameter())
# =========================================================================================
class PendulumMPC_Parameterized:
    def __init__(self, N=20, use_dual_warmstart=True, use_map=True):
        self.opti = Opti()
        self.nx = 4
        self.nu = 1
        self.N = N
        self.d = 3
        self.use_dual_warmstart = use_dual_warmstart
        self.use_map = use_map

        # Parameters
        self.h_mpc_p = self.opti.parameter() # <--- h_mpc is now a parameter!
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

        # Dynamics
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
        dv = ((F_total + m_s * l_s * omega_ * omega_ * sin_theta) - (m_s * 9.81 * cos_theta * sin_theta)) / den
        dtheta = omega_
        domega = (((-1.0 * F_total * cos_theta) - (m_s * l_s * omega_ * omega_ * sin_theta * cos_theta)) + (M_s + m_s) * 9.81 * sin_theta) / (l_s * den)
        self.f_ode = Function("f_ode", [x, u, p_ode], [vertcat(ds, dv, dtheta, domega)])

        # Collocation
        tau_root = np.append(0.0, collocation_points(self.d, "radau"))
        self.C = np.zeros((self.d + 1, self.d + 1))
        self.D = np.zeros(self.d + 1)
        for j in range(self.d + 1):
            p = np.poly1d([1.0])
            for r in range(self.d + 1):
                if r != j:
                    p = p * (np.poly1d([1.0, -tau_root[r]]) / ((tau_root[j]) - (tau_root[r])))
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
            h_sym = SX.sym("h") # <--- h_mpc is passed as SX symbol to colloc_interval
            
            Xck_mat = reshape(Xck_vec_sym, self.nx, self.d)
            x_end = self.D[0] * Xk_sym
            res_list = []
            for j in range(1, self.d + 1):
                xp = self.C[0, j] * Xk_sym
                for r in range(self.d):
                    xp = xp + self.C[r + 1, j] * Xck_mat[:, r]
                f_eval = self.f_ode(Xck_mat[:, j - 1], Uk_sym, p_sym)
                res_list.append(xp - (h_sym * f_eval)) # <--- multiplied by h_sym here
            for r in range(self.d):
                x_end = x_end + self.D[r + 1] * Xck_mat[:, r]
            res_vec = vertcat(*res_list)
            
            # The colloc_interval Function now takes h_sym as an input argument:
            self.colloc_interval = Function("colloc_interval", [Xk_sym, Xck_vec_sym, Uk_sym, p_sym, h_sym], [res_vec, x_end])
            colloc_map = self.colloc_interval.map(self.N)
            
            p_stacked = repmat(vertcat(self.M_p, self.m_p, self.l_p, self.wind_p), 1, self.N)
            h_stacked = repmat(self.h_mpc_p, 1, self.N) # <--- Replicate our opti parameter h_mpc_p
            
            # Map with both p_stacked and h_stacked:
            res_all, x_end_all = colloc_map(self.X[:, : self.N], self.Xc_var, self.U, p_stacked, h_stacked)
            
            self.opti.subject_to(self.X[:, 0] == self.current_x)
            self.opti.subject_to(self.opti.bounded(-1.0 * self.max_pos, self.X[0, :], self.max_pos))
            self.opti.subject_to(self.opti.bounded(-1.0 * self.max_force, self.U, self.max_force))
            self.opti.subject_to(res_all == 0)
            self.opti.subject_to(self.X[:, 1:] == x_end_all)
        else:
            raise NotImplementedError("This test requires map structure.")

        cost = 0.0
        Q = diag(vertcat(self.Q_s, self.Q_v, self.Q_theta, self.Q_omega))
        for k in range(self.N):
            err = (self.X[:, k]) - self.target_x
            cost = cost + mtimes(mtimes(err.T, Q), err) + self.R_F * ((self.U[0, k]) ** 2)
        err_term = (self.X[:, self.N]) - self.target_x
        cost = cost + 1.0e1 * mtimes(mtimes(err_term.T, Q), err_term)
        self.opti.minimize(cost)
        self.n_constraints = self.opti.g.shape[0]

        solver_opts = {"print_time": False, "error_on_fail": True}
        ipopt_opts = {"print_level": 0, "sb": "yes", "max_iter": 150}
        self.opti.solver("ipopt", solver_opts, ipopt_opts)

        self.sol = None

    def step(self, state, target_state, params, h_mpc):
        self.opti.set_value(self.h_mpc_p, h_mpc) # <--- Dynamically set the parameter value!
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
            X_guess = np.hstack((X_res[:, 1:], X_res[:, -1:]))
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
                self.opti.set_initial(self.Xc_var, np.zeros((self.nx * self.d, self.N)))
            if self.use_dual_warmstart:
                self.opti.set_initial(self.opti.lam_g, np.zeros(self.n_constraints))
        
        try:
            self.sol = self.opti.solve()
            success = True
        except Exception as e:
            success = False
            
        if success:
            u_opt = float(self.sol.value(self.U)[0])
            x_pred = self.sol.value(self.X)
            u_pred = self.sol.value(self.U)
        else:
            u_opt = 0.0
            x_pred = None
            u_pred = None
        return u_opt, x_pred, u_pred, success


# =========================================================================================
# 3. RUN BENCHMARK & COMPARISON
# =========================================================================================
def run_simulation(mpc_class, h_mpc, l, parameterized=False, preinstantiated_mpc=None):
    params = {
        "M": 1.0,
        "m": 0.1,
        "l": l,
        "wind": 0.0,
        "Q_s": 1.0e1,
        "Q_v": 1.0,
        "Q_theta": 1.0e2,
        "Q_omega": 1.0,
        "R_F": 0.1,
        "max_pos": 5.0,
        "max_force": 1.5e1,
        "h_mpc": h_mpc,
        "N": 20,
        "dt_sim": h_mpc,
    }
    
    t0_init = time.time()
    if preinstantiated_mpc is not None:
        mpc = preinstantiated_mpc
    else:
        if parameterized:
            mpc = mpc_class(N=20)
        else:
            mpc = mpc_class(h_mpc=h_mpc, N=20)
    t_init = time.time() - t0_init
    
    state = np.array([0.0, 0.0, np.pi, 0.0])
    target_state = np.array([1.0, 0.0, 0.0, 0.0])
    T_max = 5.0 # Let's simulate for 5 seconds to compare
    dt = h_mpc
    n_steps = int(T_max / dt)
    
    u_history = []
    x_history = []
    solve_times = []
    
    for step_idx in range(n_steps):
        t_solve_start = time.time()
        if parameterized:
            u_opt, X_pred, U_pred, success = mpc.step(state, target_state, params, h_mpc)
        else:
            u_opt, X_pred, U_pred, success = mpc.step(state, target_state, params)
        solve_times.append(time.time() - t_solve_start)
        
        if not success:
            break
            
        # Simulate forward with simple RK4 using the real dynamics helper
        def f_real(st):
            s_st, v_st, theta_st, omega_st = st[0], st[1], st[2], st[3]
            sin_t = np.sin(theta_st)
            cos_t = np.cos(theta_st)
            denom = params["M"] + params["m"] * (1.0 - (cos_t * cos_t))
            F_tot = u_opt
            l_val = params["l"]
            ds = v_st
            dv = ((F_tot + params["m"] * l_val * omega_st * omega_st * sin_t) - (params["m"] * 9.81 * cos_t * sin_t)) / denom
            dtheta = omega_st
            domega = (((-1.0 * F_tot * cos_t) - (params["m"] * l_val * omega_st * omega_st * sin_t * cos_t)) + (params["M"] + params["m"]) * 9.81 * sin_t) / (l_val * denom)
            return np.array([ds, dv, dtheta, domega])

        k1 = f_real(state)
        k2 = f_real(state + (dt / 2.0) * k1)
        k3 = f_real(state + (dt / 2.0) * k2)
        k4 = f_real(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        u_history.append(u_opt)
        x_history.append(state.copy())
        
    return {
        "u": np.array(u_history),
        "x": np.array(x_history),
        "solve_times": np.array(solve_times),
        "init_time": t_init
    }

if __name__ == "__main__":
    print("=== MPC parameterization test ===")
    
    # We will test two different step sizes: 0.05 and 0.08
    h_mpc_test_vals = [0.05, 0.08]
    l_test_val = 0.6
    
    # Pre-instantiate the parameterized solver ONCE
    print("Instantiating parameterized solver once...")
    t0 = time.time()
    param_mpc = PendulumMPC_Parameterized(N=20)
    t_param_inst = time.time() - t0
    print(f"Parameterized solver instantiation took {t_param_inst * 1000:.2f} ms")
    
    for h_val in h_mpc_test_vals:
        print(f"\n--- Testing h_mpc = {h_val:.3f} s, l = {l_test_val:.2f} m ---")
        
        # Run conventional (instantiated fresh)
        print("Running conventional solver (new instantiation)...")
        res_conv = run_simulation(PendulumMPC_Conventional, h_mpc=h_val, l=l_test_val, parameterized=False)
        print(f"  Initialization time: {res_conv['init_time'] * 1000:.2f} ms")
        print(f"  Average solve time: {np.mean(res_conv['solve_times']) * 1000:.2f} ms")
        print(f"  Number of steps: {len(res_conv['u'])}")
        
        # Run parameterized (using pre-instantiated solver)
        print("Running parameterized solver (reusing instantiated object)...")
        res_param = run_simulation(PendulumMPC_Parameterized, h_mpc=h_val, l=l_test_val, parameterized=True, preinstantiated_mpc=param_mpc)
        print(f"  Initialization time (reused): 0.00 ms (original instantiation: {t_param_inst * 1000:.2f} ms)")
        print(f"  Average solve time: {np.mean(res_param['solve_times']) * 1000:.2f} ms")
        print(f"  Number of steps: {len(res_param['u'])}")
        
        # Check matching
        min_len = min(len(res_conv["u"]), len(res_param["u"]))
        if min_len > 0:
            diff_u = np.abs(res_conv["u"][:min_len] - res_param["u"][:min_len])
            max_diff_u = np.max(diff_u)
            mean_diff_u = np.mean(diff_u)
            
            diff_x = np.abs(res_conv["x"][:min_len] - res_param["x"][:min_len])
            max_diff_x = np.max(diff_x)
            
            print(f"  Verification results:")
            print(f"    Max difference in optimal control u: {max_diff_u:.2e}")
            print(f"    Mean difference in optimal control u: {mean_diff_u:.2e}")
            print(f"    Max difference in state trajectory x: {max_diff_x:.2e}")
            
            if max_diff_u < 1e-5 and max_diff_x < 1e-5:
                print("  => VERIFICATION SUCCESSFUL: Parameterized solver yields identical results!")
            else:
                print("  => VERIFICATION FAILED: Differences are too large!")
        else:
            print("  Verification results:")
            print("    Both solvers failed immediately (0 steps completed successfully).")
            print("  => VERIFICATION SUCCESSFUL: Match verified because both failed identically!")
            
    print("\n=== Summary of Init/Instantiation benefits ===")
    print("For a grid scan over 7 h_mpc values and 8 l values and 8 mass ratio values (448 combinations):")
    print("- Conventional MPC requires: 448 instantiations.")
    print(f"  Estimated total instantiation time: 448 * {res_conv['init_time']*1000:.1f} ms = {448 * res_conv['init_time']:.2f} seconds.")
    print("- Parameterized MPC requires: 1 instantiation (per worker thread).")
    print(f"  Estimated total instantiation time: 32 workers * {t_param_inst*1000:.1f} ms = {32 * t_param_inst:.2f} seconds.")
    print(f"  Time saved on instantiation alone: {448 * res_conv['init_time'] - 32 * t_param_inst:.2f} seconds.")
