from __future__ import annotations
import argparse
from casadi import *
import numpy as np
import time
import os
import csv
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================================================
#  INVERTED PENDULUM MPC (PARAMETERIZED FOR STABILITY & SWEEP SPEED)
# =========================================================================================
class PendulumMPC_Parameterized:
    def __init__(self, N=20, use_dual_warmstart=True, use_map=True, use_jit=True):
        self.opti = Opti()
        self.nx = 4
        self.nu = 1
        self.N = N
        self.d = 3
        self.use_dual_warmstart = use_dual_warmstart
        self.use_map = use_map
        self.use_jit = use_jit

        # Parameters
        self.h_mpc_p = self.opti.parameter()
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

        # Collocation setup
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
            h_sym = SX.sym("h")
            
            Xck_mat = reshape(Xck_vec_sym, self.nx, self.d)
            x_end = self.D[0] * Xk_sym
            res_list = []
            for j in range(1, self.d + 1):
                xp = self.C[0, j] * Xk_sym
                for r in range(self.d):
                    xp = xp + self.C[r + 1, j] * Xck_mat[:, r]
                f_eval = self.f_ode(Xck_mat[:, j - 1], Uk_sym, p_sym)
                res_list.append(xp - (h_sym * f_eval))
            for r in range(self.d):
                x_end = x_end + self.D[r + 1] * Xck_mat[:, r]
            res_vec = vertcat(*res_list)
            
            self.colloc_interval = Function("colloc_interval", [Xk_sym, Xck_vec_sym, Uk_sym, p_sym, h_sym], [res_vec, x_end])
            colloc_map = self.colloc_interval.map(self.N)
            
            p_stacked = repmat(vertcat(self.M_p, self.m_p, self.l_p, self.wind_p), 1, self.N)
            h_stacked = repmat(self.h_mpc_p, 1, self.N)
            
            res_all, x_end_all = colloc_map(self.X[:, : self.N], self.Xc_var, self.U, p_stacked, h_stacked)
            
            self.opti.subject_to(self.X[:, 0] == self.current_x)
            self.opti.subject_to(self.opti.bounded(-1.0 * self.max_pos, self.X[0, :], self.max_pos))
            self.opti.subject_to(self.opti.bounded(-1.0 * self.max_force, self.U, self.max_force))
            self.opti.subject_to(res_all == 0)
            self.opti.subject_to(self.X[:, 1:] == x_end_all)
        else:
            raise NotImplementedError("This script requires constraint mapping.")

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
        if self.use_jit:
            solver_opts["jit"] = True
            solver_opts["compiler"] = "shell"
            solver_opts["jit_options"] = {"flags": ["-O3"]}
        self.opti.solver("ipopt", solver_opts, ipopt_opts)

        self.sol = None

    def step(self, state, target_state, params, h_mpc):
        self.opti.set_value(self.h_mpc_p, h_mpc)
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
        
        t0 = time.time()
        try:
            self.sol = self.opti.solve()
            success = True
        except Exception as e:
            success = False
        t_solve = time.time() - t0
            
        if success:
            u_opt = float(self.sol.value(self.U)[0])
            x_pred = self.sol.value(self.X)
            u_pred = self.sol.value(self.U)
        else:
            u_opt = 0.0
            x_pred = None
            u_pred = None
        return u_opt, x_pred, u_pred, t_solve, success


# =========================================================================================
#  MULTIPROCESSING WORKER CODE
# =========================================================================================
global_mpc = None

def init_worker(N, use_jit):
    global global_mpc
    global_mpc = PendulumMPC_Parameterized(N=N, use_jit=use_jit)

def run_simulation_task(h_val, l_val, mu_val, N, max_force_val):
    global global_mpc
    
    try:
        # Physical parameters
        params = {
            "M": 1.0,
            "m": mu_val, # m = mu * M where M = 1.0
            "l": l_val,
            "wind": 0.0,
            "Q_s": 1.0e1,
            "Q_v": 1.0,
            "Q_theta": 1.0e2,
            "Q_omega": 1.0,
            "R_F": 0.1,
            "max_pos": 5.0,
            "max_force": max_force_val,
            "h_mpc": h_val,
            "N": N,
            "dt_sim": h_val,
        }
        
        state = np.array([0.0, 0.0, np.pi, 0.0])
        target_state = np.array([1.0, 0.0, 0.0, 0.0])
        T_max = 12.0
        dt = h_val
        n_steps = int(T_max / dt)
        
        u_hist = []
        x_hist = []
        t_solve_hist = []
        success_sim = True
        
        # Reset warm start solution cache for fresh simulation
        global_mpc.sol = None
        
        for step_idx in range(n_steps):
            u_opt, X_pred, U_pred, t_solve, success = global_mpc.step(state, target_state, params, h_val)
            t_solve_hist.append(t_solve * 1000.0) # convert to ms
            
            if not success:
                success_sim = False
                break
                
            def f_real(st):
                s_st = st[0]
                v_st = st[1]
                theta_st = st[2]
                omega_st = st[3]
                sin_t = np.sin(theta_st)
                cos_t = np.cos(theta_st)
                denom = params["M"] + params["m"] * (1.0 - (cos_t * cos_t))
                F_tot = u_opt
                l_val_param = params["l"]
                ds = v_st
                dv = ((F_tot + params["m"] * l_val_param * omega_st * omega_st * sin_t) - (params["m"] * 9.81 * cos_t * sin_t)) / denom
                dtheta = omega_st
                domega = (((-1.0 * F_tot * cos_t) - (params["m"] * l_val_param * omega_st * omega_st * sin_t * cos_t)) + (params["M"] + params["m"]) * 9.81 * sin_t) / (l_val_param * denom)
                return np.array([ds, dv, dtheta, domega])

            k1 = f_real(state)
            k2 = f_real(state + (dt / 2.0) * k1)
            k3 = f_real(state + (dt / 2.0) * k2)
            k4 = f_real(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            # Check track boundary violation
            if np.abs(state[0]) > params["max_pos"]:
                success_sim = False
                break
                
            u_hist.append(u_opt)
            x_hist.append(state.copy())
            
        # Analyze stabilization
        if not success_sim or len(x_hist) < n_steps:
            t_stable = np.nan
            success_sim = False
        else:
            s_tol = 5.0e-2
            v_tol = 5.0e-2
            theta_tol = 5.0e-2
            omega_tol = 5.0e-2
            stabilization_step = -1
            for i in range(len(x_hist) - 1, -1, -1):
                s_val = x_hist[i][0]
                v_val = x_hist[i][1]
                theta_val = x_hist[i][2]
                omega_val = x_hist[i][3]
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
                t_stable = 0.0
            elif stabilization_step >= len(x_hist):
                t_stable = np.nan
            else:
                t_stable = (stabilization_step + 1) * dt
                
        # Calculate metrics
        if len(u_hist) > 0:
            avg_solve = np.mean(t_solve_hist)
            max_solve = np.max(t_solve_hist)
            control_energy = float(np.sum(np.array(u_hist)**2 * dt))
            max_control_force = float(np.max(np.abs(u_hist)))
            
            # IAE
            iae_pos = float(np.sum(np.abs(np.array([x[0] for x in x_hist]) - 1.0) * dt))
            wrapped_thetas = np.array([((x[2] + np.pi) % (2.0 * np.pi)) - np.pi for x in x_hist])
            iae_angle = float(np.sum(np.abs(wrapped_thetas) * dt))
        else:
            avg_solve = np.nan
            max_solve = np.nan
            control_energy = np.nan
            max_control_force = np.nan
            iae_pos = np.nan
            iae_angle = np.nan
            
        return {
            "h": h_val,
            "l": l_val,
            "mu": mu_val,
            "stabilization_time": t_stable,
            "avg_solve_time_ms": avg_solve,
            "max_solve_time_ms": max_solve,
            "control_energy": control_energy,
            "max_control_force": max_control_force,
            "iae_position": iae_pos,
            "iae_angle": iae_angle,
            "success": bool(success_sim and not np.isnan(t_stable))
        }
    except Exception as e:
        return {
            "h": h_val,
            "l": l_val,
            "mu": mu_val,
            "stabilization_time": np.nan,
            "avg_solve_time_ms": np.nan,
            "max_solve_time_ms": np.nan,
            "control_energy": np.nan,
            "max_control_force": np.nan,
            "iae_position": np.nan,
            "iae_angle": np.nan,
            "success": False
        }

# =========================================================================================
#  MAIN ENTRYPOINT & COMMAND LINE PARSING
# =========================================================================================
def parse_range_or_list(range_str, list_str, default_range):
    if list_str:
        return sorted([float(x.strip()) for x in list_str.split(",")])
    
    r_str = range_str if range_str else default_range
    parts = [float(x.strip()) for x in r_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Range string must be 'min,max,count', got: {r_str}")
    return sorted(list(np.linspace(parts[0], parts[1], int(parts[2]))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dimensional MPC Grid Sweep (JIT-optimized via multiprocessing)")
    
    # Core solver CLI settings
    parser.add_argument("-N", "--horizon", type=int, default=20, help="MPC prediction horizon (N)")
    parser.add_argument("--max-force", type=float, default=15.0, help="Maximum controller actuator force limit")
    parser.add_argument("--threads", type=int, default=None, help="Number of parallel worker processes (default: CPU count)")
    
    # Sweep Grid parameters
    parser.add_argument("--h-range", type=str, default="0.02,0.10,7", help="Range for h_mpc: 'min,max,count'")
    parser.add_argument("--h-vals", type=str, default=None, help="Explicit values for h_mpc (comma separated, overrides h-range)")
    
    parser.add_argument("--l-range", type=str, default="0.2,2.0,8", help="Range for length l: 'min,max,count'")
    parser.add_argument("--l-vals", type=str, default=None, help="Explicit values for length l (comma separated, overrides l-range)")
    
    parser.add_argument("--mu-range", type=str, default="0.05,0.50,5", help="Range for mass ratio m/M: 'min,max,count'")
    parser.add_argument("--mu-vals", type=str, default=None, help="Explicit values for mass ratio m/M (comma separated, overrides mu-range)")
    
    args = parser.parse_args()
    
    # Resolve sweep coordinates
    h_vals = parse_range_or_list(args.h_range, args.h_vals, "0.02,0.10,7")
    l_vals = parse_range_or_list(args.l_range, args.l_vals, "0.2,2.0,8")
    mu_vals = parse_range_or_list(args.mu_range, args.mu_vals, "0.05,0.50,5")
    
    num_workers = args.threads if args.threads else os.cpu_count()
    if num_workers is None:
        num_workers = 4
        
    has_gcc = shutil.which("gcc") is not None
    use_jit_flag = has_gcc
        
    print("=" * 70)
    print(" MPC MULTI-DIMENSIONAL GRID SWEEP (PROCESS-PARALLEL)")
    print("=" * 70)
    print(f"Prediction Horizon N: {args.horizon}")
    print(f"Max Control Force:    {args.max_force:.2f} N")
    print(f"Worker Processes:     {num_workers}")
    print(f"JIT Compilation:      {'Enabled' if use_jit_flag else 'Disabled (gcc not found)'}")
    print(f"Grid Dimensions:")
    print(f"  h_mpc (step size):  {len(h_vals)} values in {h_vals[0]:.3f}s - {h_vals[-1]:.3f}s")
    print(f"  l (pendulum len):   {len(l_vals)} values in {l_vals[0]:.2f}m - {l_vals[-1]:.2f}m")
    print(f"  m/M (mass ratio):   {len(mu_vals)} values in {mu_vals[0]:.2f} - {mu_vals[-1]:.2f}")
    total_sims = len(h_vals) * len(l_vals) * len(mu_vals)
    print(f"Total Simulations:    {total_sims}")
    print("=" * 70)
    
    # Generate tasks list
    tasks = []
    for mu in mu_vals:
        for h in h_vals:
            for l in l_vals:
                tasks.append((h, l, mu))
                
    # Warm-up JIT compilation on main process to seed the compiler cache
    if use_jit_flag:
        print("Warm-up: Compiling JIT solver once in main process to seed cache...")
        warmup_mpc = PendulumMPC_Parameterized(N=args.horizon, use_jit=True)
        warmup_params = {
            "M": 1.0, "m": 0.1, "l": 1.0, "wind": 0.0,
            "Q_s": 1.0e1, "Q_v": 1.0, "Q_theta": 1.0e2, "Q_omega": 1.0, "R_F": 0.1,
            "max_pos": 5.0, "max_force": args.max_force, "h_mpc": 0.05, "N": args.horizon, "dt_sim": 0.05
        }
        warmup_mpc.step(np.array([0.0, 0.0, np.pi, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), warmup_params, 0.05)
        print("Warm-up complete (JIT compilation cached!). Spawning processes...")

    t_start = time.time()
    results = []
    
    # Run the sweep in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(args.horizon, use_jit_flag)) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_simulation_task, h, l, mu, args.horizon, args.max_force): (h, l, mu)
            for h, l, mu in tasks
        }
        
        # Track progress using tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sweeping grid"):
            results.append(future.result())
        
    t_total = time.time() - t_start
    print(f"\nAll simulations complete in {t_total:.2f} seconds ({t_total/total_sims:.3f} s/simulation average).")
    
    # Save results to a timestamped CSV
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"p11i_grid_results_{dt_str}.csv"
    
    print(f"Saving raw numerical results to {csv_filename}...")
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "h", "l", "mu", "success", "stabilization_time",
            "avg_solve_time_ms", "max_solve_time_ms", "control_energy",
            "max_control_force", "iae_position", "iae_angle"
        ])
        writer.writeheader()
        for res in results:
            writer.writerow(res)
            
    # Visualize results (subplots of heatmaps per mass ratio)
    print("Generating visualization plots...")
    n_subplots = len(mu_vals)
    n_cols = min(3, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.0 * n_rows), squeeze=False)
    
    res_dict = {(res["h"], res["l"], res["mu"]): res for res in results}
    
    for idx, mu in enumerate(mu_vals):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        grid_data = np.zeros((len(h_vals), len(l_vals)))
        for i, h in enumerate(h_vals):
            for j, l in enumerate(l_vals):
                val = res_dict[(h, l, mu)]["stabilization_time"]
                grid_data[i, j] = val
                
        cax = ax.imshow(grid_data, interpolation="nearest", cmap="viridis", origin="lower")
        ax.set_title(f"Mass ratio m/M = {mu:.2f}")
        
        ax.set_xticks(np.arange(len(l_vals)))
        ax.set_xticklabels([f"{l:.2f}" for l in l_vals], rotation=45)
        ax.set_yticks(np.arange(len(h_vals)))
        ax.set_yticklabels([f"{h:.3f}" for h in h_vals])
        
        ax.set_xlabel("Pendulum length l [m]")
        ax.set_ylabel("Step size h_mpc [s]")
        
        # Annotate cells
        for i in range(len(h_vals)):
            for j in range(len(l_vals)):
                val = grid_data[i, j]
                text_val = "NaN" if np.isnan(val) else f"{val:.2f}"
                ax.text(j, i, text_val, ha="center", va="center", 
                        color="w" if np.isnan(val) or val > (12.0/2) else "black")
                
        fig.colorbar(cax, ax=ax, label="Stabilization Time [s]")
        
    for idx in range(n_subplots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
        
    plt.suptitle(f"MPC Sweep: Stabilization Time (N={args.horizon}, max_force={args.max_force:.1f}N)", y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_filename = f"p11i_grid_heatmaps_{dt_str}.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Heatmap visualization saved to {plot_filename}.")
    print("Sweep complete!")
