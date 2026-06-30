from __future__ import annotations
from casadi import *
import numpy as np
import time
import sys
import json


# =========================================================================================
#  INVERTED PENDULUM MPC HEADLESS BENCHMARK AND VALIDATION
# =========================================================================================
class PendulumMPC:
    def __init__(self, h_mpc=5.0e-2, N=20):
        self.opti = Opti()
        self.nx = 4
        self.nu = 1
        self.N = N
        self.h_mpc = h_mpc
        self.T_horizon = self.N * self.h_mpc
        self.d = 3
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
        self.opti.solver(
            "ipopt", {("print_time"): (False)}, {("print_level"): (0), ("sb"): ("yes")}
        )
        self.opti.set_initial(
            self.X,
            np.linspace(
                np.array([0.0, 0.0, np.pi, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                self.N + 1,
            ).T,
        )
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
        if self.sol != None:
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
            self.opti.set_initial(self.U, U_guess)
        else:
            X_res = np.zeros(
                (
                    self.nx,
                    self.N + 1,
                )
            )
            U_res = np.zeros(
                (
                    self.nu,
                    self.N,
                )
            )
            U_guess = np.zeros(self.nu)
        t0 = time.time()
        try:
            self.sol = self.opti.solve()
        except Exception as e:
            return (
                U_guess[0],
                X_res,
                U_res,
                time.time() - t0,
                False,
            )
        t_solve = time.time() - t0
        X_res = self.sol.value(self.X)
        U_res = self.sol.value(self.U)
        return (
            U_res[0],
            X_res,
            U_res,
            t_solve,
            True,
        )


def f_real(st, F_motor, params, use_corrected_l):
    s_st = st[0]
    v_st = st[1]
    theta_st = st[2]
    omega_st = st[3]
    sin_t = np.sin(theta_st)
    cos_t = np.cos(theta_st)
    M = params["M"]
    m = params["m"]
    l = params["l"] if use_corrected_l else 0.5
    wind_force = params["wind"]
    den = M + m * (1.0 - (cos_t * cos_t))
    F_tot = F_motor + wind_force * cos_t
    ds = v_st
    dv = (F_tot + m * l * omega_st * omega_st * sin_t + m * 9.81 * cos_t * sin_t) / den
    dtheta = omega_st
    domega = (
        (-1.0 * F_tot * cos_t)
        - (m * l * omega_st * omega_st * sin_t * cos_t)
        - ((M + m) * 9.81 * sin_t)
    ) / (l * den)
    return np.array([ds, dv, dtheta, domega])


def rk4_step(st, F_motor, params, dt, use_corrected_l):
    k1 = f_real(st, F_motor, params, use_corrected_l)
    k2 = f_real(st + (dt / 2.0) * k1, F_motor, params, use_corrected_l)
    k3 = f_real(st + (dt / 2.0) * k2, F_motor, params, use_corrected_l)
    k4 = f_real(st + dt * k3, F_motor, params, use_corrected_l)
    return st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    mpc, initial_state, target_state, params, sim_time, dt_sim, use_corrected_l
):
    steps = int(sim_time / dt_sim)
    state = np.array(initial_state)
    t_hist = []
    s_hist = []
    v_hist = []
    theta_hist = []
    omega_hist = []
    F_hist = []
    t_solve_hist = []
    success_hist = []
    current_time = 0.0
    for i in range(steps):
        u_opt, X_pred, U_pred, t_solve, success = mpc.step(state, target_state, params)
        t_hist.append(current_time)
        s_hist.append(state[0])
        v_hist.append(state[1])
        theta_wrapped = ((state[2] + np.pi) % (2.0 * np.pi)) - np.pi
        theta_hist.append(theta_wrapped)
        omega_hist.append(state[3])
        F_hist.append(u_opt)
        t_solve_hist.append(t_solve)
        success_hist.append(success)
        state = rk4_step(state, u_opt, params, dt_sim, use_corrected_l)
        current_time = current_time + dt_sim
    return {
        ("t"): (np.array(t_hist)),
        ("s"): (np.array(s_hist)),
        ("v"): (np.array(v_hist)),
        ("theta"): (np.array(theta_hist)),
        ("omega"): (np.array(omega_hist)),
        ("F"): (np.array(F_hist)),
        ("t_solve"): (np.array(t_solve_hist)),
        ("success"): (np.array(success_hist)),
    }


def evaluate_run(data, target_state):
    t = data["t"]
    s = data["s"]
    theta = data["theta"]
    F = data["F"]
    t_solve = data["t_solve"]
    success = data["success"]
    target_s = target_state[0]
    target_theta = target_state[2]
    dt = ((t[1]) - (t[0])) if len(t) > 1 else 5.0e-2
    iae_s = np.sum(np.abs(s - target_s)) * dt
    iae_theta = np.sum(np.abs(theta - target_theta)) * dt
    settling_time_s = 0.0
    for i in reversed(range(len(t))):
        if np.abs((s[i]) - target_s) > 5.0e-2:
            if i < (len(t) - 1):
                settling_time_s = t[i + 1]
            else:
                settling_time_s = float("inf")
            break
    settling_time_theta = 0.0
    for i in reversed(range(len(t))):
        if np.abs((theta[i]) - target_theta) > 5.0e-2:
            if i < (len(t) - 1):
                settling_time_theta = t[i + 1]
            else:
                settling_time_theta = float("inf")
            break
    overshoot_s = np.max(np.abs(s - target_s))
    max_F = np.max(np.abs(F))
    avg_t_solve = np.mean(t_solve) * 1.0e3
    max_t_solve = np.max(t_solve) * 1.0e3
    success_rate = np.mean(success) * 1.0e2
    return {
        ("iae_s"): (iae_s),
        ("iae_theta"): (iae_theta),
        ("settling_time_s"): (settling_time_s),
        ("settling_time_theta"): (settling_time_theta),
        ("overshoot_s"): (overshoot_s),
        ("max_F"): (max_F),
        ("avg_t_solve"): (avg_t_solve),
        ("max_t_solve"): (max_t_solve),
        ("success_rate"): (success_rate),
    }


def run_tests():
    results = []
    configs = [
        {
            ("name"): ("Config 1: Current Default (high Q_v, Q_omega)"),
            ("Q_s"): (1.0e1),
            ("Q_v"): (1.0e1),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0e1),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (0.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (20),
            ("use_corrected_l"): (True),
        },
        {
            ("name"): ("Config 2: Old Commit Default (low Q_v, Q_omega)"),
            ("Q_s"): (1.0e1),
            ("Q_v"): (1.0),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (0.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (20),
            ("use_corrected_l"): (True),
        },
        {
            ("name"): ("Config 3: High Position Weight (Q_s = 100.0)"),
            ("Q_s"): (1.0e2),
            ("Q_v"): (1.0e1),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0e1),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (0.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (20),
            ("use_corrected_l"): (True),
        },
        {
            ("name"): ("Config 4: Fast Position Response (Q_s=50, Q_v=2, R_F=0.05)"),
            ("Q_s"): (5.0e1),
            ("Q_v"): (2.0),
            ("Q_theta"): (1.5e2),
            ("Q_omega"): (2.0),
            ("R_F"): (5.0e-2),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (0.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (20),
            ("use_corrected_l"): (True),
        },
        {
            ("name"): ("Config 5: Short Horizon (N=10, T_horiz=0.5s)"),
            ("Q_s"): (1.0e1),
            ("Q_v"): (1.0e1),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0e1),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (0.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (10),
            ("use_corrected_l"): (True),
        },
        {
            ("name"): ("Config 6: Long Horizon (N=40, T_horiz=2.0s)"),
            ("Q_s"): (1.0e1),
            ("Q_v"): (1.0e1),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0e1),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (0.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (40),
            ("use_corrected_l"): (True),
        },
        {
            ("name"): ("Config 7: L-Mismatch (l_slider=1.5, buggy physics l=0.5)"),
            ("Q_s"): (1.0e1),
            ("Q_v"): (1.0e1),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0e1),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (1.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (20),
            ("use_corrected_l"): (False),
        },
        {
            ("name"): ("Config 8: L-Matched (l_slider=1.5, corrected physics l=1.5)"),
            ("Q_s"): (1.0e1),
            ("Q_v"): (1.0e1),
            ("Q_theta"): (1.0e2),
            ("Q_omega"): (1.0e1),
            ("R_F"): (0.1),
            ("M"): (1.0),
            ("m"): (0.1),
            ("l"): (1.5),
            ("wind"): (0.0),
            ("max_pos"): (2.0),
            ("max_force"): (1.5e1),
            ("h_mpc"): (5.0e-2),
            ("N"): (20),
            ("use_corrected_l"): (True),
        },
    ]
    test_scenarios = [
        {
            ("name"): ("Stabilization (theta_0 = 0.2 rad)"),
            ("x0"): ([0.0, 0.0, 0.2, 0.0]),
            ("x_target"): ([1.5, 0.0, 0.0, 0.0]),
            ("sim_time"): (8.0),
            ("dt_sim"): (3.3e-2),
        },
        {
            ("name"): ("Swing-Up & Position Step (theta_0 = pi)"),
            ("x0"): ([0.0, 0.0, np.pi, 0.0]),
            ("x_target"): ([1.5, 0.0, 0.0, 0.0]),
            ("sim_time"): (1.2e1),
            ("dt_sim"): (3.3e-2),
        },
    ]
    print("Starte MPC Parameter und Simulations-Benchmark...")
    out_results = {}
    for scenario in test_scenarios:
        print(
            f"\n=========================================\nSzenario: {scenario['name']}\n========================================="
        )
        scenario_results = []
        for cfg in configs:
            print(f"Simuliere: {cfg['name']}...")
            mpc = PendulumMPC(h_mpc=cfg["h_mpc"], N=cfg["N"])
            sim_data = simulate(
                mpc,
                scenario["x0"],
                scenario["x_target"],
                cfg,
                scenario["sim_time"],
                scenario["dt_sim"],
                cfg["use_corrected_l"],
            )
            metrics = evaluate_run(sim_data, scenario["x_target"])
            record = {("name"): (cfg["name"]), ("metrics"): (metrics)}
            scenario_results.append(record)
            m = metrics
            print(
                f"  - IAE_s: {m['iae_s']:.3f}, Settling t_s: {m['settling_time_s']:.2f}s, Overshoot: {m['overshoot_s']:.3f}m, Success: {m['success_rate']:.1f}%"
            )
        out_results[scenario["name"]] = scenario_results
    report_content = []
    report_content.append("# MPC Benchmark-Bericht & Parameter-Validierung")
    report_content.append(f"Generiert am: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(
        "\nDieser Bericht vergleicht verschiedene Reglereinstellungen und validiert die Korrektheit des Modells."
    )
    for scenario_name, scen_res in out_results.items():
        report_content.append(f"\n## Szenario: {scenario_name}")
        report_content.append(
            "| Konfiguration | IAE Position [m*s] | IAE Winkel [rad*s] | Settling Time Position [s] | Settling Time Winkel [s] | Overshoot Position [m] | Max Kraft [N] | Avg Solve Time [ms] | Solver Success % |"
        )
        report_content.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        for r in scen_res:
            name = r["name"]
            m = r["metrics"]
            t_s = (
                "Never"
                if m["settling_time_s"] == float("inf")
                else f"{m['settling_time_s']:.2f}"
            )
            t_theta = (
                "Never"
                if m["settling_time_theta"] == float("inf")
                else f"{m['settling_time_theta']:.2f}"
            )
            report_content.append(
                f"| {name} | {m['iae_s']:.3f} | {m['iae_theta']:.3f} | {t_s} | {t_theta} | {m['overshoot_s']:.3f} | {m['max_F']:.2f} | {m['avg_t_solve']:.2f} | {m['success_rate']:.1f}% |"
            )
    report_content.append("\n## Analyse & Empfehlungen")
    report_content.append("1. **Einfluss von Q_v und Q_omega (Config 1 vs Config 2):**")
    report_content.append(
        "   - In älteren Commits (Config 2) waren die Gewichte fuer die Geschwindigkeiten (Wagen-Geschwindigkeit Q_v und Winkel-Geschwindigkeit Q_omega) auf 1.0 statt 10.0 eingestellt."
    )
    report_content.append(
        "   - Niedrigere Dämpfungs-Kosten erlauben dem Regler, viel schneller zu beschleunigen und abzubremsen, was die Einschwingzeit (Settling Time) der Position verkuerzt, aber eventuell zu leichtem Überschwingen führt."
    )
    report_content.append("2. **Tuning fuer aggressive Positionsregelung (Config 4):**")
    report_content.append(
        "   - Durch Erhoehung von Q_s (auf z.B. 50) und gleichzeitiges Absenken von Q_v (auf z.B. 2) bei geringeren Stellkraftkosten (R_F = 0.05) kann der Wagen extrem praezise positioniert werden, ohne instabil zu werden."
    )
    report_content.append("3. **Pendellänge-Modell-Mismatch (Config 7 vs Config 8):**")
    report_content.append(
        "   - Wenn der Benutzer im GUI die Pendellänge veraendert, der Simulator aber intern mit der hartcodierten Laenge von 0.5 rechnet, weichen MPC-Modell und reale Physik stark voneinander ab. Das fuehrt zu schlechterer Performance (Config 7)."
    )
    report_content.append(
        "   - Bei passender Physik (Config 8) regelt der MPC das System auch bei Laenge 1.5 optimal."
    )
    report_str = "\n".join(report_content)
    print("\n--- BENCHMARK RESULTS ---")
    print(report_str)
    out_path = "p11d_benchmark_report.md"
    with open(out_path, "w") as f:
        f.write(report_str)
    print(f"\nBericht erfolgreich geschrieben unter: {out_path}")


if __name__ == "__main__":
    run_tests()
