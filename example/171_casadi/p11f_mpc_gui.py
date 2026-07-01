from __future__ import annotations
from casadi import *
import numpy as np
import time
import sys
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QGridLayout,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor


# =========================================================================================
#  INVERTED PENDULUM MPC GUI (MAPPED CONSTRAINTS + JIT CACHING)
# =========================================================================================
class PendulumMPC:
    def __init__(
        self,
        h_mpc=5.0e-2,
        N=20,
        use_jit=False,
        use_to_function=False,
        use_dual_warmstart=False,
        use_map=False,
    ):
        self.opti = Opti()
        self.nx = 4
        self.nu = 1
        self.N = N
        self.h_mpc = h_mpc
        self.T_horizon = self.N * self.h_mpc
        self.d = 3
        self.use_jit = use_jit
        self.use_to_function = use_to_function
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
        ipopt_opts = {("print_level"): (0), ("sb"): ("yes")}
        if self.use_jit:
            solver_opts["jit"] = True
            solver_opts["compiler"] = "shell"
            solver_opts["jit_options"] = {
                ("flags"): (["-O3", "-ffast-math", "-march=native"])
            }
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
        if self.use_map:
            self.last_Xc_var = np.zeros(
                (
                    self.nx * self.d,
                    self.N,
                )
            )
        if self.use_to_function:
            inputs = [
                self.current_x,
                self.target_x,
                self.M_p,
                self.m_p,
                self.l_p,
                self.wind_p,
                self.Q_s,
                self.Q_v,
                self.Q_theta,
                self.Q_omega,
                self.R_F,
                self.max_pos,
                self.max_force,
                self.X,
            ]
            if self.use_map:
                inputs.append(self.Xc_var)
            inputs.append(self.U)
            if self.use_dual_warmstart:
                inputs.append(self.opti.lam_g)
            outputs = [self.U[0, 0], self.X]
            if self.use_map:
                outputs.append(self.Xc_var)
            outputs.append(self.U)
            if self.use_dual_warmstart:
                outputs.append(self.opti.lam_g)
            self.mpc_func = self.opti.to_function("mpc_solve", inputs, outputs)

    def step(self, state, target_state, params):
        if self.use_to_function:
            inputs = [
                state,
                target_state,
                params["M"],
                params["m"],
                params["l"],
                params["wind"],
                params["Q_s"],
                params["Q_v"],
                params["Q_theta"],
                params["Q_omega"],
                params["R_F"],
                params["max_pos"],
                params["max_force"],
            ]
            if (self.sol is not None) or (self.last_lam_g is not None):
                X_guess = np.hstack(
                    (
                        self.last_X[:, 1:],
                        self.last_X[:, -1:],
                    )
                )
                U_guess = np.append(self.last_U[1:], self.last_U[-1])
                if self.use_map:
                    Xc_guess = self.last_Xc_var
            else:
                X_guess = np.linspace(state, target_state, self.N + 1).T
                U_guess = np.zeros(self.N)
                if self.use_map:
                    Xc_guess = np.zeros(
                        (
                            self.nx * self.d,
                            self.N,
                        )
                    )
            inputs.append(X_guess)
            if self.use_map:
                inputs.append(Xc_guess)
            inputs.append(U_guess[np.newaxis, :])
            if self.use_dual_warmstart:
                if self.last_lam_g is not None:
                    lam_g_guess = self.last_lam_g
                else:
                    lam_g_guess = np.zeros(self.n_constraints)
                inputs.append(lam_g_guess)
            t0 = time.time()
            try:
                res = self.mpc_func(*inputs)
                if self.use_map and self.use_dual_warmstart:
                    u_opt, X_val, Xc_val, U_val, lam_g_val = res
                    self.last_X = X_val.full()
                    self.last_Xc_var = Xc_val.full()
                    self.last_U = np.squeeze(U_val.full())
                    self.last_lam_g = lam_g_val.full()
                elif self.use_map:
                    u_opt, X_val, Xc_val, U_val = res
                    self.last_X = X_val.full()
                    self.last_Xc_var = Xc_val.full()
                    self.last_U = np.squeeze(U_val.full())
                elif self.use_dual_warmstart:
                    u_opt, X_val, U_val, lam_g_val = res
                    self.last_X = X_val.full()
                    self.last_U = np.squeeze(U_val.full())
                    self.last_lam_g = lam_g_val.full()
                else:
                    u_opt, X_val, U_val = res
                    self.last_X = X_val.full()
                    self.last_U = np.squeeze(U_val.full())
                u_opt = float(u_opt)
                t_solve = time.time() - t0
                return (
                    u_opt,
                    self.last_X,
                    self.last_U,
                    t_solve,
                    True,
                )
            except Exception as e:
                t_solve = time.time() - t0
                return (
                    float(U_guess[0]),
                    self.last_X,
                    self.last_U,
                    t_solve,
                    False,
                )
        else:
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


class PendulumWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setMinimumSize(400, 400)
        self.state = np.array([0.0, 0.0, np.pi, 0.0])
        self.l = 0.5
        self.force = 0.0
        self.wind = 0.0
        self.max_pos = 2.0

    def update_state(self, state, force, wind, l, max_pos):
        self.state = state
        self.force = force
        self.wind = wind
        self.l = l
        self.max_pos = max_pos
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        cx = w / 2
        cy = h / 2
        scale = w / (2.5 * self.max_pos)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        painter.setPen(pg.mkPen("gray", width=4))
        painter.drawLine(cx - (scale * self.max_pos), cy, cx + scale * self.max_pos, cy)
        s = self.state[0]
        theta = self.state[2]
        cart_w = 0.4 * scale
        cart_h = 0.2 * scale
        cart_x = cx + s * scale
        cart_y = cy
        painter.setBrush(QBrush(QColor(100, 150, 250)))
        painter.setPen(pg.mkPen("w", width=2))
        painter.drawRect(cart_x - (cart_w / 2), cart_y - (cart_h / 2), cart_w, cart_h)
        pend_x = cart_x + scale * self.l * np.sin(theta)
        pend_y = cart_y - (scale * self.l * np.cos(theta))
        painter.setPen(pg.mkPen("w", width=6))
        painter.drawLine(cart_x, cart_y, pend_x, pend_y)
        painter.setBrush(QBrush(QColor(250, 100, 100)))
        painter.drawEllipse(pend_x - 10, pend_y - 10, 20, 20)
        if np.abs(self.force) > 0.1:
            painter.setPen(pg.mkPen("g", width=3))
            painter.drawLine(cart_x, cart_y, cart_x + 5.0 * self.force, cart_y)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("MPC Inverted Pendulum - Mapped JIT Cache version")
        self.resize(1400, 900)
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        self.setCentralWidget(central_widget)
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.pendulum_widget = PendulumWidget()
        self.plot_layout = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.pendulum_widget)
        right_layout.addWidget(self.plot_layout)
        self.plots = {}
        self.history_curves = {}
        self.pred_curves = {}
        ax = self.plot_layout.addPlot(row=0, col=0, title="Wagenposition")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Position [m]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["s"] = ax
        self.history_curves["s"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "s" != "t_solve":
            self.pred_curves["s"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DashLine)
            )
        ax = self.plot_layout.addPlot(row=1, col=0, title="Pendelwinkel")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Winkel [rad]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["theta"] = ax
        self.history_curves["theta"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "theta" != "t_solve":
            self.pred_curves["theta"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DashLine)
            )
        ax = self.plot_layout.addPlot(row=2, col=0, title="Wagengeschwindigkeit")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "v [m/s]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["v"] = ax
        self.history_curves["v"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "v" != "t_solve":
            self.pred_curves["v"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DashLine)
            )
        ax = self.plot_layout.addPlot(row=0, col=1, title="Winkelgeschw.")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "omega [rad/s]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["omega"] = ax
        self.history_curves["omega"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "omega" != "t_solve":
            self.pred_curves["omega"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DashLine)
            )
        ax = self.plot_layout.addPlot(row=1, col=1, title="Aktuatorkraft")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Kraft [N]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["F"] = ax
        self.history_curves["F"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "F" != "t_solve":
            self.pred_curves["F"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DashLine)
            )
        ax = self.plot_layout.addPlot(row=2, col=1, title="Solver Rechenzeit")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Zeit [ms]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["t_solve"] = ax
        self.history_curves["t_solve"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "t_solve" != "t_solve":
            self.pred_curves["t_solve"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DashLine)
            )
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset / Apply Params")
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_reset)
        left_layout.addLayout(btn_layout)
        self.btn_start.clicked.connect(self.start_sim)
        self.btn_stop.clicked.connect(self.stop_sim)
        self.btn_reset.clicked.connect(self.reset_sim)
        self.sliders = {}
        slider_layout = QGridLayout()
        left_layout.addLayout(slider_layout)

        def add_slider(
            layout, name, label, min_val, max_val, default_val, scale, row, tooltip
        ):
            slider = QSlider(Qt.Horizontal)
            init_val = default_val * scale
            lbl = QLabel(
                f"{label}: {init_val:.2f}"
                if isinstance(scale, float)
                else f"{label}: {int(init_val)}"
            )
            slider.setToolTip(tooltip)
            lbl.setToolTip(tooltip)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            layout.addWidget(lbl, row, 0)
            layout.addWidget(slider, row, 1)
            slider.valueChanged.connect(
                lambda: lbl.setText(
                    f"{label}: {slider.value() * scale:.2f}"
                    if isinstance(scale, float)
                    else f"{label}: {int(slider.value() * scale)}"
                )
            )
            self.sliders[name] = slider

        add_slider(
            slider_layout,
            "M",
            "Wagenmasse [kg]",
            1,
            50,
            10,
            0.1,
            0,
            "Masse des Wagens. Schwerer = Träger gegen Bewegungsänderungen.",
        )
        add_slider(
            slider_layout,
            "m",
            "Pendelmasse [kg]",
            1,
            20,
            1,
            0.1,
            1,
            "Punktmasse des Pendels am Kopfende.",
        )
        add_slider(
            slider_layout,
            "l",
            "Pendellänge [m]",
            1,
            20,
            5,
            0.1,
            2,
            "Abstand vom Wagen zum Pendelschwerpunkt.",
        )
        add_slider(
            slider_layout,
            "wind",
            "Windkraft [N]",
            -300,
            300,
            0,
            0.1,
            3,
            "Konstante Störkraft, die horizontal auf das Pendel drückt.",
        )
        add_slider(
            slider_layout,
            "Q_s",
            "Gewicht Position",
            0,
            200,
            10,
            1,
            4,
            "Strafe (Penalty) für Abweichung des Wagens von der Ziel-Position.",
        )
        add_slider(
            slider_layout,
            "Q_v",
            "Gewicht Wagengeschw.",
            0,
            100,
            1,
            1,
            5,
            "Strafe für hohe Geschwindigkeit des Wagens (verhindert Überschwingen).",
        )
        add_slider(
            slider_layout,
            "Q_theta",
            "Gewicht Pendelwinkel",
            0,
            500,
            100,
            1,
            6,
            "Strafe für das Abweichen des Pendels vom instabilen Gleichgewicht (0 rad).",
        )
        add_slider(
            slider_layout,
            "Q_omega",
            "Gewicht Winkelgeschw.",
            0,
            100,
            1,
            1,
            7,
            "Strafe für schnelle Rotationen des Pendels.",
        )
        add_slider(
            slider_layout,
            "R_F",
            "Gewicht Kraftaufwand",
            1,
            200,
            10,
            1.0e-2,
            8,
            "Kostenfaktor für die Stellkraft F. Zwingt den Solver, Energie zu sparen.",
        )
        add_slider(
            slider_layout,
            "target_s",
            "Ziel-Position [m]",
            -100,
            100,
            10,
            0.1,
            9,
            "Soll-Position des Wagens auf der Schiene.",
        )
        add_slider(
            slider_layout,
            "max_pos",
            "Schiene Limit [m]",
            10,
            200,
            50,
            0.1,
            10,
            "Maximaler erlaubter Fahrweg (Constraint). Der Solver darf diesen nie überschreiten.",
        )
        add_slider(
            slider_layout,
            "max_force",
            "Max Kraft [N]",
            0,
            300,
            150,
            0.1,
            11,
            "Stellgrößenbeschränkung für den Aktuator.",
        )
        add_slider(
            slider_layout,
            "N",
            "Knotenpunkte (N)",
            1,
            500,
            20,
            1,
            12,
            "Auflösung des Solvers. N=1 bedeutet nur ein Zeitschritt in die Zukunft.",
        )
        add_slider(
            slider_layout,
            "h_mpc",
            "MPC Schritt [ms]",
            1,
            200,
            50,
            1,
            13,
            "Dauer eines MPC-Planungsschritts. T_horizon = N * h_mpc.",
        )
        add_slider(
            slider_layout,
            "dt_sim",
            "Simulations-dt [ms]",
            1,
            100,
            33,
            1,
            14,
            "Schrittweite der echten Runge-Kutta Physiksimulation. Hat keinen Einfluss auf den Solver.",
        )
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        params = self.get_params()
        self.mpc = PendulumMPC(
            h_mpc=params["h_mpc"],
            N=params["N"],
            use_jit=True,
            use_to_function=True,
            use_dual_warmstart=True,
            use_map=True,
        )
        self.reset_sim()
        self.start_sim()

    def start_sim(self):
        self.dt = (self.sliders["dt_sim"].value()) / 1.0e3
        self.timer.start(int(self.dt * 1.0e3))
        self.sliders["h_mpc"].setEnabled(False)
        self.sliders["N"].setEnabled(False)

    def stop_sim(self):
        self.timer.stop()
        self.sliders["h_mpc"].setEnabled(True)
        self.sliders["N"].setEnabled(True)

    def reset_sim(self):
        self.stop_sim()
        params = self.get_params()
        self.time = 0.0
        self.state = np.array([0.0, 0.0, np.pi, 0.0])
        self.t_hist = []
        self.dt = params["dt_sim"]
        self.hist = {
            ("s"): ([]),
            ("v"): ([]),
            ("theta"): ([]),
            ("omega"): ([]),
            ("F"): ([]),
            ("t_solve"): ([]),
        }
        new_N = int(params["N"])
        new_h = params["h_mpc"]
        # Instanziere den MPC-Solver NUR neu, wenn sich N oder h_mpc geaendert haben.
        if not hasattr(self, "mpc") or self.mpc.N != new_N or self.mpc.h_mpc != new_h:
            print(
                "N oder h_mpc haben sich geaendert. Kompiliere und generiere JIT MPC neu..."
            )
            self.mpc = PendulumMPC(
                h_mpc=new_h,
                N=new_N,
                use_jit=True,
                use_to_function=True,
                use_dual_warmstart=True,
                use_map=True,
            )
        else:
            print(
                "Parameter N und h_mpc unveraendert. Verwende bereits kompilierten JIT MPC Solver wieder."
            )
        self.pendulum_widget.update_state(
            self.state, 0.0, 0.0, params["l"], params["max_pos"]
        )
        self.history_curves["s"].setData([], [])
        self.history_curves["v"].setData([], [])
        self.history_curves["theta"].setData([], [])
        self.history_curves["omega"].setData([], [])
        self.history_curves["F"].setData([], [])
        self.history_curves["t_solve"].setData([], [])
        self.pred_curves["s"].setData([], [])
        self.pred_curves["v"].setData([], [])
        self.pred_curves["theta"].setData([], [])
        self.pred_curves["omega"].setData([], [])
        self.pred_curves["F"].setData([], [])

    def get_params(self):
        return dict(
            M=(self.sliders["M"].value()) / 1.0e1,
            m=(self.sliders["m"].value()) / 1.0e1,
            l=(self.sliders["l"].value()) / 1.0e1,
            wind=(self.sliders["wind"].value()) / 1.0e1,
            Q_s=float(self.sliders["Q_s"].value()),
            Q_v=float(self.sliders["Q_v"].value()),
            Q_theta=float(self.sliders["Q_theta"].value()),
            Q_omega=float(self.sliders["Q_omega"].value()),
            R_F=(self.sliders["R_F"].value()) / 1.0e2,
            max_pos=(self.sliders["max_pos"].value()) / 1.0e1,
            max_force=(self.sliders["max_force"].value()) / 1.0e1,
            h_mpc=(self.sliders["h_mpc"].value()) / 1.0e3,
            N=int(self.sliders["N"].value()),
            dt_sim=(self.sliders["dt_sim"].value()) / 1.0e3,
        )

    def update_loop(self):
        params = self.get_params()
        target_s = (self.sliders["target_s"].value()) / 1.0e1
        target_state = np.array([target_s, 0.0, 0.0, 0.0])
        u_opt, X_pred, U_pred, t_solve, success = self.mpc.step(
            self.state, target_state, params
        )
        F_motor = u_opt
        wind_force = params["wind"]
        self.dt = params["dt_sim"]
        self.timer.setInterval(int(self.dt * 1.0e3))

        def f_real(st):
            s_st = st[0]
            v_st = st[1]
            theta_st = st[2]
            omega_st = st[3]
            sin_t = np.sin(theta_st)
            cos_t = np.cos(theta_st)
            denom = params["M"] + params["m"] * (1.0 - (cos_t * cos_t))
            F_tot = F_motor + wind_force * cos_t
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

        k1 = f_real(self.state)
        k2 = f_real(self.state + (self.dt / 2.0) * k1)
        k3 = f_real(self.state + (self.dt / 2.0) * k2)
        k4 = f_real(self.state + self.dt * k3)
        self.state = self.state + np.array(
            np.squeeze((self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        )
        self.time = self.time + self.dt
        self.t_hist.append(self.time)
        self.hist["s"].append(self.state[0])
        self.hist["v"].append(self.state[1])
        self.hist["theta"].append(self.state[2])
        self.hist["omega"].append(self.state[3])
        self.hist["F"].append(F_motor)
        self.hist["t_solve"].append(t_solve * 1.0e3)
        if len(self.t_hist) > 100:
            self.t_hist.pop(0)
            self.hist["s"].pop(0)
            self.hist["v"].pop(0)
            self.hist["theta"].pop(0)
            self.hist["omega"].pop(0)
            self.hist["F"].pop(0)
            self.hist["t_solve"].pop(0)
        self.pendulum_widget.update_state(
            self.state, F_motor, wind_force, params["l"], params["max_pos"]
        )
        self.history_curves["s"].setData(self.t_hist, self.hist["s"])
        self.history_curves["v"].setData(self.t_hist, self.hist["v"])
        self.history_curves["theta"].setData(self.t_hist, self.hist["theta"])
        self.history_curves["omega"].setData(self.t_hist, self.hist["omega"])
        self.history_curves["F"].setData(self.t_hist, self.hist["F"])
        self.history_curves["t_solve"].setData(self.t_hist, self.hist["t_solve"])
        if success:
            t_pred = np.linspace(
                self.time, self.time + self.mpc.T_horizon, self.mpc.N + 1
            )
            self.pred_curves["s"].setData(t_pred, X_pred[0, :])
            self.pred_curves["v"].setData(t_pred, X_pred[1, :])
            self.pred_curves["theta"].setData(t_pred, X_pred[2, :])
            self.pred_curves["omega"].setData(t_pred, X_pred[3, :])
            self.pred_curves["F"].setData(t_pred[0:-1], U_pred)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
