import sys
import time
import numpy as np
import casadi as ca
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSlider,
    QLabel,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor

x_sym = ca.MX.sym("x", 4)
u_sym = ca.MX.sym("u", 1)
p_dyn = ca.MX.sym("p", 4)
dt_sym = ca.MX.sym("dt", 1)
s_ = x_sym[0]
v_ = x_sym[1]
th_ = x_sym[2]
om_ = x_sym[3]
M_ = p_dyn[0]
m_ = p_dyn[1]
l_ = p_dyn[2]
w_ = p_dyn[3]
sin_th = ca.sin(th_)
cos_th = ca.cos(th_)
denom = M_ + m_ * (1 - (cos_th**2))
f_ode = ca.Function(
    "f_ode",
    [x_sym, u_sym, p_dyn],
    [
        ca.vertcat(
            v_,
            (u_sym + w_ * cos_th + m_ * sin_th * ((l_ * (om_**2)) - (9.81 * cos_th)))
            / denom,
            om_,
            (
                (
                    (-1 * (u_sym + w_ * cos_th) * cos_th)
                    - (m_ * l_ * (om_**2) * sin_th * cos_th)
                )
                + (M_ + m_) * 9.81 * sin_th
            )
            / (l_ * denom),
        )
    ],
)
rk4_k1 = f_ode(x_sym, u_sym, p_dyn)
rk4_k2 = f_ode(x_sym + (dt_sym / 2) * rk4_k1, u_sym, p_dyn)
rk4_k3 = f_ode(x_sym + (dt_sym / 2) * rk4_k2, u_sym, p_dyn)
rk4_k4 = f_ode(x_sym + dt_sym * rk4_k3, u_sym, p_dyn)
f_rk4 = ca.Function(
    "f_rk4",
    [x_sym, u_sym, p_dyn, dt_sym],
    [x_sym + (dt_sym / 6) * (rk4_k1 + 2 * rk4_k2 + 2 * rk4_k3 + rk4_k4)],
)


def build_solvers(N_mhe, N_mpc, max_iter, use_lbfgs):
    opti_mhe = ca.Opti()
    X_mhe = opti_mhe.variable(4, N_mhe + 1)
    W_mhe = opti_mhe.variable(4, N_mhe)
    Y_meas_param = opti_mhe.parameter(4, N_mhe + 1)
    U_past_param = opti_mhe.parameter(1, N_mhe)
    X_prior_param = opti_mhe.parameter(4, 1)
    P_dyn_param = opti_mhe.parameter(4, 1)
    dt_mhe_param = opti_mhe.parameter(1, 1)
    Q_w_p = opti_mhe.parameter(4, 1)
    R_w_p = opti_mhe.parameter(4, 1)
    P_w_p = opti_mhe.parameter(4, 1)
    f_map_mhe = f_rk4.map(N_mhe)
    X_next_mhe = f_map_mhe(
        X_mhe[:, 0:N_mhe],
        U_past_param,
        ca.repmat(P_dyn_param, 1, N_mhe),
        ca.repmat(dt_mhe_param, 1, N_mhe),
    )
    opti_mhe.subject_to(X_mhe[:, 1 : N_mhe + 1] == X_next_mhe + W_mhe)
    opti_mhe.minimize(
        ca.sum1(P_w_p * (((X_mhe[:, 0]) - X_prior_param) ** 2))
        + ca.sum2(ca.sum1(Q_w_p * (W_mhe**2)))
        + ca.sum2(ca.sum1(R_w_p * ((Y_meas_param - X_mhe) ** 2)))
    )
    ipopt_opts = dict(print_level=0, sb="yes", max_iter=max_iter)
    if use_lbfgs:
        ipopt_opts["hessian_approximation"] = "limited-memory"
    opti_mhe.solver("ipopt", dict(print_time=False), ipopt_opts)
    opti_mpc = ca.Opti()
    X_mpc = opti_mpc.variable(4, N_mpc + 1)
    U_mpc = opti_mpc.variable(1, N_mpc)
    X_cur_p = opti_mpc.parameter(4, 1)
    X_tgt_p = opti_mpc.parameter(4, 1)
    P_dyn_mpc = opti_mpc.parameter(4, 1)
    dt_mpc_p = opti_mpc.parameter(1, 1)
    Q_s_p = opti_mpc.parameter()
    Q_v_p = opti_mpc.parameter()
    Q_th_p = opti_mpc.parameter()
    Q_om_p = opti_mpc.parameter()
    R_F_p = opti_mpc.parameter()
    max_pos_p = opti_mpc.parameter()
    max_F_p = opti_mpc.parameter()
    f_map_mpc = f_rk4.map(N_mpc)
    X_next_mpc = f_map_mpc(
        X_mpc[:, 0:N_mpc],
        U_mpc,
        ca.repmat(P_dyn_mpc, 1, N_mpc),
        ca.repmat(dt_mpc_p, 1, N_mpc),
    )
    opti_mpc.subject_to(X_mpc[:, 0] == X_cur_p)
    opti_mpc.subject_to(X_mpc[:, 1 : N_mpc + 1] == X_next_mpc)
    opti_mpc.subject_to(opti_mpc.bounded(-1 * max_pos_p, X_mpc[0, :], max_pos_p))
    opti_mpc.subject_to(opti_mpc.bounded(-1 * max_F_p, U_mpc, max_F_p))
    err_ = X_mpc - ca.repmat(X_tgt_p, 1, N_mpc + 1)
    opti_mpc.minimize(
        ca.sum2(ca.sum1(ca.vertcat(Q_s_p, Q_v_p, Q_th_p, Q_om_p) * (err_**2)))
        + R_F_p * ca.sum2(U_mpc**2)
    )
    ipopt_opts2 = dict(print_level=0, sb="yes", max_iter=max_iter)
    if use_lbfgs:
        ipopt_opts2["hessian_approximation"] = "limited-memory"
    opti_mpc.solver("ipopt", dict(print_time=False), ipopt_opts2)
    return (
        opti_mhe,
        X_mhe,
        W_mhe,
        Y_meas_param,
        U_past_param,
        X_prior_param,
        P_dyn_param,
        dt_mhe_param,
        Q_w_p,
        R_w_p,
        P_w_p,
        opti_mpc,
        X_mpc,
        U_mpc,
        X_cur_p,
        X_tgt_p,
        P_dyn_mpc,
        dt_mpc_p,
        Q_s_p,
        Q_v_p,
        Q_th_p,
        Q_om_p,
        R_F_p,
        max_pos_p,
        max_F_p,
        N_mpc,
    )


class PendulumWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setMinimumSize(400, 280)
        self.x_true = np.array([0.0, 0.0, 3.14159, 0.0])
        self.x_est = np.array([0.0, 0.0, 3.14159, 0.0])
        self.force = 0.0
        self.l = 0.5
        self.max_pos = 2.0

    def update_state(self, x_true, x_est, force, l, max_pos):
        self.x_true = x_true
        self.x_est = x_est
        self.force = force
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
        painter.setPen(QPen(QColor(128, 128, 128), 4))
        painter.drawLine(cx - (scale * self.max_pos), cy, cx + scale * self.max_pos, cy)

        def draw_pend(state, cart_col, pole_col):
            sx = cx + state[0] * scale
            cw = 0.4 * scale
            ch = 0.2 * scale
            painter.setBrush(QBrush(cart_col))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(sx - (cw / 2), cy - (ch / 2), cw, ch)
            px = sx + scale * self.l * np.sin(state[2])
            py = cy - (scale * self.l * np.cos(state[2]))
            painter.setPen(QPen(pole_col, 6))
            painter.drawLine(sx, cy, px, py)
            painter.setBrush(QBrush(pole_col))
            painter.setPen(QPen(pole_col, 1))
            painter.drawEllipse(px - 8, py - 8, 16, 16)

        draw_pend(self.x_est, QColor(180, 100, 30), QColor(255, 165, 0))
        draw_pend(self.x_true, QColor(100, 150, 250), QColor(250, 100, 100))
        if np.abs(self.force) > 0.1:
            ax2 = cx + self.x_true[0] * scale
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            painter.drawLine(ax2, cy, int(ax2 + 3.0 * self.force), cy)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Inverted Pendulum — MHE + MPC")
        self.resize(1500, 950)
        central = QWidget()
        root_layout = QHBoxLayout(central)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        self.setCentralWidget(central)
        root_layout.addLayout(left_col, 1)
        root_layout.addLayout(right_col, 2)
        self.pend_widget = PendulumWidget()
        left_col.addWidget(self.pend_widget)
        self.lbl_delay = QLabel("Rechenzeit: OK")
        self.lbl_delay.setStyleSheet("font-size:13px;font-weight:bold;color:green")
        left_col.addWidget(self.lbl_delay)
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset / Rebuild")
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_reset)
        left_col.addLayout(btn_row)
        self.btn_start.clicked.connect(self.start_sim)
        self.btn_stop.clicked.connect(self.stop_sim)
        self.btn_reset.clicked.connect(self.reset_sim)
        scroll_area = QScrollArea()
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        scroll_area.setWidget(ctrl_widget)
        scroll_area.setWidgetResizable(True)
        left_col.addWidget(scroll_area, 1)
        self.sliders = {}

        def make_slider(grid, name, label, min_v, max_v, default, scale, row, tip):
            sl = QSlider(Qt.Horizontal)
            lbl = QLabel(f"{label}: {default * scale:.3g}")
            sl.setRange(min_v, max_v)
            sl.setValue(default)
            sl.setToolTip(tip)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(sl, row, 1)
            sl.valueChanged.connect(
                lambda: lbl.setText(f"{label}: {sl.value() * scale:.3g}")
            )
            self.sliders[name] = sl

        def make_group(title):
            grp = QGroupBox(title)
            grd = QGridLayout()
            grp.setLayout(grd)
            ctrl_layout.addWidget(grp)
            return grd

        g = make_group("Physik / Simulation")
        make_slider(g, "M", "Wagenmasse [kg]", 1, 50, 10, 0.1, 0, "Cart mass")
        make_slider(g, "m", "Pendelmasse [kg]", 1, 20, 1, 0.1, 1, "Pole tip mass")
        make_slider(g, "l", "Pendellänge [m]", 1, 20, 5, 0.1, 2, "Pole length")
        make_slider(
            g, "wind", "Wind [N]", -300, 300, 0, 0.1, 3, "Horizontal wind force"
        )
        make_slider(
            g, "dt_sim", "Sim-dt [ms]", 1, 100, 20, 1.0, 4, "Simulation step size"
        )
        g = make_group("Sensorrauschen σ")
        make_slider(
            g,
            "sig_s",
            "σ Position s",
            0,
            100,
            2,
            1.0e-3,
            0,
            "Gaussian noise std-dev (m or rad)",
        )
        make_slider(
            g,
            "sig_v",
            "σ Geschw. v",
            0,
            100,
            5,
            1.0e-3,
            1,
            "Gaussian noise std-dev (m or rad)",
        )
        make_slider(
            g,
            "sig_th",
            "σ Winkel θ",
            0,
            100,
            2,
            1.0e-3,
            2,
            "Gaussian noise std-dev (m or rad)",
        )
        make_slider(
            g,
            "sig_om",
            "σ Winkelgeschw. ω",
            0,
            100,
            5,
            1.0e-3,
            3,
            "Gaussian noise std-dev (m or rad)",
        )
        g = make_group("MPC — Horizont & Constraints")
        make_slider(
            g, "N_mpc", "Horizont N_mpc", 2, 80, 40, 1.0, 0, "MPC steps — needs Rebuild"
        )
        make_slider(
            g, "target_s", "Ziel s [m]", -100, 100, 10, 0.1, 1, "Target cart position"
        )
        make_slider(
            g,
            "max_pos",
            "Schienenlimit [m]",
            10,
            200,
            50,
            0.1,
            2,
            "Track position limit",
        )
        make_slider(
            g,
            "max_force",
            "Max Kraft [N]",
            10,
            300,
            150,
            0.1,
            3,
            "Actuator force limit",
        )
        self.sliders["N_mpc_label"] = self.sliders["N_mpc"]
        g = make_group("MPC — Kostenfunktion")
        make_slider(g, "Q_s", "Gewicht s", 0, 200, 10, 1.0, 0, "State cost: position")
        make_slider(g, "Q_v", "Gewicht v", 0, 100, 1, 1.0, 1, "State cost: velocity")
        make_slider(g, "Q_theta", "Gewicht θ", 0, 500, 100, 1.0, 2, "State cost: angle")
        make_slider(
            g, "Q_omega", "Gewicht ω", 0, 100, 1, 1.0, 3, "State cost: angular vel"
        )
        make_slider(
            g, "R_F", "Gewicht Kraft", 1, 200, 10, 1.0e-2, 4, "Control effort cost"
        )
        g = make_group("MHE — Horizont & Sensoren")
        make_slider(
            g, "N_mhe", "Horizont N_mhe", 3, 40, 15, 1.0, 0, "MHE steps — needs Rebuild"
        )
        self.cb_s = QCheckBox("Sensor s aktiv")
        self.cb_v = QCheckBox("Sensor v aktiv")
        self.cb_th = QCheckBox("Sensor θ aktiv")
        self.cb_om = QCheckBox("Sensor ω aktiv")
        self.cb_s.setChecked(True)
        self.cb_th.setChecked(True)
        g.addWidget(self.cb_s, 1, 0, 1, 2)
        g.addWidget(self.cb_v, 2, 0, 1, 2)
        g.addWidget(self.cb_th, 3, 0, 1, 2)
        g.addWidget(self.cb_om, 4, 0, 1, 2)
        g = make_group("MHE — Gewichte")
        make_slider(
            g,
            "Q_w_scale",
            "Modellvertrauen Q_w",
            1,
            200,
            50,
            0.1,
            0,
            "Process noise weight scale (higher = trust model more)",
        )
        make_slider(
            g,
            "R_w_scale",
            "Sensorvertrauen R_w",
            1,
            200,
            50,
            0.1,
            1,
            "Measurement noise weight scale (higher = trust sensors more)",
        )
        make_slider(
            g,
            "P_w_scale",
            "Arrival Cost P_w",
            1,
            200,
            50,
            0.1,
            2,
            "Prior cost weight scale",
        )
        g = make_group("Solver (IPOPT)")
        make_slider(
            g,
            "max_iter",
            "Max. Iterationen",
            1,
            100,
            20,
            1.0,
            0,
            "IPOPT max iterations per solve — lower = faster but less accurate",
        )
        self.cb_lbfgs = QCheckBox("L-BFGS Hesse-Approximation (schneller, ungenauer)")
        g.addWidget(self.cb_lbfgs, 1, 0, 1, 2)
        self.plot_layout = pg.GraphicsLayoutWidget()
        right_col.addWidget(self.plot_layout)
        self.hist_curves = {}
        self.est_curves = {}
        self.pred_curves = {}
        ax = self.plot_layout.addPlot(row=0, col=0, title="Position")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "s [m]")
        ax.setLabel("bottom", "t [s]")
        self.hist_curves["s"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "s" != "t_solve":
            self.est_curves["s"] = ax.plot(
                pen=pg.mkPen("orange", width=2, style=Qt.DashLine)
            )
            self.pred_curves["s"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DotLine)
            )
        ax = self.plot_layout.addPlot(row=1, col=0, title="Winkel")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "rad")
        ax.setLabel("bottom", "t [s]")
        self.hist_curves["theta"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "theta" != "t_solve":
            self.est_curves["theta"] = ax.plot(
                pen=pg.mkPen("orange", width=2, style=Qt.DashLine)
            )
            self.pred_curves["theta"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DotLine)
            )
        ax = self.plot_layout.addPlot(row=2, col=0, title="Geschwindigkeit")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "m/s")
        ax.setLabel("bottom", "t [s]")
        self.hist_curves["v"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "v" != "t_solve":
            self.est_curves["v"] = ax.plot(
                pen=pg.mkPen("orange", width=2, style=Qt.DashLine)
            )
            self.pred_curves["v"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DotLine)
            )
        ax = self.plot_layout.addPlot(row=0, col=1, title="Winkelgeschw.")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "rad/s")
        ax.setLabel("bottom", "t [s]")
        self.hist_curves["omega"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "omega" != "t_solve":
            self.est_curves["omega"] = ax.plot(
                pen=pg.mkPen("orange", width=2, style=Qt.DashLine)
            )
            self.pred_curves["omega"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DotLine)
            )
        ax = self.plot_layout.addPlot(row=1, col=1, title="Kraft")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "N")
        ax.setLabel("bottom", "t [s]")
        self.hist_curves["F"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "F" != "t_solve":
            self.est_curves["F"] = ax.plot(
                pen=pg.mkPen("orange", width=2, style=Qt.DashLine)
            )
            self.pred_curves["F"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DotLine)
            )
        ax = self.plot_layout.addPlot(row=2, col=1, title="Rechenzeit")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "ms")
        ax.setLabel("bottom", "t [s]")
        self.hist_curves["t_solve"] = ax.plot(pen=pg.mkPen("w", width=2))
        if "t_solve" != "t_solve":
            self.est_curves["t_solve"] = ax.plot(
                pen=pg.mkPen("orange", width=2, style=Qt.DashLine)
            )
            self.pred_curves["t_solve"] = ax.plot(
                pen=pg.mkPen("y", width=2, style=Qt.DotLine)
            )
        self.reset_sim()
        self.start_sim()

    def sv(self, key):
        return self.sliders[key].value()

    def get_params(self):
        return dict(
            M=self.sv("M") * 0.1,
            m=self.sv("m") * 0.1,
            l=self.sv("l") * 0.1,
            wind=self.sv("wind") * 0.1,
            dt_sim=self.sv("dt_sim") * 1.0e-3,
            sig_s=self.sv("sig_s") * 1.0e-3,
            sig_v=self.sv("sig_v") * 1.0e-3,
            sig_th=self.sv("sig_th") * 1.0e-3,
            sig_om=self.sv("sig_om") * 1.0e-3,
            N_mpc=int(self.sv("N_mpc")),
            target_s=self.sv("target_s") * 0.1,
            max_pos=self.sv("max_pos") * 0.1,
            max_force=self.sv("max_force") * 0.1,
            Q_s=float(self.sv("Q_s")),
            Q_v=float(self.sv("Q_v")),
            Q_theta=float(self.sv("Q_theta")),
            Q_omega=float(self.sv("Q_omega")),
            R_F=self.sv("R_F") * 1.0e-2,
            N_mhe=int(self.sv("N_mhe")),
            Q_w_scale=self.sv("Q_w_scale") * 0.1,
            R_w_scale=self.sv("R_w_scale") * 0.1,
            P_w_scale=self.sv("P_w_scale") * 0.1,
            max_iter=int(self.sv("max_iter")),
            use_lbfgs=self.cb_lbfgs.isChecked(),
        )

    def start_sim(self):
        p = self.get_params()
        self.sliders["N_mpc"].setEnabled(False)
        self.sliders["N_mhe"].setEnabled(False)
        self.timer.start(int(p["dt_sim"] * 1.0e3))

    def stop_sim(self):
        self.timer.stop()
        self.sliders["N_mpc"].setEnabled(True)
        self.sliders["N_mhe"].setEnabled(True)

    def rebuild_solvers(self):
        p = self.get_params()
        print(
            f"Building solvers: N_mhe={p['N_mhe']} N_mpc={p['N_mpc']} max_iter={p['max_iter']} lbfgs={p['use_lbfgs']}"
        )
        (
            self.opti_mhe,
            self.X_mhe,
            self.W_mhe,
            self.Y_meas_param,
            self.U_past_param,
            self.X_prior_param,
            self.P_dyn_param,
            self.dt_mhe_param,
            self.Q_w_p,
            self.R_w_p,
            self.P_w_p,
            self.opti_mpc,
            self.X_mpc,
            self.U_mpc,
            self.X_cur_p,
            self.X_tgt_p,
            self.P_dyn_mpc,
            self.dt_mpc_p,
            self.Q_s_p,
            self.Q_v_p,
            self.Q_th_p,
            self.Q_om_p,
            self.R_F_p,
            self.max_pos_p,
            self.max_F_p,
            self.N_mpc_cur,
        ) = build_solvers(p["N_mhe"], p["N_mpc"], p["max_iter"], p["use_lbfgs"])
        self.N_mhe_cur = p["N_mhe"]

    def reset_sim(self):
        if hasattr(self, "timer"):
            self.timer.stop()
        self.rebuild_solvers()
        self.x_true = np.array([0.0, 0.0, 3.14159, 0.0])
        self.u_last = 0.0
        self.t_curr = 0.0
        self.Y_hist = []
        self.U_hist = []
        self.x_prior_val = np.copy(self.x_true)
        self.t_data = []
        self.hist = dict(s=[], v=[], theta=[], omega=[], F=[], t_solve=[])
        self.est_hist = dict(s=[], v=[], theta=[], omega=[], F=[])
        self.timer = QTimer()
        self.timer.timeout.connect(self.sim_step)

    def sim_step(self):
        t_start = time.time()
        p = self.get_params()
        dt_val = p["dt_sim"]
        p_dyn_val = np.array([p["M"], p["m"], p["l"], p["wind"]])
        res = f_rk4(self.x_true, self.u_last, p_dyn_val, dt_val)
        self.x_true = np.array(res.elements())
        y_meas = np.copy(self.x_true)
        y_meas[0] = y_meas[0] + np.random.normal(0.0, p["sig_s"])
        y_meas[1] = y_meas[1] + np.random.normal(0.0, p["sig_v"])
        y_meas[2] = y_meas[2] + np.random.normal(0.0, p["sig_th"])
        y_meas[3] = y_meas[3] + np.random.normal(0.0, p["sig_om"])
        self.Y_hist.append(np.copy(y_meas))
        self.U_hist.append([self.u_last])
        if len(self.Y_hist) > self.N_mhe_cur + 1:
            self.Y_hist.pop(0)
            self.U_hist.pop(0)
        x_est = np.copy(y_meas)
        if len(self.Y_hist) == self.N_mhe_cur + 1:
            Q_w_val = p["Q_w_scale"] * np.ones(4)
            P_w_val = p["P_w_scale"] * np.ones(4)
            r_base = p["R_w_scale"] * np.array([1.0e1, 1.0, 1.0e1, 1.0])
            R_w_val = np.array(
                [
                    r_base[0] * (1.0 if self.cb_s.isChecked() else 1.0e-4),
                    r_base[1] * (1.0 if self.cb_v.isChecked() else 1.0e-4),
                    r_base[2] * (1.0 if self.cb_th.isChecked() else 1.0e-4),
                    r_base[3] * (1.0 if self.cb_om.isChecked() else 1.0e-4),
                ]
            )
            self.opti_mhe.set_value(self.Y_meas_param, np.array(self.Y_hist).T)
            self.opti_mhe.set_value(
                self.U_past_param, np.array(self.U_hist[0 : self.N_mhe_cur]).T
            )
            self.opti_mhe.set_value(self.X_prior_param, self.x_prior_val[:, np.newaxis])
            self.opti_mhe.set_value(self.P_dyn_param, p_dyn_val[:, np.newaxis])
            self.opti_mhe.set_value(self.dt_mhe_param, dt_val)
            self.opti_mhe.set_value(self.Q_w_p, Q_w_val[:, np.newaxis])
            self.opti_mhe.set_value(self.R_w_p, R_w_val[:, np.newaxis])
            self.opti_mhe.set_value(self.P_w_p, P_w_val[:, np.newaxis])
            try:
                sol_mhe = self.opti_mhe.solve()
                X_res = sol_mhe.value(self.X_mhe)
                x_est = X_res[:, -1]
                self.x_prior_val = X_res[:, 1]
            except Exception as e:
                print(f"MHE: {e}")
        tgt = np.array([p["target_s"], 0.0, 0.0, 0.0])
        self.opti_mpc.set_value(self.X_cur_p, x_est[:, np.newaxis])
        self.opti_mpc.set_value(self.X_tgt_p, tgt[:, np.newaxis])
        self.opti_mpc.set_value(self.P_dyn_mpc, p_dyn_val[:, np.newaxis])
        self.opti_mpc.set_value(self.dt_mpc_p, dt_val)
        self.opti_mpc.set_value(self.Q_s_p, p["Q_s"])
        self.opti_mpc.set_value(self.Q_v_p, p["Q_v"])
        self.opti_mpc.set_value(self.Q_th_p, p["Q_theta"])
        self.opti_mpc.set_value(self.Q_om_p, p["Q_omega"])
        self.opti_mpc.set_value(self.R_F_p, p["R_F"])
        self.opti_mpc.set_value(self.max_pos_p, p["max_pos"])
        self.opti_mpc.set_value(self.max_F_p, p["max_force"])
        mpc_ok = False
        X_pred = None
        U_pred = None
        try:
            sol = self.opti_mpc.solve()
            self.u_last = sol.value(self.U_mpc)[0, 0]
            X_pred = sol.value(self.X_mpc)
            U_pred = sol.value(self.U_mpc)
            mpc_ok = True
        except Exception as e:
            print(f"MPC: {e}")
        t_calc = time.time() - t_start
        if t_calc > dt_val:
            self.lbl_delay.setText(
                f"WARNUNG: {(t_calc * 1000):.1f}ms > {(dt_val * 1000):.0f}ms!"
            )
            self.lbl_delay.setStyleSheet("font-size:13px;font-weight:bold;color:red")
        else:
            self.lbl_delay.setText(f"Rechenzeit: {(t_calc * 1000):.1f}ms (OK)")
            self.lbl_delay.setStyleSheet("font-size:13px;font-weight:bold;color:green")
        self.pend_widget.update_state(
            self.x_true, x_est, self.u_last, p["l"], p["max_pos"]
        )
        self.t_curr = self.t_curr + dt_val
        self.t_data.append(self.t_curr)
        self.hist["s"].append(self.x_true[0])
        self.est_hist["s"].append(x_est[0])
        self.hist["v"].append(self.x_true[1])
        self.est_hist["v"].append(x_est[1])
        self.hist["theta"].append(self.x_true[2])
        self.est_hist["theta"].append(x_est[2])
        self.hist["omega"].append(self.x_true[3])
        self.est_hist["omega"].append(x_est[3])
        self.hist["F"].append(self.u_last)
        self.est_hist["F"].append(self.u_last)
        self.hist["t_solve"].append(t_calc * 1.0e3)
        if len(self.t_data) > 200:
            self.t_data.pop(0)
            self.hist["s"].pop(0)
            self.hist["v"].pop(0)
            self.hist["theta"].pop(0)
            self.hist["omega"].pop(0)
            self.hist["F"].pop(0)
            self.hist["t_solve"].pop(0)
            self.est_hist["s"].pop(0)
            self.est_hist["v"].pop(0)
            self.est_hist["theta"].pop(0)
            self.est_hist["omega"].pop(0)
            self.est_hist["F"].pop(0)
        self.hist_curves["s"].setData(self.t_data, self.hist["s"])
        self.hist_curves["v"].setData(self.t_data, self.hist["v"])
        self.hist_curves["theta"].setData(self.t_data, self.hist["theta"])
        self.hist_curves["omega"].setData(self.t_data, self.hist["omega"])
        self.hist_curves["F"].setData(self.t_data, self.hist["F"])
        self.hist_curves["t_solve"].setData(self.t_data, self.hist["t_solve"])
        self.est_curves["s"].setData(self.t_data, self.est_hist["s"])
        self.est_curves["v"].setData(self.t_data, self.est_hist["v"])
        self.est_curves["theta"].setData(self.t_data, self.est_hist["theta"])
        self.est_curves["omega"].setData(self.t_data, self.est_hist["omega"])
        self.est_curves["F"].setData(self.t_data, self.est_hist["F"])
        if mpc_ok and (X_pred is not None):
            t_pred = np.linspace(
                self.t_curr, self.t_curr + self.N_mpc_cur * dt_val, self.N_mpc_cur + 1
            )
            self.pred_curves["s"].setData(t_pred, X_pred[0, :])
            self.pred_curves["v"].setData(t_pred, X_pred[1, :])
            self.pred_curves["theta"].setData(t_pred, X_pred[2, :])
            self.pred_curves["omega"].setData(t_pred, X_pred[3, :])
            self.pred_curves["F"].setData(t_pred[0:-1], U_pred[0, :])


app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec())
