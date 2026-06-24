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
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor

x_sym = ca.MX.sym("x", 4)
u_sym = ca.MX.sym("u", 1)
s_ = x_sym[0]
v_ = x_sym[1]
th_ = x_sym[2]
om_ = x_sym[3]
sin_th = ca.sin(th_)
cos_th = ca.cos(th_)
p_dyn = ca.MX.sym("p", 4)
M_ = p_dyn[0]
m_ = p_dyn[1]
l_ = p_dyn[2]
w_ = p_dyn[3]
denom = M_ + m_ * (1 - (cos_th**2))
s_dot = v_
v_dot = (u_sym + w_ * cos_th + m_ * sin_th * (l_ * (om_**2) + 9.81 * cos_th)) / denom
th_dot = om_
om_dot = (
    (-1 * (u_sym + w_ * cos_th) * cos_th)
    - (m_ * l_ * (om_**2) * sin_th * cos_th)
    - ((M_ + m_) * 9.81 * sin_th)
) / (l_ * denom)
f_ode = ca.Function(
    "f_ode", [x_sym, u_sym, p_dyn], [ca.vertcat(s_dot, v_dot, th_dot, om_dot)]
)
dt_sym = ca.MX.sym("dt", 1)
rk4_k1 = f_ode(x_sym, u_sym, p_dyn)
rk4_k2 = f_ode(x_sym + (dt_sym / 2) * rk4_k1, u_sym, p_dyn)
rk4_k3 = f_ode(x_sym + (dt_sym / 2) * rk4_k2, u_sym, p_dyn)
rk4_k4 = f_ode(x_sym + dt_sym * rk4_k3, u_sym, p_dyn)
x_next_expr = x_sym + (dt_sym / 6) * (rk4_k1 + 2 * rk4_k2 + 2 * rk4_k3 + rk4_k4)
f_rk4 = ca.Function("f_rk4", [x_sym, u_sym, p_dyn, dt_sym], [x_next_expr])
N_mhe = 15
opti_mhe = ca.Opti()
X_mhe = opti_mhe.variable(4, N_mhe + 1)
W_mhe = opti_mhe.variable(4, N_mhe)
Y_meas_param = opti_mhe.parameter(4, N_mhe + 1)
U_past_param = opti_mhe.parameter(1, N_mhe)
X_prior_param = opti_mhe.parameter(4, 1)
P_dyn_param = opti_mhe.parameter(4, 1)
dt_mhe_param = opti_mhe.parameter(1, 1)
Q_w = opti_mhe.parameter(4, 1)
R_w = opti_mhe.parameter(4, 1)
P_w = opti_mhe.parameter(4, 1)
f_rk4_map_mhe = f_rk4.map(N_mhe)
p_stacked_mhe = ca.repmat(P_dyn_param, 1, N_mhe)
dt_stacked_mhe = ca.repmat(dt_mhe_param, 1, N_mhe)
X_next_calc_mhe = f_rk4_map_mhe(
    X_mhe[:, 0:N_mhe], U_past_param, p_stacked_mhe, dt_stacked_mhe
)
opti_mhe.subject_to(X_mhe[:, 1 : N_mhe + 1] == X_next_calc_mhe + W_mhe)
cost_mhe = (
    ca.sum1(P_w * (((X_mhe[:, 0]) - X_prior_param) ** 2))
    + ca.sum2(ca.sum1(Q_w * (W_mhe**2)))
    + ca.sum2(ca.sum1(R_w * ((Y_meas_param - X_mhe) ** 2)))
)
opti_mhe.minimize(cost_mhe)
opti_mhe.solver("ipopt", dict(print_time=False), dict(print_level=0, sb="yes"))
N_mpc_sym = 40
opti_mpc = ca.Opti()
X_mpc = opti_mpc.variable(4, N_mpc_sym + 1)
U_mpc = opti_mpc.variable(1, N_mpc_sym)
X_cur_param = opti_mpc.parameter(4, 1)
X_tgt_param = opti_mpc.parameter(4, 1)
P_dyn_mpc = opti_mpc.parameter(4, 1)
dt_mpc_param = opti_mpc.parameter(1, 1)
Q_s_p = opti_mpc.parameter()
Q_v_p = opti_mpc.parameter()
Q_th_p = opti_mpc.parameter()
Q_om_p = opti_mpc.parameter()
R_F_p = opti_mpc.parameter()
max_pos_p = opti_mpc.parameter()
max_F_p = opti_mpc.parameter()
f_rk4_map_mpc = f_rk4.map(N_mpc_sym)
p_stacked_mpc = ca.repmat(P_dyn_mpc, 1, N_mpc_sym)
dt_stacked_mpc = ca.repmat(dt_mpc_param, 1, N_mpc_sym)
X_next_calc_mpc = f_rk4_map_mpc(
    X_mpc[:, 0:N_mpc_sym], U_mpc, p_stacked_mpc, dt_stacked_mpc
)
opti_mpc.subject_to(X_mpc[:, 0] == X_cur_param)
opti_mpc.subject_to(X_mpc[:, 1 : N_mpc_sym + 1] == X_next_calc_mpc)
opti_mpc.subject_to(opti_mpc.bounded(-1 * max_pos_p, X_mpc[0, :], max_pos_p))
opti_mpc.subject_to(opti_mpc.bounded(-1 * max_F_p, U_mpc, max_F_p))
err_mpc = X_mpc - ca.repmat(X_tgt_param, 1, N_mpc_sym + 1)
Q_diag = ca.vertcat(Q_s_p, Q_v_p, Q_th_p, Q_om_p)
cost_mpc = ca.sum2(ca.sum1(Q_diag * (err_mpc**2))) + R_F_p * ca.sum2(U_mpc**2)
opti_mpc.minimize(cost_mpc)
opti_mpc.solver("ipopt", dict(print_time=False), dict(print_level=0, sb="yes"))


class PendulumWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setMinimumSize(400, 300)
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
            s_ = state[0]
            theta = state[2]
            cart_x = cx + s_ * scale
            cart_w = 0.4 * scale
            cart_h = 0.2 * scale
            painter.setBrush(QBrush(cart_col))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(cart_x - (cart_w / 2), cy - (cart_h / 2), cart_w, cart_h)
            px = cart_x + scale * self.l * np.sin(theta)
            py = cy - (scale * self.l * np.cos(theta))
            painter.setPen(QPen(pole_col, 6))
            painter.drawLine(cart_x, cy, px, py)
            painter.setBrush(QBrush(pole_col))
            painter.setPen(QPen(pole_col, 1))
            painter.drawEllipse(px - 8, py - 8, 16, 16)

        draw_pend(self.x_est, QColor(180, 100, 30), QColor(255, 165, 0))
        draw_pend(self.x_true, QColor(100, 150, 250), QColor(250, 100, 100))
        if np.abs(self.force) > 0.1:
            ax = cx + self.x_true[0] * scale
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            painter.drawLine(ax, cy, int(ax + 3.0 * self.force), cy)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Inverted Pendulum - MHE + MPC")
        self.resize(1400, 900)
        central = QWidget()
        main_layout = QHBoxLayout(central)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        self.setCentralWidget(central)
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.pend_widget = PendulumWidget()
        left_layout.addWidget(self.pend_widget)
        self.lbl_delay = QLabel("Rechenzeit: OK")
        self.lbl_delay.setStyleSheet("font-size:13px;font-weight:bold;color:green")
        left_layout.addWidget(self.lbl_delay)
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset")
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

        def add_slider(name, label, min_v, max_v, default, scale, row, tip):
            sl = QSlider(Qt.Horizontal)
            lbl = QLabel(f"{label}: {default * scale:.2f}")
            sl.setRange(min_v, max_v)
            sl.setValue(default)
            sl.setToolTip(tip)
            slider_layout.addWidget(lbl, row, 0)
            slider_layout.addWidget(sl, row, 1)
            sl.valueChanged.connect(
                lambda: lbl.setText(f"{label}: {sl.value() * scale:.2f}")
            )
            self.sliders[name] = sl

        add_slider("M", "Wagenmasse [kg]", 1, 50, 10, 0.1, 0, "Cart mass")
        add_slider("m", "Pendelmasse [kg]", 1, 20, 1, 0.1, 1, "Pole mass")
        add_slider("l", "Pendellänge [m]", 1, 20, 5, 0.1, 2, "Pole length")
        add_slider(
            "wind", "Wind [N]", -300, 300, 0, 0.1, 3, "Horizontal wind disturbance"
        )
        add_slider("Q_s", "Gewicht s", 0, 200, 10, 1.0, 4, "MPC position cost")
        add_slider("Q_v", "Gewicht v", 0, 100, 1, 1.0, 5, "MPC velocity cost")
        add_slider("Q_theta", "Gewicht theta", 0, 500, 100, 1.0, 6, "MPC angle cost")
        add_slider("Q_omega", "Gewicht omega", 0, 100, 1, 1.0, 7, "MPC ang. vel. cost")
        add_slider("R_F", "Gewicht Kraft", 1, 200, 10, 1.0e-2, 8, "MPC force cost")
        add_slider(
            "target_s", "Ziel-s [m]", -100, 100, 10, 0.1, 9, "Target cart position"
        )
        add_slider("max_pos", "Schiene [m]", 10, 200, 50, 0.1, 10, "Position limit")
        add_slider("max_force", "Max Kraft [N]", 10, 300, 150, 0.1, 11, "Force limit")
        add_slider("dt_sim", "Sim-dt [ms]", 1, 100, 20, 1.0, 12, "Simulation step ms")
        sensor_box = QGroupBox("Sensor aktiv (MHE)")
        sb_layout = QVBoxLayout()
        self.cb_s = QCheckBox("s (Position)")
        self.cb_v = QCheckBox("v (Geschw.)")
        self.cb_th = QCheckBox("theta (Winkel)")
        self.cb_om = QCheckBox("omega (Winkelgeschw.)")
        self.cb_s.setChecked(True)
        self.cb_th.setChecked(True)
        sb_layout.addWidget(self.cb_s)
        sb_layout.addWidget(self.cb_v)
        sb_layout.addWidget(self.cb_th)
        sb_layout.addWidget(self.cb_om)
        sensor_box.setLayout(sb_layout)
        left_layout.addWidget(sensor_box)
        self.plot_layout = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.plot_layout)
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
        ax.setLabel("left", "v [m/s]")
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
        ax.setLabel("left", "F [N]")
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
            target_s=(self.sliders["target_s"].value()) / 1.0e1,
            max_pos=(self.sliders["max_pos"].value()) / 1.0e1,
            max_force=(self.sliders["max_force"].value()) / 1.0e1,
            dt_sim=(self.sliders["dt_sim"].value()) / 1.0e3,
        )

    def start_sim(self):
        p = self.get_params()
        self.timer.start(int(p["dt_sim"] * 1.0e3))

    def stop_sim(self):
        self.timer.stop()

    def reset_sim(self):
        self.stop_sim()
        self.x_true = np.array([0.0, 0.0, 3.14159, 0.0])
        self.u_last = 0.0
        self.t_curr = 0.0
        self.Y_hist = []
        self.U_hist = []
        self.x_prior_val = np.copy(self.x_true)
        self.x_est_last = np.copy(self.x_true)
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
        res_true = f_rk4(self.x_true, self.u_last, p_dyn_val, dt_val)
        self.x_true = np.array(res_true.elements())
        y_meas = np.copy(self.x_true)
        noise_std = 2.0e-2
        y_meas[0] = y_meas[0] + np.random.normal(0.0, noise_std)
        y_meas[1] = y_meas[1] + np.random.normal(0.0, noise_std)
        y_meas[2] = y_meas[2] + np.random.normal(0.0, noise_std)
        y_meas[3] = y_meas[3] + np.random.normal(0.0, noise_std)
        self.Y_hist.append(np.copy(y_meas))
        self.U_hist.append([self.u_last])
        if len(self.Y_hist) > N_mhe + 1:
            self.Y_hist.pop(0)
            self.U_hist.pop(0)
        x_est = np.copy(y_meas)
        if len(self.Y_hist) == N_mhe + 1:
            Y_mat = np.array(self.Y_hist).T
            U_mat = np.array(self.U_hist[0:N_mhe]).T
            opti_mhe.set_value(Y_meas_param, Y_mat)
            opti_mhe.set_value(U_past_param, U_mat)
            opti_mhe.set_value(X_prior_param, self.x_prior_val[:, np.newaxis])
            opti_mhe.set_value(P_dyn_param, p_dyn_val[:, np.newaxis])
            opti_mhe.set_value(dt_mhe_param, dt_val)
            opti_mhe.set_value(Q_w, np.array([1.0, 1.0, 1.0, 1.0]))
            opti_mhe.set_value(P_w, np.array([1.0e1, 1.0e1, 1.0e1, 1.0e1]))
            r_vec = [
                1.0e2 if self.cb_s.isChecked() else 1.0e-3,
                1.0e2 if self.cb_v.isChecked() else 1.0e-3,
                1.0e3 if self.cb_th.isChecked() else 1.0e-3,
                1.0e2 if self.cb_om.isChecked() else 1.0e-3,
            ]
            opti_mhe.set_value(R_w, np.array(r_vec))
            try:
                sol_mhe = opti_mhe.solve()
                X_res = sol_mhe.value(X_mhe)
                x_est = X_res[:, -1]
                self.x_prior_val = X_res[:, 1]
            except Exception as e:
                print(f"MHE failed: {e}")
        self.x_est_last = x_est
        target_s = p["target_s"]
        tgt_state = np.array([target_s, 0.0, 0.0, 0.0])
        x_est_col = x_est[:, np.newaxis]
        opti_mpc.set_value(X_cur_param, x_est_col)
        opti_mpc.set_value(X_tgt_param, tgt_state[:, np.newaxis])
        opti_mpc.set_value(P_dyn_mpc, p_dyn_val[:, np.newaxis])
        opti_mpc.set_value(dt_mpc_param, dt_val)
        opti_mpc.set_value(Q_s_p, p["Q_s"])
        opti_mpc.set_value(Q_v_p, p["Q_v"])
        opti_mpc.set_value(Q_th_p, p["Q_theta"])
        opti_mpc.set_value(Q_om_p, p["Q_omega"])
        opti_mpc.set_value(R_F_p, p["R_F"])
        opti_mpc.set_value(max_pos_p, p["max_pos"])
        opti_mpc.set_value(max_F_p, p["max_force"])
        mpc_success = False
        X_pred_val = None
        U_pred_val = None
        try:
            sol_mpc = opti_mpc.solve()
            self.u_last = sol_mpc.value(U_mpc)[0, 0]
            X_pred_val = sol_mpc.value(X_mpc)
            U_pred_val = sol_mpc.value(U_mpc)
            mpc_success = True
        except Exception as e:
            print(f"MPC failed: {e}")
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
        if mpc_success and (X_pred_val is not None):
            t_pred = np.linspace(
                self.t_curr, self.t_curr + N_mpc_sym * dt_val, N_mpc_sym + 1
            )
            self.pred_curves["s"].setData(t_pred, X_pred_val[0, :])
            self.pred_curves["v"].setData(t_pred, X_pred_val[1, :])
            self.pred_curves["theta"].setData(t_pred, X_pred_val[2, :])
            self.pred_curves["omega"].setData(t_pred, X_pred_val[3, :])
            self.pred_curves["F"].setData(t_pred[0:-1], U_pred_val[0, :])


app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec())
