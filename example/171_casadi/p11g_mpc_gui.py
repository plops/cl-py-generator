import sys
import time
import numpy as np
import casadi as ca
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from pyqtgraph import PlotWidget, plot, mkPen

m_cart = 1.0
m_pend = 0.1
l = 0.5
g = 9.81
dt = 2.0e-2
N_mpc = 40
N_mhe = 15
x_sym = ca.MX.sym("x", 4)
u_sym = ca.MX.sym("u", 1)
s_ = x_sym[0]
v_ = x_sym[1]
th_ = x_sym[2]
om_ = x_sym[3]
sin_th = ca.sin(th_)
cos_th = ca.cos(th_)
denom = m_cart + m_pend * (1 - (cos_th**2))
s_dot = v_
v_dot = (u_sym + m_pend * sin_th * (l * (om_**2) + g * cos_th)) / denom
th_dot = om_
om_dot = (
    (-1 * u_sym * cos_th)
    - (m_pend * l * (om_**2) * sin_th * cos_th)
    - ((m_cart + m_pend) * g * sin_th)
) / (l * denom)
f_ode = ca.Function("f_ode", [x_sym, u_sym], [ca.vertcat(s_dot, v_dot, th_dot, om_dot)])
k1 = f_ode(x_sym, u_sym)
k2 = f_ode(x_sym + (dt / 2) * k1, u_sym)
k3 = f_ode(x_sym + (dt / 2) * k2, u_sym)
k4 = f_ode(x_sym + dt * k3, u_sym)
x_next = x_sym + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
f_rk4 = ca.Function("f_rk4", [x_sym, u_sym], [x_next])
opti_mhe = ca.Opti()
X_mhe = opti_mhe.variable(4, N_mhe + 1)
W_mhe = opti_mhe.variable(4, N_mhe)
Y_meas_param = opti_mhe.parameter(4, N_mhe + 1)
U_past_param = opti_mhe.parameter(1, N_mhe)
X_prior_param = opti_mhe.parameter(4, 1)
Q_w = opti_mhe.parameter(4, 1)
R_w = opti_mhe.parameter(4, 1)
P_w = opti_mhe.parameter(4, 1)
f_rk4_map = f_rk4.map(N_mhe)
X_prev_slice = X_mhe[:, 0:N_mhe]
X_next_calc = f_rk4_map(X_prev_slice, U_past_param)
X_next_slice = X_mhe[:, 1 : N_mhe + 1]
opti_mhe.subject_to(X_next_slice == X_next_calc + W_mhe)
cost_mhe = 0
cost_mhe = cost_mhe + ca.sum1(P_w * (((X_mhe[:, 0]) - X_prior_param) ** 2))
cost_mhe = cost_mhe + ca.sum2(ca.sum1(Q_w * (W_mhe**2)))
cost_mhe = cost_mhe + ca.sum2(ca.sum1(R_w * ((Y_meas_param - X_mhe) ** 2)))
opti_mhe.minimize(cost_mhe)
opti_mhe.solver("ipopt", dict(print_time=False), dict(print_level=0))
opti_mpc = ca.Opti()
X_mpc = opti_mpc.variable(4, N_mpc + 1)
U_mpc = opti_mpc.variable(1, N_mpc)
X_current_param = opti_mpc.parameter(4, 1)
f_rk4_map_mpc = f_rk4.map(N_mpc)
X_next_calc_mpc = f_rk4_map_mpc(X_mpc[:, 0:N_mpc], U_mpc)
opti_mpc.subject_to(X_mpc[:, 1 : N_mpc + 1] == X_next_calc_mpc)
opti_mpc.subject_to(X_mpc[:, 0] == X_current_param)
opti_mpc.subject_to(U_mpc[:, :] <= 2.0e1)
opti_mpc.subject_to(U_mpc[:, :] >= -2.0e1)
cost_mpc = (
    ca.sum2(10 * ((X_mpc[0, :]) ** 2))
    + ca.sum2(1 * ((X_mpc[1, :]) ** 2))
    + ca.sum2(100 * ((X_mpc[2, :]) ** 2))
    + ca.sum2(1 * ((X_mpc[3, :]) ** 2))
    + ca.sum2(1.0e-2 * (U_mpc**2))
)
opti_mpc.minimize(cost_mpc)
opti_mpc.solver("ipopt", dict(print_time=False), dict(print_level=0))


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Inverted Pendulum - MHE & MPC")
        main_widget = QWidget()
        layout = QVBoxLayout()
        ui_layout = QHBoxLayout()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.lbl_delay = QLabel("Rechenzeit: OK")
        self.lbl_delay.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: green;"
        )
        layout.addWidget(self.lbl_delay)
        sensor_group = QGroupBox("Sensoren Aktiv")
        sg_layout = QVBoxLayout()
        self.cb_s = QCheckBox("Position (s)")
        self.cb_v = QCheckBox("Geschw. (v)")
        self.cb_th = QCheckBox("Winkel (th)")
        self.cb_om = QCheckBox("Winkelgeschw. (om)")
        self.cb_s.setChecked(True)
        self.cb_th.setChecked(True)
        self.cb_v.setChecked(False)
        self.cb_om.setChecked(False)
        sg_layout.addWidget(self.cb_s)
        sg_layout.addWidget(self.cb_v)
        sg_layout.addWidget(self.cb_th)
        sg_layout.addWidget(self.cb_om)
        sensor_group.setLayout(sg_layout)
        ui_layout.addWidget(sensor_group)
        noise_group = QGroupBox("Sensor Rauschen (Sigma)")
        ng_layout = QVBoxLayout()
        self.sl_noise = QSlider(Qt.Horizontal)
        self.sl_noise.setRange(0, 100)
        self.sl_noise.setValue(10)
        ng_layout.addWidget(self.sl_noise)
        noise_group.setLayout(ng_layout)
        ui_layout.addWidget(noise_group)
        layout.addLayout(ui_layout)
        self.plot_widget = PlotWidget()
        self.curve_true = self.plot_widget.plot(
            [], [], pen=mkPen(color="g", width=2, name="True")
        )
        self.curve_est = self.plot_widget.plot(
            [], [], pen=mkPen(color="r", width=2, style=Qt.DashLine, name="MHE Est")
        )
        layout.addWidget(self.plot_widget)
        self.x_true = np.array([0.0, 0.0, 0.1, 0.0])
        self.u_last = 0.0
        self.Y_hist = []
        self.U_hist = []
        self.x_prior_val = np.copy(self.x_true)
        self.t_data = []
        self.th_true_data = []
        self.th_est_data = []
        self.t_curr = 0.0
        self.timer = QTimer()
        self.timer.timeout.connect = self.sim_step
        self.timer.start(int(dt * 1000))

    def sim_step(self):
        t_start = time.time()
        res_true = f_rk4(self.x_true, self.u_last)
        self.x_true = np.array(res_true.elements())
        noise_std = self.sl_noise.value() / 1.0e2
        y_meas = np.copy(self.x_true)
        y_meas[0] = y_meas[0] + np.random.normal(0.0, noise_std)
        y_meas[1] = y_meas[1] + np.random.normal(0.0, noise_std)
        y_meas[2] = y_meas[2] + np.random.normal(0.0, noise_std)
        y_meas[3] = y_meas[3] + np.random.normal(0.0, noise_std)
        self.Y_hist.append(y_meas)
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
            opti_mhe.set_value(X_prior_param, self.x_prior_val)
            opti_mhe.set_value(Q_w, np.array([1.0, 1.0, 1.0, 1.0]))
            opti_mhe.set_value(P_w, np.array([1.0e1, 1.0e1, 1.0e1, 1.0e1]))
            r_weights = [
                1.0e2 if self.cb_s.isChecked() else 0.0,
                1.0e2 if self.cb_v.isChecked() else 0.0,
                1.0e3 if self.cb_th.isChecked() else 0.0,
                1.0e2 if self.cb_om.isChecked() else 0.0,
            ]
            opti_mhe.set_value(R_w, np.array(r_weights))
            try:
                sol_mhe = opti_mhe.solve()
                X_res = sol_mhe.value(X_mhe)
                x_est = X_res[:, -1]
                self.x_prior_val = X_res[:, 1]
            except Exception:
                e
                print("MHE Failed!")
        opti_mpc.set_value(X_current_param, x_est)
        try:
            sol_mpc = opti_mpc.solve()
            self.u_last = sol_mpc.value(U_mpc)[0]
        except Exception:
            e
            print("MPC Failed!")
        t_end = time.time()
        t_calc = t_end - t_start
        if t_calc > dt:
            self.lbl_delay.setText(
                f"GEFAHR: Rechenzeit {(t_calc * 1000):.1f}ms übersteigt Sim-Takt {(dt * 1000)}ms!"
            )
            self.lbl_delay.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: red;"
            )
        else:
            self.lbl_delay.setText(f"Rechenzeit: {(t_calc * 1000):.1f}ms (OK)")
            self.lbl_delay.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: green;"
            )
        self.t_curr = self.t_curr + dt
        self.t_data.append(self.t_curr)
        self.th_true_data.append(self.x_true[2])
        self.th_est_data.append(x_est[2])
        if len(self.t_data) > 100:
            self.t_data.pop(0)
            self.th_true_data.pop(0)
            self.th_est_data.pop(0)
        self.curve_true.setData(self.t_data, self.th_true_data)
        self.curve_est.setData(self.t_data, self.th_est_data)


app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec_())
