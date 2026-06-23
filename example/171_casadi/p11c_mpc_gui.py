from __future__ import annotations
from casadi import *
import numpy as np
import time as time
import sys as sys
import pg as pyqtgraph
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QGridLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
# =========================================================================================
#  INVERTED PENDULUM MPC (MODEL PREDICTIVE CONTROL) DASHBOARD
# =========================================================================================
#  PURPOSE AND SCOPE:
#  This application demonstrates real-time Model Predictive Control of a nonlinear, 
#  underactuated mechanical system (an inverted pendulum on a cart). 
#  The GUI provides interactive sliders to change physical parameters (like mass and wind) 
#  and tuning parameters (like cost weights) on the fly, visualizing the solver's 
#  future predictions versus the actual system trajectory.
# 
#  PHYSICAL MODEL & DYNAMICS:
#  We consider a cart of mass M moving on a 1D track, and a pendulum of mass m and length l 
#  attached to it. The system is underactuated: we only control the horizontal force F 
#  on the cart, but we want to control both the cart's position (s) and the pendulum's 
#  angle (theta). The state vector is x = [s, v, theta, omega].
#  Using Lagrangian mechanics, the nonlinear ODEs are derived considering kinetic and 
#  potential energies. A simulated wind force applies a disturbing torque.
# 
#  CONTROL THEORY (MODEL PREDICTIVE CONTROL):
#  MPC solves a finite-horizon optimal control problem (OCP) at every sampling time. 
#  It calculates a trajectory of control inputs that minimizes a cost function 
#  (e.g., deviations from the target state) while satisfying constraints (e.g., track limits). 
#  Only the very first control input is applied to the real system. In the next step, 
#  the horizon shifts and the problem is solved again (Receding Horizon Control).
# 
#  MATHEMATICS OF DIRECT COLLOCATION:
#  Instead of integrating the ODEs step-by-step (like in Multiple Shooting), Direct 
#  Collocation discretizes the state trajectory using polynomials (e.g., Lagrange polynomials). 
#  The states at specific collocation points (here: Radau points) become optimization variables. 
#  The system dynamics are enforced as equality constraints mapping the derivative of 
#  the polynomials to the vector field f(x,u). This transforms the continuous-time optimal 
#  control problem into a huge but sparse Nonlinear Programming (NLP) problem.
# 
#  CASADI & IPOPT:
#  CasADi (Computer algebra system for automatic differentiation) is used to mathematically 
#  formulate the NLP. It computes exact gradients and sparse Jacobians automatically using AD. 
#  IPOPT (Interior Point OPTimizer) is the backend solver. To achieve real-time performance 
#  (under 33ms), we use 'Warm-Starting': the optimal trajectory from the previous time step 
#  is passed as an initial guess to IPOPT, reducing the number of required iterations to just 1-3.
# =========================================================================================
class PendulumMPC:
    def __init__(self):
        self.opti=Opti()
        self.nx=4
        self.nu=1
        self.N=20
        self.T_horizon=1.    
        self.h=self.T_horizon/self.N
        self.d=3
        # Parameter für Physik (M: Wagenmasse, m: Pendelmasse, l: Länge, wind: Störkraft)
        self.M_p=self.opti.parameter()
        self.m_p=self.opti.parameter()
        self.l_p=self.opti.parameter()
        self.wind_p=self.opti.parameter()
        self.Q_s=self.opti.parameter()
        self.Q_v=self.opti.parameter()
        self.Q_theta=self.opti.parameter()
        self.Q_omega=self.opti.parameter()
        self.R_F=self.opti.parameter()
        self.max_pos=self.opti.parameter()
        self.max_force=self.opti.parameter()
        # Symbolische Variablen für die Dynamik (dx/dt = f(x,u))
        x=SX.sym("x", self.nx)
        u=SX.sym("u", self.nu)
        s_=x[0]
        v_=x[1]
        theta_=x[2]
        omega_=x[3]
        F_=u
        sin_theta=np.sin(theta_)
        cos_theta=np.cos(theta_)
        den=self.M_p+self.m_p*(1.    -(cos_theta*cos_theta))
        F_total=F_+self.wind_p*cos_theta
        ds=v_
        dv=(F_total+self.m_p*self.l_p*omega_*omega_*sin_theta+self.m_p*9.81    *cos_theta*sin_theta)/den
        dtheta=omega_
        domega=(( -1.    *F_total*cos_theta)-(self.m_p*self.l_p*omega_*omega_*sin_theta*cos_theta)-((self.M_p+self.m_p)*9.81    *sin_theta))/(self.l_p*den)
        self.f_ode=Function("f_ode", [x, u], [vertcat(ds, dv, dtheta, domega)])
        # Lagrange Collocation Polynome via Radau-Punkten berechnen
        tau_root=np.append(0.    , collocation_points(self.d, "radau"))
        self.C=np.zeros((self.d+1,self.d+1,))
        self.D=np.zeros(self.d+1)
        for j in range(self.d+1):
            p=np.poly1d([1.    ])
            for r in range(self.d+1):
                if r!=j:
                    p=p*(np.poly1d([1.    , -tau_root[r]])/((tau_root[j])-(tau_root[r])))
            self.D[j]=p(1.    )
            pder=np.polyder(p)
            for r in range(self.d+1):
                self.C[j,r]=pder(tau_root[r])
        # Decision Variables für IPOPT (X: Knoten, Xc: Collocation-Punkte, U: Stellgröße)
        self.X=self.opti.variable(self.nx, self.N+1)
        self.Xc=[]
        self.U=self.opti.variable(self.nu, self.N)
        self.current_x=self.opti.parameter(self.nx)
        self.target_x=self.opti.parameter(self.nx)
        for k in range(self.N):
            Xc_k=[]
            for r in range(self.d):
                Xc_k.append(self.opti.variable(self.nx))
            self.Xc.append(Xc_k)
        # Constraints anwenden: Startzustand, Grenzen und Collocation-Dynamik
        self.opti.subject_to(self.X[:,0]==self.current_x)
        self.opti.subject_to(self.opti.bounded( -1.    *self.max_pos, self.X[0,:], self.max_pos))
        self.opti.subject_to(self.opti.bounded( -1.    *self.max_force, self.U, self.max_force))
        for k in range(self.N):
            Xk=self.X[:,k]
            x_end=self.D[0]*Xk
            for j in range(1, self.d+1):
                xp=self.C[0,j]*Xk
                for r in range(self.d):
                    xp=xp+self.C[r+1,j]*self.Xc[k][r]
                f_eval=self.f_ode(self.Xc[k][j-1], self.U[:,k])
                self.opti.subject_to(xp==self.h*f_eval)
            for r in range(self.d):
                x_end=x_end+self.D[r+1]*self.Xc[k][r]
            self.opti.subject_to(self.X[:,k+1]==x_end)
        # Kostenfunktion (Objective): Abweichung vom Ziel und Stellenergie minimieren
        cost=0.    
        Q=np.diag(vertcat(self.Q_s, self.Q_v, self.Q_theta, self.Q_omega))
        for k in range(self.N):
            err=(self.X[:,k])-self.target_x
            cost=cost+mtimes(mtimes(err.T, Q), err)+self.R_F*((self.U[0,k])**2)
        err_term=(self.X[:,self.N])-self.target_x
        cost=cost+1.0e+1*mtimes(mtimes(err_term.T, Q), err_term)
        self.opti.minimize(cost)
        # IPOPT Konfiguration (ohne JIT Komplexität, Python API reicht aus)
        self.opti.solver("ipopt", {("print_time"):(False)}, {("print_level"):(0), ("sb"):("yes")})
        self.opti.set_initial(self.X, np.linspace(np.array([0.    , 0.    , np.pi, 0.    ]), np.array([1.    , 0.    , 0.    , 0.    ]), self.N+1).T)
        self.sol=None
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
        if self.sol!=None:
            X_res=self.sol.value(self.X)
            U_res=self.sol.value(self.U)
            X_guess=np.hstack((X_res[:,1:],X_res[:,-1:],))
            U_guess=np.append(U_res[1:], U_res[-1])
            self.opti.set_initial(self.X, X_guess)
            self.opti.set_initial(self.U, U_guess)
        t0=time.time()
        try:
            self.sol=self.opti.solve()
        except except:
            e(Exception)
            return (U_guess[0],X_res,U_res,time.time()-t0,False,)
        t_solve=time.time()-t0
        X_res=self.sol.value(self.X)
        U_res=self.sol.value(self.U)
        return (U_res[0],X_res,U_res,t_solve,True,)
class PendulumWidget(QWidget):
    def __init__(self):
        super.__init__()
        self.setMinimumSize(400, 400)
        self.state=np.array([0.    , 0.    , np.pi, 0.    ])
        self.l=0.5    
        self.force=0.    
        self.wind=0.    
        self.max_pos=2.    
    def update_state(self, state, force, wind, l, max_pos):
        self.state=state
        self.force=force
        self.wind=wind
        self.l=l
        self.max_pos=max_pos
        self.update()
    def paintEvent(self, event):
        painter=QPainter(self)
        w=self.width()
        h=self.height()
        cx=w/2
        cy=h/2
        scale=w/(2.5    *self.max_pos)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        painter.setPen(pg.mkPen("gray", width=4))
        painter.drawLine(cx-(scale*self.max_pos), cy, cx+scale*self.max_pos, cy)
        s=self.state[0]
        theta=self.state[2]
        cart_w=0.4    *scale
        cart_h=0.2    *scale
        cart_x=cx+s*scale
        cart_y=cy
        painter.setBrush(QBrush(QColor(100, 150, 250)))
        painter.setPen(pg.mkPen("w", width=2))
        painter.drawRect(cart_x-(cart_w/2), cart_y-(cart_h/2), cart_w, cart_h)
        pend_x=cart_x+scale*self.l*np.sin(theta)
        pend_y=cart_y-(scale*self.l*np.cos(theta))
        painter.setPen(pg.mkPen("w", width=6))
        painter.drawLine(cart_x, cart_y, pend_x, pend_y)
        painter.setBrush(QBrush(QColor(250, 100, 100)))
        painter.drawEllipse(pend_x-10, pend_y-10, 20, 20)
        if np.abs(self.force)>0.1    :
            painter.setPen(pg.mkPen("g", width=3))
            painter.drawLine(cart_x, cart_y, cart_x+5.    *self.force, cart_y)
class MainWindow(QMainWindow):
    def __init__(self):
        super.__init__()
        self.setWindowTitle("MPC Inverted Pendulum")
        self.resize(1400, 900)
        central_widget=QWidget()
        main_layout=QHBoxLayout(central_widget)
        left_layout=QVBoxLayout()
        right_layout=QVBoxLayout()
        self.setCentralWidget(central_widget)
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.pendulum_widget=PendulumWidget()
        self.plot_layout=pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.pendulum_widget)
        right_layout.addWidget(self.plot_layout)
        self.plots={}
        self.history_curves={}
        self.pred_curves={}
        # Plots dynamisch generieren mit Lisp Macro Expand
        ax=self.plot_layout.addPlot(row=0, col=0, title="Wagenposition")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Position [m]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["s"]=ax
        self.history_curves["s"]=ax.plot(pen=pg.mkPen("w", width=2))
        if "s"!="t_solve":
            self.pred_curves["s"]=ax.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        ax=self.plot_layout.addPlot(row=1, col=0, title="Pendelwinkel")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Winkel [rad]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["theta"]=ax
        self.history_curves["theta"]=ax.plot(pen=pg.mkPen("w", width=2))
        if "theta"!="t_solve":
            self.pred_curves["theta"]=ax.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        ax=self.plot_layout.addPlot(row=2, col=0, title="Wagengeschwindigkeit")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "v [m/s]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["v"]=ax
        self.history_curves["v"]=ax.plot(pen=pg.mkPen("w", width=2))
        if "v"!="t_solve":
            self.pred_curves["v"]=ax.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        ax=self.plot_layout.addPlot(row=0, col=1, title="Winkelgeschw.")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "omega [rad/s]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["omega"]=ax
        self.history_curves["omega"]=ax.plot(pen=pg.mkPen("w", width=2))
        if "omega"!="t_solve":
            self.pred_curves["omega"]=ax.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        ax=self.plot_layout.addPlot(row=1, col=1, title="Aktuatorkraft")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Kraft [N]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["F"]=ax
        self.history_curves["F"]=ax.plot(pen=pg.mkPen("w", width=2))
        if "F"!="t_solve":
            self.pred_curves["F"]=ax.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        ax=self.plot_layout.addPlot(row=2, col=1, title="Solver Rechenzeit")
        ax.showGrid(x=True, y=True)
        ax.setLabel("left", "Zeit [ms]")
        ax.setLabel("bottom", "Zeit [s]")
        self.plots["t_solve"]=ax
        self.history_curves["t_solve"]=ax.plot(pen=pg.mkPen("w", width=2))
        if "t_solve"!="t_solve":
            self.pred_curves["t_solve"]=ax.plot(pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
        slider_layout=QGridLayout()
        slider_widget=QWidget()
        slider_widget.setLayout(slider_layout)
        left_layout.addWidget(slider_widget)
        self.sliders={}
        def add_slider(name, label, min_val, max_val, default_val, row, col):
            slider=QSlider(Qt.Horizontal)
            lbl=QLabel(f"{label}: {default_val}")
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            slider_layout.addWidget(lbl, row, col*2)
            slider_layout.addWidget(slider, row, col*2+1)
            slider.valueChanged.connect(lambda : lbl.setText(f"{label}: {slider.value()}"))
            self.sliders[name]=slider
        add_slider("target_s", "Ziel-Position", -20, 20, 10, 0, 0)
        add_slider("wind", "Windkraft", -100, 100, 0, 1, 0)
        add_slider("M", "Wagenmasse", 1, 50, 10, 2, 0)
        add_slider("m", "Pendelmasse", 1, 20, 1, 3, 0)
        add_slider("Q_s", "Gewicht Position", 0, 100, 10, 0, 1)
        add_slider("Q_theta", "Gewicht Winkel", 0, 200, 100, 1, 1)
        add_slider("R_F", "Gewicht Kraft", 1, 100, 1, 2, 1)
        add_slider("max_pos", "Schiene Limit", 10, 50, 20, 3, 1)
        self.mpc=PendulumMPC()
        self.state=np.array([0.    , 0.    , np.pi, 0.    ])
        self.t_hist=[]
        self.hist={("s"):([]), ("v"):([]), ("theta"):([]), ("omega"):([]), ("F"):([]), ("t_solve"):([])}
        self.time=0.    
        self.dt=3.3e-2
        self.timer=QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(33)
    def get_params(self):
        return {("M"):((self.sliders["M"].value())/1.0e+1), ("m"):((self.sliders["m"].value())/1.0e+1), ("l"):(0.5    ), ("wind"):((self.sliders["wind"].value())/1.0e+1), ("Q_s"):(float(self.sliders["Q_s"].value())), ("Q_v"):(1.    ), ("Q_theta"):(float(self.sliders["Q_theta"].value())), ("Q_omega"):(1.    ), ("R_F"):((self.sliders["R_F"].value())/1.0e+2), ("max_pos"):((self.sliders["max_pos"].value())/1.0e+1), ("max_force"):(1.5e+1)}
    def update_loop(self):
        params=self.get_params()
        target_s=(self.sliders["target_s"].value())/1.0e+1
        target_state=np.array([target_s, 0.    , 0.    , 0.    ])
        u_opt, X_pred, U_pred, t_solve, success=self.mpc.step(self.state, target_state, params)
        F_motor=u_opt
        wind_force=params["wind"]
        # Physik-Simulation: Runge-Kutta 4 Integrationsschritt mit echter Windstörung
        def f_real(st):
            s_st=st[0]
            v_st=st[1]
            theta_st=st[2]
            omega_st=st[3]
            sin_t=np.sin(theta_st)
            cos_t=np.cos(theta_st)
            den=params["M"]+params["m"]*(1.    -(cos_t*cos_t))
            F_tot=F_motor+wind_force*cos_t
            ds=v_st
            dv=(F_tot+params["m"]*0.5    *omega_st*omega_st*sin_t+params["m"]*9.81    *cos_t*sin_t)/den
            dtheta=omega_st
            domega=(( -1.    *F_tot*cos_t)-(params["m"]*0.5    *omega_st*omega_st*sin_t*cos_t)-((params["M"]+params["m"])*9.81    *sin_t))/(0.5    *den)
            return np.array([ds, dv, dtheta, domega])
        k1=f_real(self.state)
        k2=f_real(self.state+(self.dt/2.    )*k1)
        k3=f_real(self.state+(self.dt/2.    )*k2)
        k4=f_real(self.state+self.dt*k3)
        self.state=self.state+np.array(np.squeeze((self.dt/6.    )*(k1+2.    *k2+2.    *k3+k4)))
        self.time=self.time+self.dt
        self.t_hist.append(self.time)
        self.hist["s"].append(self.state[0])
        self.hist["v"].append(self.state[1])
        self.hist["theta"].append(self.state[2])
        self.hist["omega"].append(self.state[3])
        self.hist["F"].append(F_motor)
        self.hist["t_solve"].append(t_solve*1.0e+3)
        if len(self.t_hist)>100:
            self.t_hist.pop(0)
            self.hist["s"].pop(0)
            self.hist["v"].pop(0)
            self.hist["theta"].pop(0)
            self.hist["omega"].pop(0)
            self.hist["F"].pop(0)
            self.hist["t_solve"].pop(0)
        self.pendulum_widget.update_state(self.state, F_motor, wind_force, 0.5    , params["max_pos"])
        self.history_curves["s"].setData(self.t_hist, self.hist["s"])
        self.history_curves["v"].setData(self.t_hist, self.hist["v"])
        self.history_curves["theta"].setData(self.t_hist, self.hist["theta"])
        self.history_curves["omega"].setData(self.t_hist, self.hist["omega"])
        self.history_curves["F"].setData(self.t_hist, self.hist["F"])
        self.history_curves["t_solve"].setData(self.t_hist, self.hist["t_solve"])
        if success:
            t_pred=np.linspace(self.time, self.time+self.mpc.T_horizon, self.mpc.N+1)
            self.pred_curves["s"].setData(t_pred, X_pred[0,:])
            self.pred_curves["v"].setData(t_pred, X_pred[1,:])
            self.pred_curves["theta"].setData(t_pred, X_pred[2,:])
            self.pred_curves["omega"].setData(t_pred, X_pred[3,:])
            self.pred_curves["F"].setData(t_pred[0:-1], U_pred[0,:])
if __name__=="__main__":
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec())