(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11c
  (:use #:cl #:cl-py-generator))

(in-package #:g11c)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11c_mpc_gui"
						 *source*))
 `(do0
   (imports-from (__future__ annotations)
		 (casadi *))
   (imports ((np numpy)
	     time
	     sys
	     (pg pyqtgraph)))
   (imports-from (PySide6.QtWidgets QApplication QMainWindow QWidget QVBoxLayout QHBoxLayout QSlider QLabel QGridLayout)
		 (PySide6.QtCore Qt QTimer)
		 (PySide6.QtGui QPainter QPen QBrush QColor))

   (comments 
    "========================================================================================="
    " INVERTED PENDULUM MPC (MODEL PREDICTIVE CONTROL) DASHBOARD"
    "========================================================================================="
    " PURPOSE AND SCOPE:"
    " This application demonstrates real-time Model Predictive Control of a nonlinear, "
    " underactuated mechanical system (an inverted pendulum on a cart). "
    " The GUI provides interactive sliders to change physical parameters (like mass and wind) "
    " and tuning parameters (like cost weights) on the fly, visualizing the solver's "
    " future predictions versus the actual system trajectory."
    ""
    " PHYSICAL MODEL & DYNAMICS:"
    " We consider a cart of mass M moving on a 1D track, and a pendulum of mass m and length l "
    " attached to it. The system is underactuated: we only control the horizontal force F "
    " on the cart, but we want to control both the cart's position (s) and the pendulum's "
    " angle (theta). The state vector is x = [s, v, theta, omega]."
    " Using Lagrangian mechanics, the nonlinear ODEs are derived considering kinetic and "
    " potential energies. A simulated wind force applies a disturbing torque."
    ""
    " CONTROL THEORY (MODEL PREDICTIVE CONTROL):"
    " MPC solves a finite-horizon optimal control problem (OCP) at every sampling time. "
    " It calculates a trajectory of control inputs that minimizes a cost function "
    " (e.g., deviations from the target state) while satisfying constraints (e.g., track limits). "
    " Only the very first control input is applied to the real system. In the next step, "
    " the horizon shifts and the problem is solved again (Receding Horizon Control)."
    ""
    " MATHEMATICS OF DIRECT COLLOCATION:"
    " Instead of integrating the ODEs step-by-step (like in Multiple Shooting), Direct "
    " Collocation discretizes the state trajectory using polynomials (e.g., Lagrange polynomials). "
    " The states at specific collocation points (here: Radau points) become optimization variables. "
    " The system dynamics are enforced as equality constraints mapping the derivative of "
    " the polynomials to the vector field f(x,u). This transforms the continuous-time optimal "
    " control problem into a huge but sparse Nonlinear Programming (NLP) problem."
    ""
    " CASADI & IPOPT:"
    " CasADi (Computer algebra system for automatic differentiation) is used to mathematically "
    " formulate the NLP. It computes exact gradients and sparse Jacobians automatically using AD. "
    " IPOPT (Interior Point OPTimizer) is the backend solver. To achieve real-time performance "
    " (under 33ms), we use 'Warm-Starting': the optimal trajectory from the previous time step "
    " is passed as an initial guess to IPOPT, reducing the number of required iterations to just 1-3."
    "=========================================================================================")

   (class PendulumMPC ()
     (def __init__ (self)
       (setf self.opti (Opti)
	     self.nx 4
	     self.nu 1
	     self.N 20
	     self.T_horizon 1.0
	     self.h (/ self.T_horizon self.N)
	     self.d 3)

       (comments "Parameter für Physik (M: Wagenmasse, m: Pendelmasse, l: Länge, wind: Störkraft)")
       (setf self.M_p (self.opti.parameter)
	     self.m_p (self.opti.parameter)
	     self.l_p (self.opti.parameter)
	     self.wind_p (self.opti.parameter)
	     self.Q_s (self.opti.parameter)
	     self.Q_v (self.opti.parameter)
	     self.Q_theta (self.opti.parameter)
	     self.Q_omega (self.opti.parameter)
	     self.R_F (self.opti.parameter)
	     self.max_pos (self.opti.parameter)
	     self.max_force (self.opti.parameter))

       (comments "Symbolische Variablen für die Dynamik (dx/dt = f(x,u))")
       (setf x (SX.sym (string "x") self.nx)
	     u (SX.sym (string "u") self.nu)
	     s_ (aref x 0) v_ (aref x 1) theta_ (aref x 2) omega_ (aref x 3) F_ u
	     sin_theta (np.sin theta_) cos_theta (np.cos theta_)
	     den (+ self.M_p (* self.m_p (- 1.0 (* cos_theta cos_theta))))
	     F_total (+ F_ (* self.wind_p cos_theta))
	     ds v_
	     dv (/ (+ F_total (* self.m_p self.l_p omega_ omega_ sin_theta) (* self.m_p 9.81 cos_theta sin_theta)) den)
	     dtheta omega_
	     domega (/ (- (* -1.0 F_total cos_theta) (* self.m_p self.l_p omega_ omega_ sin_theta cos_theta) (* (+ self.M_p self.m_p) 9.81 sin_theta)) (* self.l_p den)))
       (setf self.f_ode (Function (string "f_ode") (list x u) (list (vertcat ds dv dtheta domega))))

       (comments "Lagrange Collocation Polynome via Radau-Punkten berechnen")
       (setf tau_root (np.append 0.0 (collocation_points self.d (string "radau")))
	     self.C (np.zeros (tuple (+ self.d 1) (+ self.d 1)))
	     self.D (np.zeros (+ self.d 1)))
       (for (j (range (+ self.d 1)))
	    (setf p (np.poly1d (list 1.0)))
	    (for (r (range (+ self.d 1)))
		 (if (!= r j)
		     (setf p (* p (/ (np.poly1d (list 1.0 (- (aref tau_root r))))
				     (- (aref tau_root j) (aref tau_root r)))))))
	    (setf (aref self.D j) (p 1.0)
		  pder (np.polyder p))
	    (for (r (range (+ self.d 1)))
		 (setf (aref self.C j r) (pder (aref tau_root r)))))

       (comments "Decision Variables für IPOPT (X: Knoten, Xc: Collocation-Punkte, U: Stellgröße)")
       (setf self.X (self.opti.variable self.nx (+ self.N 1))
	     self.Xc (list)
	     self.U (self.opti.variable self.nu self.N)
	     self.current_x (self.opti.parameter self.nx)
	     self.target_x (self.opti.parameter self.nx))
       (for (k (range self.N))
	    (setf Xc_k (list))
	    (for (r (range self.d))
		 (dot Xc_k (append (self.opti.variable self.nx))))
	    (dot self.Xc (append Xc_k)))

       (comments "Constraints anwenden: Startzustand, Grenzen und Collocation-Dynamik")
       (self.opti.subject_to (== (aref self.X (slice nil nil) 0) self.current_x))
       (self.opti.subject_to (self.opti.bounded (* -1.0 self.max_pos) (aref self.X 0 (slice nil nil)) self.max_pos))
       (self.opti.subject_to (self.opti.bounded (* -1.0 self.max_force) self.U self.max_force))

       (for (k (range self.N))
	    (setf Xk (aref self.X (slice nil nil) k)
		  x_end (* (aref self.D 0) Xk))
	    (for (j (range 1 (+ self.d 1)))
		 (setf xp (* (aref self.C 0 j) Xk))
		 (for (r (range self.d))
		      (setf xp (+ xp (* (aref self.C (+ r 1) j) (aref (aref self.Xc k) r)))))
		 (setf f_eval (self.f_ode (aref (aref self.Xc k) (- j 1)) (aref self.U (slice nil nil) k)))
		 (self.opti.subject_to (== xp (* self.h f_eval))))
	    (for (r (range self.d))
		 (setf x_end (+ x_end (* (aref self.D (+ r 1)) (aref (aref self.Xc k) r)))))
	    (self.opti.subject_to (== (aref self.X (slice nil nil) (+ k 1)) x_end)))

       (comments "Kostenfunktion (Objective): Abweichung vom Ziel und Stellenergie minimieren")
       (setf cost 0.0
	     Q (np.diag (vertcat self.Q_s self.Q_v self.Q_theta self.Q_omega)))
       (for (k (range self.N))
	    (setf err (- (aref self.X (slice nil nil) k) self.target_x)
		  cost (+ cost (+ (mtimes (mtimes err.T Q) err)
				  (* self.R_F (** (aref self.U 0 k) 2))))))
       (setf err_term (- (aref self.X (slice nil nil) self.N) self.target_x)
	     cost (+ cost (* 10.0 (mtimes (mtimes err_term.T Q) err_term))))
       (self.opti.minimize cost)

       (comments "IPOPT Konfiguration (ohne JIT Komplexität, Python API reicht aus)")
       (self.opti.solver (string "ipopt") 
			 (dict ((string "print_time") False))
			 (dict ((string "print_level") 0) ((string "sb") (string "yes"))))
       
       (self.opti.set_initial self.X (dot (np.linspace (np.array (list 0.0 0.0 np.pi 0.0)) (np.array (list 1.0 0.0 0.0 0.0)) (+ self.N 1)) T))
       (setf self.sol None))

     (def step (self state target_state params)
       (self.opti.set_value self.current_x state)
       (self.opti.set_value self.target_x target_state)
       ,@(loop for key in '("M" "m" "l" "wind" "Q_s" "Q_v" "Q_theta" "Q_omega" "R_F" "max_pos" "max_force")
	       for sym = (intern (string-upcase (format nil "self.~A_p" key)))
	       collect `(self.opti.set_value ,sym (aref params (string ,key))))

       (if (!= self.sol None)
	   (do0
	    (setf X_res (self.sol.value self.X)
		  U_res (self.sol.value self.U)
		  X_guess (np.hstack (tuple (aref X_res (slice nil nil) (slice 1 nil))
					    (aref X_res (slice nil nil) (slice -1 nil))))
		  U_guess (np.append (aref U_res (slice 1 nil)) (aref U_res -1)))
	    (self.opti.set_initial self.X X_guess)
	    (self.opti.set_initial self.U U_guess)))

       (setf t0 (time.time))
       (try
	(setf self.sol (self.opti.solve))
	("Exception as e"
		(return (tuple (aref U_guess 0) X_res U_res (- (time.time) t0) False))))
       (setf t_solve (- (time.time) t0)
	     X_res (self.sol.value self.X)
	     U_res (self.sol.value self.U))
       (return (tuple (aref U_res 0) X_res U_res t_solve True))))

   (class PendulumWidget (QWidget)
     (def __init__ (self)
       (QWidget.__init__ self)
       (self.setMinimumSize 400 400)
       (setf self.state (np.array (list 0.0 0.0 np.pi 0.0))
	     self.l 0.5 self.force 0.0 self.wind 0.0 self.max_pos 2.0))
     
     (def update_state (self state force wind l max_pos)
       (setf self.state state self.force force self.wind wind self.l l self.max_pos max_pos)
       (self.update))
       
     (def paintEvent (self event)
       (setf painter (QPainter self)
	     w (self.width) h (self.height) cx (/ w 2) cy (/ h 2)
	     scale (/ w (* 2.5 self.max_pos)))
       (painter.setRenderHint QPainter.Antialiasing)
       (painter.fillRect 0 0 w h (QColor 30 30 30))
       
       (painter.setPen (pg.mkPen (string "gray") :width 4))
       (painter.drawLine (- cx (* scale self.max_pos)) cy (+ cx (* scale self.max_pos)) cy)
       
       (setf s (aref self.state 0) theta (aref self.state 2)
	     cart_w (* 0.4 scale) cart_h (* 0.2 scale)
	     cart_x (+ cx (* s scale)) cart_y cy)
       
       (painter.setBrush (QBrush (QColor 100 150 250)))
       (painter.setPen (pg.mkPen (string "w") :width 2))
       (painter.drawRect (- cart_x (/ cart_w 2)) (- cart_y (/ cart_h 2)) cart_w cart_h)
       
       (setf pend_x (+ cart_x (* scale self.l (np.sin theta)))
	     pend_y (- cart_y (* scale self.l (np.cos theta))))
       (painter.setPen (pg.mkPen (string "w") :width 6))
       (painter.drawLine cart_x cart_y pend_x pend_y)
       (painter.setBrush (QBrush (QColor 250 100 100)))
       (painter.drawEllipse (- pend_x 10) (- pend_y 10) 20 20)
       
       (if (> (np.abs self.force) 0.1)
	   (do0
	    (painter.setPen (pg.mkPen (string "g") :width 3))
	    (painter.drawLine cart_x cart_y (+ cart_x (* 5.0 self.force)) cart_y)))))

   (class MainWindow (QMainWindow)
     (def __init__ (self)
       (QMainWindow.__init__ self)
       (self.setWindowTitle (string "MPC Inverted Pendulum"))
       (self.resize 1400 900)
       
       (setf central_widget (QWidget)
	     main_layout (QHBoxLayout central_widget)
	     left_layout (QVBoxLayout)
	     right_layout (QVBoxLayout))
       (self.setCentralWidget central_widget)
       (main_layout.addLayout left_layout 1)
       (main_layout.addLayout right_layout 2)
       
       (setf self.pendulum_widget (PendulumWidget)
	     self.plot_layout (pg.GraphicsLayoutWidget))
       (left_layout.addWidget self.pendulum_widget)
       (right_layout.addWidget self.plot_layout)
       
       (setf self.plots (dict)
	     self.history_curves (dict)
	     self.pred_curves (dict))
       
       (comments "Plots dynamisch generieren mit Lisp Macro Expand")
       ,@(loop for (key title ylabel row col) in '(("s" "Wagenposition" "Position [m]" 0 0)
						   ("theta" "Pendelwinkel" "Winkel [rad]" 1 0)
						   ("v" "Wagengeschwindigkeit" "v [m/s]" 2 0)
						   ("omega" "Winkelgeschw." "omega [rad/s]" 0 1)
						   ("F" "Aktuatorkraft" "Kraft [N]" 1 1)
						   ("t_solve" "Solver Rechenzeit" "Zeit [ms]" 2 1))
	       collect
	       `(do0
		 (setf ax (self.plot_layout.addPlot :row ,row :col ,col :title (string ,title)))
		 (ax.showGrid :x True :y True)
		 (ax.setLabel (string "left") (string ,ylabel))
		 (ax.setLabel (string "bottom") (string "Zeit [s]"))
		 (setf (aref self.plots (string ,key)) ax
		       (aref self.history_curves (string ,key)) (ax.plot :pen (pg.mkPen (string "w") :width 2)))
		 (if (!= (string ,key) (string "t_solve"))
		     (setf (aref self.pred_curves (string ,key)) (ax.plot :pen (pg.mkPen (string "y") :width 2 :style QtCore.Qt.DashLine))))))
		     
       (setf slider_layout (QGridLayout)
	     slider_widget (QWidget))
       (slider_widget.setLayout slider_layout)
       (left_layout.addWidget slider_widget)
       (setf self.sliders (dict))
       
       (def add_slider (name label min_val max_val default_val row col)
	 (setf slider (QSlider Qt.Horizontal)
	       lbl (QLabel (fstring "{label}: {default_val}")))
	 (slider.setMinimum min_val)
	 (slider.setMaximum max_val)
	 (slider.setValue default_val)
	 (slider_layout.addWidget lbl row (* col 2))
	 (slider_layout.addWidget slider row (+ (* col 2) 1))
	 (slider.valueChanged.connect (lambda () (lbl.setText (fstring "{label}: {slider.value()}"))))
	 (setf (aref self.sliders name) slider))
	 
       (add_slider (string "target_s") (string "Ziel-Position") -20 20 10 0 0) 
       (add_slider (string "wind") (string "Windkraft") -100 100 0 1 0)     
       (add_slider (string "M") (string "Wagenmasse") 1 50 10 2 0)          
       (add_slider (string "m") (string "Pendelmasse") 1 20 1 3 0)          
       (add_slider (string "Q_s") (string "Gewicht Position") 0 100 10 0 1)
       (add_slider (string "Q_theta") (string "Gewicht Winkel") 0 200 100 1 1)
       (add_slider (string "R_F") (string "Gewicht Kraft") 1 100 1 2 1)     
       (add_slider (string "max_pos") (string "Schiene Limit") 10 50 20 3 1) 
       
       (setf self.mpc (PendulumMPC)
	     self.state (np.array (list 0.0 0.0 np.pi 0.0))
	     self.t_hist (list)
	     self.hist (dict ((string "s") (list)) ((string "v") (list)) ((string "theta") (list))
			     ((string "omega") (list)) ((string "F") (list)) ((string "t_solve") (list)))
	     self.time 0.0
	     self.dt 0.033)
	     
       (setf self.timer (QTimer))
       (self.timer.timeout.connect self.update_loop)
       (self.timer.start 33))

     (def get_params (self)
       (return (dictionary :M (/ (dot (aref self.sliders (string "M")) (value)) 10.0)
		     :m (/ (dot (aref self.sliders (string "m")) (value)) 10.0)
		     :l 0.5
		     :wind (/ (dot (aref self.sliders (string "wind")) (value)) 10.0)
		     :Q_s (float (dot (aref self.sliders (string "Q_s")) (value)))
		     :Q_v 1.0
		     :Q_theta (float (dot (aref self.sliders (string "Q_theta")) (value)))
		     :Q_omega 1.0
		     :R_F (/ (dot (aref self.sliders (string "R_F")) (value)) 100.0)
		     :max_pos (/ (dot (aref self.sliders (string "max_pos")) (value)) 10.0)
		     :max_force 15.0)))
       
     (def update_loop (self)
       (setf params (self.get_params)
	     target_s (/ (dot (aref self.sliders (string "target_s")) (value)) 10.0)
	     target_state (np.array (list target_s 0.0 0.0 0.0)))
	     
       (setf (ntuple u_opt X_pred U_pred t_solve success) (self.mpc.step self.state target_state params)
	     F_motor u_opt
	     wind_force (aref params (string "wind")))
	     
       (comments "Physik-Simulation: Runge-Kutta 4 Integrationsschritt mit echter Windstörung")
       (def f_real (st)
	 (setf s_st (aref st 0) v_st (aref st 1) theta_st (aref st 2) omega_st (aref st 3)
	       sin_t (np.sin theta_st) cos_t (np.cos theta_st)
	       den (+ (aref params (string "M")) (* (aref params (string "m")) (- 1.0 (* cos_t cos_t))))
	       F_tot (+ F_motor (* wind_force cos_t))
	       ds v_st
	       dv (/ (+ F_tot (* (aref params (string "m")) 0.5 omega_st omega_st sin_t) (* (aref params (string "m")) 9.81 cos_t sin_t)) den)
	       dtheta omega_st
	       domega (/ (- (* -1.0 F_tot cos_t) (* (aref params (string "m")) 0.5 omega_st omega_st sin_t cos_t) (* (+ (aref params (string "M")) (aref params (string "m"))) 9.81 sin_t)) (* 0.5 den)))
	 (return (np.array (list ds dv dtheta domega))))
	 
       (setf k1 (f_real self.state)
	     k2 (f_real (+ self.state (* (/ self.dt 2.0) k1)))
	     k3 (f_real (+ self.state (* (/ self.dt 2.0) k2)))
	     k4 (f_real (+ self.state (* self.dt k3)))
	     self.state (+ self.state (np.array (np.squeeze (* (/ self.dt 6.0) (+ k1 (* 2.0 k2) (* 2.0 k3) k4)))))
	     self.time (+ self.time self.dt))
	     
       (dot self.t_hist (append self.time))
       ,@(loop for (key val) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
	       collect `(dot (aref self.hist (string ,key)) (append (aref self.state ,val))))
       (dot (aref self.hist (string "F")) (append F_motor))
       (dot (aref self.hist (string "t_solve")) (append (* t_solve 1000.0)))
       
       (if (> (len self.t_hist) 100)
	   (do0
	    (self.t_hist.pop 0)
	    ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
		    collect `(dot (aref self.hist (string ,key)) (pop 0)))))
		    
       (self.pendulum_widget.update_state self.state F_motor wind_force 0.5 (aref params (string "max_pos")))
       
       ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
	       collect `(dot (aref self.history_curves (string ,key)) (setData self.t_hist (aref self.hist (string ,key)))))
	       
       (if success
	   (do0
	    (setf t_pred (np.linspace self.time (+ self.time self.mpc.T_horizon) (+ self.mpc.N 1)))
	    ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
		    collect `(dot (aref self.pred_curves (string ,key)) (setData t_pred (aref X_pred ,idx (slice nil nil)))))
	    (dot (aref self.pred_curves (string "F")) (setData (aref t_pred (slice 0 -1)) (aref U_pred 0 (slice nil nil))))))))

   (if (== __name__ (string "__main__"))
       (do0
	(setf app (QApplication sys.argv)
	      win (MainWindow))
	(win.show)
	(sys.exit (app.exec))))
   ))
