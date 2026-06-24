(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11f
  (:use #:cl #:cl-py-generator))

(in-package #:g11f)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11f_mpc_gui"
						 *source*))
 `(do0
   (imports-from (__future__ annotations)
		 (casadi *))
   (imports ((np numpy)
	     time
	     sys
	     (pg pyqtgraph)))
   (imports-from (PySide6.QtWidgets QApplication QMainWindow QWidget QVBoxLayout QHBoxLayout QSlider QLabel QGridLayout QPushButton)
		 (PySide6.QtCore Qt QTimer)
		 (PySide6.QtGui QPainter QPen QBrush QColor))

   (comments 
    "========================================================================================="
    " INVERTED PENDULUM MPC GUI (MAPPED CONSTRAINTS + JIT CACHING)"
    "=========================================================================================")

   (class PendulumMPC ()
     (def __init__ (self &key (h_mpc 0.05) (N 20) (use_jit False) (use_to_function False) (use_dual_warmstart False) (use_map False))
       (setf self.opti (Opti)
	     self.nx 4
	     self.nu 1
	     self.N N
	     self.h_mpc h_mpc
	     self.T_horizon (* self.N self.h_mpc)
	     self.d 3
             self.use_jit use_jit
             self.use_to_function use_to_function
             self.use_dual_warmstart use_dual_warmstart
             self.use_map use_map)

       ,@(loop for sym in '(M_p m_p l_p wind_p Q_s Q_v Q_theta Q_omega R_F max_pos max_force)
               collect `(setf (dot self ,sym) (self.opti.parameter)))

       (setf x (SX.sym (string "x") self.nx)
	     u (SX.sym (string "u") self.nu)
	     p_ode (SX.sym (string "p_ode") 4)
	     s_ (aref x 0) v_ (aref x 1) theta_ (aref x 2) omega_ (aref x 3) F_ u
	     M_s (aref p_ode 0) m_s (aref p_ode 1) l_s (aref p_ode 2) wind_s (aref p_ode 3)
	     sin_theta (np.sin theta_) cos_theta (np.cos theta_)
	     den (+ M_s (* m_s (- 1.0 (* cos_theta cos_theta))))
	     F_total (+ F_ (* wind_s cos_theta))
	     ds v_
	     dv (/ (+ F_total (* m_s l_s omega_ omega_ sin_theta) (* m_s 9.81 cos_theta sin_theta)) den)
	     dtheta omega_
	     domega (/ (- (* -1.0 F_total cos_theta) (* m_s l_s omega_ omega_ sin_theta cos_theta) (* (+ M_s m_s) 9.81 sin_theta)) (* l_s den)))
       (setf self.f_ode (Function (string "f_ode") (list x u p_ode) (list (vertcat ds dv dtheta domega))))

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

       (if self.use_map
           (do0
             (setf self.X (self.opti.variable self.nx (+ self.N 1))
                   self.Xc_var (self.opti.variable (* self.nx self.d) self.N)
                   self.U (self.opti.variable self.nu self.N)
                   self.current_x (self.opti.parameter self.nx)
                   self.target_x (self.opti.parameter self.nx))
             
             (setf Xk_sym (SX.sym (string "Xk") self.nx)
                   Xck_vec_sym (SX.sym (string "Xck_vec") (* self.nx self.d))
                   Uk_sym (SX.sym (string "Uk") self.nu)
                   p_sym (SX.sym (string "p") 4)
                   Xck_mat (reshape Xck_vec_sym self.nx self.d)
                   x_end (* (aref self.D 0) Xk_sym)
                   res_list (list))
             
             (for (j (range 1 (+ self.d 1)))
                  (setf xp (* (aref self.C 0 j) Xk_sym))
                  (for (r (range self.d))
                       (setf xp (+ xp (* (aref self.C (+ r 1) j) (aref Xck_mat (slice nil nil) r)))))
                  (setf f_eval (self.f_ode (aref Xck_mat (slice nil nil) (- j 1)) Uk_sym p_sym))
                  (res_list.append (- xp (* self.h_mpc f_eval))))
             
             (for (r (range self.d))
                  (setf x_end (+ x_end (* (aref self.D (+ r 1)) (aref Xck_mat (slice nil nil) r)))))
             
             (setf res_vec (vertcat *res_list)
                   self.colloc_interval (Function (string "colloc_interval")
                                                  (list Xk_sym Xck_vec_sym Uk_sym p_sym)
                                                  (list res_vec x_end))
                   colloc_map (self.colloc_interval.map self.N)
                   p_stacked (repmat (vertcat self.M_p self.m_p self.l_p self.wind_p) 1 self.N))
             
             (setf (ntuple res_all x_end_all) (colloc_map (aref self.X (slice nil nil) (slice nil self.N))
                                                          self.Xc_var
                                                          self.U
                                                          p_stacked))
             (self.opti.subject_to (== (aref self.X (slice nil nil) 0) self.current_x))
             (self.opti.subject_to (self.opti.bounded (* -1.0 self.max_pos) (aref self.X 0 (slice nil nil)) self.max_pos))
             (self.opti.subject_to (self.opti.bounded (* -1.0 self.max_force) self.U self.max_force))
             (self.opti.subject_to (== res_all 0))
             (self.opti.subject_to (== (aref self.X (slice nil nil) (slice 1 nil)) x_end_all)))
           (do0
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
                       (setf f_eval (self.f_ode (aref (aref self.Xc k) (- j 1))
                                                (aref self.U (slice nil nil) k)
                                                (vertcat self.M_p self.m_p self.l_p self.wind_p)))
                       (self.opti.subject_to (== xp (* self.h_mpc f_eval))))
                  (for (r (range self.d))
                       (setf x_end (+ x_end (* (aref self.D (+ r 1)) (aref (aref self.Xc k) r)))))
                  (self.opti.subject_to (== (aref self.X (slice nil nil) (+ k 1)) x_end)))))

       (setf cost 0.0
	     Q (diag (vertcat self.Q_s self.Q_v self.Q_theta self.Q_omega)))
       (for (k (range self.N))
	    (setf err (- (aref self.X (slice nil nil) k) self.target_x)
		  cost (+ cost (+ (mtimes (mtimes err.T Q) err)
				  (* self.R_F (** (aref self.U 0 k) 2))))))
       (setf err_term (- (aref self.X (slice nil nil) self.N) self.target_x)
	     cost (+ cost (* 10.0 (mtimes (mtimes err_term.T Q) err_term))))
       (self.opti.minimize cost)

       (setf self.n_constraints (aref (dot self opti g shape) 0))

       (setf solver_opts (dict ((string "print_time") False) ((string "error_on_fail") True))
             ipopt_opts (dict ((string "print_level") 0) ((string "sb") (string "yes"))))
       (if self.use_jit
           (do0
             (setf (aref solver_opts (string "jit")) True
                   (aref solver_opts (string "compiler")) (string "shell")
                   (aref solver_opts (string "jit_options")) (dict ((string "flags") (list (string "-O3")
                                                                                         (string "-ffast-math")
                                                                                         (string "-march=native")))))))
       (self.opti.solver (string "ipopt") solver_opts ipopt_opts)

       (self.opti.set_initial self.X (dot (np.linspace (np.array (list 0.0 0.0 np.pi 0.0)) (np.array (list 1.0 0.0 0.0 0.0)) (+ self.N 1)) T))
       (self.opti.set_initial self.U (np.zeros (tuple self.nu self.N)))
       (if self.use_map
           (self.opti.set_initial self.Xc_var (np.zeros (tuple (* self.nx self.d) self.N))))

       (setf self.sol None
             self.last_X (np.zeros (tuple self.nx (+ self.N 1)))
             self.last_U (np.zeros (tuple self.nu self.N))
             self.last_lam_g None)
       (if self.use_map
           (setf self.last_Xc_var (np.zeros (tuple (* self.nx self.d) self.N))))

       (if self.use_to_function
           (do0
             (setf inputs (list self.current_x self.target_x
                                self.M_p self.m_p self.l_p self.wind_p
                                self.Q_s self.Q_v self.Q_theta self.Q_omega
                                self.R_F self.max_pos self.max_force
                                self.X))
             (if self.use_map
                 (inputs.append self.Xc_var))
             (inputs.append self.U)
             (if self.use_dual_warmstart
                 (inputs.append self.opti.lam_g))

             (setf outputs (list (aref self.U 0 0) self.X))
             (if self.use_map
                 (outputs.append self.Xc_var))
             (outputs.append self.U)
             (if self.use_dual_warmstart
                 (outputs.append self.opti.lam_g))

             (setf self.mpc_func (self.opti.to_function (string "mpc_solve") inputs outputs)))))

     (def step (self state target_state params)
       (if self.use_to_function
           (do0
             (setf inputs (list state target_state
                                (aref params (string "M"))
                                (aref params (string "m"))
                                (aref params (string "l"))
                                (aref params (string "wind"))
                                (aref params (string "Q_s"))
                                (aref params (string "Q_v"))
                                (aref params (string "Q_theta"))
                                (aref params (string "Q_omega"))
                                (aref params (string "R_F"))
                                (aref params (string "max_pos"))
                                (aref params (string "max_force"))))
             (if (or (is-not self.sol None) (is-not self.last_lam_g None))
                 (do0
                   (setf X_guess (np.hstack (tuple (aref self.last_X (slice nil nil) (slice 1 nil))
                                                   (aref self.last_X (slice nil nil) (slice -1 nil))))
                         U_guess (np.append (aref self.last_U (slice 1 nil)) (aref self.last_U -1)))
                   (if self.use_map
                       (setf Xc_guess self.last_Xc_var)))
                 (do0
                   (setf X_guess (dot (np.linspace state target_state (+ self.N 1)) T)
                         U_guess (np.zeros self.N))
                   (if self.use_map
                       (setf Xc_guess (np.zeros (tuple (* self.nx self.d) self.N))))))
             (inputs.append X_guess)
             (if self.use_map
                 (inputs.append Xc_guess))
             (inputs.append (aref U_guess np.newaxis (slice nil nil)))
             (if self.use_dual_warmstart
                 (do0
                   (if (is-not self.last_lam_g None)
                       (setf lam_g_guess self.last_lam_g)
                       (setf lam_g_guess (np.zeros self.n_constraints)))
                   (inputs.append lam_g_guess)))
             (setf t0 (time.time))
             (try
               (do0
                 (setf res (self.mpc_func *inputs))
                 (cond
                   ((and self.use_map self.use_dual_warmstart)
                    (setf (ntuple u_opt X_val Xc_val U_val lam_g_val) res
                          self.last_X (dot X_val (full))
                          self.last_Xc_var (dot Xc_val (full))
                          self.last_U (np.squeeze (dot U_val (full)))
                          self.last_lam_g (dot lam_g_val (full))))
                   (self.use_map
                    (setf (ntuple u_opt X_val Xc_val U_val) res
                          self.last_X (dot X_val (full))
                          self.last_Xc_var (dot Xc_val (full))
                          self.last_U (np.squeeze (dot U_val (full)))))
                   (self.use_dual_warmstart
                    (setf (ntuple u_opt X_val U_val lam_g_val) res
                          self.last_X (dot X_val (full))
                          self.last_U (np.squeeze (dot U_val (full)))
                          self.last_lam_g (dot lam_g_val (full))))
                   (t
                    (setf (ntuple u_opt X_val U_val) res
                          self.last_X (dot X_val (full))
                          self.last_U (np.squeeze (dot U_val (full))))))
                 (setf u_opt (float u_opt)
                       t_solve (- (time.time) t0))
                 (return (tuple u_opt self.last_X self.last_U t_solve True)))
               ("Exception as e"
                 (setf t_solve (- (time.time) t0))
                 (return (tuple (float (aref U_guess 0)) self.last_X self.last_U t_solve False)))))
           (do0
             (self.opti.set_value self.current_x state)
             (self.opti.set_value self.target_x target_state)
             ,@(loop for (key sym) in '(("M" self.M_p)
                                        ("m" self.m_p)
                                        ("l" self.l_p)
                                        ("wind" self.wind_p)
                                        ("Q_s" self.Q_s)
                                        ("Q_v" self.Q_v)
                                        ("Q_theta" self.Q_theta)
                                        ("Q_omega" self.Q_omega)
                                        ("R_F" self.R_F)
                                        ("max_pos" self.max_pos)
                                        ("max_force" self.max_force))
                     collect `(self.opti.set_value ,sym (aref params (string ,key))))
             (if (is-not self.sol None)
                 (do0
                   (setf X_res (self.sol.value self.X)
                         U_res (self.sol.value self.U)
                         X_guess (np.hstack (tuple (aref X_res (slice nil nil) (slice 1 nil))
                                                   (aref X_res (slice nil nil) (slice -1 nil))))
                         U_guess (np.append (aref U_res (slice 1 nil)) (aref U_res -1)))
                   (self.opti.set_initial self.X X_guess)
                   (self.opti.set_initial self.U (aref U_guess np.newaxis (slice nil nil)))
                   (if self.use_map
                       (self.opti.set_initial self.Xc_var (self.sol.value self.Xc_var)))
                   (if self.use_dual_warmstart
                       (do0
                         (setf lam_g_res (self.sol.value self.opti.lam_g))
                         (self.opti.set_initial self.opti.lam_g lam_g_res))))
                 (do0
                   (setf X_guess (dot (np.linspace state target_state (+ self.N 1)) T)
                         U_guess (np.zeros self.N))
                   (self.opti.set_initial self.X X_guess)
                   (self.opti.set_initial self.U (aref U_guess np.newaxis (slice nil nil)))
                   (if self.use_map
                       (self.opti.set_initial self.Xc_var (np.zeros (tuple (* self.nx self.d) self.N))))
                   (if self.use_dual_warmstart
                       (self.opti.set_initial self.opti.lam_g (np.zeros self.n_constraints)))))
             (setf t0 (time.time)
                   success True)
             (try
               (setf self.sol (self.opti.solve))
               ("Exception as e"
                 (setf success False)))
             (setf t_solve (- (time.time) t0))
             (if success
                 (do0
                   (setf self.last_X (self.sol.value self.X)
                         self.last_U (np.squeeze (self.sol.value self.U))
                         u_opt (aref self.last_U 0))
                   (if self.use_map
                       (setf self.last_Xc_var (self.sol.value self.Xc_var))))
                 (do0
                   (setf self.last_X (np.hstack (tuple (aref self.last_X (slice nil nil) (slice 1 nil))
                                                       (aref self.last_X (slice nil nil) (slice -1 nil))))
                         self.last_U (np.append (aref self.last_U (slice 1 nil)) (aref self.last_U -1))
                         u_opt (aref self.last_U 0))))
             (return (tuple (float u_opt) self.last_X self.last_U t_solve success))))))

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
       (self.setWindowTitle (string "MPC Inverted Pendulum - Mapped JIT Cache version"))
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
		     (setf (aref self.pred_curves (string ,key)) (ax.plot :pen (pg.mkPen (string "y") :width 2 :style Qt.DashLine))))))
		     
       (setf btn_layout (QHBoxLayout)
             self.btn_start (QPushButton (string "Start"))
             self.btn_stop (QPushButton (string "Stop"))
             self.btn_reset (QPushButton (string "Reset / Apply Params")))
       (btn_layout.addWidget self.btn_start)
       (btn_layout.addWidget self.btn_stop)
       (btn_layout.addWidget self.btn_reset)
       (left_layout.addLayout btn_layout)
       
       (self.btn_start.clicked.connect self.start_sim)
       (self.btn_stop.clicked.connect self.stop_sim)
       (self.btn_reset.clicked.connect self.reset_sim)

       (setf self.sliders (dict)
             slider_layout (QGridLayout))
       (left_layout.addLayout slider_layout)
       
       (def add_slider (layout name label min_val max_val default_val scale row tooltip)
	 (setf slider (QSlider Qt.Horizontal)
	       init_val (* default_val scale)
	       lbl (QLabel (? (isinstance scale float)
			      (fstring "{label}: {init_val:.2f}")
			      (fstring "{label}: {int(init_val)}"))))
	 (slider.setToolTip tooltip)
	 (lbl.setToolTip tooltip)
	 (slider.setMinimum min_val)
	 (slider.setMaximum max_val)
	 (slider.setValue default_val)
	 (layout.addWidget lbl row 0)
	 (layout.addWidget slider row 1)
	 (slider.valueChanged.connect (lambda () 
					(lbl.setText (? (isinstance scale float)
							(fstring "{label}: {slider.value() * scale:.2f}")
							(fstring "{label}: {int(slider.value() * scale)}")))))
	 (setf (aref self.sliders name) slider))

       ,@(loop for (key label min-val max-val def-val scale tooltip) in
               '(("M" "Wagenmasse [kg]" 1 50 10 0.1 "Masse des Wagens. Schwerer = Träger gegen Bewegungsänderungen.")
                 ("m" "Pendelmasse [kg]" 1 20 1 0.1 "Punktmasse des Pendels am Kopfende.")
                 ("l" "Pendellänge [m]" 1 20 5 0.1 "Abstand vom Wagen zum Pendelschwerpunkt.")
                 ("wind" "Windkraft [N]" -300 300 0 0.1 "Konstante Störkraft, die horizontal auf das Pendel drückt.")
                 ("Q_s" "Gewicht Position" 0 200 10 1 "Strafe (Penalty) für Abweichung des Wagens von der Ziel-Position.")
                 ("Q_v" "Gewicht Wagengeschw." 0 100 1 1 "Strafe für hohe Geschwindigkeit des Wagens (verhindert Überschwingen).")
                 ("Q_theta" "Gewicht Pendelwinkel" 0 500 100 1 "Strafe für das Abweichen des Pendels vom instabilen Gleichgewicht (0 rad).")
                 ("Q_omega" "Gewicht Winkelgeschw." 0 100 1 1 "Strafe für schnelle Rotationen des Pendels.")
                 ("R_F" "Gewicht Kraftaufwand" 1 200 10 0.01 "Kostenfaktor für die Stellkraft F. Zwingt den Solver, Energie zu sparen.")
                 ("target_s" "Ziel-Position [m]" -100 100 10 0.1 "Soll-Position des Wagens auf der Schiene.")
                 ("max_pos" "Schiene Limit [m]" 10 200 50 0.1 "Maximaler erlaubter Fahrweg (Constraint). Der Solver darf diesen nie überschreiten.")
                 ("max_force" "Max Kraft [N]" 10 300 150 0.1 "Stellgrößenbeschränkung für den Aktuator.")
                 ("N" "Knotenpunkte (N)" 1 500 20 1 "Auflösung des Solvers. N=1 bedeutet nur ein Zeitschritt in die Zukunft.")
                 ("h_mpc" "MPC Schritt [ms]" 1 200 50 1 "Dauer eines MPC-Planungsschritts. T_horizon = N * h_mpc.")
                 ("dt_sim" "Simulations-dt [ms]" 1 100 33 1 "Schrittweite der echten Runge-Kutta Physiksimulation. Hat keinen Einfluss auf den Solver."))
               for row from 0
               collect
               `(add_slider slider_layout (string ,key) (string ,label) ,min-val ,max-val ,def-val ,scale ,row (string ,tooltip)))
       
       (setf self.timer (QTimer))
       (self.timer.timeout.connect self.update_loop)
       
       ; First time instantiation of MPC
       (setf params (self.get_params)
             self.mpc (PendulumMPC :h_mpc (aref params (string "h_mpc"))
                                   :N (aref params (string "N"))
                                   :use_jit True
                                   :use_to_function True
                                   :use_dual_warmstart True
                                   :use_map True))
       (self.reset_sim)
       (self.start_sim))

     (def start_sim (self)
       (setf self.dt (/ (dot (aref self.sliders (string "dt_sim")) (value)) 1000.0))
       (self.timer.start (int (* self.dt 1000.0)))
       (dot (aref self.sliders (string "h_mpc")) (setEnabled False))
       (dot (aref self.sliders (string "N")) (setEnabled False)))

     (def stop_sim (self)
       (self.timer.stop)
       (dot (aref self.sliders (string "h_mpc")) (setEnabled True))
       (dot (aref self.sliders (string "N")) (setEnabled True)))

     (def reset_sim (self)
       (self.stop_sim)
       (setf params (self.get_params)
             self.time 0.0
             self.state (np.array (list 0.0 0.0 np.pi 0.0))
             self.t_hist (list)
             self.dt (aref params (string "dt_sim"))
             self.hist (dict ((string "s") (list)) ((string "v") (list)) ((string "theta") (list))
                             ((string "omega") (list)) ((string "F") (list)) ((string "t_solve") (list))))
       
       (setf new_N (int (aref params (string "N")))
             new_h (aref params (string "h_mpc")))
       
       (comments "Instanziere den MPC-Solver NUR neu, wenn sich N oder h_mpc geaendert haben.")
       (if (or (not (hasattr self (string "mpc")))
               (!= self.mpc.N new_N)
               (!= self.mpc.h_mpc new_h))
           (do0
             (print (string "N oder h_mpc haben sich geaendert. Kompiliere und generiere JIT MPC neu..."))
             (setf self.mpc (PendulumMPC :h_mpc new_h
                                         :N new_N
                                         :use_jit True
                                         :use_to_function True
                                         :use_dual_warmstart True
                                         :use_map True)))
           (do0
             (print (string "Parameter N und h_mpc unveraendert. Verwende bereits kompilierten JIT MPC Solver wieder."))))
             
       (self.pendulum_widget.update_state self.state 0.0 0.0 (aref params (string "l")) (aref params (string "max_pos")))
       ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
                collect `(dot (aref self.history_curves (string ,key)) (setData (list) (list))))
       ,@(loop for key in '("s" "v" "theta" "omega" "F")
                collect `(dot (aref self.pred_curves (string ,key)) (setData (list) (list)))))

     (def get_params (self)
       (return (dictionary :M (/ (dot (aref self.sliders (string "M")) (value)) 10.0)
		     :m (/ (dot (aref self.sliders (string "m")) (value)) 10.0)
		     :l (/ (dot (aref self.sliders (string "l")) (value)) 10.0)
		     :wind (/ (dot (aref self.sliders (string "wind")) (value)) 10.0)
		     :Q_s (float (dot (aref self.sliders (string "Q_s")) (value)))
		     :Q_v (float (dot (aref self.sliders (string "Q_v")) (value)))
		     :Q_theta (float (dot (aref self.sliders (string "Q_theta")) (value)))
		     :Q_omega (float (dot (aref self.sliders (string "Q_omega")) (value)))
		     :R_F (/ (dot (aref self.sliders (string "R_F")) (value)) 100.0)
		     :max_pos (/ (dot (aref self.sliders (string "max_pos")) (value)) 10.0)
		     :max_force (/ (dot (aref self.sliders (string "max_force")) (value)) 10.0)
		     :h_mpc (/ (dot (aref self.sliders (string "h_mpc")) (value)) 1000.0)
		     :N (int (dot (aref self.sliders (string "N")) (value)))
		     :dt_sim (/ (dot (aref self.sliders (string "dt_sim")) (value)) 1000.0))))
       
     (def update_loop (self)
       (setf params (self.get_params)
	     target_s (/ (dot (aref self.sliders (string "target_s")) (value)) 10.0)
	     target_state (np.array (list target_s 0.0 0.0 0.0)))
	     
       (setf (ntuple u_opt X_pred U_pred t_solve success) (self.mpc.step self.state target_state params)
	     F_motor u_opt
	     wind_force (aref params (string "wind")))
       (setf self.dt (aref params (string "dt_sim")))
       (self.timer.setInterval (int (* self.dt 1000.0)))
	     
       (def f_real (st)
	 (setf s_st (aref st 0) v_st (aref st 1) theta_st (aref st 2) omega_st (aref st 3)
	       sin_t (np.sin theta_st) cos_t (np.cos theta_st)
	       den (+ (aref params (string "M")) (* (aref params (string "m")) (- 1.0 (* cos_t cos_t))))
	       F_tot (+ F_motor (* wind_force cos_t))
	       l_val (aref params (string "l"))
	       ds v_st
	       dv (/ (+ F_tot (* (aref params (string "m")) l_val omega_st omega_st sin_t) (* (aref params (string "m")) 9.81 cos_t sin_t)) den)
	       dtheta omega_st
	       domega (/ (- (* -1.0 F_tot cos_t) (* (aref params (string "m")) l_val omega_st omega_st sin_t cos_t) (* (+ (aref params (string "M")) (aref params (string "m"))) 9.81 sin_t)) (* l_val den)))
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
		    
       (self.pendulum_widget.update_state self.state F_motor wind_force (aref params (string "l")) (aref params (string "max_pos")))
       
       ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
	       collect `(dot (aref self.history_curves (string ,key)) (setData self.t_hist (aref self.hist (string ,key)))))
	       
       (if success
	   (do0
	    (setf t_pred (np.linspace self.time (+ self.time self.mpc.T_horizon) (+ self.mpc.N 1)))
	    ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
		    collect `(dot (aref self.pred_curves (string ,key)) (setData t_pred (aref X_pred ,idx (slice nil nil)))))
	    (dot (aref self.pred_curves (string "F")) (setData (aref t_pred (slice 0 -1)) U_pred))))))

   (if (== __name__ (string "__main__"))
       (do0
	(setf app (QApplication sys.argv)
	      win (MainWindow))
	(win.show)
	(sys.exit (app.exec))))
   ))
