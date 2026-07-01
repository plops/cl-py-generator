;; gen11h.lisp — MHE + MPC inverted pendulum GUI (full widgets, grouped)
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *source-py* "example/171_casadi/")
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-py-generator
                              (merge-pathnames #P"p11h_mhe_mpc_gui" *source-py*)))
  (defparameter *source*
    `(do0
      (imports (sys time
                    (np numpy)
                    (ca casadi)
                    (pg pyqtgraph)))
      (imports-from (PySide6.QtWidgets
                     QApplication QMainWindow QWidget QVBoxLayout QHBoxLayout
                     QGridLayout QSlider QLabel QPushButton QGroupBox QCheckBox QScrollArea)
                    (PySide6.QtCore Qt QTimer)
                    (PySide6.QtGui QPainter QPen QBrush QColor))

      ;; ==========================================
      ;; 1. SYSTEM DYNAMICS
      ;; ==========================================
      (setf x_sym (ca.MX.sym (string "x") 4)
            u_sym (ca.MX.sym (string "u") 1)
            p_dyn (ca.MX.sym (string "p") 4)
            dt_sym (ca.MX.sym (string "dt") 1))
      (setf s_ (aref x_sym 0) v_ (aref x_sym 1) th_ (aref x_sym 2) om_ (aref x_sym 3)
            M_ (aref p_dyn 0) m_ (aref p_dyn 1) l_ (aref p_dyn 2) w_ (aref p_dyn 3)
            sin_th (ca.sin th_) cos_th (ca.cos th_)
            denom (+ M_ (* m_ (- 1 (** cos_th 2)))))
      (setf f_ode (ca.Function (string "f_ode") (list x_sym u_sym p_dyn)
                               (list (ca.vertcat
                                      v_
                                      (/ (+ (+ u_sym (* w_ cos_th))
                                            (* m_ sin_th (- (* l_ (** om_ 2)) (* 9.81 cos_th))))
                                         denom)
                                      om_
                                      (/ (+ (- (* -1 (+ u_sym (* w_ cos_th)) cos_th)
                                               (* m_ l_ (** om_ 2) sin_th cos_th))
                                            (* (+ M_ m_) 9.81 sin_th))
                                         (* l_ denom))))))
      (setf rk4_k1 (f_ode x_sym u_sym p_dyn)
            rk4_k2 (f_ode (+ x_sym (* (/ dt_sym 2) rk4_k1)) u_sym p_dyn)
            rk4_k3 (f_ode (+ x_sym (* (/ dt_sym 2) rk4_k2)) u_sym p_dyn)
            rk4_k4 (f_ode (+ x_sym (* dt_sym rk4_k3)) u_sym p_dyn)
            f_rk4  (ca.Function (string "f_rk4") (list x_sym u_sym p_dyn dt_sym)
                                (list (+ x_sym (* (/ dt_sym 6)
                                                  (+ rk4_k1 (* 2 rk4_k2) (* 2 rk4_k3) rk4_k4))))))


      ;; ==========================================
      ;; 2. SOLVER FACTORY (called on build/rebuild)
      ;; ==========================================
      (def build_solvers (N_mhe N_mpc max_iter use_lbfgs)
        ;; --- MHE ---
        (setf opti_mhe (ca.Opti)
              X_mhe  (opti_mhe.variable 4 (+ N_mhe 1))
              W_mhe  (opti_mhe.variable 4 N_mhe)
              Y_meas_param  (opti_mhe.parameter 4 (+ N_mhe 1))
              U_past_param  (opti_mhe.parameter 1 N_mhe)
              X_prior_param (opti_mhe.parameter 4 1)
              P_dyn_param   (opti_mhe.parameter 4 1)
              dt_mhe_param  (opti_mhe.parameter 1 1)
              Q_w_p (opti_mhe.parameter 4 1)
              R_w_p (opti_mhe.parameter 4 1)
              P_w_p (opti_mhe.parameter 4 1))
        (setf f_map_mhe (f_rk4.map N_mhe)
              X_next_mhe (f_map_mhe (aref X_mhe (slice nil nil) (slice 0 N_mhe))
                                    U_past_param
                                    (ca.repmat P_dyn_param 1 N_mhe)
                                    (ca.repmat dt_mhe_param 1 N_mhe)))
        (opti_mhe.subject_to (== (aref X_mhe (slice nil nil) (slice 1 (+ N_mhe 1)))
                                 (+ X_next_mhe W_mhe)))
        (opti_mhe.minimize
         (+ (ca.sum1 (* P_w_p (** (- (aref X_mhe (slice nil nil) 0) X_prior_param) 2)))
            (ca.sum2 (ca.sum1 (* Q_w_p (** W_mhe 2))))
            (ca.sum2 (ca.sum1 (* R_w_p (** (- Y_meas_param X_mhe) 2))))))
        (setf ipopt_opts (dictionary :print_level 0 :sb (string "yes") :max_iter max_iter))
        (if use_lbfgs
            (setf (aref ipopt_opts (string "hessian_approximation")) (string "limited-memory")))
        (opti_mhe.solver (string "ipopt") (dictionary :print_time False) ipopt_opts)

        ;; --- MPC ---
        (setf opti_mpc (ca.Opti)
              X_mpc (opti_mpc.variable 4 (+ N_mpc 1))
              U_mpc (opti_mpc.variable 1 N_mpc)
              X_cur_p  (opti_mpc.parameter 4 1)
              X_tgt_p  (opti_mpc.parameter 4 1)
              P_dyn_mpc (opti_mpc.parameter 4 1)
              dt_mpc_p  (opti_mpc.parameter 1 1)
              Q_s_p  (opti_mpc.parameter) Q_v_p  (opti_mpc.parameter)
              Q_th_p (opti_mpc.parameter) Q_om_p (opti_mpc.parameter)
              R_F_p  (opti_mpc.parameter)
              max_pos_p (opti_mpc.parameter) max_F_p (opti_mpc.parameter))
        (setf f_map_mpc (f_rk4.map N_mpc)
              X_next_mpc (f_map_mpc (aref X_mpc (slice nil nil) (slice 0 N_mpc))
                                    U_mpc
                                    (ca.repmat P_dyn_mpc 1 N_mpc)
                                    (ca.repmat dt_mpc_p 1 N_mpc)))
        (opti_mpc.subject_to (== (aref X_mpc (slice nil nil) 0) X_cur_p))
        (opti_mpc.subject_to (== (aref X_mpc (slice nil nil) (slice 1 (+ N_mpc 1))) X_next_mpc))
        (opti_mpc.subject_to (opti_mpc.bounded (* -1 max_pos_p) (aref X_mpc 0 (slice nil nil)) max_pos_p))
        (opti_mpc.subject_to (opti_mpc.bounded (* -1 max_F_p) U_mpc max_F_p))
        (setf err_ (- X_mpc (ca.repmat X_tgt_p 1 (+ N_mpc 1))))
        (opti_mpc.minimize
         (+ (ca.sum2 (ca.sum1 (* (ca.vertcat Q_s_p Q_v_p Q_th_p Q_om_p) (** err_ 2))))
            (* R_F_p (ca.sum2 (** U_mpc 2)))))
        (setf ipopt_opts2 (dictionary :print_level 0 :sb (string "yes") :max_iter max_iter))
        (if use_lbfgs
            (setf (aref ipopt_opts2 (string "hessian_approximation")) (string "limited-memory")))
        (opti_mpc.solver (string "ipopt") (dictionary :print_time False) ipopt_opts2)

        (return (tuple opti_mhe X_mhe W_mhe Y_meas_param U_past_param X_prior_param
                       P_dyn_param dt_mhe_param Q_w_p R_w_p P_w_p
                       opti_mpc X_mpc U_mpc X_cur_p X_tgt_p P_dyn_mpc
                       dt_mpc_p Q_s_p Q_v_p Q_th_p Q_om_p R_F_p max_pos_p max_F_p N_mpc)))


      ;; ==========================================
      ;; 3. PENDULUM WIDGET
      ;; ==========================================
      (class PendulumWidget (QWidget)
        (def __init__ (self)
          (QWidget.__init__ self)
          (self.setMinimumSize 400 280)
          (setf self.x_true (np.array (list 0.0 0.0 3.14159 0.0))
                self.x_est  (np.array (list 0.0 0.0 3.14159 0.0))
                self.force 0.0 self.l 0.5 self.max_pos 2.0))
        (def update_state (self x_true x_est force l max_pos)
          (setf self.x_true x_true self.x_est x_est
                self.force force self.l l self.max_pos max_pos)
          (self.update))
        (def paintEvent (self event)
          (setf painter (QPainter self)
                w (self.width) h (self.height)
                cx (/ w 2) cy (/ h 2)
                scale (/ w (* 2.5 self.max_pos)))
          (painter.setRenderHint QPainter.Antialiasing)
          (painter.fillRect 0 0 w h (QColor 30 30 30))
          (painter.setPen (QPen (QColor 128 128 128) 4))
          (painter.drawLine (- cx (* scale self.max_pos)) cy (+ cx (* scale self.max_pos)) cy)
          (def draw_pend (state cart_col pole_col)
            (setf sx (+ cx (* (aref state 0) scale))
                  cw (* 0.4 scale) ch (* 0.2 scale))
            (painter.setBrush (QBrush cart_col))
            (painter.setPen (QPen (QColor 255 255 255) 2))
            (painter.drawRect (- sx (/ cw 2)) (- cy (/ ch 2)) cw ch)
            (setf px (+ sx (* scale self.l (np.sin (aref state 2))))
                  py (- cy (* scale self.l (np.cos (aref state 2)))))
            (painter.setPen (QPen pole_col 6))
            (painter.drawLine sx cy px py)
            (painter.setBrush (QBrush pole_col))
            (painter.setPen (QPen pole_col 1))
            (painter.drawEllipse (- px 8) (- py 8) 16 16))
          (draw_pend self.x_est  (QColor 180 100 30) (QColor 255 165 0))
          (draw_pend self.x_true (QColor 100 150 250) (QColor 250 100 100))
          (if (> (np.abs self.force) 0.1)
              (do0 (setf ax2 (+ cx (* (aref self.x_true 0) scale)))
                   (painter.setPen (QPen (QColor 0 255 0) 3))
                   (painter.drawLine ax2 cy (int (+ ax2 (* 3.0 self.force))) cy)))))


      ;; ==========================================
      ;; 4. MAIN WINDOW
      ;; ==========================================
      (class MainWindow (QMainWindow)
        (def __init__ (self)
          (QMainWindow.__init__ self)
          (self.setWindowTitle (string "Inverted Pendulum — MHE + MPC"))
          (self.resize 1500 950)

          (setf central (QWidget)
                root_layout (QHBoxLayout central)
                left_col (QVBoxLayout)
                right_col (QVBoxLayout))
          (self.setCentralWidget central)
          (root_layout.addLayout left_col 1)
          (root_layout.addLayout right_col 2)

          ;; --- pendulum canvas ---
          (setf self.pend_widget (PendulumWidget))
          (left_col.addWidget self.pend_widget)

          ;; --- timing label ---
          (setf self.lbl_delay (QLabel (string "Rechenzeit: OK")))
          (self.lbl_delay.setStyleSheet (string "font-size:13px;font-weight:bold;color:green"))
          (left_col.addWidget self.lbl_delay)

          ;; --- buttons ---
          (setf btn_row (QHBoxLayout)
                self.btn_start (QPushButton (string "Start"))
                self.btn_stop  (QPushButton (string "Stop"))
                self.btn_reset (QPushButton (string "Reset / Rebuild")))
          (btn_row.addWidget self.btn_start)
          (btn_row.addWidget self.btn_stop)
          (btn_row.addWidget self.btn_reset)
          (left_col.addLayout btn_row)
          (self.btn_start.clicked.connect self.start_sim)
          (self.btn_stop.clicked.connect  self.stop_sim)
          (self.btn_reset.clicked.connect self.reset_sim)

          ;; --- scrollable control panel ---
          (setf scroll_area (QScrollArea)
                ctrl_widget (QWidget)
                ctrl_layout (QVBoxLayout ctrl_widget))
          (scroll_area.setWidget ctrl_widget)
          (scroll_area.setWidgetResizable True)
          (left_col.addWidget scroll_area 1)

          ;; helper: slider inside a group grid
          (setf self.sliders (dict))
          (def make_slider (grid name label min_v max_v default scale row tip)
            (setf sl  (QSlider Qt.Horizontal)
                  lbl (QLabel (fstring "{label}: {default * scale:.3g}")))
            (sl.setRange min_v max_v)
            (sl.setValue default)
            (sl.setToolTip tip)
            (grid.addWidget lbl row 0)
            (grid.addWidget sl  row 1)
            (sl.valueChanged.connect (lambda () (lbl.setText (fstring "{label}: {sl.value() * scale:.3g}"))))
            (setf (aref self.sliders name) sl))

          (def make_group (title)
            (setf grp (QGroupBox title)
                  grd (QGridLayout))
            (grp.setLayout grd)
            (ctrl_layout.addWidget grp)
            (return grd))


          ;; ---- GROUP: Physik / Simulation ----
          (setf g (make_group (string "Physik / Simulation")))
          ,@(loop for (key label min-v max-v def scale tip) in
                  '(("M"      "Wagenmasse [kg]"      1  50  10  0.1   "Cart mass")
                    ("m"      "Pendelmasse [kg]"     1  20   1  0.1   "Pole tip mass")
                    ("l"      "Pendellänge [m]"      1  20   5  0.1   "Pole length")
                    ("wind"   "Wind [N]"          -300 300   0  0.1   "Horizontal wind force")
                    ("dt_sim" "Sim-dt [ms]"           1 100  20  1.0   "Simulation step size"))
                  for row from 0
                  collect `(make_slider g (string ,key) (string ,label)
                                        ,min-v ,max-v ,def ,scale ,row (string ,tip)))

          ;; ---- GROUP: Sensorrauschen (σ pro Kanal) ----
          (setf g (make_group (string "Sensorrauschen σ")))
          ,@(loop for (key label def) in
                  '(("sig_s"  "σ Position s"     2)
                    ("sig_v"  "σ Geschw. v"       5)
                    ("sig_th" "σ Winkel θ"        2)
                    ("sig_om" "σ Winkelgeschw. ω" 5))
                  for row from 0
                  collect `(make_slider g (string ,key) (string ,label)
                                        0 100 ,def 0.001 ,row (string "Gaussian noise std-dev (m or rad)")))

          ;; ---- GROUP: MPC — Horizont & Constraints ----
          (setf g (make_group (string "MPC — Horizont & Constraints")))
          ,@(loop for (key label min-v max-v def scale tip) in
                  '(("N_mpc"     "Horizont N_mpc"    2  80  40  1.0  "MPC steps — needs Rebuild")
                    ("target_s"  "Ziel s [m]"      -100 100  10  0.1  "Target cart position")
                    ("max_pos"   "Schienenlimit [m]" 10 200  50  0.1  "Track position limit")
                    ("max_force" "Max Kraft [N]"      0 300 150  0.1  "Actuator force limit"))
                  for row from 0
                  collect `(make_slider g (string ,key) (string ,label)
                                        ,min-v ,max-v ,def ,scale ,row (string ,tip)))
          (setf (aref self.sliders (string "N_mpc_label")) (aref self.sliders (string "N_mpc")))

          ;; ---- GROUP: MPC — Kostenfunktion ----
          (setf g (make_group (string "MPC — Kostenfunktion")))
          ,@(loop for (key label min-v max-v def scale tip) in
                  '(("Q_s"     "Gewicht s"        0 200  10  1.0   "State cost: position")
                    ("Q_v"     "Gewicht v"        0 100   1  1.0   "State cost: velocity")
                    ("Q_theta" "Gewicht θ"        0 500 100  1.0   "State cost: angle")
                    ("Q_omega" "Gewicht ω"        0 100   1  1.0   "State cost: angular vel")
                    ("R_F"     "Gewicht Kraft"    1 200  10  0.01  "Control effort cost"))
                  for row from 0
                  collect `(make_slider g (string ,key) (string ,label)
                                        ,min-v ,max-v ,def ,scale ,row (string ,tip)))


          ;; ---- GROUP: MHE — Horizont & Sensoren ----
          (setf g (make_group (string "MHE — Horizont & Sensoren")))
          (make_slider g (string "N_mhe") (string "Horizont N_mhe") 3 40 15 1.0 0
                       (string "MHE steps — needs Rebuild"))
          ;; sensor checkboxes in same group
          (setf self.cb_s  (QCheckBox (string "Sensor s aktiv"))
                self.cb_v  (QCheckBox (string "Sensor v aktiv"))
                self.cb_th (QCheckBox (string "Sensor θ aktiv"))
                self.cb_om (QCheckBox (string "Sensor ω aktiv")))
          (self.cb_s.setChecked True)
          (self.cb_th.setChecked True)
          (g.addWidget self.cb_s  1 0 1 2)
          (g.addWidget self.cb_v  2 0 1 2)
          (g.addWidget self.cb_th 3 0 1 2)
          (g.addWidget self.cb_om 4 0 1 2)

          ;; ---- GROUP: MHE — Gewichte ----
          (setf g (make_group (string "MHE — Gewichte")))
          ,@(loop for (key label def tip) in
                  '(("Q_w_scale" "Modellvertrauen Q_w"  50 "Process noise weight scale (higher = trust model more)")
                    ("R_w_scale" "Sensorvertrauen R_w"  50 "Measurement noise weight scale (higher = trust sensors more)")
                    ("P_w_scale" "Arrival Cost P_w"     50 "Prior cost weight scale"))
                  for row from 0
                  collect `(make_slider g (string ,key) (string ,label)
                                        1 200 ,def 0.1 ,row (string ,tip)))

          ;; ---- GROUP: Solver ----
          (setf g (make_group (string "Solver (IPOPT)")))
          (make_slider g (string "max_iter") (string "Max. Iterationen") 1 100 20 1.0 0
                       (string "IPOPT max iterations per solve — lower = faster but less accurate"))
          (setf self.cb_lbfgs (QCheckBox (string "L-BFGS Hesse-Approximation (schneller, ungenauer)")))
          (g.addWidget self.cb_lbfgs 1 0 1 2)


          ;; --- multi-panel plots ---
          (setf self.plot_layout (pg.GraphicsLayoutWidget))
          (right_col.addWidget self.plot_layout)
          (setf self.hist_curves (dict)
                self.est_curves  (dict)
                self.pred_curves (dict))
          ,@(loop for (key title ylabel row col) in
                  '(("s"       "Position"       "s [m]"   0 0)
                    ("theta"   "Winkel"         "rad"     1 0)
                    ("v"       "Geschwindigkeit" "m/s"    2 0)
                    ("omega"   "Winkelgeschw."  "rad/s"   0 1)
                    ("F"       "Kraft"          "N"       1 1)
                    ("t_solve" "Rechenzeit"     "ms"      2 1))
                  collect
                  `(do0
                    (setf ax (self.plot_layout.addPlot :row ,row :col ,col :title (string ,title)))
                    (ax.showGrid :x True :y True)
                    (ax.setLabel (string "left") (string ,ylabel))
                    (ax.setLabel (string "bottom") (string "t [s]"))
                    (setf (aref self.hist_curves (string ,key))
                          (ax.plot :pen (pg.mkPen (string "w") :width 2)))
                    (if (!= (string ,key) (string "t_solve"))
                        (do0
                         (setf (aref self.est_curves (string ,key))
                               (ax.plot :pen (pg.mkPen (string "orange") :width 2 :style Qt.DashLine)))
                         (setf (aref self.pred_curves (string ,key))
                               (ax.plot :pen (pg.mkPen (string "y") :width 2 :style Qt.DotLine)))))))

          (self.reset_sim)
          (self.start_sim))


        (def sv (self key) (return (dot (aref self.sliders key) (value))))

        (def get_params (self)
          (return (dictionary
                   :M        (* (self.sv (string "M"))        0.1)
                   :m        (* (self.sv (string "m"))        0.1)
                   :l        (* (self.sv (string "l"))        0.1)
                   :wind     (* (self.sv (string "wind"))     0.1)
                   :dt_sim   (* (self.sv (string "dt_sim"))   0.001)
                   :sig_s    (* (self.sv (string "sig_s"))    0.001)
                   :sig_v    (* (self.sv (string "sig_v"))    0.001)
                   :sig_th   (* (self.sv (string "sig_th"))   0.001)
                   :sig_om   (* (self.sv (string "sig_om"))   0.001)
                   :N_mpc    (int (self.sv (string "N_mpc")))
                   :target_s (* (self.sv (string "target_s")) 0.1)
                   :max_pos  (* (self.sv (string "max_pos"))  0.1)
                   :max_force(* (self.sv (string "max_force"))0.1)
                   :Q_s      (float (self.sv (string "Q_s")))
                   :Q_v      (float (self.sv (string "Q_v")))
                   :Q_theta  (float (self.sv (string "Q_theta")))
                   :Q_omega  (float (self.sv (string "Q_omega")))
                   :R_F      (* (self.sv (string "R_F"))      0.01)
                   :N_mhe    (int (self.sv (string "N_mhe")))
                   :Q_w_scale(* (self.sv (string "Q_w_scale"))0.1)
                   :R_w_scale(* (self.sv (string "R_w_scale"))0.1)
                   :P_w_scale(* (self.sv (string "P_w_scale"))0.1)
                   :max_iter (int (self.sv (string "max_iter")))
                   :use_lbfgs(self.cb_lbfgs.isChecked))))

        (def start_sim (self)
          (setf p (self.get_params))
          ;; disable N sliders while running
          (dot (aref self.sliders (string "N_mpc")) (setEnabled False))
          (dot (aref self.sliders (string "N_mhe")) (setEnabled False))
          (self.timer.start (int (* (aref p (string "dt_sim")) 1000.0))))

        (def stop_sim (self)
          (self.timer.stop)
          (dot (aref self.sliders (string "N_mpc")) (setEnabled True))
          (dot (aref self.sliders (string "N_mhe")) (setEnabled True)))

        (def rebuild_solvers (self)
          (setf p (self.get_params))
          (print (fstring "Building solvers: N_mhe={p['N_mhe']} N_mpc={p['N_mpc']} max_iter={p['max_iter']} lbfgs={p['use_lbfgs']}"))
          (setf (ntuple self.opti_mhe self.X_mhe self.W_mhe
                        self.Y_meas_param self.U_past_param self.X_prior_param
                        self.P_dyn_param self.dt_mhe_param
                        self.Q_w_p self.R_w_p self.P_w_p
                        self.opti_mpc self.X_mpc self.U_mpc
                        self.X_cur_p self.X_tgt_p self.P_dyn_mpc self.dt_mpc_p
                        self.Q_s_p self.Q_v_p self.Q_th_p self.Q_om_p
                        self.R_F_p self.max_pos_p self.max_F_p self.N_mpc_cur)
                (build_solvers (aref p (string "N_mhe"))
                               (aref p (string "N_mpc"))
                               (aref p (string "max_iter"))
                               (aref p (string "use_lbfgs"))))
          (setf self.N_mhe_cur (aref p (string "N_mhe"))))

        (def reset_sim (self)
          (if (hasattr self (string "timer")) (self.timer.stop))
          (self.rebuild_solvers)
          (setf self.x_true      (np.array (list 0.0 0.0 3.14159 0.0))
                self.u_last      0.0
                self.t_curr      0.0
                self.Y_hist      (list)
                self.U_hist      (list)
                self.x_prior_val (np.copy self.x_true)
                self.t_data      (list)
                self.hist        (dictionary :s (list) :v (list) :theta (list)
                                             :omega (list) :F (list) :t_solve (list))
                self.est_hist    (dictionary :s (list) :v (list) :theta (list)
                                             :omega (list) :F (list))
                self.timer       (QTimer))
          (self.timer.timeout.connect self.sim_step))


        (def sim_step (self)
          (setf t_start (time.time)
                p       (self.get_params)
                dt_val  (aref p (string "dt_sim"))
                p_dyn_val (np.array (list (aref p (string "M")) (aref p (string "m"))
                                         (aref p (string "l")) (aref p (string "wind")))))

          ;; -- true physics --
          (setf res (f_rk4 self.x_true self.u_last p_dyn_val dt_val)
                self.x_true (np.array (res.elements)))

          ;; -- noisy measurement per channel --
          (setf y_meas (np.copy self.x_true))
          ,@(loop for (i sig-key) in '((0 "sig_s") (1 "sig_v") (2 "sig_th") (3 "sig_om"))
                  collect `(setf (aref y_meas ,i)
                                 (+ (aref y_meas ,i)
                                    (np.random.normal 0.0 (aref p (string ,sig-key))))))

          ;; -- buffer --
          (self.Y_hist.append (np.copy y_meas))
          (self.U_hist.append (list self.u_last))
          (if (> (len self.Y_hist) (+ self.N_mhe_cur 1))
              (do0 (self.Y_hist.pop 0) (self.U_hist.pop 0)))

          ;; -- MHE --
          (setf x_est (np.copy y_meas))
          (if (== (len self.Y_hist) (+ self.N_mhe_cur 1))
              (do0
               (setf Q_w_val (* (aref p (string "Q_w_scale")) (np.ones 4))
                     P_w_val (* (aref p (string "P_w_scale")) (np.ones 4))
                     r_base  (* (aref p (string "R_w_scale")) (np.array (list 10.0 1.0 10.0 1.0)))
                     R_w_val (np.array (list (* (aref r_base 0) (? (self.cb_s.isChecked)  1.0 0.0001))
                                            (* (aref r_base 1) (? (self.cb_v.isChecked)  1.0 0.0001))
                                            (* (aref r_base 2) (? (self.cb_th.isChecked) 1.0 0.0001))
                                            (* (aref r_base 3) (? (self.cb_om.isChecked) 1.0 0.0001)))))
               (self.opti_mhe.set_value self.Y_meas_param (dot (np.array self.Y_hist) "T"))
               (self.opti_mhe.set_value self.U_past_param (dot (np.array (aref self.U_hist (slice 0 self.N_mhe_cur))) "T"))
               (self.opti_mhe.set_value self.X_prior_param (aref self.x_prior_val (slice nil nil) np.newaxis))
               (self.opti_mhe.set_value self.P_dyn_param   (aref p_dyn_val (slice nil nil) np.newaxis))
               (self.opti_mhe.set_value self.dt_mhe_param  dt_val)
               (self.opti_mhe.set_value self.Q_w_p (aref Q_w_val (slice nil nil) np.newaxis))
               (self.opti_mhe.set_value self.R_w_p (aref R_w_val (slice nil nil) np.newaxis))
               (self.opti_mhe.set_value self.P_w_p (aref P_w_val (slice nil nil) np.newaxis))
               (try
                (do0
                 (setf sol_mhe (self.opti_mhe.solve)
                       X_res   (sol_mhe.value self.X_mhe)
                       x_est   (aref X_res (slice nil nil) -1))
                 (setf self.x_prior_val (aref X_res (slice nil nil) 1)))
                ((as Exception e) (print (fstring "MHE: {e}"))))))

          ;; -- MPC --
          (setf tgt (np.array (list (aref p (string "target_s")) 0.0 0.0 0.0)))
          (self.opti_mpc.set_value self.X_cur_p  (aref x_est (slice nil nil) np.newaxis))
          (self.opti_mpc.set_value self.X_tgt_p  (aref tgt   (slice nil nil) np.newaxis))
          (self.opti_mpc.set_value self.P_dyn_mpc (aref p_dyn_val (slice nil nil) np.newaxis))
          (self.opti_mpc.set_value self.dt_mpc_p  dt_val)
          (self.opti_mpc.set_value self.Q_s_p   (aref p (string "Q_s")))
          (self.opti_mpc.set_value self.Q_v_p   (aref p (string "Q_v")))
          (self.opti_mpc.set_value self.Q_th_p  (aref p (string "Q_theta")))
          (self.opti_mpc.set_value self.Q_om_p  (aref p (string "Q_omega")))
          (self.opti_mpc.set_value self.R_F_p   (aref p (string "R_F")))
          (self.opti_mpc.set_value self.max_pos_p (aref p (string "max_pos")))
          (self.opti_mpc.set_value self.max_F_p   (aref p (string "max_force")))
          (setf mpc_ok False X_pred None U_pred None)
          (try
           (do0
            (setf sol (self.opti_mpc.solve)
                  self.u_last (aref (sol.value self.U_mpc) 0 0)
                  X_pred (sol.value self.X_mpc)
                  U_pred (sol.value self.U_mpc)
                  mpc_ok True))
           ((as Exception e) (print (fstring "MPC: {e}"))))

          ;; -- timing label --
          (setf t_calc (- (time.time) t_start))
          (if (> t_calc dt_val)
              (do0 (self.lbl_delay.setText (fstring "WARNUNG: {(t_calc*1000):.1f}ms > {(dt_val*1000):.0f}ms!"))
                   (self.lbl_delay.setStyleSheet (string "font-size:13px;font-weight:bold;color:red")))
              (do0 (self.lbl_delay.setText (fstring "Rechenzeit: {(t_calc*1000):.1f}ms (OK)"))
                   (self.lbl_delay.setStyleSheet (string "font-size:13px;font-weight:bold;color:green"))))

          (self.pend_widget.update_state self.x_true x_est self.u_last
                                         (aref p (string "l")) (aref p (string "max_pos")))

          ;; -- history / plots --
          (setf self.t_curr (+ self.t_curr dt_val))
          (self.t_data.append self.t_curr)
          ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
                  collect `(do0
                            (dot (aref self.hist (string ,key)) (append (aref self.x_true ,idx)))
                            (dot (aref self.est_hist (string ,key)) (append (aref x_est ,idx)))))
          (dot (aref self.hist     (string "F")) (append self.u_last))
          (dot (aref self.est_hist (string "F")) (append self.u_last))
          (dot (aref self.hist (string "t_solve")) (append (* t_calc 1000.0)))
          (if (> (len self.t_data) 200)
              (do0
               (self.t_data.pop 0)
               ,@(loop for k in '("s" "v" "theta" "omega" "F" "t_solve")
                       collect `(dot (aref self.hist (string ,k)) (pop 0)))
               ,@(loop for k in '("s" "v" "theta" "omega" "F")
                       collect `(dot (aref self.est_hist (string ,k)) (pop 0)))))
          ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
                  collect `(dot (aref self.hist_curves (string ,key))
                                (setData self.t_data (aref self.hist (string ,key)))))
          ,@(loop for key in '("s" "v" "theta" "omega" "F")
                  collect `(dot (aref self.est_curves (string ,key))
                                (setData self.t_data (aref self.est_hist (string ,key)))))
          (if (and mpc_ok (is-not X_pred None))
              (do0
               (setf t_pred (np.linspace self.t_curr
                                         (+ self.t_curr (* self.N_mpc_cur dt_val))
                                         (+ self.N_mpc_cur 1)))
               ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
                       collect `(dot (aref self.pred_curves (string ,key))
                                     (setData t_pred (aref X_pred ,idx (slice nil nil)))))
               (dot (aref self.pred_curves (string "F"))
                    (setData (aref t_pred (slice 0 -1)) (aref U_pred 0 (slice nil nil))))))))

      ;; main
      (setf app (QApplication sys.argv) win (MainWindow))
      (win.show)
      (sys.exit (app.exec))))

  (write-source *code-file* *source*))
