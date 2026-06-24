;; gen11h.lisp
;; MHE + MPC for inverted pendulum with full visualization from gen11f
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
                     QGridLayout QSlider QLabel QPushButton QGroupBox QCheckBox)
                    (PySide6.QtCore Qt QTimer)
                    (PySide6.QtGui QPainter QPen QBrush QColor))


      ;; ==========================================
      ;; 1. SYSTEM DYNAMICS
      ;; ==========================================
      (setf x_sym (ca.MX.sym (string "x") 4)
            u_sym (ca.MX.sym (string "u") 1))
      (setf s_ (aref x_sym 0) v_ (aref x_sym 1) th_ (aref x_sym 2) om_ (aref x_sym 3))
      (setf sin_th (ca.sin th_) cos_th (ca.cos th_))

      ;; Parameters as MX symbols so we can pass them into map
      (setf p_dyn (ca.MX.sym (string "p") 4))   ; [M, m, l, wind]
      (setf M_ (aref p_dyn 0) m_ (aref p_dyn 1) l_ (aref p_dyn 2) w_ (aref p_dyn 3))
      (setf denom (+ M_ (* m_ (- 1 (** cos_th 2)))))
      (setf s_dot v_
            v_dot (/ (+ (+ u_sym (* w_ cos_th))
                        (* m_ sin_th (+ (* l_ (** om_ 2)) (* 9.81 cos_th))))
                     denom)
            th_dot om_
            om_dot (/ (- (* -1 (+ u_sym (* w_ cos_th)) cos_th)
                         (* m_ l_ (** om_ 2) sin_th cos_th)
                         (* (+ M_ m_) 9.81 sin_th))
                      (* l_ denom)))
      (setf f_ode (ca.Function (string "f_ode")
                               (list x_sym u_sym p_dyn)
                               (list (ca.vertcat s_dot v_dot th_dot om_dot))))


      ;; RK4 integrator (fixed dt symbol)
      (setf dt_sym (ca.MX.sym (string "dt") 1))
      (setf rk4_k1 (f_ode x_sym u_sym p_dyn)
            rk4_k2 (f_ode (+ x_sym (* (/ dt_sym 2) rk4_k1)) u_sym p_dyn)
            rk4_k3 (f_ode (+ x_sym (* (/ dt_sym 2) rk4_k2)) u_sym p_dyn)
            rk4_k4 (f_ode (+ x_sym (* dt_sym rk4_k3)) u_sym p_dyn)
            x_next_expr (+ x_sym (* (/ dt_sym 6) (+ rk4_k1 (* 2 rk4_k2) (* 2 rk4_k3) rk4_k4))))
      (setf f_rk4 (ca.Function (string "f_rk4") (list x_sym u_sym p_dyn dt_sym) (list x_next_expr)))

      ;; ==========================================
      ;; 2. MHE SETUP
      ;; ==========================================
      (setf N_mhe 15
            opti_mhe (ca.Opti))
      (setf X_mhe  (opti_mhe.variable 4 (+ N_mhe 1))
            W_mhe  (opti_mhe.variable 4 N_mhe))
      (setf Y_meas_param  (opti_mhe.parameter 4 (+ N_mhe 1))
            U_past_param  (opti_mhe.parameter 1 N_mhe)
            X_prior_param (opti_mhe.parameter 4 1)
            P_dyn_param   (opti_mhe.parameter 4 1)  ; [M,m,l,wind]
            dt_mhe_param  (opti_mhe.parameter 1 1)
            Q_w (opti_mhe.parameter 4 1)
            R_w (opti_mhe.parameter 4 1)
            P_w (opti_mhe.parameter 4 1))


      ;; Map trick: repeat p_dyn and dt over N_mhe columns
      (setf f_rk4_map_mhe (f_rk4.map N_mhe))
      (setf p_stacked_mhe (ca.repmat P_dyn_param 1 N_mhe)
            dt_stacked_mhe (ca.repmat dt_mhe_param 1 N_mhe))
      (setf X_next_calc_mhe (f_rk4_map_mhe
                              (aref X_mhe (slice nil nil) (slice 0 N_mhe))
                              U_past_param
                              p_stacked_mhe
                              dt_stacked_mhe))
      (opti_mhe.subject_to (== (aref X_mhe (slice nil nil) (slice 1 (+ N_mhe 1)))
                               (+ X_next_calc_mhe W_mhe)))
      (setf cost_mhe (+ (ca.sum1 (* P_w (** (- (aref X_mhe (slice nil nil) 0) X_prior_param) 2)))
                        (ca.sum2 (ca.sum1 (* Q_w (** W_mhe 2))))
                        (ca.sum2 (ca.sum1 (* R_w (** (- Y_meas_param X_mhe) 2))))))
      (opti_mhe.minimize cost_mhe)
      (opti_mhe.solver (string "ipopt")
                       (dictionary :print_time False)
                       (dictionary :print_level 0 :sb (string "yes")))

      ;; ==========================================
      ;; 3. MPC SETUP
      ;; ==========================================
      (setf N_mpc_sym 40
            opti_mpc (ca.Opti))
      (setf X_mpc (opti_mpc.variable 4 (+ N_mpc_sym 1))
            U_mpc (opti_mpc.variable 1 N_mpc_sym))
      (setf X_cur_param  (opti_mpc.parameter 4 1)
            X_tgt_param  (opti_mpc.parameter 4 1)
            P_dyn_mpc    (opti_mpc.parameter 4 1)
            dt_mpc_param (opti_mpc.parameter 1 1)
            Q_s_p  (opti_mpc.parameter) Q_v_p  (opti_mpc.parameter)
            Q_th_p (opti_mpc.parameter) Q_om_p (opti_mpc.parameter)
            R_F_p  (opti_mpc.parameter)
            max_pos_p (opti_mpc.parameter) max_F_p (opti_mpc.parameter))


      (setf f_rk4_map_mpc (f_rk4.map N_mpc_sym))
      (setf p_stacked_mpc  (ca.repmat P_dyn_mpc 1 N_mpc_sym)
            dt_stacked_mpc (ca.repmat dt_mpc_param 1 N_mpc_sym))
      (setf X_next_calc_mpc (f_rk4_map_mpc
                              (aref X_mpc (slice nil nil) (slice 0 N_mpc_sym))
                              U_mpc
                              p_stacked_mpc
                              dt_stacked_mpc))
      (opti_mpc.subject_to (== (aref X_mpc (slice nil nil) 0) X_cur_param))
      (opti_mpc.subject_to (== (aref X_mpc (slice nil nil) (slice 1 (+ N_mpc_sym 1))) X_next_calc_mpc))
      (opti_mpc.subject_to (opti_mpc.bounded (* -1 max_pos_p)
                                             (aref X_mpc 0 (slice nil nil))
                                             max_pos_p))
      (opti_mpc.subject_to (opti_mpc.bounded (* -1 max_F_p) U_mpc max_F_p))

      (setf err_mpc (- X_mpc (ca.repmat X_tgt_param 1 (+ N_mpc_sym 1)))
            Q_diag  (ca.vertcat Q_s_p Q_v_p Q_th_p Q_om_p)
            cost_mpc (+ (ca.sum2 (ca.sum1 (* Q_diag (** err_mpc 2))))
                        (* R_F_p (ca.sum2 (** U_mpc 2)))))
      (opti_mpc.minimize cost_mpc)
      (opti_mpc.solver (string "ipopt")
                       (dictionary :print_time False)
                       (dictionary :print_level 0 :sb (string "yes")))


      ;; ==========================================
      ;; 4. PENDULUM WIDGET (from gen11f)
      ;; ==========================================
      (class PendulumWidget (QWidget)
        (def __init__ (self)
          (QWidget.__init__ self)
          (self.setMinimumSize 400 300)
          (setf self.x_true  (np.array (list 0.0 0.0 3.14159 0.0))
                self.x_est   (np.array (list 0.0 0.0 3.14159 0.0))
                self.force   0.0
                self.l       0.5
                self.max_pos 2.0))

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
          ;; rail
          (painter.setPen (QPen (QColor 128 128 128) 4))
          (painter.drawLine (- cx (* scale self.max_pos)) cy
                            (+ cx (* scale self.max_pos)) cy)
          ;; draw one pendulum given state, cart color, pole color
          (def draw_pend (state cart_col pole_col)
            (setf s_    (aref state 0)
                  theta (aref state 2)
                  cart_x (+ cx (* s_ scale))
                  cart_w (* 0.4 scale) cart_h (* 0.2 scale))
            (painter.setBrush (QBrush cart_col))
            (painter.setPen (QPen (QColor 255 255 255) 2))
            (painter.drawRect (- cart_x (/ cart_w 2)) (- cy (/ cart_h 2)) cart_w cart_h)
            (setf px (+ cart_x (* scale self.l (np.sin theta)))
                  py (- cy  (* scale self.l (np.cos theta))))
            (painter.setPen (QPen pole_col 6))
            (painter.drawLine cart_x cy px py)
            (painter.setBrush (QBrush pole_col))
            (painter.setPen (QPen pole_col 1))
            (painter.drawEllipse (- px 8) (- py 8) 16 16))
          ;; estimated state (dashed, orange)
          (draw_pend self.x_est (QColor 180 100 30) (QColor 255 165 0))
          ;; true state (solid, blue/red)
          (draw_pend self.x_true (QColor 100 150 250) (QColor 250 100 100))
          ;; force arrow
          (if (> (np.abs self.force) 0.1)
              (do0
               (setf ax (+ cx (* (aref self.x_true 0) scale)))
               (painter.setPen (QPen (QColor 0 255 0) 3))
               (painter.drawLine ax cy (int (+ ax (* 3.0 self.force))) cy)))))


      ;; ==========================================
      ;; 5. MAIN WINDOW
      ;; ==========================================
      (class MainWindow (QMainWindow)
        (def __init__ (self)
          (QMainWindow.__init__ self)
          (self.setWindowTitle (string "Inverted Pendulum - MHE + MPC"))
          (self.resize 1400 900)

          (setf central (QWidget)
                main_layout (QHBoxLayout central)
                left_layout (QVBoxLayout)
                right_layout (QVBoxLayout))
          (self.setCentralWidget central)
          (main_layout.addLayout left_layout 1)
          (main_layout.addLayout right_layout 2)

          ;; pendulum canvas
          (setf self.pend_widget (PendulumWidget))
          (left_layout.addWidget self.pend_widget)

          ;; delay label
          (setf self.lbl_delay (QLabel (string "Rechenzeit: OK")))
          (self.lbl_delay.setStyleSheet (string "font-size:13px;font-weight:bold;color:green"))
          (left_layout.addWidget self.lbl_delay)

          ;; buttons
          (setf btn_layout (QHBoxLayout)
                self.btn_start (QPushButton (string "Start"))
                self.btn_stop  (QPushButton (string "Stop"))
                self.btn_reset (QPushButton (string "Reset")))
          (btn_layout.addWidget self.btn_start)
          (btn_layout.addWidget self.btn_stop)
          (btn_layout.addWidget self.btn_reset)
          (left_layout.addLayout btn_layout)
          (self.btn_start.clicked.connect self.start_sim)
          (self.btn_stop.clicked.connect  self.stop_sim)
          (self.btn_reset.clicked.connect self.reset_sim)


          ;; sliders
          (setf self.sliders (dict)
                slider_layout (QGridLayout))
          (left_layout.addLayout slider_layout)

          (def add_slider (name label min_v max_v default scale row tip)
            (setf sl  (QSlider Qt.Horizontal)
                  lbl (QLabel (fstring "{label}: {default * scale:.2f}")))
            (sl.setRange min_v max_v)
            (sl.setValue default)
            (sl.setToolTip tip)
            (slider_layout.addWidget lbl row 0)
            (slider_layout.addWidget sl  row 1)
            (sl.valueChanged.connect (lambda () (lbl.setText (fstring "{label}: {sl.value() * scale:.2f}"))))
            (setf (aref self.sliders name) sl))

          ,@(loop for (key label min-v max-v def scale tip) in
                  '(("M"        "Wagenmasse [kg]"     1  50  10  0.1  "Cart mass")
                    ("m"        "Pendelmasse [kg]"    1  20   1  0.1  "Pole mass")
                    ("l"        "Pendellänge [m]"     1  20   5  0.1  "Pole length")
                    ("wind"     "Wind [N]"          -300 300  0  0.1  "Horizontal wind disturbance")
                    ("Q_s"      "Gewicht s"           0 200  10  1.0  "MPC position cost")
                    ("Q_v"      "Gewicht v"           0 100   1  1.0  "MPC velocity cost")
                    ("Q_theta"  "Gewicht theta"       0 500 100  1.0  "MPC angle cost")
                    ("Q_omega"  "Gewicht omega"       0 100   1  1.0  "MPC ang. vel. cost")
                    ("R_F"      "Gewicht Kraft"       1 200  10  0.01 "MPC force cost")
                    ("target_s" "Ziel-s [m]"       -100 100  10  0.1  "Target cart position")
                    ("max_pos"  "Schiene [m]"        10 200  50  0.1  "Position limit")
                    ("max_force""Max Kraft [N]"      10 300 150  0.1  "Force limit")
                    ("dt_sim"   "Sim-dt [ms]"         1 100  20  1.0  "Simulation step ms"))
                  for row from 0
                  collect `(add_slider (string ,key) (string ,label) ,min-v ,max-v ,def ,scale ,row (string ,tip)))

          ;; sensor checkboxes
          (setf sensor_box (QGroupBox (string "Sensor aktiv (MHE)"))
                sb_layout (QVBoxLayout)
                self.cb_s  (QCheckBox (string "s (Position)"))
                self.cb_v  (QCheckBox (string "v (Geschw.)"))
                self.cb_th (QCheckBox (string "theta (Winkel)"))
                self.cb_om (QCheckBox (string "omega (Winkelgeschw.)")))
          (self.cb_s.setChecked True)
          (self.cb_th.setChecked True)
          (sb_layout.addWidget self.cb_s)
          (sb_layout.addWidget self.cb_v)
          (sb_layout.addWidget self.cb_th)
          (sb_layout.addWidget self.cb_om)
          (sensor_box.setLayout sb_layout)
          (left_layout.addWidget sensor_box)


          ;; multi-panel plots
          (setf self.plot_layout (pg.GraphicsLayoutWidget))
          (right_layout.addWidget self.plot_layout)
          (setf self.hist_curves (dict)
                self.est_curves  (dict)
                self.pred_curves (dict))

          ,@(loop for (key title ylabel row col) in
                  '(("s"       "Position"      "s [m]"       0 0)
                    ("theta"   "Winkel"        "rad"         1 0)
                    ("v"       "Geschwindigkeit""v [m/s]"    2 0)
                    ("omega"   "Winkelgeschw." "rad/s"       0 1)
                    ("F"       "Kraft"         "F [N]"       1 1)
                    ("t_solve" "Rechenzeit"    "ms"          2 1))
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


          ;; state
          (self.reset_sim)
          (self.start_sim))

        (def get_params (self)
          (return (dictionary
                   :M        (/ (dot (aref self.sliders (string "M"))        (value)) 10.0)
                   :m        (/ (dot (aref self.sliders (string "m"))        (value)) 10.0)
                   :l        (/ (dot (aref self.sliders (string "l"))        (value)) 10.0)
                   :wind     (/ (dot (aref self.sliders (string "wind"))     (value)) 10.0)
                   :Q_s      (float (dot (aref self.sliders (string "Q_s"))     (value)))
                   :Q_v      (float (dot (aref self.sliders (string "Q_v"))     (value)))
                   :Q_theta  (float (dot (aref self.sliders (string "Q_theta")) (value)))
                   :Q_omega  (float (dot (aref self.sliders (string "Q_omega")) (value)))
                   :R_F      (/ (dot (aref self.sliders (string "R_F"))     (value)) 100.0)
                   :target_s (/ (dot (aref self.sliders (string "target_s"))(value)) 10.0)
                   :max_pos  (/ (dot (aref self.sliders (string "max_pos")) (value)) 10.0)
                   :max_force(/ (dot (aref self.sliders (string "max_force"))(value)) 10.0)
                   :dt_sim   (/ (dot (aref self.sliders (string "dt_sim"))  (value)) 1000.0))))

        (def start_sim (self)
          (setf p (self.get_params))
          (self.timer.start (int (* (aref p (string "dt_sim")) 1000.0))))

        (def stop_sim (self) (self.timer.stop))

        (def reset_sim (self)
          (if (hasattr self (string "timer"))
              (self.timer.stop))
          (setf self.x_true      (np.array (list 0.0 0.0 3.14159 0.0))
                self.u_last      0.0
                self.t_curr      0.0
                self.Y_hist      (list)
                self.U_hist      (list)
                self.x_prior_val (np.copy self.x_true)
                self.x_est_last  (np.copy self.x_true)
                self.t_data      (list)
                self.hist        (dictionary
                                  :s (list) :v (list) :theta (list) :omega (list)
                                  :F (list) :t_solve (list))
                self.est_hist    (dictionary
                                  :s (list) :v (list) :theta (list) :omega (list) :F (list))
                self.timer       (QTimer))
          (self.timer.timeout.connect self.sim_step))


        (def sim_step (self)
          (setf t_start (time.time)
                p (self.get_params)
                dt_val (aref p (string "dt_sim"))
                p_dyn_val (np.array (list (aref p (string "M"))
                                         (aref p (string "m"))
                                         (aref p (string "l"))
                                         (aref p (string "wind")))))

          ;; -- true physics --
          (setf res_true (f_rk4 self.x_true self.u_last p_dyn_val dt_val)
                self.x_true (np.array (res_true.elements)))

          ;; -- noisy measurement (all 4 states, checkbox-gated) --
          (setf y_meas (np.copy self.x_true)
                noise_std 0.02)
          ,@(loop for i from 0 to 3
                  collect `(setf (aref y_meas ,i)
                                 (+ (aref y_meas ,i) (np.random.normal 0.0 noise_std))))

          ;; -- buffer --
          (self.Y_hist.append (np.copy y_meas))
          (self.U_hist.append (list self.u_last))
          (if (> (len self.Y_hist) (+ N_mhe 1))
              (do0 (self.Y_hist.pop 0) (self.U_hist.pop 0)))

          ;; -- MHE --
          (setf x_est (np.copy y_meas))
          (if (== (len self.Y_hist) (+ N_mhe 1))
              (do0
               (setf Y_mat (dot (np.array self.Y_hist) "T")
                     U_mat (dot (np.array (aref self.U_hist (slice 0 N_mhe))) "T"))
               (opti_mhe.set_value Y_meas_param Y_mat)
               (opti_mhe.set_value U_past_param U_mat)
               (opti_mhe.set_value X_prior_param (aref self.x_prior_val (slice nil nil) np.newaxis))
               (opti_mhe.set_value P_dyn_param   (aref p_dyn_val (slice nil nil) np.newaxis))
               (opti_mhe.set_value dt_mhe_param  dt_val)
               (opti_mhe.set_value Q_w (np.array (list 1.0 1.0 1.0 1.0)))
               (opti_mhe.set_value P_w (np.array (list 10.0 10.0 10.0 10.0)))
               (setf r_vec (list (? (self.cb_s.isChecked)  100.0 0.001)
                                 (? (self.cb_v.isChecked)  100.0 0.001)
                                 (? (self.cb_th.isChecked) 1000.0 0.001)
                                 (? (self.cb_om.isChecked) 100.0 0.001)))
               (opti_mhe.set_value R_w (np.array r_vec))
               (try
                (do0
                 (setf sol_mhe (opti_mhe.solve)
                       X_res   (sol_mhe.value X_mhe)
                       x_est   (aref X_res (slice nil nil) -1))
                 (setf self.x_prior_val (aref X_res (slice nil nil) 1)))
                ((as Exception e) (print (fstring "MHE failed: {e}"))))))
          (setf self.x_est_last x_est)


          ;; -- MPC (fed with x_est) --
          (setf target_s (aref p (string "target_s"))
                tgt_state (np.array (list target_s 0.0 0.0 0.0))
                x_est_col (aref x_est (slice nil nil) np.newaxis))
          (opti_mpc.set_value X_cur_param  x_est_col)
          (opti_mpc.set_value X_tgt_param  (aref tgt_state (slice nil nil) np.newaxis))
          (opti_mpc.set_value P_dyn_mpc    (aref p_dyn_val (slice nil nil) np.newaxis))
          (opti_mpc.set_value dt_mpc_param dt_val)
          (opti_mpc.set_value Q_s_p   (aref p (string "Q_s")))
          (opti_mpc.set_value Q_v_p   (aref p (string "Q_v")))
          (opti_mpc.set_value Q_th_p  (aref p (string "Q_theta")))
          (opti_mpc.set_value Q_om_p  (aref p (string "Q_omega")))
          (opti_mpc.set_value R_F_p   (aref p (string "R_F")))
          (opti_mpc.set_value max_pos_p (aref p (string "max_pos")))
          (opti_mpc.set_value max_F_p   (aref p (string "max_force")))

          (setf mpc_success False
                X_pred_val None U_pred_val None)
          (try
           (do0
            (setf sol_mpc (opti_mpc.solve)
                  self.u_last (aref (sol_mpc.value U_mpc) 0 0)
                  X_pred_val  (sol_mpc.value X_mpc)
                  U_pred_val  (sol_mpc.value U_mpc)
                  mpc_success True))
           ((as Exception e) (print (fstring "MPC failed: {e}"))))

          ;; -- timing --
          (setf t_calc (- (time.time) t_start))
          (if (> t_calc dt_val)
              (do0
               (self.lbl_delay.setText (fstring "WARNUNG: {(t_calc*1000):.1f}ms > {(dt_val*1000):.0f}ms!"))
               (self.lbl_delay.setStyleSheet (string "font-size:13px;font-weight:bold;color:red")))
              (do0
               (self.lbl_delay.setText (fstring "Rechenzeit: {(t_calc*1000):.1f}ms (OK)"))
               (self.lbl_delay.setStyleSheet (string "font-size:13px;font-weight:bold;color:green"))))

          ;; -- update pendulum widget --
          (self.pend_widget.update_state self.x_true x_est self.u_last
                                         (aref p (string "l"))
                                         (aref p (string "max_pos")))

          ;; -- history --
          (setf self.t_curr (+ self.t_curr dt_val))
          (self.t_data.append self.t_curr)
          ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
                  collect `(do0
                            (dot (aref self.hist (string ,key)) (append (aref self.x_true ,idx)))
                            (dot (aref self.est_hist (string ,key)) (append (aref x_est ,idx)))))
          (dot (aref self.hist (string "F")) (append self.u_last))
          (dot (aref self.est_hist (string "F")) (append self.u_last))
          (dot (aref self.hist (string "t_solve")) (append (* t_calc 1000.0)))

          (if (> (len self.t_data) 200)
              (do0
               (self.t_data.pop 0)
               ,@(loop for k in '("s" "v" "theta" "omega" "F" "t_solve")
                       collect `(dot (aref self.hist (string ,k)) (pop 0)))
               ,@(loop for k in '("s" "v" "theta" "omega" "F")
                       collect `(dot (aref self.est_hist (string ,k)) (pop 0)))))

          ;; -- update plots --
          ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
                  collect `(dot (aref self.hist_curves (string ,key))
                                (setData self.t_data (aref self.hist (string ,key)))))
          ,@(loop for key in '("s" "v" "theta" "omega" "F")
                  collect `(dot (aref self.est_curves (string ,key))
                                (setData self.t_data (aref self.est_hist (string ,key)))))

          (if (and mpc_success (is-not X_pred_val None))
              (do0
               (setf t_pred (np.linspace self.t_curr
                                         (+ self.t_curr (* N_mpc_sym dt_val))
                                         (+ N_mpc_sym 1)))
               ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
                       collect `(dot (aref self.pred_curves (string ,key))
                                     (setData t_pred (aref X_pred_val ,idx (slice nil nil)))))
               (dot (aref self.pred_curves (string "F"))
                    (setData (aref t_pred (slice 0 -1))
                             (aref U_pred_val 0 (slice nil nil))))))))

      ;; main
      (setf app (QApplication sys.argv)
            win (MainWindow))
      (win.show)
      (sys.exit (app.exec))))

  (write-source *code-file* *source*))
