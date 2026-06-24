;; gen11g.lisp
;; Moving Horizon Estimation (MHE) & Model Predictive Control (MPC) für invertiertes Pendel
;; Nutzt CasADi's 'map' Funktion um C-Code Aufblähung zu verhindern.

;; run with:
;; sbcl --load gen11g.lisp --eval '(sb-ext:quit)'
;; uv run p11g_mpc_gui.py
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *source-py* "example/171_casadi/")
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p11g_mpc_gui"
						   *source-py*)))
  (defparameter *source*
    `(do0
      (imports (sys time
                    (np numpy)
                    (ca casadi)))
      (imports-from (PySide6.QtWidgets *)
                    (PySide6.QtCore *)
                    (PySide6.QtGui *))
      (imports-from (pyqtgraph PlotWidget plot mkPen))

      ;; ==========================================
      ;; 1. PHYSIKALISCHE PARAMETER
      ;; ==========================================
      (setf m_cart 1.0
            m_pend 0.1
            l 0.5
            g 9.81
            dt 0.02)
      
      (setf N_mpc 40   ; MPC blickt in die Zukunft
            N_mhe 15)  ; MHE blickt in die Vergangenheit

      ;; ==========================================
      ;; 2. SYSTEMDYNAMIK (CasADi Funktion)
      ;; ==========================================
      ;; Zustandsvektor: x = [s, v, theta, omega]
      (setf x_sym (ca.MX.sym (string "x") 4)
            u_sym (ca.MX.sym (string "u") 1))
      
      (setf s_ (aref x_sym 0)
            v_ (aref x_sym 1)
            th_ (aref x_sym 2)
            om_ (aref x_sym 3))
      
      (setf sin_th (ca.sin th_)
            cos_th (ca.cos th_)
            denom (+ m_cart (* m_pend (- 1 (** cos_th 2)))))
      
      ;; Differentialgleichungen (ODE)
      (setf s_dot v_
            v_dot (/ (+ u_sym 
                        (* m_pend sin_th (+ (* l (** om_ 2)) 
                                            (* g cos_th)))) 
                     denom)
            th_dot om_
            om_dot (/ (- (* -1 u_sym cos_th) 
                         (* m_pend l (** om_ 2) sin_th cos_th) 
                         (* (+ m_cart m_pend) g sin_th)) 
                      (* l denom)))
      
      (setf f_ode (ca.Function (string "f_ode") 
                               (list x_sym u_sym) 
                               (list (ca.vertcat s_dot v_dot th_dot om_dot))))

      ;; Runge-Kutta 4 Integrator für EINEN Zeitschritt
      (setf k1 (f_ode x_sym u_sym)
            k2 (f_ode (+ x_sym (* (/ dt 2) k1)) u_sym)
            k3 (f_ode (+ x_sym (* (/ dt 2) k2)) u_sym)
            k4 (f_ode (+ x_sym (* dt k3)) u_sym)
            x_next (+ x_sym (* (/ dt 6) (+ k1 (* 2 k2) (* 2 k3) k4))))
      
      (setf f_rk4 (ca.Function (string "f_rk4") (list x_sym u_sym) (list x_next)))

      ;; ==========================================
      ;; 3. MHE (MOVING HORIZON ESTIMATOR) SETUP
      ;; ==========================================
      (setf opti_mhe (ca.Opti))
      
      ;; Variablen für MHE: Zustände (X) und Prozessrauschen (W) über den Horizont
      ;; Dimensionen: 4 Zustände, N_mhe + 1 Zeitschritte
      (setf X_mhe (opti_mhe.variable 4 (+ N_mhe 1))
            W_mhe (opti_mhe.variable 4 N_mhe))
      
      ;; Parameter, die wir in jedem Schleifendurchlauf aus der GUI/Simulation injizieren
      (setf Y_meas_param (opti_mhe.parameter 4 (+ N_mhe 1)) ; Die verrauschten Sensordaten
            U_past_param (opti_mhe.parameter 1 N_mhe)       ; Die Motorkräfte der Vergangenheit
            X_prior_param (opti_mhe.parameter 4 1))         ; Die Ankunftskosten-Schätzung (Arrival Cost)
      
      ;; Gewichte (Kovarianz-Inversen) als Parameter, damit die GUI sie live ändern kann
      (setf Q_w (opti_mhe.parameter 4 1)  ; Vertrauen in das Systemmodell (Prozessrauschen)
            R_w (opti_mhe.parameter 4 1)  ; Vertrauen in die Sensoren (Messrauschen)
            P_w (opti_mhe.parameter 4 1)) ; Vertrauen in die Vorab-Schätzung (Prior)

      ;; ---- DER MAP-TRICK FÜR WINZIGEN C-CODE ----
      ;; Anstatt eine for-Schleife zu schreiben, mappen wir f_rk4 über N_mhe.
      ;; Das zwingt CasADi, den Graphen kompakt zu halten.
      (setf f_rk4_map (f_rk4.map N_mhe))
      
      ;; Slice arrays: Wir nehmen alle Zustände X_mhe bis auf den letzten, und alle U.
      ;; Die Map-Funktion generiert N_mhe Prädiktionen auf einmal.
      (setf X_prev_slice (aref X_mhe (slice nil nil) (slice 0 N_mhe)))
      (setf X_next_calc (f_rk4_map X_prev_slice U_past_param))
      
      ;; Constraints: X_next = berechnetes X + W (Prozessrauschen fungiert als Slack-Variable)
      (setf X_next_slice (aref X_mhe (slice nil nil) (slice 1 (+ N_mhe 1))))
      (opti_mhe.subject_to (== X_next_slice (+ X_next_calc W_mhe)))
      
      ;; Kostenfunktion (Objective) des MHE: Summe der quadratischen Fehler
      (setf cost_mhe 0)
      ;; 1. Arrival Cost: Wie sehr weichen wir von unserem Startglauben x_prior ab?
      (setf cost_mhe (+ cost_mhe (ca.sum1 (* P_w (** (- (aref X_mhe (slice nil nil) 0) X_prior_param) 2)))))
      ;; 2. Prozessrauschen: Strafkosten, wenn das System sich nicht physikalisch korrekt verhält (W != 0)
      (setf cost_mhe (+ cost_mhe (ca.sum2 (ca.sum1 (* Q_w (** W_mhe 2))))))
      ;; 3. Messrauschen: Strafkosten, wenn der Schätzer von den Sensordaten abweicht
      (setf cost_mhe (+ cost_mhe (ca.sum2 (ca.sum1 (* R_w (** (- Y_meas_param X_mhe) 2))))))
      
      (opti_mhe.minimize cost_mhe)
      (opti_mhe.solver (string "ipopt") (dictionary :print_time False) (dictionary :print_level 0))

      ;; ==========================================
      ;; 4. MPC (MODEL PREDICTIVE CONTROL) SETUP
      ;; ==========================================
      ;; Hier nutzt der Regler genau das geschätzte x_est des MHE als Startpunkt!
      (setf opti_mpc (ca.Opti))
      (setf X_mpc (opti_mpc.variable 4 (+ N_mpc 1))
            U_mpc (opti_mpc.variable 1 N_mpc))
      (setf X_current_param (opti_mpc.parameter 4 1)) ; Wird mit MHE-Output gefüttert
      
      ;; Map-Trick auch für MPC
      (setf f_rk4_map_mpc (f_rk4.map N_mpc))
      (setf X_next_calc_mpc (f_rk4_map_mpc (aref X_mpc (slice nil nil) (slice 0 N_mpc)) U_mpc))
      (opti_mpc.subject_to (== (aref X_mpc (slice nil nil) (slice 1 (+ N_mpc 1))) X_next_calc_mpc))
      
      ;; Harte Constraints im Regler (Beispiel: Schienenende und Maximalkraft)
      (opti_mpc.subject_to (== (aref X_mpc (slice nil nil) 0) X_current_param))
      (opti_mpc.subject_to (<= (aref U_mpc (slice nil nil) (slice nil nil)) 20.0))
      (opti_mpc.subject_to (>= (aref U_mpc (slice nil nil) (slice nil nil)) -20.0))
      
      ;; MPC Kostenfunktion
      (setf cost_mpc (+ (ca.sum2 (* 10 (** (aref X_mpc 0 (slice nil nil)) 2)))
                        (ca.sum2 (* 1 (** (aref X_mpc 1 (slice nil nil)) 2)))
                        (ca.sum2 (* 100 (** (aref X_mpc 2 (slice nil nil)) 2)))
                        (ca.sum2 (* 1 (** (aref X_mpc 3 (slice nil nil)) 2)))
                        (ca.sum2 (* 0.01 (** U_mpc 2)))))
      (opti_mpc.minimize cost_mpc)
      (opti_mpc.solver (string "ipopt")
		       (dictionary :print_time False)
		       (dictionary :print_level 0))

      ;; ==========================================
      ;; 5. GUI UND SIMULATIONS-SCHLEIFE
      ;; ==========================================
      (class MainWindow (QMainWindow)
             (def __init__ (self)
               (QMainWindow.__init__ self)
               (self.setWindowTitle (string "Inverted Pendulum - MHE & MPC"))
               
               ;; Hauptlayout
               (setf main_widget (QWidget)
                     layout (QVBoxLayout)
                     ui_layout (QHBoxLayout))
               (main_widget.setLayout layout)
               (self.setCentralWidget main_widget)
               
               ;; --- GUI Elemente: Delay Warnung ---
               (setf self.lbl_delay (QLabel (string "Rechenzeit: OK")))
               (self.lbl_delay.setStyleSheet (string "font-size: 16px; font-weight: bold; color: green;"))
               (layout.addWidget self.lbl_delay)

               ;; --- GUI Elemente: Sensor Checkboxen ---
               (setf sensor_group (QGroupBox (string "Sensoren Aktiv"))
                     sg_layout (QVBoxLayout))
               (setf self.cb_s (QCheckBox (string "Position (s)"))
                     self.cb_v (QCheckBox (string "Geschw. (v)"))
                     self.cb_th (QCheckBox (string "Winkel (th)"))
                     self.cb_om (QCheckBox (string "Winkelgeschw. (om)")))
               (self.cb_s.setChecked True)
               (self.cb_th.setChecked True)
               ;; Meist misst man v und omega nicht direkt, wir lassen sie wählbar
               (self.cb_v.setChecked False)
               (self.cb_om.setChecked False)
               (sg_layout.addWidget self.cb_s)
               (sg_layout.addWidget self.cb_v)
               (sg_layout.addWidget self.cb_th)
               (sg_layout.addWidget self.cb_om)
               (sensor_group.setLayout sg_layout)
               (ui_layout.addWidget sensor_group)

               ;; --- GUI Elemente: Rausch-Slider ---
               (setf noise_group (QGroupBox (string "Sensor Rauschen (Sigma)"))
                     ng_layout (QVBoxLayout))
               (setf self.sl_noise (QSlider Qt.Horizontal))
               (self.sl_noise.setRange 0 100)
               (self.sl_noise.setValue 10) ; Start mit leichtem Rauschen
               (ng_layout.addWidget self.sl_noise)
               (noise_group.setLayout ng_layout)
               (ui_layout.addWidget noise_group)
               
               (layout.addLayout ui_layout)
               
               ;; Plot Widget
               (setf self.plot_widget (PlotWidget)
                     self.curve_true (self.plot_widget.plot (list) (list) :pen (mkPen :color (string "g") :width 2 :name (string "True")))
                     self.curve_est (self.plot_widget.plot (list) (list) :pen (mkPen :color (string "r") :width 2 :style Qt.DashLine :name (string "MHE Est"))))
               (layout.addWidget self.plot_widget)

               ;; Initialisierung physikalischer Zustand (Wahre Welt)
               (setf self.x_true (np.array (list 0.0 0.0 0.1 0.0)) ; Start leicht gekippt
                     self.u_last 0.0)
               
               ;; History-Buffer für MHE (Listen der Länge N_mhe)
               (setf self.Y_hist (list)
                     self.U_hist (list))
               (setf self.x_prior_val (np.copy self.x_true))

               ;; Plot Daten
               (setf self.t_data (list)
                     self.th_true_data (list)
                     self.th_est_data (list)
                     self.t_curr 0.0)

               ;; Timer Setup
               (setf self.timer (QTimer))
               (self.timer.timeout.connect self.sim_step)
               (self.timer.start (int (* dt 1000))))

             (def sim_step (self)
               (setf t_start (time.time))

               ;; 1. Wahre Physik simulieren (Ein Schritt RK4 in Python für die reale Welt)
               (setf res_true (f_rk4 self.x_true self.u_last)
                     self.x_true (np.array (res_true.elements)))

               ;; 2. Rauschen hinzufügen basierend auf GUI (Sigma)
               (setf noise_std (/ (self.sl_noise.value) 100.0)
                     y_meas (np.copy self.x_true))
               ;; Fiktive Sensoren messen mit Gaussschem Rauschen
               (setf (aref y_meas 0) (+ (aref y_meas 0) (np.random.normal 0.0 noise_std))
                     (aref y_meas 1) (+ (aref y_meas 1) (np.random.normal 0.0 noise_std))
                     (aref y_meas 2) (+ (aref y_meas 2) (np.random.normal 0.0 noise_std))
                     (aref y_meas 3) (+ (aref y_meas 3) (np.random.normal 0.0 noise_std)))

               ;; 3. History Buffer updaten
               (self.Y_hist.append y_meas)
               (self.U_hist.append (list self.u_last))
               (if (> (len self.Y_hist) (+ N_mhe 1))
                   (do0
                    (self.Y_hist.pop 0)
                    (self.U_hist.pop 0)))

               ;; 4. State Estimation via MHE
               (setf x_est (np.copy y_meas)) ; Fallback
               (if (== (len self.Y_hist) (+ N_mhe 1))
                   (do0
                    ;; Arrays formen für CasADi
                    (setf Y_mat (dot (np.array self.Y_hist) "T")
                          U_mat (dot (np.array (aref self.U_hist (slice 0 N_mhe))) "T"))
                    
                    (opti_mhe.set_value Y_meas_param Y_mat)
                    (opti_mhe.set_value U_past_param U_mat)
                    (opti_mhe.set_value X_prior_param self.x_prior_val)
                    
                    ;; Q und P Gewichte
                    (opti_mhe.set_value Q_w (np.array (list 1.0 1.0 1.0 1.0)))
                    (opti_mhe.set_value P_w (np.array (list 10.0 10.0 10.0 10.0)))
                    
                    ;; R Gewicht dynamisch anhand Checkboxen (Wenn aus: R=0, MHE ignoriert den Sensor!)
                    (setf r_weights (list (? (self.cb_s.isChecked) 100.0 0.0)
                                          (? (self.cb_v.isChecked) 100.0 0.0)
                                          (? (self.cb_th.isChecked) 1000.0 0.0)
                                          (? (self.cb_om.isChecked) 100.0 0.0)))
                    (opti_mhe.set_value R_w (np.array r_weights))
                    
                    ;; MHE Lösen
                    (try
                     (do0
                      (setf sol_mhe (opti_mhe.solve))
                      (setf X_res (sol_mhe.value X_mhe))
                      (setf x_est (aref X_res (slice nil nil) -1)) ; Aktueller Zustand
                      ;; Update Prior für den nächsten Schritt (Moving the horizon)
                      (setf self.x_prior_val (aref X_res (slice nil nil) 1)))
                     ((as Exception e)
                      (print (string "MHE Failed!"))))))

               ;; 5. Regelung via MPC (Gefüttert ausschließlich mit x_est!)
               (opti_mpc.set_value X_current_param x_est)
               (try
                (do0
                 (setf sol_mpc (opti_mpc.solve))
                 (setf self.u_last (aref (sol_mpc.value U_mpc) 0)))
                ((as Exception e)
                 (print (string "MPC Failed!"))))

               ;; 6. Delay Warnung berechnen
               (setf t_end (time.time)
                     t_calc (- t_end t_start))
               (if (> t_calc dt)
                   (do0
                    (self.lbl_delay.setText (fstring "GEFAHR: Rechenzeit {(t_calc*1000):.1f}ms übersteigt Sim-Takt {(dt*1000)}ms!"))
                    (self.lbl_delay.setStyleSheet (string "font-size: 16px; font-weight: bold; color: red;")))
                   (do0
                    (self.lbl_delay.setText (fstring "Rechenzeit: {(t_calc*1000):.1f}ms (OK)"))
                    (self.lbl_delay.setStyleSheet (string "font-size: 16px; font-weight: bold; color: green;"))))

               ;; 7. Plot Updates
               (setf self.t_curr (+ self.t_curr dt))
               (self.t_data.append self.t_curr)
               (self.th_true_data.append (aref self.x_true 2))
               (self.th_est_data.append (aref x_est 2))
               (if (> (len self.t_data) 100)
                   (do0
                    (self.t_data.pop 0)
                    (self.th_true_data.pop 0)
                    (self.th_est_data.pop 0)))
               
               (self.curve_true.setData self.t_data self.th_true_data)
               (self.curve_est.setData self.t_data self.th_est_data)))

      ;; Main Execution
      (setf app (QApplication sys.argv)
            win (MainWindow))
      (win.show)
      (sys.exit (app.exec))))
  
  (write-source *code-file* *source*))
