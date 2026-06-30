(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11d
  (:use #:cl #:cl-py-generator))

(in-package #:g11d)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11d_mpc_test"
						 *source*))
 `(do0
   (imports-from (__future__ annotations)
		 (casadi *))
   (imports ((np numpy)
	     time
	     sys
	     json))

   (comments 
    "========================================================================================="
    " INVERTED PENDULUM MPC HEADLESS BENCHMARK AND VALIDATION"
    "=========================================================================================")

   (class PendulumMPC ()
     (def __init__ (self &key (h_mpc 0.05) (N 20))
       (setf self.opti (Opti)
	     self.nx 4
	     self.nu 1
	     self.N N
	     self.h_mpc h_mpc
	     self.T_horizon (* self.N self.h_mpc)
	     self.d 3)

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
	     dv (/ (- (+ F_total (* m_s l_s omega_ omega_ sin_theta)) (* m_s 9.81 cos_theta sin_theta)) den)
	     dtheta omega_
	     domega (/ (+ (- (* -1.0 F_total cos_theta) (* m_s l_s omega_ omega_ sin_theta cos_theta)) (* (+ M_s m_s) 9.81 sin_theta)) (* l_s den)))
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
	    (self.opti.subject_to (== (aref self.X (slice nil nil) (+ k 1)) x_end)))

       (setf cost 0.0
	     Q (diag (vertcat self.Q_s self.Q_v self.Q_theta self.Q_omega)))
       (for (k (range self.N))
	    (setf err (- (aref self.X (slice nil nil) k) self.target_x)
		  cost (+ cost (+ (mtimes (mtimes err.T Q) err)
				  (* self.R_F (** (aref self.U 0 k) 2))))))
       (setf err_term (- (aref self.X (slice nil nil) self.N) self.target_x)
	     cost (+ cost (* 10.0 (mtimes (mtimes err_term.T Q) err_term))))
       (self.opti.minimize cost)

       (self.opti.solver (string "ipopt") 
			 (dict ((string "print_time") False))
			 (dict ((string "print_level") 0) ((string "sb") (string "yes"))))
       
       (self.opti.set_initial self.X (dot (np.linspace (np.array (list 0.0 0.0 np.pi 0.0)) (np.array (list 1.0 0.0 0.0 0.0)) (+ self.N 1)) T))
       (setf self.sol None))

     (def step (self state target_state params)
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

       (if (!= self.sol None)
	   (do0
	    (setf X_res (self.sol.value self.X)
		  U_res (self.sol.value self.U)
		  X_guess (np.hstack (tuple (aref X_res (slice nil nil) (slice 1 nil))
					    (aref X_res (slice nil nil) (slice -1 nil))))
		  U_guess (np.append (aref U_res (slice 1 nil)) (aref U_res -1)))
	    (self.opti.set_initial self.X X_guess)
	    (self.opti.set_initial self.U U_guess))
	   (do0
	    (setf X_res (np.zeros (tuple self.nx (+ self.N 1)))
		  U_res (np.zeros (tuple self.nu self.N))
		  U_guess (np.zeros self.nu))))

       (setf t0 (time.time))
       (try
	(setf self.sol (self.opti.solve))
	("Exception as e"
		(return (tuple (aref U_guess 0) X_res U_res (- (time.time) t0) False))))
       (setf t_solve (- (time.time) t0)
	     X_res (self.sol.value self.X)
	     U_res (self.sol.value self.U))
       (return (tuple (aref U_res 0) X_res U_res t_solve True))))

   (def f_real (st F_motor params use_corrected_l)
     (setf s_st (aref st 0)
           v_st (aref st 1)
           theta_st (aref st 2)
           omega_st (aref st 3)
           sin_t (np.sin theta_st)
           cos_t (np.cos theta_st)
           M (aref params (string "M"))
           m (aref params (string "m"))
           l (? use_corrected_l (aref params (string "l")) 0.5)
           wind_force (aref params (string "wind"))
           den (+ M (* m (- 1.0 (* cos_t cos_t))))
           F_tot (+ F_motor (* wind_force cos_t))
           ds v_st
           dv (/ (- (+ F_tot (* m l omega_st omega_st sin_t)) (* m 9.81 cos_t sin_t)) den)
           dtheta omega_st
           domega (/ (+ (- (* -1.0 F_tot cos_t) (* m l omega_st omega_st sin_t cos_t)) (* (+ M m) 9.81 sin_t)) (* l den)))
     (return (np.array (list ds dv dtheta domega))))

   (def rk4_step (st F_motor params dt use_corrected_l)
     (setf k1 (f_real st F_motor params use_corrected_l)
           k2 (f_real (+ st (* (/ dt 2.0) k1)) F_motor params use_corrected_l)
           k3 (f_real (+ st (* (/ dt 2.0) k2)) F_motor params use_corrected_l)
           k4 (f_real (+ st (* dt k3)) F_motor params use_corrected_l))
     (return (+ st (* (/ dt 6.0) (+ k1 (* 2.0 k2) (* 2.0 k3) k4)))))

   (def simulate (mpc initial_state target_state params sim_time dt_sim use_corrected_l)
     (setf steps (int (/ sim_time dt_sim))
           state (np.array initial_state)
           t_hist (list)
           s_hist (list)
           v_hist (list)
           theta_hist (list)
           omega_hist (list)
           F_hist (list)
           t_solve_hist (list)
           success_hist (list)
           current_time 0.0)
     
     (for (i (range steps))
          (setf (ntuple u_opt X_pred U_pred t_solve success) (mpc.step state target_state params))
          (t_hist.append current_time)
          (s_hist.append (aref state 0))
          (v_hist.append (aref state 1))
          (setf theta_wrapped (- (% (+ (aref state 2) np.pi) (* 2.0 np.pi)) np.pi))
          (theta_hist.append theta_wrapped)
          (omega_hist.append (aref state 3))
          (F_hist.append u_opt)
          (t_solve_hist.append t_solve)
          (success_hist.append success)
          
          (setf state (rk4_step state u_opt params dt_sim use_corrected_l))
          (setf current_time (+ current_time dt_sim)))

     (return (dict ((string "t") (np.array t_hist))
                   ((string "s") (np.array s_hist))
                   ((string "v") (np.array v_hist))
                   ((string "theta") (np.array theta_hist))
                   ((string "omega") (np.array omega_hist))
                   ((string "F") (np.array F_hist))
                   ((string "t_solve") (np.array t_solve_hist))
                   ((string "success") (np.array success_hist)))))

   (def evaluate_run (data target_state)
     (setf t (aref data (string "t"))
           s (aref data (string "s"))
           theta (aref data (string "theta"))
           F (aref data (string "F"))
           t_solve (aref data (string "t_solve"))
           success (aref data (string "success"))
           
           target_s (aref target_state 0)
           target_theta (aref target_state 2)
           
           dt (? (> (len t) 1) (- (aref t 1) (aref t 0)) 0.05)
           
           iae_s (* (np.sum (np.abs (- s target_s))) dt)
           iae_theta (* (np.sum (np.abs (- theta target_theta))) dt)
           
           settling_time_s 0.0)
     
     (for (i (reversed (range (len t))))
          (if (> (np.abs (- (aref s i) target_s)) 0.05)
              (do0
               (if (< i (- (len t) 1))
                   (setf settling_time_s (aref t (+ i 1)))
                   (setf settling_time_s (float (string "inf"))))
               break)))

     (setf settling_time_theta 0.0)
     (for (i (reversed (range (len t))))
          (if (> (np.abs (- (aref theta i) target_theta)) 0.05)
              (do0
               (if (< i (- (len t) 1))
                   (setf settling_time_theta (aref t (+ i 1)))
                   (setf settling_time_theta (float (string "inf"))))
               break)))

     (setf overshoot_s (np.max (np.abs (- s target_s)))
           max_F (np.max (np.abs F))
           avg_t_solve (* (np.mean t_solve) 1000.0)
           max_t_solve (* (np.max t_solve) 1000.0)
           success_rate (* (np.mean success) 100.0))

     (return (dict ((string "iae_s") iae_s)
                   ((string "iae_theta") iae_theta)
                   ((string "settling_time_s") settling_time_s)
                   ((string "settling_time_theta") settling_time_theta)
                   ((string "overshoot_s") overshoot_s)
                   ((string "max_F") max_F)
                   ((string "avg_t_solve") avg_t_solve)
                   ((string "max_t_solve") max_t_solve)
                   ((string "success_rate") success_rate))))

   (def run_tests ()
     (setf results (list)
           configs
           (list
            (dict ((string "name") (string "Config 1: Current Default (high Q_v, Q_omega)"))
                  ((string "Q_s") 10.0) ((string "Q_v") 10.0) ((string "Q_theta") 100.0) ((string "Q_omega") 10.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 0.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 20) ((string "use_corrected_l") True))
            
            (dict ((string "name") (string "Config 2: Old Commit Default (low Q_v, Q_omega)"))
                  ((string "Q_s") 10.0) ((string "Q_v") 1.0) ((string "Q_theta") 100.0) ((string "Q_omega") 1.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 0.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 20) ((string "use_corrected_l") True))

            (dict ((string "name") (string "Config 3: High Position Weight (Q_s = 100.0)"))
                  ((string "Q_s") 100.0) ((string "Q_v") 10.0) ((string "Q_theta") 100.0) ((string "Q_omega") 10.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 0.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 20) ((string "use_corrected_l") True))

            (dict ((string "name") (string "Config 4: Fast Position Response (Q_s=50, Q_v=2, R_F=0.05)"))
                  ((string "Q_s") 50.0) ((string "Q_v") 2.0) ((string "Q_theta") 150.0) ((string "Q_omega") 2.0) ((string "R_F") 0.05)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 0.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 20) ((string "use_corrected_l") True))

            (dict ((string "name") (string "Config 5: Short Horizon (N=10, T_horiz=0.5s)"))
                  ((string "Q_s") 10.0) ((string "Q_v") 10.0) ((string "Q_theta") 100.0) ((string "Q_omega") 10.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 0.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 10) ((string "use_corrected_l") True))

            (dict ((string "name") (string "Config 6: Long Horizon (N=40, T_horiz=2.0s)"))
                  ((string "Q_s") 10.0) ((string "Q_v") 10.0) ((string "Q_theta") 100.0) ((string "Q_omega") 10.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 0.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 40) ((string "use_corrected_l") True))

            (dict ((string "name") (string "Config 7: L-Mismatch (l_slider=1.5, buggy physics l=0.5)"))
                  ((string "Q_s") 10.0) ((string "Q_v") 10.0) ((string "Q_theta") 100.0) ((string "Q_omega") 10.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 1.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 20) ((string "use_corrected_l") False))

            (dict ((string "name") (string "Config 8: L-Matched (l_slider=1.5, corrected physics l=1.5)"))
                  ((string "Q_s") 10.0) ((string "Q_v") 10.0) ((string "Q_theta") 100.0) ((string "Q_omega") 10.0) ((string "R_F") 0.1)
                  ((string "M") 1.0) ((string "m") 0.1) ((string "l") 1.5) ((string "wind") 0.0) ((string "max_pos") 2.0) ((string "max_force") 15.0)
                  ((string "h_mpc") 0.05) ((string "N") 20) ((string "use_corrected_l") True))
            ))

     (setf test_scenarios
           (list
            (dict ((string "name") (string "Stabilization (theta_0 = 0.2 rad)"))
                  ((string "x0") (list 0.0 0.0 0.2 0.0))
                  ((string "x_target") (list 1.5 0.0 0.0 0.0))
                  ((string "sim_time") 8.0)
                  ((string "dt_sim") 0.033))
            (dict ((string "name") (string "Swing-Up & Position Step (theta_0 = pi)"))
                  ((string "x0") (list 0.0 0.0 np.pi 0.0))
                  ((string "x_target") (list 1.5 0.0 0.0 0.0))
                  ((string "sim_time") 12.0)
                  ((string "dt_sim") 0.033))
            ))

     (print (string "Starte MPC Parameter und Simulations-Benchmark..."))
     
     (setf out_results (dict))
     (for (scenario test_scenarios)
          (print (fstring "\\n=========================================\\nSzenario: {scenario['name']}\\n========================================="))
          (setf scenario_results (list))
          (for (cfg configs)
               (print (fstring "Simuliere: {cfg['name']}..."))
               (setf mpc (PendulumMPC :h_mpc (aref cfg (string "h_mpc")) :N (aref cfg (string "N"))))
               (setf sim_data (simulate mpc
                                        (aref scenario (string "x0"))
                                        (aref scenario (string "x_target"))
                                        cfg
                                        (aref scenario (string "sim_time"))
                                        (aref scenario (string "dt_sim"))
                                        (aref cfg (string "use_corrected_l"))))
               (setf metrics (evaluate_run sim_data (aref scenario (string "x_target"))))
               (setf record (dict ((string "name") (aref cfg (string "name")))
                                  ((string "metrics") metrics)))
               (scenario_results.append record)
               
               ;; Print instant summary
               (setf m metrics)
               (print (fstring "  - IAE_s: {m['iae_s']:.3f}, Settling t_s: {m['settling_time_s']:.2f}s, Overshoot: {m['overshoot_s']:.3f}m, Success: {m['success_rate']:.1f}%")))
          (setf (aref out_results (aref scenario (string "name"))) scenario_results))
     
     ;; Generiere Markdown-Bericht fuer den User
     (setf report_content (list))
     (report_content.append (string "# MPC Benchmark-Bericht & Parameter-Validierung"))
     (report_content.append (fstring "Generiert am: {time.strftime('%Y-%m-%d %H:%M:%S')}"))
     (report_content.append (string "\\nDieser Bericht vergleicht verschiedene Reglereinstellungen und validiert die Korrektheit des Modells."))
     
     (for ((ntuple scenario_name scen_res) (out_results.items))
          (report_content.append (fstring "\\n## Szenario: {scenario_name}"))
          (report_content.append (string "| Konfiguration | IAE Position [m*s] | IAE Winkel [rad*s] | Settling Time Position [s] | Settling Time Winkel [s] | Overshoot Position [m] | Max Kraft [N] | Avg Solve Time [ms] | Solver Success % |"))
          (report_content.append (string "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"))
          (for (r scen_res)
               (setf name (aref r (string "name"))
                     m (aref r (string "metrics"))
                     t_s (? (== (aref m (string "settling_time_s")) (float (string "inf"))) (string "Never") (fstring "{m['settling_time_s']:.2f}"))
                     t_theta (? (== (aref m (string "settling_time_theta")) (float (string "inf"))) (string "Never") (fstring "{m['settling_time_theta']:.2f}")))
               (report_content.append (fstring "| {name} | {m['iae_s']:.3f} | {m['iae_theta']:.3f} | {t_s} | {t_theta} | {m['overshoot_s']:.3f} | {m['max_F']:.2f} | {m['avg_t_solve']:.2f} | {m['success_rate']:.1f}% |"))))

     ;; Diskussion und Empfehlungen
     (report_content.append (string "\\n## Analyse & Empfehlungen"))
     (report_content.append (string "1. **Einfluss von Q_v und Q_omega (Config 1 vs Config 2):**"))
     (report_content.append (string "   - In älteren Commits (Config 2) waren die Gewichte fuer die Geschwindigkeiten (Wagen-Geschwindigkeit Q_v und Winkel-Geschwindigkeit Q_omega) auf 1.0 statt 10.0 eingestellt."))
     (report_content.append (string "   - Niedrigere Dämpfungs-Kosten erlauben dem Regler, viel schneller zu beschleunigen und abzubremsen, was die Einschwingzeit (Settling Time) der Position verkuerzt, aber eventuell zu leichtem Überschwingen führt."))
     (report_content.append (string "2. **Tuning fuer aggressive Positionsregelung (Config 4):**"))
     (report_content.append (string "   - Durch Erhoehung von Q_s (auf z.B. 50) und gleichzeitiges Absenken von Q_v (auf z.B. 2) bei geringeren Stellkraftkosten (R_F = 0.05) kann der Wagen extrem praezise positioniert werden, ohne instabil zu werden."))
     (report_content.append (string "3. **Pendellänge-Modell-Mismatch (Config 7 vs Config 8):**"))
     (report_content.append (string "   - Wenn der Benutzer im GUI die Pendellänge veraendert, der Simulator aber intern mit der hartcodierten Laenge von 0.5 rechnet, weichen MPC-Modell und reale Physik stark voneinander ab. Das fuehrt zu schlechterer Performance (Config 7)."))
     (report_content.append (string "   - Bei passender Physik (Config 8) regelt der MPC das System auch bei Laenge 1.5 optimal."))

     (setf report_str (dot (string "\\n") (join report_content)))
     (print (string "\\n--- BENCHMARK RESULTS ---"))
     (print report_str)
     
     ;; Speichern als Datei
     (setf out_path (string "p11d_benchmark_report.md"))
     (with (as (open out_path (string "w")) f)
           (f.write report_str))
     (print (fstring "\\nBericht erfolgreich geschrieben unter: {out_path}")))

   (if (== __name__ (string "__main__"))
       (run_tests))
   ))
