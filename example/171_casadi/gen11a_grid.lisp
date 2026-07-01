(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11a-grid
  (:use #:cl #:cl-py-generator))

(in-package #:g11a-grid)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11a_grid"
						 *source*))
 `(do0
   (imports-from (__future__ annotations)
		 (casadi *))
   (imports ((np numpy)
	     time
	     sys
	     (plt matplotlib.pyplot)))

   (comments 
    "========================================================================================="
    " INVERTED PENDULUM MPC GRID SEARCH (HEADLESS - JIT-free)"
    "=========================================================================================")

   (class PendulumMPC ()
     (def __init__ (self &key (h_mpc 0.05) (N 20) (use_dual_warmstart True) (use_map True))
       (setf self.opti (Opti)
	     self.nx 4
	     self.nu 1
	     self.N N
	     self.h_mpc h_mpc
	     self.T_horizon (* self.N self.h_mpc)
	     self.d 3
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
             ipopt_opts (dict ((string "print_level") 0)
                              ((string "sb") (string "yes"))
                              ((string "max_iter") 150)))
       (self.opti.solver (string "ipopt") solver_opts ipopt_opts)

       (self.opti.set_initial self.X (dot (np.linspace (np.array (list 0.0 0.0 np.pi 0.0)) (np.array (list 1.0 0.0 0.0 0.0)) (+ self.N 1)) T))
       (self.opti.set_initial self.U (np.zeros (tuple self.nu self.N)))
       (if self.use_map
           (self.opti.set_initial self.Xc_var (np.zeros (tuple (* self.nx self.d) self.N))))

       (setf self.sol None
             self.last_X (np.zeros (tuple self.nx (+ self.N 1)))
             self.last_U (np.zeros (tuple self.nu self.N))
             self.last_lam_g None))

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
       (return (tuple (float u_opt) self.last_X self.last_U t_solve success))))

   (def run_simulation (h_mpc l)
     (setf params (dict ((string "M") 1.0)
                        ((string "m") 0.1)
                        ((string "l") l)
                        ((string "wind") 0.0)
                        ((string "Q_s") 10.0)
                        ((string "Q_v") 1.0)
                        ((string "Q_theta") 100.0)
                        ((string "Q_omega") 1.0)
                        ((string "R_F") 0.1)
                        ((string "max_pos") 5.0)
                        ((string "max_force") 15.0)
                        ((string "h_mpc") h_mpc)
                        ((string "N") 20)
                        ((string "dt_sim") h_mpc)))
     
     (setf mpc (PendulumMPC :h_mpc h_mpc :N 20 :use_map True))
     (setf state (np.array (list 0.0 0.0 np.pi 0.0))
           target_state (np.array (list 1.0 0.0 0.0 0.0))
           T_max 12.0
           dt h_mpc
           n_steps (int (/ T_max dt))
           s_hist (list)
           v_hist (list)
           theta_hist (list)
           omega_hist (list)
           t_hist (list))
     
     (for (step_idx (range n_steps))
          (setf (ntuple u_opt X_pred U_pred t_solve success) (mpc.step state target_state params))
          
          (def f_real (st)
            (setf s_st (aref st 0) v_st (aref st 1) theta_st (aref st 2) omega_st (aref st 3)
                  sin_t (np.sin theta_st) cos_t (np.cos theta_st)
                  denom (+ (aref params (string "M")) (* (aref params (string "m")) (- 1.0 (* cos_t cos_t))))
                  F_tot u_opt
                  l_val (aref params (string "l"))
                  ds v_st
                  dv (/ (- (+ F_tot (* (aref params (string "m")) l_val omega_st omega_st sin_t)) (* (aref params (string "m")) 9.81 cos_t sin_t)) denom)
                  dtheta omega_st
                  domega (/ (+ (- (* -1.0 F_tot cos_t) (* (aref params (string "m")) l_val omega_st omega_st sin_t cos_t)) (* (+ (aref params (string "M")) (aref params (string "m"))) 9.81 sin_t)) (* l_val denom)))
            (return (np.array (list ds dv dtheta domega))))
          
          (setf k1 (f_real state)
                k2 (f_real (+ state (* (/ dt 2.0) k1)))
                k3 (f_real (+ state (* (/ dt 2.0) k2)))
                k4 (f_real (+ state (* dt k3)))
                state (+ state (* (/ dt 6.0) (+ k1 (* 2.0 k2) (* 2.0 k3) k4))))
          
          (s_hist.append (aref state 0))
          (v_hist.append (aref state 1))
          (theta_hist.append (aref state 2))
          (omega_hist.append (aref state 3))
          (t_hist.append (* (+ step_idx 1) dt)))
     
     (if (< (len s_hist) n_steps)
         (return np.nan))
     
     (setf s_tol 0.05
           v_tol 0.05
           theta_tol 0.05
           omega_tol 0.05
           stabilization_step -1)
     
     (for (i (range (- (len s_hist) 1) -1 -1))
          (setf s_val (aref s_hist i)
                v_val (aref v_hist i)
                theta_val (aref theta_hist i)
                omega_val (aref omega_hist i)
                theta_wrapped (- (% (+ theta_val np.pi) (* 2.0 np.pi)) np.pi)
                outside (or (> (np.abs (- s_val 1.0)) s_tol)
                            (> (np.abs v_val) v_tol)
                            (> (np.abs theta_wrapped) theta_tol)
                            (> (np.abs omega_val) omega_tol)))
          (if outside
              (do0
                (setf stabilization_step (+ i 1))
                break)))
     
     (if (== stabilization_step -1)
         (return 0.0)
         (if (>= stabilization_step (len s_hist))
             (return np.nan)
             (return (aref t_hist stabilization_step)))))

   (if (== __name__ (string "__main__"))
       (do0
         (print (string "Starting Inverted Pendulum MPC Grid Search..."))
         (setf h_mpc_vals (list 0.02 0.03 0.04 0.05 0.06 0.08 0.10)
               l_vals (list 0.2 0.4 0.6 0.8 1.0 1.3 1.6 2.0))
         
         (setf grid_results (np.zeros (tuple (len h_mpc_vals) (len l_vals))))
         
         (for ((ntuple i h_val) (enumerate h_mpc_vals))
              (for ((ntuple j l_val) (enumerate l_vals))
                   (print (fstring "Simulating h_mpc={h_val:.2f}s, l={l_val:.2f}m..."))
                   (setf t_stable (run_simulation h_val l_val)
                         (aref grid_results i j) t_stable)
                   (print (fstring "  Stabilization time: {t_stable:.3f}s"))))
         
         (print (string "\\nGrid Search Results (Stabilization Time in seconds):"))
         (setf header (string "h_mpc \\ l   "))
         (for (l_val l_vals)
              (setf header (+ header (fstring "{l_val:8.2f}"))))
         (print header)
         (print (* (string "-") (len header)))
         
         (for ((ntuple i h_val) (enumerate h_mpc_vals))
              (setf line (fstring "{h_val:10.3f} |"))
              (for ((ntuple j l_val) (enumerate l_vals))
                   (setf val (aref grid_results i j))
                   (if (np.isnan val)
                       (setf line (+ line (string "     nan")))
                       (setf line (+ line (fstring "{val:8.2f}")))))
              (print line))
         
         (comments "Plot the heatmap")
         (setf (ntuple fig ax) (plt.subplots 1 1 :figsize (tuple 8 6)))
         (setf cax (ax.imshow grid_results :interpolation (string "nearest") :cmap (string "viridis") :origin (string "lower")))
         (fig.colorbar cax :label (string "Stabilisierungszeit [s]"))
         
         (ax.set_xticks (np.arange (len l_vals)))
         (ax.set_xticklabels (list (for-generator (l l_vals) (fstring "{l:.2f}"))))
         (ax.set_yticks (np.arange (len h_mpc_vals)))
         (ax.set_yticklabels (list (for-generator (h h_mpc_vals) (fstring "{h:.2f}"))))
         
         (ax.set_xlabel (string "Pendellaenge l [m]"))
         (ax.set_ylabel (string "Schrittweite h_mpc [s]"))
         (ax.set_title (string "MPC Grid Search: Stabilisierungszeit des Pendels"))
         
         (comments "Annotate text on the heatmap cells")
         (for (i (range (len h_mpc_vals)))
              (for (j (range (len l_vals)))
                   (setf val (aref grid_results i j)
                         text_val (? (np.isnan val) (string "NaN") (fstring "{val:.2f}")))
                   (ax.text j i text_val :ha (string "center") :va (string "center") :color (string "w"))))
         
         (plt.tight_layout)
         (plt.savefig (string "p11a_grid_heatmap.png"))
         (print (string "Heatmap saved as p11a_grid_heatmap.png"))
         ))
   ))
