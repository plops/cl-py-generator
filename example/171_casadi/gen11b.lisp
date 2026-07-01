(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11b
  (:use #:cl #:cl-py-generator))

(in-package #:g11b)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11b_mpc_benchmark"
						 *source*))
 `(do0
   (imports-from (__future__ annotations)
		 (casadi *))
   (imports ((np numpy)
	     (time time)))

   (def setup_mpc (use_jit)
     (setf opti (Opti))
     (setf M 1.0 m 0.1 l 0.5 g 9.81 nx 4 nu 1)
     
     (setf x (SX.sym (string "x") nx)
	   u (SX.sym (string "u") nu))
     (setf s_ (aref x 0) v_ (aref x 1) theta_ (aref x 2) omega_ (aref x 3) F_ u)
     
     (setf sin_theta (np.sin theta_) cos_theta (np.cos theta_)
	   den (+ M (* m (- 1.0 (* cos_theta cos_theta)))))
     (setf ds v_
	   dv (/ (- (+ F_ (* m l omega_ omega_ sin_theta)) (* m g cos_theta sin_theta)) den)
	   dtheta omega_
	   domega (/ (+ (- (* -1.0 F_ cos_theta) (* m l omega_ omega_ sin_theta cos_theta)) (* (+ M m) g sin_theta)) (* l den)))
     (setf f_ode (Function (string "f_ode") (list x u) (list (vertcat ds dv dtheta domega))))

     ;; Kuerzerer MPC Horizont
     (setf N 20 T 1.0 h (/ T N) d 3)
     (setf tau_root (np.append 0.0 (collocation_points d (string "radau"))))
     (setf C (np.zeros (tuple (+ d 1) (+ d 1)))
	   D (np.zeros (+ d 1)))
     (for (j (range (+ d 1)))
	  (setf p (np.poly1d (list 1.0)))
	  (for (r (range (+ d 1)))
	       (if (!= r j)
		   (setf p (* p (/ (np.poly1d (list 1.0 (- (aref tau_root r))))
				   (- (aref tau_root j) (aref tau_root r)))))))
	  (setf (aref D j) (p 1.0))
	  (setf pder (np.polyder p))
	  (for (r (range (+ d 1)))
	       (setf (aref C j r) (pder (aref tau_root r)))))

     (setf X (opti.variable nx (+ N 1)))
     (setf Xc (list))
     (for (k (range N))
	  (setf Xc_k (list))
	  (for (r (range d))
	       (dot Xc_k (append (opti.variable nx))))
	  (dot Xc (append Xc_k)))
     (setf U (opti.variable nu N))

     ;; Parameter fuer den MPC-Loop (koennen veraendert werden ohne Rekompilierung)
     (setf current_x (opti.parameter nx))
     (setf target_x (opti.parameter nx))
     (opti.set_value current_x (np.array (list 0.0 0.0 np.pi 0.0)))
     (opti.set_value target_x (np.array (list 1.0 0.0 0.0 0.0)))

     (opti.subject_to (== (aref X (slice nil nil) 0) current_x))
     
     (opti.subject_to (opti.bounded -2.0 (aref X 0 (slice nil nil)) 2.0))
     (opti.subject_to (opti.bounded -15.0 U 15.0))

     (for (k (range N))
	  (setf Xk (aref X (slice nil nil) k))
	  (setf x_end (* (aref D 0) Xk))
	  (for (j (range 1 (+ d 1)))
	       (setf xp (* (aref C 0 j) Xk))
	       (for (r (range d))
		    (setf xp (+ xp (* (aref C (+ r 1) j) (aref (aref Xc k) r)))))
	       (setf f_eval (f_ode (aref (aref Xc k) (- j 1)) (aref U (slice nil nil) k)))
	       (opti.subject_to (== xp (* h f_eval))))
	  (for (r (range d))
	       (setf x_end (+ x_end (* (aref D (+ r 1)) (aref (aref Xc k) r)))))
	  (opti.subject_to (== (aref X (slice nil nil) (+ k 1)) x_end)))

     ;; Kostenfunktion: quadratische Abweichung + Stellkosten
     (setf cost 0.0)
     (setf Q (np.diag (list 10.0 1.0 10.0 1.0)))
     (setf R 0.01)
     (for (k (range N))
	  (setf err (- (aref X (slice nil nil) k) target_x))
	  (setf cost (+ cost (+ (mtimes (mtimes err.T Q) err)
				(* R (** (aref U 0 k) 2))))))
     ;; Terminal Cost
     (setf err_term (- (aref X (slice nil nil) N) target_x))
     (setf cost (+ cost (* 10.0 (mtimes (mtimes err_term.T Q) err_term))))
     (opti.minimize cost)

     (setf solver_opts (dict ((string "print_time") False)
			     ((string "ipopt") (dict ((string "print_level") 0)))))
     (if use_jit
	 (do0
	  (print (string "JIT aktiviert: Kompiliere NLP zu C-Code... (das dauert einen Moment)"))
	  (setf (aref solver_opts (string "jit")) True)
	  (setf (aref solver_opts (string "compiler")) (string "shell"))
	  (setf (aref solver_opts (string "jit_options")) (dict ((string "flags") (list (string "-O3")))))))

     (opti.solver (string "ipopt") solver_opts)
     (return (tuple opti X U current_x target_x f_ode)))

   (def run_benchmark (opti X U current_x target_x f_ode name)
     (print (string ""))
     (print (+ (string "--- Benchmark: ") name (string " ---")))
     
     (setf x0 (np.array (list 0.0 0.0 np.pi 0.0)))
     (opti.set_value current_x x0)
     (opti.set_initial X (dot (np.linspace x0 (np.array (list 1.0 0.0 0.0 0.0)) (+ 20 1)) T))
     
     (setf t0 (time.time))
     (setf sol (opti.solve))
     (setf t_compile_and_cold (- (time.time) t0))
     
     (comments "Zuruecksetzen auf Kaltstart-Guess, um den reinen Solve ohne GCC Kompilierungszeit zu messen")
     (opti.set_value current_x x0)
     (opti.set_initial X (dot (np.linspace x0 (np.array (list 1.0 0.0 0.0 0.0)) (+ 20 1)) T))
     (opti.set_initial U (np.zeros (tuple 1 20)))
     
     (setf t0 (time.time))
     (setf sol (opti.solve))
     (setf t_cold (- (time.time) t0))
     
     (print (fstring "JIT + Cold Start Time: {t_compile_and_cold*1000:.1f} ms"))
     (print (fstring "Pure Cold Start Time : {t_cold*1000:.1f} ms"))
     
     ;; MPC Loop
     (setf n_steps 50)
     (setf times (list))
     (setf state x0)
     
     (for (i (range n_steps))
	  ;; Warm Start (Shift initialization)
	  (setf X_res (sol.value X))
	  (setf U_res (sol.value U))
	  
	  (setf X_guess (np.hstack (tuple (aref X_res (slice nil nil) (slice 1 nil))
					  (aref (aref X_res (slice nil nil) (slice -1 nil)) (slice nil nil) (slice np.newaxis nil)))))
          ;; Wait, np.hstack needs proper 2D shapes. X_res[:, 1:] is (4, 19). X_res[:, -1:] is (4, 1). So hstack is (4, 20).
	  (setf X_guess (np.hstack (tuple (aref X_res (slice nil nil) (slice 1 nil))
					  (aref X_res (slice nil nil) (slice -1 nil)))))
	  (setf U_guess (np.append (aref U_res (slice 1 nil)) (aref U_res -1)))
	  
	  (opti.set_initial X X_guess)
	  (opti.set_initial U U_guess)
	  
	  ;; RK4 Simulator Schritt (dt = 0.05)
	  (setf u_applied (aref U_res 0))
	  (setf k1 (f_ode state u_applied))
	  (setf k2 (f_ode (+ state (* 0.025 k1)) u_applied))
	  (setf k3 (f_ode (+ state (* 0.025 k2)) u_applied))
	  (setf k4 (f_ode (+ state (* 0.05 k3)) u_applied))
	  (setf state (+ state (np.array (np.squeeze (* (/ 0.05 6.0) (+ k1 (* 2 k2) (* 2 k3) k4))))))
	  
	  (opti.set_value current_x state)
	  
	  (setf t0 (time.time))
	  (setf sol (opti.solve))
	  (dot times (append (- (time.time) t0))))
	  
     (setf avg_time (* 1000.0 (np.mean times)))
     (setf max_time (* 1000.0 (np.max times)))
     (setf min_time (* 1000.0 (np.min times)))
     (print (fstring "MPC Loop (Warm Start) - {n_steps} steps:"))
     (print (fstring "  Avg: {avg_time:.2f} ms"))
     (print (fstring "  Min: {min_time:.2f} ms"))
     (print (fstring "  Max: {max_time:.2f} ms"))
     (return (tuple t_cold avg_time)))

   (setf (ntuple opti_no_jit X_no U_no c_no t_no f_ode_no) (setup_mpc False))
   (run_benchmark opti_no_jit X_no U_no c_no t_no f_ode_no (string "Python (No JIT/C-Code)"))
   
   (setf (ntuple opti_jit X_jit U_jit c_jit t_jit f_ode_jit) (setup_mpc True))
   (run_benchmark opti_jit X_jit U_jit c_jit t_jit f_ode_jit (string "C-Code Export (JIT)"))
   
   (print (string "\nBenchmark beendet."))
   ))
