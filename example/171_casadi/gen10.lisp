(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator))

(in-package #:g)

(progn
  (defparameter *source* "example/171_casadi/")

  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p10_porkchop_cpp"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    os
		    shutil
		    (time time))))

     (comments "========================================================================"
	       "PHYSIKALISCHE KONSTANTEN"
	       "========================================================================")
     (setf MU_SUN (* 4.0 (** np.pi 2)))
     (setf AU_KM 1.496e8
	   YR_S (* 365.25 24 3600)
	   V_CONV (/ AU_KM YR_S))
     (setf A_EARTH 1.0
	   T_EARTH 1.0
	   A_MARS  1.524
	   T_MARS  1.881)

     (comments "========================================================================"
	       "ANALYTISCHE PLANETENBAHNEN"
	       "========================================================================")
     (def planet_state (t_yr a period)
       (setf omega (/ (* 2 np.pi) period)
	     theta (* omega t_yr)
	     x (* a (cos theta))
	     y (* a (sin theta))
	     v (/ (* 2 np.pi a) period)
	     vx (* -1 v (sin theta))
	     vy (* v (cos theta)))
       (return (tuple x y vx vy)))

     (comments "========================================================================"
	       "SYMBOLISCHER RK4-INTEGRATOR"
	       "========================================================================")
     (def integrate_rk4 (x0 y0 vx0 vy0 tof)
       (setf s_curr (vertcat x0 y0 vx0 vy0))
       (setf N_steps 150)
       (setf h (/ tof N_steps))
       (for (i (range N_steps))
	    ;; ODE-Auswertung: ds/dt = [vx, vy, ax, ay]
	    ;; k1
	    (setf r1 (sqrt (+ (** (aref s_curr 0) 2) (** (aref s_curr 1) 2))))
	    (setf ax1 (/ (* -1 MU_SUN (aref s_curr 0)) (** r1 3))
		  ay1 (/ (* -1 MU_SUN (aref s_curr 1)) (** r1 3)))
	    (setf k1 (vertcat (aref s_curr 2) (aref s_curr 3) ax1 ay1))
	    
	    ;; k2
	    (setf s2 (+ s_curr (* 0.5 h k1)))
	    (setf r2 (sqrt (+ (** (aref s2 0) 2) (** (aref s2 1) 2))))
	    (setf ax2 (/ (* -1 MU_SUN (aref s2 0)) (** r2 3))
		  ay2 (/ (* -1 MU_SUN (aref s2 1)) (** r2 3)))
	    (setf k2 (vertcat (aref s2 2) (aref s2 3) ax2 ay2))
	    
	    ;; k3
	    (setf s3 (+ s_curr (* 0.5 h k2)))
	    (setf r3 (sqrt (+ (** (aref s3 0) 2) (** (aref s3 1) 2))))
	    (setf ax3 (/ (* -1 MU_SUN (aref s3 0)) (** r3 3))
		  ay3 (/ (* -1 MU_SUN (aref s3 1)) (** r3 3)))
	    (setf k3 (vertcat (aref s3 2) (aref s3 3) ax3 ay3))
	    
	    ;; k4
	    (setf s4 (+ s_curr (* h k3)))
	    (setf r4 (sqrt (+ (** (aref s4 0) 2) (** (aref s4 1) 2))))
	    (setf ax4 (/ (* -1 MU_SUN (aref s4 0)) (** r4 3))
		  ay4 (/ (* -1 MU_SUN (aref s4 1)) (** r4 3)))
	    (setf k4 (vertcat (aref s4 2) (aref s4 3) ax4 ay4))
	    
	    ;; Update s_curr
	    (setf s_curr (+ s_curr (* (/ h 6.0) (+ k1 (* 2.0 k2) (* 2.0 k3) k4)))))
       (return s_curr))

     (comments "========================================================================"
	       "EVALUIERUNGS-FUNKTION (DYNAMIK + JACOBIAN)"
	       "========================================================================")
     ;; z = [vx0, vy0] (Eingänge)
     ;; p = [t_dep, tof] (Eingänge)
     (setf z (SX.sym (string "z") 2)
	   p (SX.sym (string "p") 2))
     (setf vx0 (aref z 0)
	   vy0 (aref z 1)
	   t_dep (aref p 0)
	   tof (aref p 1))

     ;; Planetenpositionen
     (setf (ntuple x_E y_E vx_E vy_E) (planet_state t_dep A_EARTH T_EARTH))
     (setf (ntuple x_M y_M vx_M vy_M) (planet_state (+ t_dep tof) A_MARS T_MARS))

     ;; Integration
     (setf s_end (integrate_rk4 x_E y_E vx0 vy0 tof))

     ;; Residuen
     (setf g_eq (vertcat (- (aref s_end 0) x_M)
			 (- (aref s_end 1) y_M)))

     ;; Jacobimatrix
     (setf J (jacobian g_eq z))

     ;; Definition der Funktion, die wir exportieren wollen
     ;; Eingänge: z, p
     ;; Ausgänge: g_eq (Residuum), J (Jacobian 2x2), s_end (Endzustand Sonde), Erd-v, Mars-v
     (setf lambert_eval (Function (string "lambert_eval")
				  (list z p)
				  (list g_eq
					J
					s_end
					(vertcat vx_E vy_E)
					(vertcat vx_M vy_M))))

     (comments "========================================================================"
	       "C++ CODEGEN EXPORT & VERSCHIEBUNG NACH 'cpp_10'"
	       "========================================================================")
     (setf file_dir (dot os path (dirname (os.path.abspath __file__)))
           target_dir (dot os path (join file_dir (string "cpp_10"))))
     (dot os (makedirs target_dir :exist_ok True))

     (setf opts (dict ((string "cpp") True)
		      ((string "with_header") True)))
     
     (dot lambert_eval (generate (string "lambert_solver.cpp") opts))
     
     (dot shutil (move (string "lambert_solver.cpp") (dot os path (join target_dir (string "lambert_solver.cpp")))))
     (dot shutil (move (string "lambert_solver.h") (dot os path (join target_dir (string "lambert_solver.h")))))
     (print (string "C++-Code erfolgreich exportiert"))
     )))
