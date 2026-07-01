(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11a
  (:use #:cl #:cl-py-generator))

(in-package #:g11a)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11a_pendulum_sim"
						 *source*))
 `(do0
   (do0
    (imports-from (__future__ annotations)
		  (casadi *))
    (imports ((np numpy)
	      (plt matplotlib.pyplot))))

   (comments "--- Physikalische Modellierung (Inverted Pendulum) ---"
	     "M: Masse des Wagens (kg)"
	     "m: Masse des Pendels (kg)"
	     "l: Laenge des Pendels zum Schwerpunkt (m)"
	     "g: Erdbeschleunigung (m/s^2)")
   (setf M 1.0     
	 m 0.1     
	 l 0.5     
	 g 9.81)   

   (comments "Zustandsvariablen:"
	     "s     : Position des Wagens (m)"
	     "v     : Geschwindigkeit des Wagens (m/s)"
	     "theta : Winkel des Pendels (rad), 0 = aufrecht, pi = haengend"
	     "omega : Winkelgeschwindigkeit des Pendels (rad/s)")
   (setf nx 4
	 nu 1)
   
   (setf x (SX.sym (string "x") nx)
	 u (SX.sym (string "u") nu))
   
   (setf s_ (aref x 0)
	 v_ (aref x 1)
	 theta_ (aref x 2)
	 omega_ (aref x 3))
   
   (setf F_ u)
   
   (comments "Nichtlineare Systemdynamik (dx/dt = f(x, u))")
   (setf sin_theta (np.sin theta_)
	 cos_theta (np.cos theta_)
	 den (+ M (* m (- 1.0 (* cos_theta cos_theta)))))
   
   (setf ds v_
	 dv (/ (- (+ F_ (* m l omega_ omega_ sin_theta)) (* m g cos_theta sin_theta)) den)
	 dtheta omega_
	 domega (/ (+ (- (* -1.0 F_ cos_theta) (* m l omega_ omega_ sin_theta cos_theta)) (* (+ M m) g sin_theta)) (* l den)))
   
   (setf f_ode (Function (string "f_ode") (list x u) (list (vertcat ds dv dtheta domega))))

   (comments "--- Opti-Stack & Kollokation Setup ---")
   (setf opti (Opti))
   
   (setf N 50)     ;; Vorhersagehorizont (Anzahl Intervalle)
   (setf T 3.0)    ;; Zeithorizont (s)
   (setf h (/ T N))
   
   (comments "Kollokations-Parameter (Radau-Stützstellen, Polynomgrad d=3)")
   (setf d 3)
   (setf tau_root (np.append 0.0 (collocation_points d (string "radau"))))
   
   (comments "Konstruktion der Kollokationsmatrizen C und D fuer die Lagrange-Polynome")
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
	     
   (comments "Entscheidungsvariablen im Opti-Stack definieren")
   (setf X (opti.variable nx (+ N 1)))      ;; Zustand an den Intervallgrenzen (Knoten)
   (setf Xc (list))
   (for (k (range N))
        (setf Xc_k (list))
        (for (r (range d))
             (dot Xc_k (append (opti.variable nx))))
        (dot Xc (append Xc_k)))
   (setf U (opti.variable nu N))            ;; Konstante Stellkraft ueber das jeweilige Intervall
   
   (comments "Start- und Endbedingungen (Swing-Up)")
   (setf x_start (np.array (list 0.0 0.0 np.pi 0.0)))  ;; Start: Wagen im Ursprung, Pendel haengt nach unten
   (setf x_target (np.array (list 1.0 0.0 0.0 0.0)))   ;; Ziel: Wagen bei 1m, Pendel aufrecht ausbalanciert
   
   (opti.subject_to (== (aref X (slice nil nil) 0) x_start))
   (opti.subject_to (== (aref X (slice nil nil) N) x_target))
   
   (comments "Nebenbedingungen (Bounds)")
   (opti.subject_to (opti.bounded -2.0 (aref X 0 (slice nil nil)) 2.0))   ;; Wagenposition max +/- 2m (Schienenlaenge)
   (opti.subject_to (opti.bounded -15.0 U 15.0))                      ;; Maximale Motorkraft +/- 15 N
   
   (comments "Kollokations-Gleichungen & Intervall-Kontinuitaet erzwingen")
   (for (k (range N))
	(setf Xk (aref X (slice nil nil) k))
	(setf x_end (* (aref D 0) Xk))
	
	(for (j (range 1 (+ d 1)))
	     ;; Berechnung der Ableitung des Polynoms an Punkt j
	     (setf xp (* (aref C 0 j) Xk))
	     (for (r (range d))
		  (setf xp (+ xp (* (aref C (+ r 1) j) (aref (aref Xc k) r)))))
	     
	     ;; Die Ableitung muss der echten Systemdynamik f_ode entsprechen
	     (setf f_eval (f_ode (aref (aref Xc k) (- j 1)) (aref U (slice nil nil) k)))
	     (opti.subject_to (== xp (* h f_eval))))
	     
	;; Rekonstruktion des Zustands am Ende des Intervalls
	(for (r (range d))
	     (setf x_end (+ x_end (* (aref D (+ r 1)) (aref (aref Xc k) r)))))
	     
	;; Kontinuitaet zum naechsten Knotenpunkt
	(opti.subject_to (== (aref X (slice nil nil) (+ k 1)) x_end)))
	
   (comments "Kostenfunktion (Regelungsaufwand minimieren)")
   (setf cost 0.0)
   (for (k (range N))
	(setf cost (+ cost (* 0.01 (** (aref U 0 k) 2)))))
   (opti.minimize cost)
   
   (comments "Startschätzung fuer den Solver (Warm-Start hilft enorm)")
   (opti.set_initial X (dot (np.linspace x_start x_target (+ N 1)) T))
   
   (comments "Solver Konfiguration: IPOPT")
   (opti.solver (string "ipopt") (dict) (dict ((string "print_level") 5)))
   (setf sol (opti.solve))
   
   (comments "--- Ergebnisse auslesen und Plotten ---")
   (setf t_grid (np.linspace 0.0 T (+ N 1)))
   (setf X_res (sol.value X))
   (setf U_res (sol.value U))
   
   (setf (ntuple fig axes) (plt.subplots 3 1 :figsize (tuple 10 12)))
   (dot (aref axes 0) (plot t_grid (aref X_res 0 (slice nil nil)) :label (string "s (Wagenposition)")))
   (dot (aref axes 0) (set_ylabel (string "Position (m)")))
   (dot (aref axes 0) (legend))
   (dot (aref axes 0) (grid))
   
   (dot (aref axes 1) (plot t_grid (aref X_res 2 (slice nil nil)) :label (string "theta (Pendelwinkel)")))
   (dot (aref axes 1) (set_ylabel (string "Winkel (rad)")))
   (dot (aref axes 1) (legend))
   (dot (aref axes 1) (grid))
   
   (dot (aref axes 2) (step (aref t_grid (slice 0 -1)) U_res :label (string "F (Aktuatorkraft)") :where (string "post")))
   (dot (aref axes 2) (set_ylabel (string "Kraft (N)")))
   (dot (aref axes 2) (set_xlabel (string "Zeit (s)")))
   (dot (aref axes 2) (legend))
   (dot (aref axes 2) (grid))
   
   (plt.tight_layout)
   (plt.savefig (string "p11a_pendulum_sim.png"))
   (print (string "Simulation erfolgreich. Plot gespeichert."))
   ))
