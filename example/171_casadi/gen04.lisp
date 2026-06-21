(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator))

(in-package #:g)

#|
================================================================================
INTERPLANETARER TRANSFER: SCHNELLE KONTUR-VERFOLGUNG (CONTINUATION)
================================================================================

1. Problemstellung:
Im vorherigen Ansatz (gen03.lisp) wurde das Pork-Chop-Diagramm durch eine
vollständige Gitterberechnung (NxM) erzeugt. Dies ist teuer, da für jeden
Gitterpunkt ein NLP gelöst wird.

2. Continuation-Methode (Arc-Length Continuation):
Um die Generierung massiv zu beschleunigen, "schießen" wir direkt entlang
der Höhenlinien (Iso-Delta-v). Eine Höhenlinie Delta-v = C ist eine 1D-Kurve
im 4D-Zustandsraum z = [vx0, vy0, t_dep, t_tof].
Wir formulieren das Problem als differentiell-algebraisches System:
  g(z) = [x_end - x_M, y_end - y_M, Delta-v - C] = 0
Der Tangentenvektor T liegt im Nullraum der Jacobi-Matrix J = dg/dz.
Wir schreiten entlang der Tangente voran (Predictor) und projizieren mit einem
Newton-Rootfinder zurück auf die Kontur (Corrector).

3. Parallelisierung:
Wir identifizieren zuerst das globale Minimum. Dann bestimmen wir für
verschiedene C-Werte Startpunkte auf der jeweiligen Kontur.
Das Abfahren (Tracing) der Konturen ist unabhängig und wird parallel
auf mehrere CPUs verteilt (multiprocessing.Pool).
|#

(progn
  (defparameter *source* "example/171_casadi/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p04_porkchop_fast"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot)
		    (mp multiprocessing)
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
     
     (def planet_state (t_yr a period)
       (setf omega (/ (* 2 np.pi) period)
	     theta (* omega t_yr)
	     x (* a (np.cos theta))
	     y (* a (np.sin theta))
	     v (/ (* 2 np.pi a) period)
	     vx (* -1 v (np.sin theta))
	     vy (* v (np.cos theta)))
       (return (tuple x y vx vy)))
     
     (def create_integrator ()
       (setf x  (SX.sym (string "x"))
	     y  (SX.sym (string "y"))
	     vx (SX.sym (string "vx"))
	     vy (SX.sym (string "vy"))
	     state (vertcat x y vx vy)
	     tof (SX.sym (string "tof"))
	     r (sqrt (+ (** x 2) (** y 2)))
	     ode (* tof (vertcat vx
				 vy
				 (/ (* -1 MU_SUN x) (** r 3))
				 (/ (* -1 MU_SUN y) (** r 3))))
	     dae (dict ((string "x") state)
		       ((string "p") tof)
		       ((string "ode") ode)))
       (return (integrator (string "F")
			   (string "cvodes")
			   dae
			   (dict ((string "t0") 0)
				 ((string "tf") 1)
				 ((string "reltol") 1e-10)
				 ((string "abstol") 1e-12)))))
     
     (comments "========================================================================"
	       "WORKER-INITIALISIERUNG: SOLVER DEFINITIONEN"
	       "========================================================================"
	       "Die CasADi-Rootfinder werden einmal pro Worker-Prozess initialisiert."
	       "Das vermeidet Serialisierungsprobleme und Overhead.")
     
     (setf worker_state (dict))
     
     (def worker_init ()
       (setf F_int (create_integrator))
       
       (comments "--- Iso-Delta-v Continuation Solver ---")
       (setf z (SX.sym (string "z") 4))
       (setf vx0 (aref z 0) vy0 (aref z 1) t_dep (aref z 2) tof (aref z 3))
       
       ;; Parameter: p = [C_val, z_pred[0..3], T_pred[0..3]]
       (setf p (SX.sym (string "p") 9))
       (setf C_val (aref p 0)
	     z_pred (aref p (slice 1 5))
	     T_pred (aref p (slice 5 9)))
       
       (setf (ntuple x_E y_E vx_E vy_E) (planet_state t_dep A_EARTH T_EARTH))
       (setf (ntuple x_M y_M vx_M vy_M) (planet_state (+ t_dep tof) A_MARS T_MARS))
       
       (setf x0_vec (vertcat x_E y_E vx0 vy0))
       (setf res_int (F_int :x0 x0_vec :p tof)
	     xf (aref res_int (string "xf")))
       
       (setf dv_dep (sqrt (+ (** (- vx0 vx_E) 2) (** (- vy0 vy_E) 2)))
	     dv_arr (sqrt (+ (** (- (aref xf 2) vx_M) 2) (** (- (aref xf 3) vy_M) 2)))
	     dv_tot (* (+ dv_dep dv_arr) V_CONV)) ;; Umrechnung in km/s für C
       
       ;; Residuen für die physikalischen Gleichungen
       (setf g1 (- (aref xf 0) x_M)
	     g2 (- (aref xf 1) y_M)
	     g3 (- dv_tot C_val))
       (setf g_phys (vertcat g1 g2 g3))
       
       ;; Jacobi-Matrix der physikalischen Gleichungen (3x4)
       (setf J (jacobian g_phys z))
       (setf J_iso_dv_func (Function (string "J") (list z) (list J)))
       
       ;; Orthogonalitäts-Bedingung für den Corrector (Arc-Length)
       (setf g4 (casadi.dot (- z z_pred) T_pred))
       (setf g_all (vertcat g_phys g4))
       
       ;; Newton-Rootfinder
       (setf rf_dict (dict ((string "x") z)
			   ((string "p") p)
			   ((string "g") g_all)))
       (setf solver_iso_dv (rootfinder (string "rf") (string "newton") rf_dict
				       (dict ((string "error_on_fail") False))))
       (setf (aref worker_state (string "solver_iso_dv")) solver_iso_dv)
       (setf (aref worker_state (string "J_iso_dv_func")) J_iso_dv_func))
     
     (comments "========================================================================"
	       "WORKER-FUNKTION: KONTUR-VERFOLGUNG"
	       "========================================================================")
     
     (def trace_contour_worker (args)
       (setf solver_iso_dv (aref worker_state (string "solver_iso_dv")))
       (setf J_iso_dv_func (aref worker_state (string "J_iso_dv_func")))
       (setf (ntuple C_target z_start_arr) args)
       (setf path (list))
       (setf z_curr (np.array z_start_arr))
       (setf T_prev None)
       
       ;; Bogenlängen-Schrittweite (angepasst an die Skalierung der Variablen)
       ;; Variablen sind in der Größenordnung von 1.0 (Jahre / AU/yr).
       (setf ds 0.015)
       
       (setf should_break False)
       (for (step (range 3000))
	    (unless should_break
		    ;; 1. Tangente berechnen (Nullraum der Jacobi-Matrix)
		    (setf J_mat (np.array (J_iso_dv_func z_curr)))
		    ;; SVD liefert U, S, Vh. Vh[-1,:] ist der Vektor zum kleinsten Singulärwert.
		    (setf (ntuple U S Vh) (np.linalg.svd J_mat))
		    (setf T (aref Vh -1 (slice "" "")))
		    
		    ;; Konsistente Richtung beibehalten
		    (when (is-not T_prev None)
		      (when (< (np.dot T T_prev) 0)
			(setf T (* -1 T))))
		    (setf T_prev T)
		    
		    ;; 2. Predictor-Schritt
		    (setf z_pred (+ z_curr (* ds T)))
		    
		    ;; 3. Corrector-Schritt
		    (setf p_val (np.concatenate (list (list C_target) z_pred T)))
		    (try
		     (do0
		      (setf sol (solver_iso_dv z_pred p_val))
		      (setf z_next (dot (np.array sol) (flatten))))
		     (Exception
		      (do0
		       (setf should_break True)
		       (setf z_next (np.array (list np.nan np.nan np.nan np.nan))))))
		    
		    ;; Prüfen ob Newton konvergiert ist (enthält np.nan bei Fehlschlag)
		    (when (np.any (np.isnan z_next))
		      (setf should_break True))
		    
		    (unless should_break
			    (path.append z_next)
			    (setf z_curr z_next)
			    
			    ;; Abbruchbedingung: Kurve hat sich geschlossen
			    (when (> step 50)
			      (setf dist (np.linalg.norm (- z_curr z_start_arr)))
			      (when (< dist (* ds 1.5))
				(path.append z_start_arr) ;; Schließen
				(setf should_break True))))))
       
       (return (tuple C_target (np.array path))))
     
     (comments "========================================================================"
	       "HAUPTPROGRAMM: MINIMUM-SUCHE & PARALLELES TRACING"
	       "========================================================================")
     
     (when (== __name__ (string "__main__"))
       (print (string "Suche globales Minimum (Hohmann-Region)..."))
       (setf F_int_main (create_integrator))
       
       (setf min_dv 1e9
	     z_min None)
       
       (for (t_guess (np.linspace 0.1 2.0 10))
	    (setf opti (casadi.Opti))
	    (setf z_var (opti.variable 4))
	    (setf vx0 (aref z_var 0) vy0 (aref z_var 1)
		  t_dep (aref z_var 2) tof (aref z_var 3))
	    
	    (opti.subject_to (opti.bounded 0.0 t_dep 2.5))
	    (opti.subject_to (opti.bounded 0.2 tof 1.5))
	    
	    (setf (ntuple x_E y_E vx_E vy_E) (planet_state t_dep A_EARTH T_EARTH))
	    (setf (ntuple x_M y_M vx_M vy_M) (planet_state (+ t_dep tof) A_MARS T_MARS))
	    (setf x0_vec (vertcat x_E y_E vx0 vy0))
	    (setf res_int (F_int_main :x0 x0_vec :p tof)
		  xf (aref res_int (string "xf")))
	    
	    (opti.subject_to (== (aref xf 0) x_M))
	    (opti.subject_to (== (aref xf 1) y_M))
	    
	    (setf dv_dep (sqrt (+ (** (- vx0 vx_E) 2) (** (- vy0 vy_E) 2)))
		  dv_arr (sqrt (+ (** (- (aref xf 2) vx_M) 2) (** (- (aref xf 3) vy_M) 2)))
		  dv_tot (* (+ dv_dep dv_arr) V_CONV))
	    
	    (opti.minimize dv_tot)
	    
	    (setf hohmann_guess_vx (* 1.1 (* (/ (* 2 np.pi A_EARTH) T_EARTH) (np.cos (/ np.pi 6))))
		  hohmann_guess_vy (* 1.1 (* (/ (* 2 np.pi A_EARTH) T_EARTH) (np.sin (/ np.pi 6)))))
	    (opti.set_initial vx0 hohmann_guess_vx)
	    (opti.set_initial vy0 hohmann_guess_vy)
	    (opti.set_initial t_dep t_guess)
	    (opti.set_initial tof 0.7)
	    
	    (opti.solver (string "ipopt")
			 (dict ((string "print_time") False))
			 (dict ((string "print_level") 0)))
	    
	    (try
	     (do0
	      (setf sol_min (opti.solve))
	      (setf val (sol_min.value dv_tot))
	      (when (< val min_dv)
		(setf min_dv val)
		(setf z_min (np.array (sol_min.value z_var)))))
	     (Exception
	      pass)))
       
       (setf C_min min_dv)
       
       (print (fstring "Globales Minimum gefunden: Delta-v = {C_min:.2f} km/s at t_dep={z_min[2]:.3f}, tof={z_min[3]:.3f}"))
       
       (comments "Startpunkte für Höhenlinien suchen")
       (setf contour_levels (np.arange (+ (np.ceil C_min) 0.5) 15.0 0.5))
       (setf start_tasks (list))
       
       (for (C contour_levels)
	    (setf opti_C (casadi.Opti))
	    (setf z_C (opti_C.variable 4))
	    
	    (setf vx0_C (aref z_C 0) vy0_C (aref z_C 1) t_dep_C (aref z_C 2) tof_C (aref z_C 3))
	    (setf (ntuple x_E_C y_E_C vx_E_C vy_E_C) (planet_state t_dep_C A_EARTH T_EARTH))
	    (setf (ntuple x_M_C y_M_C vx_M_C vy_M_C) (planet_state (+ t_dep_C tof_C) A_MARS T_MARS))
	    (setf res_int_C (F_int_main :x0 (vertcat x_E_C y_E_C vx0_C vy0_C) :p tof_C))
	    (setf xf_C (aref res_int_C (string "xf")))
	    
	    (opti_C.subject_to (== (aref xf_C 0) x_M_C))
	    (opti_C.subject_to (== (aref xf_C 1) y_M_C))
	    
	    (setf dv_dep_C (sqrt (+ (** (- vx0_C vx_E_C) 2) (** (- vy0_C vy_E_C) 2)))
		  dv_arr_C (sqrt (+ (** (- (aref xf_C 2) vx_M_C) 2) (** (- (aref xf_C 3) vy_M_C) 2)))
		  dv_tot_C (* (+ dv_dep_C dv_arr_C) V_CONV))
	    
	    (opti_C.subject_to (== dv_tot_C C))
	    (opti_C.subject_to (== tof_C (aref z_min 3))) ;; Fix auf y-Höhe des Minimums
	    (opti_C.subject_to (opti_C.bounded (- (aref z_min 2) 1.5) t_dep_C (aref z_min 2))) ;; Linke Flanke
	    
	    (opti_C.minimize 0)
	    (opti_C.set_initial z_C z_min)
	    (opti_C.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))
	    
	    (try
	     (do0
	      (setf sol_C (opti_C.solve))
	      (start_tasks.append (tuple C (np.array (sol_C.value z_C)))))
	     (Exception
	      (print (fstring "Kein Startpunkt für C={C} gefunden.")))))
       
       (print (fstring "Gefundene Startpunkte: {len(start_tasks)}"))
       
       (if (== (len start_tasks) 0)
	   (print (string "FEHLER: Keine Startpunkte für die Höhenlinien gefunden!"))
	   (do0
	       (setf n_cpus (mp.cpu_count))
	       (print (fstring "Starte paralleles Tracing auf {n_cpus} Kernen..."))
	       (setf t_start (time.time))
	       
	       (with (as (mp.Pool :processes n_cpus :initializer worker_init) pool)
		     (setf results (pool.map trace_contour_worker start_tasks)))
	       
	       (setf elapsed (- (time.time) t_start))
	       (print (fstring "Tracing abgeschlossen in {elapsed:.1f} Sekunden"))
	       
	       (comments "========================================================================"
			 "VISUALISIERUNG: PORK-CHOP-DIAGRAMM"
			 "========================================================================")
	       
	       (setf fig_and_ax (plt.subplots 1 1 :figsize (tuple 12 9)))
	       (setf (ntuple fig ax) fig_and_ax)
	       
	       ;; Farbskala für die Konturen
	       (setf cmap (plt.get_cmap (string "RdYlGn_r")))
	       (setf norm (plt.Normalize :vmin (np.min contour_levels) :vmax (np.max contour_levels)))
	       
	       (for ((ntuple C path_arr) results)
		    (when (> (len path_arr) 0)
		      ;; path_arr[:, 2] = t_dep, path_arr[:, 3] = tof
		      (setf t_dep_path (aref path_arr (slice "" "") 2))
		      (setf tof_days_path (* (aref path_arr (slice "" "") 3) 365.25))
		      (setf color (cmap (norm C)))
		      (ax.plot t_dep_path tof_days_path
			       :color color
			       :linewidth 2.0)
		      ;; Kontur-Label hinzufügen (am ersten Punkt)
		      (ax.text (aref t_dep_path 0) (aref tof_days_path 0) (fstring "{C}")
			       :color color :fontsize 10 :fontweight (string "bold")
			       :ha (string "right") :va (string "center"))))
	       
	       ;; Minimum eintragen
	       (ax.plot (aref z_min 2) (* (aref z_min 3) 365.25)
			:marker (string "*") :color (string "gold") :markersize 20
			:markeredgecolor (string "black") :markeredgewidth 1.5 :zorder 10)
	       (ax.annotate (fstring "$\\\\Delta v_{{min}}$ = {C_min:.2f} km/s")
			    :xy (tuple (aref z_min 2) (* (aref z_min 3) 365.25))
			    :xytext (tuple (+ (aref z_min 2) 0.15) (+ (* (aref z_min 3) 365.25) 25))
			    :fontsize 11 :fontweight (string "bold")
			    :color (string "black")
			    :arrowprops (dict ((string "arrowstyle") (string "->")) ((string "color") (string "black")) ((string "lw") 1.5)))
	       
	       (setf sm (plt.cm.ScalarMappable :cmap cmap :norm norm))
	       (sm.set_array (list))
	       (setf cbar (fig.colorbar sm :ax ax))
	       (cbar.set_label (string "Total $\\\\Delta v$ [km/s]") :fontsize 13)
	       
	       (ax.set_xlim 0.0 2.5)
	       (ax.set_ylim 100 450)
	       
	       (ax.set_xlabel (string "Departure date [years from epoch]") :fontsize 13)
	       (ax.set_ylabel (string "Transfer duration [days]") :fontsize 13)
	       (ax.set_title (string "Earth $\\\\rightarrow$ Mars: Pork Chop Diagram (Continuation Method)")
			     :fontsize 15 :fontweight (string "bold"))
	       (ax.grid True :alpha 0.3 :linestyle (string "--"))
	       
	       (plt.tight_layout)
	       (plt.savefig (string "porkchop_fast.png") :dpi 150)
	       (print (string "Plot gespeichert: porkchop_fast.png"))
	       (plt.show)
	       ))))))
