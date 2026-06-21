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
				  (merge-pathnames #P"p06_integrator"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations))
	  (imports ((ca casadi)
		    (np numpy)
		    (plt matplotlib.pyplot))))

     (comments "========================================================================"
	       "2-STUFIGER DIODEN-GEKLEMMTER KONDENSATORFILTER (DAE-SYSTEM)"
	       "========================================================================"
	       ""
	       "1. BESCHREIBUNG DER SCHALTUNG & VERWENDUNGSZWECK:"
	       "   Diese Schaltung repraesentiert einen zweistufigen passiven Tiefpassfilter,"
	       "   bei dem jeder Kondensatorstufe eine Diode parallel geschaltet ist."
	       "   Die Dioden dienen als Spannungsbegrenzer (Clamping-Dioden)."
	       "   Die Schaltung wird von einer konstanten Ladestromquelle I_in gespeist."
	       ""
	       "   Anwendungsbereiche:"
	       "   - Signalbegrenzung (Cipping/Limiting) in HF- und Mischsignalschaltungen,"
	       "     um empfindliche nachfolgende Stufen vor Ueberspannungen zu schuetzen."
	       "   - Spitzenwertdetektoren (Peak Detector) und Huellkurvendetektoren."
	       "   - Ueberspannungsschutz- und Gleichrichterschaltungen."
	       ""
	       "2. VOR- UND NACHTEILE DER SCHALTUNG:"
	       "   Vorteile:"
	       "   - Einfaches, rein passives Design mit sehr wenigen Bauelementen."
	       "   - Sehr schnelle Reaktionszeit (Klemmung) durch das passive Diodenverhalten."
	       "   - Effektive Begrenzung von Spannungsspitzen."
	       "   Nachteile:"
	       "   - Starke Abhaengigkeit von Diodeneigenschaften (Schnittspannung, Sättigungsstrom),"
	       "     die stark temperaturabhaengig sind."
	       "   - Verlustleistung ueber die Dioden."
	       "   - Schwer manuell abzustimmen wegen der exponentiellen Nichtlinearitaeten."
	       ""
	       "3. RELEVANZ FUER ELEKTRONIKINGENIEURE & ZIELSETZUNG:"
	       "   Fuer das Design solcher Schaltungen ist die korrekte Dimensionierung"
	       "   der Kapazitaeten C1 und C2 entscheidend, um Spannungsziele bei bestimmten"
	       "   Zeitkonstanten einzuhalten. Hier nutzen wir die Sensitivitaetsanalyse und"
	       "   die DAE-beschraenkte Optimierung (IPOPT mit CasADi IDAS), um die Kapazitaeten"
	       "   automatisch so zu optimieren, dass die Ausgangsspannung V_C2 am Ende eines"
	       "   Zeitraums maximiert wird, waehrend das gesamte Kapazitaetsbudget beschraenkt ist."
	       ""
	       "4. SCHALTPLAN (ASCII-ART):"
	       ""
	       "                   Knoten 1           Knoten 2"
	       "   I_in              V_C1               V_C2"
	       "    o-------+-----------o------[ R ]-------o-----------o (Ausgang V_out)"
	       "            |           |                  |"
	       "            |           |                  |"
	       "          [I_in]      [ C1 ]             [ C2 ]"
	       "            |           |                  |"
	       "            |         + | -              + | -"
	       "            |         ( | )              ( | )"
	       "            |           |                  |"
	       "            |       D1  |              D2  |"
	       "            |          / \\                / \\ "
	       "            |         / v \\              / v \\"
	       "            |         -----              -----"
	       "            |           |                  |"
	       "            |           |                  |"
	       "   ---------+-----------+------------------+----------- GND"
	       ""
	       "5. ERGEBNISSE DIESES PROJEKTS (ZUSAMMENFASSUNG):"
	       "   - DAE-Modellierung: Formulierung eines Widerstands-Kondensator-Dioden-Netzwerks"
	       "     in symbolischen CasADi-Ausdruecken mit 2 differenziellen Zustaenden (Spannungen),"
	       "     2 algebraischen Zustaenden (Diodenstroemen) und 3 Parametern."
	       "   - Integrator-Erstellung: Initialisierung von Einschritt- und Zeitgitter-Lösern"
	       "     ueber die CasADi 'integrator'-Schnittstelle mit dem IDAS-Plugin."
	       "   - Sensitivitaetsanalyse: Symbolisches Einwickeln des DAE-Lösers zur Berechnung"
	       "     von Gradienten (1. Ordnung) und praezise Kreuzvalidierung der Hessian-Matrizen"
	       "     (2. Ordnung) mittels Adjoint-over-Adjoint (AOA) und Forward-over-Adjoint (FOA)."
	       "   - DAE-Optimierung: Erfolgreiches Auffinden der optimalen Kapazitaetsaufteilung"
	       "     (C1 = 1.9F, C2 = 0.1F), um die Ausgangsspannung V_C2 zu maximieren."
	       "========================================================================")

     ;; Symbolische Variablen fuer CasADi definieren
     (setf x (ca.SX.sym (string "x") 2) ; Differenzielle Zustaende (Knotenspannungen in Volt)
	   z (ca.SX.sym (string "z") 2) ; Algebraische Zustaende (Diodenstroeme in Ampere)
	   p (ca.SX.sym (string "p") 3)) ; Systemparameter (Eingangsstrom, Kapazitaeten)

     ;; Zustaende und Parameter fuer bessere Lesbarkeit entpacken
     (setf V_C1 (aref x 0) ; Spannung am Kondensator 1 (in Volt)
	   V_C2 (aref x 1) ; Spannung am Kondensator 2 (in Volt)
	   I_D1 (aref z 0) ; Strom durch Diode 1 (in Ampere)
	   I_D2 (aref z 1) ; Strom durch Diode 2 (in Ampere)
	   I_in (aref p 0) ; Eingangsladestrom (in Ampere)
	   C1   (aref p 1) ; Kapazitaet von Kondensator 1 (in Farad)
	   C2   (aref p 2)) ; Kapazitaet von Kondensator 2 (in Farad)

     ;; Dioden- und Bauelementkonstanten festlegen
     (setf R 2.0   ; Kopplungswiderstand zwischen den Stufen (in Ohm)
	   Is1 0.1 ; Sättigungsstrom der Diode 1 (in Ampere)
	   Is2 0.1 ; Sättigungsstrom der Diode 2 (in Ampere)
	   Vt1 0.5 ; Temperaturspannung der Diode 1 (in Volt, steuert die Kennlinie)
	   Vt2 0.5) ; Temperaturspannung der Diode 2 (in Volt, steuert die Kennlinie)

     ;; DAE-Gleichungen aufstellen (Ladungsbilanz und algebraische Nebenbedingungen)
     ;; ode: Ableitungen der Zustandsspannungen (Knotengleichungen nach KCL dividiert durch C)
     (setf ode (ca.vertcat (/ (- I_in I_D1 (/ (- V_C1 V_C2) R)) C1)
			   (/ (- (/ (- V_C1 V_C2) R) I_D2) C2))
	   ;; alg: Algebraische Gleichungen (Shockley-Dioden-Nichtlinearitaeten)
	   alg (ca.vertcat (- I_D1 (* Is1 (- (ca.exp (/ V_C1 Vt1)) 1.0)))
			   (- I_D2 (* Is2 (- (ca.exp (/ V_C2 Vt2)) 1.0)))))

     (setf dae (dict ((string "x") x)
		     ((string "z") z)
		     ((string "p") p)
		     ((string "ode") ode)
		     ((string "alg") alg)))

     ;; Zeitraster fuer die Simulation definieren
     (setf t0 0.0 ; Startzeitpunkt der Integration (in Sekunden)
	   t_grid (np.linspace 0.0 1.0 100)) ; Diskretisiertes Gitter von 0.0 bis 1.0 Sekunden mit 100 Schritten

     (comments "Integratoren unter Verwendung des 'idas'-Plugins definieren"
	       "F_tf   - Integriert bis zur Endzeit tf=1.0 (optimiert fuer NLP/Hessian)"
	       "F_grid - Integriert auf dem gesamten t_grid (fuer Trajektorien und Sensitivitaetsverlaeufe)")
     (setf F_tf (ca.integrator (string "F_tf") (string "idas") dae 0.0 1.0)
	   F_grid (ca.integrator (string "F_grid") (string "idas") dae t0 t_grid))

     (comments "========================================================================"
	       "SYMBOLIC SENSITIVITY WRAPPERS"
	       "========================================================================"
	       "Wir wickeln die DAE-Integration in eine Standard-CasADi-Funktion ein,"
	       "um analytische Ableitungen durch den Solver-Block zu ermoeglichen.")
     (setf res_tf (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p)
	   V_C2_tf (aref (aref res_tf (string "xf")) 1))

     (comments "Gradient (Sensitivitaet 1. Ordnung) von V_C2(t_f) bzgl. der Parameter [I_in, C1, C2]")
     (setf grad_V_C2 (dot (ca.jacobian V_C2_tf p) T))

     (comments "Hessian (Sensitivitaet 2. Ordnung) von V_C2(t_f) bzgl. der Parameter")
     (comments "Modus 1: Adjoint-over-Adjoint (AOA) - Rueckwaerts-ueber-Adjungierte")
     (setf (ntuple H_aoa g_aoa) (ca.hessian V_C2_tf p))
     (setf f_aoa (ca.Function (string "f_aoa") (list p) (list H_aoa g_aoa)))

     (comments "Modus 2: Forward-over-Adjoint (FOA) - Vorwaerts-Ableitung des adjungierten Gradienten")
     (setf H_foa (ca.jacobian grad_V_C2 p))
     (setf f_foa (ca.Function (string "f_foa") (list p) (list H_foa)))

     (comments "========================================================================"
	       "NOMINALE SENSITIVITAETSANALYSE"
	       "========================================================================")
     (setf p_nom (list 2.0 1.0 1.0))
     (setf (ntuple H_aoa_val g_aoa_val) (f_aoa p_nom)
	   H_foa_val (f_foa p_nom))

     (print (string "--- Nominal Parameter Sensitivity (p = [I_in=2.0, C1=1.0, C2=1.0]) ---"))
     (print (fstring "Gradient (AOA):\\n{g_aoa_val}"))
     (print (fstring "Hessian (AOA):\\n{H_aoa_val}"))
     (print (fstring "Hessian (FOA):\\n{H_foa_val}"))
     (print (fstring "Hessian agreement check (max absolute difference): {np.max(np.abs(np.array(H_aoa_val) - np.array(H_foa_val)))}"))

     (comments "========================================================================"
	       "DAE-BESCHRAENKTE PARAMETEROPTIMIERUNG"
	       "========================================================================"
	       "Maximierung der Ausgangsspannung V_C2(t_f) unter der Randbedingung"
	       "der Gesamtkapazitaet C1 + C2 = 2.0 F, bei festem Ladestrom I_in = 2.0 A.")
     (setf opti (ca.Opti)
	   p_var (opti.variable 3))

     (opti.subject_to (== (aref p_var 0) 2.0))
     (opti.subject_to (== (+ (aref p_var 1) (aref p_var 2)) 2.0))
     (opti.subject_to (>= (aref p_var 1) 0.1))
     (opti.subject_to (>= (aref p_var 2) 0.1))
     (opti.set_initial p_var p_nom)

     (setf res_opt (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var)
	   V_C2_tf_opt (aref (aref res_opt (string "xf")) 1))

     (opti.minimize (* -1.0 V_C2_tf_opt))
     (opti.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))
     (setf sol (opti.solve)
	   p_opt (sol.value p_var)
	   V_C2_tf_max (sol.value V_C2_tf_opt))

     (print (string "--- DAE-Constrained Optimization Results ---"))
     (print (fstring "Optimal Capacitances: C1 = {p_opt[1]:.4f} F, C2 = {p_opt[2]:.4f} F"))
     (print (fstring "Maximum V_C2(t_f): {V_C2_tf_max:.4f} V (Nominal: {float(F_tf(x0=[0,0], z0=[0,0], p=p_nom)[\"xf\"][1]):.4f} V)"))

     (comments "========================================================================"
	       "TRAJEKTORIEN-SIMULATION UND SENSITIVITAETS-VERLAEUFE"
	       "========================================================================")
     (setf sim_nom (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_nom)
	   sim_opt (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt))

     (setf xf_nom (np.array (aref sim_nom (string "xf")))
	   zf_nom (np.array (aref sim_nom (string "zf")))
	   xf_opt (np.array (aref sim_opt (string "xf")))
	   zf_opt (np.array (aref sim_opt (string "zf"))))

     (comments "Berechnung der Zeitverlaeufe der Sensitivitaeten erster Ordnung fuer V_C2(t)")
     (setf res_grid_sym (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p)
	   V_C2_traj_sym (aref (aref res_grid_sym (string "xf")) 1 (slice)))
     (setf J_traj_sym (ca.jacobian V_C2_traj_sym p)
	   J_func (ca.Function (string "J_func") (list p) (list J_traj_sym))
	   J_val (np.array (J_func p_nom)))

     (comments "2D Parametersweep ueber C1 und C2 zur Konturdarstellung")
     (setf C1_vals (np.linspace 0.1 3.0 40)
	   C2_vals (np.linspace 0.1 3.0 40))
     (setf (ntuple C1_grid C2_grid) (np.meshgrid C1_vals C2_vals)
	   V_C2_tf_grid (np.zeros_like C1_grid))

     (for (i (range (len C2_vals)))
	  (for (j (range (len C1_vals)))
	       (setf p_val_ij (list 2.0 (aref C1_grid i j) (aref C2_grid i j)))
	       (setf out (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_val_ij))
	       (setf (aref V_C2_tf_grid i j) (float (aref (aref out (string "xf")) 1)))))

     (comments "========================================================================"
	       "VISUALISIERUNG UND PLOT-ERSTELLUNG"
	       "========================================================================")
     (setf (ntuple fig axs) (plt.subplots 2 2 :figsize (tuple 14 10)))
     (if (in (string "seaborn-v0_8-whitegrid") plt.style.available)
	 (plt.style.use (string "seaborn-v0_8-whitegrid"))
	 (plt.style.use (string "default")))

     ;; Panel 1: Kondensatorspannungen
     (dot (aref axs 0 0) (plot t_grid (aref xf_nom 0 (slice)) (string "b-") :label (string "V_C1 (Nominal)")))
     (dot (aref axs 0 0) (plot t_grid (aref xf_nom 1 (slice)) (string "r-") :label (string "V_C2 (Nominal)")))
     (dot (aref axs 0 0) (plot t_grid (aref xf_opt 0 (slice)) (string "b--") :alpha 0.7 :label (string "V_C1 (Optimal)")))
     (dot (aref axs 0 0) (plot t_grid (aref xf_opt 1 (slice)) (string "r--") :alpha 0.7 :label (string "V_C2 (Optimal)")))
     (dot (aref axs 0 0) (set_title (string "Kondensatorspannungen ueber die Zeit") :fontsize 12 :fontweight (string "bold")))
     (dot (aref axs 0 0) (set_xlabel (string "Zeit [s]")))
     (dot (aref axs 0 0) (set_ylabel (string "Spannung [V]")))
     (dot (aref axs 0 0) (legend))
     (dot (aref axs 0 0) (grid True :alpha 0.5))

     ;; Panel 2: Diodenstroeme
     (dot (aref axs 0 1) (plot t_grid (aref zf_nom 0 (slice)) (string "g-") :label (string "I_D1 (Nominal)")))
     (dot (aref axs 0 1) (plot t_grid (aref zf_nom 1 (slice)) (string "m-") :label (string "I_D2 (Nominal)")))
     (dot (aref axs 0 1) (plot t_grid (aref zf_opt 0 (slice)) (string "g--") :alpha 0.7 :label (string "I_D1 (Optimal)")))
     (dot (aref axs 0 1) (plot t_grid (aref zf_opt 1 (slice)) (string "m--") :alpha 0.7 :label (string "I_D2 (Optimal)")))
     (dot (aref axs 0 1) (set_title (string "Diodenstroeme ueber die Zeit") :fontsize 12 :fontweight (string "bold")))
     (dot (aref axs 0 1) (set_xlabel (string "Zeit [s]")))
     (dot (aref axs 0 1) (set_ylabel (string "Strom [A]")))
     (dot (aref axs 0 1) (legend))
     (dot (aref axs 0 1) (grid True :alpha 0.5))

     ;; Panel 3: Trajektorien-Sensitivitaeten
     (dot (aref axs 1 0) (plot t_grid (aref J_val (slice) 0) (string "k-") :label (string "dV_C2/dI_in")))
     (dot (aref axs 1 0) (plot t_grid (aref J_val (slice) 1) (string "c-") :label (string "dV_C2/dC1")))
     (dot (aref axs 1 0) (plot t_grid (aref J_val (slice) 2) (string "y-") :label (string "dV_C2/dC2")))
     (dot (aref axs 1 0) (set_title (string "Sensitivitaet von V_C2(t) bzgl. der Parameter") :fontsize 12 :fontweight (string "bold")))
     (dot (aref axs 1 0) (set_xlabel (string "Zeit [s]")))
     (dot (aref axs 1 0) (set_ylabel (string "Sensitivitaet")))
     (dot (aref axs 1 0) (legend))
     (dot (aref axs 1 0) (grid True :alpha 0.5))

     ;; Panel 4: Heatmap von V_C2(tf) bzgl. C1 und C2
     (setf contour (dot (aref axs 1 1) (contourf C1_grid C2_grid V_C2_tf_grid :levels 20 :cmap (string "viridis")))
	   cbar (fig.colorbar contour :ax (aref axs 1 1)))
     (dot cbar (set_label (string "V_C2(t_f) [V]") :fontsize 11))

     (setf c1_line (np.linspace 0.1 1.9 100)
	   c2_line (- 2.0 c1_line))
     (dot (aref axs 1 1) (plot c1_line c2_line (string "r--") :lw 2.0 :label (string "Constraint: C1 + C2 = 2.0")))
     (dot (aref axs 1 1) (plot (list (aref p_nom 1)) (list (aref p_nom 2)) (string "wo") :ms 8 :markeredgecolor (string "black") :label (string "Nominal Point")))
     (dot (aref axs 1 1) (plot (list (aref p_opt 1)) (list (aref p_opt 2)) (string "r*") :ms 12 :markeredgecolor (string "black") :label (string "Optimal Point")))
     (dot (aref axs 1 1) (set_title (string "V_C2(t_f) vs Kapazitaeten") :fontsize 12 :fontweight (string "bold")))
     (dot (aref axs 1 1) (set_xlabel (string "C1 [F]")))
     (dot (aref axs 1 1) (set_ylabel (string "C2 [F]")))
     (dot (aref axs 1 1) (legend))
     (dot (aref axs 1 1) (grid True :alpha 0.3))

     (plt.suptitle (string "2-Stufiger Dioden-Geklemmter Kondensatorfilter: Simulation, Sensitivitaeten & Optimierung")
		   :fontsize 14 :fontweight (string "bold") :y 0.98)
     (plt.tight_layout)
     (plt.savefig (string "diode_sensitivities.png") :dpi 150)
     (print (string "Plot saved to diode_sensitivities.png"))
     )))
