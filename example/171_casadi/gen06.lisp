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
		    (plt matplotlib.pyplot)
		    (time time))))

     (comments "========================================================================"
	       "2-STUFIGER DIODEN-GEKLEMMTER KONDENSATORFILTER (DAE-SYSTEM)"
	       "========================================================================"
	       ""
	       "1. BESCHREIBUNG DER SCHALTUNG & TOPOLOGIE:"
	       "   Diese Schaltung ist ein zweistufiger passiver Tiefpassfilter."
	       "   An jedem Kondensatorknoten liegt eine Halbleiterdiode nach Masse."
	       "   Die Schaltung wird von einer konstanten Ladestromquelle I_in gespeist."
	       ""
	       "   Dieselbe Grundtopologie (RC + Diode) dient je nach Beschaltung"
	       "   unterschiedlichen Zwecken:"
	       ""
	       "   (a) Signalbegrenzung (Clipping/Clamping):"
	       "       Die Dioden sind PARALLEL zu einer Signalleitung geschaltet."
	       "       Sobald die Spannung am Knoten die Dioden-Schwellspannung erreicht,"
	       "       leitet die Diode und klemmt die Spannung. Die Ausgangsspannung"
	       "       am SELBEN Knoten wird so begrenzt. Anwendung: Schutz empfindlicher"
	       "       HF-Eingaenge oder ADC-Vorstufen vor Ueberspannungsspitzen."
	       ""
	       "   (b) Spitzenwertdetektor (Peak Detector):"
	       "       Die Diode ist IN SERIE zum Signalpfad geschaltet (Kathode zum"
	       "       Kondensator). Der Kondensator laedt sich ueber die Diode bis zum"
	       "       Spitzenwert auf und haelt ihn, weil die Diode den Rueckfluss sperrt."
	       "       Die Ausgangsspannung wird AM KONDENSATOR abgegriffen."
	       "       Anwendung: Huellkurvendemodulatoren, AGC-Regelschleifen."
	       ""
	       "   In unserem Modell verwenden wir die parallele Beschaltung (Fall a)."
	       "   Die Dioden sind PARALLEL zu den Kondensatoren nach Masse geschaltet."
	       ""
	       "2. VOR- UND NACHTEILE:"
	       "   Vorteile:"
	       "   - Einfaches, rein passives Design mit sehr wenigen Bauelementen."
	       "   - Sehr schnelle Reaktion (Klemmung beginnt sofort bei Ueberspannung)."
	       "   - Keine Versorgungsspannung noetig."
	       "   Nachteile:"
	       "   - Starke Abhaengigkeit von Diodeneigenschaften (Sperrstrom Is,"
	       "     Temperaturspannung Vt), die stark temperaturabhaengig sind."
	       "   - Verlustleistung ueber die Dioden (Waermeentwicklung)."
	       "   - Die exponentiellen Nichtlinearitaeten machen eine analytische"
	       "     Dimensionierung der Kapazitaeten praktisch unmoeglich."
	       ""
	       "3. RELEVANZ FUER ELEKTRONIKINGENIEURE & ZIELSETZUNG:"
	       "   Bei der Dimensionierung dieser Schaltung stellt sich die Frage:"
	       "   'Wie muss ich die Kapazitaeten C1 und C2 aufteilen, wenn mir ein"
	       "    festes Gesamtkapazitaetsbudget (z.B. C1+C2 = 2F) zur Verfuegung"
	       "    steht, um ein bestimmtes Ziel am Ausgang zu erreichen?'"
	       ""
	       "   Diese Frage beantworten wir mithilfe von CasADi (symbolisches DAE)."
	       "   Wir vergleichen hier ZWEI moegliche Entwicklungsziele (Optima):"
	       "   1. Spannungs-Maximierung: Wenn nur ein Spannungspegel triggern soll"
	       "      (z.B. CMOS-Logikgatter, hochohmiger Eingang)."
	       "   2. Energie-Maximierung: Wenn die Schaltung danach eine Last treiben"
	       "      soll oder die Energie geerntet wird (Energy Harvesting). Hier"
	       "      nuetzt eine hohe Spannung nichts, wenn die Kapazitaet C2 zu winzig"
	       "      ist, um relevante Ladungsmengen bereitzustellen."
	       ""
	       "4. SCHALTPLAN (ASCII-ART):"
	       ""
	       "                   Knoten 1           Knoten 2"
	       "   I_in              V_C1               V_C2"
	       "    o-------+-----------o------[ R ]-------o-----------o (Ausgang)"
	       "            |           |                  |"
	       "            |        ---+---            ---+---"
	       "          [I_in]     | C1  |            | C2  |"
	       "            |        ---+---            ---+---"
	       "            |           |                  |"
	       "            |         D1|                D2|"
	       "            |          /|                 /|"
	       "            |         / |                / |"
	       "            |        v  |               v  |"
	       "            |           |                  |"
	       "   ---------+-----------+------------------+--------- GND"
	       ""
	       "   Stromfluss an Knoten 1 (KCL): I_in = I_C1 + I_R + I_D1"
	       "     I_C1 = C1 * dV_C1/dt"
	       "     I_R  = (V_C1 - V_C2)/R"
	       "     I_D1 = Is1*(exp(V_C1/Vt1) - 1)"
	       ""
	       "   Stromfluss an Knoten 2 (KCL): I_R = I_C2 + I_D2"
	       "     I_C2 = C2 * dV_C2/dt"
	       "     I_D2 = Is2*(exp(V_C2/Vt2) - 1)"
	       ""
	       "5. BEOBACHTUNGEN UND INTERPRETATION DER ERGEBNISSE:"
	       "   - Verlustleistung (Panel 3):"
	       "     Die Dioden sollen Ueberspannungen ableiten. In einem schlecht"
	       "     dimensionierten (nominalen) Filter oeffnen sie zu frueh, und"
	       "     massiv viel Leistung (P = V * I) wird als Waerme verbrannt."
	       "     Das belastet die Quelle und kann die Dioden zerstoeren."
	       "     Die optimierten Designs lenken den Strom in die Kondensatoren"
	       "     und minimieren so den thermischen Verlust (Flaeche unter der Kurve)."
	       "   - Optima-Lage (Panel 5 & 6):"
	       "     Spannungs-Optimum (Panel 5): Liegt am aeussersten physikalischen Rand"
	       "     (C2=0.1F). Ein winziges C2 ist immer schneller voll. Es gibt hier keinen"
	       "     Sweet Spot im Inneren."
	       "     Energie-Optimum (Panel 6): Energie (E = 0.5*C2*V_C2^2) erfordert einen"
	       "     Kompromiss. Ein zu kleines C2 speichert keine Energie, ein zu grosses"
	       "     C2 erreicht keine Spannung. Hier liegt das Optimum tief im Inneren"
	       "     des Parameterraums! Dies ist ein klassischer Sweet Spot."
	       "========================================================================")

     (comments "========================================================================"
	       "SYMBOLISCHE DAE-DEFINITION"
	       "========================================================================")
     (setf x (ca.SX.sym (string "x") 2)
	   z (ca.SX.sym (string "z") 2)
	   p (ca.SX.sym (string "p") 3))

     (setf V_C1 (aref x 0)
	   V_C2 (aref x 1)
	   I_D1 (aref z 0)
	   I_D2 (aref z 1)
	   I_in (aref p 0)
	   C1   (aref p 1)
	   C2   (aref p 2))

     (setf R 2.0
	   Is1 0.1
	   Is2 0.1
	   Vt1 0.5
	   Vt2 0.5)

     (setf ode (ca.vertcat (/ (- I_in I_D1 (/ (- V_C1 V_C2) R)) C1)
			   (/ (- (/ (- V_C1 V_C2) R) I_D2) C2)))

     (setf alg (ca.vertcat (- I_D1 (* Is1 (- (ca.exp (/ V_C1 Vt1)) 1.0)))
			   (- I_D2 (* Is2 (- (ca.exp (/ V_C2 Vt2)) 1.0)))))

     (setf dae (dict ((string "x") x)
		     ((string "z") z)
		     ((string "p") p)
		     ((string "ode") ode)
		     ((string "alg") alg)))

     (setf t0 0.0
	   t_grid (np.linspace 0.0 1.0 100))
     (setf F_tf (ca.integrator (string "F_tf") (string "idas") dae 0.0 1.0)
	   F_grid (ca.integrator (string "F_grid") (string "idas") dae t0 t_grid))

     (comments "========================================================================"
	       "SENSITIVITAETSANALYSE: GRADIENT UND HESSIAN"
	       "========================================================================")
     (setf res_tf (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p)
	   V_C2_tf (aref (aref res_tf (string "xf")) 1))

     (setf grad_V_C2 (dot (ca.jacobian V_C2_tf p) T))

     (setf (ntuple H_aoa g_aoa) (ca.hessian V_C2_tf p))
     (setf f_aoa (ca.Function (string "f_aoa") (list p) (list H_aoa g_aoa)))

     (setf H_foa (ca.jacobian grad_V_C2 p))
     (setf f_foa (ca.Function (string "f_foa") (list p) (list H_foa)))

     (setf p_nom (list 2.0 1.0 1.0))
     (setf t_start (time.time))
     (setf (ntuple H_aoa_val g_aoa_val) (f_aoa p_nom))
     (setf t_aoa (- (time.time) t_start))

     (setf t_start (time.time))
     (setf H_foa_val (f_foa p_nom))
     (setf t_foa (- (time.time) t_start))

     (print (string "--- Nominale Sensitivitaeten (p = [I_in=2.0 A, C1=1.0 F, C2=1.0 F]) ---"))
     (print (fstring "Gradient: {g_aoa_val}"))
     (print (fstring "Hessian (AOA): {H_aoa_val}  (Rechenzeit: {t_aoa*1000:.2f} ms)"))
     (print (fstring "Hessian (FOA): {H_foa_val}  (Rechenzeit: {t_foa*1000:.2f} ms)"))
     (print (fstring "Max. Abweichung AOA vs FOA: {np.max(np.abs(np.array(H_aoa_val) - np.array(H_foa_val))):.2e}"))

     (comments "========================================================================"
	       "OPTIMIERUNG 1: SPANNUNGS-MAXIMIERUNG"
	       "========================================================================")
     (setf opti_V (ca.Opti)
	   p_var_V (opti_V.variable 3))

     (opti_V.subject_to (== (aref p_var_V 0) 2.0))
     (opti_V.subject_to (== (+ (aref p_var_V 1) (aref p_var_V 2)) 2.0))
     (opti_V.subject_to (>= (aref p_var_V 1) 0.1))
     (opti_V.subject_to (>= (aref p_var_V 2) 0.1))
     (opti_V.set_initial p_var_V p_nom)

     (setf res_opt_V (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var_V)
	   V_C2_tf_opt_V (aref (aref res_opt_V (string "xf")) 1))

     (opti_V.minimize (* -1.0 V_C2_tf_opt_V))
     (opti_V.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))

     (setf t_start (time.time))
     (setf sol_V (opti_V.solve)
	   p_opt_V (sol_V.value p_var_V)
	   V_C2_tf_max (sol_V.value V_C2_tf_opt_V))
     (setf t_opt_V (- (time.time) t_start))

     (print (string "--- DAE-beschraenkte Optimierung: Spannung ---"))
     (print (fstring "Optimale Kapazitaeten: C1 = {p_opt_V[1]:.4f} F, C2 = {p_opt_V[2]:.4f} F"))
     (print (fstring "Max V_C2(t_f): {V_C2_tf_max:.4f} V"))
     (print (fstring "Rechenzeit IPOPT: {t_opt_V*1000:.1f} ms"))

     (comments "========================================================================"
	       "OPTIMIERUNG 2: ENERGIE-MAXIMIERUNG"
	       "========================================================================")
     (setf opti_E (ca.Opti)
	   p_var_E (opti_E.variable 3))

     (opti_E.subject_to (== (aref p_var_E 0) 2.0))
     (opti_E.subject_to (== (+ (aref p_var_E 1) (aref p_var_E 2)) 2.0))
     (opti_E.subject_to (>= (aref p_var_E 1) 0.1))
     (opti_E.subject_to (>= (aref p_var_E 2) 0.1))
     (opti_E.set_initial p_var_E p_nom)

     (setf res_opt_E (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var_E)
	   V_C2_tf_opt_E (aref (aref res_opt_E (string "xf")) 1)
	   E_opt_E (* 0.5 (aref p_var_E 2) (** V_C2_tf_opt_E 2)))

     (opti_E.minimize (* -1.0 E_opt_E))
     (opti_E.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))

     (setf t_start (time.time))
     (setf sol_E (opti_E.solve)
	   p_opt_E (sol_E.value p_var_E)
	   E_tf_max (sol_E.value E_opt_E))
     (setf t_opt_E (- (time.time) t_start))

     (print (string "--- DAE-beschraenkte Optimierung: Energie ---"))
     (print (fstring "Optimale Kapazitaeten: C1 = {p_opt_E[1]:.4f} F, C2 = {p_opt_E[2]:.4f} F"))
     (print (fstring "Max Energie E(t_f): {E_tf_max:.4f} J"))
     (print (fstring "Rechenzeit IPOPT: {t_opt_E*1000:.1f} ms"))


     (comments "========================================================================"
	       "TRAJEKTORIEN-SIMULATION, VERLUSTLEISTUNG UND ZEITAUFGELOESTE SENSITIVITAETEN"
	       "========================================================================")
     (setf t_start (time.time))
     (setf sim_nom (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_nom)
	   sim_optV (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt_V)
	   sim_optE (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt_E))
     (setf t_sim (- (time.time) t_start))

     (setf xf_nom (np.array (aref sim_nom (string "xf")))
	   zf_nom (np.array (aref sim_nom (string "zf")))
	   xf_optV (np.array (aref sim_optV (string "xf")))
	   zf_optV (np.array (aref sim_optV (string "zf")))
	   xf_optE (np.array (aref sim_optE (string "xf")))
	   zf_optE (np.array (aref sim_optE (string "zf"))))

     ;; Verlustleistung P_loss = V_C1 * I_D1 + V_C2 * I_D2
     (setf P_loss_nom (+ (* (aref xf_nom 0 (slice)) (aref zf_nom 0 (slice)))
			 (* (aref xf_nom 1 (slice)) (aref zf_nom 1 (slice))))
	   P_loss_optV (+ (* (aref xf_optV 0 (slice)) (aref zf_optV 0 (slice)))
			  (* (aref xf_optV 1 (slice)) (aref zf_optV 1 (slice))))
	   P_loss_optE (+ (* (aref xf_optE 0 (slice)) (aref zf_optE 0 (slice)))
			  (* (aref xf_optE 1 (slice)) (aref zf_optE 1 (slice)))))

     (setf res_grid_sym (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p)
	   V_C2_traj_sym (aref (aref res_grid_sym (string "xf")) 1 (slice)))
     (setf t_start (time.time))
     (setf J_traj_sym (ca.jacobian V_C2_traj_sym p)
	   J_func (ca.Function (string "J_func") (list p) (list J_traj_sym))
	   J_val (np.array (J_func p_nom)))
     (setf t_jac (- (time.time) t_start))

     (comments "========================================================================"
	       "2D PARAMETERSWEEP FUER HEATMAPS (SPANNUNG UND ENERGIE)"
	       "========================================================================")
     (setf C1_vals (np.linspace 0.1 3.0 40)
	   C2_vals (np.linspace 0.1 3.0 40))
     (setf (ntuple C1_grid C2_grid) (np.meshgrid C1_vals C2_vals)
	   V_C2_tf_grid (np.zeros_like C1_grid)
	   E_tf_grid (np.zeros_like C1_grid))

     (setf t_start (time.time))
     (for (i (range (len C2_vals)))
	  (for (j (range (len C1_vals)))
	       (setf p_val_ij (list 2.0 (aref C1_grid i j) (aref C2_grid i j)))
	       (setf out (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_val_ij))
	       (setf v_tf (float (aref (aref out (string "xf")) 1)))
	       (setf (aref V_C2_tf_grid i j) v_tf)
	       (setf (aref E_tf_grid i j) (* 0.5 (aref C2_grid i j) (** v_tf 2)))))
     (setf t_sweep (- (time.time) t_start))

     (comments "========================================================================"
	       "VISUALISIERUNG (2x3 GRID)"
	       "========================================================================")
     (setf (ntuple fig axs) (plt.subplots 2 3 :figsize (tuple 18 10)))

     ;; Panel 1: Kondensatorspannungen
     (setf ax (aref axs 0 0))
     ,@(loop for (data idx style label alpha) in
	     '((xf_nom 1 "r:" "Nominal" nil)
	       (xf_optV 1 "r--" "Optimum (Spannung)" 0.8)
	       (xf_optE 1 "r-" "Optimum (Energie)" 0.8))
	     collect
	     `(dot ax (plot t_grid (aref ,data ,idx (slice)) (string ,style)
			    ,@(when alpha `(:alpha ,alpha))
			    :label (string ,label))))
     (dot ax (set_title (string "Ausgangsspannung V_C2(t)") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Spannung [V]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 2: Diodenstroeme I_D1
     (setf ax (aref axs 0 1))
     ,@(loop for (data idx style label alpha) in
	     '((zf_nom 0 "g:" "Nominal" nil)
	       (zf_optV 0 "g--" "Optimum (Spannung)" 0.8)
	       (zf_optE 0 "g-" "Optimum (Energie)" 0.8))
	     collect
	     `(dot ax (plot t_grid (aref ,data ,idx (slice)) (string ,style)
			    ,@(when alpha `(:alpha ,alpha))
			    :label (string ,label))))
     (dot ax (set_title (string "Verluststrom durch Diode 1") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Strom [A]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 3: Verlustleistung
     (setf ax (aref axs 0 2))
     (dot ax (plot t_grid P_loss_nom (string "k:") :label (string "Nominal")))
     (dot ax (plot t_grid P_loss_optV (string "k--") :label (string "Optimum (Spannung)") :alpha 0.8))
     (dot ax (plot t_grid P_loss_optE (string "k-") :label (string "Optimum (Energie)") :alpha 0.8))
     (dot ax (fill_between t_grid 0 P_loss_nom :color (string "gray") :alpha 0.2))
     (dot ax (set_title (string "Verlustleistung (Waerme in Dioden)") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Leistung [W]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 4: Sensitivitaets-Zeitverlaeufe
     (setf ax (aref axs 1 0))
     ,@(loop for (idx style label) in
	     '((0 "k-" "dV_C2/dI_in [V/A]")
	       (1 "c-" "dV_C2/dC1 [V/F]")
	       (2 "y-" "dV_C2/dC2 [V/F]"))
	     collect
	     `(dot ax (plot t_grid (aref J_val (slice) ,idx) (string ,style)
			    :label (string ,label))))
     (dot ax (set_title (string "Nominale Sensitivitaeten") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Sensitivitaet [V/Einheit]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 5: Heatmap Spannung
     (setf ax (aref axs 1 1))
     (setf contour (dot ax (contourf C1_grid C2_grid V_C2_tf_grid :levels 20 :cmap (string "Blues"))))
     (dot (fig.colorbar contour :ax ax) (set_label (string "V_C2(t_f) [V]") :fontsize 11))
     (setf c1_line (np.linspace 0.1 1.9 100)
	   c2_line (- 2.0 c1_line))
     (dot ax (plot c1_line c2_line (string "r--") :lw 2.0 :label (string "Budget C1+C2=2.0F")))
     (dot ax (plot (list (aref p_nom 1)) (list (aref p_nom 2)) (string "ko") :ms 8 :markeredgecolor (string "white") :label (string "Nominal")))
     (dot ax (plot (list (aref p_opt_V 1)) (list (aref p_opt_V 2)) (string "r*") :ms 12 :markeredgecolor (string "black") :label (string "Optimum (Spannung)")))
     (dot ax (set_title (string "Zielfunktion: Spannung") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "C1 [F]")))
     (dot ax (set_ylabel (string "C2 [F]")))
     (dot ax (legend :fontsize 9))
     (dot ax (grid True :alpha 0.3))

     ;; Panel 6: Heatmap Energie
     (setf ax (aref axs 1 2))
     (setf contour (dot ax (contourf C1_grid C2_grid E_tf_grid :levels 20 :cmap (string "Oranges"))))
     (dot (fig.colorbar contour :ax ax) (set_label (string "E(t_f) [J]") :fontsize 11))
     (dot ax (plot c1_line c2_line (string "r--") :lw 2.0 :label (string "Budget C1+C2=2.0F")))
     (dot ax (plot (list (aref p_nom 1)) (list (aref p_nom 2)) (string "ko") :ms 8 :markeredgecolor (string "white") :label (string "Nominal")))
     (dot ax (plot (list (aref p_opt_E 1)) (list (aref p_opt_E 2)) (string "r*") :ms 12 :markeredgecolor (string "black") :label (string "Optimum (Energie)")))
     (dot ax (set_title (string "Zielfunktion: Gespeicherte Energie") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "C1 [F]")))
     (dot ax (set_ylabel (string "C2 [F]")))
     (dot ax (legend :fontsize 9))
     (dot ax (grid True :alpha 0.3))

     (plt.suptitle (string "Design-Tradeoffs im Dioden-Geklemmten Filter: Spannung vs. Energie")
		   :fontsize 16 :fontweight (string "bold") :y 0.98)
     (plt.tight_layout)
     (plt.savefig (string "diode_optima.png") :dpi 150)
     (print (string "Plot gespeichert: diode_optima.png"))
     )))
