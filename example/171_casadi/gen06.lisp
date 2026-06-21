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
	       "2-STUFIGER DIODEN-GEKLEMMTER KONDENSATORFILTER - MULTI-ZIEL OPTIMIERUNG"
	       "========================================================================"
	       ""
	       "PHYSIKALISCHES MODELL:"
	       "   Zweistufiger passiver Tiefpassfilter mit Klemm-Dioden parallel"
	       "   zu den Kondensatoren. Gespeist von einer Konstantstromquelle I_in."
	       ""
	       "   Schaltplan:"
	       "                   Knoten 1           Knoten 2"
	       "   I_in              V_C1               V_C2"
	       "    o-------+-----------o------[ R ]-------o-----------o (Ausgang)"
	       "            |           |                  |"
	       "            |        ---+---            ---+---"
	       "          [I_in]     | C1  |            | C2  |"
	       "            |        ---+---            ---+---"
	       "            |         D1|                D2|"
	       "            |          / |               /  |"
	       "            |         v  |              v   |"
	       "   ---------+-----------+------------------+-- GND"
	       ""
	       "DESIGNFRAGE:"
	       "   Gegeben ein Kapazitaetsbudget C1+C2=2F und fester I_in=2A:"
	       "   Wie ist C1,C2 aufzuteilen, um ein Entwurfsziel zu erreichen?"
	       ""
	       "DREI ENTWURFSZIELE IM VERGLEICH:"
	       "   1. Spannungs-Optimum:  Maximiere V_C2(t_f)"
	       "      -> Optimum liegt am Rand (C2 minimal). Kein Sweet Spot."
	       "      -> Anwendung: Digitale Logikgatter, hochohmige Eingaenge."
	       ""
	       "   2. Energie-Optimum:   Maximiere E = 0.5*C2*V_C2(t_f)^2"
	       "      -> Optimum liegt am gegenueberligenden Rand (C2 maximal)."
	       "      -> Anwendung: Ideale Energiespeicherung ohne Nebenbedingungen."
	       "      -> Problem: Diode 1 leitet massiv (2W Verlust), zerstoert"
	       "         in der Praxis das Bauteil."
	       ""
	       "   3. Netto-Energie-Optimum (NEU): Maximiere E_netto = E - lambda * E_verlust"
	       "      -> Hier: lambda = 0.060 (Gewichtungsfaktor fuer Thermoverluste)"
	       "      -> E_verlust = Integral der Verlustleistung P_D = V_D * I_D"
	       "      -> Physikalische Bedeutung von lambda:"
	       "         lambda = Wirkungsgrad-Koeffizient. Er beschreibt, wie teuer"
	       "         ein Joule Verlust im Vergleich zu einem Joule gespeicherter"
	       "         Energie ist. Bei lambda=0.060 werden 60 mJ Verlust mit 1 mJ"
	       "         nutzbarer Energie aufgewogen."
	       "      -> DAS ERGEBNIS LIEGT IM INNEREN DES PARAMETERRAUMS!"
	       "         Das ist ein klassischer 'Sweet Spot': Ein zu kleines C2"
	       "         speichert kaum Energie, ein zu grosses C2 verbrennt zu viel"
	       "         Leistung. Der Optimierer findet automatisch den Kompromiss."
	       "      -> Anwendung: IoT-Energieernte, batterielos betriebene Sensoren,"
	       "         bei denen Bauteiltemperatur und Effizienz gleichzeitig"
	       "         beschraenkt werden muessen."
	       ""
	       "TECHNISCHER SCHLUESSEL: CasADi QUADRATURZUSTAND"
	       "   Um E_verlust = Integral(V_C1*I_D1 + V_C2*I_D2, dt) zu berechnen,"
	       "   verwenden wir das 'quad'-Feld im DAE-Dictionary. Dadurch integriert"
	       "   IDAS automatisch einen Quadraturzustand q(t) mit:"
	       "      dq/dt = V_C1*I_D1 + V_C2*I_D2  (Verlustleistung)"
	       "   Das Ergebnis F_tf(...)[qf] ist E_verlust am Integrations-Ende."
	       "   Gradienten durch diesen Quadraturzustand sind vollautomatisch"
	       "   via adjoint-Sensitivitaeten verfuegbar - keine Extraarbeit!"
	       "========================================================================")

     (comments "========================================================================"
	       "SYMBOLISCHE DAE-DEFINITION MIT QUADRATURZUSTAND"
	       "========================================================================")
     (setf x (ca.SX.sym (string "x") 2)   ; Knotenspannungen [V_C1, V_C2]
	   z (ca.SX.sym (string "z") 2)   ; Diodenstroeme [I_D1, I_D2]
	   p (ca.SX.sym (string "p") 3))  ; Parameter [I_in, C1, C2]

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

     ;; Quadraturterm: Verlustleistung in den Dioden
     ;; dq/dt = V_C1*I_D1 + V_C2*I_D2
     ;; F_tf(...)[qf] liefert dann E_verlust = Integral(P_D, 0, tf)
     (setf P_loss_sym (+ (* V_C1 I_D1) (* V_C2 I_D2)))

     ;; DAE mit 'quad'-Feld: IDAS integriert P_loss mit
     (setf dae (dict ((string "x")    x)
		     ((string "z")    z)
		     ((string "p")    p)
		     ((string "ode")  ode)
		     ((string "alg")  alg)
		     ((string "quad") P_loss_sym)))

     (setf t0 0.0
	   t_grid (np.linspace 0.0 1.0 100))
     (setf F_tf   (ca.integrator (string "F_tf")   (string "idas") dae 0.0 1.0)
	   F_grid (ca.integrator (string "F_grid") (string "idas") dae t0 t_grid))

     (comments "========================================================================"
	       "2D PARAMETERSWEEP (60x60) FUER ALLE DREI ZIELFUNKTIONEN"
	       "========================================================================")
     (setf C1_vals (np.linspace 0.1 2.8 60)
	   C2_vals (np.linspace 0.1 2.8 60))
     (setf (ntuple C1_grid C2_grid) (np.meshgrid C1_vals C2_vals)
	   V_C2_tf_grid     (np.zeros_like C1_grid)  ; Spannung am Ende
	   E_stored_grid    (np.zeros_like C1_grid)  ; Gespeicherte Energie 0.5*C2*V^2
	   E_loss_grid      (np.zeros_like C1_grid)  ; Integral der Verlustleistung
	   E_netto_grid     (np.zeros_like C1_grid)) ; Netto-Energie (Zielfunktion 3)

     (setf lam_penalty 0.060) ; Gewichtungsfaktor fuer Thermoverluste

     (setf t_start (time.time))
     (for (i (range (len C2_vals)))
	  (for (j (range (len C1_vals)))
	       (setf p_ij (list 2.0 (aref C1_grid i j) (aref C2_grid i j)))
	       (setf out (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_ij))
	       (setf v_tf    (float (aref (aref out (string "xf")) 1))
		     e_loss  (float (aref out (string "qf"))))
	       (setf e_stored (* 0.5 (aref C2_grid i j) (** v_tf 2)))
	       (setf (aref V_C2_tf_grid  i j) v_tf)
	       (setf (aref E_stored_grid i j) e_stored)
	       (setf (aref E_loss_grid   i j) e_loss)
	       (setf (aref E_netto_grid  i j) (- e_stored (* lam_penalty e_loss)))))
     (setf t_sweep (- (time.time) t_start))
     (print (fstring "Rechenzeit 2D-Sweep (60x60): {t_sweep:.1f} s"))

     (comments "========================================================================"
	       "OPTIMIERUNG 1: SPANNUNGS-MAXIMIERUNG"
	       "========================================================================")
     (setf opti_V (ca.Opti)
	   p_var_V (opti_V.variable 3))
     (opti_V.subject_to (== (aref p_var_V 0) 2.0))
     (opti_V.subject_to (== (+ (aref p_var_V 1) (aref p_var_V 2)) 2.0))
     (opti_V.subject_to (>= (aref p_var_V 1) 0.1))
     (opti_V.subject_to (>= (aref p_var_V 2) 0.1))
     (opti_V.set_initial p_var_V (list 2.0 1.0 1.0))
     (setf res_V   (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var_V)
	   V_C2_V  (aref (aref res_V (string "xf")) 1))
     (opti_V.minimize (* -1.0 V_C2_V))
     (opti_V.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))
     (setf sol_V     (opti_V.solve)
	   p_opt_V   (sol_V.value p_var_V))
     (print (fstring "Optimum Spannung: C1={p_opt_V[1]:.3f} F, C2={p_opt_V[2]:.3f} F, V_C2={-sol_V.value(opti_V.f):.4f} V"))

     (comments "========================================================================"
	       "OPTIMIERUNG 2: ENERGIE-MAXIMIERUNG"
	       "========================================================================")
     (setf opti_E (ca.Opti)
	   p_var_E (opti_E.variable 3))
     (opti_E.subject_to (== (aref p_var_E 0) 2.0))
     (opti_E.subject_to (== (+ (aref p_var_E 1) (aref p_var_E 2)) 2.0))
     (opti_E.subject_to (>= (aref p_var_E 1) 0.1))
     (opti_E.subject_to (>= (aref p_var_E 2) 0.1))
     (opti_E.set_initial p_var_E (list 2.0 1.0 1.0))
     (setf res_E   (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var_E)
	   V_C2_E  (aref (aref res_E (string "xf")) 1)
	   E_obj   (* 0.5 (aref p_var_E 2) (** V_C2_E 2)))
     (opti_E.minimize (* -1.0 E_obj))
     (opti_E.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))
     (setf sol_E     (opti_E.solve)
	   p_opt_E   (sol_E.value p_var_E))
     (print (fstring "Optimum Energie: C1={p_opt_E[1]:.3f} F, C2={p_opt_E[2]:.3f} F, E={-sol_E.value(opti_E.f):.5f} J"))

     (comments "========================================================================"
	       "OPTIMIERUNG 3: NETTO-ENERGIE (SWEET SPOT IM INNEREN!)"
	       "========================================================================"
	       "Zielfunktion: E_netto = E_gespeichert - lambda * E_verlust"
	       "  E_gespeichert = 0.5 * C2 * V_C2(tf)^2"
	       "  E_verlust = Integral(V_C1*I_D1 + V_C2*I_D2, dt)  [qf-Ausgang von F_tf]"
	       "  lambda = 0.060: Abwaegung zwischen Energie-Ernte und Thermoverlust"
	       "Da grosse C2 mehr Energie speichern, aber D1 massiv leitend machen,"
	       "und kleine C2 kaum Energie speichern, muss das Optimum irgendwo"
	       "dazwischen liegen -- nicht am Rand!")
     (setf opti_N (ca.Opti)
	   p_var_N (opti_N.variable 3))
     (opti_N.subject_to (== (aref p_var_N 0) 2.0))
     (opti_N.subject_to (== (+ (aref p_var_N 1) (aref p_var_N 2)) 2.0))
     (opti_N.subject_to (>= (aref p_var_N 1) 0.1))
     (opti_N.subject_to (>= (aref p_var_N 2) 0.1))
     (opti_N.set_initial p_var_N (list 2.0 1.0 1.0))
     (setf res_N      (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var_N)
	   V_C2_N     (aref (aref res_N (string "xf")) 1)
	   E_loss_N   (aref res_N (string "qf"))
	   E_stored_N (* 0.5 (aref p_var_N 2) (** V_C2_N 2))
	   E_netto_N  (- E_stored_N (* lam_penalty E_loss_N)))
     (opti_N.minimize (* -1.0 E_netto_N))
     (opti_N.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))
     (setf sol_N     (opti_N.solve)
	   p_opt_N   (sol_N.value p_var_N))
     (print (fstring "Optimum Netto-Energie: C1={p_opt_N[1]:.3f} F, C2={p_opt_N[2]:.3f} F, E_netto={-sol_N.value(opti_N.f):.5f} J"))

     ;; Verifikation: Ist das Optimum wirklich im Inneren?
     (setf is_interior (and (> (aref p_opt_N 1) 0.11)
			    (> (aref p_opt_N 2) 0.11)
			    (< (aref p_opt_N 1) 1.89)
			    (< (aref p_opt_N 2) 1.89)))
     (if is_interior
	 (print (string "VERIFIKATION: Optimum liegt IM INNEREN des Parameterraums [OK]"))
	 (print (string "VERIFIKATION: Optimum liegt am Rand [NICHT OK - lambda anpassen!]")))

     (comments "========================================================================"
	       "TRAJEKTORIEN FUER ALLE DREI OPTIMA"
	       "========================================================================")
     (setf sim_nom   (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p (list 2.0 1.0 1.0))
	   sim_optV  (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt_V)
	   sim_optE  (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt_E)
	   sim_optN  (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt_N))

     (setf xf_nom  (np.array (aref sim_nom  (string "xf")))
	   zf_nom  (np.array (aref sim_nom  (string "zf")))
	   xf_optV (np.array (aref sim_optV (string "xf")))
	   zf_optV (np.array (aref sim_optV (string "zf")))
	   xf_optE (np.array (aref sim_optE (string "xf")))
	   zf_optE (np.array (aref sim_optE (string "zf")))
	   xf_optN (np.array (aref sim_optN (string "xf")))
	   zf_optN (np.array (aref sim_optN (string "zf"))))

     (setf P_nom  (+ (* (aref xf_nom  0 (slice)) (aref zf_nom  0 (slice)))
		     (* (aref xf_nom  1 (slice)) (aref zf_nom  1 (slice))))
	   P_optV (+ (* (aref xf_optV 0 (slice)) (aref zf_optV 0 (slice)))
		     (* (aref xf_optV 1 (slice)) (aref zf_optV 1 (slice))))
	   P_optE (+ (* (aref xf_optE 0 (slice)) (aref zf_optE 0 (slice)))
		     (* (aref xf_optE 1 (slice)) (aref zf_optE 1 (slice))))
	   P_optN (+ (* (aref xf_optN 0 (slice)) (aref zf_optN 0 (slice)))
		     (* (aref xf_optN 1 (slice)) (aref zf_optN 1 (slice)))))

     (comments "========================================================================"
	       "VISUALISIERUNG: 2x3 PLOT-GRID"
	       "========================================================================")
     (setf (ntuple fig axs) (plt.subplots 2 3 :figsize (tuple 18 11)))

     ;; --- Reihe 1: Zeitverlaeufe ---

     ;; Panel (0,0): Ausgangsspannung V_C2(t)
     (setf ax (aref axs 0 0))
     ,@(loop for (data idx style alpha label) in
	     '((xf_nom  1 "r:"  0.6  "Nominal (C1=1, C2=1)")
	       (xf_optV 1 "b-"  0.9  "Opt. Spannung")
	       (xf_optE 1 "g-"  0.9  "Opt. Energie")
	       (xf_optN 1 "m-"  1.0  "Opt. Netto-Energie (Sweet Spot)"))
	     collect
	     `(dot ax (plot t_grid (aref ,data ,idx (slice)) (string ,style) :alpha ,alpha :label (string ,label))))
     (dot ax (set_title (string "Ausgangsspannung V_C2(t)") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Spannung [V]")))
     (dot ax (legend :fontsize 9))
     (dot ax (grid True :alpha 0.4))

     ;; Panel (0,1): Diodenstrom I_D1 (Verlustindikator)
     (setf ax (aref axs 0 1))
     ,@(loop for (data idx style alpha label) in
	     '((zf_nom  0 "r:"  0.6  "Nominal")
	       (zf_optV 0 "b-"  0.9  "Opt. Spannung")
	       (zf_optE 0 "g-"  0.9  "Opt. Energie")
	       (zf_optN 0 "m-"  1.0  "Opt. Netto-Energie"))
	     collect
	     `(dot ax (plot t_grid (aref ,data ,idx (slice)) (string ,style) :alpha ,alpha :label (string ,label))))
     (dot ax (set_title (string "Verluststrom I_D1 (Diode 1)") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Strom [A]")))
     (dot ax (legend :fontsize 9))
     (dot ax (grid True :alpha 0.4))

     ;; Panel (0,2): Verlustleistung P_loss
     (setf ax (aref axs 0 2))
     (dot ax (plot t_grid P_nom  (string "r:")  :alpha 0.6 :label (string "Nominal")))
     (dot ax (plot t_grid P_optV (string "b-")  :alpha 0.9 :label (string "Opt. Spannung")))
     (dot ax (plot t_grid P_optE (string "g-")  :alpha 0.9 :label (string "Opt. Energie")))
     (dot ax (plot t_grid P_optN (string "m-")  :alpha 1.0 :label (string "Opt. Netto-Energie")))
     (dot ax (fill_between t_grid 0 P_optN :color (string "mediumpurple") :alpha 0.2))
     (dot ax (set_title (string "Verlustleistung P_D = V_D * I_D") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Leistung [W]")))
     (dot ax (legend :fontsize 9))
     (dot ax (grid True :alpha 0.4))

     ;; --- Reihe 2: Parameterraum-Heatmaps ---

     ;; Panel (1,0): Heatmap Spannung
     (setf ax (aref axs 1 0))
     (setf contV (dot ax (contourf C1_grid C2_grid V_C2_tf_grid :levels 25 :cmap (string "Blues"))))
     (dot (fig.colorbar contV :ax ax) (set_label (string "V_C2(t_f) [V]")))
     (setf budget_c1 (np.linspace 0.1 1.9 100))
     (dot ax (plot budget_c1 (- 2.0 budget_c1) (string "w--") :lw 2.0 :label (string "Budget C1+C2=2F")))
     (dot ax (plot (list (aref p_opt_V 1)) (list (aref p_opt_V 2)) (string "r*") :ms 14 :markeredgecolor (string "white") :label (string "Opt. Spannung")))
     (dot ax (set_title (string "Zielfkt. 1: Spannung") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "C1 [F]")))
     (dot ax (set_ylabel (string "C2 [F]")))
     (dot ax (legend :fontsize 9))

     ;; Panel (1,1): Heatmap Netto-Energie (Sweet Spot!)
     (setf ax (aref axs 1 1))
     (setf contN (dot ax (contourf C1_grid C2_grid E_netto_grid :levels 25 :cmap (string "RdYlGn"))))
     (dot (fig.colorbar contN :ax ax) (set_label (string "E_netto [J]")))
     (dot ax (plot budget_c1 (- 2.0 budget_c1) (string "w--") :lw 2.0 :label (string "Budget C1+C2=2F")))
     (dot ax (plot (list (aref p_opt_N 1)) (list (aref p_opt_N 2)) (string "r*") :ms 16 :markeredgecolor (string "black") :label (string "Opt. Netto-Energie [SWEET SPOT]")))
     (dot ax (plot (list (aref p_opt_V 1)) (list (aref p_opt_V 2)) (string "b^") :ms 10 :markeredgecolor (string "white") :label (string "Opt. Spannung (Ref.)")))
     (dot ax (plot (list (aref p_opt_E 1)) (list (aref p_opt_E 2)) (string "gv") :ms 10 :markeredgecolor (string "white") :label (string "Opt. Energie (Ref.)")))
     (dot ax (set_title (string "Zielfkt. 3: E_netto = E - 0.060*E_loss") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "C1 [F]")))
     (dot ax (set_ylabel (string "C2 [F]")))
     (dot ax (legend :fontsize 8))

     ;; Panel (1,2): E_netto auf der Budget-Linie (1D Querschnitt)
     (setf ax (aref axs 1 2))
     (setf c1_scan (np.linspace 0.1 1.9 80)
	   c2_scan (- 2.0 c1_scan)
	   e_netto_scan (np.zeros_like c1_scan)
	   e_stored_scan (np.zeros_like c1_scan)
	   e_loss_scan  (np.zeros_like c1_scan))
     (for (k (range (len c1_scan)))
	  (setf out_k (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0)
			    :p (list 2.0 (aref c1_scan k) (aref c2_scan k))))
	  (setf v_k  (float (aref (aref out_k (string "xf")) 1))
		el_k (float (aref out_k (string "qf"))))
	  (setf es_k (* 0.5 (aref c2_scan k) (** v_k 2)))
	  (setf (aref e_netto_scan k)  (- es_k (* lam_penalty el_k)))
	  (setf (aref e_stored_scan k) es_k)
	  (setf (aref e_loss_scan k)   (* lam_penalty el_k)))
     (dot ax (plot c1_scan e_stored_scan (string "g-") :label (string "E_gespeichert")))
     (dot ax (plot c1_scan (* -1.0 e_loss_scan) (string "r-") :label (string "-lambda*E_verlust")))
     (dot ax (plot c1_scan e_netto_scan (string "m-") :lw 2.5 :label (string "E_netto (Summe)")))
     (dot ax (axvline (aref p_opt_N 1) :color (string "black") :ls (string "--") :lw 1.5 :label (string "Optimum")))
     (dot ax (set_title (string "E_netto auf Budget-Linie C1+C2=2F") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "C1 [F]   (C2 = 2 - C1)")))
     (dot ax (set_ylabel (string "Energie [J]")))
     (dot ax (legend :fontsize 9))
     (dot ax (grid True :alpha 0.4))

     (plt.suptitle (string "Sweet Spot im Inneren: Netto-Energie-Optimierung unter Thermoverlust-Penalty")
		   :fontsize 15 :fontweight (string "bold") :y 0.99)
     (plt.tight_layout)
     (plt.savefig (string "diode_sweetspot.png") :dpi 150)
     (print (string "Plot gespeichert: diode_sweetspot.png"))
     )))
