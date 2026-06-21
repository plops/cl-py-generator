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
	       "    steht, um ein bestimmtes Spannungsziel am Ausgang zu erreichen?'"
	       ""
	       "   Diese Frage kann man NICHT analytisch beantworten, weil die"
	       "   Shockley-Diodengleichung (exponentiell) zu einem nichtlinearen DAE"
	       "   fuehrt. Stattdessen nutzen wir hier CasADi, um:"
	       "   (1) Das DAE symbolisch zu formulieren und mit dem IDAS-Solver zu loesen."
	       "   (2) Automatische Ableitungen (Sensitivitaeten) durch den Solver zu"
	       "       propagieren, um zu verstehen, wie empfindlich die Ausgangsspannung"
	       "       auf Aenderungen von I_in, C1 und C2 reagiert."
	       "   (3) Mit IPOPT die optimale Kapazitaetsaufteilung zu finden."
	       ""
	       "   HINWEIS: Wir maximieren hier V_C2(t_f) als Demonstrationsbeispiel"
	       "   fuer DAE-gestuetzte Optimierung. In der Praxis koennte das Ziel"
	       "   auch anders lauten (z.B. Minimierung der Einschwingzeit oder"
	       "   Maximierung der Bandbreite). Die Methodik bleibt identisch."
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
	       "     I_C1 = C1 * dV_C1/dt   (Kondensatorladestrom)"
	       "     I_R  = (V_C1 - V_C2)/R (Strom durch den Kopplungswiderstand)"
	       "     I_D1 = Is1*(exp(V_C1/Vt1) - 1)  (Shockley-Diodenstrom)"
	       ""
	       "   Stromfluss an Knoten 2 (KCL): I_R = I_C2 + I_D2"
	       "     I_C2 = C2 * dV_C2/dt"
	       "     I_D2 = Is2*(exp(V_C2/Vt2) - 1)"
	       "========================================================================")

     (comments "========================================================================"
	       "SYMBOLISCHE DAE-DEFINITION"
	       "========================================================================")
     ;; CasADi-SX: Skalare symbolische Ausdruecke (duenn, effizient fuer AD)
     (setf x (ca.SX.sym (string "x") 2) ; Differenzielle Zustaende: Knotenspannungen [V]
	   z (ca.SX.sym (string "z") 2) ; Algebraische Zustaende: Diodenstroeme [A]
	   p (ca.SX.sym (string "p") 3)) ; Parameter: Ladestrom [A] und Kapazitaeten [F]

     ;; Zustaende und Parameter an benannte Variablen binden
     (setf V_C1 (aref x 0) ; Spannung am Kondensator der 1. Stufe [V]
	   V_C2 (aref x 1) ; Spannung am Kondensator der 2. Stufe [V] (= Ausgang)
	   I_D1 (aref z 0) ; Strom durch Klemmdiode D1 [A]
	   I_D2 (aref z 1) ; Strom durch Klemmdiode D2 [A]
	   I_in (aref p 0) ; Eingangsladestrom der Stromquelle [A]
	   C1   (aref p 1) ; Kapazitaet der 1. Filterstufe [F]
	   C2   (aref p 2)) ; Kapazitaet der 2. Filterstufe [F]

     ;; Bauelementekonstanten (hier fest; koennten auch Parameter sein)
     (setf R 2.0    ; Kopplungswiderstand zwischen Stufe 1 und 2 [Ohm]
	   Is1 0.1  ; Sperrstrom (Saettigungsstrom) der Diode D1 [A]
	   Is2 0.1  ; Sperrstrom (Saettigungsstrom) der Diode D2 [A]
	   Vt1 0.5  ; Thermische Spannung der Diode D1 [V] (bestimmt Steilheit der Kennlinie)
	   Vt2 0.5) ; Thermische Spannung der Diode D2 [V] (bei Raumtemperatur ca. 26 mV)

     ;; --- ODE: Zeitableitungen der Kondensatorspannungen nach KCL ---
     ;; dV_C1/dt = (I_in - I_D1 - (V_C1 - V_C2)/R) / C1
     ;;   Erklaerung: Der Gesamtstrom in Knoten 1 teilt sich in drei Pfade:
     ;;   Diode D1 (I_D1), Widerstand R zum Knoten 2, und Kondensator C1.
     ;;   Der Rest (was C1 bekommt) bestimmt die Spannungsaenderung.
     ;;
     ;; dV_C2/dt = ((V_C1 - V_C2)/R - I_D2) / C2
     ;;   Erklaerung: Der Strom durch R fliesst in Knoten 2 hinein.
     ;;   Davon geht I_D2 durch Diode D2 ab. Der Rest laedt C2 auf.
     (setf ode (ca.vertcat (/ (- I_in I_D1 (/ (- V_C1 V_C2) R)) C1)
			   (/ (- (/ (- V_C1 V_C2) R) I_D2) C2)))

     ;; --- ALG: Algebraische Nebenbedingungen (implizit, = 0) ---
     ;; Die Diodenstroeme I_D1 und I_D2 sind keine Differentialgroessen,
     ;; sondern werden zu jedem Zeitpunkt durch die Shockley-Gleichung
     ;; implizit bestimmt:
     ;;   0 = I_D1 - Is1 * (exp(V_C1/Vt1) - 1)
     ;;   0 = I_D2 - Is2 * (exp(V_C2/Vt2) - 1)
     ;; Diese exponentielle Nichtlinearitaet macht das System zu einem DAE
     ;; (nicht zu einer reinen ODE), weil die algebraischen Zustaende z
     ;; nicht durch Integration, sondern durch Newton-Iteration bestimmt werden.
     (setf alg (ca.vertcat (- I_D1 (* Is1 (- (ca.exp (/ V_C1 Vt1)) 1.0)))
			   (- I_D2 (* Is2 (- (ca.exp (/ V_C2 Vt2)) 1.0)))))

     ;; DAE-Dictionary fuer CasADi zusammenbauen
     (setf dae (dict ((string "x") x)
		     ((string "z") z)
		     ((string "p") p)
		     ((string "ode") ode)
		     ((string "alg") alg)))

     (comments "========================================================================"
	       "INTEGRATOR-ERSTELLUNG (IDAS-PLUGIN)"
	       "========================================================================"
	       "CasADi bietet zwei wesentliche DAE-Integratoren aus dem SUNDIALS-Paket:"
	       "  'cvodes' - Fuer reine ODEs (ohne algebraische Zustaende z)."
	       "  'idas'   - Fuer DAE-Systeme (mit algebraischen Zustaenden z)."
	       "Da wir hier Diodenstroeme als algebraische Variable haben, MUESSEN"
	       "wir 'idas' verwenden."
	       ""
	       "Wir erstellen ZWEI Integratoren mit unterschiedlichem Zeithorizont:"
	       ""
	       "F_tf: Integriert nur bis zur Endzeit tf = 1.0 s."
	       "  Vorteil: Liefert nur den Endwert xf. Ideal als Baustein fuer"
	       "  Optimierung (Opti/IPOPT) und Hessian-Berechnung, weil CasADi"
	       "  keine unnoetig grossen Zwischenergebnisse speichern muss."
	       ""
	       "F_grid: Integriert auf einem Zeitgitter mit 100 aequidistanten Punkten"
	       "  von t0=0.0 bis tf=1.0 Sekunden."
	       "  Vorteil: Liefert den Zustandsverlauf x(t) an JEDEM Gitterpunkt."
	       "  Das brauchen wir fuer:"
	       "  - Plotten der Spannungs-/Stromtrajektorien ueber die Zeit."
	       "  - Berechnung der Sensitivitaet dV_C2(t)/dp an jedem Zeitpunkt,"
	       "    um zu sehen, WANN im Zeitverlauf welcher Parameter den groessten"
	       "    Einfluss hat (z.B. frueh vs. spaet in der Ladeperiode).")

     (setf t0 0.0    ; Startzeitpunkt [s]
	   t_grid (np.linspace 0.0 1.0 100)) ; 100 Gitterpunkte von 0 bis 1 [s]
     (setf F_tf (ca.integrator (string "F_tf") (string "idas") dae 0.0 1.0)
	   F_grid (ca.integrator (string "F_grid") (string "idas") dae t0 t_grid))

     (comments "========================================================================"
	       "SENSITIVITAETSANALYSE: GRADIENT UND HESSIAN"
	       "========================================================================"
	       "Sensitivitaetsanalyse beantwortet die Frage:"
	       "  'Wie aendert sich die Ausgangsspannung V_C2(t_f), wenn ich einen"
	       "   Parameter (z.B. C1) um einen kleinen Betrag dp veraendere?'"
	       ""
	       "1. Ordnung (Gradient): dV_C2/dp = [dV_C2/dI_in, dV_C2/dC1, dV_C2/dC2]"
	       "   Gibt die Richtung und Staerke des linearen Einflusses jedes Parameters."
	       ""
	       "2. Ordnung (Hessian): d^2 V_C2 / (dp_i * dp_j)"
	       "   Gibt die Kruemmung der Zielfunktion an. Die Hessian-Matrix zeigt:"
	       "   - Diagonale: Wie stark ein Parameter nichtlinear wirkt."
	       "   - Nebendiagonale: Wie stark zwei Parameter GEKOPPELT sind"
	       "     (d.h. ob die Wirkung von C1 davon abhaengt, welchen Wert C2 hat)."
	       ""
	       "Wir berechnen den Hessian auf zwei Arten, um die Korrektheit zu pruefen:")

     ;; Symbolischen Aufruf des Integrators einbetten
     (setf res_tf (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p)
	   V_C2_tf (aref (aref res_tf (string "xf")) 1))

     ;; Gradient: Jacobi-Matrix (1x3) transponiert ergibt Spaltenvektor (3x1)
     (setf grad_V_C2 (dot (ca.jacobian V_C2_tf p) T))

     (comments "Modus 1: Adjoint-over-Adjoint (AOA) -- ca.hessian()"
	       "  CasADi berechnet zuerst den Gradienten durch Rueckwaerts-AD (adjoint),"
	       "  dann leitet es den Gradienten NOCHMALS rueckwaerts ab."
	       "  Vorteil: Effizient fuer skalare Zielfunktionen mit vielen Parametern,"
	       "    weil die rueckwaerts-Propagation unabhaengig von der Parameterzahl ist."
	       "  Nachteil: Erfordert die Loesung eines adjungierten DAE-Systems (IDAS"
	       "    integriert das Originalproblem + ein adjungiertes System rueckwaerts"
	       "    in der Zeit), was numerisch anspruchsvoller sein kann.")
     (setf (ntuple H_aoa g_aoa) (ca.hessian V_C2_tf p))
     (setf f_aoa (ca.Function (string "f_aoa") (list p) (list H_aoa g_aoa)))

     (comments "Modus 2: Forward-over-Adjoint (FOA) -- jacobian(gradient, p)"
	       "  Zuerst berechnen wir den Gradienten durch Rueckwaerts-AD (adjoint),"
	       "  dann leiten wir den Gradienten-Vektor VORWAERTS ab (jacobian)."
	       "  Vorteil: Numerisch robuster, weil der Vorwaerts-Pass weniger"
	       "    empfindlich auf die Anfangswert-Berechnung des adjungierten Systems ist."
	       "  Nachteil: Kosten skalieren mit der Parameterzahl (hier 3, also kein Problem)."
	       ""
	       "  Wir vergleichen AOA und FOA, um die numerische Konsistenz zu validieren."
	       "  Bei einem gut konditionierten Problem sollten beide Matrizen bis auf"
	       "  Rundungsfehler (ca. 1e-7) identisch sein.")
     (setf H_foa (ca.jacobian grad_V_C2 p))
     (setf f_foa (ca.Function (string "f_foa") (list p) (list H_foa)))

     (comments "========================================================================"
	       "AUSWERTUNG DER SENSITIVITAETEN AM NOMINALEN ARBEITSPUNKT"
	       "========================================================================")
     (setf p_nom (list 2.0 1.0 1.0)) ; I_in=2A, C1=1F, C2=1F

     ;; Benchmarks: Zeitmessung fuer die Hessian-Auswertung beider Modi
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
	       "DAE-BESCHRAENKTE PARAMETEROPTIMIERUNG MIT OPTI/IPOPT"
	       "========================================================================"
	       "CasADi bietet die 'Opti'-Klasse als komfortable High-Level-Schnittstelle"
	       "zur Formulierung nichtlinearer Optimierungsprobleme (NLP). Sie verwaltet"
	       "automatisch Variablen, Nebenbedingungen und die Solver-Kopplung."
	       ""
	       "Als NLP-Solver verwenden wir IPOPT (Interior Point OPTimizer):"
	       "  - IPOPT nutzt ein Innere-Punkte-Verfahren (Barrier-Methode) und ist"
	       "    der Standardsolver fuer allgemeine, nichtlineare, beschraenkte"
	       "    Optimierungsprobleme mit glatten Zielfunktionen."
	       "  - Alternativen waeren z.B. SNOPT (SQP-Methode, besser fuer duenn"
	       "    besetzte, grosse Probleme) oder KNITRO (kommerziell, sehr robust)."
	       "  - IPOPT benoetigt Gradienten und Hessians, die CasADi automatisch"
	       "    via AD bereitstellt -- genau das, was wir oben validiert haben."
	       ""
	       "Fragestellung: Gegeben ein festes Kapazitaetsbudget (C1+C2 = 2.0 F),"
	       "finde die Aufteilung, die V_C2(t_f=1s) maximiert.")

     ;; Opti-Instanz erstellen: Verwaltet Variablen, Constraints und Solver
     (setf opti (ca.Opti)
	   p_var (opti.variable 3)) ; 3 Optimierungsvariablen: [I_in, C1, C2]

     ;; Nebenbedingungen (Constraints) definieren:
     (opti.subject_to (== (aref p_var 0) 2.0))   ; I_in = 2.0 A festhalten
     (opti.subject_to (== (+ (aref p_var 1)       ; Kapazitaetsbudget:
			     (aref p_var 2)) 2.0)) ;   C1 + C2 = 2.0 F
     (opti.subject_to (>= (aref p_var 1) 0.1))    ; C1 >= 0.1 F (physikalische Untergrenze)
     (opti.subject_to (>= (aref p_var 2) 0.1))    ; C2 >= 0.1 F (physikalische Untergrenze)
     (opti.set_initial p_var p_nom) ; Startschaetzung: Gleichverteilung [2.0, 1.0, 1.0]

     ;; DAE-Integration symbolisch in die Zielfunktion einbetten
     (setf res_opt (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_var)
	   V_C2_tf_opt (aref (aref res_opt (string "xf")) 1))

     ;; Zielfunktion: Minimiere -V_C2(tf) <=> Maximiere V_C2(tf)
     ;; (Opti/IPOPT kennt nur Minimierung, daher Vorzeichenwechsel)
     (opti.minimize (* -1.0 V_C2_tf_opt))

     ;; Solver konfigurieren: IPOPT mit unterdrueckter Konsolenausgabe
     ;; Das erste dict() ist fuer CasADi-Plugin-Optionen (hier leer),
     ;; das zweite fuer IPOPT-Optionen (print_level=0 unterdrueckt Ausgabe).
     (opti.solver (string "ipopt") (dict) (dict ((string "print_level") 0)))

     ;; Optimierung ausfuehren und Ergebnis extrahieren
     (setf t_start (time.time))
     (setf sol (opti.solve)
	   p_opt (sol.value p_var)
	   V_C2_tf_max (sol.value V_C2_tf_opt))
     (setf t_opt (- (time.time) t_start))

     (print (string "--- DAE-beschraenkte Optimierung ---"))
     (print (fstring "Optimale Kapazitaeten: C1 = {p_opt[1]:.4f} F, C2 = {p_opt[2]:.4f} F"))
     (print (fstring "Max V_C2(t_f): {V_C2_tf_max:.4f} V (Nominal: {float(F_tf(x0=[0,0], z0=[0,0], p=p_nom)[\"xf\"][1]):.4f} V)"))
     (print (fstring "Rechenzeit IPOPT: {t_opt*1000:.1f} ms"))

     (comments "========================================================================"
	       "TRAJEKTORIEN-SIMULATION UND ZEITAUFGELOESTE SENSITIVITAETEN"
	       "========================================================================"
	       "Wir simulieren die Zustandsverlaeufe fuer den nominalen und den"
	       "optimalen Parametersatz ueber das gesamte Zeitgitter (100 Punkte).")

     ;; Nominale und optimale Trajektorien simulieren
     (setf t_start (time.time))
     (setf sim_nom (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_nom)
	   sim_opt (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_opt))
     (setf t_sim (- (time.time) t_start))
     (print (fstring "Rechenzeit Trajektorien (2x 100 Gitterpunkte): {t_sim*1000:.1f} ms"))

     (setf xf_nom (np.array (aref sim_nom (string "xf")))
	   zf_nom (np.array (aref sim_nom (string "zf")))
	   xf_opt (np.array (aref sim_opt (string "xf")))
	   zf_opt (np.array (aref sim_opt (string "zf"))))

     (comments "Zeitaufgeloeste Sensitivitaeten: dV_C2(t)/dp fuer jeden Zeitpunkt t"
	       ""
	       "Die Sensitivitaet dV_C2(t)/dC1 sagt uns z.B.:"
	       "  'Wenn ich C1 um 1 F erhoehe, um wieviel Volt aendert sich V_C2"
	       "   zum Zeitpunkt t?'"
	       "Wir berechnen das an allen 100 Gitterpunkten, um zu sehen, ob der"
	       "Parametereinfluss frueh (Einschwingphase) oder spaet (stationaerer"
	       "Zustand) dominant ist. Das hilft dem Ingenieur zu verstehen, in"
	       "welcher Betriebsphase eine Bauteiltoleranz am kritischsten ist.")
     (setf res_grid_sym (F_grid :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p)
	   V_C2_traj_sym (aref (aref res_grid_sym (string "xf")) 1 (slice)))
     ;; J_traj ist eine 100x3-Matrix: pro Zeitpunkt eine Zeile [dV_C2/dI_in, dV_C2/dC1, dV_C2/dC2]
     (setf t_start (time.time))
     (setf J_traj_sym (ca.jacobian V_C2_traj_sym p)
	   J_func (ca.Function (string "J_func") (list p) (list J_traj_sym))
	   J_val (np.array (J_func p_nom)))
     (setf t_jac (- (time.time) t_start))
     (print (fstring "Rechenzeit Sensitivitaets-Trajektorie (100 Punkte, 3 Parameter): {t_jac*1000:.1f} ms"))

     (comments "2D Parametersweep: V_C2(tf) fuer ein 40x40-Gitter ueber C1 und C2"
	       "Erzeugt die Daten fuer eine Konturdarstellung (Heatmap) der Zielfunktion.")
     (setf C1_vals (np.linspace 0.1 3.0 40)
	   C2_vals (np.linspace 0.1 3.0 40))
     (setf (ntuple C1_grid C2_grid) (np.meshgrid C1_vals C2_vals)
	   V_C2_tf_grid (np.zeros_like C1_grid))

     (setf t_start (time.time))
     (for (i (range (len C2_vals)))
	  (for (j (range (len C1_vals)))
	       (setf p_val_ij (list 2.0 (aref C1_grid i j) (aref C2_grid i j)))
	       (setf out (F_tf :x0 (list 0.0 0.0) :z0 (list 0.0 0.0) :p p_val_ij))
	       (setf (aref V_C2_tf_grid i j) (float (aref (aref out (string "xf")) 1)))))
     (setf t_sweep (- (time.time) t_start))
     (print (fstring "Rechenzeit 2D-Sweep (40x40 = 1600 DAE-Loesungen): {t_sweep*1000:.0f} ms"))

     (comments "========================================================================"
	       "VISUALISIERUNG"
	       "========================================================================")
     (setf (ntuple fig axs) (plt.subplots 2 2 :figsize (tuple 14 10)))

     ;; Panel 1: Kondensatorspannungen -- Nominal vs. Optimal
     (setf ax (aref axs 0 0))
     ,@(loop for (data idx style label alpha) in
	     '((xf_nom 0 "b-" "V_C1 (Nominal)" nil)
	       (xf_nom 1 "r-" "V_C2 (Nominal)" nil)
	       (xf_opt 0 "b--" "V_C1 (Optimal)" 0.7)
	       (xf_opt 1 "r--" "V_C2 (Optimal)" 0.7))
	     collect
	     `(dot ax (plot t_grid (aref ,data ,idx (slice)) (string ,style)
			    ,@(when alpha `(:alpha ,alpha))
			    :label (string ,label))))
     (dot ax (set_title (string "Kondensatorspannungen [V] vs Zeit") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Spannung [V]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 2: Diodenstroeme -- Nominal vs. Optimal
     (setf ax (aref axs 0 1))
     ,@(loop for (data idx style label alpha) in
	     '((zf_nom 0 "g-" "I_D1 (Nominal)" nil)
	       (zf_nom 1 "m-" "I_D2 (Nominal)" nil)
	       (zf_opt 0 "g--" "I_D1 (Optimal)" 0.7)
	       (zf_opt 1 "m--" "I_D2 (Optimal)" 0.7))
	     collect
	     `(dot ax (plot t_grid (aref ,data ,idx (slice)) (string ,style)
			    ,@(when alpha `(:alpha ,alpha))
			    :label (string ,label))))
     (dot ax (set_title (string "Diodenstroeme [A] vs Zeit") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Strom [A]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 3: Sensitivitaets-Zeitverlaeufe
     (setf ax (aref axs 1 0))
     ,@(loop for (idx style label) in
	     '((0 "k-" "dV_C2/dI_in [V/A]")
	       (1 "c-" "dV_C2/dC1 [V/F]")
	       (2 "y-" "dV_C2/dC2 [V/F]"))
	     collect
	     `(dot ax (plot t_grid (aref J_val (slice) ,idx) (string ,style)
			    :label (string ,label))))
     (dot ax (set_title (string "Sensitivitaet von V_C2(t) bzgl. Parameter") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "Zeit [s]")))
     (dot ax (set_ylabel (string "Sensitivitaet [V/Einheit]")))
     (dot ax (legend))
     (dot ax (grid True :alpha 0.5))

     ;; Panel 4: Heatmap V_C2(tf) ueber C1/C2 Parameterraum
     (setf ax (aref axs 1 1))
     (setf contour (dot ax (contourf C1_grid C2_grid V_C2_tf_grid :levels 20 :cmap (string "viridis")))
	   cbar (fig.colorbar contour :ax ax))
     (dot cbar (set_label (string "V_C2(t_f) [V]") :fontsize 11))
     (setf c1_line (np.linspace 0.1 1.9 100)
	   c2_line (- 2.0 c1_line))
     (dot ax (plot c1_line c2_line (string "r--") :lw 2.0 :label (string "C1 + C2 = 2.0 F")))
     (dot ax (plot (list (aref p_nom 1)) (list (aref p_nom 2)) (string "wo") :ms 8 :markeredgecolor (string "black") :label (string "Nominalpunkt")))
     (dot ax (plot (list (aref p_opt 1)) (list (aref p_opt 2)) (string "r*") :ms 12 :markeredgecolor (string "black") :label (string "Optimum")))
     (dot ax (set_title (string "V_C2(t_f) vs Kapazitaeten") :fontsize 12 :fontweight (string "bold")))
     (dot ax (set_xlabel (string "C1 [F]")))
     (dot ax (set_ylabel (string "C2 [F]")))
     (dot ax (legend :fontsize 8))
     (dot ax (grid True :alpha 0.3))

     (plt.suptitle (string "2-Stufiger Dioden-Geklemmter Kondensatorfilter: Simulation, Sensitivitaeten & Optimierung")
		   :fontsize 14 :fontweight (string "bold") :y 0.98)
     (plt.tight_layout)
     (plt.savefig (string "diode_sensitivities.png") :dpi 150)
     (print (string "Plot gespeichert: diode_sensitivities.png"))
     )))
