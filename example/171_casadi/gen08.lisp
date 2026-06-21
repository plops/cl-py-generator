(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator))

(in-package #:g)

;; ============================================================================
;; LISP-HILFSFUNKTIONEN (S-EXPRESSION-MACROS / GENERATOREN)
;; ============================================================================

;; Hilfsfunktion zur Generierung der Zustandskosten für einen Zeitschritt (Quadratic Stage Cost)
;; Parameter:
;; - xk: Zustandsvektor-Symbol
;; - uk: Stellgrößen-Symbol
;; - q1: Gewichtung für Karosserieeinfederweg (1/m^2)
;; - q2: Gewichtung für Aufbaugeschwindigkeit (1/(m/s)^2)
;; - q3: Gewichtung für Reifeneinfederweg (1/m^2)
;; - q4: Gewichtung für Radgeschwindigkeit (1/(m/s)^2)
;; - r:  Gewichtung für Stellkraft des Aktuators (1/N^2)
(defun stage-cost (xk uk q1 q2 q3 q4 r)
  `(+ (* ,q1 (** (aref ,xk 0) 2))
      (* ,q2 (** (aref ,xk 1) 2))
      (* ,q3 (** (aref ,xk 2) 2))
      (* ,q4 (** (aref ,xk 3) 2))
      (* ,r (** ,uk 2))))

;; Hilfsfunktion zur Formulierung der diskreten Systemgleichungen
;; Berechnet das Residuum: x_{k+1} - (A_d * x_k + B_d * u_k + G_d * v_k) = 0
;; Die genaue Herleitung der diskreten Systemmatrizen A, B, G erfolgt unten 
;; über das Matrix-Exponential (ZOH-Diskretisierung).
(defun discrete-dynamics (x-next x-curr u-curr v-curr A B G)
  `(- ,x-next (+ (@ ,A ,x-curr) (@ ,B ,u-curr) (* ,G ,v-curr))))

(progn
  (defparameter *source* "example/171_casadi/")

  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p08_active_suspension"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot)))
	  (import-from scipy.linalg expm))

     (comments
      "========================================================================"
      "AKTIVE FAHRWERKSREGELUNG MITTELS MODELLPRÄDIKTIVER REGELUNG (MPC)"
      "========================================================================"
      "Thema: Aktive Fahrwerksregelung fuer ein Viertelfahrzeug-Modell (Quarter-Car)"
      ""
      "Modellbeschreibung:"
      "Es handelt sich hierbei um ein Viertelfahrzeug-Modell (Quarter-Car Model)."
      "Dieses Modell repraesentiert genau ein Rad und die dazugehoerige Ecke der"
      "Karosserie. Es wird die Annahme getroffen, dass die Dynamik der vier Ecken"
      "des Fahrzeugs voneinander entkoppelt ist. Dies ist eine Standardannahme"
      "in der Fahrwerkstechnik fuer den Entwurf von Dämpferregelungen."
      ""
      "Zustandsdefinitionen des Vektors x = [x1, x2, x3, x4]^T:"
      "  x1: zs - zu  -> Einfederweg der Karosserie relativ zum Rad (m)."
      "                  Grenze: +/- 0.08 m (Aufhaengungsanschlag)."
      "  x2: dot_zs   -> Vertikalgeschwindigkeit der gefederten Masse (Chassis) (m/s)."
      "  x3: zu - zr  -> Einfederweg des Reifens relativ zur Strasse (m)."
      "                  Reifeneinfederung ist proportional zur Radlast."
      "  x4: dot_zu   -> Vertikalgeschwindigkeit der ungefederten Masse (Rad) (m/s)."
      ""
      "Eingangsdefinitionen:"
      "  u:  Aktive Kraft des Dämpfer-Aktuators (N). Grenze: +/- 1500 N."
      "  vr: Strassengeschwindigkeitsstoerung dot_zr (m/s) (Differentieller Strasseneinfluss)."
      ""
      "Vorschau der Strassenstoerung (Look-Ahead Preview):"
      "Modellpraediktive Regelung (MPC) erlaubt es, zukuenftige Strassenstoerungen"
      "im Voraus zu beruecksichtigen. In einem realen Fahrzeug wird dieses"
      "Strassenprofil durch LiDAR-Sensoren oder Kameras vor dem Fahrzeug erfasst,"
      "die die Fahrbahn abtasten. In dieser Simulation simulieren wir diese"
      "Vorschau (Look-Ahead), indem wir das Strassenprofil ueber den gesamten"
      "Vorhersagehorizont an den Controller uebergeben (Perfect Preview)."
      ""
      "Vorteil des QP-Lösers (Quadratic Programming):"
      "1. Einhaltung von Grenzen: Der Stellweg des Daempfers (+/- 8cm) und die"
      "   maximale Aktuatorkraft (+/- 1500N) werden als harte Grenzen"
      "   beruecksichtigt. LQR- oder PID-Regler koennen diese nicht garantieren."
      "2. Praediktion: Der Regler reagiert bereits *bevor* die Schwelle erreicht"
      "   wird, um die Beschleunigung weich zu daempfen."
      "3. Numerische Stabilitaet: Wir verwenden die exakte Matrix-Exponential-"
      "   Diskretisierung (expm) anstelle der expliziten Euler-Methode, um"
      "   Instabilitaeten bei der schnellen Raddynamik zu vermeiden."
      "========================================================================")

     ;; Physikalische Modellparameter
     (setf ms 300.0   ;; Masse des Chassis / Karosserie (kg)
	   mu 40.0    ;; Masse der Radbaugruppe / Radträger (kg)
	   ks 15000.0 ;; Steifigkeit der Aufhaengungsfeder (N/m)
	   cs 1000.0  ;; Daempfungskoeffizient des passiven Daempfers (N-s/m)
	   kt 150000.0 ;; Steifigkeit des Reifens (N/m)
	   )

     (setf dt 0.01   ;; Diskretisierungsschrittweite (s)
	   N 30      ;; Vorhersagehorizont in Schritten (entspricht 0.3 Sekunden)
	   )

     (comments "--- Physikalische Systemmatrizen (Kontinuierlich) ---"
	       "Die Matrizen A_np, B_np und G_np definieren das kontinuierliche System:"
	       "dot_x = A_c * x + B_c * u + G_c * vr"
	       "Da der Reifen sehr steif ist (hohe Eigenfrequenz), fuehrt eine einfache"
	       "Euler-Diskretisierung zu numerischer Instabilitaet. Daher diskretisieren"
	       "wir das System exakt mit dem Matrix-Exponential (ZOH-Diskretisierung).")
     (setf A_np (np.array (list (list 0.0 1.0 0.0 -1.0)
				(list (/ (- ks) ms) (/ (- cs) ms) 0.0 (/ cs ms))
				(list 0.0 0.0 0.0 1.0)
				(list (/ ks mu) (/ cs mu) (/ (- kt) mu) (/ (- cs) mu))))
	   B_np (np.array (list 0.0 (/ 1.0 ms) 0.0 (/ -1.0 mu)))
	   G_np (np.array (list 0.0 0.0 -1.0 0.0)))

     (comments "--- Exakte ZOH-Diskretisierung via Matrix-Exponential ---"
	       "Wir erstellen ein erweitertes System M, um A_d, B_d und G_d gleichzeitig exakt zu loesen.")
     (setf M (np.zeros (tuple 6 6)))
     (setf (aref M (slice 0 4) (slice 0 4)) A_np)
     (setf (aref M (slice 0 4) 4) B_np)
     (setf (aref M (slice 0 4) 5) G_np)

     ;; Diskrete Matrizen berechnen
     (setf M_disc (expm (* M dt))
	   A_d_np (aref M_disc (slice 0 4) (slice 0 4))
	   B_d_np (aref M_disc (slice 0 4) 4)
	   G_d_np (aref M_disc (slice 0 4) 5))

     ;; Uebertrage in CasADi Double Matrices (DM) fuer die symbolische Optimierung
     (setf A (DM A_d_np)
	   B (DM B_d_np)
	   G (DM G_d_np))

     (comments "--- MPC Gewichtungsfaktoren ---"
	       "q1: Strafe fuer Auslenkung der Aufhaengung (Einheit: 1/m^2)."
	       "    Verhindert das Durchschlagen auf den Endanschlag (Stroke Limit)."
	       "q2: Strafe fuer die vertikale Geschwindigkeit des Chassis (Einheit: 1/(m/s)^2)."
	       "    Dies ist das primaere Mass fuer den Fahrkomfort (Reduktion von Aufbaubeschleunigungen)."
	       "q3: Strafe fuer die Reifenauslenkung (Einheit: 1/m^2)."
	       "    Stellt den Strassenkontakt und damit die Fahrsicherheit sicher."
	       "q4: Strafe fuer die Radgeschwindigkeit (Einheit: 1/(m/s)^2)."
	       "    Daempft schnelle Radbewegungen."
	       "r:  Strafe fuer die Stellkraft des aktiven Daempfers (Einheit: 1/N^2)."
	       "    Begrenzt den Energieaufwand des Aktuators.")
     (setf q1 100000.0
	   q2 5000.0
	   q3 100000.0
	   q4 10.0
	   r  0.01
	   )

     ;; Terminale Gewichtungen am Horizontende (Schritt N) - hoeher gewichtet zur Stabilisierung
     (setf q1_N (* 10.0 q1)
	   q2_N (* 10.0 q2)
	   q3_N (* 10.0 q3)
	   q4_N (* 10.0 q4))

     (comments "--- Symbolische QP-Formulierung in CasADi ---")
     ;; Zustaende X = [x_0, x_1, ..., x_N] und Stellgroessen U = [u_0, u_1, ..., u_{N-1}]
     (setf X (list ,@(loop for i from 0 to 30 collect `(SX.sym (string ,(format nil "x_~d" i)) 4)))
	   U (list ,@(loop for i from 0 below 30 collect `(SX.sym (string ,(format nil "u_~d" i)) 1))))

     ;; Optimierungsparameter: Startzustand x_0 (4 Komponenten) + Stoerungsgeschwindigkeit vr fuer Horizont (N Komponenten)
     (setf p (SX.sym (string "p") (+ 4 N))
	   x_init (aref p (slice 0 4))
	   V_r (aref p (slice 4 nil)))

     (setf f 0.0
	   g (list))

     ;; Aufbau des quadratischen Guetekriteriums (Kosten) und der Gleichheitsnebenbedingungen (Dynamik)
     ,@(loop for k from 0 below 30 collect
	     `(do0
	       ;; Addiere quadratische Zustandskosten fuer diesen Zeitschritt k
	       (setf f (+ f ,(stage-cost `(aref X ,k) `(aref U ,k) 'q1 'q2 'q3 'q4 'r)))
	       ;; Erzwinge die exakte diskrete Systemdynamik: x_{k+1} - (A_d*x_k + B_d*u_k + G_d*v_k) = 0
	       (dot g (append ,(discrete-dynamics `(aref X ,(1+ k)) `(aref X ,k) `(aref U ,k) `(aref V_r ,k) 'A 'B 'G)))))

     ;; Terminale Kosten (Endzustand am Schritt N) fuer mathematische MPC-Stabilitaet
     (setf f (+ f (* q1_N (** (aref (aref X N) 0) 2))
		  (* q2_N (** (aref (aref X N) 1) 2))
		  (* q3_N (** (aref (aref X N) 2) 2))
		  (* q4_N (** (aref (aref X N) 3) 2))))

     ;; Anfangszustand-Gleichheitsbedingung: x_0 - x_init = 0
     (dot g (insert 0 (- (aref X 0) x_init)))

     ;; Zusammenstellen der QP-Struktur
     ;; 'x' enthaelt die Entscheidungsvariablen (Zustaende X und Stellgroessen U concatenated)
     ;; 'p' enthaelt die Parameter (Anfangszustand und Stoerungsverlauf)
     ;; 'f' enthaelt die Kostenfunktion (obj)
     ;; 'g' enthaelt die Nebenbedingungen (Systemdynamik-Residuen, die Null sein muessen)
     (setf qp (dict ((string "x") (vertcat (space * (paren (+ X U)))))
		    ((string "p") p)
		    ((string "f") f)
		    ((string "g") (vertcat (space * g)))))

     ;; Instanziieren des QP-Solvers. qpOASES loest das quadratische Programm extrem schnell.
     (setf S (qpsol (string "S") (string "qpoases") qp (dict ((string "printLevel") (string "none")))))

     (comments "--- Simulations-Setup ---")
     (setf sim_time 3.0
	   N_steps (int (/ sim_time dt))
	   t_vec (np.linspace 0.0 sim_time N_steps))

     (comments "Definition des Strassenprofils: Eine 5cm hohe Bodenschwelle von t=0.5s bis t=0.7s")
     (setf z_r_vec (np.zeros N_steps)
	   v_r_vec (np.zeros N_steps))

     (for (j (range N_steps))
	  (setf t_curr (aref t_vec j))
	  (if (and (<= 0.5 t_curr) (<= t_curr 0.7))
	      (do0
	       ;; Sinus-Foermige Bodenschwelle fuer sanften Uebergang
	       (setf (aref z_r_vec j) (* 0.025 (- 1.0 (np.cos (/ (* 2.0 np.pi (- t_curr 0.5)) 0.2)))))
	       ;; Die Ableitung (Geschwindigkeit) ist die Stoerung vr fuer das System
	       (setf (aref v_r_vec j) (* (* 0.025 (/ (* 2.0 np.pi) 0.2)) (np.sin (/ (* 2.0 np.pi (- t_curr 0.5)) 0.2)))))
	      (do0
	       (setf (aref z_r_vec j) 0.0)
	       (setf (aref v_r_vec j) 0.0))))

     (comments "Initialisierung der Messhistorie fuer das aktive (MPC) und das passive Fahrwerk")
     (setf x_hist_mpc (np.zeros (tuple 4 N_steps))
	   u_hist_mpc (np.zeros N_steps)
	   x_hist_passive (np.zeros (tuple 4 N_steps)))

     ;; Beide Systeme starten in der Ruhelage (Nullzustand)
     (setf x_curr (np.array (list 0.0 0.0 0.0 0.0))
	   x_curr_passive (np.array (list 0.0 0.0 0.0 0.0)))

     (comments "--- Erklaerung der Solver-Parameter ---"
	       "lbx / ubx: Untere und obere Grenzen fuer die Optimierungsvariablen."
	       "           Diese enthalten die zulaessigen Zustaende (z.B. Federweg) und Stellkraefte."
	       "lbg / ubg: Grenzen fuer Nebenbedingungen. Da dies Gleichheitsnebenbedingungen"
	       "           (Systemdynamik und Anfangszustand) sind, sind lbg/ubg beide 0."
	       "x0_guess:  Startschätzung fuer die Entscheidungsvariablen des Solvers."
	       "p_val:     Parameterwerte fuer die Optimierung (aktueller Zustand x_0 und"
	       "           Strassenstoerungs-Vorschau V_r fuer den Horizont).")
     (setf lb_state (np.array (list -0.08 -10.0 -0.05 -20.0))
	   ub_state (np.array (list 0.08 10.0 0.05 20.0))
	   lb_input (np.array (list -1500.0))
	   ub_input (np.array (list 1500.0)))

     ;; Grenzen fuer alle Entscheidungsvariablen (X_0, ..., X_N, U_0, ..., U_{N-1}) assemblieren
     (setf lbx (np.concatenate (tuple (np.tile lb_state (+ N 1)) (np.tile lb_input N)))
	   ubx (np.concatenate (tuple (np.tile ub_state (+ N 1)) (np.tile ub_input N))))

     ;; Gleichheitsbedingungen erzwingen, indem lbg und ubg auf 0 gesetzt werden
     (setf lbg (np.zeros (* 4 (+ N 1)))
	   ubg (np.zeros (* 4 (+ N 1))))

     (setf x0_guess (np.zeros (+ (* 4 (+ N 1)) N)))

     (comments "Simulations-Hauptschleife")
     (for (j (range (- N_steps 1)))
	  ;; Sichere aktuellen Zustand
	  (setf (aref x_hist_mpc (slice nil nil) j) x_curr)
	  (setf (aref x_hist_passive (slice nil nil) j) x_curr_passive)

	  ;; Extrahiere Fahrbahnstoerungsvorschau fuer den Horizont N
	  (setf V_r_horiz (np.zeros N))
	  (for (k (range N))
	       (setf idx (+ j k))
	       (if (< idx N_steps)
		   (setf (aref V_r_horiz k) (aref v_r_vec idx))
		   (setf (aref V_r_horiz k) 0.0)))

	  ;; Uebergebe aktuellen Zustand und Stoerungsvorschau an den Parametervektor p
	  (setf p_val (np.concatenate (tuple x_curr V_r_horiz)))

	  ;; Loese das QP fuer den aktuellen Schritt
	  (setf sol (S :x0 x0_guess :p p_val :lbx lbx :ubx ubx :lbg lbg :ubg ubg)
		x_opt (aref sol (string "x")))

	  ;; Stellkraft u_0 (erste Stellkraft des optimalen Trajektorienplans) befindet sich ab Index 4*(N+1)
	  (setf u_opt (float (aref x_opt (* 4 (+ N 1)))))
	  (setf (aref u_hist_mpc j) u_opt)

	  ;; Schrittweise Zustandsaktualisierung ueber die diskreten Systemmatrizen (ZOH)
	  (setf x_curr (+ (@ A_d_np x_curr) (* B_d_np u_opt) (* G_d_np (aref v_r_vec j))))
	  (setf x_curr_passive (+ (@ A_d_np x_curr_passive) (* G_d_np (aref v_r_vec j)))))

     ;; Letzten Schritt der Zustandstrajektorie sichern
     (setf (aref x_hist_mpc (slice nil nil) -1) x_curr)
     (setf (aref x_hist_passive (slice nil nil) -1) x_curr_passive)

     (comments "--- Nachbereitung und Rekonstruktion physikalischer Messgroessen ---")
     ;; Berechnung der Karosseriebeschleunigung (Ride Comfort Metric)
     ;; dot_x2 = a_chassis = (-ks/ms)*x1 - (cs/ms)*(x2 - x4) + (1/ms)*u
     (setf acc_mpc (+ (* (/ (- ks) ms) (aref x_hist_mpc 0))
		      (* (/ (- cs) ms) (- (aref x_hist_mpc 1) (aref x_hist_mpc 3)))
		      (* (/ 1.0 ms) u_hist_mpc))
	   acc_passive (+ (* (/ (- ks) ms) (aref x_hist_passive 0))
			  (* (/ (- cs) ms) (- (aref x_hist_passive 1) (aref x_hist_passive 3)))))

     ;; Absolute Position der Karosserie berechnen: zs = x1 + x3 + zr
     (setf zs_mpc (+ (aref x_hist_mpc 0) (aref x_hist_mpc 2) z_r_vec)
	   zs_passive (+ (aref x_hist_passive 0) (aref x_hist_passive 2) z_r_vec))

     (comments "--- Plotting Results ---")
     ;; Global plot styling adjustments for clean layout
     (plt.rcParams.update (dict ((string "font.family") (string "sans-serif"))
				((string "font.sans-serif") (list (string "DejaVu Sans") (string "Arial")))
				((string "axes.edgecolor") (string "#cccccc"))
				((string "axes.linewidth") 0.8)
				((string "grid.color") (string "#eeeeee"))
				((string "grid.linestyle") (string "-"))))

     (setf (ntuple fig axes) (plt.subplots 2 2 :figsize (tuple 14 10)))
     (dot fig (suptitle (string "Active MPC vs. Passive Suspension System Comparison") :fontsize 16 :fontweight (string "bold") :y 0.98))

     ;; Plot 1: Road Profile and Chassis Displacement
     (setf ax1 (aref axes 0 0))
     (dot ax1 (fill_between t_vec 0 z_r_vec :color (string "#e0e0e0") :alpha 0.5 :label (string "Road Profile (Bump)")))
     (dot ax1 (plot t_vec zs_passive :label (string "Passive Chassis") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax1 (plot t_vec zs_mpc :label (string "Active Chassis (MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax1 (set_title (string "Chassis Position (zs)") :fontsize 12 :fontweight (string "bold")))
     (dot ax1 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax1 (set_ylabel (string "Displacement (m)") :fontsize 10))
     (dot ax1 (grid True :alpha 0.6))
     (dot ax1 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax1 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax1 spines) (string "right")) (set_visible False))

     ;; Plot 2: Chassis Acceleration
     (setf ax2 (aref axes 0 1))
     (dot ax2 (axhspan -0.5 0.5 :color (string "#e2f0d9") :alpha 0.6 :label (string "Comfort Zone (ISO 2631)")))
     (dot ax2 (plot t_vec acc_passive :label (string "Passive") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax2 (plot t_vec acc_mpc :label (string "Active (MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax2 (set_title (string "Chassis Vertical Acceleration") :fontsize 12 :fontweight (string "bold")))
     (dot ax2 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax2 (set_ylabel (string "Acceleration (m/s^2)") :fontsize 10))
     (dot ax2 (grid True :alpha 0.6))
     (dot ax2 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax2 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax2 spines) (string "right")) (set_visible False))

     ;; Plot 3: Suspension Deflection (Stroke)
     (setf ax3 (aref axes 1 0))
     (dot ax3 (axhspan -0.08 0.08 :color (string "#f1f3f4") :alpha 0.6 :zorder 0))
     (dot ax3 (plot t_vec (aref x_hist_passive 0) :label (string "Passive") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax3 (plot t_vec (aref x_hist_mpc 0) :label (string "Active (MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax3 (axhline :y 0.08 :color (string "#cc0000") :linestyle (string ":") :lw 1.2 :label (string "Stroke Limit (+/- 8cm)")))
     (dot ax3 (axhline :y -0.08 :color (string "#cc0000") :linestyle (string ":") :lw 1.2))
     (dot ax3 (set_title (string "Suspension Deflection (Travel)") :fontsize 12 :fontweight (string "bold")))
     (dot ax3 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax3 (set_ylabel (string "Deflection (m)") :fontsize 10))
     (dot ax3 (grid True :alpha 0.6))
     (dot ax3 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax3 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax3 spines) (string "right")) (set_visible False))

     ;; Plot 4: Actuator Control Force
     (setf ax4 (aref axes 1 1))
     (dot ax4 (axhspan -1500.0 1500.0 :color (string "#f1f3f4") :alpha 0.6 :zorder 0))
     (dot ax4 (step t_vec u_hist_mpc :label (string "Active Force (MPC)") :color (string "#34a853") :lw 2 :where (string "post")))
     (dot ax4 (axhline :y 1500.0 :color (string "#cc0000") :linestyle (string ":") :lw 1.2 :label (string "Actuator Limit (+/- 1500N)")))
     (dot ax4 (axhline :y -1500.0 :color (string "#cc0000") :linestyle (string ":") :lw 1.2))
     (dot ax4 (set_title (string "Actuator Control Force") :fontsize 12 :fontweight (string "bold")))
     (dot ax4 (set_xlabel (string "Time (s)") :fontsize 10))
     (dot ax4 (set_ylabel (string "Force (N)") :fontsize 10))
     (dot ax4 (grid True :alpha 0.6))
     (dot ax4 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax4 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax4 spines) (string "right")) (set_visible False))

     (plt.tight_layout)
     (plt.savefig (string "active_suspension_mpc.png") :dpi 150)
     (print (string "Plot saved as active_suspension_mpc.png"))
     ;; (plt.show)
     )))
