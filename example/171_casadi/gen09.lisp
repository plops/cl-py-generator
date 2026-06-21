(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator))

(in-package #:g)

;; ============================================================================
;; LISP-HILFSFUNKTIONEN (S-EXPRESSION-MACROS / GENERATOREN)
;; ============================================================================

(defun stage-cost (xk uk q1 q2 q3 q4 r)
  "Berechnet die quadratischen Zustandskosten fuer einen einzelnen Zeitschritt.
Parameter:
- xk: Symbol des Zustandsvektors (4-dimensional)
- uk: Symbol der Stellgroesse (1-dimensional)
- q1: Gewichtung fuer Karosserieeinfederweg (1/m^2)
- q2: Gewichtung fuer Aufbaugeschwindigkeit (1/(m/s)^2)
- q3: Gewichtung fuer Reifeneinfederweg (1/m^2)
- q4: Gewichtung fuer Radgeschwindigkeit (1/(m/s)^2)
- r:  Gewichtung fuer Stellkraft des Aktuators (1/N^2)"
  `(+ (* ,q1 (** (aref ,xk 0) 2))
      (* ,q2 (** (aref ,xk 1) 2))
      (* ,q3 (** (aref ,xk 2) 2))
      (* ,q4 (** (aref ,xk 3) 2))
      (* ,r (** (aref ,uk 0) 2))))

(defun discrete-dynamics (x-next x-curr u-curr v-curr A B G)
  "Berechnet das Residuum der diskreten Systemgleichungen:
  x_{k+1} - (A_d * x_k + B_d * u_k + G_d * v_k) = 0
Parameter:
- x-next: Zustand am naechsten Zeitschritt
- x-curr: Zustand am aktuellen Zeitschritt
- u-curr: Stellgroesse am aktuellen Zeitschritt
- v-curr: Strassenstoerung am aktuellen Zeitschritt
- A: Systemmatrix A_d (diskretisiert)
- B: Eingangsmatrix B_d (diskretisiert)
- G: Stoerungsmatrix G_d (diskretisiert)"
  `(- ,x-next (+ (@ ,A ,x-curr) (@ ,B ,u-curr) (* ,G ,v-curr))))

(progn
  (defparameter *source* "example/171_casadi/")

  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p09_active_suspension"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot)))
	  (import-from scipy.linalg expm))

     (comments
      "========================================================================"
      "AKTIVE FAHRWERKSREGELUNG MITTELS KOMPAKTER MAP-MPC (GEN09)"
      "========================================================================"
      "Thema: Aktive Fahrwerksregelung fuer ein Viertelfahrzeug-Modell (Quarter-Car)"
      "Dieses Skript implementiert die MPC-Formulierung effizient unter Verwendung"
      "von CasADi's map() und mapaccum() Funktions-Objekten anstelle von klassischen"
      "For-Schleifen. Dies reduziert die Groesse des Ausdrucksgraphen von O(N) auf O(1)"
      "und beschleunigt die Initialisierungszeit erheblich."
      ""
      "Modellbeschreibung:"
      "Viertelfahrzeug-Modell (Quarter-Car Model) zur Entkopplung der Ecken."
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

     (comments "--- Physikalische Systemmatrizen (Kontinuierlich) und Diskretisierung ---")
     (setf A_np (np.array (list (list 0.0 1.0 0.0 -1.0)
				(list (/ (- ks) ms) (/ (- cs) ms) 0.0 (/ cs ms))
				(list 0.0 0.0 0.0 1.0)
				(list (/ ks mu) (/ cs mu) (/ (- kt) mu) (/ (- cs) mu))))
	   B_np (np.array (list 0.0 (/ 1.0 ms) 0.0 (/ -1.0 mu)))
	   G_np (np.array (list 0.0 0.0 -1.0 0.0)))

     (setf M (np.zeros (tuple 6 6)))
     (setf (aref M (slice 0 4) (slice 0 4)) A_np)
     (setf (aref M (slice 0 4) 4) B_np)
     (setf (aref M (slice 0 4) 5) G_np)

     ;; Diskrete Matrizen berechnen (ZOH)
     (setf M_disc (expm (* M dt))
	   A_d_np (aref M_disc (slice 0 4) (slice 0 4))
	   B_d_np (aref M_disc (slice 0 4) 4)
	   G_d_np (aref M_disc (slice 0 4) 5))

     ;; Uebertrage in CasADi Double Matrices (DM)
     (setf A (DM A_d_np)
	   B (DM B_d_np)
	   G (DM G_d_np))

     (comments "--- MPC Gewichtungsfaktoren (mit SI-Einheiten) ---"
	       "q1: Strafe fuer Auslenkung der Aufhaengung (1/m^2)."
	       "q2: Strafe fuer Aufbaugeschwindigkeit (1/(m/s)^2)."
	       "q3: Strafe fuer Reifeneinfederung (1/m^2)."
	       "q4: Strafe fuer Radgeschwindigkeit (1/(m/s)^2)."
	       "r:  Strafe fuer die Stellkraft des aktiven Daempfers (1/N^2).")
     (setf q1 10000.0
	   q2 500000.0
	   q3 1000.0
	   q4 1.0
	   r  0.000001)

     (setf q1_N (* 10.0 q1)
	   q2_N (* 10.0 q2)
	   q3_N (* 10.0 q3)
	   q4_N (* 10.0 q4))

     (comments "--- Symbolische QP-Formulierung mit map() in CasADi ---")
     ;; Zustaende und Stellgroessen als kompakte 2D Matrix-Symbolics
     (setf X (SX.sym (string "X") 4 (+ N 1))
	   U (SX.sym (string "U") 1 N))

     ;; Parametervektor p: Startzustand + Störungsvorschau
     (setf p (SX.sym (string "p") (+ 4 N))
	   x_init (aref p (slice 0 4))
	   V_r (dot (aref p (slice 4 nil)) T))

     ;; Lokale Symbole fuer die Einzelschritt-Definitionen
     (setf x_curr_sym (SX.sym (string "x_curr") 4)
	   x_next_sym (SX.sym (string "x_next") 4)
	   u_curr_sym (SX.sym (string "u_curr") 1)
	   v_curr_sym (SX.sym (string "v_curr") 1))

     ;; 1. Schrittfunktion fuer die diskrete Dynamik-Residualgleichung
     (setf f_dyn (Function (string "f_dyn")
			   (list x_next_sym x_curr_sym u_curr_sym v_curr_sym)
			   (list ,(discrete-dynamics 'x_next_sym 'x_curr_sym 'u_curr_sym 'v_curr_sym 'A 'B 'G))))

     ;; Abbildung der Dynamik ueber den gesamten Vorhersagehorizont
     (setf F_dyn (f_dyn.map N))

     ;; Residuen fuer alle N Zeitschritte berechnen (Zustandstransition)
     (setf g_dyn (F_dyn (aref X (slice nil nil) (slice 1 nil))
			(aref X (slice nil nil) (slice 0 N))
			U
			V_r))

     ;; Anfangsbedingung: x_0 - x_init = 0
     (setf g_init (- (aref X (slice nil nil) 0) x_init))

     ;; Gesamt-Nebenbedingungen-Vektor assemblieren (lbg/ubg sind 0)
     (setf g (vertcat g_init (reshape g_dyn -1 1)))

     ;; 2. Schrittfunktion fuer die Zustandskosten (Stage Cost)
     (setf f_stage (Function (string "f_stage")
			     (list x_curr_sym u_curr_sym)
			     (list ,(stage-cost 'x_curr_sym 'u_curr_sym 'q1 'q2 'q3 'q4 'r))))

     ;; Abbildung der Zustandskosten ueber den Vorhersagehorizont
     (setf F_stage (f_stage.map N))
     (setf stage_costs (F_stage (aref X (slice nil nil) (slice 0 N)) U))

     ;; Gesamtkostenfunktion: Summe der Zustandskosten + Terminale Kosten
     (setf f (+ (sum2 stage_costs)
		(* q1_N (** (aref X 0 N) 2))
		(* q2_N (** (aref X 1 N) 2))
		(* q3_N (** (aref X 2 N) 2))
		(* q4_N (** (aref X 3 N) 2))))

     ;; QP-Problemstruktur assemblieren
     ;; 'x' enthaelt die Entscheidungsvariablen (X und U, flach verkettet)
     (setf qp (dict ((string "x") (vertcat (reshape X -1 1) (reshape U -1 1)))
		    ((string "p") p)
		    ((string "f") f)
		    ((string "g") g)))

     ;; QP-Solver instanziieren (qpOASES fuer extrem schnelle Loesung)
     (setf S (qpsol (string "S") (string "qpoases") qp (dict ((string "printLevel") (string "none")))))

     (comments "--- Simulations-Setup ---")
     (setf sim_time 3.0
	   N_steps (int (/ sim_time dt))
	   t_vec (np.linspace 0.0 sim_time N_steps))

     (comments "Bodenschwelle (5cm Hoehe) von t=0.5s bis t=0.7s")
     (setf z_r_vec (np.zeros N_steps)
	   v_r_vec (np.zeros N_steps))

     (for (j (range N_steps))
	  (setf t_curr (aref t_vec j))
	  (if (and (<= 0.5 t_curr) (<= t_curr 0.7))
	      (do0
	       (setf (aref z_r_vec j) (* 0.025 (- 1.0 (np.cos (/ (* 2.0 np.pi (- t_curr 0.5)) 0.2)))))
	       (setf (aref v_r_vec j) (* (* 0.025 (/ (* 2.0 np.pi) 0.2)) (np.sin (/ (* 2.0 np.pi (- t_curr 0.5)) 0.2)))))
	      (do0
	       (setf (aref z_r_vec j) 0.0)
	       (setf (aref v_r_vec j) 0.0))))

     (comments "--- Simulation des passiven Systems mittels mapaccum() ---"
	       "Da das passive System ohne aktiven Regler u=0 laeuft, koennen wir"
	       "die gesamte Zustandshistorie ohne Schleife mit mapaccum() berechnen.")
     (setf x_curr_sym_p (SX.sym (string "x_curr_p") 4)
	   v_curr_sym_p (SX.sym (string "v_curr_p") 1))
     (setf f_passive_step (Function (string "f_passive_step")
				    (list x_curr_sym_p v_curr_sym_p)
				    (list (+ (@ A x_curr_sym_p) (* G v_curr_sym_p)))))
     (setf f_passive_sim (f_passive_step.mapaccum (string "f_passive_sim") (- N_steps 1)))

     (setf x_curr_passive_init (np.array (list 0.0 0.0 0.0 0.0))
	   x_hist_passive_matrix (f_passive_sim x_curr_passive_init (aref v_r_vec (slice nil -1))))
     (setf x_hist_passive (dot (hcat (list x_curr_passive_init x_hist_passive_matrix)) (full)))

     (comments "--- Simulation des aktiven (MPC) Systems ---")
     (setf x_hist_mpc (np.zeros (tuple 4 N_steps))
	   u_hist_mpc (np.zeros N_steps)
	   x_curr (np.array (list 0.0 0.0 0.0 0.0)))

     (setf lb_state (np.array (list -0.08 -10.0 -0.05 -20.0))
	   ub_state (np.array (list 0.08 10.0 0.05 20.0))
	   lb_input (np.array (list -1500.0))
	   ub_input (np.array (list 1500.0)))

     (setf lbx (np.concatenate (tuple (np.tile lb_state (+ N 1)) (np.tile lb_input N)))
	   ubx (np.concatenate (tuple (np.tile ub_state (+ N 1)) (np.tile ub_input N))))

     (setf lbg (np.zeros (* 4 (+ N 1)))
	   ubg (np.zeros (* 4 (+ N 1))))

     (setf x0_guess (np.zeros (+ (* 4 (+ N 1)) N)))

     (for (j (range (- N_steps 1)))
	  (setf (aref x_hist_mpc (slice nil nil) j) x_curr)

	  (setf V_r_horiz (np.zeros N))
	  (for (k (range N))
	       (setf idx (+ j k))
	       (if (< idx N_steps)
		   (setf (aref V_r_horiz k) (aref v_r_vec idx))
		   (setf (aref V_r_horiz k) 0.0)))

	  (setf p_val (np.concatenate (tuple x_curr V_r_horiz)))

	  (setf sol (S :x0 x0_guess :p p_val :lbx lbx :ubx ubx :lbg lbg :ubg ubg)
		x_opt (aref sol (string "x")))

	  (setf u_opt (float (aref x_opt (* 4 (+ N 1)))))
	  (setf (aref u_hist_mpc j) u_opt)

	  ;; Diskretes Update fuer den naechsten Zustand (Active)
	  (setf x_curr (+ (@ A_d_np x_curr) (* B_d_np u_opt) (* G_d_np (aref v_r_vec j)))))

     (setf (aref x_hist_mpc (slice nil nil) -1) x_curr)

     (comments "--- Berechnungen der Komfortmetriken ---")
     (setf acc_mpc (+ (* (/ (- ks) ms) (aref x_hist_mpc 0))
		      (* (/ (- cs) ms) (- (aref x_hist_mpc 1) (aref x_hist_mpc 3)))
		      (* (/ 1.0 ms) u_hist_mpc))
	   acc_passive (+ (* (/ (- ks) ms) (aref x_hist_passive 0))
			  (* (/ (- cs) ms) (- (aref x_hist_passive 1) (aref x_hist_passive 3)))))

     (setf zs_mpc (+ (aref x_hist_mpc 0) (aref x_hist_mpc 2) z_r_vec)
	   zs_passive (+ (aref x_hist_passive 0) (aref x_hist_passive 2) z_r_vec))

     (comments "--- Plotting und Visualisierung ---")
     (plt.rcParams.update (dict ((string "font.family") (string "sans-serif"))
				((string "font.sans-serif") (list (string "DejaVu Sans") (string "Arial")))
				((string "axes.edgecolor") (string "#cccccc"))
				((string "axes.linewidth") 0.8)
				((string "grid.color") (string "#eeeeee"))
				((string "grid.linestyle") (string "-"))))

     (setf (ntuple fig axes) (plt.subplots 2 2 :figsize (tuple 14 10)))
     (dot fig (suptitle (string "Kompaktes Active MPC vs. Passives Fahrwerk (CasADi Map-Formulierung)") :fontsize 16 :fontweight (string "bold") :y 0.98))

     ;; Plot 1: Road Profile and Chassis Displacement
     (setf ax1 (aref axes 0 0))
     (dot ax1 (fill_between t_vec 0 z_r_vec :color (string "#e0e0e0") :alpha 0.5 :label (string "Strassenprofil (Schwelle)")))
     (dot ax1 (plot t_vec zs_passive :label (string "Passives Fahrwerk") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax1 (plot t_vec zs_mpc :label (string "Aktives Fahrwerk (Map-MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax1 (set_title (string "Aufbauposition (zs)") :fontsize 12 :fontweight (string "bold")))
     (dot ax1 (set_xlabel (string "Zeit (s)") :fontsize 10))
     (dot ax1 (set_ylabel (string "Auslenkung (m)") :fontsize 10))
     (dot ax1 (grid True :alpha 0.6))
     (dot ax1 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax1 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax1 spines) (string "right")) (set_visible False))

     ;; Plot 2: Chassis Acceleration
     (setf ax2 (aref axes 0 1))
     (dot ax2 (axhspan -0.5 0.5 :color (string "#e2f0d9") :alpha 0.6 :label (string "Komfortzone (ISO 2631)")))
     (dot ax2 (plot t_vec acc_passive :label (string "Passiv") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax2 (plot t_vec acc_mpc :label (string "Aktiv (Map-MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax2 (set_title (string "Aufbaubeschleunigung (Chassis)") :fontsize 12 :fontweight (string "bold")))
     (dot ax2 (set_xlabel (string "Zeit (s)") :fontsize 10))
     (dot ax2 (set_ylabel (string "Beschleunigung (m/s^2)") :fontsize 10))
     (dot ax2 (grid True :alpha 0.6))
     (dot ax2 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax2 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax2 spines) (string "right")) (set_visible False))

     ;; Plot 3: Suspension Deflection (Stroke)
     (setf ax3 (aref axes 1 0))
     (dot ax3 (axhspan -0.08 0.08 :color (string "#f1f3f4") :alpha 0.6 :zorder 0))
     (dot ax3 (plot t_vec (aref x_hist_passive 0) :label (string "Passiv") :color (string "#ff4d4d") :linestyle (string "--") :lw 1.5))
     (dot ax3 (plot t_vec (aref x_hist_mpc 0) :label (string "Aktiv (Map-MPC)") :color (string "#1a73e8") :lw 2.5))
     (dot ax3 (axhline :y 0.08 :color (string "#cc0000") :linestyle (string ":") :lw 1.2 :label (string "Aufhaengungsweg-Grenze (+/- 8cm)")))
     (dot ax3 (axhline :y -0.08 :color (string "#cc0000") :linestyle (string ":") :lw 1.2))
     (dot ax3 (set_title (string "Federweg (Aufbau relativ zu Rad)") :fontsize 12 :fontweight (string "bold")))
     (dot ax3 (set_xlabel (string "Zeit (s)") :fontsize 10))
     (dot ax3 (set_ylabel (string "Federweg (m)") :fontsize 10))
     (dot ax3 (grid True :alpha 0.6))
     (dot ax3 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax3 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax3 spines) (string "right")) (set_visible False))

     ;; Plot 4: Actuator Control Force
     (setf ax4 (aref axes 1 1))
     (dot ax4 (axhspan -1500.0 1500.0 :color (string "#f1f3f4") :alpha 0.6 :zorder 0))
     (dot ax4 (step t_vec u_hist_mpc :label (string "Aktive Stellkraft (Map-MPC)") :color (string "#34a853") :lw 2 :where (string "post")))
     (dot ax4 (axhline :y 1500.0 :color (string "#cc0000") :linestyle (string ":") :lw 1.2 :label (string "Aktuator-Kraftgrenze (+/- 1500N)")))
     (dot ax4 (axhline :y -1500.0 :color (string "#cc0000") :linestyle (string ":") :lw 1.2))
     (dot ax4 (set_title (string "Aktuator-Stellkraft") :fontsize 12 :fontweight (string "bold")))
     (dot ax4 (set_xlabel (string "Zeit (s)") :fontsize 10))
     (dot ax4 (set_ylabel (string "Kraft (N)") :fontsize 10))
     (dot ax4 (grid True :alpha 0.6))
     (dot ax4 (legend :frameon True :facecolor (string "white") :edgecolor (string "none")))
     (dot (aref (dot ax4 spines) (string "top")) (set_visible False))
     (dot (aref (dot ax4 spines) (string "right")) (set_visible False))

     (plt.tight_layout)
     (plt.savefig (string "active_suspension_mpc_map.png") :dpi 150)
     (print (string "Plot gespeichert als active_suspension_mpc_map.png"))
     )))
