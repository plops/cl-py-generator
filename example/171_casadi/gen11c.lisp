(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g11c
  (:use #:cl #:cl-py-generator))

(in-package #:g11c)

(defparameter *source* "example/171_casadi/")

(write-source
 (asdf:system-relative-pathname 'cl-py-generator
				(merge-pathnames #P"p11c_mpc_gui"
						 *source*))
 `(do0
   (imports-from (__future__ annotations)
		 (casadi *))
   (imports ((np numpy)
	     time
	     sys
	     (pg pyqtgraph)))
   (imports-from (PySide6.QtWidgets QApplication QMainWindow QWidget QVBoxLayout QHBoxLayout QSlider QLabel QGridLayout QPushButton)
		 (PySide6.QtCore Qt QTimer)
		 (PySide6.QtGui QPainter QPen QBrush QColor))

   (comments 
    "========================================================================================="
    " INVERTED PENDULUM MPC (MODEL PREDICTIVE CONTROL) DASHBOARD"
    "========================================================================================="
    " ZWECK UND ZIELGRUPPE:"
    " Diese Applikation demonstriert in Echtzeit die modellprädiktive Regelung (MPC) eines"
    " nichtlinearen, unteraktuierten mechanischen Systems (invertiertes Pendel auf einem Wagen)."
    " Sie ist so gestaltet, dass Physiker und Ingenieure die Auswirkungen von physikalischen"
    " Parametern (Masse, Wind) sowie Regelungsgewichten (Q-Matrizen) interaktiv erforschen können."
    ""
    " PHYSIKALISCHES MODELL & DYNAMIK (LAGRANGE-MECHANIK):"
    " Wir betrachten einen Wagen der Masse M auf einer 1D-Schiene (Position s). Auf diesem Wagen"
    " ist ein Pendel der Masse m und Länge l montiert (Winkel theta). Das System ist unteraktuiert:"
    " Wir können nur eine horizontale Kraft F auf den Wagen ausüben, wollen aber s und theta regeln."
    " Der Zustandsvektor ist x = [s, v, theta, omega]."
    " Die Differentialgleichungen (ODEs) werden über die Euler-Lagrange-Gleichungen T - V hergeleitet:"
    " - Die kinetische Energie T berücksichtigt die Translation des Wagens und die Rotation/Translation des Pendels."
    " - Die potentielle Energie V berücksichtigt die Höhe der Pendelmasse im Gravitationsfeld."
    " Eine externe Störkraft (Wind) wirkt zusätzlich als Drehmoment auf das Pendel ein."
    ""
    " REGELUNGSTHEORIE (MODEL PREDICTIVE CONTROL - MPC):"
    " MPC löst zu jedem diskreten Zeitschritt ein Optimierungsproblem über einen endlichen"
    " Zeithorizont (T_horizon). Der Algorithmus berechnet eine zukünftige Trajektorie von"
    " Steuerkräften F, die eine Kostenfunktion (Abweichung vom Soll-Zustand + Energieverbrauch)"
    " minimiert, während Systemgrenzen (z.B. Schienenende, Maximalkraft) strikt eingehalten werden."
    " Nur der allererste berechnete Kraftwert wird tatsächlich an das System gesendet. Im nächsten"
    " Schritt verschiebt sich der Horizont (Receding Horizon) und das Problem wird neu gelöst."
    ""
    " MATHEMATIK DER DIREKTEN KOLLOKATION (DIRECT COLLOCATION):"
    " Um die kontinuierlichen Differentialgleichungen (ODE) für den NLP-Solver nutzbar zu machen,"
    " diskretisieren wir die Zustands-Trajektorie mittels Lagrange-Polynomen über Radau-Punkte."
    " Anstatt die ODE numerisch zu integrieren (Multiple Shooting), werden die Zustände an den"
    " Kollokationspunkten zu freien Optimierungsvariablen. Die Systemdynamik dx/dt = f(x,u) wird"
    " als strikte Gleichheitsbedingung (Equality Constraint) aufgezwungen."
    " Dies transformiert das Problem in ein riesiges, aber sehr dünnbesetztes (sparse) NLP-Problem."
    ""
    " CASADI & IPOPT (IMPLEMENTIERUNGSDETAILS):"
    " CasADi ist ein Computer-Algebra-System für algorithmische Differentiation (AD). Es berechnet"
    " exakte und effiziente Jacobians (erste Ableitungen) und Hessians (zweite Ableitungen) des NLP."
    " Wir nutzen 'SX' (Scalar Expression) Graphen für die ODE, da diese für mathematische Operationen"
    " auf Skalarebene deutlich schneller ausgewertet werden als Matrix-Ausdrücke ('MX')."
    " IPOPT (Interior Point Optimizer) nutzt Barrierefunktionen, um die Constraints zu lösen."
    " Für Echtzeitfähigkeit nutzen wir 'Warm-Starting': Die optimale Lösung des vorherigen Schrittes"
    " dient als Startschätzung für den aktuellen, wodurch IPOPT oft nur 1-3 Iterationen benötigt."
    "=========================================================================================")

   (class PendulumMPC ()
     (def __init__ (self &key (h_mpc 0.05) (N 20))
       (setf self.opti (Opti)
	     self.nx 4
	     self.nu 1
	     self.N N
	     self.h_mpc h_mpc
	     self.T_horizon (* self.N self.h_mpc)
	     self.d 3)

       (comments 
        " Parameter für Physik und Optimierung."
        " Wir definieren diese als CasADi 'Parameter' (opti.parameter), anstatt sie fest zu"
        " verdrahten. Das ermöglicht es uns, Massen, Wind, Grenzen oder auch den Vorhersagehorizont"
        " (T_horizon) zur Laufzeit der GUI zu ändern, ohne den kompletten CasADi Optimierungs-"
        " Graphen neu aufbauen und kompilieren zu müssen (was extrem rechenintensiv wäre).")
       ,@(loop for sym in '(M_p m_p l_p wind_p Q_s Q_v Q_theta Q_omega R_F max_pos max_force)
               collect `(setf (dot self ,sym) (self.opti.parameter)))

       (comments
        "============================================================================="
        " PHYSIK-HERLEITUNG & DYNAMISCHE GLEICHUNGEN (LAGRANGE & HAMILTON)"
        "============================================================================="
        " Systemzustand: x = [s, v, theta, omega]^T"
        " - s: Position des Wagens (Schiene) [m]"
        " - v: Geschwindigkeit des Wagens (ds/dt) [m/s]"
        " - theta: Pendelwinkel relativ zur Ruhelage [rad]"
        "          (theta = 0 ist die physikalisch stabile Ruhelage nach unten im ODE-Modell;"
        "          in der GUI-Zeichnung wird theta = 0 jedoch als aufrechte Position gezeichnet)"
        " - omega: Winkelgeschwindigkeit des Pendels (dtheta/dt) [rad/s]"
        " - u: Stellkraft F [N] (horizontale Kraft auf den Wagen)"
        ""
        " 1. KINETISCHE ENERGIE (T):"
        "    Die kinetische Energie setzt sich aus der Translation des Wagens und der Bewegung der"
        "    Pendel-Punktmasse zusammen:"
        "    T = 0.5 * M * v^2 + 0.5 * m * (v_p^2)"
        "    wobei x_p = s + l*sin(theta), y_p = -l*cos(theta) (y-Achse nach oben)."
        "    Ableitungen: dx_p = v + l*omega*cos(theta), dy_p = l*omega*sin(theta)"
        "    v_p^2 = dx_p^2 + dy_p^2 = v^2 + 2*v*l*omega*cos(theta) + l^2*omega^2"
        "    T = 0.5 * (M + m) * v^2 + m * v * l * omega * cos(theta) + 0.5 * m * l^2 * omega^2"
        ""
        " 2. POTENTIELLE ENERGIE (V):"
        "    Unter Berücksichtigung der Höhe y_p der Pendelmasse:"
        "    V = m * g * y_p = -m * g * l * cos(theta)"
        "    (Minimum bei theta = 0, d.h. die physikalisch stabile Ruhelage ist unten)"
        ""
        " 3. HAMILTONIAN (H = T + V):"
        "    Die Hamilton-Funktion repräsentiert die mechanische Gesamtenergie des Systems:"
        "    H = 0.5*(M + m)*v^2 + m*v*l*omega*cos(theta) + 0.5*m*l^2*omega^2 - m*g*l*cos(theta)"
        ""
        " 4. EULER-LAGRANGE-GLEICHUNGEN & BEWEGUNGSGLEICHUNGEN:"
        "    Lagrange-Funktion: L = T - V"
        "    Gleichungen: d/dt(dL/dv) - dL/ds = F_total"
        "                 d/dt(dL/domega) - dL/dtheta = 0"
        "    wobei F_total = F + F_wind*cos(theta) (horizontale Kraft + Windstörkraft)"
        "    Dies führt auf das gekoppelte System:"
        "    (1) (M + m)*dv/dt + m*l*domega/dt*cos(theta) - m*l*omega^2*sin(theta) = F_total"
        "    (2) m*l*dv/dt*cos(theta) + m*l^2*domega/dt + m*g*l*sin(theta) = 0"
        "    "
        "    Durch Auflösen nach den Beschleunigungen dv/dt (dv) und domega/dt (domega) erhalten wir:"
        "    Nenner: den = M + m * sin(theta)^2"
        "    dv = (F_total + m*l*omega^2*sin(theta) + m*g*cos(theta)*sin(theta)) / den"
        "    domega = (-F_total*cos(theta) - m*l*omega^2*sin(theta)*cos(theta) - (M+m)*g*sin(theta)) / (l * den)"
        ""
        " Symbolische Variablen für die Dynamik (dx/dt = f(x,u,p)):")
       (setf x (SX.sym (string "x") self.nx)
	     u (SX.sym (string "u") self.nu)
	     p_ode (SX.sym (string "p_ode") 4)
	     s_ (aref x 0) v_ (aref x 1) theta_ (aref x 2) omega_ (aref x 3) F_ u
	     M_s (aref p_ode 0) m_s (aref p_ode 1) l_s (aref p_ode 2) wind_s (aref p_ode 3)
	     sin_theta (np.sin theta_) cos_theta (np.cos theta_)
	     den (+ M_s (* m_s (- 1.0 (* cos_theta cos_theta))))
	     F_total (+ F_ (* wind_s cos_theta))
	     ds v_
	     dv (/ (+ F_total (* m_s l_s omega_ omega_ sin_theta) (* m_s 9.81 cos_theta sin_theta)) den)
	     dtheta omega_
	     domega (/ (- (* -1.0 F_total cos_theta) (* m_s l_s omega_ omega_ sin_theta cos_theta) (* (+ M_s m_s) 9.81 sin_theta)) (* l_s den)))
       (setf self.f_ode (Function (string "f_ode") (list x u p_ode) (list (vertcat ds dv dtheta domega))))

        (comments
         "============================================================================="
         " MATHEMATISCHE DIREKTE KOLLOKATION (DIRECT COLLOCATION)"
         "============================================================================="
         " Bei der direkten Kollokation wird die Zustandstrajektorie x(t) über jedes"
         " Zeitintervall h_mpc durch ein Polynom vom Grad d (hier d=3) approximiert."
         " Die Optimierungsvariablen sind die Zustände an den Intervallgrenzen (X)"
         " sowie an den d inneren Kollokationspunkten (Xc)."
         ""
         " Koordinaten & Dimensionen:"
         " - d (Polynomgrad) = 3"
         " - tau_root: Zeitstützstellen normiert auf das Intervall [0, 1]. Enthält 0.0 und die"
         "   d Wurzeln des Radau-Polynoms (Länge: d+1 = 4). Dimension: (4,)"
         " - C: Kollokationsmatrix (Koeffizienten der zeitlichen Ableitung) der Dimension (d+1) x (d+1), d.h. 4x4."
         " - D: Extrapolationsvektor (Endwerte der Lagrange-Polynome bei tau=1.0) der Dimension d+1, d.h. 4."
         ""
         " Funktionsweise von NumPy Hilfsfunktionen:"
         " - np.poly1d([1.0, -tau_r]): Erstellt ein symbolisches 1D-Polynom (tau - tau_r)."
         "   Wird verwendet, um das Lagrange-Basispolynom L_j(tau) = prod_{r != j} (tau - tau_r)/(tau_j - tau_r)"
         "   zu konstruieren, welches an tau_j den Wert 1 und an allen anderen tau_r den Wert 0 hat."
         " - np.polyder(p): Berechnet die analytische Ableitung dL_j/dtau des Lagrange-Polynoms."
         ""
         " Bedeutung der Matrizen C und D:"
         " - Matrix C (C[j, r] = dL_j/dtau(tau_r)): Repräsentiert die Ableitung des j-ten Basispolynoms"
         "   am Kollokationspunkt r. Ermöglicht die lineare Berechnung der Ableitung dx/dt"
         "   an allen Kollokationspunkten über die Zustände: dx/dt(tau_r) = 1/h * sum(C[j,r]*x_j)."
         "   Damit wird die Gleichheitsbedingung der ODE dx/dt = f(x,u) zu: sum(C[j,r]*x_j) = h * f(x_r, u)."
         " - Vektor D (D[j] = L_j(1.0)): Extrapoliert den Zustand am Ende des Intervalls (tau=1.0)"
         "   aus den Kollokationspunkten: x_end = sum(D[j]*x_j). Dies erzwingt die Kontinuität"
         "   des Zustands zum nächsten Intervallstart: X_{k+1} == x_end.")
        (setf tau_root (np.append 0.0 (collocation_points self.d (string "radau")))
	     self.C (np.zeros (tuple (+ self.d 1) (+ self.d 1)))
	     self.D (np.zeros (+ self.d 1)))
        (for (j (range (+ self.d 1)))
	    (setf p (np.poly1d (list 1.0)))
	    (for (r (range (+ self.d 1)))
		 (if (!= r j)
		     (setf p (* p (/ (np.poly1d (list 1.0 (- (aref tau_root r))))
				     (- (aref tau_root j) (aref tau_root r)))))))
	    (setf (aref self.D j) (p 1.0)
		  pder (np.polyder p))
	    (for (r (range (+ self.d 1)))
		 (setf (aref self.C j r) (pder (aref tau_root r)))))

       (comments "Decision Variables für IPOPT (X: Knoten, Xc: Collocation-Punkte, U: Stellgröße)")
       (setf self.X (self.opti.variable self.nx (+ self.N 1))
	     self.Xc (list)
	     self.U (self.opti.variable self.nu self.N)
	     self.current_x (self.opti.parameter self.nx)
	     self.target_x (self.opti.parameter self.nx))
       (for (k (range self.N))
	    (setf Xc_k (list))
	    (for (r (range self.d))
		 (dot Xc_k (append (self.opti.variable self.nx))))
	    (dot self.Xc (append Xc_k)))

       (comments "Constraints anwenden: Startzustand, Grenzen und Collocation-Dynamik")
       (self.opti.subject_to (== (aref self.X (slice nil nil) 0) self.current_x))
       (self.opti.subject_to (self.opti.bounded (* -1.0 self.max_pos) (aref self.X 0 (slice nil nil)) self.max_pos))
       (self.opti.subject_to (self.opti.bounded (* -1.0 self.max_force) self.U self.max_force))

       (for (k (range self.N))
	    (setf Xk (aref self.X (slice nil nil) k)
		  x_end (* (aref self.D 0) Xk))
	    (for (j (range 1 (+ self.d 1)))
		 (setf xp (* (aref self.C 0 j) Xk))
		 (for (r (range self.d))
		      (setf xp (+ xp (* (aref self.C (+ r 1) j) (aref (aref self.Xc k) r)))))
		 (setf f_eval (self.f_ode (aref (aref self.Xc k) (- j 1))
					  (aref self.U (slice nil nil) k)
					  (vertcat self.M_p self.m_p self.l_p self.wind_p)))
		 (self.opti.subject_to (== xp (* self.h_mpc f_eval))))
	    (for (r (range self.d))
		 (setf x_end (+ x_end (* (aref self.D (+ r 1)) (aref (aref self.Xc k) r)))))
	    (self.opti.subject_to (== (aref self.X (slice nil nil) (+ k 1)) x_end)))

        (comments 
         "============================================================================="
         " KOSTENFUNKTION (OBJECTIVE FUNCTION)"
         "============================================================================="
         " Die Kostenfunktion J minimiert die Abweichungen vom Zielzustand und den Energieaufwand."
         " J = sum_{k=0}^{N-1} ( err_k^T * Q * err_k + R_F * u_k^2 ) + 10 * err_N^T * Q * err_N"
         " "
         " Stellenergie-Formel (Control Effort Penalty):"
         " - Für jeden Zeitschritt k penalisiert der Term R_F * (U[0, k])^2 den quadratischen Stellaufwand."
         " - Formel: Stellenergie = R_F * u_k^2"
         "   wobei u_k (Kraft F) die Stellgröße und R_F der Gewichtungsfaktor (Stellkosten) ist."
         " - Dies verhindert extrem aggressive Steueraktionen und schont die Aktuatoren.")
        (setf cost 0.0
	     Q (diag (vertcat self.Q_s self.Q_v self.Q_theta self.Q_omega)))
        (for (k (range self.N))
	    (setf err (- (aref self.X (slice nil nil) k) self.target_x)
		  cost (+ cost (+ (mtimes (mtimes err.T Q) err)
				  (* self.R_F (** (aref self.U 0 k) 2))))))
        (setf err_term (- (aref self.X (slice nil nil) self.N) self.target_x)
	     cost (+ cost (* 10.0 (mtimes (mtimes err_term.T Q) err_term))))
        (self.opti.minimize cost)

        (comments 
         "============================================================================="
         " IPOPT SOLVER KONFIGURATION & PARAMETER"
         "============================================================================="
         " IPOPT (Interior Point Optimizer) löst das nichtlineare Optimierungsproblem (NLP)."
         " Zur Optimierung der Echtzeitfähigkeit konfigurieren wir folgende Parameter:"
         " - print_time = False: Deaktiviert die Ausgabe der Rechenzeitstatistik durch CasADi"
         "   nach jedem Solver-Durchlauf, um das Terminal sauber zu halten."
         " - print_level = 0: Unterdrückt alle detaillierten IPOPT-Iterationen-Logs (Standard: 5)."
         " - sb = 'yes' (Suppress Banner): Unterdrückt das IPOPT-Lizenz- und Start-Banner."
         " - Warm-Starting (siehe step-Methode): Durch Wiederverwendung der verschobenen Lösung"
         "   des vorherigen Schritts benötigt IPOPT im GUI-Betrieb meist nur 1-3 Iterationen.")
        (self.opti.solver (string "ipopt") 
			 (dict ((string "print_time") False))
			 (dict ((string "print_level") 0) ((string "sb") (string "yes"))))
        
        (self.opti.set_initial self.X (dot (np.linspace (np.array (list 0.0 0.0 np.pi 0.0)) (np.array (list 1.0 0.0 0.0 0.0)) (+ self.N 1)) T))
        (setf self.sol None))

     (def step (self state target_state params)
       (self.opti.set_value self.current_x state)
       (self.opti.set_value self.target_x target_state)
       ,@(loop for (key sym) in '(("M" self.M_p)
                                  ("m" self.m_p)
                                  ("l" self.l_p)
                                  ("wind" self.wind_p)
                                  ("Q_s" self.Q_s)
                                  ("Q_v" self.Q_v)
                                  ("Q_theta" self.Q_theta)
                                  ("Q_omega" self.Q_omega)
                                  ("R_F" self.R_F)
                                  ("max_pos" self.max_pos)
                                  ("max_force" self.max_force))
	       collect `(self.opti.set_value ,sym (aref params (string ,key))))

       (if (!= self.sol None)
	   (do0
	    (setf X_res (self.sol.value self.X)
		  U_res (self.sol.value self.U)
		  X_guess (np.hstack (tuple (aref X_res (slice nil nil) (slice 1 nil))
					    (aref X_res (slice nil nil) (slice -1 nil))))
		  U_guess (np.append (aref U_res (slice 1 nil)) (aref U_res -1)))
	    (self.opti.set_initial self.X X_guess)
	    (self.opti.set_initial self.U U_guess))
	   (do0
	    (setf X_res (np.zeros (tuple self.nx (+ self.N 1)))
		  U_res (np.zeros (tuple self.nu self.N))
		  U_guess (np.zeros self.nu))))

       (setf t0 (time.time))
       (try
	(setf self.sol (self.opti.solve))
	("Exception as e"
		(return (tuple (aref U_guess 0) X_res U_res (- (time.time) t0) False))))
       (setf t_solve (- (time.time) t0)
	     X_res (self.sol.value self.X)
	     U_res (self.sol.value self.U))
       (return (tuple (aref U_res 0) X_res U_res t_solve True))))

   (class PendulumWidget (QWidget)
     (def __init__ (self)
       (QWidget.__init__ self)
       (self.setMinimumSize 400 400)
       (setf self.state (np.array (list 0.0 0.0 np.pi 0.0))
	     self.l 0.5 self.force 0.0 self.wind 0.0 self.max_pos 2.0))
     
     (def update_state (self state force wind l max_pos)
       (setf self.state state self.force force self.wind wind self.l l self.max_pos max_pos)
       (self.update))
       
     (def paintEvent (self event)
       (setf painter (QPainter self)
	     w (self.width) h (self.height) cx (/ w 2) cy (/ h 2)
	     scale (/ w (* 2.5 self.max_pos)))
       (painter.setRenderHint QPainter.Antialiasing)
       (painter.fillRect 0 0 w h (QColor 30 30 30))
       
       (painter.setPen (pg.mkPen (string "gray") :width 4))
       (painter.drawLine (- cx (* scale self.max_pos)) cy (+ cx (* scale self.max_pos)) cy)
       
       (setf s (aref self.state 0) theta (aref self.state 2)
	     cart_w (* 0.4 scale) cart_h (* 0.2 scale)
	     cart_x (+ cx (* s scale)) cart_y cy)
       
       (painter.setBrush (QBrush (QColor 100 150 250)))
       (painter.setPen (pg.mkPen (string "w") :width 2))
       (painter.drawRect (- cart_x (/ cart_w 2)) (- cart_y (/ cart_h 2)) cart_w cart_h)
       
       (setf pend_x (+ cart_x (* scale self.l (np.sin theta)))
	     pend_y (- cart_y (* scale self.l (np.cos theta))))
       (painter.setPen (pg.mkPen (string "w") :width 6))
       (painter.drawLine cart_x cart_y pend_x pend_y)
       (painter.setBrush (QBrush (QColor 250 100 100)))
       (painter.drawEllipse (- pend_x 10) (- pend_y 10) 20 20)
       
       (if (> (np.abs self.force) 0.1)
	   (do0
	    (painter.setPen (pg.mkPen (string "g") :width 3))
	    (painter.drawLine cart_x cart_y (+ cart_x (* 5.0 self.force)) cart_y)))))

   (class MainWindow (QMainWindow)
     (def __init__ (self)
       (QMainWindow.__init__ self)
       (self.setWindowTitle (string "MPC Inverted Pendulum"))
       (self.resize 1400 900)
       
       (setf central_widget (QWidget)
	     main_layout (QHBoxLayout central_widget)
	     left_layout (QVBoxLayout)
	     right_layout (QVBoxLayout))
       (self.setCentralWidget central_widget)
       (main_layout.addLayout left_layout 1)
       (main_layout.addLayout right_layout 2)
       
       (setf self.pendulum_widget (PendulumWidget)
	     self.plot_layout (pg.GraphicsLayoutWidget))
       (left_layout.addWidget self.pendulum_widget)
       (right_layout.addWidget self.plot_layout)
       
       (setf self.plots (dict)
	     self.history_curves (dict)
	     self.pred_curves (dict))
       
       (comments "Plots dynamisch generieren mit Lisp Macro Expand")
       ,@(loop for (key title ylabel row col) in '(("s" "Wagenposition" "Position [m]" 0 0)
						   ("theta" "Pendelwinkel" "Winkel [rad]" 1 0)
						   ("v" "Wagengeschwindigkeit" "v [m/s]" 2 0)
						   ("omega" "Winkelgeschw." "omega [rad/s]" 0 1)
						   ("F" "Aktuatorkraft" "Kraft [N]" 1 1)
						   ("t_solve" "Solver Rechenzeit" "Zeit [ms]" 2 1))
	       collect
	       `(do0
		 (setf ax (self.plot_layout.addPlot :row ,row :col ,col :title (string ,title)))
		 (ax.showGrid :x True :y True)
		 (ax.setLabel (string "left") (string ,ylabel))
		 (ax.setLabel (string "bottom") (string "Zeit [s]"))
		 (setf (aref self.plots (string ,key)) ax
		       (aref self.history_curves (string ,key)) (ax.plot :pen (pg.mkPen (string "w") :width 2)))
		 (if (!= (string ,key) (string "t_solve"))
		     (setf (aref self.pred_curves (string ,key)) (ax.plot :pen (pg.mkPen (string "y") :width 2 :style Qt.DashLine))))))
		     
       (comments "Steuerungs-Buttons für Simulation")
       (setf btn_layout (QHBoxLayout)
             self.btn_start (QPushButton (string "Start"))
             self.btn_stop (QPushButton (string "Stop"))
             self.btn_reset (QPushButton (string "Reset / Apply Params")))
       (btn_layout.addWidget self.btn_start)
       (btn_layout.addWidget self.btn_stop)
       (btn_layout.addWidget self.btn_reset)
       (left_layout.addLayout btn_layout)
       
       (self.btn_start.clicked.connect self.start_sim)
       (self.btn_stop.clicked.connect self.stop_sim)
       (self.btn_reset.clicked.connect self.reset_sim)

       (setf self.sliders (dict)
             slider_layout (QGridLayout))
       (left_layout.addLayout slider_layout)
       
       (def add_slider (layout name label min_val max_val default_val scale row tooltip)
	 (setf slider (QSlider Qt.Horizontal)
	       init_val (* default_val scale)
	       lbl (QLabel (? (isinstance scale float)
			      (fstring "{label}: {init_val:.2f}")
			      (fstring "{label}: {int(init_val)}"))))
	 (slider.setToolTip tooltip)
	 (lbl.setToolTip tooltip)
	 (slider.setMinimum min_val)
	 (slider.setMaximum max_val)
	 (slider.setValue default_val)
	 (layout.addWidget lbl row 0)
	 (layout.addWidget slider row 1)
	 (slider.valueChanged.connect (lambda () 
					(lbl.setText (? (isinstance scale float)
							(fstring "{label}: {slider.value() * scale:.2f}")
							(fstring "{label}: {int(slider.value() * scale)}")))))
	 (setf (aref self.sliders name) slider))

       ,@(loop for (key label min-val max-val def-val scale tooltip) in
               '(("M" "Wagenmasse [kg]" 1 50 10 0.1 "Masse des Wagens. Schwerer = Träger gegen Bewegungsänderungen.")
                 ("m" "Pendelmasse [kg]" 1 20 1 0.1 "Punktmasse des Pendels am Kopfende.")
                 ("l" "Pendellänge [m]" 1 20 5 0.1 "Abstand vom Wagen zum Pendelschwerpunkt.")
                 ("wind" "Windkraft [N]" -300 300 0 0.1 "Konstante Störkraft, die horizontal auf das Pendel drückt.")
                 ("Q_s" "Gewicht Position" 0 200 10 1 "Strafe (Penalty) für Abweichung des Wagens von der Ziel-Position.")
                 ("Q_v" "Gewicht Wagengeschw." 0 100 1 1 "Strafe für hohe Geschwindigkeit des Wagens (verhindert Überschwingen).")
                 ("Q_theta" "Gewicht Pendelwinkel" 0 500 100 1 "Strafe für das Abweichen des Pendels vom instabilen Gleichgewicht (0 rad).")
                 ("Q_omega" "Gewicht Winkelgeschw." 0 100 1 1 "Strafe für schnelle Rotationen des Pendels.")
                 ("R_F" "Gewicht Kraftaufwand" 1 200 10 0.01 "Kostenfaktor für die Stellkraft F. Zwingt den Solver, Energie zu sparen.")
                 ("target_s" "Ziel-Position [m]" -100 100 10 0.1 "Soll-Position des Wagens auf der Schiene.")
                 ("max_pos" "Schiene Limit [m]" 10 200 50 0.1 "Maximaler erlaubter Fahrweg (Constraint). Der Solver darf diesen nie überschreiten.")
                 ("max_force" "Max Kraft [N]" 10 300 150 0.1 "Stellgrößenbeschränkung für den Aktuator.")
                 ("N" "Knotenpunkte (N)" 1 50 20 1 "Auflösung des Solvers. N=1 bedeutet nur ein Zeitschritt in die Zukunft.")
                 ("h_mpc" "MPC Schritt [ms]" 1 200 50 1 "Dauer eines MPC-Planungsschritts. T_horizon = N * h_mpc.")
                 ("dt_sim" "Simulations-dt [ms]" 1 100 33 1 "Schrittweite der echten Runge-Kutta Physiksimulation. Hat keinen Einfluss auf den Solver."))
               for row from 0
               collect
               `(add_slider slider_layout (string ,key) (string ,label) ,min-val ,max-val ,def-val ,scale ,row (string ,tooltip)))
       
       (setf self.timer (QTimer))
       (self.timer.timeout.connect self.update_loop)
       (self.reset_sim)
       (self.start_sim))

     (def start_sim (self)
       (setf self.dt (/ (dot (aref self.sliders (string "dt_sim")) (value)) 1000.0))
       (self.timer.start (int (* self.dt 1000.0)))
       (dot (aref self.sliders (string "h_mpc")) (setEnabled False))
       (dot (aref self.sliders (string "N")) (setEnabled False)))

     (def stop_sim (self)
       (self.timer.stop)
       (dot (aref self.sliders (string "h_mpc")) (setEnabled True))
       (dot (aref self.sliders (string "N")) (setEnabled True)))

     (def reset_sim (self)
       (self.stop_sim)
       (setf params (self.get_params)
             self.time 0.0
             self.state (np.array (list 0.0 0.0 np.pi 0.0))
             self.t_hist (list)
             self.dt (aref params (string "dt_sim"))
             self.hist (dict ((string "s") (list)) ((string "v") (list)) ((string "theta") (list))
                             ((string "omega") (list)) ((string "F") (list)) ((string "t_solve") (list))))
       (comments "MPC mit neuen Parametern neu kompilieren")
       (setf self.mpc (PendulumMPC :h_mpc (aref params (string "h_mpc"))
                                   :N (aref params (string "N"))))
       (self.pendulum_widget.update_state self.state 0.0 0.0 (aref params (string "l")) (aref params (string "max_pos")))
       ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
               collect `(dot (aref self.history_curves (string ,key)) (setData (list) (list))))
       ,@(loop for key in '("s" "v" "theta" "omega" "F")
               collect `(dot (aref self.pred_curves (string ,key)) (setData (list) (list)))))

     (def get_params (self)
       (return (dictionary :M (/ (dot (aref self.sliders (string "M")) (value)) 10.0)
		     :m (/ (dot (aref self.sliders (string "m")) (value)) 10.0)
		     :l (/ (dot (aref self.sliders (string "l")) (value)) 10.0)
		     :wind (/ (dot (aref self.sliders (string "wind")) (value)) 10.0)
		     :Q_s (float (dot (aref self.sliders (string "Q_s")) (value)))
		     :Q_v (float (dot (aref self.sliders (string "Q_v")) (value)))
		     :Q_theta (float (dot (aref self.sliders (string "Q_theta")) (value)))
		     :Q_omega (float (dot (aref self.sliders (string "Q_omega")) (value)))
		     :R_F (/ (dot (aref self.sliders (string "R_F")) (value)) 100.0)
		     :max_pos (/ (dot (aref self.sliders (string "max_pos")) (value)) 10.0)
		     :max_force (/ (dot (aref self.sliders (string "max_force")) (value)) 10.0)
		     :h_mpc (/ (dot (aref self.sliders (string "h_mpc")) (value)) 1000.0)
		     :N (int (dot (aref self.sliders (string "N")) (value)))
		     :dt_sim (/ (dot (aref self.sliders (string "dt_sim")) (value)) 1000.0))))
       
     (def update_loop (self)
       (setf params (self.get_params)
	     target_s (/ (dot (aref self.sliders (string "target_s")) (value)) 10.0)
	     target_state (np.array (list target_s 0.0 0.0 0.0)))
	     
       (setf (ntuple u_opt X_pred U_pred t_solve success) (self.mpc.step self.state target_state params)
	     F_motor u_opt
	     wind_force (aref params (string "wind")))
       (setf self.dt (aref params (string "dt_sim")))
       (self.timer.setInterval (int (* self.dt 1000.0)))
	     
       (comments "Physik-Simulation: Runge-Kutta 4 Integrationsschritt mit echter Windstörung")
       (def f_real (st)
	 (setf s_st (aref st 0) v_st (aref st 1) theta_st (aref st 2) omega_st (aref st 3)
	       sin_t (np.sin theta_st) cos_t (np.cos theta_st)
	       den (+ (aref params (string "M")) (* (aref params (string "m")) (- 1.0 (* cos_t cos_t))))
	       F_tot (+ F_motor (* wind_force cos_t))
	       l_val (aref params (string "l"))
	       ds v_st
	       dv (/ (+ F_tot (* (aref params (string "m")) l_val omega_st omega_st sin_t) (* (aref params (string "m")) 9.81 cos_t sin_t)) den)
	       dtheta omega_st
	       domega (/ (- (* -1.0 F_tot cos_t) (* (aref params (string "m")) l_val omega_st omega_st sin_t cos_t) (* (+ (aref params (string "M")) (aref params (string "m"))) 9.81 sin_t)) (* l_val den)))
	 (return (np.array (list ds dv dtheta domega))))
	 
       (setf k1 (f_real self.state)
	     k2 (f_real (+ self.state (* (/ self.dt 2.0) k1)))
	     k3 (f_real (+ self.state (* (/ self.dt 2.0) k2)))
	     k4 (f_real (+ self.state (* self.dt k3)))
	     self.state (+ self.state (np.array (np.squeeze (* (/ self.dt 6.0) (+ k1 (* 2.0 k2) (* 2.0 k3) k4)))))
	     self.time (+ self.time self.dt))
	     
       (dot self.t_hist (append self.time))
       ,@(loop for (key val) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
	       collect `(dot (aref self.hist (string ,key)) (append (aref self.state ,val))))
       (dot (aref self.hist (string "F")) (append F_motor))
       (dot (aref self.hist (string "t_solve")) (append (* t_solve 1000.0)))
       
       (if (> (len self.t_hist) 100)
	   (do0
	    (self.t_hist.pop 0)
	    ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
		    collect `(dot (aref self.hist (string ,key)) (pop 0)))))
		    
       (self.pendulum_widget.update_state self.state F_motor wind_force (aref params (string "l")) (aref params (string "max_pos")))
       
       ,@(loop for key in '("s" "v" "theta" "omega" "F" "t_solve")
	       collect `(dot (aref self.history_curves (string ,key)) (setData self.t_hist (aref self.hist (string ,key)))))
	       
       (if success
	   (do0
	    (setf t_pred (np.linspace self.time (+ self.time self.mpc.T_horizon) (+ self.mpc.N 1)))
	    ,@(loop for (key idx) in '(("s" 0) ("v" 1) ("theta" 2) ("omega" 3))
		    collect `(dot (aref self.pred_curves (string ,key)) (setData t_pred (aref X_pred ,idx (slice nil nil)))))
	    (dot (aref self.pred_curves (string "F")) (setData (aref t_pred (slice 0 -1)) U_pred))))))

   (if (== __name__ (string "__main__"))
       (do0
	(setf app (QApplication sys.argv)
	      win (MainWindow))
	(win.show)
	(sys.exit (app.exec))))
   ))
