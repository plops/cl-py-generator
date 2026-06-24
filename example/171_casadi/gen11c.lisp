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
   (imports-from (PySide6.QtWidgets QApplication QMainWindow QWidget QVBoxLayout QHBoxLayout QSlider QLabel QGridLayout QTabWidget QPushButton)
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
     (def __init__ (self &key (T_horizon 1.0) (N 20))
       (setf self.opti (Opti)
	     self.nx 4
	     self.nu 1
	     self.N N
	     self.T_horizon T_horizon
	     self.h (/ self.T_horizon self.N)
	     self.d 3)

       (comments 
        " Parameter für Physik und Optimierung."
        " Wir definieren diese als CasADi 'Parameter' (opti.parameter), anstatt sie fest zu"
        " verdrahten. Das ermöglicht es uns, Massen, Wind, Grenzen oder auch den Vorhersagehorizont"
        " (T_horizon) zur Laufzeit der GUI zu ändern, ohne den kompletten CasADi Optimierungs-"
        " Graphen neu aufbauen und kompilieren zu müssen (was extrem rechenintensiv wäre).")
       ,@(loop for sym in '(M_p m_p l_p wind_p Q_s Q_v Q_theta Q_omega R_F max_pos max_force)
               collect `(setf (dot self ,sym) (self.opti.parameter)))

       (comments "Symbolische Variablen für die Dynamik (dx/dt = f(x,u,p)).")
       (comments "Wir nutzen SX (Scalar Expression) anstelle von MX (Matrix Expression) für die ODE.")
       (comments "SX ist für solche mathematischen Ausdrücke auf Skalarebene deutlich effizienter bei der")
       (comments "Berechnung von Ableitungen (Jacobian/Hessian) innerhalb des NLP Solvers.")
       (comments "Um Fehler durch das Mischen von SX und den MX-Parametern des Opti-Stacks zu vermeiden,")
       (comments "übergeben wir die physikalischen Parameter explizit als SX-Vektor an die ODE-Funktion.")
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

       (comments "Lagrange Collocation Polynome via Radau-Punkten berechnen")
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
		 (self.opti.subject_to (== xp (* self.h f_eval))))
	    (for (r (range self.d))
		 (setf x_end (+ x_end (* (aref self.D (+ r 1)) (aref (aref self.Xc k) r)))))
	    (self.opti.subject_to (== (aref self.X (slice nil nil) (+ k 1)) x_end)))

       (comments "Kostenfunktion (Objective): Abweichung vom Ziel und Stellenergie minimieren")
       (setf cost 0.0
	     Q (diag (vertcat self.Q_s self.Q_v self.Q_theta self.Q_omega)))
       (for (k (range self.N))
	    (setf err (- (aref self.X (slice nil nil) k) self.target_x)
		  cost (+ cost (+ (mtimes (mtimes err.T Q) err)
				  (* self.R_F (** (aref self.U 0 k) 2))))))
       (setf err_term (- (aref self.X (slice nil nil) self.N) self.target_x)
	     cost (+ cost (* 10.0 (mtimes (mtimes err_term.T Q) err_term))))
       (self.opti.minimize cost)

       (comments "IPOPT Konfiguration (ohne JIT Komplexität, Python API reicht aus)")
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
		     
       (setf self.tabs (QTabWidget))
       (left_layout.addWidget self.tabs)

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

       (setf self.sliders (dict))
       
       (def add_slider (layout name label min_val max_val default_val row tooltip)
	 (setf slider (QSlider Qt.Horizontal)
	       lbl (QLabel (fstring "{label}: {default_val}")))
	 (slider.setToolTip tooltip)
	 (lbl.setToolTip tooltip)
	 (slider.setMinimum min_val)
	 (slider.setMaximum max_val)
	 (slider.setValue default_val)
	 (layout.addWidget lbl row 0)
	 (layout.addWidget slider row 1)
	 (slider.valueChanged.connect (lambda () (lbl.setText (fstring "{label}: {slider.value()}"))))
	 (setf (aref self.sliders name) slider))

       ,@(loop for (tab-name sliders) in
               '(("Physik"
                  (("M" "Wagenmasse [kg]" 1 50 10 "Masse des Wagens. Schwerer = Träger gegen Bewegungsänderungen.")
                   ("m" "Pendelmasse [kg]" 1 20 1 "Punktmasse des Pendels am Kopfende.")
                   ("l" "Pendellänge [m]" 1 20 5 "Abstand vom Wagen zum Pendelschwerpunkt (Wert/10 in m).")
                   ("wind" "Windkraft [N]" -300 300 0 "Konstante Störkraft, die horizontal auf das Pendel drückt.")))
                 ("Kostenfunktion"
                  (("Q_s" "Gewicht Position" 0 200 10 "Strafe (Penalty) für Abweichung des Wagens von der Ziel-Position.")
                   ("Q_v" "Gewicht Wagengeschw." 0 100 10 "Strafe für hohe Geschwindigkeit des Wagens (verhindert Überschwingen).")
                   ("Q_theta" "Gewicht Pendelwinkel" 0 500 100 "Strafe für das Abweichen des Pendels vom instabilen Gleichgewicht (0 rad).")
                   ("Q_omega" "Gewicht Winkelgeschw." 0 100 10 "Strafe für schnelle Rotationen des Pendels.")
                   ("R_F" "Gewicht Kraftaufwand" 1 200 10 "Kostenfaktor für die Stellkraft F (Wert/100). Zwingt den Solver, Energie zu sparen.")))
                 ("Grenzen & Ziel"
                  (("target_s" "Ziel-Position [m]" -20 20 10 "Soll-Position des Wagens auf der Schiene (Wert/10).")
                   ("max_pos" "Schiene Limit [m]" 10 100 20 "Maximaler erlaubter Fahrweg (Constraint). Der Solver darf diesen nie überschreiten (Wert/10).")
                   ("max_force" "Max Kraft [N]" 10 300 150 "Stellgrößenbeschränkung für den Aktuator. (Wert/10).")))
                 ("Simulation & MPC"
                  (("T_horizon" "Zeithorizont [s]" 1 50 10 "Wie weit blickt die MPC in die Zukunft? (Wert/10). Längerer Horizont plant besser, erfordert aber komplexere Trajektorien.")
                   ("N" "Knotenpunkte (N)" 5 50 20 "Auflösung des Solvers. Mehr Knoten = präzisere Planung, aber langsamer.")
                   ("dt_sim" "Simulations-dt [ms]" 1 100 33 "Schrittweite der echten Runge-Kutta Physiksimulation. Hat keinen Einfluss auf den Solver."))))
               collect
               `(do0
                 (setf tab_widget (QWidget)
                       tab_layout (QGridLayout tab_widget))
                 (self.tabs.addTab tab_widget (string ,tab-name))
                 ,@(loop for (key label min-val max-val def-val tooltip) in sliders
                         for row from 0
                         collect
                         `(add_slider tab_layout (string ,key) (string ,label) ,min-val ,max-val ,def-val ,row (string ,tooltip)))))
       
       (setf self.timer (QTimer))
       (self.timer.timeout.connect self.update_loop)
       (self.reset_sim)
       (self.start_sim))

     (def start_sim (self)
       (setf self.dt (/ (dot (aref self.sliders (string "dt_sim")) (value)) 1000.0))
       (self.timer.start (int (* self.dt 1000.0)))
       (dot (aref self.sliders (string "T_horizon")) (setEnabled False))
       (dot (aref self.sliders (string "N")) (setEnabled False)))

     (def stop_sim (self)
       (self.timer.stop)
       (dot (aref self.sliders (string "T_horizon")) (setEnabled True))
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
       (setf self.mpc (PendulumMPC :T_horizon (aref params (string "T_horizon"))
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
		     :T_horizon (/ (dot (aref self.sliders (string "T_horizon")) (value)) 10.0)
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
	       ds v_st
	       dv (/ (+ F_tot (* (aref params (string "m")) 0.5 omega_st omega_st sin_t) (* (aref params (string "m")) 9.81 cos_t sin_t)) den)
	       dtheta omega_st
	       domega (/ (- (* -1.0 F_tot cos_t) (* (aref params (string "m")) 0.5 omega_st omega_st sin_t cos_t) (* (+ (aref params (string "M")) (aref params (string "m"))) 9.81 sin_t)) (* 0.5 den)))
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
