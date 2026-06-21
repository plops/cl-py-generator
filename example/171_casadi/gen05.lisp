(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

(defun def-sym-vec (vec-name vars)
  "Erstellt eine symbolische CasADi-Vektorvariable mit dem Namen `VEC-NAME`.
   Die Dimension entspricht der Anzahl der übergebenen Variablen in `VARS`.
   Zusätzlich werden die Variablen in `VARS` an die entsprechenden Komponenten des Vektors (über `aref`) gebunden."
  (let ((size (length vars)))
    `(setf ,vec-name (SX.sym (string ,(string-downcase (symbol-name vec-name))) ,size)
           ,@(loop for var in vars
                   for i from 0
                   collect var
                   collect `(aref ,vec-name ,i)))))

(progn
  (defparameter *source* "example/171_casadi/")
  
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p05_rootfinder"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot)))
	  (imports-from (matplotlib.animation FuncAnimation)
			(matplotlib.widgets Slider)))

     (comments "========================================================================"
	       "CASADI SYMBOLIK & ROOTFINDER KONFIGURATION"
	       "========================================================================"
	       "Wir loesen die Kinematik eines ebenen Viergelenkgetriebes (Four-bar linkage)."
	       "Gegeben ist der Kurbelwinkel theta2 (Antrieb). Gesucht sind die Winkel"
	       "theta3 (Koppel) und theta4 (Schwinge)."
	       "Die Schleifenschlussgleichungen lauten:"
	       "  g0: l2 * cos(theta2) + l3 * cos(theta3) - l4 * cos(theta4) - d = 0"
	       "  g1: l2 * sin(theta2) + l3 * sin(theta3) - l4 * sin(theta4) = 0"
	       "Dabei ist:"
	       "  z = [theta3, theta4] (die Unbekannten)"
	       "  x = [theta2, d, l2, l3, l4] (die Parameter)")

     ,(def-sym-vec 'z '(theta3 theta4))
     ,(def-sym-vec 'x '(theta2 d l2 l3 l4))

     (setf g0 (+ (- (* l2 (cos theta2)) (* l4 (cos theta4)))
		 (* l3 (cos theta3))
		 (* -1 d))
	   g1 (+ (- (* l2 (sin theta2)) (* l4 (sin theta4)))
		 (* l3 (sin theta3))))

     (setf g (Function (string "g") (list z x) (list (vertcat g0 g1))))

     ;; newton solver instanziieren. error_on_fail=False stellt sicher, dass das
     ;; Skript bei Singularitaeten (z.B. unmoeglichen Geometrien) nicht abstuerzt.
     (setf G (rootfinder (string "G")
			 (string "newton")
			 g
			 (dict ((string "error_on_fail") False))))

     (comments "========================================================================"
	       "HILFSFUNKTION FUER DIE KINEMATIK"
	       "========================================================================"
	       "Loest das Gleichungssystem fuer einen gegebenen Zustand."
	       "Bei Konvergenzfehlern wird der letzte Schätzwert zurückgegeben.")
     (def solve_kinematics (theta2_val d_val l2_val l3_val l4_val z_guess_val)
       (setf x_val (list theta2_val d_val l2_val l3_val l4_val))
       (try
	(do0
	 (setf sol (G z_guess_val x_val))
	 (return (dot (np.array sol) (flatten))))
	((as Exception e)
	 (return z_guess_val))))

     (comments "========================================================================"
	       "INITIALISIERUNG DES MATPLOTLIB AXIS & WIDGETS"
	       "========================================================================")
     (setf (ntuple fig ax) (plt.subplots :figsize (tuple 8 8)))
     (plt.subplots_adjust :bottom 0.3)

     (dot ax (set_xlim -4 7))
     (dot ax (set_ylim -5 5))
     (dot ax (set_aspect (string "equal")))
     (dot ax (grid True))

     ;; Plot-Elemente anlegen
     ,@(loop for (var style color label lw ms alpha) in
             '((line_crank "o-" "royalblue" "Kurbel (l2)" 4 nil nil)
               (line_coupler "o-" "forestgreen" "Koppel (l3)" 4 nil nil)
               (line_rocker "o-" "crimson" "Schwinge (l4)" 4 nil nil)
               (line_ground "o--" "gray" "Gestell (d)" 2 nil nil)
               (point_P "ro" nil "Koppelpunkt P" nil 8 nil)
               (line_locus "r-" nil "Koppelkurve (Locus)" 2 nil 0.6))
             collect
             `(setf ,var (aref (dot ax (plot (list) (list) (string ,style)
                                            ,@(when lw `(:lw ,lw))
                                            ,@(when ms `(:ms ,ms))
                                            ,@(when alpha `(:alpha ,alpha))
                                            ,@(when color `(:color (string ,color)))
                                            ,@(when label `(:label (string ,label)))))
                               0)))

     ;; Gelenksymbole fuer O2 und O4
     (dot ax (plot (list 0.0) (list 0.0) (string "ks") :ms 10 :zorder 5))
     (setf point_O4 (aref (dot ax (plot (list) (list) (string "ks") :ms 10 :zorder 5)) 0))

     (dot ax (legend :loc (string "upper right")))

     (comments "========================================================================"
	       "SLIDER-GUI FOR PARAMETER CONTROL"
	       "========================================================================")
     ;; Slider-Achsen und Slider-Objekte anlegen
     ,@(loop for (var label y-pos min-val max-val init-val) in
             '((slider_d "d (Gestell)" 0.20 1.0 6.0 3.0)
               (slider_l2 "l2 (Kurbel)" 0.15 0.2 3.0 1.0)
               (slider_l3 "l3 (Koppel)" 0.10 1.0 6.0 3.0)
               (slider_l4 "l4 (Schwinge)" 0.05 0.5 6.0 2.5))
             collect
             (let ((ax-var (intern (format nil "ax_~a" var))))
               `(do0
                  (setf ,ax-var (plt.axes (list 0.15 ,y-pos 0.65 0.03)))
                  (setf ,var (Slider ,ax-var (string ,label) ,min-val ,max-val :valinit ,init-val)))))

     (comments "========================================================================"
	       "GLOBALE ZUSTAENDE FUER SCHAETZWERTE UND LOKUS-SPUR"
	       "========================================================================")
     (setf last_z (np.array (list (/ np.pi 4) (/ np.pi 2)))
	   locus_x (list)
	   locus_y (list)
	   theta2_vals (np.linspace 0.0 (* 2 np.pi) 200))

     (comments "========================================================================"
	       "ANIMATIONS-UPDATE FUNKTION"
	       "========================================================================"
	       "Wird fuer jeden Frame aufgerufen. Liest die Slider-Werte und loest die"
	       "Kinematik fuer den aktuellen Kurbelwinkel theta2 auf.")
     (def update (frame)
       (space global last_z)
       (setf d_val (dot slider_d val)
	     l2_val (dot slider_l2 val)
	     l3_val (dot slider_l3 val)
	     l4_val (dot slider_l4 val))

       (setf theta2_val (aref theta2_vals frame))

       ;; Loese die Gleichungen mit Warm-Start (Schätzung vom letzten Schritt)
       (setf sol (solve_kinematics theta2_val d_val l2_val l3_val l4_val last_z))
       (setf last_z sol)

       (setf theta3_val (aref sol 0)
	     theta4_val (aref sol 1))

       ;; Koordinaten berechnen
       (setf x_O2 0.0
	     y_O2 0.0)
       (setf x_A (* l2_val (np.cos theta2_val))
	     y_A (* l2_val (np.sin theta2_val)))
       (setf x_B (+ d_val (* l4_val (np.cos theta4_val)))
	     y_B (* l4_val (np.sin theta4_val)))
       (setf x_O4 d_val
	     y_O4 0.0)

       ;; Koppelpunkt P (bildet ein Dreieck auf dem Koppelglied AB)
       ;; Abstand l_AP = 1.5, Winkelversatz alpha = 0.5 rad
       (setf l_AP 1.5
	     alpha 0.5)
       (setf x_P (+ x_A (* l_AP (np.cos (+ theta3_val alpha))))
	     y_P (+ y_A (* l_AP (np.sin (+ theta3_val alpha)))))

       ;; Geometrien aktualisieren
       (dot line_crank (set_data (list x_O2 x_A) (list y_O2 y_A)))
       (dot line_coupler (set_data (list x_A x_B) (list y_A y_B)))
       (dot line_rocker (set_data (list x_B x_O4) (list y_B y_O4)))
       (dot line_ground (set_data (list x_O2 x_O4) (list y_O2 y_O4)))
       (dot point_O4 (set_data (list x_O4) (list y_O4)))
       (dot point_P (set_data (list x_P) (list y_P)))

       ;; Locus aufzeichnen
       (dot locus_x (append x_P))
       (dot locus_y (append y_P))

       ;; Verhindere unendliches Anwachsen des Locus-Speichers
       (if (> (len locus_x) (len theta2_vals))
	   (do0
	    (dot locus_x (pop 0))
	    (dot locus_y (pop 0))))

       (dot line_locus (set_data locus_x locus_y))

       (return (tuple line_crank line_coupler line_rocker line_ground point_O4 point_P line_locus)))

     (comments "========================================================================"
	       "SLIDER EVENT-HANDLER"
	       "========================================================================"
	       "Löscht die Koppelkurve und setzt die Startschätzung zurück, wenn"
	       "der Benutzer die Geometrie ändert.")
     (def on_slider_change (val)
       (space global locus_x)
       (space global locus_y)
       (space global last_z)
       (dot locus_x (clear))
       (dot locus_y (clear))
       (setf last_z (np.array (list (/ np.pi 4) (/ np.pi 2)))))

     ,@(loop for slider in '(slider_d slider_l2 slider_l3 slider_l4)
             collect `(dot ,slider (on_changed on_slider_change)))

     ;; Animation starten
     (setf ani (FuncAnimation fig update :frames (len theta2_vals) :interval 20 :blit True))
     (plt.show())
     )))
