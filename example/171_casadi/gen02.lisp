(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

#|
================================================================================
DAS HÉNON-HEILES-SYSTEM IN DER GALAXIENDYNAMIK
================================================================================

1. Historie und Entdeckung:
Im Jahr 1964 untersuchten die Astronomen Michel Hénon und Carl Heiles die Bewegung von Sternen
in einer Galaxie. Ihr Ziel war es herauszufinden, ob neben der Energie (E) und dem Drehimpuls (Lz)
ein "drittes Integral der Bewegung" (eine Erhaltungsgröße) existiert. Mithilfe früher digitaler
Computer und der von ihnen populär gemachten Poincaré-Schnitte zeigten sie, dass regelmäßige
(quasi-periodische) Bahnen bei niedrigen Energien existieren, diese aber bei höheren Energien
in Chaos übergehen.

2. Physikalischer Kontext und Dimensionsreduktion in der Meridianebene:
Eine Meridianebene ist eine Ebene, die das Zentrum der Galaxie sowie die Rotationsachse und
den aktuellen Ort des Sterns enthält. Sie rotiert mit dem Stern mit. Ein dreidimensionales,
axialsymmetrisches galaktisches Potenzial lässt sich dadurch auf eine zweidimensionale Bewegung
in dieser rotierenden Meridianebene reduzieren.
In unserem Modell sind die Koordinaten:
- x: Die radiale Abweichung von einer stabilen kreisförmigen Umlaufbahn im galaktischen Diskus.
- y: Die vertikale Auslenkung senkrecht zur galaktischen Scheibe.
- px, py: Die zugehörigen Impulse (Geschwindigkeiten, da Masse m = 1).
|#

(progn
  (defparameter *source* "example/171_casadi/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p02_hh"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot)))
	  )
     
     #|
     ==========================================================================
     WARUM IST DIE KRAFT IM ZENTRUM DER GALAXIE LINEAR?
     ==========================================================================
     - Taylor-Entwicklung: Um ein stabiles Minimum (das Zentrum) bei r = 0 ist die Kraft F = -\nabla V.
       Da es sich um ein Minimum handelt, verschwindet die erste Ableitung (die Kraft am Ursprung ist 0).
       Der erste nicht-verschwindende Term im Potenzial ist quadratisch: V(r) \approx 1/2 k r^2.
       Die Kraft F \approx -k r ist somit linear (harmonischer Oszillator).
     - Homogener Kern: Astrophysikalisch besitzt das Galaxienzentrum oft eine nahezu konstante Dichte \rho.
       Nach dem Newtonschen Schalentheorem wächst die eingeschlossene Masse M(r) proportional zu r^3.
       Die Gravitationskraft F_g = -G M(r)/r^2 ist daher proportional zu -r (also linear).
     |#
     (def simulate_hh (E y0_list &key (t_max 1000.0) (n_steps 100000))
       (declare (type float E)
		(type list y0_list)
		(type float t_max)
		(type int n_steps)
		(values list))
       ;; SX.sym: "Scalar Expression Symbolic". Erzeugt ein symbolisches Skalar-Primitiv in CasADi.
       (setf x (SX.sym (string "x")))
       (setf y (SX.sym (string "y")))
       (setf px (SX.sym (string "px")))
       (setf py (SX.sym (string "py")))
       
       ;; vertcat: "Vertical Concatenation". Kombiniert die symbolischen Skalare zu einem Zustandsvektor.
       (setf state (vertcat x y px py))
       
       ;; Parameter zur Skalierung der kubischen nichtlinearen Störung (C3-Symmetrie).
       (setf lambda_val 1.0)
       
       ;; Das Hénon-Heiles-Potenzial V(x, y) = 1/2 (x^2 + y^2) + x^2 y - 1/3 y^3.
       (setf V (+ (* 0.5 (+ (** x 2) (** y 2)))
		  (* lambda_val (- (* (** x 2) y)
				   (/ (** y 3) 3.0)))))
       
       ;; Kinetische Energie T_kin = 1/2 (px^2 + py^2).
       (setf T_kin (* 0.5 (+ (** px 2) (** py 2))))
       
       ;; Der Hamiltonian H = T_kin + V repräsentiert die Gesamtenergie E des Sterns.
       (setf H (+ T_kin V))
       
       ;; ode: "Ordinary Differential Equation". Die rechten Seiten der gewöhnlichen
       ;; Differentialgleichungen (Hamiltonsche Bewegungsgleichungen):
       ;; dx/dt = dH/dpx = px
       ;; dy/dt = dH/dpy = py
       ;; dpx/dt = -dH/dx = -dV/dx (mittels jacobian automatisch abgeleitet)
       ;; dpy/dt = -dH/dy = -dV/dy (mittels jacobian automatisch abgeleitet)
       (setf ode (vertcat px
			  py
			  (* -1 (jacobian H x))
			  (* -1 (jacobian H y))))
       
       ;; dae: "Differential-Algebraic Equation". Ein Dictionary, das CasADi übergibt.
       ;; Es enthält den Zustandsvektor 'x' und seine Ableitungen 'ode'.
       (setf dae (dict ((string "x") state)
		       ((string "ode") ode)))
       
       ;; Erzeugt ein lineares Zeitraster für die Simulationspunkte von 0 bis t_max.
       (setf t_grid (np.linspace 0.0 t_max n_steps))
       
       ;; integrator: Erstellt einen Lösungs-Funktor. Wir verwenden "cvodes" aus dem
       ;; Sundials-Paket (impliziter BDF-Löser mit adaptiver Schrittweite).
       (setf F (integrator (string "F") (string "cvodes") dae (dict ((string "grid") t_grid)
								    ((string "output_t0") True))))
       
       ;; Liste zum Speichern der Ergebnisse jeder einzelnen Trajektorie.
       (setf results (list))
       
       ;; Schleife über alle ausgewählten Anfangspositionen y0.
       (for (y0 y0_list)
	    ;; Berechnung des Potenzials am Startpunkt (x0 = 0, y0).
	    (setf V_0 (- (* 0.5 (** y0 2))
			 (/ (** y0 3) 3.0)))
	    
	    ;; Nullgeschwindigkeitskurven-Bedingung:
	    ;; Da die kinetische Energie T_kin = 1/2 (px^2 + py^2) >= 0 sein muss, gilt
	    ;; stets V(x, y) <= E. Die Kurve V(x, y) = E ist die Nullgeschwindigkeitskurve
	    ;; (ZVC), da an dieser Grenze px = py = 0 (die Geschwindigkeit ist null).
	    ;; Bereiche mit V(x, y) > E sind energetisch verboten; der Stern kann diese
	    ;; Grenze nicht überschreiten, da dies eine physikalisch unmögliche negative
	    ;; kinetische Energie (T_kin < 0) erfordern würde. Daher gibt es dort keine Pfade.
	    (if (< V_0 E)
		(do0
		 ;; Bestimmung der radialen Startgeschwindigkeit px0, so dass die Gesamtenergie H exakt E ist.
		 (setf px0 (np.sqrt (* 2 (- E V_0))))
		 
		 ;; Erzeugt das numerische Startwert-Array w0 = [x=0, y=y0, px=px0, py=0].
		 (setf w0 (np.array (list 0.0 y0 px0 0.0)))
		 
		 ;; Führt die CVodes-Integration aus.
		 (setf sol (F :x0 w0))
		 
		 ;; Konvertiert das Ergebnis-Matrix-Objekt in ein flaches NumPy-Array.
		 (setf xf (dot (aref sol (string "xf")) (full)))
		 
		 ;; Extrahiert die zeitlichen Verläufe der einzelnen Phasenraumvariablen.
		 (setf x_arr (aref xf 0 (slice)))
		 (setf y_arr (aref xf 1 (slice)))
		 (setf px_arr (aref xf 2 (slice)))
		 (setf py_arr (aref xf 3 (slice)))
		 
		 ;; Event-Detektion für Poincaré-Schnitt (x = 0, px > 0):
		 ;; Findet Indizes, bei denen die Trajektorie die y-Achse von links nach rechts kreuzt.
		 (setf idx (aref (np.where (& (& (<= (aref x_arr (slice nil -1)) 0)
						 (> (aref x_arr (slice 1 nil)) 0))
					      (> (aref px_arr (slice nil -1)) 0)))
				 0))
		 (setf y_crossings (list))
		 (setf py_crossings (list))
		 
		 ;; Lineare Interpolation für präzise Schnittpunkte bei x = 0.
		 (for (i idx)
		      (setf t_frac (/ (* -1 (aref x_arr i))
				      (- (aref x_arr (+ i 1))
					 (aref x_arr i))))
		      (setf y_cross (+ (aref y_arr i)
				       (* t_frac (- (aref y_arr (+ i 1))
						    (aref y_arr i)))))
		      (setf py_cross (+ (aref py_arr i)
					(* t_frac (- (aref py_arr (+ i 1))
						     (aref py_arr i)))))
		      (y_crossings.append y_cross)
		      (py_crossings.append py_cross))
		 
		 ;; Speichert die Trajektorie und die Schnittpunkte im Ergebnis-Dict.
		 (results.append (dict ((string "y0") y0)
				       ((string "x") x_arr)
				       ((string "y") y_arr)
				       ((string "y_cross") (np.array y_crossings))
				       ((string "py_cross") (np.array py_crossings)))))
		(print (fstring "Initial condition y0={y0} is energetically inaccessible for energy E={E}"))
		))
       (return results))
     
     (do0
      (setf fig_and_axs (plt.subplots 2 2 :figsize (tuple 12 10)))
      (setf (ntuple fig axs) fig_and_axs)
      
      ;; ax_ol / ax_oh: "Axes Orbit Low/High energy". Die linken Plots zeigen die Orbits (x, y)
      ;; in der rotierenden Meridianebene.
      ;; ax_pl / ax_ph: "Axes Poincare Low/High energy". Die rechten Plots zeigen den
      ;; Poincaré-Schnitt (y, py) bei Durchgängen durch x = 0 mit px > 0.
      (setf ax_ol (aref axs 0 0)
	    ax_pl (aref axs 0 1)
	    ax_oh (aref axs 1 0)
	    ax_ph (aref axs 1 1))
      
      ;; Niedriges Energieniveau E_low = 0.08333 (Orbits sind stabil und geordnet).
      (setf E_low 0.08333)
      (setf y0_low (dot (np.linspace -0.25 0.42 12) (tolist)))
      (setf results_low (simulate_hh E_low y0_low :t_max 5000.0 :n_steps 500000))
      
      ;; Hohes Energieniveau E_high = 0.15 (Orbits brechen großteils in Chaos aus).
      (setf E_high 0.15)
      (setf y0_high (dot (np.linspace -0.35 0.55 12) (tolist)))
      (setf results_high (simulate_hh E_high y0_high :t_max 5000.0 :n_steps 500000))
      
      ;; Erzeugt ein 2D-Gitter zur Darstellung der Potenzialkonturen.
      (setf x_g (np.linspace -1.2 1.2 200))
      (setf y_g (np.linspace -1.2 1.2 200))
      (setf (ntuple X Y) (np.meshgrid x_g y_g))
      
      ;; Berechnet das Potenzialfeld Z.
      (setf Z (+ (* 0.5 (+ (** X 2) (** Y 2)))
		 (- (* (** X 2) Y)
		    (/ (** Y 3) 3.0))))
      
      ;; Plotten der Bahnen für niedriges Energieniveau.
      ;; Jede Farbe entspricht einem Orbit, der von einer anderen vertikalen
      ;; Anfangsauslenkung y0 startet. Dadurch lässt sich die Struktur
      ;; der verschiedenen Bahnfamilien optisch unterscheiden.
      (for (res results_low)
	   (dot ax_ol (plot (aref res (string "x")) (aref res (string "y")) :alpha 0.5))
	   (dot ax_pl (scatter (aref res (string "y_cross")) (aref res (string "py_cross")) :s 0.6 :alpha 0.8))
	   )
      ;; Rote gestrichelte Kurve: Nullgeschwindigkeitskurve V(x, y) = E_low.
      (dot ax_ol (contour X Y Z :levels (list E_low) :colors (string "red") :linestyles (string "dashed")))
      
      ;; Plotten der Bahnen für hohes Energieniveau.
      ;; Bei E = 0.15 nimmt die rote Nullgeschwindigkeitskurve die Form eines gleichseitigen
      ;; Dreiecks mit abgerundeten Ecken an. Dies liegt an der C3-Rotationssymmetrie des Potenzials,
      ;; das drei Fluchtpunkte (Sattelpunkte) bei V = 1/6 (~0.1667) besitzt.
      ;; Die Orbits im Inneren präzedieren (drehen sich), wodurch sich die dreieckigen Bahnen
      ;; im Laufe der Simulation drehen und komplexe Rosetten- oder Dreiecksmuster bilden.
      (for (res results_high)
	   (dot ax_oh (plot (aref res (string "x")) (aref res (string "y")) :alpha 0.5))
	   (dot ax_ph (scatter (aref res (string "y_cross")) (aref res (string "py_cross")) :s 0.6 :alpha 0.8))
	   )
      ;; Rote gestrichelte Kurve: Nullgeschwindigkeitskurve V(x, y) = E_high.
      (dot ax_oh (contour X Y Z :levels (list E_high) :colors (string "red") :linestyles (string "dashed")))
      
      ;; Dynamische Formatierung der Subplots mittels Lisp-Loop-Makro.
      ,@(loop for (ax title xl yl eq) in `((ax_ol "Orbits (E = {E_low})" "x" "y" t)
					   (ax_pl "Poincare Section x=0, px>0 (E = {E_low})" "y" "py" nil)
					   (ax_oh "Orbits (E = {E_high})" "x" "y" t)
					   (ax_ph "Poincare Section x=0, px>0 (E = {E_high})" "y" "py" nil))
	      collect
	      `(do0
		,@(when eq `((dot ,ax (set_aspect (string "equal")))))
		(dot ,ax (set_title (fstring ,title)))
		(dot ,ax (set_xlabel (string ,xl)))
		(dot ,ax (set_ylabel (string ,yl)))
		(dot ,ax (grid True))))
      
      (plt.tight_layout)
      (plt.savefig (string "henon_heiles_chaos.png") :dpi 150)
      (print (string "Plot saved to henon_heiles_chaos.png"))
      )
    )
   ))

#|
================================================================================
ASTROPHYSIKALISCHE INTERPRETATION & BEDEUTUNG
================================================================================

1. Struktur des Phasenraums und Erhaltungsgrößen (Invariant Tori):
Bei niedriger Energie (E_low) sind die Bahnen fast vollständig regulär. Die Poincaré-Schnitte zeigen
geschlossene, glatte 1D-Kurven. Dies sind Schnitte durch zweidimensionale Tori (Invariant Tori) im
vierdimensionalen Phasenraum. Die Existenz dieser Tori beweist das Vorhandensein eines "dritten
Integrals der Bewegung". Für echte Galaxien bedeutet dies, dass Sterne auf solchen Orbits stabile,
geordnete Umlaufbahnen einnehmen und ihre Bahnparameter über Jahrmilliarden nicht driften.

2. Zusammenbruch der Stabilität und Übergang zu Chaos:
Bei hoher Energie (E_high) bricht das dritte Integral für die meisten Anfangsbedingungen zusammen.
Die Poincaré-Punkte streuen chaotisch über die Energiehyperfläche ("chaotic sea").
Einige Inseln geordneter Bewegung ("stability islands") bleiben jedoch bestehen. Sie entsprechen
starken Orbitalresonanzen (z.B. 1:1 oder 2:1 Resonanzen zwischen radialer und vertikaler Schwingung).

3. Relevanz für reale Galaxienstrukturen:
- Galaktische Balken (Galactic Bars): In Balkenspiralgalaxien wird die Struktur des Balkens durch
  stabile, nicht-achsen-symmetrische Orbit-Familien (wie die x1-Orbitfamilie) gestützt, die sich
  entlang des Balkens erstrecken. Werden Sterne durch gravitative Störungen (z. B. Vorbeiflüge anderer
  Galaxien oder Gasakkumulation im Zentrum) energiereicher, können ihre Orbits chaotisch werden.
  Dies führt dazu, dass sie den Balken verlassen, sich ungeordnet im Raum verteilen und der Balken
  sich auflöst.
- Säkulare Galaxienevolution & Verdickung von Bulges: Wenn in einer Galaxie das Zentrum an Masse gewinnt
  (z. B. durch das Wachstum eines supermassereichen Schwarzen Lochs oder eines dichten Bulges),
  wird das Potenzial im Zentrum steiler. Dies destabilisiert regelmäßige Bahnen (insb. Box-Orbits)
  und streut Sterne chaotisch in die vertikale Richtung. Die Folge ist eine dynamische Aufheizung
  der galaktischen Scheibe und die Entstehung von kasten- oder erdnussförmigen Wölbungen ("boxy/peanut bulges").
- Chaotische Diffusion: In realen Systemen führt Chaos dazu, dass Sterne sich langsam in ihrem Orbit
  verändern (chaotische Diffusion). Dies moduliert die Verteilung von Sternen im galaktischen Halo
  und der Scheibe über kosmologische Zeitskalen hinweg.
|#
