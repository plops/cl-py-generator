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

2. Physikalischer Kontext und Dimensionsreduktion:
Ein dreidimensionales, axialsymmetrisches galaktisches Potenzial lässt sich auf eine 2D-Bewegung
in der Meridianebene (r, z) reduzieren. Die Koordinaten (x, y) beschreiben die Bewegung des Sterns
in dieser rotierenden Meridianebene, während px und py die zugehörigen Impulse (radialer und vertikaler
Impuls in der Ebene) darstellen.
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

     ==========================================================================
     DIE INTEGRATIONSMETHODE UND EVENT-DETEKTION
     ==========================================================================
     Wir verwenden CasADis built-in Integrator mit der CVodes-Bibliothek. Der Integrator
     berechnet die Trajektorien über ein dichtes Zeitraster. Um die Durchgänge durch
     die Poincaré-Schnittfläche (x = 0, px > 0) präzise zu ermitteln, führt das Skript
     eine lineare Interpolation zwischen den Zeitschritten durch, bei denen sich das
     Vorzeichen von x ändert.
     |#
     (def simulate_hh (E y0_list &key (t_max 1000.0) (n_steps 100000))
       (declare (type float E)
		(type list y0_list)
		(type float t_max)
		(type int n_steps)
		(values list))
       (setf x (SX.sym (string "x"))
	     y (SX.sym (string "y"))
	     px (SX.sym (string "px"))
	     py (SX.sym (string "py"))
	     state (vertcat x y px py)
	     lambda_val 1.0
	     V (+ (* 0.5 (+ (** x 2) (** y 2)))
		  (* lambda_val (- (* (** x 2) y)
				   (/ (** y 3) 3.0)))))
       (setf T_kin (* 0.5 (+ (** px 2) (** py 2))))
       (setf H (+ T_kin V))
       (setf ode (vertcat px
			  py
			  (* -1 (jacobian H x))
			  (* -1 (jacobian H y))))
       (setf dae (dict ((string "x") state)
		       ((string "ode") ode)))
       (setf t_grid (np.linspace 0.0 t_max n_steps))
       (setf F (integrator (string "F") (string "cvodes") dae (dict ((string "grid") t_grid)
								    ((string "output_t0") True))))
       (setf results (list))
       (for (y0 y0_list)
	    (setf V_0 (- (* 0.5 (** y0 2))
			 (/ (** y0 3) 3.0)))
	    (if (< V_0 E)
		(do0
		 (setf px0 (np.sqrt (* 2 (- E V_0)))
		       w0 (np.array (list 0.0 y0 px0 0.0))
		       sol (F :x0 w0)
		       xf (dot (aref sol (string "xf")) (full))
		       x_arr (aref xf 0 (slice))
		       y_arr (aref xf 1 (slice))
		       px_arr (aref xf 2 (slice))
		       py_arr (aref xf 3 (slice))
		       idx (aref (np.where (& (& (<= (aref x_arr (slice nil -1)) 0)
						 (> (aref x_arr (slice 1 nil)) 0))
					      (> (aref px_arr (slice nil -1)) 0)))
				 0)
		       y_crossings (list)
		       py_crossings (list))
		 (for (i idx)
		      (setf t_frac (/ (* -1 (aref x_arr i))
				      (- (aref x_arr (+ i 1))
					 (aref x_arr i)))
			    y_cross (+ (aref y_arr i)
				       (* t_frac (- (aref y_arr (+ i 1))
						    (aref y_arr i)))))
		      (setf py_cross (+ (aref py_arr i)
					(* t_frac (- (aref py_arr (+ i 1))
						     (aref py_arr i)))))
		      (y_crossings.append y_cross)
		      (py_crossings.append py_cross))
		 (results.append (dict ((string "y0") y0)
				       ((string "x") x_arr)
				       ((string "y") y_arr)
				       ((string "y_cross") (np.array y_crossings))
				       ((string "py_cross") (np.array py_crossings)))))
		(print (fstring "Initial condition y0={y0} is energetically inaccessible for energy E={E}"))
		))
       (return results))
     
     #|
     ==========================================================================
     SIMULATION UND VISUALISIERUNG
     ==========================================================================
     Wir simulieren Orbits bei zwei Energieniveaus:
     - E_low = 0.08333 (quasi-integrabler Bereich, geordnete Bewegung)
     - E_high = 0.15 (chaotischer Bereich nahe der Fluchtenergie E_escape = 0.1667)
     
     Die Nullgeschwindigkeitskurve (ZVC) wird über Matplotlib contour eingezeichnet, um
     die energetischen Grenzen der Bewegung in der Galaxie aufzuzeigen.
     |#
     (do0
      (setf fig_and_axs (plt.subplots 2 2 :figsize (tuple 12 10))
	    (ntuple fig axs) fig_and_axs
	    
	    ax_ol (aref axs 0 0)
	    ax_pl (aref axs 0 1)
	    ax_oh (aref axs 1 0)
	    ax_ph (aref axs 1 1)
	    
	    E_low 0.08333
	    y0_low (dot (np.linspace -0.25 0.42 12) (tolist))
	    results_low (simulate_hh E_low y0_low :t_max 5000.0 :n_steps 500000)
	    
	    E_high 0.15
	    y0_high (dot (np.linspace -0.35 0.55 12) (tolist))
	    results_high (simulate_hh E_high y0_high :t_max 5000.0 :n_steps 500000)
	    
	    x_g (np.linspace -1.2 1.2 200)
	    y_g (np.linspace -1.2 1.2 200)
	    (ntuple X Y) (np.meshgrid x_g y_g)
	    Z (+ (* 0.5 (+ (** X 2) (** Y 2)))
		 (- (* (** X 2) Y)
		    (/ (** Y 3) 3.0))))
      
      ;; Plot low energy orbits & Poincaré
      (for (res results_low)
	   (dot ax_ol (plot (aref res (string "x")) (aref res (string "y")) :alpha 0.5))
	   (dot ax_pl (scatter (aref res (string "y_cross")) (aref res (string "py_cross")) :s 0.6 :alpha 0.8))
	   )
      ;; Zero velocity curve for low energy
      (dot ax_ol (contour X Y Z :levels (list E_low) :colors (string "red") :linestyles (string "dashed")))
      
      ;; Plot high energy orbits & Poincaré
      (for (res results_high)
	   (dot ax_oh (plot (aref res (string "x")) (aref res (string "y")) :alpha 0.5))
	   (dot ax_ph (scatter (aref res (string "y_cross")) (aref res (string "py_cross")) :s 0.6 :alpha 0.8))
	   )
      ;; Zero velocity curve for high energy
      (dot ax_oh (contour X Y Z :levels (list E_high) :colors (string "red") :linestyles (string "dashed")))
      
      ;; Formatting plots dynamically using a loop macro
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
vierdimensionalen Phasenraum. Die Existenz dieser Tori beweist das Vorhandensein eines "drittes
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
