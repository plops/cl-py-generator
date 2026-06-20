(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(defpackage #:g
  (:use #:cl #:cl-py-generator)) 

(in-package #:g)

#|
================================================================================
INTERPLANETARER TRANSFER: ERDE ZU MARS — DAS PORK-CHOP-DIAGRAMM
================================================================================

1. Physikalischer Hintergrund:
Ein interplanetarer Transfer von der Erde zum Mars erfordert mindestens zwei
impulsive Manöver: einen Abflug-Burn (Δv₁) beim Verlassen der Erdumlaufbahn und
einen Ankunfts-Burn (Δv₂) zur Einbremsung in die Marsumlaufbahn. Die Summe
Δv_total = Δv₁ + Δv₂ ist ein Maß für den Treibstoffbedarf der Mission.

2. Das Lambert-Problem:
Gegeben zwei Punkte im Raum (Abflugposition r₁, Ankunftsposition r₂) und eine
Transferzeit Δt, sucht das Lambert-Problem die Keplersche Transferbahn, die r₁ mit
r₂ in der Zeit Δt verbindet. Die Lösung liefert die benötigten Geschwindigkeiten
an beiden Endpunkten, woraus Δv₁ und Δv₂ berechnet werden.

3. Das Pork-Chop-Diagramm:
Variiert man systematisch das Abflugdatum und die Transferdauer, erhält man eine
2D-Karte des Treibstoffbedarfs. Dieses Konturdiagramm heißt in der Astrodynamik
"Pork-Chop-Diagramm", weil die Isolinien oft an eine Kotelett-Form erinnern.
Die Minima dieser Karte zeigen die optimalen Startfenster für Mars-Missionen.

4. CasADi-Nutzung:
Hier verwenden wir CasADi, um das Lambert-Problem als nichtlineares Optimierungsproblem
(NLP) zu formulieren:
- Entscheidungsvariablen: Abfluggeschwindigkeit (vx₀, vy₀)
- Nebenbedingung: Nach Integration der Keplerschen Bewegungsgleichungen über Δt
  muss die Endposition die Marsposition treffen
- Zielfunktion: Minimiere Δv_total = |v₀ - v_Erde| + |v_f - v_Mars|
Der CVODES-Integrator berechnet die Bahnpropagation, und IPOPT löst das NLP.
CasADis automatische Differentiation liefert exakte Gradienten für den Solver.

5. Parallelisierung:
Da jedes (Abflugdatum, Transferdauer)-Paar ein unabhängiges NLP ist, lässt sich
die Berechnung trivial mit multiprocessing.Pool parallelisieren.
|#

(progn
  (defparameter *source* "example/171_casadi/")
 
  (write-source
   (asdf:system-relative-pathname 'cl-py-generator
				  (merge-pathnames #P"p03_porkchop"
						   *source*))
   `(do0
     (do0 (imports-from (__future__ annotations)
			(casadi *))
	  (imports ((np numpy)
		    (plt matplotlib.pyplot)
		    (mp multiprocessing)
		    (time time))))
     
     (comments "========================================================================"
	       "PHYSIKALISCHE KONSTANTEN UND ORBITALELEMENTE"
	       "========================================================================"
	       ""
	       "Einheitensystem: Astronomische Einheiten (AU), Jahre (yr), Sonnenmassen."
	       "In diesen Einheiten gilt mu_Sonne = 4*pi^2 AU^3/yr^2 (aus dem dritten"
	       "Keplerschen Gesetz: T^2 = (4*pi^2 / GM) * a^3, mit T_Erde = 1 yr,"
	       "a_Erde = 1 AU)."
	       ""
	       "Kreisbahnnaehrung: Fuer dieses Demonstrationsbeispiel werden die Bahnen"
	       "von Erde und Mars als koplanare Kreisbahnen approximiert. Die wahren"
	       "Bahnen haben kleine Exzentrizitaeten (e_Erde = 0.017, e_Mars = 0.093)"
	       "und eine relative Inklination von ~1.85 Grad, was die Ergebnisse nur"
	       "leicht veraendert.")
     (setf MU_SUN (* 4.0 (** np.pi 2)))
     
     ;; Umrechnungsfaktoren fuer die Darstellung in physikalischen Einheiten.
     ;; 1 AU/yr = 1.496e8 km / (365.25 * 24 * 3600 s) ≈ 4.7404 km/s.
     (setf AU_KM 1.496e8
	   YR_S (* 365.25 24 3600)
	   V_CONV (/ AU_KM YR_S))
     
     ;; Orbitalelemente der Erde und des Mars (Kreisbahn-Approximation).
     ;; a: Halbachse in AU, T: Umlaufperiode in Jahren.
     (setf A_EARTH 1.0
	   T_EARTH 1.0
	   A_MARS  1.524
	   T_MARS  1.881)
     
     (comments "========================================================================"
	       "HILFSFUNKTION: PLANETENPOSITION AUF KREISBAHN"
	       "========================================================================")
     (def planet_state (t_yr a period)
       (declare (type float t_yr)
		(type float a)
		(type float period)
		(values tuple))
       (string3 "Berechnet Position und Geschwindigkeit eines Planeten auf einer Kreisbahn.

    Die Kreisbahngeschwindigkeit ist v = 2*pi*a/T (Umfang / Periode).
    Die Winkelgeschwindigkeit ist omega = 2*pi/T.
    Zum Zeitpunkt t steht der Planet beim Winkel theta = omega * t.

    Parameter:
        t_yr: Zeit in Jahren (Epoche relativ zum Referenzzeitpunkt)
        a: Halbachse der Kreisbahn in AU
        period: Umlaufperiode in Jahren

    Rueckgabe:
        (x, y, vx, vy) in AU und AU/yr")
       (setf omega (/ (* 2 np.pi) period)
	     theta (* omega t_yr)
	     x (* a (np.cos theta))
	     y (* a (np.sin theta))
	     v (/ (* 2 np.pi a) period)
	     vx (* -1 v (np.sin theta))
	     vy (* v (np.cos theta)))
       (return (tuple x y vx vy)))
     
     (comments "========================================================================"
	       "CASADI-INTEGRATOR: KEPLERSCHE ZWEIKÖRPERDYNAMIK"
	       "========================================================================"
	       ""
	       "Die Bewegungsgleichungen im heliozentrisch-kartesischen System lauten:"
	       "  d^2 r / dt^2 = -mu * r / |r|^3"
	       ""
	       "Als System erster Ordnung mit Zustand s = [x, y, vx, vy]:"
	       "  dx/dt  = vx"
	       "  dy/dt  = vy"
	       "  dvx/dt = -mu * x / r^3"
	       "  dvy/dt = -mu * y / r^3"
	       ""
	       "Zeitskalierungstrick: Um die Transferzeit als Parameter (nicht als feste"
	       "Integrationsgrenze) behandeln zu koennen, substituieren wir tau = t/TOF."
	       "Dann integrieren wir immer von tau=0 bis tau=1, und die ODE wird mit dem"
	       "Faktor TOF skaliert: ds/d(tau) = TOF * f(s)."
	       "Dies ermoeglicht es, einen einzigen Integrator fuer alle Transferzeiten"
	       "wiederzuverwenden.")
     (def create_integrator ()
       (declare (values object))
       (string3 "Erzeugt einen CasADi-CVODES-Integrator fuer die Keplersche Bahnmechanik.

    Der Integrator propagiert den Zustandsvektor [x, y, vx, vy] von tau=0 bis tau=1,
    wobei die physikalische Transferzeit TOF als Parameter uebergeben wird.

    CVODES (aus dem SUNDIALS-Paket) verwendet ein implizites BDF-Verfahren
    (Backward Differentiation Formula) mit adaptiver Schrittweitensteuerung.
    Es liefert automatisch exakte Ableitungen der Loesung nach den Anfangswerten
    und Parametern (Sensitivitaetsanalyse), die CasADi fuer die NLP-Gradienten nutzt.")
       ;; SX.sym: Erzeugt symbolische Skalarprimitive fuer CasADis Expression-Graph.
       (setf x  (SX.sym (string "x"))
	     y  (SX.sym (string "y"))
	     vx (SX.sym (string "vx"))
	     vy (SX.sym (string "vy")))
       ;; vertcat: Verkettet die Skalare zu einem 4x1-Zustandsvektor.
       (setf state (vertcat x y vx vy))
       ;; Die Transferzeit ist ein Parameter, kein Zustand.
       (setf tof (SX.sym (string "tof")))
       ;; Abstand zum Gravitationszentrum (Sonne am Ursprung).
       (setf r (sqrt (+ (** x 2) (** y 2))))
       ;; Zeitskalierte ODE: ds/d(tau) = TOF * [vx, vy, -mu*x/r^3, -mu*y/r^3]
       (setf ode (* tof (vertcat vx
				  vy
				  (/ (* -1 MU_SUN x)
				     (** r 3))
				  (/ (* -1 MU_SUN y)
				     (** r 3)))))
       ;; DAE-Dictionary: 'x' = Zustand, 'p' = Parameter, 'ode' = rechte Seite.
       (setf dae (dict ((string "x") state)
		       ((string "p") tof)
		       ((string "ode") ode)))
       ;; Erzeugt den Integrator. t0=0, tf=1 (normierte Zeit).
       ;; reltol/abstol: Relative/absolute Fehlertoleranz fuer CVODES.
       (setf F (integrator (string "F")
			   (string "cvodes")
			   dae
			   (dict ((string "t0") 0)
				 ((string "tf") 1)
				 ((string "reltol") 1e-10)
				 ((string "abstol") 1e-12))))
       (return F))
     
     (comments "========================================================================"
	       "LAMBERT-LOESER: NLP-FORMULIERUNG MIT CASADI OPTI STACK"
	       "========================================================================"
	       ""
	       "Fuer jedes Paar (Abflugdatum, Transferdauer) wird ein kleines NLP geloest:"
	       "  Entscheidungsvariablen: v0 = [vx0, vy0] (Anfangsgeschwindigkeit)"
	       "  Nebenbedingungen: Endposition = Marsposition (2 Gleichungen)"
	       "  Zielfunktion: min Delta-v_total = |v0 - v_Erde| + |v_f - v_Mars|"
	       ""
	       "Da die 2 Nebenbedingungen die 2 Variablen eindeutig festlegen (Lambert-"
	       "Theorem), ist dies effektiv ein Gleichungssystem. Die Zielfunktion dient"
	       "als Regularisierung und berechnet gleichzeitig den Treibstoffbedarf.")
     (def solve_lambert_single (args)
       (declare (values float))
       (string3 "Loest das Lambert-Problem fuer ein einzelnes (t_dep, tof)-Paar.

    Diese Funktion wird von multiprocessing.Pool.map aufgerufen und muss daher
    den Integrator selbst erzeugen (CasADi-Objekte koennen nicht zwischen
    Prozessen serialisiert/gepickelt werden).

    args: Tuple (t_dep, tof) mit:
        t_dep: Abflugzeitpunkt in Jahren
        tof:   Transferzeit in Jahren

    Rueckgabe:
        Delta-v total in km/s, oder np.nan falls die Optimierung fehlschlaegt.")
       (setf (ntuple t_dep tof) args)
       ;; Integrator im Workerprozess erzeugen (einmalig pro Aufruf, ~1 ms Overhead).
       (setf F (create_integrator))
       ;; Planetenpositionen berechnen.
       (setf (ntuple x_E y_E vx_E vy_E) (planet_state t_dep A_EARTH T_EARTH)
	     (ntuple x_M y_M vx_M vy_M) (planet_state (+ t_dep tof) A_MARS T_MARS))

       ;; Transferwinkel pruefen: Wenn Abflug- und Ankunftsposition zu nahe
       ;; beieinander liegen (Transfer nahe 0 oder 2*pi), ist das Problem
       ;; schlecht konditioniert.
       (setf dx (- x_M x_E)
	     dy (- y_M y_E)
	     dist (np.sqrt (+ (** dx 2) (** dy 2))))
       (when (< dist 0.01)
	 (return np.nan))

       ;; --- CasADi Opti Stack: NLP-Formulierung ---
       ;; Opti() erzeugt einen neuen Optimierungsproblem-Container.
       (setf opti (casadi.Opti))

       ;; Entscheidungsvariable: Abfluggeschwindigkeit [vx0, vy0] in AU/yr.
       (setf v0 (opti.variable 2))

       ;; Anfangszustand: Erdposition + unbekannte Geschwindigkeit.
       (setf x0 (vertcat x_E y_E (aref v0 0) (aref v0 1)))

       ;; Bahnpropagation mit dem CVODES-Integrator.
       ;; F(x0=..., p=tof) propagiert den Zustand ueber die normierte Zeit [0,1],
       ;; was physikalisch der Transferzeit tof entspricht.
       (setf result (F :x0 x0 :p tof)
	     xf (aref result (string "xf")))

       ;; Nebenbedingungen: Die Endposition muss die Marsposition treffen.
       ;; subject_to: Fuegt eine Gleichheitsnebenbedingung zum NLP hinzu.
       (opti.subject_to (== (aref xf 0) x_M))
       (opti.subject_to (== (aref xf 1) y_M))

       ;; Zielfunktion: Minimiere den gesamten Geschwindigkeitsaenderungsbedarf.
       ;; dv_dep: Abflug-Burn (Differenz zwischen Transfergeschwindigkeit und
       ;;         Erdbahngeschwindigkeit).
       ;; dv_arr: Ankunfts-Burn (Differenz zwischen Transfergeschwindigkeit
       ;;         am Ankunftspunkt und Marsbahngeschwindigkeit).
       (setf dv_dep (sqrt (+ (** (- (aref v0 0) vx_E) 2)
			     (** (- (aref v0 1) vy_E) 2)))
	     dv_arr (sqrt (+ (** (- (aref xf 2) vx_M) 2)
			     (** (- (aref xf 3) vy_M) 2))))
       (opti.minimize (+ dv_dep dv_arr))

       ;; Startwert (Initial Guess): Kreisbahngeschwindigkeit der Erde.
       ;; Ein guter Startwert ist entscheidend fuer die Konvergenz des NLP-Loesers.
       ;; Die Erdbahngeschwindigkeit ist ein vernuenftiger Ausgangspunkt, da die
       ;; Transferbahn typischerweise eine leicht modifizierte Erdbahn ist.
       (opti.set_initial v0 (vertcat vx_E vy_E))

       ;; Verbesserter Startwert: Approximation der Lambert-Loesung.
       ;; Der Geschwindigkeitsvektor wird in Richtung Mars gedreht und
       ;; leicht verstaerkt (Faktor 1.1), um den Hohmann-artigen Transfer
       ;; besser zu approximieren.
       (setf angle_to_mars (np.arctan2 dy dx)
	     v_earth_mag (/ (* 2 np.pi A_EARTH) T_EARTH)
	     guess_vx (* 1.1 v_earth_mag (np.cos (+ angle_to_mars (/ np.pi 6))))
	     guess_vy (* 1.1 v_earth_mag (np.sin (+ angle_to_mars (/ np.pi 6)))))
       (opti.set_initial v0 (vertcat guess_vx guess_vy))

       ;; IPOPT-Konfiguration: Interior Point Optimizer fuer nichtlineare Programme.
       ;; print_time=False: Keine CasADi-Zeitmessung ausgeben.
       ;; print_level=0: IPOPT-Ausgaben vollstaendig unterdruecken.
       ;; max_iter=1000: Maximale Iterationsanzahl.
       ;; tol=1e-6: Konvergenztoleranz fuer die KKT-Bedingungen.
       (opti.solver (string "ipopt")
		    (dict ((string "print_time") False))
		    (dict ((string "print_level") 0)
			  ((string "max_iter") 1000)
			  ((string "tol") 1e-6)))

       ;; Loesung des NLP mit Fehlerbehandlung.
       ;; Nicht alle (t_dep, tof)-Kombinationen haben eine physikalisch sinnvolle
       ;; Loesung. Bei zu kurzer Transferzeit oder unguenstigem Transferwinkel
       ;; kann IPOPT keine konvergente Loesung finden.
       (try
	(do0
	 (setf sol (opti.solve))
	 (setf total_dv (float (sol.value (+ dv_dep dv_arr))))
	 ;; Physikalische Plausibilitaetspruefung: Delta-v muss positiv und
	 ;; kleiner als ~50 km/s sein (jenseits davon sind keine chemischen
	 ;; Antriebe mehr sinnvoll).
	 (if (< total_dv (* 50.0 (/ YR_S AU_KM)))
	     (return (* total_dv V_CONV))
	     (return np.nan)))
	(Exception
	 (return np.nan))))
     
     (comments "========================================================================"
	       "HAUPTPROGRAMM: PARALLELISIERTE GITTERBERECHNUNG UND VISUALISIERUNG"
	       "========================================================================")
     
     ;; Schutzblock fuer multiprocessing: Unter Windows und mit spawn/forkserver
     ;; muss der Hauptcode hinter if __name__ == '__main__' stehen, damit
     ;; Workerprozesse das Modul importieren koennen ohne den Hauptcode auszufuehren.
     (when (== __name__ (string "__main__"))
       ;; Parametrierung des Suchgitters.
       ;; n_dep x n_tof Gitterpunkte, jeweils ein unabhaengiges NLP.
       (comments "Gitterparameter:"
		 "- Abflugdaten: 0 bis 2.5 Jahre (deckt mehr als eine synodische Periode"
		 "  Erde-Mars von 2.135 Jahren ab, um mindestens ein Startfenster zu zeigen)"
		 "- Transferdauer: 100 bis 450 Tage (Hohmann-Transfer: ~259 Tage)")
       (setf n_dep 50
	     n_tof 50
	     t_dep_arr (np.linspace 0.0 2.5 n_dep)
	     tof_days_arr (np.linspace 100.0 450.0 n_tof)
	     tof_yr_arr (/ tof_days_arr 365.25))
       
       ;; Erzeuge die vollstaendige Argumentliste fuer die Parallelisierung.
       ;; Jedes Element ist ein (t_dep, tof)-Tupel.
       (setf args_list (list))
       (for (t_dep t_dep_arr)
	    (for (tof tof_yr_arr)
		 (args_list.append (tuple t_dep tof))))
       
       ;; Parallelisierung mit multiprocessing.Pool.
       ;; mp.cpu_count() gibt die Anzahl der verfuegbaren CPU-Kerne zurueck.
       ;; Auf einem 2-Kern-System werden 2 Workerprozesse gestartet, auf einem
       ;; 32-Kern-System entsprechend 32. Die Arbeitspakete werden automatisch
       ;; auf die Worker verteilt (Lastausgleich via chunked map).
       (setf n_cpus (mp.cpu_count)
	     n_total (* n_dep n_tof))
       (print (fstring "Starte Gitterberechnung: {n_dep}x{n_tof} = {n_total} Punkte auf {n_cpus} Kernen"))
       (setf t_start (time.time))
       
       (with (as (mp.Pool :processes n_cpus) pool)
	     (setf results (pool.map solve_lambert_single args_list)))
       
       (setf elapsed (- (time.time) t_start))
       (print (fstring "Berechnung abgeschlossen in {elapsed:.1f} Sekunden"))
       
       ;; Ergebnisse in ein 2D-Array umformen: (n_dep, n_tof).
       (setf dv_grid (dot (np.array results) (reshape n_dep n_tof)))
       
       (comments "========================================================================"
		 "VISUALISIERUNG: PORK-CHOP-DIAGRAMM"
		 "========================================================================"
		 ""
		 "Das Diagramm zeigt Konturen des gesamten Delta-v in km/s als Funktion"
		 "von Abflugdatum (x-Achse) und Transferdauer (y-Achse)."
		 ""
		 "Interpretation:"
		 "- Die geschlossenen Konturen niedriger Delta-v-Werte zeigen die"
		 "  optimalen Startfenster (typischerweise nahe dem Hohmann-Transfer"
		 "  von ~259 Tagen bei ~5.6 km/s Gesamt-Delta-v)."
		 "- Die bananenartige Form ('Pork Chop') entsteht durch die relative"
		 "  Geometrie der Erde- und Marspositionen: Guenstige Konstellationen"
		 "  wiederholen sich mit der synodischen Periode von ~780 Tagen."
		 "- Rote Regionen zeigen energetisch teure Transfers (kurze Transferzeit"
		 "  bei unguentstiger Geometrie)."
		 "- Der weisse Stern markiert das globale Minimum.")
       
       ;; Gitternetzkoordinaten fuer den Konturplot.
       (setf (ntuple T_DEP TOF_DAYS) (np.meshgrid t_dep_arr tof_days_arr :indexing (string "ij")))
       
       ;; Globales Minimum finden (NaN-sichere Suche).
       (setf valid_mask (np.isfinite dv_grid))
       (when (np.any valid_mask)
	 (setf min_idx (np.nanargmin dv_grid))
	 (setf (ntuple i_min j_min) (np.unravel_index min_idx dv_grid.shape))
	 (setf dv_min (aref dv_grid i_min j_min)
	       t_dep_min (aref t_dep_arr i_min)
	       tof_min (aref tof_days_arr j_min))
	 (print (fstring "Globales Minimum: Delta-v = {dv_min:.2f} km/s"))
	 (print (fstring "  Abflug: t = {t_dep_min:.3f} yr, Transferdauer: {tof_min:.0f} Tage")))
       
       ;; Konturplot erstellen.
       (setf fig_and_ax (plt.subplots 1 1 :figsize (tuple 12 9)))
       (setf (ntuple fig ax) fig_and_ax)
       
       ;; Konturniveaus fuer Delta-v in km/s.
       ;; np.nanmin/nanmax ignorieren NaN-Werte bei der Bestimmung des Wertebereichs.
       (setf vmin (np.nanmin dv_grid)
	     vmax (np.clip (np.nanpercentile dv_grid 85) 0 30))
       (setf levels (np.linspace vmin vmax 25))
       
       ;; contourf: Gefuellter Konturplot. cmap='RdYlGn_r': Rot=teuer, Gruen=guenstig.
       (setf cf (ax.contourf T_DEP TOF_DAYS dv_grid
			     :levels levels
			     :cmap (string "RdYlGn_r")
			     :extend (string "max")))
       ;; Konturlinien darueberlegen fuer bessere Lesbarkeit.
       (setf cs (ax.contour T_DEP TOF_DAYS dv_grid
			    :levels levels
			    :colors (string "black")
			    :linewidths 0.3
			    :alpha 0.5))
       
       ;; Farbskala mit physikalischer Einheit.
       (setf cbar (plt.colorbar cf :ax ax))
       (cbar.set_label (string "Total $\\\\Delta v$ [km/s]") :fontsize 13)
       
       ;; Globales Minimum markieren.
       (when (np.any valid_mask)
	 (ax.plot t_dep_min tof_min
		  :marker (string "*")
		  :color (string "white")
		  :markersize 20
		  :markeredgecolor (string "black")
		  :markeredgewidth 1.5
		  :zorder 10)
	 (ax.annotate (fstring "$\\\\Delta v_{{min}}$ = {dv_min:.2f} km/s")
		      :xy (tuple t_dep_min tof_min)
		      :xytext (tuple (+ t_dep_min 0.15) (+ tof_min 25))
		      :fontsize 11
		      :fontweight (string "bold")
		      :color (string "white")
		      :arrowprops (dict ((string "arrowstyle") (string "->"))
				       ((string "color") (string "white"))
				       ((string "lw") 1.5))))
       
       ;; Achsenbeschriftungen und Titel.
       (ax.set_xlabel (string "Departure date [years from epoch]") :fontsize 13)
       (ax.set_ylabel (string "Transfer duration [days]") :fontsize 13)
       (ax.set_title (string "Earth $\\\\rightarrow$ Mars: Pork Chop Diagram (Impulsive $\\\\Delta v$)")
		     :fontsize 15 :fontweight (string "bold"))
       (ax.grid True :alpha 0.3 :color (string "white") :linestyle (string "--"))
       
       ;; Hohmann-Transferdauer als horizontale Referenzlinie einzeichnen.
       ;; Der Hohmann-Transfer ist die energieoptimale Bahn zwischen zwei
       ;; koplanaren Kreisbahnen: a_transfer = (a1 + a2) / 2,
       ;; T_transfer = pi * sqrt(a_transfer^3 / mu).
       (setf a_hohmann (/ (+ A_EARTH A_MARS) 2.0)
	     tof_hohmann_yr (* np.pi (np.sqrt (/ (** a_hohmann 3) (/ MU_SUN (* 4 (** np.pi 2))))))
	     tof_hohmann_days (* tof_hohmann_yr 365.25))
       (ax.axhline :y tof_hohmann_days :color (string "cyan") :linestyle (string "--")
		   :alpha 0.7 :linewidth 1.5 :label (fstring "Hohmann ({tof_hohmann_days:.0f} d)"))
       (ax.legend :loc (string "upper right") :fontsize 11)
       
       (plt.tight_layout)
       (plt.savefig (string "porkchop_earth_mars.png") :dpi 150)
       (print (string "Plot gespeichert: porkchop_earth_mars.png"))
       (plt.show)
       ))))

#|
================================================================================
PHYSIKALISCHE INTERPRETATION DER ERGEBNISSE
================================================================================

1. Hohmann-Transfer (theoretisches Minimum):
Der energieoptimale Transfer zwischen zwei koplanaren Kreisbahnen ist die
Hohmann-Ellipse mit Perihel auf der inneren Bahn (Erde, 1 AU) und Aphel auf
der aeusseren Bahn (Mars, 1.524 AU). Die Transferzeit betraegt:
  T_H = pi * sqrt(a_H^3 / mu) mit a_H = (1 + 1.524)/2 = 1.262 AU
  T_H ≈ 0.7087 yr ≈ 259 Tage
Das theoretische Gesamt-Delta-v fuer den Hohmann-Transfer Erde→Mars betraegt:
  Delta-v1 ≈ 2.94 km/s (Abflug)
  Delta-v2 ≈ 2.65 km/s (Ankunft)
  Delta-v_total ≈ 5.59 km/s
Die Abweichung des berechneten Minimums von diesem Wert haengt von der
relativen Phase der Planeten zum optimalen Abflugzeitpunkt ab.

2. Synodische Periode und Startfenster:
Die synodische Periode Erde-Mars betraegt:
  T_syn = 1/(1/T_E - 1/T_M) = 1/(1 - 1/1.881) ≈ 2.135 Jahre ≈ 780 Tage
Dies ist der Abstand zwischen aufeinanderfolgenden guenstigen Startfenstern.
Im Pork-Chop-Diagramm erscheinen die Minima daher mit einem Abstand von
~2.135 Jahren auf der x-Achse.

3. Treibstoffbedarf und Raketengleichung:
Das Delta-v ist ueber die Tsiolkowski-Raketengleichung mit dem Treibstoffbedarf
verknuepft:
  m_fuel/m_total = 1 - exp(-Delta-v / (g0 * Isp))
Fuer einen chemischen Antrieb (Isp ≈ 350 s) und Delta-v ≈ 5.6 km/s ergibt sich:
  m_fuel/m_total ≈ 80% (das Raumschiff besteht zu 80% aus Treibstoff)
Fuer einen ionischen Antrieb (Isp ≈ 3000 s) waere es nur ~17%.

4. Reale Missionen:
Tatsaechliche Mars-Missionen (Mars 2020/Perseverance, InSight, Maven) verwenden
Typ-I- oder Typ-II-Lambert-Transfers und nutzen die im Pork-Chop-Diagramm
sichtbaren Minima als Basis fuer die Missionplanung. Gravitationsassistenz-
Manoever (z.B. Venus-Flyby) koennen das effektive Delta-v weiter reduzieren,
sind aber in diesem vereinfachten Modell nicht enthalten.
|#
