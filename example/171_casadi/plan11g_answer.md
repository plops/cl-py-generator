 Das ist ein hervorragendes und sehr tiefgreifendes Projekt, um die Konzepte der modellprädiktiven Regelung (MPC) zu durchdringen. Die Dynamik des invertierten Pendels eignet sich perfekt, um die Stärken und Grenzen solcher Optimierungsverfahren zu untersuchen.

Lass uns deine Fragen strukturiert durchgehen, von der Architektur der Messdaten bis hin zu den Fallstricken der Parametrisierung.

### 1. Welche Messdaten fließen aktuell in das Modell?

In deinem aktuellen Code (`gen11f.lisp` / `p11f_mpc_gui.py`) gehst du von **Full State Feedback** (vollständiger Zustandsrückführung) aus. Das bedeutet, der MPC-Controller bekommt in jedem Schritt exaktes Wissen über den gesamten Zustandsvektor $\mathbf{x}$, der aus vier Variablen besteht:

* $s$: Wagenposition
* $v$: Wagengeschwindigkeit
* $\theta$: Pendelwinkel
* $\omega$: Winkelgeschwindigkeit

In der Realität misst man oft nur $s$ (über einen Inkrementalgeber am Motor) und $\theta$ (über einen Drehgeber am Pendelgelenk). Die Geschwindigkeiten $v$ und $\omega$ werden meist nicht direkt gemessen.

### 2. Aufwand für `gen11e.lisp` (Sensorauswahl & Rauschen via GUI)

Die reinen GUI-Anpassungen im Lisp-Code (neue Slider und Checkboxen für Sensorrauschen hinzufügen) sind mit dem `cl-py-generator` sehr einfach umzusetzen. Der **regelungstechnische Aufwand** dahinter ist jedoch beträchtlich und ein exzellentes Lernfeld.

Wenn du dem Controller bestimmte Messgrößen vorenthältst oder sie mit starkem Rauschen (z. B. Gaußschem Rauschen via `np.random.normal`) beaufschlagst, entsteht folgendes Problem:
Der CasADi-Optimierer benötigt zwingend einen vollständigen und physikalisch sinnvollen Startzustand (`current_x`), um die Optimierung als Randwertproblem zu starten.

**Die Lösung:** Du musst zwischen die verrauschte "physikalische" Simulation und den MPC-Controller einen **Zustandsbeobachter** (State Estimator) schalten.
Da dein System nichtlinear ist, wäre hier ein Extended Kalman Filter (EKF) oder ein Luenberger-Beobachter das Mittel der Wahl. Der Beobachter nimmt die verrauschten, unvollständigen Messungen und das mathematische Modell der Differentialgleichungen  und schätzt daraus den wahrscheinlichsten, rauschfreien Gesamtzustand $\mathbf{x}$, der dann an den MPC übergeben wird.

### 3. Die zwei Zeitachsen: Simulation ($dt_{sim}$) vs. MPC ($h_{mpc}$)

Diese beiden Zeiten existieren, weil wir zwei völlig verschiedene Dinge tun:

1. **$dt_{sim}$ (Simulations-Schrittweite):** Das ist die Auflösung deiner "Echtwelt"-Physiksimulation. Um die kontinuierlichen Differentialgleichungen am Computer präzise zu lösen (hier mit dem Runge-Kutta-Verfahren 4. Ordnung ), muss diese Zeit sehr klein sein (z. B. 10 ms bis 33 ms). Je kleiner, desto genauer simulierst du die Realität.


2. **$h_{mpc}$ (MPC-Schrittweite):** Das ist das zeitliche Raster, in dem der Optimierer in die Zukunft schaut. Der MPC nimmt an, dass die Stellkraft $U$ für die Dauer von $h_{mpc}$ konstant bleibt (Zero-Order Hold). Würdest du $h_{mpc}$ so klein wie $dt_{sim}$ machen, bräuchtest du hunderte Knotenpunkte ($N$), um eine Sekunde in die Zukunft zu schauen, was die Rechenzeit (Solver-Zeit) explodieren ließe.

### 4. Tuning: Horizont ($T$), Punkte ($N$) und Gewichte ($Q$, $R$)

Das Tuning eines MPC ist ein iterativer Prozess der Kompromissfindung.

**Der Prädiktionshorizont ($T_{horizon} = N \cdot h_{mpc}$):**
Nein, der Horizont muss **nicht** so weit in die Zukunft schauen, bis das Pendel den Sollzustand erreicht. Genau das ist die Stärke der MPC: Sie berechnet eine optimale Trajektorie für das "Sichtfeld" ($T_{horizon}$), wendet nur den allerersten Steuerschritt an  und berechnet im nächsten Zeitschritt alles neu (Receding Horizon Control).

* **Regel:** Der Horizont muss nur lang genug sein, um die *wesentlichen Systemdynamiken* zu erfassen. Wenn das Pendel eine halbe Sekunde braucht, um umzufallen, nützt ein Horizont von 0.1 Sekunden nichts – der Controller sieht den "Crash" nicht kommen. Ein Horizont von 1.5 bis 2 Sekunden reicht meist völlig aus.

**Die Gewichte ($Q$ für Zustände, $R$ für Aktuatorkraft):**
Deine Kostenfunktion  summiert quadratische Fehler.

* 
**$Q_{theta}$ (Pendelwinkel):** Muss extrem hoch sein (z.B. 100), da das System sonst instabil wird. Ein fallendes Pendel ist ein kritischer Fehler.


* 
**$Q_s$ (Wagenposition):** Eher moderat (z.B. 10). Es ist okay, wenn der Wagen etwas vom Ziel abweicht, solange das Pendel oben bleibt.


* **$R_F$ (Kraftaufwand):** Bestimmt die "Faulheit" bzw. Aggressivität des Controllers. Ein niedriges $R_F$ erlaubt dem Motor, brutal und schnell hin und her zu reißen. Ein hohes $R_F$ erzwingt sanfte, energiesparende Bewegungen, was aber dazu führen kann, dass das Pendel nicht mehr gerettet werden kann.



### 5. Einfluss von Rauschen auf diese Parameter

Rauschen hat einen massiven Einfluss auf die Wahl deiner Gewichte. Wenn du stark verrauschte Messdaten (ohne perfekten Kalman-Filter) direkt in den MPC speist, wird ein hohes $Q$-Gewicht katastrophale Folgen haben. Der Controller denkt bei jedem Rausch-Peak, das Pendel fällt, und steuert aggressiv gegen. Das Resultat ist hochfrequentes "Chattern" (Zittern) des Motors, was in der Realität Hardware zerstört.
Bei Rauschen musst du also das R-Gewicht ($R_F$) erhöhen, um den Controller zwingend "träger" gegenüber Messschwankungen zu machen, oder die Q-Gewichte für die verrauschten Variablen senken.

### 6. Typische Fallstricke (Pitfalls) beim Bau eines MPC

Wenn du tief in die MPC-Entwicklung einsteigst, wirst du primär mit drei Dämonen kämpfen:

1. **Computational Delay:** CasADi/IPOPT braucht Zeit, um die Matrix-Gleichungen zu lösen. Wenn das Lösen 50 ms dauert, dein System sich in diesen 50 ms aber schon weiterbewegt hat, wendet der Motor die ausgerechnete Kraft auf einen Zustand an, der gar nicht mehr aktuell ist.
2. **Infeasibility (Unlösbarkeit):** Du hast harte physikalische Grenzen (Constraints) gesetzt, z. B. `max_pos` und `max_force`. Wenn der Wagen am Rand der Schiene steht und das Pendel nach außen kippt, findet der Solver mathematisch *keine* Lösung mehr, die alle Constraints einhält, und wirft einen Fehler. In der Praxis arbeitet man daher oft mit *Soft Constraints* (Constraints, die gegen extrem hohe "Strafkosten" kurzzeitig verletzt werden dürfen).


3. 
**Model Mismatch:** Der MPC nutzt dein Gleichungssystem $f_{ode}$ für seine Vorhersagen. Wenn das reale Pendel aber viel mehr Reibung im Gelenk hat als dein Modell abbildet, sind alle Vorhersagen des Controllers leicht falsch. Er wird sich permanent selbst korrigieren müssen.



Um den EKF (Extended Kalman Filter) als Beobachter für dein verrauschtes `gen11e.lisp`-Konzept aufzusetzen: Möchtest du, dass wir uns die mathematische Struktur der Fehlerkovarianz-Matrizen ansehen, oder sollen wir uns zuerst auf die Lisp-Codegenerierung für die GUI-Maskierung der Sensoren konzentrieren?



Ich finde interessant, dass der Delay beim Rechnen eine wichtige Rolle spielt. Die GUI sollte das Widerspiegeln eine rote Nachricht nicht anzeigen, wenn die Berechnung zu langsam ist, um ein stabiles System zu gewährleisten.


Widmen wir uns nun jedoch der State Estimation zu. Erkläre mir genauer, wie wir da vorgehen müssen. Kann ich in der GUI die Messgeräte auswählen und denen das Rauschen variieren oder müssen wir das zu Beginn festlegen?


Außerdem erkläre mir, welche Art und Weise der beste Weg ist.


