# CasADi For-Loop Equivalents in active suspension MPC

Wir haben die active suspension MPC-Steuerung erfolgreich von einer ausgefalteten Schleifenformulierung ([gen08.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen08.lisp)) auf ein optimiertes Modell unter Verwendung von CasADi-Funktions-Maps ([gen09.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen09.lisp)) umgestellt.

Durch die Verwendung von `map` und `mapaccum` bleibt die Graphgröße und der Speicherbedarf zur Konstruktion des Optimierungsproblems konstant in Bezug auf den Vorhersagehorizont $N$ (bzw. wächst logarithmisch), statt linear $O(N)$ anzuwachsen.

---

## 1. Architektonische Gegenüberstellung

| Eigenschaft | Herkömmliche Formulierung (gen08) | Optimierte Map-Formulierung (gen09) |
| :--- | :--- | :--- |
| **Zustandskonstruktion** | Liste von Symbolen `[x_0, ..., x_N]` | Kompakte Matrix-Symbolic `X` ($4 \times N+1$) |
| **Stellgrößenkonstruktion** | Liste von Symbolen `[u_0, ..., u_N-1]` | Kompakte Matrix-Symbolic `U` ($1 \times N$) |
| **Systemdynamik-Residuen** | Flache Schleifen-Additionen in Python-AST | Einstufige `Function` &rarr; `.map(N)` |
| **Zustandskosten (Stage Cost)**| Flache Schleifen-Additionen in Python-AST | Einstufige `Function` &rarr; `.map(N)` &rarr; `sum2` |
| **Passives System (Simulation)** | Explicit `for`-Schleife in Python | Einzelner `mapaccum(N_steps - 1)`-Aufruf |
| **Python Zeilenanzahl (LoC)** | **684** Zeilen | **324** Zeilen (~52% kürzer) |

---

## 2. Wie es funktioniert

### A. Multiple Shooting mit `.map()`
In [gen09.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen09.lisp) definieren wir eine Einzelschritt-Differenzengleichung:

$$\mathbf{x}_{k+1} - (\mathbf{A}_d \mathbf{x}_k + \mathbf{B}_d u_k + \mathbf{G}_d v_k) = 0$$

Anstatt diese Gleichung $N$-mal in Lisp auszurollen, erzeugen wir eine CasADi `Function` für einen Schritt und mappen diese über den gesamten Zeithorizont:
```python
f_dyn = Function("f_dyn", [x_next_sym, x_curr_sym, u_curr_sym, v_curr_sym], [residual])
F_dyn = f_dyn.map(N)
g_dyn = F_dyn(X[:, 1:], X[:, 0:N], U, V_r)
```

### B. Simulation mit `.mapaccum()`
Für die Open-Loop-Simulation des passiven Systems ($u=0$) verwenden wir `mapaccum` (akkumulierendes Mapping / AkkuMap), das den Zustand akkumuliert und die Störungen $v_k$ spaltenweise einliest:
```python
f_passive_step = Function("f_passive_step", [x_curr_sym, v_curr_sym], [A @ x_curr_sym + G * v_curr_sym])
f_passive_sim = f_passive_step.mapaccum("f_passive_sim", N_steps - 1)
x_hist_passive = hcat([x0, f_passive_sim(x0, v_r[:-1])]).full()
```
Dies eliminiert die Simulations-Schleife für das passive System komplett.

---

## 3. Simulationsergebnisse und Vorhersagehorizonte

Das aktive Map-MPC-Fahrwerk verhält sich identisch zum vorherigen Ansatz, benötigt jedoch weniger Overhead für die Problemformulierung.

![Zustandsvergleich und Aktuatorkraft](active_suspension_mpc_map.png)

### Diskussion der MPC-Vorhersagetrajektorien:
Die gestrichelten Linien zeigen die vom MPC-Regler (Modellprädiktive Regelung, gelöst als quadratisches Programm / QP) zu bestimmten Zeitschritten ($t=0.35\text{s}$, $t=0.55\text{s}$, $t=0.85\text{s}$) vorhergesagten Trajektorien über den Vorhersagehorizont $N = 30$ Zeitschritte ($0.3\text{s}$):

1. **Vor der Störung ($t = 0.35\text{s}$, orange gepunktete Linie):**
   - Das Fahrzeug ist noch im Nullzustand (Ruhelage).
   - Durch die Straßenvorschau (Look-Ahead Preview, erfasst z. B. per LiDAR) sieht der Regler, dass bei $t = 0.5\text{s}$ eine Schwelle kommt.
   - Der Regler zieht das Rad vorausschauend aktiv ein (Stellkraft sinkt ab $t=0.35\text{s}$ ins Negative), um den kommenden Stoß weich abzufedern.

2. **Während der Störung ($t = 0.55\text{s}$, lila gepunktete Linie):**
   - Das Fahrzeug fährt gerade über die Bodenschwelle.
   - Der Regler reagiert mit maximaler Gegenkraft (Sättigung an der Aktuatorgrenze von $-1500\text{ N}$).
   - Die Vorhersage zeigt, wie der Regler plant, die Kraft nach Erreichen der Spitze weich abzubauen.

3. **Nach der Störung ($t = 0.85\text{s}$, türkis gepunktete Linie):**
   - Die Bodenschwelle ist überfahren, aber das System schwingt noch leicht nach.
   - Die Vorhersage zeigt, wie der Regler die Chassis-Schwingung innerhalb seines $0.3\text{s}$-Horizonts komplett auf $0$ zurückführt.
