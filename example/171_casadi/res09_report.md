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
Für die Open-Loop-Simulation des passiven Systems ($u=0$) verwenden wir `mapaccum`, das den Zustand akkumuliert und die Störungen $v_k$ spaltenweise einliest:
```python
f_passive_step = Function("f_passive_step", [x_curr_sym, v_curr_sym], [A @ x_curr_sym + G * v_curr_sym])
f_passive_sim = f_passive_step.mapaccum("f_passive_sim", N_steps - 1)
x_hist_passive = hcat([x0, f_passive_sim(x0, v_r[:-1])]).full()
```
Dies eliminiert die Simulations-Schleife für das passive System komplett.

---

## 3. Simulationsergebnisse

Das aktive Map-MPC-Fahrwerk verhält sich identisch zum vorherigen Ansatz, benötigt jedoch weniger Overhead für die Problemformulierung.

![Zustandsvergleich und Aktuatorkraft](active_suspension_mpc_map.png)

Die Aufbaubeschleunigung (Chassis Acceleration) wird um über **70%** reduziert, was den Fahrkomfort in der Komfortzone (nach ISO 2631) hält.
