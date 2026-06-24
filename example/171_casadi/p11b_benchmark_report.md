# MPC Performance Benchmark Report (Inverted Pendulum)

Dieses Dokument fasst die Ergebnisse des Zwischenschritts `gen11b.lisp` zusammen. Das Ziel war es zu verifizieren, ob die CasADi-MPC-Schleife schnell genug für eine Echtzeitanwendung (Real-Time Control) ist und welchen Einfluss das *Warm-Starting* sowie die *C-Code-Generierung (JIT)* haben.

## 1. Setup der Simulation
- **System**: Inverted Pendulum (Wagen + Pendel), 4 Zustände, 1 Stellkraft.
- **MPC-Horizont**: 1.0 Sekunde in die Zukunft ($T=1.0$).
- **Knotenpunkte**: $N=20$.
- **Diskretisierung**: Direct Collocation (Radau, Polynomgrad $d=3$).
- **Solver**: IPOPT.

Es wurden 50 fortlaufende MPC-Schritte berechnet. In jedem Schritt wurde die Lösung des vorherigen Schritts zeitlich verschoben und als exakter Startwert (`opti.set_initial()`) für den neuen Schritt verwendet.

---

## 2. Ergebnisse

### A. Reguläres CasADi (Python API)
Dies ist die Standardausführung, bei der CasADi den Graphen aus dem RAM evaluiert.

| Metrik | Wert | Bemerkung |
| :--- | :--- | :--- |
| **Initialisierung + 1. Solve** | **312.3 ms** | CasADi Graphen/Solver-Objekte aufbauen im RAM & Erster Kaltstart. |
| **Purer Kaltstart (Solve)** | **78.8 ms** | Lösungszeit von IPOPT aus dem Nichts, ohne Aufbau-Overhead. |
| **MPC Loop (Avg)** | **22.22 ms** | **Warm Start!** Durchschnittliche Zeit pro Schritt in der aktiven Regelung. |
| **MPC Loop (Min)** | **14.13 ms** | Schnellster Solve (wenn sich das System kaum ändert). |
| **MPC Loop (Max)** | **68.70 ms** | Langsamster Solve (vermutlich bei starken Richtungswechseln). |

### B. Hochoptimierter C-Code Export (JIT Compilation)
Hierbei übersetzt CasADi die gesamten Gradienten- und Jacobi-Matrizen in reinen C-Code und ruft im Hintergrund `gcc -O3` auf, bevor IPOPT startet.

| Metrik | Wert | Bemerkung |
| :--- | :--- | :--- |
| **GCC Build + 1. Solve** | **23134.7 ms** | (~23 Sekunden) Beinhaltet die `gcc`-Kompilierungszeit des extrem großen NLP-C-Codes! |
| **Purer Kaltstart (Solve)** | **51.3 ms** | Lösungszeit von IPOPT für das in C kompilierte Problem. |
| **MPC Loop (Avg)** | **16.92 ms** | Nochmals ~24% schneller als die Python-Variante. |
| **MPC Loop (Min)** | **11.10 ms** | |
| **MPC Loop (Max)** | **51.46 ms** | |

---

## 3. Schlussfolgerungen

1. **Overhead der Initialisierung:** Die bisherigen "370 ms" bestanden zum größten Teil (~230 ms) aus der Initialisierung von IPOPT im Arbeitsspeicher (Sparsity Pattern Berechnungen, Speicherallokationen). Der reine "Solve" für den Kaltstart brauchte nur schlanke **78.8 ms**.
2. **Warm-Starting ist der Schlüssel:** Durch die Übergabe des vorherigen Schritts stürzt die Lösungszeit auf **durchschnittlich 22 ms** ab.
3. **Echtzeitfähigkeit ist gegeben:** Um das System mit 30 Hz flüssig zu animieren, haben wir ein Zeitbudget von **33 Millisekunden** pro Frame. Die durchschnittliche Rechenzeit von 16-22 ms liegt perfekt in diesem Rahmen. Ein paralleles Ausführen auf 32 Kernen ist absolut nicht notwendig.
4. **Verzicht auf JIT für die GUI:** Die C-Code-Generierung spart uns zwar nochmals rund 6 Millisekunden pro Frame, blockiert aber beim Start der Applikation für 23 Sekunden den Rechner (während GCC läuft). Da 22 ms (ohne JIT) bereits echtzeitfähig sind, empfehle ich, für die interaktive GUI vorerst auf JIT zu verzichten, damit du nicht jedes Mal eine halbe Minute warten musst, wenn du die Applikation startest.

**Nächste Schritte:**
Da die Performance bestätigt ist, können wir nun `gen11c.lisp` implementieren, welches diese extrem schnelle MPC-Loop in ein interaktives PyQtGraph / PySide6 Dashboard einbettet.
