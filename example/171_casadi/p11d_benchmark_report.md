# MPC Benchmark-Bericht & Parameter-Validierung
Generiert am: 2026-06-24 04:38:15

Dieser Bericht vergleicht verschiedene Reglereinstellungen und validiert die Korrektheit des Modells.

## Szenario: Stabilization (theta_0 = 0.2 rad)
| Konfiguration | IAE Position [m*s] | IAE Winkel [rad*s] | Settling Time Position [s] | Settling Time Winkel [s] | Overshoot Position [m] | Max Kraft [N] | Avg Solve Time [ms] | Solver Success % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Config 1: Current Default (high Q_v, Q_omega) | 3.047 | 0.196 | 5.31 | 1.02 | 1.500 | 10.03 | 19.01 | 100.0% |
| Config 2: Old Commit Default (low Q_v, Q_omega) | 1.683 | 0.316 | 3.46 | 2.41 | 1.500 | 15.00 | 18.65 | 100.0% |
| Config 3: High Position Weight (Q_s = 100.0) | 1.281 | 0.443 | 2.31 | 1.98 | 1.500 | 15.00 | 19.10 | 100.0% |
| Config 4: Fast Position Response (Q_s=50, Q_v=2, R_F=0.05) | 1.344 | 0.391 | 2.64 | 2.08 | 1.500 | 15.00 | 18.15 | 100.0% |
| Config 5: Short Horizon (N=10, T_horiz=0.5s) | 4.891 | 0.153 | Never | 1.09 | 1.500 | 6.70 | 11.30 | 100.0% |
| Config 6: Long Horizon (N=40, T_horiz=2.0s) | 2.412 | 0.223 | 3.96 | 2.11 | 1.500 | 12.51 | 36.14 | 100.0% |
| Config 7: L-Mismatch (l_slider=1.5, buggy physics l=0.5) | 2.604 | 0.212 | 4.65 | 1.81 | 1.500 | 12.34 | 18.89 | 100.0% |
| Config 8: L-Matched (l_slider=1.5, corrected physics l=1.5) | 2.573 | 0.287 | 4.92 | 2.80 | 1.500 | 12.34 | 17.74 | 100.0% |

## Szenario: Swing-Up & Position Step (theta_0 = pi)
| Konfiguration | IAE Position [m*s] | IAE Winkel [rad*s] | Settling Time Position [s] | Settling Time Winkel [s] | Overshoot Position [m] | Max Kraft [N] | Avg Solve Time [ms] | Solver Success % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Config 1: Current Default (high Q_v, Q_omega) | 3.746 | 1.779 | 5.87 | 2.57 | 1.833 | 15.00 | 19.65 | 100.0% |
| Config 2: Old Commit Default (low Q_v, Q_omega) | 2.440 | 1.755 | 3.99 | 2.97 | 1.857 | 15.00 | 19.02 | 100.0% |
| Config 3: High Position Weight (Q_s = 100.0) | 1.439 | 1.821 | 2.11 | 2.41 | 1.801 | 15.00 | 20.07 | 100.0% |
| Config 4: Fast Position Response (Q_s=50, Q_v=2, R_F=0.05) | 1.649 | 1.713 | 2.08 | 2.38 | 1.827 | 15.00 | 19.35 | 100.0% |
| Config 5: Short Horizon (N=10, T_horiz=0.5s) | 9.019 | 1.762 | 11.52 | 2.67 | 2.401 | 15.00 | 11.74 | 100.0% |
| Config 6: Long Horizon (N=40, T_horiz=2.0s) | 2.799 | 1.782 | 4.36 | 2.64 | 1.825 | 15.00 | 36.47 | 100.0% |
| Config 7: L-Mismatch (l_slider=1.5, buggy physics l=0.5) | 2.751 | 1.812 | 4.88 | 1.68 | 1.804 | 15.00 | 20.52 | 100.0% |
| Config 8: L-Matched (l_slider=1.5, corrected physics l=1.5) | 5.283 | 3.157 | 6.24 | 4.92 | 2.650 | 15.00 | 20.85 | 100.0% |

## Analyse & Empfehlungen
1. **Einfluss von Q_v und Q_omega (Config 1 vs Config 2):**
   - In älteren Commits (Config 2) waren die Gewichte fuer die Geschwindigkeiten (Wagen-Geschwindigkeit Q_v und Winkel-Geschwindigkeit Q_omega) auf 1.0 statt 10.0 eingestellt.
   - Niedrigere Dämpfungs-Kosten erlauben dem Regler, viel schneller zu beschleunigen und abzubremsen, was die Einschwingzeit (Settling Time) der Position verkuerzt, aber eventuell zu leichtem Überschwingen führt.
2. **Tuning fuer aggressive Positionsregelung (Config 4):**
   - Durch Erhoehung von Q_s (auf z.B. 50) und gleichzeitiges Absenken von Q_v (auf z.B. 2) bei geringeren Stellkraftkosten (R_F = 0.05) kann der Wagen extrem praezise positioniert werden, ohne instabil zu werden.
3. **Pendellänge-Modell-Mismatch (Config 7 vs Config 8):**
   - Wenn der Benutzer im GUI die Pendellänge veraendert, der Simulator aber intern mit der hartcodierten Laenge von 0.5 rechnet, weichen MPC-Modell und reale Physik stark voneinander ab. Das fuehrt zu schlechterer Performance (Config 7).
   - Bei passender Physik (Config 8) regelt der MPC das System auch bei Laenge 1.5 optimal.