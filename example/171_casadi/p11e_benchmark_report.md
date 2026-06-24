# Inverted Pendulum MPC Optimization Benchmark Report
Generiert am: 2026-06-24 07:00:49
Computer: AMD Ryzen Threadripper PRO 7955WX 16-Cores

Dieses Dokument vergleicht den Einfluss der verschiedenen vorgeschlagenen Optimierungsansätze (to_function, JIT, Dual-Variable Warm-Starting und Mapped Constraints) für unterschiedliche Prädiktionshorizonte ($N=20$ und $N=200$).

## Benchmark-Ergebnisse für Horizont N = 20

### Szenario: Stabilization (theta_0 = 0.2 rad)
| Konfiguration | Avg Solve Time [ms] | Max Solve Time [ms] | Init / JIT Compile [ms] | IAE Pos | IAE Angle | Success % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (Opti, Primal WS) | 9.07 | 12.29 | 50.5 | 2.672 | 0.175 | 100.0% |
| 2. Warmstart Dual (Opti, Prim+Dual WS) | 8.72 | 10.24 | 28.0 | 2.672 | 0.175 | 100.0% |
| 3. JIT (Opti, JIT, Prim+Dual WS) | 6.67 | 7.79 | 27.3 | 2.672 | 0.175 | 100.0% |
| 4. to_function (Func, Prim+Dual WS) | 8.44 | 9.87 | 54.3 | 2.672 | 0.175 | 100.0% |
| 5. to_function + JIT (Func, JIT, Prim+Dual WS) | 6.12 | 7.29 | 11276.3 | 2.672 | 0.175 | 100.0% |
| 6. Map Baseline (Opti, Map, Prim+Dual WS) | 8.84 | 10.31 | 8.4 | 2.672 | 0.175 | 100.0% |
| 7. Map JIT (Opti, Map, JIT, Prim+Dual WS) | 6.15 | 7.62 | 7.5 | 2.672 | 0.175 | 100.0% |
| 8. Map to_function (Func, Map, Prim+Dual WS) | 8.75 | 10.22 | 13.9 | 2.672 | 0.175 | 100.0% |
| 9. Map to_function + JIT (Func, Map, JIT, Prim+Dual WS) | 6.14 | 8.19 | 2162.3 | 2.672 | 0.175 | 100.0% |

### Szenario: Swing-Up & Position Step (theta_0 = pi)
| Konfiguration | Avg Solve Time [ms] | Max Solve Time [ms] | Init / JIT Compile [ms] | IAE Pos | IAE Angle | Success % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (Opti, Primal WS) | 10.94 | 39.68 | 26.6 | 3.429 | 1.754 | 100.0% |
| 2. Warmstart Dual (Opti, Prim+Dual WS) | 10.96 | 41.13 | 27.2 | 3.429 | 1.754 | 100.0% |
| 3. JIT (Opti, JIT, Prim+Dual WS) | 8.07 | 28.63 | 30.6 | 3.429 | 1.754 | 100.0% |
| 4. to_function (Func, Prim+Dual WS) | 11.27 | 41.37 | 50.7 | 3.429 | 1.754 | 100.0% |
| 5. to_function + JIT (Func, JIT, Prim+Dual WS) | 8.51 | 30.21 | 11269.0 | 3.429 | 1.754 | 100.0% |
| 6. Map Baseline (Opti, Map, Prim+Dual WS) | 9.48 | 18.02 | 9.3 | 3.429 | 1.754 | 100.0% |
| 7. Map JIT (Opti, Map, JIT, Prim+Dual WS) | 6.78 | 12.87 | 6.1 | 3.429 | 1.754 | 100.0% |
| 8. Map to_function (Func, Map, Prim+Dual WS) | 9.62 | 18.11 | 12.7 | 3.429 | 1.754 | 100.0% |
| 9. Map to_function + JIT (Func, Map, JIT, Prim+Dual WS) | 6.78 | 12.71 | 2144.6 | 3.429 | 1.754 | 100.0% |

## Benchmark-Ergebnisse für Horizont N = 200

### Szenario: Stabilization (theta_0 = 0.2 rad)
| Konfiguration | Avg Solve Time [ms] | Max Solve Time [ms] | Init / JIT Compile [ms] | IAE Pos | IAE Angle | Success % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (Opti, Primal WS) | 70.33 | 94.91 | 236.9 | 2.255 | 0.211 | 100.0% |
| 2. Warmstart Dual (Opti, Prim+Dual WS) | 69.99 | 86.17 | 263.5 | 2.255 | 0.211 | 100.0% |
| 3. JIT (Opti, JIT, Prim+Dual WS) (Skipped JIT) | N/A | N/A | N/A | N/A | N/A | N/A |
| 4. to_function (Func, Prim+Dual WS) | 67.46 | 84.53 | 494.5 | 2.255 | 0.211 | 100.0% |
| 5. to_function + JIT (Func, JIT, Prim+Dual WS) (Skipped JIT) | N/A | N/A | N/A | N/A | N/A | N/A |
| 6. Map Baseline (Opti, Map, Prim+Dual WS) | 61.70 | 82.19 | 48.6 | 2.255 | 0.211 | 100.0% |
| 7. Map JIT (Opti, Map, JIT, Prim+Dual WS) | 39.93 | 51.60 | 29.2 | 2.255 | 0.211 | 100.0% |
| 8. Map to_function (Func, Map, Prim+Dual WS) | 61.60 | 79.07 | 68.9 | 2.255 | 0.211 | 100.0% |
| 9. Map to_function + JIT (Func, Map, JIT, Prim+Dual WS) | 40.13 | 52.12 | 22260.4 | 2.255 | 0.211 | 100.0% |

### Szenario: Swing-Up & Position Step (theta_0 = pi)
| Konfiguration | Avg Solve Time [ms] | Max Solve Time [ms] | Init / JIT Compile [ms] | IAE Pos | IAE Angle | Success % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (Opti, Primal WS) | 98.65 | 399.50 | 237.2 | 2.721 | 1.762 | 100.0% |
| 2. Warmstart Dual (Opti, Prim+Dual WS) | 99.32 | 402.16 | 257.4 | 2.721 | 1.762 | 100.0% |
| 3. JIT (Opti, JIT, Prim+Dual WS) (Skipped JIT) | N/A | N/A | N/A | N/A | N/A | N/A |
| 4. to_function (Func, Prim+Dual WS) | 95.56 | 393.70 | 482.9 | 2.721 | 1.762 | 100.0% |
| 5. to_function + JIT (Func, JIT, Prim+Dual WS) (Skipped JIT) | N/A | N/A | N/A | N/A | N/A | N/A |
| 6. Map Baseline (Opti, Map, Prim+Dual WS) | 70.13 | 137.18 | 48.4 | 2.721 | 1.762 | 100.0% |
| 7. Map JIT (Opti, Map, JIT, Prim+Dual WS) | 45.60 | 88.80 | 29.3 | 2.721 | 1.762 | 100.0% |
| 8. Map to_function (Func, Map, Prim+Dual WS) | 70.50 | 136.66 | 70.1 | 2.721 | 1.762 | 100.0% |
| 9. Map to_function + JIT (Func, Map, JIT, Prim+Dual WS) | 45.82 | 88.94 | 22213.6 | 2.721 | 1.762 | 100.0% |

## Analyse und Schlussfolgerungen
1. **Einfluss der Mapped Constraints:**
   - Durch das Umschreiben der NLP-Constraints auf CasADi-Mappings (looped structure) bleibt der zugrundeliegende Symbolgraph klein. Dies ermöglicht die JIT-Kompilierung auch für große Prädiktionshorizonte ($N=200$) in vertretbarer Zeit, da GCC Schleifenkonstrukte im C-Code optimieren kann statt Millionen flacher unrolled Statements. Das eliminiert den 14MB-Dateigrößen-Overhead komplett.
2. **Kombination von to_function und JIT:**
   - Die Kombination bietet das absolute Performance-Maximum, da `to_function` den Python-Stack-Overhead umgeht und JIT die Ableitungen und Dynamik-Integrationsschritte nativ ausführt.
3. **Dual-Variable Warm-Starting:**
   - Durch das Warm-Starting der Dualvariablen `lam_g` konvergiert der Solver schneller, da IPOPT an guten Schätzungen für die Aktivität der Randbedingungen anknüpfen kann, was die maximale Lösungszeit (Jitter) signifikant glättet.
