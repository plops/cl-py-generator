# Multi-Dimensional Inverted Pendulum MPC Grid Sweep

This report documents the performance benchmarks, stability analysis, and physical findings of the inverted pendulum Model Predictive Controller (MPC) across step sizes ($h_{mpc}$), pendulum lengths ($l$), and mass ratios ($m/M$) for different prediction horizons ($N=25$, $N=30$, $N=50$, and $N=100$), culminating in a high-resolution 4160-simulation grid sweep.

---

## 1. System Configuration & Architecture
* **Hardware:** 16 Cores / 32 Threads CPU.
* **Environment:** Ubuntu 26.04 LTS Docker container.
* **Software:** CPython 3.14.4, CasADi 3.6.7, GCC 15.2.0.
* **Parallelization:** Process-parallel execution using 16 worker processes (1 per physical CPU core).
* **JIT Caching:** Single compilation warm-up run in the main process. Worker processes load the cached `.so` binary instantly from the disk cache.

---

## 2. JIT vs. Pure Python Benchmark (N = 100)
To validate JIT compilation efficiency, a comparative benchmark of a single MPC step at prediction horizon $N=100$ was conducted:

| Evaluator | Solve Time per Step | Speedup |
| :--- | :--- | :--- |
| **Pure Python (CasADi VM)** | 221.7 ms | 1.0x (Baseline) |
| **JIT Compiled (GCC -O3)** | **24.3 ms** | **9.1x Faster** |

---

## 3. High-Resolution Sweep Results (N = 30, Focus Band)
To map the stability boundary in fine detail, a high-resolution grid sweep was conducted. By limiting the step size range $h_{mpc}$ to the active stable band ($[0.02, 0.06]\,\text{s}$), we eliminated the unstable regions ($h > 0.06\,\text{s}$) and focused all computational resources on the transition zone.
* **Horizon $N$:** 30
* **Max Control Force:** 20.0 N
* **Grid Dimensions:**
  * $h_{mpc}$ (step size): 20 values in $0.020\,\text{s} - 0.060\,\text{s}$
  * $l$ (pendulum length): 26 values in $0.20\,\text{m} - 2.00\,\text{m}$
  * $m/M$ (mass ratio): 8 values in $0.05 - 0.50$
* **Total Simulations:** 4,160
* **Execution Time:** **694.57 seconds (11.57 minutes)**, averaging **0.167s per simulation** on 16 processes.
* **Overall Success Rate:** **55.9% (2,325 / 4,160 runs)** successfully stabilized.

---

## 4. High-Resolution Parameter Stability Analysis

### A. Influence of Mass Ratio ($m/M$)
Increasing the mass ratio (heavier pendulum bob relative to the cart) increases inertia, raising the control force requirements:

| Mass Ratio ($m/M$) | Success Rate | Mean Stabilization Time |
| :---: | :---: | :---: |
| 0.050 | 66.0% | 5.82 s |
| 0.114 | 61.7% | 5.91 s |
| 0.179 | 61.2% | 6.00 s |
| 0.243 | 58.7% | 6.12 s |
| 0.307 | 54.2% | 6.23 s |
| 0.371 | 50.0% | 6.42 s |
| 0.436 | 48.8% | 6.43 s |
| 0.500 | 46.5% | 6.46 s |

* **Physical Trend:** We observe a perfect, monotonic decrease in success rate and an increase in mean stabilization time as the mass ratio increases. Heavier pendulum bobs require larger force inputs to swing up, causing the control output to saturate the actuator limit ($20.0\,\text{N}$), which leads to instability.

### B. Influence of Step Size ($h_{mpc}$)
| Step Size ($h_{mpc}$) | Success Rate | Lookahead Window ($N \times h_{mpc}$) |
| :---: | :---: | :---: |
| 0.020 s | 17.3% | 0.60 s (Too short / blind) |
| 0.024 s | 40.4% | 0.72 s |
| 0.028 s | 60.1% | 0.84 s |
| **0.033 s** | **72.6%** | 0.99 s (Optimal) |
| 0.037 s | 69.7% | 1.11 s |
| 0.041 s | 71.6% | 1.23 s |
| 0.045 s | 58.2% | 1.35 s |
| 0.049 s | 58.2% | 1.47 s |
| 0.054 s | 51.4% | 1.62 s |
| 0.058 s | 43.3% | 1.74 s |
| 0.060 s | 41.8% | 1.80 s |

* **Sweet Spot:** The step size success rate exhibits a clear bell curve, peaking in the region $h \in [0.033, 0.041]\,\text{s}$ (71.6% to 72.6% success). This corresponds to a preview window of $1.0 - 1.2\,\text{s}$, which provides the ideal horizon length for the swing-up trajectory while keeping discretization errors minimal.

### C. Influence of Pendulum Length ($l$)
| Length ($l$) | Success Rate | Physical Characteristics |
| :---: | :---: | :--- |
| 0.20 m | 10.6% | Rapid fall (High natural frequency $\omega \propto \sqrt{g/l}$) |
| 0.34 m | 30.0% | |
| 0.56 m | 39.4% | |
| 0.78 m | 50.6% | |
| 0.99 m | 68.1% | |
| **1.21 m** | **76.2%** | **Optimal Region** |
| 1.42 m | 73.8% | |
| 1.64 m | 66.9% | |
| 1.86 m | 61.3% | |
| 2.00 m | 53.1% | Slow dynamics, but requires large cart displacement |

* **Physical Interpretation:**
  * **Short Pendulums ($l \le 0.34\,\text{m}$):** Have very fast falling dynamics. The controller's step size is often too large to catch the pendulum in time, leading to rapid tips.
  * **Long Pendulums ($l \ge 1.86\,\text{m}$):** Fall slower (easier to catch), but their large length requires a massive cart displacement to swing up. This displacement exceeds the track boundary limits ($\pm 5\,\text{m}$) or takes longer than the 12.0s limit to settle, resulting in failure.
  * **Optimal Length ($l \approx 1.20\,\text{m}$):** Represents the perfect physical balance between manageable natural frequencies and acceptable track space requirements.

---

## 5. Comparative Summary of Sweep Horizons

To evaluate horizon scalability, we compare the original parameters across the different horizons:

| Prediction Horizon | Successful Runs | Success Rate | Average Simulation Time |
| :---: | :---: | :---: | :---: |
| **$N = 25$** | 108 / 280 | **38.6%** | **32.23 s** (0.115 s/sim) |
| **$N = 30$** | 88 / 280 | **31.4%** | **35.62 s** (0.127 s/sim) |
| **$N = 50$** | 42 / 280 | **15.0%** | **41.48 s** (0.148 s/sim) |
| **$N = 100$** | 4 / 280 | **1.4%** | **35.32 s** (0.126 s/sim) |

### The Shifting Sweet Spot and Solver Robustness
As $N$ increases, the optimal step size $h_{mpc}$ shifts steadily to the left (smaller steps) to maintain lookahead windows ($N \times h_{mpc}$) in the stable 0.8–1.5s range without accumulating excessive discretization error. Larger horizons ($N \ge 50$) combined with larger step sizes create massive, non-convex NLPs that fail to solve under cold-start conditions, confirming that shorter horizons ($N=25 - 30$) are numerically much more robust for online MPC.

---

## 6. Heatmap Visualizations

### A. High-Resolution Heatmaps (N = 30, 4160 Simulations)
![High-Res Heatmap](p11i_grid_heatmaps_20260701_181951.png)

### B. Horizon N = 25 Heatmaps (280 Simulations)
![Heatmap N=25](p11i_grid_heatmaps_20260701_174539.png)

### C. Horizon N = 30 Heatmaps (280 Simulations)
![Heatmap N=30](p11i_grid_heatmaps_20260701_175955.png)

### D. Horizon N = 50 Heatmaps (280 Simulations)
![Heatmap N=50](p11i_grid_heatmaps_20260701_180141.png)

### E. Horizon N = 100 Heatmaps (280 Simulations)
![Heatmap N=100](p11i_grid_heatmaps_20260701_175714.png)

---

## 7. Raw Data Files
* **High-Res Data (N=30):** [p11i_grid_results_20260701_181951.csv](p11i_grid_results_20260701_181951.csv)
* **N = 25 Data:** [p11i_grid_results_20260701_174539.csv](p11i_grid_results_20260701_174539.csv)
* **N = 30 Data:** [p11i_grid_results_20260701_175955.csv](p11i_grid_results_20260701_175955.csv)
* **N = 50 Data:** [p11i_grid_results_20260701_180141.csv](p11i_grid_results_20260701_180141.csv)
* **N = 100 Data:** [p11i_grid_results_20260701_175714.csv](p11i_grid_results_20260701_175714.csv)
