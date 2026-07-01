# Multi-Dimensional Inverted Pendulum MPC Grid Sweep

This report documents the performance benchmarks, stability analysis, and physical findings of the inverted pendulum Model Predictive Controller (MPC) across step sizes ($h_{mpc}$), pendulum lengths ($l$), and mass ratios ($m/M$) for different prediction horizons ($N=25$, $N=30$, $N=50$, and $N=100$), culminating in two comparative high-resolution 4160-simulation grid sweeps under different control force limits ($20.0\,\text{N}$ vs. $40.0\,\text{N}$).

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

## 3. High-Resolution Sweep Comparative Summary (N = 30)

To map the stability boundary in fine detail, two high-resolution grid sweeps of **4,160 simulations** each were conducted. We limited the step size range $h_{mpc}$ to the active stable band ($[0.02, 0.06]\,\text{s}$) and compared the system behavior under two actuator force limits ($20.0\,\text{N}$ vs. $40.0\,\text{N}$):

| Actuator Force Limit | Successful Runs | Success Rate | Average Simulation Time |
| :---: | :---: | :---: | :---: |
| **$20.0\,\text{N}$** | 2,325 / 4,160 | **55.9%** | **694.57 s** (0.167 s/sim) |
| **$40.0\,\text{N}$** | 2,680 / 4,160 | **64.4%** | **701.85 s** (0.169 s/sim) |

---

## 4. Parameter Stability Comparison: 20N vs. 40N

### A. Influence of Mass Ratio ($m/M$)
Increasing the mass ratio (heavier pendulum bob relative to the cart) increases inertia. Doubling the control force limit to $40.0\,\text{N}$ drastically mitigates actuator saturation:

| Mass Ratio ($m/M$) | 20N Success Rate | 20N Mean Settling Time | 40N Success Rate | 40N Mean Settling Time |
| :---: | :---: | :---: | :---: | :---: |
| 0.050 | 66.0% | 5.82 s | **70.6%** | **4.99 s** |
| 0.114 | 61.7% | 5.91 s | **69.4%** | **5.13 s** |
| 0.179 | 61.2% | 6.00 s | **69.0%** | **5.16 s** |
| 0.243 | 58.7% | 6.12 s | **64.4%** | **5.33 s** |
| 0.307 | 54.2% | 6.23 s | **63.7%** | **5.38 s** |
| 0.371 | 50.0% | 6.42 s | **62.1%** | **5.45 s** |
| 0.436 | 48.8% | 6.43 s | **58.8%** | **5.59 s** |
| 0.500 | 46.5% | 6.46 s | **57.3%** | **5.74 s** |

* **Analysis:** With $40.0\,\text{N}$ of force, the heaviest pendulum ($m/M = 0.500$) achieves a success rate of **57.3%** (higher than the $m/M = 0.307$ case under $20.0\,\text{N}$). Furthermore, the mean settling time for all mass ratios drops by **~0.8 seconds** because the cart can apply larger forces to pump energy into the swing-up and stabilize the pendulum much faster.

### B. Influence of Step Size ($h_{mpc}$)
| Step Size ($h_{mpc}$) | 20N Success Rate | 40N Success Rate | Lookahead Window ($N \times h_{mpc}$) |
| :---: | :---: | :---: | :---: |
| 0.020 s | 17.3% | **24.5%** | 0.60 s (Too short / blind) |
| 0.024 s | 40.4% | **61.5%** | 0.72 s |
| 0.028 s | 60.1% | **75.5%** | 0.84 s |
| **0.033 s** | **72.6%** | **88.0%** | 0.99 s (Optimal Peak) |
| 0.037 s | 69.7% | **79.8%** | 1.11 s |
| 0.041 s | 71.6% | **72.1%** | 1.23 s |
| 0.045 s | 58.2% | **68.8%** | 1.35 s |
| 0.049 s | 58.2% | **60.6%** | 1.47 s |
| 0.054 s | 51.4% | **57.7%** | 1.62 s |
| 0.058 s | 43.3% | **51.4%** | 1.74 s |
| 0.060 s | 41.8% | **45.2%** | 1.80 s |

* **Analysis:** The peak success rate at $h_{mpc} = 0.033\,\text{s}$ rises to an outstanding **88.0%** under $40.0\,\text{N}$. The overall success rate is higher across all step sizes, widening the stable operating window of the controller.

### C. Influence of Pendulum Length ($l$)
| Length ($l$) | 20N Success Rate | 40N Success Rate | Physical Characteristics |
| :---: | :---: | :---: | :--- |
| **0.20 m** | 10.6% | **78.1%** | Rapid fall (High natural frequency $\omega \propto \sqrt{g/l}$) |
| 0.34 m | 30.0% | **41.9%** | |
| 0.56 m | 39.4% | **45.6%** | |
| 0.78 m | 50.6% | **53.1%** | |
| 0.99 m | 68.1% | **63.7%** | |
| 1.21 m | 76.2% | **76.9%** | Optimal Region |
| **1.42 m** | 73.8% | **83.8%** | Peak Region (40N) |
| 1.64 m | 66.9% | **79.4%** | |
| 1.86 m | 61.3% | **75.0%** | |
| 2.00 m | 53.1% | **70.0%** | Slow dynamics, but requires large cart displacement |

* **The Short Pendulum Miracle ($l = 0.20\,\text{m}$):**
  At $20.0\,\text{N}$, the success rate for $l = 0.20\,\text{m}$ was only **10.6%**. Under $40.0\,\text{N}$, it skyrockets to **78.1%**!
  * **Physical Explanation:** Shorter pendulums fall extremely fast due to their high natural frequency. To catch them, the cart must accelerate aggressively before the angle deviates beyond a critical tip-over threshold. At $20.0\,\text{N}$, the cart was too slow to catch the pendulum. At $40.0\,\text{N}$, the cart can accelerate twice as fast, successfully catching and stabilizing the short pendulum in almost all step configurations.
* **Long Pendulums ($l = 2.00\,\text{m}$):**
  Success rate increases from 53.1% to **70.0%** because the higher force limit allows the cart to swing up the massive, long pendulum quickly enough to stabilize it within the 12.0s limit without drifting off track.

---

## 5. Comparative Summary of Sweep Horizons

To evaluate horizon scalability, we compare original parameters across different horizons:

| Prediction Horizon | Successful Runs | Success Rate | Average Simulation Time |
| :---: | :---: | :---: | :---: |
| **$N = 25$** | 108 / 280 | **38.6%** | **32.23 s** (0.115 s/sim) |
| **$N = 30$** | 88 / 280 | **31.4%** | **35.62 s** (0.127 s/sim) |
| **$N = 50$** | 42 / 280 | **15.0%** | **41.48 s** (0.148 s/sim) |
| **$N = 100$** | 4 / 280 | **1.4%** | **35.32 s** (0.126 s/sim) |

---

## 6. Heatmap Visualizations

### A. High-Resolution Heatmaps (N = 30, max_force = 40.0N, 4160 Simulations)
![High-Res Heatmap 40N](p11i_grid_results_20260701_184125_no_annotate.png)

### B. High-Resolution Heatmaps (N = 30, max_force = 20.0N, 4160 Simulations)
![High-Res Heatmap 20N](p11i_grid_results_20260701_181951_no_annotate.png)

### C. Horizon N = 25 Heatmaps (max_force = 20.0N, 280 Simulations)
![Heatmap N=25](p11i_grid_heatmaps_20260701_174539.png)

### D. Horizon N = 30 Heatmaps (max_force = 20.0N, 280 Simulations)
![Heatmap N=30](p11i_grid_heatmaps_20260701_175955.png)

### E. Horizon N = 50 Heatmaps (max_force = 20.0N, 280 Simulations)
![Heatmap N=50](p11i_grid_heatmaps_20260701_180141.png)

### F. Horizon N = 100 Heatmaps (max_force = 20.0N, 280 Simulations)
![Heatmap N=100](p11i_grid_heatmaps_20260701_175714.png)

---

## 7. Raw Data Files
* **High-Res 40N Data:** [p11i_grid_results_20260701_184125.csv](p11i_grid_results_20260701_184125.csv)
* **High-Res 20N Data:** [p11i_grid_results_20260701_181951.csv](p11i_grid_results_20260701_181951.csv)
* **N = 25 Data:** [p11i_grid_results_20260701_174539.csv](p11i_grid_results_20260701_174539.csv)
* **N = 30 Data:** [p11i_grid_results_20260701_175955.csv](p11i_grid_results_20260701_175955.csv)
* **N = 50 Data:** [p11i_grid_results_20260701_180141.csv](p11i_grid_results_20260701_180141.csv)
* **N = 100 Data:** [p11i_grid_results_20260701_175714.csv](p11i_grid_results_20260701_175714.csv)
