# Multi-Dimensional Inverted Pendulum MPC Grid Sweep

This report documents the performance benchmarks, stability analysis, and physical findings of the inverted pendulum Model Predictive Controller (MPC) across step sizes ($h_{mpc}$), pendulum lengths ($l$), and mass ratios ($m/M$) for different prediction horizons ($N=25$, $N=30$, $N=50$, and $N=100$).

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

## 3. Comparative Summary of Sweep Horizons

Across the 280 simulations of the parameter grid, changing the prediction horizon $N$ has a profound effect on the overall success rate (pendulum stabilized within 12.0s without track boundary violations):

| Prediction Horizon | Successful Runs | Success Rate | Average Simulation Time |
| :---: | :---: | :---: | :---: |
| **$N = 25$** | 108 / 280 | **38.6%** | **32.23 s** (0.115 s/sim) |
| **$N = 30$** | 88 / 280 | **31.4%** | **35.62 s** (0.127 s/sim) |
| **$N = 50$** | 42 / 280 | **15.0%** | **41.48 s** (0.148 s/sim) |
| **$N = 100$** | 4 / 280 | **1.4%** | **35.32 s** (0.126 s/sim) |

---

## 4. Horizon N = 25 Sweep Results
* **Execution Time:** 32.23 seconds (0.115s/sim average).
* **Success Rate:** 38.6% (108/280).
* **Optimal Step Size ($h_{mpc}$):** Peaked at $h_{mpc} = 0.047\,\text{s}$ with **62.5%** success rate.
* **Length ($l$):** Longer pendulums ($l \ge 1.49\,\text{m}$) were stable (54.3% success) due to slower falling dynamics.

---

## 5. Horizon N = 30 Sweep Results
* **Execution Time:** 35.62 seconds (0.127s/sim average).
* **Success Rate:** 31.4% (88/280).
* **Optimal Step Size ($h_{mpc}$):** Peaked at $h_{mpc} = 0.033\,\text{s}$ with **72.5%** success rate.
* **Step Size Analysis:**
  * $h_{mpc} = 0.020\,\text{s}$: 22.5% success (up from 10.0% at $N=25$).
  * $h_{mpc} = 0.033\,\text{s}$: **72.5%** success (up from 55.0% at $N=25$).
  * $h_{mpc} \ge 0.060\,\text{s}$: Success rates dropped sharply (e.g., 17.5% at $h=0.073\,\text{s}$).

---

## 6. Horizon N = 50 Sweep Results
* **Execution Time:** 41.48 seconds (0.148s/sim average).
* **Success Rate:** 15.0% (42/280).
* **Optimal Step Size ($h_{mpc}$):** Peaked at $h_{mpc} = 0.020\,\text{s}$ with **52.5%** success rate.
* **Step Size Analysis:**
  * $h_{mpc} = 0.020\,\text{s}$: **52.5%** success (up from 22.5% at $N=30$ and 10.0% at $N=25$).
  * $h_{mpc} = 0.033\,\text{s}$: 47.5% success.
  * $h_{mpc} \ge 0.047\,\text{s}$: Drop to 5.0% at $h=0.047\,\text{s}$ and 0.0% for all larger step sizes.
* **Length ($l$):** Shorter pendulums ($l \le 0.46\,\text{m}$) were completely uncontrollable (0.0% success).

---

## 7. Horizon N = 100 Sweep Results
* **Execution Time:** 35.32 seconds (0.126s/sim average).
* **Success Rate:** 1.4% (4/280).
* **Solver Convergence:** 276 out of 280 simulations failed due to IPOPT solver failures (`success = False`).

---

## 8. Comparative Analysis: The Shifting Sweet Spot and Solver Robustness

The multi-horizon comparison reveals a very clear, physical, and numerical trend in Model Predictive Control:

1. **The Shifting Optimal Step Size (Sweet Spot):**
   As the prediction horizon $N$ increases, the optimal step size $h_{mpc}$ **shifts steadily towards smaller values**:
   * **For $N=25$**, the optimum is at $h = 0.047\,\text{s}$ (62.5% success).
   * **For $N=30$**, the optimum shifts to $h = 0.033\,\text{s}$ (72.5% success).
   * **For $N=50$**, the optimum shifts to $h = 0.020\,\text{s}$ (52.5% success).
2. **Physical Reason (Lookahead Window):**
   To successfully swing up and stabilize the pendulum, the controller requires an absolute prediction window ($N \times h_{mpc}$) in the range of **0.8 s to 1.5 s**:
   * At $N=25$, a small step size of $0.02\,\text{s}$ covers only 0.5s of the future (too short/blind).
   * At $N=50$, a small step size of $0.02\,\text{s}$ covers exactly 1.0s of the future (perfect lookahead window).
3. **Numerical Reason (Solver Convergence & Discretization):**
   When $N$ is large, using a larger step size (e.g., $h \ge 0.060\,\text{s}$) results in a very long absolute preview window ($> 3.0\,\text{s}$ for $N=50$). Over this long window, the cumulative discretization errors of the collocation scheme are high, and the linear interpolation cold-start guess is extremely far from physical reality. This causes IPOPT to get trapped in local infeasibilities and fail.
4. **Conclusion:**
   For real-time control, **shorter horizons ($N=25$ to $30$) are numerically much more robust** and allow stable convergence across a wide range of step sizes. Larger horizons ($N \ge 50$) can only stabilize the system at very small step sizes, where the discretization error is low and the problem remains mathematically solvable.

---

## 9. Heatmap Visualizations

### A. Horizon N = 25 Heatmaps
![Heatmap N=25](p11i_grid_heatmaps_20260701_174539.png)

### B. Horizon N = 30 Heatmaps
![Heatmap N=30](p11i_grid_heatmaps_20260701_175955.png)

### C. Horizon N = 50 Heatmaps
![Heatmap N=50](p11i_grid_heatmaps_20260701_180141.png)

### D. Horizon N = 100 Heatmaps
![Heatmap N=100](p11i_grid_heatmaps_20260701_175714.png)

---

## 10. Raw Data Files
* **N = 25 Data:** [p11i_grid_results_20260701_174539.csv](p11i_grid_results_20260701_174539.csv)
* **N = 30 Data:** [p11i_grid_results_20260701_175955.csv](p11i_grid_results_20260701_175955.csv)
* **N = 50 Data:** [p11i_grid_results_20260701_180141.csv](p11i_grid_results_20260701_180141.csv)
* **N = 100 Data:** [p11i_grid_results_20260701_175714.csv](p11i_grid_results_20260701_175714.csv)
