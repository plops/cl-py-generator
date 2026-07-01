# Multi-Dimensional Inverted Pendulum MPC Grid Sweep

This report documents the performance benchmarks, stability analysis, and physical findings of the inverted pendulum Model Predictive Controller (MPC) across step sizes ($h_{mpc}$), pendulum lengths ($l$), and mass ratios ($m/M$) for different prediction horizons ($N=25$, $N=30$, and $N=100$).

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
| **$N = 100$** | 4 / 280 | **1.4%** | **35.32 s** (0.126 s/sim) |

---

## 4. Horizon N = 25 Sweep Results
* **Execution Time:** 32.23 seconds (0.115s/sim average).
* **Success Rate:** 38.6% (108/280).

### Parameter Stability Analysis (N = 25)
* **Optimal Step Size ($h_{mpc}$):** Peaked at $h_{mpc} = 0.047\,\text{s}$ with **62.5%** success rate.
* **Optimal Length ($l$):** Longer pendulums ($l \ge 1.49\,\text{m}$) were significantly more stable (54.3% success) because their slower falling dynamics gave the controller more time to react.
* **Actuator Saturation:** Heavy mass ratios ($m/M = 0.50$) had a lower success rate (30.4%) due to control force limit saturation ($20.0\,\text{N}$).

---

## 5. Horizon N = 30 Sweep Results
* **Execution Time:** 35.62 seconds (0.127s/sim average).
* **Success Rate:** 31.4% (88/280).

### Parameter Stability Analysis (N = 30)
* **Optimal Step Size ($h_{mpc}$):** Peaked at $h_{mpc} = 0.033\,\text{s}$ with **72.5%** success rate.
* **Mass Ratio:**
  * $m/M = 0.05$: 39.3% success.
  * $m/M = 0.50$: 25.0% success.
* **Step Size Analysis:**
  * $h_{mpc} = 0.020\,\text{s}$: 22.5% success (up from 10.0% at $N=25$).
  * $h_{mpc} = 0.033\,\text{s}$: **72.5%** success (up from 55.0% at $N=25$).
  * $h_{mpc} \ge 0.060\,\text{s}$: Success rates dropped sharply (e.g., 17.5% at $h=0.073\,\text{s}$, down from 50.0% at $N=25$).

---

## 6. Horizon N = 100 Sweep Results
* **Execution Time:** 35.32 seconds (0.126s/sim average).
* **Success Rate:** 1.4% (4/280).
* **Solver Convergence:** 276 out of 280 simulations failed due to IPOPT solver failures (`success = False`).

---

## 7. Comparative Analysis: Sweet Spot Shift and Solver Robustness

The comparison reveals a highly interesting physical and numerical trade-off in Model Predictive Control:

1. **Shifting Sweet Spot ($N=25$ vs. $N=30$):**
   * **For $N=25$**, the optimal step size is $h_{mpc} = 0.047\,\text{s}$ (62.5% success).
   * **For $N=30$**, the optimal step size shifts to $h_{mpc} = 0.033\,\text{s}$ (achieving a peak success rate of **72.5%**).
   * **Why?** At smaller step sizes (like $0.020\,\text{s}$ and $0.033\,\text{s}$), the slightly longer horizon $N=30$ provides a larger absolute preview window ($30 \times h_{mpc}$), giving the controller enough lookahead (0.6s to 1.0s) to plan a swing-up.
2. **Discretization and Infeasibility Failure at High Step Sizes:**
   * For larger step sizes ($h_{mpc} \ge 0.060\,\text{s}$), the success rate for $N=30$ is much worse than $N=25$.
   * **Why?** A larger horizon combined with large step sizes leads to a very long preview window (e.g., $30 \times 0.073 = 2.19\,\text{s}$). Over such a long window, the cumulative discretization errors of the collocation scheme are very high, and the linear interpolation cold-start guess is extremely far from physical reality. This causes IPOPT to fail with `EXIT: Converged to a point of local infeasibility`.
3. **Horizon Scalability Limits ($N=100$):**
   * At $N=100$ (5.0s preview), the optimization problem contains 1700 variables. The non-convexity is so high that the simple linear initial guess fails to converge on almost all steps, resulting in a 98.6% failure rate.
   * **Conclusion:** Shorter prediction horizons are numerically much more robust for online MPC unless a high-quality trajectory generator is used to seed the initial guess of the solver.

---

## 8. Heatmap Visualizations

### A. Horizon N = 25 Heatmaps
![Heatmap N=25](p11i_grid_heatmaps_20260701_174539.png)

### B. Horizon N = 30 Heatmaps
![Heatmap N=30](p11i_grid_heatmaps_20260701_175955.png)

### C. Horizon N = 100 Heatmaps
![Heatmap N=100](p11i_grid_heatmaps_20260701_175714.png)

---

## 9. Raw Data Files
* **N = 25 Data:** [p11i_grid_results_20260701_174539.csv](p11i_grid_results_20260701_174539.csv)
* **N = 30 Data:** [p11i_grid_results_20260701_175955.csv](p11i_grid_results_20260701_175955.csv)
* **N = 100 Data:** [p11i_grid_results_20260701_175714.csv](p11i_grid_results_20260701_175714.csv)
