# Multi-Dimensional Inverted Pendulum MPC Grid Sweep

This report documents the performance benchmarks, stability analysis, and physical findings of the inverted pendulum Model Predictive Controller (MPC) across step sizes ($h_{mpc}$), pendulum lengths ($l$), and mass ratios ($m/M$).

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

> [!NOTE]
> For small horizons (like $N=20$), the JIT compilation overhead (~2.5s) dominates short runs. However, for larger horizons ($N \ge 100$) or large sweeps, JIT delivers a massive **9x speedup** on active calculations.

---

## 3. Full Sweep Execution
* **Horizon $N$:** 25
* **Max Control Force:** 20.0 N
* **Total Simulations:** 280
* **Execution Time:** **32.23 seconds** (average of **0.115s** per simulation run).

---

## 4. Parameter Stability Analysis
Out of 280 simulations, **108 runs (38.6%)** successfully stabilized the pendulum at the upright position without violating track boundaries ($\pm 5\,\text{m}$) or causing solver failures.

### A. Influence of Mass Ratio ($m/M$)
A heavier pendulum bob (higher mass ratio) creates more inertia, making stabilization more difficult:

| Mass Ratio ($m/M$) | Success Rate | Mean Stabilization Time |
| :---: | :---: | :---: |
| 0.05 | 44.6% | 5.37 s |
| 0.16 | 42.9% | 5.68 s |
| 0.28 | 42.9% | 5.79 s |
| 0.39 | 32.1% | 5.97 s |
| 0.50 | 30.4% | 6.02 s |

* **Physical Trend:** As $m/M$ increases from $0.05$ to $0.50$, the success rate drops by 14.2% and the stabilization time increases. A heavier pendulum requires larger forces, which easily saturates the actuator limit of $20\,\text{N}$, leading to instability.

### B. Influence of Step Size ($h_{mpc}$)
The choice of discretization step size $h_{mpc}$ is a critical trade-off between the prediction horizon window and integration accuracy:

| Step Size ($h_{mpc}$) | Success Rate | Horizon Window ($N \times h_{mpc}$) |
| :---: | :---: | :---: |
| 0.020 s | 10.0% | 0.50 s (Too short) |
| 0.033 s | 55.0% | 0.83 s |
| 0.047 s | **62.5%** | 1.18 s (Optimal) |
| 0.060 s | 60.0% | 1.50 s |
| 0.073 s | 50.0% | 1.83 s |
| 0.087 s | 25.0% | 2.18 s |
| 0.100 s | 7.5% | 2.50 s (Integration failure) |

* **Short Horizon Blindness ($h_{mpc} = 0.020\,\text{s}$):** The control horizon covers only 0.5s of the future. The controller cannot plan the multi-second swing-up trajectory and fails.
* **Optimal Window ($h_{mpc} \in [0.033, 0.073]\,\text{s}$):** High success rates peaking at $h_{mpc}=0.047\,\text{s}$ (62.5% success).
* **Discretization Divergence ($h_{mpc} \ge 0.087\,\text{s}$):** Large step sizes lead to high discretization errors in the Runge-Kutta 4 integrator, causing the model's predictions to diverge from continuous reality.

### C. Influence of Pendulum Length ($l$)
Shorter pendulums act as faster, high-frequency unstable systems, making control challenging:

| Length ($l$) | Success Rate |
| :---: | :---: |
| 0.20 m | 11.4% (Rapid fall) |
| 0.46 m | 17.1% |
| 0.71 m | 34.3% |
| 0.97 m | 42.9% |
| 1.23 m | 45.7% |
| 1.49 m | **54.3%** |
| 1.74 m | 48.6% |
| 2.00 m | **54.3%** |

* **Physical Trend:** Longer pendulums fall slower due to a lower gravitational angular frequency ($\omega \propto \sqrt{g/l}$). This grants the controller more time to coordinate cart movements, increasing stability up to 54.3%.

---

## 5. Stability Band Analysis (Heatmap Observations)
The generated heatmaps reveal a distinct **diagonal stable operating band**:

1. **The Diagonal Stable Region:**
   * For short lengths (e.g. $l = 0.20\,\text{m}$), the system is only stable at very small step sizes ($h_{mpc} \le 0.033\,\text{s}$). If $h_{mpc}$ is larger, the rapid gravitational fall of the short pendulum occurs faster than the controller's sampling interval, leading to immediate tips.
   * For longer lengths (e.g. $l = 2.00\,\text{m}$), the system is only stable at larger step sizes ($h_{mpc} \ge 0.047\,\text{s}$). If $h_{mpc}$ is too small, the prediction window ($25 \times h_{mpc}$) is too short to capture the slow pendulum dynamics required for swing-up planning.
2. **Shrinking Window via Mass Load:**
   * Comparing $m/M = 0.05$ to $m/M = 0.50$, the diagonal stable band shrinks significantly.
   * At $m/M = 0.50$, short pendulums ($l \le 0.46\,\text{m}$) are almost completely uncontrollable within the $20\,\text{N}$ force limits.

---

## 6. Heatmap Visualization
The subplots below illustrate the stabilization times across all step sizes ($h_{mpc}$, y-axis) and pendulum lengths ($l$, x-axis) for each mass ratio panel. Cells containing `NaN` represent unstable configurations:

![Heatmap Plot](p11i_grid_heatmaps_20260701_174539.png)

---

## 7. Raw Data
The numerical results of the sweep are stored in the same repository folder:
* **Numerical CSV Data:** [p11i_grid_results_20260701_174539.csv](p11i_grid_results_20260701_174539.csv)
