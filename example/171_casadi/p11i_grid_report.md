# Multi-Dimensional Inverted Pendulum MPC Grid Sweep

This report documents the performance benchmarks, stability analysis, and physical findings of the inverted pendulum Model Predictive Controller (MPC) across step sizes ($h_{mpc}$), pendulum lengths ($l$), and mass ratios ($m/M$) for different prediction horizons ($N=25$ vs. $N=100$).

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

> [!NOTE]
> For small horizons (like $N=20$), the JIT compilation overhead (~2.5s) dominates short runs. However, for larger horizons ($N \ge 100$) or large sweeps, JIT delivers a massive **9x speedup** on active calculations.

---

## 3. Horizon N = 25 Sweep Results
* **Horizon $N$:** 25
* **Max Control Force:** 20.0 N
* **Total Simulations:** 280
* **Execution Time:** **32.23 seconds** (average of **0.115s** per simulation run).

### Parameter Stability Analysis (N = 25)
Out of 280 simulations, **108 runs (38.6%)** successfully stabilized the pendulum at the upright position without violating track boundaries ($\pm 5\,\text{m}$) or causing solver failures.

#### A. Influence of Mass Ratio ($m/M$)
| Mass Ratio ($m/M$) | Success Rate | Mean Stabilization Time |
| :---: | :---: | :---: |
| 0.05 | 44.6% | 5.37 s |
| 0.16 | 42.9% | 5.68 s |
| 0.28 | 42.9% | 5.79 s |
| 0.39 | 32.1% | 5.97 s |
| 0.50 | 30.4% | 6.02 s |

#### B. Influence of Step Size ($h_{mpc}$)
| Step Size ($h_{mpc}$) | Success Rate | Horizon Window ($N \times h_{mpc}$) |
| :---: | :---: | :---: |
| 0.020 s | 10.0% | 0.50 s (Too short) |
| 0.033 s | 55.0% | 0.83 s |
| 0.047 s | **62.5%** | 1.18 s (Optimal) |
| 0.060 s | 60.0% | 1.50 s |
| 0.073 s | 50.0% | 1.83 s |
| 0.087 s | 25.0% | 2.18 s |
| 0.100 s | 7.5% | 2.50 s (Integration failure) |

#### C. Influence of Pendulum Length ($l$)
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

---

## 4. Horizon N = 100 Sweep Results
* **Horizon $N$:** 100
* **Max Control Force:** 20.0 N
* **Total Simulations:** 280
* **Execution Time:** **35.32 seconds** (average of **0.126s** per simulation run).

### Parameter Stability Analysis (N = 100)
Out of 280 simulations, **only 4 runs (1.4%)** successfully stabilized the pendulum. The remaining **276 runs (98.6%) failed due to IPOPT solver failures** (`success = False`).

| Mass Ratio ($m/M$) | Success Rate | Mean Stabilization Time |
| :---: | :---: | :---: |
| 0.05 | 0.0% | - |
| 0.16 | 0.0% | - |
| 0.28 | 0.0% | - |
| 0.39 | 5.4% | 5.17 s |
| 0.50 | 5.4% | 5.25 s |

---

## 5. Comparative Analysis: Why N = 100 Fails

The drastic drop in success rate from **38.6%** ($N=25$) to **1.4%** ($N=100$) exposes critical limitations in the numerical robustness of long-horizon MPC:

1. **Dimensionality & Non-Convexity:**
   At $N=100$, the NLP contains **1700 optimization variables** and **1700 constraints**. With a preview window of $5.0\,\text{s}$ (at $h_{mpc}=0.05\,\text{s}$), the trajectory spans multiple pendulum swings. The resulting optimization landscape has high non-convexity and multiple local minima.
2. **Cold Start Infeasibility:**
   The controller initializes the cold start trajectory (at step 0) using a straight linear interpolation from the current state (hanging down, $\theta=\pi$) to the target state (upright, $\theta=0$).
   * For $N=25$ (1.25s preview), the linear guess is close enough for IPOPT to find a feasible path.
   * For $N=100$ (5.0s preview), the linear guess violates dynamics equations so severely over 100 steps that IPOPT gets trapped in a local minimum and fails with `EXIT: Converged to a point of local infeasibility`.
3. **Robustness Trade-Off:**
   A shorter horizon of $N=25$ is mathematically and computationally much more robust for online MPC. It converges reliably from poor initial guesses and successfully stabilizes the pendulum, whereas a larger horizon requires advanced trajectory warm-starting or a pre-computed feasible seed trajectory to converge.

---

## 6. Heatmap Visualizations

### A. Horizon N = 25 Heatmaps
![Heatmap N=25](p11i_grid_heatmaps_20260701_174539.png)

### B. Horizon N = 100 Heatmaps
![Heatmap N=100](p11i_grid_heatmaps_20260701_175714.png)

---

## 7. Raw Data Files
* **N = 25 Data:** [p11i_grid_results_20260701_174539.csv](p11i_grid_results_20260701_174539.csv)
* **N = 100 Data:** [p11i_grid_results_20260701_175714.csv](p11i_grid_results_20260701_175714.csv)
