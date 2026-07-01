# Inverted Pendulum MPC Grid Search: Step Size vs. Pendulum Length

This report analyzes the results of a headless grid search performed on the inverted pendulum model in [gen11c.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11c.lisp). The sweep varies the MPC step size $h_{mpc}$ (equal to the simulation step size $dt_{sim}$) and the pendulum length $l$ to measure the time it takes the carriage and pendulum to swing up and stabilize.

---

## 1. Grid Search Results Table

The table below shows the stabilization time (in seconds) for each pair of $(h_{mpc}, l)$. A value of `nan` indicates that the controller failed to swing up the pendulum, crashed, or did not stabilize within the $12.0\,\text{s}$ limit.

```
h_mpc \ l       0.20    0.40    0.60    0.80    1.00    1.30    1.60    2.00
----------------------------------------------------------------------------
     0.020 |     nan     nan     nan     nan     nan     nan     nan     nan
     0.030 |    7.62    9.24   11.22     nan     nan     nan     nan     nan
     0.040 |     nan    5.04    6.88    8.00    8.96     nan     nan     nan
     0.050 |     nan     nan    4.65    5.15    5.45    9.50     nan     nan
     0.060 |     nan    4.56    4.62    4.86    5.04    5.40     nan    8.46
     0.080 |     nan     nan     nan    4.56    4.72    4.88    5.12    5.60
     0.100 |     nan     nan     nan    4.10     nan    4.70    4.70    5.20
```

---

## 2. Visualization

Below is the heatmap of the stabilization times showing the distribution of stable and unstable regions:

![Grid Search Heatmap](p11a_grid_heatmap.png)

---

## 3. Physical Analysis of the Stability Band

The results show a clear **diagonal stability band** stretching from the top-left (short $l$, small $h_{mpc}$) to the bottom-right (long $l$, large $h_{mpc}$). This is a direct consequence of two physical constraints:

### A. Short-Horizon Blindness (Why small $h_{mpc}$ fails for long pendulums)
We kept the number of prediction steps in the MPC fixed at $N = 20$. The physical time horizon of the MPC is $T_{horizon} = N \cdot h_{mpc}$.
*   For $h_{mpc} = 0.02\,\text{s}$, the horizon is only $0.4\,\text{s}$.
*   For $h_{mpc} = 0.03\,\text{s}$, the horizon is $0.6\,\text{s}$.
*   **The Physics:** A long pendulum (e.g. $l = 1.0\,\text{m}$) has a natural period of $T_p = 2\pi\sqrt{l/g} \approx 2.0\,\text{s}$. A short prediction horizon of $0.4\,\text{s}$ or $0.6\,\text{s}$ is completely blind to the swing-up physics. The MPC cannot see far enough into the future to plan a multi-stage rocking motion to build energy. As a result, the solver falls into a local minimum and the pendulum hangs down (`nan`).

### B. Discretization Instability (Why large $h_{mpc}$ fails for short pendulums)
*   For $h_{mpc} = 0.08\,\text{s}$ or $0.10\,\text{s}$, the controller only updates its actuator input every $80$ or $100$ milliseconds.
*   **The Physics:** A short pendulum (e.g. $l = 0.20\,\text{m}$) has a natural period of only $T_p \approx 0.9\,\text{s}$. It falls and moves very quickly. Holding the actuator force constant (Zero-Order Hold) for $100\,\text{ms}$ causes massive overcorrection. By the time the next step arrives, the pendulum has shot past vertical and is falling in the opposite direction. This discretization delay induces wild oscillations and leads to instability (`nan`).

### C. The Optimal Band
For a given length $l$, there is an optimal step size $h_{mpc}$ that balances these two limits. For example, for $l = 1.0\,\text{m}$:
*   At $h_{mpc} \le 0.03\,\text{s}$, $T_{horizon}$ is too short to plan the swing-up.
*   At $h_{mpc} = 0.04\,\text{s}$ ($T_{horizon} = 0.8\,\text{s}$), it stabilizes in $8.96\,\text{s}$.
*   At $h_{mpc} = 0.05\,\text{s}$ ($T_{horizon} = 1.0\,\text{s}$), it stabilizes in $5.45\,\text{s}$.
*   At $h_{mpc} = 0.06\,\text{s}$ ($T_{horizon} = 1.2\,\text{s}$), it stabilizes in $5.04\,\text{s}$.
*   At $h_{mpc} = 0.08\,\text{s}$ ($T_{horizon} = 1.6\,\text{s}$), it stabilizes in $4.72\,\text{s}$.

---

## 4. Role of the Mass Ratio ($m/M$)

The mass ratio between the pendulum tip ($m = 0.1\,\text{kg}$) and the carriage ($M = 1.0\,\text{kg}$) is currently fixed at $\mu = m/M = 0.1$. 

If we were to vary this mass ratio, it would scale the coupling forces according to the equations of motion:

1.  **If $m/M$ decreases (e.g., $0.01$):**
    *   The pendulum exerts negligible reaction forces on the cart. The coupling becomes one-way (the cart drives the pendulum, but the pendulum's swing does not push the cart around).
    *   The system becomes much easier to stabilize. The diagonal stability band would **widen**, allowing larger step sizes and shorter horizons to succeed.
2.  **If $m/M$ increases (e.g., $0.5$):**
    *   The heavy pendulum heavily couples with the cart. As it swings, it pushes the cart sideways, shifting the effective inertia of the carriage between $M$ and $M+m$.
    *   This state-dependent nonlinearity degrades the accuracy of the local linearizations used inside the MPC solver.
    *   To keep the optimization models valid, the controller would require a **smaller step size $h_{mpc}$** to react frequently to the reaction forces. The stability band would **shrink** and shift towards smaller step sizes.
