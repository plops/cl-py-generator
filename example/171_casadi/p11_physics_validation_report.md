# Physics Validation & Gravity Sign Error Report

The user observed that the inverted pendulum simulation behaves in an unrealistic and non-physical manner:
1. The pendulum stable-oscillates near the vertical top ($\theta = 0$) without falling.
2. When the pendulum is near the horizontal or bottom ($\theta = \pi$), it behaves wildly, accelerating upwards and suddenly flipping direction.

This report presents a formal mathematical validation of the equations of motion implemented in the codebase, proving that the gravity vector's sign is inverted. This causes the simulation to behave as if gravity is pointing **upwards**.

---

## 1. Mathematical Derivation of the Equations of Motion

Let us define the coordinate system for a cart of mass $M$ at position $s$, with a pendulum of mass $m$ and length $l$ at angle $\theta$ (measured from the upward vertical, i.e., $\theta = 0$ is upright and $\theta = \pi$ is hanging down).

The position of the pendulum mass $m$ in Cartesian space is:
$$x_m = s + l \sin\theta$$
$$y_m = l \cos\theta$$

Taking the derivatives to obtain velocities:
$$\dot{x}_m = \dot{s} + l \dot{\theta} \cos\theta$$
$$\dot{y}_m = -l \dot{\theta} \sin\theta$$

### Kinetic Energy ($T$)
$$T = \frac{1}{2} M \dot{s}^2 + \frac{1}{2} m (\dot{x}_m^2 + \dot{y}_m^2)$$
$$T = \frac{1}{2} (M + m) \dot{s}^2 + m l \dot{s} \dot{\theta} \cos\theta + \frac{1}{2} m l^2 \dot{\theta}^2$$

### Potential Energy ($V$)
Under standard downward gravity ($g > 0$), the height of the mass is $y_m = l \cos\theta$, so:
$$V = m g y_m = m g l \cos\theta$$

### Lagrangian ($L = T - V$)
$$L = \frac{1}{2} (M + m) \dot{s}^2 + m l \dot{s} \dot{\theta} \cos\theta + \frac{1}{2} m l^2 \dot{\theta}^2 - m g l \cos\theta$$

### Euler-Lagrange Equations
1. **For cart position $s$:**
   $$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{s}}\right) - \frac{\partial L}{\partial s} = F$$
   $$(M+m)\ddot{s} + m l \ddot{\theta} \cos\theta - m l \dot{\theta}^2 \sin\theta = F \quad \text{(Eq. 1)}$$

2. **For pendulum angle $\theta$:**
   $$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\theta}}\right) - \frac{\partial L}{\partial \theta} = 0$$
   $$m l \ddot{s} \cos\theta + m l^2 \ddot{\theta} - m g l \sin\theta = 0 \implies \ddot{s} \cos\theta + l \ddot{\theta} - g \sin\theta = 0 \quad \text{(Eq. 2)}$$

---

## 2. Discrepancy in Codebase Equations

Solving (Eq. 1) and (Eq. 2) for accelerations $\ddot{s}$ (cart acceleration `dv`) and $\ddot{\theta}$ (angular acceleration `domega`) yields:

### Cart Acceleration $\ddot{s}$
$$\ddot{s} = \frac{F + m l \dot{\theta}^2 \sin\theta - m g \sin\theta \cos\theta}{M + m \sin^2\theta}$$

*   **Code Implementation** (e.g. in [p11a_pendulum_sim.py](file:///home/kiel/stage/cl-py-generator/example/171_casadi/p11a_pendulum_sim.py#L34)):
    `dv = (F_ + m * l * omega_ * omega_ * sin_theta + m * g * cos_theta * sin_theta) / den`
    > [!WARNING]
    > **Sign Mismatch:** The gravity term in the codebase has a **positive** sign (`+ m * g * cos_theta * sin_theta`), whereas the physical derivation has a **negative** sign (`- m * g * sin_theta * cos_theta`).

### Angular Acceleration $\ddot{\theta}$
$$\ddot{\theta} = \frac{- F \cos\theta - m l \dot{\theta}^2 \sin\theta \cos\theta + (M+m) g \sin\theta}{l (M + m \sin^2\theta)}$$

*   **Code Implementation** (e.g. in [p11a_pendulum_sim.py](file:///home/kiel/stage/cl-py-generator/example/171_casadi/p11a_pendulum_sim.py#L36-L40)):
    `domega = ((-1.0 * F_ * cos_theta) - (m * l * omega_ * omega_ * sin_theta * cos_theta) - ((M + m) * g * sin_theta)) / (l * den)`
    > [!WARNING]
    > **Sign Mismatch:** The gravity term in the codebase has a **negative** sign (`- (M + m) * g * sin_theta`), whereas the physical derivation has a **positive** sign (`+ (M + m) * g * sin_theta`).

---

## 3. Eigenvalue Validation Proof

To prove that this sign flip inverts gravity, we linearize the system around its equilibria with zero control force ($F=0$). The states are $x = [s, v, \theta, \omega]^T$.

We run the eigenvalue calculation script [p11_physics_validation.py](file:///home/kiel/stage/cl-py-generator/example/171_casadi/p11_physics_validation.py). The linear system matrix $A = \frac{\partial f}{\partial x}\Big|_{x_e}$ determines stability via its eigenvalues:

### Case A: Linearized around $\theta = 0$ (Upright Position)
*   **Correct Physical Dynamics:**
    $$A_{\text{correct}} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & -m g / M & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & (M+m)g/(lM) & 0 \end{bmatrix}$$
    *   **Eigenvalues:** `[0, 0, 4.65, -4.65]` (Real eigenvalues $\implies$ **Unstable Saddle Point**). The pendulum naturally falls over when perturbed.

*   **Codebase Dynamics:**
    $$A_{\text{code}} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & m g / M & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & -(M+m)g/(lM) & 0 \end{bmatrix}$$
    *   **Eigenvalues:** `[0, 0, +4.65j, -4.65j]` (Purely imaginary eigenvalues $\implies$ **Stable Center**). The pendulum behaves like a stable oscillator at the top.

### Case B: Linearized around $\theta = \pi$ (Hanging Position)
*   **Correct Physical Dynamics:**
    *   **Eigenvalues:** `[0, 0, +4.65j, -4.65j]` (Purely imaginary $\implies$ **Stable Center**). The pendulum stable-oscillates at the bottom.

*   **Codebase Dynamics:**
    *   **Eigenvalues:** `[0, 0, 4.65, -4.65]` (Real eigenvalues $\implies$ **Unstable Saddle Point**). The pendulum is unstable at the bottom and falls upward.

---

## 4. Physical Explanation of the Observed Behavior

1.  **Stable Top Position ($\theta = 0$):**
    Because of the inverted gravity signs, the potential energy landscape is flipped. The top position behaves like the bottom of a potential well. As a result, the pendulum naturally swings back to the top and oscillates stably there. This is why it never falls in the GUI when initialized upright.
2.  **Flipping Near Horizontal ($\theta \approx \pi/2$):**
    In the real world, gravity pulls a horizontal pendulum downwards. In the simulation, gravity pulls it **upwards** (towards $\theta = 0$). When the pendulum is released near the bottom ($\theta \approx \pi$), it is sitting on a potential hill. It accelerates rapidly away from the bottom and "falls upward" towards the top, creating the unrealistic whip-like snapping and direction flipping observed.

---

## 5. Affected Files

The incorrect dynamics equations appear in the following generator files:
*   [gen11a.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11a.lisp#L58-L60)
*   [gen11b.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11b.lisp#L34-L36)
*   [gen11c.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11c.lisp#L58-L60)
*   [gen11d.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11d.lisp#L52-L54)
*   [gen11e.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11e.lisp#L56-L58)
*   [gen11f.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11f.lisp#L58-L60)
*   [gen11g.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11g.lisp#L58-L66)
*   [gen11h.lisp](file:///home/kiel/stage/cl-py-generator/example/171_casadi/gen11h.lisp#L36-L44)
