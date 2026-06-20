# Hénon-Heiles Galactic Orbit Simulation & Poincaré Sections

We have successfully formulated the Hénon-Heiles system, integrated it using CasADi's `cvodes` suite, and generated the orbit plots and Poincaré sections.

## Simulation Results

![Hénon-Heiles Chaos Orbits and Poincaré Sections](henon_heiles_chaos.png)

---

## Detailed Physical Analysis of the Visualizations

### 1. The Zero-Velocity Curves (ZVCs) & Forbidden Regions
The red dashed lines in the left plots represent the **Zero-Velocity Curves (Nullgeschwindigkeitskurven)** defined by $V(x, y) = E$.
* **Why are there no paths outside?** The total energy of the system is conserved: $H = T_{\text{kin}} + V(x, y) = E$. Because the kinetic energy $T_{\text{kin}} = \frac{1}{2}(p_x^2 + p_y^2)$ must always be non-negative ($T_{\text{kin}} \ge 0$), the star is energetically confined to regions where $V(x, y) \le E$. Crossing the red dashed line would require negative kinetic energy ($T_{\text{kin}} < 0$), which is physically impossible. At the boundary itself, the star's velocity is exactly zero.
* **The "Triangles that seem to rotate":** At the higher energy level ($E = 0.15$), the zero-velocity curve takes the shape of a triangle with rounded corners. This is due to the $C_3$ (threefold) symmetry of the cubic perturbation term $x^2y - \frac{1}{3}y^3$. The three vertices of this triangle point towards the potential's three saddle points (escape channels) located at $V = 1/6 \approx 0.1667$. The star orbits inside this triangular well, and because of non-linear coupling, the orbits precess over time. This precession creates the visual appearance of a rotating triangle.

### 2. Meaning of the Colored Orbits (Left Plots)
* Each colored curve represents a single star's trajectory originating from a different initial vertical displacement $y(0)$ in the meridian plane (with $x(0)=0, p_y(0)=0$, and $p_x(0) > 0$ calculated to satisfy the energy constraint $H = E$).
* The colors allow us to visually distinguish between different orbital families (such as regular box orbits versus chaotic orbits).

### 3. The Poincaré Sections (Right Plots)
* Slicing the 4D phase space $(x, y, p_x, p_y)$ at $x=0$ with $p_x > 0$ projects the 3D energy surface onto the $(y, p_y)$ plane.
* **At Low Energy ($E = 0.08333 \approx 1/12$):** The section shows clean, nested concentric loops. Each loop represents a stable 2D torus (Invariant Torus) in phase space, demonstrating the existence of a conserved **third integral of motion**.
* **At High Energy ($E = 0.15$):** The tori break down. We see a scattered "sea" of chaotic points, representing the loss of the third integral. However, small concentric islands of stability persist, representing resonant regular orbits (like 1:1 or 2:1 vertical-to-radial oscillations).

---

## Astrophysical Interpretation & Significance

The Hénon-Heiles system represents a landmark model for understanding **secular evolution** and **stochasticity** in real galaxies.

### 1. Stability of Galactic Bars
* **The "Skeleton" of a Bar:** In barred spiral galaxies, the structural integrity of the bar is held together by stars on stable, non-axisymmetric orbits (predominantly the $x_1$ orbit family) that stretch along the bar.
* **Dissolving Bars:** If stars within the bar gain energy or angular momentum (due to gravitational perturbations, gas accretion, or interaction with giant molecular clouds), they cross the chaotic energy threshold. Their orbits transition from stable tori into the "chaotic sea," causing them to disperse randomly. This process can weaken or completely dissolve the galactic bar over cosmological timescales.

### 2. Bulge Thickening and "Boxy/Peanut" Shapes
* **Vertical Resonances:** Edge-on observations of disk galaxies often reveal boxy or peanut-shaped central bulges. This profile is formed by stars that undergo vertical resonance, bending their orbits out of the galactic plane.
* **Growth of Central Masses:** When a galaxy grows a massive central bulge or a supermassive black hole (SMBH), the central potential steepens. This change disrupts regular box orbits, scattering stars into chaotic trajectories that diffuse vertically. This "chaotic heating" puffs up the galactic core, turning a thin triaxial distribution into a thick boxy/peanut bulge.

### 3. Chaotic Diffusion
* Stars on chaotic orbits undergo slow diffusion in phase space. Over billions of years, this chaotic diffusion dynamically heats the stellar disk and alters the density profiles of galactic halos, explaining the fuzzy, non-uniform structures observed in stellar envelopes.

---

## Code Files
* **Lisp Generator Code:** [gen02.lisp](gen02.lisp)
* **Generated Python Code:** [p02_hh.py](p02_hh.py)
