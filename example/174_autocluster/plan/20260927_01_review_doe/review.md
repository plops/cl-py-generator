Hier ist ein detailliertes Review der Codebasis und Methodik sowie ein konkreter Vorschlag, wie ein **Design of Experiments (DoE)** die Analyse und Parameterauswahl wissenschaftlich robuster und verlässlicher machen kann.

---

# Teil 1: Projekt-Review

### Stärken der aktuellen Umsetzung
1. **Methodisch korrekte Validierung (Kern-Idee):**
   Der Ansatz, im reduzierten Raum zu clustern ($d \in [2, 16]$), aber **im hochdimensionalen Originalraum ($k$-D)** via Kosinus-Silhouette zu validieren, ist exzellent. Dies vermeidet die klassische Falle, bei der Projektionsartefakte von UMAP (z. B. extreme Klumpenbildung bei $d=2, \text{min\_dist}=0.0$) fälschlicherweise als „gutes Clustering“ bewertet werden.
2. **Saubere Matryoshka-Nutzung & BLOB-Handling:**
   Die Dekodierung (`little-endian float32`) und das Prefix-Trunkieren in `loader.py` sind elegant und entsprechen genau der Google-Embedding-Spezifikation.
3. **GPU-Effizienz & Caching:**
   Die Verwendung von cuML/CuPy mit `.npy`-Zwischenspeichern der UMAP-Einbettungen trennt teure Manifold-Berechnung von schneller Cluster-Iteration (1–2 Sekunden pro Run).
4. **Bestrafung von Rauschen:**
   Die modifizierte Zielfunktion $\text{Score} = \text{Silhouette} \times (1 - \text{NoiseRatio})$ verhindert wirksam, dass HDBSCAN 90 % der Punkte als Noise markiert, um eine künstlich hohe Silhouette auf einem winzigen dichten Restkern zu erzielen.

---

### Schwachstellen, Risiken und Unstimmigkeiten

#### 1. Methodischer Bias beim Breitenvergleich (Survival Bias / Äpfel mit Birnen)
In `loader.py` werden Vektoren verworfen, die kürzer als die angeforderte Breite $k$ sind:
```python
if vec.shape[0] < width:
    skipped["short"] += 1
```
* **Auswirkung:** Bei $k=128$ und $k=768$ werden $18.269$ Dokumente ausgewertet, bei $k=3072$ aber nur $16.692$ Dokumente (1.577 Einträge fallen weg).
* **Problem:** Die 1.577 herausgefallenen Vektoren könnten minderwertige Zusammenfassungen, Randfälle oder Rauschen gewesen sein. Der Silhouette-Vorteil von $k=3072$ (+4,6 %) könnte zum Teil schlicht darauf beruhen, dass **auf einer kleineren, bereinigten Teilmenge** gerechnet wurde.
* **Fix:** Für einen echten Ablationsvergleich müssen alle Breiten auf der **Schnittmenge** (also den $16.692$ Zeilen, die für alle Breiten verfügbar sind) evaluiert werden.

#### 2. Scheingenauigkeit vs. Stochastisches Rauschen (`nn_descent`)
Im Bericht wird erwähnt: *„d ∈ [4, 16] liegt überall innerhalb ±0,002 adj.“* und *„ein Refit liefert 178 statt 167 Cluster“*.
* **Problem:** cuMLs `build_algo='nn_descent'` ist inhärent nicht-deterministisch (Race Conditions beim parallelen Graph-Aufbau auf der GPU). Wenn der stochastische Jitter zwischen zwei Runs mit identischen Parametern bereits eine Score-Schwankung von z. B. $\pm 0,004$ erzeugt, ist der Unterschied zwischen $d=8, 12, 16$ reines Messrauschen. Der Sieger $d=12$ ist statistisch womöglich nicht von $d=8$ unterscheidbar.

#### 3. Unfairer Vergleich: HDBSCAN vs. DBSCAN
DBSCAN wurde mit einer fixen linearen Heuristik betrieben (`eps(d) = 0.3 + d * 0.02`), während HDBSCAN hierarchisch dichte-adaptiv arbeitet. Das Fazit *„HDBSCAN deklassiert DBSCAN überall“* ist ein Zirkelschluss: DBSCAN versagt hier vor allem, weil eine geratene lineare Epsilon-Funktion den Dichtesprung höherer Dimensionen nicht adäquat abbildet.

#### 4. GUI-/Plotting-Bugs in `plot_final.py`
```python
for c in sorted(set(labels) - {-1}):
    m = labels == c
    ax.scatter(pts[m, 0], pts[m, 1], s=4, label="c%d (n=%d)" % (c, m.sum()))
ax.legend(markerscale=3, fontsize="small")
```
Bei $167$ Clustern versucht Matplotlib, eine Legende mit 168 Einträgen zu zeichnen. Das überdeckt entweder den gesamten Plot unbrauchbar oder führt zu Layout-Warnungen.

---

# Teil 2: Mehr Robustheit durch Design of Experiments (DoE)

Aktuell wird ein starrer **Full-Factorial Grid Sweep** gefahren ($5 \times 2 \times 2 = 20$ Punkte pro Breite). Die Stützstellen sind extrem grob (nur 2 Werte für `n_neighbors` [15, 30] und `min_dist` [0.0, 0.1]). Wichtige Parameter wie `min_cluster_size` bei HDBSCAN blieben völlig fixiert ($=15$).

### Warum DoE die Auswertung drastisch verbessert:
1. **Signal-Rausch-Trennung (ANOVA):**
   DoE quantifiziert über Varianzanalysen (F-Test / p-Werte), ob ein Parameter (z. B. $d$) überhaupt einen signifikanten Haupteffekt hat oder ob seine Variation im Rauschen von `nn_descent` untergeht.
2. **Erkennung von Interaktionseffekten:**
   UMAP glättet Dichten; HDBSCAN sucht Dichten. Der Parameter `n_neighbors` (UMAP) interagiert mit hoher Wahrscheinlichkeit mit `min_cluster_size` oder `min_samples` (HDBSCAN). Diese Wechselwirkungen bleiben im aktuellen Setup unsichtbar.
3. **Robust Design (Taguchi-Prinzip):**
   In industriellen DoE-Ansätzen unterscheidet man zwischen **Steuergrößen** (z. B. $d$, `min_dist`) und **Störgrößen** (z. B. GPU-Random-Seeds, Rauschen in Web-Texten). Ziel ist es nicht, die Konfiguration zu finden, die *einmalig* den höchsten Peak-Score wirft, sondern die Konfiguration, die **minimal streut**, wenn der Seed wechselt.

---

# Teil 3: Wie ein solches DoE konkret aussehen würde

Ein solides DoE für diese Pipeline gliedert sich in drei Phasen:

```
[Phase 1: Screening & Schnittmenge]
       │
       ▼
[Phase 2: Response Surface / Robust Design mit Störfaktor]
       │
       ▼
[Phase 3: Statistische Auswertung & Optimum]
```

### 1. Faktorenraum definieren

| Faktor | Typ | Wertebereich / Stufen | Rolle |
|---|---|---|---|
| **Daten-Schnittmenge** | Fix | Nur Zeilen mit vollen 3072D ($N=16.692$) | Kontrollvariable (Verzerrungsfreiheit) |
| **Embedding-Breite $k$** | Kategorial | $[128, 768, 3072]$ | Primärer Untersuchungsfaktor |
| **UMAP $d$** | Stetig | $[4, 20]$ | Steuergröße |
| **UMAP $n\_neighbors$** | Stetig | $[15, 60]$ | Steuergröße |
| **UMAP $min\_dist$** | Stetig | $[0.0, 0.25]$ | Steuergröße |
| **HDBSCAN $min\_cluster\_size$** | Stetig | $[10, 40]$ | Steuergröße (bisher vernachlässigt!) |
| **HDBSCAN $min\_samples$** | Stetig | $[5, 25]$ | Steuergröße (Dämpft Rausch-Sensitivität) |
| **Seed / GPU-Jitter** | Diskret | $[42, 1337, 2026, 7, 99]$ (5 Replikationen) | **Störgröße (Noise Factor)** |

---

### 2. Versuchsplan (Experimentelles Design)

Statt eines unvollständigen Grids wählt man ein **Space-Filling Design** (z. B. **Latin Hypercube Sampling (LHS)** oder ein **Sobol-Sequenz-Design**) oder ein klassisches **Box-Behnken / Central Composite Design (CCD)**.

* **Screening / Exploration:** 40–50 wohldefinierte Design-Punkte via Latin Hypercube Sampling über den 5-dimensionalen kontinuierlichen Parameterraum.
* **Replikation über Störgröße:** Jeder Design-Punkt wird über $R = 3$ bis $5$ verschiedene Seeds gerechnet.

---

### 3. Statistische Metriken zur Robustheit

Für jeden Design-Punkt $i$ berechnen wir nicht nur den Mittelwert des Scores, sondern auch dessen Stabilität:

1. **Mean Adjusted Silhouette:**
   $$\bar{S}_i = \frac{1}{R} \sum_{r=1}^R \text{Score}_{i, r}$$
2. **Signal-to-Noise Ratio ($S/N$, Taguchi-Kriterium „Larger-is-Better“):**
   $$\eta_i = -10 \cdot \log_{10} \left( \frac{1}{R} \sum_{r=1}^R \frac{1}{\text{Score}_{i, r}^2} \right)$$
   *Effekt:* Ein Setup mit Score 0,130 $\pm 0,001$ gewinnt gegen ein Setup mit Score 0,132 $\pm 0,015$.
3. **Cluster-Stabilität (Pairwise ARI):**
   Adjusted Rand Index (ARI) zwischen den Cluster-Zuordnungen der verschiedenen Seeds. Ein echtes Thema muss über verschiedene Runs hinweg stabil zusammenbleiben.

---

### 4. Exemplarische Python-Implementierung (DoE-Orchestrierung)

Hier ist ein modularer Entwurf, wie das DoE-Skript unter Nutzung moderner DoE-/Sampling-Bibliotheken (`scipy.stats.qmc` für Space-Filling, `statsmodels` für ANOVA/RSM) aufgebaut wird:

```python
"""doe_cluster_search.py: Robust Design of Experiments for UMAP + HDBSCAN."""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. Definieren der Parameter-Grenzen (Bounds)
PARAM_BOUNDS = {
    "d": (4, 20),                # int
    "n_neighbors": (15, 60),      # int
    "min_dist": (0.0, 0.25),      # float
    "min_cluster_size": (10, 40), # int
    "min_samples": (5, 25),       # int
}
SEEDS = [42, 1337, 2026]         # Störfaktoren für Stabilitätsmessung

def generate_lhs_design(n_samples=40, seed=42):
    """Erzeugt einen Latin Hypercube Versuchsplan."""
    sampler = qmc.LatinHypercube(d=len(PARAM_BOUNDS), seed=seed)
    sample = sampler.random(n=n_samples)
    
    design = []
    for row in sample:
        cfg = {
            "d": int(np.round(qmc.scale(row[0], PARAM_BOUNDS["d"][0], PARAM_BOUNDS["d"][1]))),
            "n_neighbors": int(np.round(qmc.scale(row[1], PARAM_BOUNDS["n_neighbors"][0], PARAM_BOUNDS["n_neighbors"][1]))),
            "min_dist": float(qmc.scale(row[2], PARAM_BOUNDS["min_dist"][0], PARAM_BOUNDS["min_dist"][1])),
            "min_cluster_size": int(np.round(qmc.scale(row[3], PARAM_BOUNDS["min_cluster_size"][0], PARAM_BOUNDS["min_cluster_size"][1]))),
            "min_samples": int(np.round(qmc.scale(row[4], PARAM_BOUNDS["min_samples"][0], PARAM_BOUNDS["min_samples"][1]))),
        }
        design.append(cfg)
    return design

def run_experiment(design, X_norm_gpu, run_fn):
    """Führt jeden DoE-Punkt mehrfach über Störgrößen (Seeds) aus."""
    records = []
    for run_id, cfg in enumerate(design):
        scores = []
        cluster_counts = []
        noise_ratios = []
        labels_per_seed = []
        
        for seed in SEEDS:
            # run_fn kapselt cuML UMAP + HDBSCAN + high-dim Scoring
            res, labels = run_fn(cfg, seed=seed)
            if res["accepted"]:
                scores.append(res["adjusted"])
                cluster_counts.append(res["n_clusters"])
                noise_ratios.append(res["noise_ratio"])
                labels_per_seed.append(labels)
        
        if len(scores) >= 2:
            scores = np.array(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            # Taguchi S/N Ratio für 'the larger the better'
            sn_ratio = -10.0 * np.log10(np.mean(1.0 / (scores**2 + 1e-9)))
            
            records.append({
                **cfg,
                "mean_adj_score": mean_score,
                "std_adj_score": std_score,
                "sn_ratio": sn_ratio,
                "mean_clusters": np.mean(cluster_counts),
                "mean_noise": np.mean(noise_ratios),
            })
            
    return pd.DataFrame(records)

def fit_response_surface(df):
    """Fit eines Response-Surface-Modells (quadratisch + Interaktionen)."""
    formula = (
        "mean_adj_score ~ (d + n_neighbors + min_dist + min_cluster_size + min_samples)**2 "
        "+ I(d**2) + I(n_neighbors**2)"
    )
    model = ols(formula, data=df).fit()
    print(model.summary())
    return model
```

---

### Was Sie durch dieses DoE-Setup gewinnen:

1. **Echte Signifikanz statt Spekulation:**
   Die ANOVA-Tabelle zeigt sofort, ob $d$ zwischen 8 und 16 überhaupt einen statistisch signifikanten Effekt ($p < 0.05$) hat oder ob z. B. `min_cluster_size` 80 % der Varianz erklärt.
2. **Entdeckung optimaler Sweet-Spots:**
   Über das Response-Surface-Modell ($\beta$-Koeffizienten) lässt sich das mathematische Maximum der Fläche analytisch bestimmen, anstatt blind auf diskrete Gitterpunkte angewiesen zu sein.
3. **Verlässliche Produktionsparameter:**
   Durch die Maximierung des Taguchi-Verhältnisses ($S/N$) oder die Einbeziehung der paarweisen Stabilität (ARI über Seeds) wählen Sie Parameter, die nicht zufällig bei einem bestimmten Seed gut abgeschnitten haben, sondern die robust gegen stochastische Einflüsse und zukünftige Korpus-Erweiterungen sind.
