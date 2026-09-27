# deps.md — Abhängigkeiten der DoE-Untersuchung (2026-09-27)

GitHub-Pfade in `<organisation>/<projekt>`-Notation für DeepWiki-Abfragen.
Alle neu eingeführten Dependencies in neuester Version (Stand 2026-09-27,
verifiziert per `uv pip` im Projekt-`.venv`).

## Bereits vorhanden (aus `requirements.txt` / ENV.md)

- `rapidsai/cuml` — cuML UMAP, HDBSCAN, DBSCAN, Silhouette (GPU). Version `26.08.00`.
- `cupy/cupy` — CuPy GPU-Arrays, `.npy`-Cache. Version `14.2.0` (`cupy-cuda12x`).
- `numpy/numpy` — CPU-Arrays, Label-Statistik. Version `2.4.6`.
- `pandas/pandas` — Design-Tabellen, Ergebnis-CSVs. Version `3.0.3`.
- `matplotlib/matplotlib` — DoE-Plots (Main Effects, S/N, Stabilität). Version `3.11.2`.
- `scikit-learn/scikit-learn` — Adjusted Rand Index (Cluster-Stabilität über Seeds). Version `1.9.1`.
- `scipy/scipy` — `scipy.stats.qmc` Latin Hypercube / Sobol Sampling. Version `1.18.1` (bereits transitiv da, jetzt direkt genutzt).

## Neu eingeführt für DoE

- `statsmodels/statsmodels` — ANOVA (`anova_lm`) + Response-Surface-Regression (`ols`). Version `0.15.0`.
- `paulgb/formulaic` — Formel-Parser (Constraint von `statsmodels`, transitiv). Version `1.2.2`.
- `pydata/patsy` — Design-Matrizen für `statsmodels`-Formeln (transitiv). Version `1.0.3`.

## DeepWiki-Abfragen (für den Implementierungsagenten)

```text
deepwiki: rapidsai/cuml — "HDBSCAN min_samples vs min_cluster_size semantics? UMAP random_state reproducibility with nn_descent vs brute_force_knn?"
deepwiki: scipy/scipy — "scipy.stats.qmc.LatinHypercube usage example, scaling samples to bounds, seed reproducibility?"
deepwiki: statsmodels/statsmodels — "ols + anova_lm example with quadratic terms and interactions (response surface)?"
deepwiki: scikit-learn/scikit-learn — "adjusted_rand_score usage, handling of noise label -1 in clustering comparison?"
```
