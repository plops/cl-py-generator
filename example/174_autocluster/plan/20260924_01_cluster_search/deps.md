# deps.md — Dependency-Pfade (org/projekt) für DeepWiki-Abfragen

Format: `https://deepwiki.com/<org>/<projekt>`. Neueste Version nehmen (siehe plan.md §3).

| Dependency | GitHub | DeepWiki-Frage |
|---|---|---|
| cuML (UMAP, DBSCAN, HDBSCAN, Metriken) | `rapidsai/cuml` | `cuml.manifold.UMAP`-Parameter, `cuml.cluster` fit_predict, `silhouette_score`, `trustworthiness` |
| CuPy (GPU-Arrays) | `cupy/cupy` | L2-Norm / Masken-Indexing auf GPU, Dtype-Handling float32 |
| UMAP-Referenz (Parameterwirkung, CPU-Fallback `umap-learn`) | `lmcinnes/umap` | n_neighbors/min_dist/metric-Empfehlungen für Text-Embeddings |
| scikit-learn (Metrik-Fallback, ARI für Seed-Stabilität) | `scikit-learn/scikit-learn` | `adjusted_rand_score`, `silhouette_score`-Semantik |
| matplotlib (Final-Plots) | `matplotlib/matplotlib` | Scatter mit Cluster-Farben, Elbow-Plots |
| rs-summarizer (Datenquelle, read-only) | `plops/rs-summarizer` | Export-DB-Schema, Embedding-BLOB-Format |
