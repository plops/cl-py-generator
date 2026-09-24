# walkthrough.md — Cluster-Suche: was wirklich implementiert wurde

## Ergebnis

- Beste Config (`best_params.json`): **k=3072, UMAP d=12 / nn=30 / md=0.0 (cosine,
  nn_descent), HDBSCAN mcs=15** → 137 Cluster, 27.8 % Noise, adj. Score 0.1240.
- Output: `plots/best_clusters.png` (1500×1200, validiert), `plots/labels.csv`
  (12.569 identifier→cluster), `results_*.csv` (120 scored Configs), `ablation.md`.
- 3072-D-Benefit quantifiziert: +5.2 % vs 768, +13.5 % vs 128 (adj. Score).

## Was gebaut wurde (reines Python, kein Transpiler)

`loader.py` (read-only SQLite, LE-f32, Matryoshka-Prefix, Zero-Filter),
`sweep_umap.py` (cuML-Grid + `.npy`-Cache), `cluster_search.py` (DBSCAN-eps-Skala,
HDBSCAN), `validate.py` (Orig-Raum-Scoring, 40-%-Noise-Veto), `run_pilot.py`,
`ablation_width.py`, `plot_final.py`; Tests in `tests/` (14 grün, beide DBs).

## Test-bedingte Änderungen / Learnings

- `viz-tool` (Rust, CPU) clustert in 2D/4D mit O(n²)-DBSCAN — genau der
  Anti-Pattern aus dem Prompt; BLOB-/Label-Konventionen (`-1` = Noise) übernommen.
- cuML warnt: `nn_descent` ist **nicht deterministisch** trotz `random_state`
  (Refit: 131 statt 137 Cluster). Seeds fixieren + ARI-Stabilität prüfen; für
  bitgenaue Runs `brute_force_knn`.
- DBSCAN (eps-Skala 0.3+d·0.02) durchweg weit hinter HDBSCAN (adj ~0.04 vs ~0.12);
  eps-Basis gehört neu kalibriert, falls DBSCAN bleiben soll.
- Score ist flach über d ∈ [4,16]; d=2 versagt wie vorhergesagt (nur Plots).
- Live-DB: 14.146 Embeddings (12.569×3072 + 1.577×768). Update: die kompakte
  Export-DB (`summaries_compact_20260924.db`, 19.293 Rows, Export-Schema) liegt
  jetzt im Projektdir — 18.269 nutzbare Embeddings bei k=128 (1.024 NULL),
  16.692 bei k=3072. Loader + alle Treiber per `--db` umschaltbar, 14 Tests grün
  auf beiden DBs, GPU-Smoke auf Compact-DB ok. Sweeps/Plots bleiben auf
  Live-DB-Zahlen. Update 2: Compact-DB-Re-Sweep durch (120/120 Configs,
  Live-Ergebnisse in `archive_livedb_20260924/`): Sieger k=3072, d12, nn30,
  md0.1, HDBSCAN (167 Cluster, 36,5 % Noise, adj 0,1327); Plots neu gerendert.
- Trustworthiness-Elbow + TwoNN/PCA (Prompt-Strategien 2–3) sind offen:
  Config-Ranking steht, Manifold-Fidelity-Plots fehlen noch.

## Docker-Pakete für das Image

`cuml-cu12==26.08.00`, `cupy-cuda12x==14.2.0` (via `https://pypi.nvidia.com`),
`numpy==2.4.6 pandas==3.0.3 matplotlib==3.11.2 scikit-learn==1.9.1 pytest==9.1.1`
(siehe `ENV.md`, `requirements.txt`).
