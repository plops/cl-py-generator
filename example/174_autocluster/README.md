# 174_autocluster — cuML UMAP cluster search over YouTube summary embeddings

Pure-Python pipeline (no Lisp transpiler). Method: cluster in UMAP-reduced space,
score in the original L2-normalized space
(`Silhouette_orig × (1 − noise)`); see `plan/20260924_01_cluster_search/plan.md`.

## Run

```
source .venv/bin/activate
pytest tests/                                   # CPU unit tests
python run_pilot.py --width 128                 # pilot sweep -> results_pilot.csv
python ablation_width.py                        # k=768 vs k=3072 -> results_width*.csv
python plot_final.py --width 768 --d 8 --nn 15 --md 0.1 --method hdbscan
```

Alle Treiber akzeptieren `--db <pfad>` (Default: Live-DB
`/workspace/src/rs-summarizer/summaries.db`). Compact-DB im Projektdir
(`summaries_compact_20260924.db`, 19.293 Rows, 18.269 nutzbar bei k=128):
`python run_pilot.py --db summaries_compact_20260924.db --width 128`.

Outputs: `results_pilot.csv`, `results_width768.csv`, `results_width3072.csv`,
`plots/best_clusters.png`, `plots/labels.csv (identifier → cluster)`.
