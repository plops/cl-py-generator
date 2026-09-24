# 174_autocluster — cuML-UMAP-Cluster-Suche über YouTube-Summary-Embeddings

Deutsche Übersetzung von `README.md`.

Reine-Python-Pipeline (kein Lisp-Transpiler). Methode: Clustern im UMAP-reduzierten
Raum, Bewerten im originalen L2-normalisierten Raum
(`Silhouette_orig × (1 − Noise)`); siehe `plan/20260924_01_cluster_search/plan.md`.

## Ausführung

```
source .venv/bin/activate
pytest tests/                                   # CPU-Unit-Tests
python run_pilot.py --width 128                 # Pilot-Sweep -> results_pilot.csv
python ablation_width.py                        # k=768 vs k=3072 -> results_width*.csv
python plot_final.py --width 768 --d 8 --nn 15 --md 0.1 --method hdbscan
```

Alle Treiber akzeptieren `--db <Pfad>` (Default: Live-DB
`/workspace/src/rs-summarizer/summaries.db`). Compact-DB im Projektdir
(`summaries_compact_20260924.db`, 19.293 Rows, 18.269 nutzbar bei k=128):
`python run_pilot.py --db summaries_compact_20260924.db --width 128`.

Outputs: `results_pilot.csv`, `results_width768.csv`, `results_width3072.csv`,
`plots/best_clusters.png`, `plots/labels.csv (identifier → cluster)`.
