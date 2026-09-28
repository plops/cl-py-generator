# 174_autocluster — cuML UMAP cluster search over YouTube summary embeddings

Pure-Python pipeline (no Lisp transpiler). Method: cluster in UMAP-reduced space,
score in the original L2-normalized space
(`Silhouette_orig × (1 − noise)`); see `plan/20260924_01_cluster_search/plan.md`.

https://rocketrecap.com/exports/clusters.html

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

Cluster titles + interactive map: `python build_cluster_data.py` (needs GPU),
then 12 titler batches (see `cluster_data.json`), then
`python build_html.py` → `plots/clusters.html` (zoombar, offline),
`cluster_titles.json` (167 deutsche Titel mit Nachbar-Kontrast).


## Face Re-ID latent-space validation (26_onnx/source7)

Tooling to tune and validate the UMAP/HDBSCAN parameters for the Rust face
re-identification viewer (`cl-rust-generator/examples/26_onnx/source7`) on GPU
via cuML, and to prove the 512D ArcFace embeddings are computed correctly.

### Setup

```
uv venv && uv pip install -r requirements.txt
uv pip install onnxruntime opencv-python-headless   # for the LFW pipeline
```

### Ground-truth data

```
./download_ground_truth.sh          # caches LFW (Labeled Faces in the Wild)
```

### Export embeddings from the Rust DB

In the Rust project: `cargo run --release --bin export -- --db faces_db.bin --out export`
writes `export/emb.f32` (n×512 LE float32), `export/labels.i32` (person ids),
`export/meta.json`.

### Scripts

- `face_validate.py --export <DIR>` — self-consistency of the exported DB:
  L2-norm, NaN/dead-dim checks, and intra- vs inter-person cosine separation
  (Cohen's d). Confirms the embeddings themselves are sane.
- `face_dataset_validate.py --models <DIR>` — **end-to-end** validation on LFW
  through the *same* SCRFD + ArcFace ONNX models with Rust-matching preprocessing
  (5-point alignment, `(x-127.5)/127.5`, L2-norm). Reports Cohen's d, verification
  ROC-AUC, and UMAP+HDBSCAN ARI/AMI against ground-truth identities. Writes
  `plots_face/lfw_validate.png`.
- `face_sweep.py --export <DIR>` — cuML UMAP(2D)+HDBSCAN sweep scored by 2D
  trustworthiness, same-person kNN, and cluster ARI/AMI. Writes `results_face.csv`
  and `plots_face/best_2d.png`.
- `face_db_layout.py --export <DIR>` — renders all points vs. non-singleton
  persons to show the singleton effect on the layout.

### Findings (for the Rust viewer defaults)

Validated on LFW (ROC-AUC 0.9999, cluster ARI 0.991) and swept on the live DB:
small `n_neighbors` (~5–10) with `min_dist=0` makes the true same-person
micro-clusters visible; cluster in 8D with `min_cluster_size≈3`, `min_samples≈2`.
The bad-looking blob on the live DB is a data-composition effect (≈48% singletons),
not an embedding bug.
