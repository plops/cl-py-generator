# Cluster search over summary embeddings — walkthrough

Status: 2026-09-24, implemented and swept on two databases (live DB + compact
export). This document describes what was actually built, how the experiment was
run, and what it found. For the run instructions see `README.md`, for the
up-front design see `plan.md` / `task.md` in this directory.

## Summary

We searched the UMAP + clustering hyperparameter space for ~14–19k text
embeddings (YouTube summary corpus with Hacker News and other sources mixed in)
on an RTX A4000 GPU, using cuML throughout. Clustering happens in the
UMAP-reduced space, but quality is scored in the original embedding space with
`Silhouette_orig × (1 − noise)`, so scores are comparable across dimensions.
120 configurations were scored per database (3 Matryoshka widths × 20 UMAP
grids × DBSCAN/HDBSCAN). Winner on both databases: full 3072-d embeddings,
UMAP d=12 / nn=30, HDBSCAN — 137 clusters at adj. 0.1240 (live DB), 167
clusters at adj. 0.1327 (compact DB). Full width beats the 768-d prefix by ~5 %
and the 128-d pilot by ~9–14 %.

## 1. Introduction

The `rs-summarizer` project collects summaries of YouTube videos (plus Hacker
News discussions and assorted web sources) and stores a
`gemini-embedding-001` vector alongside each summary. These embeddings use
Matryoshka semantics: shorter vectors are an ordered prefix of the longer ones,
so a 3072-d vector can be truncated to 768 or 128 dims and stay meaningful.
The question this experiment answers: **which UMAP parameters produce the best
topic clusters over this corpus, and how much do the full 3072 dimensions
matter?**

There was already a clustering pipeline — the Rust `viz-tool` — but it commits
exactly the fallacy this experiment was designed around: it clusters in a 2D
(or 4D) projection with a naive O(n²) DBSCAN and implicitly treats that
low-dimensional picture as ground truth. In 2D, points are artificially packed
together and distance distributions collapse, so any quality metric computed
there is inflated and, worse, incomparable across dimensions. This experiment
replaces that methodology while reusing `viz-tool`'s conventions that proved
sound: the little-endian float32 BLOB decoding, Matryoshka truncation, and the
`-1 = noise` label convention.

The deliverables are the winning parameters (`best_params*.json`), the
cluster assignment per document (`plots/labels*.csv`), 2D plots of the best
clustering, and an interactive offline cluster map (`plots/clusters.html`)
with German titles for all 167 compact-DB clusters.

## 2. Scope of the experiment

**Goals:**

- Find the UMAP configuration (target dim, neighbors, min-dist) whose clusters
  are most cohesive in the *original* embedding space.
- Compare density clusterers (DBSCAN with dimension-scaled eps vs. HDBSCAN).
- Quantify the benefit of full-width (3072-d) vs. truncated (768/128-d)
  Matryoshka embeddings with an identical grid and criterion.
- Produce 2D diagrams of the best clustering plus reusable labels.

**Data:**

- Live DB `/workspace/src/rs-summarizer/summaries.db` (read-only): 15,262 rows,
  14,146 with embedding BLOBs — 12,569 × 3072 floats + 1,577 × 768 floats.
- Compact export `summaries_compact_20260924.db` (in this directory): 19,293
  rows in export schema; 18,269 usable at k=128 (1,024 NULL), 16,692 at
  k=3072. Includes 1,399 Hacker News items (§4.4). All drivers accept
  `--db`; live-DB results were archived to `archive_livedb_20260924/` after
  the compact re-sweep.

**Hardware:** RTX A4000 (16 GB), CUDA-UMD 13.4, Ubuntu 26 container; cuML/CuPy
GPU pipeline, no host round-trips inside the sweep loop.

**Non-goals:** no new embedding model, no supervised evaluation (there are no
labels — hence unsupervised validation), no changes to `rs-summarizer` itself
(read-only reference).

## 3. Methods

### 3.1 Loading (`loader.py`)

SQLite is opened read-only; BLOBs are decoded as little-endian float32 and
truncated to the requested Matryoshka width k (rows shorter than k are
skipped, as are NULL and zero-norm embeddings). Output is `X` (float32) plus
an L2-normalized copy `X_norm`, so Euclidean distance on `X_norm` equals
cosine distance. The loader auto-detects both schemas (live DB filters on
`summary_done = 1`, the export schema has no such column).

### 3.2 UMAP sweep (`sweep_umap.py`)

cuML UMAP grid, fixed across widths: `n_components` ∈ {2, 4, 8, 12, 16} ×
`n_neighbors` ∈ {15, 30} × `min_dist` ∈ {0.0, 0.1}, `metric='cosine'`,
`build_algo='nn_descent'`, `random_state=42`. 20 projections per width,
each cached as `.npy` (`umap_cache*/`) so clustering and re-scoring never
recompute UMAP. Typical cost is ~1–2 s per config on the A4000.

### 3.3 Clustering (`cluster_search.py`)

Every projection is clustered twice: DBSCAN with dimension-scaled
`eps(d) = 0.3 + d·0.02`, `min_samples=15`, and HDBSCAN with
`min_cluster_size=15`. Recorded per config: label vector, cluster count,
noise ratio. 20 projections × 2 methods = 40 scored configs per width,
120 per database.

### 3.4 Validation (`validate.py`) — the core criterion

Labels found in d-D are scored in the original k-D space:

`Score = Silhouette(X_norm[valid], labels[valid]) × (1 − noise_ratio)`

with noise (`-1`) filtered before scoring. Configs with > 40 % noise or a
single cluster are rejected outright. Because every config is scored against
the same high-dimensional ground truth, scores are comparable across target
dimensions — the cardinal rule from the task prompt. d=2 is swept along only
to confirm it fails; it is reserved for final plots.

### 3.5 Drivers, ablation, plots, tests

`run_pilot.py` runs the full grid per width → `results_*.csv`;
`ablation_width.py` condenses the per-width winners into `ablation.md`;
`plot_final.py` re-fits the winner at d=2 and renders
`plots/best_clusters.png` (1500×1200, noise in grey) plus `labels.csv`;
`build_cluster_data.py` / `build_html.py` produce the canonical labels,
`cluster_data.json`, and the interactive `plots/clusters.html`.
`tests/` holds 14 tests (BLOB round-trip, truncation, L2 norm, zero-filter,
score formula incl. noise penalty, eps monotonicity, synthetic-blob
end-to-end) — green on both databases.

## 4. Results

### 4.1 Winning configurations

| DB | width k | winner | clusters | noise | sil_orig | adjusted |
|---|---|---|---|---|---|---|
| live | 3072 | d12 nn30 md0.0, HDBSCAN | 137 | 27.8 % | 0.1716 | **0.1240** |
| compact | 3072 | d12 nn30 md0.1, HDBSCAN | 167 | 36.5 % | 0.2091 | **0.1327** |

The winner is the same shape on both databases: full width, d=12, nn=30,
HDBSCAN. Response over d is flat — d ∈ [4, 16] sit within ±0.002 adjusted —
while d=2 fails as predicted (DBSCAN adj. ≈ 0.03–0.08): 2D is for plots only.
HDBSCAN beats DBSCAN everywhere (adj. ~0.12 vs ~0.04); the DBSCAN eps base
would need recalibration to stay competitive.

### 4.2 Ablation: embedding width

Same grid, same criterion, best per width (compact DB; live DB shows the same
ranking, see `ablation.md`):

| width k | rows | best config | clusters | noise | adjusted |
|---|---|---|---|---|---|
| 128 | 18,269 | d16 nn15 md0.1 | 179 | 35.4 % | 0.1220 |
| 768 | 18,269 | d08 nn15 md0.0 | 226 | 30.8 % | 0.1269 |
| 3072 | 16,692 | d12 nn30 md0.1 | 167 | 36.5 % | **0.1327** |

Verdict: **3072 > 768 > 128** (+4.6 % / +8.8 % adjusted on compact;
+5.2 % / +13.5 % on live). Cost of full width: ~11 % short rows drop out and
nothing else — UMAP is ~1 s/config either way. The 128-d pilot correctly
ranked methods (HDBSCAN ≫ DBSCAN) but understated absolute quality: pilot
for speed, decide on full width.

### 4.3 Learnings and caveats

- cuML warns that `build_algo='nn_descent'` is **not deterministic** despite
  `random_state` — confirmed: a refit yields 131 instead of 137 clusters
  (live), 178 instead of 167 at 38.3 % noise (compact). Pin seeds, check ARI
  stability across seeds, and use `brute_force_knn` for bit-exact runs.
- The d=2 failure, the HDBSCAN margin, and the flat d-response replicate on
  both databases, so the ranking is robust even if individual cluster counts
  jitter under refit.
- Open from the prompt's strategies 2–3: trustworthiness-vs-d elbow and
  TwoNN/PCA intrinsic-dimension plots are still missing — the config ranking
  stands, the manifold-fidelity plots do not exist yet.

### 4.4 Hacker News articles: overlap, not separate groups

The compact DB holds 1,399 HN-linked items (vs ~16.2k YouTube), 1,237 of
which entered the k=3072 clustering. They do **not** form their own groups:
HN points spread over 79 of 167 clusters with no pure-HN cluster and only
three HN-majority ones — cl-116 "new language models & benchmarks" (118/147,
80 %), cl-111 "math research with AI proofs" (27/31, 87 %), both still
containing YouTube videos, plus cl-11, a junk cluster of empty/error entries.
Everywhere else HN is a topical minority (chip industry, LLM security,
Linux/containers, Lisp, CCC talks …): the embeddings cluster by content, not
by platform. One anomaly: 52.8 % of HN points fall into HDBSCAN noise vs
34.9 % for YouTube — HN texts scatter more widely in embedding space.

## 5. Conclusion

The experiment answered its questions cleanly. There is a stable, reproducible
recipe for this corpus — full-width Matryoshka embeddings, UMAP d≈12 / nn=30 /
cosine, HDBSCAN — that wins on two independent database snapshots, and the
high-dimensional validation criterion made the comparison across dimensions
principled instead of an artifact of 2D packing. The full 3072 dimensions are
worth ~5 % over the 768 prefix at negligible compute cost, and the pilot-then-
full-width workflow (rank cheaply at k=128, decide at k=3072) proved sound.
The main methodological caveat is UMAP's nondeterminism, which moves cluster
counts by ~5 % between refits without changing the ranking; any follow-up
that needs exact reproducibility should switch the neighbor search to
`brute_force_knn`. Remaining work: the trustworthiness elbow and TwoNN/PCA
intrinsic-dimension analysis, an ARI seed-stability study, and — if DBSCAN is
kept at all — a recalibrated eps scale. The labeled corpus
(`plots/labels_canonical.csv`, `cluster_titles.json`, `plots/clusters.html`)
is ready for downstream use, e.g. per-topic browsing at
https://rocketrecap.com/exports/clusters.html.

## Appendix A — Reproduction

```
source .venv/bin/activate
pytest tests/                                   # CPU unit tests
python run_pilot.py --width 128                 # pilot sweep -> results_pilot.csv
python ablation_width.py                        # k=768 vs k=3072 -> results_width*.csv
python plot_final.py --width 768 --d 8 --nn 15 --md 0.1 --method hdbscan
```

All drivers accept `--db <path>` (default: live DB). Cluster titles +
interactive map: `python build_cluster_data.py` (needs GPU), then the titler
batches (see `cluster_data.json`), then `python build_html.py`.

## Appendix B — Environment (Docker image)

`cuml-cu12==26.08.00`, `cupy-cuda12x==14.2.0` (via `https://pypi.nvidia.com`),
`numpy==2.4.6 pandas==3.0.3 matplotlib==3.11.2 scikit-learn==1.9.1 pytest==9.1.1`
(see `ENV.md`, `requirements.txt`).
