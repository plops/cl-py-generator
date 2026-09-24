# Implementierungsplan: GPU-Cluster-Suche über YouTube-Summary-Embeddings (EN)

Ziel: schnellste cuML-UMAP-Parametersuche auf ~14k Embeddings (RTX A4000, 16 GB),
beste Cluster finden, als Output 2D-Diagramme der besten Cluster.
Code liegt in `cl-py-generator/example/174_autocluster`, reines Python (kein Lisp-Transpiler).

## 0. Verifizierte Fakten (eigene Messung, 2026-09-24)

- DB: `/workspace/src/rs-summarizer/summaries.db`, Tabelle `summaries`, read-only nutzen.
- 14.355 Rows mit `summary_done=1`, davon 14.146 mit Embedding-BLOB.
- Dim-Histogramm (Floats): **3072 → 12.569 Rows, 768 → 1.577 Rows**. Kein 128er-Subset in der DB.
- BLOB-Format: little-endian float32, siehe `bytes_to_embedding`
  ([embedding.rs](/workspace/src/rs-summarizer/src/services/embedding.rs:104)).
- Matryoshka-Semantik: kürzere Vektoren = Prefix der längeren; Mismatch per Trunkierung
  lösen (`cosine_similarity`, [embedding.rs](/workspace/src/rs-summarizer/src/services/embedding.rs:48)).
- Modell: `gemini-embedding-001` mit 3072 Output-Dims
  ([tasks.rs](/workspace/src/rs-summarizer/src/tasks.rs:598)); `embedding_model`-Werte gemischt
  (`'', 'gemini-embedding-001', 'models/gemini-embedding-001'`).
- Compact-DB `summaries_compact_20260924.db` (Export-Schema ohne `summary_done`,
  19.293 Rows) liegt im Projektdir. Loader erkennt beide Schemas (mit/ohne Filter).
  Alle Sweeps/Plots bisher auf Live-DB (14.146 Embeddings); per `--db` auf
  Compact-DB umschaltbar (18.269 nutzbar bei k=128, 16.692 bei k=3072).
- GPU: RTX A4000, 16.376 MiB, Treiber 610.57 / CUDA-UMD 13.4. `nvcc` fehlt, System-Python hat
  **kein** numpy/cupy/cuml. Env-Aufbau per `uv` + `.venv` im Projektdir (Skill `python-env`:
  Env gehört ins Projekt, nie `/tmp`, nie System-Interpreter).
- Beispiel-Konvention in `example/`: nummerierte Dirs; hier gilt explizit **reines Python**.

## 1. Methodik (aus dem Prompt, verbindlich)

1. **High-Dim-Validation (Hauptkriterium):** clustern im reduzierten Raum (d-D),
   **scoren im originalen 768/3072-D Raum** (L2-normalisiert ≙ Kosinus):
   `Score(d) = Silhouette_orig × (1 − NoiseRatio)`, Noise (`-1`) vorher filtern,
   Lösungen mit Noise > 40 % verwerfen.
2. **Trustworthiness-Elbow:** Trustworthiness vs. d plotten, kleinstes d mit ≥ 0,92 wählen.
3. **Intrinsische Dimensionalität:** TwoNN-Schätzer + PCA-variance-Baseline (60–80 % Varianz)
   als Suchraum-Anker (Erwartung: ID ≈ 8–22).
4. **Density-Trap:** d ≥ 25 wird DBSCAN-inkompatibel (Distanzkonzentration); Sweet Spot d ∈ [5, 12]
   fürs Clustern, **d = 2 nur für Final-Plots**.

## 2. Architektur

```mermaid
flowchart LR
    DB[(summaries.db\nread-only)] --> L[0 loader\nLE-f32 decode,\nMatryoshka-truncate,\nfilter, L2-norm]
    L --> W{width k\n128 pilot → 768 → 3072}
    W --> U[1 UMAP sweep\ncuML, cosine,\nnn_descent, seed fix]
    U --> C[2 cluster in d-D\nDBSCAN + HDBSCAN]
    C --> V[3 validate in ORIG space\nSilhouette×(1-noise),\ntrustworthiness, TwoNN/PCA]
    V --> A[4 ablation\n768 vs 3072 benefit]
    V --> P[5 final 2D plots\nbest config → PNG]
    V --> J[best_params.json\n+ labels CSV]
```

Module (reines Python, je Datei ein Schritt, CLI per `argparse`, kein Framework):

| Datei | Zweck |
|---|---|
| `loader.py` | SQLite read-only lesen, LE-f32 dekodieren, auf Breite k trunkieren (nur Rows mit len ≥ k), Null-/Zero-Norm-Rows filtern, `X` (float32) + `X_norm` (L2) + `ids` liefern |
| `sweep_umap.py` | cuML-UMAP-Grid über d × n_neighbors × min_dist, Timing + `X_reduced` cachen (`.npy`) |
| `cluster_search.py` | DBSCAN (eps pro d skaliert) + HDBSCAN auf `X_reduced`, Labels + Noise-Ratio + n_clusters |
| `validate.py` | `Silhouette_orig × (1−noise)` auf `X_norm`, Trustworthiness-vs-d, TwoNN + PCA-Baseline |
| `ablation_width.py` | gleiche Pipeline auf k=768 vs k=3072 (nur 3072-Rows), Delta quantifizieren |
| `plot_final.py` | bestes Setup → 2D-Refit (d=2), Scatter pro Cluster (Noise grau), zweiter Plot: Trustworthiness-Elbow |
| `tests/` | Unit + Integration (siehe §5) |

Suchräume (Start, Pilot beschneidet):

- Breiten k: 128 (Pilot, schnell) → 768 → 3072 (nur Rows mit voller Länge).
- `n_components` d: [2, 4, 8, 12, 16, 24] (+32 nur wenn Elbow noch steigt).
- `n_neighbors`: [15, 30, 50]; `min_dist`: [0.0, 0.1, 0.5]; `metric='cosine'`,
  `build_algo='nn_descent'`, `random_state=42` fix.
- DBSCAN: `min_samples` [10, 15, 30], eps-Skala `eps(d) = base + d·step` (base/step kalibrieren,
  Start 0.3/0.02); HDBSCAN: `min_cluster_size` [10, 15, 30, 50].
- Akzeptanz: Noise < 40 %, n_clusters > 1, sonst verwerfen.

## 3. cuML-Usage (DeepWiki verifiziert, neueste Version nehmen)

UMAP (`rapidsai/cuml`, `cuml.manifold.UMAP`):

```python
from cuml.manifold.umap import UMAP
model = UMAP(n_neighbors=30, min_dist=0.0, n_components=8,
             metric='cosine', build_algo='nn_descent', random_state=42)
X_red = model.fit_transform(X_gpu)  # X_gpu: CuPy-Array, kein Host-Transfer
```

DBSCAN / HDBSCAN (`rapidsai/cuml`):

```python
from cuml.cluster import DBSCAN, HDBSCAN
labels = DBSCAN(eps=0.46, min_samples=15).fit_predict(X_red)
labels = HDBSCAN(min_cluster_size=15).fit_predict(X_red)
```

Silhouette im Originalraum (Prompt-Snippet, L2-Norm ≙ Kosinus):

```python
import cupy as cp
from cuml.metrics.cluster import silhouette_score
n = cp.linalg.norm(X_gpu, axis=1, keepdims=True)
Xn = X_gpu / cp.maximum(n, 1e-12)
s = float(silhouette_score(Xn[mask], labels[mask]))
score = s * (1.0 - noise_ratio)
```

Env (CUDA-13-Container, A4000): `.venv` im Projektdir, z. B.
`uv pip install "cudf-cu12" "cuml-cu12" cupy-cuda12x --extra-index-url https://pypi.nvidia.com`
plus `numpy pandas matplotlib scikit-learn pytest umap-learn` (letzteres nur CPU-Fallback).
Exakte Pins + Wheel-Quellen dokumentiert der Implementierungsagent in `ENV.md`.

## 4. Offene Requirements-Fragen (User-Entscheid, nicht geraten)

1. Reproduzierbarkeit: `random_state=42` fix + Seed-Stabilität (ARI über Seeds) — ok?
2. Noise-Policy: > 40 % Noise verwerfen (Prompt) — harte Grenze oder ranken?
3. Mindest-Clusterzahl/-größe für "best" (Business-Regel fehlt).
4. Semantik-Spotcheck: Top-Videos pro Cluster (identifier/link) manuell prüfen — gewünscht?
5. Output-Schema: `best_params.json` + `labels.csv (identifier → cluster)` + PNGs — reicht das?
6. 128-D-Pilot ist Matryoshka-Prefix (geordnet, daher valide), aber Bias-Check gegen 768 Pflicht.
7. 768er-Rows bei k=3072: ausschließen (12.569 bleiben) oder separat suchen?
8. CPU-Fallback (`umap-learn`) bei GPU-OOM einplanen oder GPU-only?
9. Zeitbudget pro Sweep-Lauf (Abbruch-/Pruning-Regel)?

## 5. Tests & Commit-Konvention (Kurzfassung, Details in task.md)

- Unit: LE-f32-Dekodierung, Trunkierung, L2-Norm, Zero-Filter, Score-Formel inkl.
  Noise-Strafe, DBSCAN-Eps-Skala-Monotonie.
- Integration: synthetische Blobs end-to-end (CPU, schnell); GPU-Smoke mit 500 Rows;
  Pilot-Seed-Stabilität (ARI ≥ Schwellwert über 2 Seeds).
- Commits: Conventional Commits (`feat(cluster): …`, `test(loader): …`, …), ein logischer
  Change pro Commit, Body mit Was/Warum/Validierung; nach jedem grünen Task committen.

## 6. Datei-Leseliste für den Implementierungsagenten

| Datei | Warum lesen |
|---|---|
| `plan/20260924_01_cluster_search/prompt.txt` | Originalauftrag + Methodik-Snippets |
| `plan/20260924_01_cluster_search/plan.md` (dies) | Architektur, Suchräume, offene Fragen |
| `plan/20260924_01_cluster_search/task.md` | Serielle Tasks mit Validierung |
| `plan/20260924_01_cluster_search/deps.md` | Dependency-Pfade für DeepWiki-Abfragen |
| `/workspace/src/rs-summarizer/src/services/embedding.rs` | BLOB-Format, Matryoshka-Trunkierung, Kosinus (read-only) |
| `/workspace/src/rs-summarizer/src/commands/export_db.rs` | Export-Schema der Embeddings (read-only) |
| `/workspace/src/rs-summarizer/src/models.rs` | `Summary`-Felder inkl. `embedding/full_embedding` (read-only) |
| `/workspace/src/rs-summarizer/src/db.rs` (`fetch_all_embeddings`) | Referenz-Loader-Logik (read-only) |
| DeepWiki `rapidsai/cuml` | UMAP/DBSCAN/HDBSCAN/silhouette/trustworthiness-API, neueste Version |
| DeepWiki `lmcinnes/umap` | Parameterwirkung (n_neighbors/min_dist/metric bei Text-Embeddings) |

`rs-summarizer`-Quellen sind **read-only** (keine Edits dort).

## 7. Bestand: `viz-tool` (unbedingt wiederverwenden/kennen)

DeepWiki (`plops/rs-summarizer`, verifiziert) + Code-Lesung: `export-db` kopiert nur
`summary_done=1`-Rows (ohne `transcript`) in eine kompakte DB mit
`identifier, original_source_link, model, rs_summarizer_version, embedding (BLOB, LE-f32),
embedding_model, summary, timestamps, cost, timestamped_summary_in_youtube_format`.
`viz-tool/src/` enthält bereits eine CPU-Pipeline: `data_loader.rs` (`load_compact_db`,
trunkiert Matryoshka auf Ziel-Dim, überspringt invalide BLOBs mit Warnung),
`embedding.rs` (`bytes_to_embedding_truncated`), `umap_engine.rs` (`UmapParams`-Defaults:
n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200; GPU-parametric- oder
fast-umap-CPU-Backend), `dbscan_engine.rs` (naives O(n²)-DBSCAN auf **4D**-Punkten,
`-1` = Noise), dazu `cluster_titler.rs` (Cluster-Benennung) und `nn_mapper.rs`.
Das aktuelle 2D/4D-Clustern ist genau der im Prompt kritisierte Anti-Pattern
(Informationsverlust, künstliche Silhouette-Inflation) — die cuML-Suche ersetzt es
methodisch, übernimmt aber BLOB-Dekodierung, Trunkierungs- und Label-Konventionen
(`-1` Noise) aus `viz-tool`. Leseliste ergänzen: `viz-tool/src/data_loader.rs`,
`viz-tool/src/embedding.rs`, `viz-tool/src/umap_engine.rs`,
`viz-tool/src/dbscan_engine.rs`, `viz-tool/src/cluster_titler.rs`.
