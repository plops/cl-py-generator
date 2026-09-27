# Cluster-Suche über Summary-Embeddings — Walkthrough

Stand: 2026-09-24, implementiert und auf zwei Datenbanken gesweept (Live-DB +
kompakter Export). Dieses Dokument beschreibt, was wirklich gebaut wurde, wie
das Experiment ablief und was es ergeben hat. Run-Anleitung siehe `README.md`,
vorab-Design siehe `plan.md` / `task.md` in diesem Verzeichnis.

## Kurzfassung

Wir haben den UMAP+Clustering-Hyperparameterraum für ~14–19k Text-Embeddings
(YouTube-Summary-Korpus, gemischt mit Hacker News und anderen Quellen) auf
einer RTX A4000 per cuML durchsucht. Geclustert wird im UMAP-reduzierten Raum,
bewertet aber im Originalraum mit `Silhouette_orig × (1 − Noise)` — dadurch
sind Scores über Dimensionen vergleichbar. Pro Datenbank wurden 120 Configs
gescort (3 Matryoshka-Breiten × 20 UMAP-Grids × DBSCAN/HDBSCAN). Sieger auf
beiden DBs: volle 3072-d-Embeddings, UMAP d=12 / nn=30, HDBSCAN — 137 Cluster
bei adj. 0,1240 (Live-DB), 167 Cluster bei adj. 0,1327 (Compact-DB). Volle
Breite schlägt den 768-d-Prefix um ~5 % und den 128-d-Pilot um ~9–14 %.

## 1. Einleitung

Das Projekt `rs-summarizer` sammelt Zusammenfassungen von YouTube-Videos
(dazu Hacker-News-Diskussionen und diverse Webquellen) und speichert pro
Zusammenfassung einen `gemini-embedding-001`-Vektor. Die Embeddings folgen
Matryoshka-Semantik: kürzere Vektoren sind ein geordnetes Prefix der längeren,
ein 3072-d-Vektor lässt sich also auf 768 oder 128 Dims trunkieren und bleibt
sinnvoll. Die Frage des Experiments: **Welche UMAP-Parameter erzeugen die
besten Themencluster über diesem Korpus, und wie viel bringen die vollen
3072 Dimensionen?**

Es gab bereits eine Clustering-Pipeline — das Rust-`viz-tool` — doch sie
begeht genau den Fehlschluss, um den dieses Experiment gebaut ist: Sie
clustert in einer 2D- (oder 4D-)Projektion mit naivem O(n²)-DBSCAN und
behandelt dieses niedrigdimensionale Bild implizit als Ground Truth. In 2D
werden Punkte künstlich zusammengepackt und Distanzverteilungen kollabieren —
jede dort berechnete Qualitätsmetrik ist überhöht und obendrein über
Dimensionen unvergleichbar. Dieses Experiment ersetzt die Methodik, übernimmt
aber die bewährten Konventionen aus `viz-tool`: Little-Endian-float32-
BLOB-Dekodierung, Matryoshka-Trunkierung und die Label-Konvention
(`-1` = Noise).

Liefergegenstände sind die Siegerparameter (`best_params*.json`), die
Clusterzuordnung pro Dokument (`plots/labels*.csv`), 2D-Plots des besten
Clusterings sowie eine interaktive Offline-Clusterkarte
(`plots/clusters.html`) mit deutschen Titeln für alle 167 Compact-DB-Cluster.

## 2. Scope des Experiments

**Ziele:**

- Die UMAP-Config (Zieldim, Nachbarn, min-dist) finden, deren Cluster im
  *Original*-Embeddingraum am kohäsivsten sind.
- Dichteclusterer vergleichen (DBSCAN mit dimensionsskaliertem eps vs.
  HDBSCAN).
- Den Nutzen voller (3072-d) ggü. trunkierter (768/128-d) Matryoshka-
  Embeddings bei identischem Grid und Kriterium quantifizieren.
- 2D-Diagramme des besten Clusterings plus wiederverwendbare Labels liefern.

**Daten:**

- Live-DB `/workspace/src/rs-summarizer/summaries.db` (read-only): 15.262
  Rows, 14.146 mit Embedding-BLOB — 12.569 × 3072 Floats + 1.577 × 768 Floats.
- Kompakt-Export `summaries_compact_20260924.db` (in diesem Verzeichnis):
  19.293 Rows im Export-Schema; 18.269 nutzbar bei k=128 (1.024 NULL), 16.692
  bei k=3072. Enthält 1.399 Hacker-News-Einträge (§4.4). Alle Treiber
  akzeptieren `--db`; Live-DB-Ergebnisse liegen nach dem Compact-Re-Sweep in
  `archive_livedb_20260924/`.

**Hardware:** RTX A4000 (16 GB), CUDA-UMD 13.4, Ubuntu-26-Container;
cuML/CuPy-GPU-Pipeline ohne Host-Transfers in der Sweep-Schleife.

**Nicht-Ziele:** kein neues Embedding-Modell, keine supervidierte Evaluation
(es gibt keine Labels — daher unüberwachte Validierung), keine Änderungen an
`rs-summarizer` selbst (read-only-Referenz).

## 3. Methoden

### 3.1 Laden (`loader.py`)

SQLite wird read-only geöffnet; BLOBs werden als Little-Endian-float32
dekodiert und auf die gewünschte Matryoshka-Breite k trunkiert (kürzere Rows
werden übersprungen, ebenso NULL- und Zero-Norm-Embeddings). Ausgabe ist `X`
(float32) plus eine L2-normalisierte Kopie `X_norm` — euklidische Distanz auf
`X_norm` entspricht damit Kosinus-Distanz. Der Loader erkennt beide Schemas
automatisch (Live-DB filtert auf `summary_done = 1`, das Export-Schema kennt
die Spalte nicht).

### 3.2 UMAP-Sweep (`sweep_umap.py`)

cuML-UMAP-Grid, fix über alle Breiten: `n_components` ∈ {2, 4, 8, 12, 16} ×
`n_neighbors` ∈ {15, 30} × `min_dist` ∈ {0.0, 0.1}, `metric='cosine'`,
`build_algo='nn_descent'`, `random_state=42`. 20 Projektionen pro Breite, jede
als `.npy` gecacht (`umap_cache*/`), sodass Clustering und Re-Scoring UMAP
nie neu rechnen. Kosten: ~1–2 s pro Config auf der A4000.

### 3.3 Clustering (`cluster_search.py`)

Jede Projektion wird zweimal geclustert: DBSCAN mit dimensionsskaliertem
`eps(d) = 0,3 + d·0,02`, `min_samples=15`, und HDBSCAN mit
`min_cluster_size=15`. Pro Config werden Labelvektor, Clusterzahl und
Noise-Ratio festgehalten. 20 Projektionen × 2 Verfahren = 40 gescorte Configs
pro Breite, 120 pro Datenbank.

### 3.4 Validierung (`validate.py`) — das Kernkriterium

In d-D gefundene Labels werden im originalen k-D-Raum gescort:

`Score = Silhouette(X_norm[valid], Labels[valid]) × (1 − Noise-Ratio)`

Noise (`-1`) wird vor dem Scoring gefiltert. Configs mit > 40 % Noise oder
nur einem Cluster werden verworfen. Da jede Config gegen dieselbe
hochipdimensionale Ground Truth gescort wird, sind Scores über Zieldimensionen
vergleichbar — die Kardinalregel aus dem Task-Prompt. d=2 läuft nur mit, um
sein Versagen zu bestätigen; es ist den Final-Plots vorbehalten.

### 3.5 Treiber, Ablation, Plots, Tests

`run_pilot.py` fährt das volle Grid pro Breite → `results_*.csv`;
`ablation_width.py` verdichtet die Breiten-Sieger nach `ablation.md`;
`plot_final.py` refittet den Sieger auf d=2 und rendert
`plots/best_clusters.png` (1500×1200, Noise grau) plus `labels.csv`;
`build_cluster_data.py` / `build_html.py` erzeugen kanonische Labels,
`cluster_data.json` und das interaktive `plots/clusters.html`.
`tests/` enthält 14 Tests (BLOB-Roundtrip, Trunkierung, L2-Norm, Zero-Filter,
Score-Formel inkl. Noise-Strafe, eps-Monotonie, synthetisches Blob-
End-to-End) — grün auf beiden DBs.

## 4. Ergebnisse

### 4.1 Sieger-Configs

| DB | Breite k | Sieger | Cluster | Noise | sil_orig | adj. |
|---|---|---|---|---|---|---|
| live | 3072 | d12 nn30 md0.0, HDBSCAN | 137 | 27,8 % | 0,1716 | **0,1240** |
| compact | 3072 | d12 nn30 md0.1, HDBSCAN | 167 | 36,5 % | 0,2091 | **0,1327** |

Der Sieger hat auf beiden DBs dieselbe Form: volle Breite, d=12, nn=30,
HDBSCAN. Die Antwort über d ist flach — d ∈ [4, 16] liegt überall innerhalb
±0,002 adj. — während d=2 wie vorhergesagt versagt (DBSCAN adj. ≈ 0,03–0,08):
2D ist nur für Plots. HDBSCAN schlägt DBSCAN überall (adj. ~0,12 vs ~0,04);
die DBSCAN-eps-Basis müsste neu kalibriert werden, um konkurrenzfähig zu
bleiben.

### 4.2 Ablation: Embedding-Breite

Gleiches Grid, gleiches Kriterium, Bestwert pro Breite (Compact-DB; Live-DB
mit gleichem Ranking, siehe `ablation.md`):

| Breite k | Rows | beste Config | Cluster | Noise | adj. |
|---|---|---|---|---|---|
| 128 | 18.269 | d16 nn15 md0.1 | 179 | 35,4 % | 0,1220 |
| 768 | 18.269 | d08 nn15 md0.0 | 226 | 30,8 % | 0,1269 |
| 3072 | 16.692 | d12 nn30 md0.1 | 167 | 36,5 % | **0,1327** |

Verdikt: **3072 > 768 > 128** (+4,6 % / +8,8 % adj. auf Compact;
+5,2 % / +13,5 % auf Live). Kosten der vollen Breite: ~11 % kurze Rows fallen
raus, sonst nichts — UMAP kostet ~1 s/Config so oder so. Der 128-d-Pilot hat
die Verfahren korrekt gerankt (HDBSCAN ≫ DBSCAN), aber die absolute Qualität
untertrieben: Pilot für Tempo, entscheiden auf voller Breite.

### 4.3 Learnings und Caveats

- cuML warnt, `build_algo='nn_descent'` sei trotz `random_state` **nicht
  deterministisch** — bestätigt: Ein Refit liefert 131 statt 137 Cluster
  (live) bzw. 178 statt 167 bei 38,3 % Noise (compact). Seeds fixieren,
  ARI-Stabilität über Seeds prüfen, für bitgenaue Runs `brute_force_knn`.
- Das d=2-Versagen, der HDBSCAN-Abstand und die flache d-Antwort replizieren
  auf beiden DBs — das Ranking ist robust, auch wenn einzelne Clusterzahlen
  beim Refit jittern.
- Offen aus Prompt-Strategien 2–3: Trustworthiness-vs-d-Elbow und
  TwoNN/PCA-Plots zur intrinsischen Dimensionalität fehlen noch — das
  Config-Ranking steht, die Manifold-Fidelity-Plots existieren nicht.

### 4.4 Hacker-News-Artikel: Überlappung, keine eigenen Gruppen

Die Compact-DB enthält 1.399 HN-verlinkte Einträge (ggü. ~16,2k YouTube),
1.237 davon gingen ins k=3072-Clustering ein. Sie bilden **keine eigenen
Gruppen**: HN-Punkte streuen über 79 von 167 Clustern, ohne einen einzigen
reinen HN-Cluster, mit nur drei HN-majoritären — cl-116 „Neue Sprachmodelle
und Benchmark-Vergleiche“ (118/147, 80 %), cl-111 „Mathematikforschung mit
KI-Beweisen“ (27/31, 87 %), beide trotzdem mit YouTube-Videos, plus cl-11,
ein Junk-Cluster aus Leer-/Fehlereinträgen. Überall sonst ist HN thematische
Minderheit (Chipindustrie, LLM-Sicherheit, Linux/Container, Lisp,
CCC-Vorträge …): Die Embeddings clustern nach Inhalt, nicht nach Plattform.
Eine Auffälligkeit: 52,8 % der HN-Punkte landen in HDBSCAN-Noise ggü. 34,9 %
bei YouTube — HN-Texte streuen weiter im Embedding-Raum.

## 5. Conclusion

Das Experiment hat seine Fragen sauber beantwortet. Es gibt ein stabiles,
replizierbares Rezept für diesen Korpus — Matryoshka-Embeddings voller
Breite, UMAP d≈12 / nn=30 / Kosinus, HDBSCAN — das auf zwei unabhängigen
DB-Snapshots gewinnt, und das Validierungskriterium im Hochdimensionalen
macht den Dimensionsvergleich prinzipiell statt zu einem Artefakt der
2D-Packung. Die vollen 3072 Dimensionen bringen ~5 % ggü. dem 768-Prefix bei
vernachlässigbaren Rechenkosten, und der Pilot-dann-Vollbreite-Workflow
(billig bei k=128 ranken, bei k=3072 entscheiden) hat sich bewährt. Der
wichtigste methodische Caveat ist UMAPs Nichtdeterminismus, der Clusterzahlen
zwischen Refits um ~5 % verschiebt, ohne das Ranking zu ändern; jeder
Follow-up mit exakter Reproduzierbarkeit sollte die Nachbarschaftssuche auf
`brute_force_knn` umstellen. Restarbeit: Trustworthiness-Elbow und
TwoNN/PCA-Analyse der intrinsischen Dimensionalität, eine ARI-Seed-
Stabilitätsstudie und — falls DBSCAN überhaupt bleibt — eine rekalibrierte
eps-Skala. Der gelabelte Korpus (`plots/labels_canonical.csv`,
`cluster_titles.json`, `plots/clusters.html`) ist bereit für Downstream-Nutzung,
z. B. thematisches Browsen unter
https://rocketrecap.com/exports/clusters.html.

## Anhang A — Reproduktion

```
source .venv/bin/activate
pytest tests/                                   # CPU-Unit-Tests
python run_pilot.py --width 128                 # Pilot-Sweep -> results_pilot.csv
python ablation_width.py                        # k=768 vs k=3072 -> results_width*.csv
python plot_final.py --width 768 --d 8 --nn 15 --md 0.1 --method hdbscan
```

Alle Treiber akzeptieren `--db <pfad>` (Default: Live-DB). Cluster-Titel +
interaktive Karte: `python build_cluster_data.py` (braucht GPU), dann die
Titler-Batches (siehe `cluster_data.json`), dann `python build_html.py`.

## Anhang B — Umgebung (Docker-Image)

`cuml-cu12==26.08.00`, `cupy-cuda12x==14.2.0` (via `https://pypi.nvidia.com`),
`numpy==2.4.6 pandas==3.0.3 matplotlib==3.11.2 scikit-learn==1.9.1 pytest==9.1.1`
(siehe `ENV.md`, `requirements.txt`).
