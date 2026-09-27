# task.md — Serielle DoE-Tasks (DE)

Jeder Task wird **seriell** abgearbeitet und erst nach bestandener Validierung
committet. Modi: **IMPL** = Implementierung, **HOST** = Host-/CPU-Tests,
**GPU** = GPU-Lauf (A4000, kein CI).

## Task 1 — IMPL: Schnittmengen-Loader (Bias-Fix) [fertig]

- `doe/data.py`: `load_full`, `truncate_norm`, `load_aligned` (alle k auf
  denselben N=16.692 Rows).
- Validierung: HOST — `test_truncate_is_prefix_and_unit_norm` grün.

## Task 2 — IMPL: DoE-Designmodul (LHS, S/N, ARI, RSM) [fertig]

- `doe/design.py`: `PARAM_BOUNDS`/`SEEDS`, `generate_lhs_design` (vektorisiertes
  `qmc.scale`), `taguchi_sn`, `pairwise_ari`, `aggregate_point`,
  `fit_response_surface` (RSM + ANOVA Typ II).
- Validierung: HOST — LHS-Bounds/Reproduzierbarkeit, Taguchi-Jensen-Test,
  ARI-Tests, synthetischer RSM-Fit grün.

## Task 3 — IMPL: Experiment-Hülle + Treiber [fertig]

- `doe/experiment.py`: `run_one` (UMAP+HDBSCAN+Score, Seed im Cache-Key),
  `run_dbscan_eps`; `doe/run_doe.py`: CLI-Phasen jitter/screening/width/dbscan.
- Validierung: GPU-Smoke `--phase dbscan` läuft, 28-Zeilen-CSV geschrieben.

## Task 4 — HOST: DoE-Test-Suite [fertig]

- `tests/test_doe.py`: 10 CPU-Tests (kein cuml/cupy-Import).
- Validierung: `pytest tests/` → 24/24 grün (14 Bestand + 10 neu).

## Task 5 — GPU: Jitter-Phase (R=10, Best-Config) [fertig]

- `run_doe.py --phase jitter`: Best-Config × 10 Seeds auf k=3072-Schnittmenge.
- Validierung: `results_doe_jitter.csv` hat 10 Zeilen; Ergebnis mean=0,1315,
  std=0,0012, ARI=0,72 (Jitter ≈ alte d-Spreizung!).

## Task 6 — GPU: Screening (60 LHS × 3 Seeds) [fertig]

- `run_doe.py --phase screening --n-screening 60 --seeds 42,1337,2026`.
- Validierung: `results_doe_screening.csv` = 180 Zeilen,
  `results_doe_screening_agg.csv` ≤ 60 Zeilen (≥2 Replikate/Punkt),
  `design_screening.csv` reproduzierbar (Seed 42).

## Task 7 — GPU: faire Breiten-Ablation (12 × 3k × 3 Seeds) [fertig]

- `run_doe.py --phase width --n-width 12`: gleiche 12 LHS-Punkte auf
  k ∈ {128, 768, 3072}, gleiche Rows.
- Validierung: `results_doe_width.csv` = 108 Zeilen; Pivot pro run_id zeigt
  echten Breiteneffekt ohne Survival Bias.

## Task 8 — GPU: DBSCAN-eps-Kalibrierung [fertig]

- `run_doe.py --phase dbscan`: eps-Grid × d ∈ {4,8,12,16} bei fixer UMAP.
- Validierung: 28-Zeilen-CSV; Befund eps=0,2 ≈ 0,105 (Heuristik nur ≈0,058).

## Task 9 — IMPL: `plot_final.py`-Legendenfix [fertig]

- Top-12-Cluster + Sammelpunkt („weitere N Cluster") statt 168 Einträgen.
- Validierung: HOST — `python -c "import ast; ast.parse(...)"` + GPU-Plotlauf
  (folgt nach Screening, GPU gerade belegt).

## Task 10 — IMPL: Auswertung + Plots [fertig]

- `doe/analyze.py`: Jitter-Hist, Haupteffekte, S/N-vs-Mean, ARI-Stabilität,
  Breiten-Pivot, DBSCAN-Kurven; robustes Optimum (max S/N) + Peak (max mean)
  bestimmen und als `doe/results/best_doe.json` ablegen.
- Validierung: HOST — alle 5 PNGs + JSON vorhanden; ANOVA-Tabelle druckbar.

## Task 11 — HOST: Gesamt-Suite + Review der Zahlen [offen]

- `pytest tests/` grün; CSV-Zeilen­zahlen prüfen; ANOVA-p-Werte plausibel
  (mind. 1 signifikanter Faktor erwartet: min_cluster_size/d).
- Validierung: alle Checks bestanden, sonst zurück zu Task 6/7 (mehr Punkte).

## Task 13 — GPU/FOLLOW-UP: Manifold-Fidelity (Trustworthiness, TwoNN, PCA) [fertig]

- `doe/fidelity.py` + `doe/run_fidelity.py`: Trustworthiness-vs-d (cuML, k=5/10,
  CPU-Orakel-Referenz), TwoNN-ID (mehrstufige Trims), PCA-Baseline (1024 Komp.).
- `tests/test_fidelity.py`: 10 CPU-Tests (sklearn-Orakel <1e-9, ID-Recovery,
  Trim-Stabilitaet als Regressionsfang für den F-Normierungs-Bug).
- `doe/FIDELITY_de.md`: Bericht (Elbow ab d≈6, Kerne-ID ≈13, PCA-80 % = 273).
- Validierung: `pytest tests/` 34/34 grün; GPU-Selbstcheck <5e-3 (Dubletten);
  alle 3 Plots visuell verifiziert.

## Task 14 — GPU/RUNDE-2: Phase A (CCD × Dedup) [fertig]

- `doe/run_phasea.py`: CCD mcs/ms/nn (20 Punkte) × Dedup-Block × 5 Seeds,
  Scoring rueckprojiziert auf Voll-N, RSM+ANOVA, Argmax je Block,
  Konfirmierung mit 3 frischen Seeds, 3 Plots.
- `doe/{data,design}.py`: `dedup_map`, `generate_ccd_design`, `rsm_argmax`;
  +4 Tests in `tests/test_doe.py` (CCD-Struktur, Dedup-Roundtrip, Peak-Findung).
- Befunde: Dedup Δ=−0,028 (F=860) + 5× Jitter; dedup×nn signifikant,
  dedup×mcs widerlegt; Zoom-Box flach → mcs=11 behalten. Siehe `doe/PHASEA_de.md`.
- Validierung: `pytest tests/` grün; R²=0,975; Konfirm-Seeds bestaetigen Levels.

## Task 15 — GPU/RUNDE-2: Phase B (Bake-off + Validitaet) [fertig]

- `doe/run_phaseb.py`: Bake-off (HDBSCAN/DBSCAN/Leiden/Agglo, je 8–20 Runs,
  fixe d=11-Einbettung), Winner-Seed-Stabilitaet, 80/90-%-Subsamples,
  NPMI+Kohäsion, verblindete Rating-Samples + versiegelter Key.
- `doe/coherence.py`: NPMI-Korpus (deutsch), Top-Woerter, Kohäsion;
  `tests/test_coherence.py` (7 Tests, Themen-vs-Zufall-Orakel).
- Befunde: HDBSCAN gewinnt (0,133) + stabilste Scores; Leiden peaky
  (0,115→0,06, disqualifiziert); NPMI-r≈0,05 (Null!); Rating: Agglo 3,40
  vs. Rest ~4,2–4,5. Siehe `doe/PHASEB_de.md`, `ratings_phaseb.md`.
- Validierung: `pytest tests/` grün; Leiden/Grid innen verifiziert
  (res-Erweiterung 5,0/8,0); Plots visuell geprueft.

## Task 16 — TITEL+KARTE: 219 Titel, Plotly-Map, Inkrement-Mechanismus [fertig]

- `doe/titles.py`: Store (Member-Signaturen über Identifiers) + Jaccard-Matching
  + `plan_update` (Schwelle 0,7); `tests/test_titles.py` (8 Tests).
- `doe/build_map.py`: Jobs (8 Exemplare + 3 Nachbarn) → 12 Titler-Batches →
  Store `cluster_titles_phaseb.json` (219 Titel, alle eindeutig) →
  `plots/clusters.html` (9 MB, ignoriert!) + `plots/labels_phaseb.csv`.
- Demo: Seed-1337-Reclustering → 156 keep / 64 retitle (71 % Ersparnis).
- Doku: `doe/TITLING_de.md` (Format, Algorithmus, Limitationen).
- Validierung: `pytest tests/` grün; Titel-Spotcheck; Karte (220 Traces) gebaut.

## Task 12 — DOCS: Walkthrough + Commit [fertig]

- `plan/20260927_01_review_doe/walkthrough.md` (DE): Einführung, Scope,
  Methode, Ergebnisse, Conclusion (Paper-Stil), Learnings, Erweiterungen,
  Docker-Programme (`statsmodels`, `patsy`, `formulaic`).
- `requirements.txt` um DoE-Pins ergänzen; alles committen (Conventional
  Commits pro Task); `untersuchung_technisch_de.md`-Nachtrag prüfen.
- Validierung: alle in `prompt.txt` geforderten Artefakte vorhanden.
