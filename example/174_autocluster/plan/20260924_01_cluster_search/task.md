# task.md — Serielle Tasks (Modus A: Implementierung, Modus B: Host-Tests)

**Status 2026-09-24: ausgeführt** — A0–A7, B1, B3 done (11 Unit-Tests grün,
120 GPU-Configs gescort, Plots + walkthrough committed-nah abgelegt).
Offen: B2 Seed-Stabilität per ARI (nn_descent ist nicht deterministisch, siehe
walkthrough), Trustworthiness-Elbow/TwoNN-Plots (Strategien 2–3).

Regeln: strikt seriell abarbeiten, pro Task erst Validierung grün → dann **ein**
Conventional-Commit-Format senden (Typ/Scope, Subject ≤ 72 Zeichen, Body mit
Was/Warum/Validierung). Commit-Typen: `feat`, `test`, `docs`, `chore`, `fix`.
Beispiel: `feat(loader): read embeddings from sqlite with matryoshka truncation`.
Nach Vollimplementierung + Commits: `walkthrough.md` neben `prompt.txt` ablegen
(was gebaut, Test-bedingte Änderungen, Learnings, Docker-Pakete).
Reines Python, kein Lisp-Transpiler. Env: `.venv` im Projektdir per `uv`
(Skill `python-env` beachten). DB immer read-only öffnen.

## Modus A — Implementierung

- [ ] **A0 Env + Skelett.** `.venv` anlegen, `cuml-cu12`/`cupy-cuda12x` via
  `https://pypi.nvidia.com` + `numpy pandas matplotlib scikit-learn pytest`
  installieren (neueste Versionen, Pins in `ENV.md` festhalten).
  Dateien: `loader.py sweep_umap.py cluster_search.py validate.py
  ablation_width.py plot_final.py tests/ requirements.txt ENV.md`.
  *Validierung:* `python -c "import cuml, cupy"` auf GPU-Container ok.
- [ ] **A1 Loader + Unit-Tests.** `loader.py`: read-only SQLite, LE-f32-Dekodierung,
  Trunkierung auf Breite k (nur Rows len ≥ k), Zero-Norm-Filter, `X`/`X_norm`/`ids`.
  Tests: Roundtrip `embedding_to_bytes`, 3072→k-Prefix, gemischte 768/3072-Auswahl,
  Zero-Vektor-Drop, L2-Norm == 1. *Validierung:* `pytest tests/test_loader.py` grün +
  echte Zählung (14.146 / k=128 alle, k=3072 → 12.569).
- [ ] **A2 UMAP-Sweep (Pilot k=128).** `sweep_umap.py`: Grid d×n_neighbors×min_dist
  (plan.md §2), `metric='cosine'`, `nn_descent`, Seed fix, Timing, `.npy`-Cache.
  *Validierung:* Smoke mit 500 Rows < 5 min, Shapes stimmen, rerun deterministisch.
- [ ] **A3 Cluster-Suche.** `cluster_search.py`: DBSCAN (eps-Skala pro d) + HDBSCAN
  auf Pilot-`X_reduced`; Labels, Noise-Ratio, n_clusters pro Config.
  *Validierung:* synthetische Blobs → erwartete Clusterzahl; Noise ∈ [0,1].
- [ ] **A4 Validierung.** `validate.py`: Score im Originalraum
  (`Silhouette × (1−noise)`, Noise > 40 % / 1 Cluster → verworfen),
  Trustworthiness-vs-d, TwoNN + PCA-Baseline.
  *Validierung:* Unit-Test Score-Formel (perfekte Cluster ≈ 1·(1−noise));
  Elbow-Plot auf Pilot-Daten erzeugt.
- [ ] **A5 Ablation 768 vs 3072.** `ablation_width.py`: gleiche Pipeline k=768
  (alle Rows) vs k=3072 (12.569 Rows), Delta von Score/Trustworthiness/Clustern
  in `ablation.md` quantifizieren. *Validierung:* Tabelle + Aussage Benefit ja/nein.
- [ ] **A6 Final-Plots.** `plot_final.py`: beste Config → 2D-Refit, Cluster-Scatter
  (Noise grau) + Trustworthiness-Elbow als PNGs unter `plots/`.
  *Validierung:* PNGs vorhanden, Cluster trennbar, `best_params.json` + `labels.csv`
  geschrieben.
- [ ] **A7 Docs.** `README.md` (Run-Anleitung), `ablation.md`, `ENV.md` finalisieren.
  *Validierung:* frischer Run nach README reproduziert `best_params.json`.

## Modus B — Host-Tests (GPU-Maschine, nach Modus A)

- [ ] **B1 GPU-Smoke.** `nvidia-smi` ok, `cupy`/`cuml` importieren, mini-UMAP (500×128)
  < 5 min. *Validierung:* Log + Timing.
- [ ] **B2 Pilot-Suite.** `pytest tests/` + Pilot k=128 voll: Score-Tabelle,
  Seed-Stabilität (2 Seeds, ARI berichten), Noise < 40 %-Regel greift.
  *Validierung:* Ergebnisse in `results_pilot.csv`.
- [ ] **B3 Full-Suite + Ablation.** k=768 + k=3072 Sweeps, `ablation.md`-Zahlen,
  Final-PNGs Review (2D-Diagramme der besten Cluster). *Validierung:* alle Artefakte
  vorhanden, `walkthrough.md` geschrieben und committed.
