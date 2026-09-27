# Walkthrough: Robuste Cluster-Evaluation per Design of Experiments (DE)

Datum: 2026-09-27 · Autor: Muse Code (im Auftrag von wol pumba) ·
Code: `example/174_autocluster/doe/` · GPU: NVIDIA RTX A4000

## 1. Einführung: Worum ging es?

Stellen Sie sich vor, Sie haben knapp 20.000 automatisch erzeugte
Video-Zusammenfassungen und möchten wissen: *Welche Themen stecken darin?*
Die Vorgängeruntersuchung beantwortete das mit einer GPU-Pipeline (UMAP +
HDBSCAN) und kürte per Grid-Sweep eine Best-Config. Doch ein Review stellte
die unbequemen Fragen: War der Breitenvergleich fair, wenn bei voller Breite
1.577 Dokumente stillschweigend herausfielen? Ist der hauchdünne Sieg von
`d=12` über `d=8` mehr als GPU-Zufall? Und wurde DBSCAN je fair behandelt?
Diese Untersuchung beantwortet alle drei Fragen — mit der statistischen
Gründlichkeit eines **Design of Experiments (DoE)**: 298 GPU-Runs, 5
Seed-Replikate, Varianzanalyse statt Bauchgefühl.

**Scope des Experiments:** UMAP+HDBSCAN-Hyperparameter­raum (5 Steuergrößen +
Seed-Störgröße) auf der Schnittmenge N=16.692 (Compact-DB, volle 3072-D-Rows);
Zielfunktion `Silhouette_orig × (1 − Noise)` im k-dim Originalraum; faire
Breiten-Ablation k ∈ {128, 768, 3072}; faire DBSCAN-eps-Kalibrierung;
Legendenfix in `plot_final.py`. Außerhalb des Scopes: Manifold-Fidelity
(Trustworthiness/TwoNN, weiter offen), Cluster-Betitelung, Online-Einordnung
neuer Embeddings (Folgeprojekt).

## 2. Methode: Was wurde gebaut?

Neuer Ordner `doe/` (reines Python, kein Lisp-Transpiler), ein Modul pro
Schritt: `data.py` lädt die Schnittmenge einmal und trunkiert pro Breite
(Bias-Fix); `design.py` erzeugt den Latin-Hypercube-Plan (60 + 12 Punkte),
Taguchi-S/N und ARI-Stabilität und fittet das Response-Surface-Modell
(ANOVA Typ II via `statsmodels`); `experiment.py` kapselt
UMAP+HDBSCAN+Scoring mit Seed-im-Cache-Key; `run_doe.py` fährt die vier
Phasen (jitter 10×, screening 60×3, width 12×3×3, dbscan 4×7);
`analyze.py` berichtet ANOVA, Optima und schreibt 5 Plots. Dazu 10 CPU-Tests
(`tests/test_doe.py`) und Docs (`plan.md`, `task.md`, `deps.md`, dies
Dokument). Alle Läufe auf der A4000, DB strikt read-only.

## 3. Ergebnisse: Was kam heraus?

**Alle vier Review-Befunde bestätigt — und quantifiziert:**

1. **Survival Bias existiert, ändert das Ranking aber nicht.** Faire Ablation
   auf identischen Rows: k=128 → 0,120, k=768 → 0,124, k=3072 → 0,126.
   „Volle Breite lohnt sich" gilt auch ohne Bias — aber k=128 ist
   fragiler (4 von 12 Punkten verworfen).
2. **Der d-Sieg war Rauschen — der wahre Hebel ist `min_cluster_size`.**
   Jitter der Best-Config über 10 Seeds: std=0,0012 (bei ±0,002 alter
   d-Spreizung!). ANOVA (R²=0,88): `min_cluster_size` hochsignifikant
   (p=0,001), dazu die Interaktionen `mcs:min_samples` (p=0,005) und
   `n_neighbors:min_cluster_size` (p=0,024) — die vom Review vermutete
   UMAP×HDBSCAN-Wechselwirkung, erstmals nachgewiesen. `d` (p=0,27) und
   `min_dist` (p=0,56) sind insignifikant.
3. **DBSCAN wurde unterschätzt, bleibt aber Zweiter.** Faires eps-Grid:
   bester DBSCAN 0,110 (d=4, eps=0,2) statt ≈0,058 mit der alten Heuristik —
   fast verdoppelt, doch HDBSCAN (0,134) führt weiter.
4. **Neues robustes Optimum schlägt den alten Bestwert:** d=11, nn=45,
   md=0,093, mcs=11, ms=8 → mean=0,1343 (217 Cluster, ARI=0,66) —
   besser als die alten 0,1327, und auf der *strengeren* Schnittmenge
   erzielt. Robust- und Peak-Optimum fallen zusammen.
5. **Legendenfix verifiziert:** 174-Cluster-Plot mit kompakter Legende
   (Top-12 + Sammelpunkt) rendert fehlerfrei.

## 4. Conclusion

Das DoE hat gehalten, was das Review versprach: Statt eines zufälligen
Peak-Scores auf geratener Heuristik besitzen wir nun eine **signifikante,
reproduzierbare Parameterwahl** — mit bekanntem Rausch­pegel (σ≈0,001),
nachgewiesenen Interaktionseffekten und einem Optimum, das gleichzeitig
Spitze und robust ist. Die wichtigste praktische Lehre: Wer UMAP+HDBSCAN
tunt, sollte seine Zeit in `min_cluster_size`/`min_samples` stecken, nicht
in die UMAP-Dimension. Die offene Flanke bleibt die Manifold-Fidelity
(Trustworthiness-Elbow, TwoNN) sowie die Seed-Stabilität einzelner Themen
(ARI 0,62–0,78): gut genug für Exploration, zu prüfen vor Produktion.

## 5. Learnings & mögliche Erweiterungen

- **Gelernt 1:** `qmc.scale` vektorisiert nutzen (Review-Skizze skalierte
  skalar) — schneller und weniger fehleranfällig.
- **Gelernt 2:** Taguchis „robust schlägt peaky" gilt nur bei gleichem
  Mittelwert (Jensen); +0,002 Mittelwert schlägt ±0,015 Streuung — der
  Review-Vergleich war illustrativ, der Test präzisiert ihn.
- **Gelernt 3:** k=128-Punkte scheitern überproportional (Noise > 40 %) —
  schmale Prefixe sind nicht nur schlechter, sondern instabiler.
- **Erweiterung A:** konfirmatorische 2. DoE-Runde (Box-Behnken/CCD) eng um
  das robuste Optimum zur Feinvermessung.
- **Erweiterung B:** `brute_force_knn`-Kontrollläufe als Determinismus-Anker.
- **Erweiterung C:** Rust-Online-Einordner für neue Embeddings (aus `collect.sh`
  entworfener Follow-up-Auftrag): Clustern ohne GPU, nur neue Samples ans LLM.

## 6. Neue Programme für den Docker-Container

In `requirements.txt` gepinnt (alle über Standard-PyPI, kein NVIDIA-Index
nötig): `statsmodels==0.15.0` (ANOVA/RSM), `patsy==1.0.3` und
`formulaic==1.2.2` (Formel-Parser, transitiv). Installations­befehl laut
`ENV.md`, ergänzt um `-r requirements.txt` (enthält die neuen Pins).

## 7. Artefakte (Nachweis)

- Code: `doe/{data,design,experiment,run_doe,analyze}.py`, Fix in
  `plot_final.py`, Tests in `tests/test_doe.py` (24/24 grün).
- Daten: `doe/results/` (180+108+28+10 Runs, `best_doe.json`, Report, Logs).
- Plots: `doe/plots/doe_*.png` (5) + `plotfix_check/best_clusters.png`.
- Docs: `plan/20260927_01_review_doe/{prompt,review,deps,plan,task}.md` +
  dies `walkthrough.md`. UMAP-Caches (`doe/umap_cache_doe/`, ~300 `.npy`)
  sind per `.gitignore` (`umap_cache*/`) vom Commit ausgenommen.
