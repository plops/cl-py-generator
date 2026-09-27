# FIDELITY_de.md — Manifold-Fidelity-Follow-up (Trustworthiness, TwoNN, PCA)

Datum: 2026-09-27 · Code: `doe/fidelity.py`, `doe/run_fidelity.py` ·
Tests: `tests/test_fidelity.py` (10 Tests) · GPU: RTX A4000

## 1. Ziel

Der Vorgaengerplan (§1.2–1.3) forderte zwei intrinsische (vom Clustering
unabhaengige) Kriterien zur Wahl der UMAP-Dimension d — beide blieben offen.
Dieses Follow-up holt sie auf der Schnittmenge (N=16.692, k=3072) nach:

1. **Trustworthiness-Elbow:** Trustworthiness vs. d (nn=30, md=0.1),
   Regel „kleinstes d mit T ≥ 0,92".
2. **Intrinsische Dimensionalitaet:** TwoNN-Schaetzer (Facco et al. 2017)
   + PCA-Varianz-Baseline (60/80/90/95 %).

## 2. Methode in Kuerze

- UMAP-d-Grid {2, 4, 6, 8, 12, 16, 20, 24} + d=12-Seeds {1337, 2026}
  (Mini-Stabilitaet), Trustworthiness per cuML (k=5/10, Kosinus via
  L2-Normierung — euklidisches Ranking = Kosinus-Ranking).
- CPU-Referenzimplementierung (`trustworthiness_cpu`, gechunkt) stimmt auf
  bindungsfreien Daten mit dem sklearn-Orakel auf <1e-9 ueberein (Test);
  GPU-vs-CPU-Selbstcheck im Treiber.
- TwoNN auf Xn (euklidisch/chordal — Standardnaeherung, s. Caveats),
  Fits ueber mehrere Trim-Stufen (0–75 % groesste μ); PCA randomized, 1024 Komp.

## 3. Ergebnisse

### 3.1 Trustworthiness vs. d (k=10): Elbow bei d=4, Saettigung ab d≈6

| d | 2 | 4 | 6 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| T(k=10) | 0,952 | 0,981 | 0,985 | 0,985 | 0,986 | 0,986 | 0,986 | 0,988 |

- Elbow-Regel feuert schon bei **d=2** (0,95 ≥ 0,92) — als Regel zu lax, aber
  die Kurve ist eindeutig: grosser Gewinn d=2→4 (+0,03), danach flach (+0,003
  bis d=24). Fidelity-maessig sind alle d ≥ 6 praktisch gleichwertig.
- Seed-Jitter auf d=12: 0,9856/0,9856/0,9859 — vernachlaessigbar. Pointe:
  Nachbarschaften sind seed-stabil, obwohl Cluster-Labels variieren (ARI 0,72)
  — HDBSCAN-Grenzen wackeln, die lokale Geometrie nicht.
- Einordnung: Trustworthiness misst nur *lokale* Nachbarschaft (k=10);
  dass d=2 global Themen uebereinanderstapelt (Clustering-Score bricht ein),
  widerspricht dem nicht — zwei verschiedene Qualitaeten.

### 3.2 TwoNN-ID: heterogen — Kerne ≈ 13, globaler Mittel ≈ 4–5

| Trim | 0 % | 10 % | 25 % | 50 % | 75 % |
|---|---|---|---|---|---|
| ID | 4,10 | 5,37 | 8,94 | 12,70 | 14,34 |

- Die (log μ, −log(1−F))-Kurve ist **gekruemmt**: dichte Clusterkerne haben
  hohe lokale ID (≈13–14, Klein-μ-Bereich), duenne Rausch-/Randregionen
  ziehen den globalen Fit auf ≈4–5. Eine einzige ID-Zahl ist daher irrefuehrend;
  ehrlich ist die Staffel oben. 1.387 Punkte (Dubletten, r1=0) ausgeschlossen.
- Konvergenz mit dem DoE: Das robuste Clustering-Optimum (d=11) sitzt genau
  im ID-Bereich der dichten Kerne (≈13) — UMAP-Dimension ≈ intrinsische
  Dimension der Themen, nicht der des Rauschens. Der Plan-Anker (ID ≈ 8–22)
  ist bestaetigt (Kern-Schaetzung).
- Methodik-Lehre (Bug gefunden & gefixt): F muss auf der *vollen* Stichprobe
  gezaehlt werden; Neunormierung auf der getrimmten Menge ließ die ID auf
  >100 explodieren. Regressionstest `test_twonn_trim_stable_on_uniform_data`.

### 3.3 PCA-Baseline: linear hochdimensional (273 Komp. für 80 %)

60 % → 83, **80 % → 273**, 90 % → 503, 95 % → 716 Komponenten.
Die lineare ID ist um Groessenordnungen hoeher als TwoNN (≈13) — die
Mannigfaltigkeit ist stark gekruemmt/nichtlinear; lineare Reduktion (PCA)
ist hier kein Ersatz für UMAP. (256 Komponenten reichten nicht bis 80 %,
daher 1024er-Lauf.)

## 4. Conclusion

Beide offenen Punkte sind geschlossen — mit einer Korrektur am Plan: Die
Elbow-Regel „T ≥ 0,92" ist zu schwach formuliert (trifft schon d=2), aber die
Kurve liefert die eigentliche Antwort (Saettigung ab d≈6). Zusammen mit dem
DoE („d insignifikant in [4, 20]") ist die d-Wahl im Bereich [6, 12] nun
doppelt abgesichert: weder Clustering-Score noch Fidelity widersprechen sich,
TwoNN-Kerne (≈13) und DoE-Optimum (d=11) konvergieren. Praktisch: d=11 behalten.

## 5. Nebenbefunde & Caveats

- **1.324 exakte Dubletten** in der Schnittmenge (≈8 %, plausibel: wiederholte
  Fehlermeldungs-Embeddings). Sie erzeugen Rang-Bindungen, bei denen sich
  Implementierungen legal um ~1e-3 unterscheiden (Min-Rang vs. argsort);
  dokumentiert in `trustworthiness_cpu`, Toleranz im Treiber-Selbstcheck.
- TwoNN laeuft auf chordalen (euklidischen) Distanzen normierter Vektoren —
  Naeherung an geodaetische Distaenzen, im Kleinwinkelbereich exakt genug;
  bei Bedarf mit arccos-Distanzen verfeinerbar.
- PCA auf zentrierten Roh-Vektoren (Standard), nicht auf Xn — linearer
  Referenzwert, kein Kosinus-Pendant.

## 6. Artefakte

- `doe/results/results_fidelity_trust.csv` (20 Zeilen),
  `results_fidelity_twonn.json`, `results_fidelity_pca.{json,csv}`.
- `doe/plots/doe_fidelity_{trust,twonn,pca}.png`.
- UMAP-Cache `doe/umap_cache_doe/fid_*.npy` (per `.gitignore` lokal).
