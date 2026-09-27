# PHASEB_de.md — Runde 2, Phase B: Methoden-Bake-off + externe Validitaet (DE)

Datum: 2026-09-27 · Code: `doe/run_phaseb.py`, `doe/coherence.py` ·
Tests: `tests/test_coherence.py` (7) · GPU: RTX A4000 (+ sklearn-CPU)

## 1. Ziel

Drei Fragen aus dem Runde-2-Vorschlag: (1) Ist HDBSCAN wirklich die beste
Methode — fairer Bake-off auf EINER fixen Einbettung (d=11, nn=45, md=0.09),
jede Methode mit eigenem Mini-Tuning (~8–20 Runs)? (2) Bleiben Winner stabil
(Seeds, Subsamples 80/90 %)? (3) Misst unser Score, was zaehlt — externe
Validitaet via Wort-Kohaerenz (NPMI, unabhaengig von Geometrie) + verblindetes
Author-Rating (24 Units)?

## 2. Bake-off (fixe Einbettung, Response wie immer)

| Methode | Best-Setup | adj | ncl | Noise | Bemerkung |
|---|---|---|---|---|---|
| HDBSCAN | mcs=11, ms=8 | **0,1333** | 219 | 36 % | Runde-1-Optimum bestaetigt |
| Leiden | knn=15, res=5,0 | 0,1150 | 138 | 0 % | Vollabdeckung; res-Optimum innen (8,0 kollabiert) |
| DBSCAN | eps=0,2, ms=25 | 0,1042 | 106 | 38 % | Flaches Top um eps 0,15–0,25 |
| Agglo | ward, nc=150 | 0,0308 | 150 | 0 % | average-Linkage sogar negativ → raus |

HDBSCAN gewinnt kriterium-intern; Leiden ist der einzige ernsthafte
Verfolger — mit Vollabdeckung (0 % Noise) als strukturellem Plus.

## 3. Stabilitaet: Leiden peaky, HDBSCAN felsenfest

| Methode | ARI Seeds | Score-Spanne Seeds | ARI sub-80/90 % |
|---|---|---|---|
| HDBSCAN | 0,67 | 0,133–0,135 (σ=0,0005!) | 0,54 / 0,55 |
| DBSCAN | **0,80** | 0,098–0,104 | 0,62 / 0,71 |
| Leiden | 0,70 | **0,115 → 0,056/0,065** | 0,63 / 0,66 (Score kollabiert) |
| Agglo | 0,75 | 0,028–0,031 | 0,68 / 0,73 |

Leidens Bake-Wert war Seed-42-Glueck (feste Resolution + neuer kNN-Graph =
andere Partition) — als Herausforderer **disqualifiziert**. HDBSCAN hat die
stabilsten Scores bei maessigster Label-Persistenz; DBSCAN die stabilsten
Labels. Nb.: keine Ingestions-Zeitstempel in der DB → nur Random-Subsamples.

## 4. Externe Validitaet: Geometrie ≠ Woerter (Nullresultat!)

Pro Cluster (n≥10): NPMI-Wortkohaerenz vs. geometrische Kohäsion (n≈600).

- **Pooled r = 0,05** (je Methode −0,00…0,16): geometrische Enge sagt
  Wort-Kohaerenz praktisch NICHT voraus. NPMI-Mittel je Methode fast
  identisch (0,23–0,24) — trotz Score-Spreizung 0,03…0,13!
- Blind-Rating (24 Units, Skala 1–5): DBSCAN 4,50 ≥ HDBSCAN 4,33 ≈
  Leiden 4,17 > **Agglo 3,40** (inkl. 195er Leer-Summary-Cluster).
  Details: `doe/results/ratings_phaseb.md` (vor Entblindung versiegelt).

Deutung: Der Score misst Dichte-/Abstentions-Qualitaet (HDBSCAN streicht
36 % schwerste Punkte), Menschen/Woerter sehen Themen-Qualitaet — und da
sind HDBSCAN/DBSCAN/Leiden gleichauf. Der Score ist **teil-validiert**:
er verwirft Agglo zu Recht (Mensch: 3,40), aber sein Fein-Ranking
(HDBSCAN > Leiden > DBSCAN) bestaetigt sich human nicht (4,33 ≈ 4,17 < 4,50,
alles in Klein-N-Rauschen).

## 5. Conclusion (Methoden-Entscheid)

- **Produktion: HDBSCAN** (mcs=11, ms=8, d=11, nn=45, md=0,09, k=3072) —
  bester Score, stabilste Scores, gute Themen, 217 Cluster. Offene
  Produktfrage: 36 % Noise sind gewollte Abstention — braucht einen
  Umgang (eigene Anzeige vs. Second-Level-Zuordnung im Rust-Updater).
- **Leiden verworfen** (peaky trotz Vollabdeckung), **Agglo verworfen**
  (0,03 + menschlich schlechter), **DBSCAN bleibt valider Zweiter**
  (stabilste Labels, human 4,50 — falls weniger, grobere Cluster (106)
  gewuenscht sind).
- **Wichtigste Lehre:** Silhouette×(1−Noise) optimiert Dichte, nicht
  Themen — fuer Themen-Entscheide braucht es NPMI/Rating daneben.
  Die Fragmentierungsfrage (217 vs. 465 Cluster, Phase A) bleibt damit
  extern unbeantwortet: naechster Schritt waere Rating-gegen-Count.

## 6. Artefakte

- `doe/results/results_phaseb_{bake,winners,stab,sub,coh,coh_summary}.csv/json`,
  `labels_pb_*.npy` (Winner-Labels), `samples_phaseb_{units,key}.json`,
  `ratings_phaseb.md` (Rating vor Key-Oeffnung geschrieben).
- `doe/plots/phaseb_{bake,stab,coh}.png`. Cache `pb_*.npy` lokal (gitignore).
- Neu in `requirements.txt`: igraph==1.0.0, leidenalg==0.12.0 (+texttable).
