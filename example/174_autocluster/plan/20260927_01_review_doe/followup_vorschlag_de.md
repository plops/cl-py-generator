# Follow-up-Vorschlag: Lohnt eine zweite DoE-Runde? (DE)

Datum: 2026-09-27 · Bezug: Runde 1 (LHS-Screening, 60×3) + Fidelity-Follow-up

## Kurzantwort: Ja — aber nicht dasselbe nochmal

Runde 1 hat den 5-D-Steuerraum gut vermessen (RSM-R²=0,88, klare ANOVA).
Ein blindes Re-Sampling (mehr LHS-Punkte, Sobol statt LHS, feineres d-Grid)
haette minimalen Grenznutzen. Der Erkenntnisgewinn liegt bei **neuen Faktoren**
und **neuen Responses** — also dort, wo Runde 1 Fragen aufgeworfen, aber nicht
hingeschaut hat.

## Was geschlossen ist (nicht wiederholen)

- UMAP-Haupteffekte: d, min_dist insignifikant; nn nur via Interaktion.
- Fidelity: Trustworthiness saettigt ab d≈6; TwoNN-Kerne ≈13 bestaetigen d=11.
- Breiten-Ranking 3072 > 768 > 128 (fair, auf Schnittmenge).

## Offene Fragen, rankiert nach erwartetem Erkenntnisgewinn

1. **Optimum an der Faktorgrenze?** mcs=11 bei Untergrenze 10 — das wahre
   Optimum kann unterhalb liegen. CCD-Verfeinerung mcs∈[5,20] (+ms, nn),
   analytisches Maximum + Konfidenzbereich, Konfirmierung mit frischen Seeds.
   Billig (~150–200 Runs), schließt Runde 1 sauber ab.
2. **Preprocessing-Faktoren (vermutlich groesster Hebel):** Dedup an/aus
   (8 % exakte Dubletten!), Rausch-Prefilter (Fehlermeldungs-Embeddings),
   Normierungsvarianten — gekreuzt mit mcs (Dubletten × kleine Cluster
   interagieren sicher). Erwartung: groessere Effekte als alles UMAP-Tuning.
3. **HDBSCAN-Tiefe:** `cluster_selection_method` (eom/leaf),
   `cluster_selection_epsilon`, `alpha` — Runde 1 oeffnete nur mcs/ms.
4. **Algorithmus-Bake-off, fair:** HDBSCAN vs. Leiden/Louvain auf kNN-Graph
   (aussichtsreichster Herausforderer bei Embedding-Daten!) vs. DBSCAN
   (kalibriert) vs. Agglomerative — je eigene Mini-RSM, dann Vergleich bei
   jeweils bestem Setup auf gleichem Response.
5. **Stabilitaet/Wachstum (fuer den Rust-Updater):** Cluster-Persistenz ueber
   Subsamples (80/90/100 %) und Zeit-Splits (alt vs. neu); Pareto-Front
   Score-vs-ARI statt nur Score-Maximum. Produktfrage: bleiben Cluster-IDs
   stabil, wenn die DB waechst?
6. **Externe Validitaet (wichtigste Nicht-DoE-Frage):** Korreliert
   Silhouette×(1−Noise) ueberhaupt mit LLM-bewerteter Themen-Kohaerenz?
   ~30 Cluster × 4–5 Configs raten lassen; bei schwacher Korrelation ist das
   Zielkriterium zu revidieren — wichtiger als jedes weitere Tuning.

## Explizit nicht empfohlen (niedriger Ertrag)

UMAP-Mikroparameter (epochs/init/spread), Re-Sampling-Diskussionen (Sobol),
feinere d-Grids — Runde 1 + Fidelity haben das erledigt.

## Konkreter Vorschlag Runde 2 (~1 GPU-Tag, zwei Phasen)

- **Phase A „Confirm & Preprocessing" (1 DoE):** mcs[5,20] × ms[5,15] ×
  nn[25,60] (CCD, ~20 Punkte) × Dedup{an,aus} (Block) × 5 Seeds ≈ 200 Runs.
  Responses: mean_adj, S/N, ARI, n_clusters. Liefert verifiziertes Optimum +
  Dedup-Effekt + Count-Modell.
- **Phase B „Algorithmus + Validitaet":** Bake-off (je ~15-Run-Mini-RSM) +
  LLM-Kohaerenz-Rating der Top-Configs + Subsample-Stabilitaet. Liefert
  Methoden-Entscheid + externes Guetesiegel.
- **Wenn nur eines:** Phase A (billig, schliesst ab). Vor dem Rust-Updater
  zusaetzlich Phase B (verhindert, auf dem falschen Kriterium zu produzieren).
