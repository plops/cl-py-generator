# TITLING_de.md — Cluster-Titel: Store, Karte und inkrementelle Updates (DE)

Datum: 2026-09-27 · Code: `doe/titles.py`, `doe/build_map.py` ·
Tests: `tests/test_titles.py` (8) · GPU: RTX A4000 (nur 2-D-UMAP)

## 1. Worum geht es

Titel für 219 Produktions-Cluster (HDBSCAN mcs=11/ms=8, d=11/nn=45/md=0.09)
wurden frisch per LLM erzeugt — je Cluster 8 Samples im Kontrast zu 3
Nachbar-Clustern (12 Titler-Batches). Da ein Volldurchlauf teuer ist, liegt
das Ergebnis in einem **Titel-Store mit Content-Hash**: Bei künftigem
Re-Clustering (neue Daten, neue Seeds) werden per Jaccard-Matching nur
wesentlich geänderte/neue Cluster neu betitelt.

## 2. Store-Format (`cluster_titles_phaseb.json`, ~180 KB, committet)

```json
{"version": 1, "meta": {...},
 "titles": {"5": {"title": "...", "n": 123, "member_sig": "abc…",
   "job_sig": "def…", "members": [identifier…], "exemplars": [ids…],
   "neighbors": [cluster…]}}}
```

Mitgliedschaft über **stabile DB-Identifiers** (nicht Zeilenpositionen) —
robust gegen DB-Wachstum. `member_sig` = Hash der Member-Menge (Schnellcheck),
`job_sig` = Hash dessen, was das LLM sah (Exemplare + Nachbarn, Audit-Trail).

## 3. Update-Mechanismus (`titles.plan_update`, Schwelle Jaccard ≥ 0,7)

1. Neues Clustering → `{identifier: cluster}` (Noise −1 ausgenommen).
2. Jaccard-Matching neu→alt über Identifier-Mengen (Label-Permutation egal).
3. `keep` (Titel+Signatur übernehmen) iff Jaccard ≥ 0,7 und alter Titel da;
   sonst `retitle`-Job (neue Exemplare + aktuelle Nachbar-Titel ans LLM).
4. Store neu schreiben (Version hochzählen bei Formatwechsel).

**Demo mit echten Daten** (`build_map.py --step demo`): Seed-1337-Reclustering
als „Zukunft" → **156 keep, 64 retitle** (71 % Ersparnis), Median-Jaccard
der Keeps 0,89. So sieht der Normalbetrieb aus.

## 4. Bekannte Limitationen (v1)

- Nur Membership triggert; geänderte Nachbar-Titel allein kein Retitling
  (Kontext wird zum Job-Zeitpunkt frisch gezogen — bei Batch-Retitles ggf.
  zwei Durchläufe fahren).
- Schwelle 0,7 heuristisch (an Seed-Jitter kalibrierbar: Median-Keep ≈ 0,89).
- Leere-Summary-Cluster bekommen Muster-Titel („… ohne Zusammenfassung") —
  Absicht, kein Fehler (sichtbar in der Karte).

## 5. Artefakte & Workflow

- Titel-Jobs: `doe/results/title_jobs.json` + `title_batch_*.json` (lokal,
  Regenerat aus Labels+DB — was das LLM sah).
- Batch-Ergebnisse: `title_titles_batch_*.json` (lokal, aus Titler-Logs).
- Store: `cluster_titles_phaseb.json` (Repo) · Labels: `plots/labels_phaseb.csv`
  (Repo) · Karte: `plots/clusters.html` (9 MB, **gitignoriert, nie committen**).
- Befehle: `build_map.py --step jobs|store|html|labels|demo`.
