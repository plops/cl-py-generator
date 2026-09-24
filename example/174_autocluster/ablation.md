# ablation.md — 768 vs 3072 Matryoshka widths (same grid, same criterion)

Criterion: `Silhouette_orig × (1 − noise)`, HDBSCAN (`mcs15`), best per width:

| width k | rows | best config | clusters | noise | sil_orig | adjusted |
|---|---|---|---|---|---|---|
| 128 | 14.146 | d8 nn15 md0.1 | 134 | 31.4 % | 0.1593 | **0.1092** |
| 768 | 14.146 | d12 nn15 md0.1 | 182 | 33.2 % | 0.1765 | **0.1179** |
| 3072 | 12.569 | d12 nn30 md0.0 | 137 | 27.8 % | 0.1716 | **0.1240** |

Verdict: **3072 > 768 > 128**. Full dims gain +5.2 % over the 768 prefix and
+13.5 % over the 128 pilot (adjusted score). Cost: 1.577 short rows (11 %) drop
out at k=3072, UMAP ~1 s/config either way on the A4000. The 128 pilot correctly
ranked methods (HDBSCAN ≫ DBSCAN everywhere) but understated absolute quality —
pilot for speed, decide on full width.

Flat dimension response: d ∈ [4, 16] all within ±0.002 adjusted at every width;
d=2 (DBSCAN adj ≈ 0.03–0.08) confirms the prompt's rule — 2D is for plots only.

## Compact-DB re-sweep (2026-09-24, `summaries_compact_20260924.db`)

Same grid on the compact export (live-DB results preserved in
`archive_livedb_20260924/`). Best per width:

| width k | rows | best config | clusters | noise | sil_orig | adjusted |
|---|---|---|---|---|---|---|
| 128 | 18.269 | d16 nn15 md0.1 | 179 | 35.4 % | 0.1889 | **0.1220** |
| 768 | 18.269 | d08 nn15 md0.0 | 226 | 30.8 % | 0.1833 | **0.1269** |
| 3072 | 16.692 | d12 nn30 md0.1 | 167 | 36.5 % | 0.2091 | **0.1327** |

Same verdict, stronger: **3072 > 768 > 128** (+4.6 % / +8.8 % adjusted).
Winner (`best_params_compact.json`): k=3072, d12, nn30, md0.1, HDBSCAN mcs15.
`plots/best_clusters.png` + `plots/labels.csv` re-rendered on compact data
(refit: 178 clusters, 38.3 % noise — nn_descent variance, see walkthrough).
