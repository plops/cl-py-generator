# ablation_de.md — 768 vs 3072 Matryoshka-Breiten (gleiches Grid, gleiches Kriterium)

Deutsche Übersetzung von `ablation.md` (Stand 2026-09-24).

Kriterium: `Silhouette_orig × (1 − Noise)`, HDBSCAN (`mcs15`), jeweils beste Config pro Breite:

| Breite k | Rows | beste Config | Cluster | Noise | sil_orig | adjustiert |
|---|---|---|---|---|---|---|
| 128 | 14.146 | d8 nn15 md0.1 | 134 | 31,4 % | 0,1593 | **0,1092** |
| 768 | 14.146 | d12 nn15 md0.1 | 182 | 33,2 % | 0,1765 | **0,1179** |
| 3072 | 12.569 | d12 nn30 md0.0 | 137 | 27,8 % | 0,1716 | **0,1240** |

Fazit: **3072 > 768 > 128**. Volle Dimensionen bringen +5,2 % gegenüber dem
768er-Prefix und +13,5 % gegenüber dem 128er-Piloten (adjustierter Score).
Kosten: 1.577 kurze Rows (11 %) fallen bei k=3072 weg, UMAP braucht pro Config
auf der A4000 ca. 1 s — unabhängig von der Breite. Der 128er-Pilot hat die
Methoden korrekt gereiht (HDBSCAN ≫ DBSCAN überall), aber die absolute Qualität
unterschätzt — Pilot für Tempo, Entscheidung auf voller Breite.

Flache Dimensionsantwort: d ∈ [4, 16] liegt bei jeder Breite innerhalb ±0,002
adjustiert; d=2 (DBSCAN adj ≈ 0,03–0,08) bestätigt die Regel aus dem Prompt —
2D nur für Plots.

## Compact-DB-Re-Sweep (2026-09-24, `summaries_compact_20260924.db`)

Gleiches Grid auf dem kompakten Export (Live-DB-Ergebnisse gesichert unter
`archive_livedb_20260924/`). Beste Config pro Breite:

| Breite k | Rows | beste Config | Cluster | Noise | sil_orig | adjustiert |
|---|---|---|---|---|---|---|
| 128 | 18.269 | d16 nn15 md0.1 | 179 | 35,4 % | 0,1889 | **0,1220** |
| 768 | 18.269 | d08 nn15 md0.0 | 226 | 30,8 % | 0,1833 | **0,1269** |
| 3072 | 16.692 | d12 nn30 md0.1 | 167 | 36,5 % | 0,2091 | **0,1327** |

Gleiches Fazit, deutlicher: **3072 > 768 > 128** (+4,6 % / +8,8 % adjustiert).
Sieger (`best_params_compact.json`): k=3072, d12, nn30, md0.1, HDBSCAN mcs15.
`plots/best_clusters.png` + `plots/labels.csv` auf Compact-Daten neu gerendert
(Refit: 178 Cluster, 38,3 % Noise — nn_descent-Varianz, siehe Walkthrough).
