# Von Embeddings zu Themen: GPU-Clustering von 20.000 YouTube-Summaries

Technischer Bericht, Stand 2026-09-24 · Code: `example/174_autocluster` ·
GPU: NVIDIA RTX A4000 (16 GB)

## 1. Motivation

Mehrere zehntausend automatisch erzeugte Video-Zusammenfassungen liegen als
Vektoren (Embeddings) vor — maschinenlesbar, aber für Menschen unübersichtlich.
Die Frage war, ob sich darin von selbst Themengebiete finden: Lassen sich die
Videos ohne manuelle Labels in kohärente Cluster gruppieren, und wenn ja, mit
welchen Parametern? Konkret standen drei Fragen im Raum: Erstens, welche
UMAP-Parameter und welcher Cluster-Algorithmus die besten Gruppen liefern.
Zweitens, ob Googles Matryoshka-Embeddings ihre volle Breite von 3072
Dimensionen brauchen oder ob das 768er-Prefix genügt — ein praktischer Trade-off
zwischen Rechenaufwand und Qualität. Drittens sollte am Ende etwas Anschauliches
herauskommen: Diagramme der besten Cluster und, darauf aufbauend, lesbare Namen
für jedes Cluster plus eine interaktive Karte zum Stöbern.

## 2. Die Daten

Wir arbeiteten mit zwei Datenbanken aus dem Schwesterprojekt `rs-summarizer`,
beide strikt read-only gelesen. Die Live-DB (`summaries.db`) enthält 14.355
erfolgreiche Summaries, davon 14.146 mit Embedding. Später kam die kompakte
Export-DB (`summaries_compact_20260924.db`, 19.293 Rows) ins Projektdir dazu —
sie folgt dem schlanken Export-Schema ohne `transcript`- und `summary_done`-
Spalten, unser Loader erkennt beide Varianten automatisch.

| DB | Rows | 768-D | 3072-D | NULL | nutzbar k=128 | nutzbar k=3072 |
|---|---|---|---|---|---|---|
| Live | 14.355 | 1.577 | 12.569 | 209 | 14.146 | 12.569 |
| Compact | 19.293 | 1.577 | 16.692 | 1.024 | 18.269 | 16.692 |

Jedes Embedding ist als BLOB aus little-endian float32-Zahlen gespeichert
(Format aus `viz-tool/src/embedding.rs` übernommen). Entscheidend ist die
Matryoshka-Eigenschaft der Gemini-Embeddings: Kürzere Vektoren sind ein Prefix
der längeren, man darf also bedenkenlos die ersten k Zahlen abschneiden.
Null- und Zero-Norm-Vektoren (z. B. aus fehlgeschlagenen Embedding-Läufen über
Fehlermeldungen statt Transkripten) werden herausgefiltert.

## 3. Die Methode: im Kleinen clustern, im Großen bewerten

Das methodische Herzstück ist eine Regel, die man leicht verletzt: Die Qualität
eines Clusterings darf man **nicht** in dem reduzierten Raum messen, in dem man
geclustert hat. Vergleicht man d=2 mit d=15 und rechnet jeweils dort den
Silhouette-Score aus, vergleicht man Äpfel mit Birnen — in niedrigen Dimensionen
liegen Punkte künstlich dichter, und UMAP mit `min_dist=0.0` presst sie
zusätzlich in Klumpen, was den Score schmeichelt, ohne dass die Gruppen
semantisch stimmen müssten. Deshalb: Clustern im reduzierten d-dimensionalen
Raum, aber bewerten im originalen hochdimensionalen Raum (L2-normalisiert, was
mathematisch der Kosinus-Distanz entspricht). Unser Kriterium lautet
`Score = Silhouette_orig × (1 − NoiseRatio)`; Lösungen mit über 40 % Noise oder
weniger als zwei Clustern fallen durch. Ergänzend sah der Plan
Trustworthiness-Elbow und TwoNN/PCA-Schätzer vor — das Ranking stand bereits,
diese Manifold-Fidelity-Plots bleiben offene Follow-ups.

Durchsucht haben wir Breiten k ∈ {128, 768, 3072}, Zieldimensionen
d ∈ {2, 4, 8, 12, 16, 24}, Nachbarschaften {15, 30} und `min_dist` {0.0, 0.1},
stets mit Kosinus-Metrik, approximativem `nn_descent`-Graphen und fixem Seed.
DBSCAN lief mit dimensions-skaliertem Epsilon (`0.3 + d·0.02`), HDBSCAN mit
`min_cluster_size=15`. Die 2D-Ebene war von vornherein nur für Plots reserviert:
Ab etwa d≥25 setzt die Distanzkonzentration ein und dichtebasiertes Clustern
bricht zusammen, während d=2 semantisch verschiedene Themen übereinander
stapelt — genau der Fehler, den die alte CPU-Pipeline (`viz-tool` clustert in
2D/4D mit O(n²)-DBSCAN) noch macht und den wir ersetzen.

## 4. Die Umsetzung

Entstanden ist eine schlanke Pure-Python-Pipeline ohne Lisp-Transpiler, ein
Modul pro Schritt: `loader.py` liest die DB und liefert float32-Matrizen plus
L2-normierte Kopie; `sweep_umap.py` fährt das cuML-Grid (ca. 1 Sekunde pro
Config auf der A4000, Ergebnisse als `.npy` gecacht); `cluster_search.py`
clustert; `validate.py` wertet im Originalraum aus; `run_pilot.py` und
`ablation_width.py` orchestrieren Pilot und Breiten-Ablation; `plot_final.py`
rendert. Alle Treiber akzeptieren `--db` zum Umschalten zwischen Live- und
Compact-DB. Die Umgebung (`.venv`, Pins in `requirements.txt`) nutzt
`cuml-cu12==26.08.00` und `cupy-cuda12x==14.2.0`; 14 Tests sind grün, darunter
Loader-Roundtrip, exakte Row-Counts beider DBs und die Score-Formel. Eine
Warnung sei ehrlich vermerkt: `nn_descent` ist trotz fixem Seed nicht
deterministisch (ein Refit fand 178 statt 167 Cluster) — für bitgenaue
Reproduzierbarkeit müsste man auf `brute_force_knn` wechseln.

## 5. Die Resultate

Auf der Compact-DB wurden 120 von 120 Configs akzeptiert. Die Tabelle zeigt je
Breite den Sieger:

| k | beste Config | Cluster | Noise | sil_orig | adjustiert |
|---|---|---|---|---|---|
| 128 | d16 nn15 md0.1 | 179 | 35,4 % | 0,1889 | 0,1220 |
| 768 | d08 nn15 md0.0 | 226 | 30,8 % | 0,1833 | 0,1269 |
| 3072 | d12 nn30 md0.1 | 167 | 36,5 % | 0,2091 | 0,1327 |

Der Gesamtsieger (`best_params_compact.json`, damit reproduzierbar): volle
3072 Dimensionen, UMAP mit d=12, 30 Nachbarn, `min_dist=0.1`, HDBSCAN. Drei
Befunde stechen hervor. Erstens: **Volle Breite lohnt sich** — 3072 schlägt 768
um +4,6 % und 128 um +8,8 % (adjustierter Score), bei 11 % verworfenen
Kurz-Rows; die Live-DB zeigt dieselbe Reihenfolge. Zweitens: **HDBSCAN deklassiert
DBSCAN** auf dem gesamten Grid (adjustiert ~0,12 vs. ~0,04) — die DBSCAN-eps-
Skala müsste neu kalibriert werden, falls DBSCAN bleiben soll. Drittens: Die
Antwort über d ∈ [4, 16] ist flach (±0,002), d=2 versagt wie vorhergesagt.

## 6. Von Nummern zu Namen: Betitelung und interaktive Karte

167 Cluster-IDs überzeugen niemanden — also haben wir jedem Cluster einen Namen
gegeben. Zwölf Sub-Agenten lasen je ~14 Cluster mit Beispiel-Summaries und
betitelten sie **im Kontrast zu ihren Nachbar-Clustern** (nächste Zentroide im
12-D-Raum), auf Deutsch: vom „Jugglebot-Roboter Bautagebuch" über Schuh-Reviews
bis zur „KI-Blase und Finanzierungsdebatte". Eigene Stichproben vorher bestätigten
kohärente Gruppen (religiöse Vorlesung, Schuh-Reviews, DIY-Bauten sauber
getrennt). Alle 167 Titel liegen in `cluster_titles.json`; zusammen mit
`best_params_compact.json` und den Skripten ist das Clustering reproduzierbar.

Zum Stöbern gibt es `plots/clusters.html` (bewusst **nicht** im Repo: 9 MB):
eine zoombare Plotly-Karte aller Videos, Cluster einzeln filterbar, Hover mit
Titel, Video-Link und Summary-Ausschnitt — im Browser öffnen, hineinzoomen,
loslesen.

## 7. Learnings & offene Punkte

Der 128er-Pilot reihte die Methoden korrekt, unterschätzte aber die absolute
Qualität — Pilot für Tempo, Entscheidung auf voller Breite. Die größte
methodische Falle (Scoring im reduzierten Raum) haben wir umschifft; was fehlt:
ARI-Seed-Stabilität über mehrere Seeds, Trustworthiness-Elbow und
TwoNN/PCA-Plots sowie die DBSCAN-Rekalibrierung. Details zu jedem Schritt stehen
in `plan/20260924_01_cluster_search/{plan,task,ablation}.md` und
`walkthrough.md`; die Rohtabellen in `results_pilot_compact.csv` und
`results_width*.csv`.
