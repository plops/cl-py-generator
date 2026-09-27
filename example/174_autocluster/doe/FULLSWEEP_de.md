Das Original-Dokument leidet unter einem klassischen Problem: Es ist extrem dicht geschrieben, verpackt Erklärungen in verschachtelte Klammern und ertränkt die eigentliche „Geschichte“ des Experiments in Statistik-Jargon.

Hier ist eine komplett überarbeitete Fassung. Sie behält **alle technischen Fakten und Ergebnisse** bei, ist aber so strukturiert, dass sie sich wie ein spannender technischer Bericht liest. Fachbegriffe werden dort erklärt, wo sie auftauchen, und die wichtigsten Erkenntnisse springen sofort ins Auge.

***

# Der volle Sweep: 640 Runs suchen die Überraschung — und finden zwei

**Technischer Bericht | Stand: 27.09.2026**
**Code:** `example/174_autocluster/doe/` | **Hardware:** NVIDIA RTX A4000

---

## 1. Worum geht es? (Die Ausgangslage)

In unserem Projekt gruppieren wir knapp 17.000 automatisch erstellte Video-Zusammenfassungen in thematische Cluster. Dafür nutzen wir eine zweistufige Pipeline:
1. **UMAP** faltet die hochdimensionalen Texte (3072 Dimensionen) in eine kleinere, handhabbare „Landkarte“.
2. **HDBSCAN** sucht auf dieser Karte nach dichten Punktewolken und deklariert sie als Cluster (Themen).

In einer ersten Untersuchungsrunde hatten wir bereits eine scheinbar optimale Einstellung (Konfiguration) für diese Algorithmen gefunden. Doch es blieb ein Zweifel: Unsere damalige Suche war sehr eng gefasst und wir hatten nur wenige Datenpunkte. 

**Die Frage für diesen Report lautete:** Was passiert, wenn wir die Scheuklappen abnehmen? Wenn wir den Suchraum für alle Parameter massiv vergrößern, extrem dicht abtasten (640 Durchläufe) und auch extreme Einstellungen zulassen? Finden wir ein noch besseres Optimum? Oder bricht unsere Pipeline zusammen?

---

## 2. Die 5 Stellschrauben (Parameter)

Wir haben UMAP und HDBSCAN von der Leine gelassen und diese fünf Parameter wild kombiniert:

### UMAP (Die Landkarte)
*   **`d` (Dimensionen):** Auf wie viele Achsen wird die Karte gepresst? (Getestet: 2 bis 24).
*   **`nn` (Nachbarn):** Wie weitsichtig agiert UMAP? `nn=10` achtet auf feinste Details, `nn=80` sieht nur das große Ganze. (Getestet: 10 bis 80).
*   **`md` (Mindestabstand):** Wie eng dürfen Punkte zusammenliegen? `md=0` lässt dichte Klumpen zu, `md=0.4` drückt alles künstlich auseinander. (Getestet: 0.0 bis 0.4).

### HDBSCAN (Der Cluster-Sucher)
*   **`mcs` (Mindest-Clustergröße):** Ab wann ist eine Gruppe ein Thema? (Getestet: 5 bis 60 Punkte).
*   **`ms` (Kernpunkt-Schwelle):** Wie streng prüfen wir, ob ein Punkt wirklich zum harten Kern eines Themas gehört? (Getestet: 3 bis 30).

**Wie haben wir gesucht?**
Statt eines starren Gitters oder purem Zufall haben wir eine **Sobol-Folge** genutzt. Das kann man sich wie einen Profi-Dartspieler vorstellen, der seine Pfeile systematisch und gleichmäßig über die gesamte Zielscheibe verteilt, ohne Lücken zu hinterlassen.

---

## 3. Wie messen wir Erfolg? (Der Score)

Um zu bewerten, wie gut ein Clustering ist, nutzen wir einen Score, der zwei Dinge belohnt:

1. **Trennschärfe (Silhouette):** Liegen die Videos in ihrem eigenen Cluster enger beisammen als zum nächsten Nachbar-Cluster? Dies messen wir knallhart im echten, hochdimensionalen Textraum – nicht auf der verzerrten UMAP-Karte.
2. **Mut zur Lücke (Noise-Strafe):** HDBSCAN darf schwierige Videos als „Rauschen“ (Noise) aussortieren. Wer aber 90 % wegschmeißt, um perfekte Mini-Cluster zu behalten, wird bestraft. Wir multiplizieren die Güte mit dem Anteil der behaltenen Punkte. 

Ein Score um `0.12` ist gut, alles über `0.13` ist exzellent.

---

## 4. Die Ergebnisse: Zwei echte Überraschungen

Wir haben die Ergebnisse der 640 Durchläufe an eine Varianzanalyse (ANOVA) übergeben. Man kann sich diese Statistik wie einen Buchhalter vorstellen, der genau ausrechnet, welcher der 5 Parameter wirklich den Score nach oben oder unten treibt.

Das Ergebnis sah völlig anders aus als in Runde 1.

### Überraschung 1: Der `min_dist`-Abgrund (Cliff)
In Runde 1 dachten wir, der Mindestabstand (`md`) von UMAP sei völlig egal. Jetzt wissen wir: Das lag nur daran, dass wir ihn nicht weit genug aufgedreht haben.

Sobald `md` den Wert von 0.2 überschreitet, stürzt unser Score förmlich in einen Abgrund (statistisch bewiesen durch einen extrem starken Krümmungs-Effekt in den Daten). 
*   **Flaches Land (md 0.0 bis 0.2):** Der Score bleibt stabil bei hervorragenden ~0.121.
*   **Der Abgrund (md > 0.3):** Der Score stürzt auf miserable 0.075 ab. 

**Was passiert dort physikalisch?** Wenn man UMAP zwingt, die Punkte zu weit auseinander zu drücken, schmelzen alle sauberen Themen zu 10 bis 20 diffusen Riesen-Blobs zusammen. Die Cluster-Struktur wird komplett zerstört.

### Überraschung 2: Der gefallene König `min_cluster_size`
In der kleinen Suchbox von Runde 1 war die Mindest-Clustergröße (`mcs`) der wichtigste Hebel überhaupt. In diesem riesigen Sweep geht der Effekt im Rauschen unter. 

Warum? Weil der `min_dist`-Abgrund alles andere dominiert. `mcs` ist kein grober Hebel, um ein kaputtes Clustering zu retten, sondern ein **Feinregler für die absolute Spitze**. Schaut man sich nur die Top-Ergebnisse an, haben alle ein kleines `mcs` (unter 17).

### Die Nicht-Überraschung: Die Dimension `d` ist egal
Egal ob wir die Daten auf 6 oder 24 Dimensionen falten, das Ergebnis bleibt gleich gut. Einzig `d=2` (also eine flache 2D-Karte) ist zu simpel und führt oft zu Abstürzen der Struktur. Ab 6 Dimensionen hat man freie Wahl.

---

## 5. Hat Runde 1 geirrt? (Das Optimum)

Haben wir nach 640 neuen, extremen Versuchen einen besseren Spitzenwert gefunden?
**Nein.**

Der beste neue Parametersatz, den der Sweep ausspuckte, lag bei 0.1337. Er war damit um ein Haar *schlechter* als unser alter Champion aus Runde 1 (0.1343). Mehr noch: Der neue „Sieger“ balancierte gefährlich nah am Rand des `min_dist`-Abgrunds und produzierte instabile Ergebnisse.

**Fazit:** Das alte Optimum (`d=11, nn=45, md=0.09, mcs=11, ms=8`) hat den massiven Stresstest von 640 Durchläufen unbeschadet überstanden. Eine stärkere Bestätigung für diese Einstellungen kann es kaum geben.

---

## 6. Der Methoden-Vergleich (Bake-off)

Wir haben auf unserer besten UMAP-Landkarte nicht nur HDBSCAN getestet, sondern drei weitere bekannte Clustering-Algorithmen losgelassen (jeweils mit ihren besten Einstellungen). Das Ergebnis ist eindeutig:

1. 🥇 **HDBSCAN (Score 0.135):** Bleibt ungeschlagen. Bester Score, extrem stabil auch bei neuen Zufalls-Startwerten (Seeds).
2. 🥈 **Leiden (Score 0.113):** Ein Graph-Algorithmus. Erwirbt den ehrenvollen zweiten Platz. Er sortiert keinen einzigen Punkt als Noise aus, ist aber etwas empfindlicher bei der Feinabstimmung.
3. 🥉 **DBSCAN (Score 0.106):** Langweilig, aber verlässlich. Findet zuverlässig etwas gröbere Cluster.
4. ❌ **Agglomeratives Clustering (Score ~0.03 bis 0.11):** Reine Lotterie. Je nach Startwert schwankt die Qualität extrem. Disqualifiziert.

---

## 7. Zusammenfassung und Regeln für die Praxis

Diese Untersuchung hat uns keinen neuen Rekord-Score gebracht, dafür aber echtes, tiefes **Verständnis**, wo die Gefahren in unserer Pipeline liegen. 

Für den produktiven Einsatz leiten sich daraus drei eiserne Regeln ab:

1. **Regel 1 (Überleben):** UMAPs `min_dist` darf **niemals** größer als 0.2 sein. Alles darüber zerstört die Themen-Struktur unwiderruflich.
2. **Regel 2 (Feintuning):** HDBSCANs `min_cluster_size` sollte klein bleiben (um die 11). Das garantiert feine, saubere Themen.
3. **Regel 3 (Freiheit):** Die UMAP-Dimension `d` kann völlig frei zwischen 6 und 24 gewählt werden.

Wir gehen mit exakt denselben Parametern in die Produktion wie vorher – aber ab heute wissen wir präzise, *warum* sie funktionieren.



## 8. Artefakte und Reproduzierbarkeit

- Code: `doe/run_full.py` (Sweep, Resume, eigener Cache `umap_cache_full/`),
  `doe/run_fullbake.py`, `doe/analyze_full.py`; Design in
  `doe/design.py` (`FULL_BOUNDS`, `generate_sobol_design`,
  `FULL_RESPONSE_FORMULA`); +4 CPU-Tests in `tests/test_doe.py`
  (57/57 grün per `pytest tests/`).
- Daten: `doe/results/design_full.csv`, `results_full{,_agg}.csv`
  (640 Runs), `best_full.json`, `results_fullbake_{bake,stab,winners}.*`,
  `labels_fb_*.npy`, Logs `run_full{,bake}.log`.
- Plots: `doe/plots/full_{sn_vs_mean,mcs_vs_mean,d_vs_mean}.png`,
  `fullbake_bake.png`. Caches per `.gitignore` (`umap_cache*/`) lokal.
- Reproduktion: `.venv/bin/python doe/run_full.py` (~45 min A4000),
  danach `analyze_full.py` (CPU) und `run_fullbake.py` (~5 min).

## 9. Anhang: Abkürzungsverzeichnis

| Kürzel | Bedeutung |
|---|---|
| A4000 | NVIDIA RTX A4000, die verwendete Grafikkarte (16 GB Speicher) |
| Agglo | Kurz für Agglomeratives Clustering (§5) |
| ANOVA | Analysis of Variance (Varianzanalyse); Typ II = jeder Effekt um alle anderen bereinigt (§3.4) |
| ARI | Adjusted Rand Index: 1 = identische Aufteilung, 0 = Zufallsniveau (§3.3) |
| Bake-off | Historischer Name für den Methodenvergleich; lebt nur noch in Code/Dateinamen (`run_fullbake.py`, `fullbake_*.png`) |
| Box | Der untersuchte Parameterbereich, z. B. md[0,0.4] (§3.1) |
| Cache | Zwischenspeicher: fertige UMAP-Einbettungen als `.npy`-Dateien, spart Neuberechnung |
| Community | Dicht vernetzte Gruppe in einem Graphen — Leiden-Sprache für „Cluster" |
| CPU / GPU | Hauptprozessor / Grafikprozessor. UMAP+HDBSCAN laufen auf der GPU (cuML), die Statistik auf der CPU |
| CSV / JSON / PNG / npy | Dateiformate: Tabellen, Einstellungen, Bilder, Numpy-Arrays |
| d, k, md, ms, nn (+ eps, knn, res, linkage, nc, seed) | Parameterkürzel, alle erklärt in §3.1 |
| DBSCAN | Density-Based Spatial Clustering of Applications with Noise (§5) |
| Design-Punkt | Eine Parameterkombination des Versuchsplans (§3.2) |
| DoE | Design of Experiments (statistische Versuchsplanung) |
| F-Wert | ANOVA-Kennzahl: erklärtes Signal gegen Rauschen; groß = starker Effekt (§3.4) |
| Fidelity | Treue der Dimensionsfaltung (§4.3) |
| Grid / Raster | Systematisches Suchraster beim Parameter-Tuning |
| HDBSCAN | Hierarchical Density-Based Spatial Clustering of Applications with Noise (§1) |
| kNN | k-nächste-Nachbarn |
| L2 | Euklidische Vektorlänge (§3.3) |
| Leiden | Community-Erkennungs-Algorithmus aus der Netzwerkanalyse (§1) |
| LLM | Large Language Model (großes Sprachmodell) |
| n | Stichprobenumfang, z. B. n=108 Punkte |
| Noise | Punkte ohne Cluster-Zugehörigkeit (Label −1, §3.3) |
| p-Wert | Irrtumswahrscheinlichkeit; p<0,05 gilt als signifikant (§3.4) |
| pytest | Python-Testprogramm (`pytest tests/` lässt die Test-Suite laufen) |
| R | Zahl der Replikate (Wiederholungsläufe) pro Punkt; hier R=5 |
| R² | Bestimmtheitsmaß: erklärter Streuungsanteil, 1,0 = perfekt (§4) |
| Replikat | Wiederholungslauf derselben Konfiguration mit anderem Seed |
| Response | Die Zielgröße, der Messwert eines Versuchs |
| RSM | Response-Surface-Modell: glatte Fläche durch die Messpunkte (§3.4) |
| Run | Ein Einzellauf: UMAP + Clustering + Scoring |
| S/N | Signal-zu-Rausch-Verhältnis nach Taguchi (§3.3) |
| Seed | Zufalls-Startwert; hier Störgröße, keine Stellschraube (§3.1) |
| Sieger | Siegreichste Konfiguration einer Methode pro Einbettung (§5) |
| Silhouette | Clustergüte pro Punkt, Skala −1…+1 (§3.3) |
| Smoke-Test | Kurzer Funktionstest vor dem Hauptlauf |
| Sobol(-Folge) | Quasi-zufällige, raumfüllende Punktfolge (§3.2) |
| Space-Filling | Raumfüllende Versuchsplanung (§3.2) |
| Sweep | Systematisches Durchlaufen vieler Konfigurationen |
| Taguchi | Genichi Taguchi, Qualitätsingenieur; S/N-Regel für robuste Optima (§3.3) |
| Trustworthiness | Nachbarschaftstreue der Faltung (§4.3) |
| Tuning | Feineinstellen von Parametern per Suchraster |
| UMAP | Uniform Manifold Approximation and Projection (§1) |
| Ward / average | Fusionsregeln beim agglomerativen Clustern (§3.1) |
