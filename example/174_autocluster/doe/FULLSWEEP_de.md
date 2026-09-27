# Der volle Sweep: 640 Runs suchen die Überraschung — und finden zwei

Technischer Bericht, Stand 2026-09-27 · Code: `example/174_autocluster/doe/`
(`run_full.py`, `run_fullbake.py`, `analyze_full.py`) · GPU: NVIDIA RTX A4000

## 1. Worum es geht: Vorgeschichte und Anlass

Wer diesen Bericht liest, kennt idealerweise die Vorgeschichte in zwei
Sätzen: Wir clustern knapp 17.000 Video-Zusammenfassungen, indem wir ihre
Embeddings per UMAP in wenige Dimensionen falten und dort per HDBSCAN
Themen suchen. Die Qualität messen wir bewusst nicht im gefalteten Raum,
sondern im hochdimensionalen Originalraum — sonst würde man
Projektionsartefakte mit Themen verwechseln.

Runde 1 dieser Untersuchung (Latin Hypercube, 60 Punkte × 3 Seeds, enge
Parameterbox) hatte ein klares Bild geliefert: HDBSCANs
`min_cluster_size` (Mindest-Clustergröße) ist der dominante Hebel
(p=0,001), die UMAP-Zieldimension `d` (`n_components`) und UMAPs
`min_dist` (Mindestabstand) sind egal (p=0,27 bzw. 0,56), das Optimum
liegt bei d=11, nn=45 (`n_neighbors`), md=0,09, mcs=11, ms=8
(`min_samples`) mit Score 0,1343 (alle Kürzel erklärt in §3.1).
Phase A und B hatten
nachgelegt: Dubletten nicht entfernen (−0,028 sonst!), HDBSCAN als
Produktionsmethode bestätigt, Leiden wegen Seed-Instabilität verworfen.

Doch bei der Diskussion blieb ein Unbehagen: 60 Punkte in fünf Dimensionen
sind dünn — umgerechnet etwa zwei Stützstellen pro Achse. Die Streuung wurde
aus nur drei Replikaten geschätzt. Und die Box war an allen Rändern zu:
Was jenseits von md=0,25 oder mcs=40 liegt, hatte niemand je gesehen. Kurz:
War das schöne Bild vielleicht nur ein Artefakt des kleinen Fensters? Dieser
Sweep sollte das Fenster aufreißen — doppelte Dichte, fünffache Replikation,
offene Ränder — und nachsehen, ob draußen eine Überraschung wartet.

## 2. Scope: was dieser Sweep abdeckt — und was nicht

**Drin:**

- Ein dichter Space-Filling-Sweep über den vollen UMAP+HDBSCAN-Raum auf der
  Schnittmenge N=16.692 (k=3072): 128 Sobol-Punkte × 5 Seeds = 640 Runs,
  erweiterte Box d[2,24] × nn[10,80] × md[0,0.4] × mcs[5,60] × ms[3,30]
  (Zieldimension × Nachbarschaft × Mindestabstand × Mindest-Clustergröße ×
  Kernpunkt-Schwelle — volle Namen und Anschauung in §3.1).
- Eine Response-Surface-Analyse mit allen fünf quadratischen Termen
  (Runde 1 hatte nur drei — md² und ms² fehlten per Annahme).
- Ein Methoden-Bake-off auf **drei** Einbettungen (d=8/11/16) statt einer,
  inklusive Winner-Stabilität über Seeds. Die d=11-Schiene läuft mit
  frischem Cache und ist damit zugleich eine echte Replikation von Phase B.

**Draußen (bewusst):** Die Embedding-Breite k bleibt fix (der faire
Breitenvergleich aus Runde 1 wird nicht wiederholt), Dubletten bleiben drin
(Phase A hat das entschieden), Wort-Kohärenz und menschliches Rating werden
nicht wiederholt (Phase-B-Nullresultat gilt weiter). Fixiert bleiben auch
Kosinus-Metrik, `nn_descent`-Aufbau, UMAP-Lernrate/Epochen und die
HDBSCAN-Selektionsmethode — ein Sweep kann nicht alles variieren, und diese
Achsen waren nie strittig.

## 3. Methode

### 3.1 Die Parameter: volle Namen und Anschauung

Der Sweep variiert fünf Stellschrauben plus eine Störgröße. Die Kürzel
(d, nn, md, mcs, ms) stehen im ganzen Bericht für diese vollen
Bibliotheks-Parameter:

| Kürzel | Voller Name                                       | Was es steuert (anschaulich)                                                                                                                                                                                                  | Sweep-Box |
|--------+---------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------|
| k      | Embedding-Breite (Matryoshka-Prefix-Länge)        | Wie viele der 3072 Zahlen pro Vektor verwendet werden — volle Breite = volles Wissen                                                                                                                                          | fix 3072  |
| d      | UMAP `n_components` (Zieldimension)               | Auf wie viele Achsen die „Landkarte" gepresst wird: d=2 ist ein Stadtplan (vieles überlappt), d=24 ein Atlas mit Reserve                                                                                                      | [2, 24]   |
| nn     | UMAP `n_neighbors` (Nachbarschaftsgröße)          | Aus wie vielen nächsten Nachbarn UMAP die lokale Geometrie schätzt: nn=10 ist kurzsichtig (feine Verästelung, aber zerfasert), nn=80 weitsichtig (ruhig, aber Details gehen unter)                                            | [10, 80]  |
| md     | UMAP `min_dist` (Mindestabstand)                  | Wie eng Punkte in der Einbettung liegen dürfen: md=0 presst Klumpen (dicht, aber ehrlich), md=0,4 drückt alles auseinander — bis die Struktur schmilzt (der Cliff aus §4.1!)                                                  | [0, 0,4]  |
| mcs    | HDBSCAN `min_cluster_size` (Mindest-Clustergröße) | Ab wie vielen Mitgliedern eine Gruppe als Cluster gilt: mcs=5 findet jedes Grüppchen (~450 Cluster, fragmentiert), mcs=60 lässt nur Großthemen überleben                                                                      | [5, 60]   |
| ms     | HDBSCAN `min_samples` (Kernpunkt-Schwelle)        | Wie viele Nachbarn ein Punkt braucht, um Kern- statt Randpunkt zu sein: ms=3 ist gutgläubig (wenig Noise), ms=30 misstrauisch (vieles wird Noise, Reste sind felsenfest)                                                      | [3, 30]   |
| seed   | `random_state` (Zufalls-Startwert)                | **Störgröße**, keine Stellschraube: Startwert des Zufallsgenerators. cuMLs `nn_descent` ist trotz fixem Seed nicht bit-identisch (GPU-Wettläufe beim Graphenbau) — daher je Punkt 5 Replikate mit Seeds 42, 1337, 2026, 7, 99 | R=5       |

Nur im Bake-off (§3.5) kommen hinzu — jeweils mit eigenem Mini-Tuning pro
Einbettung:

| Kürzel       | Voller Name                               | Anschauung                                                                                                                                                        |
|--------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| eps          | DBSCAN `eps` (Kugelradius)                | Punkte innerhalb dieser Distanz gelten als Nachbarn: eps=0,1 ist pingelig, eps=0,5 großzügig                                                                      |
| ms           | DBSCAN `min_samples`                      | Mindestpunkte in der eps-Kugel für einen Kernpunkt (gleiche Rolle wie bei HDBSCAN)                                                                                |
| knn          | Leiden kNN-Graph-k                        | Jede Node wird mit ihren k nächsten Nachbarn verdrahtet; auf diesem Graphen sucht Leiden Communities                                                              |
| res          | Leiden `resolution_parameter` (Auflösung) | Feinregler der Community-Größe: res=0,02 wenige Riesen-Communities, res=8,0 viele Kleinst-Communities                                                             |
| linkage / nc | Agglomerative `linkage` / `n_clusters`    | Fusionsregel (`ward` = varianzminimal, `average` = mittlere Distanz) plus **erzwungene** Clusterzahl — Zwangspartitionierung ohne Noise-Ausgang (0 % Noise immer) |

### 3.2 Der Versuchsplan: Sobol statt Gitter und Zufall

Ein Gitter (jede Achse in Stufen) explodiert kombinatorisch: 5 Stufen auf
5 Achsen sind 3.125 Punkte — unbezahlbar. Reiner Zufall dagegen klumpt:
Man kennt das vom Dartpfeil — zufällige Würfe lassen Löcher und Häufungen.
Eine **Sobol-Folge** ist der Mittelweg: eine deterministische,
quasi-zufällige Punktfolge, die den Raum systematisch gleichmäßig füllt
(„low discrepancy"), ohne Gitter-Zwang. Zwei praktische Vorteile gaben den
Ausschlag: Erstens füllt Sobol auch alle zweidimensionalen Projektionen
(z. B. mcs×ms) sauber — wichtig, weil wir gerade Interaktionen suchen.
Zweitens ist die Folge **erweiterbar**: Die ersten 2 Punkte eines
128er-Plans sind identisch mit einem 2er-Plan. Unser GPU-Smoke-Test
(2 Punkte × 1 Seed) war dadurch automatisch das echte Präfix des
Hauptlaufs — kein Wegwerf-Code, sondern die ersten beiden Messwerte.

Konkret: `scipy.stats.qmc.Sobol(d=5, scramble=True, seed=20260927)`,
128 Punkte (`random_base2(m=7)`), vektorskaliert auf die Box, ganzzahlige
Faktoren gerundet. Prüfung vorab: alle 23 d-Werte 2–24 getroffen, keine
exakten Duplikate nach Rundung, Min/Max an allen Bounds.

### 3.3 Die Zielgröße (unverändert aus Runde 1)

Pro Run: `adjusted = Silhouette_orig × (1 − NoiseRatio)`. Die Silhouette
wird auf den L2-normierten Originalvektoren (≙ Kosinus) ohne Noise-Punkte
gerechnet; der Faktor (1−Noise) bestraft Abstention. Akzeptanz nur bei
Noise ≤ 40 % und ≥ 2 Clustern, sonst ist der Run verworfen. Pro Design-Punkt
werden die Replikate aggregiert — jetzt aus bis zu fünf statt drei Werten:
Mittelwert `mean_adj` (Peak-Kriterium), Taguchi-S/N `η = −10·log10(mean(1/s²))`
(Robust-Kriterium: hoch **und** streuungsarm gewinnt) und paarweiser ARI
über Seeds (Label-Stabilität). Mindestens zwei akzeptierte Replikate sind
Pflicht, sonst fällt der ganze Punkt aus der Analyse.

### 3.4 Die Statistik: Response Surface + ANOVA

Auf die aggregierten Punkte fitten wir ein quadratisches Regressionsmodell
mit allen 2-Wege-Interaktionen (21 Koeffizienten auf n=108 — etwa fünf
Beobachtungen pro Parameter, mehr als doppelt so komfortabel wie Runde 1).
Die ANOVA-Tabelle (Typ II) zerlegt dann die erklärte Varianz: Ein Faktor mit
p < 0,05 bewegt den Score systematisch, alles andere ist im Rauschen
untergegangen. Man stelle sich die ANOVA als Buchhalter vor, der jeden
Erklärungsanteil genau einem Faktor gutschreibt — und laut meldet, wenn eine
Achse nichts beizutragen hat.

### 3.5 Der Bake-off auf drei Einbettungen

Phase B hatte vier Methoden auf einer einzigen Einbettung antreten lassen —
und prompt einen Schein-Sieger gekürt (Leiden kollabierte auf anderen
Seeds). Jetzt fixieren wir nn=45/md=0,09 (die Winner-Geometrie) und
wiederholen den kompletten Bake-off (gleiches Tuning-Grid: 15+12+20+8
Runs) auf d=8, d=11 und d=16, jeweils mit Winner-Stabilität auf zwei
weiteren Seeds. Der Clou: Der Cache-Tag ist neu (`fullbake_*`), die
d=11-Einbettung wird also **neu gerechnet statt geladen**. Stimmt das
Ergebnis trotzdem mit Phase B überein, ist das eine echte Replikation über
Prozessgrenzen hinweg — inklusive `nn_descent`-Jitter.

### 3.6 Rechenaufwand

Median ~2,5 s pro Run (UMAP+HDBSCAN+Scoring auf der A4000), 640 Runs in
44 Minuten — dank Resume-Logik (fertige Punkte werden per gecachter
Einbettung label-identisch rekonstruiert) und inkrementeller CSV
abrupt-sicher. Der Bake-off (165 Tuning- + 24 Stab-Runs auf 9 Einbettungen)
kostete weitere ~5 Minuten. Die Auswertung (RSM, Plots) ist CPU-only.

## 4. Ergebnisse Screening: zwei Überraschungen, kein neuer Peak

Überblick: 492/640 Runs akzeptiert, 108/128 Punkte aggregiert (20 fielen mit
< 2 Replikaten aus), R²=0,85. Die ANOVA-Tabelle zeigt ein völlig anderes
Gesicht als Runde 1 — der Reihe nach.

### 4.1 Überraschung 1: min_dist beherrscht alles — als Cliff

Der quadratische Term `I(md²)` schießt auf F=53,9 (p<1e-10) — mit Abstand
der stärkste Effekt des ganzen Sweeps. Zum Vergleich: In Runde 1 war md mit
p=0,56 der insignifikanteste Faktor überhaupt. Gebinnt man die Scores, sieht
man warum: md≤0,2 → ~0,121 (flach), 0,2–0,3 → 0,106, >0,3 → 0,075. Die alte
Box [0,0.25] zeigte nur das flache Stück; der Abgrund lag hinter der Kante.

Zwei konkrete Punkte machen den Cliff greifbar. Punkt 106
(d=19, nn=45, md=0,07, mcs=15): 185 Cluster, Noise 34 %, Score 0,130 —
ein typischer guter Lauf. Punkt 79 (d=6, nn=79, md=0,32, mcs=34): **10
Cluster, Noise 0,1 %, Score 0,07** — und beide Replikate sind sich
vollkommen einig (ARI 1,0). Das ist die Pointe: Hohes md erzeugt kein
Rauschen, sondern schmilzt alles zu wenigen diffusen Riesen-Blobs ein
(Silhouette 0,19→0,08, Cluster ~105→21). Jenseits des Cliffs ist es
*stabil schlecht* (127/160 Runs akzeptiert). Die Übergangszone 0,2–0,3
dagegen ist tödlich-instabil: 68/160 Runs sterben dort an Noise>40 % —
etwa Punkt 23 (d=3, nn=59, md=0,25, mcs=34, ms=29) mit 42 % Noise.

### 4.2 Überraschung 2: Der min_cluster_size-Haupteffekt löst sich auf

Runde 1 kürte `min_cluster_size` zum König (F=12, p=0,001). Jetzt: p=0,17,
nicht signifikant. Heißt das, Runde 1 lag falsch? Nein — aber der Effekt
ist kleiner als gedacht und geht in der weiten Box unter: Über [5,60]
fällt der Score nur seicht von 0,118 auf 0,099, während der md-Cliff die
Varianz dominiert. Dass mcs trotzdem am Peak entscheidet, sieht man an den
Top-5: alle haben mcs≤17. Statistisch überlebt nur die Wechselwirkung
`nn:mcs` (p=0,008, wie in Runde 1), neu hinzu kommen `nn:ms` (p=0,048) und
`md:ms` (p=0,050) — `min_samples` wirkt also nur im Zusammenspiel, nie
allein (p=0,78). Die Lehre: mcs ist ein Feinregler für die Spitze, kein
Grobregler für die Landschaft.

### 4.3 Die UMAP-Dimension bleibt egal — über [2,24]

`d` (p=0,56) und `d²` (p=0,60) sind insignifikant wie eh und je — jetzt auf
mehr als doppelter Spannweite nachgewiesen. Die binnten Mittelwerte pro d
liegen alle zwischen 0,09 und 0,12 ohne Trend. Eine Nuance gibt es doch:
d=2 ist fragil (nur 2 Überlebende), d≥3 unauffällig. Zusammen mit der
Fidelity-Analyse (Trustworthiness sättigt ab d≈6) heißt das: Die
Dimensionswahl ist im Bereich 6–24 praktisch frei; wer d=2 nimmt, spielt
mit dem Feuer, gewinnt aber nichts.

### 4.4 Das alte Optimum steht — der neue „Sieger" ist fragil

Neuer Robuster = neuer Peak: Punkt 2 (d=23, nn=15, md=0,31, mcs=6, ms=9)
mit 0,1337 — also −0,0006 *unter* dem alten Optimum 0,1343, mitten im
Jitter (σ≈0,001–0,002). Mehr noch: Der Punkt hat nur 3/5 Replikate, Noise
39 % (knapp an der 40-%-Kante) und md=0,31 — mitten im Cliff, wo ihn nur
das winzige mcs=6 rettet. Das ist kein Fundament, das ist ein
Drahtseilakt. Nach 640 Runs auf doppelter Dichte mit offenen Rändern steht
das Runde-1-Optimum (d=11, nn=45, md=0,09, mcs=11, ms=8) ungeschlagen da —
das ist die stärkste Bestätigung, die ein DoE-Optimum bekommen kann.

## 5. Ergebnisse Bake-off: Ranking stabil, zwei Methoden-Geschichten

| Emb | HDBSCAN                    | Leiden        | DBSCAN | Agglo                          |
|-----+----------------------------+---------------+--------+--------------------------------|
| d08 | **0,1349** [0,1330,0,1349] | 0,1121 stabil | 0,1067 | 0,0590 peaky                   |
| d11 | **0,1351** [0,1325,0,1351] | 0,1124 stabil | 0,1052 | 0,1160 (!) peaky [0,028,0,116] |
| d16 | **0,1324** [0,1324,0,1351] | 0,1132 stabil | 0,1071 | 0,1161 (!) peaky [0,027,0,117] |

(Spanne = Winner über 3 Seeds; ARI: DBSCAN 0,79–0,82 am label-stabilsten.)

**HDBSCAN gewinnt überall**, Leiden ist überall Zweiter, DBSCAN überall
Dritter — das Ranking hängt nicht an d. Die **Replikation** gelingt:
d=11 frisch gerechnet ergibt 0,1351 (mcs=11, ms=5) gegen 0,1333 (ms=8) in
Phase B — plus 0,0018 im Jitter, mcs=11 erneut bestätigt, über
Prozessgrenzen und `nn_descent`-Jitter hinweg.

**Leiden ist rehabilitiert — mit Stern.** An Resolution 1,5–3,0 läuft es
auf allen drei Einbettungen felsenstabil (±0,001). Die Phase-B-Diagnose
„peaky" (0,115→0,06) war also keine Methoden-, sondern eine
Resolutionsschwäche: res=5,0 baut fragile Fein-Partitionen. Für die Praxis
ändert das wenig — 0,112 bleibt Zweiter —, aber es korrigiert das Urteil:
Leiden ist stabiler Zweiter, nicht unberechenbar.

**Agglo ist Lotterie.** 0,116 auf d11/d16 sieht nach Verfolger aus —
gegen 0,031 in Phase B auf nominell gleicher Geometrie (d=11, Seed 42,
nur anderer Prozess)! Die Seed-Spanne [0,03,0,12] enthüllt den Schein:
Agglo gewinnt mal, kollabiert mal, je nach UMAP-Jitter. Der Fall ist das
schönste Argument dieser Untersuchung dafür, dass kein Bake-off ohne
Stabilitäts-Check zählen darf. DBSCANs Winner (eps=0,2, ms=25) ist
derweil auf allen drei Einbettungen buchstäblich identisch — langweilig im
besten Sinne.

## 6. Diskussion

### 6.1 Der Fenstereffekt: Warum Befunde zwischen Runden wandern

Der scheinbare Widerspruch — md unwichtig→dominant, mcs dominant→schwach —
ist kein Fehler, sondern Lehrstück: **Jeder DoE-Befund gilt nur für sein
Fenster.** Runde 1 schaute durch [0,0.25] und sah md flach; der Sweep
öffnet bis 0,4 und findet den Abgrund dahinter. Umgekehrt lebte der
mcs-Effekt vom Kontrast innerhalb [10,40]; in [5,60] bei gleichzeitigem
md-Cliff schrumpft sein Varianzanteil unter die Signifikanz. Man stelle
sich eine Taschenlampe in dunkler Landschaft vor: Was man sieht, hängt vom
Kegel ab — der Sweep hat den Kegel verbreitert, nicht die Landschaft
verändert. Die praktische Regel daraus: Haupteffekte immer mit ihrer Box
zitieren („mcs dominiert in [10,40] × md≤0,25").

### 6.2 Was das für die Praxis heißt

Drei Regeln, in Prioritätsordnung: **Erstens, md≤0,2 erzwingen.** Der Cliff
ist der einzige Fehler, der alles ruiniert — 0,121 gegen 0,075 ist kein
Tuning, das ist heil gegen kaputt. **Zweitens, mcs klein halten** (um 11),
aber keine Wunder erwarten: Der Unterschied zwischen mcs=8 und mcs=20 ist
real, aber klein. **Drittens, d frei wählen** (6–24), HDBSCAN behalten,
DBSCAN als ehrlichen Zweiten für gröbere Cluster. Der Produktions-Parametersatz
ändert sich nicht — aber jetzt wissen wir, *warum* er funktioniert und wo
die Klippen liegen.

### 6.3 Grenzen und offene Flanken

Ehrlichkeit verlangt die Schwächenliste. **Erstens** analysiert die ANOVA
nur Überlebende (108/128): Wer an Noise>40 % stirbt, hinterlässt keine
Response — der Cliff jenseits 0,3 ist also eher unter- als überschätzt.
**Zweitens** bleibt S/N aus fünf Werten unsicher; das Robust-Ranking kann
selbst rauschen (der fragile Punkt 2 ist der Beleg). **Drittens** fixiert
der Bake-off nn/md — das Ranking gilt streng nur für diese Geometrie, auch
wenn drei d-Stufen beruhigen. **Viertens** ein DB-Snapshot, keine
zeitlichen Splits, ein Rater, keine Wort-Kohärenz-Wiederholung. Und
**fünftens**: 128 Punkte sind besser als 60, aber für fünf Dimensionen
immer noch Screening, keine Vermessung — Drei-Wege-Interaktionen und feine
Krümmungen bleiben unsichtbar.

### 6.4 Was als Nächstes lohnte

Am meisten Erkenntnis pro GPU-Minute verspricht ein **Cliff-Experiment**:
md fein gestaffelt (0,15–0,35) × mcs klein/groß × 10 Seeds — wo genau
kippt Over-Merging in Noise-Tod, und rettet klein-mcs immer? Danach ein
`brute_force_knn`-Determinismus-Anker (3 Punkte × 2 Seeds) als
Reproduzierbarkeits-Eichstrich. Und irgendwann der Produktivitäts-Schritt:
LLM-Rating gegen Clusterzahl, um die Fragmentierungsfrage (217 vs. 465
Cluster aus Phase A) von außen zu entscheiden.

## 7. Conclusion

Der Sweep hielt, was sein Anlass versprach — nur anders als erhofft. Statt
eines besseren Peaks lieferte er **besseres Verständnis**: einen
md-Cliff, den niemand vermutet hatte; einen mcs-Effekt, der kleiner ist
als gedacht; ein d, das endgültig egal ist; ein altes Optimum, das 640
Runs überlebt; ein rehabilitiertes Leiden und ein entlarvtes Agglo. Die
Überraschung war nicht der Fund, sondern die Landschaft. Für die Produktion
ändert sich kein Parameter — aber jede Zeile der Begründung dahinter ist
jetzt belastbar.

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
