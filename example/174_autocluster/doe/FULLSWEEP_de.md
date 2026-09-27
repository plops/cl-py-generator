# Der volle Sweep: 640 Runs suchen die Überraschung — und finden zwei

Technischer Bericht, Stand 2026-09-27 · Code: `example/174_autocluster/doe/`
(`run_full.py`, `run_fullbake.py`, `analyze_full.py`) · GPU: NVIDIA RTX A4000

## 1. Worum es geht: Vorgeschichte und Anlass

Wer diesen Bericht liest, kennt idealerweise die Vorgeschichte in zwei
Sätzen: Wir clustern knapp 17.000 Video-Zusammenfassungen, indem wir ihre
Embeddings per UMAP (Uniform Manifold Approximation and Projection, ein
Verfahren zur Dimensionsreduktion) in wenige Dimensionen falten und dort
per HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
with Noise, ein dichtebasiertes Clustering-Verfahren) Themen suchen. Die Qualität messen wir bewusst nicht im gefalteten Raum,
sondern im hochdimensionalen Originalraum — sonst würde man
Projektionsartefakte mit Themen verwechseln.

Runde 1 dieser Untersuchung (Latin Hypercube, 60 Punkte × 3 Seeds, enge
Parameterbox) hatte ein klares Bild geliefert: HDBSCANs
`min_cluster_size` (Mindest-Clustergröße) ist der dominante Hebel
(p-Wert 0,001 — was das heißt, steht in §3.4), die UMAP-Zieldimension
`d` (`n_components`) und UMAPs
`min_dist` (Mindestabstand) sind egal (p=0,27 bzw. 0,56), das Optimum
liegt bei d=11, nn=45 (`n_neighbors`), md=0,09, mcs=11, ms=8
(`min_samples`) mit Score 0,1343 (alle Kürzel erklärt in §3.1).
Phase A und B hatten
nachgelegt: Dubletten nicht entfernen (−0,028 sonst!), HDBSCAN als
Produktionsmethode bestätigt, Leiden (ein Community-Erkennungs-Algorithmus
aus der Netzwerkanalyse) wegen Seed-Instabilität verworfen.

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
- Ein Methodenvergleich auf **drei** Einbettungen (d=8/11/16) statt einer,
  inklusive Sieger-Stabilität über Seeds. Die d=11-Schiene läuft mit
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

Nur im Methodenvergleich (§3.5) kommen hinzu — jeweils mit eigenem
Mini-Tuning pro Einbettung:

| Kürzel       | Voller Name                               | Anschauung                                                                                                                                                        |
|--------------+-------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| eps          | DBSCAN `eps` (Kugelradius)                | Punkte innerhalb dieser Distanz gelten als Nachbarn: eps=0,1 ist pingelig, eps=0,5 großzügig                                                                      |
| ms           | DBSCAN `min_samples`                      | Mindestpunkte in der eps-Kugel für einen Kernpunkt (gleiche Rolle wie bei HDBSCAN)                                                                                |
| knn          | Leiden kNN-Graph-k (k-nächste-Nachbarn)     | Jede Node wird mit ihren k nächsten Nachbarn verdrahtet; auf diesem Graphen sucht Leiden Communities (dicht vernetzte Gruppen)                                    |
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

Die Bewertung läuft in vier Schritten — vom einzelnen Punkt bis zum
Design-Punkt:

**Schritt 1: Wie gut sitzt jeder Punkt in seinem Cluster? (Silhouette).**
Für jeden Punkt stellen wir zwei Fragen: Wie nah sind im Schnitt die
anderen Mitglieder des *eigenen* Clusters (Abstand a)? Und wie nah ist im
Schnitt das *nächstgelegene fremde* Cluster (Abstand b)? Sitzt der Punkt
mitten in seiner Gruppe (a klein) und weit weg von der Konkurrenz (b
groß), bekommt er fast +1. Hockt er genau an der Grenze (a ≈ b), gibt es
~0. Wäre er im Nachbarcluster besser aufgehoben (a > b), wird der Wert
negativ bis −1. Die Formel `s = (b−a)/max(a,b)` fasst genau das zusammen;
der Mittelwert über alle Punkte ist der Silhouette-Score. Zum Einordnen:
s=0,19 (unsere guten Läufe) heißt „Punkte liegen im Schnitt deutlich näher
bei ihrer eigenen Gruppe als bei der Konkurrenz" — kein perfektes +1, aber
bei 17.000 echten Text-Embeddings mit fließenden Themengrenzen ein solider
Wert.

Gemessen werden diese Abstände an den L2-normierten Originalvektoren
(L2 = euklidische Vektorlänge; auf Einheitslänge normiert ≙ mathematisch
äquivalent zur Kosinus-Distanz), **nicht** in der gefalteten UMAP-Karte —
sonst würde man UMAPs Zeichenstil statt Themenqualität benoten.
Noise-Punkte (−1) zählen nicht mit: Wer keinem Cluster angehört, kann auch
keines bewerten.

**Schritt 2: Sich-Drücken verbieten (Noise-Strafe).** HDBSCAN darf
schwierige Punkte als Noise aussortieren — praktisch, aber eine Einladung
zum Schummeln: Wer 90 % der Punkte wegwirft, behält einen winzigen,
super-engen Restkern mit Traum-Silhouette. Deshalb multiplizieren wir mit
dem Anteil der *behaltenen* Punkte: `adjusted = Silhouette_orig ×
(1 − NoiseRatio)`. Bei 34 % Noise zählen nur 66 % der Silhouette — wer
sich vor den schwierigen Punkten drückt, zahlt dafür.

**Schritt 3: Unbrauchbares sofort verwerfen.** Noise über 40 % oder weniger
als 2 Cluster → der Run ist ungültig und bekommt gar keinen Score. (Solche
Läufe fehlen in der Statistik ersatzlos — siehe Diskussion §6.3.)

**Schritt 4: Über Seeds mitteln.** Pro Design-Punkt laufen 5 Replikate
(Seeds 42, 1337, 2026, 7, 99), daraus drei Kennzahlen: Mittelwert
`mean_adj` (Peak-Kriterium: wie gut im Schnitt?), Taguchi-S/N
`η = −10·log10(mean(1/s²))` (S/N = Signal-zu-Rausch-Verhältnis, nach dem
Qualitätsingenieur Genichi Taguchi; Robust-Kriterium: hoch **und**
streuungsarm gewinnt) und paarweiser ARI über Seeds (ARI = Adjusted Rand
Index: 1 = identische Aufteilung (Partition), 0 = Zufallsniveau; misst
Label-Stabilität — landen dieselben Punkte über Seeds in denselben
Clustern?). Mindestens zwei gültige Replikate (Wiederholungsläufe mit
anderem Seed) sind Pflicht, sonst fällt der ganze Punkt aus der Analyse.

### 3.4 Die Statistik: Response Surface + ANOVA (Varianzanalyse)

Auf die aggregierten Punkte fitten wir ein quadratisches Regressionsmodell
(Response-Surface-Modell: eine glatte Fläche durch die Messpunkte) mit
allen 2-Wege-Interaktionen (Paar-Wechselwirkungen wie nn:mcs — also der
Frage, ob zwei Parameter gemeinsam anders wirken als jeder für sich).
21 Koeffizienten auf n=108 Punkte (n = Stichprobenumfang): etwa fünf
Beobachtungen pro Parameter, mehr als doppelt so komfortabel wie Runde 1.
Die ANOVA-Tabelle (ANOVA = Analysis of Variance, Varianzanalyse; Typ II =
jeder Effekt wird um alle anderen bereinigt) zerlegt dann die erklärte
Streuung (Varianz) in Erklärungsanteile pro Faktor. Zwei Zahlen zählen:
der F-Wert (Signal gegen Rauschen — groß heißt starker Effekt) und der
p-Wert (Irrtumswahrscheinlichkeit). Faustregel: p < 0,05 (unter 5 %) →
der Faktor bewegt den Score systematisch; alles andere ist im Rauschen
untergegangen. Man stelle sich die ANOVA als Buchhalter vor, der jeden
Erklärungsanteil genau einem Faktor gutschreibt — und laut meldet, wenn eine
Achse nichts beizutragen hat.

### 3.5 Der Methodenvergleich auf drei Einbettungen

Phase B hatte vier Methoden auf einer einzigen Einbettung antreten lassen —
und prompt einen Schein-Sieger gekürt (Leiden kollabierte auf anderen
Seeds). Jetzt fixieren wir nn=45/md=0,09 (die Sieger-Geometrie: die
Geometrie der siegreichen Runde-1-Konfiguration) und wiederholen den
kompletten Methodenvergleich (jede Methode mit eigenem Tuning auf derselben
Einbettung, gleiches Tuning-Raster: 15+12+20+8 Läufe) auf d=8, d=11 und
d=16, jeweils mit Sieger-Stabilität auf zwei weiteren Seeds. Der Clou: Der
Cache-Tag ist neu (`fullbake_*`), die d=11-Einbettung wird also **neu
gerechnet statt geladen**. Stimmt das Ergebnis trotzdem mit Phase B
überein, ist das eine echte Replikation über Prozessgrenzen hinweg —
inklusive `nn_descent`-Jitter. (Code und Dateinamen tragen aus historischen
Gründen noch das Kürzel `bake`, z. B. `run_fullbake.py`.)

### 3.6 Rechenaufwand

Median ~2,5 s pro Run (UMAP+HDBSCAN+Scoring auf der A4000), 640 Runs in
44 Minuten — dank Resume-Logik (fertige Punkte werden per gecachter
Einbettung label-identisch rekonstruiert) und inkrementeller CSV
abrupt-sicher. Der Methodenvergleich (165 Tuning- + 24 Stabilitätsläufe
auf 9 Einbettungen) kostete weitere ~5 Minuten. Die Auswertung
(RSM = Response-Surface-Modell, Plots) ist CPU-only (läuft auf dem
normalen Prozessor, ohne Grafikkarte).

## 4. Ergebnisse Screening: zwei Überraschungen, kein neuer Peak

Überblick: 492/640 Runs akzeptiert, 108/128 Punkte aggregiert (20 fielen mit
< 2 Replikaten aus), R²=0,85 (R² = Bestimmtheitsmaß: Das Modell erklärt
85 % der Streuung; 1,0 wäre perfekt). Die ANOVA-Tabelle zeigt ein völlig
anderes Gesicht als Runde 1 — der Reihe nach.

### 4.1 Überraschung 1: min_dist beherrscht alles — als Cliff

Der mit Abstand stärkste Effekt des ganzen Sweeps geht auf `min_dist`
zurück — aber nicht als Gerade, sondern als Kurve. Das Modell enthält
nämlich neben md selbst auch md² (min_dist zum Quadrat). Wozu der
Quadratschnickschnack? Eine Gerade kann nur „steigt" oder „fällt"; erst
der quadratische Term erlaubt Krümmung — also Verläufe wie „erst flach,
dann Abfall". Dass ausgerechnet dieser Krümmungs-Term hochsignifikant ist,
heißt also: Der Zusammenhang zwischen md und Score ist gekrümmt, keine
Gerade.

Die Zahlen dazu: F=53,9 — der F-Wert misst Signal gegen Rauschen (je
größer, desto stärker; der nächststärkste Effekt liegt bei F≈8). Und
p<1e-10 — der p-Wert ist die Irrtumswahrscheinlichkeit, hier kleiner als
eins zu zehn Milliarden: praktisch sicher kein Zufall. (Beide Begriffe
erklärt §3.4.) Zum Vergleich: In Runde 1 hatte md einen p-Wert von 0,56,
also 56 % Irrtumswahrscheinlichkeit — meilenweit über der 5-%-Hürde
(p<0,05), damit der am klarsten wirkungslose Faktor überhaupt.

Gruppiert man die 108 Messpunkte nach md-Bereichen, sieht man die Kurve
mit bloßem Auge: md≤0,2 → Score ~0,121; 0,2–0,3 → 0,106; >0,3 → 0,075.
Zur Einordnung der Scores (adjusted = Silhouette × behaltene Punkte,
§3.3): In dieser Untersuchung bedeutet ~0,12 „gut", ~0,134 „Spitze" und
~0,07 „mäßig". „Flach" heißt also: Zwischen md=0 und md=0,2 ändert sich
fast nichts (0,121 bleibt 0,121). „Abgrund" heißt: Dahinter bricht der
Score um fast 40 % ein (0,121 → 0,075). Die alte Box [0,0.25] reichte nur
bis an den Beginn des Abfalls — sie zeigte das flache Stück plus einen
Zipfel der Kante; der eigentliche Absturz dahinter blieb unsichtbar.

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
mehr als doppelter Spannweite nachgewiesen. Die nach d gruppierten Mittelwerte
liegen alle zwischen 0,09 und 0,12 ohne Trend. Eine Nuance gibt es doch:
d=2 ist fragil (nur 2 Überlebende), d≥3 unauffällig. Zusammen mit der
Fidelity-Analyse aus dem Vorlauf (Fidelity = Treue der Dimensionsfaltung;
Trustworthiness = Nachbarschaftstreue — bleiben die k nächsten Nachbarn
nach dem Falten Nachbarn? — sie sättigt ab d≈6) heißt das: Die
Dimensionswahl ist im Bereich 6–24 praktisch frei; wer d=2 nimmt, spielt
mit dem Feuer, gewinnt aber nichts.

### 4.4 Das alte Optimum steht — der neue „Sieger" ist fragil

Neuer Robuster = neuer Peak: Punkt 2 (d=23, nn=15, md=0,31, mcs=6, ms=9)
mit 0,1337 — also −0,0006 *unter* dem alten Optimum 0,1343, mitten im
Jitter (Messrauschen, Streuung σ≈0,001–0,002). Mehr noch: Der Punkt hat nur 3/5 Replikate, Noise
39 % (knapp an der 40-%-Kante) und md=0,31 — mitten im Cliff, wo ihn nur
das winzige mcs=6 rettet. Das ist kein Fundament, das ist ein
Drahtseilakt. Nach 640 Runs auf doppelter Dichte mit offenen Rändern steht
das Runde-1-Optimum (d=11, nn=45, md=0,09, mcs=11, ms=8) ungeschlagen da —
das ist die stärkste Bestätigung, die ein DoE-Optimum (DoE = Design of
Experiments, statistische Versuchsplanung) bekommen kann.

## 5. Ergebnisse Methodenvergleich: Ranking stabil, zwei Methoden-Geschichten

| Emb | HDBSCAN                    | Leiden        | DBSCAN | Agglo                          |
|-----+----------------------------+---------------+--------+--------------------------------|
| d08 | **0,1349** [0,1330,0,1349] | 0,1121 stabil | 0,1067 | 0,0590 instabil                |
| d11 | **0,1351** [0,1325,0,1351] | 0,1124 stabil | 0,1052 | 0,1160 (!) instabil [0,028,0,116] |
| d16 | **0,1324** [0,1324,0,1351] | 0,1132 stabil | 0,1071 | 0,1161 (!) instabil [0,027,0,117] |

(Spanne = Sieger über 3 Seeds; ARI = Adjusted Rand Index, 1 = identisch;
DBSCAN 0,79–0,82 am label-stabilsten. Agglo = Agglomeratives Clustering.)

**HDBSCAN gewinnt überall**, Leiden ist überall Zweiter, DBSCAN (Density-Based Spatial Clustering of Applications with
Noise) überall
Dritter — das Ranking hängt nicht an d. Die **Replikation** gelingt:
d=11 frisch gerechnet ergibt 0,1351 (mcs=11, ms=5) gegen 0,1333 (ms=8) in
Phase B — plus 0,0018 im Jitter, mcs=11 erneut bestätigt, über
Prozessgrenzen und `nn_descent`-Jitter hinweg.

**Leiden ist rehabilitiert — mit Stern.** An Resolution 1,5–3,0 läuft es
auf allen drei Einbettungen felsenstabil (±0,001). Die Phase-B-Diagnose
„instabil" (0,115→0,06) war also keine Methoden-, sondern eine
Resolutionsschwäche: res=5,0 baut fragile Fein-Partitionen. Für die Praxis
ändert das wenig — 0,112 bleibt Zweiter —, aber es korrigiert das Urteil:
Leiden ist stabiler Zweiter, nicht unberechenbar.

**Agglomeratives Clustering (Agglo) ist Lotterie.** 0,116 auf d11/d16 sieht nach Verfolger aus —
gegen 0,031 in Phase B auf nominell gleicher Geometrie (d=11, Seed 42,
nur anderer Prozess)! Die Seed-Spanne [0,03,0,12] enthüllt den Schein:
Agglo gewinnt mal, kollabiert mal, je nach UMAP-Jitter. Der Fall ist das
schönste Argument dieser Untersuchung dafür, dass kein Methodenvergleich ohne
Stabilitätsprüfung zählen darf. DBSCANs Sieger (eps=0,2, ms=25) ist
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
nur Überlebende (108/128): Wer an Noise>40 % stirbt, hinterlässt keinen Messwert
(keine Response) — der Cliff jenseits 0,3 ist also eher unter- als
überschätzt.
**Zweitens** bleibt S/N aus fünf Werten unsicher; das Robust-Ranking kann
selbst rauschen (der fragile Punkt 2 ist der Beleg). **Drittens** fixiert
der Methodenvergleich nn/md — das Ranking gilt streng nur für diese
Geometrie, auch wenn drei d-Stufen beruhigen. **Viertens** ein DB-Snapshot, keine
zeitlichen Splits, ein Rater, keine Wort-Kohärenz-Wiederholung. Und
**fünftens**: 128 Punkte sind besser als 60, aber für fünf Dimensionen
immer noch Screening, keine Vermessung — Drei-Wege-Interaktionen und feine
Krümmungen bleiben unsichtbar.

### 6.4 Was als Nächstes lohnte

Am meisten Erkenntnis pro GPU-Minute verspricht ein **Cliff-Experiment**:
md fein gestaffelt (0,15–0,35) × mcs klein/groß × 10 Seeds — wo genau
kippt das Bild vom Einschmelzen (wenige Riesen-Blobs, §4.1) in massenhafte
Verwerfung wegen Noise>40 %, und rettet klein-mcs immer? Danach ein
`brute_force_knn`-Determinismus-Anker (exakte Nachbarschaftssuche statt
Näherung; 3 Punkte × 2 Seeds) als Reproduzierbarkeits-Eichstrich. Und
irgendwann der Produktivitäts-Schritt: LLM-Rating (LLM = Large Language
Model, großes Sprachmodell) gegen Clusterzahl, um die Fragmentierungsfrage (217 vs. 465
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
