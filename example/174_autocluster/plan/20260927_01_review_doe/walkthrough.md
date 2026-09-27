# Walkthrough: Robuste Cluster-Evaluation per Design of Experiments (DE)

Datum: 2026-09-27 · Autor: Muse Code (im Auftrag von wol pumba) ·
Code: `example/174_autocluster/doe/` · GPU: NVIDIA RTX A4000

## 1. Einführung: Worum ging es?

Stellen Sie sich vor, Sie haben knapp 20.000 automatisch erzeugte
Video-Zusammenfassungen und möchten wissen: *Welche Themen stecken darin?*
Die Vorgängeruntersuchung beantwortete das mit einer GPU-Pipeline (UMAP +
HDBSCAN) und kürte per Grid-Sweep eine Best-Config. Doch ein Review stellte
die unbequemen Fragen: War der Breitenvergleich fair, wenn bei voller Breite
1.577 Dokumente stillschweigend herausfielen? Ist der hauchdünne Sieg von
`d=12` über `d=8` mehr als GPU-Zufall? Und wurde DBSCAN je fair behandelt?
Diese Untersuchung beantwortet alle drei Fragen — mit der statistischen
Gründlichkeit eines **Design of Experiments (DoE)**: 298 GPU-Runs, 5
Seed-Replikate, Varianzanalyse statt Bauchgefühl.

**Scope des Experiments:** UMAP+HDBSCAN-Hyperparameter­raum (5 Steuergrößen +
Seed-Störgröße) auf der Schnittmenge N=16.692 (Compact-DB, volle 3072-D-Rows);
Zielfunktion `Silhouette_orig × (1 − Noise)` im k-dim Originalraum; faire
Breiten-Ablation k ∈ {128, 768, 3072}; faire DBSCAN-eps-Kalibrierung;
Legendenfix in `plot_final.py`. Außerhalb des Scopes: Manifold-Fidelity
(Trustworthiness/TwoNN, weiter offen), Cluster-Betitelung, Online-Einordnung
neuer Embeddings (Folgeprojekt).

## 2. Methode: Was wurde gebaut?

Neuer Ordner `doe/` (reines Python, kein Lisp-Transpiler), ein Modul pro
Schritt: `data.py` lädt die Schnittmenge einmal und trunkiert pro Breite
(Bias-Fix); `design.py` erzeugt den Latin-Hypercube-Plan (60 + 12 Punkte),
Taguchi-S/N und ARI-Stabilität und fittet das Response-Surface-Modell
(ANOVA Typ II via `statsmodels`); `experiment.py` kapselt
UMAP+HDBSCAN+Scoring mit Seed-im-Cache-Key; `run_doe.py` fährt die vier
Phasen (jitter 10×, screening 60×3, width 12×3×3, dbscan 4×7);
`analyze.py` berichtet ANOVA, Optima und schreibt 5 Plots. Dazu 10 CPU-Tests
(`tests/test_doe.py`) und Docs (`plan.md`, `task.md`, `deps.md`, dies
Dokument). Alle Läufe auf der A4000, DB strikt read-only.

## 3. Ergebnisse: Was kam heraus?

**Alle vier Review-Befunde bestätigt — und quantifiziert:**

1. **Survival Bias existiert, ändert das Ranking aber nicht.** Faire Ablation
   auf identischen Rows: k=128 → 0,120, k=768 → 0,124, k=3072 → 0,126.
   „Volle Breite lohnt sich" gilt auch ohne Bias — aber k=128 ist
   fragiler (4 von 12 Punkten verworfen).
2. **Der d-Sieg war Rauschen — der wahre Hebel ist `min_cluster_size`.**
   Jitter der Best-Config über 10 Seeds: std=0,0012 (bei ±0,002 alter
   d-Spreizung!). ANOVA (R²=0,88): `min_cluster_size` hochsignifikant
   (p=0,001), dazu die Interaktionen `mcs:min_samples` (p=0,005) und
   `n_neighbors:min_cluster_size` (p=0,024) — die vom Review vermutete
   UMAP×HDBSCAN-Wechselwirkung, erstmals nachgewiesen. `d` (p=0,27) und
   `min_dist` (p=0,56) sind insignifikant.
3. **DBSCAN wurde unterschätzt, bleibt aber Zweiter.** Faires eps-Grid:
   bester DBSCAN 0,110 (d=4, eps=0,2) statt ≈0,058 mit der alten Heuristik —
   fast verdoppelt, doch HDBSCAN (0,134) führt weiter.
4. **Neues robustes Optimum schlägt den alten Bestwert:** d=11, nn=45,
   md=0,093, mcs=11, ms=8 → mean=0,1343 (217 Cluster, ARI=0,66) —
   besser als die alten 0,1327, und auf der *strengeren* Schnittmenge
   erzielt. Robust- und Peak-Optimum fallen zusammen.
5. **Legendenfix verifiziert:** 174-Cluster-Plot mit kompakter Legende
   (Top-12 + Sammelpunkt) rendert fehlerfrei.

## 4. Conclusion

Das DoE hat gehalten, was das Review versprach: Statt eines zufälligen
Peak-Scores auf geratener Heuristik besitzen wir nun eine **signifikante,
reproduzierbare Parameterwahl** — mit bekanntem Rausch­pegel (σ≈0,001),
nachgewiesenen Interaktionseffekten und einem Optimum, das gleichzeitig
Spitze und robust ist. Die wichtigste praktische Lehre: Wer UMAP+HDBSCAN
tunt, sollte seine Zeit in `min_cluster_size`/`min_samples` stecken, nicht
in die UMAP-Dimension. Die offene Flanke bleibt die Seed-Stabilität einzelner Themen
(ARI 0,62–0,78): gut genug für Exploration, zu prüfen vor Produktion.
Die Manifold-Fidelity (Trustworthiness-Elbow, TwoNN) wurde am 2026-09-27 als
Follow-up nachgeholt — siehe `doe/FIDELITY_de.md`: Elbow/Saettigung ab d≈6,
TwoNN-Kerne ≈13 (DoE-Optimum d=11 bestaetigt), PCA-80 % bei 273 Komponenten.

## 5. Learnings & mögliche Erweiterungen

- **Gelernt 1:** `qmc.scale` vektorisiert nutzen (Review-Skizze skalierte
  skalar) — schneller und weniger fehleranfällig.
- **Gelernt 2:** Taguchis „robust schlägt peaky" gilt nur bei gleichem
  Mittelwert (Jensen); +0,002 Mittelwert schlägt ±0,015 Streuung — der
  Review-Vergleich war illustrativ, der Test präzisiert ihn.
- **Gelernt 3:** k=128-Punkte scheitern überproportional (Noise > 40 %) —
  schmale Prefixe sind nicht nur schlechter, sondern instabiler.
- **Erweiterung A:** konfirmatorische 2. DoE-Runde (Box-Behnken/CCD) eng um
  das robuste Optimum zur Feinvermessung.
- **Erweiterung B:** `brute_force_knn`-Kontrollläufe als Determinismus-Anker.
- **Erweiterung C:** Rust-Online-Einordner für neue Embeddings (aus `collect.sh`
  entworfener Follow-up-Auftrag): Clustern ohne GPU, nur neue Samples ans LLM.

## 6. Neue Programme für den Docker-Container

In `requirements.txt` gepinnt (alle über Standard-PyPI, kein NVIDIA-Index
nötig): `statsmodels==0.15.0` (ANOVA/RSM), `patsy==1.0.3` und
`formulaic==1.2.2` (Formel-Parser, transitiv). Installations­befehl laut
`ENV.md`, ergänzt um `-r requirements.txt` (enthält die neuen Pins).

## 7. Artefakte (Nachweis)

- Code: `doe/{data,design,experiment,run_doe,analyze}.py`, Fix in
  `plot_final.py`, Tests in `tests/test_doe.py` (24/24 grün).
- Daten: `doe/results/` (180+108+28+10 Runs, `best_doe.json`, Report, Logs).
- Plots: `doe/plots/doe_*.png` (5) + `plotfix_check/best_clusters.png`.
- Docs: `plan/20260927_01_review_doe/{prompt,review,deps,plan,task}.md` +
  dies `walkthrough.md`. UMAP-Caches (`doe/umap_cache_doe/`, ~300 `.npy`)
  sind per `.gitignore` (`umap_cache*/`) vom Commit ausgenommen.

## 8. Runde 2 (Phase A + B): Vom Optimum zum Methoden-Entscheid

### Einleitung: Zwei unbequeme Fragen blieben übrig

Nach Runde 1 hatten wir, was wir wollten: ein statistisch abgesichertes
Optimum, einen vermessenen Rauschpegel und den Nachweis, dass die
UMAP-Dimension keine Rolle spielt. Man hätte aufhören können. Aber zwei
Dinge nagten — das eine technisch, das andere grundsätzlich.

Erstens: Unser Optimum für `min_cluster_size` lag mit 11 fast genau auf der
Untergrenze des abgesuchten Bereichs (10–40). Versteckt sich das wahre Optimum
vielleicht noch darunter, bei 5 oder 6? Und zweitens, fast peinlich: 8 % unserer
Daten sind exakte Dubletten — der Fidelity-Follow-up hatte 1.324 gefunden,
vermutlich immer gleiche Fehlermeldungs-Embeddings. Bisher hatten wir sie
stillschweigend mitgeschleppt. Helfen sie dem Clustering als Anker, oder
verzerren sie alles, und wir müssten erst einmal putzen?

Und dahinter lauerte die noch größere Frage: Ist HDBSCAN eigentlich das
richtige Verfahren — oder gewinnt es nur, weil unser eigenes Maß ihm
schmeichelt? Denn unser Score streicht 36 % aller Punkte als „Noise" aus der
Wertung, bevor er die Silhouette berechnet. Ein Verfahren, das sich vor den
schwierigen Punkten drücken darf, hat es leicht, gut auszusehen. Misst unser
Maß überhaupt Themen-Qualität — oder nur geometrische Bequemlichkeit?

Runde 2 geht diesen Fragen in zwei Phasen nach: **Phase A** bestätigt und
zoomt (feineres Design ums Optimum, plus Dedup-Experiment), **Phase B** lässt
vier Verfahren fair gegeneinander antreten und prüft das Ergebnis von außen —
mit Wortstatistik statt Geometrie und mit 24 von Hand gelesenen Clustern.

### Was wir getan haben (kurz; Details in den Phasen-Berichten)

Phase A fuhr ein Central-Composite-Design über mcs/ms/nn (20 Punkte × 5 Seeds),
gekreuzt mit Dedup an/aus — 206 Fits insgesamt. Der methodische Clou: Dedup
wirkt nur auf das Fitting (UMAP + Clustering sehen 15.368 statt 16.692 Punkte),
gewertet wird aber immer auf denselben vollen 16.692 Rows, indem jede Dublette
das Label ihres Vertreters erbt. So bleibt der Blockvergleich fair.

Phase B fixierte dann eine einzige Einbettung (d=11, die Sieger-Geometrie) und
ließ vier Verfahren mit je eigenem Mini-Tuning antreten: HDBSCAN, DBSCAN, Leiden
(Graph-Communitys, neu im Werkzeugkasten) und Agglomeratives Clustering. Die
Sieger mussten zweierlei beweisen: Stabilität über UMAP-Seeds und über
80/90-%-Teilstichproben — wichtig für später: Bleiben Cluster-IDs stabil, wenn
die Datenbank wächst? Und schließlich der externe Blick: Pro Cluster berechneten
wir die NPMI-Wortkohärenz — teilen die Wörter eines Clusters gemeinsame
Kontexte in den Summaries? — und korrelierten sie mit der geometrischen Enge.
Dazu 24 blind gezogene Cluster mit je 8 Beispiel-Summaries, von Hand gelesen
und auf einer Skala von 1–5 bewertet; die Methode wurde erst nach dem Rating
enthüllt.

### Ergebnisse: Zwei Überraschungen und eine Bestätigung

**Überraschung 1: Putzen schadet.** Dedup drückt den Score um −0,028 (ANOVA:
F=860, mit Abstand der dominanteste Effekt der gesamten Untersuchung) — und
zwar bei allen 19 Configs einstimmig, mit frischen Seeds bestätigt. Schlimmer
noch: Ohne Dubletten verfünffacht sich der Seed-Jitter (σ 0,0013 → 0,0072).
Die Dubletten sind keine Verunreinigung, sondern fixierte Null-Distanz-Anker,
die UMAP-Graph und HDBSCAN stabilisieren. (Nebenbei die einzige signifikante
Interaktion: dedup×nn — ohne Anker wäscht großes nn die Struktur aus. Die von
uns vermutete dedup×mcs-Wechselwirkung wurde widerlegt.)

**Keine Überraschung, aber wichtig: Die Zoom-Box ist flach.** In mcs∈[5,19]
sind alle Configs statistisch gleichauf (alle p>0,15); der rechnerische Argmax
bei mcs=5 ist Rausch-Chasing. Dort fragmentiert das Clustering auf ~450 Cluster
bei mickriger Stabilität (ARI≈0,52) — ohne belastbaren Gewinn. Fazit: mcs=11
bleibt; die vermeintliche Grenze war keine.

**Überraschung 2 (die bittere): Leidens Sieg war Glück.** Im Bake-off sah alles
gut aus — HDBSCAN 0,133 vor Leiden 0,115, DBSCAN 0,104, Agglo abgeschlagen bei
0,031. Doch auf anderen UMAP-Seeds kollabiert Leiden (0,115 → 0,06): Feste
Resolution plus neu gebauter kNN-Graph ergibt eine andere, viel schlechtere
Partition. Ohne den Seed-Check hätten wir einen Schein-Sieger gekürt — eine
Lehre, die den Aufwand von Phase B allein rechtfertigt. HDBSCAN dagegen zeigt
die stabilsten Scores aller Verfahren (σ=0,0005); DBSCAN die stabilsten Labels
(ARI 0,80).

**Das Nullresultat, das am meisten lehrt:** Wort-Kohärenz und geometrische Enge
hängen praktisch nicht zusammen (r≈0,05 über ~600 Cluster), und alle Methoden
sind im Mittel gleich wort-kohärent (NPMI ≈ 0,23–0,24). Das blinde Rating sagt
dasselbe: DBSCAN 4,50 ≈ HDBSCAN 4,33 ≈ Leiden 4,17 — nur Agglo fällt ab (3,40,
inklusive eines 195er Clusters aus leeren Summaries, wie es nur
Zwangspartitionierung erzeugen kann).

### Diskussion: Was heißt „beste Methode" überhaupt?

Hier müssen wir ehrlich sein — auch zu uns selbst. Unser Score,
Silhouette×(1−Noise), misst **Dichte plus Abstention**: Er belohnt enge Cluster
und erlaubt, sich vor 36 % der Daten zu drücken. Menschen und Wortstatistik
messen **Themen**: Gehören diese Videos inhaltlich zusammen? Das sind
verschiedene Qualitäten, und Runde 2 zeigt, dass sie auseinanderfallen können.
Der Score ist damit teil-validiert: Er verwirft Agglo zu Recht, aber sein
Fein-Ranking (HDBSCAN > Leiden > DBSCAN) findet menschlich keine Deckung.

Für die Praxis heißt das zweierlei. Erstens, die gute Nachricht: Der
Methoden-Entscheid steht trotzdem — **HDBSCAN für Produktion** (bester Score,
stabilste Scores, gute Themen, 217 Cluster), DBSCAN als ehrlicher Zweiter für
gröbere Cluster, Leiden und Agglo verworfen. Zweitens, die unbequeme Nachricht:
36 % Noise bedeuten 36 % Videos ohne Thema. Ob das in Ordnung ist, kann keine
Kennzahl entscheiden — das ist eine Produktentscheidung (eigene Anzeige?
Second-Level-Zuordnung im geplanten Rust-Updater?). Und die Fragmentierungsfrage
aus Phase A (217 vs. 465 Cluster) bleibt offen: Nur externe Urteile — mehr
Ratings, idealerweise LLM-bewertet in größerem Maßstab — könnten sie entscheiden.

Grenzen dieser Runde, offen benannt: Das Author-Rating umfasst nur 24 Units von
einem Rater — schwach, aber protokolliert und versiegelt durchgeführt. Die
NPMI-Rechnung mit Häufigkeits-Top-Wörtern ist grob (wenn auch mit Orakel-Tests
validiert). Und statt echter zeitlicher Splits gab es nur Zufalls-Subsamples —
die DB hat keine Ingestions-Zeitstempel. Wer hier weitergehen will: LLM-Rating
gegen Clusterzahl, dann der Rust-Updater. (Details: `doe/PHASEA_de.md`,
`doe/PHASEB_de.md`, `ratings_phaseb.md`; Tests 45/45 grün.)
