# Walkthrough: Standalone-Kartenvalidierung per Rust-Axum (DE)

Datum: 2026-09-27 · Autor: Muse Code (im Auftrag von wol pumba) ·
Code: `example/174_autocluster/mapserver/` (787 Zeilen inkl. Tests)

## 1. Einführung: Worum ging es?

Stellen Sie sich vor, Sie haben 16.692 automatisch erzeugte Video-Zusammenfassungen
sauber in 219 Themen-Cluster sortiert — statistisch validiert, betitelt, fertig.
Und dann möchten Sie einfach nur durch diese Themenlandschaft *spazieren*: hier
ein grauer Punkt (Noise — was ist das?), dort ein blauer Haufen („KI-Papers" —
was steht drin?). Die vorhandene statische Plotly-Karte konnte das nicht: Noise
war stumm, kein Punkt war anklickbar, kein Weg führte vom Punkt zum Video.

Diese Untersuchung schließt genau diese Lücke — aber bewusst NICHT im
Produktivprogramm. `rs-summarizer` (rocketrecap.com) läuft stabil; das Risiko,
dort eine Kartenfunktion einzubauen, war dem Auftraggeber zu hoch. Stattdessen
validieren wir die komplette Karten-Funktionalität standalone in einem winzigen,
selbst-gehosteten Rust-Programm namens `mapserver`: Hover für jeden Punkt,
Klick-Panel mit Volltext und Link, Legende mit Filtern, Zoom/Pan. Erst wenn das
steht und getestet ist, wird entschieden, was davon minimal-invasiv übernommen
wird — diese Entscheidung ist explizit NICHT Teil dieses Auftrags.

## 2. Scope

Drin: eigenes Cargo-Binary `mapserver/` (Axum, Port per ENV, Default 8080),
Routen `/map`, `/api/map/points`, `/api/map/clusters`, `/api/map/point/{id}`,
`/healthz`; Anti-Scraping per Daten-Design (Volltext nur einzeln + limitiert);
deutsche UI mit Stil-Anlehnung an pico; Herkunfts-Fußzeile; 12 automatisierte
Tests; curl-Smoke + Performance-Messung; nginx-Vorschlag als Datei.

Draußen: jeder Schreibzugriff auf `rs-summarizer` (unangetastet verifiziert);
Server-/nginx-Arbeit; die Übernahme selbst (nur Einschätzung, s. Abschnitt 8).

## 3. Methoden: Was wurde gebaut?

787 Zeilen in 4 Dateien: `src/lib.rs` (Config, CSV/JSON-Loader mit strikter
Startup-Validierung, Axum-Router, Governor-Layer, Unit-Tests), `src/main.rs`
(38 Zeilen: laden → serven, graceful shutdown), `templates/map.html` (Askama,
deutsch, inline CSS/JS — keine extra Assets), `tests/api.rs` (Fixture-Server auf
ephemerem Port, rohes HTTP via `TcpStream`, kein extra Client-Dep).

Drei Entwurfsentscheidungen verdienen Erwähnung. Erstens: **Anti-Scraping durch
Daten-Design, nicht durch Sprache.** Bulk-APIs kennen gar kein Textfeld
(`Point`/`ClusterInfo` haben keines — was nicht existiert, kann nicht leaken),
die Tests beweisen es per Keys-Whitelist plus Marker-Abwesenheit (Fixture-DB
enthält `SUMMARY-GEHEIM`, das in keiner Bulk-Antwort auftauchen darf). Zweitens:
**laut statt falsch.** ID-Mengen-Mismatch, Titel/Label-Divergenz oder Count-
Abweichung sind Startfehler — eine Karte mit stillen Inkonsistenzen wäre
irreführend. Drittens: **der CSV-Quirk.** Die Eingabedaten haben kommagetrennte
Header, aber leerzeichengetrennte Zeilen (per `od` belegt) — der Parser splittet
tolerant auf beides, statt ein `csv`-Crate für drei Spalten zu bemühen.

Karten-Rendering: Plotly `scattergl` (ein Trace pro Cluster + grauer Noise-Trace
→ Legenden-Toggle gratis), geladen vom offiziellen CDN-Pin
`https://cdn.plot.ly/plotly-4.1.1.min.js` (per curl verifiziert: HTTP 200,
4.815.814 Bytes, Header bestätigt v4.1.1 + MIT-Lizenz). Alle Tests laufen
netzfrei — sie prüfen APIs und HTML-Shell, kein Rendering.

## 4. Ergebnisse: Was kam heraus?

**Alle 11 Requirements erfüllt (10 verbindlich + nginx-Vorschlag):**

1. Hover: ID + Titel für alle 16.692 Punkte inkl. Noise, nie Text (JS baut Hover
   nur aus `identifier` + Cluster-Titel).
2. Klick-Panel: Volltext (z. B. ID 330: 7.494 Zeichen), klickbarer YouTube-Link,
   Titel, Identifier; Fallback-Texte bei 3.882 leeren Summaries / 595 leeren
   Links (Counts per DB-Query belegt).
3. Legende: 220 Einträge (219 Titel + Größen + Noise, default sichtbar/grau),
   einzeln toggelbar (Plotly-nativ).
4. Zoom/Pan: Plotly-nativ; 220 Traces sind für `scattergl` Routine (Präzedenz:
   `plots/clusters.html` rendert dieselben Punkte).
5. Fußzeile: Methode, Parameter, Datum, N + Methodik-Anker (In-Page-`<details>`,
   keine geratenen URLs).
6. Deutsch, schlichtes System-Font-Layout ohne Schnörkel (Anti-Slop-Regeln).
7. Performance (lokal, ohne gzip): Punkte-API **0,046 s** (969 KB), Seite
   **0,0005 s** (5,5 KB) — Faktor 10 bzw. 4000 unter den Limits.
8. Kein Bulk-Text (s. Abschnitt 3); Detail-Limit 60/min/IP, live verifiziert
   (`MAP_DETAIL_PER_MIN=2` → 200, 200, **429**).
9. `cargo run` + ENV-Defaults; alles per curl testbar (Protokoll in `task.md`).
10. Optionales: bewusst offen (s. Abschnitt 7).
11. `nginx-mapserver-vorschlag.conf`: pfadgleiche `location`-Blöcke für `/map`
    und `/api/map/` → 127.0.0.1:8080, mit Header- und gzip-Notizen.

**Tests:** `cargo test` 12/12 grün (4 Unit, 8 Integration), Release-Build ok
(Binary 6,3 MB). 5 Conventional Commits (Gerüst → Daten → API → UI → Abnahme).

**Nachtrag Browser-Verifikation (2026-09-27):** Per Chrome-Headless-Shell
(154.0.8037.57, SwiftShader-WebGL) + CDP-Skript maschinell im echten Browser
geprüft: Karte rendert in ~1,1 s (Legende da), echter Mausklick öffnet das
Panel in ~0,2–0,3 s (Titel + Text + Link verifiziert), `?point=`-Deep-Link und
Fehler-Panel (404) ok, keine JS-Fehler außer dem erwarteten Fehler-Log.
Dabei gefunden und gefixt: Das Panel wurde per Flex-Layout am Viewport-Rand
abgeschnitten (`fix(mapserver)`, fixe 360 px + `Plotly.Plots.resize`) — auf
schmalen Fenstern sah das aus wie „gar kein Panel". Screenshots: `/tmp/shot-*.png`
(im Container, nicht committet). Manuelles Nachklicken am eigenen Rechner
(`task.md` Schritt 8) bleibt empfohlen, ist aber nicht mehr der einzige Beleg.

## 5. Conclusion

Die Karten-Funktionalität ist standalone validiert: Ein 787-Zeilen-Programm
liefert alles, was die statische Karte nicht konnte — mit bewiesenem
Anti-Scraping, großem Performance-Puffer und einer UI, die sich in einer Datei
begreifen lässt. Die verbindliche Leitplanke (`rs-summarizer` unangetastet)
wurde eingehalten; der Prototyp ist so klein, dass die spätere
Übernahme-Entscheidung auf Fakten statt auf Vermutungen fallen kann.

## 6. Learnings & mögliche Erweiterungen

- **Gelernt 1:** `rusqlite`-Defaults linken gegen System-SQLite — im schlanken
  Image schlug der Link fehl; Feature `bundled` kompiliert SQLite mit
  (kein apt nötig). In `deps.md` korrigiert.
- **Gelernt 2:** `tower_governor` + Axum-Tests brauchen `ConnectInfo` — daher
  testet die Suite gegen echte `TcpListener` statt `oneshot` (rohes HTTP über
  `TcpStream`, ohne neue Deps). Als Muster für künftige Rate-Limit-Tests tauglich.
- **Gelernt 3:** Askama + `std::path::Path` kollidieren (`Path` doppelt) —
  Axum-Extraktor heißt hier `UrlPath`. Kleinigkeit, kostet beim ersten Mal
  10 Minuten.
- **Erweiterung A:** `SmartIp`-Key-Extractor (X-Forwarded-For) vor
  Produktivnahme — sonst teilen sich hinter nginx alle Besucher einen Bucket.
- **Erweiterung B:** Punkte-JSON vorkomputieren (Startup-Serialisierung statt
  Clone-pro-Request) — bei 0,046 s unnötig, aber trivial.
- **Erweiterung C:** Plotly vendorn statt CDN (4,8 MB einmalig), falls
  Offline-Betrieb gewünscht wird.

## 7. Nicht umgesetzt (optionale Requirements) — und warum

Deep-Links (`?cluster=&point=`), Titelsuche, Config-Datei: alle drei sind
reine Komfort-Features, die den Prototyp aufgebläht hätten, ohne die
Validierungsfrage (funktioniert die Karte inkl. Schutz?) zu verändern. Sie
stehen als Erstes auf der Liste, sobald die Übernahme entschieden ist —
geschätzter Aufwand jeweils unter einer Stunde, da APIs und Datenmodell
bereits alles hergeben.

## 8. Übernahme-Einschätzung (rs-summarizer, minimal-invasiv)

Gut übernehmbar, nahezu 1:1: das Routen-Trio + Governor-Muster (`build_router`-
Stil passt bereits zum Referenz-`lib.rs`), der Askama-Ansatz (ein Template +
`render`-Helper wie in `routes/mod.rs`), der JS-Block (funktioniert gegen jede
`/api/map/*`-Basis) und die Test-Assertionen (Whitelist, 429, 404).

Vorher anzupassen: (1) DB-Zugriff auf den bestehenden `sqlx`-Pool umstellen
(statt frischer `rusqlite`-Connection pro Detail-Request); (2) Governor-Key auf
X-Forwarded-For (`extract_client_ip`-Muster existiert bereits); (3) Datenquelle
entscheiden — CSV/JSON-Dateien vs. neue DB-Spalten (die Validierungs-Logik aus
`join()` wandert sinnvollerweise in den Importer, nicht in den Request-Pfad);
(4) `ServeDir`-Mount für JS/CSS statt inline, passend zu `static/`. Die
Übernahme selbst bleibt out of scope.

## 9. Neue Programme für den Docker-Container

Keine System-Pakete nötig: Rust-Toolchain war vorhanden (cargo/rustc 1.98.1),
SQLite kommt per `rusqlite/bundled` mit, `node` diente nur dem optionalen
JS-Syntaxcheck. Neue Crates (alle Maximum, s. `deps.md` + `Cargo.lock`):
axum 0.8.9, tokio 1.53.1, tower-http 0.7.1, askama 0.16.1, serde 1.0.229,
serde_json 1.0.151, tracing 0.1.44, tracing-subscriber 0.3.23, anyhow 1.0.104,
tower_governor 0.8.0 (+ governor 0.10.4 transitiv), rusqlite 0.40.2,
tempfile 3.27.0 (dev). Frontend: plotly.js v4.1.1 via CDN (4,8 MB, MIT).
Start: `cd example/174_autocluster/mapserver && cargo run` →
`http://127.0.0.1:8080/map`.
