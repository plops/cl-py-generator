# Walkthrough: Mobile Zweitversion der Cluster-Karte (DE)

Datum: 2026-09-27 · Code: `example/174_autocluster/mapmobile/` (standalone) ·
Basis: `plan/20260927_03_mobile_map/review.md`

## 1. Ausgangslage

Die v1-Karte (`mapserver/`, Plotly) ist auf Smartphones kaum nutzbar. Das Review
benennt vier architektonische Gründe: 4,8-MB-JS-Bundle, Fat-Finger (kein Hover
auf Touch, 16.692 dicht liegende Punkte), 220-Zeilen-Legende, 360-px-Sidebar auf
390-px-Viewport. Auftrag: zweite Version als standalone Rust-Programm.

## 2. Was gebaut wurde

`mapmobile/`: eigenes Cargo-Binary, Backend 1:1 aus v1 übernommen (Routen,
CSV/JSON-Loader, Validierung, rusqlite-read-only, Rate-Limit, `db_ok`-Warnlogik —
v1 selbst unangetastet), Frontend komplett neu und mobil-first:

- **Leaflet 1.9.4** (BSD-2-Clause, CDN ~162 KB) mit `L.CRS.Simple` für die
  x/y-Embedding-Koordinaten: native Touch-Gesten (Pinch, Pan, Tap), Zoom-Control,
  `fitBounds`/`flyToBounds`.
- **Eigenes Canvas-Overlay** (ein `<canvas>`, rAF-gedrosselt) + **Gitter-Index**
  für Tap-Picking (20-px-Toleranz, in Daten-Einheiten aus aktuellem Zoom gerechnet).
- **Bottom Sheet** (versteckt → Peek mit Titel → expandiert mit Volltext + Link),
  **Legenden-Modal** (`<dialog>`: Suche, Checkbox-Toggle, Fokus-Zoom),
  Deep-Link `?point=` und Fehler-Sheet wie v1, deutsche UI.

## 3. Wichtigste Entscheidung: Glify evaluiert und verworfen

Das Review empfahl Leaflet.glify als Punkte-Renderer. Verifikation im echten
Browser (Headless-Chrome, WebGL ok, 16.692 Features geladen, keine JS-Fehler):
**kein einziger Punkt rendert** — leere Karte auf allen Zoomstufen. Ursache:
Glifys Bundle projiziert intern Web-Mercator und ignoriert damit `L.CRS.Simple`.
Der Ersatz (eigenes Overlay) ist kleiner (kein Plugin), CRS-nativ per Konstruktion
(Positionen kommen aus Leaflets eigener Projektion) und exakt testbar.

## 4. Ergebnisse (Messwerte)

- CDN: Leaflet JS 147.552 + CSS 14.806 Bytes = **~162 KB** (Faktor ~30 unter Plotly).
- `cargo test`: **15/15** (5 Unit + 10 Integration, inkl. Anti-Scraping- und 500-Tests).
- Node-Harness (exaktes Inline-JS, Stub-DOM): **20/20** (Init, Tap/Radius,
  Sheet-Zustände, Suche/Toggle/Fokus, Deep-Link, Fehler, CDN-Ausfall).
- Headless-Chrome, echte 390×844-Touch-Emulation: Render < 1 s; Tap exakt auf
  Punkt → Sheet mit **korrekter ID 330** in 262 ms; Suche „diffusion" → 1 Treffer;
  Toggle ohne Canvas-Leak; Fokus-Zoom 3 → 4,75; Deep-Link + Fehler-Sheet ok;
  Screenshots gelesen und korrekt (Orientierung wie v1 verifiziert).
- Unterwegs gefixt: `li[hidden]`-CSS (Suche filterte unsichtbar weiter),
  `min-height: 0` für Sheet-Scrollbox, Schließen-Button über den Text gelegt.

## 5. v1/v2-Vergleich und Grenzen

- v1 (Plotly): Desktop-Detailanalyse, 220er-Legende, Hover. v2 (Leaflet+Canvas):
  Smartphone-Bedienung, Bottom Sheet, Tap-Toleranz. Gleiche API, gleiche Daten,
  gleiche Schutzgarantien; per `PORT` nebeneinander lauffähig.
- Grenzen: Sheet ohne Drag-Geste (Tap-Toggle), Cluster-Farbpalette ≠ v1 (eigene
  HSL-Vergabe), Pinch/Pan nur Leaflet-nativ (manuell nachprüfen, s. `task.md`
  Schritt 5), kein Offline-Betrieb (CDN nötig).

## 6. Neue Programme für den Docker-Container

Keine neuen Rust-Crates (identisch zu v1, s. `deps.md`). Test-Werkzeuge (keine
Deliverables, teils aus v1-Session vorhanden): Full-Chrome für Testing 154
(`--headless=new`, nötig für echte 390-px-Emulation — chrome-headless-shell
kann kein 390-px-Viewport), Node-Harness-Skripte in `/tmp` (Proben, wie in v1
nicht committet). Neue Doku: `plan/20260927_03_mobile_map/{task,deps}.md` +
dieses Dokument. Start: `cd example/174_autocluster/mapmobile && cargo run` →
`http://127.0.0.1:8080/map`.

## 7. Nachtrag Runde 2: Performance + Bedienung (User-Feedback, 2026-09-27)

Sechs Kritikpunkte, alle im Frontend (`map.html` einzige Produkt-Datei; kein
Backend-Eingriff, `cargo test` weiter 15/15):

1. **Rendern zu langsam** → Render-Cache: Farbgruppen statisch (nur bei
   Filterwechsel neu), Container-Koordinaten in `Float32Array` (nur bei
   Zoom/Resize neu projiziert); Pannen verschiebt den Cache bloß (exakt, da
   Translation). Gemessen (Headless): Voll-Neuzeichnen 16.692 Punkte **6,1 ms**.
2. **Zoom zu flach/langsam** → `maxZoom` 5 → **8** (Vektor-Canvas, beliebig tief),
   Fokus-Flug auf 0,6 s.
3. **Kein Hover** → Maus-Hover-Label (ID + Cluster-Titel, 10-px-Radius,
   rAF-gedrosselt). Dabei ECHTER Bug gefunden: langes Label lief über und
   weitete das mobile Layout-Viewport auf (390→641)! Fix: Kanten-Umklappen +
   `max-width` + Ellipse, per Viewport-Assertion bewiesen.
4. **Sheet nicht stufenlos** → Griff-Drag per Pointer Events (140 px … 92 dvh,
   mit Clamps); Tap toggelt weiter.
5. **Kein Punktwechsel im Großzustand** → Tap auf Punkt behält Zustand (voll
   bleibt voll, Höhe bleibt); Leertap klappt stufenweise zurück (voll → Peek → zu).
6. **Zu wenig Summary im Peek** → 2-Zeilen-Auszug (160 Zeichen) im Peek.

Verifikation: Harness 27/27, CDP-Mobile (Tap exakt, Drag-Höhen + Clamps, Wechsel
mit Höhenerhalt, Peek-Auszug, Hover-Label, Zoom 8, Screenshots gelesen). Gewonnene
Erkenntnis: CDP-Viewport-Fallen (DSF/Emulation) stets per `innerWidth`-Assertion
absichern — zwei „Produkt-Bugs" unterwegs waren Test-Artefakte.
