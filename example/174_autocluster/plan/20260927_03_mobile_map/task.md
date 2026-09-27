# task.md — serielle Schritte Mobile-Zweitversion (DE)

Basis: genehmigter Plan (M0–M4), Review `review.md`. v1 (`mapserver/`) bleibt
unangetastet; keine Commits ohne explizite Bitte (D7).

## Schritt 1 — M0 Gerüst + Backend-Übernahme

- `cargo new mapmobile --bin`; Cargo.toml-Deps wie v1; `lib.rs`/`main.rs`/
  `tests/api.rs` kopieren (`mapserver::` → `mapmobile::`); v1-Template vorerst
  kopieren (M1-Parität); `mapmobile/target/` ins `.gitignore`.
- Validierung: `cargo test` 15/15 grün (Parität mit v1).

## Schritt 2 — M1 Docs

- `deps.md` (Rust-Deps + CDN-Pins mit Lizenz + Byte-Größen), diese `task.md`.
- Validierung: Versionen gegen `mapmobile/Cargo.lock` + curl-Messung geprüft.

## Schritt 3 — M2 Mobile-Template

- `mapmobile/templates/map.html` neu: Vollbild-Leaflet (`CRS.Simple`,
  y-Achse invertiert!), glify-Layer (Cluster-Farben, `sensitivity`),
  Bottom Sheet (Peek/Expand/Esc), `<dialog>`-Legenden-Modal (Suche, Toggle,
  Fokus), kompakte Fußzeile, Fehler-Hinweise wie v1.
- Shell-Test anpassen: Leaflet/glify-Marker vorhanden, kein Plotly, kein Volltext.
- Validierung: `cargo test` grün; `node --check` Inline-JS.

## Schritt 4 — M3 Logik-Harness

- `/tmp/mobile-harness.js` (Probe, nicht committet): Stub-DOM + Leaflet/glify-
  Stubs; Szenarien: Init (Layer-Daten, Farben, Bounds), Tap → Sheet-Peek mit
  Titel, Expand → Volltext + Link, Modal-Suche/Filter, Fokus-BBox, Fehler-Panel,
  Deep-Link `?point=`.
- Validierung: alle Checks PASS.

## Schritt 5 — M3 Mobile-Browser-Check (Protokoll!)

Headless-Chrome + Mobile-Emulation (390×844, Touch, mobile UA), Server mit
ECHTDATEN (eigener Port):

1. Karte rendert (Legende/Attribution ok, keine JS-Fehler), Renderzeit notieren.
2. Tap auf Punkt (Touch-Event!) → Sheet-Peek mit Cluster-Titel; Tap/Expand →
   Volltext + klickbarer Link (Ziel prüfen).
3. Modal öffnen: Suche filtert 220 → Treffer; Toggle blendet Cluster aus/ein;
   Fokus zoomt auf Cluster.
4. `?point=<id>` öffnet Sheet direkt; `?point=999999999` → Fehler-Sheet.
5. Screenshots (Karte, Sheet, Modal) lesen und bewerten.
6. Tap-Latenz notieren (Ziel < 1 s).

## Schritt 6 — M4 Walkthrough

- `walkthrough.md` (deutsch, kurz): Ergebnis, Messwerte (CDN-Bytes, Latenzen),
  was aus v1 übernommen, bekannte Grenzen, Vergleich v1/v2.
- Validierung: final `cargo test` grün; keine Prozesse/Ports übrig.
