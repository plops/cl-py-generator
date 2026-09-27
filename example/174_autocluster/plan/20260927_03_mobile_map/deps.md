# deps.md — Abhängigkeiten von mapmobile (2026-09-27)

GitHub-Pfade in `<organisation>/<projekt>`-Notation. Rust-Deps identisch zu v1
(`mapserver/Cargo.lock`-Stände, per eigenem `cargo update` aufgelöst);
Frontend komplett neu (Leaflet statt Plotly).

## Laufzeit-Dependencies (Rust, aus v1 übernommen)

- `tokio-rs/axum` `0.8.9` · `tokio-rs/tokio` `1.53.1` (full) ·
  `tower-rs/tower-http` `0.7.1` (fs) · `djc/askama` `0.16.1` ·
  `serde-rs/serde` `1.0.229` (derive) · `serde-rs/json` `1.0.151` ·
  `tokio-rs/tracing` (`0.1.44` + `0.3.23`) · `dtolnay/anyhow` `1.0.104`
- `benwis/tower-governor` `0.8.0` (MIT OR Apache-2.0; transitiv
  `antifuchs/governor` `0.10.4`) · `rusqlite/rusqlite` `0.40.2` (MIT, `bundled`)

## Neu: Mobile-Frontend (CDN, Pins per curl verifiziert, HTTP 200)

- `Leaflet/Leaflet` — Leaflet `1.9.4` (BSD-2-Clause), `L.CRS.Simple`-Karte.
  `https://unpkg.com/leaflet@1.9.4/dist/leaflet.js` — **147.552 Bytes**;
  `.../leaflet.css` — **14.806 Bytes**.
- EIGENE Canvas-Schicht (kein Plugin): Leaflet liefert Gesten/CRS, ein
  Overlay-Canvas zeichnet die Punkte (~10 ms), ein Gitter-Index pickt Taps.
  Glify 3.3.1 (MIT, 118.164 Bytes) wurde evaluiert und VERWORFEN: es projiziert
  intern Web-Mercator und rendert unter `L.CRS.Simple` nichts (Browser-Test).
- Summe CDN: **~162 KB** (nur Leaflet) statt 4.815.814 Bytes Plotly (Faktor ~30).

## Dev-Dependencies (nur Tests)

- `Stebalien/tempfile` `3.27.0`.

## DeepWiki-Abfragen

```text
deepwiki: Leaflet/Leaflet — "L.CRS.Simple for flat x/y scatter maps, y-axis orientation? Touch gestures (pinch/pan/tap) out of the box?"
deepwiki: robertleeplummerjr/Leaflet.glify — "points({data as GeoJSON, color callback IColor range, sensitivity for fat-finger taps, click(e, feature) signature})?"
```
