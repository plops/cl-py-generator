# deps.md — Abhängigkeiten von mapserver (2026-09-27)

GitHub-Pfade in `<organisation>/<projekt>`-Notation für DeepWiki-Abfragen.
Alle Dependencies in neuester Version (aufgelöst per `cargo update` auf Maximum,
Stand 2026-09-27, verifiziert in `mapserver/Cargo.lock`).

## Laufzeit-Dependencies (aus Referenz übernommen, Versionen gepinnt per Lock)

- `tokio-rs/axum` — HTTP-Routing (Axum). Version `0.8.9`.
- `tokio-rs/tokio` — Async-Runtime, Feature `full`. Version `1.53.1`.
- `tower-rs/tower-http` — `ServeDir` (statische Assets), Feature `fs`. Version `0.7.1`.
- `djc/askama` — Server-Templates für `/map`. Version `0.16.1`.
- `serde-rs/serde` — Serialisierung, Feature `derive`. Version `1.0.229`.
- `serde-rs/json` — JSON-APIs. Version `1.0.151`.
- `tokio-rs/tracing` — Logging (`tracing` `0.1.44` + `tracing-subscriber` `0.3.23`).

## Neu eingeführt für mapserver

- `benwis/tower-governor` — Rate-Limit-Layer (nur Detail-Route). Version `0.8.0`
  (MIT OR Apache-2.0). Transitiv: `antifuchs/governor` `0.10.4`.
- `rusqlite/rusqlite` — Sync-SQLite für Point-Lookups (Default-Features inkl.
  gebundeltem SQLite, kein System-lib nötig). Version `0.40.2` (MIT).
- `plotly/plotly.js` — Karten-Rendering (`scattergl`) via CDN, kein Build-Schritt
  (MIT; Pin + Byte-Größe s. Walkthrough, per curl verifiziert).

## Dev-Dependencies (nur Tests)

- `Stebalien/tempfile` — Fixture-Verzeichnisse in Integration-Tests. Version `3.27.0`.

## DeepWiki-Abfragen (für den Implementierungsagenten)

```text
deepwiki: plops/rs-summarizer — "How are Axum routes, Askama templates, and htmx/pico static assets wired together (build_router, template structs, ServeDir)?"
deepwiki: benwis/tower-governor — "GovernorLayer + GovernorConfigBuilder usage with Axum 0.8, per-IP key extractor, 429 error handling?"
deepwiki: rusqlite/rusqlite — "open_with_flags + SQLITE_OPEN_READ_ONLY example, single-row query with params?"
```
