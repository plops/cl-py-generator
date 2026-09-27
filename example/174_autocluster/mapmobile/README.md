# mapmobile — Mobile Zweitversion der Cluster-Karte (174_autocluster)

Standalone-Axum-Programm (wie `mapserver/`, aber mobil-first): Vollbild-Leaflet
mit Canvas-Overlay (16.692 Punkte), Tap mit 20-px-Toleranz, Bottom Sheet mit
Volltext + Link, Legenden-Modal mit Suche und Fokus-Zoom. Referenz-Docs:
`../plan/20260927_03_mobile_map/` (`review.md`, `task.md`, `walkthrough.md`).

## Start

```sh
cd example/174_autocluster/mapmobile
cargo run            # → http://127.0.0.1:8080/map (nebeneinander mit v1: PORT=8081)
```

Im Handy-Browser `http://<rechner>:8080/map` öffnen (Karte braucht Internet für
das Leaflet-CDN; alle Tests laufen netzfrei).

## ENV (alle mit Default, identisch zu v1)

| Variable | Default | Bedeutung |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8080` | Bind-Adresse |
| `MAP_DB` | `../summaries_compact_20260924.db` | Compact-DB (read-only!) |
| `MAP_COORDS` / `MAP_LABELS` | `../plots/coords_phaseb_2d.csv` / `../plots/labels_phaseb.csv` | Punkte + Labels |
| `MAP_TITLES` | `../cluster_titles_phaseb.json` | Cluster-Titel |
| `MAP_DETAIL_PER_MIN` | `60` | Rate-Limit Detail-API pro IP |

## Routen

- `/` → Redirect auf `/map`; `/map` mobile Karten-Seite (deutsch)
- `/api/map/points` (nie Text), `/api/map/clusters` (219 + Noise),
  `/api/map/point/{id}` (genau ein Volltext, limitiert), `/healthz` (`db_ok`)

## Tests

```sh
cargo test   # 5 Unit- + 10 Integration-Tests (Fixture-Server)
```

## Fehlerbehebung

Siehe `mapserver/README.md` (gleiche Ursachen: DB-Pfad, CSV-Pfade, Rate-Limit).
Mobil zusätzlich: Bei leerer Karte Netz prüfen (Leaflet-CDN `unpkg.com`).
