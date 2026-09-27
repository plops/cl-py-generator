# mapserver — Standalone-Kartenvalidierung (174_autocluster)

Kleines Axum-Programm: interaktive Cluster-Karte (16.692 Punkte, 219 Cluster + Noise)
mit Hover, Klick-Panel (Volltext + Link), Legende, Zoom/Pan. Referenz-Docs:
`../plan/20260927_02_webmap_route/` (`plan.md`, `task.md`, `walkthrough.md`).

## Start

```sh
cd example/174_autocluster/mapserver
cargo run            # → http://127.0.0.1:8080/map
```

Dann im Browser `http://127.0.0.1:8080/map` öffnen (Karte braucht Internet für das
Plotly-CDN; alle Tests laufen netzfrei).

## ENV (alle mit Default)

| Variable | Default | Bedeutung |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8080` | Bind-Adresse |
| `MAP_DB` | `../summaries_compact_20260924.db` | Compact-DB (read-only!) |
| `MAP_COORDS` / `MAP_LABELS` | `../plots/coords_phaseb_2d.csv` / `../plots/labels_phaseb.csv` | Punkte + Labels |
| `MAP_TITLES` | `../cluster_titles_phaseb.json` | Cluster-Titel |
| `MAP_DETAIL_PER_MIN` | `60` | Rate-Limit Detail-API pro IP |

## Routen

- `/` → Redirect auf `/map`; `/map` Karten-Seite (deutsch)
- `/api/map/points` (16.692 × id,x,y,cluster — nie Text), `/api/map/clusters`
  (219 + Noise), `/api/map/point/{id}` (genau ein Volltext, limitiert), `/healthz`

## Tests

```sh
cargo test   # 4 Unit- + 8 Integration-Tests (Fixture-Server, inkl. 429-Nachweis)
```
