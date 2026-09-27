# task.md — serielle Implementierungs- und Testschritte (DE)

Jeder Schritt wird vollständig abgearbeitet und validiert, bevor der nächste beginnt.
Basis: `plan.md` (D1–D8). Commits nach Schritt 1/2/4/6/8 (Conventional Commits).

## Schritt 1 — Gerüst + Docs-Skelett

- `cargo new mapserver --bin`; Deps auf Maximum (`cargo update`); `target/` ins
  `.gitignore`; `plan.md`/`task.md`/`deps.md` + nginx-Vorschlag schreiben.
- Validierung: `cargo build` grün; `git status` zeigt nur gewollte Dateien.
- Commit 1: `feat(mapserver): Cargo-Gerüst + Plan-Docs`.

## Schritt 2 — Daten-Layer

- Module: CSV-Loader (toleranter Delimiter), Titel-Loader (nur `title`+`n`),
  Startup-Validierung (16.692 / 219 / IDs), read-only DB-Handle (`rusqlite`).
- Unit-Tests: Quirk-CSV, Konsistenz-Counts, Quota-Mapping.
- Validierung: `cargo test daten…` grün.
- Commit 2: `feat(mapserver): Daten-Layer mit Validierung`.

## Schritt 3 — API-Routen (ohne UI)

- `/api/map/points`, `/api/map/clusters` (+ Noise-Zeile), `/api/map/point/{id}`,
  `/healthz`; `Cache-Control` auf Bulk-Routen; Governor-Layer auf Detail-Route.
- Validierung: manuell `cargo run` + `curl` (200er, JSON-Form).

## Schritt 4 — Integration-Tests

- Fixture-Server (tempdir, Mini-DB): Punkte-Count + Keys-Whitelist, Detail 200/404,
  429-Nachweis, Cluster-Counts, Shell-ohne-Volltext (nach Schritt 6, sonst rot!).
- Validierung: `cargo test` grün (Schritt-6-Test ggf. als Zweitteiler).
- Commit 3: `feat(mapserver): API + Rate-Limit + Integration-Tests`.

## Schritt 5 — CDN-Pin verifizieren

- `curl -sI <plotly-cdn-url>` + Byte-Größe messen, in Walkthrough/deps dokumentieren.
- Bei Offline-Container: CDN trotzdem verwenden (Tests netzfrei), Messung als TODO.

## Schritt 6 — UI `/map`

- Askama-Template (deutsch): Karten-Div, Panel (Button/Esc, Fokus), Legende,
  Fußzeile + Methodik-`<details>`; JS: scattergl-Traces, Hover (ID+Titel),
  Klick→fetch→Panel, Zoom/Pan nativ.
- Validierung: Route-200/Karten-Div-Test; `curl` Shell ohne Snippet.
- Commit 4: `feat(mapserver): Karten-UI mit Panel und Legende`.

## Schritt 7 — Live-Smoke + Performance (Protokoll!)

Server mit ECHTEN Daten starten (`cargo run`, Port 8080), dann:

1. `curl localhost:8080/healthz` → ok, points 16692, clusters 219.
2. `curl localhost:8080/api/map/points | python3 -c` → 16.692, nur Whitelist-Keys.
3. `curl localhost:8080/api/map/point/<id-mit-text>` → Volltext+Link+Titel.
4. `curl localhost:8080/api/map/point/999999999` → 404.
5. Rate-Limit: mit `MAP_DETAIL_PER_MIN=2` neustarten, 3× schnell → 3. ist 429.
6. Performance: `curl -w` Punkte-API < 0,5 s, `/map` < 2 s.

## Schritt 8 — Browser-Abnahme (Protokoll!)

1. Hover grau → Identifier + „Noise", kein Text. 2. Klick → Panel mit lesbarer
   Summary + klickbarem Link (neuer Tab, Ziel prüfen). 3. Klick Noise-Punkt →
   Panel ebenfalls befüllt. 4. Legende: Toggle blendet Cluster aus/ein, Noise
   toggelbar. 5. Zoom/Pan über alle Punkte. 6. Panel per Button + Esc schließbar.

## Schritt 9 — Walkthrough + finaler Commit

- `walkthrough.md` (deutsch, volle Struktur) + Messwerte + Übernahme-Einschätzung.
- `cargo test` final grün; Commit 5: `docs(map): Abnahme + Walkthrough`.
