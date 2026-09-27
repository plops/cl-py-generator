# Installation auf hetznerus (mapmobile)

Stand: 2026-09-27. Ziel: `mapmobile` läuft als systemd-Dienst hinter nginx
auf **https://rocketrecap.com/karte** — neben `rs-summarizer` (`/`) und
`rs-scrape` (`jobs.rocketrecap.com`), die unangetastet weiterlaufen.

## Überblick

| Was | Wo |
|---|---|
| Binary | `/home/kiel/mapdeploy/bin/mapmobile` (nur 127.0.0.1:8080) |
| Kartendaten (CSV/JSON) | `/home/kiel/mapdeploy/data/` |
| Datenbank (nur lesend!) | `/home/kiel/host/data/summaries.db` (+ `-wal`/`-shm`) |
| systemd-Unit | `/etc/systemd/system/mapmobile.service` |
| nginx | zwei `location`-Blöcke in `/etc/nginx/nginx.conf` (443er `rocketrecap.com`-Block) |
| URL | `https://rocketrecap.com/karte` |

Die App bindet absichtlich nur localhost — von außen antwortet ausschließlich
nginx. Das Detail-Rate-Limit ist pro Nutzer-IP (die App liest dafür
`X-Forwarded-For`, das nginx mit der echten Client-IP setzt).

## Voraussetzungen

- Root-Zugriff auf hetznerus (Key liegt in `~/.ssh`, Host `hetznerus`)
- Dateien aus diesem Repo auf dem Server (einmalig kopieren):
  `mapmobile/target/release/mapmobile` → `~/mapdeploy/bin/`,
  `plots/coords_phaseb_2d.csv`, `plots/labels_phaseb.csv`,
  `cluster_titles_phaseb.json` → `~/mapdeploy/data/`,
  `mapmobile/deploy/mapmobile.service` → `~/mapdeploy/`
- `plan/nginx.conf` ist das 1:1-Abbild der Live-Config `/etc/nginx/nginx.conf`
  (volle Datei, kein `sites-enabled`-Include — die Default-Site dort ist tot)

## Schritt 1: systemd-Dienst

Als root auf hetznerus:

```bash
cp /home/kiel/mapdeploy/mapmobile.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now mapmobile
systemctl status mapmobile
curl -s localhost:8080/healthz   # db_ok muss true sein
```

Kontrolle: `{"status":"ok","db_ok":true,...}`. Bei `db_ok:false` steht die
Ursache im Journal (`journalctl -u mapmobile`), meist falscher `MAP_DB`-Pfad —
die Karte startet trotzdem, nur Klicks liefern keine Details.

## Schritt 2: nginx (Koexistenz mit rs-summarizer)

Die Karte hängt als zwei Pfade am bestehenden 443er-Block — kein eigener
`server`, kein eigenes Zertifikat nötig (das rocketrecap.com-Zertifikat deckt
`/karte` automatisch mit ab). Pfade kollidieren nicht mit rs-summarizer
(`/, /browse, /search, /generations/*, /summaries/*, /static/*,
/process_transcript`):

```nginx
location = /karte { proxy_pass http://127.0.0.1:8080/map; ... }
location /api/map/ { proxy_pass http://127.0.0.1:8080; ... }
```

(Volltext: `mapmobile/deploy/mapmobile-nginx.conf`.)

Wichtigste Zeile: `proxy_set_header X-Forwarded-For $remote_addr` —
**überschreiben, nicht anhängen**. Die App liest den linken XFF-Eintrag;
mit `$proxy_add_x_forwarded_for` könnte jeder per eigenem Header das
Pro-Nutzer-Limit umgehen (live bewiesen und gefixt).

Aktivieren (lokal `plan/nginx.conf` pflegen, hochladen):

```bash
# auf hetznerus als root, vorher Backup:
cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak-20260927
# lokale Datei hochladen:
scp plan/nginx.conf root@hetznerus:/etc/nginx/nginx.conf
# auf hetznerus:
nginx -t && systemctl reload nginx
```

## Schritt 3: Verifikation (von außen)

```bash
curl -s https://rocketrecap.com/karte -o /dev/null -w "%{http_code}\n"  # 200
curl -s https://rocketrecap.com/api/map/points | python3 -c "..."       # 16692 Punkte
curl -s https://rocketrecap.com/ | md5sum     # unverändert (rs-summarizer lebt)
curl -s -o /dev/null -w "%{http_code}\n" https://jobs.rocketrecap.com/  # 200
```

Rate-Limit: 70 parallele Requests auf `/api/map/point/330` → danach `429`,
nach ~1 min wieder `200` (60 Burst, 1/s Refill). Per-Nutzer-Trennung sowie
Spoof-Schutz wurden bei der Installation live nachgewiesen.

## Updates

Neues Binary bauen (`cargo build --release` in `mapmobile/`), kopieren,
Dienst neu starten — Config-Dateien bleiben unangetastet:

```bash
scp mapmobile/target/release/mapmobile hetznerus:~/mapdeploy/bin/
ssh root@hetznerus systemctl restart mapmobile
```

## Fehlersuche

- `systemctl status mapmobile` / `journalctl -u mapmobile -f` — Startfehler, DB-Warnungen
- `nginx -t` — Syntax prüfen; `tail /var/log/nginx/error.log`
- `429` für alle Nutzer: `X-Forwarded-For`-Header fehlt (siehe Schritt 2)
- `502 Bad Gateway` auf `/karte`: Dienst läuft nicht (`systemctl start mapmobile`)
- Rollback nginx: `cp /etc/nginx/nginx.conf.bak-20260927 /etc/nginx/nginx.conf && nginx -t && systemctl reload nginx`
