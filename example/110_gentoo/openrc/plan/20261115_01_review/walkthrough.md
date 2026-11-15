# Walkthrough: Binpkg-Export, OverlayFS-Nachinstallation und Rebuild (2026-11-15)

Diese Anleitung erklärt, was am Gentoo-OpenRC-Image geändert wurde, warum
einzelne Stellen nach Test-Fehlern umgebaut werden mussten und wie du die
Binpkgs im laufenden System benutzt. Alle Zahlen und Dateinamen stammen aus
[evidence.md](evidence.md) — dort stehen die Original-Logs, Hashes und
Testergebnisse.

## Glossar (Abkürzungen bei erster Verwendung eingeführt)

- **Binpkg (BINPKG):** fertig gebautes Binärpaket (`.gpkg.tar`), das Portage
  (Gentoos Paketmanager) ohne Kompilieren installieren kann.
- **PKGDIR:** Verzeichnis, in dem Portage gebaute Binpkgs ablegt
  (hier `/var/cache/binpkgs`).
- **Binhost:** Server mit fertigen Binpkgs (hier der offizielle
  Gentoo-Binhost `distfiles.gentoo.org`).
- **BDEPEND:** Build-Abhängigkeit — zum Kompilieren nötig, läuft aber nicht
  im fertigen Image (z. B. `dev-lang/go`).
- **`@world`:** die Menge aller explizit gewünschten Pakete
  (`config/var/lib/portage/world`).
- **VDB:** `/var/db/pkg`, Portages Datenbank aller installierten Pakete.
- **OverlayFS:** Schreibschicht über dem (nur lesbaren) Squashfs-Image im
  laufenden System.
- **Stage:** Bauabschnitt im `Dockerfile` (`builder` kompiliert, `target`
  installiert sauber neu, `variant-nv`/`variant-e14` erzeugen die Images).

## 1. Was implementiert wurde

### Phase 1 (cache-erhaltend, kein Neubau)

1. **`.gitignore`** und sechs thematische Conventional Commits sichern den
   Stand (`10fe0626` … `5c623f77`).
2. **`config/extract_local_binpkgs.py`** filtert im `builder`-Container streng
   nach *echt lokal kompilierten* Paketen und schreibt sie nach
   `out_export/gentoo-z6-min-openrc_20261115/binpkgs/` (150 Dateien,
   2160,4 MiB, Dateianzahl = Index-Einträge), plus
   `upstream_binpkgs.tsv`/`.txt` mit Download-URLs für alle übrigen
   488 Upstream-Pakete. Entscheidend ist allein der jeweils letzte
   Merge-Eintrag in `/var/log/emerge.log` (Abschnitt 2 erklärt warum).
3. **`config/emerge-binpkg`** (`/usr/local/sbin/emerge-binpkg` im Image)
   installiert Pakete aus `binpkgs/` nach: `--offline` strikt lokal,
   Default Hybrid (lokal + signierter Upstream-Binhost), `--remove`
   inkl. `--depclean` zum Entfernen. `BINPKG_DIR` wird nie automatisch
   gemountet — siehe [binpkg_overlay_usage.md](../doc/binpkg_overlay_usage.md).
4. Der Export läuft über eine ans `Dockerfile`-Ende angehängte Stage, sodass
   `builder`/`target`/`variant-*` zu 100 % im Cache blieben.

### Phase 2 (Neubau: `PORTAGE_TAG=20261115`, `STAGE3_TAG=nomultilib-20261109`, Kernel 6.18.63)

- `package.use` korrigiert (`media-libs/libxkbcommon` → `x11-libs/…`,
  `mesa` wieder mit `llvm`, `LLVM_TARGETS="X86 AMDGPU"`, `installkernel`
  mit `dracut -grub`), `world` auf `firefox-bin` + `installkernel` umgestellt.
  Ein Solver-Preflight mit Standard-Backtracking (`--backtrack=10`) löst
  `@world` sauber auf (106 `[ebuild N]`, kein `grub`).
- Der `builder` trennt jetzt sauber: `xkeyboard`-Rebuild → Export nach
  `/out_binpkgs` → Kopie nach `/tmp/target-binpkgs` + `quickpkg` dort →
  `genlop` mit `FEATURES="-buildpkg"`. Das Target installiert nur noch aus
  `/tmp/target-binpkgs`.
- Varianten setzen den Hostnamen (`hpz6`/`e14`, inkl. `boot`-Runlevel),
  entfernen alle statischen `*.a`-Archive (außer Glibc-Stubs und GCC-eigenen)
  und bestehen einen `g++`-Smoke-Test. `/etc/hosts` kommt per
  `mksquashfs`-Pseudo-File ins Image, `/etc/local.d/10-hosts.start` schützt
  zur Boot-Zeit vor veralteten Overlays.

```mermaid
flowchart LR
    B[builder: emerge @world + BDEPENDs] --> E[extract_local_binpkgs.py]
    E --> L[binpkgs/: nur lokal Gebautes]
    E --> U[upstream_binpkgs.tsv: URLs]
    B --> T[/tmp/target-binpkgs: + quickpkg-Snapshots]
    T --> I[target: saubere Neuinstallation]
    I --> V1[variant-nv: hpz6 + nvidia]
    I --> V2[variant-e14: E14 + Firmware-Minimum]
    L --> O[OverlayFS: emerge-binpkg --offline/--remove]
    U --> O
```

```mermaid
flowchart TD
    LOG[/var/log/emerge.log] --> P{letzter Merge-Eintrag je CPV?}
    P -->|Compiling/...| LOC[lokal: kopiere .gpkg.tar]
    P -->|Merging Binary| UP[upstream: URL aus Binhost-Index]
    P -->|kein Eintrag| ST[stage3-Basis: nur vermerken]
    VDB[/var/db/pkg] --> P
```

## 2. Was Tests erzwungen haben (Log-Auszug, Ursache, Fix)

**Klassifikator wirkungslos (Phase 1.2).** Erster Export: `local_built=0`.
Ursache: Portage loggt `Compiling/Packaging` (nie `Compiling/Merging`) und
`Merging Binary` (nie `Binary (`); `BINPKGMD5` steht auf *allen* Paketen
(Portage schreibt es auch bei `buildpkg` aus Quellbuilds). Fix: nur die
Log-Merge-Art zählt; Upstream-Index wird vom `sync-uri` geladen. Danach
157 (Phase 1) bzw. 150 (Phase 2) lokale Binpkgs.

**`dracut` meldet `Failed to find module 'nd_blk'`** (nicht fatal, Step
`DONE`, Initramfs ok). Ursache: Der Treiber existiert in Kernel 6.18 nicht
mehr (kein `drivers/nvdimm/*blk*`, kein `CONFIG_ND_BLK`). Fix auf
Planner-Empfehlung: `nd_blk` ersatzlos aus `--add-drivers` gestrichen —
die `FAILED:`-Zeile ist weg.

**`/etc/hosts` ist im Docker-Build read-only (fatal, Exit 1).**
`printf ... > /etc/hosts` brach `variant-nv` ab
(`/bin/sh: line 1: /etc/hosts: Read-only file system`). Fix auf
Planner-Empfehlung: `hosts`-Inhalt per `mksquashfs`-Pseudo-File
(`-p "/etc/hosts f 0644 …"`, echte Datei per `-e etc/hosts`
ausgeschlossen) ins Image schreiben und per `unsquashfs -cat` + `cmp`
verifizieren; `/etc/local.d/10-hosts.start` repariert zur Boot-Zeit alte
Overlays. Rebuild danach grün.

**Hybrid-Modus ohne Keyring tot.** Realer `emerge-binpkg dev-lang/go`
scheiterte mit `NO_PUBKEY 2C44695DB9F6043D`. Fix: `COPY --from=builder
/etc/portage/gnupg` ins Target plus Modi-Härtung (0755/0644 — Portage
verifiziert als unprivilegierter User). Danach `go1.27.3`-Install/Remove
grün, ohne Keyring-Seed.

Kleinigkeiten mit Protokollwert: Docker-`-v`-Mounts brauchen
Daemon-Pfade (`/home/kiel/stage/…`, sonst stille leere Dirs);
`unsquashfs -ll` bricht lange Zeilen bei 80 Spalten um — `.a`-Check läuft
darum mit `-l`; `qlist | grep -q` unter `pipefail` stirbt an SIGPIPE trotz
Treffer (Datei-Zwischenschritt verwenden).

## 3. Learnings und mögliche Erweiterungen

- **Env schlägt `make.conf`:** `FEATURES="-buildpkg"` bzw. `PKGDIR=…`
  als Env-Variablen wirken verlässlich (per `portageq envvar` und
  `quickpkg`-Probe bewiesen) — darauf bauen `genlop`-Schutz, Wrapper und
  Target-Bind-Mount.
- **Upstream-Index zur Build-Zeit laden** (statt lokal vorauszusetzen)
  machte den Export robust; der Binhost-Cache als Fallback blieb ungenutzt.
- **Mögliche Erweiterungen:** `firefox-bin` ↔ Quell-`firefox` per
  eine-Zeilen-Umschalter in `world` (dokumentiert); `binpkgs/`-Größe
  (2,1 GiB) bei Bedarf per USB statt `/1p2` transportieren
  (`install_on_e14.sh` kopiert es bewusst nicht); `evidence.md`-Muster
  (Fakten vor Prosa) für künftige Läufe beibehalten.

## 4. Paketänderungen im Container (Auswahl)

Neu: `sys-kernel/installkernel` (Kernel-Updates auf dem gebooteten System),
`www-client/firefox-bin` (330 MiB, 65 s statt ~20 min Quellbau).
Geändert: `media-libs/mesa` wieder mit `llvm`, `llvm-core/llvm` ohne
`NVPTX`. Entfernt/Abwesend (verifiziert): `dev-lang/rust`, `dev-lang/go`,
`app-portage/gentoolkit`, `sys-boot/grub` — BDEPENDs landen nicht mehr im
Image, sondern kommen bei Bedarf aus `binpkgs/` bzw. dem Upstream-Binhost.
Größte Brocken: `linux-firmware` (1991 MiB), `gentoo-sources` (1475 MiB),
`nvidia-drivers` (1135 MiB, inkl. `nvidia.ko` für 6.18.63).
