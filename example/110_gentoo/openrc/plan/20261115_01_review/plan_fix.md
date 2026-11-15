# Review & Ausführungsplan: Binpkg-Export, OverlayFS-Workflow, Rebuild (`2026-11-15`)

## Teil 1: Review des Ist-Stands (nach `plan_fix2.md`)

### 1.1 Was gut umgesetzt wurde und direkt weiterverwendet wird
- Die **2-Phasen-Architektur** (`stage3_base` → `builder` → `target` → `variant-nv`/`variant-e14` → `export`) funktioniert (`logs/setup01_run_20261115_051847.log`, `exit_code=0`, 8 Export-Artefakte in `out_export/gentoo-z6-min-openrc_20261115/`).
- `slstatus` baut korrekt gegen `config.h` (`BAT0`), `variant-e14` schützt `libnvme.so`, `rc_sys=""` und die Single-Owner-OpenRC-User-Session sind sauber umgesetzt, und `logs/last_build_args.env` hält die Export-Cache-Treffer stabil.

### 1.2 Gefundene Fehler und notwendige Verbesserungen

| Nr. | Befund im Ist-Stand | Maßnahme | Phase |
| :--- | :--- | :--- | :--- |
| **1** | **Uncommitteter Stand & fehlende `.gitignore`:** Alle Änderungen aus `plan_fix2.md` sind noch nicht committed; im Repo dürfen niemals `logs/`, `out_export/`, `*.squashfs*`, `vmlinuz`, `*.img`, `*.png` oder `binpkgs/` landen. `collect.sh` soll unberührt bleiben. | `openrc/.gitignore` anlegen und alle bisherigen Änderungen (inkl. `plan/20261114_01_review/`, strikt begrenzt auf `openrc/`, **ohne** `collect.sh`) in thematischen Conventional Commits sichern. | Phase 1.1 |
| **2** | **Binpkg-Export für OverlayFS fehlt noch (`quickpkg`-Vermischung im Cache):** Im aktuellen `builder`-Cache liegen in `/var/cache/binpkgs` sowohl **echte selbst gebaute** Pakete (`buildpkg`, ~158 Pakete + `xkeyboard-config`) als auch **`quickpkg`-Snapshots** (~270 Pakete aus Stage3 und Gentoo-Upstream-Binhost, plus ggf. `genlop`). Upstream-Pakete liegen dagegen getrennt in `/var/cache/binhost/*`. | **Phase 1.2 (cache-erhaltend):** `config/extract_local_binpkgs.py` filtert strikt über `/var/log/emerge.log` (`Compiling/Merging`) + `/var/db/pkg/*/*` (kein `BINPKGMD5`, kein `genlop`-only) **nur** echt gebaute `.gpkg.tar` heraus, erzeugt einen sauberen `binpkgs/Packages`-Index und speichert für alle Nicht-Lokal-Pakete eine `upstream_binpkgs.tsv`/`.txt` mit `sync-uri` + exakter URL aus `/var/cache/binhost/*/Packages`. Durch Anhängen der Stage `FROM builder AS export-binpkgs` ganz am **Ende** des `Dockerfile` bleiben `builder`, `target`, `variant-*` und `export` zu 100 % im Cache. | Phase 1.2 |
| **3** | **Nachinstallations-Workflow im laufenden OverlayFS:** Benötigte Build-Abhängigkeiten sollen später über `BINPKG_DIR` nachinstalliert und wieder entfernt werden können. | Wrapper `config/emerge-binpkg` (`/usr/local/sbin/emerge-binpkg`): prüft `BINPKG_DIR` (kein Auto-Mount), nutzt im Default **Online-Hybrid** (`PKGDIR=... FEATURES="-binpkg-request-signature" emerge -1 --usepkg ...`, sodass lokale unsignierte Pakete und signierte Upstream-Binpkgs zusammen funktionieren), bietet `--offline` (`-getbinpkg`) sowie `--remove` (`emerge -C` + `--depclean`). Wird in Phase 1.3 im Container gegen das bestehende Image getestet und in Phase 2 fest ins Squashfs eingebaut. | Phase 1.3 & 2.2 |
| **4** | **Ursache des `libxkbcommon`-Backtrack-Fehlers noch in `package.use`:** Zeile 17 in `config/etc/portage/package.use` lautet weiterhin **`media-libs/libxkbcommon X`** (falsche Kategorie; korrekt ist `x11-libs/libxkbcommon` [4.1.1]). Deshalb griff das Flag nicht und der Agent baute `--backtrack=30 --autounmask-backtrack=y` in `make.conf` ein. | Auf **`x11-libs/libxkbcommon X`** korrigieren und im Solver-Preflight nachweisen, dass `@world` auch mit Standard-Backtracking (`--backtrack=10`) auflöst. `--backtrack=30 --autounmask-backtrack=y` bleibt als Sicherheitsnetz in `make.conf`. | Phase 2.1 |
| **5** | **`mesa` wieder mit `llvm`, ohne `NVPTX` auf E14, und keine `*.a`-Dateien im Squashfs:** `media-libs/mesa -llvm` nimmt GPU-/Shader-Beschleunigung weg; `LLVM_TARGETS="X86 NVPTX AMDGPU"` baut unnötiges `NVPTX` mit; aktuell exkludiert `mksquashfs` nur `usr/lib64/libQt*.a`. | `media-libs/mesa llvm` aktivieren; `LLVM_TARGETS="X86 AMDGPU"` (weder E14 noch shared `llvm` brauchen `NVPTX`); in `variant-nv` und `variant-e14` vor `mksquashfs` alle statischen `*.a`-Archive außer glibc-Link-Stubs (`*_nonshared.a`) und GCC-internen (`/usr/lib*/gcc/**`) entfernen + sofortigen `g++`-Compile/Link-Smoke-Test ausführen. | Phase 2.1 & 2.2 |
| **6** | **Firefox, Hostname, gepinnte Basis-Tags, `installkernel` in `world`, Kernel `6.18.63`:** | `www-client/firefox-bin` aktivieren (`#www-client/firefox` als 1-Zeilen-Schalter in `world` belassen); Hostname je Variante (`hpz6` in `variant-nv`, `e14` in `variant-e14`, `hostname` im `boot`-Runlevel); `PORTAGE_TAG=20261115`, `STAGE3_TAG` auf `20261109` pinnen (vorher via `docker manifest inspect` prüfen); `sys-kernel/installkernel` in `world` + `dracut -grub` in `package.use`; `KVER_PURE=6.18.63` mit `config6.18.48` + `olddefconfig`; in `builder` `quickpkg` und `genlop` sauber von `/var/cache/binpkgs` trennen. | Phase 2.1 & 2.2 |
| **7** | **Report-Qualität (`walkthrough.md`):** Das bisherige `walkthrough.md` enthielt zu viel Ausschmückung und zu wenig harte Rohdaten als Grundlage für stärkere Modelle. | Zweiteilung in Phase 2.4: zuerst **`evidence.md`** (nur Fakten, Tabellen, Log-Belege, Diffs, Test-Outputs), danach **`walkthrough.md`** gemäß Prompt mit strikten Anti-Geschwafel-Regeln. | Phase 2.4 |

---

## Teil 2: Expliziter Ausführungsplan (für den Agenten)

> **Verbindliche Regeln für den Agenten:**
> 1. Arbeitsverzeichnis ist **`/workspace/src/cl-py-generator/example/110_gentoo/openrc`** (Host: `/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc`).
> 2. **Git-Grenze:** Das `.git`-Verzeichnis liegt ggf. oberhalb von `openrc/`. Führe **alle** Git-Befehle strikt auf `.` bezogen aus (`git status -- .`, `git add -- ...`). Verändere und committe **niemals** Dateien außerhalb von `openrc/` und **niemals** `collect.sh`.
> 3. **Nicht löschen:** `/tmp/openrc_cleanup_backup` und `out_export/` werden **nicht** gelöscht (erst nach Boot-Test auf dem E14 durch den Benutzer).
> 4. **Reihenfolge strikt einhalten:**
>    - **Phase 0 & Phase 1 (cache-erhaltend!)** zuerst komplett abschließen und verifizieren. In Phase 1 dürfen `builder`, `target`, `variant-nv`, `variant-e14` und `export` im `Dockerfile` um kein einziges Byte verändert werden!
>    - **Phase 2 (Neubau)** erst nach grünem Abschluss von Phase 1 starten.
> 5. Nach jedem Abschnitt alle **Verifikationskommandos** ausführen. Bei Fehler sofort stoppen und beheben.

---

### Phase 0: Ausgangszustand prüfen & sichern

```bash
set -euo pipefail
cd /workspace/src/cl-py-generator/example/110_gentoo/openrc

docker info >/dev/null
df -h /workspace/src /tmp
git rev-parse --show-toplevel
git status -s -- .

# Frisches Backup dieses Zwischenstands (altes /tmp/openrc_cleanup_backup bleibt unberuehrt!)
mkdir -p /tmp/openrc_20261115_backup
cp -a Dockerfile *.sh config/ /tmp/openrc_20261115_backup/
```

**Test:**
```bash
test -f /tmp/openrc_20261115_backup/Dockerfile
test -f logs/last_build_args.env
test -d out_export/gentoo-z6-min-openrc_20261115
```

---

### Phase 1: Git-Commit & Cache-erhaltender Binpkg-Export + Test

#### Phase 1.1: `.gitignore` anlegen & bisherigen Stand thematisch committen

1. Erstelle **`openrc/.gitignore`** mit exakt diesem Inhalt (`collect.sh` bleibt **unignoriert** und **uncommitted**, da der Benutzer es selbst pflegt):
   ```gitignore
   # Build- und Laufzeitlogs
   logs/

   # Exportierte Bundles und temporaere Arbeitsverzeichnisse
   out_export/
   out_test/
   *.tmp.*

   # Gebaute Images, Kernel, Initramfs und Microcode
   *.squashfs*
   *.erofs*
   *.img
   *.qcow2
   *.raw
   vmlinuz
   vmlinuz-*

   # Exportierte Binaerpaket-Caches und Manifeste
   binpkgs/
   Packages
   Packages.gz
   upstream_binpkgs*

   # Generierte Diagramme/Bilder
   *.png

   # Editor-/Backup-Reste
   *~
   *.bak-*
   .DS_Store
   ```
2. Prüfe, ob ignorierte Dateien (`logs/`, `*.png`, `out_export/`) bereits früher getrackt wurden — falls ja, nur aus dem Index entfernen (nicht von Disk löschen):
   ```bash
   tracked_ignored="$(git ls-files -ci --exclude-standard -- . || true)"
   if [ -n "${tracked_ignored}" ]; then
     printf '%s\n' "${tracked_ignored}" | xargs -r git rm --cached --
   fi
   ```
3. **Thematische Conventional Commits (strikt nur innerhalb von `openrc/`, ohne `collect.sh`):**
   Führe jeden Block aus; `git diff --cached --quiet` verhindert leere Commits, falls eine Gruppe bereits committed war:
   ```bash
   # 1. .gitignore
   git add -- .gitignore
   git diff --cached --quiet || git commit -m "chore(openrc): add .gitignore for build artifacts, logs, binpkgs, and plots"

   # 2. Struktur-Aufraeumen (Host-Referenzen, Archiv, Skripte)
   git add -A -- archive/ config/host-reference/ config/config-gentoo.sh config/resolv.conf
   git restore --staged -- collect.sh 2>/dev/null || true
   git diff --cached --quiet || git commit -m "refactor(openrc): archive obsolete scripts and organize host references"

   # 3. Portage-Konfiguration
   git add -A -- config/etc/portage/ config/var/lib/portage/world
   git diff --cached --quiet || git commit -m "fix(portage): clean up package.use, keywords, world, and solver opts"

   # 4. OpenRC / Session / Bringup
   git add -A -- config/xinitrc config/activate config/
   git diff --cached --quiet || git commit -m "fix(openrc): make xinitrc and activate host-aware and single-owner"

   # 5. Dockerfile & Build-/Export-Skripte
   git add -A -- Dockerfile setup01_build_image.sh setup03_copy_from_container.sh
   git diff --cached --quiet || git commit -m "feat(docker): switch to two-stage binpkg build and atomic export"

   # 6. Review-Plan und Reports 2026-11-14 (ohne *.png dank .gitignore)
   git add -A -- plan/20261114_01_review/
   git diff --cached --quiet || git commit -m "docs(openrc): archive 2026-11-14 review plan and walkthrough"

   # Restliche Aenderungen innerhalb openrc/ pruefen (ausser collect.sh!)
   git status -s -- .
   ```
   Falls `git status -s -- .` noch andere gewollte geänderte/neue Dateien (außer `collect.sh` und ignorierten Pfaden) zeigt, diese in einem sinnvollen Conventional Commit nachziehen.

**Test Phase 1.1:**
```bash
# Nur collect.sh darf ggf. noch als modified/untracked auftauchen
git status -s -- . | grep -vE '^\?\? collect\.sh$|^ M collect\.sh$' && { echo "UNCOMMITTED FILES REMAIN"; exit 1; } || true
! git ls-files -- . | grep -qE '(^logs/|^out_export/|\.png$|\.squashfs)'
git log -n 6 --oneline -- .
```

---

#### Phase 1.2: Cache-erhaltender Binpkg-Export (`extract_local_binpkgs.py`, `emerge-binpkg`, `setup03`)

1. Erstelle **`config/extract_local_binpkgs.py`**:
   Dieses Skript läuft im `builder`-Container. Es filtert streng nach **echt lokal kompilierten** Paketen (anhand `/var/log/emerge.log` + Fehlen von `BINPKGMD5` in `/var/db/pkg`, ohne `genlop`-only-Pakete und ohne `quickpkg`-Snapshots) und erstellt für alle übrigen installierten Nicht-Virtual-Pakete eine Upstream-Liste (`upstream_binpkgs.tsv` + `upstream_binpkgs.txt`) mit Download-URLs aus `/etc/portage/binrepos.conf/*.conf` und `/var/cache/binhost/*/Packages`:
   ```python
   #!/usr/bin/env python3
   """Exportiert nur echt lokal gebaute @world/BDEPEND-Binpkgs aus /var/cache/binpkgs
   und dokumentiert alle per Upstream-Binhost oder Stage3 bezogenen Pakete."""

   from __future__ import annotations

   import argparse
   import configparser
   import os
   import re
   import shutil
   from pathlib import Path


   def parse_emerge_log(log_path: Path) -> tuple[dict[str, str], set[str]]:
       """Liest /var/log/emerge.log und liefert:
       - last_mode[cpv]: 'source' (Compiling/Merging) oder 'binary' (Binary)
       - genlop_only_cpvs: CPVs, die erst nach '--oneshot app-portage/genlop' gemerged wurden.
       """
       last_mode: dict[str, str] = {}
       pre_genlop_cpvs: set[str] = set()
       post_genlop_cpvs: set[str] = set()
       in_genlop_phase = False

       re_cmd = re.compile(r"^\d+:\s+\*\*\*\s+emerge\s+(.*)$")
       re_compile = re.compile(r"^\d+:\s+===\s+\(\d+\s+of\s+\d+\)\s+Compiling/Merging\s+\(([^:]+)::")
       re_binary = re.compile(r"^\d+:\s+===\s+\(\d+\s+of\s+\d+\)\s+Binary\s+\(([^:]+)::")

       if not log_path.is_file():
           return last_mode, set()

       for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
           m_cmd = re_cmd.match(line)
           if m_cmd:
               args = m_cmd.group(1)
               if "app-portage/genlop" in args and "--oneshot" in args:
                   in_genlop_phase = True
               elif "x11-misc/xkeyboard-config" in args:
                   in_genlop_phase = False
               continue

           m_c = re_compile.match(line)
           if m_c:
               cpv = m_c.group(1)
               last_mode[cpv] = "source"
               (post_genlop_cpvs if in_genlop_phase else pre_genlop_cpvs).add(cpv)
               continue

           m_b = re_binary.match(line)
           if m_b:
               cpv = m_b.group(1)
               last_mode[cpv] = "binary"
               (post_genlop_cpvs if in_genlop_phase else pre_genlop_cpvs).add(cpv)

       genlop_only = post_genlop_cpvs - pre_genlop_cpvs
       return last_mode, genlop_only


   def read_binrepos(conf_dir: Path) -> list[dict[str, str]]:
       """Liest /etc/portage/binrepos.conf/*.conf sortiert nach absteigender Prioritaet."""
       repos: list[dict[str, str]] = []
       if not conf_dir.is_dir():
           return repos

       cp = configparser.ConfigParser()
       for p in sorted(conf_dir.glob("*.conf")):
           cp.read(p, encoding="utf-8")

       for sec in cp.sections():
           if sec == "DEFAULT":
               continue
           sync_uri = cp.get(sec, "sync-uri", fallback="").strip()
           if not sync_uri:
               continue
           location = cp.get(sec, "location", fallback=f"/var/cache/binhost/{sec}").strip()
           try:
               priority = int(cp.get(sec, "priority", fallback="0").strip())
           except ValueError:
               priority = 0
           repos.append({
               "name": sec,
               "sync_uri": sync_uri.rstrip("/"),
               "location": location,
               "priority": str(priority),
               "_prio": priority,
           })

       repos.sort(key=lambda r: int(r["_prio"]), reverse=True)
       return repos


   def parse_packages_index(index_path: Path) -> list[dict[str, str]]:
       """Parst eine Gentoo-Binhost-Packages-Datei in eine Liste von Stanza-Dicts."""
       if not index_path.is_file():
           return []
       stanzas: list[dict[str, str]] = []
       cur: dict[str, str] = {}
       for raw in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
           line = raw.strip()
           if not line:
               if "CPV" in cur:
                   stanzas.append(cur)
               cur = {}
               continue
           if ":" in line:
               k, v = line.split(":", 1)
               cur[k.strip()] = v.strip()
       if "CPV" in cur:
           stanzas.append(cur)
       return stanzas


   def stanza_url(sync_uri: str, stanza: dict[str, str]) -> str:
       rel = stanza.get("PATH", "").strip()
       if not rel:
           cpv = stanza["CPV"]
           cat, pf = cpv.split("/", 1)
           pn = re.sub(r"-[0-9].*$", "", pf)
           bid = stanza.get("BUILD_ID", "").strip()
           fmt = stanza.get("BINPKG_FORMAT", "gpkg").strip()
           ext = ".gpkg.tar" if fmt == "gpkg" else ".tbz2"
           if bid:
               rel = f"{cat}/{pn}/{pf}-{bid}{ext}"
           else:
               rel = f"{cat}/{pn}/{pf}{ext}"
       return f"{sync_uri}/{rel.lstrip('/')}"


   def find_local_binpkg(pkgdir: Path, cat: str, pf: str) -> Path | None:
       """Sucht passendes .gpkg.tar im PKGDIR; bei mehreren BUILD_IDs gewinnt die neueste."""
       cat_dir = pkgdir / cat
       if not cat_dir.is_dir():
           return None
       # Portage legt unter <cat>/<pn>/<pf>-<build_id>.gpkg.tar ab (oder direkt <cat>/<pf>.gpkg.tar)
       candidates: list[tuple[int, float, Path]] = []
       pattern = re.compile(rf"^{re.escape(pf)}(?:-(\d+))?\.gpkg\.tar$")
       for root_dir in [cat_dir] + [d for d in cat_dir.iterdir() if d.is_dir()]:
           for f in root_dir.iterdir():
               if not f.is_file():
                   continue
               m = pattern.match(f.name)
               if m:
                   bid = int(m.group(1)) if m.group(1) else 0
                   candidates.append((bid, f.stat().st_mtime, f))
       if not candidates:
           return None
       candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
       return candidates[0][2]


   def main() -> None:
       ap = argparse.ArgumentParser()
       ap.add_argument("--pkgdir", default="/var/cache/binpkgs")
       ap.add_argument("--vdb", default="/var/db/pkg")
       ap.add_argument("--emerge-log", default="/var/log/emerge.log")
       ap.add_argument("--binrepos-conf", default="/etc/portage/binrepos.conf")
       ap.add_argument("--out-binpkgs", required=True)
       ap.add_argument("--out-upstream-tsv", required=True)
       ap.add_argument("--out-upstream-txt", required=True)
       args = ap.parse_args()

       pkgdir = Path(args.pkgdir)
       vdb = Path(args.vdb)
       out_binpkgs = Path(args.out_binpkgs)
       out_tsv = Path(args.out_upstream_tsv)
       out_txt = Path(args.out_upstream_txt)

       if out_binpkgs.exists():
           shutil.rmtree(out_binpkgs)
       out_binpkgs.mkdir(parents=True, exist_ok=True)
       out_tsv.parent.mkdir(parents=True, exist_ok=True)

       last_mode, genlop_only = parse_emerge_log(Path(args.emerge_log))
       repos = read_binrepos(Path(args.binrepos_conf))

       # Upstream-Index laden: Map CPV -> Liste[(repo, stanza)] und CP -> Liste[(repo, stanza)]
       by_cpv: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
       by_cp: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
       for repo in repos:
           idx_path = Path(repo["location"]) / "Packages"
           for st in parse_packages_index(idx_path):
               cpv = st["CPV"]
               by_cpv.setdefault(cpv, []).append((repo, st))
               cat, pf = cpv.split("/", 1)
               pn = re.sub(r"-[0-9].*$", "", pf)
               by_cp.setdefault(f"{cat}/{pn}", []).append((repo, st))

       counts = {
           "local_built": 0,
           "upstream_binhost": 0,
           "stage3_superseded": 0,
           "stage3_only": 0,
           "skipped_genlop": 0,
           "skipped_virtual": 0,
           "missing_local_file": 0,
       }
       local_bytes = 0
       upstream_rows: list[tuple[str, str, str, str, str, str, str]] = []

       for cat_dir in sorted(p for p in vdb.iterdir() if p.is_dir()):
           cat = cat_dir.name
           for pkg_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
               pf = pkg_dir.name
               cpv = f"{cat}/{pf}"

               if cat == "virtual":
                   counts["skipped_virtual"] += 1
                   continue
               if cpv in genlop_only:
                   counts["skipped_genlop"] += 1
                   continue

               has_binpkgmd5 = (pkg_dir / "BINPKGMD5").is_file()
               mode = last_mode.get(cpv, "")
               is_locally_compiled = (mode == "source") and (not has_binpkgmd5)

               if is_locally_compiled:
                   src_file = find_local_binpkg(pkgdir, cat, pf)
                   if src_file is None:
                       counts["missing_local_file"] += 1
                       raise SystemExit(f"FEHLER: Lokal gebautes Paket ohne .gpkg.tar in {pkgdir}: {cpv}")
                   rel_sub = src_file.relative_to(pkgdir)
                   dst_file = out_binpkgs / rel_sub
                   dst_file.parent.mkdir(parents=True, exist_ok=True)
                   shutil.copy2(src_file, dst_file)
                   counts["local_built"] += 1
                   local_bytes += dst_file.stat().st_size
                   continue

               use_val = (pkg_dir / "USE").read_text(encoding="utf-8", errors="replace").strip() if (pkg_dir / "USE").is_file() else ""
               md5_val = (pkg_dir / "BINPKGMD5").read_text(encoding="utf-8", errors="replace").strip() if has_binpkgmd5 else ""
               pn = re.sub(r"-[0-9].*$", "", pf)
               cp = f"{cat}/{pn}"

               if cpv in by_cpv:
                   chosen_repo, chosen_st = by_cpv[cpv][0]
                   if md5_val:
                       for r, st in by_cpv[cpv]:
                           if st.get("MD5", "") == md5_val:
                               chosen_repo, chosen_st = r, st
                               break
                   url = stanza_url(chosen_repo["sync_uri"], chosen_st)
                   upstream_rows.append((
                       cpv,
                       "upstream_binhost",
                       chosen_repo["name"],
                       url,
                       chosen_st.get("SIZE", "0"),
                       chosen_st.get("SHA512", "-"),
                       use_val,
                   ))
                   counts["upstream_binhost"] += 1
               elif cp in by_cp:
                   chosen_repo, chosen_st = by_cp[cp][-1]
                   url = stanza_url(chosen_repo["sync_uri"], chosen_st)
                   upstream_rows.append((
                       cpv,
                       f"stage3_superseded_by_{chosen_st['CPV']}",
                       chosen_repo["name"],
                       url,
                       chosen_st.get("SIZE", "0"),
                       chosen_st.get("SHA512", "-"),
                       use_val,
                   ))
                   counts["stage3_superseded"] += 1
               else:
                   upstream_rows.append((cpv, "stage3_only", "-", "-", "0", "-", use_val))
                   counts["stage3_only"] += 1

       with out_tsv.open("w", encoding="utf-8") as fh:
           fh.write("cpv\torigin\tbinrepo\turl\tsize_bytes\tsha512\tuse\n")
           for row in upstream_rows:
               fh.write("\t".join(row) + "\n")

       with out_txt.open("w", encoding="utf-8") as fh:
           fh.write("Gentoo Builder Package Origin Summary\n")
           fh.write("=====================================\n")
           for k, v in counts.items():
               fh.write(f"{k}: {v}\n")
           fh.write(f"local_binpkgs_bytes: {local_bytes} ({local_bytes / (1024*1024):.1f} MiB)\n\n")
           fh.write(f"{'CPV':<45} {'ORIGIN':<32} {'BINREPO':<16} URL\n")
           for cpv, origin, repo, url, *_ in upstream_rows:
               fh.write(f"{cpv:<45} {origin:<32} {repo:<16} {url}\n")

       print(f"extract_local_binpkgs: {counts}, local_mib={local_bytes / (1024*1024):.1f}")
       if counts["local_built"] == 0:
           raise SystemExit("FEHLER: 0 lokale Binpkgs gefunden!")


   if __name__ == "__main__":
       main()
   ```
   ```bash
   python3 -m py_compile config/extract_local_binpkgs.py
   ```

2. Erstelle den OverlayFS-Wrapper **`config/emerge-binpkg`** (mit `--binpkg-dir` / `BINPKG_DIR`, Hybrid-Online als Default, `--offline`, `--remove`, `--list`, `--yes`):
   ```bash
   cat > config/emerge-binpkg <<'EOF'
   #!/usr/bin/env bash
   # Wrapper zum Nachinstallieren und Entfernen von Paketen (z. B. BDEPENDs) im
   # laufenden OverlayFS aus den exportierten lokalen Binpkgs + Upstream-Binhost.
   set -euo pipefail

   BINPKG_DIR="${BINPKG_DIR:-}"
   MODE="install"
   OFFLINE=0
   AUTO_YES=0
   EXTRA_OPTS=()
   ATOMS=()

   usage() {
     cat <<'USAGE'
   Verwendung:
     BINPKG_DIR=/pfad/zu/binpkgs emerge-binpkg [OPTIONEN] <paket...>
     emerge-binpkg --binpkg-dir /pfad/zu/binpkgs [OPTIONEN] <paket...>
     emerge-binpkg --remove [--yes] <paket...>
     BINPKG_DIR=/pfad/zu/binpkgs emerge-binpkg --list

   Optionen:
     --binpkg-dir DIR   Pfad zum binpkgs-Verzeichnis (alternativ Env BINPKG_DIR;
                        kein automatisches Mounten)
     --offline          Nur lokale Binpkgs nutzen:
                        FEATURES="-getbinpkg -binpkg-request-signature"
                        (Default ist Online-Hybrid: lokale PKGDIR ohne Signaturzwang
                        + Upstream-Binhost via binrepos.conf mit Signaturpruefung)
     --list             Listet alle lokal verfuegbaren Binpkgs im PKGDIR auf
     --remove           Entfernt angegebene Pakete aus dem OverlayFS (emerge -C)
                        und raeumt ungenutzte Abhaengigkeiten (emerge --depclean)
     --force            Erlaubt --remove auch fuer @world-Pakete
     -y, --yes          Kein interaktives Nachfragen beim Entfernen (--depclean)
     -p, --pretend      Nur Trockenlauf an emerge durchreichen
     -v, --verbose      Ausfuehrliche emerge-Ausgabe
     -h, --help         Diese Hilfe

   Beispiele:
     export BINPKG_DIR=/mnt/usb/gentoo-z6-min-openrc_20261115/binpkgs
     sudo -E emerge-binpkg dev-lang/rust-bin
     sudo -E emerge-binpkg --offline dev-build/cmake
     sudo emerge-binpkg --remove --yes dev-lang/rust-bin
   USAGE
   }

   FORCE_REMOVE=0
   while [ $# -gt 0 ]; do
     case "$1" in
       --binpkg-dir) BINPKG_DIR="${2:?--binpkg-dir braucht ein Argument}"; shift 2 ;;
       --offline)    OFFLINE=1; shift ;;
       --list)       MODE="list"; shift ;;
       --remove)     MODE="remove"; shift ;;
       --force)      FORCE_REMOVE=1; shift ;;
       -y|--yes)     AUTO_YES=1; shift ;;
       -p|--pretend|-v|--verbose|-pv|-vp) EXTRA_OPTS+=("$1"); shift ;;
       -h|--help)    usage; exit 0 ;;
       --)           shift; ATOMS+=("$@"); break ;;
       -*)           EXTRA_OPTS+=("$1"); shift ;;
       *)            ATOMS+=("$1"); shift ;;
     esac
   done

   resolve_pkgdir() {
     local d="${1:-}"
     if [ -z "${d}" ]; then
       echo "[FEHLER] BINPKG_DIR ist nicht gesetzt und --binpkg-dir wurde nicht angegeben." >&2
       echo "Hinweis: Binpkg-Verzeichnis wird nicht automatisch gemountet." >&2
       echo "Beispiel: export BINPKG_DIR=/pfad/zu/gentoo-z6-min-openrc_YYYYMMDD/binpkgs" >&2
       exit 1
     fi
     if [ -f "${d}/Packages" ]; then
       printf '%s' "${d}"
     elif [ -f "${d}/binpkgs/Packages" ]; then
       printf '%s' "${d}/binpkgs"
     else
       echo "[FEHLER] Kein gueltiges binpkgs-Verzeichnis (Packages-Index fehlt in: ${d})." >&2
       exit 1
     fi
   }

   if [ "${MODE}" = "list" ]; then
     PKGDIR_PATH="$(resolve_pkgdir "${BINPKG_DIR}")"
     awk '/^CPV:/ {cpv=$2} /^SIZE:/ {printf "%-48s %8.2f MiB\n", cpv, $2/1048576}' "${PKGDIR_PATH}/Packages" | sort
     exit 0
   fi

   if [ "${#ATOMS[@]}" -eq 0 ]; then
     echo "[FEHLER] Mindestens ein Paket-Atom angeben (siehe --help)." >&2
     exit 1
   fi

   if [ "${MODE}" = "remove" ]; then
     if [ "${FORCE_REMOVE}" -ne 1 ] && [ -f /var/lib/portage/world ]; then
       for a in "${ATOMS[@]}"; do
         if grep -qx "${a}" /var/lib/portage/world; then
           echo "[FEHLER] ${a} steht in /var/lib/portage/world. Abbruch (mit --force erzwingen)." >&2
           exit 1
         fi
       done
     fi
     emerge --unmerge "${EXTRA_OPTS[@]}" "${ATOMS[@]}"
     if [ "${AUTO_YES}" -eq 1 ]; then
       emerge --depclean "${EXTRA_OPTS[@]}"
     else
       emerge --depclean --ask "${EXTRA_OPTS[@]}"
     fi
     exit 0
   fi

   PKGDIR_PATH="$(resolve_pkgdir "${BINPKG_DIR}")"
   if [ "${OFFLINE}" -eq 1 ]; then
     FEATURES_ARG="-getbinpkg -binpkg-request-signature"
   else
     FEATURES_ARG="-binpkg-request-signature"
   fi

   exec env PKGDIR="${PKGDIR_PATH}" FEATURES="${FEATURES_ARG}" \
     emerge -1 --usepkg "${EXTRA_OPTS[@]}" "${ATOMS[@]}"
   EOF
   chmod 0755 config/emerge-binpkg
   bash -n config/emerge-binpkg
   ```

3. **Cache-erhaltende Erweiterung am ENDE des `Dockerfile`:**
   Hänge **ausschließlich ans Dateiende** von `Dockerfile` (hinter `COPY --from=target /IMAGE-BUILD-INFO.txt /`, ohne auch nur eine einzige frühere Zeile zu ändern!) folgende zwei Stages an:
   ```dockerfile
   # ==============================================================================
   # Stage 4: Export locally built binpkgs + upstream manifest (cache-preserving)
   # ==============================================================================
   FROM builder AS export-binpkgs
   COPY config/extract_local_binpkgs.py /tmp/extract_local_binpkgs.py
   RUN python3 /tmp/extract_local_binpkgs.py \
         --out-binpkgs /out_binpkgs/binpkgs \
         --out-upstream-tsv /out_binpkgs/upstream_binpkgs.tsv \
         --out-upstream-txt /out_binpkgs/upstream_binpkgs.txt \
    && rm -f /out_binpkgs/binpkgs/Packages /out_binpkgs/binpkgs/Packages.gz \
    && PKGDIR=/out_binpkgs/binpkgs emaint --fix binhost \
    && test -s /out_binpkgs/binpkgs/Packages \
    && test -s /out_binpkgs/upstream_binpkgs.tsv

   FROM scratch AS export-binpkgs-out
   COPY --from=export-binpkgs /out_binpkgs/ /
   ```

4. Aktualisiere **`setup03_copy_from_container.sh`** (exportiert sowohl `--target export` als auch `--target export-binpkgs-out` in `${TARGET}/`, steuerbar über `EXPORT_BINPKGS=1`, Platzcheck `MIN_GIB`, liest `logs/last_build_args.env` für 100 % Cache-Treffer):
   ```bash
   cat > setup03_copy_from_container.sh <<'EOF'
   #!/usr/bin/env bash
   set -euo pipefail

   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   cd "${SCRIPT_DIR}"

   TODAY="$(date +%Y%m%d)"
   EXPORT_BASE="${EXPORT_BASE:-/dev/shm}"
   EXPORT_BINPKGS="${EXPORT_BINPKGS:-1}"
   MIN_GIB="${MIN_GIB:-8}"
   mkdir -p "${EXPORT_BASE}"

   AVAIL_BYTES="$(df -B1 --output=avail "${EXPORT_BASE}" | tail -n 1 | tr -d ' ')"
   MIN_BYTES=$((MIN_GIB * 1024 * 1024 * 1024))
   if [ "${AVAIL_BYTES}" -lt "${MIN_BYTES}" ]; then
     echo "[FEHLER] ${EXPORT_BASE} hat nur ${AVAIL_BYTES} Bytes frei (< ${MIN_GIB} GiB)." >&2
     echo "Im Agent-Container z. B. setzen: EXPORT_BASE=${SCRIPT_DIR}/out_export bash $0" >&2
     exit 1
   fi

   if [ -f logs/last_build_args.env ]; then
     # shellcheck disable=SC1091
     . logs/last_build_args.env
   fi
   BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT:-unknown}"
   BUILD_REPOSITORY="${BUILD_REPOSITORY:-local}"
   BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

   TARGET="${EXPORT_BASE}/gentoo-z6-min-openrc_${TODAY}"
   TMP_TARGET="${TARGET}.tmp.$$"
   TMP_BINPKGS="${TARGET}.binpkgs.$$"

   rm -rf "${TMP_TARGET}" "${TMP_BINPKGS}"
   mkdir -p "${TMP_TARGET}"

   DOCKER_BUILDKIT=1 docker build \
     --target export \
     --build-arg BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT}" \
     --build-arg BUILD_REPOSITORY="${BUILD_REPOSITORY}" \
     --build-arg BUILD_DATE="${BUILD_DATE}" \
     --output "type=local,dest=${TMP_TARGET}" \
     --progress=plain \
     .

   if [ "${EXPORT_BINPKGS}" = "1" ]; then
     mkdir -p "${TMP_BINPKGS}"
     DOCKER_BUILDKIT=1 docker build \
       --target export-binpkgs-out \
       --build-arg BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT}" \
       --build-arg BUILD_REPOSITORY="${BUILD_REPOSITORY}" \
       --build-arg BUILD_DATE="${BUILD_DATE}" \
       --output "type=local,dest=${TMP_BINPKGS}" \
       --progress=plain \
       .
     mv "${TMP_BINPKGS}/binpkgs" "${TMP_TARGET}/binpkgs"
     mv "${TMP_BINPKGS}/upstream_binpkgs.tsv" "${TMP_TARGET}/upstream_binpkgs.tsv"
     mv "${TMP_BINPKGS}/upstream_binpkgs.txt" "${TMP_TARGET}/upstream_binpkgs.txt"
     rmdir "${TMP_BINPKGS}"
   fi

   rm -rf "${TARGET}"
   mv "${TMP_TARGET}" "${TARGET}"
   ls -lh "${TARGET}"
   if [ -d "${TARGET}/binpkgs" ]; then
     du -sh "${TARGET}/binpkgs"
   fi
   EOF
   chmod +x setup03_copy_from_container.sh
   bash -n setup03_copy_from_container.sh
   ```

5. **Cache-erhaltenden Export jetzt ausführen & prüfen:**
   ```bash
   EXPORT_BASE="$(pwd)/out_export" bash ./setup03_copy_from_container.sh
   OUT="$(pwd)/out_export/gentoo-z6-min-openrc_$(date +%Y%m%d)"
   ```

**Tests Phase 1.2:**
```bash
test -s "${OUT}/gentoo.squashfs_nv"
test -s "${OUT}/gentoo.squashfs_e14"
test -s "${OUT}/binpkgs/Packages"
test -s "${OUT}/upstream_binpkgs.tsv"
test -s "${OUT}/upstream_binpkgs.txt"

# Pruefen, dass Anzahl .gpkg.tar exakt der Anzahl CPV-Eintraege im Index entspricht
n_files=$(find "${OUT}/binpkgs" -name '*.gpkg.tar' | wc -l)
n_idx=$(grep -c '^CPV:' "${OUT}/binpkgs/Packages")
echo "binpkgs files=${n_files} index=${n_idx}"
test "${n_files}" -gt 0
test "${n_files}" -eq "${n_idx}"

# Kein genlop im lokalen Export, Upstream-URLs gueltig
! find "${OUT}/binpkgs" -name 'genlop-*.gpkg.tar' | grep -q .
grep -q '^.*	upstream_binhost	.*	https://' "${OUT}/upstream_binpkgs.tsv"
head -n 20 "${OUT}/upstream_binpkgs.txt"
```

---

#### Phase 1.3: Container-Test des `emerge-binpkg`-Flows (auf bestehendem `gentoo-z6-min-openrc:latest`)

Teste **alle drei Modi** (`--list`, `--offline`, Hybrid-Online + `--remove`) im ephemeren Container gegen das aktuelle `target`-Image (`gentoo-z6-min-openrc:latest`):

```bash
OUT="$(pwd)/out_export/gentoo-z6-min-openrc_$(date +%Y%m%d)"

docker run --rm \
  -v "${OUT}/binpkgs:/mnt/binpkgs:ro" \
  -v "$(pwd)/config/emerge-binpkg:/usr/local/sbin/emerge-binpkg:ro" \
  gentoo-z6-min-openrc:latest \
  bash -euo pipefail -c '
    # 1. Fehlerfall ohne BINPKG_DIR testen
    if emerge-binpkg dev-build/cmake 2>/dev/null; then
      echo "FEHLER: erwartete Abbruchmeldung ohne BINPKG_DIR"
      exit 1
    fi

    export BINPKG_DIR=/mnt/binpkgs

    # 2. --list pruefen
    emerge-binpkg --list | head -n 10

    # 3. --offline Test: lokal gebautes Blatt-Paket entfernen und strikt offline aus PKGDIR neu installieren
    TEST_PKG=$(emerge-binpkg --list | awk "/^(app-misc\/jq|dev-build\/cmake|dev-util\/pkgconf) / {print \$1; exit}")
    test -n "${TEST_PKG}"
    echo "=== Offline-Test mit ${TEST_PKG} ==="
    emerge --unmerge "${TEST_PKG}"
    emerge-binpkg --offline -p "${TEST_PKG}" | tee /tmp/offline_plan.txt
    grep -q "\[binary " /tmp/offline_plan.txt
    ! grep -q "\[ebuild " /tmp/offline_plan.txt
    emerge-binpkg --offline "${TEST_PKG}"

    # 4. Hybrid-Online Test: getbinpkg bleibt aktiv, lokale PKGDIR ohne Signaturzwang;
    #    echte Installation eines Upstream-BDEPENDs (z.B. dev-lang/go) + --remove Cleanup
    echo "=== Hybrid-Online Test (Pretend rust-bin + Real install/remove dev-lang/go) ==="
    emerge-binpkg -p dev-lang/rust-bin | tee /tmp/hybrid_rust.txt
    grep -q "\[binary " /tmp/hybrid_rust.txt
    emerge-binpkg dev-lang/go
    go version
    emerge-binpkg --remove --yes dev-lang/go
    ! command -v go >/dev/null
    echo "BINPKG FLOW TEST: OK"
  '
```
*(Hinweis: Falls in `upstream_binpkgs.tsv` kein `dev-lang/go` sondern `dev-lang/rust-bin` / ein anderes BDEPEND steht, oder im Offline-Test `cmake` in `binpkgs` fehlt, das jeweilige Test-Atom passend aus `emerge-binpkg --list` bzw. `upstream_binpkgs.tsv` wählen und im Protokoll dokumentieren.)*

> **Zwischen-Meilenstein für den Benutzer:** Sobald Phase 1.3 grün ist, liegen alle 8 Image-Artefakte **plus** `binpkgs/` und `upstream_binpkgs.*` in `out_export/gentoo-z6-min-openrc_<HEUTE>/`. Du kannst diesen Stand jetzt bereits mit `sudo bash ./install_on_e14.sh --src out_export/... --dry-run` (ohne `binpkgs/`) auf dem E14 testen, während Phase 2 läuft.

---

### Phase 2: Neubau-Änderungen (Hostname, Wrapper im Squashfs, `6.18.63`, `mesa +llvm`, `firefox-bin`, Gepinnte Tags)

#### Phase 2.1: Tags/Manifeste verifizieren, Portage-Config korrigieren & Solver-Preflight

1. **Docker-Tags und Kernel-Version im Voraus prüfen:**
   ```bash
   docker manifest inspect gentoo/portage:20261115 >/dev/null
   # Stage3 OpenRC-Tag fuer 20261109 pruefen (nomultilib-20261109 vs nomultilib-openrc-20261109)
   STAGE3_TAG_VERIFIED=""
   for tag in nomultilib-20261109 nomultilib-openrc-20261109 amd64-nomultilib-openrc-20261109; do
     if docker manifest inspect "gentoo/stage3:${tag}" >/dev/null 2>&1; then
       # Sicherstellen, dass es OpenRC ist (nicht systemd):
       if docker run --rm "gentoo/stage3:${tag}" test -x /sbin/openrc; then
         STAGE3_TAG_VERIFIED="${tag}"
         break
       fi
     fi
   done
   test -n "${STAGE3_TAG_VERIFIED}"
   echo "STAGE3_TAG_VERIFIED=${STAGE3_TAG_VERIFIED}"

   # Kernel 6.18.63 im Portage-Snapshot 20261115 pruefen
   docker run --rm gentoo/portage:20261115 \
     test -f /var/db/repos/gentoo/sys-kernel/gentoo-sources/gentoo-sources-6.18.63.ebuild
   ```

2. **`config/etc/portage/make.conf` aktualisieren:**
   - `LLVM_TARGETS` auf `"X86 AMDGPU"` ändern (kein `NVPTX` für E14 oder shared LLVM).
   - `--backtrack=30 --autounmask-backtrack=y` als Sicherheitsnetz behalten (wie gewünscht; im Solver-Test unten prüfen wir trotzdem gegen `--backtrack=10`, dass die USE-Flags selbst sauber sind!).
   ```bash
   sed -i 's/^LLVM_TARGETS=.*/LLVM_TARGETS="X86 AMDGPU"/' config/etc/portage/make.conf
   grep -qx 'LLVM_TARGETS="X86 AMDGPU"' config/etc/portage/make.conf
   ```

3. **`config/etc/portage/package.use` korrigieren:**
   - **`media-libs/libxkbcommon X` → `x11-libs/libxkbcommon X`** (Bugfix!).
   - **`media-libs/mesa -llvm` → `media-libs/mesa llvm`** (Hardware-/Shader-Beschleunigung über LLVM wieder aktiv!).
   - **`sys-kernel/installkernel dracut -grub`** ergänzen (kein `grub` im Image!).
   - **`www-client/firefox ...`** Flags behalten und klar kommentieren (für einfaches Zurückwechseln auf Source-Build in `world`).
   ```bash
   cat > config/etc/portage/package.use <<'EOF'
   x11-libs/libdrm video_cards_radeon
   media-libs/vulkan-loader X
   media-libs/libpulse X
   media-plugins/alsa-plugins pulseaudio
   sys-libs/minizip-ng compat
   x11-wm/dwm savedconfig -xinerama
   sys-fs/lvm2 lvm readline
   media-video/pipewire alsa dbus sound-server udev
   media-video/wireplumber dbus
   sys-fs/squashfs-tools xattr -debug -lz4 -lzma -lzo zstd
   sys-auth/elogind pam policykit
   sys-auth/polkit elogind
   sys-apps/dbus elogind
   app-text/xmlto text
   media-libs/libglvnd X
   x11-libs/libxkbcommon X
   app-editors/emacs -dbus -gmp gtk gui -libxml2 -xpm
   net-analyzer/wireshark -capinfos -captype -dftest -dumpcap -editcap -filecaps -mergecap -minizip -netlink -randpkt -randpktdump -reordercap -sharkd -text2pcap -tshark -udpdump -zstd
   app-text/mupdf -javascript -jpeg2k
   app-text/zathura-pdf-mupdf -javascript
   net-misc/freerdp X aad client -fuse
   app-text/zathura -man
   dev-vcs/git -gpg -nls -pcre -perl -python_single_target_python3_14 -webdav
   net-misc/mosh -utempter
   net-print/cups -dbus
   sys-power/tlp -ppd -rdw
   # Source-Firefox-Flags (greifen, sobald www-client/firefox in world aktiv ist):
   www-client/firefox -clang -dbus -gmp-autoupdate hwaccel -jumbo-build system-pipewire system-png -telemetry
   www-client/firefox-bin -gmp-autoupdate -telemetry
   dev-debug/gdb -debuginfod -python -python_single_target_python3_14
   media-video/mpv -egl -lua -uchardet vaapi -vulkan
   x11-base/xorg-server xorg
   app-misc/mc -edit -slang
   dev-debug/strace ncurses
   dev-libs/nss -utils
   net-dns/dnsmasq -dumpfile
   sys-apps/nvme-cli -json
   sys-apps/smartmontools -daemon
   app-admin/sudo -sendmail
   net-dialup/picocom -man
   sys-fs/btrfs-progs -man
   media-libs/mesa llvm
   sys-kernel/installkernel dracut -grub
   # Vom Solver gefordert (implizit durch obige Flags):
   # firefox[system-png] braucht libpng[apng]; mpv[vaapi] braucht ffmpeg[vaapi] + libva[X]
   media-libs/libpng apng
   media-video/ffmpeg vaapi
   media-libs/libva X
   EOF
   ```

4. **`config/var/lib/portage/world` aktualisieren:**
   - `www-client/firefox-bin` aktivieren, `#www-client/firefox` auskommentieren (einfaches Umschalten direkt durch Aus-/Einkommentieren dieser beiden Zeilen!).
   - `sys-kernel/installkernel` aufnehmen.
   ```bash
   sed -i 's|^#www-client/firefox-bin$|www-client/firefox-bin|' config/var/lib/portage/world
   sed -i 's|^www-client/firefox$|#www-client/firefox|' config/var/lib/portage/world
   grep -qx 'sys-kernel/installkernel' config/var/lib/portage/world \
     || printf 'sys-kernel/installkernel\n' >> config/var/lib/portage/world
   ```

5. **Portage-Solver-Preflight gegen `gentoo/portage:20261115` + verifiziertes `STAGE3_TAG`:**
   Prüft vor dem 2-h-Bau, dass:
   - `emerge -p` mit `--backtrack=10` (ohne `--backtrack=30`!) sauber auflöst,
   - `media-libs/mesa` `llvm` aktiv hat, `llvm-core/llvm` kein `NVPTX` baut,
   - `sys-kernel/installkernel` `dracut -grub` hat und **kein** `sys-boot/grub` gezogen wird,
   - `www-client/firefox-bin` ohne `clang` / `rust` und ohne USE-Konflikte auflöst.
   ```bash
   TEST_PREFLIGHT_IMG="openrc-preflight-20261115:tmp"
   docker build -q -t "${TEST_PREFLIGHT_IMG}" \
     --build-arg PORTAGE_TAG=20261115 \
     --build-arg STAGE3_TAG="${STAGE3_TAG_VERIFIED}" \
     -f - . <<'EOF'
   ARG PORTAGE_TAG
   ARG STAGE3_TAG
   FROM gentoo/portage:${PORTAGE_TAG} AS portage
   FROM gentoo/stage3:${STAGE3_TAG}
   COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo
   RUN eselect profile set default/linux/amd64/23.0/no-multilib
   COPY config/etc/portage/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
   RUN mkdir -p /etc/portage/package.mask
   COPY config/etc/portage/package.mask /etc/portage/package.mask/package.mask
   COPY config/etc/portage/make.conf /etc/portage/make.conf
   COPY config/etc/portage/package.use /etc/portage/package.use
   COPY config/var/lib/portage/world /var/lib/portage/world
   EOF

   docker run --rm "${TEST_PREFLIGHT_IMG}" bash -euo pipefail -c '
     # 1. Strenger Solver-Test mit Default-Backtrack 10 (beweist saubere USE-Flags!)
     grep -vE "^[[:space:]]*(#|$)" /var/lib/portage/world \
       | xargs -r -d "\n" emerge -pv --backtrack=10 --autounmask-backtrack=n \
       | tee /tmp/preflight_solve.txt

     # 2. Erwartete Flags/Pakete pruefen
     grep -E "media-libs/mesa.*llvm" /tmp/preflight_solve.txt
     ! grep -iE "llvm_targets_nvptx|\+nvptx" /tmp/preflight_solve.txt
     ! grep -q "sys-boot/grub" /tmp/preflight_solve.txt
     grep -E "sys-kernel/installkernel.*dracut.*-grub" /tmp/preflight_solve.txt
     grep -q "www-client/firefox-bin" /tmp/preflight_solve.txt
     ! grep -q "www-client/firefox-[0-9]" /tmp/preflight_solve.txt
   ' | tee /tmp/preflight_phase2_emerge.txt

   docker rmi -f "${TEST_PREFLIGHT_IMG}" >/dev/null || true
   ```
   *(Falls `firefox-bin` im Preflight-Output unbekannte USE-Flags wie `-gmp-autoupdate` meldet oder noch ein einzelnes Flag ergänzt werden muss: `config/etc/portage/package.use` hier direkt in Phase 2.1 anpassen und den Preflight wiederholen, bis `exit 0` steht.)*

---

#### Phase 2.2: `Dockerfile` & Dokumentation aktualisieren

1. Passe **`Dockerfile`** gezielt für den Neubau an (ersetze `STAGE3_TAG` durch den in Phase 2.1 verifizierten `20261109`-Tag):
   - **`ARG`s & Kernel:**
     - `ARG PORTAGE_TAG=20261115`
     - `ARG STAGE3_TAG=<STAGE3_TAG_VERIFIED>` (z. B. `nomultilib-20261109`)
     - `ENV KVER_PURE=6.18.63` (in `builder` **und** `target`!)
   - **`builder` sauber in echte Binpkgs vs. Target-Cache trennen:**
     Nach dem Konvergenz-`emerge -uDN @world`:
     1. Zuerst (bevor `quickpkg` und bevor `genlop` laufen!) `python3 /tmp/extract_local_binpkgs.py` auf `/var/cache/binpkgs` ausführen → erzeugt `/out_binpkgs/binpkgs` (reine lokal kompilierte `@world`/BDEPEND-Pakete) und `/out_binpkgs/upstream_binpkgs.{tsv,txt}`, anschließend dort `PKGDIR=/out_binpkgs/binpkgs emaint --fix binhost`.
     2. Kopiere `/var/cache/binpkgs` → `/tmp/target-binpkgs` und führe `quickpkg` + `xkeyboard-config`-Source-Rebuild ausschließlich mit `PKGDIR=/tmp/target-binpkgs` durch (Achtung: auch `xkeyboard-config` nach `/out_binpkgs/binpkgs` kopieren und dort den Index neu erzeugen, falls es lokal neu gebaut wurde; noch sauberer: den `xkeyboard-config`-Rebuild **vor** Schritt 1 direkt nach der Konvergenz durchführen, dann liegt es automatisch als echt lokal gebautes Binpkg in `/var/cache/binpkgs` **und** in `/out_binpkgs/binpkgs`!).
     3. Installiere `app-portage/genlop` mit `FEATURES="-buildpkg"` (`FEATURES="-buildpkg" emerge --ask=n --oneshot app-portage/genlop`), damit `genlop` niemals im Binpkg-Export landet.
     *(Die Stage `FROM builder AS export-binpkgs` in Phase 1.2 schrumpft dadurch auf das einfache Übernehmen von `/out_binpkgs` — oder prüft sie einfach noch einmal und liefert sie an `export-binpkgs-out` weiter!)*
   - **`target` ergänzen:**
     - `--mount=type=bind,from=builder,source=/tmp/target-binpkgs,target=/var/cache/binpkgs` verwenden.
     - `COPY config/emerge-binpkg /usr/local/sbin/emerge-binpkg` + `RUN chmod 0755 /usr/local/sbin/emerge-binpkg`.
     - Firefox-Komfort-Symlink (falls `firefox-bin` nur `/usr/bin/firefox-bin` installiert):
       ```dockerfile
       RUN if [ ! -e /usr/bin/firefox ] && command -v firefox-bin >/dev/null 2>&1; then \
             ln -sf "$(command -v firefox-bin)" /usr/local/bin/firefox; \
           fi
       ```
     - OpenRC-Boot-Runlevel: `rc-update add hostname boot` hinzufügen.
     - Smoke-Test vor den Varianten erweitern:
       - `hostname` im `boot`-Runlevel prüfen (`test -e /etc/runlevels/boot/hostname`),
       - `command -v emerge-binpkg >/dev/null` und `emerge-binpkg --help >/dev/null`,
       - `(command -v firefox >/dev/null || command -v firefox-bin >/dev/null)`,
       - `qlist -I | grep -qx 'sys-kernel/installkernel'`,
       - `! qlist -I | grep -qE '^(dev-lang/rust|dev-lang/go|app-portage/gentoolkit|sys-boot/grub)'`.
   - **`variant-nv` (HP Z6) erweitern (vor `mksquashfs`):**
     - Hostname setzen:
       ```dockerfile
       RUN printf 'hostname="hpz6"\n' > /etc/conf.d/hostname \
        && printf '127.0.0.1\thpz6 localhost\n::1\t\thpz6 localhost\n' > /etc/hosts \
        && grep -qx 'hostname="hpz6"' /etc/conf.d/hostname
       ```
     - Statische `*.a`-Dateien bereinigen + `g++`-Compile/Link/Run-Smoke-Test direkt danach:
       ```dockerfile
       RUN find /usr/lib /usr/lib64 -type f -name '*.a' \
             ! -name '*_nonshared.a' \
             ! -path '/usr/lib/gcc/*' \
             ! -path '/usr/lib64/gcc/*' \
             -delete \
        && printf '#include <iostream>\nint main(){std::cout << "cpp-ok\\n"; return 0;}\n' \
           | g++ -x c++ - -o /tmp/cpp_smoke \
        && /tmp/cpp_smoke | grep -qx 'cpp-ok' \
        && rm -f /tmp/cpp_smoke
       ```
   - **`variant-e14` (ThinkPad E14) erweitern (vor `mksquashfs`):**
     - Hostname setzen:
       ```dockerfile
       && printf 'hostname="e14"\n' > /etc/conf.d/hostname \
       && printf '127.0.0.1\te14 localhost\n::1\t\te14 localhost\n' > /etc/hosts \
       && grep -qx 'hostname="e14"' /etc/conf.d/hostname \
       ```
     - Statische `*.a`-Dateien bereinigen + `g++`-Smoke-Test (identisch zu `variant-nv`) und prüfen:
       `! ldd "$(command -v nvme)" | grep -q 'not found'` und `test -e /etc/runlevels/boot/hostname`.

2. **Dokumentation (`doc/binpkg_overlay_usage.md` + Links in `doc/install_on_e14.md` & `doc/install_on_hpz6.md`):**
   Lege `doc/binpkg_overlay_usage.md` an, das dokumentiert:
   - Exportstruktur (`binpkgs/`, `upstream_binpkgs.tsv`, `upstream_binpkgs.txt`, nicht automatisch in `install_on_e14.sh` kopiert).
   - Nutzung von `BINPKG_DIR` ohne Auto-Mount:
     ```bash
     export BINPKG_DIR=/pfad/zu/gentoo-z6-min-openrc_YYYYMMDD/binpkgs
     sudo -E emerge-binpkg --list
     sudo -E emerge-binpkg dev-lang/rust-bin            # Hybrid: lokal + Upstream-Binhost
     sudo -E emerge-binpkg --offline dev-build/cmake     # strikt lokal: FEATURES="-getbinpkg -binpkg-request-signature"
     sudo emerge-binpkg --remove --yes dev-lang/rust-bin # sauber aus OverlayFS entfernen
     ```
   - Wie man zwischen `www-client/firefox-bin` und `www-client/firefox` umschaltet (`config/var/lib/portage/world`, eine Zeile kommentieren/entkommentieren).
   Verlinke `doc/binpkg_overlay_usage.md` kurz am Ende von `doc/install_on_e14.md` und `doc/install_on_hpz6.md`.

**Test Phase 2.2 (Syntax vor Build):**
```bash
grep -E '^\s*COPY\s+' Dockerfile | grep -v -- '--from=' | awk '{print $2}' | sort -u | while read -r p; do
  test -e "${p}" || { echo "MISSING COPY SOURCE: ${p}"; exit 1; }
done
grep -q 'KVER_PURE=6.18.63' Dockerfile
grep -q 'PORTAGE_TAG=20261115' Dockerfile
grep -q 'hostname="hpz6"' Dockerfile
grep -q 'hostname="e14"' Dockerfile
grep -q 'emerge-binpkg' Dockerfile
```

---

#### Phase 2.3: Vollständiger Build, Export & End-to-End-Verifikation

1. **Build mit Log starten:**
   ```bash
   bash ./setup01_run_with_log.sh
   ```
   *(Prüfen: `EXIT_CODE=0` in der neuesten `logs/setup01_run_*.meta`.)*

2. **Export inkl. `binpkgs/` und Upstream-Manifest:**
   ```bash
   EXPORT_BASE="$(pwd)/out_export" bash ./setup03_copy_from_container.sh
   OUT="$(pwd)/out_export/gentoo-z6-min-openrc_$(date +%Y%m%d)"
   ```

3. **Artefakt-, Squashfs- und `*.a`-/Hostname-/LLVM-Checks:**
   ```bash
   for f in gentoo.squashfs_nv gentoo.squashfs_e14 vmlinuz \
            initramfs_squash_sda1-x86_64.img amd-uc.img \
            packages.txt packages.tsv IMAGE-BUILD-INFO.txt \
            binpkgs/Packages upstream_binpkgs.tsv upstream_binpkgs.txt; do
     test -s "${OUT}/${f}" || { echo "MISSING OR EMPTY: ${f}"; exit 1; }
   done

   grep -q '^sys-kernel/installkernel' "${OUT}/packages.tsv"
   grep -q '^www-client/firefox-bin' "${OUT}/packages.tsv"
   grep -q '^llvm-core/llvm' "${OUT}/packages.tsv"
   ! grep -qE '^(dev-lang/rust|dev-lang/go|app-portage/gentoolkit|sys-boot/grub)\b' "${OUT}/packages.tsv"

   # Squashfs-Stichproben (beachte Merged-Usr: usr/lib/modules, usr/lib64)
   docker run --rm -v "${OUT}:/out:ro" gentoo-z6-min-openrc:latest bash -euo pipefail -c '
     unsquashfs -cat /out/gentoo.squashfs_nv etc/conf.d/hostname | grep -qx '\''hostname="hpz6"'\''
     unsquashfs -cat /out/gentoo.squashfs_e14 etc/conf.d/hostname | grep -qx '\''hostname="e14"'\''
     unsquashfs -ll /out/gentoo.squashfs_nv etc/runlevels/boot | grep -q "hostname"
     unsquashfs -ll /out/gentoo.squashfs_e14 etc/runlevels/boot | grep -q "hostname"
     unsquashfs -ll /out/gentoo.squashfs_e14 usr/local/sbin | grep -q "emerge-binpkg"
     unsquashfs -ll /out/gentoo.squashfs_e14 usr/lib64 | grep -q "libnvme\.so"
     unsquashfs -ll /out/gentoo.squashfs_nv usr/lib/modules | grep -q "nvidia\.ko"

     # Keine unerlaubten *.a-Dateien in beiden Squashfs-Images (ausser *_nonshared.a und gcc/)
     for img in /out/gentoo.squashfs_nv /out/gentoo.squashfs_e14; do
       stray=$(unsquashfs -ll "${img}" usr/lib usr/lib64 \
         | awk "/\\.a$/ {print \$NF}" \
         | grep -vE "_nonshared\\.a$|^usr/lib(64)?/gcc/" || true)
       if [ -n "${stray}" ]; then
         echo "STRAY .a IN ${img}:"
         printf "%s\n" "${stray}"
         exit 1
       fi
     done
   '
   ```

4. **Binpkg-Nachinstallations-Flow auf dem NEUEN Image testen** (inkl. im Image eingebautem `/usr/local/sbin/emerge-binpkg` + `g++`-Check):
   ```bash
   docker run --rm -v "${OUT}/binpkgs:/mnt/binpkgs:ro" gentoo-z6-min-openrc:latest bash -euo pipefail -c '
     test -x /usr/local/sbin/emerge-binpkg
     export BINPKG_DIR=/mnt/binpkgs
     emerge-binpkg --list | tee /tmp/new_binpkg_list.txt | head -n 15

     # Offline-Install-Test mit einem Paket aus binpkgs/
     TEST_PKG=$(awk "/^(dev-build\/cmake|dev-util\/pkgconf|app-misc\/jq) / {print \$1; exit}" /tmp/new_binpkg_list.txt)
     test -n "${TEST_PKG}"
     emerge --unmerge "${TEST_PKG}"
     emerge-binpkg --offline "${TEST_PKG}"

     # Online-Hybrid + Remove Test mit dev-lang/go (oder rust-bin)
     emerge-binpkg dev-lang/go
     go version
     emerge-binpkg --remove --yes dev-lang/go
     ! command -v go >/dev/null
     echo "NEW IMAGE BINPKG TEST: OK"
   '
   ```

5. **`install_on_e14.sh` Preconditions prüfen** (kopiert explizit **kein** `binpkgs/` auf `/1p2`):
   ```bash
   bash -n install_on_e14.sh
   ! grep -q 'binpkgs' install_on_e14.sh
   ```

---

#### Phase 2.4: Strukturierte Rohdaten (`evidence.md`), `walkthrough.md` & finale Commits

1. **Zuerst `plan/20261115_01_review/evidence.md` schreiben (NUR harte Fakten, Tabellen, Original-Outputs — KEIN Prosa-Geschwafel!):**
   Das Dokument muss folgende Pflichtsektionen vollständig mit echten Daten aus diesem Lauf füllen:
   - **Sektion 1 — Build-Metadaten:** Exakter `PORTAGE_TAG`, `STAGE3_TAG` + Image-Digest (`docker inspect --format '{{index .RepoDigests 0}}'`), `KVER_PURE`, Build-Dauer aus `logs/setup01_run_*.meta`, Git-Commits (Phase 1.1 & Phase 2).
   - **Sektion 2 — Artefakt- & Größenvergleich:** Tabelle Alt (`20261115_05`) vs. Neu (alle 8 Artefakte in Bytes/MiB, `binpkgs/` Dateianzahl + Gesamt-MiB, Anzahl Zeilen nach `origin` in `upstream_binpkgs.tsv`, `sha256sum` der Export-Dateien).
   - **Sektion 3 — Solver- & Feature-Nachweise:** Exakte Ausgaben aus `/tmp/preflight_phase2_emerge.txt` (`--backtrack=10`, `x11-libs/libxkbcommon`, `mesa[llvm]`, `llvm` ohne `NVPTX`, `installkernel[dracut,-grub]`, `firefox-bin`, kein `grub`).
   - **Sektion 4 — Fehlerprotokoll (falls bei Phase 2 Fehler auftraten):** Pro Fehler zwingend: *Phase*, *exakter 5–10-zeiliger Log-Auszug*, *Root Cause*, *konkrete Änderung*, *Verifikations-Output*.
   - **Sektion 5 — Paket-Diff (`packages.tsv` alt vs. neu) & Top-Pakete:** Neu hinzugekommene, entfernte und im Volumen geänderte Pakete inkl. LLVM/Mesa-Größe und Firefox-Bauzeit-Ersparnis.
   - **Sektion 6 — Testergebnisse:** Gekürzte Original-Ausgaben der Tests für `emerge-binpkg` (`--list`, `--offline`, Hybrid + `--remove`), `unsquashfs`-Hostname (`hpz6`/`e14`), `*.a`-Freiheit und `g++`-Smoke-Test.

2. **Danach `plan/20261115_01_review/walkthrough.md` schreiben:**
   Halte dich strikt an folgende Regeln **ohne Ausnahme**:
   - **Sprache & Stil:** zwingend Deutsch; didaktisch, holt den Leser ab, flüssig lesbar. Abkürzungen bei erster Verwendung einführen und im Glossar erklären.
   - **Anti-Geschwafel-Regel (HART!):**
     - Keine erfundenen Witze, Metaphern oder literarischen Einleitungen.
     - Jede technische Aussage und jede Zahl muss exakt aus `evidence.md` oder verifizierten Logs/Artefakten stammen (nichts schätzen!).
     - Nur wirklich existierende Dateien verlinken; nichts aus vorherigen Läufen als neu verkaufen.
     - Diagramme auf maximal 2–3 präzise Mermaid-Diagramme beschränken (Architektur mit `binpkgs` + OverlayFS-Nachinstallation, Datenfluss `extract_local_binpkgs.py`).
     - Falls PNG-Diagramme erzeugt werden (z. B. Paketgrößen-/Bauzeitvergleich via Python/matplotlib in `plan/20261115_01_review/`), diese einbinden und sachlich erklären (bleiben dank `.gitignore` uncommittet).
   - **Pflichtstruktur:**
     1. Was exakt implementiert wurde (Phase 1 cache-erhaltender Binpkg-Export + Phase 2 Rebuild).
     2. Welche Stellen aufgrund von Tests geändert werden mussten (mit Log-Auszug + Ursache + Fix).
     3. Learnings und mögliche zukünftige Erweiterungen.
     4. Liste der neuen/geänderten/entfernten Pakete im Docker-Container (inkl. `installkernel`, `firefox-bin`, `mesa[llvm]` / `llvm`, `*.a`-Bereinigung).

3. **Phase-2-Änderungen thematisch committen (weiterhin ohne `collect.sh`, ohne `*.png`, `logs/`, `out_export/`):**
   ```bash
   git add -A -- config/extract_local_binpkgs.py config/emerge-binpkg setup03_copy_from_container.sh
   git diff --cached --quiet || git commit -m "feat(binpkg): export genuinely local binpkgs, upstream manifest, and overlay wrapper"

   git add -A -- config/etc/portage/ config/var/lib/portage/world
   git diff --cached --quiet || git commit -m "fix(portage): pin kernel/deps, fix libxkbcommon category, enable mesa llvm, switch to firefox-bin"

   git add -A -- Dockerfile doc/
   git diff --cached --quiet || git commit -m "feat(docker): pin stage3/portage, bump kernel 6.18.63, set variant hostnames, strip .a"

   git add -A -- plan/20261115_01_review/
   git diff --cached --quiet || git commit -m "docs(openrc): add 2026-11-15 evidence report and walkthrough"

   # Endkontrolle: nichts ausserhalb openrc/ geaendert, Backups & out_export/ bleiben erhalten!
   git status -s -- .
   test -d /tmp/openrc_cleanup_backup
   test -d /tmp/openrc_20261115_backup
   test -d out_export
   ```
