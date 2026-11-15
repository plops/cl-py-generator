# Evidence — Phase 1 & 2, Lauf 2026-11-15 (Fakten, keine Prosa)

## 1. Build-Metadaten

| Item | Wert |
| :--- | :--- |
| `PORTAGE_TAG` | `20261115` (`docker manifest inspect` OK) |
| `STAGE3_TAG` | `nomultilib-20261109` (Manifest OK, `/sbin/openrc` im Image verifiziert) |
| Kernel | `KVER_PURE=6.18.63`, `gentoo-sources-6.18.63.ebuild` im Portage-Snapshot vorhanden, `config6.18.48` + `olddefconfig` |
| Image-Digest (`gentoo-z6-min-openrc:latest`) | `gentoo-z6-min-openrc@sha256:14fbf73bf587eb00e1c2424bcf12217bf713ae9567185847d0c19234cfd4b271` |
| Build-Läufe (aus `logs/setup01_run_*.meta`) | `084945`: 5071 s, exit 1 (variant-nv `/etc/hosts`); `101604`: 98 s, exit 0 (Cache-Rebuild nach Hosts-Fix); `103311`: 1099 s, exit 0 (final, mit Planer-Patches) |
| Git-Commits Phase 1.1 | `10fe0626 chore(openrc): .gitignore`, `6ff7c6b7 refactor(openrc): Archiv/Host-Referenzen`, `95efce48 fix(portage)`, `5bb7b326 fix(openrc)`, `7536a2b3 feat(docker)`, `5c623f77 docs(openrc): 2026-11-14` |
| Git-Phase-2-Stand | uncommitted: `Dockerfile`, `setup03_*`, `make.conf`, `package.use`, `world`, `install_on_*.md`; neu: `extract_local_binpkgs.py`, `emerge-binpkg`, `doc/binpkg_overlay_usage.md` (Commits folgen nach diesem Dokument) |
| Uncommitted gelassen | `collect.sh` (Root, User-Datei), `plan/20261106_02_slownet/wifirecon` (fremder Plan) |

## 2. Artefakte & Größen (neu, gemessen 2026-11-15 ~10:51 UTC)

Verzeichnis: `out_export/gentoo-z6-min-openrc_20261115/` (Phase-1-Bundle gleichen Datums überschrieben).

| Datei | Bytes | MiB |
| :--- | ---: | ---: |
| `gentoo.squashfs_nv` | 1601835008 | 1527.6 |
| `gentoo.squashfs_e14` | 1191858176 | 1136.6 |
| `vmlinuz` (6.18.63-gentoo-dist) | 20673024 | 19.7 |
| `initramfs_squash_sda1-x86_64.img` | 14130539 | 13.5 |
| `amd-uc.img` | 305664 | 0.3 |
| `binpkgs/` (150 `.gpkg.tar` = 150 Index-Einträge) | 2265556726 | 2160.4 |
| `packages.txt` | 44457 | — |
| `packages.tsv` | 16349 | — |
| `upstream_binpkgs.tsv` | 109923 | — |
| `upstream_binpkgs.txt` | 103671 | — |
| `IMAGE-BUILD-INFO.txt` | 569 | — |

Vergleich Phase 1 → Phase 2 (Binpkg-Export, protokolliert, nicht geraten):
Phase 1: `local_built=157, upstream_binhost=501, stage3_superseded=1,
stage3_only=1, skipped_genlop=2, skipped_virtual=54, local_mib=2379.3`.
Phase 2: `local_built=150, upstream_binhost=488, stage3_only=1,
skipped_genlop=0, skipped_virtual=53, local_mib=2160.4`
(150+488+1+53 = 692 VDB-Pakete; `skipped_genlop=0`, weil der Export jetzt
*vor* `genlop` läuft — `genlop`/`Date-Manip` existieren da noch nicht).
Alte Squashfs-Größen wurden nicht festgehalten (gleiches Datumsverzeichnis).

sha256 (Auszug): `packages.tsv 36f7d7a0a…`, `upstream_binpkgs.tsv
6160d0d5…`, `upstream_binpkgs.txt 3125deb8…`, `vmlinuz 7866af78…`,
`initramfs 3fe8fab3…`, `amd-uc.img f5071115…`
(vollständig: siehe Build-Protokoll; Squashfs-Hashes auf Anfrage via
`sha256sum out_export/…`).

Größte Pakete (`packages.txt`): `linux-firmware 1991 MiB (81 s)`,
`gentoo-sources 1475 MiB (100 s)`, `nvidia-drivers 1135 MiB (82 s)`,
`llvm 395 MiB (7 s)`, `signal-desktop-bin 364 MiB`, `firefox-bin 330 MiB
(65 s)`, `gcc 305 MiB (19 min 32 s)`.

## 3. Solver- & Feature-Nachweise

Preflight-Image `openrc-preflight-20261115:tmp`
(`PORTAGE_TAG=20261115`, `STAGE3_TAG=nomultilib-20261109`), danach gelöscht.
Befehl: `emerge -pv --backtrack=10 --autounmask-backtrack=n @world`
(Default-Backtracking, ohne `--backtrack=30`).

Original-Zeilen aus `/tmp/preflight_phase2_emerge.txt` (106 `[ebuild N]`):

```text
[ebuild  N     ] media-libs/mesa-26.1.8-r1::gentoo  USE="X llvm (opengl) proprietary-codecs zstd ..." CPU_FLAGS_X86="sse2" LLVM_SLOT="22 -20 -21" VIDEO_CARDS="radeonsi ..." 76004 KiB
[ebuild  N     ] sys-kernel/installkernel-70-r1::gentoo  USE="dracut -efistub -grub -refind -systemd -systemd-boot -ugrd -uki -ukify" 33 KiB
[ebuild  N     ] www-client/firefox-bin-160.0:rapid::gentoo  USE="-gmp-autoupdate (-selinux) -wayland" ... 88836 KiB
```

Geprüft und bestanden (`PREFLIGHT_SOLVE_OK`, Exit 0):
`x11-libs/libxkbcommon`-Kategorie (kein Backtrack-Alarm),
`media-libs/mesa` mit `llvm`, kein `NVPTX`/`+nvptx` im Output,
`installkernel[dracut,-grub]`, kein `sys-boot/grub`,
`www-client/firefox-bin` vorhanden, kein `www-client/firefox-[0-9]`.

Env-Override-Verhalten (gegen `openrc-builder-dbg:tmp` verifiziert):
`FEATURES="-buildpkg" portageq envvar FEATURES` entfernt nur `buildpkg`
(Diff: genau 1 Zeile); `PKGDIR=/tmp/qtest quickpkg ...` schreibt nach
`/tmp/qtest` (quickpkg beachtet `PKGDIR`-Env).

## 4. Fehlerprotokoll

### 4a. `extract_local_binpkgs.py`: Plan-Klassifikator war wirkungslos (Phase 1.2)

Befund Export-Versuch 1: `local_built=0, upstream_binhost=0,
stage3_only=662` → `SystemExit: FEHLER: 0 lokale Binpkgs gefunden!`.
Ursachen (alle am Builder verifiziert): Portage loggt `Compiling/Packaging`
(nie `Compiling/Merging`) bzw. `Merging Binary` (nie `Binary (`);
`BINPKGMD5` steht auf allen 716 VDB-Paketen (Portage schreibt es auch bei
`buildpkg` aus Quellbuilds); kein lokaler `Packages`-Index
(`/var/cache/binhost/gentoo` enthält nur 319 heruntergeladene `.gpkg.tar`).
Fix: letzte Merge-Art entscheidet allein; Upstream-Index wird vom `sync-uri`
geladen (22 MB, Netzwerk im Build ok), Fallback exakte Cache-Datei.
Danach: `local_built=157, upstream_binhost=501` (Phase 1).

### 4b. `dracut`: `nd_blk` fehlt (nicht fatal)

Original aus `logs/setup01_run_20261115_084945.log:25024-25025`
(Step `#98`, danach `DONE 4.7s`, Initramfs gebaut, Smoke grün):

```text
#98  2.215 dracut-install: Failed to find module 'nd_blk'
#98  2.216 dracut[E]: FAILED:  /usr/bin/dracut/dracut-install ... -m libnvdimm nd_pmem nd_btt nd_blk dax device_dax dax_pmem dax_pmem_core nd_e820
```

Verifiziert: kein `drivers/nvdimm/*blk*`, kein `CONFIG_ND_BLK` in
6.18.63-Sourcen/`.config` (Treiber upstream entfernt); übrige
NVDIMM/DAX-Treiber sind `=y` builtin. Fix (Planer): `nd_blk` ersatzlos aus
`--add-drivers` gestrichen. Danach keine `FAILED:`-Zeile mehr im finalen
Log (`103311`).

### 4c. `variant-nv`: `/etc/hosts` read-only (fatal, Exit 1)

Original aus `logs/setup01_run_20261115_084945.log:25388`
(`DOCKERFILE:471`, Build-Meta `exit_code=1`, Dauer 5071 s):

```text
#104 ERROR: ... recipes ... did not complete successfully: exit code: 1
```

Vom User aus dem Log gezogen:

```text
2.175 /bin/sh: line 1: /etc/hosts: Read-only file system
```

`printf ... > /etc/hosts` scheitert (Docker-Mount). Fix (Planer):
`hosts`-Zeilen aus beiden Variant-RUNs entfernt (nur
`/etc/conf.d/hostname` bleibt); stattdessen `mksquashfs`-Pseudo-File
`-p "/etc/hosts f 0644 0 0 cat /tmp/hosts.variant"` + `-e etc/hosts`
(+ `tmp/hosts.variant` im `-e`, damit es nicht ins Image fällt) sowie
`/etc/local.d/10-hosts.start` als Boot-Schutz im Target. Rebuild danach
grün (`101604`: 98 s, Exit 0).

### 4d. `emerge-binpkg` Hybrid-Modus braucht Keyring (Phase 1.3)

Ohne Keys: `Binary package is not usable ... NO_PUBKEY 2C44695DB9F6043D`
(realer `dev-lang/go`-Install im Container schlug fehl). Mit Builder-Keyring
als Seed: `go version go1.27.3`, danach Remove ok. Fix (Planer bestätigt):
`COPY --from=builder /etc/portage/gnupg` + Modi-Härtung (Dirs 0755,
Files 0644 — Verifikation läuft als `portage`-User). Die Modi-Härtung wurde
vom Agenten um File-Sockets/Locks-Bereinigung ergänzt.

### 4e. Docker-`-v`-Mounts brauchen Daemon-Pfade

`/workspace/...` ist im Daemon-Namespace unsichtbar (Bind-Mount
`/home/kiel/stage/...`); `-v "$(pwd)/..."` mountet stille leere Dirs.
Alle Container-Tests liefen daher mit `/home/kiel/stage/...`-Pfaden.
(Kein Code-Fix nötig; Test-Hinweis für Phase 2.3.)

## 5. Paket-Delta (bekannte Änderungen, kein Voll-Diff)

Alte `packages.tsv` wurde mit dem gleichnamigen Datumsverzeichnis
überschrieben; daher kein Zeilen-Diff, sondern grep-belegte Deltas im neuen
Stand (`packages.tsv`, 692 Pakete):

```text
llvm-core/llvm     395 MiB  (llvm ohne NVPTX: LLVM_TARGETS="X86 AMDGPU")
media-libs/mesa     26 MiB  (wieder mit llvm)
sys-kernel/installkernel   (neu in world; USE dracut,-grub per Preflight)
www-client/firefox-bin 330 MiB  (statt Quell-firefox; -gmp-autoupdate ok)
```

Negativ-Probe bestanden: kein `dev-lang/rust`, `dev-lang/go`,
`app-portage/gentoolkit`, `sys-boot/grub` in `packages.tsv`;
`nvidia.ko` liegt für `6.18.63-gentoo-dist` vor
(`usr/lib/modules/6.18.63-gentoo-dist/video/nvidia.ko` im Squashfs).

## 6. Testergebnisse (Original-Ausgaben, gekürzt)

### 6a. Phase 1.2 (alter Stand, Protokoll)

```text
FILES_OK
binpkgs files=157 index=157
COUNT_OK
NOGENLOP_OK
URLS_OK
```

### 6b. Phase 1.3 (alter Stand, `/tmp/p13_full2.log`, Exit 0)

```text
SEED_KEYS_OK
OFFLINE_OK
go version go1.27.3-X:nodwarf5 linux/amd64
BINPKG FLOW TEST: OK
```

(Test-Atom `app-misc/jq` lokal+installiert; Hybrid-Real-Install mit
Builder-Keyring als Seed. Bekannte Test-Skript-Fallen dokumentiert:
`qlist | grep -q` unter `pipefail` → SIGPIPE/Exit 141 trotz Treffer;
`--list`-Zeilen enthalten Versionen.)

### 6c. Phase 2.3 (neuer Stand, alle Exit 0)

Artefakte: `ARTIFACTS_OK` (11 Dateien), `PKGS_OK`, `NO_BDEPENDS_OK`,
`BINPKG_EXPORT_OK` (150/150, kein `genlop-*.gpkg.tar`).
Squashfs (`gentoo-z6-min-openrc:latest`, Daemon-Pfade):
`HOSTNAME_OK` (`hpz6`/`e14` in `conf.d/hostname` + `boot`-Runlevel),
beide `etc/hosts` exakt wie per `-p` geschrieben, `emerge-binpkg`,
`libnvme.so`, `nvidia.ko`, `etc/local.d/10-hosts.start` vorhanden,
`NO_STRAY_A_OK` (`unsquashfs -l`, prefix-bereinigt; `-ll` wrappt bei
80 Spalten und ist dafür ungeeignet).
Neues Image (`/tmp/phase23_full.log`):

```text
BUILTIN_WRAPPER_OK
TEST_PKG=app-misc/jq
OFFLINE_NEW_OK
go version go1.27.3-X:nodwarf5 linux/amd64
g++ (Gentoo 16.2.1_p20260926 p1) 16.2.1 20260926
NEW IMAGE BINPKG TEST: OK
```

(Ohne Keyring-Seed — Keys sind jetzt eingebaut.)
`install_on_e14.sh`: `bash -n` ok, kein `binpkgs`-Bezug.
