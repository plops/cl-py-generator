# Walkthrough: Umbau auf 2-Phasen-Binpkg-Build (`plan_fix2.md`)

## 1. Kurzfassung: Was hat sich geändert und was ist das Ergebnis?

Bisher wurde alles in einem einzigen Container-Rootfs kompiliert und Build-Werkzeuge (`rust`, `go`, `gentoolkit`) mussten nachträglich per `emerge -C` entfernt werden. Jetzt baut das `Dockerfile` in **zwei getrennten Stufen**:

1. **`builder`**: Kompiliert Kernel `6.18.60-gentoo-dist`, `slstatus` (mit `config.h`) und `@world` (aufgeteilt in 10 Layer-Chunks + Konvergenz) und erzeugt dabei für alle Pakete lokale Binärpakete in `/var/cache/binpkgs`.
2. **`target`**: Startet von einem frischen `stage3_base`, übernimmt nur `/boot`, `/lib/modules/6.18.60-gentoo-dist` und `/usr/local/bin/slstatus` aus `builder` und installiert `@world` per `emerge --usepkg --with-bdeps=n` direkt aus `/var/cache/binpkgs` (Bind-Mount). Dadurch bleiben reine Build-Abhängigkeiten (`rust`, `go`, `genlop`) komplett aus dem Zielsystem draußen, während gewünschte Entwickler-Tools (`gcc`, `g++`, `gdb`, `cmake`, `pkgconf`, `sbcl`) über `world` installiert werden.
3. **Varianten & Export (`variant-nv`, `variant-e14`, `export`)**: Aus dem geprüften `target` entstehen zwei getrennte Squashfs-Images (HP Z6 mit NVIDIA + Reverse-SSH bzw. ThinkPad E14 mit `iwd`/Minimal-Firmware und ohne NVIDIA). Die `export`-Stage sammelt alle 8 Release-Dateien via `docker build --target export --output` ein — ohne `docker run --privileged` und ohne `copy_files.sh`.

**Ergebnis des letzten erfolgreichen Laufs (`logs/setup01_run_20261115_051847.log`):**
- **Status / Dauer:** `exit_code=0`, 1045 s (~17 Min mit Layer-Cache)
- **Export-Verzeichnis:** `out_export/gentoo-z6-min-openrc_20261115/`
- **Inhalt (630 Pakete im Target):**
  - `gentoo.squashfs_nv` (1,5 GB), `gentoo.squashfs_e14` (1,1 GB)
  - `vmlinuz` (21 MB), `initramfs_squash_sda1-x86_64.img` (14 MB), `amd-uc.img` (299 KB)
  - `packages.txt` (44 KB), `packages.tsv` (16 KB), `IMAGE-BUILD-INFO.txt` (569 B)

---

## 2. Geänderte Dateien im Überblick

### Build & Export
- **`Dockerfile`**: Umbau auf `stage3_base` → `builder` → `target` → `variant-nv` / `variant-e14` → `export`.
  - `slstatus`: `COPY config/slstatus_config.h config.h` (vorher wirkungslos als `slstatus_config.h` kopiert) + `strings`-Check auf `BAT0`.
  - `variant-e14`: `emerge -C` + gezieltes `rm` für NVIDIA statt `rm -rf /usr/lib64/libnv*` (schützte `libnvme` für `nvme-cli`); `|| true` nach `mksquashfs` entfernt; `modules="amdgpu mt7921e"` und `openresolv`/`iwd`-Konfig nur noch hier.
  - `variant-nv`: `reverse-ssh-eu`/`-us` nur noch hier aktiviert; Firmware exakt über `modinfo -F firmware nvidia.ko` gefiltert.
  - Microcode: `/boot/amd-uc.img` wird aus `/usr/lib/firmware/amd-ucode/` via `cpio` mitgebaut und exportiert.
  - Smoke-Tests vor den Varianten: prüft `rc_sys=""`, Runlevels, 18 erwartete Binaries (u. a. `cmake`, `pkgconf`, `gdb`, `sbcl`), Abwesenheit von `rustc`/`cargo`/`go`/`gentoolkit`, Initramfs und `amd-uc.img`.
- **`setup01_build_image.sh`**: Ungenutzte/unsichere Build-Args (`INITRAMFS_AUTH_KEY`, `TUNNEL_KEY`) entfernt; speichert `BUILD_GIT_COMMIT`, `BUILD_REPOSITORY`, `BUILD_DATE` in `logs/last_build_args.env`.
- **`setup03_copy_from_container.sh`**: Nutzt `docker build --target export --output` mit `logs/last_build_args.env` (damit Cache-Treffer beim Export); `EXPORT_BASE` konfigurierbar (Default `/dev/shm` mit 6-GiB-Check, im Container z. B. `out_export`).

### Portage-Konfiguration
- **`config/var/lib/portage/world`**: `dev-build/cmake` und `dev-util/pkgconf` ergänzt (damit sie trotz `--with-bdeps=n` im Target landen); `app-portage/gentoolkit` entfernt.
- **`config/etc/portage/make.conf`**: `EMERGE_DEFAULT_OPTS` um `--backtrack=30 --autounmask-backtrack=y` erweitert (verhindert vorzeitigen Solver-Abbruch bei `x11-libs/libxkbcommon`).
- **`config/etc/portage/package.use`**: Doppelte Zeilen (`lvm2`, `firefox`) zusammengeführt; `media-libs/mesa -llvm` (vorab per Solver im Container geprüft) sowie vom Solver geforderte Flags (`media-libs/libpng apng`, `media-video/ffmpeg vaapi`, `media-libs/libva X`) ergänzt.
- **`config/etc/portage/package.accept_keywords`**: Totes `sys-kernel/dracut-crypt-ssh ~amd64` entfernt.

### Runtime-Skripte & Repo-Struktur
- **`config/xinitrc`**: `xrandr` nur noch bei erkanntem `DP-0 connected` + `DP-2 connected` (HP Z6); `~/start-pipewire.sh &` entfernt (OpenRC `user.kiel` ist Single Owner); `dbus-run-session` nur als Fallback.
- **`config/activate`**: `STAGE_ROOT` konfigurierbar; `env-update`/`ldconfig` nur noch, wenn `/IMAGE-BUILD-INFO.txt -nt /etc/ld.so.cache`; `tlp` nur auf E14 via `rc-service`.
- **Aufgeräumt**: Host-Referenzen nach `config/host-reference/hpz6/`, Altskripte (`copy_files.sh`, `nethex`, `NetworkManager.conf`, `start-reverse-tunnels.sh`) nach `archive/`, `config/config-gentoo.sh` pfadunabhängig gemacht.

---

## 3. Probleme während des Builds und konkrete Fixes

Beim Durchlauf von Phase 6 traten 8 konkrete Fehler auf, die direkt behoben wurden:

1. **Builder verweigerte eigene Binärpakete (`binpkg appears unsigned`)**
   - *Ursache:* `make.conf` verlangt `binpkg-request-signature`, im Builder gibt es aber keinen Signierschlüssel für frisch gebaute Pakete (`buildpkg`).
   - *Fix:* Im Builder `-binpkg-request-signature` an `make.conf` angehängt. Die finale `make.conf` im Target behält die Signaturprüfung; `target` umgeht sie nur beim lokalen `emerge`-Aufruf per `FEATURES="-binpkg-request-signature -getbinpkg"`.

2. **`make install` scheiterte mit `Cannot find LILO`**
   - *Ursache:* Das ungepinnte Rolling-Stage3 (`gentoo/stage3:nomultilib`) enthält kein `installkernel` mehr.
   - *Fix:* `emerge -1 sys-apps/portage sys-kernel/installkernel` im Builder vor dem Kernelbau.

3. **Syntaxfehler beim Aufteilen von `world`**
   - *Ursache:* Im `awk`-Einzeiler fehlte beim Übertragen ein schließender `/` im Regex (`^[[:space:]]*(#|$)`).
   - *Fix:* Schrägstrich ergänzt.

4. **`sys-power/tlp` brach in `pkg_setup` ab**
   - *Ursache:* `sys-power/tlp -ppd -python_single_target_python3_14 -rdw` in `package.use` schaltete das einzige verfügbare Python-Target ab. Der vorherige `emerge -p`-Trockenlauf prüft keine `pkg_setup`-Phasen.
   - *Fix:* `-python_single_target_python3_14` bei `tlp` entfernt.

5. **`target` fand keine Binärpakete für Stage3-/Binhost-Pakete**
   - *Ursache:* Nur lokal gebaute Pakete lagen in `/var/cache/binpkgs`; vorinstallierte Stage3-Pakete und vom Upstream-Binhost gezogene Pakete fehlten dort.
   - *Fix:* Im Builder ergänzt: alle installierten Nicht-Virtual-Pakete ohne lokales Binpkg werden per `quickpkg --include-config=y` nachgepackt. Zudem muss vor `emaint --fix binhost` der alte `Packages`-Index gelöscht werden (`rm -f Packages Packages.gz`), da `emaint` inkrementell 3 Pakete (`acct-user/messagebus`, `ntp`, `polkitd`) ausgelassen hatte.

6. **`preinst`-Abbruch bei `quickpkg`-Snapshots (`nvidia-drivers`, `xkeyboard-config`)**
   - *Ursache:* Beide Ebuilds erwarten in `pkg_preinst` Dateien aus dem Build-Layout (`supported-gpus.json` bzw. `xkb.workaround`), die ein nachträglicher `quickpkg`-Snapshot des installierten Zustands nicht mehr enthält.
   - *Fix:*
     - `quickpkg` läuft nur noch für Pakete **ohne** vorhandenes Buildzeit-Binpkg (dadurch gewinnt bei `nvidia-drivers` das Buildzeit-Binpkg).
     - `x11-misc/xkeyboard-config` (kam als Binhost-Paket ohne lokales Buildzeit-Binpkg) wird einmalig mit `FEATURES="-getbinpkg" emerge -1 --buildpkg` aus Quelltext in ein leeres `PKGDIR` gebaut und nach `/var/cache/binpkgs` kopiert.

7. **`--usepkgonly` scheiterte an `@system`-Virtuals**
   - *Ursache:* Virtuelle Pakete (`virtual/ssh` usw.) besitzen kein Binpkg und sind unter `--usepkgonly` unlösbar.
   - *Fix:* `target` verwendet `--usepkg --with-bdeps=n` statt `--usepkgonly`. Trockenlauf bestätigt: Nur 16 `virtual/*`-Pakete werden als `[ebuild]` aufgelöst (installieren keine Dateien), alle echten Pakete kommen aus `/var/cache/binpkgs`.

8. **Smoke-Test scheiterte an `/boot/vmlinuz`**
   - *Ursache:* `sys-kernel/installkernel` legt nur versionierte Dateien (`vmlinuz-6.18.60-gentoo-dist`) an, kein generisches `/boot/vmlinuz`.
   - *Fix:* In `target` (um den `builder`-Cache nicht zu entwerten) `cp /boot/vmlinuz-${KVER_RELEASE} /boot/vmlinuz` ergänzt.
   - *Nebenbefund beim Squashfs-Check:* Wegen Merged-Usr muss `unsquashfs` auf `usr/lib/modules` statt `lib/modules` prüfen (`lib` ist nur ein Symlink).

---

## 4. Verifikation & offene Punkte

- **Geprüft:** Alle 8 Export-Dateien vorhanden; `cmake`, `pkgconf`, `gdb`, `sbcl` in `packages.tsv` enthalten; `rust`, `go`, `gentoolkit` fehlen im Target; Stichproben auf `gentoo.squashfs_e14` (`libnvme.so`, `iwd`, `modules`, kein `nvidia-smi`, kein `reverse-ssh`) und `gentoo.squashfs_nv` (`reverse-ssh`, `nvidia.ko`) bestanden.
- **Ausstehend auf echter Hardware (ThinkPad E14):**
  ```bash
  sudo bash ./install_on_e14.sh --src out_export/gentoo-z6-min-openrc_20261115 --dry-run
  ```
- **Für zukünftige Builds offen:**
  - `sys-kernel/installkernel` in `config/var/lib/portage/world` aufnehmen (damit auch Kernel-Updates auf dem gebooteten System funktionieren).
  - `STAGE3_TAG` pinnen (`nomultilib` ist Rolling), um unerwartete Stage3-Änderungen zu vermeiden.
  - `/tmp/openrc_cleanup_backup` und `out_export/` erst nach erfolgreichem Boot-Test auf dem E14 löschen.
