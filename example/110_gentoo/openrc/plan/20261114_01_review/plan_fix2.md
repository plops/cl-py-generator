Hier ist das vollständige Review (abgeglichen mit allen Antworten und deinem Runner-`Dockerfile`) sowie der finale, schrittweise ausführbare Plan für den Agenten in `/workspace/src/cl-py-generator/example/110_gentoo/openrc`.

---

## Teil 1: Vollständiges Review & Zielarchitektur

### 1.1 Gefundene Bugs, Widersprüche und Altlasten

| Bereich | Konkreter Befund | Maßnahme |
| :--- | :--- | :--- |
| **`slstatus` (`Dockerfile`)** | `COPY config/slstatus_config.h .` kopiert nach `slstatus_config.h`; das `Makefile` baut aber aus `config.h` (Standard-`config.def.h`). Deine Konfiguration wird ignoriert. | `COPY config/slstatus_config.h config.h` + Smoke-Test (`strings /usr/local/bin/slstatus \| grep -q 'BAT0'`). |
| **E14-Stripping (`Dockerfile`)** | `rm -rf /usr/lib64/libnv*` löscht auch `/usr/lib64/libnvme*.so*` (`sys-libs/libnvme`) und macht `nvme` (`sys-apps/nvme-cli`) unbrauchbar. | Sauber per `emerge -C x11-drivers/nvidia-drivers dev-util/nvidia-cuda-toolkit` + gezieltem `rm -rf` auf `libnvidia*`, `libnvcuvid*`, `libnvoptix*`, `libcuda*` strippen + `ldd $(command -v nvme)`-Check. |
| **Fehlerunterdrückung (`Dockerfile`)** | `|| true` am Ende des E14-`mksquashfs`-Blocks verschluckt abgebrochene Builds. | Entfernen; beide `mksquashfs`-Schritte müssen hart mit `test -s` geprüft werden. |
| **2-Phasen-Build (`BDEPEND`-Trennung)** | Bisher wird alles im selben Rootfs gebaut; Build-Deps (`rust`, `go`, `clang`, `genlop`) verschmutzen das System oder werden nachträglich unsauber per `emerge -C` entfernt (`gentoolkit` steht sogar in `world`, wird aber gelöscht). | Umbau auf **2-Phasen-Binpkg-Architektur** (`stage3_base` → `builder` mit `FEATURES="buildpkg"` → `target` via `--usepkgonly --with-bdeps=n`). |
| **Dev-Tools (`world`)** | `dev-build/cmake` und `dev-util/pkgconf` stehen nicht in `world`. Bei `--with-bdeps=n` würden `BDEPEND`-Tools wie `cmake` nicht im Zielsystem landen. `app-portage/gentoolkit` steht noch in `world`, soll aber raus. | `dev-build/cmake` und `dev-util/pkgconf` in `world` aufnehmen; `app-portage/gentoolkit` entfernen. `rust`/`go` bleiben draußen. |
| **`llvm` vs. AMD GPU (`package.use`)** | `media-libs/mesa` unterstützt für `radeonsi` (ACO) und RADV grundsätzlich Builds ohne `llvm`. Da `firefox`, `mpv[vaapi]` oder andere Pakete aber ggf. Nebenbedingungen haben, darf `-llvm` nicht blind gesetzt werden. | In Phase 2 im `gentoo/portage:20261106`-Container per `emerge -pv` prüfen, ob `media-libs/mesa -llvm` sauber auflöst, und nur dann aktivieren. |
| **Portage `package.use`** | Doppelte Einträge für `sys-fs/lvm2` (`lvm` vs. `lvm readline`) und `www-client/firefox` (`-clang ...` vs. `hwaccel ...`). | Zusammenführen (`lvm readline`; Firefox mit `-clang` **und** `hwaccel`). `app-text/mupdf -javascript -jpeg2k` bleibt minimal für `zathura`. |
| **D-Bus & PipeWire (Single Owner)** | `Dockerfile` kommentiert `pipewire`/`pipewire-pulse`/`wireplumber` im User-Runlevel aus, `config/user-runtime.initd` legt sie beim Boot dennoch an, und `config/xinitrc` startet zusätzlich `~/start-pipewire.sh &` sowie `exec dbus-run-session dwm`. | OpenRC (`user.kiel` + `user-runtime`) als **Single Owner**: alle 4 User-Links im `Dockerfile` aktivieren, `~/start-pipewire.sh &` aus `xinitrc` entfernen (`~/start-pipewire.sh` bleibt als manuelles Notfall-Skript liegen), `dbus-run-session` nur als Fallback. |
| **`xrandr` (`config/xinitrc`)** | Hartkodiert auf HP Z6 (`DP-0`, `DP-2`), während ThinkPad E14 `eDP` / `DisplayPort-0..5` hat. | Nur anwenden, wenn `DP-0 connected` **und** `DP-2 connected` in `xrandr --query` vorkommen. |
| **Boot-Overlay-Drift (`config/activate`)** | `activate` ruft bei jedem Boot `env-update`, `ldconfig` und `depmod -a` auf (schreibt unnötig ins persistente Overlay) und `/etc/init.d/tlp start` direkt (auch auf Z6). | `env-update`/`ldconfig` nur noch, wenn `/IMAGE-BUILD-INFO.txt -nt /etc/ld.so.cache`; `tlp` nur auf E14 per `rc-service`; `STAGE_ROOT` konfigurierbar machen (`/home/kiel/stage` bzw. `/workspace/src`). |
| **Secrets & Initramfs-Reste** | `setup01_build_image.sh` übergibt ungenutzte `INITRAMFS_AUTH_KEY` und `TUNNEL_KEY` (`~/.ssh/tinyus_key.pem`). `dracut-crypt-ssh` steht noch in `package.accept_keywords`. | Aus `setup01_build_image.sh` und `package.accept_keywords` entfernen; `config/start-reverse-tunnels.sh` archivieren. |
| **Host-spezifische Defaults** | `reverse-ssh-eu`/`-us` und `modules="amdgpu mt7921e"` stehen im gemeinsamen Basis-Image. | `reverse-ssh-*` nur in `variant-nv` aktivieren; `modules="amdgpu mt7921e"` nur in `variant-e14` schreiben. |
| **Microcode & Export** | `amd-uc.img` wird nicht exportiert (`install_on_e14.sh` muss von alten Slots kopieren). `copy_files.sh` referenziert tote `/gentoo.squashfs` und `/gentoo.ext4` und nutzt `docker run --privileged` + `chmod -R a+rwx`. | `/boot/amd-uc.img` im Build via `cpio` aus `/usr/lib/firmware/amd-ucode/` erzeugen; Export über `FROM scratch AS export` + `docker build --target export --output`. `copy_files.sh` archivieren. |
| **`Dockerfile`-Altlasten** | Wirkungsloses `RUN env-update && source /etc/profile`, wirkungsloses `mkdir /run/user/1000`, `sed` auf `.config` ins Leere, doppeltes `sshd`/`cgroups`, `dwm-6.8`-Savedconfig-Versionierung, fehlende `chown kiel:kiel` auf `.xinitrc`, doppelte `packages.txt`-Erzeugung, tote Kommentarblöcke (`ryzen_smu`, `ext4_root`, `ocaml`/`gnatwhy3`), fehlende Prüfung von `70overlayfs`. | Alles bereinigen; die wichtigen Kommentare (10-Block-`@world`-Split + E14-Minimal-Firmware) bleiben vollständig erhalten. `config/config-gentoo.sh` bleibt funktionsfähig in `config/`. |
| **Container-in-Container-Export** | Der Agent läuft im Container (`/workspace/src` -> Host `/home/kiel/stage`, Docker über `/var/run/docker.sock`). Container-`/dev/shm` ist oft nur 64 MiB groß und schlägt bei Multi-GB-Exports fehl. | `setup03_copy_from_container.sh` unterstützt `EXPORT_BASE` (Default `/dev/shm` auf dem Host, automatischer Platzcheck / Override auf `/workspace/src/...` oder `/tmp/...` im Agent-Container). |

---

### 1.2 Zielarchitektur (2-Phasen-Binpkg + Multi-Stage)

```text
gentoo/portage:20261106 ──► portage
gentoo/stage3:nomultilib ─► stage3_base (einmalig aufgelöst, identisch für builder & target)
                              │
                              ├──► builder
                              │      • FEATURES="... buildpkg"
                              │      • Kernel 6.18.60 + Module + slstatus (config.h!)
                              │      • @world in 10 Blöcken (xargs -d '\n') + Konvergenz-@world
                              │      • genlop-Laufzeiten -> /tmp/build_times.tsv
                              │      • emaint --fix binhost -> /var/cache/binpkgs
                              │
                              └──► target
                                     • frisches stage3_base + saubere Portage-Configs
                                     • /boot + /lib/modules/${KVER_RELEASE} + /usr/local/bin/slstatus aus builder
                                     • bind-mount /var/cache/binpkgs (FEATURES="-binpkg-request-signature -getbinpkg")
                                     • emerge -uDN --usepkgonly --with-bdeps=n @world  (OHNE rust/go/clang/genlop!)
                                     • OpenRC-Konfig (rc_sys="", Single-Owner User-Session, kein reverse-ssh/modules)
                                     • Dracut initramfs + /boot/amd-uc.img + /packages.{txt,tsv}
                                     • Reality- & Smoke-Tests
                                        │
                                        ├──► variant-nv  (reverse-ssh-*, nur nvidia.ko-Firmware) -> /out/gentoo.squashfs_nv
                                        ├──► variant-e14 (iwd/openresolv, E14-Firmware, -NVIDIA) -> /out/gentoo.squashfs_e14
                                        └──► export (FROM scratch): 8 Artefakte
```

**Wichtige Portage-Details im 2-Phasen-Flow:**
1. Da `config/etc/portage/make.conf` `FEATURES="${FEATURES} binpkg-request-signature"` setzt, würde Portage unsignierte lokale Binärpakete aus `builder` ablehnen. Deshalb wird im `target`-Installationsschritt lokal `FEATURES="-binpkg-request-signature -getbinpkg"` über die Kommandozeile übergeben; die finale `/etc/portage/make.conf` im Image behält die Signaturprüfung für den späteren Betrieb.
2. Für den Build-Cache zwischen `setup01_build_image.sh` und `setup03_copy_from_container.sh` speichert `setup01_build_image.sh` die `BUILD_*`-Argumente in `logs/last_build_args.env`. `setup03_copy_from_container.sh` liest diese wieder ein, sodass `--target export --output` exakt denselben `BUILD_DATE`-Layer trifft und nicht wegen eines neuen Zeitstempels Dracut und beide `mksquashfs` neu ausführt.

---

## Teil 2: Expliziter Ausführungsplan (für den Agenten)

> **Regeln für den ausführenden Agenten:**
> 1. Arbeitsverzeichnis ist **`/workspace/src/cl-py-generator/example/110_gentoo/openrc`** (Host-Pfad `/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc`).
> 2. Arbeite **Phase 0 bis Phase 6 strikt der Reihe nach** ab.
> 3. Führe nach jeder Phase alle zugehörigen **Verifikationskommandos** aus. Bei jedem Fehler sofort stoppen und in derselben Phase beheben.
> 4. Verändere `install_on_e14.sh` und `doc/*` **nicht** (außer ggf. Lesezugriff).
> 5. Für den Docker-Export im Container **nicht** das Container-`/dev/shm` verwenden, sondern `EXPORT_BASE` explizit auf ein Verzeichnis mit genügend Plattenplatz setzen (z. B. `/workspace/src/cl-py-generator/example/110_gentoo/openrc/out_test`).

---

### Phase 0: Ausgangszustand sichern & Umgebung prüfen

1. Ins Projektverzeichnis wechseln, Docker-Zugriff und freien Speicher prüfen, Backup anlegen:
   ```bash
   set -euo pipefail
   cd /workspace/src/cl-py-generator/example/110_gentoo/openrc
   docker info >/dev/null
   df -h /workspace/src /tmp
   mkdir -p /tmp/openrc_cleanup_backup
   cp -a Dockerfile *.sh config/ /tmp/openrc_cleanup_backup/
   ```
2. **Test:**
   ```bash
   test -f /tmp/openrc_cleanup_backup/Dockerfile
   docker version --format '{{.Server.Version}}'
   ```

---

### Phase 1: Verzeichnisstruktur, `config-gentoo.sh` & `collect.sh` aufräumen

1. HP-Z6-Host-Referenzen in `config/host-reference/hpz6/` verschieben:
   ```bash
   mkdir -p config/host-reference/hpz6
   mv config/50-activate.start config/host-reference/hpz6/
   mv config/docker-disk.initd config/host-reference/hpz6/
   mv config/containerd-config.toml config/host-reference/hpz6/
   ```
2. Veraltete Skripte/Konfigs nach `archive/` verschieben und Emacs-Backup löschen:
   ```bash
   mkdir -p archive
   mv config/start-reverse-tunnels.sh archive/
   mv config/nethex archive/
   mv config/NetworkManager.conf archive/
   mv copy_files.sh archive/
   rm -f config/config-gentoo.sh~
   ```
3. `config/config-gentoo.sh` **behalten** und pfad-unabhängig + konfigurierbar machen (funktioniert sowohl aus `openrc/` als auch aus `openrc/config/`). Ersetze `config/config-gentoo.sh` exakt mit:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   GENTOO_DEP_TUI="${GENTOO_DEP_TUI:-${HOME}/Downloads/gentoo-dep-tui}"
   REPO_DIR="${REPO_DIR:-/var/db/repos/gentoo}"
   BINHOST_DIR="${BINHOST_DIR:-/tmp/Packages}"

   exec "${GENTOO_DEP_TUI}" \
       --repo "${REPO_DIR}" \
       --portage-dir "${SCRIPT_DIR}/etc/portage" \
       --world "${SCRIPT_DIR}/var/lib/portage/world" \
       --binhost "${BINHOST_DIR}"
   ```
   ```bash
   chmod +x config/config-gentoo.sh
   ```
4. `collect.sh` härten (sauberes Quoting, `archive/` und `out*` ausschließen, `config6*` weiterhin auslassen). Ersetze `collect.sh` exakt mit:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   find Dockerfile ./*.sh config/ doc/ plan/20261114_01_review/ -type f \
     | grep -v 'config6' \
     | sort \
     | while IFS= read -r f; do
         echo "//start of  ${f}"
         cat "${f}"
         echo ""
       done
   ```
   ```bash
   chmod +x collect.sh
   ```
5. **Test:**
   ```bash
   bash -n config/config-gentoo.sh collect.sh
   test -f config/config-gentoo.sh
   test ! -e config/config-gentoo.sh~
   for f in 50-activate.start docker-disk.initd containerd-config.toml; do
     test -f "config/host-reference/hpz6/${f}"
   done
   for f in start-reverse-tunnels.sh nethex NetworkManager.conf copy_files.sh; do
     test -f "archive/${f}"
   done
   ```

---

### Phase 2: Portage-Konfiguration (`package.use`, `package.accept_keywords`, `world`, `resolv.conf`)

1. **`mesa -llvm` Machbarkeit gegen den exakten Snapshot prüfen** (`gentoo/portage:20261106` + `gentoo/stage3:nomultilib`):
   ```bash
   MESA_LLVM_FLAG=""
   if docker run --rm \
     -v "$(pwd)/config:/tmp/cfg:ro" \
     gentoo/portage:20261106 \
     sh -c 'grep -q "llvm" /var/db/repos/gentoo/media-libs/mesa/*.ebuild'; then
     # Vollstaendiger Dep-Check gegen stage3 + portage + lokale Configs:
     if docker run --rm \
       -v "$(pwd)/config:/tmp/cfg:ro" \
       --mount type=volume,dst=/tmp/empty_ignore \
       gentoo/stage3:nomultilib \
       sh -c 'echo "skip"' >/dev/null 2>&1; then
       true
     fi
   fi
   ```
   Führe diesen konkreten **Portage-Solver-Test** im Container aus (Mount des `portage:20261106`-Images via `docker create` oder `COPY` in einem Mini-Inline-Dockerfile dauert ~10s und prüft `package.use` + `world` + `mesa -llvm` mit echtem `emerge -p`):
   ```bash
   TEST_PORTAGE_IMG="openrc-portage-check:tmp"
   docker build -q -t "${TEST_PORTAGE_IMG}" -f - . <<'EOF'
   FROM gentoo/portage:20261106 AS portage
   FROM gentoo/stage3:nomultilib
   COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo
   RUN eselect profile set default/linux/amd64/23.0/no-multilib
   COPY config/etc/portage/make.conf /etc/portage/make.conf
   COPY config/etc/portage/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
   RUN mkdir -p /etc/portage/package.mask
   COPY config/etc/portage/package.mask /etc/portage/package.mask/package.mask
   COPY config/etc/portage/package.use /etc/portage/package.use
   EOF

   if docker run --rm "${TEST_PORTAGE_IMG}" sh -c \
     'printf "media-libs/mesa -llvm\n" >> /etc/portage/package.use && emerge -p --quiet media-libs/mesa media-video/mpv www-client/firefox'; then
     echo "MESA_NO_LLVM=OK"
     USE_MESA_NO_LLVM=1
   else
     echo "MESA_NO_LLVM=CONFLICT"
     USE_MESA_NO_LLVM=0
   fi
   ```
2. Schreibe **`config/etc/portage/package.use`** bereinigt (ohne Duplikate; `mupdf` bleibt minimal ohne `X`/OpenGL für `zathura`; `firefox` mit `-clang` **und** `hwaccel`; `media-libs/mesa -llvm` nur, wenn `USE_MESA_NO_LLVM=1`):
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
   sys-power/tlp -ppd -python_single_target_python3_14 -rdw
   www-client/firefox -clang -dbus -gmp-autoupdate hwaccel -jumbo-build system-pipewire system-png -telemetry
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
   EOF
   if [ "${USE_MESA_NO_LLVM:-0}" -eq 1 ]; then
     printf 'media-libs/mesa -llvm\n' >> config/etc/portage/package.use
   fi
   ```
3. Entferne das tote `sys-kernel/dracut-crypt-ssh ~amd64` aus **`config/etc/portage/package.accept_keywords`**:
   ```bash
   sed -i '/^sys-kernel\/dracut-crypt-ssh/d' config/etc/portage/package.accept_keywords
   ```
4. Aktualisiere **`config/var/lib/portage/world`** (`app-portage/gentoolkit` raus; `dev-build/cmake` und `dev-util/pkgconf` rein; abschließendes `\n` sicherstellen):
   ```bash
   sed -i '/^app-portage\/gentoolkit$/d' config/var/lib/portage/world
   for atom in dev-build/cmake dev-util/pkgconf; do
     grep -qx "${atom}" config/var/lib/portage/world || printf '\n%s\n' "${atom}" >> config/var/lib/portage/world
   done
   # Leerzeilen am Ende normalisieren auf genau ein \n
   awk 'NF || /^#/' config/var/lib/portage/world > config/var/lib/portage/world.tmp
   mv config/var/lib/portage/world.tmp config/var/lib/portage/world
   ```
5. **`config/resolv.conf`** (Inhalt unverändert, nur abschließendes `\n` ergänzen):
   ```bash
   printf 'nameserver 194.11.90.70\nsearch lgs-net.com\nnameserver 8.8.8.8\n' > config/resolv.conf
   ```
6. **Portage-Preflight-Check** (prüft vorab in ~20 Sekunden, ob `world` + `package.use` + `package.mask` + `package.accept_keywords` ohne Blocker auflösen; danach temporäres Check-Image löschen):
   ```bash
   docker run --rm \
     -v "$(pwd)/config/etc/portage/package.use:/etc/portage/package.use:ro" \
     -v "$(pwd)/config/etc/portage/package.accept_keywords:/etc/portage/package.accept_keywords/package.accept_keywords:ro" \
     -v "$(pwd)/config/var/lib/portage/world:/var/lib/portage/world:ro" \
     "${TEST_PORTAGE_IMG}" \
     sh -c 'grep -vE "^[[:space:]]*(#|$)" /var/lib/portage/world | grep -v "gentoo-sources" | xargs -r -d "\n" emerge -p --quiet >/tmp/emerge_plan.txt && tail -n 5 /tmp/emerge_plan.txt'
   docker rmi -f "${TEST_PORTAGE_IMG}" >/dev/null || true
   ```
7. **Test:**
   ```bash
   test -z "$(awk '!/^\s*(#|$)/ {print $1}' config/etc/portage/package.use | sort | uniq -d)"
   ! grep -q 'dracut-crypt-ssh' config/etc/portage/package.accept_keywords
   ! grep -qx 'app-portage/gentoolkit' config/var/lib/portage/world
   grep -qx 'dev-build/cmake' config/var/lib/portage/world
   grep -qx 'dev-util/pkgconf' config/var/lib/portage/world
   test "$(tail -c1 config/var/lib/portage/world | wc -l)" -eq 1
   test "$(tail -c1 config/resolv.conf | wc -l)" -eq 1
   test "$(tail -c1 config/etc/portage/package.use | wc -l)" -eq 1
   ```

---

### Phase 3: OpenRC-Session, `config/xinitrc` & `config/activate`

1. **`config/xinitrc`** exakt ersetzen durch (OpenRC ist Single Owner; `xrandr` nur wenn `DP-0` und `DP-2` verbunden sind):
   ```sh
   #!/bin/sh

   # HP Z6 dual-vertical layout only when those outputs exist (no-op on ThinkPad E14 eDP)
   if xrandr --query 2>/dev/null | grep -q '^DP-0 connected' \
     && xrandr --query 2>/dev/null | grep -q '^DP-2 connected'; then
     xrandr --output DP-0 --scale 1x1 --rotate left
     xrandr --output DP-2 --scale 1x1 --rotate right
     xrandr --output DP-2 --right-of DP-0
   fi

   export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
   export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
   export PULSE_SERVER="${PULSE_SERVER:-unix:${XDG_RUNTIME_DIR}/pulse/native}"

   setxkbmap -layout us -variant colemak

   xterm -e btop &
   slstatus &

   cd /home/kiel || exit 1
   if [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
     exec dwm
   else
     exec dbus-run-session dwm
   fi
   ```
2. **`config/activate`** exakt ersetzen durch (`STAGE_ROOT` und `OPENRC_CONFIG_ROOT` konfigurierbar, `ldconfig` nur bei veraltetem Cache gegenüber `/IMAGE-BUILD-INFO.txt`, `tlp` nur auf E14):
   ```sh
   #!/bin/sh
   set -eu

   STAGE_ROOT="${STAGE_ROOT:-/home/kiel/stage}"
   config_root="${OPENRC_CONFIG_ROOT:-/usr/local/share/openrc-host-config}"
   if [ ! -d "${config_root}" ] && [ -d "${STAGE_ROOT}/cl-py-generator/example/110_gentoo/openrc/config" ]; then
     config_root="${STAGE_ROOT}/cl-py-generator/example/110_gentoo/openrc/config"
   fi

   try_modprobe() {
     modprobe "$1" 2>/dev/null || true
   }

   sync_initd() {
     src="$1"
     dst="$2"
     if [ -r "${src}" ]; then
       cp "${src}" "${dst}"
       chmod 0755 "${dst}"
     fi
   }

   is_e14=0
   if [ -r /sys/class/dmi/id/product_version ] \
     && grep -q "ThinkPad E14" /sys/class/dmi/id/product_version 2>/dev/null; then
     is_e14=1
   fi

   rc-service udev start 2>/dev/null || true
   rc-service udev-trigger start 2>/dev/null || true
   rc-service udev-settle start 2>/dev/null || true
   rc-service hostname start 2>/dev/null || true
   rc-service elogind start 2>/dev/null || true
   rc-service dbus start 2>/dev/null || true
   rc-service user-runtime start 2>/dev/null || true

   sync_initd "${config_root}/reverse-ssh-eu.initd" /etc/init.d/reverse-ssh-eu
   sync_initd "${config_root}/reverse-ssh-us.initd" /etc/init.d/reverse-ssh-us
   rc-update -u 2>/dev/null || true

   for svc in reverse-ssh-eu reverse-ssh-us; do
     if [ "${is_e14}" -eq 1 ]; then
       rc-update del "${svc}" default 2>/dev/null || true
     else
       rc-update add "${svc}" default 2>/dev/null || true
       rc-service "${svc}" start 2>/dev/null || true
     fi
   done

   if [ "${is_e14}" -eq 1 ]; then
     if grep -q '^rc_autostart_user=' /etc/rc.conf 2>/dev/null; then
       sed -i 's/^rc_autostart_user=.*/rc_autostart_user="NO"/' /etc/rc.conf
     else
       printf '\n%s\n' 'rc_autostart_user="NO"' >> /etc/rc.conf
     fi
   fi

   KEYMAP_PATH="$(find /usr/share/keymaps -path '*/colemak/en.latin*.gz' | sort | head -n 1 || true)"
   if [ -n "${KEYMAP_PATH}" ]; then
     loadkeys "${KEYMAP_PATH}"
   fi

   # Refresh linker cache only when persistent overlay cache is older than the booted image
   if [ -e /IMAGE-BUILD-INFO.txt ] && [ /IMAGE-BUILD-INFO.txt -nt /etc/ld.so.cache ]; then
     env-update
     ldconfig
   fi

   try_modprobe snd_hda_intel
   try_modprobe snd_pci_ps
   try_modprobe snd_soc_ps_mach
   try_modprobe snd_sof_amd_acp63
   if [ "${is_e14}" -eq 1 ]; then
     rc-service tlp start 2>/dev/null || true
   fi
   ```
3. **Test:**
   ```bash
   sh -n config/xinitrc
   sh -n config/activate
   ! grep -q 'start-pipewire\.sh' config/xinitrc
   grep -q 'STAGE_ROOT' config/activate
   ```

---

### Phase 4: `Dockerfile` auf 2-Phasen-Binpkg + Multi-Stage umbauen

Ersetze **`Dockerfile`** exakt mit folgendem Inhalt (die 10-Block-`@world`-Kommentare und E14-Firmware-Kommentare bleiben erhalten; der 10-Block-Split ist reaktiviert; `--mount=type=cache` für Distfiles beschleunigt Wiederholungs-Builds):

```dockerfile
# syntax=docker/dockerfile:1

ARG PORTAGE_TAG=20261106
ARG STAGE3_TAG=nomultilib

FROM gentoo/portage:${PORTAGE_TAG} AS portage
FROM gentoo/stage3:${STAGE3_TAG} AS stage3_base

# ==============================================================================
# Stage 1: Builder (compiles kernel, @world binary packages, slstatus)
# ==============================================================================
FROM stage3_base AS builder

COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

RUN eselect profile set default/linux/amd64/23.0/no-multilib
COPY config/etc/portage/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
RUN mkdir -p /etc/portage/package.mask
COPY config/etc/portage/package.mask /etc/portage/package.mask/package.mask
COPY config/etc/portage/make.conf /etc/portage/make.conf
COPY config/etc/portage/package.use /etc/portage/package.use
COPY config/etc/portage/package.env /etc/portage/package.env
COPY config/etc/portage/env /etc/portage/env
COPY config/var/lib/portage/world /var/lib/portage/world
COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm

# Build binary packages for all emerged packages so target stage installs without BDEPENDs
RUN printf '\nFEATURES="${FEATURES} buildpkg"\n' >> /etc/portage/make.conf

ENV KVER_PURE=6.18.60
ENV KVER_SOURCE=${KVER_PURE}-gentoo
ENV KVER_RELEASE=${KVER_SOURCE}-dist

RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked \
    emerge -1 sys-apps/portage

RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked \
    emerge =sys-kernel/gentoo-sources-${KVER_PURE} \
 && eselect kernel list \
 && eselect kernel set linux-${KVER_SOURCE}

WORKDIR /usr/src/linux
COPY config/config6.18.48 .config
RUN ./scripts/config \
      --set-str LOCALVERSION "-dist" \
      --disable DEBUG_INFO \
      --disable DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
      --disable DEBUG_INFO_BTF \
      --disable DEBUG_INFO_BTF_MODULES \
 && make olddefconfig \
 && make prepare \
 && make -j32 \
 && make modules_install \
 && depmod -a ${KVER_RELEASE} \
 && make install \
 && test -s /boot/vmlinuz-${KVER_RELEASE} \
 && test -d /lib/modules/${KVER_RELEASE}

# --- World-Update in 10 Blöcken (aktiviert) ---
# Letzter erfolgreicher Chunk-Build: logs/setup01_run_20261106_182536.log (exit 0).
# Ein Fehler stoppt nur seinen Block, die davor liegenden Blöcke sind im
# Layer-Cache. Atome werden über xargs -d '\n' übergeben, damit Sonderzeichen
# wie das '<' in sys-kernel/gentoo-sources-6.19 nicht von der Shell
# interpretiert werden.
# (Fuer ein Ein-Schritt-Update diesen Chunk-Block auskommentieren und nur die
# abschliessende `emerge -uDN @world`-Zeile aktiv lassen.)
RUN awk '!/^[[:space:]]*(#|$)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}' /var/lib/portage/world > /tmp/world_atoms.txt \
 && split -n l/10 -d /tmp/world_atoms.txt /tmp/world_stage_ \
 && wc -l /tmp/world_stage_*

RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_00 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_01 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_02 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_03 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_04 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_05 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_06 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_07 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_08 emerge -uDN
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked xargs -r -d '\n' -a /tmp/world_stage_09 emerge -uDN

# Konvergenz-@world + Binhost-Index aktualisieren
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked \
    emerge -uDN @world \
 && rm -f /tmp/world_atoms.txt /tmp/world_stage_* \
 && emaint --fix binhost \
 && test -s /var/cache/binpkgs/Packages \
 && gcc-config -l

# Build slstatus against config.h (verified for BAT0 from slstatus_config.h)
WORKDIR /usr/src
RUN git clone https://git.suckless.org/slstatus
WORKDIR /usr/src/slstatus
COPY config/slstatus_config.h config.h
RUN make -j32 \
 && make install \
 && make clean \
 && strings /usr/local/bin/slstatus | grep -q 'BAT0'

# Extract build times from builder's emerge.log without polluting target with genlop/perl
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked \
    set -e \
 && emerge --ask=n --oneshot app-portage/genlop \
 && : > /tmp/build_times.tsv \
 && for pkg in $(qlist -I); do \
      time_human=$(genlop -t "$pkg" 2>/dev/null | sed -nE 's/.*merge time: (.*)/\1/p' | tail -n 1 | sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g'); \
      if [ -n "$time_human" ]; then \
        build_seconds=$(TIME_HUMAN="$time_human" awk 'BEGIN{t=ENVIRON["TIME_HUMAN"]; gsub(/,/, "", t); n=split(t, a, / +/); s=0; for (i=1; i<=n; i++) { if (a[i] ~ /^[0-9]+$/) { v=a[i]+0; u=a[i+1]; if (u ~ /^day/) s+=v*86400; else if (u ~ /^hour/) s+=v*3600; else if (u ~ /^minute/) s+=v*60; else if (u ~ /^second/) s+=v; } } print s }'); \
      else \
        time_human='-'; \
        build_seconds=0; \
      fi; \
      printf '%s\t%s\t%s\n' "$pkg" "${build_seconds:-0}" "${time_human:-N/A}" >> /tmp/build_times.tsv; \
    done

# ==============================================================================
# Stage 2: Clean Target Rootfs (installed strictly from binpkgs, no BDEPENDs)
# ==============================================================================
FROM stage3_base AS target

COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

ENV KVER_PURE=6.18.60
ENV KVER_SOURCE=${KVER_PURE}-gentoo
ENV KVER_RELEASE=${KVER_SOURCE}-dist

RUN eselect profile set default/linux/amd64/23.0/no-multilib
COPY config/etc/portage/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
RUN mkdir -p /etc/portage/package.mask
COPY config/etc/portage/package.mask /etc/portage/package.mask/package.mask
COPY config/etc/portage/make.conf /etc/portage/make.conf
COPY config/etc/portage/package.use /etc/portage/package.use
COPY config/etc/portage/package.env /etc/portage/package.env
COPY config/etc/portage/env /etc/portage/env
COPY config/var/lib/portage/world /var/lib/portage/world
COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm

RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Copy prebuilt kernel + modules before emerging @world so nvidia-drivers postinst sees the kernel tree
COPY --from=builder /boot /boot
COPY --from=builder /lib/modules/${KVER_RELEASE} /lib/modules/${KVER_RELEASE}
COPY --from=builder /usr/local/bin/slstatus /usr/local/bin/slstatus

# Install strictly from local builder binpkgs without build-only dependencies (e.g. rust, go, genlop)
RUN --mount=type=bind,from=builder,source=/var/cache/binpkgs,target=/var/cache/binpkgs \
    FEATURES="-binpkg-request-signature -getbinpkg" \
    emerge -1 --usepkgonly sys-apps/portage \
 && FEATURES="-binpkg-request-signature -getbinpkg" \
    emerge -uDN --usepkgonly --with-bdeps=n @world \
 && depmod -a ${KVER_RELEASE} \
 && rm -rf /var/tmp/portage/*

# Sudo policy & user kiel
RUN mkdir -p /etc/sudoers.d \
 && echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel \
 && groupadd -f input \
 && useradd -m -G users,wheel,audio,video,input -s /bin/bash kiel \
 && if getent group tty >/dev/null; then usermod -aG tty kiel; fi \
 && for grp in libvirt kvm qemu plugdev pcap docker dialout; do \
      if getent group "${grp}" >/dev/null; then usermod -aG "${grp}" kiel; fi; \
    done \
 && passwd -d kiel

COPY config/xinitrc /home/kiel/.xinitrc
RUN mkdir -p /etc/X11/xorg.conf.d
COPY config/40-touchpad.conf /etc/X11/xorg.conf.d/40-touchpad.conf
COPY config/activate /home/kiel/activate
COPY config/start2 /home/kiel/start2
COPY config/start-pipewire.sh /home/kiel/start-pipewire.sh
RUN chmod +x /home/kiel/activate /home/kiel/start2 /home/kiel/start-pipewire.sh \
 && chown kiel:kiel /home/kiel/.xinitrc /home/kiel/activate /home/kiel/start2 /home/kiel/start-pipewire.sh

COPY config/user-runtime.initd /etc/init.d/user-runtime
COPY config/zz-openrc-user-session.sh /etc/profile.d/zz-openrc-user-session.sh
COPY config/user-dbus.initd /etc/user/init.d/dbus
COPY config/user-pipewire.initd /etc/user/init.d/pipewire
COPY config/user-pipewire-pulse.initd /etc/user/init.d/pipewire-pulse
COPY config/user-wireplumber.initd /etc/user/init.d/wireplumber

COPY config/resolv.conf /etc/
COPY config/reverse-ssh-eu.initd /etc/init.d/reverse-ssh-eu
COPY config/reverse-ssh-us.initd /etc/init.d/reverse-ssh-us
RUN chmod +x /etc/init.d/reverse-ssh-*

# Install Emacs packages from MELPA in batch mode
USER kiel
ENV HOME=/home/kiel
RUN emacs --batch \
    --eval "(require 'package)" \
    --eval "(add-to-list 'package-archives '(\"melpa\" . \"https://melpa.org/packages/\") t)" \
    --eval "(package-initialize)" \
    --eval "(package-refresh-contents)" \
    --eval "(dolist (pkg '(magit paredit slime)) (package-install pkg))"

USER root
ENV HOME=/root

RUN mkdir -p /usr/local/share/openrc-host-config
COPY config/activate /usr/local/share/openrc-host-config/activate
COPY config/start2 /usr/local/share/openrc-host-config/start2
COPY config/reverse-ssh-eu.initd /usr/local/share/openrc-host-config/reverse-ssh-eu.initd
COPY config/reverse-ssh-us.initd /usr/local/share/openrc-host-config/reverse-ssh-us.initd

RUN fc-cache -fv

# OpenRC configuration (Single Owner for dbus/pipewire/wireplumber; host-specific units enabled per variant)
RUN printf '%s\n' 'keymap="colemak/en-latin9"' > /etc/conf.d/keymaps \
 && sed -i 's/^rc_sys=.*/rc_sys=""/' /etc/rc.conf \
 && if grep -q '^rc_logger=' /etc/rc.conf 2>/dev/null; then \
      sed -i 's/^rc_logger=.*/rc_logger="YES"/' /etc/rc.conf; \
    else \
      printf '%s\n' 'rc_logger="YES"' >> /etc/rc.conf; \
    fi \
 && if grep -q '^rc_log_path=' /etc/rc.conf 2>/dev/null; then \
      sed -i 's|^rc_log_path=.*|rc_log_path="/var/log/rc.log"|' /etc/rc.conf; \
    else \
      printf '%s\n' 'rc_log_path="/var/log/rc.log"' >> /etc/rc.conf; \
    fi \
 && if grep -q '^rc_verbose=' /etc/rc.conf 2>/dev/null; then \
      sed -i 's/^rc_verbose=.*/rc_verbose="YES"/' /etc/rc.conf; \
    else \
      printf '%s\n' 'rc_verbose="YES"' >> /etc/rc.conf; \
    fi \
 && if grep -q '^rc_autostart_user=' /etc/rc.conf 2>/dev/null; then \
      sed -i 's/^rc_autostart_user=.*/rc_autostart_user="NO"/' /etc/rc.conf; \
    else \
      printf '%s\n' 'rc_autostart_user="NO"' >> /etc/rc.conf; \
    fi \
 && if ! grep -q '^s0:12345:respawn:/sbin/agetty 115200 ttyS0 vt100$' /etc/inittab; then \
      printf '%s\n' 's0:12345:respawn:/sbin/agetty 115200 ttyS0 vt100' >> /etc/inittab; \
    fi \
 && if ! grep -qx 'ttyS0' /etc/securetty; then \
      printf '%s\n' 'ttyS0' >> /etc/securetty; \
    fi \
 && chmod 0755 /etc/init.d/user-runtime /etc/profile.d/zz-openrc-user-session.sh \
      /etc/user/init.d/dbus /etc/user/init.d/pipewire /etc/user/init.d/pipewire-pulse /etc/user/init.d/wireplumber \
 && test -e /etc/init.d/user \
 && ln -sf /etc/init.d/user /etc/init.d/user.kiel \
 && install -d -o kiel -g kiel -m 0755 /home/kiel/.config/rc/runlevels/default \
 && ln -sf /etc/user/init.d/dbus /home/kiel/.config/rc/runlevels/default/dbus \
 && ln -sf /etc/user/init.d/pipewire /home/kiel/.config/rc/runlevels/default/pipewire \
 && ln -sf /etc/user/init.d/pipewire-pulse /home/kiel/.config/rc/runlevels/default/pipewire-pulse \
 && ln -sf /etc/user/init.d/wireplumber /home/kiel/.config/rc/runlevels/default/wireplumber \
 && rc-update add udev sysinit \
 && rc-update add udev-trigger sysinit \
 && rc-update add udev-settle sysinit \
 && rc-update add cgroups sysinit \
 && rc-update add modules boot \
 && rc-update add keymaps boot \
 && rc-update add elogind boot \
 && rc-update add user-runtime default \
 && rc-update add user.kiel default \
 && rc-update add dbus default \
 && rc-update add iwd default \
 && rc-update add sshd default \
 && rc-update add docker default \
 && rc-update add local default \
 && chown -R kiel:kiel /home/kiel/.config

# Dracut Module documentation https://dracut-ng.github.io/dracut-ng/modules/core.html
WORKDIR /
RUN test -d /usr/lib/dracut/modules.d/70overlayfs \
 && test -f /usr/lib/dracut/modules.d/70overlayfs/module-setup.sh
COPY config/mount-overlayfs.sh /usr/lib/dracut/modules.d/70overlayfs/mount-overlayfs.sh
RUN chmod +x /usr/lib/dracut/modules.d/70overlayfs/*.sh \
 && dracut \
      -m "kernel-modules base dmsquash-live dm udev-rules crypt lvm overlayfs" \
      --filesystems "squashfs erofs vfat ext4 overlay btrfs" \
      --add-drivers "libnvdimm nd_pmem nd_btt nd_blk dax device_dax dax_pmem dax_pmem_core nd_e820" \
      --kver=${KVER_RELEASE} \
      --install "stat blockdev" \
      --force \
      /boot/initramfs_squash_sda1-x86_64.img

# Build external AMD microcode cpio archive from linux-firmware
RUN set -e \
 && mkdir -p /tmp/ucode/kernel/x86/microcode \
 && cat /usr/lib/firmware/amd-ucode/microcode_amd*.bin > /tmp/ucode/kernel/x86/microcode/AuthenticAMD.bin \
 && (cd /tmp/ucode && find kernel | cpio -o -H newc > /boot/amd-uc.img) \
 && rm -rf /tmp/ucode \
 && test -s /boot/amd-uc.img

# Generate final package reports from target sizes + builder merge times
RUN --mount=type=bind,from=builder,source=/tmp/build_times.tsv,target=/tmp/build_times.tsv \
    set -e \
 && printf 'package\tsize_mib\tbuild_seconds\tbuild_time_human\n' > /packages.tsv \
 && time_threshold_mib=50 \
 && for pkg in $(qlist -I); do \
      size_line=$(qsize -m "$pkg" | head -n 1 || true); \
      size_mib=$(printf '%s\n' "$size_line" | sed -nE 's/.*,[[:space:]]*([0-9.]+)[[:space:]]+MiB$/\1/p'); \
      if [ -z "$size_mib" ]; then \
        size_kib=$(printf '%s\n' "$size_line" | sed -nE 's/.*,[[:space:]]*([0-9.]+)[[:space:]]+KiB$/\1/p'); \
        if [ -n "$size_kib" ]; then \
          size_mib=$(awk "BEGIN { printf \"%.3f\", ${size_kib}/1024 }"); \
        else \
          size_mib=0; \
        fi; \
      fi; \
      is_large=$(awk "BEGIN { print (${size_mib} >= ${time_threshold_mib}) ? 1 : 0 }"); \
      if [ "$is_large" -eq 1 ]; then \
        bt_line=$(awk -F '\t' -v p="$pkg" '$1 == p {print $2 "\t" $3; exit}' /tmp/build_times.tsv); \
        if [ -n "$bt_line" ]; then \
          build_seconds="${bt_line%%	*}"; \
          time_human="${bt_line#*	}"; \
        else \
          build_seconds=0; \
          time_human='-'; \
        fi; \
      else \
        time_human='-'; \
        build_seconds=0; \
      fi; \
      printf '%s\t%s\t%s\t%s\n' "$pkg" "$size_mib" "${build_seconds:-0}" "${time_human:-N/A}" >> /packages.tsv; \
    done \
 && { \
      printf '%-40s %10s %14s  %s\n' "Package" "Size MiB" "Build Seconds" "Build Time"; \
      tail -n +2 /packages.tsv | sort -t "$(printf '\t')" -k2,2nr -k3,3nr | awk -F '\t' '{printf "%-40s %10s %14s  %s\n", $1, $2, $3, $4}'; \
    } > /packages.txt

# Keep BUILD_* ARGs as late as possible so export cache hits cleanly
ARG BUILD_GIT_COMMIT=unknown
ARG BUILD_REPOSITORY=unknown
ARG BUILD_DATE=unknown

RUN printf '%s\n' \
  'OpenRC Gentoo image build' \
  '==========================' \
  "Git commit: ${BUILD_GIT_COMMIT}" \
  "Repository: ${BUILD_REPOSITORY}" \
  'Source Dockerfile: openrc/Dockerfile' \
  "Build date (UTC): ${BUILD_DATE}" \
  '' \
  'Audio note' \
  '----------' \
  'PipeWire/WirePlumber are enabled, but the ALSA card is hardware-dependent.' \
  'On the ThinkPad E14, mpv may need:' \
  '  mpv --ao=alsa --audio-device=alsa/sysdefault:CARD=Generic_1' \
  'Do not use Generic_1 as a global default for the HP Z6; inspect aplay -l,' \
  'wpctl status, or pactl list short sinks there and select the host-specific' \
  'device/profile.' \
  > /IMAGE-BUILD-INFO.txt

# Reality & smoke tests on clean target rootfs before building variants
RUN set -e \
 && grep -q '^rc_sys=""' /etc/rc.conf \
 && for s in udev udev-trigger udev-settle cgroups; do \
      test -e "/etc/runlevels/sysinit/${s}"; \
    done \
 && for s in modules keymaps elogind; do \
      test -e "/etc/runlevels/boot/${s}"; \
    done \
 && for s in user-runtime user.kiel dbus iwd sshd docker local; do \
      test -e "/etc/runlevels/default/${s}"; \
    done \
 && for s in dbus pipewire pipewire-pulse wireplumber; do \
      test -e "/home/kiel/.config/rc/runlevels/default/${s}"; \
    done \
 && for bin in dwm slstatus emacs xfreerdp nvme firefox zathura gcc g++ gdb cmake pkgconf sbcl git iwctl; do \
      command -v "${bin}" >/dev/null || { echo "MISSING BIN: ${bin}"; exit 1; }; \
    done \
 && strings /usr/local/bin/slstatus | grep -q 'BAT0' \
 && ! command -v rustc >/dev/null \
 && ! command -v cargo >/dev/null \
 && ! command -v go >/dev/null \
 && ! qlist -I | grep -qE '^(dev-lang/rust|dev-lang/go|app-portage/gentoolkit)' \
 && lsinitrd /boot/initramfs_squash_sda1-x86_64.img | grep -q 'mount-overlayfs.sh' \
 && cpio -it < /boot/amd-uc.img | grep -q 'AuthenticAMD.bin' \
 && test -s /boot/vmlinuz \
 && test -s /packages.txt \
 && test -s /packages.tsv

# ==============================================================================
# Variant 1: HP Z6 (NVIDIA + reverse SSH tunnels)
# ==============================================================================
FROM target AS variant-nv

# Stage the exact firmware paths requested by the installed nvidia.ko
RUN set -e \
 && echo "Preparing NVIDIA squashfs" \
 && rc-update add reverse-ssh-eu default \
 && rc-update add reverse-ssh-us default \
 && test -e /lib/modules/${KVER_RELEASE}/video/nvidia.ko \
 && rm -rf /tmp/fw_nv_root /out \
 && mkdir -p /tmp/fw_nv_root/usr/lib/firmware /out \
 && modinfo -F firmware /lib/modules/${KVER_RELEASE}/video/nvidia.ko \
      | sort -u \
      | while IFS= read -r rel; do \
          test -n "${rel}"; \
          src="/usr/lib/firmware/${rel}"; \
          dst="/tmp/fw_nv_root/usr/lib/firmware/${rel}"; \
          test -e "${src}"; \
          mkdir -p "$(dirname "${dst}")"; \
          cp -a "${src}" "${dst}"; \
        done \
 && rm -rf /usr/lib/firmware/* \
 && cp -a /tmp/fw_nv_root/usr/lib/firmware/. /usr/lib/firmware/ \
 && rm -rf /tmp/fw_nv_root \
 && mksquashfs / /out/gentoo.squashfs_nv \
   -comp zstd \
   -Xcompression-level 19 \
   -b 256K \
   -mem 10G \
   -xattrs \
   -noappend \
   -not-reproducible \
   -progress \
   -one-file-system-x \
   -p "/dev d 755 0 0" \
   -p "/proc d 555 0 0" \
   -p "/sys d 555 0 0" \
   -noI -noX \
   -wildcards \
   -e \
     out \
     usr/src \
     var/cache/binhost \
     var/cache/binpkgs \
     var/cache/distfiles \
     "gentoo*squashfs*" \
     "gentoo*ext4" \
     "usr/lib64/libQt*.a" \
     usr/share/genkernel/distfiles \
     usr/share/doc \
     usr/src/linux \
     usr/share/sgml \
     var/cache/eix/previous.eix \
     boot \
     persistent \
     home/*/.cache \
     var/log/journal \
     var/cache/genkernel \
     var/tmp \
     opt/Signal/LICENSES.chromium.html \
     lost+found \
 && test -s /out/gentoo.squashfs_nv

# ==============================================================================
# Variant 2: ThinkPad E14 (iwd owns wlan0, minimal AMD/MT7922 firmware, no NVIDIA)
# ==============================================================================
FROM target AS variant-e14

# ThinkPad E14 only: iwd owns wlan0 (WLAN, DHCP, DNS via openresolv).
# Background: doc/hotspot_disconnect_report.md
COPY config/e14-iwd-main.conf /tmp/e14/iwd-main.conf
COPY config/e14-iwd.confd /tmp/e14/iwd.confd
COPY config/e14-nm-no-connectivity.conf /tmp/e14/nm-no-connectivity.conf

# Minimal firmware set for the ThinkPad E14 Gen 6 (21M30030MZ, Ryzen 7 7735HS),
# extracted as the unique "Loading firmware:" lines in dmesg_e14 (also stable
# across the suspend/resume reload cycles at 21210s and 26879s):
#
# regulatory.db
# regulatory.db.p7s
# mediatek/WIFI_RAM_CODE_MT7922_1.bin
# mediatek/WIFI_MT7922_patch_mcu_1_1_hdr.bin
# mediatek/BT_RAM_CODE_MT7922_1_1_hdr.bin
# amdgpu/yellow_carp_toc.bin
# amdgpu/yellow_carp_ta.bin
# amdgpu/yellow_carp_dmcub.bin
# amdgpu/yellow_carp_pfp.bin
# amdgpu/yellow_carp_me.bin
# amdgpu/yellow_carp_ce.bin
# amdgpu/yellow_carp_rlc.bin
# amdgpu/yellow_carp_mec.bin
# amdgpu/yellow_carp_mec2.bin
# amdgpu/yellow_carp_sdma.bin
# amdgpu/yellow_carp_vcn.bin
#
# Additionally kept beyond the dmesg set:
# - amd-ucode/*: required for microcode late loading
#   (CONFIG_MICROCODE_LATE_LOADING=y); applied at runtime with
#   echo 1 > /sys/devices/system/cpu/microcode/reload
# - rtl_nic/* + rtl_bt/*: the onboard RTL8168h needs no firmware, but a
#   not-always-plugged USB adapter does, so it never showed in dmesg_e14
#
# Deliberately dropped (no load request in dmesg_e14, no driver errors):
# - amdgpu/rembrandt*: VBIOS string says REMBRANDT but the driver loads yellow_carp
# - mediatek/* beyond the 3 files above (other chipsets: MT7925/MT7996/...)
# - amd/* beyond amd-ucode (SEV/TEE blobs for server silicon)
RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked \
    set -e \
 && echo "Preparing ThinkPad E14 squashfs (iwd, minimal firmware, remove NVIDIA)" \
 && printf 'modules="amdgpu mt7921e"\n' > /etc/conf.d/modules \
 && emerge --ask=n --noreplace net-dns/openresolv \
 && install -d -m 0755 /etc/iwd /etc/NetworkManager/conf.d \
 && install -m 0644 /tmp/e14/iwd-main.conf /etc/iwd/main.conf \
 && install -m 0644 /tmp/e14/iwd.confd /etc/conf.d/iwd \
 && install -m 0644 /tmp/e14/nm-no-connectivity.conf \
      /etc/NetworkManager/conf.d/90-e14-no-connectivity.conf \
 && (rc-update del NetworkManager default 2>/dev/null || true) \
 && rm -rf /tmp/e14 /var/tmp/portage/* \
 \
 && mkdir -p /tmp/tmpfw/amdgpu /tmp/tmpfw/mediatek /out \
 && cp -a /usr/lib/firmware/regulatory.db /usr/lib/firmware/regulatory.db.p7s /tmp/tmpfw/ \
 && cp -a /usr/lib/firmware/mediatek/WIFI_RAM_CODE_MT7922_1.bin \
      /usr/lib/firmware/mediatek/WIFI_MT7922_patch_mcu_1_1_hdr.bin \
      /usr/lib/firmware/mediatek/BT_RAM_CODE_MT7922_1_1_hdr.bin /tmp/tmpfw/mediatek/ \
 && cp -a /usr/lib/firmware/amdgpu/yellow_carp_toc.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_ta.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_dmcub.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_pfp.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_me.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_ce.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_rlc.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_mec.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_mec2.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_sdma.bin \
      /usr/lib/firmware/amdgpu/yellow_carp_vcn.bin /tmp/tmpfw/amdgpu/ \
 && cp -a /usr/lib/firmware/amd-ucode \
      /usr/lib/firmware/rtl_nic \
      /usr/lib/firmware/rtl_bt /tmp/tmpfw/ \
 && rm -rf /usr/lib/firmware/* \
 && mv /tmp/tmpfw/* /usr/lib/firmware/ \
 && rmdir /tmp/tmpfw \
 \
 && emerge -C x11-drivers/nvidia-drivers dev-util/nvidia-cuda-toolkit 2>/dev/null || true \
 && rm -rf /opt/cuda /usr/lib64/lib{cuda,nvidia}* \
      /usr/lib64/libnvcuvid* /usr/lib64/libnvoptix* \
      /usr/bin/nvidia* \
      /usr/lib/elogind/system-sleep/nvidia \
      /usr/lib/systemd/system/nvidia-* \
      /usr/lib/systemd/system-sleep/nvidia \
      /usr/lib/dracut/dracut.conf.d/10-nvidia-drivers.conf \
      /usr/share/applications/nvidia-settings.desktop \
      /usr/share/pixmaps/nvidia-settings.png \
      /usr/share/doc/nvidia-drivers-* \
      /usr/share/nvidia \
      /usr/lib64/nvidia \
      /usr/lib64/gbm/nvidia-drm_gbm.so \
      /usr/lib64/vdpau/libvdpau_nvidia.so* \
      /nvidia${KVER_RELEASE} \
 && ! ldd "$(command -v nvme)" | grep -q 'not found' \
 && ! test -e /etc/runlevels/default/reverse-ssh-eu \
 && test -e /etc/iwd/main.conf \
 && mksquashfs / /out/gentoo.squashfs_e14 \
   -comp zstd \
   -Xcompression-level 19 \
   -b 256K \
   -mem 10G \
   -xattrs \
   -noappend \
   -not-reproducible \
   -progress \
   -one-file-system-x \
   -p "/dev d 755 0 0" \
   -p "/proc d 555 0 0" \
   -p "/sys d 555 0 0" \
   -noI -noX \
   -wildcards \
   -e \
     out \
     usr/src \
     var/cache/binpkgs \
     var/cache/distfiles \
     var/cache/binhost \
     "gentoo*squashfs*" \
     "gentoo*ext4" \
     "usr/lib64/libQt*.a" \
     usr/share/genkernel/distfiles \
     usr/share/doc \
     usr/src/linux \
     usr/share/sgml \
     var/cache/eix/previous.eix \
     boot \
     persistent \
     home/*/.cache \
     var/log/journal \
     var/cache/genkernel \
     var/tmp \
     opt/Signal/LICENSES.chromium.html \
     lost+found \
 && test -s /out/gentoo.squashfs_e14

# ==============================================================================
# Stage 3: Export all release artifacts (`docker build --target export --output ...`)
# ==============================================================================
FROM scratch AS export
COPY --from=variant-nv  /out/gentoo.squashfs_nv /
COPY --from=variant-e14 /out/gentoo.squashfs_e14 /
COPY --from=target      /boot/vmlinuz /
COPY --from=target      /boot/initramfs_squash_sda1-x86_64.img /
COPY --from=target      /boot/amd-uc.img /
COPY --from=target      /packages.txt /
COPY --from=target      /packages.tsv /
COPY --from=target      /IMAGE-BUILD-INFO.txt /
```

**Tests vor dem Build:**
```bash
grep -E '^\s*COPY\s+' Dockerfile | grep -v -- '--from=' | awk '{print $2}' | sort -u | while read -r p; do
  test -e "${p}" || { echo "MISSING COPY SOURCE: ${p}"; exit 1; }
done
grep -q '^COPY config/slstatus_config.h config.h$' Dockerfile
grep -q '^RUN --mount=type=cache.*world_stage_00 emerge -uDN$' Dockerfile
grep -q '^RUN --mount=type=cache.*world_stage_09 emerge -uDN$' Dockerfile
! grep -n '|| true' Dockerfile | grep -iE 'mksquashfs|squashfs'
```

---

### Phase 5: Build- und Export-Skripte aktualisieren

1. **`setup01_build_image.sh`** exakt ersetzen durch (ohne `INITRAMFS_AUTH_KEY`/`TUNNEL_KEY`; speichert die genauen `BUILD_*`-Werte in `logs/last_build_args.env` für 100 % Cache-Treffer in `setup03` und taggt zusätzlich `gentoo-z6-min-openrc:target` zum Inspizieren):
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   cd "${SCRIPT_DIR}"
   mkdir -p logs

   GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
   if [ "${GIT_COMMIT}" != "unknown" ] && ! git diff --quiet 2>/dev/null; then
     GIT_COMMIT="${GIT_COMMIT}-dirty"
   fi
   REPO_URL="$(git config --get remote.origin.url 2>/dev/null || echo local)"
   BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

   cat > logs/last_build_args.env <<EOF
   BUILD_GIT_COMMIT=${GIT_COMMIT}
   BUILD_REPOSITORY=${REPO_URL}
   BUILD_DATE=${BUILD_DATE}
   EOF

   # Build debuggable target image AND full export graph
   DOCKER_BUILDKIT=1 docker build \
     --target target \
     --build-arg BUILD_GIT_COMMIT="${GIT_COMMIT}" \
     --build-arg BUILD_REPOSITORY="${REPO_URL}" \
     --build-arg BUILD_DATE="${BUILD_DATE}" \
     -t gentoo-z6-min-openrc \
     --progress=plain \
     .

   DOCKER_BUILDKIT=1 docker build \
     --target export \
     --build-arg BUILD_GIT_COMMIT="${GIT_COMMIT}" \
     --build-arg BUILD_REPOSITORY="${REPO_URL}" \
     --build-arg BUILD_DATE="${BUILD_DATE}" \
     --progress=plain \
     .
   ```
2. **`setup03_copy_from_container.sh`** exakt ersetzen durch (`EXPORT_BASE` konfigurierbar, Default `/dev/shm` mit Platzcheck ≥ 6 GiB, sonst Hinweis auf `EXPORT_BASE`):
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   cd "${SCRIPT_DIR}"

   TODAY="$(date +%Y%m%d)"
   EXPORT_BASE="${EXPORT_BASE:-/dev/shm}"
   mkdir -p "${EXPORT_BASE}"

   AVAIL_BYTES="$(df -B1 --output=avail "${EXPORT_BASE}" | tail -n 1 | tr -d ' ')"
   MIN_BYTES=$((6 * 1024 * 1024 * 1024))
   if [ "${AVAIL_BYTES}" -lt "${MIN_BYTES}" ]; then
     echo "[FEHLER] ${EXPORT_BASE} hat nur ${AVAIL_BYTES} Bytes frei (< 6 GiB)." >&2
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

   rm -rf "${TMP_TARGET}"
   mkdir -p "${TMP_TARGET}"

   DOCKER_BUILDKIT=1 docker build \
     --target export \
     --build-arg BUILD_GIT_COMMIT="${BUILD_GIT_COMMIT}" \
     --build-arg BUILD_REPOSITORY="${BUILD_REPOSITORY}" \
     --build-arg BUILD_DATE="${BUILD_DATE}" \
     --output "type=local,dest=${TMP_TARGET}" \
     --progress=plain \
     .

   rm -rf "${TARGET}"
   mv "${TMP_TARGET}" "${TARGET}"
   ls -lh "${TARGET}"
   ```
3. **Test:**
   ```bash
   chmod +x setup01_build_image.sh setup03_copy_from_container.sh
   bash -n setup01_build_image.sh setup03_copy_from_container.sh setup01_run_with_log.sh install_on_e14.sh
   ```

---

### Phase 6: Build, Export & End-to-End-Verifikation

1. **Vollständigen Build mit Log starten** (ca. 2h; jeder fertige Chunk landet im Layer-Cache):
   ```bash
   bash ./setup01_run_with_log.sh
   ```
   Prüfen, dass `EXIT_CODE=0` ausgegeben wurde und in `logs/setup01_run_*.meta` steht.
2. **Artefakte exportieren:**
   - Im **Agent-Container** (weil Container-`/dev/shm` meist zu klein ist):
     ```bash
     EXPORT_BASE="$(pwd)/out_export" bash ./setup03_copy_from_container.sh
     OUT="$(pwd)/out_export/gentoo-z6-min-openrc_$(date +%Y%m%d)"
     ```
   - Auf dem **Host**:
     ```bash
     bash ./setup03_copy_from_container.sh
     OUT="/dev/shm/gentoo-z6-min-openrc_$(date +%Y%m%d)"
     ```
3. **Export-Vollständigkeit & Paketlisten prüfen:**
   ```bash
   for f in gentoo.squashfs_nv gentoo.squashfs_e14 vmlinuz \
            initramfs_squash_sda1-x86_64.img amd-uc.img \
            packages.txt packages.tsv IMAGE-BUILD-INFO.txt; do
     test -s "${OUT}/${f}" || { echo "MISSING OR EMPTY: ${f}"; exit 1; }
   done

   # Verify desired tools ARE listed and stripped BDEPENDs are NOT listed
   grep -q '^dev-build/cmake' "${OUT}/packages.tsv"
   grep -q '^dev-util/pkgconf' "${OUT}/packages.tsv"
   grep -q '^dev-debug/gdb' "${OUT}/packages.tsv"
   grep -q '^dev-lisp/sbcl' "${OUT}/packages.tsv"
   ! grep -qE '^(dev-lang/rust|dev-lang/go|app-portage/gentoolkit)\b' "${OUT}/packages.tsv"
   ```
4. **Inhaltliche Stichproben auf beiden Squashfs-Images** (nutzt `unsquashfs` direkt aus dem frisch gebauten `gentoo-z6-min-openrc:latest`, funktioniert also auch wenn `unsquashfs` auf dem Agent-Container fehlt):
   ```bash
   docker run --rm -v "${OUT}:/out:ro" gentoo-z6-min-openrc sh -c '
     set -eu
     unsquashfs -ll /out/gentoo.squashfs_e14 usr/lib64 | grep -q "libnvme\.so"
     ! unsquashfs -ll /out/gentoo.squashfs_e14 | grep -q "nvidia-smi"
     unsquashfs -ll /out/gentoo.squashfs_e14 etc/iwd/main.conf | grep -q "main.conf"
     unsquashfs -ll /out/gentoo.squashfs_e14 etc/conf.d/modules | grep -q "modules"
     ! unsquashfs -ll /out/gentoo.squashfs_e14 etc/runlevels/default | grep -q "reverse-ssh-eu"
     unsquashfs -ll /out/gentoo.squashfs_nv etc/runlevels/default | grep -q "reverse-ssh-eu"
     unsquashfs -ll /out/gentoo.squashfs_nv lib/modules | grep -q "nvidia\.ko"
   '
   ```
5. **Nach Abschluss (auf dem Zielrechner E14):**
   ```bash
   sudo bash ./install_on_e14.sh --src "${OUT}" --dry-run
   ```
   (Das Backup unter `/tmp/openrc_cleanup_backup` und ggf. `out_export/` erst nach erfolgreichem Boot-Test löschen.)
