Hier ist ein vollständiges Review des aktuellen Stands (Stand deiner gepasteten Dateien) sowie ein **expliziter, phasenweiser Ausführungsplan**, der direkt von einem nachrangigen/einfachen AI-Agenten Schritt für Schritt abgearbeitet werden kann.

---

## Teil 1: Review & konkrete Befunde (Ist-Zustand)

Viele Punkte aus `doc/review_dockerfile_and_install_concept.md` sind bereits umgesetzt (`rc_sys=""`, `keymap="colemak/en-latin9"`, späte `packages.tsv`-Generierung). Im aktuellen Code stecken aber noch **mehrere echte Bugs**, **tote Reste** und **strukturelle Altlasten**:

### Kritische Bugs
1. **`slstatus` baut deine Konfiguration gar nicht mit (`Dockerfile`)**:
   - `COPY config/slstatus_config.h .` kopiert nach `/usr/src/slstatus/slstatus_config.h`.
   - Das `Makefile` von `slstatus` baut aber aus `config.h` (Standardkopie von `config.def.h`). Deine `slstatus_config.h` wird komplett ignoriert. Muss `COPY config/slstatus_config.h config.h` heißen.
2. **E14-Stripping löscht `libnvme` (`Dockerfile`)**:
   - `rm -rf /usr/lib64/libnv*` löscht nicht nur NVIDIA-Bibliotheken, sondern auch `/usr/lib64/libnvme*.so*` (`sys-libs/libnvme`), was `sys-apps/nvme-cli` (in `world`) auf dem ThinkPad E14 zerstört.
3. **`|| true` schluckt Build-Fehler bei `gentoo.squashfs_e14` (`Dockerfile`)**:
   - Der gesamte E14-Block endet mit `|| true`. Ein abgebrochenes `mksquashfs` oder fehlende Firmware lässt den Build grün durchlaufen.
4. **Doppelte / kollidierende Session-Verwaltung (D-Bus & PipeWire)**:
   - Im `Dockerfile` sind die User-Links für `pipewire`, `pipewire-pulse` und `wireplumber` auskommentiert – **`config/user-runtime.initd` legt genau diese Symlinks bei jedem Boot aber trotzdem wieder an**.
   - Gleichzeitig startet `config/xinitrc` nochmal `~/start-pipewire.sh &` **und** `exec dbus-run-session dwm` (was einen zweiten Session-D-Bus öffnet und `DBUS_SESSION_BUS_ADDRESS` aus `zz-openrc-user-session.sh` überschattet).
5. **Destruktive In-Place-Mutation + 3,5 GB Squashfs-Bloat im Image (`Dockerfile` + `copy_files.sh`)**:
   - Firmware und `/usr/lib64` werden im selben Container-Dateisystem hin- und herkopiert und gelöscht; beide Squashfs-Images bleiben in den Docker-Layern liegen.
   - `copy_files.sh` versucht noch `/gentoo.squashfs` (ohne `|| true`) und `/gentoo.ext4` zu kopieren, die gar nicht mehr gebaut werden.
6. **Ungenutzte & unsichere Secrets im Build-Script (`setup01_build_image.sh`)**:
   - `INITRAMFS_AUTH_KEY` und `TUNNEL_KEY="$(cat ~/.ssh/tinyus_key.pem)"` werden als `--build-arg` übergeben, aber im `Dockerfile` nie als `ARG` konsumiert (Rest von `dracut-crypt-ssh`, genau wie `config/start-reverse-tunnels.sh`). Existiert `~/.ssh/tinyus_key.pem` nicht, bricht `setup01_build_image.sh` unnötig ab.

### Widersprüche, Redundanzen & Kleinigkeiten
7. **`app-portage/gentoolkit` steht in `world`, wird im `Dockerfile` aber per `emerge -C` wieder gelöscht** – und `/packages.txt` wird zweimal erzeugt (früh und spät).
8. **`config/etc/portage/package.use` hat doppelte Einträge**:
   - `sys-fs/lvm2` steht zweimal drin (`lvm` vs. `lvm readline`).
   - `www-client/firefox` steht zweimal drin (`-clang ...` vs. `hwaccel ...`).
   - `app-text/mupdf` fehlt ggf. `X` (siehe `doc/walkthrough.md`), falls `mupdf-x11` erwartet wird.
9. **`Dockerfile` Kleinigkeiten**:
   - `FROM gentoo/stage3:nomultilib` ist nicht auf ein Datum gepinnt (`portage:20261106` schon).
   - `COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm-6.8` ist fragil bei Portage-Versionssprüngen (`x11-wm/dwm` ohne Versionssuffix gilt für alle Versionen).
   - `sed` auf `config/config6.18.48` läuft komplett ins Leere (`CONFIG_LOCALVERSION="-dist"` und `CONFIG_DEBUG_INFO_NONE=y` stehen bereits so drin).
   - `RUN env-update && source /etc/profile` ist wirkungslos (Export stirbt am Ende des `RUN`-Layers).
   - `chown kiel:kiel` fehlt auf `/home/kiel/.xinitrc` (gehört aktuell `root:root`).
   - `rc-update add sshd default` ist doppelt; `rc-update add cgroups boot` ist doppelt zu `sysinit`.
   - `install -d ... /run/user/1000` im `Dockerfile` ist wirkungslos, weil `/run` zur Laufzeit ein frisches `tmpfs` ist (`user-runtime.initd` legt es an).
   - `reverse-ssh-eu`/`reverse-ssh-us` und `modules="amdgpu mt7921e"` werden im gemeinsamen Basis-Image aktiviert, obwohl sie maschinenspezifisch sind.
   - `COPY config/mount-overlayfs.sh /usr/lib/dracut/modules.d/70overlayfs` ohne End-Slash und ohne Build-Check, ob `70overlayfs` im aktuellen `dracut` noch so heißt.
   - `emerge --ask=n sys-fs/e2fsprogs sys-fs/erofs-utils || true` mit auskommentiertem `erofs`-Block und auskommentierten Alt-Blöcken (`ryzen_smu` mit undefiniertem `${KVER}`, altes `mksquashfs`, `ext4_root`, `gnatwhy3`/`ocaml`, `freerdp`-Check).
   - `amd-uc.img` wird nicht aus dem Build exportiert (`install_on_e14.sh` muss auf alte Slots zurückgreifen).
10. **Fehlende abschließende Newlines & tote Dateien in `config/`**:
    - `config/var/lib/portage/world` und `config/resolv.conf` enden ohne `\n`.
    - Ungenutzte Altdateien im `config/`-Verzeichnis: `config/start-reverse-tunnels.sh`, `config/nethex`, `config/config-gentoo.sh`, `config/NetworkManager.conf`, sowie Host-Referenzdateien (`50-activate.start`, `docker-disk.initd`, `containerd-config.toml`), die nicht ins Image kopiert werden.
    - `config/xinitrc` hat hartkodierte HP-Z6-`xrandr`-Ausgänge (`DP-0`, `DP-2`).
    - `config/activate` ruft `/etc/init.d/tlp start` direkt statt `rc-service tlp start` auf (und prüft nicht, ob `tlp` auf der Maschine sinnvoll ist) und lädt `ldconfig` hart.

---

## Teil 2: Zielarchitektur

Das `Dockerfile` wird auf **Multi-Stage-Build mit BuildKit-Export** umgestellt. Damit gehen beide Varianten (`nv` für HP Z6, `e14` für ThinkPad E14) von derselben unveränderten `rootfs`-Stage aus, und `setup03_copy_from_container.sh` braucht kein `docker run --privileged` und kein `copy_files.sh` mehr:

```text
gentoo/portage:20261106  (Stage: portage)
gentoo/stage3:<TAG>      (Stage: rootfs)
  ├── Portage-Setup, Kernel, Locale
  ├── Chunked @world-Build (10 Blöcke REAKTIVIERT + Konvergenz-@world)
  ├── slstatus (config.h!), User kiel, Emacs/MELPA, OpenRC-Basis
  ├── Dracut initramfs + amd-uc.img + /packages.{txt,tsv}
  ├── Reality-Check & Smoke-Tests
  │
  ├──► variant-nv  (FROM rootfs): reverse-ssh-*, nur NVIDIA-Firmware → /out/gentoo.squashfs_nv
  ├──► variant-e14 (FROM rootfs): E14-iwd/openresolv, E14-Firmware,
  │                               NVIDIA per `emerge -C` entfernen → /out/gentoo.squashfs_e14
  │
  └──► export (FROM scratch): sammelt Artefakte beider Varianten + Kernel/Initramfs/Listen
```

Verzeichnisstruktur nach dem Aufräumen:
- `archive/` (oder Löschung per `git rm`): alte Einmal-Skripte (`nethex`, `start-reverse-tunnels.sh`, `config-gentoo.sh`, `NetworkManager.conf`, `copy_files.sh`)
- `config/host-reference/hpz6/`: Referenzdateien (`50-activate.start`, `docker-disk.initd`, `containerd-config.toml`), klar getrennt von Dateien, die ins Image kopiert werden
- `config/`: nur noch Dateien, die tatsächlich vom `Dockerfile` verwendet werden
- `install_on_e14.sh` & `doc/`: bleiben funktional intakt (`install_on_e14.sh` nicht anfassen!)

---

## Teil 3: Expliziter Ausführungsplan (für einen AI-Agenten)

> **Anweisungen an den ausführenden Agenten:**
> Arbeite die Phasen **strikt der Reihe nach** ab. Überspringe keine Verifikationskommandos. Wenn ein Check fehlschlägt, brich ab und behebe den Fehler in derselben Phase, bevor du weitergehst. Ändere `install_on_e14.sh` **nicht**.

---

### Phase 0: Ausgangszustand sichern

**Ziel:** Cleanes Git-Backup / Snapshot vor Änderungen, sodass jeder Schritt per `git diff` prüfbar ist.

1. Prüfe Arbeitsverzeichnis:
   ```bash
   cd /home/kiel/stage/cl-py-generator/example/110_gentoo/openrc
   git status
   ```
2. Erzeuge ein lokales Backup-Verzeichnis für die zu verändernden Dateien:
   ```bash
   mkdir -p /tmp/openrc_cleanup_backup
   cp -a Dockerfile *.sh config/ /tmp/openrc_cleanup_backup/
   ```
3. **Test:** `test -f /tmp/openrc_cleanup_backup/Dockerfile` muss Exit-Code `0` liefern.

---

### Phase 1: Verzeichnisstruktur & Altlasten aufräumen

**Ziel:** Ungenutzte/veraltete Dateien aus `config/` entfernen bzw. Host-Referenzdateien sauber trennen.

1. Verschiebe die HP-Z6-Host-Referenzdateien (sie werden nicht ins Image kopiert, siehe `doc/docker_data_disk_and_boot.md`) nach `config/host-reference/hpz6/`:
   ```bash
   mkdir -p config/host-reference/hpz6
   mv config/50-activate.start config/host-reference/hpz6/
   mv config/docker-disk.initd config/host-reference/hpz6/
   mv config/containerd-config.toml config/host-reference/hpz6/
   ```
2. Verschiebe obsolete Altlasten in ein `archive/`-Verzeichnis (damit nichts verloren geht, aber `config/` sauber ist):
   ```bash
   mkdir -p archive
   mv config/start-reverse-tunnels.sh archive/
   mv config/nethex archive/
   mv config/config-gentoo.sh archive/
   mv config/NetworkManager.conf archive/
   ```
   *(Hinweis: `copy_files.sh` wird in Phase 5 ebenfalls nach `archive/` verschoben.)*
3. Härte `collect.sh`, damit Dateinamen sauber gequotet werden und das neue `archive/` nicht mitgesammelt wird:
   Schreibe `collect.sh` neu mit exakt diesem Inhalt:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   find Dockerfile ./*.sh config/ doc/ -type f | sort | while IFS= read -r f; do
     echo "//start of  ${f}"
     cat "${f}"
     echo ""
   done
   ```
   ```bash
   chmod +x collect.sh
   ```
4. **Test:**
   ```bash
   bash -n collect.sh
   for f in 50-activate.start docker-disk.initd containerd-config.toml; do
     test -f "config/host-reference/hpz6/$f"
   done
   for f in start-reverse-tunnels.sh nethex config-gentoo.sh NetworkManager.conf; do
     test -f "archive/$f"
     test ! -e "config/$f"
   done
   ```

---

### Phase 2: Portage-Konfiguration, `world` & `resolv.conf` bereinigen

**Ziel:** Doppelte `package.use`-Zeilen zusammenführen, `gentoolkit`-Widerspruch auflösen, fehlende Newlines ergänzen.

1. **`config/etc/portage/package.use`**:
   - Entferne doppelte Zeilen für `sys-fs/lvm2` und `www-client/firefox` und führe die Flags jeweils in einer Zeile zusammen:
     - `sys-fs/lvm2 lvm readline`
     - `www-client/firefox -clang -dbus -gmp-autoupdate hwaccel -jumbo-build system-pipewire system-png -telemetry`
   - Ergänze bei `app-text/mupdf` das `X`-Flag **nur**, wenn die Prüfung im Portage-Snapshot ergibt, dass `X` ein gültiges IUSE-Flag von `app-text/mupdf` ist:
     ```bash
     docker run --rm gentoo/portage:20261106 grep -E 'IUSE=.*\bX\b' /var/db/repos/gentoo/app-text/mupdf/*.ebuild || true
     ```
     Falls Ausgabe nicht leer ist, setze in `package.use`: `app-text/mupdf X -javascript -jpeg2k`. Falls leer, belasse `app-text/mupdf -javascript -jpeg2k`.
   - Stelle sicher, dass die Datei mit `\n` endet.
   - Exakte bereinigte `config/etc/portage/package.use` (ohne Duplikate):
     ```text
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
     ```
     *(Falls der Portage-Check oben `IUSE=...X...` für `mupdf` bestätigt hat, ersetze `app-text/mupdf -javascript -jpeg2k` durch `app-text/mupdf X -javascript -jpeg2k`.)*

2. **`config/var/lib/portage/world`**:
   - Behalte alle aktuellen Pakete (inkl. `app-portage/gentoolkit`, da es im `Dockerfile` künftig **nicht** mehr unmerged wird) und ergänze eine abschließende Newline `\n`:
     ```bash
     printf '\n' >> config/var/lib/portage/world
     # Doppelt-Newlines am Ende vermeiden:
     sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' config/var/lib/portage/world
     printf '\n' >> config/var/lib/portage/world
     ```
3. **`config/resolv.conf`**:
   - Abschließende Newline ergänzen:
     ```bash
     printf 'nameserver 194.11.90.70\nsearch lgs-net.com\nnameserver 8.8.8.8\n' > config/resolv.conf
     ```
4. **Test:**
   ```bash
   # Keine doppelten Paketatome in package.use:
   awk '!/^\s*(#|$)/ {print $1}' config/etc/portage/package.use | sort | uniq -d
   # Erwartet: leere Ausgabe!
   test "$(tail -c1 config/var/lib/portage/world | wc -l)" -eq 1
   test "$(tail -c1 config/resolv.conf | wc -l)" -eq 1
   ```

---

### Phase 3: OpenRC-Session, `xinitrc` und `activate` bereinigen

**Ziel:** Single-Owner-Prinzip für D-Bus und PipeWire herstellen, `xrandr` auf HP Z6 eingrenzen, `activate` härten.

1. **`config/xinitrc`**:
   - `xrandr` nur ausführen, wenn `DP-0` und `DP-2` verbunden sind (oder nicht auf E14).
   - **Kein** zweiter `dbus-run-session`-Start und **kein** doppeltes `~/start-pipewire.sh &`, da `user.kiel` (OpenRC User-Session) bereits `dbus`, `pipewire`, `pipewire-pulse` und `wireplumber` startet. Falls `DBUS_SESSION_BUS_ADDRESS`-Socket wider Erwarten fehlt, nur dann Fallback auf `dbus-run-session`.
   - Ersetze `config/xinitrc` exakt durch:
     ```sh
     #!/bin/sh

     # HP Z6 multi-monitor layout only when those outputs exist
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
2. **`config/user-runtime.initd`**:
   - Ist bereits konsistent (erstellt `/run/user/1000` und aktiviert `dbus`, `pipewire`, `pipewire-pulse`, `wireplumber` in `/home/kiel/.config/rc/runlevels/default/`). Im `Dockerfile` werden diese 3 PipeWire-Symlinks ab sofort ebenfalls wieder **aktiv** (nicht mehr auskommentiert), sodass `Dockerfile` und `user-runtime.initd` übereinstimmen.
3. **`config/activate`**:
   - Entferne den veralteten `config_root`-Fallback auf `/home/kiel/stage/...` nicht komplett, wenn gewünscht, aber benutze `rc-service` statt `/etc/init.d/tlp start` und mach `tlp` fehlertolerant (`rc-service tlp start 2>/dev/null || true`).
   - Entferne `depmod -a` / `env-update` / `ldconfig`? **Wichtig:** `doc/install_on_e14.md` dokumentiert, dass bei einem persistenten Overlay ein altes `/etc/ld.so.cache` aus dem vorherigen Slot liegen bleiben kann. Daher soll `ldconfig` **nur** laufen, wenn `/etc/ld.so.cache` älter ist als `/etc/os-release` (oder `/IMAGE-BUILD-INFO.txt`) – dann wird das Overlay nicht bei jedem Boot unnötig beschrieben!
   - Ersetze `config/activate` exakt durch:
     ```sh
     #!/bin/sh
     set -eu

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

     config_root=/usr/local/share/openrc-host-config
     if [ ! -d "${config_root}" ] && [ -d /home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/config ]; then
       config_root=/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc/config
     fi

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

     # Refresh linker cache only if stale compared to current image build info
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
4. **Test:**
   ```bash
   sh -n config/xinitrc
   sh -n config/activate
   ```

---

### Phase 4: `Dockerfile` bereinigen, Chunked-`@world` reaktivieren & Multi-Stage-Build umsetzen

**Wichtige Vorgaben für den Agenten:**
- Die Kommentare zur **Aufteilung des `@world`-Builds in 10 Blöcke** (inkl. des Hinweises auf `xargs -d '\n'` und `logs/setup01_run_20261106_182536.log`) **MÜSSEN erhalten bleiben**!
- Die 10-Chunk-Variante für `@world` **MUSS wieder aktiviert** sein (mit abschließendem `emerge -uDN @world` als Konvergenzschritt und Cleanup der `/tmp/world_*`-Dateien; die reine Ein-Schritt-Variante bleibt als auskommentierte Alternative dokumentiert).
- Die ausführlichen Kommentare zur **Minimal-Firmware des ThinkPad E14** **MÜSSEN erhalten bleiben**.
- `COPY config/slstatus_config.h config.h` (nicht `.`!).
- `COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm` (ohne Versionssuffix, damit Portage `savedconfig` immer greift).
- `chown kiel:kiel` auf `/home/kiel/.xinitrc` ergänzen.
- Keine `|| true` auf `mksquashfs`!
- In `variant-e14`: NVIDIA sauber per `emerge -C x11-drivers/nvidia-drivers` + gezieltem `rm -rf` entfernen, **NICHT** `rm -rf /usr/lib64/libnv*` verwenden!
- `amd-uc.img` direkt im Build aus `/usr/lib/firmware/amd-ucode/microcode_amd*.bin` als `cpio` (`newc`) nach `/boot/amd-uc.img` bauen.
- `app-portage/gentoolkit` **nicht** mehr per `emerge -C` entfernen (es steht in `world`).
- Entferne die toten Blöcke (`ryzen_smu`, altes `mksquashfs`, `ext4_root`, `gnatwhy3`/`ocaml`, `freerdp` accept_keywords-Snippet, `emerge --depclean`-Kommentar, doppeltes `sshd` / `cgroups boot`, frühe doppelte `packages.txt`-Erzeugung, nutzloses `RUN env-update && source /etc/profile`, wirkungsloses `mkdir /run/user/1000`, `erofs`-Kommentarblock falls nicht genutzt sowie `emerge sys-fs/e2fsprogs sys-fs/erofs-utils || true`).
  *(Hinweis: Falls du den EROFS-Kommentarblock als Doku-Referenz behalten willst, lass ihn als kurzen Verweis auf `doc/erofs.md` stehen, ohne tote `emerge`-Befehle.)*

1. Schreibe `Dockerfile` exakt mit folgendem Inhalt:

```dockerfile
# syntax=docker/dockerfile:1

# Base images
ARG PORTAGE_TAG=20261106
# Pin stage3 tag for reproducibility (override via --build-arg STAGE3_TAG=...)
ARG STAGE3_TAG=nomultilib

FROM gentoo/portage:${PORTAGE_TAG} AS portage

# Stage3 base (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:${STAGE3_TAG} AS rootfs

# Copy full Portage tree (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

ARG BUILD_GIT_COMMIT=unknown
ARG BUILD_REPOSITORY=unknown
ARG BUILD_DATE=unknown

# Portage configuration
RUN eselect profile set default/linux/amd64/23.0/no-multilib
COPY config/etc/portage/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
RUN mkdir -p /etc/portage/package.mask
COPY config/etc/portage/package.mask /etc/portage/package.mask/package.mask

# Repository tooling and package configuration
COPY config/etc/portage/make.conf /etc/portage/make.conf
COPY config/etc/portage/package.use /etc/portage/package.use
COPY config/etc/portage/package.env /etc/portage/package.env
COPY config/etc/portage/env /etc/portage/env
COPY config/var/lib/portage/world /var/lib/portage/world
COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm

ENV KVER_PURE=6.18.60
ENV KVER_SOURCE=${KVER_PURE}-gentoo
ENV KVER_RELEASE=${KVER_SOURCE}-dist

RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE} \
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
 && make install

# Clean up the locales
RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN emerge -1 sys-apps/portage

# --- World-Update in 10 Blöcken (aktiviert) ---
# Letzter erfolgreicher Chunk-Build: logs/setup01_run_20261106_182536.log (exit 0).
# Ein Fehler stoppt nur seinen Block, die davor liegenden Blöcke sind im
# Layer-Cache. Atome werden über xargs -d '\n' übergeben, damit Sonderzeichen
# wie das '<' in sys-kernel/gentoo-sources-6.19 nicht von der Shell
# interpretiert werden.
# (Für ein Ein-Schritt-Update diesen Chunk-Block auskommentieren und nur die
# abschließende `RUN emerge -uDN @world`-Zeile aktiv lassen.)
RUN awk '!/^[[:space:]]*(#|$)/ {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print}' /var/lib/portage/world > /tmp/world_atoms.txt \
 && split -n l/10 -d /tmp/world_atoms.txt /tmp/world_stage_ \
 && wc -l /tmp/world_stage_*
RUN xargs -r -d '\n' -a /tmp/world_stage_00 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_01 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_02 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_03 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_04 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_05 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_06 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_07 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_08 emerge -uDN
RUN xargs -r -d '\n' -a /tmp/world_stage_09 emerge -uDN

# Abschluss-Konvergenz (stellt sicher, dass auch @system und Paketabhängigkeiten
# zwischen den Blöcken konsistent sind) und Aufräumen der temporären Listen.
RUN emerge -uDN @world \
 && rm -f /tmp/world_atoms.txt /tmp/world_stage_*

RUN gcc-config -l && qlist -Iv sys-devel/gcc

# Sudo policy
RUN mkdir -p /etc/sudoers.d \
 && echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel

# slstatus (config.h must match slstatus Makefile target)
WORKDIR /usr/src
RUN git clone https://git.suckless.org/slstatus
WORKDIR /usr/src/slstatus
COPY config/slstatus_config.h config.h
RUN make -j32 \
 && make install \
 && make clean

RUN groupadd -f input \
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

# Network and services (host-specific reverse-ssh-* runlevels are enabled only in variant-nv)
COPY config/resolv.conf /etc/
COPY config/reverse-ssh-eu.initd /etc/init.d/reverse-ssh-eu
COPY config/reverse-ssh-us.initd /etc/init.d/reverse-ssh-us
RUN chmod +x /etc/init.d/reverse-ssh-*

# Switch to user 'kiel' to install Emacs packages from MELPA in batch mode
USER kiel
ENV HOME=/home/kiel
RUN emacs --batch \
    --eval "(require 'package)" \
    --eval "(add-to-list 'package-archives '(\"melpa\" . \"https://melpa.org/packages/\") t)" \
    --eval "(package-initialize)" \
    --eval "(package-refresh-contents)" \
    --eval "(dolist (pkg '(magit paredit slime)) (package-install pkg))"

# Switch back to root user for the remaining steps
USER root
ENV HOME=/root

RUN mkdir -p /usr/local/share/openrc-host-config
COPY config/activate /usr/local/share/openrc-host-config/activate
COPY config/start2 /usr/local/share/openrc-host-config/start2
COPY config/reverse-ssh-eu.initd /usr/local/share/openrc-host-config/reverse-ssh-eu.initd
COPY config/reverse-ssh-us.initd /usr/local/share/openrc-host-config/reverse-ssh-us.initd

# For chinese font in chrome / desktop
RUN fc-cache -fv

RUN printf '%s\n' \
      'keymap="colemak/en-latin9"' \
      > /etc/conf.d/keymaps \
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

# Build-time dependencies (not needed at runtime)
RUN emerge -C dev-lang/rust-bin virtual/rust dev-lang/go dev-lang/go-bootstrap dev-util/cargo-c 2>/dev/null || true \
 && rm -rf /var/tmp/portage/*

WORKDIR /
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

# Dracut Module documentation https://dracut-ng.github.io/dracut-ng/modules/core.html
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

# Generate external AMD microcode image for GRUB initrd preloading
RUN set -e \
 && mkdir -p /tmp/ucode/kernel/x86/microcode \
 && cat /usr/lib/firmware/amd-ucode/microcode_amd*.bin > /tmp/ucode/kernel/x86/microcode/AuthenticAMD.bin \
 && (cd /tmp/ucode && find kernel | cpio -o -H newc > /boot/amd-uc.img) \
 && rm -rf /tmp/ucode \
 && test -s /boot/amd-uc.img

# Generate package reports late so they match the final image contents.
RUN set -e \
 && emerge --ask=n --oneshot app-portage/genlop \
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
        time_human=$(genlop -t "$pkg" 2>/dev/null | sed -nE 's/.*merge time: (.*)/\1/p' | tail -n 1 | sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g'); \
        build_seconds=$(TIME_HUMAN="$time_human" awk 'BEGIN{t=ENVIRON["TIME_HUMAN"]; gsub(/,/, "", t); n=split(t, a, / +/); s=0; for (i=1; i<=n; i++) { if (a[i] ~ /^[0-9]+$/) { v=a[i]+0; u=a[i+1]; if (u ~ /^day/) s+=v*86400; else if (u ~ /^hour/) s+=v*3600; else if (u ~ /^minute/) s+=v*60; else if (u ~ /^second/) s+=v; } } print s }'); \
      else \
        time_human='-'; \
        build_seconds=0; \
      fi; \
      printf '%s\t%s\t%s\t%s\n' "$pkg" "$size_mib" "${build_seconds:-0}" "${time_human:-N/A}" >> /packages.tsv; \
    done \
 && { \
      printf '%-40s %10s %14s  %s\n' "Package" "Size MiB" "Build Seconds" "Build Time"; \
      tail -n +2 /packages.tsv | sort -t "$(printf '\t')" -k2,2nr -k3,3nr | awk -F '\t' '{printf "%-40s %10s %14s  %s\n", $1, $2, $3, $4}'; \
    } > /packages.txt \
 && emerge -C app-portage/genlop \
 && rm -rf /var/tmp/portage/*

# Reality & smoke checks on shared rootfs before creating variants
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
 && for bin in dwm slstatus emacs xfreerdp nvme firefox; do \
      command -v "${bin}" >/dev/null; \
    done \
 && lsinitrd /boot/initramfs_squash_sda1-x86_64.img | grep -q 'mount-overlayfs.sh' \
 && cpio -it < /boot/amd-uc.img | grep -q 'AuthenticAMD.bin' \
 && test -s /boot/vmlinuz

# ==============================================================================
# Variant 1: HP Z6 (NVIDIA + reverse SSH tunnels)
# ==============================================================================
FROM rootfs AS variant-nv

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
FROM rootfs AS variant-e14

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
RUN set -e \
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
 # Strip NVIDIA/CUDA cleanly without touching unrelated libs like libnvme
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
# Export stage (used via `docker build --target export --output ...`)
# ==============================================================================
FROM scratch AS export
COPY --from=variant-nv  /out/gentoo.squashfs_nv /
COPY --from=variant-e14 /out/gentoo.squashfs_e14 /
COPY --from=rootfs      /boot/vmlinuz /
COPY --from=rootfs      /boot/initramfs_squash_sda1-x86_64.img /
COPY --from=rootfs      /boot/amd-uc.img /
COPY --from=rootfs      /packages.txt /
COPY --from=rootfs      /packages.tsv /
COPY --from=rootfs      /IMAGE-BUILD-INFO.txt /
```

2. **Test (Syntax-Check vor dem Build):**
   - Prüfe, ob alle vom `Dockerfile` referenzierten `COPY`-Quellen existieren:
     ```bash
     grep -E '^\s*COPY\s+' Dockerfile | grep -v -- '--from=' | awk '{print $2}' | sort -u | while read -r p; do
       test -e "$p" || { echo "MISSING COPY SOURCE: $p"; exit 1; }
     done
     ```
   - Prüfe, dass `|| true` nicht mehr bei `mksquashfs` steht:
     ```bash
     ! grep -n '|| true' Dockerfile | grep -iE 'mksquashfs|squashfs'
     ```
   - Prüfe, dass `slstatus` nach `config.h` kopiert wird:
     ```bash
     grep -q '^COPY config/slstatus_config.h config.h$' Dockerfile
     ```
   - Prüfe, dass die Chunked-`@world`-Zeilen aktiv sind (nicht auskommentiert):
     ```bash
     grep -q '^RUN xargs -r -d .* -a /tmp/world_stage_00 emerge -uDN$' Dockerfile
     grep -q '^RUN xargs -r -d .* -a /tmp/world_stage_09 emerge -uDN$' Dockerfile
     ```

---

### Phase 5: Build- und Export-Skripte aktualisieren

**Ziel:** Unsichere/tote `--build-arg`-Secrets entfernen, Metadaten (`BUILD_GIT_COMMIT`, `BUILD_DATE`) sauber übergeben und Export ohne `--privileged docker run` / `copy_files.sh` durchführen.

1. **`setup01_build_image.sh`** exakt ersetzen durch:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
   if ! git diff --quiet 2>/dev/null; then
     GIT_COMMIT="${GIT_COMMIT}-dirty"
   fi
   REPO_URL="$(git config --get remote.origin.url 2>/dev/null || echo local)"
   BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

   DOCKER_BUILDKIT=1 docker build \
     --build-arg BUILD_GIT_COMMIT="${GIT_COMMIT}" \
     --build-arg BUILD_REPOSITORY="${REPO_URL}" \
     --build-arg BUILD_DATE="${BUILD_DATE}" \
     -t gentoo-z6-min-openrc \
     --progress=plain \
     .
   ```
   *(Beachte: Damit baut `setup01` die letzte Stage `export`, löst also `rootfs`, `variant-nv` und `variant-e14` aus und füllt den BuildKit-Cache.)*

2. **`setup03_copy_from_container.sh`** exakt ersetzen durch:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   TODAY="$(date +%Y%m%d)"
   TARGET="/dev/shm/gentoo-z6-min-openrc_${TODAY}"
   TMP_TARGET="${TARGET}.tmp.$$"

   rm -rf "${TMP_TARGET}"
   mkdir -p "${TMP_TARGET}"

   DOCKER_BUILDKIT=1 docker build \
     --target export \
     --output "type=local,dest=${TMP_TARGET}" \
     .

   rm -rf "${TARGET}"
   mv "${TMP_TARGET}" "${TARGET}"
   ls -lh "${TARGET}"
   ```
   *(Atomares Umbenennen nach abgeschlossenem Export; kein `chmod -R a+rwx`, kein `--privileged docker run` mehr nötig.)*

3. **`copy_files.sh`** ins Archiv verschieben:
   ```bash
   mv copy_files.sh archive/
   ```

4. **Test:**
   ```bash
   chmod +x setup01_build_image.sh setup03_copy_from_container.sh
   bash -n setup01_build_image.sh
   bash -n setup03_copy_from_container.sh
   bash -n setup01_run_with_log.sh
   bash -n install_on_e14.sh
   ```

---

### Phase 6: Build & End-to-End-Verifikation

**Ziel:** Gesamten Build mit Log ausführen, Artefakte exportieren und sowohl die Export-Dateien als auch `install_on_e14.sh --dry-run` testen.

1. **Docker-Build mit Logging starten:**
   ```bash
   bash ./setup01_run_with_log.sh
   ```
   - Prüfe `EXIT_CODE=0` am Ende der Ausgabe und in `logs/setup01_run_*.meta`.
   - Falls ein Chunk `world_stage_XX` wegen deinen Änderungen in `world` / `package.use` fehlschlägt: im Log steht exakt der fehlgeschlagene Block; alle vorherigen Chunks liegen im Docker-Cache.

2. **Artefakte nach `/dev/shm` exportieren:**
   ```bash
   bash ./setup03_copy_from_container.sh
   ```
   - Prüfe, dass exakt diese 8 Dateien im Export-Ordner `/dev/shm/gentoo-z6-min-openrc_$(date +%Y%m%d)/` liegen und nicht leer (`-s`) sind:
     ```bash
     OUT="/dev/shm/gentoo-z6-min-openrc_$(date +%Y%m%d)"
     for f in gentoo.squashfs_nv gentoo.squashfs_e14 vmlinuz \
              initramfs_squash_sda1-x86_64.img amd-uc.img \
              packages.txt packages.tsv IMAGE-BUILD-INFO.txt; do
       test -s "${OUT}/${f}" || { echo "MISSING OR EMPTY: ${f}"; exit 1; }
     done
     ```

3. **Inhaltliche Stichproben auf den Squashfs-Images (falls `unsquashfs` auf dem Host verfügbar):**
   ```bash
   # 1. slstatus wurde wirklich mit slstatus_config.h gebaut (nicht Default-config.def.h)
   # 2. E14 hat libnvme.so, aber KEINE nvidia-Treiberreste; Z6 hat nvidia.ko-Firmware
   unsquashfs -ll "${OUT}/gentoo.squashfs_e14" usr/lib64 | grep -q 'libnvme\.so'
   ! unsquashfs -ll "${OUT}/gentoo.squashfs_e14" | grep -q 'nvidia-smi'
   unsquashfs -ll "${OUT}/gentoo.squashfs_e14" etc/iwd/main.conf | grep -q 'main.conf'
   unsquashfs -ll "${OUT}/gentoo.squashfs_nv" etc/runlevels/default | grep -q 'reverse-ssh-eu'
   ! unsquashfs -ll "${OUT}/gentoo.squashfs_e14" etc/runlevels/default | grep -q 'reverse-ssh-eu'
   ```

4. **Dry-Run des E14-Installers gegen den neuen Export (auf dem E14-Host):**
   ```bash
   sudo bash ./install_on_e14.sh --src "${OUT}" --dry-run
   ```

5. **Abschluss-Check des Repos:**
   ```bash
   git status
   # Aufräumen des Phase-0-Backups erst nach erfolgreichem Build & Boot-Test:
   # rm -rf /tmp/openrc_cleanup_backup
   ```

---

### Kurzer Hinweis noch für dich vor dem Start:
1. Prüfe vor dem Ausführen kurz `config/dwm-6.8` bzw. `emerge =sys-kernel/gentoo-sources-6.18.60`: Wenn du in `world` viele Pakete ergänzt hast, dauert der erste Durchlauf von `world_stage_00`..`09` entsprechend – dank Layer-Cache bleibt jeder bereits gebaute Block bei einem Abbruch erhalten.
2. In `config/resolv.conf` steht aktuell noch `194.11.90.70` als erster Nameserver (auf dem E14 überschreibt `openresolv`/`iwd` das nach WLAN-Connect; auf dem HP Z6 bleibt es statisch). Falls das beabsichtigt ist, so lassen – falls nicht, in Phase 2 auf `8.8.8.8` / `1.1.1.1` umstellen.
