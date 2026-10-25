# syntax=docker/dockerfile:1
# Minimal-OpenRC-Testimage (openrc2): Konsolen-System für QEMU, ohne ASan.
# Vollsystem mit Hardware-Support: siehe ../openrc/Dockerfile (bleibt unverändert).

FROM gentoo/portage:20261021 AS portage

FROM gentoo/stage3:nomultilib-20261019 AS base
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# HINWEIS Build-Args: BUILD_* stehen ABSICHTLICH erst spaet (vor Squashfs).
# BUILD_DATE aendert sich pro Run und wuerde sonst den gesamten Cache ab
# Zeile 12 entwerten (jeder Build = Voll-Rebuild, belegt Run 12-18).
RUN eselect profile set default/linux/amd64/23.0/no-multilib

# Minimal-Konfiguration (kein X, keine HW-Treiber). Cache-Regel: Hier nur
# STABILE Dateien — package.env (Batch-wechselnd), Suppressions und ENV kommen
# spaet (vor der Oneshot-Stufe), damit Kernel-/Welt-Layer aus dem Cache kommen.
COPY config/make.conf /etc/portage/make.conf
COPY config/package.use /etc/portage/package.use
COPY config/package.env.base /etc/portage/package.env
COPY config/env /etc/portage/env
COPY config/world /var/lib/portage/world
RUN mkdir -p /etc/portage/package.mask
COPY config/package.mask /etc/portage/package.mask/package.mask

# Gleicher Kernel-Pin wie openrc/ (Kompatibilität zum Hardware-Pfad)
ENV KVER_PURE=6.18.57
ENV KVER_SOURCE=${KVER_PURE}-gentoo
ENV KVER_RELEASE=${KVER_SOURCE}-dist

RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

RUN eselect kernel list
RUN eselect kernel set linux-${KVER_SOURCE}
# Oracle-Basis: 1:1-Kopie von ../openrc/config/config6.18.48 (Stand 2026-10-09).
# Nur das Fragment kernel-qemu.fragment weicht davon ab (dokumentiert).
WORKDIR /usr/src/linux
COPY config/config-base .config
COPY config/kernel-qemu.fragment /tmp/kernel-qemu.fragment
RUN sed -i \
    -e 's/^CONFIG_LOCALVERSION="-gentoo-dist"/CONFIG_LOCALVERSION="-dist"/' \
    -e 's/^CONFIG_DEBUG_INFO=y/# CONFIG_DEBUG_INFO is not set/' \
    -e 's/^CONFIG_DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT=y/# CONFIG_DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT is not set/' \
    -e 's/^CONFIG_DEBUG_INFO_BTF=y/# CONFIG_DEBUG_INFO_BTF is not set/' \
    -e 's/^CONFIG_DEBUG_INFO_BTF_MODULES=y/# CONFIG_DEBUG_INFO_BTF_MODULES is not set/' \
    .config \
 && for sym in VIRTIO VIRTIO_PCI VIRTIO_BLK VIRTIO_NET VIRTIO_CONSOLE VIRTIO_FS \
      SERIAL_8250 SERIAL_8250_CONSOLE SQUASHFS SQUASHFS_ZSTD SQUASHFS_XATTR \
      OVERLAY_FS BLK_DEV_DM DM_CRYPT EXT4_FS VFAT_FS; do \
      ./scripts/config --enable "${sym}"; \
    done \
 && for sym in DRM_AMDGPU DRM_NOUVEAU DRM_I915 SND WLAN BT; do \
      ./scripts/config --disable "${sym}"; \
    done \
 && make olddefconfig && make prepare

# Kernel kompilieren
RUN make -j32
RUN make modules_install
RUN depmod -a ${KVER_RELEASE}
RUN make install

# Locales
RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update \
 && . /etc/profile
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN env-update && source /etc/profile
RUN emerge -1 sys-apps/portage

# Welt-Update (Binärpakete OK: Sanitizer kommen pro Batch per package.env,
# s. naechste Stufe). Ohne X11/Desktop/HW-Treiber geht das schnell.
RUN emerge -uDN @world
RUN gcc-config -l

# Sanitizer-Batches: config/package.env (von tools/apply-batch.sh erzeugt)
# weist Paketen env/sanitizers zu. Ein CFLAGS-Wechsel loest allein keinen
# Rebuild aus und Binärpakete wuerden die Flags aushebeln — daher hier
# explizit aus Quellen neu bauen. Alles Flüchtige (package.env, Suppressions,
# ENV, Portage-Workaround) liegt HIER, damit darueber alles gecacht bleibt.
COPY config/package.env /etc/portage/package.env
COPY config/patches /etc/portage/patches
COPY config/sanitized-packages.txt /tmp/sanitized-packages.txt
COPY config/asan-suppressions /etc/asan-suppressions
COPY config/bashrc /etc/portage/bashrc
COPY config/dracut-00-lsan.sh /etc/portage/openrc2-dracut-00-lsan.sh
COPY config/dracut-99-unset-preload.sh /etc/portage/openrc2-dracut-99-unset-preload.sh
# Meson-Wrapper (Run-39): nacktes `meson` der Eclass (setup/test/install)
# landet hier, wenn config/env/sanitizers-meson den PATH voranstellt.
# NUR Meson bekommt die Vorladung (python+Meson brauchen libasan zuerst wegen
# dlopen sanitisierter Libs); Portages eigene Mechanik bleibt unberuehrt
# (globales LD_PRELOAD bricht __save_ebuild_env, Run-38 rot).
# Pfad als Literal (kein $(...), Run-37 rot); gcc-Slot bei Wechsel pruefen.
RUN mkdir -p /usr/local/asan-wrappers \
 && printf '%s\n' '#!/bin/sh' \
   'exec env LD_PRELOAD="/usr/lib/gcc/x86_64-pc-linux-gnu/16/libasan.so.8" LSAN_OPTIONS="exitcode=0" /usr/bin/meson "$@"' \
   > /usr/local/asan-wrappers/meson \
 && chmod 755 /usr/local/asan-wrappers/meson \
 && /usr/local/asan-wrappers/meson --version
# Leak-Policy (s. asan-worklog.md): NUR via package.env (config/env/sanitizers)!
# KEIN Docker-ENV LSAN_OPTIONS: Vererbte Werte ueberschreiben package.env in
# Merge-Phasen (Gast-/Docker-Beweis: Phasen-Env zeigte Docker-Form statt plain)
# und suppressions-Pfade voiden exitcode bei Unlesbarkeit. ASan/UBSan lethal.
# Portage-Workaround (s. asan-worklog.md): __save_ebuild_env() unsetet sich
# selbst + `declare -fp` in Pipe = Exit 1 unter ASan-Bash (0 sonst) -> alle
# Merge-Phasen tot. MUSS nach `emerge portage` stehen (Binärpaket wuerde die
# Datei sonst ueberschreiben). Bis Bash das sauber kann.
RUN f=/usr/lib/portage/python3.14/save-ebuild-env.sh \
 && echo "vorher:" && grep -n '__save_ebuild_env$' "$f" \
 && sed -i '/^\t\t__save_ebuild_env$/d' "$f" \
 && echo "nachher:" && grep -n '__save_ebuild_env$' "$f" \
 && test "$(grep -c '__save_ebuild_env$' "$f")" -eq 1
RUN if [ -s /tmp/sanitized-packages.txt ]; then \
      FEATURES="-getbinpkg -binpkg-request-signature" \
      emerge --oneshot $(grep -vE '^\s*(#|$)' /tmp/sanitized-packages.txt); \
    fi || { echo "=== EVIDENCE-DUMP ==="; \
      for l in /var/tmp/portage/*/*/temp/build.log; do \
        [ -f "$l" ] || continue; echo "--- $l"; \
        grep -a -m2 -B2 -A22 -i "sanitizer\|direct leak\|indirect leak\|runtime error\|READ of size\|heap-\|does not come first\|meson setup\|No such file or directory" "$l" | head -n 50; \
        echo "--- tail $l"; tail -n 25 "$l"; \
      done; \
      for e in /var/tmp/portage/*/*/temp/environment; do \
        [ -f "$e" ] || continue; echo "--- env $e"; \
        grep -a -o "^declare -x \(LSAN_OPTIONS\|ASAN_OPTIONS\|LD_PRELOAD\|FEATURES\|SANDBOX_ON\)=.*" "$e" | head -n 8; \
      done; echo "=== DUMP-ENDE ==="; exit 1; }

# openssh plain AUS QUELLEN neu bauen (package.use: -seccomp, Stapel 2+).
# Das Binärpaket käme MIT Seccomp-Filter; der erschiesst (SIGSYS) jede
# Anmeldung, sobald libasan im Prozess ist (pam_openrc.so, Batch-02).
# Eigener Schritt (nicht im Batch): bleibt plain bis Stapel 3.
# LSAN-Export, weil configure mit sanitisierten Tools arbeitet.
RUN export LSAN_OPTIONS="exitcode=0" \
 && FEATURES="-getbinpkg -binpkg-request-signature" \
    emerge --oneshot net-misc/openssh

# Sudo + passwortloser Konsolen-User für QEMU-Tests
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel

# Gast-Netz + schneller SSH-Login: DHCP auf eth0 (QEMU user-net), kein
# DNS-Reverse-Lookup beim Login (Haenger im Banner ohne DNS).
RUN export LSAN_OPTIONS="exitcode=0" \
 && printf 'config_eth0="dhcp"\n' > /etc/conf.d/net \
 && ln -sf /etc/init.d/net.lo /etc/init.d/net.eth0 \
 && rc-update add net.eth0 default \
 && printf 'UseDNS no\n' >> /etc/ssh/sshd_config

# Udev-Dienst unter Sanitizer-Libs: udevd laedt libblkid (Batch-01) per
# dlopen. Ohne Preload an erster Stelle startet der Dienst nicht (Beweis:
# Gast-7, Dienst gestoppt). conf.d wird vom Dienst gesourct, Env vererbt.
RUN export LSAN_OPTIONS="exitcode=0" \
 && printf 'export LSAN_OPTIONS="exitcode=0"\nfor _f in /usr/lib/gcc/*/*/libasan.so; do [ -f "$_f" ] && { LD_PRELOAD="$_f"; break; }; done\nexport LD_PRELOAD\nunset _f\n' >> /etc/conf.d/udev

# Gleiches Muster fuer sshd: dessen PAM-Stapel laedt pam_openrc.so
# (Batch-02, DT_NEEDED libasan). Ohne Preload stirbt das Kind beim
# Schluessel-Angebot ("Connection closed", Docker-Beweis). Mit Preload
# konsistent. Bleibt auch unter sanitisiertem openssh (Stapel 3) gueltig.
RUN export LSAN_OPTIONS="exitcode=0" \
 && printf 'export LSAN_OPTIONS="exitcode=0"\nfor _f in /usr/lib/gcc/*/*/libasan.so; do [ -f "$_f" ] && { LD_PRELOAD="$_f"; break; }; done\nexport LD_PRELOAD\nunset _f\n' >> /etc/conf.d/sshd

# rc_sys autodetect statt "docker": OpenRCs Installation schreibt beim Bauen
# im Behaelter rc_sys="docker" fest (Container-Erkennung). Im KVM-Gast wuerde
# das ewig alle Container-Keywords ziehen (kein devfs/udev). Leer = pro
# Umgebung neu erkennen (Gast: echt, Behaelter: weiter Container).
RUN export LSAN_OPTIONS="exitcode=0" \
 && sed -i 's/^rc_sys=.*/rc_sys=""/' /etc/rc.conf \
 && grep -q '^rc_sys=""$' /etc/rc.conf

# Udev-Dienste aus dem Boot nehmen (Stapel 2+, Testgast): Der Aufwand, den
# Daemon unter Aufsicht zu starten, haengt (supervise-daemon kehrt nie
# zurueck; von Hand gestartet laeuft er). Der Testgast braucht ihn nicht:
# keine Hotplug-Geraete, keine Module (alles builtin), Geraeteknoten kommen
# aus der Initramfs. Manuell weiter nutzbar. Falls spaetere Stapel ihn
# brauchen (elogind?), mit Shell-Zugang neu bewerten.
# MUSS VOR mksquashfs stehen (Gast startet aus dem Squashfs!).
RUN export LSAN_OPTIONS="exitcode=0" \
 && rc-update del udev sysinit || true \
 && rc-update del udev-trigger sysinit || true \
 && rc-update del udev-settle sysinit || true

ARG SSH_PUBKEY=""
RUN export LSAN_OPTIONS="exitcode=0" \
 && groupadd -f input \
 && useradd -m -G users,wheel,audio,video,input -s /bin/bash kiel \
 && if [ -n "${SSH_PUBKEY}" ]; then \
      mkdir -p /home/kiel/.ssh && chmod 700 /home/kiel/.ssh \
      && printf '%s\n' "${SSH_PUBKEY}" > /home/kiel/.ssh/authorized_keys \
      && chmod 600 /home/kiel/.ssh/authorized_keys \
      && chown -R kiel:kiel /home/kiel/.ssh; \
    fi \
 && if getent group tty >/dev/null; then usermod -aG tty kiel; fi \
 && for grp in libvirt kvm qemu plugdev pcap docker dialout; do \
      if getent group "${grp}" >/dev/null; then usermod -aG "${grp}" kiel; fi; \
    done && passwd -d kiel

# Serielle Konsole + minimale Dienste (kein X, kein elogind-GUI-Stack nötig,
# elogind/dbus bleiben für Kompatibilität zum Hardware-Pfad aktiv).
# Hinweis: udev/udev-trigger/udev-settle bleiben ABSICHTLICH aus sysinit
# (weiter oben per rc-update del entfernt). eudev ist noch nicht sanitisiert;
# sein Start haengt den Boot auf. Reaktivieren mit Batch-2.
RUN if ! grep -q '^s0:12345:respawn:/sbin/agetty 115200 ttyS0 vt100$' /etc/inittab; then \
      printf '%s\n' 's0:12345:respawn:/sbin/agetty 115200 ttyS0 vt100' >> /etc/inittab; \
    fi \
 && if ! grep -qx 'ttyS0' /etc/securetty; then \
      printf '%s\n' 'ttyS0' >> /etc/securetty; \
    fi \
 && rc-update add cgroups sysinit \
 && rc-update add modules boot \
 && rc-update add elogind boot \
 && rc-update add dbus default \
 && rc-update add sshd default
# SSH-Host-Schluessel beim BAU erzeugen (Run-39): sanitisiertes ssh-keygen -A
# stuerzt ab (SEGV, separater Hunt); Einzeltypen laufen. Stabile Schluessel
# ersparen die Erzeugung beim ersten Boot und stabilisieren known_hosts.
# LSAN-Policy: nur Build-Zeit (Exit-Leaks).
RUN export LSAN_OPTIONS="exitcode=0"; set -o pipefail \
 && for t in rsa ecdsa ed25519; do \
      if [ ! -f "/etc/ssh/ssh_host_${t}_key" ]; then \
        ssh-keygen -q -t "${t}" -f "/etc/ssh/ssh_host_${t}_key" -N ""; \
      fi; \
    done \
 && test "$(ls /etc/ssh/ssh_host_*_key 2>/dev/null | wc -l)" -eq 3

RUN rm -rf /var/tmp/portage/*

WORKDIR /
# Build-Args spaet (s. Hinweis oben): nur Tail-Layer werden pro Run neu.
ARG BUILD_GIT_COMMIT=unknown
ARG BUILD_REPOSITORY=unknown
ARG BUILD_DATE=unknown
RUN printf '%s\n' \
  'Minimal OpenRC QEMU test image (openrc2, no ASan)' \
  '=================================================' \
  "Git commit: ${BUILD_GIT_COMMIT}" \
  "Repository: ${BUILD_REPOSITORY}" \
  'Source Dockerfile: openrc2/Dockerfile' \
  "Build date (UTC): ${BUILD_DATE}" \
  > /IMAGE-BUILD-INFO.txt

# Hinweis: KEIN custom mount-overlayfs-Hook hier. ../openrc/config/mount-overlayfs.sh
# ist Hardware-spezifisch (LUKS /dev/mapper/enc + persistente Upper/Work-Dirs) und
# ersetzt dracuts Stock-Hook mount/01-mount-overlayfs.sh — damit würde /sysroot nie
# gemountet ("Can't mount root filesystem"). QEMU nutzt Stock-dracut mit tmpfs-Overlay.
# LSAN-Policy NUR fuer diesen Bauschritt (Batch-1-Run-16): dracut ruft
# Sanitizer-Werkzeuge (sort & Co., 8-B-Exit-Leaks, belegt) und bricht bei deren
# Exit-Code ab. Wirkt nur zur Build-Zeit, nicht im Image.
RUN LSAN_OPTIONS="exitcode=0" dracut \
      -m "kernel-modules base dmsquash-live dm udev-rules lvm overlayfs" \
  --filesystems "squashfs erofs vfat ext4 overlay btrfs" \
  --add-drivers "virtio virtio_pci virtio_blk virtio_net virtio_console" \
      --kver=${KVER_RELEASE} \
      --install "stat blockdev" \
      --include /etc/portage/openrc2-dracut-00-lsan.sh \
        /var/lib/dracut/hooks/pre-udev/00-lsan.sh \
      --include /etc/portage/openrc2-dracut-99-unset-preload.sh \
        /var/lib/dracut/hooks/pre-pivot/99-unset-preload.sh \
      --force \
      /boot/initramfs_squash_sda1-x86_64.img

# Paketbericht (vereinfacht: nur Groessen, keine Build-Zeiten -> kein genlop nötig).
# gentoolkit (qlist/qsize) steht in config/world.
# LSAN-Policy (Batch-1): sed/sort sind sanitisiert (Exit-Leaks), set -e wuerde
# den Loop beim ersten Leak abbrechen. Nur Build-Zeit, nicht im Image.
RUN export LSAN_OPTIONS="exitcode=0"; set -e \
 && printf 'package\tsize_mib\n' > /packages.tsv \
 && for pkg in $(qlist -I); do \
      size_line=$(qsize -m "$pkg" | head -n 1 || true); \
      size_mib=$(printf '%s\n' "$size_line" | sed -nE 's/.*,[[:space:]]*([0-9.]+)[[:space:]]+MiB$/\1/p'); \
      if [ -z "$size_mib" ]; then size_mib=0; fi; \
      printf '%s\t%s\n' "$pkg" "$size_mib" >> /packages.tsv; \
    done \
 && sort -t "$(printf '\t')" -k2,2nr /packages.tsv > /packages.txt

# Minimalitäts-Garantie: kein LLVM, NVIDIA, Mesa, Emacs oder Firefox im Image.
# (llvm-core/llvm allein ~395 MiB; wird ohne mesa/nvidia/rust nicht gezogen.)
# LSAN-Policy: grep ist sanitisiert; falsche Exit-Codes wuerden den Check
# blind machen (Fund mit Leak-Exit 1 gaelte als "nicht gefunden").
RUN export LSAN_OPTIONS="exitcode=0"; set -e \
 && for pkg in llvm-core/llvm x11-drivers/nvidia-drivers media-libs/mesa \
      app-editors/emacs www-client/firefox-bin net-im/signal-desktop-bin \
      x11-base/xorg-server sys-kernel/linux-firmware; do \
      if qlist -I | grep -qx "${pkg}"; then \
        echo "FEHLER: Schwergewicht ${pkg} im Minimal-Image!" >&2; exit 1; \
      fi; \
    done \
 && echo "Minimal-Check OK: kein LLVM/NVIDIA/Mesa/Emacs/Firefox/Firmware im Image."

# EINZIGES QEMU-Squashfs (kein NVIDIA-, kein E14-Image; minimale Firmware reicht
# für virtio, daher kein Firmware-Tausch nötig)
# LSAN-Policy: df (coreutils, sanitisiert) darf den Schritt nicht kippen.
RUN export LSAN_OPTIONS="exitcode=0"; df -h
# Bau-Artefakt entfernen: /.dockerenv gaukelt OpenRC einen Container vor
# (Keywords -docker/-containers) -> devfs/udev/trigger wuerden nie starten.
# Muss VOR mksquashfs weg (liegt sonst im Squashfs).
RUN rm -f /.dockerenv
# Systemweite ASan-Vorladung (Run-42): libasan fuer JEDEN Prozess zuerst.
# Mischt sanitisierte und unsanitisierte Programme (z.B. uptime -> libproc2
# -> sanitisiertes libelogind, sonst "does not come first", Gast-Beweis).
# Behaelter-belegt (uptime/python/ls/mountpoint RC=0). SPAET hier, damit der
# Neubau klein bleibt; Gast braucht sie im Squashfs -> VOR mksquashfs.
RUN export LSAN_OPTIONS="exitcode=0"; printf '%s\n' '/usr/lib/gcc/x86_64-pc-linux-gnu/16/libasan.so.8' \
  > /etc/ld.so.preload && cat /etc/ld.so.preload
# LSAN-Policy: mksquashfs laeuft jetzt vorbelastet (plain, Exit-Leaks).
RUN export LSAN_OPTIONS="exitcode=0"; mksquashfs / /gentoo.squashfs_qemu \
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
    usr/src \
    var/cache/binpkgs \
    var/cache/distfiles \
    var/cache/binhost \
    "gentoo*squashfs*" \
    "gentoo*ext4" \
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
    initramfs-with-squashfs.img \
    lost+found \
    .dockerenv

# Letzter Schritt: /.dockerenv erneut entfernen. Jeder RUN-Schritt davor legt
# sie neu an (Docker-Deamon), auch nach dem rm vor mksquashfs. Das Squashfs
# ist zusaetzlich ueber die -e-Liste geschuetzt; das hier haelt das Image sauber.
# LSAN-Policy: rm ist sanitisiert und laeuft jetzt vorbelastet.
RUN export LSAN_OPTIONS="exitcode=0"; rm -f /.dockerenv
