# syntax=docker/dockerfile:1
# Minimal-OpenRC-Testimage (openrc2): Konsolen-System für QEMU, ohne ASan.
# Vollsystem mit Hardware-Support: siehe ../openrc/Dockerfile (bleibt unverändert).

FROM gentoo/portage:20261021 AS portage

FROM gentoo/stage3:nomultilib-20261019 AS base
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

ARG BUILD_GIT_COMMIT=unknown
ARG BUILD_REPOSITORY=unknown
ARG BUILD_DATE=unknown

RUN eselect profile set default/linux/amd64/23.0/no-multilib

# Minimal-Konfiguration (kein X, keine HW-Treiber)
COPY config/make.conf /etc/portage/make.conf
COPY config/package.use /etc/portage/package.use
COPY config/package.env /etc/portage/package.env
COPY config/env /etc/portage/env
COPY config/world /var/lib/portage/world
# Leak-Policy (s. config/asan-suppressions + asan-worklog.md): GNU-CLI-Tools lecken
# bauartbedingt beim Beenden (Hash-Tabellen, Einmal-Allocs) — das darf weder Builds
# noch Boot brechen. Daher: bekannte Stellen via suppressions-Datei still,
# alle UEBRIGEN Leak-Meldungen bleiben sichtbar, zaehlen aber exit 0
# (Triage statt Abbruch). Echte Speicherfehler (ASan) + UBSan bleiben lethal.
COPY config/asan-suppressions /etc/asan-suppressions
ENV LSAN_OPTIONS="suppressions=/etc/asan-suppressions/lsan.supp:exitcode=0"
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
# explizit aus Quellen neu bauen.
COPY config/patches /etc/portage/patches
COPY config/sanitized-packages.txt /tmp/sanitized-packages.txt
RUN if [ -s /tmp/sanitized-packages.txt ]; then \
      FEATURES="-getbinpkg -binpkg-request-signature" \
      emerge --oneshot $(grep -vE '^\s*(#|$)' /tmp/sanitized-packages.txt); \
    fi

# Sudo + passwortloser Konsolen-User für QEMU-Tests
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel

ARG SSH_PUBKEY=""
RUN groupadd -f input \
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
# elogind/dbus bleiben für Kompatibilität zum Hardware-Pfad aktiv)
RUN if ! grep -q '^s0:12345:respawn:/sbin/agetty 115200 ttyS0 vt100$' /etc/inittab; then \
      printf '%s\n' 's0:12345:respawn:/sbin/agetty 115200 ttyS0 vt100' >> /etc/inittab; \
    fi \
 && if ! grep -qx 'ttyS0' /etc/securetty; then \
      printf '%s\n' 'ttyS0' >> /etc/securetty; \
    fi \
 && rc-update add udev sysinit \
 && rc-update add udev-trigger sysinit \
 && rc-update add udev-settle sysinit \
 && rc-update add cgroups sysinit \
 && rc-update add modules boot \
 && rc-update add elogind boot \
 && rc-update add dbus default \
 && rc-update add sshd default

RUN rm -rf /var/tmp/portage/*

WORKDIR /
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
RUN dracut \
      -m "kernel-modules base dmsquash-live dm udev-rules crypt lvm overlayfs" \
  --filesystems "squashfs erofs vfat ext4 overlay btrfs" \
  --add-drivers "virtio virtio_pci virtio_blk virtio_net virtio_console" \
      --kver=${KVER_RELEASE} \
      --install "stat blockdev" \
      --force \
      /boot/initramfs_squash_sda1-x86_64.img

# Paketbericht (vereinfacht: nur Groessen, keine Build-Zeiten -> kein genlop nötig).
# gentoolkit (qlist/qsize) steht in config/world.
RUN set -e \
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
RUN set -e \
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
RUN df -h
RUN mksquashfs / /gentoo.squashfs_qemu \
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
    lost+found
