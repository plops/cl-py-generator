# syntax=docker/dockerfile:1

ARG PORTAGE_TAG=20261115
ARG STAGE3_TAG=nomultilib-20261109

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

# Build binary packages for all emerged packages so target stage installs without BDEPENDs.
# make.conf requests signed binpkgs (binpkg-request-signature), but the builder
# has no signing key, so freshly built local binpkgs would fail verification
# ("binpkg appears unsigned") and abort the emerge. Disable only the signature
# *request* here; the final make.conf in the image keeps it for later operation.
RUN printf '\nFEATURES="${FEATURES} buildpkg -binpkg-request-signature"\n' >> /etc/portage/make.conf

ENV KVER_PURE=6.18.63
ENV KVER_SOURCE=${KVER_PURE}-gentoo
ENV KVER_RELEASE=${KVER_SOURCE}-dist

RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN --mount=type=cache,target=/var/cache/distfiles,sharing=locked \
    emerge -1 sys-apps/portage sys-kernel/installkernel

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

# xkeyboard-config's preinst unconditionally renames xkb.workaround->xkb, so a
# post-merge quickpkg snapshot (xkb/ already in final layout) cannot be
# reinstalled (preinst dies with "cannot stat xkb.workaround"). Rebuild it
# from source FIRST (directly into the default PKGDIR; without --usepkg no
# binary short-circuits the build) to get a proper build-time binpkg, so the
# export below picks it up as genuinely local. The =atom forces a reinstall
# of the exact version.
RUN rm -f /var/cache/binpkgs/x11-misc/xkeyboard-config/*.gpkg.tar \
 && FEATURES="-getbinpkg" emerge -1 --usepkg=n --buildpkg =$(qlist -Iv x11-misc/xkeyboard-config) \
 && rm -f /var/cache/binpkgs/Packages /var/cache/binpkgs/Packages.gz \
 && emaint --fix binhost \
 && test -s /var/cache/binpkgs/Packages

# Export genuinely local binpkgs + upstream manifest BEFORE quickpkg/genlop
# touch /var/cache/binpkgs: /out_binpkgs holds only truly compiled @world/
# BDEPEND binpkgs, /tmp/target-binpkgs (below) feeds the target stage.
COPY config/extract_local_binpkgs.py /tmp/extract_local_binpkgs.py
RUN python3 /tmp/extract_local_binpkgs.py \
      --out-binpkgs /out_binpkgs/binpkgs \
      --out-upstream-tsv /out_binpkgs/upstream_binpkgs.tsv \
      --out-upstream-txt /out_binpkgs/upstream_binpkgs.txt \
   && rm -f /out_binpkgs/binpkgs/Packages /out_binpkgs/binpkgs/Packages.gz \
   && PKGDIR=/out_binpkgs/binpkgs emaint --fix binhost \
   && test -s /out_binpkgs/binpkgs/Packages \
   && test -s /out_binpkgs/upstream_binpkgs.tsv

# Snapshot installed packages lacking a proper build-time binpkg as local
# binpkgs for the TARGET stage only (/tmp/target-binpkgs stays out of the
# export above). The builder installs most dependencies as upstream binaries,
# which are NOT stored in /var/cache/binpkgs, so the target stage would fail
# with "no binary packages to satisfy ...". Virtuals provide no files and are
# skipped. Only missing ones are snapshotted: where a build-time binpkg
# already exists, it MUST win — a post-merge quickpkg snapshot can be
# uninstallable (e.g. nvidia-drivers' preinst unconditionally removes
# supported-gpus.json, which no longer exists post-merge).
# The index is regenerated from scratch (rm first): incremental
# `emaint --fix binhost` silently skipped newly added binpkgs (acct-user/
# messagebus, ntp, polkitd were missing from the index although present).
RUN cp -al /var/cache/binpkgs /tmp/target-binpkgs \
 && missing=$(for atom in $(qlist -I | grep -v '^virtual/'); do \
        [ -n "$(ls -A /tmp/target-binpkgs/${atom} 2>/dev/null)" ] || printf '%s\n' "${atom}"; \
      done) \
 && if [ -n "${missing}" ]; then \
      printf '%s\n' "${missing}" | PKGDIR=/tmp/target-binpkgs xargs quickpkg --include-config=y; \
    fi \
 && rm -f /tmp/target-binpkgs/Packages /tmp/target-binpkgs/Packages.gz \
 && PKGDIR=/tmp/target-binpkgs emaint --fix binhost \
 && test -s /tmp/target-binpkgs/Packages

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
 && FEATURES="-buildpkg" emerge --ask=n --oneshot app-portage/genlop \
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

ENV KVER_PURE=6.18.63
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

# Copy prebuilt kernel + modules before emerging @world so nvidia-drivers postinst sees the kernel tree.
# sys-kernel/installkernel installs only versioned names, but the smoke test and
# the export stage expect the generic /boot/vmlinuz, so copy it here in the
# target (builder stays untouched, keeping its layer cache valid).
COPY --from=builder /boot /boot
COPY --from=builder /lib/modules/${KVER_RELEASE} /lib/modules/${KVER_RELEASE}
COPY --from=builder /usr/local/bin/slstatus /usr/local/bin/slstatus
RUN cp /boot/vmlinuz-${KVER_RELEASE} /boot/vmlinuz && test -s /boot/vmlinuz

# Install strictly from local builder binpkgs without build-only dependencies (e.g. rust, go, genlop).
# NOTE: --usepkg (not --usepkgonly): virtuals from @system (virtual/ssh, ...)
# cannot be satisfied by any binary, so --usepkgonly fails. With --usepkg the
# virtuals resolve via ebuilds (they install no files) while every real package
# still comes from the local binpkgs (verified: pretend shows only virtual/*
# as ebuilds, zero source builds).
RUN --mount=type=bind,from=builder,source=/tmp/target-binpkgs,target=/var/cache/binpkgs \
    FEATURES="-binpkg-request-signature -getbinpkg" \
    emerge -1 --usepkgonly sys-apps/portage \
 && FEATURES="-binpkg-request-signature -getbinpkg" \
    emerge -uDN --usepkg --with-bdeps=n @world \
 && depmod -a ${KVER_RELEASE} \
 && rm -rf /var/tmp/portage/*

# Offline/Hybrid binpkg helper (host provides BINPKG_DIR at runtime, no automount)
COPY config/emerge-binpkg /usr/local/sbin/emerge-binpkg
RUN chmod 0755 /usr/local/sbin/emerge-binpkg

# Upstream binhost signing keys for the hybrid-online mode of emerge-binpkg.
# Same keyring the builder verified upstream binpkgs against; permissions
# must stay world-readable (portage verifies as unprivileged user), so strip
# sockets/locks and enforce modes explicitly (builder may leave 0600/0700).
COPY --from=builder /etc/portage/gnupg /etc/portage/gnupg
RUN rm -f /etc/portage/gnupg/S.* /etc/portage/gnupg/.#* \
 && find /etc/portage/gnupg -type d -exec chmod 0755 {} + \
 && find /etc/portage/gnupg -type f -exec chmod 0644 {} +

# Firefox comfort symlink (firefox-bin normally provides /usr/bin/firefox itself)
RUN if [ ! -e /usr/bin/firefox ] && command -v firefox-bin >/dev/null 2>&1; then \
      ln -sf "$(command -v firefox-bin)" /usr/local/bin/firefox; \
    fi

# Boot-time guard for a stale /etc/hosts in a persistent overlay upperdir:
# ensures 127.0.0.1/::1 map the live hostname (runs via OpenRC local service).
RUN install -d -m 0755 /etc/local.d \
 && printf '%s\n' \
      '#!/bin/sh' \
      'h="$(hostname 2>/dev/null || true)"' \
      '[ -n "${h}" ] && [ "${h}" != "(none)" ] && [ "${h}" != "localhost" ] || exit 0' \
      'if ! grep -Eq "^127\.0\.0\.1[[:space:]].* ${h}( |$)" /etc/hosts 2>/dev/null; then' \
      '  printf "127.0.0.1\t%s localhost\n::1\t\t%s localhost\n" "${h}" "${h}" > /etc/hosts' \
      'fi' \
      > /etc/local.d/10-hosts.start \
 && chmod 0755 /etc/local.d/10-hosts.start

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
 && rc-update add hostname boot \
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
      --add-drivers "libnvdimm nd_pmem nd_btt dax device_dax dax_pmem dax_pmem_core nd_e820" \
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
 && test -e /etc/runlevels/boot/hostname \
 && command -v emerge-binpkg >/dev/null \
 && emerge-binpkg --help >/dev/null \
 && (command -v firefox >/dev/null || command -v firefox-bin >/dev/null) \
 && qlist -I | grep -qx 'sys-kernel/installkernel' \
 && ! command -v rustc >/dev/null \
 && ! command -v cargo >/dev/null \
 && ! command -v go >/dev/null \
 && ! qlist -I | grep -qE '^(dev-lang/rust|dev-lang/go|app-portage/gentoolkit|sys-boot/grub)' \
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
 && printf 'hostname="hpz6"\n' > /etc/conf.d/hostname \
 && grep -qx 'hostname="hpz6"' /etc/conf.d/hostname \
 && find /usr/lib /usr/lib64 -type f -name '*.a' \
      ! -name '*_nonshared.a' \
      ! -path '/usr/lib/gcc/*' \
      ! -path '/usr/lib64/gcc/*' \
      -delete \
 && printf '#include <iostream>\nint main(){std::cout << "cpp-ok\\n"; return 0;}\n' \
    | g++ -x c++ - -o /tmp/cpp_smoke \
 && /tmp/cpp_smoke | grep -qx 'cpp-ok' \
 && rm -f /tmp/cpp_smoke \
 && printf '127.0.0.1\thpz6 localhost\n::1\t\thpz6 localhost\n' > /tmp/hosts.variant \
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
   -p "/etc/hosts f 0644 0 0 cat /tmp/hosts.variant" \
   -noI -noX \
   -wildcards \
   -e \
     etc/hosts \
     tmp/hosts.variant \
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
 && rm -f /tmp/hosts.variant \
 && unsquashfs -cat /out/gentoo.squashfs_nv etc/hosts > /tmp/hosts.check \
 && printf '127.0.0.1\thpz6 localhost\n::1\t\thpz6 localhost\n' | cmp -s - /tmp/hosts.check \
 && rm -f /tmp/hosts.check \
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
 && printf 'hostname="e14"\n' > /etc/conf.d/hostname \
 && grep -qx 'hostname="e14"' /etc/conf.d/hostname \
 && find /usr/lib /usr/lib64 -type f -name '*.a' \
      ! -name '*_nonshared.a' \
      ! -path '/usr/lib/gcc/*' \
      ! -path '/usr/lib64/gcc/*' \
      -delete \
 && printf '#include <iostream>\nint main(){std::cout << "cpp-ok\\n"; return 0;}\n' \
    | g++ -x c++ - -o /tmp/cpp_smoke \
 && /tmp/cpp_smoke | grep -qx 'cpp-ok' \
 && rm -f /tmp/cpp_smoke \
 && test -e /etc/runlevels/boot/hostname \
 && printf '127.0.0.1\te14 localhost\n::1\t\te14 localhost\n' > /tmp/hosts.variant \
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
   -p "/etc/hosts f 0644 0 0 cat /tmp/hosts.variant" \
   -noI -noX \
   -wildcards \
   -e \
     etc/hosts \
     tmp/hosts.variant \
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
 && rm -f /tmp/hosts.variant \
 && unsquashfs -cat /out/gentoo.squashfs_e14 etc/hosts > /tmp/hosts.check \
 && printf '127.0.0.1\te14 localhost\n::1\t\te14 localhost\n' | cmp -s - /tmp/hosts.check \
 && rm -f /tmp/hosts.check \
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

# ==============================================================================
# Stage 4: Export locally built binpkgs + upstream manifest (cache-preserving)
# ==============================================================================
# The builder produced /out_binpkgs before quickpkg/genlop touched the cache;
# here we only re-verify and refresh the index (no rebuild, no re-download).
FROM builder AS export-binpkgs
RUN test -s /out_binpkgs/binpkgs/Packages \
 && test -s /out_binpkgs/upstream_binpkgs.tsv \
 && test -s /out_binpkgs/upstream_binpkgs.txt \
 && PKGDIR=/out_binpkgs/binpkgs emaint --fix binhost \
 && test -s /out_binpkgs/binpkgs/Packages

FROM scratch AS export-binpkgs-out
COPY --from=export-binpkgs /out_binpkgs/ /
