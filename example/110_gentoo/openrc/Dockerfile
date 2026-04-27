# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# Base images
FROM gentoo/portage:20260427 AS portage

# Stage3 base (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-20260427 AS base
# https://hub.docker.com/r/gentoo/portage/tags
# Copy full Portage tree (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Portage configuration
RUN eselect profile set default/linux/amd64/23.0/no-multilib
COPY config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords


# Repository tooling and package configuration
RUN emerge --ask=n app-eselect/eselect-repository dev-vcs/git
#RUN emerge --ask=n sys-devel/mold
COPY config/make.conf /etc/portage/make.conf
COPY config/package.use /etc/portage/package.use
COPY config/package.env /etc/portage/package.env
COPY config/env /etc/portage/env
# Keep the world set on PipeWire and let PipeWire provide PulseAudio compatibility.
COPY config/world /var/lib/portage/world
COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm-6.8
ENV KVER_PURE=6.18.18
ENV KVER_SOURCE=${KVER_PURE}-gentoo
ENV KVER_RELEASE=${KVER_SOURCE}-dist

RUN df -h
RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

RUN eselect kernel list
RUN eselect kernel set linux-${KVER_SOURCE}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config6.18.18 .config
RUN sed -i \
    -e 's/^CONFIG_LOCALVERSION="-gentoo-dist"/CONFIG_LOCALVERSION="-dist"/' \
    -e 's/^CONFIG_DEBUG_INFO=y/# CONFIG_DEBUG_INFO is not set/' \
    -e 's/^CONFIG_DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT=y/# CONFIG_DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT is not set/' \
    -e 's/^CONFIG_DEBUG_INFO_BTF=y/# CONFIG_DEBUG_INFO_BTF is not set/' \
    -e 's/^CONFIG_DEBUG_INFO_BTF_MODULES=y/# CONFIG_DEBUG_INFO_BTF_MODULES is not set/' \
    .config
# amd/sof/* and amd/sof-tplg/* firmware blobs

RUN make prepare

# compile the kernel
RUN make -j32
RUN make modules_install
RUN depmod -a ${KVER_RELEASE}

# I found that linux hangs after booting (just before you can log into the console). I also found that
# removing the modules helps.
# In order for the computer to boot the nvidia modules must be not present but they must
# be present for X11 to start. So we will move the modules to /nvidia and add a script /nvidia_start.sh
# to run depmod -a and modprobe. The user can execute this before starting X11

RUN make install

# Clean up the locales
RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update \
 && . /etc/profile
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN sed -i \
    -e '/^net-analyzer\/arping\>/d' \
    -e '/^net-misc\/speedtest-cli\>/d' \
    -e '/^sys-process\/progress\>/d' \
    -e '/^sys-process\/sysstat\>/d' \
    /var/lib/portage/world
RUN mkdir -p /etc/portage/package.use \
 && touch /etc/portage/package.use/zz-late-world \
 && grep -qxF 'app-text/xmlto text' /etc/portage/package.use/zz-late-world \
 || printf '\napp-text/xmlto text\n' >> /etc/portage/package.use/zz-late-world
RUN --mount=type=tmpfs,target=/var/tmp/portage,size=8g \
    PORTAGE_TMPDIR=/var/tmp/portage emerge -uDN @world
RUN gcc-config -l

# System package build in cached chunks:
# 1) Compute an ordered empty-tree world plan during the Docker build.
# 2) Split it into 10 files with roughly equal number of packages.
# 3) Emerge each chunk in its own layer to reduce rebuild cost after failures.

# RUN emerge -pqe --columns @world | awk '$2 ~ /\// {print $2}' > /tmp/world_packages.txt \
#  && total_lines=$(wc -l < /tmp/world_packages.txt) \
#  && lines_per_stage=$(( (total_lines + 9) / 10 )) \
#  && for i in $(seq 1 10); do \
#       start=$(( (i - 1) * lines_per_stage + 1 )); \
#       end=$(( i * lines_per_stage )); \
#       sed -n "${start},${end}p" /tmp/world_packages.txt > "/tmp/world_stage_${i}.txt"; \
#     done

# RUN if [ -s /tmp/world_stage_1.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_1.txt); fi
# RUN if [ -s /tmp/world_stage_2.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_2.txt); fi
# RUN if [ -s /tmp/world_stage_3.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_3.txt); fi
# RUN if [ -s /tmp/world_stage_4.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_4.txt); fi
# RUN if [ -s /tmp/world_stage_5.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_5.txt); fi
# RUN if [ -s /tmp/world_stage_6.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_6.txt); fi
# RUN if [ -s /tmp/world_stage_7.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_7.txt); fi
# RUN if [ -s /tmp/world_stage_8.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_8.txt); fi
# RUN if [ -s /tmp/world_stage_9.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_9.txt); fi
# RUN mkdir -p /etc/portage/package.accept_keywords \
#  && touch /etc/portage/package.accept_keywords/package.accept_keywords \
#  && grep -q '^>=net-misc/freerdp-3.17.2-r1 ~amd64$' /etc/portage/package.accept_keywords/package.accept_keywords \
#     || printf '>=net-misc/freerdp-3.17.2-r1 ~amd64\n' >> /etc/portage/package.accept_keywords/package.accept_keywords
# RUN if [ -s /tmp/world_stage_10.txt ]; then emerge --ask=n $(grep -E '.+/.+' /tmp/world_stage_10.txt); fi

# Temporarily install package analysis tools
RUN emerge --ask=n app-portage/genlop app-portage/gentoolkit \
 && qlist -Iv sys-devel/gcc \
 && echo "Generating package statistics..." \
 && for pkg in $(qlist -I); do \
      SIZE=$(qsize -m "$pkg" | awk '{print $5$6}'); \
      TIME=$(genlop -t "$pkg" | grep "merge time" | tail -n 1 | sed 's/.*merge time: //'); \
      echo "$pkg | Size: ${SIZE:-N/A} | Build Time: ${TIME:-N/A}"; \
    done > /packages.txt \
 && emerge -C app-portage/genlop app-portage/gentoolkit


RUN emerge --depclean

# Sudo policy
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel


# # Ryzen SMU kernel module
# WORKDIR /usr/src
# RUN git clone https://github.com/amkillam/ryzen_smu
# WORKDIR /usr/src/ryzen_smu
# # Path should be like /lib/modules/6.13.5-gentoo/
# # Replace $(shell uname -r) with ${KVER}-x86_64, otherwise the TARGET folder is not defined correctly
# RUN sed -i "s/\$(shell uname -r)/${KVER}/g" Makefile
# RUN make -j7
# RUN cp ryzen_smu.ko /lib/modules/${KVER}/kernel/
# RUN depmod -a ${KVER}
# RUN rm *.o

# Keep video modules out of /lib/modules during boot. They are activated later by /video.sh.
# I suspect the RT_PREEMPT kernel is causing issues with the nvidia video modules, i've also abserved instability on amd apu driver on the thinkpad e14
# RUN if [ -d /lib/modules/${KVER}/video ]; then \
#       mv /lib/modules/${KVER}/video /video${KVER}; \
#     fi
# COPY config/video.sh /video.sh
# RUN chmod 0755 /video.sh


# slstatus
WORKDIR /usr/src
RUN git clone https://git.suckless.org/slstatus
WORKDIR /usr/src/slstatus
COPY config/slstatus_config.h .
RUN make -j32
RUN make install
RUN make clean

RUN groupadd -f input \
 && useradd -m -G users,wheel,audio,video,input -s /bin/bash kiel \
 && for grp in libvirt kvm qemu; do \
      if getent group "${grp}" >/dev/null; then usermod -aG "${grp}" kiel; fi; \
    done

COPY config/xinitrc /home/kiel/.xinitrc
COPY config/activate /home/kiel/activate
COPY config/start2 /home/kiel/start2
COPY config/start-pipewire.sh /home/kiel/start-pipewire.sh
RUN chmod +x /home/kiel/activate /home/kiel/start2 /home/kiel/start-pipewire.sh \
 && chown kiel:kiel /home/kiel/activate /home/kiel/start2 /home/kiel/start-pipewire.sh
COPY config/user-runtime.initd /etc/init.d/user-runtime
COPY config/zz-openrc-user-session.sh /etc/profile.d/zz-openrc-user-session.sh
COPY config/user-dbus.initd /etc/user/init.d/dbus
COPY config/user-pipewire.initd /etc/user/init.d/pipewire
COPY config/user-pipewire-pulse.initd /etc/user/init.d/pipewire-pulse
COPY config/user-wireplumber.initd /etc/user/init.d/wireplumber

RUN emerge --depclean

# Network and services
COPY config/resolv.conf /etc/
COPY config/reverse-ssh-eu.initd /etc/init.d/reverse-ssh-eu
COPY config/reverse-ssh-us.initd /etc/init.d/reverse-ssh-us

RUN chmod +x /etc/init.d/reverse-ssh-* \
 && rc-update add sshd default \
 && rc-update add reverse-ssh-eu default \
 && rc-update add reverse-ssh-us default
# Switch to user 'kiel'
USER kiel
ENV HOME=/home/kiel

# Install Emacs packages from MELPA in batch mode
RUN emacs --batch \
    --eval "(require 'package)" \
    --eval "(add-to-list 'package-archives '(\"melpa\" . \"https://melpa.org/packages/\") t)" \
    --eval "(package-initialize)" \
    --eval "(package-refresh-contents)" \
    --eval "(dolist (pkg '(magit paredit slime)) (package-install pkg))"

# Switch back to root user for the remaining steps
USER root


RUN mkdir -p /usr/local/share/openrc-host-config
COPY config/activate /usr/local/share/openrc-host-config/activate
COPY config/start2 /usr/local/share/openrc-host-config/start2
COPY config/reverse-ssh-eu.initd /usr/local/share/openrc-host-config/reverse-ssh-eu.initd
COPY config/reverse-ssh-us.initd /usr/local/share/openrc-host-config/reverse-ssh-us.initd


# For chinese font in chrome
RUN fc-cache -fv

RUN printf '%s\n' \
      'modules="amdgpu mt7921e"' \
      > /etc/conf.d/modules \
 && printf '%s\n' \
      'keymap="colemak"' \
      > /etc/conf.d/keymaps \
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
 && install -d -m 0755 /run/user \
 && install -d -o kiel -g kiel -m 0700 /run/user/1000 \
 && test -e /etc/init.d/user \
 && ln -sf /etc/init.d/user /etc/init.d/user.kiel \
 && install -d -o kiel -g kiel -m 0755 /home/kiel/.config/rc/runlevels/default \
 && ln -sf /etc/user/init.d/dbus /home/kiel/.config/rc/runlevels/default/dbus \
#  && ln -sf /etc/user/init.d/pipewire /home/kiel/.config/rc/runlevels/default/pipewire \
#  && ln -sf /etc/user/init.d/pipewire-pulse /home/kiel/.config/rc/runlevels/default/pipewire-pulse \
#  && ln -sf /etc/user/init.d/wireplumber /home/kiel/.config/rc/runlevels/default/wireplumber \
 && rc-update add udev sysinit \
 && rc-update add udev-trigger sysinit \
 && rc-update add udev-settle sysinit \
 && rc-update add modules boot \
 && rc-update add keymaps boot \
 && rc-update add user-runtime default \
 && rc-update add user.kiel default \
 && rc-update add dbus default \
 && rc-update add iwd default \
 && chown -R kiel:kiel /home/kiel/.config

RUN rm -rf /var/tmp/portage/* # reduce docker layer size

RUN emerge -C dev-lang/rust-bin virtual/rust dev-lang/go dev-lang/go-bootstrap dev-util/cargo-c || true

WORKDIR /
RUN emerge --ask=n sys-fs/e2fsprogs || true
RUN mksquashfs \
/ \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-noappend \
-not-reproducible \
-Xcompression-level 19 \
-progress \
-one-file-system-x \
-p "/dev d 755 0 0" \
-p "/proc d 555 0 0" \
-p "/sys d 555 0 0" \
-b 256K -noI -noX \
-mem 10G \
-wildcards \
-e \
bin \
sbin \
lib \
lib64 \
usr/bin \
usr/sbin \
usr/lib \
usr/lib64 \
opt \
usr/src \
var/cache/binpkgs \
var/cache/distfiles \
"gentoo*squashfs" \
"gentoo*ext4" \
"usr/lib64/libQt*.a" \
usr/share/genkernel/distfiles \
usr/src/linux \
usr/share/sgml \
var/cache/eix/previous.eix \
boot \
persistent \
home/kiel/.cache/mozilla \
home/kiel/.cache/google-chrome \
home/kiel/.cache/mesa_shader_cache \
home/kiel/.cache/fontconfig \
home/kiel/Downloads \
home/kiel/.config \
home/kiel/.mozilla \
home/kiel/stage \
var/log/journal \
var/cache/genkernel \
var/tmp/portage \
mnt/ \
mnt2/ \
mnt4/ \
mnt5/ \
var/tmp \
initramfs-with-squashfs.img \
lost+found \
var/log \
usr/share/gtk-doc \
usr/share/doc \
usr/share/locale

# Ext4 image creation with e2fsprogs (sparse initial file)
# Contains ONLY binaries and libraries for DAX Execute-In-Place
RUN set -e \
 && echo "Building gentoo.ext4 with DAX capabilities..." \
 && mkdir -p /tmp/ext4_root/usr /tmp/ext4_root/opt \
 && rsync -a /bin /sbin /lib /lib64 /tmp/ext4_root/ \
 && rsync -a /usr/bin /usr/sbin /usr/lib /usr/lib64 /tmp/ext4_root/usr/ \
 && rsync -a /opt/ /tmp/ext4_root/opt/ \
 && truncate -s 10G /gentoo.ext4 \
 && mke2fs -t ext4 -b 4096 -O ^has_journal -d /tmp/ext4_root /gentoo.ext4 \
 && rm -rf /tmp/ext4_root

# Dracut Module documentation https://dracut-ng.github.io/dracut-ng/modules/core.html

COPY config/mount-overlayfs.sh /usr/lib/dracut/modules.d/70overlayfs
RUN chmod +x /usr/lib/dracut/modules.d/70overlayfs/*.sh
RUN dracut \
      -m "kernel-modules base dmsquash-live dm udev-rules crypt lvm overlayfs" \
      --filesystems "squashfs vfat ext4 overlay btrfs" \
      --add-drivers "libnvdimm nd_pmem nd_btt nd_blk dax device_dax dax_pmem dax_pmem_core nd_e820" \
      --kver=${KVER_RELEASE} \
      --install "stat blockdev" \
      --force \
      /boot/initramfs_squash_sda1-x86_64.img

# Ensure SPARK proof tooling can find the Gentoo-installed Why3 helper.
#RUN ln -sf /usr/bin/gnatwhy3 /usr/libexec/spark/bin/gnatwhy3 \
# && if [ -x /usr/bin/why3 ]; then ln -sf /usr/bin/why3 /usr/libexec/spark/bin/why3; fi

# The top-level OCaml package is not needed for runtime proof use in the image:
# gnatprove/why3/alt-ergo still work after removing it, while build-time OCaml
# was required earlier to install the prover stack.
#RUN emerge -C dev-lang/ocaml || true

# Regenerate package reports late so they match the final image contents.
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
 && emerge -C app-portage/genlop || true

# Create NVIDIA/full-runtime squashfs image.
# Stage the exact firmware paths requested by the installed nvidia.ko so the
# exported HP Z6 image contains the files at their runtime paths instead of as
# leftover build scratch state under /tmp.
RUN set -e \
 && echo "Preparing NVIDIA squashfs" \
 && test -e /lib/modules/${KVER_RELEASE}/video/nvidia.ko \
 && rm -rf /tmp/fw_original /tmp/fw_nv_root \
 && mkdir -p /tmp/fw_original /tmp/fw_nv_root/usr/lib/firmware \
 && cp -a /usr/lib/firmware/. /tmp/fw_original/ \
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
 && mksquashfs / /gentoo.squashfs_nv \
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
     "gentoo*squashfs*" \
     "gentoo*ext4" \
     "usr/lib64/libQt*.a" \
     usr/share/genkernel/distfiles \
     usr/src/linux \
     usr/share/sgml \
     var/cache/eix/previous.eix \
     boot \
     persistent \
     home/*/.cache \
     tmp/fw_original \
     tmp/fw_nv_root \
     var/log/journal \
    var/cache/genkernel \
    var/tmp \
    initramfs-with-squashfs.img \
    lost+found \
 && rm -rf /usr/lib/firmware/* \
 && cp -a /tmp/fw_original/. /usr/lib/firmware/ \
 && rm -rf /tmp/fw_original /tmp/fw_nv_root

# Create ThinkPad E14 squashfs image (keep Firefox, trim NVIDIA userspace,
# keep only firmware required on the ThinkPad E14, and preserve the SPARK prover stack)
RUN set -e \
 && echo "Preparing ThinkPad E14 squashfs (remove NVIDIA)" \
 # gather firmware for E14 (from dmesg: regulatory, mediatek, amdgpu yellow_carp)
 && mkdir -p /tmp/tmpfw/amdgpu /tmp/tmpfw/mediatek /tmp/tmpfw/amd /tmp/tmpfw/rtl_bt || true \
 && cp -a /usr/lib/firmware/regulatory.db* /tmp/tmpfw/ 2>/dev/null || true \
 && cp -a /usr/lib/firmware/mediatek /tmp/tmpfw/ 2>/dev/null || true \
 && cp -a /usr/lib/firmware/amdgpu/yellow_carp* /tmp/tmpfw/amdgpu/ 2>/dev/null || true \
 && cp -a /usr/lib/firmware/amdgpu/rembrandt* /tmp/tmpfw/amdgpu/ 2>/dev/null || true \
 && cp -a /usr/lib/firmware/amd /tmp/tmpfw/ 2>/dev/null || true \
 && rm -rf /usr/lib/firmware/* \
 && mv /tmp/tmpfw/* /usr/lib/firmware/ 2>/dev/null || true \
 \
 # Build ThinkPad E14 image: keep Firefox and the SPARK prover stack, strip NVIDIA userspace
 && echo "Stripping NVIDIA/CUDA userspace for E14 squashfs" \
 && rm -rf /opt/cuda /usr/lib64/lib{cuda,nvidia,nv}* /usr/lib64/libnv* 2>/dev/null || true \
 && rm -rf /usr/bin/nvidia* \
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
      /usr/lib64/vdpau/libvdpau_nvidia.so* 2>/dev/null || true \
 && rm -rf /nvidia${KVER_RELEASE} 2>/dev/null || true \
 && mksquashfs / /gentoo.squashfs_e14 \
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
     "gentoo*squashfs*" \
     "gentoo*ext4" \
     "usr/lib64/libQt*.a" \
     usr/share/genkernel/distfiles \
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
   || true
