# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# Base images
FROM gentoo/portage:20260320 AS portage

# Stage3 base (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-20260316 AS base
# https://hub.docker.com/r/gentoo/portage/tags
# Copy full Portage tree (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Portage configuration
RUN eselect profile set default/linux/amd64/23.0/no-multilib
COPY config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
COPY config/package.mask /etc/portage/package.mask/package.mask

# Repository tooling and package configuration
RUN emerge --ask=n app-eselect/eselect-repository dev-vcs/git
COPY config/make.conf /etc/portage/make.conf
COPY config/package.mask /etc/portage/package.mask/package.mask
COPY config/package.license /etc/portage/package.license
ENV KVER_PURE=6.12.74
ENV KVER=${KVER_PURE}-gentoo

RUN df -h
RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

RUN eselect kernel list
RUN eselect kernel set linux-${KVER}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config6.12.41 .config
# amd/sof/* and amd/sof-tplg/* firmware blobs
RUN <<EOF
set -e
CONFIG_OPTS=(
  # =========================================================================
  # --- Hardware & Kernel Profile (HP Z6 / AMD Ryzen Threadripper PRO) ---
  # =========================================================================
  # To make this Dockerfile generic, comment out the "Specific Options"
  # and uncomment the "Generic Options" below.

  # --- Specific Options (Optimized for Threadripper / Workstation) ---
  -d PREEMPT_RT
  -e PREEMPT
  -d PREEMPT_VOLUNTARY
  -e GENERIC_CPU
  -d MK8
  -m KVM_AMD

  # --- DAX / Ext4 Options ---
  -e FS_DAX
  -e DAX
  -e EXT4_FS
  -m EXT4_FS_POSIX_ACL
  -m EXT4_FS_SECURITY
  -m BLK_DEV_PMEM
  -e ZONE_DEVICE
  -m LIBNVDIMM
  -m BTT
  -m NVDIMM_PFN
  -m NVDIMM_BLK
  -m DEV_DAX
  -m DEV_DAX_PMEM

  # --- Generic Options (Commented out by default) ---
  # -e PREEMPT_RT
  # -e PREEMPT
  # -d PREEMPT_VOLUNTARY
  # -e GENERIC_CPU
  # -d MK8
  # -m KVM_INTEL
  # -m KVM_AMD
  # =========================================================================

  # --- System / Core (Docker, Namespaces, Storage) ---
  -e CUSE
  -e EFI_PARTITION
  -e SQUASHFS
  -e SQUASHFS_4K_DEVBLK_SIZE
  -e CGROUPS
  -e NAMESPACES
  -e OVERLAY_FS
  -e TUN
  -m PPDEV

  # --- Boot / Crypto ---
  -e BLK_DEV_INITRD
  -m DM_CRYPT
  -m CRYPTO_AES_NI_INTEL
  -m CRYPTO_XTS
  -e MD

  # --- Virtualization / QEMU Base ---
  -m KVM
  -e VETH
  -m MACVTAP
  -m VHOST_NET
  -m VIRTIO_BALLOON
  -m VIRTIO_BLK
  -m VIRTIO_NET

  # libvirt default NAT (virbr0) via iptables needs netfilter reject/nat/conntrack modules
  -m NETFILTER_XTABLES
  -m NF_CONNTRACK
  -m NF_NAT
  -m IP_NF_IPTABLES
  -m IP_NF_FILTER
  -m IP_NF_NAT
  -m IP_NF_TARGET_MASQUERADE
  -m NF_REJECT_IPV4
  -m IP_NF_TARGET_REJECT
  -m NF_REJECT_IPV6
  -m IP6_NF_TARGET_REJECT
  -m BRIDGE
  -m BR_NETFILTER

  # --- Bluetooth ---
  -m BT_BCM
  -m BT_BNEP
  -m BT_HCIBTUSB
  -m BT_INTEL
  -m BT_INTEL_PCIE
  -m BT_MTK
  -m BT_RFCOMM
  -e BT_RFCOMM_TTY
  -m BT_RTL

  # --- Network (Ethernet, WiFi, WWAN) ---
  -m MT7921E
  -m QRTR
  -m R8169
  -m RTW88_8821C
  -m RTW88_8821CU
  -m RTW89_8852B
  -m RTW89_8852BE
  -m RTW89_8852B_COMMON
  -m USB_IPHETH
  -m USB_NET_CDCETHER
  -m USB_NET_CDC_NCM
  -m USB_NET_QMI_WWAN
  -m USB_USBNET

  # --- HID / Input Devices ---
  -m APPLE_MFI_FASTCHARGE
  -m HID_CHERRY
  -m HID_GENERIC
  -m HID_LENOVO
  -m HID_LOGITECH
  -m HID_LOGITECH_HIDPP
  -m HID_MULTITOUCH
  -m HID_PLANTRONICS
  -d TOUCHSCREEN_SUR40

  # --- Laptop / Platform / WMI ---
  -m AMD_PMC
  -m ASUS_WMI
  -m EEEPC_WMI
  -m HP_WMI
  -m IDEAPAD_LAPTOP
  -m LENOVO_WMI_CAMERA
  -m THINKPAD_ACPI
  -m THINKPAD_LMI

  # --- Hardware Sensors / Power ---
  -m EDAC_DECODE_MCE
  -m EEPROM_EE1004
  -m INTEL_RAPL
  -m INTEL_RAPL_CORE
  -m SENSORS_K10TEMP
  -m SENSORS_SPD5118
  -e SENSORS_SPD5118_DETECT

  # --- Sound / Audio ---
  -m SND_HDA_CODEC_CONEXANT
  -m SND_HDA_CODEC_REALTEK
  -m SND_HDA_INTEL
  -m SND_SOC_AMD_ACP6x
  -m SND_SOC_AMD_ACP_COMMON
  -m SND_SOC_AMD_ACP_LEGACY_COMMON
  -m SND_SOC_AMD_ACP_PCI
  -m SND_SOC_AMD_PS
  -m SND_SOC_AMD_PS_MACH
  -m SND_SOC_DMIC
  -m SND_SOC_SDCA
  -m SND_SOC_SOF_AMD_ACP63
  -m SND_SOC_SOF_AMD_REMBRANDT
  -m SND_SOC_SOF_AMD_RENOIR
  -m SND_SOC_SOF_AMD_TOPLEVEL
  -m SND_SOC_SOF_AMD_VANGOGH
  -m SND_SOC_SOF_PCI
  -m SND_SOC_SOF_TOPLEVEL
  -m SOUNDWIRE
  -m SOUNDWIRE_AMD
  -m SOUNDWIRE_BUS
  -m SOUNDWIRE_GENERIC_ALLOCATION
  -d MFD_CS42L43_SDW

  # --- Media / Video ---
  -e MEDIA_USB_SUPPORT
  -m MEDIA_CAMERA_SUPPORT
  -m MEDIA_SUPPORT
  -m USB_VIDEO_CLASS
  -m VIDEO_DEV
  -m VIDEO_V4L2
  -d USB_GSPCA
  -d USB_PWC

  # --- Storage ---
  -m NVME_AUTH
  -m NVME_KEYRING

  # --- Crypto ---
  -m CRYPTO_CMAC
  -m CRYPTO_GHASH_CLMUL_NI_INTEL
  -m CRYPTO_SHA1_SSSE3
  -m CRYPTO_SHA256_SSSE3
  -m CRYPTO_SHA512_SSSE3
  -m CRYPTO_USER_API_HASH
  -m CRYPTO_USER_API_SKCIPHER

  # --- USB / Thunderbolt ---
  -m THUNDERBOLT
  -m USB4
  -d USB4_NET
  -m USB_SERIAL_CH341
  -m USB_SERIAL_FTDI_SIO
)

./scripts/config --file .config "${CONFIG_OPTS[@]}"
make olddefconfig # accept default for new options to avoid interactive prompts
EOF


RUN make prepare

# compile the kernel
RUN make -j32
RUN make modules_install

# I found that linux hangs after booting (just before you can log into the console). I also found that
# removing the modules helps.
# In order for the computer to boot the nvidia modules must be not present but they must
# be present for X11 to start. So we will move the modules to /nvidia and add a script /nvidia_start.sh
# to run depmod -a and modprobe. The user can execute this before starting X11

RUN make install

# Delaying the copy of volatile configs (package.use, world, etc) until after the kernel build
# enables Docker to cache the long kernel compilation even if world/use flags change!
COPY config/package.use /etc/portage/package.use
COPY config/package.env /etc/portage/package.env
COPY config/env /etc/portage/env
# Keep the world set on PipeWire and let PipeWire provide PulseAudio compatibility.
COPY config/world /var/lib/portage/world
COPY config/dwm-6.8 /etc/portage/savedconfig/x11-wm/dwm-6.8

# Clean up the locales
RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update \
 && . /etc/profile
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN emerge -uDN @world

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

RUN qlist -Iv > /packages.txt

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

RUN useradd -m -G users,wheel,audio,video -s /bin/bash kiel \
 && for grp in libvirt kvm qemu; do \
      if getent group "${grp}" >/dev/null; then usermod -aG "${grp}" kiel; fi; \
    done

COPY config/xinitrc /home/kiel/.xinitrc

RUN emerge --depclean

# Network and services
COPY config/resolv.conf /etc/
COPY config/reverse-ssh-eu.initd /etc/init.d/reverse-ssh-eu
COPY config/reverse-ssh-us.initd /etc/init.d/reverse-ssh-us
RUN chmod +x /etc/init.d/reverse-ssh-* \
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

COPY config/sysctl.conf /etc/sysctl.conf

# For virt-manager (otherwise it crashes (!))
RUN glib-compile-schemas /usr/share/glib-2.0/schemas/

# For chinese font in chrome
RUN fc-cache -fv

RUN echo 'keymap="colemak"' > /etc/conf.d/keymaps

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
      --kver=${KVER} \
      --install "stat blockdev" \
      --force \
      /boot/initramfs_squash_sda1-x86_64.img

# Create ThinkPad E14 squashfs image (no NVIDIA runtime files, includes AMD/WiFi firmware)
RUN set -e \
 && echo "Preparing ThinkPad E14 squashfs (remove NVIDIA)" \
 && mkdir -p /tmp/firmware_e14 || true \
 \
 # gather firmware for E14 (from dmesg: regulatory, mediatek, amdgpu yellow_carp)
 && cp --parents regulatory.db* /tmp/firmware_e14/ 2>/dev/null || true \
 && cp --parents mediatek/WIFI_RAM_CODE_MT7922_1.bin /tmp/firmware_e14/ 2>/dev/null || true \
 && cp --parents mediatek/WIFI_MT7922_patch_mcu_1_1_hdr.bin /tmp/firmware_e14/ 2>/dev/null || true \
 && cp --parents amdgpu/yellow_carp* /tmp/firmware_e14/ 2>/dev/null || true \
 \
 # Build ThinkPad E14 image: hide/remove NVIDIA runtime to minimize size
 && echo "Stripping NVIDIA/CUDA and Firefox for E14 squashfs" \
 && rm -rf /opt/cuda /usr/lib64/lib{cuda,nvidia}* /usr/lib64/libnvidia* 2>/dev/null || true \
 && rm -rf /usr/lib64/firefox /usr/lib/firefox /usr/bin/firefox /usr/share/applications/firefox.desktop /usr/share/icons/hicolor/*/apps/firefox* /usr/share/pixmaps/firefox* /usr/share/man/man1/firefox.1* 2>/dev/null || true \
 && rm -rf /usr/lib64/libnvoptix.so* /usr/share/nvidia/nvoptix.bin 2>/dev/null || true \
 && rm -rf /nvidia${KVER} 2>/dev/null || true \
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
   -noI -noX \
   -wildcards \
   -e \
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
     home/*/.cache \
     var/log/journal \
     var/cache/genkernel \
     var/tmp \
     initramfs-with-squashfs.img \
     lost+found \
   || true
