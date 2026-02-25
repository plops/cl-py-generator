# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20260225 AS portage

# based on stage3 image (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-systemd-20260223
# https://hub.docker.com/r/gentoo/portage/tags
# copy the entire portage volume in (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Configure the system
RUN eselect profile set default/linux/amd64/23.0/no-multilib/systemd
COPY config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords

# Install GCC 16
RUN emerge --ask=n --verbose sys-devel/gcc:16 \
 && gcc_16=$(eselect gcc list | grep "16" | awk '{print $1}' | tr -d '[]') \
 && eselect gcc set "${gcc_16}" \
 && source /etc/profile \
 && emerge --ask=n --oneshot --verbose libtool \
 # Use || true to prevent failure if it's not in world
 && (emerge --deselect=y sys-devel/gcc:14 || true) \
 # Use || true so build doesn't die if Portage refuses to remove it
 && (emerge --ask=n --depclean sys-devel/gcc:14 || true) \
 && emerge --ask=n @preserved-rebuild

 
#RUN emerge -1 gentoolkit eix app-eselect/eselect-repository dev-vcs/git dev-lang/rust-bin
COPY config/package.use /etc/portage/package.use
COPY config/make.conf /etc/portage/make.conf
COPY config/world /var/lib/portage/world
COPY config/package.license /etc/portage/package.license
COPY config/dwm-6.6 /etc/portage/savedconfig/x11-wm/dwm-6.6

# Install additional packages
RUN mv /etc/portage/package.license /etc/portage/package.license.bak
RUN mkdir -p /etc/portage/package.license
RUN mv /etc/portage/package.license.bak /etc/portage/package.license/package.license.1
COPY config/package.license.2 /etc/portage/package.license/
COPY config/package.accept_keywords.2 /etc/portage/package.accept_keywords/package.accept_keywords.2

# Clean up the locales
RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update \
 && . /etc/profile
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8
#RUN eselect repository enable guru
#RUN emaint sync -r guru
#RUN emerge --update --deep --changed-use @world --fetchonly

ENV KVER_PURE=6.12.58
ENV KVER=${KVER_PURE}-gentoo

RUN df -h
RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

RUN eselect kernel list
RUN eselect kernel set linux-${KVER}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config6.12.41 .config
#  Update current config utilising a provided .config as base
RUN make oldconfig
# cd example/110_gentoo/docker_min_hpz6/systems
# find .|grep lsmod|xargs cat|sort|uniq|grep -v nouveau > /dev/shm/lsmod
# yes "" | make LSMOD=/dev/shm/lsmod localmodconfig


# RUN ./scripts/config --file .config \
# --set-val MODPROBE_PATH "/usr/sbin/modprobe" \
# --set-val EXTRA_FIRMWARE "amd-ucode/microcode_amd_fam19h.bin rtw89/rtw8852b_fw.bin regulatory.db" \
# --set-val EXTRA_FIRMWARE_DIR "/usr/lib/firmware" \
# -e SQUASHFS
# amd/sof/* and amd/sof-tplg/* firmware blobs
RUN <<EOF
set -e
OPTIONS="
BT_BCM
BT_INTEL
BT_INTEL_PCIE
BT_MTK
BT_RTL
BT_BNEP
BT_HCIBTUSB
APPLE_MFI_FASTCHARGE
USB_NET_CDCETHER
EEEPC_WMI
ASUS_WMI
HP_WMI
IDEAPAD_LAPTOP
USB_NET_CDC_NCM
EEPROM_EE1004
EDAC_DECODE_MCE
THINKPAD_LMI
THINKPAD_ACPI
HID_CHERRY
HID_GENERIC
HID_LOGITECH
HID_LOGITECH_HIDPP
HID_MULTITOUCH
HID_LENOVO
LENOVO_WMI_CAMERA
HID_PLANTRONICS
CRYPTO_GHASH_CLMUL_NI_INTEL
INTEL_RAPL_CORE
INTEL_RAPL
USB_IPHETH
SENSORS_K10TEMP
NVME_KEYRING
NVME_AUTH
PPDEV
USB_NET_QMI_WWAN
QRTR
R8169
RTW88_8821C
RTW88_8821CU
RTW89_8852B
RTW89_8852BE
RTW89_8852B_COMMON
SND_HDA_CODEC_REALTEK
SND_HDA_INTEL
CRYPTO_SHA1_SSSE3
CRYPTO_SHA256_SSSE3
CRYPTO_SHA512_SSSE3
SND_SOC_AMD_ACP_LEGACY_COMMON
SND_SOC_AMD_ACP_COMMON
SND_SOC_AMD_ACP_PCI
USB_USBNET
USB_VIDEO_CLASS
USB_SERIAL_CH341
MT7921E
SND_SOC_SOF_TOPLEVEL
SND_SOC_SOF_PCI
SND_SOC_SOF_AMD_TOPLEVEL
SND_SOC_SOF_AMD_RENOIR
SND_SOC_SOF_AMD_REMBRANDT
SND_SOC_SOF_AMD_VANGOGH
SND_SOC_SOF_AMD_ACP63
SND_SOC_AMD_ACP6x
SND_SOC_AMD_PS
SND_SOC_DMIC
SND_HDA_CODEC_CONEXANT
SOUNDWIRE
SOUNDWIRE_AMD
SOUNDWIRE_GENERIC_ALLOCATION
SND_SOC_SDCA
AMD_PMC
THUNDERBOLT
BT_RFCOMM
SENSORS_SPD5118
SOUNDWIRE
SOUNDWIRE_AMD
SOUNDWIRE_GENERIC_ALLOCATION
SOUNDWIRE_BUS
SND_SOC_SDCA
AMD_PMC
SND_SOC_AMD_PS_MACH
THUNDERBOLT
USB4
BT_RFCOMM
BT_BNEP
CRYPTO_USER_API_HASH
CRYPTO_USER_API_SKCIPHER
CRYPTO_CMAC
MEDIA_SUPPORT
MEDIA_USB_SUPPORT
USB_VIDEO_CLASS
MEDIA_CAMERA_SUPPORT
VIDEO_DEV
VIDEO_V4L2
"
# Use printf to format each option with a preceding "-m"
# The result is captured into the M_OPTIONS variable
M_OPTIONS=$(printf -- '-m %s ' $OPTIONS)

# Execute the script a single time with the dynamically built list of options
./scripts/config --file .config $M_OPTIONS -e EFI_PARTITION -e SQUASHFS_4K_DEVBLK_SIZE -e CUSE -d USB4_NET -e SENSORS_SPD5118_DETECT -d MFD_CS42L43_SDW -e MEDIA_USB_SUPPORT -d USB_GSPCA -d TOUCHSCREEN_SUR40 -d USB_PWC -e BT_RFCOMM_TTY 
make olddefconfig # apply default configuration for whatever options are still open
EOF


RUN make prepare

# compile the kernel
RUN make -j32
RUN make modules_install
RUN make install


RUN emerge -e --ask=n @world

## Generate the full list of packages to build and split into 10 stages
#RUN emerge -pqe --columns @world | awk '{print $2}' > /tmp/world_packages.txt && \
#    total_lines=$(wc -l < /tmp/world_packages.txt) && \
#    lines_per_stage=$((($total_lines + 9) / 10)) && \
#    for i in {1..10}; do \
#      start=$(((i-1) * lines_per_stage + 1)); \
#      sed -n "${start},$((start + lines_per_stage - 1))p" /tmp/world_packages.txt > /tmp/stage_${i}_packages.txt; \
#    done
#
## Stage 1 of 10
#RUN if [ -s /tmp/stage_1_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_1_packages.txt | awk '{print $NF}'); fi
#
## Stage 2 of 10
#RUN if [ -s /tmp/stage_2_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_2_packages.txt | awk '{print $NF}'); fi
#
## Stage 3 of 10
#RUN if [ -s /tmp/stage_3_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_3_packages.txt | awk '{print $NF}'); fi
#
## Stage 4 of 10
#RUN if [ -s /tmp/stage_4_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_4_packages.txt | awk '{print $NF}'); fi
#
## Stage 5 of 10
#RUN if [ -s /tmp/stage_5_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_5_packages.txt | awk '{print $NF}'); fi
#
## Stage 6 of 10
#RUN if [ -s /tmp/stage_6_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_6_packages.txt | awk '{print $NF}'); fi
#
## Stage 7 of 10
#RUN if [ -s /tmp/stage_7_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_7_packages.txt | awk '{print $NF}'); fi
#
## Stage 8 of 10
#RUN if [ -s /tmp/stage_8_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_8_packages.txt | awk '{print $NF}'); fi
#
## Stage 9 of 10
#RUN if [ -s /tmp/stage_9_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_9_packages.txt | awk '{print $NF}'); fi
#
### Generate the full list of packages to build and split into 10 stages
### Using 'awk' to extract only the package name and avoiding empty lines
##RUN emerge -pqe --columns @world | awk '$2 ~ /\// {print $2}' > /tmp/world_packages.txt && \
##    total_lines=$(wc -l < /tmp/world_packages.txt) && \
##    lines_per_stage=$(( (total_lines + 9) / 10 )) && \
##    for i in {1..10}; do \
##      start=$(((i-1) * lines_per_stage + 1)); \
##      sed -n "${start},$((start + lines_per_stage - 1))p" /tmp/world_packages.txt > /tmp/stage_${i}_packages.txt; \
##    done
#
#
## Stage 10 of 10
#RUN if [ -s /tmp/stage_10_packages.txt ]; then emerge --ask=n $(cat /tmp/stage_10_packages.txt | grep / | awk '{print $NF}'); fi

RUN emerge --depclean

# # Allow members of the wheel group to execute any command without a password
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel


# # Add a tool to read the Ryzen temperatures, frequencies and voltages
WORKDIR /usr/src
RUN git clone https://github.com/amkillam/ryzen_smu
WORKDIR /usr/src/ryzen_smu
# Path should be like /lib/modules/6.13.5-gentoo/
# Replace $(shell uname -r) with ${KVER}-x86_64, otherwise the TARGET folder is not defined correctly
RUN sed -i "s/\$(shell uname -r)/${KVER}/g" Makefile
RUN make -j7
RUN cp ryzen_smu.ko /lib/modules/${KVER}/kernel/
RUN depmod -a ${KVER}
RUN rm *.o


# # Add slstatus to show battery status and time
WORKDIR /usr/src
RUN git clone https://git.suckless.org/slstatus
WORKDIR /usr/src/slstatus
COPY config/slstatus_config.h .
RUN make -j32
RUN make install
RUN make clean

RUN useradd -m -G users,wheel,audio,video -s /bin/bash kiel

COPY config/xinitrc /home/kiel/.xinitrc


RUN emerge --depclean

COPY config/20-enp65s0.network /etc/systemd/network/
COPY config/resolv.conf /etc/
COPY config/reverse-ssh-eu.service /etc/systemd/system/
COPY config/reverse-ssh-us.service /etc/systemd/system/
RUN systemctl enable reverse-ssh-eu
RUN systemctl enable reverse-ssh-us


# on my laptop i only need
# [   61.916836] Loading firmware: regulatory.db
# [   61.934422] Loading firmware: regulatory.db.p7s
# [   63.134619] Loading firmware: mediatek/WIFI_RAM_CODE_MT7922_1.bin
# [   63.136954] Loading firmware: amdgpu/yellow_carp_toc.bin
# [   63.139761] Loading firmware: amdgpu/yellow_carp_ta.bin
# [   63.142223] Loading firmware: amdgpu/yellow_carp_dmcub.bin
# [   63.143726] Loading firmware: amdgpu/yellow_carp_pfp.bin
# [   63.144158] Loading firmware: amdgpu/yellow_carp_me.bin
# [   63.144554] Loading firmware: amdgpu/yellow_carp_ce.bin
# [   63.144991] Loading firmware: amdgpu/yellow_carp_rlc.bin
# [   63.145354] Loading firmware: amdgpu/yellow_carp_mec.bin
# [   63.145729] Loading firmware: amdgpu/yellow_carp_mec2.bin
# [   63.146091] Loading firmware: amdgpu/yellow_carp_sdma.bin
# [   63.146270] Loading firmware: amdgpu/yellow_carp_vcn.bin
# [   63.206931] Loading firmware: mediatek/WIFI_MT7922_patch_mcu_1_1_hdr.bin
# [   63.589195] Loading firmware: mediatek/WIFI_RAM_CODE_MT7922_1.bin
# [   63.634733] Loading firmware: mediatek/WIFI_RAM_CODE_MT7922_1.bin

# 1. Create a clean temporary workspace
RUN mkdir -p /tmp/firmware_safe
WORKDIR /usr/lib/firmware

# 2. Copy the specific files you need, preserving their directory paths
RUN cp --parents regulatory.db* /tmp/firmware_safe/
RUN cp --parents mediatek/WIFI_RAM_CODE_MT7922_1.bin /tmp/firmware_safe/
RUN cp --parents mediatek/WIFI_MT7922_patch_mcu_1_1_hdr.bin /tmp/firmware_safe/
RUN cp --parents amdgpu/yellow_carp* /tmp/firmware_safe/

# 3. Wipe the original directory (and /opt/cuda as per your first message)
WORKDIR /
RUN rm -rf /usr/lib/firmware/*

# 4. Move everything back in one go
RUN mv /tmp/firmware_safe/* /usr/lib/firmware/

# 5. Cleanup
RUN rm -rf /tmp/firmware_safe

# 12.8GB uncompressed
RUN rm -rf /opt/cuda /usr/lib64/lib{cuda,nvidia}*

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

WORKDIR /
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
usr/src \
var/cache/binpkgs \
var/cache/distfiles \
"gentoo*squashfs" \
"usr/lib64/libQt*.a" \
usr/share/genkernel/distfiles \
usr/src/linux \
usr/share/sgml \
var/cache/eix/previous.eix \
boot \
persistent \
home/martin/.cache/mozilla \
home/martin/.cache/google-chrome \
home/martin/.cache/mesa_shader_cache \
home/martin/.cache/fontconfig \
home/martin/Downloads \
home/martin/.config \
home/martin/.mozilla \
home/martin/stage \
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

COPY config/mount-overlayfs.sh_hpz6 /usr/lib/dracut/modules.d/70overlayfs
RUN chmod +x /usr/lib/dracut/modules.d/70overlayfs/*.sh
RUN dracut \
      -m "kernel-modules base dmsquash-live dm udev-rules lvm crypt overlayfs" \
      --filesystems "squashfs vfat ext4 overlay btrfs" \
      --kver=${KVER} \
      --install "stat blockdev" \
      --force \
      /boot/initramfs-hpz6-x86_64.img



