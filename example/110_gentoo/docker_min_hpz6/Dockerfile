# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20260121 AS portage

# based on stage3 image (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-systemd-20260121
# https://hub.docker.com/r/gentoo/portage/tags
# copy the entire portage volume in (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Configure the system
RUN eselect profile set default/linux/amd64/23.0/no-multilib/systemd

# Install GCC 15
RUN emerge --ask=n --verbose sys-devel/gcc:15 \
 && gcc_15=$(eselect gcc list | grep "15" | awk '{print $1}' | tr -d '[]') \
 && eselect gcc set "${gcc_15}" \
 && source /etc/profile \
 && emerge --ask=n --oneshot --verbose libtool \
 # Use || true to prevent failure if it's not in world
 && (emerge --deselect=y sys-devel/gcc:14 || true) \
 # Use || true so build doesn't die if Portage refuses to remove it
 && (emerge --ask=n --depclean sys-devel/gcc:14 || true) \
 && emerge --ask=n @preserved-rebuild
 
RUN emerge -1 gentoolkit eix app-eselect/eselect-repository dev-vcs/git dev-lang/rust-bin
COPY config/package.use /etc/portage/package.use
COPY config/make.conf /etc/portage/make.conf
COPY config/world /var/lib/portage/world
COPY config/package.license /etc/portage/package.license
COPY config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
COPY config/dwm-6.6 /etc/portage/savedconfig/x11-wm/dwm-6.6

# Install additional packages
RUN mv /etc/portage/package.license /etc/portage/package.license.bak
RUN mkdir -p /etc/portage/package.license
RUN mv /etc/portage/package.license.bak /etc/portage/package.license/package.license.1
COPY config/package.license.2 /etc/portage/package.license
COPY config/package.accept_keywords.2 /etc/portage/package.accept_keywords/package.accept_keywords.2

# Clean up the locales
RUN printf 'en_US.UTF-8 UTF-8\n' > /etc/locale.gen \
 && locale-gen \
 && env-update \
 && . /etc/profile
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8
RUN eselect repository enable guru
RUN emaint sync -r guru
RUN emerge --update --deep --changed-use @world --fetchonly

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
./scripts/config --file .config $M_OPTIONS -e EFI_PARTITION -d USB4_NET -e SENSORS_SPD5118_DETECT -d MFD_CS42L43_SDW -e MEDIA_USB_SUPPORT -d USB_GSPCA -d TOUCHSCREEN_SUR40 -d USB_PWC 
make olddefconfig # apply default configuration for whatever options are still open
EOF


RUN make prepare


RUN emerge -e @world --fetchonly

# compile the kernel
RUN make -j32
RUN make modules_install
RUN make install

#iRUN emerge --update --deep --newuse --with-bdeps=y @world
RUN emerge -e @world
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
#RUN echo "kiel:Cosmos3fed*debtor" | chpasswd

COPY config/xinitrc /home/kiel/.xinitrc


RUN emerge --depclean
#RUN chmod a+rx /opt/clion/bin/clang/linux/x64/bin/clang-format

#RUN IGNORE_PREEMPT_RT_PRESENCE=1 emerge nvidia-drivers

COPY config/20-enp65s0.network /etc/systemd/network/
COPY config/resolv.conf /etc/
COPY config/reverse-ssh-eu.service /etc/systemd/system/
COPY config/reverse-ssh-us.service /etc/systemd/system/
RUN systemctl enable reverse-ssh-eu
RUN systemctl enable reverse-ssh-us

#COPY config/list_files.py /usr/bin/list_files.py
#RUN chmod a+x /usr/bin/list_files.py
#RUN python /usr/bin/list_files.py > /list_files.txt
# -ef /list_files.txt
# "... *.a"
# "opt/cuda/targets/x86_64-linux/lib/lib*.a"


WORKDIR /
RUN mksquashfs \
/ \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-noappend \
-not-reproducible \
-Xcompression-level 22 \
-progress \
-one-file-system-x \
-b 1M \
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

# "usr/lib/llvm/*/lib64/lib*.a" 
# 'opt/rust-bin*'
#usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,ath11k,ath10k,ath12k,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,i915,qca,cirrus}
#usr/lib/firmware/{iwlwifi,phanfw}*

# Create an initramfs that loads the squashfs from disk and overlays it
# on the folder persistent/ on that partition

# Dracut Module documentation https://dracut-ng.github.io/dracut-ng/modules/core.html

#COPY config/init_dracut.sh /usr/lib/dracut/modules.d/99base/init.sh
COPY config/init_dracut_crypt.sh /usr/lib/dracut/modules.d/99base/init.sh
RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

RUN dracut \
      -m " kernel-modules base rootfs-block dm udev-rules crypt lvm " \
      --filesystems " squashfs vfat overlay btrfs " \
   --kver=${KVER} \
  --force \
  /boot/initramfs_squash_sda1-x86_64.img


COPY config/init_dracut_crypt_disk.sh /usr/lib/dracut/modules.d/99base/init.sh
RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

RUN dracut \
      -m " kernel-modules base rootfs-block dm udev-rules crypt lvm " \
      --filesystems " squashfs vfat overlay btrfs " \
   --kver=${KVER} \
  --force \
  /boot/initramfs_squash_from_disk.img

# allow passing the initramfs public key and tunnel private key into the build
ARG INITRAMFS_AUTH_KEY=""
ARG TUNNEL_KEY=""

# Create dracut assets for remote-unlock initramfs and include them in a new initramfs image.
# Ensure the directory exists and write any provided keys; fall back to generating tunnel key if not provided.
RUN mkdir -p /etc/dracut-crypt-ssh \
    && mkdir -p /root/.ssh \
 && if [ -n "$INITRAMFS_AUTH_KEY" ]; then printf '%s\n' "$INITRAMFS_AUTH_KEY" > /root/.ssh/authorized_keys; else :; fi \
 && if [ -n "$TUNNEL_KEY" ]; then printf '%s\n' "$TUNNEL_KEY" > /etc/dracut-crypt-ssh/tunnel_id_rsa && chmod 600 /etc/dracut-crypt-ssh/tunnel_id_rsa; else ssh-keygen -f /etc/dracut-crypt-ssh/tunnel_id_rsa -t rsa -N "" >/dev/null 2>&1; fi \
 && ssh-keygen -f /etc/dracut-crypt-ssh/ssh_host_ecdsa_key -t ecdsa -N "" >/dev/null 2>&1 \
 && chmod 600 /etc/dracut-crypt-ssh/ssh_host_ecdsa_key || true

# Add the tunnel starter script and (optional) static authorized_keys fallback file
COPY config/start-reverse-tunnels.sh /etc/dracut-crypt-ssh/start-reverse-tunnels.sh
RUN chmod +x /etc/dracut-crypt-ssh/start-reverse-tunnels.sh || true

# Build the third initramfs which includes dracut-crypt-ssh and network so dropbear + autossh can run
# RUN dracut \
#       -m "kernel-modules base rootfs-block dm udev-rules crypt lvm network crypt-ssh" \
#       --add-drivers "r8169" \
#       --filesystems "squashfs vfat overlay btrfs" \
#       --install "/etc/dracut-crypt-ssh/ssh_host_ecdsa_key /etc/dracut-crypt-ssh/start-reverse-tunnels.sh /etc/dracut-crypt-ssh/tunnel_id_rsa" \
#       --kver=${KVER} \
#       --force \
#       /boot/initramfs-with-ssh.img

# real    174m37.518s
#70 198.6 Filesystem size 7895377.73 Kbytes (7710.33 Mbytes)
#70 198.6       36.59% of uncompressed filesystem size (21579915.17 Kbytes)