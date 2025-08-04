# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20250804 AS portage

# based on stage3 image (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-systemd-20250804
# https://hub.docker.com/r/gentoo/portage/tags
# copy the entire portage volume in (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Configure the system
RUN eselect profile set default/linux/amd64/23.0/no-multilib/systemd
RUN emerge -1 gentoolkit eix
COPY config/package.use /etc/portage/package.use
COPY config/make.conf /etc/portage/make.conf
COPY config/world /var/lib/portage/world
COPY config/package.license /etc/portage/package.license
COPY config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
COPY config/dwm-6.5 /etc/portage/savedconfig/x11-wm/dwm-6.5

# Clean up the locales
RUN echo "C.UTF-8 UTF-8" >> /etc/locale.gen
RUN rm -rf /usr/share/locale/*
RUN locale-gen

RUN emerge --update --deep --changed-use @world --fetchonly

ENV KVER_PURE=6.12.31
ENV KVER=${KVER_PURE}-gentoo

RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

RUN eselect kernel list
RUN eselect kernel set linux-${KVER}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config6.12.31 .config
RUN make oldconfig

# pesto is what you find on google when you search for fat penguin
# Maybe: -e CONFIG_RFKILL instead of -m
# --set-val BUILD_SALT "6.13.5-gentoo"
RUN ./scripts/config --file .config \
-d DEBUG_INFO \
-d DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
-d DEBUG_INFO_DWARF4 \
-d DEBUG_INFO_DWARF5 \
-e DEBUG_INFO_NONE \
--set-val DEFAULT_HOSTNAME "pesto" \
-e MZEN4 \
-d GENERIC_CPU \
-d CPU_MITIGATIONS \
--set-val MODPROBE_PATH "/usr/sbin/modprobe" \
-m NETFILTER_XT_NAT \
-m NETFILTER_XT_TARGET_MASQUERADE \
-m IP_NF_IPTABLES \
--set-val EXTRA_FIRMWARE "amd-ucode/microcode_amd_fam19h.bin" \
--set-val EXTRA_FIRMWARE_DIR "/usr/lib/firmware" \
-m DM_CRYPT \
-m MII \
-m USB_NET_DRIVERS \
-m USB_USBNET \
-m USB_NET_CDCETHER \
-m USB_NET_CDC_EEM \
-m USB_NET_CDC_NCM \
-m USB_NET_RNDIS_HOST \
-m USB_NET_CDC_SUBSET_ENABLE \
-m USB_NET_CDC_SUBSET \
-e USB_BELKIN \
-e USB_ARMLINUX \
-e HID_CHICONY \
-e HID_LOGITECH \
-e USB_STORAGE \
-e OVERLAY_FS \
-e OVERLAY_FS_REDIRECT_ALWAYS_FOLLOW \
-m ISO9660_FS \
-e SQUASHFS \
-d SQUASHFS_FILE_CACHE \
-e SQUASHFS_FILE_DIRECT \
-e SQUASHFS_DECOMP_MULTI_PERCPU \
-d SQUASHFS_CHOICE_DECOMP_BY_MOUNT \
-d SQUASHFS_COMPILE_DECOMP_SINGLE \
-d SQUASHFS_COMPILE_DECOMP_MULTI \
-e SQUASHFS_COMPILE_DECOMP_MULTI_PERCPU \
-e SQUASHFS_XATTR \
-e SQUASHFS_ZLIB \
-d SQUASHFS_LZ4 \
-d SQUASHFS_LZO \
-e SQUASHFS_XZ \
-e SQUASHFS_ZSTD \
-d SQUASHFS_4K_DEVBLK_SIZE \
-d SQUASHFS_EMBEDDED \
--set-val SQUASHFS_FRAGMENT_CACHE_SIZE 3 \
-m ENCRYPTED_KEYS \
        -e CONFIG_BLK_DEV_NVME \
        -e CONFIG_NVME_CORE \
        -d DRM_NOUVEAU \
        -e HID_CHERRY \
        -e ACPI_IPMI \
        -d FB_NVIDIA -d FB_RIVA -d DRM_SIMPLEDRM \
        -e SIMPLEFB -e FB_VESA -e FB_EFI -e FB_SIMPLE \
        -d DRM \
        -e BLK_DEV_LOOP \
        --set-val LOCALVERSION "-gentoo"
RUN make prepare

RUN emerge -e @world --fetchonly

# compile the kernel
RUN make -j32
RUN make modules_install
RUN make install

RUN emerge -e @world
RUN emerge --depclean

# # Allow members of the wheel group to execute any command without a password
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel

# # Add slstatus to show battery status and time
WORKDIR /usr/src
RUN git clone https://git.suckless.org/slstatus
WORKDIR /usr/src/slstatus
COPY config/slstatus_config.h .
RUN make -j32
RUN make install
RUN make clean

RUN useradd -m -G users,wheel,audio,video -s /bin/bash kiel
RUN echo "kiel:kiel" | chpasswd

COPY config/xinitrc /home/kiel/.xinitrc

# Install additional packages
RUN mv /etc/portage/package.license /etc/portage/package.license.bak
RUN mkdir -p /etc/portage/package.license
RUN mv /etc/portage/package.license.bak /etc/portage/package.license/package.license.1
COPY config/package.license.2 /etc/portage/package.license
COPY config/package.accept_keywords.2 /etc/portage/package.accept_keywords/package.accept_keywords.2
COPY config/package.use.2 /etc/portage/package.use/package.use.2

RUN emerge \
    dev-util/clion \
    sys-process/btop \
    www-client/firefox-bin \
    app-containers/docker \
    app-containers/docker-buildx \
    app-containers/docker-cli \
    app-editors/emacs \
    app-editors/vscode \
    app-misc/mc \
    app-text/mupdf \
    dev-debug/gdb \
    dev-lisp/sbcl \
    dev-python/uv \
    dev-util/nvidia-cuda-toolkit \
    dev-util/ruff \
    media-gfx/feh \
    media-gfx/scrot \
    net-fs/sshfs \
    net-im/signal-desktop-bin \
    net-misc/apt-cacher-ng \
    net-misc/chrony \
    net-misc/freerdp \
    x11-misc/xclip \
    x11-misc/xtrlock \
    app-crypt/p11-kit \
    net-misc/chrony \
    net-fs/sshfs \
    sys-apps/qdirstat \
    dev-debug/ltrace \
    dev-debug/dtrace \
    dev-util/ruff \
    app-text/poppler

RUN emerge --depclean

COPY config/list_files.py /usr/bin/list_files.py
RUN chmod a+x /usr/bin/list_files.py
RUN python /usr/bin/list_files.py > /list_files.txt

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
-mem 10G \
-ef /list_files.txt     \
-wildcards \
-e \
usr/src \
var/cache/binpkgs \
var/cache/distfiles \
'gentoo*squashfs' \
'usr/lib/llvm/*/lib64/lib*.a' \
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
usr/lib64/libQt*.a \
lost+found \
var/log \
usr/share/gtk-doc \
usr/share/doc \
usr/share/locale

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


