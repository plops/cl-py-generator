# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20260217 AS portage

# based on stage3 image
FROM gentoo/stage3:nomultilib-20260216 AS base

# copy the entire portage volume in (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Configure the system and copy all initial configuration files
COPY config/make.conf /etc/portage/make.conf
COPY config/package.accept_keywords /etc/portage/package.accept_keywords/package.accept_keywords
COPY config/world /var/lib/portage/world
COPY config/package.license /etc/portage/package.license
COPY config/dwm-6.5 /etc/portage/savedconfig/x11-wm/dwm-6.5

RUN eselect profile set default/linux/amd64/23.0/no-multilib && \
    echo "C.UTF-8 UTF-8" >> /etc/locale.gen && \
    emerge -1 gentoolkit eix

# Install GCC 16, switch to it, and prepare for world update
COPY config/package.use /etc/portage/package.use
RUN emerge -vp sys-devel/gcc:16 > /emerge-gcc16-stage1.txt && \
    emerge --ask=n --verbose sys-devel/gcc:16 && \
    gcc_16=$(eselect gcc list | grep "16" | awk '{print $1}' | tr -d '[]') && \
    eselect gcc set "${gcc_16}" && \
    source /etc/profile && \
    emerge --ask=n --oneshot --verbose libtool && \
    (emerge --deselect=y sys-devel/gcc:14 || true) && \
    (emerge --ask=n --depclean sys-devel/gcc:14 || true) && \
    emerge --ask=n @preserved-rebuild && \
    emerge --update --deep --changed-use @world --fetchonly

# Install kernel sources and configure
ENV KVER_PURE=6.12.58
ENV KVER=${KVER_PURE}-gentoo
RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE} && \
    eselect kernel list && \
    eselect kernel set linux-${KVER}

WORKDIR /usr/src/linux
RUN make defconfig && \
    make kvm_guest.config

# Configure kernel - disable unnecessary drivers and enable required features
RUN for i in \
	DRM_NOUVEAU DRM_EXYNOS DRM_ROCKCHIP DRM_RCAR_DU DRM_RCAR_DW_HDMI \
	DRM_RCAR_USE_LVDS DRM_RCAR_USE_MIPI_DSI DRM_IMX_DCSS DRM_ETNAVIV \
	DRM_HISI_HIBMC DRM_HISI_KIRIN DRM_MEDIATEK DRM_MSM DRM_MXSFB \
	DRM_MESON DRM_PL111 DRM_TIDSS DRM_LEGACY DRM_SUN4I DRM_TEGRA DRM_I915 \
	TEGRA_HOST1X SCSI_UFSHCD FPGA RC_CORE NEW_LEDS CHROME_PLATFORMS \
	SURFACE_PLATFORMS XEN_BLKDEV_FRONTEND LOGO SOUND SLIMBUS SOUNDWIRE \
	MEDIA_SUPPORT MMC NFS_FS 9P_FS SUSPEND HIBERNATION VIRTUALIZATION \
	WLAN PINCTRL GPIOLIB PWM IPMI_HANDLER CAN BT WIRELESS MD RFKILL \
	NET_9P NFC SPI SPMI HWMON THERMAL IIO USB_NET_DRIVERS \
	XEN_NETDEV_FRONTEND ETHERNET QCOM_IPA REGULATOR STAGING DEBUG_KERNEL XEN \
    ; do ./scripts/config --disable $i; done && \
    for i in \
    MODULES OVERLAY_FS_REDIRECT_DIR OVERLAY_FS_INDEX OVERLAY_FS_XINO_AUTO \
    OVERLAY_FS_METACOPY SQUASHFS_ZSTD SQUASHFS_XATTR SQUASHFS_4K_DEVBLK_SIZE \
    SQUASHFS_DECOMP_MULTI SQUASHFS_COMPILE_DECOMP_MULTI SQUASHFS_FILE_DIRECT \
    MD CUSE \
    ; do ./scripts/config --enable $i; done && \
    for i in \
    OVERLAY_FS BLK_DEV_DM DM_CRYPT DM_SNAPSHOT DM_MIRROR SQUASHFS \
    VFAT_FS EXT4_FS BTRFS_FS CRYPTO_AES CRYPTO_XTS CRYPTO_SHA256 \
    CRYPTO_USER_API_SKCIPHER CRYPTO_USER_API_HASH CRYPTO_AES_X86_64 \
    CRPYTO_AES_NI_INTEL FANOTIFY CGROUP_BPF \
    ; do ./scripts/config --module $i; done && \
    ./scripts/config --set-str CMDLINE "" && \
    make prepare

# Install and configure steve jobserver, then prepare for world rebuild
RUN emerge --ask=n dev-build/steve && \
    cat > /usr/local/bin/with-steve.sh << 'EOF' && \
chmod +x /usr/local/bin/with-steve.sh
#!/bin/bash
set -e
mkdir -p /run/steve
if ! pgrep -x steve > /dev/null; then
    /usr/bin/steve --jobs=32 --load-average=32 --verbose &
    STEVE_PID=$!
    sleep 2
    "$@"
    RESULT=$?
    kill $STEVE_PID 2>/dev/null || true
    wait $STEVE_PID 2>/dev/null || true
    exit $RESULT
else
    "$@"
fi
EOF

# Configure package.use.ssl and package.mask for systemd-free world rebuild
COPY config/package.use.ssl /tmp/package.use.ssl
COPY config/package.mask.nosystemd /tmp/package.mask.nosystemd
RUN install -d /etc/portage/package.use /etc/portage/package.mask && \
    cat /tmp/package.use.ssl >> /etc/portage/package.use/zz-local && \
    cat /tmp/package.mask.nosystemd >> /etc/portage/package.mask/zz-nosystemd && \
    rm /tmp/package.use.ssl /tmp/package.mask.nosystemd && \
    emerge --ask=n --oneshot dev-lang/python:3.13 sys-apps/systemd-utils

# Download all packages first (cached layer - won't re-download if build fails)
RUN with-steve.sh emerge -e @world --fetchonly

# Rebuild world with steve jobserver coordination (separate layer for debugging)
RUN with-steve.sh emerge -e @world

# Clean up dependencies (separate layer)
RUN with-steve.sh emerge --depclean

# Build kernel with steve jobserver coordination (can be re-run if build fails)
RUN with-steve.sh make -j32

# Install kernel modules and kernel (separate layer - faster if only kernel config changes)
RUN make modules_install && \
    make install

# Configure sudo and create user with password
RUN mkdir -p /etc/sudoers.d && \
    echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel && \
    useradd -m -G wheel -s /bin/bash martin

# Set up password using expect script
COPY config/simple_password.expect /simple_password.expect
RUN GENTOO_MIRRORS="" emerge dev-tcltk/expect && \
    expect /simple_password.expect && \
    rm /simple_password.expect

# Generate file list for exclusions
COPY config/list_files.py /usr/bin/list_files.py
RUN chmod a+x /usr/bin/list_files.py && \
    python /usr/bin/list_files.py > /list_files.txt

# Create final squashfs image
WORKDIR /
RUN emerge --info -v > emerge_info_v.txt && \
    rm -rf /usr/lib/lib*.a \
           /usr/x86_64-pc-linux-gnu/gcc-bin/15/lto-dump \
           /usr/share/i18n/ \
           /usr/var/db/pkg/sys-kernel/gentoo-sources-6.12.58/CONTENTS \
           /usr/lib/gcc/x86_64-pc-linux-gnu/15/lib*.a

RUN mksquashfs \
/ \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-noappend \
-not-reproducible \
-Xcompression-level 19 \
-progress \
-b 256K -noI -noX \
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
'opt/rust-bin*' \
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

# Create initramfs with overlayfs support
COPY config/mount-overlayfs.sh /usr/lib/dracut/modules.d/70overlayfs
RUN chmod +x /usr/lib/dracut/modules.d/70overlayfs/*.sh && \
    dracut \
      -m "kernel-modules base dmsquash-live dm udev-rules lvm crypt overlayfs" \
      --filesystems "squashfs vfat ext4 overlay btrfs" \
      --kver=${KVER} \
      --install "stat blockdev" \
      --force \
      /boot/initramfs_squash_sda1-x86_64.img


