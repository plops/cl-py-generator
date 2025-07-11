# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20250711 AS portage

# based on stage3 image (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-systemd-20250707
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
#COPY config/lsmod_hpz6_6.11.4-301.fc41.x86_64 /lsmod_fc41
#RUN make defconfig
#RUN make LSMOD=/lsmod_fc41 localmodconfig

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
-m ENCRYPTED_KEYS
RUN make prepare



RUN emerge -e @world --fetchonly
# The fetch should replace he one earlier

RUN emerge -e @world
RUN emerge --depclean
# only binutils is deleted

RUN make -j32
RUN make modules_install
RUN make install


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

# # Create a user
RUN useradd -m -G wheel -s /bin/bash martin
COPY config/simple_password.expect /simple_password.expect
RUN expect /simple_password.expect
RUN rm /simple_password.expect

COPY config/list_files.py /usr/bin/list_files.py
RUN chmod a+x /usr/bin/list_files.py
RUN python /usr/bin/list_files.py > /list_files.txt


WORKDIR /usr/src/linux
RUN ./scripts/config --file .config \
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

RUN make -j32
RUN make modules_install
RUN make install

# add to license: x11-drivers/nvidia-drivers NVIDIA-2025
# add to package.accept_keywords: x11-drivers/nvidia-drivers ~amd64
# make sure newline in front and after the line

RUN echo -e "\nx11-drivers/nvidia-drivers ~amd64" >> /etc/portage/package.accept_keywords/package.accept_keywords
RUN echo -e "\nx11-drivers/nvidia-drivers NVIDIA-2025" >> /etc/portage/package.license
#use flags: gnome-base/librsvg vala -debug -gtk-doc introspection
# RUN echo -e "\ngnome-base/librsvg vala -debug -gtk-doc introspection" >> /etc/portage/package.use/package.use
RUN echo -e "\nx11-drivers/nvidia-drivers X modules static-libs strip -tools -dist-kernel kernel-open -modules-compress -modules-sign -persistenced -powerd -wayland" >> /etc/portage/package.use/package.use
RUN echo "*/* VIDEO_CARDS: -* nvidia" > /etc/portage/package.use/00video


RUN emerge x11-drivers/nvidia-drivers

RUN useradd -m -G users,wheel,audio -s /bin/bash kiel
RUN echo "kiel:kiel" | chpasswd
#COPY config/passwd /etc/passwd
#COPY config/shadow /etc/shadow

RUN echo -e "\nVIDEO_CARDS=\"nvidia\"" >> /etc/portage/make.conf
RUN emerge --update --deep --changed-use @world
RUN emerge --depclean

WORKDIR /
# 22
RUN mksquashfs \
/ \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-noappend \
-not-reproducible \
-Xcompression-level 1 \
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
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,ath11k,ath10k,ath12k,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,i915,qca,cirrus} \
usr/lib/firmware/{iwlwifi,phanfw}* \
var/tmp \
initramfs-with-squashfs.img \
usr/lib64/libQt*.a \
lost+found \
var/log \
usr/share/gtk-doc \
usr/share/doc \
usr/share/locale


# Create an initramfs that loads the squashfs from disk and overlays it
# on the folder persistent/ on that partition

# Dracut Module documentation https://dracut-ng.github.io/dracut-ng/modules/core.html

COPY config/init_dracut.sh /usr/lib/dracut/modules.d/99base/init.sh
RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

RUN ln -s /lib/modules/6.12.31-gentoo-gentoo-dist /lib/modules/6.12.31-gentoo
RUN dracut \
  -m "base" \
  --kver=${KVER} \
  --filesystems " squashfs fat vfat overlay nvme fuse hid-cherry kvm nvidia_uvm ip6table_filter ip6_tables overlay vfat fat nvidia_drm nvidia_modeset nvidia intel_rapl_msr amd_atl intel_rapl_common amd64_edac edac_mce_amd kvm_amd kvm irqbypass snd_hda_codec_realtek snd_hda_codec_generic snd_hda_scodec_component snd_hda_codec_hdmi hp_wmi sparse_keymap snd_hda_intel platform_profile rfkill snd_intel_dspcfg wmi_bmof snd_hda_codec snd_hwdep snd_hda_core rapl pcspkr snd_pcm video igc snd_timer i2c_piix4 snd r8169 k10temp i2c_smbus soundcore wmi gpio_amdpt joydev gpio_generic fuse loop hid_cherry crct10dif_pclmul crc32_pclmul crc32c_intel polyval_clmulni polyval_generic ghash_clmulni_intel sha512_ssse3 sha256_ssse3 sha1_ssse3 serio_raw nvme nvme_core ucsi_acpi ccp sp5100_tco typec_ucsi typec " \
  --force \
  /boot/initramfs_squash_sda1-x86_64.img


