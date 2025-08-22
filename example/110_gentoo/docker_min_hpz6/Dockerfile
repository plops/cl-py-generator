# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20250822 AS portage

# based on stage3 image (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-systemd-20250818
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

# Install additional packages
RUN mv /etc/portage/package.license /etc/portage/package.license.bak
RUN mkdir -p /etc/portage/package.license
RUN mv /etc/portage/package.license.bak /etc/portage/package.license/package.license.1
COPY config/package.license.2 /etc/portage/package.license
COPY config/package.accept_keywords.2 /etc/portage/package.accept_keywords/package.accept_keywords.2

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
--set-val EXTRA_FIRMWARE "amd-ucode/microcode_amd_fam19h.bin rtw89/rtw8852b_fw.bin regulatory.db" \
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
--set-val LOCALVERSION "-gentoo" \
-m WIREGUARD \
-e MACVLAN \
-e VXLAN \
-e VETH \
-e IPVALN \
-e IPVLAN_L3S \
-m IGC \
-m R8169 \
-e REALTEK_PHY \
-m GPIO_AMDPT \
-m DRM_GPUVM \
-e FB_DEVICE \
-m HID_CHERRY \
-m EDAC_AMD64 \
-m MXM_WMI \
-m HP_WMI \
-m RAS_FMPM \
-d CPU_MITIGATIONS \
-m NF_CONNTRACK \
-m NF_NAT \
-m NETFILTER_XTABLES \
-m NF_DEFRAG_IPV4 \
-m NF_DEFRAG_IPV6 \
-m STP \
-m BRIDGE \
-m LLC \
-e BT \
-m BT_BREDR \
-m BT_RFCOMM \
-e BT_RFCOM_TTY \
-m BT_BNEP \
-e BT_BNEP_MC_FILTER \
-e BT_BNEP_PROTO_FILTER \
-m BT_HIDP \
-e BT_LE_L2CAP_ECRED \
-e BT_MSFTEXT \
-e BT_AOSPEXT \
-e BT_DEBUGFS \
-m BT_INTEL \
-m BT_BCM \
-m BT_RTL \
-m BT_MTK \
-m BT_HCIBTUSB \
-m WEXT_CORE \
-m WEXT_PROC \
-m CFG80211 \
-m MAC80211 \
-e CFG80211_REQUIRE_SIGNED_REGDB \
-e CFG80211_USE_KERNEL_REGDB_KEYS \
-e CFG80211_DEFAULT_PS \
-e CFG80211_DEBUGFS \
-e CFG80211_CRDA_SUPPORT \
-e CFG80211_WEXT \
-e MAC80211_HAS_RC \
-e MAC80211_RC_MINSTREL \
-e MAC80211_RC_DEFAULT_MINSTREL \
-e MAC80211_MESH \
-e MAC80211_LEDS \
-e MAC80211_DEBUGFS \
-e RFKILL \
-m MAC_EMUMOUSEBTN \
-m RTW89 \
-m RTW89_CORE \
-m RTW89_PCI \
-m RTW89_8852B_COMMON \
-m RTW89_8852B \
-m RTW89_8852BE \
-e RTW89_DEBUG \
-e RTW89_DEBUGMSG \
-e RTW89_DEBUGFS \
-m VIRT_WIFI \
-m INPUT_MOUSEDEV \
-m KEYBOARD_ATKBD \
-e GPIO_PALMAS \
-e GPIO_RC5T583 \
-e GPIO_TPS6586X \
-e GPIO_TPS65910 \
-e POWER_RESET_RESTART \
-e CHARGER_MANAGER \
-m MEDIA_SUPPORT \
-m MEDIA_SUPPORT_FILTER \
-m MEDIA_SUBDRV_AUTOSELECT \
-m MEDIA_CAMERA_SUPPORT \
-m VIDEO_DEV \
-m VIDEO_V4L2_I2C \
-m VIDEO_V4L2_SUBDEV_API \
-m V4L2_FWNODE \
-m V4L2_ASYNC \
-m MEDIA_USB_SUPPORT \
-m USB_VIDEO_CLASS \
-m USB_VIDEO_CLASS_INPUT_EVDEV \
-m V4L_PLATFORM_DRIVERS \
-m V4L_MEM2MEM_DRIVERS \
-m UVC_COMMON \
-m VIDEOBUF2_CORE \
-m VIDEOBUF2_V4L2 \
-m VIDEOBUF2_MEMOPS \
-m VIDEOBUF2_VMALLOC \
-m VIDEO_CAMERA_SENSOR \
-e SCREEN_INFO \
-e VIDEO \
-e DRM \
-m DRM_KMS_HELPER \
-m DRM_FBDEV_EMULATION \
-m DRM_LOAD_EDID_FIRMWARE \
-m DRM_DISPLAY_HELPER \
-m DRM_DISPLAY_DP_HELPER \
-m DRM_DISPLAY_HDCP_HELPER \
-m DRM_DISPLAY_HDMI_HELPER \
-m DRM_TTM \
-m DRM_EXEC \
-m DRM_BUDDY \
-m DRM_TTM_HELPER \
-e DRM_GEM_SHMEM_HELPER \
-m DRM_SUBALLOC_HELPER \
-m DRM_SCHED \
-m DRM_AMDGPU \
-e DRM_AMDGPU_USERPTR \
-m DRM_AMD_ACP \
-e DRM_AMD_DC \
-e DRM_AMD_DC_FP \
-e DRM_AMD_SECURE_DISPLAY \
-e HSA_AMD \
-e HSA_AMD_SVM \
-e DRM_PANEL \
-e DRM_BRIDGE \
-e DRM_PANEL_BRIDGE \
-e DRM_SIMPLEDRM \
-e HDMI \
-m SND_SOC \
-m SND_SOC_COMPRESS \
-m SND_SOC_TOPOLOGY \
-m SND_SOC_ACPI \
-m SND_SOC_AMD_ACP3x \
-m SND_SOC_AMD_RENOIR \
-m SND_SOC_AMD_ACP5x \
-m SND_SOC_AMD_ACP6x \
-m SND_AMD_ACP_CONFIG \
-m SND_SOC_AMD_ACP_COMMON \
-m SND_SOC_ACPI_AMD_MATCH \
-m SND_SOC_AMD_ACP_LEGACY_COMMON \
-m SND_SOC_AMD_ACP_PCI \
-m SND_AMD_SOUNDWIRE_ACPI \
-m SND_SOC_AMD_RPL_ACP6x \
-m SND_SOC_AMD_ACP63_TOPLEVEL \
-m SND_SOC_AMD_SOUNDWIRE_LINK_BASELINE \
-m SND_SOC_AMD_PS \
-m SND_SOC_IMG \
-m SND_SOC_INTEL_SST_TOPLEVEL \
-m SND_SOC_ACPI_INTEL_MATCH \
-m SND_SOC_INTEL_MACH \
-m SND_SOC_INTEL_USER_FRIENDLY_LONG_NAMES \
-m SND_SOC_SOF_TOPLEVEL \
-m SND_SOC_SOF_PCI_DEV \
-m SND_SOC_SOF_PCI \
-m SND_SOC_SOF_ACPI \
-m SND_SOC_SOF_ACPI_DEV \
-m SND_SOC_SOF_DEBUG_PROBES \
-m SND_SOC_SOF_CLIENT \
-m SND_SOC_SOF \
-m SND_SOC_SOF_IPC3 \
-m SND_SOC_SOF_IPC4 \
-m SND_SOC_SOF_AMD_TOPLEVEL \
-m SND_SOC_SOF_AMD_COMMON \
-m SND_SOC_SOF_AMD_RENOIR \
-m SND_SOC_SOF_AMD_REMBRANDT \
-m SND_SOC_SOF_ACP_PROBES \
-m SND_SOC_SOF_INTEL_TOPLEVEL \
-m SND_SOC_SOF_INTEL_HIFI_EP_IPC \
-m SND_SOC_SOF_INTEL_ATOM_HIFI_EP \
-m SND_SOC_SOF_INTEL_COMMON \
-m SND_SOC_SOF_BAYTRAIL \
-m SND_SOC_SOF_INTEL_MTL \
-m SND_SOC_SOF_INTEL_LNL \
-m SND_SOC_SOF_LUNARLAKE \
-m SND_SOC_SOF_INTEL_PTL \
-m SND_SOC_SOF_PANTHERLAKE \
-m SND_SOC_SOF_HDA_COMMON \
-m SND_SOC_SOF_HDA_GENERIC \
-m SND_SOC_SOF_HDA_LINK_BASELINE \
-m SND_SOC_SOF_HDA_PROBES \
-m SND_SOC_SOF_INTEL_SOUNDWIRE_LINK_BASELINE \
-m SND_SOC_SOF_XTENSA \
-m SND_SOC_I2C_AND_SPI \
-m UHID \
-m HID_MULTITOUCH \
-m I2C_HID_ACPI \
-m I2C_HID_CORE \
-m USB_ROLE_SWITCH \
-m MMC \
-e MMC_CRYPTO \
-m MMC_SDHCI \
-m MMC_SDHCI_PCI \
-m MMC_RICOH_MMC \
-m MMC_CQHCI \
-e LEDS_TRIGGER_CPU \
-m HP_BIOSCFG \
-m IDEAPAD_LAPTOP \
-m THINKPAD_LMI \
-m FAT_FS \
-m VFAT_FS \
-e FAT_DEFAULT_UTF8 \
    -e CRYPTO_DES \
    -e CRYPTO_MD4 \
    -e CRYPTO_DES3_EDE_X86_64 \
    -e PKCS8_PRIVATE_KEY_PARSER


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
RUN echo "kiel:kiel" | chpasswd

COPY config/xinitrc /home/kiel/.xinitrc


RUN emerge --depclean
RUN sudo chmod a+rx /opt/clion/bin/clang/linux/x64/bin/clang-format

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
"usr/lib/llvm/*/lib64/lib*.a" \
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


