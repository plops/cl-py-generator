# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20260131 AS portage

# based on stage3 image (1GB)
FROM gentoo/stage3:nomultilib-systemd-20260126

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
#RUN rm -rf /usr/share/locale/*
#RUN locale-gen

RUN emerge --update --deep --changed-use @world --fetchonly

# # Note: The kernel needs linux-firmware (for amdgpu) and  wireless-regdb for regulatory.db 

ENV KVER_PURE=6.12.58
ENV KVER=${KVER_PURE}-gentoo

RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

#ENV KVER=6.12.16-gentoo
RUN eselect kernel list
RUN eselect kernel set linux-${KVER}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
#COPY config/config-${KVER_PURE} .config
#COPY config/config-6.13.5 .config


#RUN make olddefconfig
# Disable debug info for a smaller amdgpu module (20MB instead of 400MB)
# tobogganing is the behavior where penguins slide on their bellies across ice or snow
# Maybe: -e CONFIG_RFKILL instead of -m
# --set-val BUILD_SALT "6.13.5-gentoo" 
#RUN ./scripts/config --file .config \
#-d DEBUG_INFO \
#-d DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
#-d DEBUG_INFO_DWARF4 \
#-d DEBUG_INFO_DWARF5 \
#-e DEBUG_INFO_NONE \
#--set-val DEFAULT_HOSTNAME "toboggan" \
#-e MZEN3 \
#-d GENERIC_CPU \
#-d CPU_MITIGATIONS \
#--set-val MODPROBE_PATH "/usr/sbin/modprobe" \
#-m NETFILTER_XT_NAT \
#-m NETFILTER_XT_TARGET_MASQUERADE \
#-m IP_NF_IPTABLES \
#--set-val EXTRA_FIRMWARE "rtw89/rtw8852b_fw.bin regulatory.db amd-ucode/microcode_amd_fam19h.bin" \
#--set-val EXTRA_FIRMWARE_DIR "/usr/lib/firmware" \
#-m DM_CRYPT \
#-m MII \
#-m USB_NET_DRIVERS \
#-m USB_USBNET \
#-m USB_NET_CDCETHER \
#-m USB_NET_CDC_EEM \
#-m USB_NET_CDC_NCM \
#-m USB_NET_RNDIS_HOST \
#-m USB_NET_CDC_SUBSET_ENABLE \
#-m USB_NET_CDC_SUBSET \
#-e USB_BELKIN \
#-e USB_ARMLINUX \
#-m VIRT_WIFI \
#-e POWER_RESET_RESTART \
#-e CHARGER_MANAGER \
#-e THERMAL_GOV_POWER_ALLOCATOR \
#-e DEVFREQ_THERMAL \
#-d DRM_AMDGPU_SI \
#-d DRM_AMDGPU_CIK \
#-e HID_CHICONY \
#-e HID_LOGITECH \
#-e USB_STORAGE \
#-e OVERLAY_FS \
#-e OVERLAY_FS_REDIRECT_ALWAYS_FOLLOW \
#-m ISO9660_FS \
#-e SQUASHFS \
#-d SQUASHFS_FILE_CACHE \
#-e SQUASHFS_FILE_DIRECT \
#-e SQUASHFS_DECOMP_MULTI_PERCPU \
#-d SQUASHFS_CHOICE_DECOMP_BY_MOUNT \
#-d SQUASHFS_COMPILE_DECOMP_SINGLE \
#-d SQUASHFS_COMPILE_DECOMP_MULTI \
#-e SQUASHFS_COMPILE_DECOMP_MULTI_PERCPU \
#-e SQUASHFS_XATTR \
#-e SQUASHFS_ZLIB \
#-d SQUASHFS_LZ4 \
#-d SQUASHFS_LZO \
#-e SQUASHFS_XZ \
#-e SQUASHFS_ZSTD \
#-d SQUASHFS_4K_DEVBLK_SIZE \
#-d SQUASHFS_EMBEDDED \
#-m MMC_BLOCK \
#-m CRYPTO_USER_API_SKCIPHER \
#-m CRYPTO_ECB \
#-m CRYPTO_CBC \
#-m CRYPTO_DES \
#-m CRYPTO_DES3_EDE_X86_64 \
#--set-val SQUASHFS_FRAGMENT_CACHE_SIZE 3 \
#-m ENCRYPTED_KEYS \
#--set-val SECURITY_TOMOYO_POLICY_LOADER "/usr/bin/tomoyo-init"
# crypto_des is needed for iwd

# https://wiki.gentoo.org/wiki/Hetzner_Cloud_(ARM64)

RUN make defconfig
RUN make kvm_guest.config
RUN for i in \
	DRM_NOUVEAU \
	DRM_EXYNOS \
	DRM_ROCKCHIP \
	DRM_RCAR_DU \
	DRM_RCAR_DW_HDMI \
	DRM_RCAR_USE_LVDS \
	DRM_RCAR_USE_MIPI_DSI \
	DRM_IMX_DCSS \
	DRM_ETNAVIV \
	DRM_HISI_HIBMC \
	DRM_HISI_KIRIN \
	DRM_MEDIATEK \
	DRM_MSM \
	DRM_MXSFB \
	DRM_MESON \
	DRM_PL111 \
	DRM_TIDSS \
	DRM_LEGACY \
	DRM_SUN4I \
	DRM_TEGRA \
	TEGRA_HOST1X \
	SCSI_UFSHCD \
	FPGA \
	RC_CORE \
	NEW_LEDS \
	CHROME_PLATFORMS \
	SURFACE_PLATFORMS \
	XEN_BLKDEV_FRONTEND \
	LOGO \
	SOUND \
	SLIMBUS \
	SOUNDWIRE \
	MEDIA_SUPPORT \
	MMC \
	NFS_FS \
	9P_FS \
	SUSPEND \
	HIBERNATION \
	BLK_DEV_INITRD \
	VIRTUALIZATION \
	WLAN \
	PINCTRL \
	GPIOLIB \
	PWM \
	IPMI_HANDLER \
	CAN \
	BT \
	WIRELESS \
	MD \
	RFKILL \
	NET_9P \
	NFC \
	SPI \
	SPMI \
	HWMON \
	THERMAL \
	IIO \
	USB_NET_DRIVERS \
	XEN_NETDEV_FRONTEND \
	ETHERNET \
	QCOM_IPA \
	REGULATOR \
	STAGING \
	DEBUG_KERNEL \
	XEN \
	MODULES; do ./scripts/config --disable $i; done && \
    ./scripts/config --set-str CMDLINE "init=/usr/lib/systemd/systemd root=/dev/sda3 rootwait rootfstype=ext4"
RUN make prepare



RUN emerge -e @world --fetchonly
RUN emerge -e @world
RUN emerge --depclean

RUN make -j32
RUN make modules_install
RUN make install




# # Remove rust, it needs a lot of space
# RUN emerge --unmerge virtual/rust dev-lang/rust-bin
# RUN emerge --depclean

# # Allow members of the wheel group to execute any command without a password
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel


# # Add a tool to read the Ryzen temperatures, frequencies and voltages
WORKDIR /usr/src
RUN git clone https://github.com/leogx9r/ryzen_smu
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

# # Create a user 
RUN useradd -m -G wheel -s /bin/bash martin
# # Create a random passphrase of 3 words, set as the password and print to console
# RUN echo "" >> /etc/portage/package.accept_keywords/package.accept_keywords
# RUN echo "app-admin/xkcdpass ~amd64" >> /etc/portage/package.accept_keywords/package.accept_keywords
# RUN emerge app-admin/xkcdpass
# RUN xkcdpass -n 3 -d '.' > /tmp/password.txt \
# && echo "Password: $(cat /tmp/password.txt)" \
# && echo "martin:$(cat /tmp/password.txt)" | chpasswd

RUN GENTOO_MIRRORS="" emerge dev-tcltk/expect
COPY config/simple_password.expect /simple_password.expect
RUN expect /simple_password.expect
RUN rm /simple_password.expect

#RUN systemctl enable tlp
#RUN systemctl enable bluetooth

COPY config/list_files.py /usr/bin/list_files.py
RUN chmod a+x /usr/bin/list_files.py
RUN python /usr/bin/list_files.py > /list_files.txt

# RUN USE="static" emerge app-shells/dash


COPY config/list_files.py /usr/bin/list_files.py
RUN chmod a+x /usr/bin/list_files.py
RUN python /usr/bin/list_files.py > /list_files.txt

# Add additional programs to world
# COPY config/world.2 /
# COPY config/package.use.2 /
# RUN cat /world.2 >> /var/lib/portage/world
# RUN cat /package.use.2 >> /etc/portage/package.use/package.use
# RUN emerge \
# app-editors/emacs \
# app-emacs/magit \
# app-emacs/paredit \
# app-misc/tmux \
# dev-lisp/sbcl \
# dev-util/tkdiff \
# net-analyzer/netcat \
# dev-libs/capnproto \
# app-text/mupdf \
# x11-terms/zutty \
# sys-fs/cryptsetup \
# sys-fs/lvm2 \
# media-libs/glfw \
# media-sound/pulseaudio \
# dev-cpp/popl \
# dev-db/sqlite \
# dev-libs/spdlog

# # jumbo-build combines files to compile faster but requires more memory

# RUN echo "" >> /etc/portage/package.use/package.use
# RUN cat <<EOF >> /etc/portage/package.use/package.use
# dev-libs/glib dbus elf mime xattr -debug -doc introspection -selinux -static-libs -sysprof -systemtap -test -utils
# x11-libs/gdk-pixbuf -gif -gtk-doc introspection -jpeg -test -tif
# www-client/firefox X -clang -gmp-autoupdate jumbo-build system-av1 system-harfbuzz system-icu system-jpeg system-libevent system-libvpx system-webp -telemetry -dbus -debug -eme-free -gnome-shell -hardened -hwaccel -jack -libproxy -openh264 -pgo pulseaudio -selinux -sndio -system-png -wasm-sandbox -wayland -wifi
# media-video/ffmpeg X dav1d encode gpl libdrm opus pulseaudio static-libs threads vaapi vulkan x264 -alsa -amf -amr -amrenc -appkit -bluray -bs2b -bzip2 -cdio -chromaprint -chromium -codec2 -cpudetection (-cuda) -debug -doc -fdk -flite -fontconfig -frei0r -fribidi -gcrypt -glslang -gme -gmp -gnutls -gsm -hardcoded-tables -iconv -iec61883 -ieee1394 -jack -jpeg2k -jpegxl -kvazaar -ladspa -lcms -libaom -libaribb24 -libass -libcaca -libilbc -libplacebo -librtmp -libsoxr -libtesseract -libv4l -libxml2 -lv2 -lzma (-mipsdspr1) (-mipsdspr2) (-mipsfpu) -modplug -mp3 -network -nvenc -openal -opencl -opengl -openh264 -openssl -oss -pic -postproc (-qsv) -rav1e -rubberband -samba -sdl -shaderc -snappy -sndio -soc -speex -srt -ssh -svg -svt-av1 -test -theora -truetype -twolame -v4l -vdpau -verify-sig -vidstab -vmaf -vorbis -vpx -webp -x265 -xvid -zeromq -zimg zlib -zvbi
# media-libs/libvpx highbitdepth threads -doc postproc -static-libs -test
# gnome-base/librsvg vala -debug -gtk-doc introspection
# EOF
# RUN emerge -v www-client/firefox

# # 622MB compressed, 2.1GB uncompressed, 7min for compilation of world.2

# # 32 minutes for the firefox build
# # 847MB compressed, 2.9GB uncompressed

# RUN emerge --unmerge virtual/rust dev-lang/rust-bin
# RUN emerge --depclean

# RUN emerge \
#  net-wireless/iwgtk

# #net-wireless/blueman # need gtk 3.24

#  #83 37.29 Filesystem size 876395.57 Kbytes (855.86 Mbytes)
# #83 37.29       29.95% of uncompressed filesystem size (2926143.56 Kbytes)

# RUN eclean-dist
# RUN eclean-pkg


# # COPY config/world.3 /
# # COPY config/package.use.3 /
# # COPY config/package.license.3 /
# # COPY config/package.accept_keywords.3 /
# # RUN cat /world.3 >> /var/lib/portage/world
# # RUN cat /package.use.3 >> /etc/portage/package.use/package.use
# # RUN cat /package.license.3 >> /etc/portage/package.license
# # RUN cat /package.accept_keywords.3 >> /etc/portage/package.accept_keywords/package.accept_keywords
# # #RUN emerge --update --deep --changed-use @world --fetchonly
# # #RUN emerge --update --deep --changed-use @world
# # #RUN emerge -v www-client/chromium

# RUN echo "" >> /etc/portage/package.accept_keywords/package.accept_keywords
# RUN echo "dev-util/clion ~amd64" >> /etc/portage/package.accept_keywords/package.accept_keywords
# RUN echo "dev-util/clion IDEA" >> /etc/portage/package.license 
# RUN emerge dev-util/clion

# #89 87.70 Filesystem size 2149035.79 Kbytes (2098.67 Mbytes)
# #89 87.70       31.35% of uncompressed filesystem size (6855828.81 Kbytes)
# # 5min24s to install clion (and recompress)

# RUN emerge  dev-debug/gdb
# RUN emerge dev-ada/gprbuild 

# #90 86.38 Filesystem size 2154112.32 Kbytes (2103.63 Mbytes)
# #90 86.38       31.36% of uncompressed filesystem size (6868545.41 Kbytes)
# # 2min35s to install gdb (and recompress)

# # RUN echo "" >> /etc/portage/package.use/package.use
# # RUN cat <<EOF >> /etc/portage/package.use/package.use
# # net-libs/nodejs inspector
# # sys-libs/zlib minizip
# # media-libs/harfbuzz icu
# # dev-libs/libxml2 icu
# # www-client/chromium X -hangouts official -proprietary-codecs -screencast system-harfbuzz system-png system-zstd -wayland -widevine -bindist -bundled-toolchain -cups -custom-cflags -debug -ffmpeg-chromium gtk4 -headless -kerberos -pax-kernel -pgo -pulseaudio -qt6 -selinux -system-icu -test -vaapi
# # EOF
# # RUN emerge www-client/chromium --jobs=4

# #87 5223.1  * ERROR: www-client/chromium-135.0.7049.52::gentoo failed (compile phase):
# #87 5223.1  *   ninja -v -j33 -l0 -C out/Release chrome chromedriver chrome_sandbox failed
# #87 5223.1  *
# #87 5223.1  * Call stack:
# #87 5223.1  *     ebuild.sh, line  136:  Called src_compile
# #87 5223.1  *   environment, line 4489:  Called chromium_compile
# #87 5223.1  *   environment, line 1745:  Called eninja '-C' 'out/Release' 'chrome' 'chromedriver' 'chrome_sandbox'
# #87 5223.1  *   environment, line 2118:  Called die
# #87 5223.1  * The specific snippet of code:
# #87 5223.1  *       "$@" || die -n "${*} failed"
# #87 ERROR: process "/bin/sh -c emerge www-client/chromium" did not complete successfully: exit code: 1
# # failed after real    87m4.507s

# # #-e 'usr/lib/firmware/amdgpu/(?!green_sardine.*\.bin$)' 

# # # The full emerge of all the updated packet took 2224 seconds
# # # The kernel build took 194 seconds
# # #'usr/lib/firmware/!({regulator.db*,amdgpu,rtl_bt,rtw89})' 

# For my laptop: -march=znver3 
# -one-file-system-x 
# Prepare the squashfs image
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



# # intel/vsc 11MB   Visual Sensing Controller
# # This hardware is found with modern Intel Core laptops 
# # and tied into the web camera can be used for proximity sensors and ultimately 
# # enabling functionality like locking the screen when walking away from the computer
# # intel/ice 4.7MB Intel Ethernet Connection E800 Series

# # Create a new initramfs with cryptsetup
# # It will mount the encrypted partition /dev/nvme0n1p4,
# # mount gentoo.squashfs into a overlayfs and pivot_root into it
# COPY config/init_dracut_crypt.sh /usr/lib/dracut/modules.d/99base/init.sh
# RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh
# RUN dracut \
#   -m " kernel-modules base rootfs-block crypt dm " \
#   --filesystems " squashfs vfat overlay " \
#   --kver=${KVER} \
#   --force \
#   /boot/initramfs_squash_crypt-x86_64.img


# # Configure grub
# # This will load kernel and initramfs from nvme0n1p1 (EFI partition with UUID F63D-5318)
# # It the initramfs (which calls config/init_dracut_crypt.sh) will mount the encrypted partition /dev/nvme0n1p4
# # create overlay with squashfs image and pivot_root into it
# COPY config/40_custom /etc/grub.d/40_custom

# # Create an initramfs that loads the squashfs from nvme0n1p5 and overlays it 
# # on the folder persistent/ on that partition

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


# entire build takes 40min16sec


# we need the kernel before the full build
