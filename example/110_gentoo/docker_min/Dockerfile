# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20250305 AS portage

# based on stage3 image (1GB)
FROM gentoo/stage3:nomultilib-systemd-20250310

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
RUN emerge --update --deep --changed-use @world

# 310 packages

RUN emerge --depclean

# # Note: The kernel needs linux-firmware (for amdgpu) and  wireless-regdb for regulatory.db 

ENV KVER_PURE=6.13.5
ENV KVER=${KVER_PURE}-gentoo

RUN echo "sys-kernel/gentoo-sources ~amd64" >> /etc/portage/package.accept_keywords/gentoo-sources
RUN USE="experimental" emerge =sys-kernel/gentoo-sources-${KVER_PURE}

#ENV KVER=6.12.16-gentoo
RUN eselect kernel list
RUN eselect kernel set linux-${KVER}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config-${KVER_PURE} .config


RUN make olddefconfig
# Disable debug info for a smaller amdgpu module (20MB instead of 400MB)
# tobogganing is the behavior where penguins slide on their bellies across ice or snow
# Maybe: -e CONFIG_RFKILL instead of -m
# --set-val BUILD_SALT "6.13.5-gentoo" 
RUN ./scripts/config --file .config \
-d DEBUG_INFO \
-d DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
-d DEBUG_INFO_DWARF4 \
-d DEBUG_INFO_DWARF5 \
-e DEBUG_INFO_NONE \
--set-val DEFAULT_HOSTNAME "toboggan" \
-e MZEN3 \
-d GENERIC_CPU \
-d CPU_MITIGATIONS \
--set-val MODPROBE_PATH "/usr/sbin/modprobe" \
-m NETFILTER_XT_NAT \
-m NETFILTER_XT_TARGET_MASQUERADE \
-m IP_NF_IPTABLES \
--set-val EXTRA_FIRMWARE "rtw89/rtw8852b_fw.bin regulatory.db amd-ucode/microcode_amd_fam19h.bin" \
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
-m VIRT_WIFI \
-e POWER_RESET_RESTART \
-e CHARGER_MANAGER \
-e THERMAL_GOV_POWER_ALLOCATOR \
-e DEVFREQ_THERMAL \
-d DRM_AMDGPU_SI \
-d DRM_AMDGPU_CIK \
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
--set-val SECURITY_TOMOYO_POLICY_LOADER "/usr/bin/tomoyo-init"

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
# Create a random passphrase of 3 words, set as the password and print to console
RUN echo "" >> /etc/portage/package.accept_keywords/package.accept_keywords
RUN echo "app-admin/xkcdpass ~amd64" >> /etc/portage/package.accept_keywords/package.accept_keywords
RUN emerge app-admin/xkcdpass
RUN xkcdpass -n 3 -d '.' > /tmp/password.txt \
&& echo "Password: $(cat /tmp/password.txt)" \
&& echo "martin:$(cat /tmp/password.txt)" | chpasswd

RUN systemctl enable tlp
RUN systemctl enable bluetooth

COPY config/list_files.py /usr/bin/list_files.py
RUN chmod a+x /usr/bin/list_files.py
RUN python /usr/bin/list_files.py > /list_files.txt

# #-e 'usr/lib/firmware/amdgpu/(?!green_sardine.*\.bin$)' 

# # The full emerge of all the updated packet took 2224 seconds
# # The kernel build took 194 seconds
# #'usr/lib/firmware/!({regulator.db*,amdgpu,rtl_bt,rtw89})' 

# For my laptop: -march=znver3 
# -one-file-system-x 
# Prepare the squashfs image
RUN mksquashfs /lib /sbin /bin /etc /home /dev \
/proc /sys /run /var /tmp /mnt \
/usr  \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 22 \
-progress \
-mem 10G \
-no-strip \
-noappend \
-wildcards \
-ef /list_files.txt \
-e \
lib/modules/6.6.52-gentoo-x86_64 \
lib/modules/6.6.74-gentoo-x86_64 \
lib/modules/6.6.58-gentoo-r1-x86_64 \
usr/lib/modules/6.6.52-gentoo-x86_64 \
usr/lib/modules/6.6.74-gentoo-x86_64 \
usr/lib/modules/6.6.58-gentoo-r1-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
usr/share/locale/* \
usr/share/gtk-doc/* \
opt/rust-bin* \
boot/* \
proc \
sys/* \
run/* \
dev/pts/* \
dev/shm/* \
dev/hugepages/* \
dev/mqueue/* \
home/martin/.cache/mozilla \
home/martin/.cache/google-chrome \
home/martin/.cache/mesa_shader_cache \
home/martin/.cache/fontconfig \
home/martin/Downloads/* \
home/martin/.config/* \
home/martin/.mozilla/* \
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt2/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/iwlwifi* \
usr/lib/firmware/intel/ipu \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,ath11k,ath10k,ath12k,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia,i915,qca,cirrus} \
usr/lib/firmware/{iwlwifi,phanfw}* \
persistent \
var/tmp/portage/* \
usr/lib/llvm/19/bin/llvm-exegesis \
usr/lib/grub/i386-pc \
usr/lib/firmware/intel/vsc \
usr/lib/firmware/intel/ice \
usr/share/sgml/docbook \
usr/share/doc/openssl* \
usr/share/doc/docutils*/html \
tmp/password.txt



# # intel/vsc 11MB   Visual Sensing Controller
# # This hardware is found with modern Intel Core laptops 
# # and tied into the web camera can be used for proximity sensors and ultimately 
# # enabling functionality like locking the screen when walking away from the computer
# # intel/ice 4.7MB Intel Ethernet Connection E800 Series

# # Create a new initramfs with cryptsetup
# # It will mount the encrypted partition /dev/nvme0n1p4,
# # mount gentoo.squashfs into a overlayfs and pivot_root into it
COPY config/init_dracut_crypt.sh /usr/lib/dracut/modules.d/99base/init.sh
RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh
RUN dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=${KVER} \
  --force \
  /boot/initramfs_squash_crypt-x86_64.img


# # Configure grub
# # This will load kernel and initramfs from nvme0n1p1 (EFI partition with UUID F63D-5318)
# # It the initramfs (which calls config/init_dracut_crypt.sh) will mount the encrypted partition /dev/nvme0n1p4
# # create overlay with squashfs image and pivot_root into it
# COPY config/40_custom /etc/grub.d/40_custom

# # Create an initramfs that loads the squashfs from nvme0n1p5 and overlays it 
# # on the folder persistent/ on that partition
COPY config/init_dracut.sh /usr/lib/dracut/modules.d/99base/init.sh
RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh
RUN dracut \
  -m " kernel-modules base rootfs-block dm " \
  --filesystems " squashfs vfat overlay btrfs " \
  --kver=${KVER} \
  --force \
  /boot/initramfs_squash_nvme0n1p5-x86_64.img

RUN rm /tmp/password.txt


# entire build takes 40min16sec


# we need the kernel before the full build
