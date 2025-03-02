# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:latest AS portage

# based on stage3 image (1GB)
FROM gentoo/stage3:nomultilib-systemd-20250224

# gcc 14.2.1

# copy the entire portage volume in (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo

# Configure the system
RUN eselect profile set default/linux/amd64/23.0/no-multilib/systemd
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

RUN emerge --unmerge dev-lang/python:3.13
# We need rust for rsvg (and it takes a lot time to compile, so we use the)
RUN emerge --oneshot dev-lang/rust-bin

# RUN emerge -e @world --fetchonly

# Build 674 packages in 5840 seconds
RUN emerge -e @world
RUN emerge --depclean

# Note: The kernel needs linux-firmware (for amdgpu) and  wireless-regdb for regulatory.db 

ENV KVER=6.6.74-gentoo
RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config-6.6.74 .config
RUN make oldconfig
RUN make -j32
RUN make modules_install
RUN make install

# Remove rust, it needs a lot of space
RUN emerge --unmerge virtual/rust dev-lang/rust-bin
RUN emerge --depclean

# Allow members of the wheel group to execute any command without a password
RUN mkdir -p /etc/sudoers.d
RUN echo "%wheel ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/wheel

# Install some qt5 libraries with documentation
RUN echo "dev-qt/qt-docs:5 3d assistant bluetooth concurrent dbus -declarative -designer -gamepad graphicaleffects gui help html -imageformats -linguist -location multimedia network -opengl -positioning -printsupport -qch -qdoc -quickcontrols -quickcontrols2 -scxml -sensors -serialbus -serialport -speech sql svg test -wayland -webchannel -websockets -webview widgets x11extras -xml -xmlpatterns" >> /etc/portage/package.use/qt-docs
RUN emerge \
dev-qt/qtcore:5 \
dev-qt/qtgui:5 \
dev-qt/qtmultimedia:5 \
dev-qt/qtnetwork:5 \
dev-qt/qtsql:5 \
dev-qt/qttest:5 \
dev-qt/qtwidgets:5 \
dev-qt/qtcharts:5 \
dev-qt/qtconcurrent:5 \
dev-qt/qtx11extras:5 \
dev-qt/qt-docs:5 \
sys-apps/qdirstat



# Add a tool to read the Ryzen temperatures, frequencies and voltages
WORKDIR /usr/src
RUN git clone https://github.com/leogx9r/ryzen_smu
WORKDIR /usr/src/ryzen_smu
# Replace $(shell uname -r) with ${KVER}-x86_64, otherwise the TARGET folder is not defined correctly
RUN sed -i "s/\$(shell uname -r)/${KVER}-x86_64/g" Makefile
RUN make -j7
RUN cp ryzen_smu.ko /lib/modules/${KVER}-x86_64/kernel/drivers/
RUN rm *.o


# Add slstatus to show battery status and time
WORKDIR /usr/src
RUN git clone https://git.suckless.org/slstatus
WORKDIR /usr/src/slstatus
COPY config/slstatus_config.h .
RUN make -j32
RUN make install
RUN make clean

# Create a user 
RUN useradd -m -G wheel -s /bin/bash martin
# Create a random passphrase of 3 words, set as the password and print to console
RUN echo "" >> /etc/portage/package.accept_keywords/package.accept_keywords
RUN echo "app-admin/xkcdpass ~amd64" >> /etc/portage/package.accept_keywords/package.accept_keywords
RUN emerge app-admin/xkcdpass
RUN xkcdpass -n 3 -d '.' > /tmp/password.txt \
&& echo "Password: $(cat /tmp/password.txt)" \
&& echo "martin:$(cat /tmp/password.txt)" | chpasswd
RUN rm /tmp/password.txt





RUN systemctl enable tlp

RUN emerge net-wireless/iw

# Prepare the squashfs image
RUN mksquashfs / /gentoo.squashfs \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 22 \
-progress \
-mem 10G \
-wildcards \
-e \
lib/modules/6.6.52-gentoo-x86_64 \
lib/modules/6.6.58-gentoo-r1-x86_64 \
usr/lib/modules/6.6.52-gentoo-x86_64 \
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
usr/lib/firmware/intel/ice

# intel/vsc 11MB   Visual Sensing Controller
# This hardware is found with modern Intel Core laptops 
# and tied into the web camera can be used for proximity sensors and ultimately 
# enabling functionality like locking the screen when walking away from the computer
# intel/ice 4.7MB Intel Ethernet Connection E800 Series

# Create a new initramfs with cryptsetup
# It will mount the encrypted partition /dev/nvme0n1p4,
# mount gentoo.squashfs into a overlayfs and pivot_root into it
COPY config/init_dracut_crypt.sh /usr/lib/dracut/modules.d/99base/init.sh
RUN chmod a+x /usr/lib/dracut/modules.d/99base/init.sh
RUN dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=${KVER}-x86_64 \
  --force \
  /boot/initramfs_squash_crypt-x86_64.img


# Configure grub
# This will load kernel and initramfs from nvme0n1p1 (EFI partition with UUID F63D-5318)
# It the initramfs (which calls config/init_dracut_crypt.sh) will mount the encrypted partition /dev/nvme0n1p4
# create overlay with squashfs image and pivot_root into it
COPY config/40_custom /etc/grub.d/40_custom


  # Runtime
  # umerge 8s
  # rust-bin 32s
  # world 5830s
  # depclean 12s
  # kernel 180s
  # umerge, deplcean 15s
  # squashfs 60s
  # dracut 4s
  # export 15s

