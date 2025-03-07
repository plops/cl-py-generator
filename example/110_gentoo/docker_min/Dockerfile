# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20250305 AS portage

# based on stage3 image (1GB)
FROM gentoo/stage3:nomultilib-systemd-20250303

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

RUN USE="experimental" emerge =sys-kernel/gentoo-sources-6.6.79
ENV KVER=6.6.79-gentoo

#ENV KVER=6.12.16-gentoo
RUN eselect kernel list
RUN eselect kernel set linux-6.6.79-gentoo
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config-6.6.79 .config
RUN make olddefconfig
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
# Replace $(shell uname -r) with ${KVER}-x86_64, otherwise the TARGET folder is not defined correctly
RUN sed -i "s/\$(shell uname -r)/${KVER}-x86_64/g" Makefile
RUN make -j7
RUN cp ryzen_smu.ko /lib/modules/${KVER}-x86_64/kernel/drivers/
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
RUN rm /tmp/password.txt





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

# Prepare the squashfs image
RUN mksquashfs /lib /bin /etc /home /dev \
/usr/lib /usr/share \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 22 \
-progress \
-mem 10G \
-no-strip \
-one-file-system-x \
-noappend \
-wildcards \
-ef /list_files.txt \
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
usr/lib/firmware/intel/ice \
usr/share/sgml/docbook \
usr/share/doc/openssl* \
usr/share/doc/docutils*/html



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
  --kver=${KVER}-x86_64 \
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
  --kver=${KVER}-x86_64 \
  --force \
  /boot/initramfs_squash_nvme0n1p5-x86_64.img

# entire build takes 43min