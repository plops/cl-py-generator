# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# name the portage image
FROM gentoo/portage:20250911 AS portage

# based on stage3 image (1GB)
# https://hub.docker.com/r/gentoo/stage3/tags
FROM gentoo/stage3:nomultilib-systemd-20250908
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

ENV KVER_PURE=6.12.41
ENV KVER=${KVER_PURE}-gentoo

RUN emerge =sys-kernel/gentoo-sources-${KVER_PURE}

RUN eselect kernel list
RUN eselect kernel set linux-${KVER}
#RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config6.12.31 .config
RUN make oldconfig

# RUN ./scripts/config --file .config \
# --set-val MODPROBE_PATH "/usr/sbin/modprobe" \
# --set-val EXTRA_FIRMWARE "amd-ucode/microcode_amd_fam19h.bin rtw89/rtw8852b_fw.bin regulatory.db" \
# --set-val EXTRA_FIRMWARE_DIR "/usr/lib/firmware" \
# -e SQUASHFS 

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
RUN echo "kiel:kiel" | chpasswd

COPY config/xinitrc /home/kiel/.xinitrc


RUN emerge --depclean
RUN sudo chmod a+rx /opt/clion/bin/clang/linux/x64/bin/clang-format

#RUN IGNORE_PREEMPT_RT_PRESENCE=1 emerge nvidia-drivers



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


