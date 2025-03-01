# syntax=docker/dockerfile:1  # REQUIRED for build-time mounts

# sudo emerge -av app-containers/docker-buildx
# DOCKER_BUILDKIT=1 docker build -t gentoo-ideapad  .
# name the portage image
FROM gentoo/portage:latest AS portage

# based on stage3 image (1GB)
FROM gentoo/stage3:nomultilib-systemd-20250217
# gcc 14.2.1

# copy the entire portage volume in (570MB)
COPY --from=portage /var/db/repos/gentoo /var/db/repos/gentoo
COPY package.use /etc/portage/package.use
COPY make.conf /etc/portage/make.conf
#COPY update_package_use.sh /usr/local/bin/
#RUN chmod +x /usr/local/bin/update_package_use.sh
RUN eselect profile set default/linux/amd64/23.0/no-multilib/systemd

# RUN emerge -v --update --deep --newuse  @world --fetchonly
RUN emerge -e @world --fetchonly
#RUN emerge gentoolkit
# --quiet-build=y -a
#RUN emerge -v --update --deep --newuse  @world
RUN emerge -e @world

COPY config/package.license /etc/portage/package.license
RUN emerge \
sys-kernel/gentoo-sources \
sys-kernel/linux-firmware

# The kernel needs firmware blobs (for amdgpu) and  regulatory.db 
RUN emerge \
net-wireless/wireless-regdb

ENV KVER=6.6.74-gentoo
RUN eselect kernel set 1
WORKDIR /usr/src/linux
COPY config/config-6.6.74 .config
RUN make oldconfig
RUN make -j32
RUN make modules_install
RUN make install

# The kernel compilation needs firmware blobs

WORKDIR /
COPY config/dwm-6.5 /etc/portage/savedconfig/x11-wm/dwm-6.5
RUN emerge \
sys-fs/squashfs-tools \
sys-kernel/dracut \
emerge sys-fs/cryptsetup \
app-editors/emacs \
x11-base/xorg-server \
net-wireless/bluez \
net-wireless/iwd \
net-misc/dhcp \
dev-vcs/git \
media-plugins/alsa-plugins \
media-libs/libaom \
media-sound/pulseaudio \
net-wireless/blueman \
x11-wm/dwm \
dev-libs/glib \
dev-libs/dbus-glib \
app-text/poppler \
app-text/mupdf \
media-video/mpv \
x11-terms/xterm \
www-client/firefox-bin

RUN echo "net-wireless/iwgtk ~amd64" >> /etc/portage/package.accept_keywords/package.accept_keywords
RUN emerge \
dev-lisp/sbcl \
net-wireless/iwgtk \
app-emacs/magit \
app-emacs/paredit \
x11-misc/xclip

# Remove rust, it needs a lot of space
RUN emerge --unmerge virtual/rust dev-lang/rust-bin
RUN emerge --unmerge dev-lang/python:3.13
RUN emerge --depclean

RUN echo "C.UTF-8 UTF-8" >> /etc/locale.gen
RUN rm -rf /usr/share/locale/*
RUN locale-gen

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
usr/lib/firmwar/{iwlwifi,phanfw}* \
persistent \
var/tmp/portage/* 

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



#RUN <<-EOF
#    set -e
#
#    # configure portage
#    echo '*/* ~amd64' >> /etc/portage/package.accept_keywords/base.conf
#    echo 'dev-lang/python **' >> /etc/portage/package.accept_keywords/python.conf
#    echo '*/* full-stdlib sqlite' >> /etc/portage/package.use/python.conf
#    echo 'dev-vcs/git -perl' >> /etc/portage/package.use/git.conf
#
#    # install ::gentoo
#    wget --progress=dot:mega -O - https://github.com/gentoo-mirror/gentoo/archive/master.tar.gz | tar -xz
#    mv gentoo-master /var/db/repos/gentoo
#
#    # main job
#    emerge -1vnt --jobs dev-python/tox app-arch/lzip dev-vcs/git \
#        dev-python/pypy{,3_10}-exe-bin dev-db/sqlite dev-libs/mpdecimal
#    emerge -1v --jobs --nodeps dev-lang/python:{2.7,3.8,3.9,3.10,3.11,3.12,{3.13,3.14}{,t}} \
#        dev-lang/pypy:{2.7,3.10}
#
#    # cleanup
#    rm -r /var/db/repos/* /var/cache/distfiles/*
#EOF


# docker run -it gentoo-ideapad