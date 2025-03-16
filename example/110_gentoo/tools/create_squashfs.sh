#!/bin/bash

python list_files.py > /dev/shm/list_files.txt

mksquashfs /lib /sbin /bin /etc /home /dev \
/proc /sys /run /var /tmp /mnt \
/usr  \
/gentoo.squashfs \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 1 \
-progress \
-mem 10G \
-no-strip \
-noappend \
-wildcards \
-ef /dev/shm/list_files.txt \
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
	   tmp/password.txt \
	   usr/lib64/libQt*.a
