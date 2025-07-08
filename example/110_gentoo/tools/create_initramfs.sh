#!/bin/bash

cp init_dracut_crypt.sh /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh


dracut \
    -m " kernel-modules base rootfs-block crypt dm " \
    --filesystems " squashfs vfat overlay " \
    --kver=6.12.31-gentoo-x86_64 \
    --force \
    "/boot/initramfs_squash_crypt-6.12.31-gentoo-x86_64.img"
