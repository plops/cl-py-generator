#!/bin/bash

dracut \
    -m " kernel-modules base rootfs-block crypt dm " \
    --filesystems " squashfs vfat overlay " \
    --kver=6.12.16-gentoo-x86_64 \
    --force \
    "/boot/initramfs_squash_crypt-6.12.16-gentoo-x86_64.img
