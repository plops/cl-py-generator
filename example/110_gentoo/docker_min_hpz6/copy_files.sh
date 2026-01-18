#!/bin/bash

# This script copies the resulting gentoo files (squashfs image and kernel) to the host
TODAY=$(date +%Y%m%d)
TARGET=gentoo-z6-min_${TODAY}
mkdir -p /tmp/outside/${TARGET}
cp /gentoo.squashfs /tmp/outside/${TARGET}/
cp /boot/vmlinuz /tmp/outside/${TARGET}/
#cp /boot/initramfs_squash_crypt-x86_64.img /tmp/outside/${TARGET}/
cp /boot/initramfs_squash_sda1-x86_64.img /tmp/outside/${TARGET}/
cp /boot/initramfs_squash_from_disk.img /tmp/outside/${TARGET}/
#cp /boot/initramfs-with-ssh.img /tmp/outside/${TARGET}/
qlist -Iv > /tmp/outside/packages.txt
