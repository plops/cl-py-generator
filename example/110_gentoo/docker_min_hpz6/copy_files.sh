#!/bin/bash

# This script copies the resulting gentoo files (both original and E14 squashfs images, plus kernel) to the host
TODAY=$(date +%Y%m%d)
TARGET=gentoo-z6-min_${TODAY}
mkdir -p /tmp/outside/${TARGET}
cp /gentoo.squashfs /tmp/outside/${TARGET}/
cp /gentoo.squashfs_e14 /tmp/outside/${TARGET}/
cp /boot/vmlinuz /tmp/outside/${TARGET}/
cp /boot/initramfs_squash_sda1-x86_64.img /tmp/outside/${TARGET}/
qlist -Iv > /tmp/outside/${TARGET}/packages.txt
chmod -R a+rwx /tmp/outside/${TARGET}
