#!/bin/bash

# This script copies the resulting gentoo files (squashfs image and kernel) to the host
TODAY=$(date +%Y%m%d)
TARGET=gentoo-ideapad_${TODAY}
mkdir -p /tmp/outside/${TARGET}
cp /gentoo.squashfs /tmp/outside/${TARGET}/
cp /boot/vmlinuz /tmp/outside/${TARGET}/
cp /boot/initramfs_squash_crypt-x86_64.img /tmp/outside/${TARGET}/
cp /boot/initramfs_squash_nvme0n1p5-x86_64.img /tmp/outside/${TARGET}/