#!/bin/bash
# Läuft IM Container: exportiert die Build-Artefakte nach /tmp/outside/<TARGET>.

TODAY=$(date +%Y%m%d)
TARGET=gentoo-qemu-min-openrc_${TODAY}
mkdir -p /tmp/outside/${TARGET}
cp /gentoo.squashfs_qemu /tmp/outside/${TARGET}/ || true
cp /boot/vmlinuz /tmp/outside/${TARGET}/
cp /boot/initramfs_squash_sda1-x86_64.img /tmp/outside/${TARGET}/
cp /packages.txt /tmp/outside/${TARGET}/packages.txt || qlist -Iv > /tmp/outside/${TARGET}/packages.txt
cp /packages.tsv /tmp/outside/${TARGET}/packages.tsv || true
chmod -R a+rwx /tmp/outside/${TARGET}
