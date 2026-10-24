#!/bin/bash
# Kopiert QEMU-Artefakte (Squashfs, Kernel, Initramfs, Paketbericht) aus dem Container.
TODAY=$(date +%Y%m%d)
TARGET=gentoo-qemu-min-openrc_${TODAY}
mkdir -p /dev/shm/${TARGET}
cp copy_files.sh /dev/shm/${TARGET}
docker run \
--rm \
--privileged \
-v /dev/shm:/tmp/outside \
gentoo-qemu-min-openrc \
/bin/bash /tmp/outside/${TARGET}/copy_files.sh
