#!/bin/sh

command -v getarg > /dev/null || . /lib/dracut-lib.sh

# Kernel args / dracut switches
getargbool 0 rd.live.overlay.overlayfs && overlayfs="yes"
getargbool 0 rd.live.overlay.readonly && readonly_overlay="--readonly" || readonly_overlay=""
ROOTFLAGS="$(getarg rootflags)"

if [ -n "$overlayfs" ]; then
    # Build overlay lowerdir options
    if [ -n "$readonly_overlay" ] && [ -h /run/overlayfs-r ]; then
        ovlfs=lowerdir=/run/overlayfs-r:/run/rootfsbase
    else
        ovlfs=lowerdir=/run/rootfsbase
    fi

    # Persistent encrypted upper/work layers
    mkdir -p /run/enc
    mount -t ext4 /dev/mapper/enc /run/enc

    # Remove transient dirs from previous initramfs step.
    rm -rf /run/overlayfs
    rm -rf /run/ovlwork

    # Reuse persistent upper/work dirs from encrypted disk.
    ln -s /run/enc/persistent/upper /run/overlayfs
    ln -s /run/enc/persistent/work /run/ovlwork

    # Mount overlay root once.
    if ! strstr "$(cat /proc/mounts)" LiveOS_rootfs; then
        mount -t overlay LiveOS_rootfs -o "$ROOTFLAGS,$ovlfs",upperdir=/run/overlayfs,workdir=/run/ovlwork "$NEWROOT"
    fi
fi
