#!/bin/sh

command -v getarg > /dev/null || . /lib/dracut-lib.sh

getargbool 0 rd.live.overlay.overlayfs && overlayfs="yes"
getargbool 0 rd.live.overlay.readonly && readonly_overlay="--readonly" || readonly_overlay=""

ROOTFLAGS="$(getarg rootflags)"

if [ -n "$overlayfs" ]; then
    if [ -n "$readonly_overlay" ] && [ -h /run/overlayfs-r ]; then
        ovlfs=lowerdir=/run/overlayfs-r:/run/rootfsbase
    else
        ovlfs=lowerdir=/run/rootfsbase
    fi

    mkdir -p /run/enc
    mount -t ext4 /dev/mapper/enc /run/enc
    rm -rf /run/overlayfs
    rm -rf /run/ovlwork
    ln -s /run/enc/overlayfs /run/overlayfs
    ln -s /run/enc/ovlwork /run/ovlwork
    
    if ! strstr "$(cat /proc/mounts)" LiveOS_rootfs; then
        mount -t overlay LiveOS_rootfs -o "$ROOTFLAGS,$ovlfs",upperdir=/run/overlayfs,workdir=/run/ovlwork "$NEWROOT"
    fi
fi
