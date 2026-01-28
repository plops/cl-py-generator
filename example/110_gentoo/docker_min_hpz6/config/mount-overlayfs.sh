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
    # mount persistent encrypted disk that will be the upper layer of the overlayfs
    mount -t ext4 /dev/mapper/enc /run/enc
    # a previous step creates these folders. we dont want them
    rm -rf /run/overlayfs
    rm -rf /run/ovlwork
    # we have the folders persistent/upper and persistent/work on the encrypted partition
    ln -s /run/enc/persistent/upper /run/overlayfs
    ln -s /run/enc/persistent/work /run/ovlwork
    
    if ! strstr "$(cat /proc/mounts)" LiveOS_rootfs; then
        mount -t overlay LiveOS_rootfs -o "$ROOTFLAGS,$ovlfs",upperdir=/run/overlayfs,workdir=/run/ovlwork "$NEWROOT"
    fi
fi
