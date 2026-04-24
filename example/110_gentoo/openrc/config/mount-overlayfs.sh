#!/bin/sh

command -v getarg > /dev/null || . /lib/dracut-lib.sh

# Kernel args / dracut switches
getargbool 0 rd.live.overlay.overlayfs && overlayfs="yes"
getargbool 0 rd.live.overlay.readonly && readonly_overlay="--readonly" || readonly_overlay=""
ROOTFLAGS="$(getarg rootflags)"
LOWER_EXT4_DEV="$(getarg rd.live.overlay.lower.ext4dev=)"
LOWER_EXT4_OPTS="$(getarg rd.live.overlay.lower.ext4opts=)"
[ -n "${LOWER_EXT4_OPTS}" ] || LOWER_EXT4_OPTS="ro"

mount_lower_ext4() {
    [ -n "${LOWER_EXT4_DEV}" ] || return 1
    mkdir -p /run/rootfsdax
    if mount -t ext4 -o "${LOWER_EXT4_OPTS}" "${LOWER_EXT4_DEV}" /run/rootfsdax; then
        return 0
    fi
    warn "failed to mount ${LOWER_EXT4_DEV} with options '${LOWER_EXT4_OPTS}', retrying read-only without DAX"
    mount -t ext4 -o ro "${LOWER_EXT4_DEV}" /run/rootfsdax
}

if [ -n "$overlayfs" ]; then
    # Build overlay lowerdir options
    if [ -n "$readonly_overlay" ] && [ -h /run/overlayfs-r ]; then
        ovlfs=lowerdir=/run/overlayfs-r:/run/rootfsbase
    else
        ovlfs=lowerdir=/run/rootfsbase
    fi

    if mount_lower_ext4; then
        ovlfs=lowerdir=/run/rootfsdax:/run/rootfsbase
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
