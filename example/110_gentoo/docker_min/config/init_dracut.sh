#!/bin/sh


# Define a simple emergency shell
emergency_shell() {
    echo "Entering emergency shell..."
    /bin/sh
}

# Define NEWROOT (this MUST be done before switch_root)
NEWROOT=/sysroot


# mount some important things
if [ ! -d /proc/self ]; then
    if ! mount -t proc -o nosuid,noexec,nodev proc /proc > /dev/null; then
        echo "Cannot mount proc on /proc! Compile the kernel with CONFIG_PROC_FS!"
        exit 1
    fi
fi

if [ ! -d /sys/kernel ]; then
    if ! mount -t sysfs -o nosuid,noexec,nodev sysfs /sys > /dev/null; then
        echo "Cannot mount sysfs on /sys! Compile the kernel with CONFIG_SYSFS!"
        exit 1
    fi
fi

RD_DEBUG=""
. /lib/dracut-lib.sh

setdebug

if ! ismounted /dev; then
    mount -t devtmpfs -o mode=0755,noexec,nosuid,strictatime devtmpfs /dev > /dev/null
fi

if ! ismounted /dev; then
    echo "Cannot mount devtmpfs on /dev! Compile the kernel with CONFIG_DEVTMPFS!"
    exit 1
fi

# prepare the /dev directory
[ ! -h /dev/fd ] && ln -s /proc/self/fd /dev/fd > /dev/null 2>&1
[ ! -h /dev/stdin ] && ln -s /proc/self/fd/0 /dev/stdin > /dev/null 2>&1
[ ! -h /dev/stdout ] && ln -s /proc/self/fd/1 /dev/stdout > /dev/null 2>&1
[ ! -h /dev/stderr ] && ln -s /proc/self/fd/2 /dev/stderr > /dev/null 2>&1

if ! ismounted /dev/pts; then
    mkdir -m 0755 -p /dev/pts
    mount -t devpts -o gid=5,mode=620,noexec,nosuid devpts /dev/pts > /dev/null
fi

if ! ismounted /dev/shm; then
    mkdir -m 0755 -p /dev/shm
    mount -t tmpfs -o mode=1777,noexec,nosuid,nodev,strictatime tmpfs /dev/shm > /dev/null
fi

if ! ismounted /run; then
    mkdir -m 0755 -p /newrun
    if ! str_starts "$(readlink -f /bin/sh)" "/run/"; then
        mount -t tmpfs -o mode=0755,noexec,nosuid,nodev,strictatime tmpfs /newrun > /dev/null
    else
        # the initramfs binaries are located in /run, so don't mount it with noexec
        mount -t tmpfs -o mode=0755,nosuid,nodev,strictatime tmpfs /newrun > /dev/null
    fi
    cp -a /run/* /newrun > /dev/null 2>&1
    mount --move /newrun /run
    rm -fr -- /newrun
fi

if command -v kmod > /dev/null 2> /dev/null; then
    kmod static-nodes --format=tmpfiles 2> /dev/null \
        | while read -r type file mode _ _ _ majmin || [ -n "$type" ]; do
            type=${type%\!}
            case $type in
                d)
                    mkdir -m "$mode" -p "$file"
                    ;;
                c)
                    mknod -m "$mode" "$file" "$type" "${majmin%:*}" "${majmin#*:}"
                    ;;
            esac
        done
fi


mkdir -p /mnt /squash 
# Check if /dev/sda1 exists.
if [ ! -b /dev/sda1 ]; then
    echo "Error: /dev/sda1 does not exist!"
    emergency_shell
fi

mount -t ext4 /dev/sda1 /mnt || { echo "Failed to mount /dev/sda1"; emergency_shell; }

# Check if /mnt/gentoo.squashfs exists.
if [ ! -f /mnt/gentoo.squashfs ]; then
   echo "Error: /mnt/gentoo.squashfs does not exist!"
    emergency_shell
fi

mount /mnt/gentoo.squashfs /squash || { echo "Failed to mount /mnt/gentoo.squashfs"; emergency_shell; }

echo "Mounting overlay..."
mkdir -p /mnt/persistent/lower /mnt/persistent/work
mount -t overlay overlay -o upperdir=/mnt/persistent/lower,lowerdir=/squash,workdir=/mnt/persistent/work "$NEWROOT" || { echo "Failed to mount overlay"; emergency_shell; }


CAPSH=$(command -v capsh)
SWITCH_ROOT=$(command -v switch_root)

# You *might* want to set a default INIT if it's not provided externally
INIT="${INIT:-/sbin/init}"  # Default to /sbin/init if INIT is not set

# initargs might be empty, but we'll define it for clarity
initargs="${initargs:-}"


if [ -f /etc/capsdrop ]; then
    . /etc/capsdrop
    echo "Calling $INIT with capabilities $CAPS_INIT_DROP dropped."
    unset RD_DEBUG
    exec "$CAPSH" --drop="$CAPS_INIT_DROP" -- \
        -c "exec $SWITCH_ROOT \"$NEWROOT\" \"$INIT\" $initargs" \
        || {
            echo "Command:"
            echo "$CAPSH --drop=\"$CAPS_INIT_DROP\" -- -c exec $SWITCH_ROOT \"$NEWROOT\" \"$INIT\" $initargs"
            echo "failed."
            emergency_shell
        }
else
    unset RD_DEBUG
    # shellcheck disable=SC2086
    exec "$SWITCH_ROOT" "$NEWROOT" "$INIT" $initargs || {
        echo "Something went very badly wrong in the initramfs.  Please "
        echo "file a bug against dracut."
        emergency_shell
    }
fi

    