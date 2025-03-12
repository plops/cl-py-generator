#!/bin/sh


# Define a simple emergency shell
emergency_shell() {
    echo "Entering emergency shell..."
    /bin/sh
}

# Define NEWROOT (this MUST be done before switch_root)
NEWROOT=/sysroot

OLDPATH=$PATH
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH

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

# only needed for ssh, which we don't use in initramfs
# if ! ismounted /dev/pts; then
#     mkdir -m 0755 -p /dev/pts
#     mount -t devpts -o gid=5,mode=620,noexec,nosuid devpts /dev/pts > /dev/null
# fi

if ! ismounted /dev/shm; then
    mkdir -m 0755 -p /dev/shm
    mount -t tmpfs -o mode=1777,exec,nosuid,nodev,strictatime tmpfs /dev/shm > /dev/null
fi

# if ! ismounted /run; then
#     mkdir -m 0755 -p /newrun
#     if ! str_starts "$(readlink -f /bin/sh)" "/run/"; then
#         mount -t tmpfs -o mode=0755,noexec,nosuid,nodev,strictatime tmpfs /newrun > /dev/null
#     else
#         # the initramfs binaries are located in /run, so don't mount it with noexec
#         mount -t tmpfs -o mode=0755,nosuid,nodev,strictatime tmpfs /newrun > /dev/null
#     fi
#     cp -a /run/* /newrun > /dev/null 2>&1
#     mount --move /newrun /run
#     rm -fr -- /newrun
# fi

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

# trap "emergency_shell Signal caught!" 0

# export UDEVRULESD=/run/udev/rules.d
# [ -d /run/udev ] || mkdir -p -m 0755 /run/udev
# [ -d "$UDEVRULESD" ] || mkdir -p -m 0755 "$UDEVRULESD"


# udevadm control --reload > /dev/null 2>&1 || :
# # then the rest
# udevadm trigger --type=subsystems --action=add > /dev/null 2>&1
# udevadm trigger --type=devices --action=add > /dev/null 2>&1


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

cp /mnt/gentoo.squashfs /dev/shm/gentoo.squashfs || { echo "Failed to copy /mnt/gentoo.squashfs"; emergency_shell; }
mount /dev/shm/gentoo.squashfs /squash || { echo "Failed to mount /dev/shm/gentoo.squashfs"; emergency_shell; }

echo "Mounting overlay..."
mkdir -p /mnt/persistent/lower /mnt/persistent/work "$NEWROOT"
mount -t overlay overlay -o upperdir=/mnt/persistent/lower,lowerdir=/squash,workdir=/mnt/persistent/work "$NEWROOT" || { echo "Failed to mount overlay"; emergency_shell; }

echo "Contents of /:"
ls -l /
echo "Contents of /mnt:"
ls -l /mnt
echo "Contents of /squash:"
ls -l /squash
echo "Contents of /sysroot:"
ls -l "$NEWROOT"
echo "df -h:"
/sysroot/usr/bin/df -h
#
export PATH=/bin:/usr/bin:/sysroot/usr/bin
export LD_LIBRARY_PATH=/lib:/usr/lib:/sysroot/usr/lib64:/sysroot/usr/lib64/systemd/:/usr/lib64:/usr/lib64/systemd/
#/bin/bash
# clean up. The init process will remount proc sys and dev later
umount /proc
umount /sys

rm -rf /dev/fd /dev/stdin /dev/stdout /dev/stderr
#umount /dev

# exec /usr/bin/switch_root /sysroot /lib/systemd/systemd
exec /usr/bin/switch_root /sysroot /usr/bin/bash
