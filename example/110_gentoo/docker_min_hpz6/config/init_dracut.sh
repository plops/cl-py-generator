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

 export UDEVRULESD=/run/udev/rules.d
 [ -d /run/udev ] || mkdir -p -m 0755 /run/udev
 [ -d "$UDEVRULESD" ] || mkdir -p -m 0755 "$UDEVRULESD"


 udevadm control --reload > /dev/null 2>&1 || :
 # then the rest
 udevadm trigger --type=subsystems --action=add > /dev/null 2>&1
 udevadm trigger --type=devices --action=add > /dev/null 2>&1


#PARTNAME=/dev/sda1
PARTNAME=/dev/disk/by-label/gentoo
PARTNAME2=/dev/disk/by-label/docker

mkdir -p /mnt1 /mnt /squash 
# Check if /dev/sda1 exists.
if [ ! -b $PARTNAME ]; then
    echo "Error: $PARTNAME does not exist!"
    emergency_shell
fi

mount $PARTNAME /mnt1 || { echo "Failed to mount $PARTNAME"; emergency_shell; }
mount $PARTNAME2 /mnt || { echo "Failed to mount $PARTNAME2"; emergency_shell; }

# Check if /mnt/gentoo.squashfs exists.
if [ ! -f /mnt1/gentoo.squashfs ]; then
   echo "Error: /mnt1/gentoo.squashfs does not exist!"
    emergency_shell
fi

cp /mnt1/gentoo.squashfs /dev/shm/gentoo.squashfs || { echo "Failed to copy /mnt1/gentoo.squashfs"; emergency_shell; }
mount /dev/shm/gentoo.squashfs /squash || { echo "Failed to mount /dev/shm/gentoo.squashfs"; emergency_shell; }

echo "Mounting overlay..."
mkdir -p /mnt/persistent/lower /mnt/persistent/work "$NEWROOT"
mount -t overlay overlay -o upperdir=/mnt/persistent/lower,lowerdir=/squash,workdir=/mnt/persistent/work "$NEWROOT" || { echo "Failed to mount overlay"; emergency_shell; }


exec /usr/bin/switch_root /sysroot /bin/init
