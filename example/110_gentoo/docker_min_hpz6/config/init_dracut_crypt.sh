#!/bin/sh

export -p > /tmp/export.orig

NEWROOT="/sysroot"
[ -d $NEWROOT ] || mkdir -p -m 0755 $NEWROOT

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

trap "emergency_shell Signal caught!" 0

export UDEVRULESD=/run/udev/rules.d
[ -d /run/udev ] || mkdir -p -m 0755 /run/udev
[ -d "$UDEVRULESD" ] || mkdir -p -m 0755 "$UDEVRULESD"

if [ "$RD_DEBUG" = "yes" ]; then
    mkfifo /run/initramfs/loginit.pipe
    loginit "$DRACUT_QUIET" < /run/initramfs/loginit.pipe > /dev/console 2>&1 &
    exec > /run/initramfs/loginit.pipe 2>&1
else
    exec 0<> /dev/console 1<> /dev/console 2<> /dev/console
fi

[ -f /usr/lib/initrd-release ] && . /usr/lib/initrd-release
[ -n "$VERSION_ID" ] && info "$NAME-$VERSION_ID"

source_conf /etc/conf.d

if getarg "rd.cmdline=ask"; then
    echo "Enter additional kernel command line parameter (end with ctrl-d or .)"
    while read -r -p "> " ${BASH:+-e} line || [ -n "$line" ]; do
        [ "$line" = "." ] && break
        echo "$line" >> /etc/cmdline.d/99-cmdline-ask.conf
    done
fi

if ! getargbool 1 'rd.hostonly'; then
    [ -f /etc/cmdline.d/99-cmdline-ask.conf ] && mv /etc/cmdline.d/99-cmdline-ask.conf /tmp/99-cmdline-ask.conf
    remove_hostonly_files
    [ -f /tmp/99-cmdline-ask.conf ] && mv /tmp/99-cmdline-ask.conf /etc/cmdline.d/99-cmdline-ask.conf
fi

# run scriptlets to parse the command line
make_trace_mem "hook cmdline" '1+:mem' '1+:iomem' '3+:slab'
getarg 'rd.break=cmdline' -d 'rdbreak=cmdline' && emergency_shell -n cmdline "Break before cmdline"
source_hook cmdline

[ -z "$root" ] && die "No or empty root= argument"
[ -z "$rootok" ] && die "Don't know how to handle 'root=$root'"

export root rflags fstype netroot NEWROOT

# pre-udev scripts run before udev starts, and are run only once.
make_trace_mem "hook pre-udev" '1:shortmem' '2+:mem' '3+:slab'
getarg 'rd.break=pre-udev' -d 'rdbreak=pre-udev' && emergency_shell -n pre-udev "Break before pre-udev"
source_hook pre-udev

UDEV_LOG=err
getargbool 0 rd.udev.info -d -y rdudevinfo && UDEV_LOG=info
getargbool 0 rd.udev.debug -d -y rdudevdebug && UDEV_LOG=debug

# start up udev and trigger cold plugs
UDEV_LOG=$UDEV_LOG "$systemdutildir"/systemd-udevd --daemon --resolve-names=never

UDEV_QUEUE_EMPTY="udevadm settle --timeout=0"

udevproperty "hookdir=$hookdir"

make_trace_mem "hook pre-trigger" '1:shortmem' '2+:mem' '3+:slab'
getarg 'rd.break=pre-trigger' -d 'rdbreak=pre-trigger' && emergency_shell -n pre-trigger "Break before pre-trigger"
source_hook pre-trigger

udevadm control --reload > /dev/null 2>&1 || :
# then the rest
udevadm trigger --type=subsystems --action=add > /dev/null 2>&1
udevadm trigger --type=devices --action=add > /dev/null 2>&1

make_trace_mem "hook initqueue" '1:shortmem' '2+:mem' '3+:slab'
getarg 'rd.break=initqueue' -d 'rdbreak=initqueue' && emergency_shell -n initqueue "Break before initqueue"

RDRETRY=$(getarg rd.retry -d 'rd_retry=')
RDRETRY=${RDRETRY:-180}
RDRETRY=$((RDRETRY * 2))
export RDRETRY
main_loop=0
export main_loop
while :; do

    check_finished && break

    udevsettle

    check_finished && break

    if [ -f "$hookdir"/initqueue/work ]; then
        rm -f -- "$hookdir"/initqueue/work
    fi

    for job in "$hookdir"/initqueue/*.sh; do
        [ -e "$job" ] || break
        # shellcheck disable=SC2097 disable=SC1090 disable=SC2098
        job=$job . "$job"
        check_finished && break 2
    done

    $UDEV_QUEUE_EMPTY > /dev/null 2>&1 || continue

    for job in "$hookdir"/initqueue/settled/*.sh; do
        [ -e "$job" ] || break
        # shellcheck disable=SC2097 disable=SC1090 disable=SC2098
        job=$job . "$job"
        check_finished && break 2
    done

    $UDEV_QUEUE_EMPTY > /dev/null 2>&1 || continue

    # no more udev jobs and queues empty.
    sleep 0.5

    if [ $main_loop -gt $((2 * RDRETRY / 3)) ]; then
        for job in "$hookdir"/initqueue/timeout/*.sh; do
            [ -e "$job" ] || break
            # shellcheck disable=SC2097 disable=SC1090 disable=SC2098
            job=$job . "$job"
            udevadm settle --timeout=0 > /dev/null 2>&1 || main_loop=0
            [ -f "$hookdir"/initqueue/work ] && main_loop=0
        done
    fi

    main_loop=$((main_loop + 1))
    [ $main_loop -gt $RDRETRY ] \
        && {
            flock -s 9
            emergency_shell "Could not boot."
        } 9> /.console_lock
done
unset job
unset queuetriggered
unset main_loop
unset RDRETRY

# pre-mount happens before we try to mount the root filesystem,
# and happens once.
make_trace_mem "hook pre-mount" '1:shortmem' '2+:mem' '3+:slab'
getarg 'rd.break=pre-mount' -d 'rdbreak=pre-mount' && emergency_shell -n pre-mount "Break pre-mount"
source_hook pre-mount

getarg 'rd.break=mount' -d 'rdbreak=mount' && emergency_shell -n mount "Break mount"


# mount scripts actually try to mount the root filesystem, and may
# be sourced any number of times. As soon as one suceeds, no more are sourced.
_i_mount=0
while :; do
    if ismounted "$NEWROOT"; then
        usable_root "$NEWROOT" && break
        umount "$NEWROOT"
    fi
    for f in "$hookdir"/mount/*.sh; do
        # shellcheck disable=SC1090
        [ -f "$f" ] && . "$f"
        if ismounted "$NEWROOT"; then
            usable_root "$NEWROOT" && break
            warn "$NEWROOT has no proper rootfs layout, ignoring and removing offending mount hook"
            umount "$NEWROOT"
            rm -f -- "$f"
        fi
    done

    _i_mount=$((_i_mount + 1))
    [ $_i_mount -gt 20 ] \
        && {
            flock -s 9
            emergency_shell "Can't mount root filesystem"
        } 9> /.console_lock
done

{
    printf "Mounted root filesystem "
    while read -r dev mp _ || [ -n "$dev" ]; do [ "$mp" = "$NEWROOT" ] && echo "$dev"; done < /proc/mounts
} | vinfo

# pre pivot scripts are sourced just before we doing cleanup and switch over
# to the new root.
make_trace_mem "hook pre-pivot" '1:shortmem' '2+:mem' '3+:slab'
getarg 'rd.break=pre-pivot' -d 'rdbreak=pre-pivot' && emergency_shell -n pre-pivot "Break pre-pivot"
source_hook pre-pivot

make_trace_mem "hook cleanup" '1:shortmem' '2+:mem' '3+:slab'
# pre pivot cleanup scripts are sourced just before we switch over to the new root.
getarg 'rd.break=cleanup' -d 'rdbreak=cleanup' && emergency_shell -n cleanup "Break cleanup"
source_hook cleanup

#GRUB_CMDLINE_LINUX_DEFAULT="quiet splash \
#squashfs.part=LABEL=gentoo \
#squashfs.file=gentoo.squashfs \
#persist.part=UUID=42bbab57-7cb0-465f-8ef9-6b34379da7d3 \
#persist.lv=/dev/mapper/ubuntu--vg-ubuntu--lv \
#persist.mapname=vg"

#menuentry 'Gentoo from ram (Configurable)' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-4f708c84-185d-437b-a03a-7a565f598a23' {
#	load_video
#	insmod gzio
#	insmod part_gpt
#	insmod btrfs
#	search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23
#	echo	'Loading Linux 6.12.31-gentoo-gentoo-dist ...'
#	linux	/boot/vmlinuz root=UUID=4f708c84-185d-437b-a03a-7a565f598a23 ro squashfs.part=LABEL=gentoo squashfs.file=gentoo.squashfs persist.part=UUID=42bbab57-7cb0-465f-8ef9-6b34379da7d3 persist.lv=/dev/mapper/ubuntu--vg-ubuntu--lv persist.mapname=vg
#	echo	'Loading initial ramdisk ...'
#	initrd	/boot/initramfs_squash_sda1-x86_64.img
#}

#
# Martin's init script for encrypted rootfs with configurable parameters
#
info "Martin's init: Starting custom overlay boot sequence."

# --- Get kernel parameters with sane defaults ---
# Use `getarg <param>` which is provided by dracut-lib.sh.
# The `|| echo "default"` part provides a fallback if the parameter isn't set.

SQUASH_PART=$(getarg squashfs.part || echo "/dev/disk/by-label/gentoo")
SQUASH_FILE=$(getarg squashfs.file || echo "gentoo.squashfs")
PERSIST_PART=$(getarg persist.part || echo "/dev/disk/by-uuid/42bbab57-7cb0-465f-8ef9-6b34379da7d3")
PERSIST_LV=$(getarg persist.lv || echo "/dev/mapper/ubuntu--vg-ubuntu--lv")
PERSIST_MAPNAME=$(getarg persist.mapname || echo "vg")

info "SquashFS source: ${SQUASH_PART}"
info "SquashFS file: ${SQUASH_FILE}"
info "Persistence source: ${PERSIST_PART}"
info "Persistence LV: ${PERSIST_LV}"
info "Persistence LUKS mapping name: ${PERSIST_MAPNAME}"

# --- Open encrypted persistence volume ---
info "Opening LUKS volume ${PERSIST_PART} as ${PERSIST_MAPNAME}"
cryptsetup luksOpen "${PERSIST_PART}" "${PERSIST_MAPNAME}" || {
    warn "Failed to open LUKS device ${PERSIST_PART}"
    emergency_shell
}
info "Activating LVM volume groups"
lvm vgchange -ay || {
    warn "Failed to activate LVM volume groups"
    emergency_shell
}

# --- Prepare and mount filesystems ---
mkdir -p /mnt_squash /mnt_persist /squash

# Mount partition containing the squashfs file
info "Mounting ${SQUASH_PART} on /mnt_squash"
if [ ! -b "${SQUASH_PART}" ]; then
    # Wait for the device to appear, udev might still be working
    udevadm settle
    if [ ! -b "${SQUASH_PART}" ]; then
        warn "SquashFS partition ${SQUASH_PART} does not exist!"
        emergency_shell
    fi
fi
mount "${SQUASH_PART}" /mnt_squash || {
    warn "Failed to mount ${SQUASH_PART}"
    emergency_shell
}

# Mount the persistent storage LV
info "Mounting ${PERSIST_LV} on /mnt_persist"
if [ ! -b "${PERSIST_LV}" ]; then
    warn "Persistent LV ${PERSIST_LV} does not exist!"
    emergency_shell
fi
mount "${PERSIST_LV}" /mnt_persist || {
    warn "Failed to mount encrypted LV ${PERSIST_LV}"
    emergency_shell
}

# Copy squashfs to RAM for better performance and to free the source device
SQUASH_SRC_PATH="/mnt_squash/${SQUASH_FILE}"
SQUASH_RAM_PATH="/dev/shm/${SQUASH_FILE}"
info "Checking for ${SQUASH_SRC_PATH}"
if [ ! -f "${SQUASH_SRC_PATH}" ]; then
    warn "${SQUASH_SRC_PATH} does not exist!"
    emergency_shell
fi

info "Copying ${SQUASH_SRC_PATH} to RAM"
cp "${SQUASH_SRC_PATH}" "${SQUASH_RAM_PATH}" || {
    warn "Failed to copy ${SQUASH_SRC_PATH}"
    emergency_shell
}

info "Mounting SquashFS image from RAM"
mount "${SQUASH_RAM_PATH}" /squash || {
    warn "Failed to mount ${SQUASH_RAM_PATH}"
    emergency_shell
}

# --- Setup the OverlayFS ---
info "Mounting overlay filesystem..."
UPPER_DIR="/mnt_persist/persistent/upper"
WORK_DIR="/mnt_persist/persistent/work"
LOWER_DIR="/squash"

# Create overlay directories if they don't exist
mkdir -p "${UPPER_DIR}" "${WORK_DIR}"

mount -t overlay overlay -o "upperdir=${UPPER_DIR},lowerdir=${LOWER_DIR},workdir=${WORK_DIR}" "${NEWROOT}" || {
    warn "Failed to mount overlay filesystem"
    emergency_shell
}

info "Successfully mounted overlayfs on ${NEWROOT}"

# --- Clean up before switching root ---
#umount /mnt_squash
#umount /mnt_persist
# The crypt-device will be closed by the real system shutdown

INIT_BIN="/sbin/init"

info "Switching root to ${NEWROOT} and executing ${INIT_BIN}"
exec /usr/bin/switch_root "${NEWROOT}" "${INIT_BIN}"