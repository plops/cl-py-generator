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


echo "Martin's init script for encrypted rootfs"


PARTNAME=/dev/disk/by-label/gentoo
#PARTNAME2=/dev/disk/by-label/docker
PARTNAME2C=/dev/disk/by-uuid/42bbab57-7cb0-465f-8ef9-6b34379da7d3

cryptsetup luksOpen $PARTNAME2C vg
lvm vgchange -ay


mkdir -p /mnt1 /mnt /squash
# Check if /dev/sda1 exists.
if [ ! -b $PARTNAME ]; then
    echo "Error: $PARTNAME does not exist!"
    emergency_shell
fi

mount $PARTNAME /mnt1 || { echo "Failed to mount $PARTNAME"; emergency_shell; }
#mount $PARTNAME2 /mnt || { echo "Failed to mount $PARTNAME2"; emergency_shell; }
mount /dev/mapper/ubuntu--vg-ubuntu--lv /mnt || { echo "Failed to mount encrypted $PARTNAME2C"; emergency_shell; }

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


#
#
#
## Create necessary directories
#mkdir -p /mnt /squash
#
## Unmount potential pre-mounted partitions
#umount /dev/nvme0n1p1
#umount /dev/nvme0n1p3
#umount /dev/nvme0n1p5
#
## Decrypt the LUKS encrypted partition
#cryptsetup luksOpen /dev/nvme0n1p4 vg
#
## If you use lvm you might need to activate all known volume groups
## vgchange -ay
#
## Mount the ext4 filesystems
#mount -t ext4 /dev/mapper/vg /mnt
#
#echo "Mount gentoo.squashfs from /dev/nvme0n1p4"
## Mount the squashfs
#mount /mnt/gentoo.squashfs /squash
#
## Set up directories for overlayfs
#mkdir -p /mnt/persistent/lower
#mkdir -p /mnt/persistent/work
#
## Mount overlayfs
#mount -t overlay overlay -o upperdir=/mnt/persistent/lower,lowerdir=/squash,workdir=/mnt/persistent/work /sysroot
#
#
## By the time we get here, the root filesystem should be mounted.
## Try to find init.
#for i in "$(getarg real_init=)" "$(getarg init=)" $(getargs rd.distroinit=) /sbin/init; do
#    [ -n "$i" ] || continue
#
#    __p="${NEWROOT}/${i}"
#    if [ -h "$__p" ]; then
#        # relative links need to be left alone,
#        # while absolute links need to be resolved and prefixed.
#        __pt=$(readlink "$__p")
#        [ "${__pt#/}" = "$__pt" ] || __p="${NEWROOT}/$__pt"
#    fi
#    if [ -x "$__p" ]; then
#        INIT="$i"
#        break
#    fi
#done
#
#[ "$INIT" ] || {
#    echo "Cannot find init!"
#    echo "Please check to make sure you passed a valid root filesystem!"
#    emergency_shell
#}
#
#udevadm control --exit
#udevadm info --cleanup-db
#
#debug_off # Turn off debugging for this section
#
## unexport some vars
#export_n root rflags fstype netroot NEWROOT
#unset CMDLINE
#export RD_TIMESTAMP
## Clean up the environment
#for i in $(export -p); do
#    i=${i#declare -x}
#    i=${i#export}
#    strstr "$i" "=" || continue
#    i=${i%%=*}
#    [ -z "$i" ] && continue
#    case $i in
#        root | PATH | HOME | TERM | PS4 | RD_*)
#            :
#            ;;
#        *)
#            unset "$i"
#            ;;
#    esac
#done
#. /tmp/export.orig 2> /dev/null || :
#rm -f -- /tmp/export.orig
#
#initargs=""
#read -r CLINE < /proc/cmdline
#if getarg init= > /dev/null; then
#    ignoreargs="console BOOT_IMAGE"
#    # only pass arguments after init= to the init
#    CLINE=${CLINE#*init=}
#    set -- "$CLINE"
#    shift # clear out the rest of the "init=" arg
#    for x in "$@"; do
#        for s in $ignoreargs; do
#            [ "${x%%=*}" = "$s" ] && continue 2
#        done
#        initargs="$initargs $x"
#    done
#    unset CLINE
#else
#    debug_off # Turn off debugging for this section
#    set -- "$CLINE"
#    for x in "$@"; do
#        case "$x" in
#            [0-9] | s | S | single | emergency | auto)
#                initargs="$initargs $x"
#                ;;
#        esac
#    done
#fi
#debug_on
#
#if ! [ -d "$NEWROOT"/run ]; then
#    NEWRUN=/dev/.initramfs
#    mkdir -m 0755 -p "$NEWRUN"
#    mount --rbind /run/initramfs "$NEWRUN"
#fi
#
#wait_for_loginit
#
## remove helper symlink
#[ -h /dev/root ] && rm -f -- /dev/root
#
#bv=$(getarg rd.break -d rdbreak) && [ -z "$bv" ] \
#    && emergency_shell -n switch_root "Break before switch_root"
#unset bv
#info "Switching root"
#
#unset PS4
#
#CAPSH=$(command -v capsh)
#SWITCH_ROOT=$(command -v switch_root)
#PATH=$OLDPATH
#export PATH
#
#if [ -f /etc/capsdrop ]; then
#    . /etc/capsdrop
#    info "Calling $INIT with capabilities $CAPS_INIT_DROP dropped."
#    unset RD_DEBUG
#    exec "$CAPSH" --drop="$CAPS_INIT_DROP" -- \
#        -c "exec switch_root \"$NEWROOT\" \"$INIT\" $initargs" \
#        || {
#            warn "Command:"
#            warn capsh --drop="$CAPS_INIT_DROP" -- -c exec switch_root "$NEWROOT" "$INIT" "$initargs"
#            warn "failed."
#            emergency_shell
#        }
#else
#    unset RD_DEBUG
#    # shellcheck disable=SC2086
#    exec "$SWITCH_ROOT" "$NEWROOT" "$INIT" $initargs || {
#        warn "Something went very badly wrong in the initramfs.  Please "
#        warn "file a bug against dracut."
#        emergency_shell
#    }
#fi
#
#
