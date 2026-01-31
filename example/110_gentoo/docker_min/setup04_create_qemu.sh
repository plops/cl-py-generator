#!/bin/bash

# Exit immediately on any error
set -e
#set -x  # Enable command tracing for debugging

# --- Helper Functions ---

# Function to log messages with timestamp
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check disk usage
check_disk_usage() {
    log_message "Checking disk usage on ${1}:"
    df -h "$1"
}


# --- Main Script ---

log_message "Starting script..."

# Create qemu directory if it doesn't exist
mkdir -p qemu
log_message "Created qemu directory."

# Create the raw disk image
# it will contain 2 partitions: vfat with grub, kernel and initramfs, squashfs; encrypted ext4 for persistence
qemu-img create -f raw qemu/sda1.img 560M
log_message "Created raw disk image: qemu/sda1.img"

# Detach all loop devices (cleanup from previous runs)
log_message "Detaching all existing loop devices..."
losetup -D

# Attach the image to a loop device
log_message "Attaching image to loop device..."
losetup -fP qemu/sda1.img

# List loop devices
log_message "Listing loop devices:"
losetup -l
# List partitions (using lsblk for better output and error handling)
log_message "Listing partitions (lsblk):"
lsblk /dev/loop0 || true # 'true' prevents exit if /dev/loop0 doesn't exist yet (first run)

# Get the loop device name (robustly)
LOOP_DEVICE=$(losetup -j qemu/sda1.img | awk '{print $1}' | sed 's/://')

if [ -z "$LOOP_DEVICE" ]; then
  log_message "Error: Could not determine loop device."
  exit 1
fi

log_message "Loop device is: $LOOP_DEVICE"


# Create msdos disk label
log_message "Creating msdos disk label..."
parted "$LOOP_DEVICE" mklabel msdos
# Create ext4 partition
log_message "Creating 2 partitions..."
parted "$LOOP_DEVICE" mkpart primary fat32 1MiB 553MB # grub, kernel, initramfs, squashfs
parted "$LOOP_DEVICE" mkpart primary ext4 553MB 100% # encrypted persistence
# Set the boot flag
log_message "Setting boot flag on partition 1..."
parted "$LOOP_DEVICE" set 1 boot on

# List partitions again, after creation, to verify
log_message "Listing partitions after creation:"
lsblk "$LOOP_DEVICE"


# Create a file system
log_message "Creating vfat filesystem on ${LOOP_DEVICE}p1..."
mkfs.vfat "${LOOP_DEVICE}p1"
log_message "Creating encrypted LUKS partition on ${LOOP_DEVICE}p2..."
# Set up LUKS encryption
echo -n "123" | cryptsetup luksFormat "${LOOP_DEVICE}p2"
# Open the LUKS partition
echo -n "123" | cryptsetup luksOpen "${LOOP_DEVICE}p2" enc

# --- NEU: LVM Setup ---
log_message "Setting up LVM (PV, VG, LV) on /dev/mapper/enc..."
# Physical Volume erstellen
pvcreate /dev/mapper/enc
# Volume Group 'vg' erstellen
vgcreate vg /dev/mapper/enc
# Logical Volume 'lv_persistence' erstellen (nutzt 100% des Platzes)
lvcreate -l 100%FREE -n lv_persistence vg

# Aktivieren (zur Sicherheit, meistens automatisch aktiv nach create)
vgchange -ay vg

log_message "Creating ext4 filesystem on /dev/mapper/vg-lv_persistence..."
# Dateisystem auf dem LV erstellen
mkfs.ext4 -L "persistence" /dev/mapper/vg-lv_persistence

# Mount the persistence partition to create required directories
log_message "Setting up persistence structure on Logical Volume..."
# Hier mounten wir das LV, nicht mehr 'enc' direkt
mount /dev/mapper/vg-lv_persistence /mnt
mkdir -p /mnt/overlayfs
mkdir -p /mnt/ovlwork
umount /mnt

# Close LVM and LUKS
log_message "Deactivating LVM and closing LUKS partition..."
vgchange -an vg
cryptsetup luksClose enc

# print the uuid of the encrypted partition
UUID=$(blkid -s UUID -o value "${LOOP_DEVICE}p2")
log_message "UUID of encrypted partition (${LOOP_DEVICE}p2): $UUID"





# Mount the file system
log_message "Mounting ${LOOP_DEVICE}p1 to /mnt..."
mount "${LOOP_DEVICE}p1" /mnt
# Create folder for grub
log_message "Creating /mnt/boot/grub directory..."
mkdir -p /mnt/boot/grub
# Install grub
log_message "Installing GRUB to $LOOP_DEVICE..."
grub-install --target=i386-pc --boot-directory=/mnt/boot "$LOOP_DEVICE"


# Note: you should have only one /dev/shm/gentoo-*/gentoo.squashfs file
# sudo rm -rf /dev/shm/gentoo-*;  ./setup03_copy_from_container.sh
# Copy squashfs, initramfs and kernel
log_message "Copying files from /dev/shm/..."
find /dev/shm/ -maxdepth 2 -name "gentoo*" -type d | while read -r gentoo_dir; do
    if [ -d "$gentoo_dir" ]; then
        log_message "Processing directory: $gentoo_dir"
        find "$gentoo_dir" -maxdepth 1 \( -name "*.squashfs" -o -name "*.img" -o -name "vmlinuz" \) -print0 |
            while IFS= read -r -d $'\0' file; do
                log_message "Copying file: $file"
                cp "$file" /mnt/
                check_disk_usage /mnt  # Check after each copy
            done
    else
        log_message "Warning: Directory not found: $gentoo_dir"
    fi
done

check_disk_usage /mnt #final check

# https://github.com/blickers/livebackup
# https://www.gnu.org/software/grub/manual/grub/html_node/Loopback-booting.html
# Create grub.cfg


# mapping:
# root=live:<device>            -> Where the squashfs file is
# rd.live.dir=/                 -> File is in root, not /LiveOS
# rd.live.squashimg=filename    -> The name of the file
# rd.live.ram=1                 -> Copy to RAM
# rd.overlay=<device>           -> The persistence partition
# rd.live.overlay.overlayfs=1   -> Use OverlayFS (not DeviceMapper)



# Wir müssen Dracut sagen, dass LVM genutzt wird.
# rd.lvm.vg=vg aktiviert die Volume Group 'vg'.
# rd.lvm.lv=vg/lv_persistence aktiviert das spezifische LV.

log_message "Creating /mnt/boot/grub/grub.cfg..."
cat << EOF > /mnt/boot/grub/grub.cfg
set default=0
set timeout=5
set root=(hd0,1)

menuentry 'Gentoo Dracut (LVM on LUKS)' {
    insmod part_msdos
    echo 'Loading Linux ...'
    linux /vmlinuz \
    root=live:/dev/sda1 \
    rd.live.dir=/ \
    rd.live.squashimg=gentoo.squashfs \
    rd.luks.uuid=${UUID} \
    rd.luks.name=${UUID}=enc \
    rd.lvm.vg=vg \
    rd.lvm.lv=vg/lv_persistence \
    rd.overlay=LABEL=persistence:/overlayfs \
    rd.live.overlay.overlayfs=1 \
    console=ttyS0
    initrd /initramfs_squash_sda1-x86_64.img
}

menuentry 'Gentoo Dracut (LVM on LUKS) debug' {
    insmod part_msdos
    echo 'Loading Linux ...'
    linux /vmlinuz \
    root=live:/dev/sda1 \
    rd.live.dir=/ \
    rd.live.squashimg=gentoo.squashfs \
    rd.luks.uuid=${UUID} \
    rd.luks.name=${UUID}=enc \
    rd.lvm.vg=vg \
    rd.lvm.lv=vg/lv_persistence \
    rd.overlay=LABEL=persistence:/overlayfs \
    rd.live.overlay.overlayfs=1 \
    rd.break=cleanup \
    console=ttyS0 rd.debug
    initrd /initramfs_squash_sda1-x86_64.img
}
EOF
# The AI expects that mounting the overlay takes too long and the dracut script defaults to ramfs for upper layer
# of the overlayfs

# Print the grub.cfg (so that i can see if the UUID is correct)
log_message "Contents of /mnt/boot/grub/grub.cfg:"
cat /mnt/boot/grub/grub.cfg



# Summary of what happens on boot:
# Dracut starts.
# It sees root=live:/dev/sda1. It mounts sda1 to /run/initramfs/live.
# It sees rd.live.ram=1. It copies /run/initramfs/live/gentoo.squashfs to RAM.
# It sees rd.overlay=/dev/sda2. It mounts sda2 temporarily.
# It checks if sda2 contains directories /overlayfs and /ovlwork (which we created in the updated script).
# It creates the union mount using those directories over the RAM-backed system.




log_message "grub.cfg created."

# Umount the file system and detach the loop device
log_message "Unmounting /mnt..."
umount /mnt
log_message "Detaching loop device: $LOOP_DEVICE..."
losetup -d "$LOOP_DEVICE"

# Verify loop device is detached
log_message "Verifying loop device detachment:"
losetup -l

log_message "Script completed successfully."
exit 0
