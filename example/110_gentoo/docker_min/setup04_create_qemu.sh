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
qemu-img create -f raw qemu/sda1.img 653M
log_message "Created raw disk image: qemu/sda1.img (653M)"

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
log_message "Creating ext4 partition..."
parted "$LOOP_DEVICE" mkpart primary ext4 1MiB 100%
# Set the boot flag
log_message "Setting boot flag on partition 1..."
parted "$LOOP_DEVICE" set 1 boot on

# List partitions again, after creation, to verify
log_message "Listing partitions after creation:"
lsblk "$LOOP_DEVICE"


# Create a file system
log_message "Creating ext4 filesystem on ${LOOP_DEVICE}p1..."
mkfs.ext4 "${LOOP_DEVICE}p1"
# Mount the file system
log_message "Mounting ${LOOP_DEVICE}p1 to /mnt..."
mount "${LOOP_DEVICE}p1" /mnt
# Create folder for grub
log_message "Creating /mnt/boot/grub directory..."
mkdir -p /mnt/boot/grub
# Install grub
log_message "Installing GRUB to $LOOP_DEVICE..."
grub-install --target=i386-pc --boot-directory=/mnt/boot "$LOOP_DEVICE"

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

# Create grub.cfg
log_message "Creating /mnt/boot/grub/grub.cfg..."
cat << EOF > /mnt/boot/grub/grub.cfg
set default=0
set timeout=5
set root=(hd0,1)
menuentry "Gentoo" {
    linux /vmlinuz root=/dev/sda1
    initrd /initramfs_squash_sda1-x86_64.img
}
EOF

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