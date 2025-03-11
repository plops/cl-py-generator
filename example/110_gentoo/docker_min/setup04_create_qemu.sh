#!/bin/bash

# Exit immediately on any error
set -e

mkdir -p qemu
qemu-img create -f raw qemu/sda1.img 900M

# Detach all loop devices
losetup -D

losetup -fP qemu/sda1.img

# List loop devices
losetup -l
# List partitions (using lsblk for better output and error handling)
lsblk /dev/loop0 || true # 'true' prevents exit if /dev/loop0 doesn't exist yet (first run)


# Get the loop device name (robustly)
LOOP_DEVICE=$(losetup -j qemu/sda1.img | awk '{print $1}' | sed 's/://')

if [ -z "$LOOP_DEVICE" ]; then
  echo "Error: Could not determine loop device."
  exit 1
fi


# Create msdos disk label
parted "$LOOP_DEVICE" mklabel msdos
# Create ext4 partition  (fixed size to match image size, no need to overcomplicate it)
parted "$LOOP_DEVICE" mkpart primary ext4 1MiB 100%
# Set the boot flag
parted "$LOOP_DEVICE" set 1 boot on

# Update partition table (no need to detach and reattach; parted handles this)
# Removed the redundant detach/reattach

# Create a file system
mkfs.ext4 "${LOOP_DEVICE}p1"
# Mount the file system
mount "${LOOP_DEVICE}p1" /mnt
# Create folder for grub
mkdir -p /mnt/boot/grub
# Install grub
grub-install --target=i386-pc --boot-directory=/mnt/boot "$LOOP_DEVICE"
# Copy squashfs, initramfs and kernel (using a more robust find)

find /dev/shm/ -maxdepth 2 -name "gentoo*" -type d | while read -r gentoo_dir; do
    if [ -d "$gentoo_dir" ]; then
      find "$gentoo_dir" -maxdepth 1 \( -name "*.squashfs" -o -name "*.img" -o -name "vmlinuz" \) -print0 |
          xargs -0 -I {} cp {} /mnt/
    fi
done


# Create grub.cfg
# https://wiki.archlinux.org/title/GRUB
# Boot vmlinuz with initramfs_squash_sda1-x86_64.img
# Added quotes around variables for safety.
cat << EOF > /mnt/boot/grub/grub.cfg
set default=0
set timeout=5
set root=(hd0,1)
menuentry "Gentoo" {
    linux /vmlinuz root=/dev/sda1
    initrd /initramfs_squash_sda1-x86_64.img
}
EOF


# Umount the file system and detach the loop device
umount /mnt
losetup -d "$LOOP_DEVICE"

# Verify loop device is detached
losetup -l

echo "Script completed successfully."
exit 0

# mkdir -p qemu
# qemu-img create -f raw qemu/sda1.img 600M
# losetup -fP qemu/sda1.img 

#   #  -f find the first unused loop device
#   #  -P scans for the partitions

# # List loop devices
# losetup -l
# # List partitions
# parted /dev/loop0 print
# # Create msdos disk label
# parted /dev/loop0 mklabel msdos
# # Create ext4 partition
# parted /dev/loop0 mkpart primary ext4 1Mib 580Gb
# # Set the boot flag
# parted /dev/loop0 set 1 boot on

# # Update partition table
# losetup -d /dev/loop0
# losetup -fP qemu/sda1.img 

# # Create a file system
# mkfs.ext4 /dev/loop0p1
# # Mount the file system
# mount /dev/loop0p1 /mnt
# # Create folder for grub
# mkdir -p /mnt/boot/grub
# # Install grub
# grub-install --target=i386-pc --boot-directory=/mnt/boot /dev/loop0
# # Copy squashfs, initramfs and kernel
# cp /dev/shm/gentoo*/{*.squashfs,*.img,vmlinuz} /mnt


# # Create grub.cfg
# # https://wiki.archlinux.org/title/GRUB
# # Boot vmlinuz with initramfs_squash_sda1-x86_64.img
# cat << EOF > /mnt/boot/grub/grub.cfg
# set default=0
# set timeout=5
# set root=(hd0,1)
# menuentry "Gentoo" {
#     linux /vmlinuz root=/dev/sda1
#     initrd /initramfs_squash_sda1-x86_64.img
# }
# EOF


# # Umount the file system and detach the loop device
# umount /mnt
# losetup -d /dev/loop0

# # Verify loop device is detached
# losetup -l

