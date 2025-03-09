mkdir -p qemu
qemu-img create -f raw qemu/nvme0n1.img 1100M
losetup -fP qemu/nvme0n1.img 

  #  -f find the first unused loop device
  #  -P scans for the partitions

# List loop devices
losetup -l
# List partitions
parted /dev/loop0 print
# Create msdos disk label
parted /dev/loop0 mklabel msdos
# Create ext4 partition
parted /dev/loop0 mkpart primary ext4 1Mib 1Gib
# Set the boot flag
parted /dev/loop0 set 1 boot on

# Update partition table
losetup -d /dev/loop0
losetup -fP qemu/nvme0n1.img 

# Create a file system
mkfs.ext4 /dev/loop0p1
# Mount the file system
mount /dev/loop0p1 /mnt
# Create folder for grub
mkdir -p /mnt/boot/grub
# Install grub
grub-install --target=i386-pc --boot-directory=/mnt/boot /dev/loop0
# Copy squashfs, initramfs and kernel
cp /dev/shm/gentoo*/{*.squashfs,*.img,vmlinuz} /mnt


# Create grub.cfg
# https://wiki.archlinux.org/title/GRUB
# Boot vmlinuz with initramfs_squash_sda1-x86_64.img
cat << EOF > /mnt/boot/grub/grub.cfg
set default=0
set timeout=5
set root=(hd0,1)
menuentry "Gentoo" {
    linux /vmlinuz root=/dev/sda1
    initrd /initramfs_squash_nvme0n1p5-x86_64.img
}
EOF


# Umount the file system and detach the loop device
umount /mnt
losetup -d /dev/loop0

# Verify loop device is detached
losetup -l