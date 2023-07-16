
I asked chatgpt how to boot my system from squashfs:


What you are trying to do is somewhat complex, but definitely achievable. Here is a step-by-step guide:

**Step 1: Prepare the SquashFS and OverlayFS directories**

Make sure your squashfs file and the directories for overlayfs exist and are located properly. For the sake of this example, let's assume the SquashFS file (`/gentoo_20230716b.squashfs`) is in the root of your SSD, and the overlay directory will be `/persist`.

**Step 2: Generate the fstab file for initramfs**

Firstly, you need to create an `/etc/fstab` file that will be included in your initramfs. This file will define the mount points for your SquashFS and overlay filesystems.

Create a new `/etc/fstab.initramfs` file with the following content:

```
/tmp/gentoo_20230716b.squashfs /sfs/sfs squashfs defaults 0 0
/persist /sfs/overlay overlay defaults,upperdir=/persist/upper,lowerdir=/sfs/sfs,workdir=/persist/work 0 0
```

Here, `/sfs/sfs` is the mount point for your SquashFS filesystem, and `/sfs/overlay` is the mount point for your overlay filesystem.

**Step 3: Generate the initramfs**

Now generate the initramfs with dracut. 

```
mkdir -p /tmp/sfs/sfs /tmp/sfs/overlay
dracut  -i /etc/fstab.initramfs /etc/fstab     \
        -i /tmp/sfs /sfs              \
        --add-drivers overlay   \
        --add-drivers squashfs  \
        /boot/initramfs.img
```

This will include both the `fstab.initramfs` and the required drivers for overlay and squashfs into the initramfs.

**Step 4: Update the GRUB configuration**

Add a new entry to your GRUB configuration (`/etc/grub.d/40_custom`):

```bash
menuentry "Gentoo from SquashFS" {
    set root=(hd0,1)
    linux /boot/vmlinuz-VERSION-OF-YOUR-KERNEL root=/dev/sda1 ro rootfstype=overlay rootflags=lowerdir=/sfs/sfs,upperdir=/persist/upper,workdir=/persist/work
    initrd /boot/initramfs.img
}
```

Replace `(hd0,1)` with the appropriate GRUB disk/partition identifier for your root partition (in your case, where the SSD is), and `root=/dev/sda1` with the correct `/dev` entry for your root partition. Substitute `vmlinuz-VERSION-OF-YOUR-KERNEL` with your actual kernel image name.

**Step 5: Update GRUB**

Finally, update GRUB to incorporate the new configuration:

```bash
update-grub
```

Now, when you boot your system, you should see a new entry in your GRUB boot menu for "Gentoo from SquashFS". Selecting this option will boot your system from the SquashFS filesystem with an overlay filesystem for changes.

Please remember that the above commands should be run with superuser (root) privileges, and they are quite potent, potentially altering the way your system boots. Be careful to have backups and/or a rescue system at hand.

Also note that SquashFS is a read-only filesystem, so any changes you make during your session will only persist until you reboot if you use an overlay filesystem. With an overlay filesystem, you can create a writable layer (the "upperdir") that stores changes to files and directories in the SquashFS filesystem (the "lowerdir"), but the original SquashFS filesystem remains unaltered. Changes are only stored in the overlay filesystem, and they can be persisted across reboots if you store the overlay filesystem on a persistent storage medium (like your SSD).