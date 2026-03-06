# Install Gentoo Image on HP Z6 (using existing GRUB)

This guide describes how to install a new live Gentoo image on an HP Z6 workstation **without** reinstalling GRUB.

It is not a clean-slate install:
- Existing bootloader (GRUB) is reused.
- Existing Linux installation is used as the working environment.
- A new encrypted persistent partition is prepared on `nvme0n1p5`.
- Kernel, initramfs, and squashfs remain on an unencrypted partition.

Current image set used in this guide:
- Build directory: `gentoo-z6-min_20260306`
- Version suffix for copied boot artifacts: `0306`

## 0. Preflight and safety checks

Run these checks first:

```bash
lsblk -f
```

On the target HP Z6 used for this guide, the relevant identifiers are:
- Boot/squashfs partition: `/dev/nvme0n1p3` (label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`)
- Persistent encrypted partition: `/dev/nvme0n1p5` (LUKS UUID `0d7c5e23-6bab-4dce-b744-a5d61d497aca`)
- Open mapper for persistence: `/dev/mapper/enc` (ext4 label `persist`)

Before editing GRUB config directly, always create a backup:

```bash
sudo cp -av /0p3/boot/grub/grub.cfg /0p3/boot/grub/grub.cfg.bak_$(date +%Y%m%d_%H%M%S)
```

## 1. Inspect disks and choose persistent partition

Boot into an existing Linux installation and inspect disks:

```bash
lsblk
```

Example layout (shortened):

```text
nvme0n1
├─nvme0n1p3   (contains existing Gentoo boot artifacts and squashfs)
└─nvme0n1p5   (will be repurposed as encrypted persistent storage)
```

Decision:
- Wipe `nvme0n1p5`
- Create `LUKS2 -> ext4` for persistence

Rationale:
- Keep boot artifacts (`vmlinuz`, `initramfs`, `squashfs`) on unencrypted storage.
- Encrypt only the persistent writable layer.

## 2. Verify where current Gentoo boots from

Check current kernel command line:

```bash
cat /proc/cmdline
```

Current setup references:
- `squashfs.part=/dev/disk/by-label/gentoo`

Verify label target:

```bash
ls -ltr /dev/disk/by-label/gentoo
```

Expected result:
- `/dev/disk/by-label/gentoo -> ../../nvme0n1p3`

Mount it to inspect contents:

```bash
sudo mkdir -p /0p3
mountpoint -q /0p3 || sudo mount /dev/nvme0n1p3 /0p3
ls /0p3
ls /0p3/boot
```

You should see previous dated artifacts such as:
- `/0p3/gentoo.squashfs_XXXX`
- `/0p3/boot/vmlinuz_XXXX`
- `/0p3/boot/initramfs_squash_sda1-x86_64.img_XXXX`

## 3. Check existing GRUB entry

Inspect GRUB config on that partition:

```bash
grep -i gentoo -A 10 /0p3/boot/grub/grub.cfg | head -n 12
```

Confirm:
- `search --fs-uuid` points to the partition that hosts `/boot` and squashfs (`UUID=4f708c84-...` in this setup).

## 4. Copy new build artifacts with date suffix

Artifacts were copied from container output (`setup03_copy_from_container.sh`) and are located in:
- `gentoo-z6-min_20260306/`

Example listing:

```bash
ls -tlrh gentoo-z6-min_20260306/
```

Expected files:
- `gentoo.squashfs`
- `vmlinuz`
- `initramfs_squash_sda1-x86_64.img`
- `packages.txt`

Copy to boot partition with rollback-friendly suffixes:

```bash
sudo cp -av gentoo-z6-min_20260306/gentoo.squashfs /0p3/gentoo.squashfs_0306
sudo cp -av gentoo-z6-min_20260306/vmlinuz /0p3/boot/vmlinuz_0306
sudo cp -av gentoo-z6-min_20260306/initramfs_squash_sda1-x86_64.img /0p3/boot/initramfs_squash_sda1-x86_64.img_0306
```

Verify copied files:

```bash
ls -lh /0p3/gentoo.squashfs_0306
ls -lh /0p3/boot/vmlinuz_0306 /0p3/boot/initramfs_squash_sda1-x86_64.img_0306
```

## 5. Create encrypted persistent partition (`nvme0n1p5`)

If persistence on `nvme0n1p5` is already configured and working, **skip this section** to avoid destructive reformatting.

Set variables:

```bash
export PERSIST_PART=/dev/nvme0n1p5
export CRYPT_NAME=persist
```

Create LUKS2 container and ext4 filesystem:

```bash
sudo cryptsetup luksFormat --type luks2 "$PERSIST_PART"
sudo cryptsetup open "$PERSIST_PART" "$CRYPT_NAME"
sudo mkfs.ext4 -L persist "/dev/mapper/$CRYPT_NAME"
sudo mkdir -p /mnt/persist
sudo mount "/dev/mapper/$CRYPT_NAME" /mnt/persist
sudo blkid "$PERSIST_PART" "/dev/mapper/$CRYPT_NAME"
```

Example UUIDs from this machine:

```text
/dev/nvme0n1p5: UUID="0d7c5e23-6bab-4dce-b744-a5d61d497aca" TYPE="crypto_LUKS"
/dev/mapper/persist: UUID="3ca5dfb2-35c9-4ed5-906a-f965dbcd1c7b" TYPE="ext4"
```

Boot partition UUID (where squashfs is loaded from) stays unchanged:

```bash
sudo blkid /dev/disk/by-label/gentoo
```

In this setup:
- Boot/squashfs UUID: `4f708c84-185d-437b-a03a-7a565f598a23`

## 6. Add GRUB menu entry for dracut live boot + encrypted overlay

Preferred location:
- Put custom entries into `/0p3/boot/grub/custom.cfg` (it is sourced by `grub.cfg`).
- This survives `grub-mkconfig` regeneration better than editing `grub.cfg` directly.

Use this entry (adapt UUIDs/paths if your system differs):

```cfg
menuentry 'Gentoo Dracut (persist on nvme0n1p5)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0306 \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0306 \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/initramfs_squash_sda1-x86_64.img_0306
}
```

Recommended menu title convention:
- `Gentoo Dracut (persist on nvme0n1p5 0306)`

Example append command:

```bash
cat <<'EOF' | sudo tee -a /0p3/boot/grub/custom.cfg >/dev/null
menuentry 'Gentoo Dracut (persist on nvme0n1p5 0306)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0306 \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0306 \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/initramfs_squash_sda1-x86_64.img_0306
}
EOF
```

Notes:
- This setup uses stock dracut behavior.
- `modprobe.blacklist=hp_bioscfg` is included to avoid a kernel issue seen on this platform.

`hp_bioscfg` is HP's BIOS configuration driver (sysfs interface under `/sys/class/firmware_attributes/hp-bioscfg/`). If BIOS configuration from Linux is not needed, blacklisting is acceptable.

After editing, validate syntax if available:

```bash
sudo grub-script-check /0p3/boot/grub/grub.cfg
```

On minimal live environments this tool may be absent. In that case, do a careful visual check and keep your backup file for rollback.

## 7. Prepare required overlay directories on persistent volume

Dracut overlay mode expects specific directories on the unlocked persistent filesystem.

Mount and create them:

```bash
sudo mount "/dev/mapper/$CRYPT_NAME" /mnt
sudo mkdir -p /mnt/persistent/upper
sudo mkdir -p /mnt/persistent/work
```

If `enc` is already open and mounted (for example at `/run/enc`), you can use that mountpoint directly:

```bash
sudo mkdir -p /run/enc/persistent/upper
sudo mkdir -p /run/enc/persistent/work
```

Meaning:
- `upper`: writable OverlayFS layer
- `work`: OverlayFS work directory (must be on same filesystem as `upper`)

## 8. Seed user/account data into overlay (optional but practical)

If you want existing users/passwords/SSH config available immediately, pre-seed files into `upper`:

```bash
sudo mkdir -p /mnt/persistent/upper/etc
sudo cp -av /etc/shadow /mnt/persistent/upper/etc/shadow
sudo cp -av /etc/passwd /mnt/persistent/upper/etc/passwd
sudo cp -av /etc/group /mnt/persistent/upper/etc/group

sudo mkdir -p /mnt/persistent/upper/home/kiel
sudo chown -R kiel:kiel /mnt/persistent/upper/home/kiel
sudo cp -av /home/kiel/.ssh /mnt/persistent/upper/home/kiel
sudo cp -av /home/kiel/.xinitrc /mnt/persistent/upper/home/kiel
sudo cp -av /home/kiel/.bashrc /mnt/persistent/upper/home/kiel
```

Finally unmount:

```bash
sudo umount /mnt
```

## 9. Reboot and test

1. Reboot and select `Gentoo Dracut (persist on nvme0n1p5)` from GRUB.
2. Unlock LUKS when prompted.
3. Verify persistence by creating a test file and rebooting again.
4. Keep older `_MMDD` artifacts for rollback until the new boot path is validated.
