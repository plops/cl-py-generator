# Install Gentoo OpenRC Image on HP Z6 (using existing GRUB)

This guide describes how to install the new HP Z6 OpenRC live image on the HP Z6 workstation by copying new boot artifacts to the existing Gentoo artifacts partition and adding a new GRUB entry.

It follows the same partition layout already in use on this machine:

- Boot artifacts and squashfs images stay on `/dev/nvme0n1p3` (label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`)
- Persistent writable storage stays on `/dev/nvme0n1p5` via LUKS (`UUID=0d7c5e23-6bab-4dce-b744-a5d61d497aca`) and `/dev/mapper/enc`
- Existing GRUB installation is reused

Current deployment target:

- New build directory: `/dev/shm/gentoo-z6-min-openrc_20260323/`
- New HP Z6 squashfs source: `/dev/shm/gentoo-z6-min-openrc_20260323/gentoo.squashfs_nv`
- New version suffix: `0323`
- Current booted fallback that must be kept: `0311`

## 0. Current machine state

On this HP Z6, the relevant runtime state is:

- `/run/initramfs/live` is mounted from `/dev/nvme0n1p3`
- `/proc/cmdline` currently uses:
  - `root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23`
  - `rd.live.squashimg=gentoo.squashfs_0311`
  - `rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca`
  - `rd.overlay=/dev/mapper/enc:persistent`
- Current GRUB custom entry uses the `0311` kernel/initramfs/squashfs set

Currently present squashfs files on `/dev/nvme0n1p3`:

- `gentoo.squashfs_0225` about `1.4G`
- `gentoo.squashfs_0302` about `2.4G`
- `gentoo.squashfs_0306` about `2.7G`
- `gentoo.squashfs_0309` about `2.6G`
- `gentoo.squashfs_0311` about `2.4G`

There is currently about `19G` free on `/dev/nvme0n1p3`, so there should be enough space to keep all existing squashfs files and add the new `0323` set.

## 1. Preflight checks

Run these first:

```bash
lsblk -f
df -h /run/initramfs/live
cat /proc/cmdline
```

Expected identifiers on this machine:

- Artifacts partition: `/dev/nvme0n1p3`, label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`
- Persistent LUKS partition: `/dev/nvme0n1p5`, UUID `0d7c5e23-6bab-4dce-b744-a5d61d497aca`
- Open persistence mapper: `/dev/mapper/enc`

Inspect the existing artifacts and GRUB entry:

```bash
ls -lh /run/initramfs/live/gentoo.squashfs_*
ls -lh /run/initramfs/live/boot/vmlinuz_* /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_* 2>/dev/null
sed -n '1,200p' /run/initramfs/live/boot/grub/custom.cfg
```

## 2. Remount the artifacts partition read-write

The live artifacts partition is usually mounted read-only while booted from it, so remount it before copying:

```bash
sudo mount -o remount,rw /run/initramfs/live
mount | rg '/run/initramfs/live'
```

Expected result:

- `/run/initramfs/live` remains mounted from `/dev/nvme0n1p3`
- mount flags now include `rw`

## 3. Verify the new build output

Check that the expected OpenRC HP Z6 artifacts exist:

```bash
ls -lh /dev/shm/gentoo-z6-min-openrc_20260323/
ls -lh \
  /dev/shm/gentoo-z6-min-openrc_20260323/gentoo.squashfs_nv \
  /dev/shm/gentoo-z6-min-openrc_20260323/vmlinuz \
  /dev/shm/gentoo-z6-min-openrc_20260323/initramfs_squash_sda1-x86_64.img
```

Important for HP Z6:

- Use `gentoo.squashfs_nv`
- Do not use the E14-specific `gentoo.squashfs_e14`

## 4. Storage policy: what to keep and what to delete only if needed

Recommended keep set:

- Keep `gentoo.squashfs_0311` because that is the currently booted known-good fallback
- Keep the older squashfs files as long as free space remains comfortable
- Keep matching kernel and initramfs files for the versions you keep bootable in GRUB

With about `19G` free on `/dev/nvme0n1p3`, the current recommendation is:

- Keep all existing squashfs files
- Add the new `0323` files without deleting anything first

If the copy later fails because space becomes tight, delete in this order:

1. `gentoo.squashfs_0225` and its matching `boot/vmlinuz_0225` and `boot/initramfs_squash_sda1-x86_64.img_0225`
2. `gentoo.squashfs_0302` and its matching `boot/vmlinuz_0302` and `boot/initramfs_squash_sda1-x86_64.img_0302`
3. `gentoo.squashfs_0306` and its matching `boot/vmlinuz_0306` and `boot/initramfs_squash_sda1-x86_64.img_0306`
4. `gentoo.squashfs_0309` and its matching `boot/vmlinuz_0309` and `boot/initramfs_squash_sda1-x86_64.img_0309`

Do not delete:

- `gentoo.squashfs_0311`
- `boot/vmlinuz_0311`
- `boot/initramfs_squash_sda1-x86_64.img_0311`

Optional cleanup commands if needed later:

```bash
sudo rm -v /run/initramfs/live/gentoo.squashfs_0225
sudo rm -v /run/initramfs/live/boot/vmlinuz_0225
sudo rm -v /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0225
```

Re-check free space after each cleanup step:

```bash
df -h /run/initramfs/live
```

## 5. Copy the new `0323` artifacts

Copy the new OpenRC HP Z6 artifacts with versioned names:

```bash
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260323/gentoo.squashfs_nv \
  /run/initramfs/live/gentoo.squashfs_0323
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260323/vmlinuz \
  /run/initramfs/live/boot/vmlinuz_0323
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260323/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0323
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260323/packages.txt \
  /run/initramfs/live/packages_0323.txt
```

Verify:

```bash
ls -lh /run/initramfs/live/gentoo.squashfs_0323
ls -lh /run/initramfs/live/boot/vmlinuz_0323
ls -lh /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0323
df -h /run/initramfs/live
```

## 6. Add a new GRUB entry and keep `0311` as fallback

Append a new entry to `/run/initramfs/live/boot/grub/custom.cfg` and keep the existing `0311` entry unchanged.

Recommended new entry:

```bash
cat <<'EOF' | sudo tee -a /run/initramfs/live/boot/grub/custom.cfg >/dev/null

menuentry 'Gentoo Dracut (persist on nvme0n1p5 0323 OpenRC NV)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/vmlinuz_0323 \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/ \
      rd.live.squashimg=gentoo.squashfs_0323 \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/initramfs_squash_sda1-x86_64.img_0323
}
EOF
```

Verify that both the old and new entries are present:

```bash
rg -n "0311|0323" /run/initramfs/live/boot/grub/custom.cfg
```

## 7. Final checks before reboot

Run:

```bash
sync
ls -lh /run/initramfs/live/gentoo.squashfs_0311 /run/initramfs/live/gentoo.squashfs_0323
ls -lh /run/initramfs/live/boot/vmlinuz_0311 /run/initramfs/live/boot/vmlinuz_0323
ls -lh /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0311 /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0323
sed -n '1,220p' /run/initramfs/live/boot/grub/custom.cfg
```

Checklist:

- `0311` files still exist
- `0323` files exist and have non-zero size
- `custom.cfg` contains both the existing `0311` entry and the new `0323` entry
- `/dev/mapper/enc` is still the overlay target referenced by the new entry

## 8. Reboot and select the new entry

Reboot the machine:

```bash
sudo reboot
```

At the GRUB menu:

- Select `Gentoo Dracut (persist on nvme0n1p5 0323 OpenRC NV)` for the new image
- Keep using the `0311` entry as rollback fallback if the new image has a regression

## 9. Rollback

If the new image does not boot cleanly:

- Reboot
- Select the existing `0311` menu entry
- Leave the `0323` files on disk for later inspection unless you need the space immediately

If you want to remove the `0323` deployment after a failed test:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -v /run/initramfs/live/gentoo.squashfs_0323
sudo rm -v /run/initramfs/live/boot/vmlinuz_0323
sudo rm -v /run/initramfs/live/boot/initramfs_squash_sda1-x86_64.img_0323
```

Then remove the corresponding `0323` GRUB entry from `/run/initramfs/live/boot/grub/custom.cfg`.
