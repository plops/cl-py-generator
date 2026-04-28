# Install Gentoo OpenRC Image on HP Z6

This document records the HP Z6 OpenRC squashfs deployments from April 27-28,
2026.

- April 27, 2026: initial `0427` squashfs deployment.
- April 28, 2026: new dated `0428` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260428/`.

## Stable Device References

Do not rely on `/dev/nvme0...` and `/dev/nvme1...` names here. They can swap
between reboots. Use `/dev/disk/by-id/` links as the canonical references.

Relevant stable links from:

```bash
ls -l /dev/disk/by-id/ | grep nvme
```

Artifact disk and partitions:

```text
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3  -> ../../nvme1n1p3
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part5  -> ../../nvme1n1p5
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part6  -> ../../nvme1n1p6
```

These correspond to:

- `/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3`
  `label=gentoo`
  `UUID=4f708c84-185d-437b-a03a-7a565f598a23`
  Btrfs artifact partition with GRUB and `/boot/0427`, `/boot/0428`.
- `/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part5`
  `UUID=0d7c5e23-6bab-4dce-b744-a5d61d497aca`
  LUKS partition used for the persistent overlay.
- `/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part6`
  `label=gentoo_orc`
  `UUID=19b4ec1e-0403-4820-98e6-4ed57ba819f0`
  Direct OpenRC disk install fallback.

During this April 28 work, those stable links happened to resolve to
`/dev/nvme1n1p3`, `/dev/nvme1n1p5`, and `/dev/nvme1n1p6`, but the by-id links
should be treated as the source of truth in future updates.

## Mount Target

Do not use the currently booted root `/boot` for squashfs deployment work. The
artifact partition is mounted at:

```text
/run/initramfs/live
```

Check it first:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

For April 28, it started read-only and was temporarily remounted read-write:

```bash
sudo mount -o remount,rw /run/initramfs/live
```

Return it to read-only after the copy and GRUB update:

```bash
sudo mount -o remount,ro /run/initramfs/live
```

## Current Boot State

Current kernel command line before rebooting into `0428`:

```text
BOOT_IMAGE=/boot/0427/vmlinuz root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 rd.live.dir=/boot/0427 rd.live.squashimg=gentoo.squashfs rd.live.ram=1 rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc rd.overlay=/dev/mapper/enc:persistent rd.live.overlay.overlayfs=1 pcie_aspm=off modprobe.blacklist=hp_bioscfg
```

This means the system was still running the `0427` entry while the new `0428`
payload was installed.

## April 27 Deployment

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260427/
```

Installed target:

```text
/run/initramfs/live/boot/0427/
```

The `0427` slot remains intact after the April 28 correction. Its active files
were restored from the `.before-20260428` backups:

```text
/run/initramfs/live/boot/0427/gentoo.squashfs
/run/initramfs/live/boot/0427/vmlinuz
/run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img
/run/initramfs/live/boot/0427/packages.txt
/run/initramfs/live/boot/0427/packages.tsv
```

Restored checksums:

```text
1c7079884537fcb8a968b815a452b7dbafed0fee1e41dd129c57ace2c5950a67  /run/initramfs/live/boot/0427/vmlinuz
1ac0af63181724b548cfd2213f08f20734897885eafb327b70b750ad4b70b646  /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img
```

## April 28 Deployment

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260428/
```

Create the new dated folder:

```bash
sudo mkdir -p /run/initramfs/live/boot/0428
```

Copy the new HP Z6 artifacts into `0428`:

```bash
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/gentoo.squashfs_nv \
  /run/initramfs/live/boot/0428/gentoo.squashfs
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/vmlinuz \
  /run/initramfs/live/boot/0428/vmlinuz
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/0428/initramfs_squash_sda1-x86_64.img
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/packages.txt \
  /run/initramfs/live/boot/0428/packages.txt
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/packages.tsv \
  /run/initramfs/live/boot/0428/packages.tsv
sync
```

Installed files:

```text
/run/initramfs/live/boot/0428/gentoo.squashfs
/run/initramfs/live/boot/0428/vmlinuz
/run/initramfs/live/boot/0428/initramfs_squash_sda1-x86_64.img
/run/initramfs/live/boot/0428/packages.txt
/run/initramfs/live/boot/0428/packages.tsv
```

Observed sizes and mtimes:

```text
/run/initramfs/live/boot/0428/gentoo.squashfs 2208034816 bytes 2026-04-28 07:37
/run/initramfs/live/boot/0428/vmlinuz 20189696 bytes 2026-04-28 07:37
/run/initramfs/live/boot/0428/initramfs_squash_sda1-x86_64.img 13578769 bytes 2026-04-28 07:37
/run/initramfs/live/boot/0428/packages.txt 50168 bytes 2026-04-28 07:37
/run/initramfs/live/boot/0428/packages.tsv 18350 bytes 2026-04-28 07:37
```

Verification checksums:

```text
2b795b555752c7e4cd061f06104e94112fd50c69196f4dc4ab4100180782488c  /run/initramfs/live/boot/0428/gentoo.squashfs
53a3d0d6e5a91e157faab0789db6b95c25decdbf6a6e118a1ef802e3e3d53464  /run/initramfs/live/boot/0428/vmlinuz
d49f62bc3e1ee2117c1e9a2e4fac106a8310a7bb7100553bcd4aae3f9eb0ff42  /run/initramfs/live/boot/0428/initramfs_squash_sda1-x86_64.img
```

## GRUB Update

Before editing, back up the existing custom GRUB config:

```bash
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-0428
```

Append this new entry:

```grub
menuentry 'Gentoo Dracut (persist on luks overlay 0428 OpenRC NV folder)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/0428/vmlinuz \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/boot/0428 \
      rd.live.squashimg=gentoo.squashfs \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      pcie_aspm=off \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/amd-uc.img /boot/0428/initramfs_squash_sda1-x86_64.img
}
```

Important details:

- `rd.live.dir=/boot/0428` and `rd.live.squashimg=gentoo.squashfs` must match
  the dated folder layout.
- `pcie_aspm=off` remains in place for the HP Z6 ASPM and ACPI boot issues.
- The entry preloads `/boot/amd-uc.img` before the dated initramfs.

The older `0427` GRUB entry was left in place as a fallback.

## Expected GRUB Choice

Select this on the next reboot:

```text
Gentoo Dracut (persist on luks overlay 0428 OpenRC NV folder)
```

Fallbacks still available:

- `Gentoo Dracut (persist on nvme0n1p5 0427 OpenRC NV folder)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV debug minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV rescue minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV shell bare)`
- `Gentoo OpenRC disk`

## Verification Commands

Run these before reboot:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
ls -lh /run/initramfs/live/boot/0427
ls -lh /run/initramfs/live/boot/0428
grep -n "0427 OpenRC NV folder\\|0428\\|Gentoo OpenRC disk" \
  /run/initramfs/live/boot/grub/custom.cfg
sha256sum /run/initramfs/live/boot/0428/gentoo.squashfs \
  /dev/shm/gentoo-z6-min-openrc_20260428/gentoo.squashfs_nv
sha256sum /run/initramfs/live/boot/0428/vmlinuz \
  /dev/shm/gentoo-z6-min-openrc_20260428/vmlinuz
sha256sum /run/initramfs/live/boot/0428/initramfs_squash_sda1-x86_64.img \
  /dev/shm/gentoo-z6-min-openrc_20260428/initramfs_squash_sda1-x86_64.img
sync
```

## Rollback

To remove only the April 28 deployment:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -rf /run/initramfs/live/boot/0428
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg.before-0428 \
  /run/initramfs/live/boot/grub/custom.cfg
sudo mount -o remount,ro /run/initramfs/live
sync
```

If the new entry fails, reboot and select either the `0427` squashfs entry or
the `Gentoo OpenRC disk` entry.
