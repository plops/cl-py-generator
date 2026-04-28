# Install Gentoo OpenRC Image on HP Z6

This document now records two related HP Z6 steps:

- April 27, 2026: initial `0427` squashfs deployment.
- April 28, 2026: in-place `0427` kernel and initramfs refresh from `/dev/shm/gentoo-z6-min-openrc_20260428`.

## Current Layout

The current workstation has two Gentoo boot styles on disk:

- `/dev/nvme1n1p3`, label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`: the Btrfs artifact partition containing GRUB and the squashfs boot payloads.
- `/dev/nvme1n1p5`, UUID `0d7c5e23-6bab-4dce-b744-a5d61d497aca`: the LUKS partition used for the persistent overlay.
- `/dev/nvme1n1p6`, label `gentoo_orc`, UUID `19b4ec1e-0403-4820-98e6-4ed57ba819f0`: the direct OpenRC disk install fallback.

Do not use the current root `/boot` for this squashfs deployment work. The relevant target is the Btrfs artifact partition mounted at `/run/initramfs/live`.

Current kernel command line during the April 28 refresh:

```text
BOOT_IMAGE=/boot/0427/vmlinuz root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 rd.live.dir=/boot/0427 rd.live.squashimg=gentoo.squashfs rd.live.ram=1 rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc rd.overlay=/dev/mapper/enc:persistent rd.live.overlay.overlayfs=1 pcie_aspm=off modprobe.blacklist=hp_bioscfg
```

This confirms the machine was already booted from the `0427` squashfs entry while the kernel and initramfs were refreshed in place.

## April 27, 2026 Deployment Summary

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260427/
```

Installed target:

```bash
/dev/nvme1n1p3:/boot/0427/
```

Installed files:

```text
/boot/0427/gentoo.squashfs
/boot/0427/vmlinuz
/boot/0427/initramfs_squash_sda1-x86_64.img
/boot/0427/packages.txt
/boot/0427/packages.tsv
```

The HP Z6 squashfs source is `gentoo.squashfs_nv`. The E14-specific `gentoo.squashfs_e14` was not used.

The OpenRC image also carries the host bring-up helpers as installed files:

```text
/home/kiel/activate
/home/kiel/start2
```

These are built from the repo-managed sources in `example/110_gentoo/openrc/config/`
so the image no longer depends on ad-hoc copies under `/home/kiel`.

## Host Bring-Up Notes

On the HP Z6, `kiel` had to be in the `input` group for `startx` to get working
keyboard and mouse input. The image build now creates that group if needed and
adds `kiel` to it in the Dockerfile.

Compared with the Lenovo ThinkPad E14 Gen 2 setup:

- `/home/kiel/activate` is now stored in the repo and uses `/usr/local/share/openrc-host-config` as its primary config source instead of the stale `docker_min_hpz6_openrc` fallback path.
- `/home/kiel/start2` is now stored in the repo and keeps the HP Z6 network bring-up defaults (`eno1`, `enp65s0`, `r8169`, `igc`) while tolerating other machines by skipping missing interfaces and switching to `mt7921e` module loading on ThinkPad E14 hardware.
- Reverse SSH OpenRC services remain enabled by default on the HP Z6 path and are still disabled on ThinkPad E14 hardware by the `activate` script.

## GRUB Entry Used

The `0427` GRUB entry on `/dev/nvme1n1p3` remains:

```grub
menuentry 'Gentoo Dracut (persist on nvme0n1p5 0427 OpenRC NV folder)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/0427/vmlinuz \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/boot/0427 \
      rd.live.squashimg=gentoo.squashfs \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      pcie_aspm=off \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/amd-uc.img /boot/0427/initramfs_squash_sda1-x86_64.img
}
```

Important detail: because the squashfs is inside `/boot/0427`, the entry uses:

```text
rd.live.dir=/boot/0427
rd.live.squashimg=gentoo.squashfs
```

This differs from the older flat layout, which used `rd.live.dir=/` and filenames such as `gentoo.squashfs_0407`.

The `pcie_aspm=off` kernel option is included to work around the ASPM and ACPI
errors seen on boot, and the entry preloads the shared AMD microcode image from
`/boot/amd-uc.img` before the `0427` initramfs.

## April 28, 2026 Kernel And Initramfs Refresh

Source refresh payload:

```bash
/dev/shm/gentoo-z6-min-openrc_20260428/
```

Only the kernel and initramfs were refreshed. The `0427` squashfs and GRUB
entry were left unchanged.

Observed source files:

```text
/dev/shm/gentoo-z6-min-openrc_20260428/vmlinuz
/dev/shm/gentoo-z6-min-openrc_20260428/initramfs_squash_sda1-x86_64.img
```

Mount state before writing:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
sudo mount -o remount,rw /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

Expected transition:

```text
/dev/nvme1n1p3 /run/initramfs/live btrfs ro,...
/dev/nvme1n1p3 /run/initramfs/live btrfs rw,...
```

Backup and copy commands used:

```bash
sudo cp -av /run/initramfs/live/boot/0427/vmlinuz \
  /run/initramfs/live/boot/0427/vmlinuz.before-20260428
sudo cp -av /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img.before-20260428
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/vmlinuz \
  /run/initramfs/live/boot/0427/vmlinuz
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260428/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img
sync
```

Installed `0427` file state after refresh:

```text
/run/initramfs/live/boot/0427/gentoo.squashfs                          2203664384 bytes  2026-04-27 08:25
/run/initramfs/live/boot/0427/vmlinuz                                    20189696 bytes  2026-04-28 07:37
/run/initramfs/live/boot/0427/vmlinuz.before-20260428                    20128256 bytes  2026-04-27 08:25
/run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img           13578769 bytes  2026-04-28 07:37
/run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img.before-20260428 13404040 bytes  2026-04-27 08:25
```

Verification checksums:

```text
53a3d0d6e5a91e157faab0789db6b95c25decdbf6a6e118a1ef802e3e3d53464  /run/initramfs/live/boot/0427/vmlinuz
53a3d0d6e5a91e157faab0789db6b95c25decdbf6a6e118a1ef802e3e3d53464  /dev/shm/gentoo-z6-min-openrc_20260428/vmlinuz
d49f62bc3e1ee2117c1e9a2e4fac106a8310a7bb7100553bcd4aae3f9eb0ff42  /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img
d49f62bc3e1ee2117c1e9a2e4fac106a8310a7bb7100553bcd4aae3f9eb0ff42  /dev/shm/gentoo-z6-min-openrc_20260428/initramfs_squash_sda1-x86_64.img
```

Because the GRUB entry still points at `/boot/0427/vmlinuz` and
`/boot/0427/initramfs_squash_sda1-x86_64.img`, no `custom.cfg` edit was needed
for the April 28 refresh.

## Existing Fallbacks

The following existing GRUB entries were left untouched:

- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV debug minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV rescue minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV shell bare)`
- `Gentoo OpenRC disk`, which boots `/boot/0424/vmlinuz` with root UUID `19b4ec1e-0403-4820-98e6-4ed57ba819f0`

Expected GRUB selection for the refreshed path:

```text
Gentoo Dracut (persist on nvme0n1p5 0427 OpenRC NV folder)
```

If the refreshed `0427` entry fails, reboot and select either the `0407`
squashfs entry or the `Gentoo OpenRC disk` entry.

## Verification Commands

Run these before reboot:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
ls -lh /run/initramfs/live/boot/0427
sha256sum /run/initramfs/live/boot/0427/vmlinuz \
  /dev/shm/gentoo-z6-min-openrc_20260428/vmlinuz
sha256sum /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img \
  /dev/shm/gentoo-z6-min-openrc_20260428/initramfs_squash_sda1-x86_64.img
sync
```

## Rollback

To roll back only the April 28 kernel/initramfs refresh:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo cp -av /run/initramfs/live/boot/0427/vmlinuz.before-20260428 \
  /run/initramfs/live/boot/0427/vmlinuz
sudo cp -av /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img.before-20260428 \
  /run/initramfs/live/boot/0427/initramfs_squash_sda1-x86_64.img
sync
```

To remove the whole `0427` deployment instead, restore the previous `custom.cfg`
backup from the initial April 27 deployment and delete `/boot/0427`.
