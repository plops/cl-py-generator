# Install Gentoo OpenRC Image on HP Z6

This document records the April 27, 2026 HP Z6 OpenRC squashfs deployment.

The current workstation has two Gentoo boot styles on disk:

- `/dev/nvme0n1p6`, label `gentoo_orc`, UUID `19b4ec1e-0403-4820-98e6-4ed57ba819f0`: the currently booted from-scratch OpenRC install.
- `/dev/nvme0n1p3`, label `gentoo`, UUID `4f708c84-185d-437b-a03a-7a565f598a23`: the older Btrfs partition that still contains GRUB, the squashfs artifacts, and `/boot/0424`.

Do not use the current root `/boot` for this squashfs deployment. The relevant target is the Btrfs artifact partition `/dev/nvme0n1p3`.

## Deployment Summary

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260427/
```

Installed target:

```bash
/dev/nvme0n1p3:/boot/0427/
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

## Verified Partition Layout

Relevant identifiers:

```text
/dev/nvme0n1p3  btrfs        label=gentoo      UUID=4f708c84-185d-437b-a03a-7a565f598a23
/dev/nvme0n1p5  crypto_LUKS                  UUID=0d7c5e23-6bab-4dce-b744-a5d61d497aca
/dev/nvme0n1p6  ext4         label=gentoo_orc  UUID=19b4ec1e-0403-4820-98e6-4ed57ba819f0
```

Current kernel command line during installation:

```text
BOOT_IMAGE=/boot/0424/vmlinuz root=UUID=19b4ec1e-0403-4820-98e6-4ed57ba819f0
```

This confirms the machine was booted from the direct OpenRC install while the squashfs artifacts were installed onto `/dev/nvme0n1p3`.

## Mount Target

The artifact partition was mounted at:

```bash
/tmp/gentoo_artifacts_probe
```

Read-only inspection:

```bash
sudo mount -o ro /dev/nvme0n1p3 /tmp/gentoo_artifacts_probe
```

Write phase:

```bash
sudo mount -o remount,rw /tmp/gentoo_artifacts_probe
```

Free space after installation:

```text
/dev/nvme0n1p3  196G  186G  8.3G  96% /tmp/gentoo_artifacts_probe
```

## Copy Commands Used

Create the date-based folder:

```bash
sudo mkdir -p /tmp/gentoo_artifacts_probe/boot/0427
```

Copy the new HP Z6 artifacts:

```bash
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260427/gentoo.squashfs_nv \
  /tmp/gentoo_artifacts_probe/boot/0427/gentoo.squashfs
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260427/vmlinuz \
  /tmp/gentoo_artifacts_probe/boot/0427/vmlinuz
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260427/initramfs_squash_sda1-x86_64.img \
  /tmp/gentoo_artifacts_probe/boot/0427/initramfs_squash_sda1-x86_64.img
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260427/packages.txt \
  /tmp/gentoo_artifacts_probe/boot/0427/packages.txt
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260427/packages.tsv \
  /tmp/gentoo_artifacts_probe/boot/0427/packages.tsv
```

Observed installed sizes:

```text
2.1G  /boot/0427/gentoo.squashfs
20M   /boot/0427/vmlinuz
13M   /boot/0427/initramfs_squash_sda1-x86_64.img
49K   /boot/0427/packages.txt
18K   /boot/0427/packages.tsv
```

## GRUB Update

Before editing, the existing custom GRUB config was backed up:

```bash
sudo cp -a \
  /tmp/gentoo_artifacts_probe/boot/grub/custom.cfg \
  /tmp/gentoo_artifacts_probe/boot/grub/custom.cfg.before-0427
```

The new entry appended to `/boot/grub/custom.cfg` on `/dev/nvme0n1p3` is:

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

    initrd /boot/0427/initramfs_squash_sda1-x86_64.img
}
```

The `pcie_aspm=off` kernel option is included to work around the ASPM and ACPI
errors seen on boot:

```text
[ 1379.718135] igc 0000:41:00.0: can't disable ASPM; OS doesn't have ASPM control
[ 1386.880909] ACPI BIOS Error (bug): Could not resolve symbol [\_SB.PCI1.GPPB.UP00.DP60.XH00], AE_NOT_FOUND (20250807/psargs-332)
[ 1386.880931] ACPI Error: Aborting method \_GPE._L08 due to previous error (AE_NOT_FOUND) (20250807/psparse-529)
[ 1386.880946] ACPI Error: AE_NOT_FOUND, while evaluating GPE method [_L08] (20250807/evgpe-511)
```

Important detail: because the squashfs is inside `/boot/0427`, the entry uses:

```text
rd.live.dir=/boot/0427
rd.live.squashimg=gentoo.squashfs
```

This differs from the older flat layout, which used `rd.live.dir=/` and filenames such as `gentoo.squashfs_0407`.

## Existing Fallbacks

The following existing GRUB entries were left untouched:

- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV debug minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV rescue minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV shell bare)`
- `Gentoo OpenRC disk`, which boots `/boot/0424/vmlinuz` with root UUID `19b4ec1e-0403-4820-98e6-4ed57ba819f0`

## Verification Commands

Run these before reboot:

```bash
ls -lh /tmp/gentoo_artifacts_probe/boot/0427
ls -lh /tmp/gentoo_artifacts_probe/boot/grub/custom.cfg.before-0427
tail -n 35 /tmp/gentoo_artifacts_probe/boot/grub/custom.cfg
df -h /tmp/gentoo_artifacts_probe
sync
```

Expected new GRUB selection:

```text
Gentoo Dracut (persist on nvme0n1p5 0427 OpenRC NV folder)
```

If the new entry fails, reboot and select either the `0407` squashfs entry or the `Gentoo OpenRC disk` entry.

## Rollback

To remove only the 0427 deployment:

```bash
sudo mount -o remount,rw /tmp/gentoo_artifacts_probe
sudo rm -rf /tmp/gentoo_artifacts_probe/boot/0427
sudo cp -a \
  /tmp/gentoo_artifacts_probe/boot/grub/custom.cfg.before-0427 \
  /tmp/gentoo_artifacts_probe/boot/grub/custom.cfg
sync
```
