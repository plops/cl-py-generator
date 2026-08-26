# Install Gentoo OpenRC Image on HP Z6

This document records the HP Z6 OpenRC squashfs deployments from April 27,
2026 through August 26, 2026.

- April 27, 2026: initial `0427` squashfs deployment.
- April 28, 2026: new dated `0428` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260428/`.
- June 25, 2026: new dated `0625` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260625/`.
- August 26, 2026: new dated `0826` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260826/`. The old `0427` slot was removed
  for space; the running `0625` slot was preserved.

## Stable Device References

Do not rely on `/dev/nvme0...` and `/dev/nvme1...` names here. They can swap
between reboots. Use `/dev/disk/by-id/` links as the canonical references.

Relevant stable links from:

```bash
ls -l /dev/disk/by-id/ | grep nvme
```

Artifact disk and partitions:

```text
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part5
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part6
```

These correspond to:

- `/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3`
  `label=gentoo`
  `UUID=4f708c84-185d-437b-a03a-7a565f598a23`
  Btrfs artifact partition with GRUB and `/boot/0428`, `/boot/0608`,
  `/boot/0625`, `/boot/0826`.
- `/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part5`
  `UUID=0d7c5e23-6bab-4dce-b744-a5d61d497aca`
  LUKS partition used for the persistent overlay.
- `/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part6`
  `label=gentoo_orc`
  `UUID=19b4ec1e-0403-4820-98e6-4ed57ba819f0`
  Direct OpenRC disk install fallback.

During the August 26 work, those stable links resolved to `/dev/nvme0n1p3`,
`/dev/nvme0n1p5`, and `/dev/nvme0n1p6`. They had resolved to `nvme1n1` during
earlier work, so the by-id links must remain the source of truth.

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

For the April 28 and August 26 deployments, it started read-only and was
temporarily remounted read-write:

```bash
sudo mount -o remount,rw /run/initramfs/live
```

Return it to read-only after the copy and GRUB update:

```bash
sudo mount -o remount,ro /run/initramfs/live
```

## Current Boot State

Current kernel command line while installing `0826`:

```text
BOOT_IMAGE=/boot/0625/vmlinuz root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 rd.live.dir=/boot/0625 rd.live.squashimg=gentoo.squashfs rd.live.ram=1 rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc rd.overlay=/dev/mapper/enc:persistent rd.live.overlay.overlayfs=1 pcie_aspm=off acpi_mask_gpe=0x08 modprobe.blacklist=hp_bioscfg
```

This confirms that `0625`, not `0624`, was the running slot. Both the
user-named `0624` slot and the detected running `0625` slot were protected
during cleanup; no `0624` directory existed.

## April 27 Deployment

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260427/
```

Installed target:

```text
/run/initramfs/live/boot/0427/
```

The `0427` slot remained intact after the April 28 correction. Its active files
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

On August 26, only 2.1 GiB was free, which was insufficient for the 2.3 GB new
NV squashfs plus its supporting files. The `0427` directory, including the two
`.before-20260428` files, was removed and its GRUB entry was deleted. This
freed about 2.1 GiB and left `0428`, `0608`, and the running `0625` as dated
fallbacks. The deleted slot was not moved to trash and requires its original
artifacts or another backup to recover.

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

## June 25 Deployment

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260625/
```

Create the new dated folder:

```bash
sudo mkdir -p /run/initramfs/live/boot/0625
```

Copy the new HP Z6 artifacts into `0625`:

```bash
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260625/gentoo.squashfs_nv \
  /run/initramfs/live/boot/0625/gentoo.squashfs
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260625/vmlinuz \
  /run/initramfs/live/boot/0625/vmlinuz
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260625/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/0625/initramfs_squash_sda1-x86_64.img
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260625/packages.txt \
  /run/initramfs/live/boot/0625/packages.txt
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260625/packages.tsv \
  /run/initramfs/live/boot/0625/packages.tsv
sync
```

Installed files:

```text
/run/initramfs/live/boot/0625/gentoo.squashfs
/run/initramfs/live/boot/0625/vmlinuz
/run/initramfs/live/boot/0625/initramfs_squash_sda1-x86_64.img
/run/initramfs/live/boot/0625/packages.txt
/run/initramfs/live/boot/0625/packages.tsv
```

Observed sizes and mtimes:

```text
/run/initramfs/live/boot/0625/gentoo.squashfs 2095058944 bytes 2026-06-25 04:43
/run/initramfs/live/boot/0625/vmlinuz 26423808 bytes 2026-06-25 04:43
/run/initramfs/live/boot/0625/initramfs_squash_sda1-x86_64.img 13810908 bytes 2026-06-25 04:43
/run/initramfs/live/boot/0625/packages.txt 52879 bytes 2026-06-25 04:43
/run/initramfs/live/boot/0625/packages.tsv 19556 bytes 2026-06-25 04:43
```

Verification checksums:

```text
9856ff825356d44fe0fe94439272e79fbec46fac31309dd5de84c45dc9df2ca2  /run/initramfs/live/boot/0625/gentoo.squashfs
4769d97df83de7bceac637f22d422bcef63a256568fe67c56b08735e1b140795  /run/initramfs/live/boot/0625/vmlinuz
4f85afb903e2e152f2c72dc74ffcd0fd32129eddad7f8840495d3a7fb17b9a64  /run/initramfs/live/boot/0625/initramfs_squash_sda1-x86_64.img
```

## August 26 Deployment

Source build:

```bash
/dev/shm/gentoo-z6-min-openrc_20260826/
```

The bundle was exported from `gentoo-z6-min-openrc:latest`, container image ID
`sha256:712140a0f697ebfb1bc94b95022b033ee51921d596f39afe388a261b64e0b717`,
created on August 25 at 15:07:11 UTC. Docker history showed the direct
`COPY config/reverse-ssh-*.initd` layers from this build project:

```text
/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc
```

The alternative Common Lisp Dockerfile generator under
`/home/kiel/stage/cl-cl-generator/example/05_dockerfile_meta/source01/examples/01_gentoo`
did not create this image. It uses generated heredoc layers instead of the
direct copy layers present in the container history.

The repository was at commit
`230dd59314a3555bab18103ccdf5c285fb4f8d1d`. The files `openrc/Dockerfile` and
`openrc/config/config6.18.18` had local modifications, so the commit alone is
not a complete description of the build input.

The HP Z6 uses `gentoo.squashfs_nv`; `gentoo.squashfs_e14` is for the ThinkPad
E14 and was not installed. The installed filesystem is Squashfs 4.0 with zstd
compression, created on August 25 at 15:06:24 UTC. The build used
`gentoo/stage3:nomultilib-20260824`, `gentoo/portage:20260824`, and the amd64
23.0 no-multilib profile.

The kernel is built from `sys-kernel/gentoo-sources-6.18.41`, using
`openrc/config/config6.18.18` as its starting configuration. Its release is
`6.18.41-gentoo-dist`; it was built with GCC 15.3.0 and GNU ld 2.46.1 on
August 25. The matching dracut initramfs copies the squashfs into RAM.

Create the target and copy the HP Z6 artifacts:

```bash
sudo mkdir /run/initramfs/live/boot/0826
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260826/gentoo.squashfs_nv \
  /run/initramfs/live/boot/0826/gentoo.squashfs
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260826/vmlinuz \
  /run/initramfs/live/boot/0826/vmlinuz
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260826/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/0826/initramfs_squash_sda1-x86_64.img
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260826/packages.txt \
  /run/initramfs/live/boot/0826/packages.txt
sudo cp -av /dev/shm/gentoo-z6-min-openrc_20260826/packages.tsv \
  /run/initramfs/live/boot/0826/packages.tsv
sync
```

A human-readable provenance file was installed alongside the artifacts:

```text
/run/initramfs/live/boot/0826/SOURCE.txt
```

It records the artifact bundle, build project and commit, dirty-build warning,
container image ID, Gentoo base snapshots, kernel source and toolchain, and
all installed-file hashes.

Observed sizes and mtimes:

```text
/run/initramfs/live/boot/0826/gentoo.squashfs 2398760960 bytes 2026-08-26 09:11
/run/initramfs/live/boot/0826/vmlinuz 20238848 bytes 2026-08-26 09:11
/run/initramfs/live/boot/0826/initramfs_squash_sda1-x86_64.img 13785945 bytes 2026-08-26 09:11
/run/initramfs/live/boot/0826/packages.txt 52424 bytes 2026-08-26 09:11
/run/initramfs/live/boot/0826/packages.tsv 19375 bytes 2026-08-26 09:11
```

Verification checksums:

```text
c960454e8ca01cbab3497de3bdbaa4f23b57d406a96b420d9002c383160faedc  /run/initramfs/live/boot/0826/gentoo.squashfs
8ecdfa5c8c5e97664ad9bcaa35711d95e888a2064c13c25aa33c849d396eb795  /run/initramfs/live/boot/0826/vmlinuz
6bf3f8a299c3ba4ec4c95b36e63762949f1f6bd18da490300eca0d747c6669cf  /run/initramfs/live/boot/0826/initramfs_squash_sda1-x86_64.img
cdbe3d3a6224602d8f86238da99dd68434813d9bbe4f5a861a2b07a8f32ea5d5  /run/initramfs/live/boot/0826/packages.txt
0869d2c073b155c0d685a2ffff3dc4ec1132d1b0d8e65e7345ac5e767a80fb7e  /run/initramfs/live/boot/0826/packages.tsv
```

## GRUB Update

Before editing, back up the existing custom GRUB config:

```bash
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-0826
```

Install this entry before the older entries:

```grub
menuentry 'Gentoo Dracut (persist on luks overlay 0826 OpenRC NV folder)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/0826/vmlinuz \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/boot/0826 \
      rd.live.squashimg=gentoo.squashfs \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      pcie_aspm=off acpi_mask_gpe=0x08 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/amd-uc.img /boot/0826/initramfs_squash_sda1-x86_64.img
}
```

Important details:

- `rd.live.dir=/boot/0826` and `rd.live.squashimg=gentoo.squashfs` must match
  the dated folder layout.
- `pcie_aspm=off` remains in place for the HP Z6 ASPM and ACPI boot issues.
- `acpi_mask_gpe=0x08` is retained from the known-good `0625` command line.
- The entry preloads `/boot/amd-uc.img` before the dated initramfs.

The `0625`, `0608`, and `0428` entries remain as dated fallbacks. The deleted
`0427` slot's entry was removed so GRUB does not advertise a missing payload.

## Expected GRUB Choice

Select this on the next reboot:

```text
Gentoo Dracut (persist on luks overlay 0826 OpenRC NV folder)
```

Fallbacks still available:

- `Gentoo Dracut (persist on luks overlay 0625 OpenRC NV folder)`
- `Gentoo Dracut (persist on luks overlay 0608 OpenRC NV folder)`
- `Gentoo Dracut (persist on luks overlay 0428 OpenRC NV folder)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV debug minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV rescue minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV shell bare)`
- `Gentoo OpenRC disk`

## Verification Commands

Run these before reboot:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
ls -lh /run/initramfs/live/boot/0608
ls -lh /run/initramfs/live/boot/0625
ls -lh /run/initramfs/live/boot/0826
grep -n "0826\\|0625\\|0608 OpenRC NV folder\\|0428\\|Gentoo OpenRC disk" \
  /run/initramfs/live/boot/grub/custom.cfg
sha256sum /run/initramfs/live/boot/0826/gentoo.squashfs \
  /dev/shm/gentoo-z6-min-openrc_20260826/gentoo.squashfs_nv
sha256sum /run/initramfs/live/boot/0826/vmlinuz \
  /dev/shm/gentoo-z6-min-openrc_20260826/vmlinuz
sha256sum /run/initramfs/live/boot/0826/initramfs_squash_sda1-x86_64.img \
  /dev/shm/gentoo-z6-min-openrc_20260826/initramfs_squash_sda1-x86_64.img
sha256sum /run/initramfs/live/boot/0826/packages.txt \
  /dev/shm/gentoo-z6-min-openrc_20260826/packages.txt
sha256sum /run/initramfs/live/boot/0826/packages.tsv \
  /dev/shm/gentoo-z6-min-openrc_20260826/packages.tsv
sync
```

Each pair of checksum lines must be identical. After verification, return the
artifact filesystem to read-only and confirm the `ro` mount option:

```bash
sudo mount -o remount,ro /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

On August 26 all checksum pairs matched, the final checks passed, and the
artifact filesystem was returned to read-only. About 1.9 GiB remained free;
`df` rounded the utilization display to 100%.

## Rollback

To remove only the August 26 deployment and restore the pre-install GRUB file:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -rf /run/initramfs/live/boot/0826
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg.before-0826 \
  /run/initramfs/live/boot/grub/custom.cfg
sync
sudo mount -o remount,ro /run/initramfs/live
```

If the new entry fails, reboot and select `0625`, `0608`, `0428`, or the direct
`Gentoo OpenRC disk` entry. Do not remove `0625` until `0826` has booted and
been validated independently.
