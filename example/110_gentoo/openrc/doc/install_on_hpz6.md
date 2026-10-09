# Install Gentoo OpenRC Image on HP Z6

This document records the HP Z6 OpenRC squashfs deployments from April 27,
2026 through October 9, 2026.

- April 27, 2026: initial `0427` squashfs deployment.
- April 28, 2026: new dated `0428` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260428/`.
- June 25, 2026: new dated `0625` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260625/`.
- August 26, 2026: new dated `0826` deployment from
  `/dev/shm/gentoo-z6-min-openrc_20260826/`. The old `0427` slot was removed
  for space; the running `0625` slot was preserved.
- September 7, 2026: replacement of the existing `0907` deployment from
  `/home/kiel/gentoo-z6-min-openrc_20260907/`. The files were overwritten
  in place to avoid requiring a second full squashfs in the nearly full
  artifact partition.
- September 8, 2026: the existing `0907` deployment was overwritten again
  from the same source directory while running `0826`; the GRUB entry was
  retained and the artifact partition was returned to read-only.
- October 9, 2026: new dated `1009` deployment from
  `/home/kiel/gentoo-z6-min-openrc_20261009/` while running `0907`. The
  artifact partition had only 2.4 GiB free, so the old `0608` slot (2.1 GiB)
  was removed to make room; the running `0907` slot and the `0625`/`0826`
  fallbacks were preserved. A new GRUB entry for `1009` was prepended and the
  `0608` entry was removed.

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
  Btrfs artifact partition with GRUB and `/boot/0625`, `/boot/0826`,
  `/boot/0907`, `/boot/1009`.
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

Current kernel command line while installing `1009`:

```text
BOOT_IMAGE=/boot/0907/vmlinuz root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 rd.live.dir=/boot/0907 rd.live.squashimg=gentoo.squashfs rd.live.ram=1 rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc rd.overlay=/dev/mapper/enc:persistent rd.live.overlay.overlayfs=1 pcie_aspm=off acpi_mask_gpe=0x08 modprobe.blacklist=hp_bioscfg
```

This confirms that `0907` was the running slot during the October 9 work. The
running `0907` slot was protected during cleanup; the old `0608` slot was the
one removed for space.

The command line below was recorded earlier while installing `0826`:

```text
BOOT_IMAGE=/boot/0625/vmlinuz root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 rd.live.dir=/boot/0625 rd.live.squashimg=gentoo.squashfs rd.live.ram=1 rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc rd.overlay=/dev/mapper/enc:persistent rd.live.overlay.overlayfs=1 pcie_aspm=off acpi_mask_gpe=0x08 modprobe.blacklist=hp_bioscfg
```

It confirmed that `0625`, not `0624`, was the running slot at that time.

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

## September 7 Deployment

Source build:

```bash
/home/kiel/gentoo-z6-min-openrc_20260907/
```

The existing `0907` slot was overwritten in place while the machine was
running the known-good `0826` slot. This avoids needing enough free space for
both the old and new squashfs images. The same replacement procedure was
repeated on September 8, 2026 after another `0907` build had been installed
the previous day. The HP Z6 uses
`gentoo.squashfs_nv`; `gentoo.squashfs_e14` was not installed.

Install or replace the HP Z6 artifacts:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-0907
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260907/gentoo.squashfs_nv \
  /run/initramfs/live/boot/0907/gentoo.squashfs
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260907/vmlinuz \
  /run/initramfs/live/boot/0907/vmlinuz
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260907/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/0907/initramfs_squash_sda1-x86_64.img
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260907/packages.txt \
  /run/initramfs/live/boot/0907/packages.txt
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20260907/packages.tsv \
  /run/initramfs/live/boot/0907/packages.tsv
sync
```

Observed installed sizes:

```text
/run/initramfs/live/boot/0907/gentoo.squashfs 1734602752 bytes
/run/initramfs/live/boot/0907/vmlinuz 20111872 bytes
/run/initramfs/live/boot/0907/initramfs_squash_sda1-x86_64.img 13850346 bytes
/run/initramfs/live/boot/0907/packages.txt 53125 bytes
/run/initramfs/live/boot/0907/packages.tsv 19699 bytes
```

Verification checksums:

```text
fac11f249185251378ff2e14b8ff1bc6cddfb0e582d4d602d9125e19dd99c064  /run/initramfs/live/boot/0907/gentoo.squashfs
c5fe9aedb2933fc650112feb10ed885225303c2063170b1ecc298034bbf72b8a  /run/initramfs/live/boot/0907/vmlinuz
e810f55518f7692f0ddf3b58e48c989942c95ce5914de08b12b45881d23cce9e  /run/initramfs/live/boot/0907/initramfs_squash_sda1-x86_64.img
827a7795ed806edf2b85c55c78bb914ee58b4ff6758a7f3829799bd94645ab08  /run/initramfs/live/boot/0907/packages.txt
51373472c3b8d95c0762ec800b47b1f70ed0ab00437696c5dd55171c33273233  /run/initramfs/live/boot/0907/packages.tsv
```

## September 7 GRUB Update

Before editing, back up the existing custom GRUB config:

```bash
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-0907
```

Install this entry before the older entries:

```grub
menuentry 'Gentoo Dracut (persist on luks overlay 0907 OpenRC NV folder)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/0907/vmlinuz \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/boot/0907 \
      rd.live.squashimg=gentoo.squashfs \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      pcie_aspm=off acpi_mask_gpe=0x08 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/amd-uc.img /boot/0907/initramfs_squash_sda1-x86_64.img
}
```

Important details:

- `rd.live.dir=/boot/0907` and `rd.live.squashimg=gentoo.squashfs` must match
  the dated folder layout.
- `pcie_aspm=off` remains in place for the HP Z6 ASPM and ACPI boot issues.
- `acpi_mask_gpe=0x08` is retained from the known-good `0625` command line.
- The entry preloads `/boot/amd-uc.img` before the dated initramfs.

The `0826`, `0625`, `0608`, and `0428` entries remain as dated fallbacks. The
deleted `0427` slot's entry was removed so GRUB does not advertise a missing
payload.

## October 9 Deployment

Source build:

```bash
/home/kiel/gentoo-z6-min-openrc_20261009/
```

The source directory was produced by the project `Dockerfile` and contained
`gentoo.squashfs_nv`, `vmlinuz`, `initramfs_squash_sda1-x86_64.img`,
`packages.txt`, `packages.tsv`, and `copy_files.sh`. The HP Z6 uses
`gentoo.squashfs_nv`.

Unlike the September in-place overwrites, this deployment created a new dated
`1009` slot while the machine was running `0907`. The artifact partition had
only 2.4 GiB free (99% used), so the old `0608` slot (2.1 GiB) was removed to
make room. The running `0907` slot and the `0625` and `0826` fallbacks were
preserved. The `0424` and `0806` slots are small (76 MiB and 25 MiB) and were
left untouched.

Remount read-write, back up GRUB, free space, then create and populate the
slot:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-1009
sudo rm -rf /run/initramfs/live/boot/0608
sudo mkdir -p /run/initramfs/live/boot/1009
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20261009/gentoo.squashfs_nv \
  /run/initramfs/live/boot/1009/gentoo.squashfs
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20261009/vmlinuz \
  /run/initramfs/live/boot/1009/vmlinuz
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20261009/initramfs_squash_sda1-x86_64.img \
  /run/initramfs/live/boot/1009/initramfs_squash_sda1-x86_64.img
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20261009/packages.txt \
  /run/initramfs/live/boot/1009/packages.txt
sudo cp -av /home/kiel/gentoo-z6-min-openrc_20261009/packages.tsv \
  /run/initramfs/live/boot/1009/packages.tsv
sync
```

Observed installed sizes:

```text
/run/initramfs/live/boot/1009/gentoo.squashfs 1759891456 bytes
/run/initramfs/live/boot/1009/vmlinuz 20599296 bytes
/run/initramfs/live/boot/1009/initramfs_squash_sda1-x86_64.img 13966570 bytes
/run/initramfs/live/boot/1009/packages.txt 53135 bytes
/run/initramfs/live/boot/1009/packages.tsv 19708 bytes
```

Verification checksums:

```text
97aad43f54d716dc20b5c4296276d0dd89156011dc946e7c9746912fd3e6e1d7  /run/initramfs/live/boot/1009/gentoo.squashfs
3a797e320e09fcd0fb0c2878676c6cdbbafeb6dccff2a9c8c65f6533768cc320  /run/initramfs/live/boot/1009/vmlinuz
03cceb4cd00fc5cc3139337c2a3d4fc4a0dcbb296421eb0db033b73928f61a94  /run/initramfs/live/boot/1009/initramfs_squash_sda1-x86_64.img
66bfb8ce3a434c4e56feb75247d4537b1c3b0ae01b90ce104a4ad6bb417fa16a  /run/initramfs/live/boot/1009/packages.txt
ebeede5cc77ceb26e50b6b6ae8918ce43aa25bb3b9c9a4df00f7bcebf532e40e  /run/initramfs/live/boot/1009/packages.tsv
```

All five checksum pairs matched the source files in
`/home/kiel/gentoo-z6-min-openrc_20261009/`.

## October 9 GRUB Update

Before editing, the existing custom GRUB config was backed up to
`custom.cfg.before-1009`. The new `1009` entry was prepended before all other
entries, and the stale `0608` entry was removed because its slot was deleted.

Installed entry (first in the file):

```grub
menuentry 'Gentoo Dracut (persist on luks overlay 1009 OpenRC NV folder)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23

    linux /boot/1009/vmlinuz \
      root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
      rd.live.dir=/boot/1009 \
      rd.live.squashimg=gentoo.squashfs \
      rd.live.ram=1 \
      rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
      rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
      rd.overlay=/dev/mapper/enc:persistent \
      rd.live.overlay.overlayfs=1 \
      pcie_aspm=off acpi_mask_gpe=0x08 \
      modprobe.blacklist=hp_bioscfg

    initrd /boot/amd-uc.img /boot/1009/initramfs_squash_sda1-x86_64.img
}
```

After the GRUB edit the partition was synced and returned to read-only. The
remaining dated folder entries are, in order, `0907` (the slot that was
running during the install), `0826`, and `0625`.

## Änderungen in diesem Update (1009 gegenüber 0907)

Die folgende Zusammenfassung beschreibt, was sich im Gentoo-Image mit dem
`1009`-Build gegenüber dem bis dahin laufenden `0907`-Build ändert. Die
exakten Versionsnummern stammen aus dem Build-Log
`openrc/logs/setup01_run_20261009_060806.log` (geschrieben von
`setup01_run_with_log.sh`) im Vergleich zum vorherigen Log
`setup01_run_20260909_064910.log`. Größen-/Build-Zeit-Angaben stammen
zusätzlich aus den mitgelieferten `packages.txt`/`packages.tsv` beider Slots;
diese enthalten selbst keine Versionsnummern.

**Basis-Snapshots (Dockerfile):**

- stage3-Basisimage: `gentoo/stage3:nomultilib-20260907` →
  `gentoo/stage3:nomultilib-20261005`
- Portage-Baum: `gentoo/portage:20260909` → `gentoo/portage:20261008`

Diese beiden Snapshot-Wechsel bringen den Großteil der Toolchain- und
Paketaktualisierungen mit sich, da GCC, glibc und binutils aus dem
stage3-Image stammen und nicht im Dockerfile gepinnt werden.

**Toolchain / GCC (Hauptänderung):**

- `sys-devel/gcc` springt von `15.3.0` auf `16.2.1_p20260926` — also ein
  Major-Versionswechsel auf **GCC 16**. Das deckt sich mit dem Wachstum des
  gcc-Pakets von 301 MiB auf 319 MiB in `packages.txt`.
- `sys-devel/gcc-config` bleibt bei `2.12.2`.
- `llvm-core/llvm` bleibt bei `22.1.8`.
- glibc und binutils werden nicht neu gebaut (keine Versionszeile im Log,
  Größen unverändert); sie kommen unverändert aus dem stage3-Snapshot.

**Kernel:**

- `KVER_PURE` von `6.18.48` auf `6.18.54` angehoben
  (`sys-kernel/gentoo-sources-6.18.54`, Release `6.18.54-gentoo-dist`). Im
  Portage-Baum war zusätzlich bereits `6.18.55` verfügbar, gebaut wurde aber
  die im Dockerfile gepinnte `6.18.54`.
- Dadurch neu gebautes dracut-initramfs und neuer, passend gebauter
  NVIDIA-Kernelmodul-Stack.

**NVIDIA-Treiber:**

- `x11-drivers/nvidia-drivers` von `610.57.04` auf `615.78.08`
  (Paketgröße 1019 → 1083 MiB), passend zum 6.18.54-Kernel.

**Firmware:**

- `sys-kernel/linux-firmware` unverändert bei 1841 MiB.

**Weitere Paketänderungen (Auswahl aus dem Größen-/Build-Diff):**

- `www-client/firefox-bin` 315 → 319 MiB.
- `net-im/signal-desktop-bin` 408 → 361 MiB (kleineres Paket).
- `net-vpn/tailscale` 57 → 50 MiB.
- Container-Stack: `app-containers/docker` 94 → 98 MiB,
  `containerd` 93 → 90 MiB, `docker-cli` 43 → 41 MiB,
  `docker-buildx` 63 → 62 MiB.
- `app-editors/emacs` 172 → 173 MiB, `media-libs/mesa` 25 → 26 MiB,
  `dev-libs/openssl` 19 → 20 MiB, `net-misc/networkmanager` 24 → 25 MiB,
  `sys-apps/portage` 24 → 25 MiB, `sys-devel/mold` 23 → 24 MiB.
- Neu hinzugekommen: `dev-python/tomlkit`.
- Entfernt: `app-text/asciidoc`.
- Python-Build-Tooling aufgeräumt: `dev-python/setuptools` 10 → 7 MiB,
  `dev-python/flit-core` 1 → 0 MiB, `dev-python/uv` 64 → 65 MiB.

Das installierte `gentoo.squashfs` wird dabei nur geringfügig größer
(0907: 1 734 602 752 Bytes, 1009: 1 759 891 456 Bytes, +~24 MiB).

## Expected GRUB Choice

Select this on the next reboot:

```text
Gentoo Dracut (persist on luks overlay 1009 OpenRC NV folder)
```

Fallbacks still available:

- `Gentoo Dracut (persist on luks overlay 0907 OpenRC NV folder)`
- `Gentoo Dracut (persist on luks overlay 0826 OpenRC NV folder)`
- `Gentoo Dracut (persist on luks overlay 0625 OpenRC NV folder)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV debug minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV rescue minimal)`
- `Gentoo Dracut (persist on nvme0n1p5 0407 OpenRC NV shell bare)`
- `Gentoo OpenRC disk`

## Verification Commands

Run these before reboot:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
ls -lh /run/initramfs/live/boot/0625
ls -lh /run/initramfs/live/boot/0826
ls -lh /run/initramfs/live/boot/0907
ls -lh /run/initramfs/live/boot/1009
grep -n "1009\|0907\|0826\|0625\|Gentoo OpenRC disk" \
  /run/initramfs/live/boot/grub/custom.cfg
sha256sum /run/initramfs/live/boot/1009/gentoo.squashfs \
  /home/kiel/gentoo-z6-min-openrc_20261009/gentoo.squashfs_nv
sha256sum /run/initramfs/live/boot/1009/vmlinuz \
  /home/kiel/gentoo-z6-min-openrc_20261009/vmlinuz
sha256sum /run/initramfs/live/boot/1009/initramfs_squash_sda1-x86_64.img \
  /home/kiel/gentoo-z6-min-openrc_20261009/initramfs_squash_sda1-x86_64.img
sha256sum /run/initramfs/live/boot/1009/packages.txt \
  /home/kiel/gentoo-z6-min-openrc_20261009/packages.txt
sha256sum /run/initramfs/live/boot/1009/packages.tsv \
  /home/kiel/gentoo-z6-min-openrc_20261009/packages.tsv
sync
```

Each pair of checksum lines must be identical. After verification, return the
artifact filesystem to read-only and confirm the `ro` mount option:

```bash
sudo mount -o remount,ro /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

On October 9 all five checksum pairs matched, the `1009` entry was first in
`custom.cfg`, the stale `0608` entry was gone, and the artifact filesystem was
returned to read-only.

## Rollback

To remove only the October 9 deployment and restore the pre-install GRUB file:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -rf /run/initramfs/live/boot/1009
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg.before-1009 \
  /run/initramfs/live/boot/grub/custom.cfg
sync
sudo mount -o remount,ro /run/initramfs/live
```

Note that the rollback restores the GRUB file but cannot restore the deleted
`0608` slot, which was removed for space during this deployment.

If the new entry fails, reboot and select `0907`, `0826`, `0625`, or the
direct `Gentoo OpenRC disk` entry. Do not remove `0907` until `1009` has
booted and been validated independently.
