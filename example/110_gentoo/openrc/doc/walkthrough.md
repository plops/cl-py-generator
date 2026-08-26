# HP Z6 OpenRC Squashfs Deployment Walkthrough

This walkthrough explains how to deploy a dated Gentoo OpenRC squashfs bundle
to the HP Z6 without overwriting the running system. The concrete example is
the August 26, 2026 deployment from
`/dev/shm/gentoo-z6-min-openrc_20260826` into `/boot/0826`.

For the historical record, exact hashes, and older deployments, see
[install_on_hpz6.md](install_on_hpz6.md).

## Storage Layout

The boot artifacts and GRUB configuration live on the Btrfs filesystem with
UUID `4f708c84-185d-437b-a03a-7a565f598a23`. While a live image is running,
that filesystem is mounted at:

```text
/run/initramfs/live
```

The persistent overlay is the LUKS partition with UUID
`0d7c5e23-6bab-4dce-b744-a5d61d497aca`.

NVMe kernel names can change between boots. Use these stable references when
identifying the disks:

```text
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part5
/dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part6
```

Do not deploy into the running root's ordinary `/boot`. Use
`/run/initramfs/live/boot`.

## 1. Define the New Deployment

For the August deployment:

```bash
SOURCE_DATE=20260826
SLOT=0826
ARTIFACT_DIR=/dev/shm/gentoo-z6-min-openrc_20260826
TARGET_DIR=/run/initramfs/live/boot/0826
```

Keep the date and slot explicit and inspect their values before using them:

```bash
printf 'source date: %s\nslot: %s\nsource: %s\ntarget: %s\n' \
  "$SOURCE_DATE" "$SLOT" "$ARTIFACT_DIR" "$TARGET_DIR"
```

The HP Z6 artifact is `gentoo.squashfs_nv`. Do not accidentally install the
ThinkPad E14 artifact, `gentoo.squashfs_e14`.

## 2. Verify the Mount and Running Slot

Inspect the artifact filesystem and stable device mapping:

```bash
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
readlink -f \
  /dev/disk/by-id/nvme-MTFDKBA1T0TFH-1BC1AABHA_UMDMD01J1IZ9A9-part3
lsblk -f
```

Read the running slot from the kernel command line:

```bash
cat /proc/cmdline
```

Look for both `BOOT_IMAGE=/boot/<slot>/vmlinuz` and
`rd.live.dir=/boot/<slot>`. During the August installation both identified
`0625`. A previously expected value of `0624` was not trusted over the running
kernel's evidence.

Never delete:

- The slot named by `/proc/cmdline`.
- A slot the operator explicitly asked to preserve.
- The direct-disk fallback.
- Every older known-good squashfs; retain at least one fallback.

For the August installation, both `0624` and `0625` were treated as protected,
even though no `0624` directory existed.

## 3. Inspect and Identify the Source Bundle

Require the five deployment artifacts:

```bash
ls -lah "$ARTIFACT_DIR"
stat -c '%n %s bytes %y' \
  "$ARTIFACT_DIR/gentoo.squashfs_nv" \
  "$ARTIFACT_DIR/vmlinuz" \
  "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img" \
  "$ARTIFACT_DIR/packages.txt" \
  "$ARTIFACT_DIR/packages.tsv"
```

Describe the image and kernel:

```bash
file "$ARTIFACT_DIR/gentoo.squashfs_nv"
file "$ARTIFACT_DIR/vmlinuz"
strings "$ARTIFACT_DIR/vmlinuz" | grep -m 2 'Linux version'
```

Compute source hashes before copying:

```bash
sha256sum \
  "$ARTIFACT_DIR/gentoo.squashfs_nv" \
  "$ARTIFACT_DIR/vmlinuz" \
  "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img" \
  "$ARTIFACT_DIR/packages.txt" \
  "$ARTIFACT_DIR/packages.tsv"
```

When the container image still exists, its history can distinguish similar
build projects:

```bash
sudo docker image inspect gentoo-z6-min-openrc:latest \
  --format 'created={{.Created}} id={{.Id}}'
sudo docker history --no-trunc --format '{{.CreatedBy}}' \
  gentoo-z6-min-openrc:latest | grep -E \
  'reverse-ssh|config6|KVER_PURE|gentoo-sources'
```

For `0826`, direct `COPY config/reverse-ssh-*.initd` layers confirmed that the
image came from:

```text
/home/kiel/stage/cl-py-generator/example/110_gentoo/openrc
```

The installed kernel is `6.18.41-gentoo-dist`, built from
`sys-kernel/gentoo-sources-6.18.41` with GCC 15.3.0 and GNU ld 2.46.1.

## 4. Check Space Before Writing

Compare free space with the source bundle and list the dated slots:

```bash
df -h /run/initramfs/live /dev/shm
du -sh /run/initramfs/live/boot/* 2>/dev/null | sort -h
du -sh "$ARTIFACT_DIR"
```

The August deployment began with 2.1 GiB free. The new NV squashfs was
2,398,760,960 bytes, so cleanup was required before copying.

The oldest dated squashfs slot was `0427`, about 2.2 GiB. It was not running,
was not operator-protected, and newer fallbacks `0428`, `0608`, and `0625`
remained. Its contents were listed before deletion:

```bash
find /run/initramfs/live/boot/0427 -mindepth 1 -maxdepth 1 \
  -printf '%y %f %s bytes\n' | sort
```

## 5. Remount, Back Up GRUB, and Free Space

The live artifact filesystem normally starts read-only. Temporarily remount it
read-write and back up GRUB before changing either slots or menu entries:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg \
  /run/initramfs/live/boot/grub/custom.cfg.before-0826
```

Delete only the previously inspected, unprotected old slot. For the August
deployment this was `/run/initramfs/live/boot/0427`. Verify that it is absent,
sync, and check the recovered space before continuing:

```bash
sudo rm -rf -- /run/initramfs/live/boot/0427
sync
test ! -e /run/initramfs/live/boot/0427
df -h /run/initramfs/live
```

This increased available space to about 4.2 GiB. The deletion was direct, not
a move to trash; recovery requires the old source artifacts or another backup.
Remove the matching GRUB menu entry when installing the new configuration so
the menu does not point to missing files.

## 6. Copy the New Artifacts

Require a new target rather than silently merging with a partial deployment:

```bash
test ! -e "$TARGET_DIR"
sudo mkdir "$TARGET_DIR"
```

Copy and rename the HP Z6 squashfs while preserving the other filenames:

```bash
sudo cp -av "$ARTIFACT_DIR/gentoo.squashfs_nv" \
  "$TARGET_DIR/gentoo.squashfs"
sudo cp -av "$ARTIFACT_DIR/vmlinuz" "$TARGET_DIR/vmlinuz"
sudo cp -av "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img" \
  "$TARGET_DIR/initramfs_squash_sda1-x86_64.img"
sudo cp -av "$ARTIFACT_DIR/packages.txt" "$TARGET_DIR/packages.txt"
sudo cp -av "$ARTIFACT_DIR/packages.tsv" "$TARGET_DIR/packages.tsv"
sync
```

## 7. Add `SOURCE.txt`

Every new dated directory should contain a human-readable `SOURCE.txt`. Record:

- Installation date, source bundle, and target slot.
- Build-project path and repository commit.
- Whether relevant build files had uncommitted modifications.
- Container tag, image ID, and creation time when available.
- Why `gentoo.squashfs_nv` was selected.
- Gentoo stage3, Portage snapshot, and profile.
- Kernel source version, release, configuration source, compiler, and linker.
- The role of the accompanying initramfs.
- SHA-256 hashes using the installed filenames.

The deployed example is:

```text
/run/initramfs/live/boot/0826/SOURCE.txt
```

Including the dirty-build warning matters: a Git commit cannot reproduce an
image if `Dockerfile` or kernel configuration changes were not committed.

## 8. Add the GRUB Entry

Add the new entry before older entries in
`/run/initramfs/live/boot/grub/custom.cfg`:

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

The directory in `linux`, `rd.live.dir`, and `initrd` must match. Keep
`rd.live.squashimg=gentoo.squashfs` aligned with the installed filename. The
AMD microcode image must precede the dated initramfs.

## 9. Verify Before Returning to Read-Only

List the installed files and compare every source/target hash pair:

```bash
ls -lah "$TARGET_DIR"
sha256sum "$TARGET_DIR/gentoo.squashfs" \
  "$ARTIFACT_DIR/gentoo.squashfs_nv"
sha256sum "$TARGET_DIR/vmlinuz" "$ARTIFACT_DIR/vmlinuz"
sha256sum "$TARGET_DIR/initramfs_squash_sda1-x86_64.img" \
  "$ARTIFACT_DIR/initramfs_squash_sda1-x86_64.img"
sha256sum "$TARGET_DIR/packages.txt" "$ARTIFACT_DIR/packages.txt"
sha256sum "$TARGET_DIR/packages.tsv" "$ARTIFACT_DIR/packages.tsv"
```

Each pair must have identical hashes. Check the new entry, retained fallbacks,
and backup:

```bash
grep -nE 'menuentry.*(0826|0625|0608|0428)|rd.live.dir=/boot/(0826|0625|0608|0428)' \
  /run/initramfs/live/boot/grub/custom.cfg
test -d /run/initramfs/live/boot/0625
test -f /run/initramfs/live/boot/grub/custom.cfg.before-0826
```

If `grub-script-check` is installed, run it against `custom.cfg`. It was not
installed during the August deployment, so the entry was checked by inspecting
its structure and verifying every referenced file.

Sync, restore read-only mode, and confirm it:

```bash
sync
sudo mount -o remount,ro /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
df -h /run/initramfs/live
```

The final August state had about 1.9 GiB free. `df` displayed 100% utilization
because of rounding, but all artifacts had already been copied and verified.

## 10. Reboot and Validate

On the next reboot select:

```text
Gentoo Dracut (persist on luks overlay 0826 OpenRC NV folder)
```

After login, confirm that the new slot supplied the kernel and live root:

```bash
cat /proc/cmdline
uname -r
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

Expected values include:

```text
BOOT_IMAGE=/boot/0826/vmlinuz
rd.live.dir=/boot/0826
6.18.41-gentoo-dist
```

Do not remove `0625` merely because `0826` copied successfully. Keep it until
the new image has booted and its networking, overlay, graphics, and required
services have been validated.

## Rollback

If `0826` fails to boot, select `0625` from GRUB. To remove only the new slot
and restore the previous menu after booting a fallback:

```bash
sudo mount -o remount,rw /run/initramfs/live
sudo rm -rf -- /run/initramfs/live/boot/0826
sudo cp -av /run/initramfs/live/boot/grub/custom.cfg.before-0826 \
  /run/initramfs/live/boot/grub/custom.cfg
sync
sudo mount -o remount,ro /run/initramfs/live
findmnt /run/initramfs/live -o SOURCE,TARGET,FSTYPE,OPTIONS
```

The additional fallbacks are `0608`, `0428`, and the direct `Gentoo OpenRC
disk` entry.
