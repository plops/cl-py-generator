# Gentoo minimal squashfs image (ThinkPad + HP Z6)

This project builds a Gentoo-based image from scratch in Docker, produces a
compressed squashfs root, and packages kernel + initramfs artifacts for booting
from disk or from RAM. The resulting system is tuned to boot on a ThinkPad laptop
and an HP Z6 workstation, with tooling and configs captured here for repeatable
rebuilds.

Reference notes from the earliest setup live in `../README.md`.

## What this repository does

- Builds a Gentoo system inside Docker using stage3 + portage bases.
- Compiles a custom kernel and builds multiple initramfs variants via dracut.
- Produces a squashfs root filesystem and copies artifacts to the host.
- Optionally packages a bootable raw disk image for QEMU testing.
- Uses hardware probes from a modern live OS (Ubuntu/Fedora) to guide kernel
  module selection via `lsmod`.

## Update cadence

- The Docker base images (Gentoo portage and stage3 tags) are periodically
  updated, typically every few weeks, and the image is rebuilt.
- A clean rebuild from scratch takes about 2.5 hours.

## Primary workflow

1. Build the image (passing initramfs SSH keys for optional remote unlock):
   - `setup01_build_image.sh`
2. Run and extract artifacts to the host:
   - `setup03_copy_from_container.sh`
3. Create a bootable raw disk image for QEMU:
   - `setup04_create_qemu.sh`
4. Boot the raw image in QEMU:
   - `setup05_run_qemu.sh`

For a one-shot local run, use `build_and_run.sh`.

## Artifacts

When `setup03_copy_from_container.sh` runs, artifacts are copied to
`/dev/shm/gentoo-z6-min_YYYYMMDD/`:

- `gentoo.squashfs` (compressed root filesystem)
- `vmlinuz` (kernel)
- `initramfs_squash_sda1-x86_64.img` (squashfs copied to RAM)
- `initramfs_squash_from_disk.img` (squashfs mounted from disk)
- `initramfs-with-ssh.img` (if built; remote unlock assets)
- `packages.txt` (installed packages snapshot)

## Hardware support workflow

To support new hardware, a modern Ubuntu or Fedora live boot is used to ensure
drivers load. The `lsmod` output from that environment is then used to enable
the required kernel modules in the Gentoo build.

## Image size note

The squashfs image can be large enough that it may not fit into RAM on the
laptop (16GB), so the disk-mount initramfs variant is used in those cases.
Large packages are commented out in `config/world` for the laptop-sized build.
On the 64GB workstation, the full image works fine.

## Kernel config cadence

When Gentoo bumps the kernel, a new kernel config file is created. This has
been needed only once or twice over the last two years of running this project.

## CUDA/NVIDIA toolchain note

As of early 2026, GCC 15.2.1 is in use but NVIDIA nvcc does not yet support
GCC 15, so the NVIDIA CUDA SDK is currently disabled. Once nvcc supports GCC 15,
CUDA will be enabled again. Keeping GCC 14 and 15 side-by-side is avoided to
reduce confusion and disk usage.

## Initramfs status

- The remote-unlock initramfs with an SSH server is a work in progress and is
  not currently used.
- Day-to-day boots use the RAM-copy initramfs and the disk-mount initramfs.

## Dockerfile iteration workflow

To avoid expensive rebuilds while experimenting (for example, adding kernel
config options), temporary changes are typically appended to the end of the
`Dockerfile`. Once validated, those lines are removed and a full clean build is
run again.

## Partitions, GRUB, and disk identification

- GRUB config and kernel are stored on unencrypted partitions to avoid the
  complexity of encrypted boot in GRUB.
- The squashfs and the persistent partition are encrypted.
- Disks/partitions are identified by UUID.
- `grub.txt` contains an example GRUB entry used as a template.

On a running system, `/proc/cmdline` shows where the squashfs and persistence
layers come from. Example:

```
BOOT_IMAGE=/boot/vmlinuz root=/dev/nvme0n1p2 squashfs.part=/dev/disk/by-uuid/df544e10-90c0-4315-860c-92a58ec8499e squashfs.file=gentoo.squashfs persist.part=/dev/disk/by-uuid/bbac9bb8-39d9-42fa-8d04-94610ced9839 persist.lv=/dev/mapper/enc persist.mapname=enc
```

## GRUB maintenance caveats

- Ubuntu-installed GRUB updates can overwrite dependencies and custom config.
- GRUB is edited directly in `grub.cfg` for speed, even though the proper
  approach would be to update Ubuntu GRUB config sources.
- Keeping an Ubuntu/Fedora install around is useful for recovery and for
  probing hardware when drivers are unknown.

## Persistent storage workflow

- Both boot variants use an encrypted partition with overlayfs for persistence.
- It is possible to `emerge` programs and store them on the persistent layer.
- Occasionally the persistent partition is reset by deleting directories such
  as `usr/`, `etc/`, or `var/`, then the updated configuration is committed back
  into this repo and rebuilt into the image.

## Notable configuration areas

- Docker build: `Dockerfile`
- Gentoo settings: `config/make.conf`, `config/world`,
  `config/package.use`, `config/package.accept_keywords*`,
  `config/package.license*`
- Kernel config: `config/config6.12.41`
- Initramfs init scripts: `config/init_dracut_crypt.sh`,
  `config/init_dracut_crypt_disk.sh`
- Reverse tunnel units: `config/reverse-ssh-*.service`
- Networking: `config/20-enp65s0.network`, `config/resolv.conf`
- X session: `config/xinitrc`, `config/slstatus_config.h`

## Hardware inventory snapshots

The `systems/` folder stores non-intrusive hardware inventory snapshots
and helper scripts used to capture/anonymize them. This is used to guide
kernel module selection and hardware support.
