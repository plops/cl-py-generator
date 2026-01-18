# Project specification (current state)

This document summarizes the current implementation details for the Gentoo
minimal image build. It reflects the present files and configuration in
this directory.

## Build inputs

- Base images:
  - `gentoo/portage:20260118` (full portage tree)
  - `gentoo/stage3:nomultilib-systemd-20260112` (stage3 base)
- Portage tree copied into the stage3 image at `/var/db/repos/gentoo`.
- Profile: `default/linux/amd64/23.0/no-multilib/systemd`.

## Dockerfile iteration approach

While experimenting (for example, kernel configuration tweaks), temporary
changes are appended at the end of the `Dockerfile` to avoid re-running the
full build. Once the changes are validated, those lines are removed and a
clean rebuild is done.

## Toolchain and base packages

- GCC 15 is installed and selected via `eselect gcc`.
- As of early 2026, GCC 15.2.1 is not supported by NVIDIA nvcc, so the
  NVIDIA CUDA SDK is currently disabled. Once nvcc supports GCC 15, CUDA
  will be enabled again. Keeping GCC 14 and 15 side-by-side is avoided to
  reduce confusion and disk usage.
- Additional tools: `gentoolkit`, `eix`, `eselect-repository`, `git`,
  `rust-bin`.
- World/package configuration is taken from:
  - `config/make.conf` (CPU flags, USE flags, mirrors, build settings)
  - `config/world` (base package set)
  - `config/package.use`
  - `config/package.license` and `config/package.license.2`
  - `config/package.accept_keywords` and `config/package.accept_keywords.2`
  - `config/dwm-6.6` (savedconfig for dwm)

## Kernel build

- Kernel sources: `sys-kernel/gentoo-sources` with `KVER_PURE=6.12.58` and
  `KVER=6.12.58-gentoo`.
- Kernel config starts from `config/config6.12.41`, then:
  - `make oldconfig`
  - A scripted option tweak via `./scripts/config` to set specific
    driver and platform options (Bluetooth, Wi-Fi, audio, sensors, USB,
    Thunderbolt, etc.), mostly as modules.
  - `make olddefconfig` to fill remaining defaults.
- Kernel build and install:
  - `make -j32`, `make modules_install`, `make install`.
- Hardware support is validated by booting a modern Ubuntu/Fedora live
  environment and using its `lsmod` output to decide which kernel modules
  to enable.
- When Gentoo bumps the kernel version, a new kernel config file is created.
  This has been needed only once or twice over the last two years.

## Root filesystem build

- `emerge -e @world` followed by `emerge --depclean`.
- Sudo configured for `wheel` group with passwordless sudo.
- User `kiel` created with groups `users,wheel,audio,video`.
- X session config copied to `/home/kiel/.xinitrc`.

## Extra components

- `ryzen_smu` kernel module is built from GitHub and installed into
  `/lib/modules/${KVER}/kernel/`.
- `slstatus` built from source and installed, with config from
  `config/slstatus_config.h`.
- Reverse SSH systemd units are installed and enabled:
  - `config/reverse-ssh-eu.service`
  - `config/reverse-ssh-us.service`

## Network configuration

- `config/20-enp65s0.network` enables DHCP on `enp65s0`.
- `config/resolv.conf` is copied into the image.

## Squashfs image creation

- The filesystem is compressed via `mksquashfs / /gentoo.squashfs` with
  zstd level 22 and a number of exclusions:
  - kernel sources, distfiles, binpkgs, logs, cache, user caches, etc.
  - `boot` and `persistent` are excluded (persistent storage is external).

## Initramfs variants

Two dracut init scripts are shipped, both placed at
`/usr/lib/dracut/modules.d/99base/init.sh` before running dracut:

1. `config/init_dracut_crypt.sh`:
   - Copies squashfs into RAM (tmpfs) before mounting.
   - Creates an overlay with lower filesystem on an encrypted partition.
2. `config/init_dracut_crypt_disk.sh`:
   - Mounts squashfs directly from disk.
   - Uses the same encrypted-overlay strategy as above.

Two initramfs images are built:

- `/boot/initramfs_squash_sda1-x86_64.img` (RAM-copy variant)
- `/boot/initramfs_squash_from_disk.img` (disk-mount variant)
- The disk-mount variant is used when the squashfs image is too large to
  fit into laptop RAM (16GB). Large packages are commented out in
  `config/world` for the laptop-sized build; the full image works on the
  64GB workstation.

The remote-unlock initramfs (with SSH in early boot) is not currently working
and is not used for regular boots.

## Remote unlock assets (optional)

Build arguments allow injecting keys into the image:

- `INITRAMFS_AUTH_KEY` (public key for initramfs shell access)
- `TUNNEL_KEY` (private key used by reverse tunnel script)

The build provisions `/etc/dracut-crypt-ssh` and drops in
`config/start-reverse-tunnels.sh`, which attempts to open reverse SSH
connections to `tinyeu` and `tinyus` on port 2332.

## Persistent storage model

Both the RAM-copy and disk-mount boot paths use an encrypted partition with
overlayfs for persistence. It is possible to `emerge` new programs and store
them on the persistent layer. Periodically, the persistent partition is reset
by deleting directories such as `usr/`, `etc/`, or `var/`, then the resulting
configuration changes (typically from `etc/`) are committed back into this repo
and rebuilt into the image.

## Partition and GRUB layout notes

- GRUB config and kernel live on unencrypted partitions.
- The squashfs and persistent storage partitions are encrypted.
- Disks/partitions are identified by UUID.
- `grub.txt` holds an example GRUB entry template.

On a running system, `/proc/cmdline` is the definitive source for the squashfs
and persistence locations. Example:

```
BOOT_IMAGE=/boot/vmlinuz root=/dev/nvme0n1p2 squashfs.part=/dev/disk/by-uuid/df544e10-90c0-4315-860c-92a58ec8499e squashfs.file=gentoo.squashfs persist.part=/dev/disk/by-uuid/bbac9bb8-39d9-42fa-8d04-94610ced9839 persist.lv=/dev/mapper/enc persist.mapname=enc
```

## GRUB maintenance caveats

- Ubuntu GRUB updates can overwrite dependencies and custom config.
- GRUB is edited directly in `grub.cfg` for speed, even though the proper
  approach would be to update the Ubuntu GRUB config sources.
- Keeping an Ubuntu/Fedora install around is useful for recovery and hardware
  probing.

## Artifact export

Inside the container, `copy_files.sh` copies the generated artifacts to the
host via `/tmp/outside` (bind-mounted to `/dev/shm`):

- `/gentoo.squashfs`
- `/boot/vmlinuz`
- `/boot/initramfs_squash_sda1-x86_64.img`
- `/boot/initramfs_squash_from_disk.img`
- `/boot/initramfs-with-ssh.img` (if built)
- `packages.txt` (output of `qlist -Iv`)

## QEMU test image

`setup04_create_qemu.sh` creates `qemu/sda1.img` and installs GRUB:

- Partition table: MBR (msdos).
- `p1`: FAT32 (1MiB-900MB), mounted at `/mnt`, contains kernel/initramfs
  and `gentoo.squashfs`.
- `p2`: ext4 (900MB-end), created but not mounted by the script.

`setup05_run_qemu.sh` boots the raw image in QEMU with KVM enabled.

## Hardware inventory snapshots

The `systems/` folder contains scripts and captured outputs that describe
hardware for multiple hosts. These are used to guide kernel module selection
and configuration.

- `systems/00_setup_gather_dependencies.sh` installs the minimal tooling
  needed to gather hardware info.
- `systems/01_gather_info.sh` collects hardware-only outputs.
- `systems/03_anonymize_files.sh` redacts sensitive tokens in captured logs.

## Build timing notes

- Clean rebuild time is roughly 2.5 hours on the host hardware.
- Docker base image tags are periodically updated to newer timestamps and
  rebuilt.
