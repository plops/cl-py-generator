# Gentoo HP Z6 Workstation Build

A Docker-based Gentoo Linux build system that creates a compressed squashfs root filesystem optimized for HP Z6 workstations. ThinkPad E14 support is prepared but currently disabled. The system uses an immutable SquashFS base with encrypted OverlayFS persistence.

## Quick Start

```bash
# Build the Docker image
./setup01_run_with_log.sh

# Extract artifacts to host
./setup03_copy_from_container.sh

# Create a QEMU test image with:
#   p1 = unencrypted boot/live partition with kernel + initramfs + gentoo.squashfs
#   p2 = LUKS2 -> ext4 persistent overlay storage
sudo ./setup04_create_qemu.sh

# Boot in QEMU and unlock with the passphrase printed by setup04
./setup05_run_qemu.sh
```

## Artifacts

Build outputs are copied to `/dev/shm/gentoo-z6-min_YYYYMMDD/`:

- `gentoo.squashfs` - Compressed root filesystem (~3GB)
- `vmlinuz` - Linux kernel
- `initramfs_squash_sda1-x86_64.img` - Boot initramfs
- `packages.txt` - Installed package list

## Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build configuration with 10-stage package installation |
| `config/world` | List of packages to install (~50 active) |
| `config/make.conf` | GCC 14, -march=native optimization |
| `config/config6.12.41` | Kernel configuration base |
| `config/mount-overlayfs.sh` | Custom dracut module for encrypted overlay mounting |
| `setup04_create_qemu.sh` | Builds a BIOS/QEMU disk image that boots squashfs from LUKS-ext4 |
| `setup05_run_qemu.sh` | Runs the test image on the serial console |

## QEMU Test Flow

`setup04_create_qemu.sh` looks for the newest `/dev/shm/gentoo-z6-min-openrc_*` export by default. You can override that with `ARTIFACT_DIR=/dev/shm/gentoo-z6-min-openrc_YYYYMMDD`.

Useful overrides:

```bash
sudo IMAGE_SIZE=12G ./setup04_create_qemu.sh
QEMU_MEMORY=8192 QEMU_SMP=8 ./setup05_run_qemu.sh
```

At boot, enter the fixed test LUKS passphrase `openrc-test` on the serial console when dracut prompts for it. The builder uses the hygienic host-side mapper name `qemuenc` during image construction, but the guest kernel command line opens the encrypted persistence partition as `/dev/mapper/enc` so it matches the current initramfs logic in the image. The live squashfs remains on the unencrypted boot partition for this QEMU test setup.

## Documentation

- **[doc/README.md](doc/README.md)** - Comprehensive project overview and workflow
- **[doc/spec.md](doc/spec.md)** - Detailed implementation specifications
- **[doc/install_on_hpz6.md](doc/install_on_hpz6.md)** - Installation guide for HP Z6
- **[doc/dependency_audit.md](doc/dependency_audit.md)** - Package size analysis

## Build Features

- **GCC 14 toolchain** (reverted from experimental GCC 16)
- **10-stage Docker caching** for efficient rebuilds
- **Hardware-specific optimization** for AMD Threadripper PRO 7955WX and Ryzen 7735HS (via -march=native)
- **Encrypted persistence** via LUKS + LVM + OverlayFS
- **SquashFS compression** at level 19 (~36% size reduction)

## Notes

This build is part of a larger Gentoo live system project. See the parent directory for other variants including a minimal QEMU-optimized build.

Wiki pages you might want to explore:
- [Gentoo Linux Live Systems (plops/cl-py-generator)](https://deepwiki.com/plops/cl-py-generator/5-gentoo-linux-live-systems)
