# Gentoo HP Z6 Workstation Build

A Docker-based Gentoo Linux build system that creates a compressed squashfs root filesystem optimized for HP Z6 workstations. ThinkPad E14 support is prepared but currently disabled. The system uses an immutable SquashFS base with encrypted OverlayFS persistence.

## Quick Start

```bash
# Build the Docker image
./setup01_run_with_log.sh

# Extract artifacts to host
./setup03_copy_from_container.sh

# Create QEMU test image (optional)
sudo ./setup04_create_qemu.sh

# Boot in QEMU (optional)
sudo ./setup05_run_qemu.sh
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
