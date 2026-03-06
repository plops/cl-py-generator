# Gentoo System Information - ThinkPad E14 (0306 Update)

This directory contains system snapshots of the Gentoo installation on the ThinkPad E14 workstation, updated on **2026-03-06** (Build `0306`).

## Version Summary

- **Kernel**: `6.12.74-gentoo` (Build date: 2026-03-05)
- **Base Build**: `gentoo-z6-min_20260306`
- **Boot Method**: Dracut Live (SquashFS) with LUKS persistence on `nvme0n1p4`.

## Key Changes (0226 -> 0306)

### 1. Kernel and Build
- Upgraded kernel from `6.12.58` to `6.12.74`.
- Updated compiler/toolchain signatures in `dmesg` (Gentoo 16.0.1 experimental).

### 2. Boot Configuration
- Boot artifacts now use the `_0306` suffix:
    - `vmlinuz_0306`
    - `initramfs_squash_sda1-x86_64.img_0306`
    - `gentoo.squashfs_0306`
- Active command line updated to reference these artifacts.

### 3. Storage and Memory
- **Image Size**: The SquashFS image size increased significantly (~1.4G -> ~2.7G), as reflected in `df.txt` and `lsblk`.
- **Rootfs Usage**: `/run/rootfsbase` (the mounted squashfs) now consumes 2.7G of RAM (when `rd.live.ram=1` is used).
- **Persistent Storage**: `nvme0n1p4` (decrypted to `enc`) remains the primary persistent layer.

### 4. Hardware and Runtime
- **Battery**: Cycle count increased from 86 to 95. Capacity slightly adjusted (95.3% -> 95.0%).
- **NVMe Endurance**:
    - `nvme0n1` (1TB) reached **20.9 TB** total writes.
    - `nvme1n1` (512GB) reached **5.70 TB** total writes.
    - Logged ~95 additional Power On Hours since the 0226 snapshot.
- **CPU**: `cpuinfo.txt` reflects active frequency scaling during the snapshot.
- **Modules**: Minor adjustments in loaded kernel modules.

## Snapshot Files
- `blkid.txt`: Partition UUIDs and labels.
- `cpuinfo.txt` / `lscpu.txt`: Processor details (AMD Ryzen 7 7735HS).
- `df.txt` / `lsblk_all.txt`: Filesystem and block device status.
- `dmesg.txt`: Kernel ring buffer after successful boot.
- `tlp-stat.txt`: Power management and battery health status.
- `uname.txt`: Current kernel signature.
